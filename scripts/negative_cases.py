"""
Negative-control case builder  (segments that FAILED the persistence gate)
=========================================================================

All six gated cases in the thesis came back likely_breach, so the agent was never
shown a pair that should NOT be a breach. This script re-runs Stage 2 over the
held-out sequence exactly as batch_cases.py does, but captures the pair episodes
that lasted >= 3 s yet failed the persistence gate (never held h_gap < 3 s for a
continuous 3 s). Those are the natural negative controls: real following pairs,
same pipeline, same evidence format, but not sustained close following.

It makes NO API calls. It writes a desk/repeat-compatible case file; you then send
chosen cases to the agent with the existing repeat_agent.py.

    # 1. FREE: build negative-control cases from the held-out run
    python scripts/negative_cases.py --weights weights/best.pt --dataset ./dataset_eval \
        --out batch_neg/negative_cases.json

    # 2. Read batch_neg/negative_summary.txt, pick 3-4 labels (N1, N2, ...)
    #    (the summary marks suggested picks), check the cost:
    python scripts/repeat_agent.py batch_neg/negative_cases.json --cases N1,N2,N3 --runs 2 --dry-run

    # 3. Run them (costs money; ~US$0.13 per call in practice)
    python scripts/repeat_agent.py batch_neg/negative_cases.json --cases N1,N2,N3 --runs 2

    # 4. Optional: view them in the desk
    python scripts/merge_verdicts.py batch_neg/negative_cases.json   # -> batch_neg/cases.json

Outputs (next to --out):
    negative_cases.json     {"cases": [...]} - same case format as batch_cases.json
    negative_summary.txt    table of candidates with headway stats + suggested picks
    case_frames/            dual-panel review frames for each candidate
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import batch_cases as bc                      # reuse the exact Stage 2 setup
from batch_cases import (cfg, SIM_DT, TRANSIENCY_S, SplitResolver, index_frames,
                         load_meta, preset_for, env_for, _longest_deficit,
                         CameraIntrinsics, CameraPose, TrafficCamera,
                         TrafficCamPerceptionPipeline, FrameInput,
                         export_segment_frames, load_rgb)


def capture(pipe, log: dict, negatives: list):
    """Wrap _close_pair. Gate outcomes are computed exactly as in batch_cases.instrument.
    Episodes that fail ONLY the persistence gate are force-closed into a segment and
    kept as negative controls; they are NOT yielded by the pipeline, so normal
    behaviour (what counts as a case) is unchanged."""
    orig_close = pipe._close_pair

    def close(key, force=False):
        obs = list(pipe._pairs.get(key, []))
        rec = {"key": list(key), "n_obs": len(obs),
               "duration_s": round(obs[-1].t - obs[0].t, 2) if len(obs) >= 2 else 0.0}
        if len(obs) < 2:
            rec["outcome"] = "too_few_obs"
        elif force:
            rec["outcome"] = "force_flushed_end_of_stream"
        elif rec["duration_s"] < pipe.min_segment_duration_s:
            rec["outcome"] = "failed_duration_gate"
        else:
            rec["longest_deficit_s"] = round(_longest_deficit(pipe, obs), 2)
            rec["outcome"] = ("passed" if rec["longest_deficit_s"] >= pipe.deficit_min_seconds
                              else "failed_persistence_gate")
        log["episodes"].append(rec)

        if rec["outcome"] == "failed_persistence_gate":
            seg = orig_close(key, force=True)       # bypass the gates only for capture
            if seg is not None:
                negatives.append((seg, rec["longest_deficit_s"]))
            return None                             # pipeline does not emit it
        return orig_close(key, force=force)

    pipe._close_pair = close


def stats(seg) -> dict:
    a = seg.aggregates()
    g3 = a.get("sustained_below_3s_seconds") or 0
    c3 = a.get("sustained_clearance_below_3s_seconds") or 0
    g2 = a.get("sustained_below_2s_seconds") or 0
    c2 = a.get("sustained_clearance_below_2s_seconds") or 0
    return {
        "duration_s": round(seg.duration_s, 2),
        "usable": a.get("usable_samples"),
        "v_kmh": a.get("mean_follower_speed_kmh"),
        "median_gap_s": a.get("median_gap_s"),
        "min_gap_s": a.get("min_gap_s"),
        "median_clear_s": a.get("median_clearance_s"),
        "gap3": g3, "clr3": c3, "gap2": g2, "clr2": c2,
        "min_ttc": a.get("min_ttc_s"),
        "heur_gap": g3 >= TRANSIENCY_S or g2 >= 3.0,
        "heur_clear": c3 >= TRANSIENCY_S or c2 >= 3.0,
    }


def suggest(rows: list[dict], n: int) -> set[str]:
    """Half 'clean' negatives (most time at/above 3 s on clearance), half borderline
    (closest to the gate). Only rows with usable samples."""
    ok = [r for r in rows if (r["s"]["usable"] or 0) > 0]
    clean = sorted(ok, key=lambda r: (r["s"]["clr3"], -(r["s"]["median_clear_s"] or 0)))
    border = sorted(ok, key=lambda r: -r["deficit"])
    pick = []
    for r in clean[: (n + 1) // 2] + border:
        if r["label"] not in pick:
            pick.append(r["label"])
        if len(pick) >= n:
            break
    return set(pick)


def run(weights, dataset, conf, frames_per_preset, n_suggest, out, detector=None):
    dataset, out = Path(dataset), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "case_frames"

    idx = index_frames(dataset)
    if not idx:
        raise SystemExit(f"No frame_*.png under {dataset}/images/*")
    ids = sorted(idx)
    resolver = SplitResolver({p.name: p for p in idx.values()})
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z,
        calibration_rmse_m=cfg.CALIBRATION_RMSE_M,
        max_reliable_range_m=cfg.MAX_RELIABLE_RANGE_M)

    meta = load_meta(dataset)
    t_of = (lambda f: meta[f]["t"]) if meta else (lambda f: f * SIM_DT)
    preset_of = (lambda f: meta[f]["preset"]) if meta else (lambda f: preset_for(f, frames_per_preset))
    ep_of = (lambda f: meta[f]["episode"]) if meta else (lambda f: 0)
    print(f"{len(ids)} frames; "
          + (f"frames.csv found ({len({m['episode'] for m in meta.values()})} episodes)" if meta
             else "no frames.csv (timestamps = frame_id * dt)"))

    if detector is None:
        from ultralytics import YOLO
        detector = YOLO(weights)

    log = {"episodes": []}
    frame_boxes: dict[int, dict] = {}
    captured, n_passed = [], 0
    episodes = sorted({ep_of(f) for f in ids})
    for ep in episodes:
        ep_ids = [f for f in ids if ep_of(f) == ep]
        pipe = TrafficCamPerceptionPipeline(camera=cam, yolo_weights=weights,
                                            conf_threshold=conf, detector=detector)
        negs: list = []
        capture(pipe, log, negs)
        orig_update = pipe.tracker.update

        def logged_update(dets, t, frame_id, _u=orig_update):
            tracks = _u(dets, t, frame_id)
            frame_boxes[frame_id] = {tid: (tr.last_bbox, tr.cls) for tid, tr in tracks.items()}
            return tracks
        pipe.tracker.update = logged_update

        def stream(ep_ids=ep_ids):
            for fid in ep_ids:
                if fid % 250 == 0:
                    print(f"  frame {fid}  preset={preset_of(fid)}  episode={ep}")
                yield FrameInput(frame_id=fid, timestamp_s=t_of(fid),
                                 rgb_image=load_rgb(idx[fid]),
                                 environment=env_for(preset_of(fid)))

        for _ in pipe.process_stream(stream()):
            n_passed += 1                           # gated cases (already in the thesis)
        for seg, deficit in negs:
            if len(episodes) > 1:
                seg.segment_id = f"ep{ep:02d}_{seg.segment_id}"
            captured.append((seg, deficit))

    outcomes = Counter(e["outcome"] for e in log["episodes"])
    captured.sort(key=lambda x: x[0].samples[0].frame_id if x[0].samples else 0)

    rows, cases = [], []
    for i, (seg, deficit) in enumerate(captured, 1):
        label = f"N{i}"
        start = seg.samples[0].frame_id if seg.samples else None
        preset = preset_of(start) if start is not None else "?"
        s = stats(seg)
        frames = export_segment_frames(seg, frame_boxes, resolver, frames_dir, cam) if seg.samples else []
        flags = [f"negative control: failed persistence gate "
                 f"(longest continuous run with h_gap < 3 s = {deficit:.2f} s)"]
        cases.append({"case_id": seg.segment_id, "label": label, "status": "pending_review",
                      "carla_preset": preset, "start_frame": start, "flags": flags,
                      "evidence": seg.evidence_brief(), "frames": frames, "verdict": None})
        rows.append({"label": label, "seg": seg.segment_id, "preset": preset, "deficit": deficit,
                     "pair": f"{seg.follower_vehicle_class} behind {seg.lead_vehicle_class}",
                     "frames": f"{start}-{seg.samples[-1].frame_id}" if seg.samples else "-",
                     "s": s})

    picks = suggest(rows, n_suggest)
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site_id": "TOWN04_HW_01", "model": None, "temperature": None, "reproducible": False,
        "purpose": "negative controls: pair episodes >= 3 s that failed the persistence gate",
        "batch_summary": {"frames_processed": len(ids), "conf_threshold": conf,
                          "pair_episodes": len(log["episodes"]), "outcomes": dict(outcomes),
                          "gated_cases_emitted": n_passed, "negative_candidates": len(cases),
                          "suggested": sorted(picks, key=lambda x: int(x[1:]))},
        "cases": cases,
    }
    out.write_text(json.dumps(doc, indent=2))

    L = [f"Negative-control build {doc['generated_utc']}  frames {ids[0]}..{ids[-1]}  conf={conf}", "",
         "Gate outcomes (should match the thesis held-out row):"]
    for k in ("too_few_obs", "failed_duration_gate", "failed_persistence_gate",
              "passed", "force_flushed_end_of_stream"):
        L.append(f"  {k:30s} {outcomes.get(k, 0)}")
    L += ["", f"NEGATIVE CANDIDATES ({len(rows)})   * = suggested pick", "",
          f"{'':1s}{'case':4s} {'pair':34s} {'preset':14s} {'frames':11s} {'dur':>5s} {'v':>5s} "
          f"{'deficit':>7s} {'med_g':>6s} {'med_c':>6s} {'g<3':>5s} {'c<3':>5s} {'g<2':>5s} "
          f"{'c<2':>5s} {'heur g/c':>8s} {'n':>4s}"]
    for r in rows:
        s = r["s"]
        f = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "-"
        L.append(f"{'*' if r['label'] in picks else ' '}{r['label']:4s} {r['pair'][:34]:34s} "
                 f"{r['preset'][:14]:14s} {r['frames']:11s} {s['duration_s']:5.2f} "
                 f"{(s['v_kmh'] or 0):5.1f} {r['deficit']:7.2f} {f(s['median_gap_s']):>6s} "
                 f"{f(s['median_clear_s']):>6s} {s['gap3']:5.2f} {s['clr3']:5.2f} {s['gap2']:5.2f} "
                 f"{s['clr2']:5.2f} {('Y' if s['heur_gap'] else 'N') + '/' + ('Y' if s['heur_clear'] else 'N'):>8s} "
                 f"{(s['usable'] or 0):4d}")
    L += ["", "deficit = longest continuous run with h_gap < 3 s (gate needs >= 3 s)",
          "heur g/c = prompt heuristic (>=5 s below 3 s or >=3 s below 2 s) met on gap / clearance",
          "Expected agent answer for a clean negative: unlikely_breach or possible_breach.",
          "A 'Y' under clearance means the clearance convention alone would call it - a useful",
          "borderline test of the convention effect.", "", "Segment IDs:"]
    L += [f"  {r['label']}: {r['seg']}" for r in rows]
    txt = "\n".join(L)
    (out.parent / "negative_summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nWrote {out.resolve()}\nWrote {(out.parent / 'negative_summary.txt').resolve()}"
          f"\nNext: python scripts/repeat_agent.py {out} --cases "
          f"{','.join(sorted(picks, key=lambda x: int(x[1:])))} --runs 2 --dry-run")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="./dataset_eval")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--frames-per-preset", type=int, default=250)
    p.add_argument("--suggest", type=int, default=4, help="how many picks to mark in the summary")
    p.add_argument("--out", default="batch_neg/negative_cases.json")
    a = p.parse_args()
    run(a.weights, a.dataset, a.conf, a.frames_per_preset, a.suggest, a.out)
