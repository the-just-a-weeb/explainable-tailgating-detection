"""
Batch case builder  (Stage 3 over the whole collection -> cases + gate counts)
==============================================================================

Runs the Stage 3 pipeline over the FULL frame sequence (train + val merged, so the
stream is contiguous at 20 Hz), logs why every candidate pair was kept or dropped,
and writes a desk-compatible case file plus a summary table for the thesis.

Why this exists (vs build_case_file.py)
---------------------------------------
build_case_file.py streams a window of ONE split (default: train) and timestamps
frames as i * dt. Because the split is a random 85/15 per-frame split, consecutive
train frames are not consecutive in time, so time is compressed by ~6-15%: speeds
come out too high and headways/durations too low. Here, timestamps are
frame_id * dt over the merged, contiguous sequence.

    # 1. FREE dry run: segments, gate counts, review frames. No API calls.
    python scripts/batch_cases.py --weights weights/best.pt

    # 2. Look at batch_summary.txt, then send chosen cases to the agent
    python scripts/batch_cases.py --weights weights/best.pt --call-agent --max-cases 8

    # restrict to a frame range
    python scripts/batch_cases.py --weights weights/best.pt --start-frame 0 --end-frame 750

Outputs (next to --out):
    batch_cases.json      desk-compatible ({"cases": [...]}) + "batch_summary"
    batch_summary.txt     gate counts + per-case table (paste into Results)
    case_frames/          dual-panel review frames, INCLUDING every frame the
                          agent cites (fixes the "cited frame not exported" gap)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import string
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "scripts"))

import site_config as cfg  # noqa
from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
from perception_pipeline import (TrafficCamPerceptionPipeline, FrameInput,  # noqa
                                 NOMINAL_LENGTH_M)
from reasoning_agent import TailgatingReasoningAgent, EnvironmentContext  # noqa
from build_case_file import export_segment_frames, load_rgb  # noqa

SIM_DT = 1.0 / cfg.SIM_HZ
TRANSIENCY_S = 5.0          # prompt heuristic: sustained_below_3s >= 5 s

# CARLA preset -> (weather, road_surface) in the agent's EnvironmentContext vocab.
PRESET_ENV = {
    "ClearNoon":      ("dry_clear",  "dry"),
    "CloudyNoon":     ("overcast",   "dry"),
    "WetNoon":        ("dry_clear",  "wet"),
    "WetCloudyNoon":  ("overcast",   "wet"),
    "MidRainyNoon":   ("rain_light", "wet"),
    "HardRainNoon":   ("rain_heavy", "wet"),
    "ClearSunset":    ("dawn_dusk",  "dry"),
    "WetSunset":      ("dawn_dusk",  "wet"),
    "SoftRainSunset": ("rain_light", "wet"),
    "ClearNight":     ("night",      "dry"),
    "CloudyNight":    ("night",      "dry"),
    "WetNight":       ("night",      "wet"),
}


# ---------------------------------------------------------------------------
# frame index (train + val merged)
# ---------------------------------------------------------------------------

class SplitResolver:
    """Stands in for `img_dir` in export_segment_frames: `resolver / name` returns
    the real path of that frame, whichever split it landed in."""
    def __init__(self, by_name: dict[str, Path]):
        self.by_name = by_name

    def __truediv__(self, name: str) -> Path:
        return self.by_name.get(name, Path("__missing__") / name)


def index_frames(dataset: Path) -> dict[int, Path]:
    idx: dict[int, Path] = {}
    for split in ("train", "val", "test"):
        for p in (dataset / "images" / split).glob("frame_*.png"):
            idx[int(p.stem.split("_")[-1])] = p
    return idx


def load_meta(dataset: Path) -> dict[int, dict]:
    """frames.csv from collect_eval.py: real sim time, preset, traffic episode."""
    f = dataset / "frames.csv"
    if not f.exists():
        return {}
    import csv
    out = {}
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            out[int(r["name"].split("_")[-1])] = {
                "t": float(r["sim_time_s"]), "preset": r["preset"], "episode": int(r["episode"])}
    return out


def preset_for(frame_id: int, frames_per_preset: int) -> str:
    names = cfg.WEATHER_PRESET_NAMES
    return names[min(len(names) - 1, frame_id // frames_per_preset)]


def env_for(preset: str) -> EnvironmentContext:
    weather, surface = PRESET_ENV.get(preset, ("dry_clear", "dry"))
    return EnvironmentContext(weather=weather, road_surface=surface,
                              road_class="motorway", speed_limit_kmh=100,
                              is_built_up_area=False, is_safe_t_cam_zone=True)


# ---------------------------------------------------------------------------
# gate logging (wraps the pipeline without changing its behaviour)
# ---------------------------------------------------------------------------

def _longest_deficit(pipe, obs) -> float:
    """Exact replica of the persistent-deficit computation in _close_pair."""
    below, run_start, prev_t = 0.0, None, None
    for o in obs:
        spd = max(o.follower_speed_kmh / 3.6, 0.1)
        if o.gap_m / spd < pipe.deficit_gap_s:
            if run_start is None:
                run_start = o.t
            prev_t = o.t
        else:
            if run_start is not None:
                below = max(below, prev_t - run_start)
            run_start = None
    if run_start is not None and prev_t is not None:
        below = max(below, prev_t - run_start)
    return below


def instrument(pipe, log: dict):
    """Attach gate logging. log gets: plausibility rejections and one record per
    closed pair episode with its outcome."""
    orig_find = pipe._find_following_pairs
    orig_close = pipe._close_pair

    def find(tracks):
        pairs = orig_find(tracks)
        for f, l in pairs:
            gap = math.hypot(l.last_ground[0] - f.last_ground[0],
                             l.last_ground[1] - f.last_ground[1])
            flen = NOMINAL_LENGTH_M.get(f.cls, 4.6)
            if gap < flen * pipe.min_gap_length_factor:
                log["plausibility_rejected_obs"] += 1
                log["plausibility_rejected_pairs"].add((f.track_id, l.track_id))
        return pairs

    def close(key, force=False):
        obs = list(pipe._pairs.get(key, []))
        rec = {"key": list(key), "n_obs": len(obs),
               "duration_s": round(obs[-1].t - obs[0].t, 2) if len(obs) >= 2 else 0.0,
               "t0": round(obs[0].t, 2) if obs else None}
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
        seg = orig_close(key, force=force)
        rec["segment_id"] = seg.segment_id if seg else None
        log["episodes"].append(rec)
        return seg

    pipe._find_following_pairs = find
    pipe._close_pair = close


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def seg_t0(seg) -> float:
    return int(seg.segment_id.rsplit("_", 1)[-1]) / 100.0


def seg_ep(seg) -> str:
    return seg.segment_id.split("_", 1)[0] if seg.segment_id.startswith("ep") else ""


def flag_duplicates(segments) -> dict[str, str]:
    """Same follower, overlapping in time, different lead -> likely one physical
    lead seen as two tracks (e.g. articulated truck split into cab + trailer)."""
    flags = {}
    for i, a in enumerate(segments):
        for b in segments[i + 1:]:
            fa = a.samples[0].follower_track_id
            fb = b.samples[0].follower_track_id
            if fa != fb or seg_ep(a) != seg_ep(b):
                continue
            a0, b0 = seg_t0(a), seg_t0(b)
            if a0 <= b0 + b.duration_s and b0 <= a0 + a.duration_s:
                flags[a.segment_id] = f"overlaps {b.segment_id} (same follower, different lead)"
                flags[b.segment_id] = f"overlaps {a.segment_id} (same follower, different lead)"
    return flags


def case_label(i: int) -> str:
    letters = string.ascii_uppercase
    return letters[i] if i < 26 else letters[i // 26 - 1] + letters[i % 26]


class UsageMeter:
    """Wraps client.messages.create to record real token usage per call."""
    def __init__(self, agent, price_in, price_out):
        self.calls, self.tin, self.tout = 0, 0, 0
        self.price_in, self.price_out = price_in, price_out
        orig = agent.client.messages.create

        def create(*a, **k):
            r = orig(*a, **k)
            u = getattr(r, "usage", None)
            self.calls += 1
            self.tin += getattr(u, "input_tokens", 0) or 0
            self.tout += getattr(u, "output_tokens", 0) or 0
            return r
        agent.client.messages.create = create

    def cost(self):
        return self.tin / 1e6 * self.price_in + self.tout / 1e6 * self.price_out

    def line(self):
        return (f"{self.calls} API call(s), {self.tin:,} input + {self.tout:,} output tokens "
                f"~= ${self.cost():.2f} at ${self.price_in}/${self.price_out} per M")


def export_with_citations(seg, verdict, frame_boxes, resolver, out_dir, cam, base):
    """Base frames + any frame the agent cited that is not already exported."""
    have = {f["frame_id"] for f in base}
    cited = set()
    if verdict:
        for c in (verdict.get("evidence_assessment", {}) or {}).get("sample_citations", []) or []:
            try:
                cited.add(int(c.get("frame_id")))
            except (TypeError, ValueError):
                pass
    missing = cited - have
    extra = []
    if missing:
        sub = [s for s in seg.samples if s.frame_id in missing]
        if sub:
            extra = export_segment_frames(replace(seg, samples=sub), frame_boxes,
                                          resolver, out_dir, cam, max_frames=len(sub))
            for f in extra:
                f["cited_by_agent"] = True
    for f in base:
        if f["frame_id"] in cited:
            f["cited_by_agent"] = True
    return sorted(base + extra, key=lambda f: f["t"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run(weights, dataset, start_frame, end_frame, conf, frames_per_preset,
        max_cases, include_flushed, call_agent, yes, model, max_tokens,
        price_in, price_out, out, detector=None):
    dataset = Path(dataset)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / "case_frames"

    idx = index_frames(dataset)
    if not idx:
        raise SystemExit(f"No frame_*.png under {dataset}/images/train|val")
    ids = sorted(i for i in idx if i >= start_frame and (end_frame is None or i <= end_frame))
    gaps = [(a, b) for a, b in zip(ids, ids[1:]) if b - a != 1]
    print(f"{len(ids)} frames in range [{ids[0]}..{ids[-1]}] (train+val merged)")
    if gaps:
        print(f"  WARNING: {len(gaps)} gap(s) in frame ids, e.g. {gaps[:3]} - "
              "tracking across a gap may break pairs.")
    resolver = SplitResolver({p.name: p for p in idx.values()})

    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z,
        calibration_rmse_m=cfg.CALIBRATION_RMSE_M,
        max_reliable_range_m=cfg.MAX_RELIABLE_RANGE_M)

    meta = load_meta(dataset)
    if meta:
        print(f"  frames.csv found: using recorded sim time, presets and "
              f"{len({m['episode'] for m in meta.values()})} traffic episode(s)")
    t_of = (lambda fid: meta[fid]["t"]) if meta else (lambda fid: fid * SIM_DT)
    preset_of = ((lambda fid: meta[fid]["preset"]) if meta
                 else (lambda fid: preset_for(fid, frames_per_preset)))
    ep_of = (lambda fid: meta[fid]["episode"]) if meta else (lambda fid: 0)

    if detector is None:
        from ultralytics import YOLO
        detector = YOLO(weights)

    log = {"plausibility_rejected_obs": 0, "plausibility_rejected_pairs": set(),
           "episodes": []}
    frame_boxes: dict[int, dict] = {}
    segments = []
    episodes = sorted({ep_of(f) for f in ids})
    print(f"Stage 3 (conf={conf}) ...")
    for ep in episodes:
        ep_ids = [f for f in ids if ep_of(f) == ep]
        # fresh pipeline per traffic episode: re-spawned traffic must not inherit
        # tracks or open pairs from the previous episode
        pipe = TrafficCamPerceptionPipeline(camera=cam, yolo_weights=weights,
                                            conf_threshold=conf, detector=detector)
        instrument(pipe, log)
        orig_update = pipe.tracker.update

        def logged_update(dets, t, frame_id, _u=orig_update):
            tracks = _u(dets, t, frame_id)
            frame_boxes[frame_id] = {tid: (tr.last_bbox, tr.cls) for tid, tr in tracks.items()}
            return tracks
        pipe.tracker.update = logged_update

        def stream(ep_ids=ep_ids):
            for n, fid in enumerate(ep_ids):
                if fid % 250 == 0:
                    print(f"  frame {fid}  preset={preset_of(fid)}  episode={ep}")
                yield FrameInput(frame_id=fid, timestamp_s=t_of(fid),
                                 rgb_image=load_rgb(idx[fid]),
                                 environment=env_for(preset_of(fid)))

        for seg in pipe.process_stream(stream()):
            if len(episodes) > 1:
                old = seg.segment_id
                seg.segment_id = f"ep{ep:02d}_{old}"
                for e in log["episodes"]:
                    if e.get("segment_id") == old and e.get("ep") is None:
                        e["segment_id"] = seg.segment_id
            segments.append(seg)
        for e in log["episodes"]:
            e.setdefault("ep", ep)

    outcomes = Counter(e["outcome"] for e in log["episodes"])
    dup_flags = flag_duplicates(segments)

    # selection: gated segments (longest first), then optionally flushed ones
    passed_ids = {e["segment_id"] for e in log["episodes"] if e["outcome"] == "passed"}
    gated = sorted([s for s in segments if s.segment_id in passed_ids],
                   key=lambda s: s.duration_s, reverse=True)
    flushed = [s for s in segments if s.segment_id not in passed_ids]
    chosen = gated + (flushed if include_flushed else [])
    chosen = chosen[:max_cases]

    agent = meter = None
    if call_agent and chosen:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set.")
        est = len(chosen) * (7000 / 1e6 * price_in + 4500 / 1e6 * price_out)
        print(f"\nAbout to call {model} on {len(chosen)} case(s): est. ~${est:.2f} "
              f"(assumes ~7k in / ~4.5k out tokens per call at ${price_in}/${price_out} per M).")
        if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("Aborted - no API calls made.")
        agent = TailgatingReasoningAgent(model=model, max_tokens=max_tokens)
        meter = UsageMeter(agent, price_in, price_out)

    cases, rows = [], []
    for i, seg in enumerate(chosen):
        label = case_label(i)
        start_fid = seg.samples[0].frame_id
        preset = preset_of(start_fid)
        base = export_segment_frames(seg, frame_boxes, resolver, frames_dir, cam)
        verdict = None
        if agent is not None:
            print(f"  [{label}] reasoning on {seg.segment_id} ...")
            try:
                res = agent.analyze(seg)
                verdict = {"verdict": res.verdict, "confidence": res.confidence,
                           "headline": res.headline,
                           **{k: v for k, v in res.raw.items()
                              if k not in ("verdict", "confidence", "headline")}}
            except Exception as e:              # keep going; record the failure
                print(f"      agent failed: {e}")
                verdict = None
        frames = export_with_citations(seg, verdict, frame_boxes, resolver,
                                       frames_dir, cam, base)
        a = seg.aggregates()
        gap3 = a.get("sustained_below_3s_seconds", 0) or 0
        clr3 = a.get("sustained_clearance_below_3s_seconds", 0) or 0
        flags = []
        if seg.segment_id not in passed_ids:
            flags.append("not gated (end-of-stream flush)")
        if seg.segment_id in dup_flags:
            flags.append(dup_flags[seg.segment_id])
        cases.append({
            "case_id": seg.segment_id, "label": label, "status": "pending_review",
            "carla_preset": preset, "start_frame": start_fid,
            "flags": flags, "evidence": seg.evidence_brief(),
            "frames": frames, "verdict": verdict,
        })
        rows.append({
            "label": label, "segment_id": seg.segment_id,
            "pair": f"{seg.follower_vehicle_class} behind {seg.lead_vehicle_class}",
            "preset": preset, "frames": f"{start_fid}-{seg.samples[-1].frame_id}",
            "duration_s": round(seg.duration_s, 2),
            "mean_speed_kmh": a.get("mean_follower_speed_kmh"),
            "gap_below3_s": gap3, "clear_below3_s": clr3,
            # prompt heuristic: >=5 s below 3 s  OR  >=3 s below 2 s
            "heuristic_gap": gap3 >= TRANSIENCY_S or (a.get("sustained_below_2s_seconds") or 0) >= 3.0,
            "heuristic_clear": clr3 >= TRANSIENCY_S or (a.get("sustained_clearance_below_2s_seconds") or 0) >= 3.0,
            "verdict": (verdict or {}).get("verdict", "-"),
            "confidence": (verdict or {}).get("confidence", "-"),
            "flags": "; ".join(flags),
        })

    summary = {
        "frames_processed": len(ids),
        "frame_range": [ids[0], ids[-1]],
        "conf_threshold": conf,
        "pair_episodes": len(log["episodes"]),
        "outcomes": dict(outcomes),
        "plausibility_rejected_observations": log["plausibility_rejected_obs"],
        "plausibility_rejected_pairs": len(log["plausibility_rejected_pairs"]),
        "segments_emitted": len(segments),
        "cases_written": len(cases),
        "convention_disagreements": sum(r["heuristic_gap"] != r["heuristic_clear"] for r in rows),
        "usage": meter.line() if meter else "no API calls (dry run)",
        "episodes": log["episodes"],
    }
    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site_id": "TOWN04_HW_01",
        "model": model if agent else None,
        "temperature": None,
        "reproducible": False,
        "batch_summary": summary,
        "cases": cases,
    }
    out.write_text(json.dumps(doc, indent=2))

    # human-readable summary
    L = []
    L.append(f"Batch run {doc['generated_utc']}  frames {ids[0]}..{ids[-1]} ({len(ids)})  conf={conf}")
    L.append("")
    L.append("GATE COUNTS (pair episodes = follower/lead pair tracked until lost)")
    L.append(f"  pair episodes:                 {summary['pair_episodes']}")
    L.append("  (force_flushed = pair still open when the stream or a traffic episode ended)")
    for k in ("too_few_obs", "failed_duration_gate", "failed_persistence_gate",
              "passed", "force_flushed_end_of_stream"):
        L.append(f"  {k:30s} {outcomes.get(k, 0)}")
    L.append(f"  plausibility rejections:       {summary['plausibility_rejected_observations']} "
             f"observation(s) across {summary['plausibility_rejected_pairs']} pair(s)")
    L.append(f"  segments emitted:              {len(segments)}")
    L.append(f"  convention disagreements:      {summary['convention_disagreements']} of {len(rows)} "
             f"(heuristic: >=5 s below 3 s or >=3 s below 2 s; met under one convention only)")
    L.append("")
    L.append("CASES")
    hdr = (f"{'case':4s} {'pair':38s} {'preset':15s} {'frames':11s} {'dur':>5s} {'v_kmh':>6s} "
           f"{'gap<3':>6s} {'clr<3':>6s} {'heur g/c':>10s}  verdict")
    L.append(hdr)
    for r in rows:
        L.append(f"{r['label']:4s} {r['pair'][:38]:38s} {r['preset']:15s} {r['frames']:11s} "
                 f"{r['duration_s']:5.2f} {r['mean_speed_kmh'] or 0:6.1f} "
                 f"{r['gap_below3_s']:6.2f} {r['clear_below3_s']:6.2f} "
                 f"{('Y' if r['heuristic_gap'] else 'N') + '/' + ('Y' if r['heuristic_clear'] else 'N'):>10s}  "
                 f"{r['verdict']} ({r['confidence']})" + (f"   [{r['flags']}]" if r["flags"] else ""))
    L.append("")
    L.append("Segment IDs:")
    for r in rows:
        L.append(f"  {r['label']}: {r['segment_id']}")
    L.append("")
    L.append(f"API: {summary['usage']}")
    txt = "\n".join(L)
    (out.parent / "batch_summary.txt").write_text(txt)
    print("\n" + txt)
    print(f"\nWrote {out.resolve()}\nWrote {(out.parent / 'batch_summary.txt').resolve()}"
          f"\nFrames in {frames_dir.resolve()}")
    return doc


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--conf", type=float, default=0.4,
                   help="detector confidence threshold (pipeline default 0.4)")
    p.add_argument("--frames-per-preset", type=int, default=250,
                   help="collection cycled presets in blocks of this many frames")
    p.add_argument("--max-cases", type=int, default=10)
    p.add_argument("--include-flushed", action="store_true",
                   help="also write ungated end-of-stream segments as cases")
    p.add_argument("--call-agent", action="store_true",
                   help="send cases to the reasoning agent (costs money). Default: dry run")
    p.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--price-in", type=float, default=3.0, help="USD per M input tokens (check console)")
    p.add_argument("--price-out", type=float, default=15.0, help="USD per M output tokens (check console)")
    p.add_argument("--out", default="batch/batch_cases.json")
    a = p.parse_args()
    run(a.weights, a.dataset, a.start_frame, a.end_frame, a.conf, a.frames_per_preset,
        a.max_cases, a.include_flushed, a.call_agent, a.yes, a.model, a.max_tokens,
        a.price_in, a.price_out, a.out)