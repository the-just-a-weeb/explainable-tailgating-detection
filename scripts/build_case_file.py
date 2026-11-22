"""
Stage 5 feed  —  generate verdicts and export them as JSON for the dashboard
============================================================================

Runs Stage 3 over a frame window to extract following segments, sends each to the
reasoning agent (temperature=0, so re-running gives identical verdicts), and
writes a single JSON file the adjudication dashboard loads.

    # dry run - segments only, no API calls, no cost
    python scripts/build_case_file.py --weights weights/best.pt --dry-run

    # real verdicts
    python scripts/build_case_file.py --weights weights/best.pt --max-cases 3

Output: cases.json  — a list of cases, each containing the frozen evidence brief
and the agent's verdict, reasoning chain, caveats and counterfactual.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

import site_config as cfg  # noqa
from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
from perception_pipeline import TrafficCamPerceptionPipeline, FrameInput  # noqa
from reasoning_agent import TailgatingReasoningAgent, EnvironmentContext  # noqa

SIM_DT = 1.0 / cfg.SIM_HZ


def load_rgb(path: Path) -> np.ndarray:
    img = mpimg.imread(str(path))
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    elif img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return np.ascontiguousarray(img)



def export_segment_frames(seg, frame_boxes, img_dir: Path, out_dir: Path,
                          cam, max_frames: int = 12) -> list[dict]:
    """Write review frames: camera view (left) + bird's-eye homography projection
    (right), with the follower and lead highlighted in both.

    Two panels because they answer different questions. The camera view lets the
    reviewer confirm the system tracked the *right two vehicles*. The BEV lets them
    read the *actual spacing* - at 90 m, perspective compresses 16 m into a few
    pixels, so distance is not judgeable from the camera frame alone.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = seg.samples
    if not samples:
        return []
    step = max(1, len(samples) // max_frames)
    picked = samples[::step][:max_frames]
    # track ids live on the samples, not the segment
    f_id, l_id = samples[0].follower_track_id, samples[0].lead_track_id

    yaw = math.radians(cfg.CAMERA_ROTATION[1])
    fwd = (math.cos(yaw), math.sin(yaw))
    ox, oy, _ = cfg.CAMERA_LOCATION

    def road_frame(gx, gy):
        dx, dy = gx - ox, gy - oy
        return dx * fwd[0] + dy * fwd[1], -dx * fwd[1] + dy * fwd[0]

    lane_hw = cfg.LANE_HALF_WIDTH_M
    manifest = []
    for smp in picked:
        src = img_dir / f"frame_{smp.frame_id:08d}.png"
        if not src.exists():
            continue
        fig, (ax_img, ax_bev) = plt.subplots(
            1, 2, figsize=(16, 6.4), gridspec_kw={"width_ratios": [1.55, 1]})

        ax_img.imshow(mpimg.imread(str(src)))
        pair_pts = {}
        labels_to_place = []          # (x, y, text, colour) - placed after boxes
        for tid, (bbox, cls) in frame_boxes.get(smp.frame_id, {}).items():
            x1, y1, x2, y2 = bbox
            if tid == f_id:
                col, lab, lw = "#e8433d", "FOLLOWER", 2.6
            elif tid == l_id:
                col, lab, lw = "#2f9e44", "LEAD", 2.6
            else:
                col, lab, lw = "#8A94A6", "", 1.0
            ax_img.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                               fill=False, edgecolor=col, linewidth=lw))
            gx, gy = cam.pixel_to_ground((x1 + x2) / 2.0, y2)
            if not math.isnan(gx):
                d, lat = road_frame(gx, gy)
                if lab:
                    pair_pts[lab] = (lat, d, col, cls)
                    # FOLLOWER label goes BELOW its box, LEAD above, so the two
                    # never collide when the vehicles are close in image space
                    # (at range the boxes are only a few pixels apart).
                    ly = (y2 + 14) if lab == "FOLLOWER" else (y1 - 6)
                    labels_to_place.append((x1, ly, f"{lab} {cls} {d:.0f}m", col))
        for lx, ly, txt, col in labels_to_place:
            ax_img.text(lx, ly, txt, color=col, fontsize=9, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                  edgecolor="none", alpha=0.72))
        ax_img.set_title(f"Camera view - frame {smp.frame_id}", fontsize=11)
        ax_img.axis("off")
        ax_img.text(0.012, 0.982,
                    f"t = {smp.t:.2f}s\n"
                    f"gap = {smp.gap_m:.1f} m  ({smp.gap_s:.2f} s headway)\n"
                    f"follower {smp.follower_speed_kmh:.0f} km/h   "
                    f"lead {smp.lead_speed_kmh:.0f} km/h",
                    transform=ax_img.transAxes, va="top", ha="left", fontsize=10.5,
                    color="white", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="#1B2432", alpha=0.85))

        # --- bird's-eye panel: metric, so the gap is directly readable ---
        max_r = cfg.MAX_LABEL_RANGE_M
        for d in range(0, int(max_r) + 1, 10):
            ax_bev.axhline(d, color="#DCE2E8", lw=0.8, zorder=0)
            ax_bev.text(-lane_hw * 4.6, d + 0.6, f"{d} m", fontsize=7.5,
                        color="#9AA5B1", zorder=1)
        for lx in (-lane_hw, lane_hw):
            ax_bev.axvline(lx, color="#C3CBD6", lw=1.0, ls="--", zorder=0)
        ax_bev.plot(0, 0, "^", color="#1B2432", markersize=13, zorder=4)
        ax_bev.text(0, -4.5, "camera", ha="center", fontsize=9, weight="bold",
                    color="#1B2432")
        # offset BEV labels vertically when the two markers nearly coincide
        _ds = [v[1] for v in pair_pts.values()]
        _close = len(_ds) == 2 and abs(_ds[0] - _ds[1]) < 6.0
        for lab, (lat, d, col, cls) in pair_pts.items():
            ax_bev.plot(lat, d, "o", color=col, markersize=11,
                        markeredgecolor="white", markeredgewidth=1.2, zorder=5)
            dy = 0.0
            if _close:
                dy = 3.0 if lab == "LEAD" else -3.0
            ax_bev.text(lat + lane_hw * 0.5, d + dy, f"{lab}\n{cls}", fontsize=8.5,
                        weight="bold", color=col, va="center", zorder=5)
        if {"FOLLOWER", "LEAD"} <= set(pair_pts):
            (flat, fd, *_), (llat, ld, *_) = pair_pts["FOLLOWER"], pair_pts["LEAD"]
            ax_bev.annotate("", xy=(llat, ld), xytext=(flat, fd),
                            arrowprops=dict(arrowstyle="<->", color="#C92A2A", lw=1.8))
            ax_bev.text(min(flat, llat) - lane_hw * 0.35, (fd + ld) / 2,
                        f"{smp.gap_m:.1f} m", color="#C92A2A", fontsize=11,
                        weight="bold", va="center", ha="right",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor="none", alpha=0.8))
        ax_bev.set_xlim(-lane_hw * 5, lane_hw * 5)
        ax_bev.set_ylim(-8, max_r + 5)
        ax_bev.set_xlabel("lateral offset (m)", fontsize=9)
        ax_bev.set_ylabel("down-range (m)", fontsize=9)
        ax_bev.set_title("Bird's-eye (homography) - true metric spacing", fontsize=11)
        ax_bev.tick_params(labelsize=8)

        fig.tight_layout(pad=0.6)
        name = f"{seg.segment_id}_f{smp.frame_id:08d}.jpg"
        fig.savefig(out_dir / name, dpi=100, bbox_inches="tight")
        plt.close(fig)
        manifest.append({
            "file": f"{out_dir.name}/{name}",
            "frame_id": smp.frame_id,
            "t": round(smp.t, 3),
            "gap_m": round(smp.gap_m, 2),
            "gap_s": round(smp.gap_s, 3),
        })
    return manifest


def main(weights, dataset, split, start, count, conf, max_cases,
         model, max_tokens, dry_run, out, temperature):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z,
        calibration_rmse_m=cfg.CALIBRATION_RMSE_M,
        max_reliable_range_m=cfg.MAX_RELIABLE_RANGE_M)
    pipe = TrafficCamPerceptionPipeline(camera=cam, yolo_weights=weights,
                                        conf_threshold=conf)

    img_dir = Path(dataset) / "images" / split
    window = sorted(img_dir.glob("*.png"))[start:start + count]
    if not window:
        raise SystemExit(f"No frames in window [{start}:{start+count}] of {img_dir}")

    env = EnvironmentContext(
        weather="dry_clear", road_surface="dry", road_class="motorway",
        speed_limit_kmh=100, is_built_up_area=False, is_safe_t_cam_zone=True)

    def stream():
        for i, p in enumerate(window):
            yield FrameInput(frame_id=int(p.stem.split("_")[-1]),
                             timestamp_s=i * SIM_DT,
                             rgb_image=load_rgb(p), environment=env)

    print(f"Stage 3 over {len(window)} frames from {window[0].stem} ...")
    # Capture track_id -> bbox per frame so the reviewer can see the actual
    # vehicles, not just the numbers. HITL review of a verdict is incomplete if
    # the human cannot check that the system tracked the right two vehicles.
    frame_boxes: dict[int, dict[str, tuple]] = {}
    _orig_update = pipe.tracker.update

    def _logged_update(dets, t, frame_id):
        tracks = _orig_update(dets, t, frame_id)
        frame_boxes[frame_id] = {tid: (tr.last_bbox, tr.cls)
                                 for tid, tr in tracks.items()}
        return tracks

    pipe.tracker.update = _logged_update
    segments = list(pipe.process_stream(stream()))
    pipe.tracker.update = _orig_update
    if not segments:
        raise SystemExit("No following segments in this window.")

    # longest (most sustained) first - those are the substantive cases
    segments.sort(key=lambda s: s.duration_s, reverse=True)
    segments = segments[:max_cases]
    print(f"  {len(segments)} segment(s) selected for adjudication:")
    for s in segments:
        print(f"    {s.segment_id}: {s.follower_vehicle_class} following "
              f"{s.lead_vehicle_class}, {s.duration_s:.1f}s")

    agent = None
    if not dry_run:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set - export it or use --dry-run.")
        # temperature is only sent when --temperature is given; it is deprecated
        # on newer models and sending it raises a 400.
        agent = TailgatingReasoningAgent(model=model, max_tokens=max_tokens,
                                         temperature=temperature)

    cases = []
    for i, seg in enumerate(segments, 1):
        brief = seg.evidence_brief()
        frames = export_segment_frames(seg, frame_boxes, img_dir,
                                       Path(out).parent / "case_frames", cam)
        case = {
            "case_id": seg.segment_id,
            "status": "pending_review",
            "evidence": brief,
            "frames": frames,
            "verdict": None,
        }
        print(f"    exported {len(frames)} annotated frame(s)")
        if agent is not None:
            print(f"  [{i}/{len(segments)}] reasoning on {seg.segment_id} ...")
            res = agent.analyze(seg)
            case["verdict"] = {
                "verdict": res.verdict,
                "confidence": res.confidence,
                "headline": res.headline,
                **{k: v for k, v in res.raw.items()
                   if k not in ("verdict", "confidence", "headline")},
            }
        cases.append(case)

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "site_id": brief["camera"]["site_id"],
        "model": None if dry_run else model,
        "temperature": None if dry_run else temperature,
        "reproducible": (not dry_run) and temperature == 0.0,
        "cases": cases,
    }
    Path(out).write_text(json.dumps(doc, indent=2))
    frames_dir = (Path(out).parent / "case_frames").resolve()
    n_frames = sum(len(c["frames"]) for c in cases)
    print(f"\nWrote {Path(out).resolve()}  ({len(cases)} case(s))")
    print(f"Wrote {n_frames} review frame(s) to {frames_dir}")
    if dry_run:
        print("[--dry-run] evidence only, no verdicts. Re-run without --dry-run "
              "to populate them.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--max-cases", type=int, default=3,
                   help="how many segments to adjudicate (longest first)")
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-tokens", type=int, default=8000)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--temperature", type=float, default=None,
                   help="omit on models where temperature is deprecated")
    p.add_argument("--out", default="cases.json")
    a = p.parse_args()
    main(a.weights, a.dataset, a.split, a.start, a.count, a.conf, a.max_cases,
         a.model, a.max_tokens, a.dry_run, a.out, a.temperature)