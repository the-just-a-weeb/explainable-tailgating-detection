"""
Stage 3 sequence test  (tracking + TTC filtration on consecutive frames)
=========================================================================

Stages 1-2 proved detection + homography give correct per-frame ground
positions. Stage 3 is the first TEMPORAL stage: it tracks vehicles across frames,
estimates speeds from ground displacement, finds follower->lead pairs, computes
time-headway (gap_s) and time-to-collision (ttc_s), and applies the
persistent-deficit gate (sustained close-following, not transient flickers).

This runs the real trained detector over a SEQUENCE of consecutive dataset
frames and reports what the temporal logic produces, so we can sanity-check:
  - tracks persist across frames (IDs stable, not churning every frame)
  - speeds are physically plausible (not 0, not 500 km/h)
  - gaps/TTC are sane for the close-following pairs
  - the persistent-deficit gate emits FollowingSegments for sustained tailgating

    python scripts/test_stage3_sequence.py --weights weights/best.pt \
        --dataset ./dataset --start 20 --count 60

Pick a --start where the busy/tailgating frames are (consecutive frame numbers
are 20 Hz apart within the run, which is what tracking needs).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

import site_config as cfg  # noqa
from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
from perception_pipeline import TrafficCamPerceptionPipeline, FrameInput  # noqa
from reasoning_agent import EnvironmentContext  # noqa

SIM_DT = 1.0 / cfg.SIM_HZ   # seconds between consecutive frames (20 Hz -> 0.05)


def default_env() -> EnvironmentContext:
    # Neutral environment for the geometry/tracking test; real runs would set
    # this from the frame's weather. Motorway, dry, not built-up.
    return EnvironmentContext(
        weather="dry_clear", road_surface="dry", road_class="motorway",
        speed_limit_kmh=100, is_built_up_area=False, is_safe_t_cam_zone=True)


def main(weights, dataset, split, start, count, conf):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)

    pipe = TrafficCamPerceptionPipeline(camera=cam, yolo_weights=weights,
                                        conf_threshold=conf)

    img_dir = Path(dataset) / "images" / split
    imgs = sorted(img_dir.glob("*.png"))
    # take a consecutive window by frame index
    window = imgs[start:start + count]
    if not window:
        raise SystemExit(f"No frames in window [{start}:{start+count}] of {img_dir}")
    print(f"Running {len(window)} consecutive frames from {window[0].stem} "
          f"at {cfg.SIM_HZ} Hz (dt={SIM_DT:.3f}s)\n")

    # Diagnostic: are these frames actually consecutive in real frame-id terms?
    fids = [int(p.stem.split("_")[-1]) for p in window]
    gaps = [fids[i+1] - fids[i] for i in range(len(fids) - 1)]
    n_gap = sum(1 for g in gaps if g != 1)
    if n_gap:
        print(f"  WARNING: {n_gap}/{len(gaps)} frame steps are NOT +1 "
              f"(val-split removed frames). Max gap={max(gaps)}. Tracking needs")
        print(f"  truly consecutive frames - results here may show broken tracks.")
        print(f"  Use a continuous --sequence collection for a clean Stage 3 test.\n")
    else:
        print(f"  frames are consecutive (+1 steps) - good for tracking.\n")

    def frame_stream():
        for i, p in enumerate(window):
            # frame_id from the filename so IoU/age logic sees consecutive ids
            fid = int(p.stem.split("_")[-1])
            img = mpimg.imread(str(p))
            # normalise to uint8 RGB (drop alpha if present; PNGs load as RGBA)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]          # RGBA -> RGB
            elif img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)   # grayscale -> RGB
            rgb = np.ascontiguousarray(img)
            yield FrameInput(frame_id=fid, timestamp_s=i * SIM_DT,
                             rgb_image=rgb, environment=default_env())

    # Collect emitted segments and also peek at per-frame tracking via the tracker.
    # Instrument: wrap the tracker to log detections and track counts per frame.
    orig_update = pipe.tracker.update
    frame_log = []
    def logged_update(detections, t, frame_id):
        tracks = orig_update(detections, t, frame_id)
        frame_log.append((frame_id, len(detections), len(tracks)))
        return tracks
    pipe.tracker.update = logged_update

    segments = list(pipe.process_stream(frame_stream()))

    # Show the first/last few frames' detection & track counts
    print("--- Per-frame detections / tracks (first 10, last 5) ---")
    for fid, nd, nt in frame_log[:10]:
        print(f"    frame {fid}: {nd} detections -> {nt} active tracks")
    if len(frame_log) > 15:
        print("    ...")
        for fid, nd, nt in frame_log[-5:]:
            print(f"    frame {fid}: {nd} detections -> {nt} active tracks")
    total_det = sum(nd for _, nd, _ in frame_log)
    max_tracks = max((nt for _, _, nt in frame_log), default=0)
    print(f"  totals: {total_det} detections across {len(frame_log)} frames, "
          f"peak active tracks={max_tracks}\n")

    # Report tracker state after the run
    tracks = pipe.tracker.tracks
    print(f"--- Tracking summary ---")
    print(f"  active tracks at end: {len(tracks)}")
    for tid, tr in list(tracks.items())[:12]:
        print(f"    {tid}: cls={tr.cls:12s} speed={tr.speed_kmh:5.1f} km/h "
              f"ground=({tr.last_ground[0]:.1f},{tr.last_ground[1]:.1f})m "
              f"seen_frames={len(tr.ground_hist)}")

    print(f"\n--- Following segments emitted (persistent-deficit gate) ---")
    if not segments:
        print("  none - either no sustained close-following in this window, or")
        print("  the window is too short for the deficit_min_seconds gate.")
    for s in segments:
        gaps = [fs.gap_s for fs in s.samples if fs.gap_s > 0]
        ttcs = [fs.ttc_s for fs in s.samples if fs.ttc_s and fs.ttc_s > 0]
        min_gap_s = min(gaps) if gaps else float("nan")
        min_ttc = min(ttcs) if ttcs else float("nan")
        print(f"  {s.segment_id}: {s.follower_vehicle_class} following "
              f"{s.lead_vehicle_class}, {s.duration_s:.1f}s, "
              f"{len(s.samples)} samples, min gap_s={min_gap_s:.2f}s, "
              f"min TTC={min_ttc:.1f}s")

    print(f"\n--- Sanity checks ---")
    speeds = [tr.speed_kmh for tr in tracks.values()]
    if speeds:
        print(f"  speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h "
              f"({'OK' if all(0 <= s < 200 for s in speeds) else 'SUSPICIOUS - check tracking'})")
    print(f"  segments emitted: {len(segments)}")
    print(f"  (a healthy result: stable tracks, speeds ~40-110 km/h on a motorway,")
    print(f"   and >=1 segment if the window contains sustained tailgating)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--start", type=int, default=20, help="index into sorted frames")
    p.add_argument("--count", type=int, default=60, help="consecutive frames to run")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    main(a.weights, a.dataset, a.split, a.start, a.count, a.conf)
