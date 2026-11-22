"""
Stage 4 test  (real Stage-3 segment -> LLM legal reasoning -> verdict)
=====================================================================

Chains the whole back half of the pipeline: runs Stage 3 over a sequence to
produce real FollowingSegments, picks the most significant one (longest sustained
close-follow), and passes it to the TailgatingReasoningAgent, which applies NSW
Rule 126/127 and returns an auditable, rule-cited verdict. This is the
"de-black-boxing" payoff - not a score, but reasoning a human can check.

    python scripts/test_stage4_reasoning.py --weights weights/best.pt \
        --dataset ./dataset --start 0 --count 200

Requires ANTHROPIC_API_KEY in the environment (the agent calls Claude).
Use --dry-run to skip the API call and just show the evidence brief that would
be sent (useful to sanity-check the payload without spending tokens).
"""

from __future__ import annotations
import argparse
import json
import os
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
from reasoning_agent import (TailgatingReasoningAgent, EnvironmentContext,  # noqa
                             format_result_for_log)

SIM_DT = 1.0 / cfg.SIM_HZ


def env_for_night(is_night: bool) -> EnvironmentContext:
    return EnvironmentContext(
        weather="night" if is_night else "dry_clear",
        road_surface="dry", road_class="motorway",
        speed_limit_kmh=100, is_built_up_area=False, is_safe_t_cam_zone=True)


def main(weights, dataset, split, start, count, conf, dry_run, model, max_tokens):
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
        raise SystemExit(f"No frames in window [{start}:{start+count}]")

    def stream():
        for i, p in enumerate(window):
            fid = int(p.stem.split("_")[-1])
            img = mpimg.imread(str(p))
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            yield FrameInput(frame_id=fid, timestamp_s=i * SIM_DT,
                             rgb_image=np.ascontiguousarray(img),
                             environment=env_for_night(False))

    print(f"Running Stage 3 over {len(window)} frames to extract segments...")
    segments = list(pipe.process_stream(stream()))
    if not segments:
        raise SystemExit("No following segments found in this window - try a "
                         "different --start or a window with tailgating.")

    # Pick the most significant: longest duration (most sustained tailgating)
    segments.sort(key=lambda s: s.duration_s, reverse=True)
    seg = segments[0]
    print(f"\nSelected segment: {seg.segment_id}")
    print(f"  {seg.follower_vehicle_class} following {seg.lead_vehicle_class}, "
          f"{seg.duration_s:.1f}s, {len(seg.samples)} samples")

    brief = seg.evidence_brief()
    print("\n--- Evidence brief (what the agent receives) ---")
    print(json.dumps(brief, indent=2)[:2000])

    if dry_run:
        print("\n[--dry-run] Skipping the API call. Above is the payload that "
              "would be sent to the reasoning agent.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set - export it or use --dry-run.")

    print(f"\nCalling reasoning agent ({model})...")
    agent = TailgatingReasoningAgent(model=model, max_tokens=max_tokens)
    result = agent.analyze(seg)
    print("\n" + format_result_for_log(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--dry-run", action="store_true", help="show payload, skip API call")
    p.add_argument("--max-tokens", type=int, default=8000,
                   help="raise if the verdict comes back truncated")
    p.add_argument("--model", default="claude-sonnet-5",
                   help="reasoning model (update to a current model string)")
    a = p.parse_args()
    main(a.weights, a.dataset, a.split, a.start, a.count, a.conf, a.dry_run,
         a.model, a.max_tokens)
