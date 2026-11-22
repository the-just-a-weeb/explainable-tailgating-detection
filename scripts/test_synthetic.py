"""
Synthetic end-to-end test (no CARLA, no GPU)
============================================

Exercises two things the live path can't easily unit-test:

  1. The perception pipeline's geometry + pairing + speed-estimation + segment
     gating, using a FAKE detector that emits known vehicle ground-truth boxes
     (so you can confirm Stages 2-3 independent of YOLO accuracy).

  2. The reasoning agent on the resulting FollowingSegment (needs ANTHROPIC_API_KEY;
     pass --no-agent to skip and just print the evidence brief).

    python scripts/test_synthetic.py
    python scripts/test_synthetic.py --no-agent     # geometry + schema only
"""

from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import perception_pipeline as pp  # noqa
from perception_pipeline import FrameInput, TrafficCamPerceptionPipeline  # noqa
from reasoning_agent import EnvironmentContext  # noqa
import site_config as cfg  # noqa


class FakeBox:
    def __init__(self, xyxy, cls_id, conf):
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.cls = [np.array([cls_id], dtype=float)]
        self.conf = [np.array([conf], dtype=float)]


class FakeResult:
    def __init__(self, boxes): self.boxes = boxes


class FakeDetector:
    """Stands in for YOLO. Returns boxes from a per-frame scripted scene by
    projecting known world vehicles through the real camera geometry."""
    def __init__(self, camera: TrafficCamera, scene_fn):
        self.camera = camera
        self.scene_fn = scene_fn
        self.frame = 0

    def predict(self, image, verbose=False, conf=0.4):
        vehicles = self.scene_fn(self.frame)
        self.frame += 1
        boxes = []
        for (wx, wy, cls_id, length) in vehicles:
            # Bottom-centre of the bbox must sit at the vehicle's ground contact
            # point (wx, wy, 0) so that the pipeline's bottom-centre->ground
            # projection recovers the true position - exactly as a real YOLO box
            # bottom edge does.
            bcx, bcy, d = self.camera.world_to_pixel(wx, wy, 0.0)      # ground contact
            topx, topy, dt = self.camera.world_to_pixel(wx, wy, 1.6)   # roof
            sidex, sidey, ds = self.camera.world_to_pixel(wx, wy + 1.1, 0.0)  # half-width
            if any(math.isnan(v) for v in (bcx, bcy, topx, sidex)):
                continue
            half_w = abs(bcx - sidex)
            x1, x2 = bcx - half_w, bcx + half_w
            y1, y2 = min(topy, bcy), bcy        # bottom edge y2 == ground-contact py
            boxes.append(FakeBox([x1, y1, x2, y2], cls_id, 0.9))
        return [FakeResult(boxes)]


def make_camera():
    intr = CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG)
    pose = CameraPose((0.0, 0.0, 6.0), (-8.0, 0.0, 0.0))  # simple +X-looking pose
    cam = TrafficCamera(intr, pose, ground_z=0.0)
    cam.calibration_rmse_m = 0.5
    cam.max_reliable_range_m = cfg.MAX_RELIABLE_RANGE_M
    return cam


def sustained_tailgating_scene(frame):
    """Car closely following an articulated truck, ~0.8s headway, 20Hz.
    Truck (cls 7) ahead, car (cls 2) behind, both at 100 km/h, gap ~22m."""
    t = frame / cfg.SIM_HZ
    speed_ms = 100 / 3.6
    lead_x = 40 + speed_ms * t
    gap = 22.0 + math.sin(t / 2) * 1.0
    fol_x = lead_x - gap
    return [
        (lead_x, 0.0, 7, 16.0),   # articulated truck
        (fol_x, 0.0, 2, 4.6),     # car
    ]


def run(no_agent: bool):
    cam = make_camera()
    fake = FakeDetector(cam, sustained_tailgating_scene)
    pipe = TrafficCamPerceptionPipeline(camera=cam, yolo_weights="unused",
                                        lane_half_width_m=cfg.LANE_HALF_WIDTH_M,
                                        min_segment_duration_s=3.0,
                                        detector=fake)

    env = EnvironmentContext("rain_light", "wet", "motorway", 110, False, False)
    frames = [FrameInput(1000 + i, i / cfg.SIM_HZ, np.zeros((cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3), np.uint8), env)
              for i in range(200)]   # 10 s

    segments = list(pipe.process_stream(frames))
    print(f"Segments emitted: {len(segments)}")
    if not segments:
        print("No segment passed the persistent-deficit gate. (Unexpected for this scene.)")
        return
    seg = segments[0]
    print(f"\nSegment {seg.segment_id}: {seg.follower_vehicle_class} behind {seg.lead_vehicle_class}, {seg.duration_s:.1f}s")
    print("\nAggregates:")
    print(json.dumps(seg.aggregates(), indent=2))

    if no_agent:
        print("\n(--no-agent) Evidence brief sample count:",
              len(seg.evidence_brief()["representative_samples"]))
        return

    from reasoning_agent import TailgatingReasoningAgent, format_result_for_log
    agent = TailgatingReasoningAgent()
    result = agent.analyze(seg)
    print("\n" + format_result_for_log(result))
    assert result.verdict in ("likely_breach", "possible_breach"), \
        f"expected a breach finding for 0.8s sustained headway, got {result.verdict}"
    print("\nOK: sustained 0.8s headby behind a truck on a wet motorway -> breach finding.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-agent", action="store_true")
    a = p.parse_args()
    run(a.no_agent)
