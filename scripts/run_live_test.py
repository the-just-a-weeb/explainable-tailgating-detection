"""
Live test: CARLA Town04 fixed camera -> perception -> reasoning agent
=====================================================================

Spins up the fixed traffic camera on Town04, runs trained (or stock) YOLO26
through the perception pipeline, and sends each closed FollowingSegment to the
reasoning agent. Prints an evidence-cited verdict per following pair.

    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/run_live_test.py --weights runs/tailgating/yolo26_town04/weights/best.pt --duration 120

Use --weights yolo26m.pt to test with stock COCO weights before training.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import carla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
from perception_pipeline import FrameInput, TrafficCamPerceptionPipeline  # noqa
from reasoning_agent import (EnvironmentContext, TailgatingReasoningAgent,  # noqa
                             format_result_for_log)
import site_config as cfg  # noqa


def environment_from_world(world) -> EnvironmentContext:
    w = world.get_weather()
    if w.precipitation > 50:
        weather, surface = "rain_heavy", "wet"
    elif w.precipitation > 10:
        weather, surface = "rain_light", "wet"
    elif w.fog_density > 30:
        weather, surface = "fog", "dry"
    elif w.sun_altitude_angle < -5:
        weather, surface = "night", "dry"
    elif w.sun_altitude_angle < 10:
        weather, surface = "dawn_dusk", "dry"
    else:
        weather, surface = "dry_clear", "dry"
    return EnvironmentContext(weather, surface, "motorway", 100, False, False)


def main(weights, duration, rmse, host, port):
    intr = CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG)
    pose = CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION)
    cam = TrafficCamera(intr, pose, ground_z=cfg.GROUND_Z)
    cam.calibration_rmse_m = rmse
    cam.max_reliable_range_m = cfg.MAX_RELIABLE_RANGE_M

    pipeline = TrafficCamPerceptionPipeline(camera=cam, site_id="TOWN04_HW_01",
                                            yolo_weights=weights,
                                            lane_half_width_m=cfg.LANE_HALF_WIDTH_M)
    agent = TailgatingReasoningAgent()

    client = carla.Client(host, port); client.set_timeout(60.0)
    world = client.load_world(cfg.CARLA_MAP)
    tm = client.get_trafficmanager(); tm.set_synchronous_mode(True)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg.SIM_HZ
    world.apply_settings(settings)

    # spawn traffic
    bp_lib = world.get_blueprint_library()
    vbps = [b for b in bp_lib.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) >= 2]
    sps = world.get_map().get_spawn_points()
    import random; random.shuffle(sps)
    actors = []
    for sp in sps[:cfg.NUM_TRAFFIC_VEHICLES]:
        v = world.try_spawn_actor(random.choice(vbps), sp)
        if v:
            v.set_autopilot(True, tm.get_port()); actors.append(v)
    for v in random.sample(actors, int(len(actors)*cfg.CLOSE_FOLLOW_FRACTION)):
        tm.distance_to_leading_vehicle(v, random.uniform(*cfg.CLOSE_FOLLOW_GAP_M))

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
    cam_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
    cam_bp.set_attribute("fov", str(cfg.FOV_DEG))
    cam_tf = carla.Transform(carla.Location(*cfg.CAMERA_LOCATION),
                             carla.Rotation(pitch=cfg.CAMERA_ROTATION[0],
                                            yaw=cfg.CAMERA_ROTATION[1],
                                            roll=cfg.CAMERA_ROTATION[2]))
    camera = world.spawn_actor(cam_bp, cam_tf)
    buffer = []
    camera.listen(lambda img: buffer.append(img))

    for _ in range(60):
        world.tick()
    buffer.clear()

    def frames():
        start = time.time()
        while time.time() - start < duration:
            world.tick()
            while buffer:
                img = buffer.pop(0)
                arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
                rgb = arr[:, :, :3][:, :, ::-1].copy()
                yield FrameInput(img.frame, img.timestamp, rgb, environment_from_world(world))

    try:
        for seg in pipeline.process_stream(frames()):
            print(f"\n>>> Following pair segment {seg.segment_id}  ({seg.duration_s:.1f}s)")
            print(f"    {seg.follower_vehicle_class} behind {seg.lead_vehicle_class}")
            print(f"    aggregates: {seg.aggregates()}")
            result = agent.analyze(seg)
            print(format_result_for_log(result))
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        camera.stop(); camera.destroy()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="yolo26m.pt")
    p.add_argument("--duration", type=float, default=120.0)
    p.add_argument("--rmse", type=float, default=0.5, help="homography RMSE from validate_geometry --carla")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    main(a.weights, a.duration, a.rmse, a.host, a.port)
