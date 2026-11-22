"""
Validate the fixed-camera geometry  (run before a long collection / test run)
=============================================================================

Two modes:

1. --offline  (no CARLA): round-trips known ground points through projection and
   the homography to confirm the math is internally exact, and prints the road
   range that lands inside the frame for the configured pose.

2. --carla    (CARLA running): spawns vehicles at known world positions on Town04,
   measures their gap via the homography (using each vehicle's bottom-centre
   pixel) and compares to the true world gap. Prints RMSE, which you then put in
   site_config / CameraMetadata.calibration_rmse_m so the agent can propagate it.

    python scripts/validate_geometry.py --offline
    python scripts/validate_geometry.py --carla
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import site_config as cfg  # noqa


def make_camera() -> TrafficCamera:
    intr = CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG)
    pose = CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION)
    return TrafficCamera(intr, pose, ground_z=cfg.GROUND_Z)


def offline():
    cam = make_camera()
    print("=== OFFLINE GEOMETRY CHECK ===")
    print(f"Pose: loc={cfg.CAMERA_LOCATION} rot={cfg.CAMERA_ROTATION} fov={cfg.FOV_DEG}")

    # Step along the camera's actual viewing direction (its yaw), not world X.
    yaw = math.radians(cfg.CAMERA_ROTATION[1])
    fwd = (math.cos(yaw), math.sin(yaw))      # unit vector along view axis
    left = (-fwd[1], fwd[0])                   # lateral (lane offset) direction
    ox, oy, _ = cfg.CAMERA_LOCATION

    max_err = 0.0
    for X in (15, 25, 40, 60, 80, 100, 120):
        for Y in (-cfg.LANE_HALF_WIDTH_M, 0.0, cfg.LANE_HALF_WIDTH_M):
            wx = ox + fwd[0] * X + left[0] * Y
            wy = oy + fwd[1] * X + left[1] * Y
            px, py, d = cam.world_to_pixel(wx, wy, cfg.GROUND_Z)
            if math.isnan(px):
                continue
            rx, ry = cam.pixel_to_ground(px, py)
            max_err = max(max_err, math.hypot(rx - wx, ry - wy))
    print(f"Max round-trip error over grid: {max_err:.6f} m  (should be ~0)")

    print("\nIn-frame road coverage (lane centre, along view axis):")
    visible = []
    for X in range(5, 200, 1):
        wx = ox + fwd[0] * X
        wy = oy + fwd[1] * X
        px, py, d = cam.world_to_pixel(wx, wy, 0.5)
        if 0 <= px <= cfg.IMAGE_WIDTH and 0 <= py <= cfg.IMAGE_HEIGHT and d > 0:
            visible.append(X)
    if visible:
        ok = min(visible) <= 20 and max(visible) >= 100
        print(f"  visible down-range: {min(visible)} .. {max(visible)} m")
        print(f"  tailgating band 15-120 m: {'COVERED' if ok else 'NOT fully covered - retune pose'}")
    else:
        print("  WARNING: no road visible - retune CAMERA_ROTATION pitch/height")


def carla_validate(host, port):
    import carla
    import numpy as np
    cam = make_camera()
    client = carla.Client(host, port); client.set_timeout(60.0)
    world = client.load_world(cfg.CARLA_MAP)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    spawned = []
    errs = []
    try:
        base = carla.Location(*cfg.CAMERA_LOCATION)
        # Place vehicles down-range at known offsets along the camera's yaw axis.
        yaw = math.radians(cfg.CAMERA_ROTATION[1])
        fwd = (math.cos(yaw), math.sin(yaw))
        positions = [25, 40, 55, 70, 90]
        actors = []
        for X in positions:
            loc = carla.Location(base.x + fwd[0]*X, base.y + fwd[1]*X, 0.3)
            tf = carla.Transform(loc, carla.Rotation(yaw=cfg.CAMERA_ROTATION[1]))
            a = world.try_spawn_actor(bp, tf)
            if a:
                actors.append((X, a)); spawned.append(a)
        world.tick() if world.get_settings().synchronous_mode else None
        import time; time.sleep(1.0)

        print("=== CARLA GEOMETRY VALIDATION (Town04) ===")
        for i in range(len(actors) - 1):
            Xa, aa = actors[i]; Xb, ab = actors[i+1]
            # bottom-centre of each vehicle bbox projected to ground
            def ground_of(actor):
                bb = actor.bounding_box
                vs = bb.get_world_vertices(actor.get_transform())
                # bottom-centre approx: mean of 4 lowest vertices
                vs_sorted = sorted(vs, key=lambda v: v.z)[:4]
                cx = sum(v.x for v in vs_sorted)/4; cy = sum(v.y for v in vs_sorted)/4
                cz = sum(v.z for v in vs_sorted)/4
                px, py, d = cam.world_to_pixel(cx, cy, cz)
                return cam.pixel_to_ground(px, py)
            gx_a, gy_a = ground_of(aa); gx_b, gy_b = ground_of(ab)
            measured = math.hypot(gx_b-gx_a, gy_b-gy_a)
            true_gap = abs(Xb - Xa)
            err = abs(measured - true_gap); errs.append(err)
            print(f"  gap {Xa}->{Xb}m: true={true_gap:.1f}  measured={measured:.2f}  err={err:.3f} m")
        if errs:
            rmse = (sum(e*e for e in errs)/len(errs))**0.5
            print(f"\nRMSE: {rmse:.3f} m")
            print(f"Set CameraMetadata.calibration_rmse_m = {rmse:.3f} (or a little higher to be safe)")
    finally:
        client.apply_batch([carla.command.DestroyActor(a) for a in spawned])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--offline", action="store_true")
    p.add_argument("--carla", action="store_true")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    if a.carla:
        carla_validate(a.host, a.port)
    else:
        offline()
