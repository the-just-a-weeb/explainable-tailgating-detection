"""
Decode the CARLA instance-seg ID encoding empirically
=====================================================

Stop guessing the channel order. This spawns ONE vehicle in clear view, captures
the instance-segmentation frame, finds the pixels that clearly belong to that
vehicle (by location), and prints the actual raw R/G/B/A byte values found there
alongside the actor.id. From that we can read off exactly how CARLA encodes the
instance id and fix decode_instance_seg for certain.

    python scripts/diagnose_seg_encoding.py
"""

from __future__ import annotations
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import carla

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))
from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import site_config as cfg  # noqa


def main(host="localhost", port=2000):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)
    client = carla.Client(host, port); client.set_timeout(60.0)
    if not client.get_world().get_map().name.endswith(cfg.CARLA_MAP):
        world = client.load_world(cfg.CARLA_MAP)
    else:
        world = client.get_world()
    s = world.get_settings()
    orig = (s.synchronous_mode, s.fixed_delta_seconds)
    s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    yaw = math.radians(cfg.CAMERA_ROTATION[1]); fwd = (math.cos(yaw), math.sin(yaw))
    ox, oy, _ = cfg.CAMERA_LOCATION
    vx, vy = ox + fwd[0]*30, oy + fwd[1]*30
    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    actor = world.try_spawn_actor(bp, carla.Transform(
        carla.Location(vx, vy, 0.3), carla.Rotation(yaw=cfg.CAMERA_ROTATION[1])))
    if actor is None:
        raise SystemExit("spawn failed")

    seg_bp = world.get_blueprint_library().find("sensor.camera.instance_segmentation")
    seg_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
    seg_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
    seg_bp.set_attribute("fov", str(cfg.FOV_DEG))
    seg_cam = world.spawn_actor(seg_bp, carla.Transform(
        carla.Location(*cfg.CAMERA_LOCATION),
        carla.Rotation(pitch=cfg.CAMERA_ROTATION[0], yaw=cfg.CAMERA_ROTATION[1],
                       roll=cfg.CAMERA_ROTATION[2])))
    buf = []
    seg_cam.listen(lambda im: buf.append(im))
    try:
        for _ in range(30):
            world.tick()
        buf.clear()
        fid = world.tick()
        seg = None
        for _ in range(40):
            m = [im for im in buf if im.frame == fid]
            if m:
                seg = m[-1]; break
            time.sleep(0.01)
        arr = np.frombuffer(seg.raw_data, dtype=np.uint8).reshape((seg.height, seg.width, 4))

        # find where the vehicle projects, sample a patch around it
        bb = actor.bounding_box
        loc = actor.get_transform().location
        px, py, d = cam.world_to_pixel(loc.x, loc.y, loc.z + 0.5)
        px, py = int(px), int(py)
        print(f"actor.id = {actor.id}  (low16 = {actor.id & 0xFFFF}, "
              f"hi={(actor.id>>8)&0xFF}, lo={actor.id&0xFF})")
        print(f"vehicle projects near pixel ({px},{py})")
        patch = arr[max(0,py-40):py+40, max(0,px-40):px+40]
        # list the distinct non-background pixel values in the patch
        pixels = patch.reshape(-1, 4)
        # drop pure-zero / road pixels: keep where G or B nonzero
        nz = pixels[(pixels[:,0] != 0) | (pixels[:,1] != 0)]
        if len(nz) == 0:
            print("No non-zero seg pixels in patch - vehicle may not be where expected.")
        c = Counter(map(tuple, nz))
        print("\nMost common (B,G,R,A) byte tuples on the vehicle patch:")
        for val, cnt in c.most_common(8):
            B, G, R, A = val
            print(f"  B={B:3d} G={G:3d} R={R:3d} A={A:3d}  count={cnt}   "
                  f"|  (G<<8|B)={(G<<8)|B}  (R<<8|G)={(R<<8)|G}  (B<<8|G)={(B<<8)|G}")
        print(f"\nMatch these against actor.id low16 = {actor.id & 0xFFFF} to read the encoding.")
    finally:
        seg_cam.stop(); seg_cam.destroy(); actor.destroy()
        s2 = world.get_settings(); s2.synchronous_mode, s2.fixed_delta_seconds = orig
        world.apply_settings(s2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    main(a.host, a.port)
