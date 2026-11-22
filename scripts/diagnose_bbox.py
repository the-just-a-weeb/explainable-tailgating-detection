"""
Diagnose the bounding-box offset  (why labels sit camera-ward of vehicles)
==========================================================================

Spawns one vehicle in the camera's view and prints, for it:
  - the actor world location (vehicle origin)
  - the bounding_box.location (offset from origin to box centre) and extent
  - the 8 world vertices CARLA reports
  - where the box centre and the actor origin each project in the image
  - the projected box (what becomes the label)

This tells us whether the box corners are anchored at the actor ORIGIN vs the
box CENTRE, which is the likely cause of labels sitting closer to the camera
than the vehicle.

    python scripts/diagnose_bbox.py
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

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
    world = client.get_world()        # use already-loaded world (no reload)

    # Place a car ~25 m down-range in the camera's view.
    yaw = math.radians(cfg.CAMERA_ROTATION[1])
    fwd = (math.cos(yaw), math.sin(yaw))
    ox, oy, _ = cfg.CAMERA_LOCATION
    down = 25.0
    loc = carla.Location(ox + fwd[0] * down, oy + fwd[1] * down, 0.3)

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    tf = carla.Transform(loc, carla.Rotation(yaw=cfg.CAMERA_ROTATION[1]))
    actor = world.try_spawn_actor(bp, tf)
    if actor is None:
        print("Could not spawn at that spot; trying a small offset...")
        loc.z = 0.5
        actor = world.try_spawn_actor(bp, carla.Transform(loc, tf.rotation))
    if actor is None:
        raise SystemExit("Spawn failed - move `down` or check the camera location.")

    try:
        for _ in range(20):
            world.tick() if world.get_settings().synchronous_mode else time.sleep(0.05)

        t = actor.get_transform()
        bb = actor.bounding_box
        print("=== ACTOR ===")
        print(f"  actor.location        = ({t.location.x:.2f}, {t.location.y:.2f}, {t.location.z:.2f})")
        print(f"  bounding_box.location = ({bb.location.x:.2f}, {bb.location.y:.2f}, {bb.location.z:.2f})  (offset origin->box centre)")
        print(f"  bounding_box.extent   = ({bb.extent.x:.2f}, {bb.extent.y:.2f}, {bb.extent.z:.2f})  (half-sizes)")

        # box centre in world = actor transform applied to bb.location
        # (this is what the vertices should be centred on)
        m = t.get_matrix()
        def xf(lx, ly, lz):
            wx = m[0][0]*lx + m[0][1]*ly + m[0][2]*lz + m[0][3]
            wy = m[1][0]*lx + m[1][1]*ly + m[1][2]*lz + m[1][3]
            wz = m[2][0]*lx + m[2][1]*ly + m[2][2]*lz + m[2][3]
            return wx, wy, wz
        cx, cy, cz = xf(bb.location.x, bb.location.y, bb.location.z)
        print(f"\n  TRUE box centre (world) = ({cx:.2f}, {cy:.2f}, {cz:.2f})")

        verts = bb.get_world_vertices(t)
        vs = [(v.x, v.y, v.z) for v in verts]
        vminx = min(v[0] for v in vs); vmaxx = max(v[0] for v in vs)
        vminy = min(v[1] for v in vs); vmaxy = max(v[1] for v in vs)
        vminz = min(v[2] for v in vs); vmaxz = max(v[2] for v in vs)
        vcx = (vminx+vmaxx)/2; vcy=(vminy+vmaxy)/2; vcz=(vminz+vmaxz)/2
        print(f"  vertices centre (world) = ({vcx:.2f}, {vcy:.2f}, {vcz:.2f})")
        print(f"  vertices z-range        = {vminz:.2f} .. {vmaxz:.2f}  (ground should be ~0, roof ~1.5)")

        # Where do the actor origin, box centre, and vertices project?
        apx, apy, ad = cam.world_to_pixel(t.location.x, t.location.y, t.location.z)
        bpx, bpy, bd = cam.world_to_pixel(cx, cy, cz)
        print(f"\n  actor origin projects to px=({apx:.0f},{apy:.0f})")
        print(f"  box centre   projects to px=({bpx:.0f},{bpy:.0f})")
        box = cam.project_bbox_vertices(vs)
        if box:
            x1,y1,x2,y2 = box
            print(f"  PROJECTED LABEL BOX = ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})  centre=({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})")
            print(f"\n  >>> Does label-box centre ({(x1+x2)/2:.0f},{(y1+y2)/2:.0f}) match box-centre projection ({bpx:.0f},{bpy:.0f})?")
            print(f"  >>> If the label sits LOWER (bigger y) than box-centre, vertices are anchored too low.")
    finally:
        actor.destroy()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    main(a.host, a.port)
