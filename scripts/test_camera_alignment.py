"""
Camera alignment test  (does the rendered image match camera_geometry?)
=======================================================================

We've ruled out timing and near-field framing. The remaining suspect is a
mismatch between what the CARLA camera SENSOR actually renders and what
camera_geometry.py assumes the camera pose/convention is. If those disagree,
labels are geometrically self-consistent but offset from the rendered vehicles -
exactly what the motion diagnostic showed.

This test removes all ambiguity:
  1. Spawn ONE vehicle at a precisely known world position in front of the camera.
  2. Render the actual camera frame.
  3. Overlay where camera_geometry predicts the vehicle's base and box should be.
  4. Save the frame. If the prediction sits on the car -> geometry matches the
     sensor (and the bug is elsewhere). If it's offset -> we can read the pixel
     offset directly and correct the pose/convention.

It also re-projects a row of ground markers at known down-range distances
(20/40/60/80 m) so you can see whether the ground plane itself is aligned.

    python scripts/test_camera_alignment.py
"""

from __future__ import annotations
import math
import sys
import time
from pathlib import Path

import numpy as np
import carla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import site_config as cfg  # noqa
sys.path.insert(0, str(ROOT / "scripts"))
from collect_data import bbox_world_vertices  # noqa

OUT = ROOT / "alignment_test.png"


def main(host="localhost", port=2000):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)

    client = carla.Client(host, port); client.set_timeout(60.0)
    # IMPORTANT: load the SAME map the camera pose was calibrated for. Using
    # whatever world happens to be loaded (client.get_world()) will place the
    # camera in the wrong map and the test is meaningless.
    if world_already_loaded := (client.get_world().get_map().name.endswith(cfg.CARLA_MAP)):
        world = client.get_world()
        print(f"Town04 already loaded.")
    else:
        print(f"Loading {cfg.CARLA_MAP} (current map is not it)...")
        world = client.load_world(cfg.CARLA_MAP)

    settings = world.get_settings()
    orig = (settings.synchronous_mode, settings.fixed_delta_seconds)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Known position: 30 m down-range along the camera yaw, lane centre.
    yaw = math.radians(cfg.CAMERA_ROTATION[1])
    fwd = (math.cos(yaw), math.sin(yaw))
    ox, oy, _ = cfg.CAMERA_LOCATION
    down = 30.0
    vx, vy = ox + fwd[0] * down, oy + fwd[1] * down

    bp = world.get_blueprint_library().find("vehicle.tesla.model3")
    # face the vehicle along the road (same yaw as camera view direction)
    actor = world.try_spawn_actor(
        bp, carla.Transform(carla.Location(vx, vy, 0.3),
                            carla.Rotation(yaw=cfg.CAMERA_ROTATION[1])))
    if actor is None:
        # try a couple of nearby down-ranges
        for d in (25, 35, 40, 45):
            vx, vy = ox + fwd[0] * d, oy + fwd[1] * d
            actor = world.try_spawn_actor(
                bp, carla.Transform(carla.Location(vx, vy, 0.3),
                                    carla.Rotation(yaw=cfg.CAMERA_ROTATION[1])))
            if actor:
                down = d; break
    if actor is None:
        raise SystemExit("Could not spawn the test vehicle on the road - adjust `down`.")

    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
    cam_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
    cam_bp.set_attribute("fov", str(cfg.FOV_DEG))
    cam_tf = carla.Transform(carla.Location(*cfg.CAMERA_LOCATION),
                             carla.Rotation(pitch=cfg.CAMERA_ROTATION[0],
                                            yaw=cfg.CAMERA_ROTATION[1],
                                            roll=cfg.CAMERA_ROTATION[2]))
    sensor = world.spawn_actor(cam_bp, cam_tf)
    buf = []
    sensor.listen(lambda im: buf.append(im))

    try:
        for _ in range(30):
            world.tick()
        # grab a settled frame
        buf.clear()
        frame_id = world.tick()
        img = None
        for _ in range(40):
            m = [im for im in buf if im.frame == frame_id]
            if m:
                img = m[-1]; break
            time.sleep(0.01)
        if img is None and buf:
            img = buf[-1]
        rgb = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            (img.height, img.width, 4))[:, :, :3][:, :, ::-1].copy()

        t = actor.get_transform()
        loc = t.location
        print(f"Test vehicle at world ({loc.x:.2f},{loc.y:.2f},{loc.z:.2f}), "
              f"~{down:.0f} m down-range")

        # geometry predictions
        base_px = cam.world_to_pixel(loc.x, loc.y, 0.0)        # ground contact
        origin_px = cam.world_to_pixel(loc.x, loc.y, loc.z)    # actor origin
        verts = bbox_world_vertices(actor)
        box = cam.project_bbox_vertices(verts)

        fig, ax = plt.subplots(figsize=(18, 10))
        ax.imshow(rgb)

        # ground reference markers down the lane (independent of the vehicle)
        for d in (20, 40, 60, 80):
            mx, my = ox + fwd[0] * d, oy + fwd[1] * d
            mpx, mpy, md = cam.world_to_pixel(mx, my, 0.0)
            if not math.isnan(mpx):
                ax.plot(mpx, mpy, "s", color="yellow", markersize=8)
                ax.text(mpx + 8, mpy, f"{d}m ground", color="yellow", fontsize=10, weight="bold")

        if box:
            x1, y1, x2, y2 = box
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor="#00ff00", linewidth=2.5))
            ax.text(x1, y1 - 8, "predicted box", color="#00ff00", fontsize=11, weight="bold")
        ax.plot(*base_px[:2], "o", color="red", markersize=10, label="predicted ground contact")
        ax.plot(*origin_px[:2], "x", color="cyan", markersize=12, markeredgewidth=2.5,
                label="actor origin proj")
        ax.legend(loc="upper right", fontsize=11)
        ax.set_title(f"Alignment: vehicle at {down:.0f} m. "
                     f"Do the green box / red dot sit ON the car?", fontsize=13)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT, dpi=120)
        print(f"\nSaved {OUT}")
        print("Open it and look:")
        print("  - GREEN box / RED dot on the car  -> geometry matches sensor (bug is elsewhere)")
        print("  - Offset from the car             -> sensor pose != camera_geometry pose;")
        print("    measure the pixel offset and we correct the pose/convention.")
        print("  - YELLOW squares show the ground plane at 20/40/60/80 m for reference.")
    finally:
        sensor.stop(); sensor.destroy()
        actor.destroy()
        s = world.get_settings()
        s.synchronous_mode, s.fixed_delta_seconds = orig
        world.apply_settings(s)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    main(a.host, a.port)
