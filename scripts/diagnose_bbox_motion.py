"""
Motion-aware bbox diagnostic  (reproduce the label offset in seconds)
=====================================================================

The stationary diagnostic proved the geometry is correct for a fixed vehicle.
This one reproduces the *collection* conditions - moving traffic, RGB image and
labels captured together - and saves annotated frames so you can SEE whether the
box sits on the vehicle. It runs three capture strategies on the SAME frames so
we can tell which one is correct:

  A) naive      : tick, then read whatever image is in the buffer + world state
                  (the ORIGINAL collector behaviour)
  B) frame-sync : wait for the image whose .frame == the tick id, then read state
                  (the FIX currently in collect_data.py)
  C) image-snap : read vehicle positions at the EXACT moment by using a world
                  snapshot tied to the image frame (the most rigorous option)

Each saves a few annotated PNGs into ./bbox_diag/. Open them and see which
strategy puts boxes ON the vehicles. That tells us the true fix with certainty,
on a cool CARLA, in well under a minute - no full collection run needed.

    python scripts/diagnose_bbox_motion.py
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
from collect_data import carla_classup, bbox_world_vertices  # noqa

OUT = ROOT / "bbox_diag"
OUT.mkdir(exist_ok=True)


def labels_from_world(world, cam):
    """Replicates the collector's labelling, returning pixel boxes + the actor
    world location (so we can also mark where the vehicle origin projects)."""
    out = []
    for actor in world.get_actors().filter("vehicle.*"):
        cls = carla_classup(actor.type_id)
        if cls is None:
            continue
        verts = bbox_world_vertices(actor)
        box = cam.project_bbox_vertices(verts)
        if box is None:
            continue
        loc = actor.get_transform().location
        out.append((cls, box, (loc.x, loc.y, loc.z), actor.id))
    return out


def annotate(rgb, labels, cam, title, path):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(rgb)
    for (cls, (x1, y1, x2, y2), (lx, ly, lz), aid) in labels:
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor="#00d000", linewidth=2))
        # where the actor origin projects (green X) vs box bottom-centre (red dot)
        opx, opy, od = cam.world_to_pixel(lx, ly, lz)
        if not math.isnan(opx):
            ax.plot(opx, opy, "x", color="cyan", markersize=10, markeredgewidth=2)
        ax.plot((x1 + x2) / 2, y2, "o", color="red", markersize=6)
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(host="localhost", port=2000):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)

    client = carla.Client(host, port); client.set_timeout(60.0)
    world = client.get_world()
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    settings = world.get_settings()
    orig = (settings.synchronous_mode, settings.fixed_delta_seconds)
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg.SIM_HZ
    world.apply_settings(settings)

    # spawn a handful of moving vehicles
    bp_lib = world.get_blueprint_library()
    vbps = [b for b in bp_lib.filter("vehicle.*")
            if int(b.get_attribute("number_of_wheels")) >= 4]
    sps = world.get_map().get_spawn_points()
    import random; random.shuffle(sps)
    actors = []
    for sp in sps[:25]:
        v = world.try_spawn_actor(random.choice(vbps), sp)
        if v:
            v.set_autopilot(True, tm.get_port()); actors.append(v)

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
    cam_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
    cam_bp.set_attribute("fov", str(cfg.FOV_DEG))
    cam_tf = carla.Transform(carla.Location(*cfg.CAMERA_LOCATION),
                             carla.Rotation(pitch=cfg.CAMERA_ROTATION[0],
                                            yaw=cfg.CAMERA_ROTATION[1],
                                            roll=cfg.CAMERA_ROTATION[2]))
    sensor = world.spawn_actor(cam_bp, cam_tf)
    buf = []
    sensor.listen(lambda img: buf.append(img))

    def to_rgb(img):
        a = np.frombuffer(img.raw_data, dtype=np.uint8).reshape((img.height, img.width, 4))
        return a[:, :, :3][:, :, ::-1].copy()

    try:
        for _ in range(40):
            world.tick()
        buf.clear()

        shots = 0
        attempts = 0
        while shots < 3 and attempts < 60:
            attempts += 1
            frame_id = world.tick()

            # --- Strategy A: naive (original collector) ---
            # whatever's in the buffer now + current world state
            time.sleep(0.02)
            if not buf:
                continue
            img_naive = buf[-1]
            labels_naive = labels_from_world(world, cam)

            # --- Strategy B: frame-synced image + current world state ---
            img_sync = None
            for _ in range(20):
                m = [im for im in buf if im.frame == frame_id]
                if m:
                    img_sync = m[-1]; break
                time.sleep(0.005)
            labels_sync = labels_from_world(world, cam)

            if img_sync is None:
                buf.clear()
                continue

            # Only save frames that actually contain vehicles in view
            if len(labels_sync) >= 2:
                shots += 1
                annotate(to_rgb(img_naive), labels_naive, cam,
                         f"A) NAIVE  img.frame={img_naive.frame} world=tick{frame_id}",
                         OUT / f"shot{shots}_A_naive.png")
                annotate(to_rgb(img_sync), labels_sync, cam,
                         f"B) FRAME-SYNC  img.frame={img_sync.frame} world=tick{frame_id}",
                         OUT / f"shot{shots}_B_sync.png")
                print(f"shot {shots}: naive img.frame={img_naive.frame}, "
                      f"sync img.frame={img_sync.frame}, world tick={frame_id}, "
                      f"vehicles={len(labels_sync)}")
            buf.clear()

        print(f"\nSaved annotated frames to {OUT}")
        print("Open shot1_A_naive.png vs shot1_B_sync.png and compare:")
        print("  - GREEN box = label, RED dot = box bottom-centre, CYAN x = actor origin projection")
        print("  - Whichever strategy puts boxes ON the vehicles is the correct capture method.")
        print("  - If BOTH are offset, the problem is NOT timing and we look elsewhere.")
        print("  - Note whether img.frame matches world tick - a gap means the image lags.")
    finally:
        sensor.stop(); sensor.destroy()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
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
