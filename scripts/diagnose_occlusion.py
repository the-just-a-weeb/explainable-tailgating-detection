"""
Occlusion filter diagnostic  (does the seg filter keep/drop the right vehicles?)
================================================================================

Connects to a running CARLA Town04, captures one RGB frame + the instance
segmentation, and annotates every vehicle with:
  - whether the occlusion filter would KEEP or DROP it
  - how many segmentation pixels it actually owns (its real visibility)
  - its down-range distance

This turns "is that car behind the mountain?" into a number: a labelled car
with only a handful of seg pixels is genuinely occluded (filter too loose); one
with hundreds/thousands is truly visible (correct to keep).

    python scripts/diagnose_occlusion.py

Green box  = KEPT by the filter (visible enough to label)
Red box    = DROPPED by the filter (occluded / too few visible pixels)
Each box is annotated with its seg-pixel count.
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
from collect_data import (carla_classup, bbox_world_vertices,  # noqa
                          decode_instance_seg, is_bicycle)

OUT = ROOT / "occlusion_diag.png"


def main(host="localhost", port=2000, n_traffic=30):
    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)

    client = carla.Client(host, port); client.set_timeout(60.0)
    if not client.get_world().get_map().name.endswith(cfg.CARLA_MAP):
        world = client.load_world(cfg.CARLA_MAP)
    else:
        world = client.get_world()
    tm = client.get_trafficmanager(); tm.set_synchronous_mode(True)
    s = world.get_settings()
    orig = (s.synchronous_mode, s.fixed_delta_seconds)
    s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    bp_lib = world.get_blueprint_library()
    vbps = [b for b in bp_lib.filter("vehicle.*")
            if int(b.get_attribute("number_of_wheels")) >= 2 and not is_bicycle(b.id)]
    sps = world.get_map().get_spawn_points()
    import random; random.shuffle(sps)
    actors = []
    for sp in sps[:n_traffic]:
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
    rgb_cam = world.spawn_actor(cam_bp, cam_tf)
    seg_bp = bp_lib.find("sensor.camera.instance_segmentation")
    seg_w = max(1, int(cfg.IMAGE_WIDTH * cfg.SEG_SCALE))
    seg_h = max(1, int(cfg.IMAGE_HEIGHT * cfg.SEG_SCALE))
    seg_bp.set_attribute("image_size_x", str(seg_w))
    seg_bp.set_attribute("image_size_y", str(seg_h))
    seg_bp.set_attribute("fov", str(cfg.FOV_DEG))
    seg_cam = world.spawn_actor(seg_bp, cam_tf)

    rgb_buf, seg_buf = [], []
    rgb_cam.listen(lambda im: rgb_buf.append(im))
    seg_cam.listen(lambda im: seg_buf.append(im))

    try:
        for _ in range(40):
            world.tick()
        rgb_buf.clear(); seg_buf.clear()
        fid = world.tick()
        rgb = seg = None
        for _ in range(40):
            r = [im for im in rgb_buf if im.frame == fid]
            sg = [im for im in seg_buf if im.frame == fid]
            if r and sg:
                rgb, seg = r[-1], sg[-1]; break
            time.sleep(0.01)
        if rgb is None:
            raise SystemExit("No matching frame captured.")

        rgb_arr = np.frombuffer(rgb.raw_data, dtype=np.uint8).reshape(
            (rgb.height, rgb.width, 4))[:, :, :3][:, :, ::-1].copy()
        seg_ids = decode_instance_seg(seg)

        yaw = math.radians(cfg.CAMERA_ROTATION[1])
        fwd = (math.cos(yaw), math.sin(yaw)); ox, oy, _ = cfg.CAMERA_LOCATION

        fig, ax = plt.subplots(figsize=(20, 11))
        ax.imshow(rgb_arr)
        kept = dropped = 0
        for actor in world.get_actors().filter("vehicle.*"):
            cls = carla_classup(actor.type_id)
            if cls is None:
                continue
            box = cam.project_bbox_vertices(bbox_world_vertices(actor))
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < cfg.MIN_BBOX_PIXELS or h < cfg.MIN_BBOX_PIXELS:
                continue
            # seg pixel count (same logic as the collector's _seg_visible)
            shape_h, shape_w = seg_ids.shape[:2]
            sx, sy = shape_w / cfg.IMAGE_WIDTH, shape_h / cfg.IMAGE_HEIGHT
            xi1, yi1 = max(0, int(x1*sx)), max(0, int(y1*sy))
            xi2, yi2 = min(shape_w, int(x2*sx)), min(shape_h, int(y2*sy))
            vis = 0
            if xi2 > xi1 and yi2 > yi1:
                crop = seg_ids[yi1:yi2, xi1:xi2]
                vis = int(np.count_nonzero(crop == (actor.id & 0xFFFF)))
            box_area = max((xi2-xi1)*(yi2-yi1), 1)
            coverage = vis / box_area
            keep = (vis >= cfg.MIN_VISIBLE_SEG_PIXELS and coverage >= cfg.MIN_SEG_BOX_COVERAGE)
            # down-range
            gx, gy = cam.pixel_to_ground((x1+x2)/2, y2)
            down = (gx-ox)*fwd[0] + (gy-oy)*fwd[1] if not math.isnan(gx) else -1
            col = "#00d000" if keep else "#ff2020"
            ax.add_patch(patches.Rectangle((x1, y1), w, h, fill=False,
                                           edgecolor=col, linewidth=2))
            ax.text(x1, y1-6, f"{'KEEP' if keep else 'DROP'} px={vis} cov={coverage:.2f} {down:.0f}m",
                    color=col, fontsize=7, weight="bold")
            kept += keep; dropped += not keep

        ax.set_title(f"Occlusion filter check - GREEN=kept ({kept})  RED=dropped ({dropped})  "
                     f"[MIN_PX={cfg.MIN_VISIBLE_SEG_PIXELS}, MIN_COV={cfg.MIN_SEG_BOX_COVERAGE}, SEG_SCALE={cfg.SEG_SCALE}]",
                     fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT, dpi=120)
        print(f"Saved {OUT}")
        print(f"  kept={kept}  dropped={dropped}")
        print("  A labelled car near terrain with px=<small> is genuinely occluded -> tighten filter.")
        print("  One with px=<hundreds+> is truly visible -> correct to keep.")
    finally:
        rgb_cam.stop(); rgb_cam.destroy()
        seg_cam.stop(); seg_cam.destroy()
        client.apply_batch([carla.command.DestroyActor(a) for a in actors])
        s2 = world.get_settings()
        s2.synchronous_mode, s2.fixed_delta_seconds = orig
        world.apply_settings(s2)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--traffic", type=int, default=30)
    a = p.parse_args()
    main(a.host, a.port, a.traffic)
