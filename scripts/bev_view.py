"""
Bird's-eye-view (BEV) homography visualization
==============================================

Renders a collected frame next to a top-down map of where each labelled vehicle
sits on the road, projected through the SAME ground homography the perception
pipeline uses. A metric grid (down-range lines every 10 m, lane lines) is drawn
so vehicle positions can be read against fixed, independent reference distances.

WHAT THIS DOES AND DOESN'T SHOW
-------------------------------
The BEV is built FROM the homography, so on its own it demonstrates internal
consistency (vehicles fall into lanes, gaps look sensible), not metric accuracy
- a wrong calibration would warp self-consistently and still look plausible.
The rigorous accuracy proof is validate_geometry.py --carla (RMSE vs CARLA
ground truth, measured at 0.018 m). This BEV is the qualitative companion to
that number: the metric grid lets a viewer see that vehicles land on sensible
distances, consistent with the measured RMSE.

    python scripts/bev_view.py --dataset ./dataset --frame frame_00000123
    python scripts/bev_view.py --dataset ./dataset            # picks the busiest frame
    python scripts/bev_view.py --dataset ./dataset --out bev.png
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                # headless; saves to file
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import site_config as cfg  # noqa

CLASS_COLORS = {0: "#1e6091", 1: "#9d0208", 2: "#6a4c93", 3: "#e07a00"}  # car/moto/bus/truck
CLASS_NAMES = {i: n for i, n in enumerate(cfg.CLASS_NAMES)}

_YAW = math.radians(cfg.CAMERA_ROTATION[1])
_FWD = (math.cos(_YAW), math.sin(_YAW))
_LEFT = (-_FWD[1], _FWD[0])
_OX, _OY, _ = cfg.CAMERA_LOCATION


def make_camera() -> TrafficCamera:
    intr = CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG)
    pose = CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION)
    return TrafficCamera(intr, pose, ground_z=cfg.GROUND_Z)


def road_frame(gx: float, gy: float) -> tuple[float, float]:
    dx, dy = gx - _OX, gy - _OY
    return (dx * _FWD[0] + dy * _FWD[1], dx * _LEFT[0] + dy * _LEFT[1])


def read_boxes(label_path: Path, cam: TrafficCamera):
    """Return list of (cls, down_range, lateral, u, v) for each labelled vehicle."""
    out = []
    W, H = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls, xc, yc, w, h = int(p[0]), *(float(v) for v in p[1:])
        u, v = xc * W, (yc + h / 2.0) * H
        gx, gy = cam.pixel_to_ground(u, v)
        if math.isnan(gx):
            continue
        down, lat = road_frame(gx, gy)
        if down <= 0 or down > cfg.MAX_RELIABLE_RANGE_M + 20:
            continue
        out.append((cls, down, lat, u, v, xc, yc, w, h))
    return out


def pick_frame(dataset: Path, split: str, name: str | None) -> tuple[Path, Path]:
    img_dir = dataset / "images" / split
    lbl_dir = dataset / "labels" / split
    if name:
        return img_dir / f"{name}.png", lbl_dir / f"{name}.txt"
    # otherwise pick the label file with the most vehicles (most informative)
    best, best_n = None, -1
    for lf in lbl_dir.glob("*.txt"):
        n = len([1 for ln in lf.read_text().splitlines() if ln.strip()])
        if n > best_n:
            best, best_n = lf, n
    if best is None:
        raise SystemExit(f"No label files found in {lbl_dir}")
    return img_dir / f"{best.stem}.png", best


def render(dataset: Path, split: str, name: str | None, out_path: Path):
    cam = make_camera()
    img_path, lbl_path = pick_frame(dataset, split, name)
    if not lbl_path.exists():
        raise SystemExit(f"Label file not found: {lbl_path}")
    boxes = read_boxes(lbl_path, cam)

    fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(18, 7),
                                         gridspec_kw={"width_ratios": [1.4, 1]})

    # --- left: camera frame with boxes -----------------------------------
    if img_path.exists():
        ax_img.imshow(mpimg.imread(str(img_path)))
    W, H = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    for (cls, down, lat, u, v, xc, yc, w, h) in boxes:
        x1, y1 = (xc - w / 2) * W, (yc - h / 2) * H
        col = CLASS_COLORS.get(cls, "#444")
        ax_img.add_patch(plt.Rectangle((x1, y1), w * W, h * H, fill=False,
                                       edgecolor=col, linewidth=1.5))
        ax_img.plot(u, v, "o", color=col, markersize=4)        # ground-contact point
        ax_img.text(x1, y1 - 4, f"{down:.0f}m", color=col, fontsize=8, weight="bold")
    ax_img.set_title(f"Camera frame — {img_path.stem}  ({len(boxes)} vehicles)", fontsize=11)
    ax_img.set_xlim(0, W); ax_img.set_ylim(H, 0); ax_img.axis("off")

    # --- right: bird's-eye view with metric grid -------------------------
    max_r = cfg.MAX_RELIABLE_RANGE_M
    lane_hw = cfg.LANE_HALF_WIDTH_M
    # down-range gridlines every 10 m (independent reference distances)
    for d in range(0, int(max_r) + 1, 10):
        ax_bev.axhline(d, color="#ccc", linewidth=0.8, zorder=1)
        ax_bev.text(-lane_hw * 4.2, d, f"{d} m", fontsize=8, va="center", color="#666")
    # lateral lane reference lines
    for x in (-lane_hw, 0, lane_hw):
        ax_bev.axvline(x, color="#e0a96d" if x == 0 else "#ddd",
                       linewidth=1.0, linestyle="--", zorder=1)
    # monitored-lane band
    ax_bev.axvspan(-lane_hw, lane_hw, color="#f0f4f8", zorder=0)
    # camera at origin
    ax_bev.plot(0, 0, "^", color="black", markersize=12, zorder=5)
    ax_bev.text(0, -4, "camera", ha="center", fontsize=9, weight="bold")

    for (cls, down, lat, *_rest) in boxes:
        col = CLASS_COLORS.get(cls, "#444")
        ax_bev.plot(lat, down, "o", color=col, markersize=9,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=4)

    # following-gap annotation: nearest in-lane pair
    in_lane = sorted([(d, l, c) for (c, d, l, *_x) in boxes if abs(l) <= lane_hw])
    for i in range(len(in_lane) - 1):
        d0, l0, _ = in_lane[i]; d1, l1, _ = in_lane[i + 1]
        gap = d1 - d0
        if 1.0 < gap <= 40.0:
            ax_bev.annotate("", xy=(0, d1), xytext=(0, d0),
                            arrowprops=dict(arrowstyle="<->", color="#9d0208", lw=1.4))
            ax_bev.text(lane_hw * 0.4, (d0 + d1) / 2, f"{gap:.1f} m",
                        color="#9d0208", fontsize=9, weight="bold")

    ax_bev.set_xlim(-lane_hw * 5, lane_hw * 5)
    ax_bev.set_ylim(-8, max_r + 5)
    ax_bev.set_xlabel("lateral offset (m)"); ax_bev.set_ylabel("down-range (m)")
    ax_bev.set_title("Bird's-eye view — homography projection", fontsize=11)
    ax_bev.set_aspect("equal", adjustable="box")

    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=9, label=CLASS_NAMES.get(i, str(i)))
               for i, c in CLASS_COLORS.items() if i in CLASS_NAMES]
    ax_bev.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Homography BEV check · grid = fixed reference distances "
                 "(rigorous accuracy: validate_geometry.py --carla, RMSE 0.018 m)",
                 fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    print(f"Saved BEV to {out_path}")
    print(f"Frame: {img_path.stem}   vehicles plotted: {len(boxes)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--frame", default=None, help="frame stem, e.g. frame_00000123 (default: busiest)")
    p.add_argument("--out", default="bev_view.png")
    a = p.parse_args()
    render(Path(a.dataset), a.split, a.frame, Path(a.out))
