"""
Count close-following pairs in a collected dataset  (data-quality check)
========================================================================

After a collection run, this scans the YOLO label files and reports how many
frames contain a genuine TAILGATING-RANGE following pair: two motor vehicles in
the same lane whose ground-plane gap falls in a configurable band.

Crucially, it does NOT guess closeness from pixel positions. It projects each
box's bottom-centre through the SAME camera homography the perception pipeline
uses (src/camera_geometry.py), so "close" means close in metres on the road -
the same definition Stage 3 applies. That makes this an honest preview of how
much tailgating material your dataset actually contains.

    python scripts/count_following_pairs.py --dataset ./dataset
    python scripts/count_following_pairs.py --dataset ./dataset --band 5 40
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


# Camera view axis (yaw) and lateral axis, in world coordinates. Used to convert
# a ground (X, Y) into along-road / across-road components regardless of how the
# camera is oriented on the map.
_YAW = math.radians(cfg.CAMERA_ROTATION[1])
_FWD = (math.cos(_YAW), math.sin(_YAW))         # down-range (along the road)
_LEFT = (-_FWD[1], _FWD[0])                      # lateral (across lanes)
_OX, _OY, _ = cfg.CAMERA_LOCATION


def road_frame(gx: float, gy: float) -> tuple[float, float]:
    """Convert a world ground point to (down_range, lateral) in the camera's
    road frame, so range/lane tests work for any camera yaw."""
    dx, dy = gx - _OX, gy - _OY
    down = dx * _FWD[0] + dy * _FWD[1]
    lat = dx * _LEFT[0] + dy * _LEFT[1]
    return down, lat


def ground_points_from_label(path: Path, cam: TrafficCamera):
    """Read one YOLO label file, return each vehicle's (cls, down_range, lateral)
    road-frame position via the bottom-centre of its box through the homography."""
    pts = []
    W, H = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(parts[0]), *(float(v) for v in parts[1:])
        u = xc * W
        v = (yc + h / 2.0) * H                 # bottom-centre = ground contact
        gx, gy = cam.pixel_to_ground(u, v)
        if math.isnan(gx):
            continue
        down, lat = road_frame(gx, gy)
        if down <= 0 or down > cfg.MAX_RELIABLE_RANGE_M:
            continue
        pts.append((cls, down, lat))
    return pts


def has_following_pair(pts, band, lane_half_width):
    """True if any two vehicles share a lane and sit within `band` metres along
    the road (a tailgating-range pair)."""
    lane = [p for p in pts if abs(p[2]) <= lane_half_width]
    lane.sort(key=lambda p: p[1])             # sort by down-range distance
    lo, hi = band
    for i in range(len(lane) - 1):
        for j in range(i + 1, len(lane)):
            gap = abs(lane[j][1] - lane[i][1])
            lateral = abs(lane[j][2] - lane[i][2])
            if gap < 1.0:                      # same vehicle / overlap
                continue
            if lo <= gap <= hi and lateral <= lane_half_width:
                return True
            if lane[j][1] - lane[i][1] > hi:   # sorted: no closer pair ahead
                break
    return False


def scan(dataset: Path, band, lane_half_width):
    cam = make_camera()
    totals = {}
    grand_frames = grand_pairs = 0
    for split in ("train", "val"):
        ldir = dataset / "labels" / split
        if not ldir.exists():
            continue
        files = sorted(ldir.glob("*.txt"))
        n_frames = len(files)
        n_with_pair = 0
        n_vehicles = 0
        for f in files:
            pts = ground_points_from_label(f, cam)
            n_vehicles += len(pts)
            if has_following_pair(pts, band, lane_half_width):
                n_with_pair += 1
        totals[split] = (n_frames, n_with_pair, n_vehicles)
        grand_frames += n_frames
        grand_pairs += n_with_pair

    print(f"=== Close-following audit: {dataset} ===")
    print(f"Tailgating band: {band[0]}-{band[1]} m   lane half-width: {lane_half_width} m\n")
    for split, (nf, np_, nv) in totals.items():
        if nf == 0:
            continue
        pct = 100.0 * np_ / nf if nf else 0
        avg = nv / nf if nf else 0
        print(f"  {split:5}: {nf:6d} frames | {np_:6d} with a following pair ({pct:5.1f}%) | avg {avg:.1f} vehicles/frame")
    if grand_frames:
        pct = 100.0 * grand_pairs / grand_frames
        print(f"\n  TOTAL: {grand_pairs}/{grand_frames} frames have a tailgating-range pair ({pct:.1f}%)")
        if pct < 5:
            print("  ⚠ Low. Consider raising CLOSE_FOLLOW_FRACTION or widening CLOSE_FOLLOW_GAP_M in site_config.")
        elif pct < 15:
            print("  ~ Moderate. Usable, but more close-follow material would strengthen Stages 3-4.")
        else:
            print("  ✓ Healthy supply of close-following frames for the tailgating logic.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--band", type=float, nargs=2, default=[5.0, 40.0],
                   help="tailgating gap band in metres, e.g. --band 5 40")
    p.add_argument("--lane-half-width", type=float, default=cfg.LANE_HALF_WIDTH_M)
    a = p.parse_args()
    scan(Path(a.dataset), tuple(a.band), a.lane_half_width)
