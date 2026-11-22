"""
Stage 2 integration test  (trained detector + homography -> ground distances)
=============================================================================

Loads the trained best.pt, runs it on real dataset frames, takes each detection's
bottom-centre pixel through the (now-corrected) homography, and shows every
vehicle with its class, confidence, and computed down-range distance. This is the
first check that the DETECTOR and the GEOMETRY work together on live detections
(not ground-truth labels) - the payoff of the Stage-1/geometry work.

    python scripts/test_stage2_integration.py --weights best.pt --dataset ./dataset --n 6

IMPORTANT - class id mapping:
Your model was trained with the dataset's own class order (0=car, 1=motorcycle,
2=bus, 3=truck), NOT COCO ids. The pipeline's COCO_TO_NSW map (2=car,3=moto,...)
is for a COCO-pretrained detector and will mislabel/drop your model's outputs.
This test uses the CORRECT trained-class map below and flags the discrepancy.
"""

from __future__ import annotations
import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))
from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa
import site_config as cfg  # noqa

# Correct mapping for the MODEL WE TRAINED (matches site_config.CLASS_NAMES order)
TRAINED_CLASS_MAP = {i: name for i, name in enumerate(cfg.CLASS_NAMES)}
CLASS_COLORS = {"car": "#00b300", "motorcycle": "#d00000", "bus": "#7b2cbf", "truck": "#ff8800"}


def main(weights, dataset, n, split, conf, out):
    from ultralytics import YOLO

    cam = TrafficCamera(
        CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG),
        CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION), cfg.GROUND_Z)
    model = YOLO(weights)

    img_dir = Path(dataset) / "images" / split
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        raise SystemExit(f"No images in {img_dir}")
    sample = random.sample(imgs, min(n, len(imgs)))

    rows = (len(sample) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(20, 6 * rows))
    axes = np.array(axes).ravel()

    yaw = math.radians(cfg.CAMERA_ROTATION[1]); fwd = (math.cos(yaw), math.sin(yaw))
    ox, oy, _ = cfg.CAMERA_LOCATION

    total_det = 0
    for ax, img_path in zip(axes, sample):
        res = model.predict(str(img_path), verbose=False, conf=conf)[0]
        import matplotlib.image as mpimg
        ax.imshow(mpimg.imread(str(img_path)))
        ndet = 0
        for box in res.boxes:
            cid = int(box.cls[0].item())
            cls = TRAINED_CLASS_MAP.get(cid, f"id{cid}")
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            confv = float(box.conf[0].item())
            u, v = (x1 + x2) / 2.0, y2
            gx, gy = cam.pixel_to_ground(u, v)
            # down-range distance along the road axis
            down = (gx - ox) * fwd[0] + (gy - oy) * fwd[1] if not math.isnan(gx) else float("nan")
            col = CLASS_COLORS.get(cls, "#888")
            ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                           fill=False, edgecolor=col, linewidth=2))
            ax.plot(u, v, "o", color=col, markersize=5, markeredgecolor="white")
            label = f"{cls} {confv:.2f}\n{down:.0f}m" if not math.isnan(down) else f"{cls} {confv:.2f}\n(off-ground)"
            ax.text(x1, y1 - 6, label, color=col, fontsize=8, weight="bold")
            ndet += 1
        total_det += ndet
        ax.set_title(f"{img_path.stem}  ({ndet} detections)", fontsize=10)
        ax.axis("off")
    for ax in axes[len(sample):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")
    print(f"  {total_det} detections across {len(sample)} frames "
          f"(avg {total_det/len(sample):.1f}/frame)")
    print("\nWhat to check:")
    print("  - Boxes on vehicles, correct class labels, sensible confidences.")
    print("  - Down-range distances increase with vehicles further up the road,")
    print("    and match roughly what the BEV/label-check showed for similar frames.")
    print("  - If classes look wrong (e.g. everything 'car'), the pipeline's")
    print("    COCO_TO_NSW map needs replacing with the trained-class map (see note).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="path to trained best.pt")
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", default="stage2_integration.png")
    a = p.parse_args()
    main(a.weights, a.dataset, a.n, a.split, a.conf, a.out)
