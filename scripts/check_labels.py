"""
Label overlay check  (verify training labels are correct)
=========================================================

Draws the ACTUAL stored YOLO labels directly onto the raw collected images -
no homography, no re-projection, just "here is exactly what YOLO will train on".
Use this to confirm boxes envelop the vehicles before committing to training.

    python scripts/check_labels.py --dataset ./dataset
    python scripts/check_labels.py --dataset ./dataset --frame frame_00000944
    python scripts/check_labels.py --dataset ./dataset --n 6      # montage of 6 frames

Output: annotated PNG(s) you open and eyeball. Green box = label; the small dot
is the box's bottom-centre (the ground-contact point used for distance).
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "config"))
import site_config as cfg  # noqa

CLASS_COLORS = {0: "#00b300", 1: "#d00000", 2: "#7b2cbf", 3: "#ff8800"}
CLASS_NAMES = {i: n for i, n in enumerate(cfg.CLASS_NAMES)}


def draw_one(ax, img_path: Path, lbl_path: Path):
    img = mpimg.imread(str(img_path))
    H, W = img.shape[:2]
    ax.imshow(img)
    n = 0
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cls, xc, yc, w, h = int(p[0]), *(float(v) for v in p[1:])
            # YOLO normalised -> pixel corners
            bw, bh = w * W, h * H
            x1 = (xc * W) - bw / 2
            y1 = (yc * H) - bh / 2
            col = CLASS_COLORS.get(cls, "#888")
            ax.add_patch(patches.Rectangle((x1, y1), bw, bh, fill=False,
                                           edgecolor=col, linewidth=1.8))
            # bottom-centre ground-contact dot
            ax.plot(xc * W, y1 + bh, "o", color=col, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.text(x1, y1 - 5, CLASS_NAMES.get(cls, str(cls)),
                    color=col, fontsize=8, weight="bold")
            n += 1
    ax.set_title(f"{img_path.stem}  ({n} labels)", fontsize=10)
    ax.axis("off")


def pick_frames(dataset: Path, split: str, name: str | None, n: int):
    img_dir = dataset / "images" / split
    lbl_dir = dataset / "labels" / split
    if name:
        return [(img_dir / f"{name}.png", lbl_dir / f"{name}.txt")]
    # pick the n frames with the MOST labels (most informative to check)
    scored = []
    for lf in lbl_dir.glob("*.txt"):
        cnt = len([1 for ln in lf.read_text().splitlines() if ln.strip()])
        scored.append((cnt, lf))
    scored.sort(reverse=True)
    out = []
    for _, lf in scored[:n]:
        out.append((img_dir / f"{lf.stem}.png", lf))
    return out


def main(dataset: Path, split: str, name: str | None, n: int, out: Path):
    frames = pick_frames(dataset, split, name, n)
    if not frames:
        raise SystemExit(f"No frames found in {dataset/'labels'/split}")
    if len(frames) == 1:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        draw_one(ax, *frames[0])
    else:
        cols = 2
        rows = (len(frames) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
        for axx, fr in zip(np.array(axes).ravel(), frames):
            draw_one(axx, *fr)
        for axx in np.array(axes).ravel()[len(frames):]:
            axx.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"Saved {out}  ({len(frames)} frame(s))")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="./dataset")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--frame", default=None)
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--out", default="label_check.png")
    a = p.parse_args()
    main(Path(a.dataset), a.split, a.frame, a.n, Path(a.out))
