"""
Fine-tune YOLO26 on the CARLA Town04 traffic-camera dataset  (Stage 1->2 bridge)
================================================================================

    python scripts/train_yolo26.py --data dataset/dataset.yaml --model yolo26m --epochs 100

Model sizes: yolo26n (fastest CPU) / s / m (recommended) / l / x (highest mAP).
Add --p2 to use the small-object head (yolo26<size>-p2.yaml) for reliable
detection of vehicles far down-range (60-120 m), where a fixed traffic camera
sees the most distant part of its monitored stretch.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from ultralytics import YOLO


def train(data, model="yolo26m", epochs=100, imgsz=960, batch=16, device="0",
          project="runs/tailgating", name="yolo26_town04", resume=False, p2=False):
    if p2:
        size = model.replace("yolo26", "")
        net = YOLO(f"yolo26{size}-p2.yaml").load(f"yolo26{size}.pt")
        print(f"P2 small-object head: yolo26{size}-p2.yaml (backbone from yolo26{size}.pt)")
    else:
        net = YOLO(model if model.endswith(".pt") or Path(model).exists() else f"{model}.pt")

    net.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device,
        project=project, name=name, resume=resume,
        optimizer="MuSGD",        # YOLO26 default SGD+Muon hybrid
        close_mosaic=10, augment=True, cache=True, patience=30,
        save_period=10, plots=True,
        lr0=0.001, lrf=0.01, warmup_epochs=3, box=7.5, cls=0.5,
    )
    best = Path(project) / name / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    m = net.val(data=data, imgsz=imgsz, device=device)
    print(f"mAP50: {m.box.map50:.4f}   mAP50-95: {m.box.map:.4f}")
    print("\nUse in the pipeline:")
    print(f"  TrafficCamPerceptionPipeline(camera=cam, yolo_weights='{best}')")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="yolo26m")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)   # 960 suits 1920x1080 captures
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/tailgating")
    p.add_argument("--name", default="yolo26_town04")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--p2", action="store_true")
    a = p.parse_args()
    train(a.data, a.model, a.epochs, a.imgsz, a.batch, a.device, a.project, a.name, a.resume, a.p2)
