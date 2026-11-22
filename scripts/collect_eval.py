"""
Held-out EVALUATION collection  (separate from the training dataset)
====================================================================

Same camera, same labelling and occlusion filtering as collect_data.py, but built
for evaluation rather than training:

  * new random seed            -> different traffic than the training run
  * every frame is saved       -> contiguous 20 Hz timeline (training collection
                                  skipped frames with no labels)
  * traffic re-spawned every N frames -> density stays high all run, instead of
                                  the training run's early cluster then thin traffic
  * frames.csv                 -> sim time, CARLA frame, weather preset and
                                  episode for every frame, so Stage 3 uses real time
  * everything goes to images/test + labels/test (nothing here is ever trained on)

Run with CARLA open:
    python scripts/collect_eval.py --output ./dataset_eval --frames 3000 --seed 2026

Then:
    # held-out detector score (no train/val leakage)
    yolo val model=weights/best.pt data=dataset_eval/dataset.yaml imgsz=960 batch=4
    # Stage 3 cases (free dry run)
    python scripts/batch_cases.py --weights weights/best.pt --dataset ./dataset_eval --out batch_eval/batch_cases.json
"""

from __future__ import annotations
import argparse
import csv
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm
import carla

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_data import TrafficCamCollector, decode_instance_seg, resolve_weather  # noqa: E402
import site_config as cfg  # noqa: E402  (collect_data puts config/ on sys.path)

SPLIT = "test"


class EvalCollector(TrafficCamCollector):
    def __init__(self, output_dir: str):
        super().__init__(output_dir, val_split=0.0)
        for sub in ("images", "labels"):
            (self.out / sub / SPLIT).mkdir(parents=True, exist_ok=True)
            # remove the empty train/val dirs the parent created
            for s in ("train", "val"):
                d = self.out / sub / s
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
        self.n_empty = 0

    def _save_eval(self, image, labels):
        name = f"frame_{self.n_frames:08d}"
        image.save_to_disk(str(self.out / "images" / SPLIT / f"{name}.png"))
        with open(self.out / "labels" / SPLIT / f"{name}.txt", "w") as f:
            for cls, xc, yc, w, h in labels:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        self.n_frames += 1
        self.n_labels += len(labels)
        if not labels:
            self.n_empty += 1
        return name

    def _destroy(self, client, actors):
        if actors:
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])

    def collect_eval(self, target_frames, respawn_every, seed, settle_ticks,
                     host="localhost", port=2000):
        random.seed(seed)
        client = carla.Client(host, port)
        client.set_timeout(120.0)
        world = client.get_world()
        if world.get_map().name.split("/")[-1] != cfg.CARLA_MAP:
            world = client.load_world(cfg.CARLA_MAP)
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(seed)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / cfg.SIM_HZ
        world.apply_settings(settings)

        actors, camera, seg_camera = [], None, None
        meta_f = open(self.out / "frames.csv", "w", newline="")
        meta = csv.writer(meta_f)
        meta.writerow(["name", "carla_frame", "sim_time_s", "preset", "episode", "n_labels"])
        try:
            bl = world.get_blueprint_library()
            cam_bp = bl.find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
            cam_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
            cam_bp.set_attribute("fov", str(cfg.FOV_DEG))
            cam_tf = carla.Transform(
                carla.Location(*cfg.CAMERA_LOCATION),
                carla.Rotation(pitch=cfg.CAMERA_ROTATION[0], yaw=cfg.CAMERA_ROTATION[1],
                               roll=cfg.CAMERA_ROTATION[2]))
            camera = world.spawn_actor(cam_bp, cam_tf)
            buffer: list = []
            camera.listen(lambda img: buffer.append(img))

            seg_buffer: list = []
            if cfg.OCCLUSION_FILTER:
                seg_bp = bl.find("sensor.camera.instance_segmentation")
                seg_bp.set_attribute("image_size_x", str(max(1, int(cfg.IMAGE_WIDTH * cfg.SEG_SCALE))))
                seg_bp.set_attribute("image_size_y", str(max(1, int(cfg.IMAGE_HEIGHT * cfg.SEG_SCALE))))
                seg_bp.set_attribute("fov", str(cfg.FOV_DEG))
                seg_camera = world.spawn_actor(seg_bp, cam_tf)
                seg_camera.listen(lambda img: seg_buffer.append(img))

            n_presets = len(cfg.WEATHER_PRESET_NAMES)
            per_preset = max(1, target_frames // n_presets)
            widx, episode, since_spawn = -1, -1, respawn_every
            pbar = tqdm(total=target_frames, desc="Town04 EVAL")

            while self.n_frames < target_frames:
                # (re)spawn traffic at episode boundaries
                if since_spawn >= respawn_every:
                    self._destroy(client, actors)
                    world.tick()
                    actors = self._spawn_traffic(world, tm)
                    for _ in range(settle_ticks):
                        world.tick()
                    buffer.clear(); seg_buffer.clear()
                    episode += 1
                    since_spawn = 0

                want = min(n_presets - 1, self.n_frames // per_preset)
                if want != widx:
                    widx = want
                    world.set_weather(resolve_weather(cfg.WEATHER_PRESET_NAMES[widx]))
                    pbar.set_postfix_str(f"{cfg.WEATHER_PRESET_NAMES[widx]} ep{episode}")

                frame_id = world.tick()
                image = None
                for _ in range(10):
                    m = [im for im in buffer if im.frame == frame_id]
                    if m:
                        image = m[-1]; break
                    time.sleep(0.005)
                buffer.clear()
                if image is None:
                    # a dropped image would break the 20 Hz timeline; record it
                    print(f"\n  WARNING: no image for CARLA frame {frame_id}; skipped")
                    continue

                seg_ids = None
                if cfg.OCCLUSION_FILTER:
                    sm = [im for im in seg_buffer if im.frame == frame_id]
                    if sm:
                        seg_ids = decode_instance_seg(sm[-1])
                    seg_buffer.clear()

                labels = self._labels_for_frame(world, seg_ids)   # may be empty: kept
                name = self._save_eval(image, labels)
                snap = world.get_snapshot()
                meta.writerow([name, frame_id, f"{snap.timestamp.elapsed_seconds:.3f}",
                               cfg.WEATHER_PRESET_NAMES[widx], episode, len(labels)])
                since_spawn += 1
                pbar.update(1)
            pbar.close()
        finally:
            meta_f.close()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            for s in (camera, seg_camera):
                if s:
                    s.stop(); s.destroy()
            self._destroy(client, actors)

        (self.out / "dataset.yaml").write_text(
            f"path: {self.out.resolve()}\n"
            f"train: images/{SPLIT}\nval: images/{SPLIT}\n\n"
            f"nc: {len(cfg.CLASS_NAMES)}\nnames: {cfg.CLASS_NAMES}\n")
        print(f"\nDone: {self.n_frames} frames ({self.n_empty} with no labels), "
              f"{self.n_labels} labels, {episode + 1} traffic episode(s) -> {self.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="./dataset_eval")
    p.add_argument("--frames", type=int, default=3000)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--respawn-every", type=int, default=250,
                   help="re-spawn traffic every N saved frames (keeps density up)")
    p.add_argument("--settle-ticks", type=int, default=60,
                   help="unsaved ticks after each spawn so traffic gets moving")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    EvalCollector(a.output).collect_eval(a.frames, a.respawn_every, a.seed,
                                         a.settle_ticks, a.host, a.port)
