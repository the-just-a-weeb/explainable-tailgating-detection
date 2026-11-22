"""
CARLA Town04 Fixed Traffic-Camera Data Collector  (Stage 1)
============================================================

Mounts a STATIONARY camera over the Town04 highway (as a Safe-T-Cam style
roadside unit would be), spawns roaming traffic with a configurable fraction of
close-followers, and exports YOLO-format detection labels by projecting each
vehicle's exact CARLA 3D bounding box into the fixed camera.

Run CARLA first:
    ./CarlaUE4.sh -quality-level=Epic
Then:
    python scripts/collect_data.py --output ./dataset --frames 12000

Output (Ultralytics YOLO layout):
    dataset/
      images/{train,val}/*.png
      labels/{train,val}/*.txt        # "<cls> <xc> <yc> <w> <h>" normalised
      dataset.yaml

The camera pose, map, classes and scenario parameters all come from
config/site_config.py so collection and inference stay consistent.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import carla

# make src/ and config/ importable when run from scripts/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

from camera_geometry import CameraIntrinsics, CameraPose, TrafficCamera  # noqa: E402
import site_config as cfg  # noqa: E402


# CARLA pedal-cycle blueprints (two-wheeled but NOT motor vehicles). These are
# excluded entirely: bicycles are not subject to the following-distance rules
# this system models, so they must never be spawned or labelled.
BICYCLE_KEYS = ("crossbike", "omafiets", "century", "bh.crossbike",
                "diamondback", "gazelle")


def is_bicycle(type_id: str) -> bool:
    return any(k in type_id.lower() for k in BICYCLE_KEYS)


def carla_classup(type_id: str) -> int | None:
    b = type_id.lower()
    if is_bicycle(b):
        return None                       # pedal cycle - not in scope, skip
    # 'bus' class was dropped (Town04 spawns ~none); fold any bus into truck,
    # since a bus is a large/long vehicle and behaves like one for Rule 127.
    if any(k in b for k in ("truck", "bus", "carlamotors", "cybertruck", "sprinter", "ambulance", "firetruck")):
        return cfg.CLASS_TRUCK
    if any(k in b for k in ("harley", "yamaha", "kawasaki", "vespa", "motorcycle")):
        return cfg.CLASS_MOTORCYCLE
    return cfg.CLASS_CAR


def resolve_weather(name: str):
    return getattr(carla.WeatherParameters, name)


def bbox_world_vertices(actor: carla.Actor) -> list[tuple[float, float, float]]:
    bb = actor.bounding_box
    verts = bb.get_world_vertices(actor.get_transform())
    return [(v.x, v.y, v.z) for v in verts]


def decode_instance_seg(image: "carla.Image") -> np.ndarray:
    """Decode CARLA instance segmentation into a per-pixel actor-id array.

    Encoding verified empirically (diagnose_seg_encoding.py) for this CARLA
    build: raw_data is BGRA per pixel [B, G, R, A]; R is the semantic tag and the
    actor instance id is  id = (B << 8) | G  (G = low byte, B = high byte).
    A vehicle hidden behind terrain/another car owns no pixels with its id here,
    which is the occlusion signal the geometric projection lacks.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    b = arr[:, :, 0].astype(np.uint32)   # byte 0 = B (high byte of id)
    g = arr[:, :, 1].astype(np.uint32)   # byte 1 = G (low byte of id)
    return ((b << 8) | g).astype(np.uint32)


class TrafficCamCollector:
    def __init__(self, output_dir: str, val_split: float = 0.15):
        self.out = Path(output_dir)
        self.val_split = val_split
        for split in ("train", "val"):
            (self.out / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.out / "labels" / split).mkdir(parents=True, exist_ok=True)

        intr = CameraIntrinsics(cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT, cfg.FOV_DEG)
        pose = CameraPose(cfg.CAMERA_LOCATION, cfg.CAMERA_ROTATION)
        self.cam = TrafficCamera(intr, pose, ground_z=cfg.GROUND_Z)

        # camera forward unit vector (road axis) for down-range distance
        _yaw = math.radians(cfg.CAMERA_ROTATION[1])
        self._fwd = (math.cos(_yaw), math.sin(_yaw))

        self.n_frames = 0
        self.n_labels = 0

    def collect(self, target_frames: int, host="localhost", port=2000):
        client = carla.Client(host, port)
        client.set_timeout(60.0)

        world = client.load_world(cfg.CARLA_MAP)
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / cfg.SIM_HZ
        world.apply_settings(settings)

        actors: list[carla.Actor] = []
        camera = None
        seg_camera = None
        try:
            actors = self._spawn_traffic(world, tm)

            # Fixed camera as a world sensor (NOT attached to any vehicle)
            cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            cam_bp.set_attribute("image_size_x", str(cfg.IMAGE_WIDTH))
            cam_bp.set_attribute("image_size_y", str(cfg.IMAGE_HEIGHT))
            cam_bp.set_attribute("fov", str(cfg.FOV_DEG))
            cam_tf = carla.Transform(
                carla.Location(*cfg.CAMERA_LOCATION),
                carla.Rotation(pitch=cfg.CAMERA_ROTATION[0],
                               yaw=cfg.CAMERA_ROTATION[1],
                               roll=cfg.CAMERA_ROTATION[2]),
            )
            camera = world.spawn_actor(cam_bp, cam_tf)

            buffer: list[carla.Image] = []
            camera.listen(lambda img: buffer.append(img))

            # Instance-segmentation camera, co-located with the RGB camera
            # (identical transform, resolution, FOV) so their pixels line up.
            # Used only to test real visibility, never saved.
            seg_buffer: list[carla.Image] = []
            seg_camera = None
            if cfg.OCCLUSION_FILTER:
                seg_bp = world.get_blueprint_library().find(
                    "sensor.camera.instance_segmentation")
                seg_w = max(1, int(cfg.IMAGE_WIDTH * cfg.SEG_SCALE))
                seg_h = max(1, int(cfg.IMAGE_HEIGHT * cfg.SEG_SCALE))
                seg_bp.set_attribute("image_size_x", str(seg_w))
                seg_bp.set_attribute("image_size_y", str(seg_h))
                seg_bp.set_attribute("fov", str(cfg.FOV_DEG))
                seg_camera = world.spawn_actor(seg_bp, cam_tf)
                seg_camera.listen(lambda img: seg_buffer.append(img))

            # settle traffic
            for _ in range(60):
                world.tick()
            buffer.clear()
            seg_buffer.clear()

            pbar = tqdm(total=target_frames, desc="Town04 traffic-cam")
            # Distribute ALL presets evenly across the whole run so every
            # condition - including the night presets at the end of the list -
            # gets sampled. (Previously we advanced one preset per 400 frames,
            # so a 3000-frame run stopped ~7/12 presets in and never reached
            # ClearNight/CloudyNight/WetNight.)
            n_presets = len(cfg.WEATHER_PRESET_NAMES)
            frames_per_preset = max(1, target_frames // n_presets)
            widx = -1
            while self.n_frames < target_frames:
                desired_idx = min(n_presets - 1, self.n_frames // frames_per_preset)
                if desired_idx != widx:
                    widx = desired_idx
                    world.set_weather(resolve_weather(cfg.WEATHER_PRESET_NAMES[widx]))
                    pbar.set_postfix_str(cfg.WEATHER_PRESET_NAMES[widx])

                frame_id = world.tick()   # synchronous: returns the frame id of THIS tick

                # Wait for the camera image belonging to exactly this frame. In
                # synchronous mode the sensor callback is delivered for the tick;
                # we match on image.frame so the RGB image and the vehicle
                # positions we read are from the SAME instant. Without this, the
                # image lags the world state by a tick and labels appear shifted
                # ahead of fast-moving vehicles.
                image = None
                for _ in range(10):
                    matches = [im for im in buffer if im.frame == frame_id]
                    if matches:
                        image = matches[-1]
                        break
                    time.sleep(0.005)
                buffer.clear()
                if image is None:
                    continue

                seg_ids = None
                if cfg.OCCLUSION_FILTER:
                    seg_match = [im for im in seg_buffer if im.frame == frame_id]
                    if seg_match:
                        seg_ids = decode_instance_seg(seg_match[-1])
                    seg_buffer.clear()

                # Read vehicle positions for THIS SAME frame_id.
                labels = self._labels_for_frame(world, seg_ids)
                if not labels:
                    continue
                split = "val" if random.random() < self.val_split else "train"
                self._save(image, labels, split)
                pbar.update(1)
            pbar.close()

        finally:
            settings.synchronous_mode = False
            world.apply_settings(settings)
            if camera:
                camera.stop(); camera.destroy()
            if seg_camera:
                seg_camera.stop(); seg_camera.destroy()
            client.apply_batch([carla.command.DestroyActor(a) for a in actors])

        self._write_yaml()
        print(f"\nDone: {self.n_frames} frames, {self.n_labels} labels -> {self.out/'dataset.yaml'}")

    def _spawn_traffic(self, world, tm) -> list[carla.Actor]:
        bp_lib = world.get_blueprint_library()
        # Motor vehicles only: >=2 wheels AND not a pedal cycle. Bicycles are
        # out of scope for following-distance enforcement, so we never spawn them.
        vehicle_bps = [b for b in bp_lib.filter("vehicle.*")
                       if int(b.get_attribute("number_of_wheels")) >= 2
                       and not is_bicycle(b.id)]
        carla_map = world.get_map()
        spawned = []

        # --- Cluster most traffic ON THE CAMERA'S ROAD, in-lane, so vehicles
        #     flow through the view in queues and actually form the tailgating
        #     pairs the project is about. Scattering across all of Town04 (the
        #     old behaviour) left ~1.4 vehicles/frame in view and ~0 pairs. ---
        cam_loc = carla.Location(*cfg.CAMERA_LOCATION)
        yaw = math.radians(cfg.CAMERA_ROTATION[1])
        fwd = carla.Vector3D(math.cos(yaw), math.sin(yaw), 0.0)

        # Start a little BEHIND the camera so vehicles drive toward/through view,
        # and walk waypoints FORWARD along the road for the reliable range.
        start = carla.Location(cam_loc.x - fwd.x * 20.0,
                               cam_loc.y - fwd.y * 20.0, 0.0)
        wp = carla_map.get_waypoint(start, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)

        road_spawns = []            # (transform, is_on_camera_road)
        if wp is not None:
            # Collect this lane and its drivable neighbours, walking forward.
            lanes_to_seed = [wp]
            for nb in (wp.get_left_lane(), wp.get_right_lane()):
                if nb and nb.lane_type == carla.LaneType.Driving:
                    lanes_to_seed.append(nb)
            step_m = cfg.CAMERA_ROAD_SPAWN_STEP_M
            reach = cfg.CAMERA_ROAD_SPAWN_REACH_M
            for lane_wp in lanes_to_seed:
                cur = lane_wp
                dist = 0.0
                while cur and dist < reach:
                    tf = cur.transform
                    tf.location.z += 0.3        # lift slightly to avoid ground clip
                    road_spawns.append(tf)
                    nxts = cur.next(step_m)
                    if not nxts:
                        break
                    cur = nxts[0]
                    dist += step_m

        random.shuffle(road_spawns)
        n_road = min(cfg.CAMERA_ROAD_VEHICLES, len(road_spawns))
        for tf in road_spawns[:n_road]:
            bp = random.choice(vehicle_bps)
            v = world.try_spawn_actor(bp, tf)
            if v:
                v.set_autopilot(True, tm.get_port())
                spawned.append(v)

        # A few scattered vehicles elsewhere for variety / cross traffic.
        scatter_pts = carla_map.get_spawn_points()
        random.shuffle(scatter_pts)
        n_scatter = min(cfg.NUM_SCATTER_VEHICLES, len(scatter_pts))
        for sp in scatter_pts[:n_scatter]:
            bp = random.choice(vehicle_bps)
            v = world.try_spawn_actor(bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                spawned.append(v)

        # Make a high fraction follow closely to generate tailgating positives.
        # Because they're queued in-lane on the camera's road, close-following
        # now happens IN VIEW rather than off somewhere on the map.
        n_close = int(len(spawned) * cfg.CLOSE_FOLLOW_FRACTION)
        for v in random.sample(spawned, n_close) if n_close else []:
            lo, hi = cfg.CLOSE_FOLLOW_GAP_M
            tm.distance_to_leading_vehicle(v, random.uniform(lo, hi))
            tm.vehicle_percentage_speed_difference(v, random.uniform(-20, 0))  # a bit faster
        print(f"  spawned {len(spawned)} vehicles "
              f"({n_road} on camera road, {len(spawned)-n_road} scattered), "
              f"{n_close} set to close-follow")
        return spawned

    def _labels_for_frame(self, world, seg_ids=None) -> list[tuple[int, float, float, float, float]]:
        labels = []
        W, H = cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT
        for actor in world.get_actors().filter("vehicle.*"):
            cls = carla_classup(actor.type_id)
            if cls is None:
                continue
            verts = bbox_world_vertices(actor)
            box = self.cam.project_bbox_vertices(verts)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < cfg.MIN_BBOX_PIXELS or h < cfg.MIN_BBOX_PIXELS:
                continue
            # Range cutoff: drop vehicles beyond the reliable labelling range.
            # The bottom-centre ground point gives down-range distance; distant
            # vehicles (near the vanishing point / against terrain) are where
            # occlusion is ambiguous and measurement is unreliable, and they sit
            # past the range where tailgating enforcement matters. Cutting them
            # keeps labels in the band the geometry is trusted for.
            u, v = (x1 + x2) / 2.0, y2
            gx, gy = self.cam.pixel_to_ground(u, v)
            if not math.isnan(gx):
                down = (gx - cfg.CAMERA_LOCATION[0]) * self._fwd[0] + \
                       (gy - cfg.CAMERA_LOCATION[1]) * self._fwd[1]
                if down <= 0 or down > cfg.MAX_LABEL_RANGE_M:
                    continue
            # discard boxes mostly clipped off-screen (edge-of-frame case)
            full_w, full_h = self._unclipped_size(verts)
            if full_w > 0 and (w * h) / max(full_w * full_h, 1) < cfg.MIN_VISIBLE_FRACTION:
                continue
            # OCCLUSION CHECK: require the vehicle to actually be visible in the
            # instance-segmentation view (handles overpass / barrier / rock /
            # vehicle-behind-vehicle occlusion, which geometry alone cannot see).
            if seg_ids is not None:
                if not self._seg_visible(actor, seg_ids, x1, y1, x2, y2):
                    continue
            labels.append((cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, w / W, h / H))
        return labels

    def _seg_visible(self, actor, seg_ids, x1, y1, x2, y2) -> bool:
        """True if `actor` genuinely occupies enough pixels in the seg view.

        We look only inside the actor's projected bounding box (a small crop, so
        this stays cheap) and count pixels whose instance id matches the low 16
        bits of the actor id. Two tests must pass:
          1. absolute: at least MIN_VISIBLE_SEG_PIXELS belong to the vehicle;
          2. relative: those pixels cover at least MIN_SEG_BOX_COVERAGE of the
             projected box area (guards against a box that mostly wraps an
             occluder while only a sliver of the real vehicle shows).
        """
        target = actor.id & 0xFFFF                  # seg encodes low 16 bits
        # The seg image may be rendered smaller than the RGB image. Derive the
        # scale from the seg array's own shape so the box maps to the right crop
        # regardless of SEG_SCALE.
        seg_h, seg_w = seg_ids.shape[:2]
        sx = seg_w / cfg.IMAGE_WIDTH
        sy = seg_h / cfg.IMAGE_HEIGHT
        xi1, yi1 = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        xi2, yi2 = min(seg_w, int(x2 * sx)), min(seg_h, int(y2 * sy))
        if xi2 <= xi1 or yi2 <= yi1:
            return False
        crop = seg_ids[yi1:yi2, xi1:xi2]
        visible_px = int(np.count_nonzero(crop == target))
        if visible_px < cfg.MIN_VISIBLE_SEG_PIXELS:
            return False
        box_area = (xi2 - xi1) * (yi2 - yi1)
        if visible_px / max(box_area, 1) < cfg.MIN_SEG_BOX_COVERAGE:
            return False
        return True

    def _unclipped_size(self, verts) -> tuple[float, float]:
        xs, ys = [], []
        for (x, y, z) in verts:
            px, py, d = self.cam.world_to_pixel(x, y, z)
            if d > 0.1 and not np.isnan(px):
                xs.append(px); ys.append(py)
        if len(xs) < 2:
            return 0.0, 0.0
        return max(xs) - min(xs), max(ys) - min(ys)

    def _save(self, image, labels, split):
        name = f"frame_{self.n_frames:08d}"
        image.save_to_disk(str(self.out / "images" / split / f"{name}.png"))
        with open(self.out / "labels" / split / f"{name}.txt", "w") as f:
            for cls, xc, yc, w, h in labels:
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        self.n_frames += 1
        self.n_labels += len(labels)

    def _write_yaml(self):
        (self.out / "dataset.yaml").write_text(
            f"path: {self.out.resolve()}\n"
            f"train: images/train\nval: images/val\n\n"
            f"nc: {len(cfg.CLASS_NAMES)}\nnames: {cfg.CLASS_NAMES}\n"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Collect YOLO26 training data from a fixed CARLA Town04 traffic camera")
    p.add_argument("--output", default="./dataset")
    p.add_argument("--frames", type=int, default=12000)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=2000)
    a = p.parse_args()
    TrafficCamCollector(a.output, a.val_split).collect(a.frames, a.host, a.port)
