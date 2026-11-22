"""
Fixed Traffic-Camera Perception Pipeline  (Stages 2 + 3)
=========================================================

Consumes frames from a stationary roadside camera and produces FollowingSegment
objects for the reasoning agent. Because the camera is fixed:

  * Distances come straight from the static ground homography (no ego motion).
  * Speeds are estimated from how far a tracked vehicle's ground point moves
    between frames (vision odometry), since there is no speedometer.
  * "Following pairs" are found geometrically: for each vehicle in the monitored
    lane, the nearest vehicle ahead of it in the same lane is its lead.

Stage 2 (Spatial Perception): YOLO26 -> bottom-centre pixel -> homography -> (X,Y)
Stage 3 (Spatio-Temporal Filtration): FIFO buffer -> per-frame D_t, v_rel, TTC
                                       -> persistent-deficit gate -> segment

The persistent-deficit gate is what the diagram calls the "Persistent deficit?"
decision: a segment is only emitted to Stage 4 if a following pair stays below a
headway/TTC deficit for a sustained window, suppressing transient flickers.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

import numpy as np

from camera_geometry import TrafficCamera
from reasoning_agent import (
    CameraMetadata, EnvironmentContext, FollowingSegment, FrameSample,
    TailgatingEvent, VehicleClass,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# COCO ids (YOLO26 default) -> NSW vehicle class. Length-refined post hoc.
# The detector is trained with the dataset's own class order (see
# site_config.CLASS_NAMES): 0=car, 1=motorcycle, 2=truck. Bus was dropped
# (Town04 spawns ~none; any bus is folded into truck at collection time).
# These are NOT COCO ids.
COCO_TO_NSW: dict[int, VehicleClass] = {
    0: "car", 1: "motorcycle", 2: "truck_rigid",
}
# Approx vehicle lengths (m) for Rule 127 long-vehicle test and bbox sanity.
NOMINAL_LENGTH_M: dict[VehicleClass, float] = {
    "car": 4.6, "motorcycle": 2.1, "light_commercial": 5.5,
    "truck_rigid": 9.0, "truck_articulated": 16.0, "road_train": 36.0, "bus": 12.5,
}


# ---------------------------------------------------------------------------
# Detection + track structures
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox: tuple[float, float, float, float]   # x1,y1,x2,y2 px
    cls: VehicleClass
    conf: float
    ground_xy: tuple[float, float]            # world X (downrange), Y (lateral)


@dataclass
class TrackState:
    track_id: str
    cls: VehicleClass
    last_bbox: tuple[float, float, float, float]
    last_ground: tuple[float, float]
    last_t: float
    last_seen_frame: int
    speed_kmh: float = 0.0
    ground_hist: deque = field(default_factory=lambda: deque(maxlen=10))
    t_hist: deque = field(default_factory=lambda: deque(maxlen=10))


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class GroundTracker:
    """IoU tracker that also estimates each track's ground speed from how far its
    ground point travels between frames. Replace with ByteTrack for production;
    the speed-estimation logic carries over unchanged."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 8):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: dict[str, TrackState] = {}
        self._next = 0

    def update(self, detections: list[Detection], t: float, frame_id: int) -> dict[str, TrackState]:
        used = set()
        # Greedy IoU association
        for tid, tr in list(self.tracks.items()):
            best, best_iou = -1, self.iou_threshold
            for i, d in enumerate(detections):
                if i in used:
                    continue
                iou = _iou(tr.last_bbox, d.bbox)
                if iou > best_iou:
                    best_iou, best = iou, i
            if best >= 0:
                d = detections[best]
                self._advance(tr, d, t, frame_id)
                used.add(best)
        # New tracks
        for i, d in enumerate(detections):
            if i in used:
                continue
            tid = f"trk_{self._next:06d}"; self._next += 1
            tr = TrackState(tid, d.cls, d.bbox, d.ground_xy, t, frame_id)
            tr.ground_hist.append(d.ground_xy); tr.t_hist.append(t)
            self.tracks[tid] = tr
        # Drop stale
        self.tracks = {k: v for k, v in self.tracks.items()
                       if frame_id - v.last_seen_frame <= self.max_age}
        return self.tracks

    def _advance(self, tr: TrackState, d: Detection, t: float, frame_id: int) -> None:
        tr.ground_hist.append(d.ground_xy); tr.t_hist.append(t)
        # Speed from displacement over the buffered window (robust to per-frame noise)
        if len(tr.ground_hist) >= 2:
            (x0, y0), (x1, y1) = tr.ground_hist[0], tr.ground_hist[-1]
            dt = tr.t_hist[-1] - tr.t_hist[0]
            if dt > 1e-3:
                dist = math.hypot(x1 - x0, y1 - y0)
                tr.speed_kmh = (dist / dt) * 3.6
        tr.last_bbox = d.bbox; tr.last_ground = d.ground_xy
        tr.cls = d.cls; tr.last_t = t; tr.last_seen_frame = frame_id


# ---------------------------------------------------------------------------
# Frame I/O
# ---------------------------------------------------------------------------

@dataclass
class FrameInput:
    frame_id: int
    timestamp_s: float
    rgb_image: np.ndarray
    environment: EnvironmentContext


@dataclass
class _PairObs:
    """Per-frame observation of one follower->lead pair."""
    t: float
    frame_id: int
    follower_id: str
    lead_id: str
    follower_cls: VehicleClass
    lead_cls: VehicleClass
    gap_m: float                      # rear-to-rear ground distance
    clearance_m: float                # gap_m - follower length: clear road between
    follower_speed_kmh: float
    lead_speed_kmh: float
    conf: float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrafficCamPerceptionPipeline:
    def __init__(
        self,
        camera: TrafficCamera,
        site_id: str = "TOWN04_HW_01",
        yolo_weights: str = "yolo26m.pt",
        conf_threshold: float = 0.4,
        lane_half_width_m: float = 2.0,
        min_segment_duration_s: float = 3.0,
        max_pair_loss_s: float = 1.0,
        # persistent-deficit gate: only emit segments where the pair spends
        # at least `deficit_min_seconds` below `deficit_gap_s` headway.
        deficit_gap_s: float = 3.0,
        deficit_min_seconds: float = 3.0,
        min_gap_length_factor: float = 0.9,
        detector=None,
    ):
        # `detector` lets tests inject a fake; production passes None and we
        # load YOLO26 from weights.
        if detector is not None:
            self.detector = detector
        elif YOLO is None:
            raise ImportError("ultralytics not installed: pip install ultralytics")
        else:
            self.detector = YOLO(yolo_weights)
        self.camera = camera
        self.site_id = site_id
        self.conf_threshold = conf_threshold
        self.lane_half_width_m = lane_half_width_m
        self.min_segment_duration_s = min_segment_duration_s
        self.max_pair_loss_s = max_pair_loss_s
        self.deficit_gap_s = deficit_gap_s
        self.deficit_min_seconds = deficit_min_seconds
        # a pair closer than (lead length x this) is physically impossible
        # and is treated as a perception artefact rather than tailgating
        self.min_gap_length_factor = min_gap_length_factor

        # Camera forward unit vector and origin, for DOWN-RANGE distance along the
        # road axis. Critical: the camera looks along its yaw direction (e.g. +Y
        # at yaw 90), so "down-range" is NOT world-X. Projecting onto _fwd gives
        # the true along-road distance; assuming world-X wrongly treats every
        # vehicle as behind the camera when yaw != 0.
        import math as _m
        _yaw = _m.radians(camera.pose.rotation_pyr[1])
        self._fwd = (_m.cos(_yaw), _m.sin(_yaw))
        self._origin = (camera.pose.location_xyz[0], camera.pose.location_xyz[1])

        self.tracker = GroundTracker()
        # Open pair buffers keyed by (follower_id, lead_id)
        self._pairs: dict[tuple[str, str], list[_PairObs]] = {}
        self._pair_last_seen: dict[tuple[str, str], float] = {}

    # -- main streaming entry point ----------------------------------------

    def process_stream(self, frames: Iterable[FrameInput]) -> Iterator[FollowingSegment]:
        for frame in frames:
            yield from self._process_frame(frame)
        # Flush any open pairs at end of stream
        for key in list(self._pairs.keys()):
            seg = self._close_pair(key, force=True)
            if seg:
                yield seg

    def _process_frame(self, frame: FrameInput) -> Iterator[FollowingSegment]:
        results = self.detector.predict(frame.rgb_image, verbose=False, conf=self.conf_threshold)[0]
        detections: list[Detection] = []
        for box in results.boxes:
            cid = int(box.cls[0].item())
            if cid not in COCO_TO_NSW:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            # bottom-centre pixel -> ground (Stage 2 coordinate reduction)
            u = (x1 + x2) / 2.0
            v = y2
            gx, gy = self.camera.pixel_to_ground(u, v)
            # DOWN-RANGE distance along the road axis (project onto forward vec).
            # Using raw gx (world-X) as "behind camera" is wrong when yaw != 0.
            down = (gx - self._origin[0]) * self._fwd[0] + \
                   (gy - self._origin[1]) * self._fwd[1]
            if math.isnan(gx) or down <= 0:         # behind camera / invalid
                continue
            cls = self._refine_class(COCO_TO_NSW[cid], x1, y1, x2, y2)
            detections.append(Detection((x1, y1, x2, y2), cls, conf, (gx, gy)))

        tracks = self.tracker.update(detections, frame.timestamp_s, frame.frame_id)
        pairs = self._find_following_pairs(tracks)

        seen_keys = set()
        for follower, lead in pairs:
            key = (follower.track_id, lead.track_id)
            seen_keys.add(key)
            gap = math.hypot(lead.last_ground[0] - follower.last_ground[0],
                             lead.last_ground[1] - follower.last_ground[1])

            # Geometric plausibility gate. Both ground points are the bottom-centre
            # of the bounding box, i.e. each vehicle's REAR ground contact as seen
            # from behind. So `gap` is a rear-to-rear distance, and the FOLLOWER's
            # own body must fit inside it: clearance = gap - follower_length. A gap
            # smaller than the follower's length means the two boxes occupy the same
            # road space, which is physically impossible.
            #
            # In practice this rejects tracker-splitting artefacts: an articulated
            # vehicle split into two IDs (cab + trailer, or a duplicate) presents as
            # a "pair" a couple of metres apart at identical speed, which the
            # telemetry alone reports as an extreme violation.
            follower_len = NOMINAL_LENGTH_M.get(follower.cls, 4.6)
            if gap < follower_len * self.min_gap_length_factor:
                continue
            clearance_m = max(0.0, gap - follower_len)

            obs = _PairObs(
                t=frame.timestamp_s, frame_id=frame.frame_id,
                follower_id=follower.track_id, lead_id=lead.track_id,
                follower_cls=follower.cls, lead_cls=lead.cls,
                gap_m=gap, clearance_m=clearance_m,
                follower_speed_kmh=follower.speed_kmh,
                lead_speed_kmh=lead.speed_kmh,
                conf=min(0.99, 0.5 + 0.5 * min(1.0, gap / self.camera.max_reliable_range_m) * 0),
            )
            # conf: use detector-ish proxy; here approximate via track persistence
            obs.conf = 0.9 if gap <= self.camera.max_reliable_range_m else 0.4
            self._pairs.setdefault(key, []).append(obs)
            self._pair_last_seen[key] = frame.timestamp_s
            self._env = frame.environment

        # Close pairs not seen recently
        for key in list(self._pairs.keys()):
            if key in seen_keys:
                continue
            if frame.timestamp_s - self._pair_last_seen.get(key, 0) > self.max_pair_loss_s:
                seg = self._close_pair(key)
                if seg:
                    yield seg

    # -- following-pair geometry -------------------------------------------

    def _road_frame(self, ground_xy: tuple[float, float]) -> tuple[float, float]:
        """Convert a world ground point to (down_range, lateral) in the road
        frame: down_range = distance along the camera forward axis, lateral =
        signed offset perpendicular to it. Works for any camera yaw (the raw
        world X/Y axes only coincide with down-range/lateral at yaw 0/90 by luck).
        """
        dx = ground_xy[0] - self._origin[0]
        dy = ground_xy[1] - self._origin[1]
        down = dx * self._fwd[0] + dy * self._fwd[1]
        lateral = -dx * self._fwd[1] + dy * self._fwd[0]   # perpendicular (left +)
        return down, lateral

    def _find_following_pairs(self, tracks: dict[str, TrackState]) -> list[tuple[TrackState, TrackState]]:
        """For each vehicle in the monitored lane, find the nearest vehicle ahead
        in the same lane (its lead). Returns (follower, lead) pairs. Uses the
        road frame (down-range, lateral) so it's correct for any camera yaw."""
        rf = {t.track_id: self._road_frame(t.last_ground) for t in tracks.values()}
        lane = [t for t in tracks.values() if abs(rf[t.track_id][1]) <= self.lane_half_width_m]
        lane.sort(key=lambda t: rf[t.track_id][0])   # sort by down-range
        pairs = []
        for i, follower in enumerate(lane):
            lead = None
            for cand in lane[i + 1:]:
                if rf[cand.track_id][0] > rf[follower.track_id][0] + 1.0:
                    lead = cand
                    break
            if lead is not None:
                pairs.append((follower, lead))
        return pairs

    def _refine_class(self, base: VehicleClass, x1, y1, x2, y2) -> VehicleClass:
        """Refine truck_rigid -> truck_articulated using projected ground length."""
        if base != "truck_rigid":
            return base
        # Estimate vehicle ground length from the bbox footprint (down-range span).
        g_near = self.camera.pixel_to_ground((x1 + x2) / 2.0, y2)
        g_far = self.camera.pixel_to_ground((x1 + x2) / 2.0, y1)
        if not (math.isnan(g_near[0]) or math.isnan(g_far[0])):
            near_d, _ = self._road_frame(g_near)
            far_d, _ = self._road_frame(g_far)
            length_m = abs(far_d - near_d)
            if length_m > 12.0:
                return "truck_articulated"
        return base

    # -- segment close + event detection -----------------------------------

    def _close_pair(self, key: tuple[str, str], force: bool = False) -> Optional[FollowingSegment]:
        obs = self._pairs.pop(key, [])
        self._pair_last_seen.pop(key, None)
        if len(obs) < 2:
            return None
        duration = obs[-1].t - obs[0].t
        if duration < self.min_segment_duration_s and not force:
            return None

        # Persistent-deficit gate (Stage 3 decision)
        t0 = obs[0].t
        below = 0.0
        run_start = None
        prev_t = None
        for o in obs:
            spd = max(o.follower_speed_kmh / 3.6, 0.1)
            gs = o.gap_m / spd
            # Clearance-based headway. gap_m is rear-to-rear (both ground points
            if gs < self.deficit_gap_s:
                if run_start is None:
                    run_start = o.t
                prev_t = o.t
            else:
                if run_start is not None:
                    below = max(below, prev_t - run_start)
                run_start = None
        if run_start is not None and prev_t is not None:
            below = max(below, prev_t - run_start)
        if below < self.deficit_min_seconds and not force:
            return None   # transient - do not escalate to Stage 4

        samples: list[FrameSample] = []
        for o in obs:
            spd = max(o.follower_speed_kmh / 3.6, 0.1)
            gs = o.gap_m / spd
            # Clearance-based headway. gap_m is rear-to-rear (both ground points
            # are bottom-of-box, i.e. each vehicle's rear as seen from behind),
            # so gap_s counts the follower's own body as part of the "gap". The
            # conventional 2/3-second rule measures clear road, which is
            # stricter. Both are reported so the convention is explicit.
            cs = o.clearance_m / spd
            v_rel = o.follower_speed_kmh - o.lead_speed_kmh
            ttc = (o.gap_m / (v_rel / 3.6)) if v_rel > 0.5 else None
            # A track needs a few frames before its vision-estimated speed is
            # meaningful. Until the follower is moving plausibly (>5 km/h),
            # mark the sample low-confidence so it is excluded from aggregates
            # (otherwise gap_s = gap_m / ~0 explodes).
            conf = o.conf if o.follower_speed_kmh > 5.0 else 0.0
            samples.append(FrameSample(
                t=o.t - t0, frame_id=o.frame_id,
                follower_speed_kmh=o.follower_speed_kmh, lead_speed_kmh=o.lead_speed_kmh,
                gap_m=o.gap_m, gap_s=gs,
                clearance_m=o.clearance_m, clearance_s=cs,
                v_rel_kmh=v_rel, ttc_s=ttc,
                follower_track_id=o.follower_id, lead_track_id=o.lead_id,
                perception_confidence=conf, gap_uncertainty_m=self.camera.calibration_rmse_m,
            ))

        events = self._detect_events(samples)
        follower_cls = obs[-1].follower_cls
        lead_cls = obs[-1].lead_cls
        return FollowingSegment(
            segment_id=f"seg_{key[0]}_{key[1]}_{int(t0 * 100)}",
            follower_vehicle_class=follower_cls,
            lead_vehicle_class=lead_cls,
            follower_length_m=NOMINAL_LENGTH_M.get(follower_cls, 4.6),
            lead_length_m=NOMINAL_LENGTH_M.get(lead_cls, 4.6),
            duration_s=duration,
            samples=samples,
            environment=self._env,
            events=events,
            camera=CameraMetadata(
                site_id=self.site_id,
                calibration_rmse_m=self.camera.calibration_rmse_m,
                max_reliable_range_m=self.camera.max_reliable_range_m,
                speed_estimation="vision_tracked",
                warnings=list(self.camera.warnings),
            ),
        )

    def _detect_events(self, samples: list[FrameSample]) -> list[TailgatingEvent]:
        events = []
        for i in range(1, len(samples)):
            dt = samples[i].t - samples[i - 1].t
            if 0 < dt < 1.5:
                dgap = samples[i].gap_m - samples[i - 1].gap_m
                if dgap < -8.0 and samples[i].lead_track_id != samples[i - 1].lead_track_id:
                    events.append(TailgatingEvent(samples[i].t, "cut_in",
                        {"gap_drop_m": round(-dgap, 2)}))
        for i in range(2, len(samples)):
            d_lead_spd = samples[i].lead_speed_kmh - samples[i - 1].lead_speed_kmh
            if d_lead_spd < -5.0:
                events.append(TailgatingEvent(samples[i].t, "lead_brake",
                    {"lead_speed_drop_kmh": round(-d_lead_spd, 1)}))
        for i in range(1, len(samples)):
            d_fol = samples[i].follower_speed_kmh - samples[i - 1].follower_speed_kmh
            if d_fol < -5.0:
                events.append(TailgatingEvent(samples[i].t, "follower_brake",
                    {"speed_drop_kmh": round(-d_fol, 1)}))
        return events


__all__ = ["FrameInput", "TrafficCamPerceptionPipeline", "COCO_TO_NSW"]