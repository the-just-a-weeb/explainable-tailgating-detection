"""
Camera geometry for a FIXED traffic camera.
===========================================

This module is the single source of truth for all camera math, shared by:
  - the data collector (to project ground-truth 3D boxes -> YOLO labels)
  - the perception pipeline (to project detected boxes -> world distances)
  - the tests (to validate distance estimates against CARLA ground truth)

Keeping the intrinsics, extrinsics, projection and homography in one place means
the labels you train on and the distances you measure at inference come from the
*exact same* camera model. That equivalence is the backbone of the
"de-black-boxing" argument: every metric the reasoning agent sees is traceable
to this geometry.

Coordinate conventions
----------------------
CARLA / UE4 world: X forward, Y right, Z up, LEFT-handed, units = metres.
OpenCV camera:     X right, Y down, Z forward (optical axis).

A fixed traffic camera does NOT move, so its extrinsic is computed once from its
world Transform and reused for every frame. This is simpler and more accurate
than a vehicle-mounted camera, whose extrinsic changes as the car pitches/rolls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

@dataclass
class CameraIntrinsics:
    image_width: int
    image_height: int
    fov_deg: float

    @property
    def focal_px(self) -> float:
        # CARLA defines FOV horizontally.
        return self.image_width / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))

    def matrix(self) -> np.ndarray:
        f = self.focal_px
        cx = self.image_width / 2.0
        cy = self.image_height / 2.0
        return np.array([[f, 0, cx],
                         [0, f, cy],
                         [0, 0, 1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Fixed-camera pose (the traffic camera on its pole/gantry)
# ---------------------------------------------------------------------------

@dataclass
class CameraPose:
    """World pose of a fixed camera, matching a carla.Transform.

    location_xyz: (x, y, z) in CARLA world metres.
    rotation_pyr: (pitch, yaw, roll) in DEGREES, CARLA convention.
                  pitch < 0 looks downward (typical for a pole camera).
    """
    location_xyz: tuple[float, float, float]
    rotation_pyr: tuple[float, float, float]

    def rotation_matrix(self) -> np.ndarray:
        pitch, yaw, roll = (math.radians(a) for a in self.rotation_pyr)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)
        # CARLA/UE4 rotation, matching carla.Transform.get_matrix() exactly
        # (LibCarla geom/Math.cpp). The previous version had the pitch (sp)
        # terms sign-inverted, which compressed the projected ground plane and
        # threw every label off; this matches the actual camera sensor.
        return np.array([
            [cp * cy,  cy * sp * sr - sy * cr,  -cy * sp * cr - sy * sr],
            [cp * sy,  sy * sp * sr + cy * cr,  -sy * sp * cr + cy * sr],
            [sp,       -cp * sr,                 cp * cr                ],
        ], dtype=np.float64)

    def world_to_camera(self) -> np.ndarray:
        """4x4 matrix mapping homogeneous world points into the camera frame
        (still in UE4 axis convention; convert to OpenCV separately)."""
        R = self.rotation_matrix()
        t = np.array(self.location_xyz, dtype=np.float64)
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = t
        return np.linalg.inv(M)


# UE4 camera axes -> OpenCV optical axes (X right, Y down, Z forward).
_UE4_TO_CV = np.array([
    [0, 1,  0],
    [0, 0, -1],
    [1, 0,  0],
], dtype=np.float64)


class TrafficCamera:
    """A fixed traffic camera: intrinsics + a static world pose.

    Provides forward projection (world point -> pixel) and a ground-plane
    homography (pixel -> world X,Y on the road surface) for distance measurement.
    """

    def __init__(self, intr: CameraIntrinsics, pose: CameraPose, ground_z: float = 0.0,
                 calibration_rmse_m: float = 0.0, max_reliable_range_m: float = 120.0):
        self.intr = intr
        self.pose = pose
        self.ground_z = ground_z  # world Z of the road plane under the camera
        self.K = intr.matrix()
        self.W2C = pose.world_to_camera()

        # Reliability metadata. calibration_rmse_m MUST be set from the measured
        # validate_geometry --carla result, not left at 0.0 - the reasoning layer
        # reports it as the measurement uncertainty, and claiming zero error would
        # misrepresent the evidence.
        self.calibration_rmse_m: float = calibration_rmse_m
        self.max_reliable_range_m: float = max_reliable_range_m
        self.warnings: list[str] = []

        self._H_img_to_ground = self._build_ground_homography()

    # -- forward projection ------------------------------------------------

    def world_to_pixel(self, x: float, y: float, z: float) -> tuple[float, float, float]:
        """Project a world point to (px, py, depth). depth>0 means in front."""
        p_world = np.array([x, y, z, 1.0])
        p_cam_ue4 = (self.W2C @ p_world)[:3]
        p_cam = _UE4_TO_CV @ p_cam_ue4
        depth = p_cam[2]
        if depth <= 1e-6:
            return float("nan"), float("nan"), depth
        px = self.K[0, 0] * p_cam[0] / depth + self.K[0, 2]
        py = self.K[1, 1] * p_cam[1] / depth + self.K[1, 2]
        return px, py, depth

    # -- ground homography (pixel -> world X,Y on the road) ----------------

    def _build_ground_homography(self) -> np.ndarray:
        """Homography mapping image pixels to world (X, Y) on the ground plane.

        For a point on the plane Z = ground_z, the full projection reduces to a
        3x3 homography. We construct it from the projection of the plane's basis:
        columns correspond to world X, world Y, and the plane offset.
        """
        R_w2c_ue4 = self.W2C[:3, :3]
        t_w2c_ue4 = self.W2C[:3, 3]
        R = _UE4_TO_CV @ R_w2c_ue4         # world -> OpenCV-cam rotation
        t = _UE4_TO_CV @ t_w2c_ue4         # world -> OpenCV-cam translation
        # World point on plane: [X, Y, ground_z, 1]. The Z column is constant.
        # Image (homogeneous) = K [ r1 r2 (r3*ground_z + t) ] [X Y 1]^T
        H_ground_to_img = self.K @ np.column_stack([
            R[:, 0],
            R[:, 1],
            R[:, 2] * self.ground_z + t,
        ])
        return np.linalg.inv(H_ground_to_img)

    def pixel_to_ground(self, px: float, py: float) -> tuple[float, float]:
        """Project an image pixel that lies on the road to world (X, Y) metres."""
        v = self._H_img_to_ground @ np.array([px, py, 1.0])
        if abs(v[2]) < 1e-12:
            return float("nan"), float("nan")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def pixel_to_ground_distance(self, px1, py1, px2, py2) -> float:
        """Planar metric distance between two image points assumed on the road."""
        x1, y1 = self.pixel_to_ground(px1, py1)
        x2, y2 = self.pixel_to_ground(px2, py2)
        if any(math.isnan(v) for v in (x1, y1, x2, y2)):
            return float("nan")
        return math.hypot(x2 - x1, y2 - y1)

    # -- 3D bbox projection (used by the data collector) -------------------

    def project_bbox_vertices(self, vertices_world: list[tuple[float, float, float]],
                              ) -> tuple[float, float, float, float] | None:
        """Project 8 world-space AABB corners to an axis-aligned image box.

        Returns (x1, y1, x2, y2) in pixels (already clipped), or None if the box
        is entirely behind the camera or off-screen.
        """
        xs, ys = [], []
        any_in_front = False
        for (x, y, z) in vertices_world:
            px, py, depth = self.world_to_pixel(x, y, z)
            if depth <= 0.1 or math.isnan(px):
                continue
            any_in_front = True
            xs.append(px)
            ys.append(py)
        if not any_in_front or len(xs) < 2:
            return None
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(float(self.intr.image_width), x2)
        y2 = min(float(self.intr.image_height), y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


__all__ = ["CameraIntrinsics", "CameraPose", "TrafficCamera"]
