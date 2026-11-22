"""
Central configuration for the fixed traffic-camera tailgating pipeline.
=======================================================================

Everything site- or hardware-specific lives here so the collector, the
perception pipeline, and the tests stay in lock-step.

CARLA map: Town04
-----------------
Town04 is chosen deliberately. It contains a large multi-lane highway with a
long figure-of-eight loop and gentle curves - the closest CARLA map to the
inter-urban motorway setting that NSW Safe-T-Cam units actually monitor. It has
long straight stretches where vehicles hold steady speeds and genuine following
behaviour develops, unlike the stop-start grids of Town01/Town03.

The camera pose below was validated (see scripts/validate_geometry.py) to frame
the road from ~15 m to ~150 m down-range, which is the band where tailgating is
measurable and where Rule 126 / 127 distances live.
"""

from __future__ import annotations

# --- CARLA map -------------------------------------------------------------
CARLA_MAP = "Town04"

# A highway pose on Town04. (x, y, z) in metres, world frame.
# This sits the camera ~6 m above the road on a notional gantry, looking along
# the highway. Tune CAMERA_LOCATION to a spot on YOUR Town04 highway segment;
# the helper in scripts/find_highway_pose.py prints good candidates.
CAMERA_LOCATION = (-14.0, -208.0, 6.0)     # over the Town04 highway straight
CAMERA_ROTATION = (-8.0, 90.0, 0.0)        # (pitch, yaw, roll) deg; pitch down 8

# --- Camera intrinsics -----------------------------------------------------
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
FOV_DEG = 90.0

# Road plane height in world Z at the camera site (Town04 highway ~ 0.0).
GROUND_Z = 0.0

# Range past which monocular ground-plane distance is not trusted.
# Measured homography accuracy from validate_geometry.py --carla. This is
# reported to the reasoning layer as the measurement uncertainty - never leave
# it at 0, which would claim perfect precision.
CALIBRATION_RMSE_M = 0.018

MAX_RELIABLE_RANGE_M = 120.0
# Labelling range cutoff: vehicles beyond this down-range distance are not
# labelled. Distant vehicles (near the vanishing point / against terrain) are
# where occlusion is ambiguous, boxes are a few pixels, and measurement is least
# reliable - and they sit past the range where tailgating enforcement matters.
# Keeping labels within this band focuses the detector on the trusted range.
MAX_LABEL_RANGE_M = 90.0
# --- Classes (YOLO label ids) ---------------------------------------------
# Kept aligned with the NSW vehicle taxonomy used by the reasoning agent.
CLASS_NAMES = ["car", "motorcycle", "truck"]
CLASS_CAR, CLASS_MOTORCYCLE, CLASS_TRUCK = 0, 1, 2

# --- Long-vehicle threshold (Rule 127) ------------------------------------
LONG_VEHICLE_LENGTH_M = 7.5

# --- Scenario / collection defaults ---------------------------------------
SIM_HZ = 20                       # fixed_delta_seconds = 1/20 = 0.05 s
NUM_TRAFFIC_VEHICLES = 40         # (legacy) total vehicles; see split below
CLOSE_FOLLOW_FRACTION = 0.25      # fraction told to follow closely (tailgating)
CLOSE_FOLLOW_GAP_M = (2.0, 8.0)   # TM leading-distance range for close followers

# Camera-road-clustered spawning: put most vehicles IN-LANE on the camera's own
# road so they queue through the view and form visible tailgating pairs. A few
# scattered vehicles add cross-traffic variety. Tuned for REALISTIC density -
# vehicles spaced at natural distances with a believable MINORITY following too
# closely, not a bumper-to-bumper jam.
CAMERA_ROAD_VEHICLES = 11         # spawned in-lane along the camera's road
NUM_SCATTER_VEHICLES = 8          # spawned randomly elsewhere for variety
CAMERA_ROAD_SPAWN_REACH_M = 100.0 # how far along the road (from ~20 m behind cam)
CAMERA_ROAD_SPAWN_STEP_M = 22.0   # spacing between in-lane spawn slots
MIN_BBOX_PIXELS = 12              # discard boxes smaller than this (px, w or h)
MIN_VISIBLE_FRACTION = 0.35       # discard boxes mostly outside the frame

# --- Occlusion filtering (instance-segmentation visibility check) ----------
# A label is only kept if the vehicle is ACTUALLY visible to the camera, not
# merely geometrically in front of it. We co-locate an instance-segmentation
# camera with the RGB camera and require that a vehicle occupies at least this
# many pixels, and that the pixels the seg camera sees for it cover at least
# this fraction of its projected (geometric) bounding-box area. This removes
# vehicles hidden behind the overpass, the rock face, the barrier, or behind
# other vehicles - none of which the pure geometric projection can detect.
OCCLUSION_FILTER = True
MIN_VISIBLE_SEG_PIXELS = 12       # vehicle must own at least this many seg pixels
MIN_SEG_BOX_COVERAGE = 0.10       # visible seg pixels / projected box area >= this
# The segmentation camera only needs enough resolution to COUNT visible vehicle
# pixels, not to look good - so we render it at a fraction of the RGB resolution
# to cut GPU cost (a half-res seg pass is ~1/4 the pixels). The RGB images you
# train on stay full resolution. 0.5 = half res; 1.0 = same as RGB.
SEG_SCALE = 0.5
# When the seg camera is downscaled, the absolute pixel threshold above is in
# *seg* pixels. Keep it modest so small/distant vehicles still register.

# Weather presets cycled during collection for condition robustness.
# (names resolved against carla.WeatherParameters at runtime)
WEATHER_PRESET_NAMES = [
    "ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon",
    "MidRainyNoon", "HardRainNoon", "ClearSunset", "WetSunset",
    "SoftRainSunset", "ClearNight", "CloudyNight", "WetNight",
]

# --- Lane geometry for the monitored approach -----------------------------
# Vehicles within this half-width of the camera's principal axis are treated as
# "in the monitored lane" for following-pair selection.
LANE_HALF_WIDTH_M = 2.0
