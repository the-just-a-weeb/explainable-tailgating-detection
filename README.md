# Explainable Tailgating Detection — Fixed Traffic Camera (CARLA Town04)

A five-stage pipeline that detects tailgating from a **fixed roadside traffic camera** (Safe-T-Cam style) and produces a legally-grounded, evidence-cited verdict under the NSW Road Rules 2014 — not a black-box score.

```
Stage 1  Data Generation      CARLA Town04 fixed camera → RGB frames + exact 3D-bbox labels
Stage 2  Spatial Perception   YOLO26 detection → bottom-centre (u,v) → homography → metric (X,Y)
Stage 3  Spatio-Temporal      FIFO buffer → per-frame Dt, v_rel, TTC → persistent-deficit gate
Stage 4  Agentic Reasoning    frozen JSON payload → LLM applies Rule 126/127 → evidentiary JSON
Stage 5  Human Control        adjudicator dashboard → confirm / override → enforcement decision
```

## Why a fixed camera (and why Town04)

This simulates **fixed infrastructure**, exactly like the NSW Safe-T-Cam network that enforces heavy-vehicle following distances. A fixed camera is *easier and more accurate* than a dashcam for this task: the homography is static (calibrate once), there is no ego-motion to compensate, and the monitored road stretch is known. **Town04** is CARLA's large multi-lane highway with long straights — the closest map to the inter-urban motorway setting Safe-T-Cam actually polices, where genuine sustained following develops.

The trade-off: there is no speedometer, so vehicle speeds are **estimated from vision** (how far a tracked vehicle's ground point moves between frames). The pipeline carries this honestly — `camera.speed_estimation = "vision_tracked"` — and the reasoning agent is told to treat speeds as noisy and lower its confidence accordingly.

## Why this is not a black box

1. **Perception emits physical quantities, not scores** — `gap_m`, `gap_s`, `ttc_s`, `v_rel`, each with units and a known uncertainty (`gap_s_uncertainty_estimate`, propagated from the homography RMSE).
2. **All statistics are computed in Python** (`FollowingSegment.aggregates()`), never by the LLM. The model interprets numbers; it does not invent them.
3. **Every legal claim cites evidence** — `reasoning_chain[].evidence_refs` point to an aggregate name, a `sample_t`, or an `event_t`.
4. **Uncertainty propagates to legal confidence**, and failure modes (samples dropped for low confidence / out of range, vision-speed noise, short segments) are stated explicitly.
5. **Every verdict ships a counterfactual** — the smallest evidence change that flips it. That is the recourse mechanism a court or adjudicator can actually argue with.

## Two closeness metrics (matches the Stage 3 design)

- `gap_s = gap_m / follower_speed` — **time headway**: the Rule 126 "can you stop safely" proxy.
- `ttc_s = gap_m / closing_speed` — **time-to-collision**: only defined when the follower is closing; captures active, dynamic hazard.

Steady close-following shows a low `gap_s` with absent/large `ttc_s`; an active closing-in shows a falling `ttc_s`. The agent sees both and distinguishes them.

## Layout

```
tailgating_traffic_cam/
├── config/
│   ├── site_config.py          camera pose, Town04 site, classes, scenario params
│   └── requirements.txt
├── src/
│   ├── camera_geometry.py      intrinsics, fixed extrinsics, 3D-bbox projection, ground homography
│   ├── perception_pipeline.py  Stage 2+3: YOLO26, tracking, pairing, speed est, deficit gate
│   └── reasoning_agent.py      Stage 4: evidence schema + NSW Rule 126/127 agent
├── scripts/
│   ├── validate_geometry.py    confirm calibration (offline + against CARLA)
│   ├── collect_data.py         Stage 1: collect YOLO26 training data
│   ├── train_yolo26.py         fine-tune YOLO26 on the CARLA dataset
│   ├── run_live_test.py        Stage 1→4 live in CARLA
│   ├── test_synthetic.py       end-to-end test, no CARLA / no GPU
│   └── adjudicator_dashboard.jsx   Stage 5: HITL review UI
└── README.md
```

## Setup

```bash
pip install -r config/requirements.txt
# install the CARLA python wheel matching your server build, e.g.:
pip install carla==0.9.15
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run order

**0. Validate the geometry (always do this first).**
```bash
python scripts/validate_geometry.py --offline       # math check + frame coverage
# with CARLA running:
python scripts/validate_geometry.py --carla          # measures real RMSE → put in --rmse later
```
Offline confirms zero round-trip error and that the 15–120 m tailgating band is in frame. The CARLA mode spawns vehicles at known distances and prints the homography RMSE to use downstream.

**1. Collect training data (CARLA running):**
```bash
python scripts/collect_data.py --output ./dataset --frames 12000
```
Fixed camera over the Town04 highway; cycles 12 weather/lighting presets; exports YOLO labels by projecting each vehicle's exact 3D bbox. A configurable fraction of traffic is set to follow closely so tailgating positives are well represented.

**2. Fine-tune YOLO26:**
```bash
python scripts/train_yolo26.py --data dataset/dataset.yaml --model yolo26m --epochs 100
# add --p2 for the small-object head (far-field vehicles at 80–120 m)
```

**3. Test the full pipeline.**

Without CARLA or a GPU (verifies geometry + Stages 2–4 logic):
```bash
python scripts/test_synthetic.py --no-agent     # geometry + schema only
python scripts/test_synthetic.py                # also runs the reasoning agent
```

Live in CARLA with your trained weights:
```bash
python scripts/run_live_test.py \
    --weights runs/tailgating/yolo26_town04/weights/best.pt \
    --rmse 0.5 --duration 120
```

**4. Human review (Stage 5):** open `scripts/adjudicator_dashboard.jsx` as an artifact, paste a segment's `evidence_brief()` JSON, generate the report, and confirm or override.

## Verified

The geometry and Stage 2–3 logic are tested and exact:

- Homography round-trip error: **0.000000 m** over a 15–120 m grid.
- A known 25 m vehicle gap measures **25.000 m**; a scripted 0.79 s headway recovers `mean_gap_s = 0.795`.
- The synthetic test produces a clean `FollowingSegment` (car behind articulated truck, 0.8 s sustained headway on a wet motorway) that the agent classes as a breach.

## Known simplifications (research scaffold, not production)

- `GroundTracker` is a greedy-IoU tracker. Swap for ByteTrack/BoT-SORT for production; the vision-speed estimation carries over unchanged.
- Vehicle length (for the Rule 127 long-vehicle test) uses nominal per-class values plus a bbox-footprint refinement; in production, estimate from the tracked 3D extent.
- `high_beam_flash` / `horn` events (aggravating signals for predatory driving) are in the schema but not yet detected from imagery.
- Penalty figures in the agent prompt reflect 2024–25 NSW schedules; verify current values before any real use. **General information only — not legal advice.**
