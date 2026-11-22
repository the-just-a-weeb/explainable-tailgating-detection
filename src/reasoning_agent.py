"""
NSW Tailgating Reasoning Agent  (Stage 4 - Agentic Reasoning / XAI)
===================================================================

Fixed-traffic-camera variant. The perception layer observes a *following pair*
(a FOLLOWER behind a LEAD) from a roadside camera and produces structured
evidence. This module applies NSW Road Rules 2014 (Rule 126, and Rule 127 for
long vehicles) and returns a verdict in which every claim cites evidence.

Two complementary closeness metrics are carried, matching the pipeline diagram:

  gap_s = D_t / v_follower        (time headway: can the follower stop in time)
  TTC   = D_t / v_rel             (time-to-collision: closing-rate hazard)

TTC is only meaningful when the follower is closing (v_rel > 0). When the pair
holds steady speed, TTC -> infinity while gap_s stays finite - the agent is told
about both so it can distinguish "sitting too close" from "actively closing in".

Nothing here calls the LLM to do arithmetic: all statistics are computed in
Python (auditable), and the model interprets them against the statute.
"""

from __future__ import annotations

import json
import re
import os
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import anthropic


# ---------------------------------------------------------------------------
# Evidence schema  (the frozen JSON contract between perception and reasoning)
# ---------------------------------------------------------------------------

Weather = Literal[
    "dry_clear", "overcast", "rain_light", "rain_heavy", "fog", "dawn_dusk", "night"
]
RoadSurface = Literal["dry", "wet", "icy", "gravel"]
RoadClass = Literal["urban_local", "urban_arterial", "rural", "motorway"]
VehicleClass = Literal[
    "car", "motorcycle", "light_commercial",
    "truck_rigid", "truck_articulated", "road_train", "bus",
]
EventType = Literal[
    "cut_in", "lead_brake", "follower_brake", "lane_change",
    "overtake", "high_beam_flash", "horn",
]


@dataclass
class FrameSample:
    """One time-step for a following pair, as produced by the fixed camera."""
    t: float                          # seconds since segment start
    frame_id: int
    follower_speed_kmh: float         # vision-estimated (tracked displacement)
    lead_speed_kmh: float             # vision-estimated
    gap_m: float                      # rear-to-rear ground-plane longitudinal gap
    gap_s: float                      # gap_m / follower_speed  (time headway)
    v_rel_kmh: float                  # follower_speed - lead_speed (closing if >0)
    ttc_s: Optional[float]            # gap_m / v_rel  (None if not closing)
    follower_track_id: str
    lead_track_id: str
    perception_confidence: float      # min detection conf of the pair
    gap_uncertainty_m: float          # from homography calibration error
    # defaulted fields must come last in a dataclass
    clearance_m: float = 0.0          # clear road between vehicles (gap_m - follower len)
    clearance_s: float = 0.0          # clearance_m / follower_speed - the STRICTER headway


@dataclass
class TailgatingEvent:
    t: float
    type: EventType
    details: dict = field(default_factory=dict)


@dataclass
class EnvironmentContext:
    weather: Weather
    road_surface: RoadSurface
    road_class: RoadClass
    speed_limit_kmh: int
    is_built_up_area: bool
    is_safe_t_cam_zone: bool = False


@dataclass
class CameraMetadata:
    """Replaces the dashcam 'homography' block. Notes that speeds are estimated
    from vision (not a speedometer), so the agent calibrates trust accordingly."""
    site_id: str
    calibration_rmse_m: float
    max_reliable_range_m: float
    speed_estimation: str = "vision_tracked"   # vs "speedometer"
    warnings: list[str] = field(default_factory=list)


@dataclass
class FollowingSegment:
    """A continuous segment where one vehicle followed another past the camera."""
    segment_id: str
    follower_vehicle_class: VehicleClass
    lead_vehicle_class: VehicleClass
    follower_length_m: float
    lead_length_m: float
    duration_s: float
    samples: list[FrameSample]
    environment: EnvironmentContext
    events: list[TailgatingEvent]
    camera: CameraMetadata

    # -- derived aggregates (computed here, not by the LLM) -----------------

    def aggregates(self) -> dict[str, Any]:
        usable = [
            s for s in self.samples
            if s.perception_confidence >= 0.5
            and s.gap_m <= self.camera.max_reliable_range_m
        ]
        if not usable:
            return {"usable_samples": 0}

        gaps_s = [s.gap_s for s in usable]
        clears_s = [s.clearance_s for s in usable] or [0.0]
        gaps_m = [s.gap_m for s in usable]
        fspeeds = [s.follower_speed_kmh for s in usable]
        ttcs = [s.ttc_s for s in usable if s.ttc_s is not None and s.ttc_s > 0]

        long_pair = (
            self._is_long(self.follower_vehicle_class)
            and self._is_long(self.lead_vehicle_class)
            and not self.environment.is_built_up_area
        )

        return {
            "usable_samples": len(usable),
            "total_samples": len(self.samples),
            "excluded_low_confidence": sum(1 for s in self.samples if s.perception_confidence < 0.5),
            "excluded_out_of_range": sum(
                1 for s in self.samples if s.gap_m > self.camera.max_reliable_range_m
            ),
            # time-headway statistics
            "mean_gap_s": round(statistics.mean(gaps_s), 3),
            "median_gap_s": round(statistics.median(gaps_s), 3),
            "min_gap_s": round(min(gaps_s), 3),
            "p10_gap_s": round(_pct(gaps_s, 10), 3),
            # Clearance-based equivalents: clear road between the vehicles, i.e.
            # gap_m minus the follower's own length. Stricter than gap_s.
            "mean_clearance_s": round(statistics.mean(clears_s), 3),
            "median_clearance_s": round(statistics.median(clears_s), 3),
            "min_clearance_s": round(min(clears_s), 3),
            "sustained_clearance_below_2s_seconds": round(
                self._longest_below(usable, "clearance_s", 2.0), 2),
            "sustained_clearance_below_3s_seconds": round(
                self._longest_below(usable, "clearance_s", 3.0), 2),
            "mean_gap_m": round(statistics.mean(gaps_m), 2),
            "min_gap_m": round(min(gaps_m), 2),
            "mean_follower_speed_kmh": round(statistics.mean(fspeeds), 1),
            "max_follower_speed_kmh": round(max(fspeeds), 1),
            # sustained headway breaches
            "sustained_below_2s_seconds": round(self._longest_below(usable, "gap_s", 2.0), 2),
            "sustained_below_3s_seconds": round(self._longest_below(usable, "gap_s", 3.0), 2),
            "sustained_below_4s_seconds": round(self._longest_below(usable, "gap_s", 4.0), 2),
            # TTC (closing-rate hazard)
            "min_ttc_s": round(min(ttcs), 2) if ttcs else None,
            "mean_ttc_s": round(statistics.mean(ttcs), 2) if ttcs else None,
            "seconds_closing": round(sum(
                1 for s in usable if s.v_rel_kmh > 1.0
            ) / max(len(usable), 1) * self.duration_s, 2),
            "sustained_ttc_below_3s_seconds": round(
                self._longest_below_ttc(usable, 3.0), 2
            ),
            # Rule 127 metric distance (long vehicles outside built-up area)
            "sustained_below_60m_seconds": (
                round(self._longest_below(usable, "gap_m", 60.0), 2) if long_pair else None
            ),
            # propagated measurement uncertainty on the headway
            "gap_s_uncertainty_estimate": round(
                self.camera.calibration_rmse_m / max(statistics.mean(fspeeds) / 3.6, 0.1), 3
            ),
        }

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _longest_below(samples: list[FrameSample], attr: str, thr: float) -> float:
        longest = run_start = prev_t = 0.0
        in_run = False
        for s in samples:
            if getattr(s, attr) < thr:
                if not in_run:
                    in_run, run_start = True, s.t
                prev_t = s.t
            else:
                if in_run:
                    longest = max(longest, prev_t - run_start)
                in_run = False
        if in_run:
            longest = max(longest, prev_t - run_start)
        return longest

    @staticmethod
    def _longest_below_ttc(samples: list[FrameSample], thr: float) -> float:
        longest = run_start = prev_t = 0.0
        in_run = False
        for s in samples:
            ttc = s.ttc_s
            if ttc is not None and 0 < ttc < thr:
                if not in_run:
                    in_run, run_start = True, s.t
                prev_t = s.t
            else:
                if in_run:
                    longest = max(longest, prev_t - run_start)
                in_run = False
        if in_run:
            longest = max(longest, prev_t - run_start)
        return longest

    @staticmethod
    def _is_long(cls: VehicleClass) -> bool:
        return cls in ("truck_rigid", "truck_articulated", "road_train", "bus")

    # -- evidence brief (frozen payload to the agent) -----------------------

    def evidence_brief(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "follower_vehicle_class": self.follower_vehicle_class,
            "lead_vehicle_class": self.lead_vehicle_class,
            "follower_length_m": self.follower_length_m,
            "lead_length_m": self.lead_length_m,
            "duration_s": round(self.duration_s, 2),
            "environment": asdict(self.environment),
            "camera": asdict(self.camera),
            "aggregates": self.aggregates(),
            "representative_samples": [asdict(s) for s in self._reps(20)],
            "events": [asdict(e) for e in self.events],
        }

    def _reps(self, max_n: int) -> list[FrameSample]:
        usable = [
            s for s in self.samples
            if s.perception_confidence >= 0.5 and s.gap_m <= self.camera.max_reliable_range_m
        ]
        if len(usable) <= max_n:
            return usable
        smin = min(usable, key=lambda s: s.gap_s)
        smax = max(usable, key=lambda s: s.gap_s)
        picked = {id(smin), id(smax)}
        out = [smin, smax]
        step = len(usable) / (max_n - 2)
        i = 0.0
        while len(out) < max_n and int(i) < len(usable):
            s = usable[int(i)]
            if id(s) not in picked:
                out.append(s); picked.add(id(s))
            i += step
        return sorted(out, key=lambda s: s.t)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


# ---------------------------------------------------------------------------
# Reasoning agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an explainable NSW road-law reasoning agent in a fixed-traffic-camera enforcement-research pipeline. A roadside camera (CARLA Town04 highway, calibrated homography) observed a following pair: a FOLLOWER behind a LEAD vehicle. Upstream perception (YOLO26 detection -> ground-plane homography -> tracking) produced the structured evidence below. Apply NSW Road Rules 2014, primarily Rule 126, and Rule 127 where the geometry triggers it.

WHAT IS MEASURED (and its limits)
- Distances come from a calibrated ground-plane homography. Trust them within max_reliable_range_m; beyond that they are dropped.
- Speeds are estimated from vision (tracked displacement between frames), NOT from a speedometer. Treat them as having a few km/h of noise. The camera.speed_estimation field tells you the method.
- Two closeness metrics are provided:
    gap_s (= gap_m / follower_speed), where gap_m is REAR-TO-REAR: the distance between
      the two vehicles' rear ground-contact points. This counts the follower's own body
      length as part of the "gap".
    clearance_s (= (gap_m - follower_length) / follower_speed): the CLEAR ROAD between
      the vehicles, expressed in seconds. This is the stricter and arguably the correct
      reading of the "2/3-second rule", which concerns road ahead, not bumper positions.

    MEASUREMENT CONVENTION - you must address this explicitly. Reason PRIMARILY from
    clearance_s (and its sustained_clearance_* aggregates) as the governing test. Then
    state what gap_s would imply, and where the two diverge enough to change the verdict,
    say so plainly in the reasoning chain. The choice of convention is a real source of
    uncertainty in this evidence and a human adjudicator is entitled to see it, not have
    it silently resolved. Do not treat either figure as self-evidently correct.
    ttc_s (time-to-collision = gap_m / closing_speed): only present when the follower is actively closing; captures dynamic hazard.
  A short gap_s with infinite/absent TTC means steady close-following. A falling TTC means active closing - more dangerous.

NON-NEGOTIABLE REQUIREMENTS
1. Every legal claim cites evidence: an aggregate statistic name, a sample timestamp, or an event timestamp.
2. Propagate perception uncertainty (use gap_s_uncertainty_estimate; remember speeds are vision-estimated).
3. State failure modes explicitly (samples excluded for low confidence or range; vision-speed noise; short segments).
4. Provide a counterfactual: the smallest evidence change that flips the verdict.
5. Output machine-consumable JSON only - no markdown, no preamble.

LAW (verbatim)
Rule 126, Road Rules 2014 (NSW) - Keeping a safe distance behind vehicles:
"A driver must drive a sufficient distance behind a vehicle travelling in front of the driver so the driver can, if necessary, stop safely to avoid a collision with the vehicle."
Maximum penalty: 20 penalty units (court max $2,200). Penalty notice approx $464 + 3 demerit points (standard vehicle); higher (approx $1,636 + 4 demerit points) for heavy vehicles in some Safe-T-Cam zones.

Rule 127 - Minimum distance between long vehicles:
A driver of a long vehicle (7.5 m or longer, including any load) must stay at least 60 m behind another long vehicle (200 m if the follower is a road train), unless driving on a multi-lane road or in a built-up area, or overtaking (r 127(1)(a)-(b)).

SUFFICIENCY CALIBRATION (Rule 126 prescribes no fixed distance; the test is sufficiency to stop. NSW publishes this widely-applied calibration)
- 3 seconds in good conditions (dry, day, sealed)
- 4+ seconds in poor conditions (wet, gravel, night, fog)
- 5+ seconds for heavy vehicles or obscured forward view
- Rule 127: 60 m / 200 m are fixed minimum distances, unless an r 127(1) exemption applies

TRANSIENCY DOCTRINE
A brief dip below threshold (< 5 s, e.g. just after a cut_in) is not a breach. Sustained behaviour supports a breach: sustained_below_3s_seconds >= 5 s in good conditions, or sustained_below_2s_seconds >= 3 s, strongly supports Rule 126. A falling/low min_ttc_s while closing is an aggravating dynamic signal.

ESCALATION (consider; do not allege without strong evidence)
- Negligent driving (s117 Road Transport Act 2013 (NSW))
- Negligent driving occasioning GBH/death
- Dangerous driving (s52A/s52AB Crimes Act 1900 (NSW))
- Predatory driving (s51A Crimes Act 1900 (NSW)) - using a vehicle to menace/pursue (high_beam_flash, horn, sustained sub-2s closing)

OUTPUT FORMAT - RAW JSON ONLY
{
  "verdict": "likely_breach | possible_breach | unlikely_breach | insufficient_evidence",
  "confidence": "low | medium | high",
  "headline": "<= 25 words",
  "applicable_rules": [{"rule": "Rule 126", "reason": "<why engaged>", "text": "<verbatim>"}],
  "evidence_assessment": {
    "summary": "<2-3 sentences>",
    "key_statistics": [{"stat": "<aggregate name>", "value": "<value>", "interpretation": "<legal meaning>"}],
    "sample_citations": [{"sample_t": <float>, "frame_id": <int>, "claim": "<supported claim>"}],
    "event_citations": [{"event_t": <float>, "type": "<type>", "legal_significance": "<...>"}]
  },
  "perception_caveats": ["<e.g. vision-estimated speed +/- 3 km/h>", "<e.g. N samples dropped beyond range>"],
  "reasoning_chain": [{"step": 1, "claim": "<...>", "evidence_refs": ["aggregate.sustained_below_2s_seconds"], "legal_principle": "<...>"}],
  "aggravating_factors": [{"factor": "<...>", "evidence_ref": "<...>"}],
  "mitigating_factors": [{"factor": "<...>", "evidence_ref": "<...>"}],
  "likely_outcome": {"primary_charge": "Rule 126 | Rule 127 | no charge", "penalty_notice_estimate_aud": <int>, "demerit_points": <int>, "court_max_aud": <int>, "escalation_risk": "low|medium|high", "escalation_charges_to_consider": ["<charge>"]},
  "counterfactual": "<one sentence>",
  "explanation_for_lay_user": "<3-4 plain-English sentences>"
}

If usable_samples == 0, the segment is < 3 s, or camera warnings make distances untrustworthy, return verdict "insufficient_evidence" and explain in perception_caveats."""


@dataclass
class AgentResult:
    verdict: str
    confidence: str
    headline: str
    raw: dict

    @property
    def is_breach(self) -> bool:
        return self.verdict == "likely_breach"

    @property
    def is_inconclusive(self) -> bool:
        return self.verdict in ("possible_breach", "insufficient_evidence")


class TailgatingReasoningAgent:
    def __init__(self, api_key: Optional[str] = None,
                 model: str = "claude-sonnet-5", max_tokens: int = 8000,
                 temperature: float | None = None):
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        # Determinism note: non-determinism was observed at default sampling (the
        # same segment returned possible_breach on one run and likely_breach on
        # another). temperature=0 is the usual fix, but `temperature` is DEPRECATED
        # on newer models (e.g. claude-sonnet-5) and sending it is a 400 error.
        # So it is only sent when explicitly set; on models that reject it, leave
        # it None and use ensembling / escalate-on-disagreement instead.
        self.temperature = temperature
        self.parse_retries = 0   # how often JSON repair was needed

    def analyze(self, segment: FollowingSegment) -> AgentResult:
        payload = segment.evidence_brief()
        user = (
            "FROZEN TELEMETRY PAYLOAD (fixed traffic camera, CARLA Town04 + YOLO26 + homography):\n\n"
            + json.dumps(payload, indent=2)
            + "\n\nApply NSW Rule 126. Apply Rule 127 only if BOTH the FOLLOWING and LEAD "
            "vehicles are long vehicles (7.5 m or longer), and neither r 127(1) exemption "
            "applies (multi-lane road or built-up area; overtaking). "
            "Respond in the required JSON schema."
        )
        kwargs = dict(
            model=self.model, max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT, messages=[{"role": "user", "content": user}],
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", None) == "text")
        if not text.strip():
            # No text block at all. Usually means the token budget was consumed
            # before any output text was emitted (e.g. by reasoning blocks), or
            # the model stopped for another reason. Surface enough to diagnose.
            kinds = [getattr(b, "type", "?") for b in resp.content]
            raise ValueError(
                f"Agent returned no text content. stop_reason="
                f"{getattr(resp, 'stop_reason', '?')}, content block types={kinds}, "
                f"max_tokens={self.max_tokens}, usage="
                f"{getattr(resp, 'usage', None)}. "
                "If stop_reason is 'max_tokens', raise --max-tokens.")
        try:
            parsed = _robust_json_parse(text)
        except ValueError as first_err:
            # One repair attempt: hand the malformed output back and ask for
            # valid JSON only. Content is usually sound; only the serialisation
            # is broken, so this is cheaper than re-reasoning from scratch.
            self.parse_retries += 1
            repair = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content":
                        "That response was not valid JSON: "
                        f"{first_err}\n\nRe-emit the SAME analysis as a single "
                        "valid JSON object. No code, no method calls, no prose "
                        "outside the object. Do not change any judgement."},
                ],
            )
            text = "".join(getattr(b, "text", "") for b in repair.content
                           if getattr(b, "type", None) == "text")
            parsed = _robust_json_parse(text)
        return AgentResult(
            verdict=parsed.get("verdict", "insufficient_evidence"),
            confidence=parsed.get("confidence", "low"),
            headline=parsed.get("headline", ""),
            raw=parsed,
        )

    def analyze_batch(self, segments: list[FollowingSegment]) -> list[AgentResult]:
        return [self.analyze(s) for s in segments]


def _robust_json_parse(text: str) -> dict:
    """Extract the JSON object from a model response.

    The response may be bare JSON, or wrapped in a ```json ... ``` fence (open or
    closed). Slicing on the first "{" and last "}" handles all of these, and is
    robust to a closed fence - an earlier version split on "```" and took the last
    part, which for a *closed* fence is the empty string after it, silently
    discarding a perfectly good response.
    """
    raw = text
    text = text.strip()
    if not text:
        raise ValueError("Agent response was empty.")
    # Models occasionally splice code into string values, e.g.
    #   "...unlikely_breach.".replace(""," "),
    # which is syntactically invalid JSON. Strip trailing method calls that sit
    # between a closing quote and the following delimiter.
    text = re.sub(r'"\s*\.\s*\w+\([^)]*\)\s*(?=[,}\]])', '"', text)
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b == -1:
        raise ValueError(
            "Agent response was TRUNCATED mid-JSON (opening brace found, no closing "
            "brace). Raise max_tokens on TailgatingReasoningAgent.\n"
            f"Received {len(text)} chars, tail:\n...{text[-300:]}")
    if a == -1:
        raise ValueError(
            "No JSON object in agent response (no opening brace found).\n"
            f"Received {len(raw)} chars. First 400:\n{text[:400]}")
    try:
        return json.loads(text[a:b + 1])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Agent returned malformed JSON ({e}). Usually the response was cut "
            f"short or contains prose inside the object.\n"
            f"Payload {len(text)} chars. Around the error:\n"
            f"{text[max(0, e.pos - 200):e.pos + 200]}") from e


def format_result_for_log(result: AgentResult) -> str:
    r = result.raw
    out = ["=" * 72,
           f"VERDICT: {result.verdict.upper()} (confidence: {result.confidence})",
           f"  {result.headline}", "-" * 72, "REASONING:"]
    for step in r.get("reasoning_chain", []):
        out.append(f"  [{step.get('step')}] {step.get('claim')}")
        for ref in step.get("evidence_refs", []):
            out.append(f"        - cites: {ref}")
        if step.get("legal_principle"):
            out.append(f"        - principle: {step['legal_principle']}")
    out += ["-" * 72, "PERCEPTION CAVEATS:"]
    out += [f"  - {c}" for c in r.get("perception_caveats", [])]
    out += ["-" * 72, f"COUNTERFACTUAL: {r.get('counterfactual', '')}", "=" * 72]
    return "\n".join(out)


__all__ = [
    "FrameSample", "TailgatingEvent", "EnvironmentContext", "CameraMetadata",
    "FollowingSegment", "TailgatingReasoningAgent", "AgentResult",
    "format_result_for_log", "VehicleClass",
]