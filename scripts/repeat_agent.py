"""
Repeat-run the reasoning agent to measure verdict stability
===========================================================

Sends the SAME frozen evidence payload (from a cases.json / batch_cases.json) to
the agent N times and reports how often the verdict, confidence and convention
handling agree. Uses the stored payload directly, so every run sees byte-identical
input - any variation is the model, not the pipeline.

    # see what it would cost, no calls
    python scripts/repeat_agent.py batch/batch_cases.json --cases A --runs 5 --dry-run

    # Case A x5, Cases B,C x3
    python scripts/repeat_agent.py batch/batch_cases.json --cases A --runs 5
    python scripts/repeat_agent.py batch/batch_cases.json --cases B,C --runs 3

Cases can be given by label (A, B, ...) or full segment ID.
Writes <input>_repeats.json (all raw verdicts) and prints a stability table.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))
sys.path.insert(0, str(ROOT / "scripts"))

from reasoning_agent import TailgatingReasoningAgent  # noqa
from batch_cases import UsageMeter  # noqa


class FrozenPayload:
    """Duck-types FollowingSegment for agent.analyze(): only evidence_brief() is used."""
    def __init__(self, brief: dict):
        self._brief = brief

    def evidence_brief(self) -> dict:
        return self._brief


def mentions_convention(v: dict) -> bool:
    txt = json.dumps(v).lower()
    return "gap_s" in txt and "clearance_s" in txt


def main(path, case_keys, runs, model, max_tokens, price_in, price_out, dry_run, yes):
    doc = json.loads(Path(path).read_text())
    cases = doc["cases"]
    by_key = {}
    for i, c in enumerate(cases):
        by_key[c["case_id"]] = c
        by_key[c.get("label") or chr(65 + i)] = c
    chosen = []
    for k in case_keys:
        if k not in by_key:
            raise SystemExit(f"Unknown case '{k}'. Available: "
                             + ", ".join(f"{c.get('label', '?')}={c['case_id']}" for c in cases))
        chosen.append(by_key[k])

    n_calls = len(chosen) * runs
    est = n_calls * (7000 / 1e6 * price_in + 4500 / 1e6 * price_out)
    print(f"{len(chosen)} case(s) x {runs} run(s) = {n_calls} call(s), est. ~${est:.2f} "
          f"(~7k in / ~4.5k out tokens per call at ${price_in}/${price_out} per M)")
    if dry_run:
        print("[--dry-run] no calls made.")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set.")
    if not yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Aborted - no API calls made.")

    agent = TailgatingReasoningAgent(model=model, max_tokens=max_tokens)
    meter = UsageMeter(agent, price_in, price_out)

    out_path = Path(path).with_name(Path(path).stem + "_repeats.json")
    record = json.loads(out_path.read_text()) if out_path.exists() else {"runs": []}

    results = {}
    for c in chosen:
        label = c.get("label", c["case_id"])
        payload = FrozenPayload(c["evidence"])
        vs = []
        for r in range(1, runs + 1):
            print(f"  {label} run {r}/{runs} ...", end=" ", flush=True)
            try:
                res = agent.analyze(payload)
                v = {"verdict": res.verdict, "confidence": res.confidence,
                     "headline": res.headline, **res.raw}
                print(f"{res.verdict} ({res.confidence})")
            except Exception as e:
                v = {"verdict": "ERROR", "confidence": "-", "error": str(e)}
                print(f"ERROR: {e}")
            vs.append(v)
            record["runs"].append({
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "case_id": c["case_id"], "label": label, "model": model, "verdict": v})
            out_path.write_text(json.dumps(record, indent=2))   # save after every call
        results[label] = (c, vs)

    print("\nSTABILITY (identical input each run)")
    print(f"{'case':5s} {'n':>2s}  {'verdicts':45s} {'confidence':28s} {'modal %':>7s} {'conv. noted':>11s}")
    for label, (c, vs) in results.items():
        ok = [v for v in vs if v["verdict"] != "ERROR"]
        vc = Counter(v["verdict"] for v in ok)
        cc = Counter(v["confidence"] for v in ok)
        modal = vc.most_common(1)[0][1] / len(ok) * 100 if ok else 0
        conv = sum(mentions_convention(v) for v in ok)
        print(f"{label:5s} {len(ok):2d}  {', '.join(f'{k} {n}' for k, n in vc.items()):45s} "
              f"{', '.join(f'{k} {n}' for k, n in cc.items()):28s} {modal:6.0f}% {conv:>5d}/{len(ok)}")
        if len(ok) < len(vs):
            print(f"      {len(vs) - len(ok)} run(s) errored")
    print(f"\nJSON repair retries: {agent.parse_retries}")
    print(f"API: {meter.line()}")
    print(f"All verdicts saved to {out_path.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases_json")
    p.add_argument("--cases", required=True, help="comma list of labels or segment IDs, e.g. A,B")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--model", default="claude-sonnet-5")
    p.add_argument("--max-tokens", type=int, default=16000)
    p.add_argument("--price-in", type=float, default=3.0)
    p.add_argument("--price-out", type=float, default=15.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true")
    a = p.parse_args()
    main(a.cases_json, [k.strip() for k in a.cases.split(",") if k.strip()], a.runs,
         a.model, a.max_tokens, a.price_in, a.price_out, a.dry_run, a.yes)
