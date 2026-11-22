"""
Merge agent verdicts from <cases>_repeats.json back into the case file so the
adjudication desk shows them. Uses the MOST RECENT run per case by default.

    python scripts/merge_verdicts.py batch_eval/batch_cases.json
    -> writes batch_eval/cases.json (open adjudication_desk.html from that folder)
"""
import json, sys
from pathlib import Path

src = Path(sys.argv[1])
rep = src.with_name(src.stem + "_repeats.json")
out = src.with_name(sys.argv[2] if len(sys.argv) > 2 else "cases.json")
doc = json.loads(src.read_text())
runs = json.loads(rep.read_text())["runs"] if rep.exists() else []

latest, counts = {}, {}
for r in runs:
    if r["verdict"].get("verdict") == "ERROR":
        continue
    latest[r["case_id"]] = r
    counts.setdefault(r["case_id"], []).append(r["verdict"]["verdict"])

n = 0
for c in doc["cases"]:
    r = latest.get(c["case_id"])
    if r:
        c["verdict"] = r["verdict"]
        c["repeat_verdicts"] = counts[c["case_id"]]
        n += 1
doc["model"] = next(iter(latest.values()))["model"] if latest else doc.get("model")
out.write_text(json.dumps(doc, indent=2))
print(f"Merged {n}/{len(doc['cases'])} verdict(s) -> {out.resolve()}")
for c in doc["cases"]:
    print(f"  {c.get('label','?')}: {c['case_id']}  runs={c.get('repeat_verdicts', '-')}")
