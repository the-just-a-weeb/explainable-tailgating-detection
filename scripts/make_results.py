import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open(sys.argv[1]))
for c in d["cases"]:
    e, a, v = c["evidence"], c["evidence"]["aggregates"], c["verdict"]
    print(f"\n== {c['case_id']}  {e['follower_vehicle_class']} behind {e['lead_vehicle_class']}  {e['duration_s']}s  {e['environment']['weather']}")
    rows = [("min headway (s)", a["min_gap_s"], a["min_clearance_s"]),
            ("median headway (s)", a["median_gap_s"], a["median_clearance_s"]),
            ("mean headway (s)", a["mean_gap_s"], a["mean_clearance_s"]),
            ("sustained < 3 s (s)", a["sustained_below_3s_seconds"], a["sustained_clearance_below_3s_seconds"]),
            ("sustained < 2 s (s)", a["sustained_below_2s_seconds"], a["sustained_clearance_below_2s_seconds"])]
    for n, g, cl in rows: print(f"  {n:22s} gap={g:<7} clear={cl:<7} delta={round(cl-g,3)}")
    for k in ["usable_samples","mean_follower_speed_kmh","max_follower_speed_kmh","mean_gap_m","min_gap_m",
              "min_ttc_s","mean_ttc_s","seconds_closing","sustained_ttc_below_3s_seconds","sustained_below_4s_seconds","gap_s_uncertainty_estimate"]:
        print(f"  {k} = {a[k]}")
    from collections import Counter
    print("  events:", dict(Counter(x["type"] for x in e["events"])))
    print(f"  VERDICT: {v['verdict']} ({v['confidence']})" if v else "  VERDICT: none in this file (run merge_verdicts.py)")

c = d["cases"][int(sys.argv[2]) if len(sys.argv) > 2 else 0]; s = c["evidence"]["representative_samples"]
t = [x["t"] for x in s]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
ax.axhline(3.0, color="#888", ls="--", lw=1.2, label="3.0 s guidance threshold")
ax.axhline(2.0, color="#bbb", ls=":", lw=1.2, label="2.0 s")
ax.plot(t, [x["gap_s"] for x in s], "o-", color="#1f4e9c", lw=1.8, ms=4, label="Rear-to-rear headway $h_{gap}$")
ax.plot(t, [x["clearance_s"] for x in s], "s-", color="#c0392b", lw=1.8, ms=4, label="Clear-road headway $h_{clear}$")
for ev in c["evidence"]["events"]:
    if ev["type"] == "lead_brake": ax.axvline(ev["t"], color="#2e8b57", alpha=.35, lw=1)
ax.set_ylim(0, 6); ax.set_xlim(-0.1, max(t) + 0.3)
ax.set_xlabel("Time within segment (s)"); ax.set_ylabel("Headway (s)")
ax.set_title(f"{c.get('label','Case')} ({c['case_id']}): {c['evidence']['follower_vehicle_class']} behind {c['evidence']['lead_vehicle_class']}", fontsize=10)
ax.plot([], [], color="#2e8b57", alpha=.5, label="lead_brake event")
ax.legend(fontsize=7.5, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(sys.argv[3] if len(sys.argv) > 3 else "fig_convention_divergence.png")
print("\nsaved fig3_3_convention_divergence.png  (green lines = lead_brake events; plotted from", len(s), "representative samples)")