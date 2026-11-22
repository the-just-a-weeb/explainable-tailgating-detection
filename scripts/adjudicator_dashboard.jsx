import { useState } from "react";

// Stage 5: Meaningful Human Control - adjudicator dashboard.
// Paste a frozen telemetry payload (the evidence_brief() JSON from the
// perception pipeline) and the agent returns an evidence-cited verdict for
// human review. The HITL adjudicator can then confirm or override.

const SYSTEM_PROMPT = `You are an explainable NSW road-law reasoning agent in a fixed-traffic-camera enforcement-research pipeline (CARLA Town04). A roadside camera observed a following pair (FOLLOWER behind LEAD). Perception (YOLO26 -> homography -> tracking) produced the structured evidence. Distances are from a calibrated ground homography (trust within max_reliable_range_m). Speeds are vision-estimated (a few km/h of noise), not from a speedometer. Two metrics: gap_s (time headway = gap_m/follower_speed) for the Rule 126 stop-safely test, and ttc_s (time-to-collision = gap_m/closing_speed, present only when closing) for dynamic hazard.

Apply NSW Rule 126 and Rule 127 (long vehicles >7.5m outside built-up areas: 60m, road trains 200m). Calibration: 3s good conditions, 4s+ poor, 5s+ heavy vehicles. Transiency: brief dips (<5s, e.g. after a cut_in) are not breaches; sustained_below_3s_seconds>=5s or sustained_below_2s_seconds>=3s supports a breach.

CRITICAL: cite evidence (aggregate name / sample_t / event_t) for every claim; propagate uncertainty; state failure modes; give a counterfactual.

Respond as RAW JSON only with keys: verdict (likely_breach|possible_breach|unlikely_breach|insufficient_evidence), confidence (low|medium|high), headline, applicable_rules (array of {rule, reason, text}), evidence_assessment ({summary, key_statistics:[{stat,value,interpretation}], sample_citations:[{sample_t,frame_id,claim}], event_citations:[{event_t,type,legal_significance}]}), perception_caveats (array), reasoning_chain (array of {step, claim, evidence_refs, legal_principle}), aggravating_factors (array of {factor, evidence_ref}), mitigating_factors (array of {factor, evidence_ref}), likely_outcome ({primary_charge, penalty_notice_estimate_aud, demerit_points, court_max_aud, escalation_risk, escalation_charges_to_consider}), counterfactual, explanation_for_lay_user.`;

const SAMPLE_EVIDENCE = {
  segment_id: "seg_trk_000001_trk_000000_0",
  follower_vehicle_class: "car",
  lead_vehicle_class: "truck_articulated",
  follower_length_m: 4.6,
  lead_length_m: 16.0,
  duration_s: 9.95,
  environment: { weather: "rain_light", road_surface: "wet", road_class: "motorway", speed_limit_kmh: 110, is_built_up_area: false, is_safe_t_cam_zone: false },
  camera: { site_id: "TOWN04_HW_01", calibration_rmse_m: 0.5, max_reliable_range_m: 120.0, speed_estimation: "vision_tracked", warnings: [] },
  aggregates: {
    usable_samples: 199, total_samples: 200, excluded_low_confidence: 1, excluded_out_of_range: 0,
    mean_gap_s: 0.795, median_gap_s: 0.803, min_gap_s: 0.752, p10_gap_s: 0.753,
    mean_gap_m: 22.15, min_gap_m: 21.0, mean_follower_speed_kmh: 99.8, max_follower_speed_kmh: 101.8,
    sustained_below_2s_seconds: 9.9, sustained_below_3s_seconds: 9.9, sustained_below_4s_seconds: 9.9,
    min_ttc_s: null, mean_ttc_s: null, seconds_closing: 0.0, sustained_ttc_below_3s_seconds: 0.0,
    sustained_below_60m_seconds: null, gap_s_uncertainty_estimate: 0.018,
  },
  representative_samples: [
    { t: 0.05, frame_id: 1001, follower_speed_kmh: 99.8, lead_speed_kmh: 99.8, gap_m: 22.0, gap_s: 0.794, v_rel_kmh: 0.0, ttc_s: null, follower_track_id: "trk_000001", lead_track_id: "trk_000000", perception_confidence: 0.9, gap_uncertainty_m: 0.5 },
    { t: 5.0, frame_id: 1100, follower_speed_kmh: 100.2, lead_speed_kmh: 100.0, gap_m: 22.6, gap_s: 0.812, v_rel_kmh: 0.2, ttc_s: null, follower_track_id: "trk_000001", lead_track_id: "trk_000000", perception_confidence: 0.9, gap_uncertainty_m: 0.5 },
    { t: 9.9, frame_id: 1199, follower_speed_kmh: 99.9, lead_speed_kmh: 99.9, gap_m: 21.5, gap_s: 0.775, v_rel_kmh: 0.0, ttc_s: null, follower_track_id: "trk_000001", lead_track_id: "trk_000000", perception_confidence: 0.9, gap_uncertainty_m: 0.5 },
  ],
  events: [],
};

export default function AdjudicatorDashboard() {
  const [evidenceJson, setEvidenceJson] = useState(JSON.stringify(SAMPLE_EVIDENCE, null, 2));
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [decision, setDecision] = useState(null); // 'confirmed' | 'overridden'

  const analyze = async () => {
    setLoading(true); setError(null); setResult(null); setDecision(null);
    try { JSON.parse(evidenceJson); }
    catch (e) { setError("Invalid JSON: " + e.message); setLoading(false); return; }
    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-20250514",
          max_tokens: 3000,
          system: SYSTEM_PROMPT,
          messages: [{ role: "user", content: `FROZEN TELEMETRY PAYLOAD:\n\n${evidenceJson}\n\nApply NSW Rule 126 (and 127 if both long & outside built-up area). Respond in the required JSON schema.` }],
        }),
      });
      if (!response.ok) throw new Error("Agent request failed (" + response.status + ")");
      const data = await response.json();
      const text = data.content.filter(b => b.type === "text").map(b => b.text).join("");
      const a = text.indexOf("{"), b = text.lastIndexOf("}");
      if (a === -1) throw new Error("No JSON in response.");
      setResult(JSON.parse(text.slice(a, b + 1)));
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const vColor = (v) => v === "likely_breach" ? "var(--oxblood)" : v === "possible_breach" ? "var(--amber)" : v === "unlikely_breach" ? "var(--forest)" : "var(--ink-muted)";
  const vLabel = (v) => ({ likely_breach: "Likely breach", possible_breach: "Possible breach", unlikely_breach: "Unlikely breach", insufficient_evidence: "Insufficient evidence" }[v] || v);

  let parsed = null;
  try { parsed = JSON.parse(evidenceJson); } catch { /* */ }

  const renderChart = () => {
    if (!parsed?.representative_samples?.length || parsed.representative_samples.length < 2) return null;
    const s = parsed.representative_samples;
    const W = 720, H = 200, PAD = 40;
    const tMax = Math.max(...s.map(d => d.t));
    const gMax = Math.max(...s.map(d => d.gap_s), 4);
    const xs = t => PAD + (t / tMax) * (W - 2 * PAD);
    const ys = g => H - PAD - (g / gMax) * (H - 2 * PAD);
    const path = s.map((d, i) => `${i ? "L" : "M"} ${xs(d.t).toFixed(1)} ${ys(d.gap_s).toFixed(1)}`).join(" ");
    return (
      <div style={{ marginTop: 18, padding: "16px 18px", background: "var(--bg-panel)", border: "1px solid var(--line-soft)" }}>
        <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Time-headway profile (gap_s)</div>
        <svg width="100%" viewBox={`0 0 ${W} ${H}`}>
          {[2, 3].map(thr => (
            <g key={thr}>
              <line x1={PAD} y1={ys(thr)} x2={W - PAD} y2={ys(thr)} stroke={thr === 2 ? "var(--oxblood)" : "var(--forest)"} strokeWidth="1" strokeDasharray="4 4" />
              <text x={W - PAD + 4} y={ys(thr) + 4} fontSize="10" fill={thr === 2 ? "var(--oxblood)" : "var(--forest)"} fontFamily="JetBrains Mono">{thr.toFixed(1)}s</text>
            </g>
          ))}
          <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="var(--line)" />
          <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="var(--line)" />
          <path d={path} fill="none" stroke="var(--ink)" strokeWidth="1.5" />
          {s.map((d, i) => <circle key={i} cx={xs(d.t)} cy={ys(d.gap_s)} r="3" fill={d.gap_s < 2 ? "var(--oxblood)" : d.gap_s < 3 ? "var(--amber)" : "var(--forest)"} />)}
          <text x={PAD} y={H - PAD + 14} fontSize="10" fill="var(--ink-muted)" fontFamily="JetBrains Mono">t=0s</text>
          <text x={W - PAD - 30} y={H - PAD + 14} fontSize="10" fill="var(--ink-muted)" fontFamily="JetBrains Mono">t={tMax.toFixed(1)}s</text>
        </svg>
      </div>
    );
  };

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Manrope:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        :root { --bg:#f4ede2; --bg-panel:#faf6ee; --bg-deep:#ebe2d2; --ink:#14110c; --ink-soft:#2a2520; --ink-muted:#6b5d4a; --line:#c4b59b; --line-soft:#d9cdb5; --oxblood:#8b1a1a; --amber:#a65f00; --forest:#2d5016; --navy:#1e3a5f; }
        .display{font-family:'Instrument Serif',serif;font-weight:400} .display-italic{font-family:'Instrument Serif',serif;font-style:italic}
        .body{font-family:'Manrope',sans-serif} .mono{font-family:'JetBrains Mono',monospace}
        .smallcaps{font-family:'Manrope',sans-serif;text-transform:uppercase;letter-spacing:.18em;font-size:11px;font-weight:600}
        .field{background:var(--bg-panel);border:1px solid var(--line-soft);color:var(--ink);font-family:'JetBrains Mono',monospace;font-size:12px;line-height:1.5;padding:10px 12px;width:100%;outline:none}
        .field:focus{border-color:var(--ink)}
        textarea.field{resize:vertical}
        .btn{background:var(--ink);color:var(--bg);font-family:'Manrope',sans-serif;font-weight:600;font-size:13px;text-transform:uppercase;letter-spacing:.14em;padding:14px 28px;border:1px solid var(--ink);cursor:pointer;transition:all .18s}
        .btn:hover:not(:disabled){background:var(--oxblood);border-color:var(--oxblood)} .btn:disabled{opacity:.4;cursor:not-allowed}
        .btn-decision{font-family:'Manrope',sans-serif;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.12em;padding:12px 22px;border:1px solid currentColor;cursor:pointer;background:transparent;transition:all .15s}
        .btn-decision:hover{background:var(--bg-deep)}
        .rule-quote{background:var(--bg-deep);border-left:3px solid var(--navy);padding:14px 18px;font-family:'Instrument Serif',serif;font-style:italic;font-size:17px;line-height:1.5;color:var(--ink-soft)}
        .divider{border-top:1px solid var(--line);margin:28px 0}
        .pill{display:inline-block;font-family:'Manrope',sans-serif;font-size:11px;font-weight:600;letter-spacing:.12em;text-transform:uppercase;padding:4px 10px;border:1px solid currentColor}
        .rcard{padding:14px 18px;background:var(--bg-panel);border-left:2px solid var(--ink);margin-bottom:10px}
        .rnum{font-family:'JetBrains Mono',monospace;font-size:11px;color:var(--ink-muted);display:block;margin-bottom:4px}
        .rclaim{font-family:'Manrope',sans-serif;font-size:14.5px;line-height:1.5;color:var(--ink);margin-bottom:6px}
        .eref{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:10.5px;padding:1px 6px;margin:2px 4px 2px 0;background:var(--bg-deep);color:var(--ink-soft);border:1px solid var(--line-soft)}
        .principle{font-family:'Instrument Serif',serif;font-style:italic;font-size:13px;color:var(--ink-muted);margin-top:6px;display:block}
        .srow{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dotted var(--line-soft);font-family:'Manrope',sans-serif;font-size:13.5px}
        .srow:last-child{border-bottom:none}
        .sname{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ink-muted)} .sval{font-weight:600;color:var(--ink)} .sint{color:var(--ink-soft);font-style:italic;flex:1;padding:0 12px;font-size:12.5px}
        .cf{background:linear-gradient(to right,var(--bg-deep),var(--bg-panel));border:1px solid var(--navy);padding:18px 22px;margin-top:8px}
        .caveat{font-family:'JetBrains Mono',monospace;font-size:11.5px;color:var(--ink-muted);padding:6px 0;border-left:2px solid var(--amber);padding-left:10px;margin-bottom:4px}
        @keyframes pd{0%,100%{opacity:.3}50%{opacity:1}} .ld{animation:pd 1.4s infinite} .ld:nth-child(2){animation-delay:.2s} .ld:nth-child(3){animation-delay:.4s}
      `}</style>

      <div style={{ maxWidth: 960, margin: "0 auto", padding: "48px 32px 96px" }}>
        <header style={{ borderBottom: "2px solid var(--ink)", paddingBottom: 20, marginBottom: 32 }}>
          <div className="mono" style={{ fontSize: 10, letterSpacing: ".15em", color: "var(--ink-muted)", marginBottom: 12 }}>
            STAGE 5 — MEANINGFUL HUMAN CONTROL · TOWN04 SAFE-T-CAM RESEARCH
          </div>
          <h1 className="display" style={{ fontSize: 50, lineHeight: 1.02, margin: 0, color: "var(--ink)" }}>
            Adjudicator <span className="display-italic">review</span>
          </h1>
          <p className="body" style={{ marginTop: 14, fontSize: 15, color: "var(--ink-muted)", lineHeight: 1.55, maxWidth: 680 }}>
            A schema-validated evidentiary report from the reasoning agent, for human-in-the-loop review. Every claim is traced to perception evidence; the counterfactual states what would change the verdict. The adjudicator confirms or overrides — no enforcement action is automatic.
          </p>
        </header>

        <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Frozen telemetry payload (evidence_brief JSON)</div>
        <textarea className="field" style={{ minHeight: 240 }} value={evidenceJson} onChange={e => setEvidenceJson(e.target.value)} />
        {renderChart()}

        <div style={{ marginTop: 24, display: "flex", alignItems: "center", gap: 20 }}>
          <button className="btn" onClick={analyze} disabled={loading}>{loading ? "Reasoning..." : "Generate report"}</button>
          {loading && <span className="mono" style={{ fontSize: 12, color: "var(--ink-muted)" }}>applying Rule 126<span className="ld">.</span><span className="ld">.</span><span className="ld">.</span></span>}
        </div>

        {error && <div style={{ marginTop: 20, padding: "12px 14px", border: "1px solid var(--oxblood)", color: "var(--oxblood)", fontFamily: "Manrope,sans-serif", fontSize: 13 }}>{error}</div>}

        {result && (
          <section style={{ marginTop: 52 }}>
            <div style={{ borderTop: "2px solid var(--ink)", paddingTop: 20 }}>
              <div className="mono" style={{ fontSize: 10, letterSpacing: ".15em", color: "var(--ink-muted)", marginBottom: 8 }}>EVIDENTIARY REPORT</div>
              <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 20, flexWrap: "wrap" }}>
                <h2 className="display" style={{ fontSize: 28, lineHeight: 1.15, margin: 0, color: "var(--ink)", maxWidth: 640 }}>{result.headline}</h2>
                <span className="pill" style={{ color: vColor(result.verdict), whiteSpace: "nowrap" }}>{vLabel(result.verdict)} · {result.confidence}</span>
              </div>
            </div>

            {result.perception_caveats?.length > 0 && (
              <><div className="divider" /><div>
                <div className="smallcaps" style={{ color: "var(--amber)", marginBottom: 8 }}>Perception caveats</div>
                {result.perception_caveats.map((c, i) => <div key={i} className="caveat">{c}</div>)}
              </div></>
            )}

            <div className="divider" />
            <div>
              <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Applicable rule(s)</div>
              {(result.applicable_rules || []).map((r, i) => (
                <div key={i} style={{ marginBottom: 16 }}>
                  <div className="body" style={{ fontSize: 13.5, color: "var(--navy)", fontWeight: 600, marginBottom: 6 }}>{r.rule}</div>
                  <p className="body" style={{ fontSize: 14, color: "var(--ink-soft)", margin: "0 0 10px" }}>{r.reason}</p>
                  <div className="rule-quote">"{r.text}"</div>
                </div>
              ))}
            </div>

            {result.evidence_assessment && (
              <><div className="divider" /><div>
                <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Evidence assessment</div>
                <p className="body" style={{ fontSize: 14.5, lineHeight: 1.6, color: "var(--ink-soft)", margin: "0 0 16px" }}>{result.evidence_assessment.summary}</p>
                {result.evidence_assessment.key_statistics && (
                  <div style={{ background: "var(--bg-panel)", border: "1px solid var(--line-soft)", padding: "8px 16px" }}>
                    {result.evidence_assessment.key_statistics.map((s, i) => (
                      <div key={i} className="srow"><span className="sname">{s.stat}</span><span className="sval">{s.value}</span><span className="sint">{s.interpretation}</span></div>
                    ))}
                  </div>
                )}
              </div></>
            )}

            <div className="divider" />
            <div>
              <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Reasoning chain</div>
              {(result.reasoning_chain || []).map((s, i) => (
                <div key={i} className="rcard">
                  <span className="rnum">STEP {s.step || i + 1}</span>
                  <div className="rclaim">{s.claim}</div>
                  {s.evidence_refs?.length > 0 && <div style={{ marginTop: 6 }}>{s.evidence_refs.map((r, j) => <span key={j} className="eref">{r}</span>)}</div>}
                  {s.legal_principle && <span className="principle">— {s.legal_principle}</span>}
                </div>
              ))}
            </div>

            <div className="divider" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
              <div>
                <div className="smallcaps" style={{ color: "var(--oxblood)", marginBottom: 10 }}>Aggravating</div>
                {(result.aggravating_factors || []).length ? result.aggravating_factors.map((f, i) => (
                  <div key={i} style={{ fontSize: 13.5, padding: "6px 0", color: "var(--ink-soft)", borderBottom: "1px dotted var(--line-soft)" }}>
                    <div>{typeof f === "string" ? f : f.factor}</div>
                    {f.evidence_ref && <span className="eref">{f.evidence_ref}</span>}
                  </div>
                )) : <p className="body" style={{ fontSize: 13, color: "var(--ink-muted)", fontStyle: "italic" }}>None on these facts.</p>}
              </div>
              <div>
                <div className="smallcaps" style={{ color: "var(--forest)", marginBottom: 10 }}>Mitigating</div>
                {(result.mitigating_factors || []).length ? result.mitigating_factors.map((f, i) => (
                  <div key={i} style={{ fontSize: 13.5, padding: "6px 0", color: "var(--ink-soft)", borderBottom: "1px dotted var(--line-soft)" }}>
                    <div>{typeof f === "string" ? f : f.factor}</div>
                    {f.evidence_ref && <span className="eref">{f.evidence_ref}</span>}
                  </div>
                )) : <p className="body" style={{ fontSize: 13, color: "var(--ink-muted)", fontStyle: "italic" }}>None on these facts.</p>}
              </div>
            </div>

            {result.likely_outcome && typeof result.likely_outcome === "object" && (
              <><div className="divider" /><div>
                <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Likely outcome</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: "12px 24px" }}>
                  <div><span className="sname">Primary charge</span><div className="sval">{result.likely_outcome.primary_charge}</div></div>
                  <div><span className="sname">Penalty notice</span><div className="sval">${result.likely_outcome.penalty_notice_estimate_aud} AUD</div></div>
                  <div><span className="sname">Demerit points</span><div className="sval">{result.likely_outcome.demerit_points}</div></div>
                  <div><span className="sname">Court max</span><div className="sval">${result.likely_outcome.court_max_aud} AUD</div></div>
                  <div style={{ gridColumn: "1 / -1" }}><span className="sname">Escalation risk</span><div className="sval">{result.likely_outcome.escalation_risk}</div>
                    {result.likely_outcome.escalation_charges_to_consider?.length > 0 && <div style={{ marginTop: 4 }}>{result.likely_outcome.escalation_charges_to_consider.map((c, i) => <span key={i} className="eref">{c}</span>)}</div>}
                  </div>
                </div>
              </div></>
            )}

            {result.counterfactual && (
              <><div className="divider" /><div>
                <div className="smallcaps" style={{ color: "var(--navy)", marginBottom: 8 }}>Counterfactual — how the verdict would flip</div>
                <div className="cf"><p className="display" style={{ fontStyle: "italic", fontSize: 20, lineHeight: 1.4, margin: 0, color: "var(--ink-soft)" }}>{result.counterfactual}</p></div>
              </div></>
            )}

            {result.explanation_for_lay_user && (
              <><div className="divider" /><div style={{ background: "var(--bg-panel)", border: "1px solid var(--line-soft)", padding: "22px 24px" }}>
                <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 10 }}>Plain English</div>
                <p className="body" style={{ fontSize: 15, lineHeight: 1.65, color: "var(--ink-soft)", margin: 0 }}>{result.explanation_for_lay_user}</p>
              </div></>
            )}

            {/* HITL decision - the Stage 5 control point */}
            <div className="divider" />
            <div>
              <div className="smallcaps" style={{ color: "var(--ink-muted)", marginBottom: 12 }}>Adjudicator decision</div>
              {decision ? (
                <div className="body" style={{ fontSize: 15, color: decision === "confirmed" ? "var(--oxblood)" : "var(--forest)", fontWeight: 600 }}>
                  {decision === "confirmed"
                    ? "✓ Violation confirmed — enforcement action: penalty notice issued."
                    : "✕ Case overridden — no enforcement action recorded."}
                  <button className="btn-decision" style={{ marginLeft: 16, color: "var(--ink-muted)" }} onClick={() => setDecision(null)}>Reset</button>
                </div>
              ) : (
                <div style={{ display: "flex", gap: 12 }}>
                  <button className="btn-decision" style={{ color: "var(--oxblood)" }} onClick={() => setDecision("confirmed")}>Confirm violation</button>
                  <button className="btn-decision" style={{ color: "var(--forest)" }} onClick={() => setDecision("overridden")}>Override / dismiss</button>
                </div>
              )}
            </div>

            <p className="body" style={{ marginTop: 28, fontSize: 12, color: "var(--ink-muted)", fontStyle: "italic", textAlign: "center" }}>
              Research pipeline. General information only — not legal advice.
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
