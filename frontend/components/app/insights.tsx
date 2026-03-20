"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import type { ModelInfo, SimState, TeamOverrides } from "@/components/app/types";
import {
  BG_ALT,
  BG_HEADER,
  BORDER_INNER,
  BORDER_OUTER,
  BORDER_SUBTLE,
  GREEN,
  PAD_BODY,
  PAD_HEADER,
  Seed,
  SURFACE,
  TEXT,
  TEXT_MUTED,
  TEXT_SUBTLE,
  PROGRESS_TRACK,
  pct,
  sumPct,
  toML,
  top,
} from "@/components/app/shared";
import {
  getDisagreement,
  getFirstFour,
  getHistoricalComps,
  getMatchup,
  getModelLog,
  getRegionStats,
  getSeedHistory,
  getTeams,
  getWhatIf,
} from "@/lib/api";

export function DiagnosticsSection({ sim, overrides }: { sim: SimState; overrides: TeamOverrides }) {
  if (!sim.complete) return null;

  const champTotal = sumPct(sim.champion_pct);
  const titleTotal = sumPct(sim.title_game_pct);
  const f4Total = sumPct(sim.final_four_pct);
  const e8Total = sumPct(sim.elite_eight_pct);

  const top4Champ = top(sim.champion_pct, 4).reduce((acc, [, p]) => acc + p, 0);
  const top8Champ = top(sim.champion_pct, 8).reduce((acc, [, p]) => acc + p, 0);
  const maxChamp = top(sim.champion_pct, 1)[0]?.[1] || 0;
  const maxTeam = top(sim.champion_pct, 1)[0]?.[0] || "N/A";
  const highUpset = (sim.upset_watch || []).filter((u: any) => u.upset_prob >= 35).length;
  const avgUpset = (sim.upset_watch || []).length
    ? (sim.upset_watch || []).reduce((acc: number, u: any) => acc + u.upset_prob, 0) / (sim.upset_watch || []).length
    : 0;
  const assumptionCount = Object.keys(overrides).length;
  const p = maxChamp / 100;
  const n = Math.max(sim.n_sims, 1);
  const moe = 1.96 * Math.sqrt((p * (1 - p)) / n) * 100;

  const checks = [
    { label: "Champion %", actual: champTotal, target: 100, tol: 1.5, hint: "All teams' championship % should sum to ~100% (exactly 1 winner)" },
    { label: "Title game %", actual: titleTotal, target: 200, tol: 2.5, hint: "All teams' title-game % should sum to ~200% (2 finalists)" },
    { label: "Final Four %", actual: f4Total, target: 400, tol: 4.0, hint: "All teams' Final Four % should sum to ~400% (4 F4 teams)" },
    { label: "Elite Eight %", actual: e8Total, target: 800, tol: 6.0, hint: "All teams' Elite Eight % should sum to ~800% (8 E8 teams)" },
  ];

  const statusColor = (ok: boolean) => (ok ? "#065f46" : "#7f1d1d");
  const statusBg = (ok: boolean) => (ok ? "#dcfce7" : "#fee2e2");

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>Verifies probabilities sum correctly (e.g. all championship %s ≈ 100%) and reports simulation precision.</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1fr" }}>
        <div style={{ padding: "12px 14px", borderRight: `1px solid ${BORDER_INNER}` }}>
          {checks.map((c) => {
            const delta = c.actual - c.target;
            const ok = Math.abs(delta) <= c.tol;
            return (
              <div key={c.label} title={c.hint} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "7px 0", borderBottom: `1px solid ${BORDER_SUBTLE}`, cursor: "help" }}>
                <div style={{ fontSize: 11, color: TEXT_MUTED }}>{c.label}</div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ fontSize: 10, color: TEXT_SUBTLE }}>{c.actual.toFixed(1)}% vs {c.target}%</span>
                  <span style={{ fontSize: 9, fontWeight: 700, padding: "2px 6px", borderRadius: 14, color: statusColor(ok), background: statusBg(ok) }}>
                    {ok ? "OK" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}pp`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{ padding: "12px 14px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px 12px" }}>
          {[
            { label: "Top-4 title share", value: pct(top4Champ, 1), hint: "Combined championship % of the top 4 teams" },
            { label: "Top-8 title share", value: pct(top8Champ, 1), hint: "Combined championship % of the top 8 teams" },
            { label: "Highest title odds", value: pct(maxChamp, 1), hint: `${maxTeam}'s chance to win the tournament` },
            { label: `${maxTeam} 95% MOE`, value: `±${moe.toFixed(2)}pp`, hint: "Margin of error at this sample size" },
            { label: "Avg upset chance", value: pct(avgUpset, 1), hint: "Average first-round upset probability across all R64 games" },
            { label: "35%+ upset games", value: String(highUpset), hint: "Number of R64 games where the underdog has ≥35% win chance" },
            { label: "Assumptions active", value: String(assumptionCount), hint: "Elo adjustments you applied before simulating" },
            { label: "Monte Carlo runs", value: sim.n_sims.toLocaleString(), hint: "Total bracket simulations run. More runs = higher precision." },
          ].map((m) => (
            <div key={m.label} title={m.hint} style={{ border: `1px solid ${BORDER_SUBTLE}`, padding: "8px 10px", cursor: "help" }}>
              <div style={{ fontSize: 9, letterSpacing: "0.06em", textTransform: "uppercase", color: TEXT_SUBTLE, marginBottom: 4 }}>{m.label}</div>
              <div style={{ fontSize: 16, fontWeight: 600, lineHeight: 1.1 }}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function DataProvenance({ info }: { info: ModelInfo | null }) {
  if (!info?.inference_data || !info?.training_data) return null;
  const inf = info.inference_data;
  const tr = info.training_data;
  const md = info.model_details;
  const stack = (md?.stack || ["logistic_regression", "xgboost", "lightgbm", "meta_logistic"]).join(" + ");
  const Row = ({ label, value }: { label: string; value: ReactNode }) => (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "6px 0", borderBottom: `1px solid ${BORDER_INNER}`, fontSize: 12 }}>
      <span style={{ color: TEXT_MUTED, flexShrink: 0 }}>{label}</span>
      <span style={{ color: TEXT, textAlign: "right" }}>{value}</span>
    </div>
  );
  return (
    <div style={{ padding: "14px 16px", background: SURFACE }}>
      <Row label="Bracket" value={<>{inf.dataset} · {inf.season} · {inf.teams_in_bracket} teams</>} />
      <Row label="Freeze date" value={inf.selection_sunday_freeze} />
      <Row label="Training" value={<>{tr.season_min} to {tr.season_max} · {tr.rows} rows</>} />
      <Row label="Recency" value={tr.recency_weighting} />
      <Row label="Stack" value={stack} />
      <Row label="Calibration" value={md?.calibration || "isotonic"} />
      <Row label="Features" value={<>{md?.core_feature_count ?? "n/a"} core + {md?.market_feature_count ?? "n/a"} market/meta</>} />
      <div style={{ paddingTop: 8, fontSize: 11, color: TEXT_MUTED, lineHeight: 1.5 }}>
        {md?.market_usage || "Vegas odds used as a feature in the model blend"} · {md?.variance_modeling || "Random per-team strength perturbations each sim run to capture game-day variance"}.
      </div>
    </div>
  );
}

export function TeamProbabilityMath({ sim, teams }: { sim: SimState; teams: string[] }) {
  const [team, setTeam] = useState(teams[0] || "");

  useEffect(() => {
    if (!teams.length) {
      setTeam("");
      return;
    }
    if (!team || !teams.includes(team)) {
      setTeam(teams[0]);
    }
  }, [teams, team]);

  if (teams.length === 0 || !team) return null;

  const n = Math.max(sim.n_sims, 1);
  const rows = [
    { label: "Reach Round of 32", value: sim.round_of_32_pct?.[team] || 0 },
    { label: "Reach Sweet 16", value: sim.sweet_sixteen_pct?.[team] || 0 },
    { label: "Reach Elite Eight", value: sim.elite_eight_pct?.[team] || 0 },
    { label: "Reach Final Four", value: sim.final_four_pct?.[team] || 0 },
    { label: "Reach Championship Game", value: sim.title_game_pct?.[team] || 0 },
    { label: "Win Championship", value: sim.champion_pct?.[team] || 0 },
  ];

  const explain = (value: number) => {
    const p = Math.max(0, Math.min(1, value / 100));
    const expectedHits = p * n;
    const se = Math.sqrt((p * (1 - p)) / n);
    const moe = 1.96 * se;
    const lo = Math.max(0, p - moe);
    const hi = Math.min(1, p + moe);
    return { p, expectedHits, lo, hi };
  };

  const r32 = (sim.round_of_32_pct?.[team] || 0) / 100;
  const s16 = (sim.sweet_sixteen_pct?.[team] || 0) / 100;
  const e8 = (sim.elite_eight_pct?.[team] || 0) / 100;
  const f4 = (sim.final_four_pct?.[team] || 0) / 100;
  const fin = (sim.title_game_pct?.[team] || 0) / 100;
  const ch = (sim.champion_pct?.[team] || 0) / 100;
  const pathBuckets = [
    { label: "Out in Round of 64", p: Math.max(0, 1 - r32) },
    { label: "Out in Round of 32", p: Math.max(0, r32 - s16) },
    { label: "Out in Sweet 16", p: Math.max(0, s16 - e8) },
    { label: "Out in Elite Eight", p: Math.max(0, e8 - f4) },
    { label: "Out in Final Four", p: Math.max(0, f4 - fin) },
    { label: "Lose Championship Game", p: Math.max(0, fin - ch) },
    { label: "Win Championship", p: Math.max(0, ch) },
  ];
  const likelyOutcome = [...pathBuckets].sort((a, b) => b.p - a.p)[0];

  return (
    <div style={{ background: SURFACE }}>
      <div style={{ padding: PAD_HEADER, borderBottom: `1px solid ${BORDER_SUBTLE}`, background: BG_HEADER }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <span style={{ fontSize: 10, color: TEXT_SUBTLE }}>Team</span>
            <select value={team} onChange={(e) => setTeam(e.target.value)} style={{ height: 30, padding: "0 10px", border: `1px solid ${BORDER_SUBTLE}`, fontSize: 11, background: SURFACE, minWidth: 180 }}>
              {teams.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>
        </div>
        <div style={{ fontSize: 11, color: TEXT_MUTED, lineHeight: 1.6, marginTop: 8 }}>
          Plays out every game thousands of times with randomized team-strength noise. No two brackets are the same. Percentages show how often each outcome occurred.
        </div>
        <div style={{ fontSize: 10, color: TEXT_SUBTLE, marginTop: 4 }}>
          Model: Logistic Regression + XGBoost + LightGBM ensemble with isotonic calibration, trained on 77 first-round games (2005 to 2025).
        </div>
      </div>
      {!sim.complete ? (
        <div style={{ padding: PAD_BODY, fontSize: 10, color: TEXT_SUBTLE }}>
          Run a simulation to see round-by-round probabilities.
        </div>
      ) : (
        <>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ borderBottom: `1px solid ${BORDER_SUBTLE}`, background: BG_HEADER }}>
                  <th style={{ padding: "8px 10px", textAlign: "left", fontSize: 9, color: TEXT_MUTED, letterSpacing: "0.06em" }}>OUTCOME</th>
                  <th title="% of runs where this team reached this round" style={{ padding: "8px 10px", textAlign: "right", fontSize: 9, color: TEXT_MUTED, letterSpacing: "0.06em", cursor: "help" }}>CHANCE</th>
                  <th title="Average times this happens in N runs (e.g., 1,480 / 10,000)" style={{ padding: "8px 10px", textAlign: "right", fontSize: 9, color: TEXT_MUTED, letterSpacing: "0.06em", cursor: "help" }}>EXPECTED HITS</th>
                  <th title="95% confidence interval for the true probability" style={{ padding: "8px 10px", textAlign: "right", fontSize: 9, color: TEXT_MUTED, letterSpacing: "0.06em", cursor: "help" }}>95% CI</th>
                  <th title="Implied American moneyline odds (e.g., +650 means bet $100 to win $650)" style={{ padding: "8px 10px", textAlign: "right", fontSize: 9, color: TEXT_MUTED, letterSpacing: "0.06em", cursor: "help" }}>FAIR ODDS</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r, idx) => {
                  const m = explain(r.value);
                  const fairOdds = m.p <= 0 || m.p >= 1 ? "N/A" : toML(m.p);
                  return (
                    <tr key={r.label} style={{ borderBottom: `1px solid ${BORDER_SUBTLE}`, background: idx % 2 === 0 ? SURFACE : BG_HEADER }}>
                      <td style={{ padding: "8px 10px", fontWeight: 500 }}>{r.label}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", fontWeight: 600 }}>{pct(r.value, 2)}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", color: "#555" }}>{m.expectedHits.toFixed(0)} / {n.toLocaleString()}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", color: "#555" }}>{pct(m.lo * 100, 2)} - {pct(m.hi * 100, 2)}</td>
                      <td style={{ padding: "8px 10px", textAlign: "right", color: "#555" }}>{fairOdds}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{ padding: PAD_BODY, borderTop: `1px solid ${BORDER_SUBTLE}`, display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12, flexWrap: "wrap", background: BG_HEADER }}>
            <div style={{ fontSize: 10, color: TEXT_SUBTLE, letterSpacing: "0.06em", textTransform: "uppercase" }}>Most likely finish</div>
            <div style={{ fontSize: 12, fontWeight: 600 }}>
              {likelyOutcome?.label || "N/A"} <span style={{ fontSize: 10, color: TEXT_MUTED, fontWeight: 500 }}>({pct((likelyOutcome?.p || 0) * 100, 2)})</span>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export function ModelTrainingOutput() {
  const [log, setLog] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [fetched, setFetched] = useState(false);
  const [height, setHeight] = useState(280);
  const dragRef = useRef<{ y: number; h: number } | null>(null);

  const load = async () => {
    if (fetched) return;
    setLoading(true);
    try {
      const response = await getModelLog();
      setLog(response.log);
      setFetched(true);
    } catch {
      setLog("Failed to load.");
    }
    setLoading(false);
  };

  const onDrag = (e: React.MouseEvent) => {
    dragRef.current = { y: e.clientY, h: height };
    const move = (ev: MouseEvent) => {
      if (dragRef.current) {
        setHeight(Math.max(80, Math.min(800, dragRef.current.h + ev.clientY - dragRef.current.y)));
      }
    };
    const up = () => {
      dragRef.current = null;
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  return (
    <div style={{ marginTop: 12, border: `1px solid ${BORDER_INNER}`, borderRadius: 14, background: BG_HEADER, overflow: "hidden" }}>
      <div style={{ padding: "10px 12px", display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>Training log</span>
        {!fetched && !loading && (
          <button onClick={load} style={{ padding: "4px 12px", background: "#0f172a", color: "#fff", border: "none", fontSize: 10, fontWeight: 600, cursor: "pointer", borderRadius: 14 }}>Load</button>
        )}
      </div>
      {loading && <div style={{ padding: "14px", fontSize: 11, color: "#aaa" }}>Loading precomputed model summary...</div>}
      {log && (
        <>
          <div style={{ height, overflowY: "auto", overflowX: "auto", background: "#1a1a1a", padding: "12px 16px", fontSize: 11, lineHeight: 1.8, whiteSpace: "pre", fontFamily: "ui-monospace,monospace" }}>
            {log.split("\n").map((line, i) => (
              <div key={i} style={{ color: line.trim().startsWith("✓") ? "#86efac" : line.trim().startsWith("✗") ? "#f87171" : /\d+\.\d{4}/.test(line) ? "#94a3b8" : "#a1a1aa" }}>{line || " "}</div>
            ))}
          </div>
          <div onMouseDown={onDrag} style={{ height: 6, background: BORDER_SUBTLE, cursor: "ns-resize", display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div style={{ width: 32, height: 2, background: "#bbb", borderRadius: 14 }} />
          </div>
        </>
      )}
    </div>
  );
}

export function AnalysisSection({ allTeams }: { allTeams: string[] }) {
  const [a, setA] = useState(allTeams[0] || "Duke");
  const [b, setB] = useState(allTeams[4] || "UConn");
  const [matchup, setMatchup] = useState<any>(null);
  const [disagree, setDisagree] = useState<any>(null);
  const [comps, setComps] = useState<any>(null);
  const [whatif, setWhatif] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  const run = async () => {
    if (a === b) return;
    setLoading(true);
    try {
      const [matchupData, disagreementData, compsData, whatIfData] = await Promise.all([
        getMatchup(a, b),
        getDisagreement(a, b),
        getHistoricalComps(a, b),
        getWhatIf(a, b),
      ]);
      setMatchup(matchupData);
      setDisagree(disagreementData);
      setComps(compsData);
      setWhatif(whatIfData.results || []);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  const selectStyle = { flex: 1, height: 30, background: SURFACE, border: `1px solid ${BORDER_OUTER}`, fontSize: 11, padding: "0 8px", appearance: "none" as const, cursor: "pointer" };

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>Head-to-head win probability from the ML ensemble. Signal breakdown shows which factors favor each team. Right panel tests scenario shifts.</div>
      </div>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, display: "flex", gap: 10, alignItems: "center" }}>
        <select value={a} onChange={(e) => setA(e.target.value)} style={selectStyle}>{allTeams.map((team) => <option key={team}>{team}</option>)}</select>
        <span style={{ fontSize: 11, color: TEXT_SUBTLE }}>vs</span>
        <select value={b} onChange={(e) => setB(e.target.value)} style={selectStyle}>{allTeams.map((team) => <option key={team}>{team}</option>)}</select>
        <button onClick={run} disabled={loading || a === b} style={{ height: 30, padding: "0 16px", background: "#000", color: "#fff", border: "none", fontSize: 10, fontWeight: 600, cursor: "pointer", letterSpacing: "0.06em", whiteSpace: "nowrap" }}>{loading ? "..." : "ANALYZE"}</button>
      </div>
      {matchup && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
          <div style={{ borderRight: `1px solid ${BORDER_INNER}` }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", borderBottom: `1px solid ${BORDER_SUBTLE}` }}>
              {[{ name: a, prob: matchup.win_prob_a, score: matchup.score_a }, { name: b, prob: matchup.win_prob_b, score: matchup.score_b }].map(({ name, prob, score }, i) => (
                <div key={name} style={{ padding: "16px", borderRight: i === 0 ? `1px solid ${BORDER_SUBTLE}` : "none", background: prob > 50 ? BG_ALT : SURFACE }}>
                  <div style={{ fontSize: 10, color: TEXT_SUBTLE, marginBottom: 4 }}>{name}</div>
                  <div style={{ fontSize: 36, fontWeight: 300, lineHeight: 1 }}>{prob.toFixed(1)}%</div>
                  {score && <div title="Simulated final score from the model" style={{ fontSize: 10, color: "#bbb", marginTop: 4, cursor: "help" }}>sim: {score}</div>}
                  {prob > 50 && <div style={{ fontSize: 9, fontWeight: 600, marginTop: 8, letterSpacing: "0.1em" }}>PROJECTED WINNER</div>}
                </div>
              ))}
            </div>
            {matchup.score_note && (
              <div style={{ padding: "8px 16px", borderBottom: `1px solid ${BORDER_SUBTLE}`, fontSize: 10, color: TEXT_SUBTLE }}>
                {matchup.score_note}
              </div>
            )}
            {disagree && (
              <div style={{ padding: "12px 16px", borderBottom: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ fontSize: 9, letterSpacing: "0.1em", fontWeight: 600, color: TEXT_SUBTLE, marginBottom: 4 }}>SIGNAL BREAKDOWN</div>
                <div style={{ fontSize: 8, color: "#bbb", marginBottom: 10 }}>Bar direction and value show each category&apos;s deviation from 50/50</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5px 20px" }}>
                  {Object.entries(disagree.signals).map(([key, value]: any) => {
                    const adv = value - 50;
                    return (
                      <div key={key} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <span style={{ fontSize: 9, color: TEXT_SUBTLE, minWidth: 72 }}>{key.replace(/_/g, " ")}</span>
                        <div style={{ flex: 1, height: 2, background: BORDER_SUBTLE, position: "relative" }}>
                          <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#bbb" }} />
                          <div style={{ position: "absolute", height: "100%", background: "#000", left: adv >= 0 ? "50%" : `${50 + adv}%`, width: `${Math.abs(adv)}%` }} />
                        </div>
                        <span style={{ fontSize: 9, minWidth: 24, textAlign: "right" }}>{adv > 0 ? "+" : ""}{adv.toFixed(0)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            {comps?.comps && (
              <div style={{ padding: "12px 16px" }}>
                <div style={{ fontSize: 9, letterSpacing: "0.1em", fontWeight: 600, color: TEXT_SUBTLE, marginBottom: 8 }}>HISTORICAL COMPS: {a}-analog won {((comps.historical_win_rate || 0) * 100).toFixed(0)}% of {comps.comps.length} similar games</div>
                {comps.comps.slice(0, 4).map((comp: any, i: number) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "4px 0", borderBottom: `1px solid ${BORDER_SUBTLE}`, fontSize: 10 }}>
                    <span style={{ color: "#ccc", minWidth: 30 }}>{comp.season}</span>
                    <span style={{ fontWeight: comp.team_a_won ? 600 : 400, flex: 1 }}>({comp.seed_a}) {comp.team_a}</span>
                    <span style={{ color: "#ccc" }}>vs</span>
                    <span style={{ fontWeight: !comp.team_a_won ? 600 : 400, flex: 1, textAlign: "right" }}>({comp.seed_b}) {comp.team_b}</span>
                    {comp.upset && <span style={{ fontSize: 8, border: "1px solid #d1d5db", padding: "0 3px" }}>U</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <div style={{ padding: "8px 14px", borderBottom: `1px solid ${BORDER_SUBTLE}`, background: BG_HEADER }}>
              <div style={{ fontSize: 9, letterSpacing: "0.1em", fontWeight: 600, color: TEXT_SUBTLE }}>WHAT IF: {a} WIN PROBABILITY</div>
              <div style={{ fontSize: 9, color: "#aaa", marginTop: 2 }}>How {a}&apos;s odds shift under different game-day scenarios</div>
            </div>
            {whatif.map((result: any, i: number) => {
              const base = whatif[0]?.new_prob || 50;
              const delta = result.new_prob - base;
              const isBase = i === 0;
              const absDelta = Math.abs(delta);
              const deltaColor = delta > 1.5 ? "#16a34a" : delta < -1.5 ? "#dc2626" : "#9ca3af";
              const barMax = Math.max(...whatif.map((r: any) => Math.abs(r.new_prob - base)), 1);
              return (
                <div key={i} style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_SUBTLE}`, background: isBase ? BG_ALT : SURFACE }}>
                  <div style={{ display: "flex", alignItems: "flex-start", gap: 10 }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 11, fontWeight: isBase ? 700 : 500, color: isBase ? "#111" : "#333" }}>{result.scenario}</div>
                      {result.desc && <div style={{ fontSize: 9, color: "#9ca3af", marginTop: 2, lineHeight: 1.4 }}>{result.desc}</div>}
                    </div>
                    <div style={{ textAlign: "right", flexShrink: 0, minWidth: 72 }}>
                      <div style={{ fontSize: 16, fontWeight: isBase ? 700 : 400, color: isBase ? "#111" : "#333", lineHeight: 1 }}>{result.new_prob.toFixed(1)}%</div>
                      {!isBase && (
                        <div style={{ fontSize: 11, fontWeight: 700, color: deltaColor, marginTop: 3 }}>
                          {delta > 0 ? "+" : ""}{delta.toFixed(1)}pp
                        </div>
                      )}
                      {isBase && <div style={{ fontSize: 9, color: "#9ca3af", marginTop: 3 }}>baseline</div>}
                    </div>
                  </div>
                  {!isBase && (
                    <div style={{ marginTop: 6, display: "flex", alignItems: "center", gap: 6 }}>
                      <div style={{ flex: 1, height: 4, background: PROGRESS_TRACK, borderRadius: 14, position: "relative", overflow: "hidden" }}>
                        <div style={{ position: "absolute", left: "50%", top: 0, bottom: 0, width: 1, background: "#d4d4d4" }} />
                        {delta !== 0 && (
                          <div style={{
                            position: "absolute",
                            top: 0,
                            bottom: 0,
                            borderRadius: 14,
                            background: deltaColor,
                            left: delta >= 0 ? "50%" : `${50 - (absDelta / barMax) * 50}%`,
                            width: `${(absDelta / barMax) * 50}%`,
                            transition: "all 0.3s ease",
                          }} />
                        )}
                      </div>
                      <span style={{ fontSize: 9, color: "#aaa", flexShrink: 0, minWidth: 40, textAlign: "right" }}>
                        {delta > 0 ? "helps" : delta < 0 ? "hurts" : "neutral"}
                      </span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export function FirstFourSection({ sim }: { sim: SimState }) {
  const ff = sim.first_four_pct || {};
  const [games, setGames] = useState<any[]>([]);

  useEffect(() => {
    getFirstFour().then((data) => setGames(data.games || [])).catch(() => {});
  }, []);

  if (games.length === 0) return null;

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        {games.map((game: any, i: number) => {
          const pa = ff[game.team_a] || 50;
          const pb = ff[game.team_b] || 50;
          const tot = pa + pb || 100;
          const pA = (pa / tot) * 100;
          const pB = (pb / tot) * 100;
          const winner = pA >= pB ? game.team_a : game.team_b;
          return (
            <div key={i} style={{ padding: "12px 14px", borderRight: i % 2 === 0 ? `1px solid ${BORDER_INNER}` : "none", borderBottom: i < 2 ? `1px solid ${BORDER_INNER}` : "none" }}>
              <div style={{ fontSize: 9, color: TEXT_SUBTLE, marginBottom: 8 }}>{game.region} {game.seed}-seed, winner plays {game.plays}</div>
              {[{ name: game.team_a, p: pA }, { name: game.team_b, p: pB }].map(({ name, p: probability }) => (
                <div key={name} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                  <span style={{ flex: 1, fontSize: 11, fontWeight: name === winner ? 600 : 400, color: name === winner ? "#000" : "#999" }}>{name}</span>
                  <span style={{ fontSize: 10 }}>{pct(probability)}</span>
                  {name === winner && <span style={{ fontSize: 9 }}>→</span>}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function ModelVsMarket({ sim }: { sim: SimState }) {
  const [teams, setTeams] = useState<Record<string, any>>({});

  useEffect(() => {
    getTeams().then(setTeams).catch(() => {});
  }, []);

  if (!sim.complete || Object.keys(teams).length === 0) return null;

  const rows = Object.entries(sim.champion_pct)
    .filter(([, probability]) => probability > 0.3)
    .map(([name, modelPct]) => {
      const team = teams[name];
      const vegasPct = team?.championship_odds_pct || 0;
      const edge = modelPct - vegasPct;
      return { name, modelPct, vegasPct, edge, seed: team?.seed || 16 };
    })
    .sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge));

  const maxModel = Math.max(...rows.map((row) => row.modelPct), 1);

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>MODEL = simulated championship %. VEGAS = implied % from pre-tournament futures. EDGE = difference (positive = model more bullish).</div>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr style={{ borderBottom: "2px solid #000", background: "#000" }}>
              <th style={{ padding: "8px 10px", textAlign: "left", color: "#fff", fontSize: 9, fontWeight: 600 }}>TEAM</th>
              <th style={{ padding: "8px 8px", textAlign: "center", color: "#fff", fontSize: 9, fontWeight: 600 }}>MODEL</th>
              <th style={{ padding: "8px 8px", textAlign: "center", color: "#fff", fontSize: 9, fontWeight: 600 }}>VEGAS</th>
              <th style={{ padding: "8px 10px", textAlign: "center", color: GREEN, fontSize: 9, fontWeight: 600 }}>EDGE</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 16).map((row, i) => {
              const edgeColor = row.edge > 2 ? "#000" : row.edge < -2 ? "#999" : "#ccc";
              return (
                <tr key={row.name} style={{ borderBottom: `1px solid ${BORDER_SUBTLE}`, background: i % 2 === 0 ? SURFACE : BG_HEADER }}>
                  <td style={{ padding: "6px 10px", whiteSpace: "nowrap" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                      <Seed n={row.seed} />
                      <span style={{ fontWeight: row.seed <= 4 ? 600 : 400 }}>{row.name}</span>
                    </span>
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "center" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 6, justifyContent: "center" }}>
                      <div style={{ width: 60, height: 4, background: PROGRESS_TRACK, borderRadius: 14, overflow: "hidden" }}>
                        <div style={{ width: `${(row.modelPct / maxModel) * 100}%`, height: "100%", background: "#000", borderRadius: 14 }} />
                      </div>
                      <span style={{ minWidth: 40 }}>{pct(row.modelPct, 1)}</span>
                    </div>
                  </td>
                  <td style={{ padding: "6px 8px", textAlign: "center", color: TEXT_MUTED }}>{pct(row.vegasPct, 1)}</td>
                  <td style={{ padding: "6px 10px", textAlign: "center", fontWeight: 600, color: edgeColor }}>
                    {row.edge > 0 ? "+" : ""}{pct(row.edge, 1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function RegionDifficulty({ sim }: { sim: SimState }) {
  const [stats, setStats] = useState<any[]>([]);

  useEffect(() => {
    getRegionStats().then(setStats).catch(() => {});
  }, []);

  if (!sim.complete || stats.length === 0) return null;

  const maxEM = Math.max(...stats.map((s: any) => Math.abs(s.avg_em)), 1);
  const maxChamp = Math.max(...stats.map((s: any) => s.total_championship_pct), 1);

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>Efficiency margin is KenPom-style (pts/100 possessions above avg). Combined title odds = total championship % in the region.</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)" }}>
        {stats.map((stat: any, i: number) => {
          const difficulty = stat.total_championship_pct;
          const rank = [...stats].sort((a: any, b: any) => b.total_championship_pct - a.total_championship_pct).findIndex((x: any) => x.region === stat.region) + 1;
          const label = rank === 1 ? "TOUGHEST" : rank === 4 ? "EASIEST" : "";
          return (
            <div key={stat.region} style={{ padding: "14px 16px", borderRight: i < 3 ? `1px solid ${BORDER_INNER}` : "none" }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 4, marginBottom: 10 }}>
                <div style={{ fontSize: 11, fontWeight: 700 }}>{stat.region.toUpperCase()}</div>
                {label && <span style={{ fontSize: 8, fontWeight: 600, letterSpacing: "0.08em", padding: "2px 6px", border: `1px solid ${BORDER_OUTER}`, background: rank === 1 ? "#000" : SURFACE, color: rank === 1 ? "#fff" : TEXT, alignSelf: "flex-start" }}>{label}</span>}
              </div>
              <div style={{ fontSize: 9, color: TEXT_SUBTLE, marginBottom: 4 }}>Avg. efficiency margin</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                <div style={{ flex: 1, height: 4, background: PROGRESS_TRACK, borderRadius: 14, overflow: "hidden" }}>
                  <div style={{ width: `${(Math.abs(stat.avg_em) / maxEM) * 100}%`, height: "100%", background: "#000", borderRadius: 14 }} />
                </div>
                <span style={{ fontSize: 11, fontWeight: 600, minWidth: 32 }}>{stat.avg_em > 0 ? "+" : ""}{stat.avg_em}</span>
              </div>
              <div style={{ fontSize: 9, color: TEXT_SUBTLE, marginBottom: 4 }}>Combined title odds</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 12 }}>
                <div style={{ flex: 1, height: 4, background: PROGRESS_TRACK, borderRadius: 14, overflow: "hidden" }}>
                  <div style={{ width: `${(difficulty / maxChamp) * 100}%`, height: "100%", background: rank <= 2 ? "#000" : "#bbb", borderRadius: 14 }} />
                </div>
                <span style={{ fontSize: 11, fontWeight: 600, minWidth: 32 }}>{pct(difficulty, 0)}</span>
              </div>
              <div style={{ fontSize: 9, color: TEXT_SUBTLE, marginBottom: 6 }}>Top seeds</div>
              {stat.top_seeds.map((team: any) => (
                <div key={team.name} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "3px 0", borderBottom: `1px solid ${BORDER_SUBTLE}` }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                    <Seed n={team.seed} />
                    <span style={{ fontSize: 10, fontWeight: 500 }}>{team.name}</span>
                  </span>
                  <span style={{ fontSize: 9, color: TEXT_SUBTLE }}>{pct(team.championship_odds_pct, 1)}</span>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function SeedHistory({ sim }: { sim: SimState }) {
  const [history, setHistory] = useState<Record<string, number>>({});

  useEffect(() => {
    getSeedHistory().then(setHistory).catch(() => {});
  }, []);

  if (!sim.complete || Object.keys(history).length === 0) return null;

  const upsets = (sim.upset_watch || []).slice(0, 8);
  const entries = Object.entries(history)
    .map(([seed, rate]) => ({ seed: parseInt(seed, 10), rate }))
    .sort((a, b) => a.seed - b.seed);

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>First-round win rate by seed (1985 to 2024). Compare this year&apos;s predicted upset chances against historical averages.</div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr" }}>
        <div style={{ padding: "14px 16px", borderRight: `1px solid ${BORDER_INNER}` }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {entries.map(({ seed, rate }) => (
              <div key={seed} style={{ display: "grid", gridTemplateColumns: "28px 1fr 44px", alignItems: "center", gap: 8 }}>
                <Seed n={seed} />
                <div style={{ height: 5, background: PROGRESS_TRACK, borderRadius: 14, overflow: "hidden" }}>
                  <div style={{ width: `${rate * 100}%`, height: "100%", background: rate > 0.8 ? "#000" : rate > 0.5 ? "#555" : rate > 0.3 ? "#999" : "#ccc", borderRadius: 14 }} />
                </div>
                <span style={{ fontSize: 10, fontWeight: rate > 0.5 ? 600 : 400, textAlign: "right" }}>{(rate * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ padding: "14px 16px" }}>
          <div style={{ fontSize: 9, letterSpacing: "0.1em", color: TEXT_SUBTLE, marginBottom: 10, textTransform: "uppercase" }}>This year&apos;s upset candidates vs history</div>
          {upsets.map((upset: any, i: number) => {
            const historicalUpsetRate = 1 - (history[String(upset.dog_seed)] || 0);
            return (
              <div key={i} style={{ marginBottom: 8, paddingBottom: 8, borderBottom: `1px solid ${BORDER_SUBTLE}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 3 }}>
                  <span style={{ fontSize: 10, fontWeight: 600 }}>({upset.dog_seed}) {upset.underdog} over ({upset.fav_seed}) {upset.favorite}</span>
                  <span style={{ fontSize: 10, fontWeight: 600 }}>{pct(upset.upset_prob, 0)}</span>
                </div>
                <div style={{ fontSize: 9, color: TEXT_SUBTLE }}>
                  Historical: {upset.dog_seed}-seeds win {(historicalUpsetRate * 100).toFixed(0)}% of R64 games
                  {upset.upset_prob > historicalUpsetRate * 100 + 5 && (
                    <span style={{ color: "#000", fontWeight: 600 }}> (above-average upset chance)</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

const SEED_BASELINE_CHAMP: Record<number, number> = {
  1: 12.5, 2: 5.0, 3: 3.0, 4: 1.8, 5: 0.8, 6: 0.5, 7: 0.4, 8: 0.2,
  9: 0.15, 10: 0.1, 11: 0.08, 12: 0.05, 13: 0.01, 14: 0.005, 15: 0.002, 16: 0.001,
};

export function BracketValuePicks({ sim, teamsCatalog }: { sim: SimState; teamsCatalog: Record<string, any> }) {
  if (!sim.complete) return null;

  const picks = Object.entries(sim.champion_pct)
    .filter(([, p]) => p > 0.05)
    .map(([name, champPct]) => {
      const team = teamsCatalog[name];
      const seed = team?.seed || 16;
      const region = team?.region || "?";
      const baseline = SEED_BASELINE_CHAMP[seed] || 0.01;
      const valueRatio = champPct / baseline;
      const f4Pct = sim.final_four_pct?.[name] || 0;
      const vegasChamp = team?.championship_odds_pct || 0;
      return { name, seed, region, champPct, f4Pct, baseline, valueRatio, vegasChamp };
    })
    .filter(t => t.seed >= 5 && t.champPct > 0.3)
    .sort((a, b) => b.valueRatio - a.valueRatio);

  if (picks.length === 0) return null;

  return (
    <div style={{ border: `1px solid ${BORDER_OUTER}`, background: SURFACE }}>
      <div style={{ padding: "10px 14px", borderBottom: `1px solid ${BORDER_INNER}`, background: BG_HEADER }}>
        <div style={{ fontSize: 10, color: TEXT_MUTED }}>
          Mid-seeds and underdogs whose simulated championship odds beat their historical seed average. High-ceiling bracket picks.
        </div>
      </div>
      <div style={{ padding: "10px 14px" }}>
        {picks.slice(0, 8).map((t, i) => {
          const multiplier = t.valueRatio.toFixed(1);
          const hot = t.valueRatio >= 3;
          return (
            <div key={t.name} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", borderBottom: i < picks.length - 1 ? `1px solid ${BORDER_SUBTLE}` : "none" }}>
              <Seed n={t.seed} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 11, fontWeight: 600 }}>{t.name}</span>
                  <span style={{ fontSize: 8, color: "#9ca3af", letterSpacing: "0.06em" }}>{t.region}</span>
                </div>
                <div style={{ fontSize: 9, color: "#9ca3af", marginTop: 2 }}>
                  F4: {pct(t.f4Pct, 1)} · Champ: {pct(t.champPct, 1)} · Vegas: {t.vegasChamp > 0 ? pct(t.vegasChamp, 1) : "n/a"}
                </div>
              </div>
              <div style={{ textAlign: "right", flexShrink: 0 }}>
                <div style={{ fontSize: 14, fontWeight: 700, color: hot ? TEXT : TEXT_MUTED }}>{multiplier}x</div>
                <div style={{ fontSize: 8, color: hot ? "#16a34a" : "#9ca3af", fontWeight: 600 }}>vs seed avg</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
