"use client";
import { useState, useEffect, useRef, useCallback, memo } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";
const GREEN = "#22c55e";
const ACCENT = "#000";
const ACCENT_SOFT = "#fafafa";
const PANEL_BORDER = "#e5e7eb";
const PANEL_DIVIDER = "#f1f5f9";
const ROUNDS = ["R32","S16","E8","F4","Final","Champ"] as const;
const REGIONS = ["East","West","Midwest","South"] as const;
const R64_ORDER: [number, number][] = [[1,16],[8,9],[5,12],[4,13],[6,11],[3,14],[7,10],[2,15]];

// ── Types ─────────────────────────────────────────────────────────────────────
interface Game {
  team_a: string; team_b: string; seed_a: number; seed_b: number;
  win_prob_a: number; winner: string; upset: boolean;
  expected_margin: number; score_a?: number; score_b?: number;
}
interface SimState {
  champion_pct: Record<string,number>;
  title_game_pct: Record<string,number>;
  final_four_pct: Record<string,number>;
  elite_eight_pct: Record<string,number>;
  sweet_sixteen_pct: Record<string,number>;
  round_of_32_pct: Record<string,number>;
  first_four_pct: Record<string,number>;
  predicted_bracket: Record<string, Game[][]>;
  upset_watch: any[];
  matchup_probs: Record<string,number>;
  n_sims: number; done: number; total: number;
  model_used: string; elapsed_sec?: number;
  complete: boolean; bracketStarted: boolean;
  sims_per_sec?: number;
  first_four_games?: any[];
  teams?: Record<string,any>;
  seed_history?: Record<number,number>;
  region_stats?: any[];
}
interface ModelInfo {
  inference_data?: {
    dataset?: string;
    season?: number;
    selection_sunday_freeze?: string;
    teams_in_bracket?: number;
  };
  training_data?: {
    season_min?: number;
    season_max?: number;
    rows?: number;
    recency_weighting?: string;
  };
  model_details?: {
    stack?: string[];
    calibration?: string;
    core_feature_count?: number;
    market_feature_count?: number;
    market_usage?: string;
    variance_modeling?: string;
  };
}
type Resolved = Record<string, Record<number, Record<number, boolean>>>;
type TeamOverrides = Record<string, number>;
type ForcedPicks = Record<string, string>;
type SimRunConfig = {
  n_sims:number;
  latent_sigma:number;
  team_overrides: TeamOverrides;
  forced_picks: ForcedPicks;
};
const pickKey = (region: string, round: number, gi: number) => `${region}:${round}:${gi}`;

const EMPTY: SimState = {
  champion_pct:{}, title_game_pct:{}, final_four_pct:{}, elite_eight_pct:{},
  sweet_sixteen_pct:{}, round_of_32_pct:{}, first_four_pct:{},
  predicted_bracket:{}, upset_watch:[], matchup_probs:{},
  n_sims:0, done:0, total:10000,
  model_used:"", complete:false, bracketStarted:false,
};

const top = (o: Record<string,number>, n=10) =>
  Object.entries(o).sort((a,b)=>b[1]-a[1]).slice(0,n);
const pct = (n:number, d=1) => `${n.toFixed(d)}%`;
const clampProb = (p:number) => Math.min(0.999, Math.max(0.001, p));
const toML = (p: number) => {
  const q = clampProb(p);
  return q >= 0.5 ? `-${Math.round((q/(1-q))*100)}` : `+${Math.round(((1-q)/q)*100)}`;
};
const sumPct = (o: Record<string, number>) =>
  Object.values(o).reduce((acc, v) => acc + v, 0);

// ── Shared UI ─────────────────────────────────────────────────────────────────
function SectionGap() { return <div style={{marginTop:14}}/>; }
function SH({ label }: { label:string }) {
  return <div style={{background:"#000",padding:"7px 14px"}}><span style={{fontSize:10,fontWeight:600,letterSpacing:"0.1em",color:"#fff"}}>{label}</span></div>;
}
function Seed({ n }: { n:number }) {
  return <span style={{display:"inline-flex",alignItems:"center",justifyContent:"center",width:17,height:17,fontSize:9,fontWeight:600,flexShrink:0,border:"1px solid #d1d5db",background:n<=4?"#111":"transparent",color:n<=4?"#fff":"#111"}}>{n}</span>;
}
function GroupTitle({ label, hint }: { label: string; hint?: string }) {
  return (
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",padding:"2px 2px 0"}}>
      <div style={{fontSize:10,fontWeight:600,letterSpacing:"0.08em",textTransform:"uppercase",color:"#6b7280"}}>{label}</div>
      {hint && <div style={{fontSize:10,color:"#9ca3af"}}>{hint}</div>}
    </div>
  );
}

// Collapsible section
function Collapse({ label, children, defaultOpen=false }: { label:string; children:React.ReactNode; defaultOpen?:boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{border:"1px solid #ebeff5",marginTop:0,background:"#fff",borderRadius:12,overflow:"hidden"}}>
      <button onClick={()=>setOpen(o=>!o)} style={{
        width:"100%",padding:"10px 12px",background:open?"#fafafa":"#fff",
        border:"none",cursor:"pointer",display:"flex",alignItems:"center",
        justifyContent:"space-between",fontSize:11,fontWeight:600,letterSpacing:"0.02em"
      }}>
        <span>{label}</span>
        <span style={{fontSize:14,color:"#999",fontWeight:300}}>{open?"−":"+"}</span>
      </button>
      {open && <div style={{borderTop:`1px solid ${PANEL_DIVIDER}`}}>{children}</div>}
    </div>
  );
}

// ── Sim Controls ──────────────────────────────────────────────────────────────
function SimControls({ running, onRun, assumptionCount, lockedPickCount }: {
  running: boolean;
  onRun: (cfg: {n_sims:number; latent_sigma:number}) => void;
  assumptionCount: number;
  lockedPickCount: number;
}) {
  const [nSims, setNSims] = useState(10000);
  const [sigma, setSigma] = useState(0.06);

  const NSIMS_OPTS = [1000,2500,5000,10000,25000];
  const nSimsIdx = Math.max(0, NSIMS_OPTS.indexOf(nSims));
  const SIGMA_LABELS: Record<number,string> = {
    0.02:"Low variance (chalky)",
    0.04:"Mild",
    0.06:"Default",
    0.09:"High variance",
    0.12:"Chaotic"
  };

  return (
    <div style={{padding:"18px 20px 20px",maxWidth:760,margin:"0 auto",display:"flex",flexDirection:"column",gap:16}}>
      <div style={{textAlign:"center",marginBottom:4}}>
        <div style={{fontSize:11,color:"#555",lineHeight:1.6}}>
          Run the full bracket thousands of times — every game from first round to the title.
          <br />
          Based on ML (LR, XGBoost, LightGBM) and calibrated stats from 77 R64 games, 2005–2025.
        </div>
      </div>

      <div>
        <div style={{fontSize:10,letterSpacing:"0.1em",color:"#888",textTransform:"uppercase",textAlign:"center"}}>Simulations</div>
        <div style={{fontSize:24,fontWeight:700,color:"#111",textAlign:"center",margin:"4px 0 8px"}}>{nSims.toLocaleString()}</div>
        <input
          type="range"
          min={0}
          max={NSIMS_OPTS.length - 1}
          step={1}
          value={nSimsIdx}
          onChange={(e)=>setNSims(NSIMS_OPTS[parseInt(e.target.value, 10)])}
          style={{width:"100%",accentColor:ACCENT}}
        />
        <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#a3a3a3",marginTop:4}}>
          <span>1k</span><span>2.5k</span><span>5k</span><span>10k</span><span>25k</span>
        </div>
        <div style={{fontSize:10,color:"#aaa",marginTop:8,textAlign:"center"}}>
          Est. runtime ~{Math.round(nSims*0.0082)}s
        </div>
      </div>

      <div>
        <div style={{fontSize:10,letterSpacing:"0.1em",color:"#888",textTransform:"uppercase",textAlign:"center"}}>
          Tournament variance: <span style={{color:"#111"}}>{SIGMA_LABELS[sigma]||sigma}</span>
        </div>
        <input type="range" min={0.02} max={0.12} step={0.01} value={sigma}
          onChange={e=>setSigma(parseFloat(e.target.value))}
          style={{width:"100%",accentColor:ACCENT,marginTop:8}}
        />
        <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#bbb",marginTop:4}}>
          <span>chalk</span><span>chaos</span>
        </div>
      </div>

      <div style={{display:"flex",justifyContent:"center",gap:12,flexWrap:"wrap",fontSize:10,color:"#999"}}>
        <span>Variance: <span style={{color:"#444",fontWeight:500}}>{SIGMA_LABELS[sigma]||sigma}</span></span>
        <span style={{color:"#ddd"}}>·</span>
        <span>{assumptionCount} assumption{assumptionCount!==1?"s":""} active</span>
        <span style={{color:"#ddd"}}>·</span>
        <span>{lockedPickCount} pick{lockedPickCount!==1?"s":""} locked</span>
      </div>

      <div style={{display:"flex",justifyContent:"center",paddingTop:4}}>
        <button
          onClick={()=>onRun({n_sims:nSims,latent_sigma:sigma})}
          disabled={running}
          style={{
            height:36,minWidth:180,padding:"0 20px",background:running?"#f0f0f0":"#111",
            color:running?"#aaa":"#fff",border:"none",borderRadius:8,
            fontSize:11,fontWeight:600,cursor:running?"not-allowed":"pointer",
            letterSpacing:"0.03em",whiteSpace:"nowrap"
          }}
        >
          {running?"SIMULATING...":"RUN SIMULATION"}
        </button>
      </div>
    </div>
  );
}

function AssumptionsPanel({
  teams,
  overrides,
  onChange,
  running,
}: {
  teams: string[];
  overrides: TeamOverrides;
  onChange: (next: TeamOverrides) => void;
  running: boolean;
}) {
  const [team, setTeam] = useState("");
  const [delta, setDelta] = useState(25);

  const entries = Object.entries(overrides).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const hasAssumptions = entries.length > 0;

  const addAssumption = () => {
    if (!team) return;
    const value = Math.max(-250, Math.min(250, Math.round(delta)));
    onChange({ ...overrides, [team]: value });
  };

  const removeAssumption = (name: string) => {
    const next = { ...overrides };
    delete next[name];
    onChange(next);
  };

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"8px 10px",borderBottom:"1px solid #e8e8e8",background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Apply team strength deltas (Elo points) before simulation. Positive favors the team, negative fades it.</div>
      </div>
      <div style={{padding:"10px 14px",display:"grid",gridTemplateColumns:"1fr 1fr auto",gap:10,alignItems:"end",borderBottom:"1px solid #e8e8e8"}}>
        <div>
          <div style={{fontSize:9,color:"#888",marginBottom:6,textTransform:"uppercase",letterSpacing:"0.08em"}}>Team</div>
          <select value={team} onChange={(e)=>setTeam(e.target.value)} disabled={running} style={{width:"100%",height:32,border:`1px solid ${PANEL_BORDER}`,padding:"0 8px",fontSize:11,background:"#fff"}}>
            <option value="">Select team...</option>
            {teams.map((t)=><option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
            <span style={{fontSize:9,color:"#888",textTransform:"uppercase",letterSpacing:"0.08em"}}>Adjustment</span>
            <span style={{fontSize:11,fontWeight:600}}>{delta > 0 ? "+" : ""}{delta}</span>
          </div>
          <input type="range" min={-120} max={120} step={5} value={delta} disabled={running}
            onChange={(e)=>setDelta(parseInt(e.target.value, 10))}
            style={{width:"100%",accentColor:"#000"}} />
        </div>
        <button onClick={addAssumption} disabled={running || !team}
          style={{height:32,padding:"0 14px",border:`1px solid ${PANEL_BORDER}`,background:"#000",color:"#fff",fontSize:10,fontWeight:600,letterSpacing:"0.06em",cursor:running?"not-allowed":"pointer"}}>
          APPLY
        </button>
      </div>
      <div style={{padding:"8px 10px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:10}}>
        <div style={{display:"flex",flexWrap:"wrap",gap:6,flex:1}}>
          {!hasAssumptions && <span style={{fontSize:10,color:"#aaa"}}>No custom assumptions active.</span>}
          {entries.slice(0, 8).map(([name, val])=>(
            <span key={name} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"3px 8px",border:`1px solid ${PANEL_BORDER}`,fontSize:10,background:"#fff"}}>
              <span>{name}</span>
              <span style={{fontWeight:700}}>{val>0?"+":""}{val}</span>
              <button onClick={()=>removeAssumption(name)} disabled={running}
                style={{border:"none",background:"transparent",fontSize:11,cursor:"pointer",color:"#888",padding:0}}>×</button>
            </span>
          ))}
        </div>
        {hasAssumptions && (
          <button onClick={()=>onChange({})} disabled={running}
            style={{height:28,padding:"0 10px",border:`1px solid ${PANEL_BORDER}`,background:"#fff",fontSize:10,cursor:"pointer"}}>
            CLEAR ALL
          </button>
        )}
      </div>
    </div>
  );
}

function InitialPickTree({
  teamsCatalog,
  sim,
  forcedPicks,
  onTogglePick,
  onClearPicks,
}: {
  teamsCatalog: Record<string, any>;
  sim: SimState;
  forcedPicks: ForcedPicks;
  onTogglePick: (key: string, winner: string) => void;
  onClearPicks: () => void;
}) {
  const [activeRegion, setActiveRegion] = useState<(typeof REGIONS)[number]>("East");

  const buildRegionRound0 = (region: string) => {
    const seedToTeam: Record<number, string> = {};
    Object.entries(teamsCatalog).forEach(([name, t]: any) => {
      if (t?.region === region && typeof t?.seed === "number") seedToTeam[t.seed] = name;
    });
    return R64_ORDER.map(([a, b], gi) => ({
      key: pickKey(region, 0, gi),
      seedA: a,
      seedB: b,
      teamA: seedToTeam[a] || `Seed ${a}`,
      teamB: seedToTeam[b] || `Seed ${b}`,
    }));
  };

  const RegionPanel = ({ region }: { region: string }) => {
    const r64 = buildRegionRound0(region);
    const RoundMatchups = ({
      round,
      label,
      pickable = false,
    }: {
      round: number;
      label: string;
      pickable?: boolean;
    }) => {
      const games = sim.predicted_bracket[region]?.[round] || [];
      if (!pickable && !sim.complete) return null;
      if (!pickable && games.length === 0) return null;
      return (
        <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
          <div style={{padding:"7px 10px",borderBottom:`1px solid ${PANEL_DIVIDER}`,fontSize:9,color:"#777",letterSpacing:"0.08em",textTransform:"uppercase"}}>{label}</div>
          <div style={{padding:8,display:"grid",gridTemplateColumns:pickable?"repeat(2,minmax(0,1fr))":"repeat(3,minmax(0,1fr))",gap:10}}>
            {(pickable ? r64 : games).map((g: any, idx: number) => {
              if (pickable) {
                const key = g.key;
                return (
                  <div key={key} style={{border:`1px solid ${PANEL_BORDER}`,borderRadius:6,overflow:"hidden",background:"#fff"}}>
                    <div style={{padding:"3px 8px",background:"#fafafa",borderBottom:`1px solid ${PANEL_BORDER}`,display:"flex",alignItems:"center",justifyContent:"center",gap:4,fontSize:9,color:"#888"}}>
                      <span style={{fontWeight:600,color:"#555"}}>({g.seedA})</span>
                      <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g.teamA}</span>
                      <span style={{color:"#ccc",fontWeight:700,fontSize:8}}>vs</span>
                      <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g.teamB}</span>
                      <span style={{fontWeight:600,color:"#555"}}>({g.seedB})</span>
                    </div>
                    {[{name:g.teamA,seed:g.seedA},{name:g.teamB,seed:g.seedB}].map(({name, seed}, ti) => {
                      const locked = forcedPicks[key] === name;
                      return (
                        <button
                          key={name}
                          onClick={()=>onTogglePick(key, name)}
                          style={{
                            width:"100%",display:"flex",alignItems:"center",gap:6,padding:"5px 8px",
                            border:"none",borderTop:ti===1?`1px solid ${PANEL_DIVIDER}`:"none",
                            background:locked?"#111":"#fff",
                            color:locked?"#fff":"#111",cursor:"pointer",textAlign:"left"
                          }}
                        >
                          <Seed n={seed}/>
                          <span style={{flex:1,fontSize:10,fontWeight:locked?700:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
                          {locked && <span style={{fontSize:7,background:"#fff",color:"#111",padding:"1px 4px",borderRadius:3,fontWeight:700,letterSpacing:"0.04em"}}>LOCK</span>}
                        </button>
                      );
                    })}
                  </div>
                );
              }
              const pA = g?.win_prob_a ?? 50;
              const pB = 100 - pA;
              return (
                <div key={`${label}-${idx}`} style={{border:`1px solid ${PANEL_BORDER}`,borderRadius:6,overflow:"hidden",background:"#fff"}}>
                  <div style={{padding:"3px 8px",background:"#fafafa",borderBottom:`1px solid ${PANEL_BORDER}`,display:"flex",alignItems:"center",justifyContent:"center",gap:4,fontSize:9,color:"#888"}}>
                    <span style={{fontWeight:600,color:"#555"}}>({g?.seed_a})</span>
                    <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g?.team_a}</span>
                    <span style={{color:"#ccc",fontWeight:700,fontSize:8}}>vs</span>
                    <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g?.team_b}</span>
                    <span style={{fontWeight:600,color:"#555"}}>({g?.seed_b})</span>
                  </div>
                  {[{name:g?.team_a,seed:g?.seed_a,p:pA},{name:g?.team_b,seed:g?.seed_b,p:pB}].map((t: any, ti: number) => {
                    const won = g?.winner === t.name;
                    return (
                      <div key={t.name} style={{display:"flex",alignItems:"center",gap:6,padding:"5px 8px",borderTop:ti===1?`1px solid ${PANEL_DIVIDER}`:"none",background:won?"#f8f8f8":"#fff"}}>
                        <Seed n={t.seed || 0}/>
                        <span style={{flex:1,fontSize:10,fontWeight:won?700:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{t.name}</span>
                        <span style={{fontSize:9,color:won?"#111":"#999",fontWeight:won?600:400}}>{pct(t.p,1)}</span>
                        {won && <span style={{fontSize:11,color:"#22c55e",lineHeight:1}}>✓</span>}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      );
    };

    return (
      <div style={{display:"flex",flexDirection:"column",gap:10}}>
        <div style={{padding:"6px 10px",fontSize:10,fontWeight:600,letterSpacing:"0.08em",color:"#333",border:`1px solid ${PANEL_BORDER}`,background:"#fafafa"}}>
          {region.toUpperCase()} REGION
        </div>
        <RoundMatchups round={0} label="Round of 64 — Pick Winners" pickable={true}/>
        <RoundMatchups round={1} label="Round of 32"/>
        <RoundMatchups round={2} label="Sweet 16"/>
        <RoundMatchups round={3} label="Elite Eight"/>
      </div>
    );
  };

  const count = Object.keys(forcedPicks).length;
  const f4Candidates = top(sim.final_four_pct, 4).map(([name]) => name);
  const titleCandidates = top(sim.champion_pct, 2).map(([name]) => name);

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"8px 10px",borderBottom:"1px solid #e8e8e8",display:"flex",justifyContent:"space-between",alignItems:"center",background:"#fafafa"}}>
        <span style={{fontSize:10,color:"#444",letterSpacing:"0.06em"}}>Lock first-round picks, then run the sim.</span>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:10,color:"#666"}}>{count} locked</span>
          {count > 0 && (
            <button onClick={onClearPicks} style={{fontSize:9,padding:"3px 8px",border:`1px solid ${PANEL_BORDER}`,background:"#fff",cursor:"pointer"}}>
              clear all
            </button>
          )}
        </div>
      </div>
      <div style={{padding:"8px 8px 0",display:"flex",gap:6,flexWrap:"wrap"}}>
        {REGIONS.map((r) => (
          <button
            key={r}
            onClick={() => setActiveRegion(r)}
            style={{
              padding:"4px 10px",
              border:`1px solid ${PANEL_BORDER}`,
              background:activeRegion===r?"#000":"#fff",
              color:activeRegion===r?"#fff":"#111",
              fontSize:10,
              fontWeight:600,
              cursor:"pointer",
              letterSpacing:"0.04em",
            }}
          >
            {r.toUpperCase()}
          </button>
        ))}
      </div>
      <div style={{padding:10}}>
        <RegionPanel region={activeRegion}/>
      </div>
      <div style={{borderTop:"1px solid #eee",padding:"10px",display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        <div style={{border:"1px solid #e8e8e8",padding:"8px 10px"}}>
          <div style={{fontSize:8,color:"#888",letterSpacing:"0.08em",marginBottom:6}}>FINAL FOUR OUTLOOK</div>
          {Array.from({length:4}).map((_, i) => {
            const name = f4Candidates[i];
            return name ? <div key={i} style={{border:"1px solid #d9d9d9",background:"#fff",color:"#111",padding:"6px 8px",fontSize:10,fontWeight:600,marginBottom:6}}>{name}</div>
            : <div key={i} style={{height:26,borderLeft:"1px solid #ddd",borderBottom:"1px solid #ddd",marginBottom:6}}/>;
          })}
        </div>
        <div style={{border:"1px solid #e8e8e8",padding:"8px 10px"}}>
          <div style={{fontSize:8,color:"#888",letterSpacing:"0.08em",marginBottom:6}}>TITLE GAME OUTLOOK</div>
          {Array.from({length:2}).map((_, i) => {
            const name = titleCandidates[i];
            return name ? <div key={`t-${i}`} style={{border:"1px solid #d9d9d9",background:"#fff",color:"#111",padding:"6px 8px",fontSize:10,fontWeight:600,marginBottom:6}}>{name}</div>
            : <div key={`t-${i}`} style={{height:26,borderLeft:"1px solid #ddd",borderBottom:"1px solid #ddd",marginBottom:6}}/>;
          })}
        </div>
      </div>
    </div>
  );
}

// ── Live sim panel (progress + leaderboard + log) ─────────────────────────────
function LiveSimPanel({ sim, logLines }: { sim: SimState; logLines: string[] }) {
  const logRef = useRef<HTMLDivElement>(null);
  useEffect(()=>{ if(logRef.current) logRef.current.scrollTop=logRef.current.scrollHeight; },[logLines]);

  const progress = sim.total>0 ? sim.done/sim.total : 0;
  const champ = top(sim.champion_pct,4);
  const maxPct = champ[0]?.[1] || 1;

  return (
    <div style={{background:"#fff"}}>
      <div style={{background:"#fafafa",padding:"8px 10px",display:"flex",alignItems:"center",justifyContent:"space-between",borderBottom:"1px solid #e8e8e8"}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:10,fontWeight:600,letterSpacing:"0.1em",color:"#444"}}>RUN STATUS</span>
          {sim.done>0&&!sim.complete&&<span style={{width:6,height:6,borderRadius:"50%",background:GREEN,display:"inline-block",animation:"blink 1s step-end infinite"}}/>}
          {sim.complete&&<span style={{fontSize:9,color:GREEN,letterSpacing:"0.06em"}}>COMPLETE</span>}
        </div>
        <div style={{fontSize:9,color:"#666",display:"flex",gap:12}}>
          {sim.sims_per_sec&&!sim.complete&&<span>{sim.sims_per_sec.toFixed(0)} sims/sec</span>}
          {sim.elapsed_sec&&<span>{sim.elapsed_sec}s</span>}
          {sim.model_used&&<span style={{color:"#444"}}>{sim.model_used.replace(/_/g," ")}</span>}
        </div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr"}}>
        {/* Left: progress + live leaderboard */}
        <div style={{borderRight:"1px solid #e8e8e8",padding:"14px 16px"}}>
          {/* Progress bar */}
          <div style={{marginBottom:14}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#888",marginBottom:5}}>
              <span>{sim.done.toLocaleString()} / {sim.total.toLocaleString()} simulations</span>
              <span style={{fontWeight:600,color:sim.complete?GREEN:"#000"}}>{(progress*100).toFixed(0)}%</span>
            </div>
            <div style={{height:4,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
              <div className={!sim.complete && sim.done > 0 ? "progress-bar-running" : ""} style={{height:"100%",width:`${progress*100}%`,background:sim.complete?GREEN:ACCENT,transition:"width 0.4s ease",borderRadius:2}}/>
            </div>
          </div>

          {/* Live champion leaderboard */}
          <div style={{fontSize:9,letterSpacing:"0.1em",color:"#888",marginBottom:8,textTransform:"uppercase"}}>
            {sim.complete?"Final Championship Odds":"Live Championship Odds"}
          </div>
          {champ.length===0 ? (
            <div style={{fontSize:11,color:"#ccc"}}>Waiting for simulations...</div>
          ) : (
            <div style={{display:"flex",flexDirection:"column",gap:6}}>
              {champ.map(([team,p],i)=>(
                <div key={team}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:3}}>
                    <span style={{fontWeight:i===0?700:400}}>{team}</span>
                    <div style={{display:"flex",gap:10,alignItems:"center"}}>
                      <span style={{fontSize:10,color:"#aaa"}}>{toML(p/100)}</span>
                      <span style={{fontWeight:i===0?700:400}}>{pct(p)}</span>
                    </div>
                  </div>
                  <div style={{height:3,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
                    <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${(p/maxPct)*100}%`,height:"100%",background:i===0?ACCENT:"#ccc",transition:"width 0.5s ease",borderRadius:2}}/>
                  </div>
                </div>
              ))}
              {sim.complete&&top(sim.champion_pct,1)[0]&&(
                <div style={{marginTop:6,fontSize:10,color:"#666",borderTop:"1px solid #f0f0f0",paddingTop:8}}>
                  Implied odds: {top(sim.champion_pct,1)[0][0]} at {toML(top(sim.champion_pct,1)[0][1]/100)}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: console log */}
        <div ref={logRef} className="terminal-log" style={{height:220,overflowY:"auto",background:"#0d0d0d",padding:"10px 14px",fontSize:10,lineHeight:1.9,color:"#555",borderLeft:"1px solid #e8e8e8",scrollbarWidth:"none"} as React.CSSProperties}>
          {logLines.map((line,i)=>{
            const isRecent = i>=logLines.length-3;
            const isChamp = line.includes("champion")||line.includes("CHAMPION")||line.includes("complete");
            const isCheck = line.includes("✓")||line.includes("passed");
            return (
              <div key={i} style={{color:isChamp?"#fff":isCheck?"#86efac":isRecent?"#999":"#444",animation:i===logLines.length-1?"fadeIn 0.2s ease":"none"}}>
                {line}
              </div>
            );
          })}
          {logLines.length===0&&<div style={{color:"#333"}}>ready to simulate...</div>}
        </div>
      </div>
    </div>
  );
}

// ── Advancement Table ─────────────────────────────────────────────────────────
function AdvancementTable({ sim }: { sim: SimState }) {
  if (!sim.complete) return null;

  const regions = ["East","West","Midwest","South"];
  const teamsMap: Record<string, {name:string;seed:number;region:string}> = {};

  regions.forEach(region=>{
    (sim.predicted_bracket[region]?.[0]||[]).forEach((g:Game)=>{
      teamsMap[g.team_a] = {name:g.team_a,seed:g.seed_a,region};
      teamsMap[g.team_b] = {name:g.team_b,seed:g.seed_b,region};
    });
  });
  const allTeams = Object.values(teamsMap);

  const getData = (team:string) => ({
    r32:  sim.round_of_32_pct[team]||0,
    s16:  sim.sweet_sixteen_pct[team]||0,
    e8:   sim.elite_eight_pct[team]||0,
    f4:   sim.final_four_pct[team]||0,
    fin:  sim.title_game_pct[team]||0,
    champ:sim.champion_pct[team]||0,
  });

  const rows = allTeams
    .map((t) => ({ ...t, ...getData(t.name) }))
    .filter((t) => t.r32 > 0 || t.champ > 0 || t.seed <= 13)
    .sort((a,b)=>(
      b.champ - a.champ ||
      b.fin - a.fin ||
      b.f4 - a.f4 ||
      a.seed - b.seed ||
      a.name.localeCompare(b.name)
    ));

  // Color scale: 0=white, 100=black
  const cell = (v:number, isChamp=false) => {
    const intensity = Math.min(v/85, 1);
    const bg = isChamp ? `rgba(34,197,94,${intensity*0.8})` : `rgba(0,0,0,${intensity*0.15})`;
    const color = isChamp&&v>30 ? "#065f46" : intensity>0.6 ? "#000" : "#333";
    return {background:bg, color, fontWeight:v>40?700:v>10?500:400};
  };

  return (
    <div style={{overflowX:"auto",border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
        <thead>
          <tr style={{borderBottom:"2px solid #000",background:"#000"}}>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>RK</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>SEED</th>
            <th style={{padding:"8px 10px",textAlign:"left",color:"#fff",fontSize:10,letterSpacing:"0.06em",fontWeight:600}}>TEAM</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>R32</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>S16</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>E8</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>F4</th>
            <th style={{padding:"8px 8px",textAlign:"center",color:"#aaa",fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>FINAL</th>
            <th style={{padding:"8px 10px",textAlign:"center",color:GREEN,fontSize:9,letterSpacing:"0.08em",fontWeight:600}}>CHAMP</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t,i)=>{
            return (
              <tr key={t.name} style={{borderBottom:"1px solid #f4f4f4",background:i%2===0?"#fff":"#fafafa"}}>
                <td style={{padding:"6px 8px",textAlign:"center",fontSize:10,fontWeight:600,color:"#666"}}>{i+1}</td>
                <td style={{padding:"6px 8px",textAlign:"center"}}><Seed n={t.seed}/></td>
                <td style={{padding:"6px 10px",whiteSpace:"nowrap"}}>
                  <span style={{display:"inline-flex",alignItems:"center",gap:6}}>
                    <span style={{fontWeight:t.seed<=4?600:400}}>{t.name}</span>
                    <span style={{fontSize:9,color:"#bbb"}}>{t.region}</span>
                  </span>
                </td>
                {([t.r32,t.s16,t.e8,t.f4,t.fin] as number[]).map((v,j)=>(
                  <td key={j} style={{padding:"6px 8px",textAlign:"center",...cell(v)}}>
                    {v>0.5?pct(v,v<10?1:0):"—"}
                  </td>
                ))}
                <td style={{padding:"6px 10px",textAlign:"center",...cell(t.champ,true)}}>
                  {t.champ>0.5?<strong>{pct(t.champ,t.champ<5?1:0)}</strong>:"—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── Bracket grid (replaces tree — much less re-render churn) ──────────────────
// Each region is an independent memoized column. Resolved state passed per-game.
const GameCard = memo(function GameCard({
  g,
  isResolved,
  region,
  round,
  gi,
  forcedPicks,
  onTogglePick,
}: {
  g: Game|null;
  isResolved: boolean;
  region: string;
  round: number;
  gi: number;
  forcedPicks: ForcedPicks;
  onTogglePick: (key: string, winner: string) => void;
}) {
  if (!isResolved || !g) {
    return (
      <div style={{border:`1px solid ${PANEL_BORDER}`,borderRadius:6,background:"#fafafa",marginBottom:6,overflow:"hidden"}}>
        <div style={{padding:"5px 10px",minHeight:22,display:"flex",alignItems:"center",gap:6,borderBottom:"1px solid #f0f0f0"}}>
          <div style={{width:20,height:20,borderRadius:4,border:"1px solid #e0e0e0",background:"#eee",flexShrink:0}}/>
          <div style={{height:8,background:"#efefef",flex:1,borderRadius:3}}/>
        </div>
        <div style={{padding:"5px 10px",minHeight:22,display:"flex",alignItems:"center",gap:6}}>
          <div style={{width:20,height:20,borderRadius:4,border:"1px solid #e0e0e0",background:"#eee",flexShrink:0}}/>
          <div style={{height:8,background:"#efefef",width:"60%",borderRadius:3}}/>
        </div>
      </div>
    );
  }

  const key = pickKey(region, round, gi);
  const forced = forcedPicks[key];
  const aWon = (forced ? forced === g.team_a : g.winner===g.team_a);
  const pA = g.win_prob_a;
  const pB = Math.max(0, 100 - pA);

  const SeedBadge = ({seed}:{seed:number}) => (
    <span style={{
      width:20,height:20,fontSize:10,fontWeight:700,flexShrink:0,
      display:"inline-flex",alignItems:"center",justifyContent:"center",
      borderRadius:4,background:seed<=4?"#111":"#e8e8e8",color:seed<=4?"#fff":"#444",
    }}>{seed}</span>
  );

  const TeamRow = ({team,seed,score,won,p,isForced}:{team:string;seed:number;score?:number;won:boolean;p:number;isForced:boolean}) => (
    <button
      onClick={()=>onTogglePick(key, team)}
      style={{
        display:"flex",alignItems:"center",gap:8,padding:"6px 10px",border:"none",width:"100%",textAlign:"left",
        background:won?"#f8f8f8":"#fff",minHeight:26,cursor:"pointer",
        borderLeft:won?`3px solid ${ACCENT}`:"3px solid transparent",
      }}
      title={isForced ? "Click to clear pick" : "Click to force this winner"}
    >
      <SeedBadge seed={seed}/>
      <span style={{flex:1,fontSize:11,fontWeight:won?700:400,color:won?"#000":"#777",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{team}</span>
      <span style={{fontSize:10,fontWeight:won?600:400,color:won?"#111":"#aaa",minWidth:38,textAlign:"right"}}>{pct(p,1)}</span>
      {score!=null&&<span style={{fontSize:9,fontWeight:won?700:400,color:won?"#000":"#ccc",minWidth:18,textAlign:"right"}}>{score}</span>}
      {isForced
        ? <span style={{fontSize:7,color:"#fff",background:"#111",padding:"1px 4px",borderRadius:3,fontWeight:700,letterSpacing:"0.04em"}}>LOCK</span>
        : won
          ? <span style={{fontSize:12,color:"#22c55e",marginLeft:2,lineHeight:1}}>✓</span>
          : <span style={{width:14}}/>
      }
    </button>
  );

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,borderRadius:6,background:"#fff",marginBottom:6,overflow:"hidden",animation:"fadeIn 0.3s ease"}}>
      <div style={{
        display:"flex",alignItems:"center",justifyContent:"center",gap:6,
        padding:"3px 8px",background:"#fafafa",borderBottom:`1px solid ${PANEL_BORDER}`,
        fontSize:9,color:"#888",letterSpacing:"0.02em",
      }}>
        <span style={{fontWeight:600,color:"#555"}}>({g.seed_a})</span>
        <span style={{maxWidth:80,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{g.team_a}</span>
        <span style={{color:"#ccc",fontWeight:700,fontSize:8}}>vs</span>
        <span style={{maxWidth:80,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{g.team_b}</span>
        <span style={{fontWeight:600,color:"#555"}}>({g.seed_b})</span>
      </div>
      <TeamRow team={g.team_a} seed={g.seed_a} score={g.score_a} won={aWon} p={pA} isForced={forced===g.team_a}/>
      <div style={{display:"flex",alignItems:"center",gap:8,padding:"0 10px",background:"#fff"}}>
        <div style={{flex:1,height:1,background:"#e8e8e8"}}/>
        <span style={{fontSize:8,color:"#ccc",fontWeight:600}}>VS</span>
        <div style={{flex:1,height:1,background:"#e8e8e8"}}/>
      </div>
      <TeamRow team={g.team_b} seed={g.seed_b} score={g.score_b} won={!aWon} p={pB} isForced={forced===g.team_b}/>
      <div style={{height:3,background:"#f0f0f0",borderRadius:"0 0 5px 5px"}}>
        <div style={{width:`${pA}%`,height:"100%",background:aWon?"#111":"#888",borderRadius:"0 0 0 5px"}}/>
      </div>
    </div>
  );
});

const RegionColumn = memo(function RegionColumn({ region, sim, resolved, forcedPicks, onTogglePick }: {
  region: string; sim: SimState; resolved: Resolved; forcedPicks: ForcedPicks; onTogglePick: (key: string, winner: string) => void;
}) {
  const ROUND_LABELS = ["FIRST ROUND","ROUND OF 32","SWEET 16","ELITE 8"];
  const rounds = sim.predicted_bracket[region] || [];

  return (
    <div style={{flex:1,borderRight:"1px solid #e8e8e8"}}>
      <div style={{background:"#000",padding:"5px 8px"}}>
        <span style={{fontSize:9,fontWeight:600,letterSpacing:"0.1em",color:"#fff"}}>{region.toUpperCase()}</span>
      </div>
      {[0,1,2,3].map(ri=>{
        const count = ri===0?8:ri===1?4:ri===2?2:1;
        return (
          <div key={ri}>
            <div style={{padding:"2px 8px",background:"#f4f4f4",borderBottom:"1px solid #e8e8e8",borderTop:ri>0?"1px solid #ddd":"none"}}>
              <span style={{fontSize:8,letterSpacing:"0.1em",fontWeight:700,color:"#888"}}>{ROUND_LABELS[ri]}</span>
            </div>
            {Array.from({length:count}).map((_,gi)=>{
              const g = rounds[ri]?.[gi] || null;
              const isResolved = !!(resolved[region]?.[ri]?.[gi]);
              return (
                <GameCard
                  key={`${ri}-${gi}`}
                  g={g}
                  isResolved={isResolved}
                  region={region}
                  round={ri}
                  gi={gi}
                  forcedPicks={forcedPicks}
                  onTogglePick={onTogglePick}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
});

function BracketGrid({
  sim,
  resolved,
  forcedPicks,
  onTogglePick,
  onClearPicks,
}: {
  sim: SimState;
  resolved: Resolved;
  forcedPicks: ForcedPicks;
  onTogglePick: (key: string, winner: string) => void;
  onClearPicks: () => void;
}) {
  if (!sim.bracketStarted) return null;
  const pickCount = Object.keys(forcedPicks).length;
  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,overflowX:"auto",background:"#fff"}}>
      <div style={{padding:"6px 10px",background:"#f8f8f8",borderBottom:"1px solid #e8e8e8",display:"flex",alignItems:"center",justifyContent:"space-between",gap:12}}>
        <span style={{fontSize:9,color:"#666",letterSpacing:"0.06em"}}>Click any team to lock a winner. Locked picks are used next simulation run while all other games are simulated.</span>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:9,color:"#444"}}>{pickCount} picks locked</span>
          {pickCount > 0 && (
            <button onClick={onClearPicks} style={{fontSize:9,padding:"3px 8px",border:`1px solid ${PANEL_BORDER}`,background:"#fff",cursor:"pointer"}}>
              clear
            </button>
          )}
        </div>
      </div>
      <div style={{display:"flex",minWidth:800}}>
        {["East","West","Midwest","South"].map(r=>(
          <RegionColumn
            key={r}
            region={r}
            sim={sim}
            resolved={resolved}
            forcedPicks={forcedPicks}
            onTogglePick={onTogglePick}
          />
        ))}
      </div>
      {sim.complete && (
        <div style={{borderTop:`1px solid ${PANEL_DIVIDER}`,display:"grid",gridTemplateColumns:"1fr 1fr 1fr"}}>
          {[
            {label:"FINAL FOUR · SF1", teams:top(sim.final_four_pct,4).slice(0,2)},
            {label:"FINAL FOUR · SF2", teams:top(sim.final_four_pct,4).slice(2,4)},
            {label:"NATIONAL CHAMPIONSHIP", teams:top(sim.champion_pct,3)},
          ].map(({label,teams},i)=>(
            <div key={label} style={{padding:"10px 12px",borderRight:i<2?"1px solid #000":"none"}}>
              <div style={{fontSize:8,letterSpacing:"0.1em",fontWeight:700,color:"#888",marginBottom:6}}>{label}</div>
              {teams.map(([team,p],j)=>(
                <div key={team} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:"1px solid #f4f4f4"}}>
                  <span style={{fontSize:11,fontWeight:j===0?700:400}}>{team}</span>
                  <span style={{fontSize:10,color:"#666"}}>{pct(p)}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Charts (collapsible) ──────────────────────────────────────────────────────
function ChampionshipChart({ sim }: { sim: SimState }) {
  const data = top(sim.champion_pct, 12);
  const max = data[0]?.[1] || 1;
  return (
    <div style={{padding:"16px 20px"}}>
      <div style={{fontSize:9,letterSpacing:"0.1em",color:"#888",marginBottom:12,textTransform:"uppercase"}}>Championship probability — all {data.length} teams above 0%</div>
      <div style={{display:"flex",flexDirection:"column",gap:5}}>
        {data.map(([team,p],i)=>(
          <div key={team} style={{display:"grid",gridTemplateColumns:"140px 1fr 54px",alignItems:"center",gap:10}}>
            <div style={{fontSize:11,fontWeight:i<4?600:400,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{team}</div>
            <div style={{height:6,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${(p/max)*100}%`,height:"100%",background:i===0?"#000":i<4?"#333":"#bbb",borderRadius:3,transition:"width 0.5s"}}/>
            </div>
            <div style={{fontSize:11,fontWeight:i===0?700:400,textAlign:"right"}}>{pct(p)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function OddsComparisonChart({ sim }: { sim: SimState }) {
  const teams = top(sim.champion_pct,8).map(([t])=>t);
  const rounds = [
    {label:"Sweet 16", data:sim.sweet_sixteen_pct, color:"#d4d4d4"},
    {label:"Elite 8",  data:sim.elite_eight_pct, color:"#a3a3a3"},
    {label:"Final Four",data:sim.final_four_pct, color:"#525252"},
    {label:"Champion", data:sim.champion_pct,  color:"#000"},
  ];
  const maxVal = Math.max(...teams.map(t=>sim.sweet_sixteen_pct[t]||0));
  return (
    <div style={{padding:"16px 20px"}}>
      <div style={{fontSize:9,letterSpacing:"0.1em",color:"#888",marginBottom:16,textTransform:"uppercase"}}>Advancement probability — top 8 title contenders</div>
      <div style={{display:"grid",gridTemplateColumns:`140px repeat(${teams.length},1fr)`,gap:0}}>
        {/* Round labels */}
        <div/>
        {teams.map(t=>(
          <div key={t} style={{fontSize:9,textAlign:"center",padding:"0 2px 8px",fontWeight:600,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{t.split(" ").slice(-1)[0]}</div>
        ))}
        {rounds.map(({label,data})=>(
          <div key={label} style={{display:"contents"}}>
            <div style={{fontSize:10,color:"#666",paddingRight:10,paddingBottom:6,display:"flex",alignItems:"center"}}>{label}</div>
            {teams.map(t=>{
              const v = data[t]||0;
              const intensity = v/maxVal;
              return (
                <div key={t} title={`${t}: ${pct(v)}`} style={{
                  background:`rgba(0,0,0,${intensity*0.85})`,
                  margin:"0 1px 1px",
                  height:28,
                  display:"flex",alignItems:"center",justifyContent:"center",
                }}>
                  <span style={{fontSize:9,color:intensity>0.5?"#fff":"#333",fontWeight:v>50?700:400}}>
                    {v>1?pct(v,0):""}
                  </span>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function UpsetChart({ sim }: { sim: SimState }) {
  const upsets = (sim.upset_watch||[]).sort((a:any,b:any)=>b.upset_prob-a.upset_prob).slice(0,8);
  const max = upsets[0]?.upset_prob||1;
  return (
    <div style={{padding:"16px 20px"}}>
      <div style={{fontSize:9,letterSpacing:"0.1em",color:"#888",marginBottom:12,textTransform:"uppercase"}}>Top upset candidates — R64</div>
      <div style={{display:"flex",flexDirection:"column",gap:6}}>
        {upsets.map((u:any,i:number)=>(
          <div key={i} style={{display:"grid",gridTemplateColumns:"180px 1fr 48px",alignItems:"center",gap:10}}>
            <div style={{fontSize:10,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
              <span style={{fontWeight:600}}>({u.dog_seed}) {u.underdog}</span>
              <span style={{color:"#aaa",fontSize:9}}> over ({u.fav_seed}) {u.favorite}</span>
            </div>
            <div style={{height:6,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
              <div style={{width:`${(u.upset_prob/max)*100}%`,height:"100%",background:u.upset_prob>40?"#000":u.upset_prob>32?"#555":"#999",borderRadius:3}}/>
            </div>
            <div style={{fontSize:11,fontWeight:600,textAlign:"right"}}>{pct(u.upset_prob,0)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function DiagnosticsSection({ sim, overrides }: { sim: SimState; overrides: TeamOverrides }) {
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
    { label: "Champion mass", actual: champTotal, target: 100, tol: 1.5 },
    { label: "Title game mass", actual: titleTotal, target: 200, tol: 2.5 },
    { label: "Final Four mass", actual: f4Total, target: 400, tol: 4.0 },
    { label: "Elite Eight mass", actual: e8Total, target: 800, tol: 6.0 },
  ];

  const statusColor = (ok: boolean) => (ok ? "#065f46" : "#7f1d1d");
  const statusBg = (ok: boolean) => (ok ? "#dcfce7" : "#fee2e2");

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Quick confidence checks on probability accounting and bracket volatility.</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1.2fr 1fr"}}>
        <div style={{padding:"12px 14px",borderRight:`1px solid ${PANEL_DIVIDER}`}}>
          {checks.map((c) => {
            const delta = c.actual - c.target;
            const ok = Math.abs(delta) <= c.tol;
            return (
              <div key={c.label} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"7px 0",borderBottom:"1px solid #f2f2f2"}}>
                <div style={{fontSize:11,color:"#444"}}>{c.label}</div>
                <div style={{display:"flex",alignItems:"center",gap:8}}>
                  <span style={{fontSize:10,color:"#888"}}>{c.actual.toFixed(1)}% vs {c.target}%</span>
                  <span style={{fontSize:9,fontWeight:700,padding:"2px 6px",borderRadius:10,color:statusColor(ok),background:statusBg(ok)}}>
                    {ok ? "OK" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}pp`}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
        <div style={{padding:"12px 14px",display:"grid",gridTemplateColumns:"1fr 1fr",gap:"10px 12px"}}>
          {[
            {label:"Top-4 title share", value:pct(top4Champ,1)},
            {label:"Top-8 title share", value:pct(top8Champ,1)},
            {label:"Highest title odds", value:pct(maxChamp,1)},
            {label:`${maxTeam} 95% MOE`, value:`±${moe.toFixed(2)}pp`},
            {label:"Avg upset chance", value:pct(avgUpset,1)},
            {label:"35%+ upset games", value:String(highUpset)},
            {label:"Assumptions active", value:String(assumptionCount)},
            {label:"Simulation sample", value:sim.n_sims.toLocaleString()},
          ].map((m) => (
            <div key={m.label} style={{border:"1px solid #eee",padding:"8px 10px"}}>
              <div style={{fontSize:9,letterSpacing:"0.06em",textTransform:"uppercase",color:"#888",marginBottom:4}}>{m.label}</div>
              <div style={{fontSize:16,fontWeight:600,lineHeight:1.1}}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function DataProvenance({ info }: { info: ModelInfo | null }) {
  if (!info?.inference_data || !info?.training_data) return null;
  const inf = info.inference_data;
  const tr = info.training_data;
  const md = info.model_details;
  const stack = (md?.stack || ["logistic_regression", "xgboost", "lightgbm", "meta_logistic"]).join(" + ");
  const Row = ({ label, value }: { label: string; value: React.ReactNode }) => (
    <div style={{ display: "flex", justifyContent: "space-between", gap: 12, padding: "6px 0", borderBottom: "1px solid #f1f5f9", fontSize: 12 }}>
      <span style={{ color: "#64748b", flexShrink: 0 }}>{label}</span>
      <span style={{ color: "#0f172a", textAlign: "right" }}>{value}</span>
    </div>
  );
  return (
    <div style={{ padding: "14px 16px", background: "#fff" }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: "#334155", marginBottom: 10 }}>Data & model</div>
      <Row label="Bracket" value={<>{inf.dataset} · {inf.season} · {inf.teams_in_bracket} teams</>} />
      <Row label="Freeze date" value={inf.selection_sunday_freeze} />
      <Row label="Training" value={<>{tr.season_min}–{tr.season_max} · {tr.rows} rows</>} />
      <Row label="Recency" value={tr.recency_weighting} />
      <Row label="Stack" value={stack} />
      <Row label="Calibration" value={md?.calibration || "isotonic"} />
      <Row label="Features" value={<>{md?.core_feature_count ?? "—"} core + {md?.market_feature_count ?? "—"} market/meta</>} />
      <div style={{ paddingTop: 8, fontSize: 11, color: "#64748b", lineHeight: 1.5 }}>
        {md?.market_usage || "Market features in blend"} · {md?.variance_modeling || "Latent strength variance in sims"}.
      </div>
    </div>
  );
}

function TeamProbabilityMath({ sim, teams }: { sim: SimState; teams: string[] }) {
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
    <div style={{border:`1px solid ${PANEL_DIVIDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,display:"flex",justifyContent:"space-between",alignItems:"center",gap:10,flexWrap:"wrap"}}>
        <span style={{fontSize:10,color:"#666"}}>Round-by-round odds</span>
        <div style={{display:"flex",alignItems:"center",gap:6}}>
          <span style={{fontSize:10,color:"#888"}}>Team</span>
          <select value={team} onChange={(e)=>setTeam(e.target.value)} style={{height:30,padding:"0 10px",border:`1px solid ${PANEL_DIVIDER}`,fontSize:11,background:"#fff",minWidth:180}}>
            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>
      {!sim.complete ? (
        <div style={{padding:"10px 14px",fontSize:10,color:"#888",borderBottom:`1px solid ${PANEL_DIVIDER}`}}>
          Pick a team and run a simulation to see probabilities.
        </div>
      ) : (
      <>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
          <thead>
            <tr style={{borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fcfcfc"}}>
              <th style={{padding:"8px 10px",textAlign:"left",fontSize:9,color:"#777",letterSpacing:"0.06em"}}>OUTCOME</th>
              <th style={{padding:"8px 10px",textAlign:"right",fontSize:9,color:"#777",letterSpacing:"0.06em"}}>CHANCE</th>
              <th style={{padding:"8px 10px",textAlign:"right",fontSize:9,color:"#777",letterSpacing:"0.06em"}}>EXPECTED HITS</th>
              <th style={{padding:"8px 10px",textAlign:"right",fontSize:9,color:"#777",letterSpacing:"0.06em"}}>95% CI</th>
              <th style={{padding:"8px 10px",textAlign:"right",fontSize:9,color:"#777",letterSpacing:"0.06em"}}>FAIR ODDS</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, idx) => {
              const m = explain(r.value);
              const fairOdds = m.p <= 0 || m.p >= 1 ? "N/A" : toML(m.p);
              return (
                <tr key={r.label} style={{borderBottom:`1px solid ${PANEL_DIVIDER}`,background:idx % 2 === 0 ? "#fff" : "#fcfcfc"}}>
                  <td style={{padding:"8px 10px",fontWeight:500}}>{r.label}</td>
                  <td style={{padding:"8px 10px",textAlign:"right",fontWeight:600}}>{pct(r.value, 2)}</td>
                  <td style={{padding:"8px 10px",textAlign:"right",color:"#555"}}>{m.expectedHits.toFixed(0)} / {n.toLocaleString()}</td>
                  <td style={{padding:"8px 10px",textAlign:"right",color:"#555"}}>{pct(m.lo * 100, 2)} - {pct(m.hi * 100, 2)}</td>
                  <td style={{padding:"8px 10px",textAlign:"right",color:"#555"}}>{fairOdds}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div style={{padding:"10px 14px",borderTop:`1px solid ${PANEL_DIVIDER}`,display:"flex",justifyContent:"space-between",alignItems:"center",gap:12,flexWrap:"wrap",background:"#fcfcfc"}}>
        <div style={{fontSize:10,color:"#888",letterSpacing:"0.06em",textTransform:"uppercase"}}>Most likely finish</div>
        <div style={{fontSize:12,fontWeight:600}}>
          {likelyOutcome?.label || "N/A"} <span style={{fontSize:10,color:"#666",fontWeight:500}}>({pct((likelyOutcome?.p || 0) * 100, 2)})</span>
        </div>
      </div>
      </>
      )}
    </div>
  );
}

// ── Model training output ──────────────────────────────────────────────────────
function ModelTrainingOutput() {
  const [log,setLog] = useState<string|null>(null);
  const [loading,setLoading] = useState(false);
  const [fetched,setFetched] = useState(false);
  const [height,setHeight] = useState(280);
  const dragRef = useRef<{y:number;h:number}|null>(null);

  const load = async()=>{
    if(fetched) return;
    setLoading(true);
    try { const r=await fetch(`${API}/model-log`).then(r=>r.json()); setLog(r.log); setFetched(true); }
    catch { setLog("Failed to load."); }
    setLoading(false);
  };

  const onDrag = (e:React.MouseEvent)=>{
    dragRef.current={y:e.clientY,h:height};
    const mv=(ev:MouseEvent)=>{ if(dragRef.current) setHeight(Math.max(80,Math.min(800,dragRef.current.h+ev.clientY-dragRef.current.y))); };
    const up=()=>{ dragRef.current=null; window.removeEventListener("mousemove",mv); window.removeEventListener("mouseup",up); };
    window.addEventListener("mousemove",mv); window.addEventListener("mouseup",up);
  };

  return (
    <div style={{marginTop:12,border:`1px solid ${PANEL_DIVIDER}`,borderRadius:8,background:"#fafafa",overflow:"hidden"}}>
      <div style={{padding:"10px 12px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:8}}>
        <span style={{fontSize:11,fontWeight:600,color:"#475569"}}>Training log</span>
        {!fetched && !loading && (
          <button onClick={load} style={{padding:"4px 12px",background:"#0f172a",color:"#fff",border:"none",fontSize:10,fontWeight:600,cursor:"pointer",borderRadius:6}}>Load</button>
        )}
      </div>
      {loading&&<div style={{padding:"14px",fontSize:11,color:"#aaa"}}>Running model pipeline (~14s)...</div>}
      {log&&(
        <>
          <div style={{height,overflowY:"auto",overflowX:"auto",background:"#1a1a1a",padding:"12px 16px",fontSize:11,lineHeight:1.8,whiteSpace:"pre"}}>
            {log.split("\n").map((line,i)=>(
              <div key={i} style={{color:line.startsWith("===")?"#fff":line.trim().startsWith("✓")?"#86efac":/\d+\.\d{4}/.test(line)?"#d1d5db":"#9ca3af",fontWeight:line.startsWith("===")?600:400,borderTop:line.startsWith("===")?"1px solid #333":"none",paddingTop:line.startsWith("===")?8:0,marginTop:line.startsWith("===")?4:0}}>{line||" "}</div>
            ))}
          </div>
          <div onMouseDown={onDrag} style={{height:6,background:"#e8e8e8",cursor:"ns-resize",display:"flex",alignItems:"center",justifyContent:"center"}}>
            <div style={{width:32,height:2,background:"#bbb",borderRadius:1}}/>
          </div>
        </>
      )}
    </div>
  );
}

// ── Analysis ──────────────────────────────────────────────────────────────────
function AnalysisSection({ allTeams }: { allTeams: string[] }) {
  const [a,setA] = useState(allTeams[0]||"Duke");
  const [b,setB] = useState(allTeams[4]||"UConn");
  const [matchup,setMatchup] = useState<any>(null);
  const [disagree,setDisagree] = useState<any>(null);
  const [comps,setComps] = useState<any>(null);
  const [whatif,setWhatif] = useState<any[]>([]);
  const [loading,setLoading] = useState(false);

  const run = async()=>{
    if(a===b) return; setLoading(true);
    try {
      const [m,d,c,w] = await Promise.all([
        fetch(`${API}/matchup`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({team_a:a,team_b:b})}).then(r=>r.json()),
        fetch(`${API}/disagreement/${encodeURIComponent(a)}/${encodeURIComponent(b)}`).then(r=>r.json()),
        fetch(`${API}/comps/${encodeURIComponent(a)}/${encodeURIComponent(b)}`).then(r=>r.json()),
        fetch(`${API}/whatif`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({team_a:a,team_b:b,scenarios:[
          {label:"Baseline",target:"a",injury_factor:1.0},
          {label:`${a} minor inj`,target:"a",injury_factor:0.93},
          {label:`${a} major inj`,target:"a",injury_factor:0.82},
          {label:`${b} injured`,target:"b",injury_factor:0.90},
          {label:"Hot streak",target:"a",form_score:8.0},
          {label:"Slow tempo",target:"a",possessions_per_game:62.0},
        ]})}).then(r=>r.json()),
      ]);
      setMatchup(m); setDisagree(d); setComps(c); setWhatif(w.results||[]);
    } catch(e){console.error(e);}
    setLoading(false);
  };
  const sel={flex:1,height:30,background:"#fff",border:`1px solid ${PANEL_BORDER}`,fontSize:11,padding:"0 8px",appearance:"none" as const,cursor:"pointer"};

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Win probability and what drives it. Right panel shows impact of injuries, hot streaks, pace changes.</div>
      </div>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,display:"flex",gap:10,alignItems:"center"}}>
        <select value={a} onChange={e=>setA(e.target.value)} style={sel}>{allTeams.map(t=><option key={t}>{t}</option>)}</select>
        <span style={{fontSize:11,color:"#999"}}>vs</span>
        <select value={b} onChange={e=>setB(e.target.value)} style={sel}>{allTeams.map(t=><option key={t}>{t}</option>)}</select>
        <button onClick={run} disabled={loading||a===b} style={{height:30,padding:"0 16px",background:"#000",color:"#fff",border:"none",fontSize:10,fontWeight:600,cursor:"pointer",letterSpacing:"0.06em",whiteSpace:"nowrap"}}>{loading?"...":"ANALYZE"}</button>
      </div>
      {matchup&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr"}}>
          <div style={{borderRight:`1px solid ${PANEL_DIVIDER}`}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",borderBottom:"1px solid #e8e8e8"}}>
              {[{name:a,prob:matchup.win_prob_a,score:matchup.score_a},{name:b,prob:matchup.win_prob_b,score:matchup.score_b}].map(({name,prob,score},i)=>(
                <div key={name} style={{padding:"16px",borderRight:i===0?"1px solid #e8e8e8":"none",background:prob>50?"#f9f9f9":"#fff"}}>
                  <div style={{fontSize:10,color:"#888",marginBottom:4}}>{name}</div>
                  <div style={{fontSize:36,fontWeight:300,lineHeight:1}}>{prob.toFixed(1)}%</div>
                  {score&&<div style={{fontSize:10,color:"#bbb",marginTop:4}}>sim: {score}</div>}
                  {prob>50&&<div style={{fontSize:9,fontWeight:600,marginTop:8,letterSpacing:"0.1em"}}>PROJECTED WINNER</div>}
                </div>
              ))}
            </div>
            {disagree&&(
              <div style={{padding:"12px 16px",borderBottom:"1px solid #e8e8e8"}}>
                <div style={{fontSize:9,letterSpacing:"0.1em",fontWeight:600,color:"#888",marginBottom:10}}>SIGNAL BREAKDOWN</div>
                <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"5px 20px"}}>
                  {Object.entries(disagree.signals).map(([k,v]:any)=>{
                    const adv=v-50;
                    return (<div key={k} style={{display:"flex",alignItems:"center",gap:8}}>
                      <span style={{fontSize:9,color:"#888",minWidth:72}}>{k.replace(/_/g," ")}</span>
                      <div style={{flex:1,height:2,background:"#e8e8e8",position:"relative"}}>
                        <div style={{position:"absolute",left:"50%",top:0,bottom:0,width:1,background:"#bbb"}}/>
                        <div style={{position:"absolute",height:"100%",background:"#000",left:adv>=0?"50%":`${50+adv}%`,width:`${Math.abs(adv)}%`}}/>
                      </div>
                      <span style={{fontSize:9,minWidth:24,textAlign:"right"}}>{adv>0?"+":""}{adv.toFixed(0)}</span>
                    </div>);
                  })}
                </div>
              </div>
            )}
            {comps?.comps&&(
              <div style={{padding:"12px 16px"}}>
                <div style={{fontSize:9,letterSpacing:"0.1em",fontWeight:600,color:"#888",marginBottom:8}}>HISTORICAL COMPS — {a}-analog won {((comps.historical_win_rate||0)*100).toFixed(0)}% of {comps.comps.length} similar games</div>
                {comps.comps.slice(0,4).map((c:any,i:number)=>(
                  <div key={i} style={{display:"flex",alignItems:"center",gap:8,padding:"4px 0",borderBottom:"1px solid #f4f4f4",fontSize:10}}>
                    <span style={{color:"#ccc",minWidth:30}}>{c.season}</span>
                    <span style={{fontWeight:c.team_a_won?600:400,flex:1}}>({c.seed_a}) {c.team_a}</span>
                    <span style={{color:"#ccc"}}>vs</span>
                    <span style={{fontWeight:!c.team_a_won?600:400,flex:1,textAlign:"right"}}>({c.seed_b}) {c.team_b}</span>
                    {c.upset&&<span style={{fontSize:8,border:"1px solid #d1d5db",padding:"0 3px"}}>U</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
          <div>
            <div style={{padding:"8px 14px",borderBottom:"1px solid #e8e8e8",background:"#fafafa"}}>
              <div style={{fontSize:9,letterSpacing:"0.1em",fontWeight:600,color:"#888"}}>WHAT IF — {a} win probability</div>
            </div>
            {whatif.map((r:any,i:number)=>{
              const base=whatif[0]?.new_prob||50; const delta=r.new_prob-base; const isBase=i===0;
              return (<div key={i} style={{display:"flex",alignItems:"center",gap:12,padding:"10px 16px",borderBottom:"1px solid #f4f4f4",background:isBase?"#f9f9f9":"#fff"}}>
                <div style={{flex:1,fontSize:11,fontWeight:isBase?600:400}}>{r.scenario}</div>
                <div style={{textAlign:"right",minWidth:60}}>
                  <div style={{fontSize:15,fontWeight:300}}>{r.new_prob.toFixed(1)}%</div>
                  {!isBase&&<div style={{fontSize:10,fontWeight:600,color:delta>2?"#000":delta<-2?"#888":"#ccc"}}>{delta>0?"+":""}{delta.toFixed(1)}%</div>}
                </div>
              </div>);
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── First Four (dynamic from API) ─────────────────────────────────────────
function FirstFourSection({ sim }: { sim: SimState }) {
  const ff = sim.first_four_pct || {};
  const [games, setGames] = useState<any[]>([]);

  useEffect(() => {
    fetch(`${API}/first-four`).then(r => r.json()).then(d => setGames(d.games || [])).catch(() => {});
  }, []);

  if (games.length === 0) return null;

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr"}}>
        {games.map((g: any, i: number) => {
          const pa = ff[g.team_a] || 50, pb = ff[g.team_b] || 50, tot = pa + pb || 100;
          const pA = (pa / tot) * 100, pB = (pb / tot) * 100, winner = pA >= pB ? g.team_a : g.team_b;
          return (
            <div key={i} style={{padding:"12px 14px",borderRight:i%2===0?`1px solid ${PANEL_DIVIDER}`:"none",borderBottom:i<2?`1px solid ${PANEL_DIVIDER}`:"none"}}>
              <div style={{fontSize:9,color:"#888",marginBottom:8}}>{g.region} {g.seed}-seed — winner plays {g.plays}</div>
              {[{name:g.team_a,p:pA},{name:g.team_b,p:pB}].map(({name,p})=>(
                <div key={name} style={{display:"flex",alignItems:"center",gap:10,marginBottom:4}}>
                  <span style={{flex:1,fontSize:11,fontWeight:name===winner?600:400,color:name===winner?"#000":"#999"}}>{name}</span>
                  <span style={{fontSize:10}}>{pct(p)}</span>
                  {name===winner&&<span style={{fontSize:9}}>→</span>}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Model vs Market ──────────────────────────────────────────────────────
function ModelVsMarket({ sim }: { sim: SimState }) {
  const [teams, setTeams] = useState<Record<string,any>>({});

  useEffect(() => {
    fetch(`${API}/teams`).then(r => r.json()).then(setTeams).catch(() => {});
  }, []);

  if (!sim.complete || Object.keys(teams).length === 0) return null;

  const rows = Object.entries(sim.champion_pct)
    .filter(([, p]) => p > 0.3)
    .map(([name, modelPct]) => {
      const t = teams[name];
      const vegasPct = t?.championship_odds_pct || 0;
      const edge = modelPct - vegasPct;
      return { name, modelPct, vegasPct, edge, seed: t?.seed || 16 };
    })
    .sort((a, b) => Math.abs(b.edge) - Math.abs(a.edge));

  const maxModel = Math.max(...rows.map(r => r.modelPct), 1);

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Where our simulation disagrees with betting markets. Positive edge = model likes them more than Vegas.</div>
      </div>
      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}>
          <thead>
            <tr style={{borderBottom:"2px solid #000",background:"#000"}}>
              <th style={{padding:"8px 10px",textAlign:"left",color:"#fff",fontSize:9,fontWeight:600}}>TEAM</th>
              <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,fontWeight:600}}>MODEL</th>
              <th style={{padding:"8px 8px",textAlign:"center",color:"#fff",fontSize:9,fontWeight:600}}>VEGAS</th>
              <th style={{padding:"8px 10px",textAlign:"center",color:GREEN,fontSize:9,fontWeight:600}}>EDGE</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 16).map((r, i) => {
              const edgeColor = r.edge > 2 ? "#000" : r.edge < -2 ? "#999" : "#ccc";
              return (
                <tr key={r.name} style={{borderBottom:"1px solid #f4f4f4",background:i%2===0?"#fff":"#fafafa"}}>
                  <td style={{padding:"6px 10px",whiteSpace:"nowrap"}}>
                    <span style={{display:"inline-flex",alignItems:"center",gap:6}}>
                      <Seed n={r.seed}/>
                      <span style={{fontWeight:r.seed<=4?600:400}}>{r.name}</span>
                    </span>
                  </td>
                  <td style={{padding:"6px 8px",textAlign:"center"}}>
                    <div style={{display:"flex",alignItems:"center",gap:6,justifyContent:"center"}}>
                      <div style={{width:60,height:4,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
                        <div style={{width:`${(r.modelPct/maxModel)*100}%`,height:"100%",background:"#000",borderRadius:2}}/>
                      </div>
                      <span style={{minWidth:40}}>{pct(r.modelPct,1)}</span>
                    </div>
                  </td>
                  <td style={{padding:"6px 8px",textAlign:"center",color:"#666"}}>{pct(r.vegasPct,1)}</td>
                  <td style={{padding:"6px 10px",textAlign:"center",fontWeight:600,color:edgeColor}}>
                    {r.edge > 0 ? "+" : ""}{pct(r.edge,1)}
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

// ── Region Difficulty ────────────────────────────────────────────────────
function RegionDifficulty({ sim }: { sim: SimState }) {
  const [stats, setStats] = useState<any[]>([]);

  useEffect(() => {
    fetch(`${API}/region-stats`).then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  if (!sim.complete || stats.length === 0) return null;

  const maxEM = Math.max(...stats.map((s: any) => Math.abs(s.avg_em)), 1);
  const maxChamp = Math.max(...stats.map((s: any) => s.total_championship_pct), 1);

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Which regions are stacked and which have soft paths to the Final Four.</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)"}}>
        {stats.map((s: any, i: number) => {
          const difficulty = s.total_championship_pct;
          const rank = [...stats].sort((a: any, b: any) => b.total_championship_pct - a.total_championship_pct).findIndex((x: any) => x.region === s.region) + 1;
          const label = rank === 1 ? "TOUGHEST" : rank === 4 ? "EASIEST" : "";
          return (
            <div key={s.region} style={{padding:"14px 16px",borderRight:i<3?`1px solid ${PANEL_DIVIDER}`:"none"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
                <div style={{fontSize:11,fontWeight:700}}>{s.region.toUpperCase()}</div>
                {label && <span style={{fontSize:8,fontWeight:600,letterSpacing:"0.08em",padding:"2px 6px",border:`1px solid ${PANEL_BORDER}`,background:rank===1?"#000":"#fff",color:rank===1?"#fff":"#000"}}>{label}</span>}
              </div>
              <div style={{fontSize:9,color:"#888",marginBottom:4}}>Avg. efficiency margin</div>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:10}}>
                <div style={{flex:1,height:4,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
                  <div style={{width:`${(Math.abs(s.avg_em)/maxEM)*100}%`,height:"100%",background:"#000",borderRadius:2}}/>
                </div>
                <span style={{fontSize:11,fontWeight:600,minWidth:32}}>{s.avg_em > 0 ? "+" : ""}{s.avg_em}</span>
              </div>
              <div style={{fontSize:9,color:"#888",marginBottom:4}}>Combined title odds</div>
              <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:12}}>
                <div style={{flex:1,height:4,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
                  <div style={{width:`${(difficulty/maxChamp)*100}%`,height:"100%",background:rank<=2?"#000":"#bbb",borderRadius:2}}/>
                </div>
                <span style={{fontSize:11,fontWeight:600,minWidth:32}}>{pct(difficulty,0)}</span>
              </div>
              <div style={{fontSize:9,color:"#888",marginBottom:6}}>Top seeds</div>
              {s.top_seeds.map((t: any) => (
                <div key={t.name} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"3px 0",borderBottom:"1px solid #f4f4f4"}}>
                  <span style={{display:"inline-flex",alignItems:"center",gap:5}}>
                    <Seed n={t.seed}/>
                    <span style={{fontSize:10,fontWeight:500}}>{t.name}</span>
                  </span>
                  <span style={{fontSize:9,color:"#888"}}>{pct(t.championship_odds_pct,1)}</span>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── Seed History ─────────────────────────────────────────────────────────
function SeedHistory({ sim }: { sim: SimState }) {
  const [history, setHistory] = useState<Record<string,number>>({});

  useEffect(() => {
    fetch(`${API}/seed-history`).then(r => r.json()).then(setHistory).catch(() => {});
  }, []);

  if (!sim.complete || Object.keys(history).length === 0) return null;

  const upsets = (sim.upset_watch || []).slice(0, 8);
  const entries = Object.entries(history)
    .map(([seed, rate]) => ({seed: parseInt(seed), rate: rate as number}))
    .sort((a, b) => a.seed - b.seed);

  return (
    <div style={{border:`1px solid ${PANEL_BORDER}`,background:"#fff"}}>
      <div style={{padding:"10px 14px",borderBottom:`1px solid ${PANEL_DIVIDER}`,background:"#fafafa"}}>
        <div style={{fontSize:10,color:"#666"}}>Historical first-round win rate by seed. Context for how often upsets actually happen.</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr"}}>
        <div style={{padding:"14px 16px",borderRight:`1px solid ${PANEL_DIVIDER}`}}>
          <div style={{display:"flex",flexDirection:"column",gap:4}}>
            {entries.map(({seed, rate}) => (
              <div key={seed} style={{display:"grid",gridTemplateColumns:"28px 1fr 44px",alignItems:"center",gap:8}}>
                <Seed n={seed}/>
                <div style={{height:5,background:"#f0f0f0",borderRadius:3,overflow:"hidden"}}>
                  <div style={{width:`${rate*100}%`,height:"100%",background:rate>0.8?"#000":rate>0.5?"#555":rate>0.3?"#999":"#ccc",borderRadius:3}}/>
                </div>
                <span style={{fontSize:10,fontWeight:rate>0.5?600:400,textAlign:"right"}}>{(rate*100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{padding:"14px 16px"}}>
          <div style={{fontSize:9,letterSpacing:"0.1em",color:"#888",marginBottom:10,textTransform:"uppercase"}}>This year's upset candidates vs history</div>
          {upsets.map((u: any, i: number) => {
            const historicalUpsetRate = 1 - (history[String(u.dog_seed)] || 0);
            return (
              <div key={i} style={{marginBottom:8,paddingBottom:8,borderBottom:"1px solid #f4f4f4"}}>
                <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
                  <span style={{fontSize:10,fontWeight:600}}>({u.dog_seed}) {u.underdog} over ({u.fav_seed}) {u.favorite}</span>
                  <span style={{fontSize:10,fontWeight:600}}>{pct(u.upset_prob,0)}</span>
                </div>
                <div style={{fontSize:9,color:"#888"}}>
                  Historical: {u.dog_seed}-seeds win {(historicalUpsetRate*100).toFixed(0)}% of R64 games
                  {u.upset_prob > historicalUpsetRate * 100 + 5 &&
                    <span style={{color:"#000",fontWeight:600}}> — model sees ABOVE-average upset chance</span>
                  }
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [sim, setSim]         = useState<SimState>(EMPTY);
  const [logLines, setLog]    = useState<string[]>([]);
  const [resolved, setResolved] = useState<Resolved>({});
  const [running, setRunning]   = useState(false);
  const [elapsed, setElapsed]   = useState(0);
  const [teamsCatalog, setTeamsCatalog] = useState<Record<string, any>>({});
  const [overrides, setOverrides] = useState<TeamOverrides>({});
  const [forcedPicks, setForcedPicks] = useState<ForcedPicks>({});
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const esRef = useRef<EventSource|null>(null);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    fetch(`${API}/teams`).then(r => r.json()).then(setTeamsCatalog).catch(() => {});
    fetch(`${API}/model-info`).then(r => r.json()).then(setModelInfo).catch(() => {});
  }, []);

  const startSim = useCallback((cfg:SimRunConfig)=>{
    if(esRef.current){ esRef.current.close(); esRef.current=null; }
    setSim({...EMPTY,total:cfg.n_sims});
    setResolved({});
    setLog([`connecting...`]);
    setRunning(true);
    startTimeRef.current = Date.now();

    const encodedOverrides = encodeURIComponent(JSON.stringify(cfg.team_overrides || {}));
    const encodedForced = encodeURIComponent(JSON.stringify(cfg.forced_picks || {}));
    const url = `${API}/simulate/stream?n_sims=${cfg.n_sims}&emit_every=${Math.max(100,Math.floor(cfg.n_sims/40))}&latent_sigma=${cfg.latent_sigma}&team_overrides=${encodedOverrides}&forced_picks=${encodedForced}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (e)=>{
      const data = JSON.parse(e.data);

      if(data.type==="start"){
        setLog(prev=>[...prev,
          `simulation started`,
          `${data.n_sims.toLocaleString()} runs · sigma=${cfg.latent_sigma} · random seed`,
          `${data.assumptions_applied || 0} team assumptions applied`,
          `${data.forced_picks_applied || 0} game picks locked`,
          ...Object.entries(cfg.team_overrides || {})
            .slice(0, 4)
            .map(([name, v]) => `assumption: ${name} ${v > 0 ? "+" : ""}${v} elo`),
          ...Object.entries(cfg.forced_picks || {})
            .slice(0, 4)
            .map(([k, winner]) => `pick: ${k} -> ${winner}`),
          `model: calibrated_ml_stack`,
        ]);
      }

      if(data.type==="progress"){
        const elapsed = (Date.now()-startTimeRef.current)/1000;
        const sps = elapsed>0 ? data.done/elapsed : 0;
        setSim(prev=>({
          ...prev,
          champion_pct:data.champion_pct,
          final_four_pct:data.final_four_pct,
          done:data.done,
          total:data.total,
          sims_per_sec:sps,
        }));
        const t1=Object.entries(data.champion_pct as Record<string,number>).sort((a,b)=>b[1]-a[1])[0];
        if(t1) setLog(prev=>[...prev.slice(-80),
          `[${data.done.toLocaleString()}/${data.total.toLocaleString()}] ${t1[0]} leading at ${t1[1].toFixed(1)}%  (${sps.toFixed(0)} sims/sec)`
        ]);
      }

      if(data.type==="game"){
        setSim(prev=>{
          const pb={...prev.predicted_bracket};
          if(!pb[data.region]) pb[data.region]=[];
          const rounds=[...(pb[data.region]||[])];
          if(!rounds[data.round]) rounds[data.round]=[];
          const round=[...rounds[data.round]];
          round[data.gi]=data.game;
          rounds[data.round]=round;
          pb[data.region]=rounds;
          return {...prev,predicted_bracket:pb,bracketStarted:true};
        });
        setResolved(prev=>({
          ...prev,
          [data.region]:{
            ...(prev[data.region]||{}),
            [data.round]:{...((prev[data.region]||{})[data.round]||{}),[data.gi]:true}
          }
        }));
      }

      if(data.type==="complete"){
        setSim(prev=>({
          ...prev,
          champion_pct:data.champion_pct,
          title_game_pct:data.title_game_pct||{},
          final_four_pct:data.final_four_pct,
          elite_eight_pct:data.elite_eight_pct,
          sweet_sixteen_pct:data.sweet_sixteen_pct,
          round_of_32_pct:data.round_of_32_pct,
          first_four_pct:data.first_four_pct||{},
          predicted_bracket:data.predicted_bracket,
          upset_watch:data.upset_watch,
          matchup_probs:data.matchup_probs||{},
          n_sims:data.n_sims,
          done:data.n_sims,
          total:data.n_sims,
          model_used:data.model_used,
          elapsed_sec:data.elapsed_sec,
          complete:true,
          bracketStarted:true,
        }));
        const champ=Object.entries(data.champion_pct as Record<string,number>).sort((a,b)=>b[1]-a[1])[0];
        setLog(prev=>[...prev,``,`complete — ${data.elapsed_sec}s`,`champion: ${champ?.[0]} ${champ?.[1].toFixed(1)}%`]);
        setRunning(false);
        es.close();
      }

      if(data.type==="error"){
        setLog(prev=>[...prev,`error: ${data.message}`]);
        setRunning(false); es.close();
      }
    };
    es.onerror=()=>{
      setLog(prev=>[...prev,"connection error"]);
      setRunning(false); es.close();
    };
  },[]);

  useEffect(()=>()=>esRef.current?.close(),[]);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      setElapsed((Date.now() - startTimeRef.current) / 1000);
    }, 100);
    return () => clearInterval(id);
  }, [running]);

  const simTeams = Object.keys(sim.predicted_bracket).length>0
    ? [...new Set(Object.values(sim.predicted_bracket).flatMap(rounds=>(rounds[0]||[]).flatMap((g:Game)=>[g.team_a,g.team_b])))].sort()
    : [];
  const catalogTeams = Object.keys(teamsCatalog).sort();
  const allTeams = simTeams.length > 0 ? simTeams : catalogTeams;
  const hasFF = sim.complete&&Object.keys(sim.first_four_pct||{}).length>0;
  const champ = top(sim.champion_pct,1)[0];
  const toggleForcedPick = useCallback((key: string, winner: string) => {
    setForcedPicks(prev => prev[key] === winner
      ? Object.fromEntries(Object.entries(prev).filter(([k]) => k !== key))
      : ({ ...prev, [key]: winner })
    );
  }, []);
  const leaderName = champ?.[0] || "Waiting for simulation";
  const leaderPct = champ ? pct(champ[1]) : "—";
  const leaderPctNum = champ?.[1] ?? 0;
  const finalFourTop = top(sim.final_four_pct || {}, 4);
  const ffMax = finalFourTop[0]?.[1] || 1;
  const lockedCount = Object.keys(forcedPicks).length;

  return (
    <div style={{maxWidth:1280,margin:"0 auto",padding:"20px 18px 52px"}}>
      <div style={{padding:"2px 2px 12px",display:"flex",justifyContent:"space-between",alignItems:"flex-end"}}>
        <div>
          <div style={{fontSize:10,color:"#9ca3af",letterSpacing:"0.14em",marginBottom:3}}>2026 NCAA TOURNAMENT</div>
          <div style={{fontSize:24,fontWeight:700,letterSpacing:"-0.02em"}}>Bracket Simulator</div>
        </div>
        <a href="https://github.com/alexh212" target="_blank" rel="noopener noreferrer" style={{fontSize:10,color:"#6b7280",textDecoration:"none",fontWeight:500}}>github.com/alexh212 ↗</a>
      </div>

      <div style={{border:"1px solid #ebeff5",borderRadius:14,padding:12,marginBottom:14,background:"#fff"}}>
        <div className="hero-grid">
          <div style={{padding:"8px 10px",border:"1px solid #f1f5f9",borderRadius:10}}>
            <div style={{fontSize:9,letterSpacing:"0.12em",color:"#9ca3af",marginBottom:4,textTransform:"uppercase"}}>{sim.complete?"Projected Champion":"Current Leader"}</div>
            <div style={{fontSize:26,fontWeight:700,lineHeight:1.05}}>{leaderName}</div>
            <div style={{fontSize:12,color:"#6b7280",marginTop:5}}>{leaderPct}</div>
            {leaderPctNum > 0 && (
              <div style={{height:3,background:"#f0f0f0",borderRadius:2,overflow:"hidden",marginTop:8}}>
                <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${Math.min(100, leaderPctNum)}%`,height:"100%",background:sim.complete?GREEN:ACCENT,transition:"width 0.5s ease",borderRadius:2}}/>
              </div>
            )}
          </div>
          <div style={{padding:"8px 10px",border:"1px solid #f1f5f9",borderRadius:10}}>
            <div style={{fontSize:9,letterSpacing:"0.12em",color:"#9ca3af",marginBottom:5,textTransform:"uppercase"}}>Final Four Leaders</div>
            <div style={{display:"grid",gap:6}}>
              {finalFourTop.map(([t,p],i)=>(
                <div key={t}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:2}}>
                    <span style={{fontWeight:500}}>{t}</span>
                    <span style={{color:"#6b7280"}}>{pct(p)}</span>
                  </div>
                  <div style={{height:3,background:"#f0f0f0",borderRadius:2,overflow:"hidden"}}>
                    <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${(p/ffMax)*100}%`,height:"100%",background:i===0?ACCENT:"#ccc",transition:"width 0.5s ease",borderRadius:2}}/>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{padding:"8px 10px",border:"1px solid #f1f5f9",borderRadius:10,display:"grid",alignContent:"space-between"}}>
            <div style={{fontSize:10,color:"#6b7280",lineHeight:1.6}}>
              {sim.done>0?`${sim.done.toLocaleString()} sims run`:"Ready to simulate"}<br/>
              {sim.complete
                ? `${sim.elapsed_sec}s elapsed`
                : running
                  ? <span style={{color:"#111",fontWeight:600}}>{elapsed.toFixed(1)}s elapsed</span>
                  : "Waiting to start"}
            </div>
            <div style={{fontSize:10,color:"#9ca3af"}}>{lockedCount} picks locked</div>
          </div>
        </div>
      </div>

      <div className="dashboard-grid">
        <div className="stack">
          <Collapse label="Team Deep Dive" defaultOpen={true}>
            <TeamProbabilityMath sim={sim} teams={catalogTeams}/>
          </Collapse>

          <GroupTitle label="Run & Track" hint="simulate and monitor live output" />
          <Collapse label="Simulation Setup" defaultOpen={true}>
            <SimControls
              running={running}
              onRun={(cfg)=>startSim({...cfg, team_overrides: overrides, forced_picks: forcedPicks})}
              assumptionCount={Object.keys(overrides).length}
              lockedPickCount={Object.keys(forcedPicks).length}
            />
          </Collapse>
          <Collapse label="Live Simulation" defaultOpen={true}>
            <LiveSimPanel sim={sim} logLines={logLines}/>
          </Collapse>

          <GroupTitle label="Pick & Scenario" hint="interactive bracket + assumptions" />
          <Collapse label="Bracket Picks & Path" defaultOpen={true}>
            <InitialPickTree
              teamsCatalog={teamsCatalog}
              sim={sim}
              forcedPicks={forcedPicks}
              onTogglePick={toggleForcedPick}
              onClearPicks={()=>setForcedPicks({})}
            />
          </Collapse>
          <Collapse label="Assumptions" defaultOpen={false}>
            <AssumptionsPanel
              teams={catalogTeams}
              overrides={overrides}
              onChange={setOverrides}
              running={running}
            />
          </Collapse>
        </div>

        <div className="stack insights-rail" style={{position:"sticky",top:12}}>
          <GroupTitle label="Insights" hint="probability surfaces and diagnostics" />
          {sim.complete ? (
            <>
              <Collapse label="Simulation Quality" defaultOpen={true}>
                <DiagnosticsSection sim={sim} overrides={overrides}/>
              </Collapse>
              <Collapse label="Championship Probability" defaultOpen={false}>
                <ChampionshipChart sim={sim}/>
              </Collapse>
              <Collapse label="Advancement Heatmap — Top 8 Contenders" defaultOpen={false}>
                <OddsComparisonChart sim={sim}/>
              </Collapse>
              <Collapse label="Upset Watch" defaultOpen={false}>
                <UpsetChart sim={sim}/>
              </Collapse>
              <Collapse label="Model vs Vegas" defaultOpen={false}>
                <ModelVsMarket sim={sim}/>
              </Collapse>
              <Collapse label="Region Difficulty" defaultOpen={false}>
                <RegionDifficulty sim={sim}/>
              </Collapse>
              <Collapse label="Seed Win Rates" defaultOpen={false}>
                <SeedHistory sim={sim}/>
              </Collapse>
              {hasFF && (
                <Collapse label="First Four" defaultOpen={false}>
                  <FirstFourSection sim={sim}/>
                </Collapse>
              )}
              {allTeams.length>0 && (
                <Collapse label="Head-to-Head Analysis" defaultOpen={false}>
                  <AnalysisSection allTeams={allTeams}/>
                </Collapse>
              )}
              <Collapse label="Advancement Table" defaultOpen={false}>
                <div style={{maxHeight:520,overflowY:"auto"}}>
                  <AdvancementTable sim={sim}/>
                </div>
              </Collapse>
            </>
          ) : (
            <div style={{border:"1px dashed #e5e7eb",borderRadius:10,padding:"12px 14px",fontSize:11,color:"#6b7280",background:"#fff"}}>
              Run a simulation to unlock advanced insights.
            </div>
          )}

          <GroupTitle label="Model details" hint="data and training log" />
          <Collapse label="Data & model" defaultOpen={false}>
            <DataProvenance info={modelInfo}/>
            <ModelTrainingOutput/>
          </Collapse>
        </div>
      </div>

      <div style={{borderTop:"1px solid #e8e8e8",marginTop:48,paddingTop:14,display:"flex",flexDirection:"column",alignItems:"flex-end",gap:4}}>
        <a href="https://github.com/alexh212" target="_blank" rel="noopener noreferrer" style={{fontSize:9,color:"#bbb",textDecoration:"none"}}>github.com/alexh212</a>
        <span style={{fontSize:9,color:"#bbb"}}>LR + XGBoost + LightGBM · isotonic calibration · latent draws · rolling-origin CV · 77 R64 games 2005-2025 · 20/20 benchmarks</span>
        <span style={{fontSize:9,color:"#bbb"}}>Next.js · React · TypeScript · Recharts</span>
      </div>
    </div>
  );
}
