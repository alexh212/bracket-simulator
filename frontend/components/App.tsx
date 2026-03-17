"use client";
import { useEffect, useRef, useState, memo } from "react";

import type { ForcedPicks, Game, Resolved, SimRunConfig, SimState, TeamOverrides } from "@/components/app/types";
import {
  ACCENT,
  ACCENT_SOFT,
  BG_ALT,
  BG_HEADER,
  BORDER_INNER,
  BORDER_OUTER,
  BORDER_SUBTLE,
  Collapse,
  GREEN,
  GroupTitle,
  PAD_BODY,
  PAD_HEADER,
  R64_ORDER,
  REGIONS,
  ROUNDS,
  SH,
  SectionGap,
  Seed,
  pct,
  pickKey,
  toML,
  top,
} from "@/components/app/shared";
import {
  AnalysisSection,
  DataProvenance,
  DiagnosticsSection,
  FirstFourSection,
  ModelTrainingOutput,
  ModelVsMarket,
  RegionDifficulty,
  SeedHistory,
  TeamProbabilityMath,
} from "@/components/app/insights";
import { useSimulation } from "@/components/app/useSimulation";

// ── Sim Controls ──────────────────────────────────────────────────────────────
function SimControls({ running, onRun, assumptionCount, lockedPickCount }: {
  running: boolean;
  onRun: (cfg: {n_sims:number; latent_sigma:number}) => void;
  assumptionCount: number;
  lockedPickCount: number;
}) {
  const [nSims, setNSims] = useState(10000);
  const [sigma, setSigma] = useState(0.06);

  const NSIMS_OPTS = [250,500,1000,2500,5000,10000,25000,50000];
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
          <span>250</span><span>500</span><span>1k</span><span>2.5k</span><span>5k</span><span>10k</span><span>25k</span><span>50k</span>
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
            color:running?"#aaa":"#fff",border:"none",borderRadius:14,
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
    <div style={{border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
      <div style={{padding:PAD_HEADER,borderBottom:`1px solid ${BORDER_SUBTLE}`,background:BG_HEADER}}>
        <div style={{fontSize:10,color:"#666"}}>Apply team strength deltas (Elo points) before simulation. Positive favors the team, negative fades it.</div>
      </div>
      <div style={{padding:"10px 14px",display:"grid",gridTemplateColumns:"1fr 1fr auto",gap:10,alignItems:"end",borderBottom:`1px solid ${BORDER_SUBTLE}`}}>
        <div>
          <div style={{fontSize:9,color:"#888",marginBottom:6,textTransform:"uppercase",letterSpacing:"0.08em"}}>Team</div>
          <select value={team} onChange={(e)=>setTeam(e.target.value)} disabled={running} style={{width:"100%",height:32,border:`1px solid ${BORDER_OUTER}`,padding:"0 8px",fontSize:11,background:"#fff"}}>
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
          style={{height:32,padding:"0 14px",border:`1px solid ${BORDER_OUTER}`,background:"#000",color:"#fff",fontSize:10,fontWeight:600,letterSpacing:"0.06em",cursor:running?"not-allowed":"pointer"}}>
          APPLY
        </button>
      </div>
      <div style={{padding:"8px 10px",display:"flex",justifyContent:"space-between",alignItems:"center",gap:10}}>
        <div style={{display:"flex",flexWrap:"wrap",gap:6,flex:1}}>
          {!hasAssumptions && <span style={{fontSize:10,color:"#aaa"}}>No custom assumptions active.</span>}
          {entries.slice(0, 8).map(([name, val])=>(
            <span key={name} style={{display:"inline-flex",alignItems:"center",gap:6,padding:"3px 8px",border:`1px solid ${BORDER_OUTER}`,fontSize:10,background:"#fff"}}>
              <span>{name}</span>
              <span style={{fontWeight:700}}>{val>0?"+":""}{val}</span>
              <button onClick={()=>removeAssumption(name)} disabled={running}
                style={{border:"none",background:"transparent",fontSize:11,cursor:"pointer",color:"#888",padding:0}}>×</button>
            </span>
          ))}
        </div>
        {hasAssumptions && (
          <button onClick={()=>onChange({})} disabled={running}
            style={{height:28,padding:"0 10px",border:`1px solid ${BORDER_OUTER}`,background:"#fff",fontSize:10,cursor:"pointer"}}>
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
        <div style={{border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
          <div style={{padding:"7px 10px",borderBottom:`1px solid ${BORDER_INNER}`,fontSize:9,color:"#777",letterSpacing:"0.08em",textTransform:"uppercase"}}>{label}</div>
          <div style={{padding:8,display:"grid",gridTemplateColumns:pickable?"repeat(2,minmax(0,1fr))":"repeat(3,minmax(0,1fr))",gap:10}}>
            {(pickable ? r64 : games).map((g: any, idx: number) => {
              if (pickable) {
                const key = g.key;
                return (
                  <div key={key} style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,overflow:"hidden",background:"#fff"}}>
                    <div style={{padding:"3px 8px",background:BG_HEADER,borderBottom:`1px solid ${BORDER_OUTER}`,display:"flex",alignItems:"center",justifyContent:"center",gap:4,fontSize:9,color:"#888"}}>
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
                            border:"none",borderTop:ti===1?`1px solid ${BORDER_INNER}`:"none",
                            background:locked?"#111":"#fff",
                            color:locked?"#fff":"#111",cursor:"pointer",textAlign:"left"
                          }}
                        >
                          <Seed n={seed}/>
                          <span style={{flex:1,fontSize:10,fontWeight:locked?700:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{name}</span>
                          {locked && <span style={{fontSize:7,background:"#fff",color:"#111",padding:"1px 4px",borderRadius:14,fontWeight:700,letterSpacing:"0.04em"}}>LOCK</span>}
                          {locked && <span style={{fontSize:11,color:locked?"#22c55e":"#fff",lineHeight:1}}>✓</span>}
                        </button>
                      );
                    })}
                  </div>
                );
              }
              const pA = g?.win_prob_a ?? 50;
              const pB = 100 - pA;
              return (
                <div key={`${label}-${idx}`} style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,overflow:"hidden",background:"#fff"}}>
                  <div style={{padding:"3px 8px",background:BG_HEADER,borderBottom:`1px solid ${BORDER_OUTER}`,display:"flex",alignItems:"center",justifyContent:"center",gap:4,fontSize:9,color:"#888"}}>
                    <span style={{fontWeight:600,color:"#555"}}>({g?.seed_a})</span>
                    <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g?.team_a}</span>
                    <span style={{color:"#ccc",fontWeight:700,fontSize:8}}>vs</span>
                    <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g?.team_b}</span>
                    <span style={{fontWeight:600,color:"#555"}}>({g?.seed_b})</span>
                  </div>
                  {[{name:g?.team_a,seed:g?.seed_a,p:pA},{name:g?.team_b,seed:g?.seed_b,p:pB}].map((t: any, ti: number) => {
                    const won = g?.winner === t.name;
                    return (
                      <div key={t.name} style={{display:"flex",alignItems:"center",gap:6,padding:"5px 8px",borderTop:ti===1?`1px solid ${BORDER_INNER}`:"none",background:won?BG_ALT:"#fff"}}>
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
        <div style={{padding:"6px 10px",fontSize:10,fontWeight:600,letterSpacing:"0.08em",color:"#333",border:`1px solid ${BORDER_OUTER}`,background:BG_HEADER}}>
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

  // Derive F4 and Title matchups from predicted bracket (East vs Midwest = SF1, West vs South = SF2)
  const getRegionChamp = (region: string): Game | null => {
    const rounds = sim.predicted_bracket[region];
    if (!rounds?.[3]?.[0]) return null;
    return rounds[3][0] as Game;
  };
  const matchupProb = (a: string, b: string): number => {
    const mp = sim.matchup_probs || {};
    const key1 = `${a} vs ${b}`;
    const key2 = `${b} vs ${a}`;
    if (mp[key1] != null) return mp[key1];
    if (mp[key2] != null) return 100 - mp[key2];
    return 50;
  };
  const sf1 = getRegionChamp("East") && getRegionChamp("Midwest")
    ? { team_a: getRegionChamp("East")!.winner, team_b: getRegionChamp("Midwest")!.winner, seed_a: teamsCatalog[getRegionChamp("East")!.winner]?.seed ?? 0, seed_b: teamsCatalog[getRegionChamp("Midwest")!.winner]?.seed ?? 0 }
    : null;
  const sf2 = getRegionChamp("West") && getRegionChamp("South")
    ? { team_a: getRegionChamp("West")!.winner, team_b: getRegionChamp("South")!.winner, seed_a: teamsCatalog[getRegionChamp("West")!.winner]?.seed ?? 0, seed_b: teamsCatalog[getRegionChamp("South")!.winner]?.seed ?? 0 }
    : null;
  const sf1Prob = sf1 ? matchupProb(sf1.team_a, sf1.team_b) : 50;
  const sf2Prob = sf2 ? matchupProb(sf2.team_a, sf2.team_b) : 50;
  const sf1Winner = sf1 ? (sf1Prob >= 50 ? sf1.team_a : sf1.team_b) : null;
  const sf2Winner = sf2 ? (sf2Prob >= 50 ? sf2.team_a : sf2.team_b) : null;
  const titleGame = sf1Winner && sf2Winner
    ? { team_a: sf1Winner, team_b: sf2Winner, seed_a: teamsCatalog[sf1Winner]?.seed ?? 0, seed_b: teamsCatalog[sf2Winner]?.seed ?? 0 }
    : null;
  const titleProb = titleGame ? matchupProb(titleGame.team_a, titleGame.team_b) : 50;
  const titleWinner = titleGame ? (titleProb >= 50 ? titleGame.team_a : titleGame.team_b) : null;

  const renderGameCard = (g: { team_a: string; team_b: string; seed_a: number; seed_b: number } | null, pA: number, winner: string | null) => {
    if (!g) return <div style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,background:BG_HEADER,minHeight:70}}/>;
    const pB = 100 - pA;
    const aWon = winner === g.team_a;
    return (
      <div style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,overflow:"hidden",background:"#fff"}}>
        <div style={{padding:"3px 8px",background:BG_HEADER,borderBottom:`1px solid ${BORDER_OUTER}`,display:"flex",alignItems:"center",justifyContent:"center",gap:6,fontSize:9,color:"#888"}}>
          <span style={{fontWeight:600,color:"#555"}}>({g.seed_a})</span>
          <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g.team_a}</span>
          <span style={{color:"#ccc",fontWeight:700,fontSize:8}}>vs</span>
          <span style={{overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",maxWidth:70}}>{g.team_b}</span>
          <span style={{fontWeight:600,color:"#555"}}>({g.seed_b})</span>
        </div>
        {[{name:g.team_a,seed:g.seed_a,p:pA},{name:g.team_b,seed:g.seed_b,p:pB}].map((t: any, ti: number) => {
          const won = winner === t.name;
          return (
            <div key={t.name} style={{display:"flex",alignItems:"center",gap:6,padding:"5px 8px",borderTop:ti===1?`1px solid ${BORDER_INNER}`:"none",background:won?BG_ALT:"#fff"}}>
              <Seed n={t.seed || 0}/>
              <span style={{flex:1,fontSize:10,fontWeight:won?700:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{t.name}</span>
              <span style={{fontSize:9,color:won?"#111":"#999",fontWeight:won?600:400}}>{pct(t.p,1)}</span>
              {won && <span style={{fontSize:11,color:GREEN,lineHeight:1}}>✓</span>}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
      <div style={{padding:PAD_HEADER,borderBottom:`1px solid ${BORDER_SUBTLE}`,display:"flex",justifyContent:"space-between",alignItems:"center",background:BG_HEADER}}>
        <span style={{fontSize:10,color:"#444",letterSpacing:"0.06em"}}>Lock first-round picks, then run the sim.</span>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:10,color:"#666"}}>{count} locked</span>
          {count > 0 && (
            <button onClick={onClearPicks} style={{fontSize:9,padding:"3px 8px",border:`1px solid ${BORDER_OUTER}`,background:"#fff",cursor:"pointer"}}>
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
              border:`1px solid ${BORDER_OUTER}`,
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
      <div style={{borderTop:`1px solid ${BORDER_SUBTLE}`,padding:"10px",display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        <div style={{border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
          <div style={{padding:"7px 10px",borderBottom:`1px solid ${BORDER_INNER}`,fontSize:9,color:"#777",letterSpacing:"0.08em",textTransform:"uppercase"}}>Final Four Outlook</div>
          <div style={{padding:8,display:"grid",gridTemplateColumns:"repeat(2,minmax(0,1fr))",gap:10}}>
            <div key="sf1">{renderGameCard(sf1, sf1Prob, sf1Winner)}</div>
            <div key="sf2">{renderGameCard(sf2, sf2Prob, sf2Winner)}</div>
          </div>
        </div>
        <div style={{border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
          <div style={{padding:"7px 10px",borderBottom:`1px solid ${BORDER_INNER}`,fontSize:9,color:"#777",letterSpacing:"0.08em",textTransform:"uppercase"}}>Title Game Outlook</div>
          <div style={{padding:8,display:"grid",gridTemplateColumns:"1fr",gap:10}}>
            <div key="title">{renderGameCard(titleGame, titleProb, titleWinner)}</div>
          </div>
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
      <div style={{background:BG_HEADER,padding:PAD_HEADER,display:"flex",alignItems:"center",justifyContent:"space-between",borderBottom:`1px solid ${BORDER_SUBTLE}`}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:10,fontWeight:600,letterSpacing:"0.1em",color:"#444"}}>RUN STATUS</span>
          {sim.done>0&&!sim.complete&&<span style={{width:6,height:6,borderRadius:14,background:GREEN,display:"inline-block",animation:"blink 1s step-end infinite"}}/>}
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
        <div style={{borderRight:`1px solid ${BORDER_SUBTLE}`,padding:PAD_BODY}}>
          {/* Progress bar */}
          <div style={{marginBottom:14}}>
            <div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:"#888",marginBottom:5}}>
              <span>{sim.done.toLocaleString()} / {sim.total.toLocaleString()} simulations</span>
              <span style={{fontWeight:600,color:sim.complete?GREEN:"#000"}}>{(progress*100).toFixed(0)}%</span>
            </div>
            <div style={{height:4,background:"#f0f0f0",borderRadius:14,overflow:"hidden"}}>
              <div className={!sim.complete && sim.done > 0 ? "progress-bar-running" : ""} style={{height:"100%",width:`${progress*100}%`,background:sim.complete?GREEN:ACCENT,transition:"width 0.4s ease",borderRadius:14}}/>
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
                  <div style={{height:3,background:"#f0f0f0",borderRadius:14,overflow:"hidden"}}>
                    <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${(p/maxPct)*100}%`,height:"100%",background:i===0?ACCENT:"#ccc",transition:"width 0.5s ease",borderRadius:14}}/>
                  </div>
                </div>
              ))}
              {sim.complete&&top(sim.champion_pct,1)[0]&&(
                <div style={{marginTop:6,fontSize:10,color:"#666",borderTop:`1px solid ${BORDER_SUBTLE}`,paddingTop:8}}>
                  Implied odds: {top(sim.champion_pct,1)[0][0]} at {toML(top(sim.champion_pct,1)[0][1]/100)}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right: console log */}
        <div ref={logRef} className="terminal-log" style={{height:220,overflowY:"auto",background:"#0d0d0d",padding:"10px 14px",fontSize:10,lineHeight:1.9,color:"#555",borderLeft:`1px solid ${BORDER_SUBTLE}`,scrollbarWidth:"none"} as React.CSSProperties}>
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
    <div style={{overflowX:"auto",border:`1px solid ${BORDER_OUTER}`,background:"#fff"}}>
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
              <tr key={t.name} style={{borderBottom:`1px solid ${BORDER_SUBTLE}`,background:i%2===0?"#fff":BG_HEADER}}>
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
      <div style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,background:BG_HEADER,marginBottom:6,overflow:"hidden"}}>
        <div style={{padding:"5px 10px",minHeight:22,display:"flex",alignItems:"center",gap:6,borderBottom:`1px solid ${BORDER_SUBTLE}`}}>
          <div style={{width:20,height:20,borderRadius:14,border:"1px solid #e0e0e0",background:BG_ALT,flexShrink:0}}/>
          <div style={{height:8,background:BG_ALT,flex:1,borderRadius:14}}/>
        </div>
        <div style={{padding:"5px 10px",minHeight:22,display:"flex",alignItems:"center",gap:6}}>
          <div style={{width:20,height:20,borderRadius:14,border:"1px solid #e0e0e0",background:BG_ALT,flexShrink:0}}/>
          <div style={{height:8,background:BG_ALT,width:"60%",borderRadius:14}}/>
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
      borderRadius:14,background:seed<=4?"#111":BORDER_SUBTLE,color:seed<=4?"#fff":"#444",
    }}>{seed}</span>
  );

  const TeamRow = ({team,seed,score,won,p,isForced}:{team:string;seed:number;score?:number;won:boolean;p:number;isForced:boolean}) => (
    <button
      onClick={()=>onTogglePick(key, team)}
      style={{
        display:"flex",alignItems:"center",gap:8,padding:"6px 10px",border:"none",width:"100%",textAlign:"left",
        background:won?BG_ALT:"#fff",minHeight:26,cursor:"pointer",
        borderLeft:won?`3px solid ${ACCENT}`:"3px solid transparent",
      }}
      title={isForced ? "Click to clear pick" : "Click to force this winner"}
    >
      <SeedBadge seed={seed}/>
      <span style={{flex:1,fontSize:11,fontWeight:won?700:400,color:won?"#000":"#777",overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{team}</span>
      <span style={{fontSize:10,fontWeight:won?600:400,color:won?"#111":"#aaa",minWidth:38,textAlign:"right"}}>{pct(p,1)}</span>
      {score!=null&&<span style={{fontSize:9,fontWeight:won?700:400,color:won?"#000":"#ccc",minWidth:18,textAlign:"right"}}>{score}</span>}
      {isForced
        ? <span style={{fontSize:7,color:"#fff",background:"#111",padding:"1px 4px",borderRadius:14,fontWeight:700,letterSpacing:"0.04em"}}>LOCK</span>
        : won
          ? <span style={{fontSize:12,color:"#22c55e",marginLeft:2,lineHeight:1}}>✓</span>
          : <span style={{width:14}}/>
      }
    </button>
  );

  return (
    <div style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,background:"#fff",marginBottom:6,overflow:"hidden",animation:"fadeIn 0.3s ease"}}>
      <div style={{
        display:"flex",alignItems:"center",justifyContent:"center",gap:6,
        padding:"3px 8px",background:BG_HEADER,borderBottom:`1px solid ${BORDER_OUTER}`,
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
        <div style={{flex:1,height:1,background:BORDER_SUBTLE}}/>
        <span style={{fontSize:8,color:"#ccc",fontWeight:600}}>VS</span>
        <div style={{flex:1,height:1,background:BORDER_SUBTLE}}/>
      </div>
      <TeamRow team={g.team_b} seed={g.seed_b} score={g.score_b} won={!aWon} p={pB} isForced={forced===g.team_b}/>
      <div style={{height:3,background:"#f0f0f0",borderRadius:"0 0 14px 14px"}}>
        <div style={{width:`${pA}%`,height:"100%",background:aWon?"#111":"#888",borderRadius:"0 0 0 14px"}}/>
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
    <div style={{flex:1,borderRight:`1px solid ${BORDER_SUBTLE}`}}>
      <div style={{background:"#000",padding:"5px 8px"}}>
        <span style={{fontSize:9,fontWeight:600,letterSpacing:"0.1em",color:"#fff"}}>{region.toUpperCase()}</span>
      </div>
      {[0,1,2,3].map(ri=>{
        const count = ri===0?8:ri===1?4:ri===2?2:1;
        return (
          <div key={ri}>
            <div style={{padding:"2px 8px",background:BG_ALT,borderBottom:`1px solid ${BORDER_SUBTLE}`,borderTop:ri>0?`1px solid ${BORDER_SUBTLE}`:"none"}}>
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
  const finalFourRounds = sim.predicted_bracket["FinalFour"] || [];
  const finalFourGames = finalFourRounds[0] || [];
  const titleGames = finalFourRounds[1] || [];
  return (
    <div style={{border:`1px solid ${BORDER_OUTER}`,overflowX:"auto",background:"#fff"}}>
      <div style={{padding:"6px 10px",background:BG_ALT,borderBottom:`1px solid ${BORDER_SUBTLE}`,display:"flex",alignItems:"center",justifyContent:"space-between",gap:12}}>
        <span style={{fontSize:9,color:"#666",letterSpacing:"0.06em"}}>Click any team to lock a winner. Locked picks are used next simulation run while all other games are simulated.</span>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <span style={{fontSize:9,color:"#444"}}>{pickCount} picks locked</span>
          {pickCount > 0 && (
            <button onClick={onClearPicks} style={{fontSize:9,padding:"3px 8px",border:`1px solid ${BORDER_OUTER}`,background:"#fff",cursor:"pointer"}}>
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
        <div style={{borderTop:`1px solid ${BORDER_INNER}`,display:"grid",gridTemplateColumns:"1fr 1fr 1fr"}}>
          {[
            {label:"FINAL FOUR · SF1", game:finalFourGames[0]},
            {label:"FINAL FOUR · SF2", game:finalFourGames[1]},
            {label:"NATIONAL CHAMPIONSHIP", game:titleGames[0]},
          ].map(({label,game},i)=>(
            <div key={label} style={{padding:"10px 12px",borderRight:i<2?"1px solid #000":"none"}}>
              <div style={{fontSize:8,letterSpacing:"0.1em",fontWeight:700,color:"#888",marginBottom:6}}>{label}</div>
              {game ? (
                [
                  {team: game.team_a, p: game.win_prob_a, won: game.winner === game.team_a},
                  {team: game.team_b, p: 100 - game.win_prob_a, won: game.winner === game.team_b},
                ].map(({team,p,won})=>(
                  <div key={team} style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:`1px solid ${BORDER_SUBTLE}`}}>
                    <span style={{fontSize:11,fontWeight:won?700:400}}>{team}</span>
                    <span style={{fontSize:10,color:"#666"}}>{pct(p,1)}</span>
                  </div>
                ))
              ) : (
                <div style={{fontSize:10,color:"#999"}}>Waiting for projected finals...</div>
              )}
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
            <div style={{height:6,background:"#f0f0f0",borderRadius:14,overflow:"hidden"}}>
              <div style={{width:`${(p/max)*100}%`,height:"100%",background:i===0?"#000":i<4?"#333":"#bbb",borderRadius:14,transition:"width 0.5s"}}/>
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
            <div style={{height:6,background:"#f0f0f0",borderRadius:14,overflow:"hidden"}}>
              <div style={{width:`${(u.upset_prob/max)*100}%`,height:"100%",background:u.upset_prob>40?"#000":u.upset_prob>32?"#555":"#999",borderRadius:14}}/>
            </div>
            <div style={{fontSize:11,fontWeight:600,textAlign:"right"}}>{pct(u.upset_prob,0)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Main ──────────────────────────────────────────────────────────────────────
export default function App() {
  const {
    sim,
    logLines,
    resolved,
    running,
    elapsed,
    teamsCatalog,
    overrides,
    setOverrides,
    forcedPicks,
    setForcedPicks,
    modelInfo,
    startSim,
    toggleForcedPick,
  } = useSimulation();

  const simTeams = Object.keys(sim.predicted_bracket).length>0
    ? [...new Set(Object.values(sim.predicted_bracket).flatMap(rounds=>(rounds[0]||[]).flatMap((g:Game)=>[g.team_a,g.team_b])))].sort()
    : [];
  const catalogTeams = Object.keys(teamsCatalog).sort();
  const allTeams = simTeams.length > 0 ? simTeams : catalogTeams;
  const hasFF = sim.complete&&Object.keys(sim.first_four_pct||{}).length>0;
  const champ = top(sim.champion_pct,1)[0];
  const projectedChampionGame = sim.predicted_bracket["FinalFour"]?.[1]?.[0];
  const projectedChampionName = projectedChampionGame?.winner;
  const leaderName = sim.complete ? (projectedChampionName || champ?.[0] || "Waiting for simulation") : (champ?.[0] || "Waiting for simulation");
  const leaderPctNum = leaderName && sim.champion_pct[leaderName] ? sim.champion_pct[leaderName] : (champ?.[1] ?? 0);
  const leaderPct = leaderPctNum > 0 ? pct(leaderPctNum) : "—";
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

      <div style={{border:`1px solid ${BORDER_OUTER}`,borderRadius:14,padding:12,marginBottom:14,background:"#fff"}}>
        <div className="hero-grid">
          <div style={{padding:"8px 10px",border:`1px solid ${BORDER_INNER}`,borderRadius:14}}>
            <div style={{fontSize:9,letterSpacing:"0.12em",color:"#9ca3af",marginBottom:4,textTransform:"uppercase"}}>{sim.complete?"Projected Bracket Champion":"Current Leader"}</div>
            <div style={{fontSize:26,fontWeight:700,lineHeight:1.05}}>{leaderName}</div>
            <div style={{fontSize:12,color:"#6b7280",marginTop:5}}>{leaderPct}</div>
            {leaderPctNum > 0 && (
              <div style={{height:3,background:"#f0f0f0",borderRadius:14,overflow:"hidden",marginTop:8}}>
                <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${Math.min(100, leaderPctNum)}%`,height:"100%",background:sim.complete?GREEN:ACCENT,transition:"width 0.5s ease",borderRadius:14}}/>
              </div>
            )}
          </div>
          <div style={{padding:"8px 10px",border:`1px solid ${BORDER_INNER}`,borderRadius:14}}>
            <div style={{fontSize:9,letterSpacing:"0.12em",color:"#9ca3af",marginBottom:5,textTransform:"uppercase"}}>Final Four Leaders</div>
            <div style={{display:"grid",gap:6}}>
              {finalFourTop.map(([t,p],i)=>(
                <div key={t}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:11,marginBottom:2}}>
                    <span style={{fontWeight:500}}>{t}</span>
                    <span style={{color:"#6b7280"}}>{pct(p)}</span>
                  </div>
                  <div style={{height:3,background:"#f0f0f0",borderRadius:14,overflow:"hidden"}}>
                    <div className={!sim.complete ? "progress-bar-running" : ""} style={{width:`${(p/ffMax)*100}%`,height:"100%",background:i===0?ACCENT:"#ccc",transition:"width 0.5s ease",borderRadius:14}}/>
                  </div>
                </div>
              ))}
            </div>
          </div>
          <div style={{padding:"8px 10px",border:`1px solid ${BORDER_INNER}`,borderRadius:14,display:"grid",alignContent:"space-between"}}>
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
            <div style={{border:`1px dashed ${BORDER_OUTER}`,borderRadius:14,padding:"12px 14px",fontSize:11,color:"#6b7280",background:"#fff"}}>
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

      <div style={{borderTop:`1px solid ${BORDER_SUBTLE}`,marginTop:48,paddingTop:14,display:"flex",flexDirection:"column",alignItems:"flex-end",gap:4}}>
        <a href="https://github.com/alexh212" target="_blank" rel="noopener noreferrer" style={{fontSize:9,color:"#bbb",textDecoration:"none"}}>github.com/alexh212</a>
        <span style={{fontSize:9,color:"#bbb"}}>LR + XGBoost + LightGBM · isotonic calibration · latent draws · rolling-origin CV · 77 R64 games 2005-2025 · 20/20 benchmarks</span>
        <span style={{fontSize:9,color:"#bbb"}}>Next.js · React · TypeScript · Tailwind CSS</span>
      </div>
    </div>
  );
}
