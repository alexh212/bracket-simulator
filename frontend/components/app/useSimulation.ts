"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { createSimulationStreamUrl, getModelInfo, getTeams } from "@/lib/api";
import type { ForcedPicks, ModelInfo, Resolved, SimRunConfig, SimState, TeamOverrides } from "@/components/app/types";
import { EMPTY_SIM_STATE } from "@/components/app/types";

export type SimPhase = "idle" | "simulating" | "building_bracket" | "done";

export function useSimulation() {
  const [sim, setSim] = useState<SimState>(EMPTY_SIM_STATE);
  const [logLines, setLog] = useState<string[]>([]);
  const [resolved, setResolved] = useState<Resolved>({});
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<SimPhase>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [teamsCatalog, setTeamsCatalog] = useState<Record<string, any>>({});
  const [overrides, setOverrides] = useState<TeamOverrides>({});
  const [forcedPicks, setForcedPicks] = useState<ForcedPicks>({});
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const esRef = useRef<EventSource | null>(null);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    getTeams().then(setTeamsCatalog).catch(() => {});
    getModelInfo().then(setModelInfo).catch(() => {});
  }, []);

  const startSim = useCallback((cfg: SimRunConfig) => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    setSim({ ...EMPTY_SIM_STATE, total: cfg.n_sims });
    setResolved({});
    setLog(["connecting..."]);
    setRunning(true);
    setPhase("simulating");
    startTimeRef.current = Date.now();

    const es = new EventSource(createSimulationStreamUrl(cfg));
    esRef.current = es;

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);

      if (data.type === "start") {
        setLog((prev) => [
          ...prev,
          "simulation started",
          `${data.n_sims.toLocaleString()} runs · sigma=${cfg.latent_sigma} · random seed`,
          `${data.assumptions_applied || 0} team assumptions applied`,
          `${data.forced_picks_applied || 0} game picks locked`,
          ...Object.entries(cfg.team_overrides || {})
            .slice(0, 4)
            .map(([name, value]) => `assumption: ${name} ${value > 0 ? "+" : ""}${value} elo`),
          ...Object.entries(cfg.forced_picks || {})
            .slice(0, 4)
            .map(([key, winner]) => `pick: ${key} -> ${winner}`),
          "model: calibrated_ml_stack",
        ]);
      }

      if (data.type === "progress") {
        const elapsedSec = (Date.now() - startTimeRef.current) / 1000;
        const simsPerSec = elapsedSec > 0 ? data.done / elapsedSec : 0;
        setSim((prev) => ({
          ...prev,
          champion_pct: data.champion_pct,
          final_four_pct: data.final_four_pct,
          done: data.done,
          total: data.total,
          sims_per_sec: simsPerSec,
        }));
        if (data.done >= data.total) {
          setElapsed(elapsedSec);
          setPhase("building_bracket");
        }
        const leader = Object.entries(data.champion_pct as Record<string, number>).sort((a, b) => b[1] - a[1])[0];
        if (leader) {
          setLog((prev) => [
            ...prev.slice(-80),
            `[${data.done.toLocaleString()}/${data.total.toLocaleString()}] ${leader[0]} leading at ${leader[1].toFixed(1)}%  (${simsPerSec.toFixed(0)} sims/sec)`,
          ]);
        }
      }

      if (data.type === "game") {
        setSim((prev) => {
          const predictedBracket = { ...prev.predicted_bracket };
          if (!predictedBracket[data.region]) {
            predictedBracket[data.region] = [];
          }
          const rounds = [...(predictedBracket[data.region] || [])];
          if (!rounds[data.round]) {
            rounds[data.round] = [];
          }
          const round = [...rounds[data.round]];
          round[data.gi] = data.game;
          rounds[data.round] = round;
          predictedBracket[data.region] = rounds;
          return { ...prev, predicted_bracket: predictedBracket, bracketStarted: true };
        });
        setResolved((prev) => ({
          ...prev,
          [data.region]: {
            ...(prev[data.region] || {}),
            [data.round]: { ...((prev[data.region] || {})[data.round] || {}), [data.gi]: true },
          },
        }));
      }

      if (data.type === "complete") {
        setSim((prev) => ({
          ...prev,
          champion_pct: data.champion_pct,
          title_game_pct: data.title_game_pct || {},
          final_four_pct: data.final_four_pct,
          elite_eight_pct: data.elite_eight_pct,
          sweet_sixteen_pct: data.sweet_sixteen_pct,
          round_of_32_pct: data.round_of_32_pct,
          first_four_pct: data.first_four_pct || {},
          predicted_bracket: data.predicted_bracket,
          upset_watch: data.upset_watch,
          matchup_probs: data.matchup_probs || {},
          n_sims: data.n_sims,
          done: data.n_sims,
          total: data.n_sims,
          model_used: data.model_used,
          elapsed_sec: data.elapsed_sec,
          complete: true,
          bracketStarted: true,
        }));
        const champ = Object.entries(data.champion_pct as Record<string, number>).sort((a, b) => b[1] - a[1])[0];
        setLog((prev) => [...prev, "", `complete ${data.elapsed_sec}s`, `champion: ${champ?.[0]} ${champ?.[1].toFixed(1)}%`]);
        setRunning(false);
        setPhase("done");
        es.close();
      }

      if (data.type === "error") {
        setLog((prev) => [...prev, `error: ${data.message}`]);
        setRunning(false);
        setPhase("idle");
        es.close();
      }
    };

    es.onerror = () => {
      setLog((prev) => [...prev, "connection error"]);
      setRunning(false);
      setPhase("idle");
      es.close();
    };
  }, []);

  useEffect(() => () => esRef.current?.close(), []);

  useEffect(() => {
    if (phase !== "simulating") {
      return;
    }
    const id = setInterval(() => {
      setElapsed((Date.now() - startTimeRef.current) / 1000);
    }, 100);
    return () => clearInterval(id);
  }, [phase]);

  const toggleForcedPick = useCallback((key: string, winner: string) => {
    setForcedPicks((prev) =>
      prev[key] === winner
        ? Object.fromEntries(Object.entries(prev).filter(([existingKey]) => existingKey !== key))
        : { ...prev, [key]: winner }
    );
  }, []);

  return {
    sim,
    logLines,
    resolved,
    running,
    phase,
    elapsed,
    teamsCatalog,
    overrides,
    setOverrides,
    forcedPicks,
    setForcedPicks,
    modelInfo,
    startSim,
    toggleForcedPick,
  };
}
