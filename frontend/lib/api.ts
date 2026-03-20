"use client";

import type { ModelInfo, SimRunConfig } from "@/components/app/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

/** How often the UI refetches `/results` (live + final scores). */
export const RESULTS_POLL_MS = 60_000;

async function fetchJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

export function createSimulationStreamUrl(cfg: SimRunConfig) {
  const encodedOverrides = encodeURIComponent(JSON.stringify(cfg.team_overrides || {}));
  const encodedForced = encodeURIComponent(JSON.stringify(cfg.forced_picks || {}));
  return `${API_BASE}/simulate/stream?n_sims=${cfg.n_sims}&emit_every=${Math.max(100, Math.floor(cfg.n_sims / 40))}&latent_sigma=${cfg.latent_sigma}&team_overrides=${encodedOverrides}&forced_picks=${encodedForced}`;
}

export interface RealGame {
  region: string;
  round: number;
  game_index: number;
  team_a: string;
  team_b: string;
  seed_a: number;
  seed_b: number;
  score_a: number;
  score_b: number;
  winner: string;
  status: "final" | "live" | "upcoming";
  /** Clock / period when status is live (from scoreboard API) */
  status_detail?: string;
}

export interface RealResults {
  last_updated: string | null;
  tournament_status: string;
  games: RealGame[];
  live_scores_enabled?: boolean;
  live_scores_matched?: number;
  live_scores_error?: boolean;
  live_scores_source?: string;
}

export interface PerfectBracket {
  perfect_remaining: number | null;
  brackets_fallen: number | null;
}

export const getResults = () => fetchJson<RealResults>("/results");
export const getPerfectBracket = () => fetchJson<PerfectBracket>("/perfect-bracket");
export const getTeams = () => fetchJson<Record<string, any>>("/teams");
export const getModelInfo = () => fetchJson<ModelInfo>("/model-info");
export const getModelLog = () => fetchJson<{ log: string }>("/model-log");
export const getFirstFour = () => fetchJson<{ games: any[] }>("/first-four");
export const getRegionStats = () => fetchJson<any[]>("/region-stats");
export const getSeedHistory = () => fetchJson<Record<string, number>>("/seed-history");

export function getMatchup(teamA: string, teamB: string) {
  return fetchJson<any>("/matchup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ team_a: teamA, team_b: teamB }),
  });
}

export function getDisagreement(teamA: string, teamB: string) {
  return fetchJson<any>(`/disagreement/${encodeURIComponent(teamA)}/${encodeURIComponent(teamB)}`);
}

export function getHistoricalComps(teamA: string, teamB: string) {
  return fetchJson<any>(`/comps/${encodeURIComponent(teamA)}/${encodeURIComponent(teamB)}`);
}

export interface WhatIfScenario {
  label: string;
  desc: string;
  params: Record<string, any>;
}

export function buildWhatIfScenarios(teamA: string, teamB: string): WhatIfScenario[] {
  return [
    {
      label: "Baseline",
      desc: "Current ratings, no adjustments",
      params: { target: "a", injury_factor: 1.0 },
    },
    {
      label: `${teamA} role player out`,
      desc: "Rotation player injured — slightly reduced offense & power rating",
      params: { target: "a", injury_factor: 0.93, delta_kenpom_adj_off: -1.0 },
    },
    {
      label: `${teamA} star player out`,
      desc: "Best player unavailable — major drop in efficiency, shooting, and power",
      params: { target: "a", injury_factor: 0.80, delta_kenpom_adj_off: -3.5, delta_efg_pct: -0.02 },
    },
    {
      label: `${teamB} key player out`,
      desc: "Opponent loses a starter — their offense and power rating take a hit",
      params: { target: "b", injury_factor: 0.88, delta_kenpom_adj_off: -2.0 },
    },
    {
      label: `${teamA} hot streak`,
      desc: "Playing well above season average — boosted offense, Elo, and efficiency",
      params: { target: "a", delta_kenpom_adj_off: 3.0, delta_elo_current: 60, delta_efg_pct: 0.015 },
    },
    {
      label: "Grind-it-out pace",
      desc: "Game played at 62 possessions — favors disciplined defenses, hurts fast teams",
      params: { target: "a", possessions_per_game: 62.0 },
    },
    {
      label: `${teamA} cold shooting`,
      desc: "Off night from three — reduced eFG% and three-point rate",
      params: { target: "a", delta_efg_pct: -0.03, delta_three_pt_rate: -0.04 },
    },
  ];
}

export function getWhatIf(teamA: string, teamB: string) {
  const scenarios = buildWhatIfScenarios(teamA, teamB);
  return fetchJson<{ results: any[] }>("/whatif", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      team_a: teamA,
      team_b: teamB,
      scenarios: scenarios.map((s) => ({ label: s.label, ...s.params })),
    }),
  }).then((data) => ({
    results: (data.results || []).map((r: any, i: number) => ({
      ...r,
      desc: scenarios[i]?.desc || "",
    })),
  }));
}
