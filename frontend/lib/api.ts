"use client";

import type { ModelInfo, SimRunConfig } from "@/components/app/types";

export const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8001";

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

export function getWhatIf(teamA: string, teamB: string) {
  return fetchJson<{ results: any[] }>("/whatif", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      team_a: teamA,
      team_b: teamB,
      scenarios: [
        { label: "Baseline", target: "a", injury_factor: 1.0 },
        { label: `${teamA} minor inj`, target: "a", injury_factor: 0.93 },
        { label: `${teamA} major inj`, target: "a", injury_factor: 0.82 },
        { label: `${teamB} injured`, target: "b", injury_factor: 0.9 },
        { label: "Hot streak", target: "a", form_score: 8.0 },
        { label: "Slow tempo", target: "a", possessions_per_game: 62.0 },
      ],
    }),
  });
}
