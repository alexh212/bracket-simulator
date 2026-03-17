"use client";

export interface Game {
  team_a: string;
  team_b: string;
  seed_a: number;
  seed_b: number;
  win_prob_a: number;
  winner: string;
  upset: boolean;
  expected_margin: number;
  score_a?: number;
  score_b?: number;
  score_note?: string;
}

export interface SimState {
  champion_pct: Record<string, number>;
  title_game_pct: Record<string, number>;
  final_four_pct: Record<string, number>;
  elite_eight_pct: Record<string, number>;
  sweet_sixteen_pct: Record<string, number>;
  round_of_32_pct: Record<string, number>;
  first_four_pct: Record<string, number>;
  predicted_bracket: Record<string, Game[][]>;
  upset_watch: any[];
  matchup_probs: Record<string, number>;
  n_sims: number;
  done: number;
  total: number;
  model_used: string;
  elapsed_sec?: number;
  complete: boolean;
  bracketStarted: boolean;
  sims_per_sec?: number;
  first_four_games?: any[];
  teams?: Record<string, any>;
  seed_history?: Record<number, number>;
  region_stats?: any[];
}

export interface ModelInfo {
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

export type Resolved = Record<string, Record<number, Record<number, boolean>>>;
export type TeamOverrides = Record<string, number>;
export type ForcedPicks = Record<string, string>;

export type SimRunConfig = {
  n_sims: number;
  latent_sigma: number;
  team_overrides: TeamOverrides;
  forced_picks: ForcedPicks;
};

export const EMPTY_SIM_STATE: SimState = {
  champion_pct: {},
  title_game_pct: {},
  final_four_pct: {},
  elite_eight_pct: {},
  sweet_sixteen_pct: {},
  round_of_32_pct: {},
  first_four_pct: {},
  predicted_bracket: {},
  upset_watch: [],
  matchup_probs: {},
  n_sims: 0,
  done: 0,
  total: 10000,
  model_used: "",
  complete: false,
  bracketStarted: false,
};
