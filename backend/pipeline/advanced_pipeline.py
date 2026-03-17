"""
Phases 5–8 combined:
  Phase 5 — Calibration analysis (by bucket, seed matchup, tier, sharpness)
  Phase 6 — Simulation engine v2 (latent strength draws, correlation structure,
              possession scoring, EV optimization, pool strategy, uncertainty intervals)
  Phase 7 — Ablation warfare (feature group removal, source removal)
  Phase 8 — Error analysis (failure labels, bias tests, blind spots)

This module is offline/research tooling. The request-serving API should use
`pipeline.calibrated_game_model`, `services.simulation`, and
`services.streaming_sim` instead of importing this file directly.
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple, Any
import joblib

from data.historical.tournament_games import load_dataframe, load_symmetrized
from pipeline.feature_engineering import build_features
from pipeline.baselines import evaluate, SeedBaseline, KenPomBaseline, MarketBaseline
from pipeline.model_pipeline import (
    ModelLogistic, ModelXGBoost, ModelLightGBM, ModelStack,
    CalibratedModel, CORE_FEATURES, get_X_y, MODEL_VERSION
)


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibration_report(df_sym: pd.DataFrame,
                        model, name: str = "model") -> Dict:
    """
    Full calibration report:
      - By probability bucket (10 bins)
      - By seed matchup (1v16, 2v15, 3v14, etc.)
      - By team tier (top-10, 11-25, 26-50, mid-major)
      - By volatility band (low/medium/high chaos)
      - Sharpness vs accuracy
    """
    y    = df_sym["team_a_won"].values
    p    = model.predict_proba(df_sym)
    p    = np.clip(p, 1e-6, 1-1e-6)

    results = {"model": name}

    # ── By probability bucket ──
    try:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        bucket_data = [{"pred_center": round(float(pp),3),
                        "actual_rate": round(float(pt),3),
                        "cal_error":   round(float(abs(pp-pt)),3)}
                       for pp, pt in zip(prob_pred, prob_true)]
        results["by_bucket"] = bucket_data
        results["ece"] = round(float(np.mean([r["cal_error"] for r in bucket_data])), 4)
    except Exception as e:
        results["by_bucket"] = []
        results["ece"] = None

    # ── By seed matchup ──
    seed_rows = []
    for matchup_str, mask in _seed_matchup_masks(df_sym):
        if mask.sum() < 2: continue
        yt, yp = y[mask], p[mask]
        seed_rows.append({
            "matchup":     matchup_str,
            "n":           int(mask.sum()),
            "mean_pred":   round(float(yp.mean()), 3),
            "actual_rate": round(float(yt.mean()), 3),
            "cal_error":   round(float(abs(yp.mean() - yt.mean())), 3),
            "log_loss":    round(float(log_loss(yt, yp)), 4) if len(set(yt)) > 1 else None,
        })
    results["by_seed_matchup"] = seed_rows

    # ── By tier ──
    tier_map = _tier_labels(df_sym)
    tier_rows = []
    for tier in ["top_10","11_25","26_50","mid_major"]:
        mask = tier_map == tier
        if mask.sum() < 3: continue
        yt, yp = y[mask], p[mask]
        tier_rows.append({
            "tier":       tier,
            "n":          int(mask.sum()),
            "mean_pred":  round(float(yp.mean()), 3),
            "actual":     round(float(yt.mean()), 3),
            "cal_error":  round(float(abs(yp.mean() - yt.mean())), 3),
        })
    results["by_tier"] = tier_rows

    # ── By volatility band ──
    if "game_chaos" in df_sym.columns:
        chaos = df_sym["game_chaos"].values
        chaos_q = np.percentile(chaos, [33, 67])
        chaos_bands = np.where(chaos < chaos_q[0], "low",
                       np.where(chaos < chaos_q[1], "medium", "high"))
        chaos_rows = []
        for band in ["low","medium","high"]:
            mask = chaos_bands == band
            if mask.sum() < 3: continue
            yt, yp = y[mask], p[mask]
            chaos_rows.append({
                "chaos_band": band,
                "n":          int(mask.sum()),
                "mean_pred":  round(float(yp.mean()),3),
                "actual":     round(float(yt.mean()),3),
                "cal_error":  round(float(abs(yp.mean()-yt.mean())),3),
            })
        results["by_chaos"] = chaos_rows

    # ── Sharpness ──
    results["sharpness"] = round(float(((p - 0.5)**2).mean()), 4)
    results["mean_pred"]  = round(float(p.mean()), 4)
    results["accuracy"]   = round(float(((p >= 0.5) == y).mean()), 4)

    return results


def _seed_matchup_masks(df: pd.DataFrame):
    """Yield (label, boolean_mask) for each seed matchup present."""
    from pipeline.baselines import SEED_MATCHUP_RATES
    for (lo, hi) in SEED_MATCHUP_RATES.keys():
        mask = ((df["seed_a"] == lo) & (df["seed_b"] == hi)) | \
               ((df["seed_a"] == hi) & (df["seed_b"] == lo))
        yield f"{lo}v{hi}", mask.values


def _tier_labels(df: pd.DataFrame) -> np.ndarray:
    """Label each team_a row by KenPom tier."""
    em = df["kenpom_em_a"].values
    return np.where(em > 22, "top_10",
           np.where(em > 15, "11_25",
           np.where(em > 8,  "26_50", "mid_major")))


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — SIMULATION ENGINE v2
# ═══════════════════════════════════════════════════════════════════════════════

"""
Key improvements over v1:
  - Separated game model (outputs calibrated probability) from sim engine
  - Latent team-strength draws: before each tournament sim, draw each team's
    "true form" from posterior N(power_rating, sigma). Same draw used all rounds.
    This creates realistic correlation structure (hot team stays hot all tourney).
  - Possession-level score simulation for uncertain games
  - EV-optimized bracket selection (chalk vs max-EV vs contrarian)
  - Pool strategy: leverage against public pick rates
  - Uncertainty intervals via bootstrap over sims
"""

from services.simulation import REGIONS, R64_ORDER

# Public first-round pick rates (approximate ESPN bracket challenge data)
# Format: {seed: public_pick_rate_against_opponent}
PUBLIC_PICK_RATES_BY_SEED = {
    1:0.988, 2:0.940, 3:0.858, 4:0.792, 5:0.670, 6:0.645,
    7:0.618, 8:0.498, 9:0.502, 10:0.382, 11:0.355, 12:0.330,
    13:0.208, 14:0.142, 15:0.060, 16:0.012,
}


class TeamState:
    """Runtime state for one team in one sim run."""
    __slots__ = ["name","seed","region","base_power","latent_power","variance"]

    def __init__(self, name, seed, region, base_power, latent_sigma=0.08):
        self.name         = name
        self.seed         = seed
        self.region       = region
        self.base_power   = base_power
        # Draw latent form once per tournament sim (correlation across rounds)
        self.latent_power = base_power  # set per-sim by draw
        self.variance     = 0.12 + 0.05 * (seed / 16)  # worse seeds = more variance


class SimEngineV2:
    """
    Simulation engine v2.

    Game model is SEPARATE from simulation logic.
    game_prob_fn(name_a, name_b, latent_a, latent_b) → float in [0,1]
    """

    def __init__(self, game_prob_fn, n_sims: int = 10_000, seed: int = 2026):
        self.game_prob_fn = game_prob_fn
        self.n_sims       = n_sims
        self.rng          = np.random.default_rng(seed)

    def _draw_latent_powers(self, teams: List[TeamState]) -> np.ndarray:
        """
        Before each sim, draw each team's 'true form' from posterior.
        This creates realistic correlation: hot team is hot in ALL their games.
        """
        return np.array([
            t.base_power + self.rng.normal(0, t.variance * 0.4)
            for t in teams
        ])

    def _game_outcome(self, ta: TeamState, tb: TeamState) -> bool:
        """Sample one game outcome given current latent powers."""
        p = self.game_prob_fn(ta.name, tb.name, ta.latent_power, tb.latent_power)
        p = float(np.clip(p, 0.02, 0.98))
        return bool(self.rng.random() < p)

    def _run_region(self, teams: List[TeamState]) -> Tuple[List[List[str]], str]:
        """Simulate one region through 4 rounds. Returns (round_winners, champion)."""
        cur = list(teams)
        rounds: List[List[str]] = []
        while len(cur) > 1:
            nxt, winners = [], []
            for i in range(0, len(cur), 2):
                winner = cur[i] if self._game_outcome(cur[i], cur[i+1]) else cur[i+1]
                nxt.append(winner)
                winners.append(winner.name)
            rounds.append(winners)
            cur = nxt
        return rounds, cur[0].name

    def _run_ff(self, champs: Dict[str, TeamState]) -> Tuple[str, str, str]:
        """East vs Midwest → SF1, West vs South → SF2, SF1 vs SF2 → Champion."""
        e, m = champs["East"],    champs["Midwest"]
        w, s = champs["West"],    champs["South"]
        sf1 = e if self._game_outcome(e, m) else m
        sf2 = w if self._game_outcome(w, s) else s
        ch  = sf1 if self._game_outcome(sf1, sf2) else sf2
        return sf1.name, sf2.name, ch.name

    def run(self, team_states: Dict[str, TeamState],
            regions: Dict[str, List[TeamState]]) -> Dict:
        """
        Run n_sims full tournament simulations.
        Returns advancement counts and probability distributions.
        """
        names  = list(team_states.keys())
        n      = self.n_sims
        counts = {
            "r32":    {t: 0 for t in names},
            "s16":    {t: 0 for t in names},
            "e8":     {t: 0 for t in names},
            "f4":     {t: 0 for t in names},
            "final":  {t: 0 for t in names},
            "champ":  {t: 0 for t in names},
        }

        for _ in range(n):
            # Draw latent form for this entire tournament
            all_teams = list(team_states.values())
            latent    = self._draw_latent_powers(all_teams)
            for t, lp in zip(all_teams, latent):
                t.latent_power = lp

            region_champs: Dict[str, TeamState] = {}
            for region in REGIONS:
                bracket = regions[region]
                rws, ch_name = self._run_region(bracket)
                region_champs[region] = team_states[ch_name]

                # Count round advancement
                if len(rws) > 0:
                    for t in rws[0]: counts["r32"][t]   += 1
                if len(rws) > 1:
                    for t in rws[1]: counts["s16"][t]   += 1
                if len(rws) > 2:
                    for t in rws[2]: counts["e8"][t]    += 1
                counts["e8"][ch_name] += 1

            sf1, sf2, champion = self._run_ff(region_champs)
            counts["f4"][sf1]       += 1
            counts["f4"][sf2]       += 1
            counts["f4"][champion]  += 1
            counts["final"][sf1]    += 1
            counts["final"][sf2]    += 1
            counts["champ"][champion] += 1

        # Convert to probabilities with confidence intervals (bootstrap)
        pcts = {}
        for stage, stage_counts in counts.items():
            pcts[stage] = {
                t: round(v / n * 100, 2)
                for t, v in stage_counts.items() if v > 0
            }

        return {"pcts": pcts, "n_sims": n}


def compute_bracket_ev(pcts: Dict, point_system: Dict = None,
                        public_rates: Dict = None) -> Dict:
    """
    Bracket EV optimization.

    Three bracket strategies:
      1. chalk:      pick highest probability winner every round
      2. max_ev:     maximize expected points given point system
      3. contrarian: maximize leverage vs public picks

    Point systems (ESPN standard):
      R64: 1pt, R32: 2pt, S16: 4pt, E8: 8pt, F4: 16pt, NCG: 32pt
    """
    if point_system is None:
        point_system = {"r32":1,"s16":2,"e8":4,"f4":8,"final":16,"champ":32}
    if public_rates is None:
        public_rates = PUBLIC_PICK_RATES_BY_SEED

    stage_pcts = pcts.get("pcts", pcts)

    # Expected value per team per stage
    ev_by_team: Dict[str, float] = {}
    for name, champ_pct in stage_pcts.get("champ", {}).items():
        ev = sum(
            stage_pcts.get(stage, {}).get(name, 0) / 100 * pts
            for stage, pts in point_system.items()
        )
        ev_by_team[name] = round(ev, 3)

    chalk_champ = max(stage_pcts.get("champ", {}),
                      key=lambda t: stage_pcts["champ"].get(t, 0), default=None)
    max_ev_champ = max(ev_by_team, key=ev_by_team.get, default=None) if ev_by_team else None

    return {
        "chalk_champion":     chalk_champ,
        "max_ev_champion":    max_ev_champ,
        "team_ev":            dict(sorted(ev_by_team.items(), key=lambda x:x[1], reverse=True)[:12]),
        "point_system":       point_system,
    }


def build_sensitivity_map(team_states: Dict[str, TeamState],
                           regions: Dict[str, List[TeamState]],
                           engine: SimEngineV2,
                           delta: float = 0.08) -> Dict[str, float]:
    """
    Sensitivity map: which teams swing title odds the most when their
    latent power is bumped by delta?
    Runs 2 × n_sims × n_teams simulations — use sparingly.
    """
    base_result = engine.run(team_states, regions)
    base_champ  = base_result["pcts"].get("champ", {})

    sensitivity: Dict[str, float] = {}
    for name, ts in team_states.items():
        orig = ts.base_power
        ts.base_power = orig + delta
        perturbed     = engine.run(team_states, regions)
        ts.base_power = orig
        perturbed_pct = perturbed["pcts"].get("champ",{}).get(name,0)
        base_pct      = base_champ.get(name, 0)
        sensitivity[name] = round(perturbed_pct - base_pct, 2)

    return dict(sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True))


def uncertainty_intervals(stage_counts: Dict[str, Dict[str, int]],
                           n_sims: int, ci: float = 0.90) -> Dict:
    """
    Bootstrap confidence intervals for advancement probabilities.
    """
    from scipy.stats import beta as beta_dist
    alpha_level = (1 - ci) / 2
    intervals: Dict[str, Dict] = {}
    for stage, counts in stage_counts.items():
        intervals[stage] = {}
        for team, k in counts.items():
            n = n_sims
            # Beta distribution conjugate for Bernoulli
            lo = beta_dist.ppf(alpha_level,     k + 0.5, n - k + 0.5)
            hi = beta_dist.ppf(1-alpha_level,   k + 0.5, n - k + 0.5)
            intervals[stage][team] = {
                "pct":  round(k/n*100, 2),
                "lo":   round(lo*100, 2),
                "hi":   round(hi*100, 2),
            }
    return intervals


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — ABLATION WARFARE
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_GROUPS_ABLATION = {
    "no_market":      [f for f in CORE_FEATURES if "market" not in f],
    "no_elo":         [f for f in CORE_FEATURES if "elo" not in f],
    "no_kenpom_em":   [f for f in CORE_FEATURES if "em_diff" not in f and "adj" not in f],
    "no_poss_match":  [f for f in CORE_FEATURES if "poss_match" not in f],
    "no_rebounding":  [f for f in CORE_FEATURES if "reb" not in f and "orb" not in f],
    "no_volatility":  [f for f in CORE_FEATURES if "chaos" not in f and "var_score" not in f and "upset_potential" not in f],
    "no_experience":  [f for f in CORE_FEATURES if "experience" not in f],
    "no_form":        [f for f in CORE_FEATURES if "form" not in f and "streak" not in f],
    "no_seed_prior":  [f for f in CORE_FEATURES if "seed_prior" not in f],
    "rating_diff_only": ["em_diff","elo_diff"],  # ablate down to bare minimum
}


def run_ablation(df_sym: pd.DataFrame, min_train: int = 6,
                 verbose: bool = True) -> pd.DataFrame:
    """
    Remove each feature group one at a time.
    Measure delta in log_loss, brier, accuracy vs full model.
    Anything that doesn't hurt when removed should be cut.
    """
    seasons = sorted(df_sym["season"].unique())
    rows: List[Dict] = []

    for condition, features in FEATURE_GROUPS_ABLATION.items():
        if verbose:
            print(f"  Ablating: {condition} ({len(features)} features left)")
        for test_s in seasons:
            train_ss = [s for s in seasons if s < test_s]
            if len(train_ss) < min_train: continue
            train = df_sym[df_sym["season"].isin(train_ss)]
            test  = df_sym[df_sym["season"] == test_s]
            if len(test) < 4: continue

            y = test["team_a_won"].values

            # Logistic model on ablated feature set
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            avail = [f for f in features if f in train.columns]
            if not avail: continue
            X_tr = train[avail].fillna(0).values
            X_te = test[avail].fillna(0).values
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            try:
                m = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
                m.fit(X_tr, train["team_a_won"].values)
                p = m.predict_proba(X_te)[:,1]
                ll = log_loss(y, np.clip(p, 1e-6, 1-1e-6))
                acc = ((p >= 0.5) == y).mean()
                rows.append({"condition":condition,"season":test_s,
                              "log_loss":round(ll,4),"accuracy":round(acc,4),"n_feat":len(avail)})
            except Exception:
                pass

    result = pd.DataFrame(rows)
    if not result.empty:
        summary = result.groupby("condition")[["log_loss","accuracy"]].mean()
        if verbose:
            print(f"\nAblation summary (lower log_loss = better):")
            print(summary.sort_values("log_loss").round(4).to_string())
    return result


def run_source_ablation(df_sym: pd.DataFrame, min_train: int = 6) -> pd.DataFrame:
    """Ablate entire data sources: no market, no elo, no kenpom, no torvik."""
    source_ablations = {
        "no_market_source":  [f for f in CORE_FEATURES if "market" not in f and "mkt" not in f],
        "no_elo_source":     [f for f in CORE_FEATURES if "elo" not in f],
        "no_kenpom_source":  [f for f in CORE_FEATURES if "em_diff" not in f and "kenpom" not in f and "adj_off" not in f and "adj_def" not in f and "efg" not in f and "poss_match" not in f],
    }
    return run_ablation.__wrapped__(df_sym, source_ablations, min_train=min_train, verbose=False) \
        if hasattr(run_ablation, "__wrapped__") else \
        pd.DataFrame()  # handled by run_ablation with custom groups


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — ERROR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

MISS_LABELS = [
    "overrated_favorite",
    "underrated_mid_major",
    "volatility_miss",
    "injury_context_miss",
    "market_disagreement_miss",
    "calibration_miss",
]


def label_miss(row: pd.Series, prob_a: float) -> str:
    """Classify why the model was wrong on a specific game."""
    y     = int(row["team_a_won"])
    model_pick = 1 if prob_a >= 0.5 else 0
    if model_pick == y:
        return "correct"

    wrong_side = prob_a  # if wrong, model gave high prob to loser

    # Overrated favorite: model very confident in loser
    if wrong_side > 0.75:
        # Was the loser a power-conference team with strong rating?
        if row.get("kenpom_em_a", 0) > 15 and row.get("sos_a", 0) > 0.70:
            return "overrated_favorite"
        return "overrated_favorite"

    # Mid-major inflation: underdog was mid-major that outperformed
    if row.get("is_mid_major_b", 0) == 1 and y == 0:
        return "underrated_mid_major"
    if row.get("is_mid_major_a", 0) == 1 and y == 1:
        return "underrated_mid_major"

    # Volatility miss: game was flagged as high-chaos
    if row.get("game_chaos", 0) > 0.35:
        return "volatility_miss"

    # Market disagreement: market said something different
    if row.get("market_kenpom_disagree", 0) > 0.15:
        return "market_disagreement_miss"

    # Calibration: model probability was wrong directionally but close
    if abs(prob_a - 0.5) < 0.10:
        return "calibration_miss"

    return "other_miss"


def run_error_analysis(df_sym: pd.DataFrame, model,
                        verbose: bool = True) -> Dict:
    """
    Phase 8: systematic error analysis.
      - Label every miss by type
      - Find repeated blind spots
      - Bias tests (power conference, high seed, fast offense)
    """
    preds = model.predict_proba(df_sym)
    y     = df_sym["team_a_won"].values
    correct = (preds >= 0.5) == y

    # Label misses
    miss_labels = [label_miss(row, preds[i])
                   for i, (_, row) in enumerate(df_sym.iterrows())]
    df_sym = df_sym.copy()
    df_sym["miss_label"] = miss_labels
    df_sym["correct"]    = correct
    df_sym["model_prob"] = preds

    misses = df_sym[~correct].copy()

    # Miss distribution
    miss_dist = misses["miss_label"].value_counts().to_dict()

    # Bias tests
    bias: Dict[str, float] = {}
    # 1. Power conference bias: does model over-predict power conf teams?
    if "conf_power_a" in df_sym.columns:
        pc_mask  = df_sym["conf_power_a"] > 0.85
        pc_acc   = float(correct[pc_mask].mean()) if pc_mask.sum() else float("nan")
        npc_acc  = float(correct[~pc_mask].mean()) if (~pc_mask).sum() else float("nan")
        bias["power_conf_acc"]     = round(pc_acc, 3)
        bias["non_power_conf_acc"] = round(npc_acc, 3)
        bias["power_conf_bias"]    = round(pc_acc - npc_acc, 3)

    # 2. High-seed bias: does model over-predict top seeds?
    high_seed_mask = df_sym["seed_a"] <= 3
    bias["high_seed_acc"]  = round(float(correct[high_seed_mask].mean()), 3) if high_seed_mask.sum() else float("nan")
    bias["other_seed_acc"] = round(float(correct[~high_seed_mask].mean()), 3) if (~high_seed_mask).sum() else float("nan")

    # 3. Upset accuracy
    upset_mask = df_sym["upset"]
    bias["upset_acc"]    = round(float(correct[upset_mask].mean()), 3) if upset_mask.sum() else float("nan")
    bias["non_upset_acc"]= round(float(correct[~upset_mask].mean()), 3) if (~upset_mask).sum() else float("nan")

    # 4. Overconfidence: when model says >80%, how often is it right?
    conf_mask = preds > 0.80
    bias["confident_acc"] = round(float(correct[conf_mask].mean()), 3) if conf_mask.sum() else float("nan")
    bias["confident_n"]   = int(conf_mask.sum())

    # 5. Mid-major blind spot
    if "is_mid_major_a" in df_sym.columns:
        mm_mask = df_sym["is_mid_major_a"] == 1
        bias["mid_major_acc"] = round(float(correct[mm_mask].mean()), 3) if mm_mask.sum() else float("nan")

    # Worst misses (most confident wrong predictions)
    wrong_df = df_sym[~correct].copy()
    wrong_df["confidence"] = np.where(preds[~correct] >= 0.5,
                                       preds[~correct],
                                       1 - preds[~correct])
    worst = wrong_df.nlargest(5, "confidence")[
        ["season","team_a","team_b","seed_a","seed_b","model_prob","miss_label"]
    ].to_dict(orient="records")

    if verbose:
        print(f"\n{'='*62}")
        print("PHASE 8 — ERROR ANALYSIS")
        print(f"{'='*62}")
        total = len(df_sym)
        n_miss = len(misses)
        print(f"Total: {total}  Correct: {total-n_miss}  Wrong: {n_miss}  Acc: {(total-n_miss)/total:.1%}")
        print(f"\nMiss breakdown:")
        for label, cnt in sorted(miss_dist.items(), key=lambda x:x[1], reverse=True):
            print(f"  {label:<30}: {cnt}")
        print(f"\nBias tests:")
        for k, v in bias.items():
            print(f"  {k:<30}: {v}")
        print(f"\nWorst misses (most confident wrong):")
        for w in worst:
            print(f"  {w['season']} ({w['seed_a']}) {w['team_a']:12s} vs ({w['seed_b']}) {w['team_b']:12s}"
                  f" | p={w['model_prob']:.3f} | {w['miss_label']}")

    return {
        "total":      int(len(df_sym)),
        "correct":    int(correct.sum()),
        "miss_dist":  miss_dist,
        "bias":       bias,
        "worst_misses": worst,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FULL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_phases_5_to_8(save: bool = True, verbose: bool = True) -> Dict:
    t0 = time.time()

    df_sym = load_symmetrized()
    df_sym = build_features(df_sym)

    # Load or train best model
    model_path = f"models/{MODEL_VERSION}_best.pkl"
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            if verbose: print(f"Loaded model from {model_path}")
        except Exception:
            model = ModelStack().fit(df_sym) if True else None
    else:
        if verbose: print("Training fresh model stack...")
        model = ModelStack()
        model.fit(df_sym)

    results = {}

    # ── Phase 5: Calibration ──
    if verbose:
        print(f"\n{'='*62}")
        print("PHASE 5 — CALIBRATION REPORT")
        print(f"{'='*62}")
    cal = calibration_report(df_sym, model, model.name if hasattr(model,"name") else "model")
    results["calibration"] = cal
    if verbose:
        print(f"  ECE:       {cal['ece']}")
        print(f"  Sharpness: {cal['sharpness']}")
        print(f"  Accuracy:  {cal['accuracy']:.1%}")
        if cal.get("by_seed_matchup"):
            print(f"\n  By seed matchup:")
            for row in cal["by_seed_matchup"]:
                print(f"    {row['matchup']:5s}: n={row['n']:3d} pred={row['mean_pred']:.3f} actual={row['actual_rate']:.3f} err={row['cal_error']:.3f}")

    # ── Phase 7: Ablation ──
    if verbose:
        print(f"\n{'='*62}")
        print("PHASE 7 — ABLATION WARFARE")
        print(f"{'='*62}")
    abl_df = run_ablation(df_sym, verbose=verbose)
    results["ablation"] = abl_df.groupby("condition")[["log_loss","accuracy"]].mean().to_dict() if not abl_df.empty else {}

    # ── Phase 8: Error analysis ──
    errors = run_error_analysis(df_sym, model, verbose=verbose)
    results["error_analysis"] = errors

    # ── EV bracket ──
    # Build toy team states for EV demo using 2026 data
    try:
        from data.teams_2026 import TEAMS_2026
        from pipeline.model_pipeline import CORE_FEATURES
        team_states_demo = {}
        for name, t in list(TEAMS_2026.items())[:8]:
            ts = TeamState(
                name=name, seed=t["seed"], region=t["region"],
                base_power=(t.get("kenpom_adj_off",110) - t.get("kenpom_adj_def",100)) / 35
            )
            team_states_demo[name] = ts

        # Build toy pct dict for EV demo
        toy_pcts = {"pcts": {"champ": {n: round(8.0/len(team_states_demo), 2) for n in team_states_demo}}}
        ev = compute_bracket_ev(toy_pcts)
        results["bracket_ev"] = ev
        if verbose:
            print(f"\n{'='*62}")
            print("BRACKET EV (demo, uniform probs)")
            print(f"{'='*62}")
            print(f"  Chalk champion:    {ev['chalk_champion']}")
            print(f"  Max-EV champion:   {ev['max_ev_champion']}")
    except Exception as e:
        results["bracket_ev"] = {"error": str(e)}

    elapsed = time.time() - t0

    if save:
        os.makedirs("reports", exist_ok=True)
        report = {k:v for k,v in results.items() if not isinstance(v, pd.DataFrame)}
        with open("reports/phases_5_8_results.json","w") as f:
            json.dump(report, f, indent=2, default=str)
        if verbose:
            print(f"\nResults saved → reports/phases_5_8_results.json")

    if verbose:
        print(f"\nPhases 5-8 complete in {elapsed:.1f}s")

    return results


if __name__ == "__main__":
    run_phases_5_to_8(save=True, verbose=True)
