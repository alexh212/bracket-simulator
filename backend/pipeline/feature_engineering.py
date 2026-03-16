"""
Phase 3 — Feature engineering pipeline.

Feature classes (per the spec):
  A. Team strength      — ratings, efficiency, Elo, market
  B. Matchup-specific   — differential features, interaction terms
  C. Volatility         — 3pt dependence, tov variance, scoring SD, tempo
  D. Context            — SOS, conference, travel, experience
  E. Calibration-only   — seed prior, market disagreement, uncertainty flags
  F. Style clustering   — archetype labels (bomb-heavy, rim-bully, slow-elite-D, chaos)

All features frozen at Selection Sunday.
Leakage check: no feature uses game-day info except market (flagged).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from typing import Dict, List, Tuple

from data.historical.tournament_games import load_symmetrized
from pipeline.baselines import SeedBaseline, KenPomBaseline, EloBaseline

FEATURE_VERSION = "features_v1_selection_sunday"

# ─── Feature class registry ──────────────────────────────────────────────────

FEATURE_CLASSES = {
    "strength":    "Team power ratings and composite score",
    "matchup":     "Differential and interaction features specific to this game",
    "volatility":  "Game-to-game consistency and chaos metrics",
    "context":     "Schedule strength, conference, experience, travel",
    "calibration": "Seed prior, market disagreement, uncertainty flags",
    "style":       "Team archetype clustering labels",
}

FEATURE_REGISTRY: Dict[str, Dict] = {}

def register(name, cls, source, freeze, leakage, sign, confidence, transform="raw"):
    FEATURE_REGISTRY[name] = dict(
        cls=cls, source=source, freeze=freeze, leakage=leakage,
        sign=sign, confidence=confidence, transform=transform
    )


# ─── A. Strength features ────────────────────────────────────────────────────

def add_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite power ratings from multiple sources.
    Key insight: normalise each source to same scale before blending,
    then check if blend beats individuals (ablation in Phase 7).
    """
    # Normalise KenPom EM to [0,1] (range -35 to +38 in practice)
    df["kp_norm_a"] = (df["kenpom_em_a"] + 35) / 73
    df["kp_norm_b"] = (df["kenpom_em_b"] + 35) / 73

    # Normalise Elo to [0,1] (range 1400-2200)
    df["elo_norm_a"] = (df["elo_a"] - 1400) / 800
    df["elo_norm_b"] = (df["elo_b"] - 1400) / 800

    # Market prob is already [0,1]
    # Seed strength proxy (16-seed) / 15 so lower seed = higher value
    df["seed_strength_a"] = (17 - df["seed_a"]) / 15
    df["seed_strength_b"] = (17 - df["seed_b"]) / 15

    # SOS credibility: soft discount for weak schedules
    df["sos_cred_a"] = np.clip(0.68 + 0.32 * (df["sos_a"] / 0.85), 0, 1)
    df["sos_cred_b"] = np.clip(0.68 + 0.32 * (df["sos_b"] / 0.85), 0, 1)

    # Luck-adjusted KenPom (subtract luck component)
    # luck field not in our historical data — skip, mark as missing
    df["luck_adj_em_a"] = df["kenpom_em_a"]  # v1: no luck data
    df["luck_adj_em_b"] = df["kenpom_em_b"]

    # Pythagorean win expectancy (PPP^11.5 / (PPP^11.5 + defPPP^11.5))
    # adj_off/adj_def are per-100 possession, convert to PPP space
    ppp_a = df["adj_off_a"] / 100
    ppp_b = df["adj_off_b"] / 100
    dppp_a = df["adj_def_a"] / 100
    dppp_b = df["adj_def_b"] / 100
    EXP = 11.5
    df["pyth_wp_a"] = (ppp_a**EXP) / (ppp_a**EXP + dppp_a**EXP)
    df["pyth_wp_b"] = (ppp_b**EXP) / (ppp_b**EXP + dppp_b**EXP)

    # Composite power rating (weights from v1; will be tuned in Phase 4)
    df["power_a"] = (0.32 * df["kp_norm_a"]
                   + 0.20 * df["elo_norm_a"]
                   + 0.12 * df["market_prob_a"]
                   + 0.08 * df["seed_strength_a"]
                   + 0.08 * (df["pyth_wp_a"] - 0.5) * 2   # centre around 0
                   )
    df["power_b"] = (0.32 * df["kp_norm_b"]
                   + 0.20 * df["elo_norm_b"]
                   + 0.12 * df["market_prob_b"]
                   + 0.08 * df["seed_strength_b"]
                   + 0.08 * (df["pyth_wp_b"] - 0.5) * 2
                   )

    register("power_a","strength","composite","selection_sunday","none","+","high")
    register("pyth_wp_a","strength","KenPom PPP","selection_sunday","none","+","high","derived")
    return df


# ─── B. Matchup features (differentials + interactions) ──────────────────────

def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Differentials are the primary signal — model cares more about A-B than raw values.
    Also engineer interaction terms.
    """
    # Power differential (primary feature)
    df["power_diff"]    = df["power_a"]   - df["power_b"]
    df["pyth_diff"]     = df["pyth_wp_a"] - df["pyth_wp_b"]
    df["sos_cred_diff"] = df["sos_cred_a"]- df["sos_cred_b"]

    # SOS-adjusted efficiency differential
    df["adj_em_sos_a"] = df["kenpom_em_a"] * df["sos_cred_a"]
    df["adj_em_sos_b"] = df["kenpom_em_b"] * df["sos_cred_b"]
    df["adj_em_sos_diff"] = df["adj_em_sos_a"] - df["adj_em_sos_b"]

    # Possession-level matchup: A's off vs B's def
    # Simulate: expected PPP for A against B = adj_off_A * (adj_def_B / league_avg)
    LEAGUE_DEF = 100.0  # baseline
    df["poss_match_a"] = df["adj_off_a"] * (LEAGUE_DEF / df["adj_def_b"].clip(lower=88))
    df["poss_match_b"] = df["adj_off_b"] * (LEAGUE_DEF / df["adj_def_a"].clip(lower=88))
    df["poss_match_diff"] = df["poss_match_a"] - df["poss_match_b"]

    # eFG differential (offense advantage in shooting quality)
    df["efg_net_a"] = df["efg_a"]     - df["def_efg_a"]   # A off - A def allowed
    df["efg_net_b"] = df["efg_b"]     - df["def_efg_b"]
    df["efg_net_diff"] = df["efg_net_a"] - df["efg_net_b"]

    # Turnover creation vs prevention
    # A wants to force B turnovers and protect its own ball
    df["tov_matchup_a"] = df["tov_b"]  - df["tov_a"]      # positive = A wins TOV battle
    df["tov_matchup_b"] = df["tov_a"]  - df["tov_b"]

    # Offensive rebound vs defensive rebound matchup
    # A.ORB vs B's implied DRB (1 - orb_b is a proxy for DRB quality)
    df["reb_matchup_a"] = df["orb_a"] * (1 - df["orb_b"])  # A ORB × B DRB proxy
    df["reb_matchup_diff"] = df["orb_a"] - df["orb_b"]

    # Free throw edge
    df["ft_edge_a"] = df["ft_rate_a"] * df["ft_pct_a"]
    df["ft_edge_b"] = df["ft_rate_b"] * df["ft_pct_b"]
    df["ft_edge_diff"] = df["ft_edge_a"] - df["ft_edge_b"]

    # 3PA rate vs opponent perimeter defense (proxy: def_efg is a mix; no explicit 3pt def)
    df["three_pt_threat_a"] = df["three_pt_rate_a"] * (1 - df["def_efg_b"])

    # Tempo mismatch: absolute difference → game pace effect
    df["tempo_abs_diff"] = (df["tempo_a"] - df["tempo_b"]).abs()
    df["game_tempo_est"] = (df["tempo_a"] * 0.45 + df["tempo_b"] * 0.55).where(
        df["tempo_a"] >= df["tempo_b"],
        other=(df["tempo_a"] * 0.55 + df["tempo_b"] * 0.45)
    )

    # Non-linear transforms: squared terms capture extreme advantages
    df["em_diff_sq"]    = df["em_diff"]   ** 2
    df["elo_diff_sq"]   = df["elo_diff"]  ** 2
    df["power_diff_sq"] = df["power_diff"]** 2

    # Capped transforms: prevent extreme outliers from dominating
    df["em_diff_cap"]   = df["em_diff"].clip(-25, 25)
    df["elo_diff_cap"]  = df["elo_diff"].clip(-400, 400)

    # Percentile rank within this dataset (cross-sectional)
    df["em_diff_pctile"] = df["em_diff"].rank(pct=True)

    register("power_diff","matchup","composite","selection_sunday","none","+","high","diff")
    register("poss_match_diff","matchup","KenPom PPP","selection_sunday","none","+","high","derived")
    register("efg_net_diff","matchup","KenPom","selection_sunday","none","+","high","diff")
    register("tov_matchup_a","matchup","KenPom","selection_sunday","none","+","medium","interaction")
    register("reb_matchup_diff","matchup","KenPom","selection_sunday","none","+","medium","diff")
    register("ft_edge_diff","matchup","KenPom","selection_sunday","none","+","medium","interaction")
    register("tempo_abs_diff","matchup","KenPom","selection_sunday","none","context","medium","abs")
    register("em_diff_sq","matchup","KenPom","selection_sunday","none","+","medium","squared")
    return df


# ─── C. Volatility features ───────────────────────────────────────────────────

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Real volatility from:
      - 3pt attempt rate (binary outcomes per possession)
      - turnover rate (unpredictable ball security)
      - combined game-level chaos score

    NOTE: game-log SD features (scoring_SD, shooting_SD) are in FEATURE_DICT
    but not available in our v1 dataset — marked as missing for future v2.
    """
    # Team-level variance scores
    df["var_score_a"] = (
        0.45 * df["three_pt_rate_a"]
      + 0.30 * df["tov_a"]
      + 0.15 * np.clip(df["tempo_a"] / 76, 0, 1)
      + 0.10 * (1 - df["sos_cred_a"])  # weak schedule inflates uncertainty
    )
    df["var_score_b"] = (
        0.45 * df["three_pt_rate_b"]
      + 0.30 * df["tov_b"]
      + 0.15 * np.clip(df["tempo_b"] / 76, 0, 1)
      + 0.10 * (1 - df["sos_cred_b"])
    )

    # Game-level chaos: average of both team variance scores
    df["game_chaos"] = (df["var_score_a"] + df["var_score_b"]) / 2

    # Chaos asymmetry: one team much more volatile than other
    df["chaos_asymmetry"] = (df["var_score_a"] - df["var_score_b"]).abs()

    # 3pt rate differential: team leaning on 3s against opponent that forces 3s
    df["three_pt_rate_sum"] = df["three_pt_rate_a"] + df["three_pt_rate_b"]

    # Pace-induced variance: faster game → more possessions → less variance per game
    # (more possessions reduces per-game variance via law of large numbers)
    df["pace_variance_factor"] = 1.0 / np.clip(df["game_tempo_est"] / 68, 0.8, 1.3)

    # Upset likelihood proxy: high chaos + close power ratings
    df["upset_potential"] = df["game_chaos"] / (1 + df["power_diff"].abs())

    # Missing features (would be in v2 with game log data):
    # scoring_sd_a, scoring_sd_b, shooting_sd_a, tov_sd_a, margin_sd_a

    register("var_score_a","volatility","KenPom","selection_sunday","none","context","medium","derived")
    register("game_chaos","volatility","derived","selection_sunday","none","context","medium","derived")
    register("upset_potential","volatility","derived","selection_sunday","none","context","medium","derived")
    return df


# ─── D. Context features ─────────────────────────────────────────────────────

def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament context:
      - Conference strength corrections (not just SOS multiplier)
      - Experience and roster depth
      - Mid-major inflation flag
    """
    # Conference power differential
    if "conf_power_a" in df.columns:
        df["conf_power_diff"] = df["conf_power_a"] - df["conf_power_b"]
    else:
        df["conf_power_diff"] = 0.0

    # Mid-major flag (proxy: low SOS + low conf_power)
    df["is_mid_major_a"] = ((df["sos_a"] < 0.55) & (df.get("conf_power_a", pd.Series([1]*len(df))) < 0.60)).astype(float)
    df["is_mid_major_b"] = ((df["sos_b"] < 0.55) & (df.get("conf_power_b", pd.Series([1]*len(df))) < 0.60)).astype(float)

    # Mid-major matchup interaction: model should be skeptical of mid-major ratings
    df["mid_major_mismatch"] = (df["is_mid_major_a"] != df["is_mid_major_b"]).astype(float)

    # Experience differential
    df["experience_diff"] = df["experience_a"] - df["experience_b"]

    # Tournament-readiness proxy: experienced team in unfamiliar conference
    df["tourney_readiness_a"] = df["experience_a"] * df["sos_cred_a"]
    df["tourney_readiness_b"] = df["experience_b"] * df["sos_cred_b"]
    df["tourney_readiness_diff"] = df["tourney_readiness_a"] - df["tourney_readiness_b"]

    # SOS quintile (1=weakest, 5=strongest schedule)
    df["sos_quintile_a"] = pd.qcut(df["sos_a"], q=5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
    df["sos_quintile_b"] = pd.qcut(df["sos_b"], q=5, labels=[1,2,3,4,5], duplicates="drop").astype(float)

    register("conf_power_diff","context","derived","selection_sunday","none","+","medium","diff")
    register("mid_major_mismatch","context","derived","selection_sunday","none","context","medium","binary")
    register("tourney_readiness_diff","context","derived","selection_sunday","none","+","low","interaction")
    return df


# ─── E. Calibration / meta features ──────────────────────────────────────────

def add_calibration_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features used for probability calibration and model confidence:
      - Seed prior probability (historical matchup rates)
      - Market disagreement with KenPom model
      - Uncertainty flags (small conference, injured team, weak schedule)
      - Model confidence score
    """
    from pipeline.baselines import seed_win_prob, KenPomBaseline

    # Seed prior
    df["seed_prior_a"] = [seed_win_prob(int(r.seed_a), int(r.seed_b))
                          for _, r in df.iterrows()]

    # KenPom implied probability
    kp_prob = expit(df["em_diff"].values * 0.178)
    df["kenpom_prob_a"] = kp_prob

    # Market disagreement: |market - kenpom| → large = interesting signal
    df["market_kenpom_disagree"] = (df["market_prob_a"] - df["kenpom_prob_a"]).abs()

    # Market direction: is market above or below KenPom?
    df["market_above_kenpom"] = (df["market_prob_a"] > df["kenpom_prob_a"]).astype(float)

    # Elo disagreement with KenPom
    elo_prob = expit((df["elo_a"] - df["elo_b"]).values / 400 * np.log(10))
    df["elo_prob_a"] = elo_prob
    df["elo_kenpom_disagree"] = (df["elo_prob_a"] - df["kenpom_prob_a"]).abs()

    # Ensemble spread: max disagreement across all signals
    stacked = np.stack([df["seed_prior_a"].values,
                        df["kenpom_prob_a"].values,
                        df["elo_prob_a"].values,
                        df["market_prob_a"].values], axis=1)
    df["ensemble_spread"] = stacked.max(axis=1) - stacked.min(axis=1)
    df["ensemble_mean"]   = stacked.mean(axis=1)

    # Model confidence: high when all signals agree AND underdog is clearly worse
    df["model_confidence"] = (1 - df["ensemble_spread"]) * (1 - df["game_chaos"])

    # Uncertainty flags
    df["flag_mid_major"]     = ((df["sos_a"] < 0.50) | (df["sos_b"] < 0.50)).astype(int)
    df["flag_close_game"]    = (df["ensemble_spread"] > 0.25).astype(int)
    df["flag_large_em_diff"] = (df["em_diff"].abs() > 20).astype(int)
    df["flag_high_chaos"]    = (df["game_chaos"] > 0.35).astype(int)

    register("seed_prior_a","calibration","historical","selection_sunday","none","weak+","high","lookup")
    register("market_kenpom_disagree","calibration","derived","selection_sunday","none","context","high","abs_diff")
    register("ensemble_spread","calibration","derived","selection_sunday","none","context","high","derived")
    register("model_confidence","calibration","derived","selection_sunday","none","+","medium","derived")
    return df


# ─── F. Style clustering ─────────────────────────────────────────────────────

def add_style_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Team style archetypes:
      - bomb_heavy:    high 3pt_rate + fast tempo
      - rim_bully:     high ORB + low 3pt_rate
      - slow_elite_D:  low tempo + low adj_def
      - chaos_agent:   high TOV + high 3pt_rate
      - balanced:      none of the above dominate

    These create matchup interaction features.
    """
    def classify_style(row, suffix="a"):
        tpa  = row[f"three_pt_rate_{suffix}"]
        orb  = row[f"orb_{suffix}"]
        tov  = row[f"tov_{suffix}"]
        tempo = row[f"tempo_{suffix}"]
        adj_def = row[f"adj_def_{suffix}"]

        if tpa > 0.40 and tempo > 70:
            return "bomb_heavy"
        elif orb > 0.29 and tpa < 0.36:
            return "rim_bully"
        elif tempo < 65 and adj_def < 95:
            return "slow_elite_d"
        elif tov > 0.185 and tpa > 0.38:
            return "chaos_agent"
        else:
            return "balanced"

    df["style_a"] = [classify_style(r, "a") for _, r in df.iterrows()]
    df["style_b"] = [classify_style(r, "b") for _, r in df.iterrows()]

    # Style matchup interaction
    # bomb_heavy vs slow_elite_d → interesting clash (slow team neutralises)
    df["style_clash"] = (
        ((df["style_a"] == "bomb_heavy") & (df["style_b"] == "slow_elite_d")) |
        ((df["style_b"] == "bomb_heavy") & (df["style_a"] == "slow_elite_d"))
    ).astype(int)

    # Chaos matchup: both chaotic teams
    df["double_chaos"] = (
        (df["style_a"] == "chaos_agent") & (df["style_b"] == "chaos_agent")
    ).astype(int)

    # One-hot encode style_a for model (style_b as interaction)
    for style in ["bomb_heavy","rim_bully","slow_elite_d","chaos_agent","balanced"]:
        df[f"style_a_{style}"] = (df["style_a"] == style).astype(int)

    register("style_a","style","derived","selection_sunday","none","context","low","categorical")
    register("style_clash","style","derived","selection_sunday","none","context","low","binary")
    return df


# ─── Full feature pipeline ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Run all feature engineering phases in order.
    Input:  raw or symmetrized matchup dataframe
    Output: dataframe with all engineered features appended
    """
    import time
    t0 = time.time()

    df = df.copy()
    phases = [
        ("strength",    add_strength_features),
        ("matchup",     add_matchup_features),
        ("volatility",  add_volatility_features),
        ("context",     add_context_features),
        ("calibration", add_calibration_features),
        ("style",       add_style_features),
    ]
    for name, fn in phases:
        before = len(df.columns)
        df = fn(df)
        if verbose:
            print(f"  {name:<14}: +{len(df.columns)-before} features → {len(df.columns)} total")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Feature build complete: {len(df.columns)} total features in {elapsed:.2f}s")
    return df


def get_feature_groups() -> Dict[str, List[str]]:
    """Return features grouped by class."""
    groups: Dict[str, List[str]] = {c:[] for c in FEATURE_CLASSES}
    for name, meta in FEATURE_REGISTRY.items():
        groups[meta["cls"]].append(name)
    return groups


def get_model_features(df: pd.DataFrame, include_classes: List[str] = None) -> List[str]:
    """
    Return the canonical list of model input features.
    Excludes: raw identifiers, metadata, target, and calibration-only features.
    """
    exclude_patterns = [
        "team_a","team_b","region","round","season","source_conf","notes",
        "team_a_won", "style_a", "style_b",  # categorical, encoded separately
    ]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    model_feats = [c for c in numeric_cols
                   if not any(c.startswith(p) or c == p for p in exclude_patterns)
                   and c not in ["sos_quintile_a","sos_quintile_b"]]  # has NaN from qcut
    return model_feats


def feature_missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Log missingness by feature."""
    miss = df.isnull().sum()
    pct  = miss / len(df)
    out  = pd.DataFrame({"missing":miss,"pct":pct}).query("missing > 0").sort_values("pct",ascending=False)
    return out


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    print("=" * 60)
    print(f"Feature Engineering Pipeline — {FEATURE_VERSION}")
    print("=" * 60)

    df = load_symmetrized()
    print(f"\nInput: {len(df)} rows × {len(df.columns)} columns")

    df_feat = build_features(df, verbose=True)
    print(f"\nOutput: {len(df_feat)} rows × {len(df_feat.columns)} columns")

    # Missingness report
    miss = feature_missingness_report(df_feat)
    if len(miss):
        print(f"\nMissingness ({len(miss)} features):")
        print(miss.to_string())
    else:
        print("\nNo missing values in engineered features ✓")

    # Feature groups
    groups = get_feature_groups()
    print("\nFeature registry by class:")
    for cls, feats in groups.items():
        if feats:
            print(f"  {cls:<14}: {len(feats)} features — {feats}")

    # Sample model feature list
    model_feats = get_model_features(df_feat)
    print(f"\nModel feature count: {len(model_feats)}")
    print(f"Sample: {model_feats[:8]}")

    # Style distribution
    print("\nStyle archetype distribution:")
    print(df_feat["style_a"].value_counts().to_string())

    # Correlation with target (top 15 most correlated features)
    corrs = df_feat[model_feats + ["team_a_won"]].corr()["team_a_won"].drop("team_a_won")
    top_corr = corrs.abs().sort_values(ascending=False).head(15)
    print("\nTop 15 features by |correlation| with team_a_won:")
    for feat, val in top_corr.items():
        sign = "+" if corrs[feat] > 0 else "-"
        print(f"  {feat:<35}: {sign}{abs(val):.4f}")
