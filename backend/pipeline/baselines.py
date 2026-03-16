"""
Phase 2 — Baseline models.
Every future model MUST beat ALL of these on rolling-origin CV.

Hard metrics:
  - game-level log loss
  - Brier score
  - ECE (expected calibration error)
  - straight-up accuracy
  - upset-pick accuracy (correctly calling lower seed wins)

Target:
  - Beat seed baseline LL by ≥8%
  - Beat KenPom-only LL by ≥3%
  - Match or beat market baseline calibration
  - Beat simple blend accuracy by ≥2%
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple

from data.historical.tournament_games import load_dataframe, load_symmetrized

# ── Historical seed matchup win rates (1985-2024, ~1500 games) ─────────────
SEED_MATCHUP_RATES = {
    (1,16):0.993,(2,15):0.939,(3,14):0.848,(4,13):0.791,
    (5,12):0.648,(6,11):0.628,(7,10):0.604,(8,9): 0.490,
}

def seed_win_prob(sa: int, sb: int) -> float:
    """Win prob for seed sa facing seed sb."""
    lo, hi = min(sa,sb), max(sa,sb)
    base = SEED_MATCHUP_RATES.get((lo,hi), 0.5)
    return base if sa <= sb else (1.0 - base)


# ── Core metric suite ─────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str,
             seed_a: np.ndarray = None, seed_b: np.ndarray = None) -> Dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-6, 1-1e-6)

    ll    = log_loss(y_true, y_pred)
    bs    = brier_score_loss(y_true, y_pred)
    acc   = float(((y_pred >= 0.5) == y_true).mean())

    # Expected Calibration Error — 10 quantile bins
    try:
        prob_t, prob_p = calibration_curve(y_true, y_pred, n_bins=10, strategy="quantile")
        ece = float(np.mean(np.abs(prob_t - prob_p)))
    except Exception:
        ece = float("nan")

    # Upset accuracy: rows where lower seed (underdog) wins
    upset_acc = float("nan")
    if seed_a is not None and seed_b is not None:
        underdog_wins = ((seed_a > seed_b) & (y_true == 1)) | ((seed_b > seed_a) & (y_true == 0))
        if underdog_wins.sum() > 0:
            # Correct upset call = predicted prob for team_a > 0.5 when A is the dog and won
            upset_preds = ((seed_a > seed_b) & (y_pred >= 0.5)) | ((seed_b > seed_a) & (y_pred < 0.5))
            upset_acc = float((upset_preds & underdog_wins).sum() / underdog_wins.sum())

    # Sharpness: mean squared deviation from 0.5 (confident = high sharpness)
    sharpness = float(((y_pred - 0.5) ** 2).mean())

    return {
        "model":       name,
        "log_loss":    round(ll,   4),
        "brier":       round(bs,   4),
        "accuracy":    round(acc,  4),
        "ece":         round(ece,  4),
        "upset_acc":   round(upset_acc, 4) if not np.isnan(upset_acc) else None,
        "sharpness":   round(sharpness, 4),
        "n":           len(y_true),
    }


def calibration_by_bucket(y_true: np.ndarray, y_pred: np.ndarray,
                           bucket_col: np.ndarray, label: str = "") -> pd.DataFrame:
    """Calibration table split by an external grouping (e.g. seed matchup, tier)."""
    rows = []
    for grp in sorted(set(bucket_col)):
        mask = bucket_col == grp
        if mask.sum() < 3:
            continue
        yt = y_true[mask]
        yp = np.clip(y_pred[mask], 1e-6, 1-1e-6)
        rows.append({
            "group":       f"{label}={grp}",
            "n":           mask.sum(),
            "mean_pred":   round(float(yp.mean()), 3),
            "actual_rate": round(float(yt.mean()), 3),
            "cal_error":   round(float(abs(yp.mean() - yt.mean())), 3),
        })
    return pd.DataFrame(rows)


# ── Baseline model classes ────────────────────────────────────────────────

class SeedBaseline:
    name = "seed_only"
    conf = "low"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([seed_win_prob(int(r.seed_a), int(r.seed_b))
                         for _, r in df.iterrows()])


class KenPomBaseline:
    name = "kenpom_only"
    conf = "high"
    SCALE = 0.178  # calibrated: logistic(em_diff * scale) → historical accuracy

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return expit(df["em_diff"].values * self.SCALE)


class EloBaseline:
    name = "elo_only"
    conf = "high"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return expit((df["elo_a"].values - df["elo_b"].values) / 400.0 * np.log(10))


class MarketBaseline:
    name = "market_only"
    conf = "high"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return df["market_prob_a"].values.astype(float)


class BlendBaseline:
    """KenPom 35% + Elo 25% + Market 25% + Seed 15%."""
    name = "simple_blend"
    conf = "high"

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        kp   = expit(df["em_diff"].values * 0.178)
        elo  = expit((df["elo_a"].values - df["elo_b"].values) / 400 * np.log(10))
        mkt  = df["market_prob_a"].values.astype(float)
        seed = np.array([seed_win_prob(int(r.seed_a), int(r.seed_b))
                         for _, r in df.iterrows()])
        return 0.35*kp + 0.25*elo + 0.25*mkt + 0.15*seed


class LogisticRatingBaseline:
    """Logistic regression on [em_diff, elo_diff] — no calibration."""
    name = "logistic_rating"
    conf = "medium"

    def __init__(self):
        self.model: Optional[LogisticRegression] = None
        self.coefs: Dict = {}

    def fit(self, train_df: pd.DataFrame) -> "LogisticRatingBaseline":
        X = train_df[["em_diff", "elo_diff"]].values
        y = train_df["team_a_won"].values
        self.model = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="lbfgs")
        self.model.fit(X, y)
        self.coefs = dict(zip(["em_diff","elo_diff"], self.model.coef_[0]))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Must call fit() before predict()")
        return self.model.predict_proba(df[["em_diff","elo_diff"]].values)[:, 1]


# ── Benchmark targets (what our full model must beat) ──────────────────────

BENCHMARK_TARGETS = {
    # Our model must beat these on rolling-origin CV
    "seed_only":       {"log_loss": 0.680, "accuracy": 0.680},
    "kenpom_only":     {"log_loss": 0.580, "accuracy": 0.710},
    "elo_only":        {"log_loss": 0.590, "accuracy": 0.705},
    "market_only":     {"log_loss": 0.500, "accuracy": 0.740},
    "simple_blend":    {"log_loss": 0.510, "accuracy": 0.745},
    "logistic_rating": {"log_loss": 0.570, "accuracy": 0.715},
}


# ── Rolling-origin cross-validation ───────────────────────────────────────

def rolling_origin_cv(df_sym: pd.DataFrame,
                      min_train_seasons: int = 6) -> pd.DataFrame:
    """
    Season-based rolling-origin CV. Train on [2005..season-1], test on [season].
    Uses symmetrized data so both teams appear as team_a in training.
    """
    seasons = sorted(df_sym["season"].unique())
    rows: List[Dict] = []
    log_bl = LogisticRatingBaseline()
    static_bls = [SeedBaseline(), KenPomBaseline(), EloBaseline(),
                  MarketBaseline(), BlendBaseline()]

    for test_s in seasons:
        train_ss = [s for s in seasons if s < test_s]
        if len(train_ss) < min_train_seasons:
            continue
        train = df_sym[df_sym["season"].isin(train_ss)]
        test  = df_sym[df_sym["season"] == test_s]
        if len(test) < 4:
            continue

        y = test["team_a_won"].values
        log_bl.fit(train)

        for bl in static_bls + [log_bl]:
            m = evaluate(y, bl.predict(test), bl.name,
                         seed_a=test["seed_a"].values,
                         seed_b=test["seed_b"].values)
            m["season"]  = int(test_s)
            m["n_train"] = len(train)
            rows.append(m)

    return pd.DataFrame(rows)


# ── Calibration deep-dive ──────────────────────────────────────────────────

def calibration_deep_dive(df_sym: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calibration tables for each baseline:
      - by fav_seed (1,2,3,4,5...)
      - by seed matchup tier (heavy_fav, moderate_fav, close)
    """
    y = df_sym["team_a_won"].values
    bls = [SeedBaseline(), KenPomBaseline(), EloBaseline(),
           MarketBaseline(), BlendBaseline()]

    results = {}
    for bl in bls:
        preds = bl.predict(df_sym)
        by_seed = calibration_by_bucket(
            y, preds, df_sym["fav_seed"].values, label="fav_seed")
        # Tier: heavy = seed diff ≥ 8, moderate = 3-7, close = 1-2
        tiers = pd.cut(df_sym["seed_diff"].abs(),
                       bins=[-1, 2, 7, 20],
                       labels=["close_1-2","moderate_3-7","heavy_8+"]).astype(str)
        by_tier = calibration_by_bucket(y, preds, tiers.values, label="tier")
        results[bl.name] = {"by_seed": by_seed, "by_tier": by_tier}

    return results


# ── Market comparison ─────────────────────────────────────────────────────

def compare_to_market(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    Compare each model to market baseline game-by-game.
    Report: where does model beat market? Where does market win?
    """
    y    = df_sym["team_a_won"].values
    mkt  = df_sym["market_prob_a"].values.astype(float)
    mkt_ll = log_loss(y, np.clip(mkt, 1e-6, 1-1e-6))

    bls = [SeedBaseline(), KenPomBaseline(), EloBaseline(), BlendBaseline()]
    rows = []
    for bl in bls:
        preds = bl.predict(df_sym)
        bl_ll = log_loss(y, np.clip(preds, 1e-6, 1-1e-6))
        # Blend 50/50
        blend = 0.5*preds + 0.5*mkt
        blend_ll = log_loss(y, np.clip(blend, 1e-6, 1-1e-6))
        rows.append({
            "model":       bl.name,
            "model_ll":    round(bl_ll,   4),
            "market_ll":   round(mkt_ll,  4),
            "blend_ll":    round(blend_ll,4),
            "beats_market": bl_ll < mkt_ll,
            "blend_beats_market": blend_ll < mkt_ll,
        })
    return pd.DataFrame(rows)


# ── Data integrity tests ──────────────────────────────────────────────────

def run_integrity_tests(df: pd.DataFrame) -> List[str]:
    failures = []
    for col in ["market_prob_a"]:
        bad = df[(df[col]<0)|(df[col]>1)]
        if len(bad):
            failures.append(f"FAIL {col} out of [0,1]: {len(bad)} rows")
    for col in ["seed_a","seed_b"]:
        bad = df[(df[col]<1)|(df[col]>16)]
        if len(bad):
            failures.append(f"FAIL {col} out of [1,16]: {bad[col].tolist()}")
    for col in ["elo_a","elo_b"]:
        bad = df[(df[col]<1350)|(df[col]>2350)]
        if len(bad):
            failures.append(f"FAIL {col} implausible: {bad[col].tolist()}")
    bad = df[~df["team_a_won"].isin([0,1])]
    if len(bad):
        failures.append(f"FAIL team_a_won not binary: {len(bad)} rows")
    dupes = df.duplicated(subset=["season","team_a","team_b"], keep=False)
    if dupes.sum():
        failures.append(f"FAIL {dupes.sum()} duplicate games")
    for col in ["season","seed_a","seed_b","kenpom_em_a","kenpom_em_b","elo_a","elo_b","team_a_won"]:
        n = df[col].isnull().sum()
        if n:
            failures.append(f"FAIL {n} nulls in required column {col}")
    return failures


# ── Full runner ───────────────────────────────────────────────────────────

def run_all(save_path: Optional[str] = None, verbose: bool = True) -> Dict:
    df_raw = load_dataframe()
    df_sym = load_symmetrized()

    print("=" * 62)
    print("DATA INTEGRITY TESTS")
    print("=" * 62)
    fails = run_integrity_tests(df_raw)
    if fails:
        for f in fails: print(f"  {f}")
        print("  INTEGRITY FAILURES — fix data before proceeding")
        return {}
    print(f"  All PASSED  ({len(df_raw)} games, {len(df_sym)} symmetrized rows)\n")

    y    = df_sym["team_a_won"].values
    sa   = df_sym["seed_a"].values
    sb   = df_sym["seed_b"].values

    # Fit logistic on full symmetrized data
    log_bl = LogisticRatingBaseline().fit(df_sym)
    all_bls = [SeedBaseline(), KenPomBaseline(), EloBaseline(),
               MarketBaseline(), BlendBaseline(), log_bl]

    print("=" * 62)
    print("PHASE 2 — BASELINE RESULTS (full in-sample, symmetrized)")
    print("=" * 62)
    print(f"{'Model':<22} {'LogLoss':>8} {'Brier':>7} {'Acc':>7} {'ECE':>7} {'UpsetAcc':>9} {'Sharp':>7}")
    print("-" * 65)

    in_sample: Dict[str, Dict] = {}
    for bl in all_bls:
        m = evaluate(y, bl.predict(df_sym), bl.name, seed_a=sa, seed_b=sb)
        in_sample[bl.name] = m
        ua = f"{m['upset_acc']:.3f}" if m["upset_acc"] else "  n/a"
        print(f"{m['model']:<22} {m['log_loss']:>8.4f} {m['brier']:>7.4f} "
              f"{m['accuracy']:>7.3%} {m['ece']:>7.4f} {ua:>9} {m['sharpness']:>7.4f}")

    # Rolling-origin CV
    print(f"\n{'='*62}")
    print("ROLLING-ORIGIN CV (season holdout, min 6 train seasons)")
    print(f"{'='*62}")
    cv_df = rolling_origin_cv(df_sym)
    if not cv_df.empty:
        cv_summary = cv_df.groupby("model")[["log_loss","brier","accuracy"]].mean()
        print(cv_summary.round(4).to_string())
    else:
        cv_summary = pd.DataFrame()
        print("  Not enough seasons for CV")

    # Market comparison
    print(f"\n{'='*62}")
    print("MARKET COMPARISON")
    print(f"{'='*62}")
    mkt_df = compare_to_market(df_sym)
    print(mkt_df.to_string(index=False))

    # Calibration deep-dive
    print(f"\n{'='*62}")
    print("CALIBRATION BY FAV SEED (KenPom baseline)")
    print(f"{'='*62}")
    cal = calibration_deep_dive(df_sym)
    kp_cal = cal["kenpom_only"]["by_seed"]
    if not kp_cal.empty:
        print(kp_cal.to_string(index=False))

    # Logistic coefficients
    if log_bl.coefs:
        print(f"\n{'='*62}")
        print("LOGISTIC BASELINE COEFFICIENTS")
        print(f"{'='*62}")
        for feat, coef in log_bl.coefs.items():
            print(f"  {feat:<20}: {coef:+.4f}")

    result = {
        "data_version":  "matchups_v1_raw",
        "n_games_raw":   len(df_raw),
        "n_games_sym":   len(df_sym),
        "in_sample":     in_sample,
        "cv_summary":    cv_summary.to_dict() if not cv_summary.empty else {},
        "market_compare": mkt_df.to_dict(orient="records"),
        "logistic_coefs": log_bl.coefs,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({k:v for k,v in result.items() if k != "cv_summary"}, f, indent=2)
        print(f"\nResults saved → {save_path}")

    return result


if __name__ == "__main__":
    run_all(save_path="reports/baseline_results.json", verbose=True)
