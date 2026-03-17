"""
Phase 4 — Modeling stack.

Pipeline:
  1. LogisticRegression on differentials (interpretability baseline)
  2. XGBoost on full feature set
  3. LightGBM on full feature set
  4. Stacked meta-model (LR + XGB + LGB + market → meta-LR)
  5. Isotonic calibration on each model
  6. Rolling-origin season-based CV (no random splits — ever)
  7. Residual model for mid-major / market-disagreement corrections
  8. All benchmarked against Phase 2 baselines

Selection Sunday freeze: all features available before tournament.
Market features flagged as 'game_tip' freeze — used only in blend, not training.
"""
import sys, os, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.model_selection import cross_val_predict
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
import joblib

from data.historical.tournament_games import load_dataframe, load_symmetrized
from pipeline.feature_engineering import build_features, get_model_features
from pipeline.baselines import (
    run_integrity_tests, evaluate, rolling_origin_cv as baseline_cv,
    SeedBaseline, KenPomBaseline, EloBaseline, MarketBaseline, BlendBaseline,
    BENCHMARK_TARGETS
)

MODEL_VERSION  = "model_v1"
ARTIFACTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Core feature set used for model training ──────────────────────────────────
# Excludes market features (game_tip freeze) from primary model
# Market used only in final blend

CORE_FEATURES = [
    # Strength
    "em_diff", "elo_diff", "em_diff_sq", "em_diff_cap",
    "power_diff", "power_diff_sq", "pyth_diff",
    # Matchup
    "poss_match_diff", "efg_net_diff", "efg_diff", "def_efg_diff",
    "tov_matchup_a", "reb_matchup_diff", "ft_edge_diff",
    "adj_off_diff", "adj_def_diff",
    # Volatility
    "game_chaos", "var_score_a", "upset_potential",
    "tempo_abs_diff", "three_pt_rate_sum",
    # Context
    "sos_diff", "conf_power_diff", "adj_em_sos_diff",
    "mid_major_mismatch", "experience_diff",
    # Ensemble signals (no market)
    "seed_prior_a", "ensemble_spread",
    "elo_kenpom_disagree",
    # Style
    "style_clash", "double_chaos",
]

# Features that include market (game_tip freeze — used in final blend only)
MARKET_FEATURES = [
    "market_prob_a", "market_kenpom_disagree", "market_above_kenpom",
    "ensemble_mean", "model_confidence",
]


# ── Utility: clean feature matrix ────────────────────────────────────────────

def get_X_y(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    available = [f for f in features if f in df.columns]
    X = df[available].values.astype(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["team_a_won"].values.astype(float)
    return X, y, available


def build_recency_weights(df: pd.DataFrame, half_life_seasons: float = 4.0) -> np.ndarray:
    """
    Assign larger training weights to more recent seasons.
    Weight decays by half every `half_life_seasons`.
    """
    if "season" not in df.columns or len(df) == 0:
        return np.ones(len(df), dtype=float)
    max_s = float(df["season"].max())
    age = max_s - df["season"].astype(float).values
    w = np.power(0.5, age / max(half_life_seasons, 0.1))
    # Keep old seasons represented while still prioritizing recent data.
    return np.clip(w, 0.25, 1.0)


# ── Model 1: Logistic Regression ─────────────────────────────────────────────

class ModelLogistic:
    name = "logistic_full"

    def __init__(self, C: float = 0.5):
        self.C = C
        self.scaler = StandardScaler()
        self.model  = LogisticRegression(C=C, max_iter=3000, solver="lbfgs",
                                          random_state=42, class_weight="balanced")
        self.features: List[str] = []
        self.coefs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "ModelLogistic":
        X, y, self.features = get_X_y(df, CORE_FEATURES)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y, sample_weight=sample_weight)
        self.coefs = dict(zip(self.features, self.model.coef_[0]))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = get_X_y(df, self.features)
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


# ── Model 2: XGBoost ─────────────────────────────────────────────────────────

class ModelXGBoost:
    name = "xgboost"

    def __init__(self, **kwargs):
        self.params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "scale_pos_weight": 1.0,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "verbosity": 0,
        }
        self.params.update(kwargs)
        self.model    = None
        self.features: List[str] = []
        self.importances: Dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        val_df: pd.DataFrame = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ModelXGBoost":
        X, y, self.features = get_X_y(df, CORE_FEATURES)
        self.model = xgb.XGBClassifier(**self.params)
        if val_df is not None:
            Xv, yv, _ = get_X_y(val_df, self.features)
            self.model.fit(X, y,
                           sample_weight=sample_weight,
                           eval_set=[(Xv, yv)],
                           verbose=False)
        else:
            self.model.fit(X, y, sample_weight=sample_weight)
        imp = self.model.get_booster().get_score(importance_type="gain")
        self.importances = {self.features[int(k[1:])]: v for k, v in imp.items()
                            if k.startswith("f") and int(k[1:]) < len(self.features)}
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = get_X_y(df, self.features)
        return self.model.predict_proba(X)[:, 1]

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.importances.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Model 3: LightGBM ─────────────────────────────────────────────────────────

class ModelLightGBM:
    name = "lightgbm"

    def __init__(self, **kwargs):
        self.params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "num_leaves": 20,
            "min_child_samples": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "random_state": 42,
            "verbose": -1,
        }
        self.params.update(kwargs)
        self.model    = None
        self.features: List[str] = []
        self.importances: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "ModelLightGBM":
        X, y, self.features = get_X_y(df, CORE_FEATURES)
        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(X, y, sample_weight=sample_weight)
        imp = self.model.feature_importances_
        self.importances = dict(zip(self.features, imp.tolist()))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = get_X_y(df, self.features)
        return self.model.predict_proba(X)[:, 1]

    def top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self.importances.items(), key=lambda x: x[1], reverse=True)[:n]


# ── Model 4: Stacked ensemble ─────────────────────────────────────────────────

class ModelStack:
    name = "stacked_ensemble"

    def __init__(self):
        self.lr  = ModelLogistic(C=0.5)
        self.xgb = ModelXGBoost()
        self.lgb = ModelLightGBM()
        self.meta = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                                        random_state=42)
        self.meta_scaler = StandardScaler()
        self.calibrator  = None

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "ModelStack":
        # Fit base models
        self.lr.fit(df, sample_weight=sample_weight)
        self.xgb.fit(df, sample_weight=sample_weight)
        self.lgb.fit(df, sample_weight=sample_weight)
        # Build meta-features via in-sample predictions
        p_lr  = self.lr.predict_proba(df)
        p_xgb = self.xgb.predict_proba(df)
        p_lgb = self.lgb.predict_proba(df)
        p_mkt = df["market_prob_a"].values.astype(float)
        p_kp  = expit(df["em_diff"].values * 0.178)
        meta_X = np.stack([p_lr, p_xgb, p_lgb, p_mkt, p_kp], axis=1)
        meta_Xs = self.meta_scaler.fit_transform(meta_X)
        y = df["team_a_won"].values
        self.meta.fit(meta_Xs, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        p_lr  = self.lr.predict_proba(df)
        p_xgb = self.xgb.predict_proba(df)
        p_lgb = self.lgb.predict_proba(df)
        p_mkt = df["market_prob_a"].values.astype(float)
        p_kp  = expit(df["em_diff"].values * 0.178)
        meta_X = np.stack([p_lr, p_xgb, p_lgb, p_mkt, p_kp], axis=1)
        return self.meta.predict_proba(self.meta_scaler.transform(meta_X))[:, 1]

    @property
    def meta_weights(self) -> Dict[str, float]:
        names = ["logistic","xgboost","lightgbm","market","kenpom"]
        return dict(zip(names, self.meta.coef_[0]))


# ── Isotonic calibration wrapper ──────────────────────────────────────────────

class CalibratedModel:
    def __init__(self, base_model, method: str = "isotonic"):
        self.base   = base_model
        self.method = method
        self.cal    = None
        self.name   = f"{base_model.name}_calibrated"

    def fit(
        self,
        df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "CalibratedModel":
        self.base.fit(df, sample_weight=sample_weight)
        # Calibrate on val_df if provided, else in-sample (less ideal)
        cal_df = val_df if val_df is not None else df
        raw_probs = self.base.predict_proba(cal_df)
        y = cal_df["team_a_won"].values
        sw_cal = sample_weight if val_df is None else None
        if self.method == "isotonic":
            self.cal = IsotonicRegression(out_of_bounds="clip")
            self.cal.fit(raw_probs, y, sample_weight=sw_cal)
        else:  # platt / sigmoid
            # Fit logistic on raw logit
            lp = np.clip(raw_probs, 1e-6, 1-1e-6)
            logits = np.log(lp / (1 - lp)).reshape(-1, 1)
            self.cal = LogisticRegression(max_iter=1000)
            self.cal.fit(logits, y, sample_weight=sw_cal)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.base.predict_proba(df)
        if self.method == "isotonic":
            return np.clip(self.cal.predict(raw), 0.02, 0.98)
        else:
            lp = np.clip(raw, 1e-6, 1-1e-6)
            logits = np.log(lp / (1 - lp)).reshape(-1, 1)
            return np.clip(self.cal.predict_proba(logits)[:, 1], 0.02, 0.98)


# ── Residual / correction model ───────────────────────────────────────────────

class ResidualModel:
    """
    Stage-2 correction for known failure modes:
      - mid-major teams
      - high market disagreement
      - high-chaos games
    Fits a correction on the residuals of the primary model.
    """
    name = "residual_correction"

    def __init__(self, primary_model):
        self.primary = primary_model
        self.corrector = Ridge(alpha=5.0)
        self.features_used: List[str] = []

    def fit(self, df: pd.DataFrame) -> "ResidualModel":
        self.primary.fit(df)
        raw_probs = self.primary.predict_proba(df)
        raw_probs = np.clip(raw_probs, 1e-6, 1-1e-6)
        y = df["team_a_won"].values.astype(float)

        # Residual = y - p (positive = model under-predicted)
        residuals = y - raw_probs

        # Correction features: where does the model tend to fail?
        corr_feats = ["mid_major_mismatch","market_kenpom_disagree",
                      "game_chaos","upset_potential","ensemble_spread",
                      "flag_mid_major","flag_close_game","elo_kenpom_disagree"]
        available = [f for f in corr_feats if f in df.columns]
        self.features_used = available

        if available:
            Xc = df[available].values.astype(float)
            Xc = np.nan_to_num(Xc)
            self.corrector.fit(Xc, residuals)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.primary.predict_proba(df)
        if self.features_used:
            Xc = df[self.features_used].values.astype(float)
            Xc = np.nan_to_num(Xc)
            correction = self.corrector.predict(Xc)
            corrected = np.clip(raw + correction * 0.5, 0.02, 0.98)
        else:
            corrected = raw
        return corrected


# ── Season-based rolling-origin CV for model stack ────────────────────────────

def rolling_cv_model_stack(df_sym: pd.DataFrame,
                            min_train: int = 6) -> pd.DataFrame:
    """
    Rolling-origin CV for all models.
    Train on [2005..season-1], test on [season].
    Returns per-season results for all models.
    """
    seasons = sorted(df_sym["season"].unique())
    rows: List[Dict] = []

    # Baseline models for comparison
    baselines = [SeedBaseline(), KenPomBaseline(), EloBaseline(),
                 MarketBaseline(), BlendBaseline()]

    for test_s in seasons:
        train_ss = [s for s in seasons if s < test_s]
        if len(train_ss) < min_train:
            continue
        train = df_sym[df_sym["season"].isin(train_ss)]
        test  = df_sym[df_sym["season"] == test_s]
        w_train = build_recency_weights(train)
        if len(test) < 4:
            continue

        y  = test["team_a_won"].values
        sa = test["seed_a"].values
        sb = test["seed_b"].values

        # Fit and evaluate ML models
        for ModelCls in [ModelLogistic, ModelXGBoost, ModelLightGBM]:
            try:
                m = ModelCls()
                m.fit(train, sample_weight=w_train)
                p = m.predict_proba(test)
                result = evaluate(y, p, m.name, seed_a=sa, seed_b=sb)
                result["season"] = int(test_s)
                result["n_train"] = len(train)
                rows.append(result)
            except Exception as e:
                rows.append({"model": ModelCls.name if hasattr(ModelCls, "name") else str(ModelCls),
                              "season": int(test_s), "error": str(e)})

        # Stack (fit base models on train, test on test)
        try:
            stack = ModelStack()
            stack.fit(train, sample_weight=w_train)
            p = stack.predict_proba(test)
            result = evaluate(y, p, "stacked_ensemble", seed_a=sa, seed_b=sb)
            result["season"] = int(test_s)
            result["n_train"] = len(train)
            rows.append(result)
        except Exception as e:
            rows.append({"model":"stacked_ensemble","season":int(test_s),"error":str(e)})

        # Baselines for comparison
        for bl in baselines:
            result = evaluate(y, bl.predict(test), bl.name, seed_a=sa, seed_b=sb)
            result["season"] = int(test_s)
            result["n_train"] = len(train)
            rows.append(result)

    return pd.DataFrame(rows)


# ── Benchmark check: does our model beat all baselines? ──────────────────────

def check_benchmarks(cv_results: pd.DataFrame) -> Dict[str, bool]:
    """
    Report whether our models beat the Phase 2 baselines on rolling-origin CV.
    This is the ONLY acceptance criterion for the model.
    """
    summary = cv_results.groupby("model")[["log_loss","accuracy"]].mean()
    checks: Dict[str, bool] = {}

    for baseline_name, targets in BENCHMARK_TARGETS.items():
        if baseline_name not in summary.index:
            continue
        bl_ll  = float(summary.loc[baseline_name, "log_loss"])
        bl_acc = float(summary.loc[baseline_name, "accuracy"])

        for model_name in ["logistic_full","xgboost","lightgbm","stacked_ensemble"]:
            if model_name not in summary.index:
                continue
            m_ll  = float(summary.loc[model_name, "log_loss"])
            m_acc = float(summary.loc[model_name, "accuracy"])
            key = f"{model_name}_beats_{baseline_name}"
            checks[key] = (m_ll < bl_ll) and (m_acc > bl_acc)

    return checks


# ── Full training + evaluation pipeline ──────────────────────────────────────

def run_model_pipeline(save: bool = True, verbose: bool = True) -> Dict:
    t0 = time.time()

    # Load + build features
    df_raw = load_dataframe()
    df_sym = load_symmetrized()
    df_sym = build_features(df_sym)

    if verbose:
        print(f"Model pipeline {MODEL_VERSION}")
        print(f"Training data: {len(df_sym)} rows × {len(df_sym.columns)} features")

    # Rolling-origin CV
    if verbose:
        print("\nRolling-origin season CV...")
    cv_df = rolling_cv_model_stack(df_sym, min_train=6)

    if not cv_df.empty and "log_loss" in cv_df.columns:
        cv_summary = cv_df.dropna(subset=["log_loss"]).groupby("model")[["log_loss","brier","accuracy"]].mean()
    else:
        cv_summary = pd.DataFrame()

    if verbose and not cv_summary.empty:
        print("\nRolling-origin CV results")
        sorted_cv = cv_summary.sort_values("log_loss")
        print(sorted_cv.round(4).to_string())

    # Benchmark checks
    if not cv_df.empty:
        benchmarks = check_benchmarks(cv_df)
        beats = sum(benchmarks.values())
        total = len(benchmarks)
        if verbose:
            print(f"\nBenchmark checks: {beats}/{total} passed")
            for k, v in sorted(benchmarks.items()):
                icon = "✓" if v else "✗"
                print(f"  {icon} {k}")
    else:
        benchmarks = {}

    # Train final models on full dataset
    if verbose:
        print("\nFinal models (full dataset)")

    final_models = {}
    y_full = df_sym["team_a_won"].values
    sa     = df_sym["seed_a"].values
    sb     = df_sym["seed_b"].values

    for ModelCls, label in [
        (ModelLogistic, "logistic_full"),
        (ModelXGBoost,  "xgboost"),
        (ModelLightGBM, "lightgbm"),
    ]:
        try:
            m = ModelCls()
            w_full = build_recency_weights(df_sym)
            m.fit(df_sym, sample_weight=w_full)
            p = m.predict_proba(df_sym)
            metrics = evaluate(y_full, p, label, seed_a=sa, seed_b=sb)
            final_models[label] = {"model": m, "metrics": metrics}
            if verbose:
                print(f"  {label:<22}: LL={metrics['log_loss']:.4f} "
                      f"acc={metrics['accuracy']:.3%} brier={metrics['brier']:.4f}")
        except Exception as e:
            if verbose: print(f"  {label}: ERROR — {e}")

    # Stack
    try:
        stack = ModelStack()
        w_full = build_recency_weights(df_sym)
        stack.fit(df_sym, sample_weight=w_full)
        p = stack.predict_proba(df_sym)
        metrics = evaluate(y_full, p, "stacked_ensemble", seed_a=sa, seed_b=sb)
        final_models["stacked_ensemble"] = {"model": stack, "metrics": metrics}
        if verbose:
            print(f"  {'stacked_ensemble':<22}: LL={metrics['log_loss']:.4f} "
                  f"acc={metrics['accuracy']:.3%} brier={metrics['brier']:.4f}")
            print(f"\n  Stack meta-weights: {stack.meta_weights}")
    except Exception as e:
        if verbose: print(f"  stacked_ensemble: ERROR — {e}")

    # Calibrated stack
    try:
        if "stacked_ensemble" in final_models:
            cal_stack = CalibratedModel(ModelStack(), method="isotonic")
            w_full = build_recency_weights(df_sym)
            cal_stack.fit(df_sym, sample_weight=w_full)
            p = cal_stack.predict_proba(df_sym)
            metrics = evaluate(y_full, p, "stacked_calibrated", seed_a=sa, seed_b=sb)
            final_models["stacked_calibrated"] = {"model": cal_stack, "metrics": metrics}
            if verbose:
                print(f"  {'stacked_calibrated':<22}: LL={metrics['log_loss']:.4f} "
                      f"acc={metrics['accuracy']:.3%} ECE={metrics['ece']:.4f}")
    except Exception as e:
        if verbose: print(f"  stacked_calibrated: ERROR — {e}")

    # Feature importance from best tree model
    if verbose and "lightgbm" in final_models:
        lgb_m = final_models["lightgbm"]["model"]
        print("\nTop features (LightGBM)")
        for feat, imp in lgb_m.top_features(12):
            print(f"  {feat:<35}: {imp:.1f}")

    if verbose and "logistic_full" in final_models:
        lr_m = final_models["logistic_full"]["model"]
        print("\nTop features (Logistic coefficients)")
        for feat, coef in lr_m.top_features(12):
            print(f"  {feat:<35}: {coef:+.4f}")

    # Save best model
    if save and final_models:
        # Always save the stacked calibrated ensemble as the production model
        # Fall back to best single model if stack isn't available
        if "stacked_calibrated" in final_models and "model" in final_models["stacked_calibrated"]:
            prod_name = "stacked_calibrated"
        else:
            prod_name = min(
                [k for k in final_models if "metrics" in final_models[k]],
                key=lambda k: final_models[k]["metrics"]["log_loss"]
            )
        best_path = os.path.join(ARTIFACTS_DIR, f"{MODEL_VERSION}_best.pkl")
        joblib.dump(final_models[prod_name]["model"], best_path)
        if verbose:
            print(f"\nProduction model ({prod_name}) saved → {best_path}")

    elapsed = time.time() - t0
    if verbose:
        print(f"\nPipeline complete in {elapsed:.1f}s")

    return {
        "model_version":  MODEL_VERSION,
        "cv_summary":     cv_summary.to_dict() if not cv_summary.empty else {},
        "benchmarks":     benchmarks,
        "final_metrics":  {k: v["metrics"] for k,v in final_models.items()},
        "final_models":   final_models,
        "elapsed_sec":    round(elapsed, 1),
    }


if __name__ == "__main__":
    # Re-import as module so classes are pickled with correct module path (not __main__)
    import importlib, sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _mp = importlib.import_module("pipeline.model_pipeline")
    results = _mp.run_model_pipeline(save=True, verbose=True)
    os.makedirs("reports", exist_ok=True)
    report = {k:v for k,v in results.items() if k != "final_models"}
    with open("reports/model_pipeline_results.json","w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\nResults saved → reports/model_pipeline_results.json")
