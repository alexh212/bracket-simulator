"""
Calibrated game model — Phase 6 separation of concerns.

This module is the ONLY bridge between:
  - The ML training pipeline (model_pipeline.py)
  - The simulation engine (services/simulation.py)

Responsibility: given two team feature dicts (frozen at Selection Sunday),
output a calibrated win probability for team_a.

The simulation engine calls this. It does NOT touch training code.
"""
import sys, os, hashlib, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from scipy.special import expit
from typing import Dict, Optional, Tuple

from pipeline.model_pipeline import CORE_FEATURES, ModelStack
from pipeline.feature_engineering import build_features
from pipeline.baselines import seed_win_prob, SEED_MATCHUP_RATES
from data.historical.tournament_games import load_symmetrized

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

logger = logging.getLogger("bracket_api")
PYTH_EXP   = 11.5
LEAGUE_DEF = 100.0
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "models", "model_v1_best.pkl")


def _sos_cred(sos: float) -> float:
    return float(np.clip(0.68 + 0.32 * (sos / 0.85), 0, 1))


def _pyth_wp(adj_off: float, adj_def: float) -> float:
    ppp  = adj_off / 100
    dppp = adj_def / 100
    ppp  = max(ppp, 0.01)
    dppp = max(dppp, 0.01)
    return float((ppp ** PYTH_EXP) / (ppp ** PYTH_EXP + dppp ** PYTH_EXP))


def _power_rating(t: Dict) -> float:
    """Composite power rating — same formula as feature_engineering.py."""
    em    = t.get("kenpom_adj_off", 110) - t.get("kenpom_adj_def", 100)
    sos_c = _sos_cred(t.get("sos", 0.70))
    kp    = np.clip((em * sos_c + 35) / 73, 0, 1)
    elo   = np.clip((t.get("elo_current", 1750) - 1400) / 800, 0, 1)
    mkt   = float(t.get("moneyline_prob", 0.05)) * 8
    mkt   = np.clip(mkt, 0, 1)
    pyth  = _pyth_wp(t.get("kenpom_adj_off", 110), t.get("kenpom_adj_def", 100))
    inj   = float(t.get("injury_factor", 1.0)) ** 1.6
    return float(inj * (0.32*kp + 0.20*elo + 0.12*mkt + 0.08*0.5 + 0.08*(pyth - 0.5)*2))


def _var_score(t: Dict) -> float:
    sos_c = _sos_cred(t.get("sos", 0.70))
    return float(
        0.45 * t.get("three_pt_rate", 0.36)
      + 0.30 * t.get("tov_rate", 0.17)
      + 0.15 * np.clip(t.get("possessions_per_game", 70) / 76, 0, 1)
      + 0.10 * (1 - sos_c)
    )


def _style(t: Dict) -> str:
    tpa   = t.get("three_pt_rate", 0.36)
    orb   = t.get("orb_rate", 0.28)
    tov   = t.get("tov_rate", 0.17)
    tempo = t.get("possessions_per_game", 70)
    adef  = t.get("kenpom_adj_def", 100)
    if tpa > 0.40 and tempo > 70:    return "bomb_heavy"
    if orb > 0.29 and tpa < 0.36:   return "rim_bully"
    if tempo < 65 and adef < 95:     return "slow_elite_d"
    if tov > 0.185 and tpa > 0.38:  return "chaos_agent"
    return "balanced"


def build_matchup_row(a: Dict, b: Dict) -> Dict:
    """
    Build a single matchup feature dict from two team dicts.
    Mirrors the transformations in feature_engineering.py exactly.
    This is the canonical mapping from team data → model features.
    """
    # Raw fields
    em_a   = a.get("kenpom_adj_off", 110) - a.get("kenpom_adj_def", 100)
    em_b   = b.get("kenpom_adj_off", 110) - b.get("kenpom_adj_def", 100)
    elo_a  = float(a.get("elo_current", 1750))
    elo_b  = float(b.get("elo_current", 1750))
    mkt_a  = float(a.get("moneyline_prob", 0.05))
    mkt_b  = float(b.get("moneyline_prob", 0.05))
    mkt_sum = mkt_a + mkt_b
    mkt_prob_a = (mkt_a / mkt_sum) if mkt_sum > 0.002 else 0.5

    sa = int(a.get("seed", 8))
    sb = int(b.get("seed", 8))

    sos_a  = float(a.get("sos", 0.70))
    sos_b  = float(b.get("sos", 0.70))
    sc_a   = _sos_cred(sos_a)
    sc_b   = _sos_cred(sos_b)

    # Core differentials
    em_diff    = em_a   - em_b
    elo_diff   = elo_a  - elo_b

    # Strength
    pow_a = _power_rating(a)
    pow_b = _power_rating(b)
    pyth_a = _pyth_wp(a.get("kenpom_adj_off",110), a.get("kenpom_adj_def",100))
    pyth_b = _pyth_wp(b.get("kenpom_adj_off",110), b.get("kenpom_adj_def",100))

    # Possession matchup
    aoa = a.get("kenpom_adj_off", 110)
    ada = a.get("kenpom_adj_def", 100)
    aob = b.get("kenpom_adj_off", 110)
    adb = b.get("kenpom_adj_def", 100)
    poss_a = aoa * (LEAGUE_DEF / max(adb, 88))
    poss_b = aob * (LEAGUE_DEF / max(ada, 88))

    # eFG
    efg_a   = float(a.get("efg_pct", 0.51))
    efg_b   = float(b.get("efg_pct", 0.51))
    defg_a  = float(a.get("def_efg_pct", 0.49))
    defg_b  = float(b.get("def_efg_pct", 0.49))

    # TOV
    tov_a = float(a.get("tov_rate", 0.17))
    tov_b = float(b.get("tov_rate", 0.17))

    # ORB
    orb_a = float(a.get("orb_rate", 0.28))
    orb_b = float(b.get("orb_rate", 0.28))

    # FT
    ftr_a = float(a.get("ft_rate", 0.38))
    ftr_b = float(b.get("ft_rate", 0.38))
    ftp_a = float(a.get("ft_pct", 0.72))
    ftp_b = float(b.get("ft_pct", 0.72))

    # 3PT
    tpa_a = float(a.get("three_pt_rate", 0.36))
    tpa_b = float(b.get("three_pt_rate", 0.36))

    # Tempo
    tmp_a = float(a.get("possessions_per_game", 70))
    tmp_b = float(b.get("possessions_per_game", 70))

    # Conf power
    cp_a  = float(a.get("conference_power", 0.75))
    cp_b  = float(b.get("conference_power", 0.75))

    # Experience
    exp_a = float(a.get("experience", 2.2))
    exp_b = float(b.get("experience", 2.2))

    # Variance
    vs_a = _var_score(a)
    vs_b = _var_score(b)
    game_chaos = (vs_a + vs_b) / 2
    pow_diff   = pow_a - pow_b
    pyth_diff  = pyth_a - pyth_b

    # Baseline probs (for ensemble spread)
    kp_prob  = float(expit(em_diff * 0.178))
    elo_prob = float(expit(elo_diff / 400 * np.log(10)))
    sp_a     = seed_win_prob(sa, sb)
    ensemble_spread = max(kp_prob, elo_prob, mkt_prob_a, sp_a) - min(kp_prob, elo_prob, mkt_prob_a, sp_a)

    # Mid-major
    mm_a = 1.0 if (sos_a < 0.55 and cp_a < 0.60) else 0.0
    mm_b = 1.0 if (sos_b < 0.55 and cp_b < 0.60) else 0.0

    # Style
    sty_a = _style(a)
    sty_b = _style(b)

    row = {
        # Baseline features
        "em_diff":          em_diff,
        "elo_diff":         elo_diff,
        "em_diff_sq":       em_diff ** 2,
        "em_diff_cap":      float(np.clip(em_diff, -25, 25)),
        "power_diff":       pow_diff,
        "power_diff_sq":    pow_diff ** 2,
        "pyth_diff":        pyth_diff,
        # Matchup
        "poss_match_diff":  poss_a - poss_b,
        "efg_net_diff":     (efg_a - defg_a) - (efg_b - defg_b),
        "efg_diff":         efg_a - efg_b,
        "def_efg_diff":     defg_a - defg_b,
        "tov_matchup_a":    tov_b - tov_a,
        "reb_matchup_diff": orb_a - orb_b,
        "ft_edge_diff":     (ftp_a * ftr_a) - (ftp_b * ftr_b),
        "adj_off_diff":     aoa - aob,
        "adj_def_diff":     ada - adb,
        # Volatility
        "game_chaos":       game_chaos,
        "var_score_a":      vs_a,
        "upset_potential":  game_chaos / (1 + abs(pow_diff)),
        "tempo_abs_diff":   abs(tmp_a - tmp_b),
        "three_pt_rate_sum":tpa_a + tpa_b,
        # Context
        "sos_diff":         sos_a - sos_b,
        "conf_power_diff":  cp_a - cp_b,
        "adj_em_sos_diff":  em_a * sc_a - em_b * sc_b,
        "mid_major_mismatch": abs(mm_a - mm_b),
        "experience_diff":  exp_a - exp_b,
        # Calibration
        "seed_prior_a":     sp_a,
        "ensemble_spread":  float(ensemble_spread),
        "elo_kenpom_disagree": abs(elo_prob - kp_prob),
        # Style
        "style_clash": int(
            (sty_a == "bomb_heavy"   and sty_b == "slow_elite_d") or
            (sty_b == "bomb_heavy"   and sty_a == "slow_elite_d")
        ),
        "double_chaos": int(sty_a == "chaos_agent" and sty_b == "chaos_agent"),
        # Market (needed for meta-model blend)
        "market_prob_a":    mkt_prob_a,
    }
    return row


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CalibratedGameModel:
    """
    The single, clean game model interface.
    Training pipeline calls .fit() once.
    Simulation engine calls .predict() per matchup.

    Separation of concerns:
      - This class knows about ML; simulation.py does NOT import sklearn/xgb/lgb
      - simulation.py only sees: predict(team_a_dict, team_b_dict) → float
    """

    def __init__(self):
        self.model: Optional[ModelStack] = None
        self.is_fitted = False
        self._cache: Dict[Tuple[str,str], float] = {}

    def fit(self, df_sym=None) -> "CalibratedGameModel":
        """Train on historical symmetrized data."""
        if df_sym is None:
            df_sym = build_features(load_symmetrized())
        self.model = ModelStack()
        self.model.fit(df_sym)
        self.is_fitted = True
        self._cache.clear()
        return self

    def load(self, path: str = MODEL_PATH) -> "CalibratedGameModel":
        """Load pre-trained model from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model artifact not found: {path}")

        expected_sha256 = os.getenv("MODEL_SHA256", "").strip().lower()
        if expected_sha256:
            actual_sha256 = _sha256_file(path)
            if actual_sha256 != expected_sha256:
                raise RuntimeError("Model artifact checksum mismatch")

        self.model = joblib.load(path)
        self.is_fitted = True
        self._cache.clear()
        return self

    def predict(self, a: Dict, b: Dict,
                latent_a: float = 0.0, latent_b: float = 0.0) -> float:
        """
        Return calibrated P(team_a wins).

        latent_a/latent_b: per-sim latent strength draws from simulation engine.
        When non-zero, these perturb the power ratings before prediction,
        creating the correlation structure described in Phase 6.
        """
        name_a = a.get("name", id(a))
        name_b = b.get("name", id(b))
        cache_key = (str(name_a), str(name_b), round(latent_a, 3), round(latent_b, 3))

        if cache_key in self._cache:
            return self._cache[cache_key]

        row = build_matchup_row(a, b)

        # Apply latent perturbation to key differential features
        if latent_a != 0.0 or latent_b != 0.0:
            delta = latent_a - latent_b
            row["em_diff"]        += delta * 4.0   # scale to EM units
            row["elo_diff"]       += delta * 80.0  # scale to Elo units
            row["power_diff"]     += delta
            row["em_diff_sq"]      = row["em_diff"] ** 2
            row["power_diff_sq"]   = row["power_diff"] ** 2
            row["em_diff_cap"]     = float(np.clip(row["em_diff"], -25, 25))
            row["poss_match_diff"]+= delta * 3.0

        import pandas as pd
        df_row = pd.DataFrame([row])

        if self.is_fitted and self.model is not None:
            try:
                p = float(self.model.predict_proba(df_row)[0])
            except Exception:
                p = self._fallback(row)
        else:
            p = self._fallback(row)

        p = float(np.clip(p, 0.02, 0.98))
        self._cache[cache_key] = p
        return p

    def _fallback(self, row: Dict) -> float:
        """Blend baseline when model unavailable."""
        kp  = float(expit(row.get("em_diff", 0) * 0.178))
        elo = float(expit(row.get("elo_diff", 0) / 400 * np.log(10)))
        mkt = float(row.get("market_prob_a", 0.5))
        sp  = float(row.get("seed_prior_a", 0.5))
        return 0.35*kp + 0.25*elo + 0.25*mkt + 0.15*sp

    def clear_cache(self):
        self._cache.clear()

    def predict_with_breakdown(self, a: Dict, b: Dict) -> Dict:
        """Full breakdown of what each signal contributes."""
        row    = build_matchup_row(a, b)
        import pandas as pd
        df_row = pd.DataFrame([row])

        # Individual signal contributions
        em_prob  = float(expit(row["em_diff"] * 0.178))
        elo_prob = float(expit(row["elo_diff"] / 400 * np.log(10)))
        mkt_prob = float(row["market_prob_a"])
        sp_prob  = float(row["seed_prior_a"])
        poss_contrib = float(expit(row["poss_match_diff"] * 0.04))

        full_prob = self.predict(a, b)

        # Signal-level breakdown (as % advantage for team_a)
        breakdown = {
            "base_power":   round(em_prob * 100, 1),
            "elo":          round(elo_prob * 100, 1),
            "possession":   round(poss_contrib * 100, 1),
            "efficiency":   round(float(expit(row["efg_net_diff"] * 5)) * 100, 1),
            "rebounding":   round(float(expit(row["reb_matchup_diff"] * 8)) * 100, 1),
            "turnovers":    round(float(expit(row["tov_matchup_a"] * 6)) * 100, 1),
            "free_throws":  round(float(expit(row["ft_edge_diff"] * 10)) * 100, 1),
            "experience":   round(float(expit(-row["experience_diff"] * 0.8)) * 100, 1),
            "coaching":     50.0,
            "market":       round(mkt_prob * 100, 1),
            "pythagorean":  round(float(expit(row["pyth_diff"] * 4)) * 100, 1),
        }

        return {
            "win_prob_a":      round(full_prob * 100, 1),
            "win_prob_b":      round((1 - full_prob) * 100, 1),
            "expected_margin": round(abs(row["poss_match_diff"]) * 0.4, 1),
            "upset":           a.get("seed", 1) > b.get("seed", 1) and full_prob > 0.5,
            "volatility":      round(row["game_chaos"] * 100, 1),
            "breakdown":       breakdown,
            "model_confidence":round((1 - row["ensemble_spread"]) * (1 - row["game_chaos"]) * 100, 1),
        }


# ── Singleton for use by simulation engine ────────────────────────────────────

_GAME_MODEL: Optional[CalibratedGameModel] = None


def get_game_model(force_retrain: bool = False) -> CalibratedGameModel:
    """
    Get or initialize the global game model.
    Loads from disk if available; trains fresh if not.
    """
    global _GAME_MODEL
    if _GAME_MODEL is None or force_retrain:
        _GAME_MODEL = CalibratedGameModel()
        if force_retrain:
            logger.warning("Force retrain requested; loading artifact after cache reset")
        _GAME_MODEL.load(MODEL_PATH)
    return _GAME_MODEL


if __name__ == "__main__":
    # Smoke test
    from data.teams_2026 import TEAMS_2026
    model = get_game_model()

    pairs = [
        ("Duke", "Siena"),
        ("Arizona", "Michigan"),
        ("UMBC", "Virginia"),    # historic upset: UMBC was 16 seed
        ("Yale", "Auburn"),      # 2024 upset
    ]

    print(f"{'Matchup':<35} {'P(A)':>6} {'P(B)':>6} {'Upset':>6} {'Conf':>6}")
    print("-" * 62)
    for na, nb in pairs:
        if na in TEAMS_2026 and nb in TEAMS_2026:
            ta = {**TEAMS_2026[na], "name": na}
            tb = {**TEAMS_2026[nb], "name": nb}
            r = model.predict_with_breakdown(ta, tb)
            print(f"{na:15s} vs {nb:15s}"
                  f" {r['win_prob_a']:>5.1f}%"
                  f" {r['win_prob_b']:>5.1f}%"
                  f" {'YES' if r['upset'] else 'no':>6}"
                  f" {r['model_confidence']:>5.1f}%")
        else:
            # Use mock data for historic matchups
            print(f"{na:15s} vs {nb:15s}  (not in 2026 bracket)")
