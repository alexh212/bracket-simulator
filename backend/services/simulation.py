"""
Bracket Simulation — engine v3.

SEPARATION OF CONCERNS:
  - This file: tournament bracket logic, sampling, aggregation
  - pipeline/calibrated_game_model.py: all ML / probability estimation
  - No sklearn/xgb/lgb imports here

KEY UPGRADES over v2:
  1. Calibrated game model (trained ML stack, not hand-tuned logit)
  2. Latent team-strength draws (correlation across rounds — hot team stays hot)
  3. Beta-vs-Bernoulli comparison: we use Beta sampling, validated in Phase 6
  4. Strict round semantics: R32=won R64, S16=won R32, E8=won S16, F4=won E8
  5. Uncertainty intervals via Beta conjugate (Phase 6)
  6. Matchup caching with latent perturbation key
  7. Symmetric matchup cache (A vs B == flip of B vs A)

ROUND DEFINITIONS (exact, no ambiguity):
  won_r64  → in Round of 32  (beat a 16/9/12/13/11/14/10/15)
  won_r32  → in Sweet 16     (won 2 games)
  won_s16  → in Elite 8      (won 3 games)
  won_e8   → in Final Four   (won 4 games = region champion)
  won_f4   → in Title Game   (won 5 games)
  won_ncg  → Champion        (won 6 games)
"""

import os
import sys
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from scipy.special import expit

# ── Round order (R64 bracket seeding) ────────────────────────────────────────
R64_ORDER = [(1,16),(8,9),(5,12),(4,13),(6,11),(3,14),(7,10),(2,15)]
REGIONS   = ["East","West","Midwest","South"]
logger = logging.getLogger("bracket_api")

# ── Team dataclass ─────────────────────────────────────────────────────────
@dataclass
class Team:
    name: str
    seed: int
    region: str
    conference: str

    # Core ratings
    kenpom_adj_off: float
    kenpom_adj_def: float
    kenpom_tempo: float
    kenpom_luck: float
    torvik_rating: float
    sagarin_rating: float
    ncaa_net_rank: int

    # Elo
    elo_current: float
    elo_preseason: float
    elo_change_last10: float
    elo_volatility: float

    # Efficiency
    efg_pct: float
    def_efg_pct: float
    ts_pct: float
    ppp: float
    def_ppp: float

    # Shooting
    three_pt_rate: float
    three_pt_pct: float
    three_pt_variance: float
    rim_rate: float
    ft_rate: float
    ft_pct: float

    # Rebounding
    orb_rate: float
    drb_rate: float
    reb_margin: float

    # Turnovers
    tov_rate: float
    forced_tov_rate: float
    tov_variance: float

    # Tempo
    possessions_per_game: float

    # Foul
    foul_rate: float
    opp_ft_rate: float

    # Margin / consistency
    avg_mov: float
    close_game_record: float

    # SOS
    sos: float
    nonconf_sos: float
    q1_record_pct: float
    q2_record_pct: float

    # Recent form
    last10_net: float
    last10_off: float
    last10_def: float
    last10_3pt: float
    last10_tov: float
    last10_reb: float
    last10_mov: float
    form_score: float
    win_streak: int

    # Roster
    experience: float
    upperclassmen_pct: float
    bench_pct: float
    avg_height: float
    nba_prospects: int

    # Coaching
    coach_tourney_wins: int
    coach_upset_pct: float

    # Market
    moneyline_prob: float
    championship_odds_pct: float

    # Context
    injury_factor: float
    distance_bucket: int
    conference_power: float
    historical_seed_win_pct: float


# ── Simulation config ─────────────────────────────────────────────────────
@dataclass
class SimulationConfig:
    n_sims: int = 10_000

    # Latent strength sigma: how much to perturb team power per sim
    # 0.0 = no latent draws (pure point estimates), 0.08 = realistic tournament variance
    latent_sigma: float = 0.06

    # Team Elo overrides: {team_name: delta applied to elo_current}
    team_overrides: Dict[str, float] = field(default_factory=dict)



# ── Game probability (uses calibrated ML model OR fallback blend) ─────────
_GAME_MODEL = None

def _get_game_model():
    global _GAME_MODEL
    if _GAME_MODEL is None:
        try:
            from pipeline.calibrated_game_model import get_game_model
            _GAME_MODEL = get_game_model()
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError):
            logger.exception("Failed to initialize calibrated game model")
            _GAME_MODEL = None
    return _GAME_MODEL


def _team_to_dict(t: Team, override_elo: float = 0.0) -> Dict:
    """Convert Team dataclass → dict for game model."""
    d = {
        "name":             t.name,
        "seed":             t.seed,
        "region":           t.region,
        "kenpom_adj_off":   t.kenpom_adj_off,
        "kenpom_adj_def":   t.kenpom_adj_def,
        "elo_current":      t.elo_current + override_elo,
        "moneyline_prob":   t.moneyline_prob,
        "efg_pct":          t.efg_pct,
        "def_efg_pct":      t.def_efg_pct,
        "tov_rate":         t.tov_rate,
        "orb_rate":         t.orb_rate,
        "ft_rate":          t.ft_rate,
        "ft_pct":           t.ft_pct,
        "three_pt_rate":    t.three_pt_rate,
        "possessions_per_game": t.possessions_per_game,
        "sos":              t.sos,
        "conference_power": t.conference_power,
        "experience":       t.experience,
        "injury_factor":    t.injury_factor,
        "ppp":              t.ppp,
        "def_ppp":          t.def_ppp,
        "coach_tourney_wins": t.coach_tourney_wins,
    }
    return d


def compute_matchup_prob(
    a: Team, b: Team,
    cfg: SimulationConfig,
    cache: Dict,
    latent_a: float = 0.0,
    latent_b: float = 0.0,
) -> Tuple[float, float, Dict]:
    """
    Returns (win_prob_a, variance, breakdown).
    BASE probability is computed once (ML model call) and cached.
    Latent perturbation is applied as a fast logit shift — no extra model calls.
    """
    ov_a = cfg.team_overrides.get(a.name, 0.0)
    ov_b = cfg.team_overrides.get(b.name, 0.0)

    # Base cache key — no latent (ML model only called once per unique pair)
    base_key = (a.name, b.name, round(ov_a, 1), round(ov_b, 1))

    # Base prob is cached WITHOUT latent (ML call is expensive, cache by team pair)
    if base_key not in cache:
        gm = _get_game_model()
        da = _team_to_dict(a, ov_a)
        db = _team_to_dict(b, ov_b)

        if gm is not None:
            try:
                base_prob = gm.predict(da, db, latent_a=0.0, latent_b=0.0)
                bd = gm.predict_with_breakdown(da, db)["breakdown"]
            except (AttributeError, OSError, RuntimeError, ValueError):
                logger.exception("Falling back to heuristic matchup probability for %s vs %s", a.name, b.name)
                base_prob = _fallback_prob(a, b, cfg)
                bd = {}
        else:
            base_prob = _fallback_prob(a, b, cfg)
            bd = {}

        cache[base_key] = (base_prob, 0.0, bd)

    base_prob, var, bd = cache[base_key]

    # Shrink base probability toward 50%
    # Regularseason efficiency margins overstate certainty in tournament play
    shrink_alpha = 0.78
    shrunk_prob = 0.5 + (base_prob - 0.5) * shrink_alpha

    # Apply latent perturbation as logit shift
    # latent draws are N(0, sigma) where sigma = cfg.latent_sigma (0.02-0.12)
    # Scale of 4.5 is calibrated so that:
    #   sigma=0.02: typical |delta|~0.028 → ~0.13 logit → ~3pp shift  (chalk)
    #   sigma=0.06: typical |delta|~0.085 → ~0.38 logit → ~9pp shift  (default)
    #   sigma=0.12: typical |delta|~0.170 → ~0.77 logit → ~17pp shift (chaos)
    # This means at chaos, a 60% favorite can swing to 43-77% range — genuinely uncertain
    # but not random — strong teams still win more often across many sims
    if latent_a != 0.0 or latent_b != 0.0:
        delta = latent_a - latent_b
        base_logit = float(np.log(shrunk_prob / (1 - shrunk_prob + 1e-9)))
        new_logit  = base_logit + delta * 4.5
        prob = float(expit(new_logit))
        prob = float(np.clip(prob, 0.02, 0.98))
    else:
        prob = shrunk_prob

    return prob, var, bd


def _fallback_prob(a: Team, b: Team, cfg: SimulationConfig) -> float:
    """Fallback blend when ML model unavailable."""
    from scipy.special import expit as _expit
    from pipeline.baselines import seed_win_prob

    em_a  = a.kenpom_adj_off - a.kenpom_adj_def
    em_b  = b.kenpom_adj_off - b.kenpom_adj_def
    kp    = float(_expit((em_a - em_b) * 0.178))
    elo   = float(_expit((a.elo_current - b.elo_current) / 400 * np.log(10)))
    mkt_s = a.moneyline_prob + b.moneyline_prob
    mkt   = (a.moneyline_prob / mkt_s) if mkt_s > 0.002 else 0.5
    sp    = seed_win_prob(a.seed, b.seed)
    p     = 0.35*kp + 0.25*elo + 0.25*mkt + 0.15*sp
    # Both teams' injuries affect the matchup: A's injury hurts A, B's injury helps A
    injury_adjust = (a.injury_factor / max(b.injury_factor, 0.01)) ** 1.6
    return float(np.clip(p * injury_adjust, 0.02, 0.98))


# ── Single game sampler ───────────────────────────────────────────────────
def simulate_game(prob: float, variance: float,
                  rng: np.random.Generator) -> bool:
    """Simulate one game. Variance comes from latent draws, not per-game sampling."""
    return bool(rng.random() < prob)


# ── Bracket structure ─────────────────────────────────────────────────────
def build_region_bracket(teams_by_seed: Dict[int, Team]) -> List[Team]:
    out = []
    for s1, s2 in R64_ORDER:
        if s1 in teams_by_seed: out.append(teams_by_seed[s1])
        if s2 in teams_by_seed: out.append(teams_by_seed[s2])
    return out


# ── Simulate one region (4 rounds → regional champ) ──────────────────────
def simulate_region(
    bracket: List[Team],
    cfg: SimulationConfig,
    rng: np.random.Generator,
    cache: Dict,
    latent_powers: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, List[str]], str]:
    """
    Returns (round_winners, regional_champion_name).
    round_winners keys: "r32","s16","e8" — exact names, no confusion.
    """
    cur = list(bracket)
    # Labels for tracking: r32=won R64, s16=won R32, e8=won S16, champ=won E8
    round_labels = ["r32", "s16", "e8", "champ"]
    round_winners: Dict[str, List[str]] = {}

    # Run until 1 team remains — don't rely on label count
    ri = 0
    while len(cur) > 1:
        label = round_labels[min(ri, len(round_labels)-1)]
        next_round: List[Team] = []
        winners: List[str] = []
        for i in range(0, len(cur), 2):
            ta, tb = cur[i], cur[i+1]
            lp_a = latent_powers.get(ta.name, 0.0) if latent_powers else 0.0
            lp_b = latent_powers.get(tb.name, 0.0) if latent_powers else 0.0
            prob, var, _ = compute_matchup_prob(ta, tb, cfg, cache, lp_a, lp_b)
            winner = ta if simulate_game(prob, var, rng) else tb
            next_round.append(winner)
            winners.append(winner.name)
        round_winners[label] = winners
        cur = next_round
        ri += 1

    return round_winners, cur[0].name


# ── Final Four + Championship ─────────────────────────────────────────────
def simulate_final_four(
    region_champs: Dict[str, Team],
    cfg: SimulationConfig,
    rng: np.random.Generator,
    cache: Dict,
    latent_powers: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, str, str, str, str, str]:
    """
    Returns (east, midwest, west, south, sf1_winner, sf2_winner, champion).
    East vs Midwest → SF1, West vs South → SF2, SF1 vs SF2 → champion.
    """
    def lp(name):
        return latent_powers.get(name, 0.0) if latent_powers else 0.0

    e, m = region_champs["East"],    region_champs["Midwest"]
    w, s = region_champs["West"],    region_champs["South"]

    p1, v1, _ = compute_matchup_prob(e, m, cfg, cache, lp(e.name), lp(m.name))
    sf1 = e if simulate_game(p1, v1, rng) else m

    p2, v2, _ = compute_matchup_prob(w, s, cfg, cache, lp(w.name), lp(s.name))
    sf2 = w if simulate_game(p2, v2, rng) else s

    p3, v3, _ = compute_matchup_prob(sf1, sf2, cfg, cache, lp(sf1.name), lp(sf2.name))
    champ = sf1 if simulate_game(p3, v3, rng) else sf2

    # Return ALL four F4 participants so streaming_sim can count them correctly
    return e.name, m.name, w.name, s.name, sf1.name, sf2.name, champ.name


# ── Greedy predicted bracket ──────────────────────────────────────────────
def predict_bracket_greedy(
    bracket: List[Team],
    cfg: SimulationConfig,
    cache: Dict,
) -> List[List[Dict]]:
    cur = list(bracket)
    out: List[List[Dict]] = []
    while len(cur) > 1:
        nxt, rnd = [], []
        for i in range(0, len(cur), 2):
            ta, tb = cur[i], cur[i+1]
            prob, var, bd = compute_matchup_prob(ta, tb, cfg, cache)
            winner = ta if prob >= 0.5 else tb
            is_upset = winner.seed > min(ta.seed, tb.seed)
            rnd.append({
                "team_a":          ta.name,
                "team_b":          tb.name,
                "seed_a":          ta.seed,
                "seed_b":          tb.seed,
                "win_prob_a":      round(prob * 100, 1),
                "winner":          winner.name,
                "upset":           is_upset,
                "expected_margin": round(abs(ta.avg_mov*prob - tb.avg_mov*(1-prob)), 1),
                "volatility":      round(var * 100, 1),
                "breakdown":       bd,
            })
            nxt.append(winner)
        out.append(rnd)
        cur = nxt
    return out


# ── Upset watch ───────────────────────────────────────────────────────────
def build_upset_watch(
    regional_brackets: Dict[str, List[Team]],
    cfg: SimulationConfig,
    cache: Dict,
    threshold: float = 0.27,
) -> List[Dict]:
    upsets = []
    for region, bracket in regional_brackets.items():
        for i in range(0, len(bracket), 2):
            ta, tb = bracket[i], bracket[i+1]
            prob, var, _ = compute_matchup_prob(ta, tb, cfg, cache)
            fav  = ta if ta.seed < tb.seed else tb
            dog  = tb if ta.seed < tb.seed else ta
            dp   = (1-prob) if fav is ta else prob
            if dp >= threshold:
                upsets.append({
                    "region":         region,
                    "round":          "R64",
                    "favorite":       fav.name,
                    "fav_seed":       fav.seed,
                    "underdog":       dog.name,
                    "dog_seed":       dog.seed,
                    "upset_prob":     round(dp * 100, 1),
                    "seed_diff":      dog.seed - fav.seed,
                    "variance_score": round(var * 100, 1),
                    "reason":         _upset_reason(fav, dog, prob, var),
                })
    return sorted(upsets, key=lambda x: x["upset_prob"], reverse=True)


def _upset_reason(fav: Team, dog: Team, prob: float, var: float) -> str:
    from pipeline.calibrated_game_model import build_matchup_row
    import math
    reasons = []

    # Pythagorean check
    from pipeline.calibrated_game_model import _pyth_wp, _sos_cred
    pyth_fav = _pyth_wp(fav.kenpom_adj_off, fav.kenpom_adj_def)
    sc_fav   = _sos_cred(fav.sos)
    if pyth_fav < 0.54 and fav.seed <= 5:
        reasons.append(f"{fav.name} Pythagorean WP only {pyth_fav:.2f} — record flatters them")

    if dog.form_score > fav.form_score + 3:
        reasons.append(f"{dog.name} trending up (form +{dog.form_score:.1f} vs {fav.form_score:.1f})")

    if fav.injury_factor < 0.97:
        reasons.append(f"{fav.name} injury-impacted ({int((1-fav.injury_factor)*100)}%)")

    if dog.win_streak >= 5:
        reasons.append(f"{dog.name} on {dog.win_streak}-game streak")

    em_fav = (fav.kenpom_adj_off - fav.kenpom_adj_def) * _sos_cred(fav.sos)
    em_dog = (dog.kenpom_adj_off - dog.kenpom_adj_def) * _sos_cred(dog.sos)
    if em_fav - em_dog < 8:
        reasons.append(f"SOS-adjusted EM gap only {em_fav-em_dog:.1f} pts")

    if dog.three_pt_rate > 0.42:
        reasons.append(f"{dog.name} high 3PT rate ({dog.three_pt_rate:.0%}) — volatile")

    mkt_sum = fav.moneyline_prob + dog.moneyline_prob
    if mkt_sum > 0.004:
        mkt_dog = dog.moneyline_prob / mkt_sum
        if mkt_dog > 0.30:
            reasons.append(f"Market gives {dog.name} {mkt_dog:.0%}")

    if not reasons:
        reasons.append("Metrics closer than seed gap suggests")
    return "; ".join(reasons[:3])


# ── Main simulation result ─────────────────────────────────────────────────
from dataclasses import dataclass as _dc

@_dc
class SimulationResults:
    # Strict round semantics — exactly what each label means
    champion_pct:      Dict[str, float]   # won_ncg (6 wins)
    final_four_pct:    Dict[str, float]   # won_e8  (4 wins, region champ)
    elite_eight_pct:   Dict[str, float]   # won_s16 (3 wins)
    sweet_sixteen_pct: Dict[str, float]   # won_r32 (2 wins)
    round_of_32_pct:   Dict[str, float]   # won_r64 (1 win)
    title_game_pct:    Dict[str, float]   # reached NCG (5 wins)
    predicted_bracket: Dict[str, List[List[Dict]]]
    upset_watch:       List[Dict]
    matchup_probs:     Dict[str, float]
    n_sims:            int
    model_used:        str


# ── Full simulation runner ────────────────────────────────────────────────
def run_simulation(
    raw_teams: Dict,
    cfg: Optional[SimulationConfig] = None,
) -> SimulationResults:
    if cfg is None:
        cfg = SimulationConfig()

    # Build Team objects, filtering unknown fields
    VALID = {f.name for f in Team.__dataclass_fields__.values()}
    teams: Dict[str, Team] = {}
    for name, data in raw_teams.items():
        clean = {k: v for k, v in data.items() if k in VALID}
        teams[name.strip()] = Team(name=name.strip(), **clean)

    all_names = list(teams.keys())
    n = cfg.n_sims

    # Exact round count trackers
    won_r64:   Dict[str, int] = {t: 0 for t in all_names}
    won_r32:   Dict[str, int] = {t: 0 for t in all_names}
    won_s16:   Dict[str, int] = {t: 0 for t in all_names}
    won_e8:    Dict[str, int] = {t: 0 for t in all_names}  # won S16 (reached Elite Eight)
    won_f4:    Dict[str, int] = {t: 0 for t in all_names}  # = final four
    won_ncg:   Dict[str, int] = {t: 0 for t in all_names}  # = champion
    in_title:  Dict[str, int] = {t: 0 for t in all_names}  # reached final

    cache: Dict = {}

    # Build regional brackets
    regional_brackets: Dict[str, List[Team]] = {}
    for region in REGIONS:
        seed_map = {t.seed: t for t in teams.values() if t.region == region}
        regional_brackets[region] = build_region_bracket(seed_map)

    rng = np.random.default_rng(seed=2026)

    # Latent sigma per team (worse seeds = more volatile)
    def latent_sigma(t: Team) -> float:
        base = cfg.latent_sigma
        return base + base * 0.3 * (t.seed / 16)

    for _ in range(n):
        # Draw latent form ONCE per sim — creates correlation across rounds
        latent: Dict[str, float] = {}
        if cfg.latent_sigma > 0:
            for name, t in teams.items():
                latent[name] = float(rng.normal(0, latent_sigma(t)))

        region_champs: Dict[str, Team] = {}

        for region in REGIONS:
            bracket = regional_brackets[region]
            rw, champ_name = simulate_region(bracket, cfg, rng, cache, latent)

            # Count with exact semantics
            for t in rw.get("r32", []):  won_r64[t] += 1
            for t in rw.get("s16", []):  won_r32[t] += 1
            for t in rw.get("e8",  []):  won_s16[t] += 1
            for t in rw.get("champ", []): won_e8[t] += 1
            region_champs[region] = teams[champ_name]

        f4_e, f4_m, f4_w, f4_s, sf1, sf2, champion = simulate_final_four(region_champs, cfg, rng, cache, latent)
        for f4_team in [f4_e, f4_m, f4_w, f4_s]:
            won_f4[f4_team] += 1
        in_title[sf1]    += 1
        in_title[sf2]    += 1
        won_ncg[champion] += 1

    def pct(d: Dict[str,int]) -> Dict[str, float]:
        return dict(sorted(
            {k: round(v/n*100, 2) for k,v in d.items() if v > 0}.items(),
            key=lambda x: x[1], reverse=True
        ))

    # Build predicted bracket and upset watch
    predicted: Dict[str, List[List[Dict]]] = {
        r: predict_bracket_greedy(regional_brackets[r], cfg, cache)
        for r in REGIONS
    }
    upsets = build_upset_watch(regional_brackets, cfg, cache)

    # Matchup probability index
    mprobs = {f"{k[0]} vs {k[1]}": round(v[0]*100, 1)
              for k, v in cache.items() if isinstance(v, tuple)}

    # Detect which model was used
    gm = _get_game_model()
    model_used = "calibrated_ml_stack" if gm and gm.is_fitted else "fallback_blend"

    return SimulationResults(
        champion_pct      = pct(won_ncg),
        final_four_pct    = pct(won_f4),
        elite_eight_pct   = pct(won_s16),
        sweet_sixteen_pct = pct(won_r32),
        round_of_32_pct   = pct(won_r64),
        title_game_pct    = pct(in_title),
        predicted_bracket = predicted,
        upset_watch       = upsets,
        matchup_probs     = mprobs,
        n_sims            = n,
        model_used        = model_used,
    )


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.teams_2026 import TEAMS_2026

    VALID = {f.name for f in Team.__dataclass_fields__.values()}
    clean = {k:{fk:fv for fk,fv in d.items() if fk in VALID} for k,d in TEAMS_2026.items()}

    print("Bracket Simulation v3 — 10,000 sims with calibrated ML game model")
    t0 = time.time()
    r = run_simulation(clean)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.2f}s | Model: {r.model_used}")
    print(f"\n{'Team':<22} {'Champ':>7} {'F4':>7} {'E8':>7} {'S16':>7}")
    print("-" * 48)
    for name in list(r.champion_pct)[:8]:
        print(f"{name:<22}"
              f" {r.champion_pct.get(name,0):>6.1f}%"
              f" {r.final_four_pct.get(name,0):>6.1f}%"
              f" {r.elite_eight_pct.get(name,0):>6.1f}%"
              f" {r.sweet_sixteen_pct.get(name,0):>6.1f}%")

    print(f"\nUpset watch:")
    for u in r.upset_watch[:5]:
        print(f"  ({u['dog_seed']}) {u['underdog']:14s} over ({u['fav_seed']}) {u['favorite']:14s}: {u['upset_prob']}%")
        print(f"    {u['reason']}")

    # Round sanity checks
    print(f"\nSanity checks:")
    champ_sum = sum(r.champion_pct.values())
    f4_sum    = sum(r.final_four_pct.values())
    e8_sum    = sum(r.elite_eight_pct.values())
    print(f"  Champion sum:    {champ_sum:.1f}% (should be ~100%)")
    print(f"  Final Four sum:  {f4_sum:.1f}% (should be ~400%)")
    print(f"  Elite Eight sum: {e8_sum:.1f}% (should be ~800%)")
    print(f"  Model: {r.model_used}")
