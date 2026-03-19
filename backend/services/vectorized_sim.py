"""
Vectorized bracket simulation — pre-computes pairwise probs, runs all sims in parallel.

Replaces the per-sim Python loop with NumPy array operations for ~10–50x speedup.
"""
import numpy as np
from scipy.special import expit
from typing import Dict, List, Optional, Tuple

from services.simulation import (
    Team, SimulationConfig, REGIONS, R64_ORDER,
    _team_to_dict, _fallback_prob, _get_game_model,
)

SHRINK_ALPHA = 0.78
LATENT_SCALE = 4.5


def precompute_pairwise_probs(
    teams: Dict[str, Team],
    team_names: List[str],
    cfg: SimulationConfig,
) -> np.ndarray:
    """
    Pre-compute all pairwise win probabilities (team_a beats team_b).
    Returns (n_teams, n_teams) matrix with shrunk base probs.
    ML model is called once per unique pair — no inference in sim loop.
    """
    n = len(team_names)
    P = np.full((n, n), 0.5, dtype=np.float64)

    gm = _get_game_model()
    for i in range(n):
        for j in range(i + 1, n):
            a, b = teams[team_names[i]], teams[team_names[j]]
            ov_a = cfg.team_overrides.get(a.name, 0.0)
            ov_b = cfg.team_overrides.get(b.name, 0.0)
            da = _team_to_dict(a, ov_a)
            db = _team_to_dict(b, ov_b)

            if gm is not None:
                try:
                    base = gm.predict(da, db, latent_a=0.0, latent_b=0.0)
                except (AttributeError, OSError, RuntimeError, ValueError):
                    base = _fallback_prob(a, b, cfg)
            else:
                base = _fallback_prob(a, b, cfg)

            shrunk = 0.5 + (base - 0.5) * SHRINK_ALPHA
            p_ij = np.clip(shrunk, 0.02, 0.98)
            P[i, j] = p_ij
            P[j, i] = 1.0 - p_ij

    return P


def _apply_latent_to_probs(
    base_probs: np.ndarray,
    latent_a: np.ndarray,
    latent_b: np.ndarray,
) -> np.ndarray:
    """Apply latent perturbation as logit shift (vectorized)."""
    delta = latent_a - latent_b
    p = np.clip(base_probs, 0.02, 0.98)
    logit = np.log(p / (1.0 - p))
    return np.clip(expit(logit + delta * LATENT_SCALE), 0.02, 0.98)


def _latent_sigma_per_team(team: Team, base_sigma: float) -> float:
    return base_sigma * (1 + 0.3 * team.seed / 16)


def _build_region_bracket_indices(
    teams: Dict[str, Team],
    team_names: List[str],
    name_to_idx: Dict[str, int],
    region: str,
) -> np.ndarray:
    seed_to_idx = {}
    for name, t in teams.items():
        if t.region == region:
            seed_to_idx[t.seed] = name_to_idx[name]
    bracket = []
    for s1, s2 in R64_ORDER:
        if s1 in seed_to_idx:
            bracket.append(seed_to_idx[s1])
        if s2 in seed_to_idx:
            bracket.append(seed_to_idx[s2])
    return np.array(bracket, dtype=np.int32)


def _simulate_region_round(
    cur: np.ndarray,
    prob_matrix: np.ndarray,
    latent: np.ndarray,
    rng: np.random.Generator,
    forced: Dict[str, str],
    region: str,
    round_i: int,
    name_to_idx: Dict[str, int],
    n_sims: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate one round in a region. Returns (winners, round_winners for counting)."""
    n_games = cur.shape[1] // 2

    def _forced_key(r: str, ri: int, gi: int) -> str:
        return f"{r}:{ri}:{gi}"

    winners_team = np.zeros((n_sims, n_games), dtype=np.int32)
    for g in range(n_games):
        idx_a, idx_b = 2 * g, 2 * g + 1
        team_a = cur[:, idx_a]
        team_b = cur[:, idx_b]
        base_p = prob_matrix[team_a, team_b]
        la = latent[np.arange(n_sims), team_a]
        lb = latent[np.arange(n_sims), team_b]
        probs = _apply_latent_to_probs(base_p, la, lb)
        forced_w = forced.get(_forced_key(region, round_i, g))
        if forced_w is not None:
            idx_f = name_to_idx.get(forced_w, -1)
            if idx_f >= 0:
                mask_a = (team_a == idx_f)
                mask_b = (team_b == idx_f)
                rand_w = (rng.random(n_sims) < probs).astype(np.int32)  # 1=team_a
                winners = np.where(mask_a, 1, np.where(mask_b, 0, rand_w))
            else:
                winners = (rng.random(n_sims) < probs).astype(np.int32)
        else:
            winners = (rng.random(n_sims) < probs).astype(np.int32)  # 1 = team_a wins
        winners_team[:, g] = np.where(winners == 1, team_a, team_b)  # 1=team_a wins

    return winners_team, winners_team


def run_vectorized_simulation(
    teams: Dict[str, Team],
    team_names: List[str],
    prob_matrix: np.ndarray,
    n_sims: int,
    latent_sigma: float,
    rng: np.random.Generator,
    forced_picks: Optional[Dict[str, str]] = None,
    first_four_games: Optional[List[Dict]] = None,
    slot_to_default: Optional[Dict[str, str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Run n_sims tournaments in parallel using vectorized NumPy ops.

    Returns:
        counts: dict with keys won_r64, won_r32, won_s16, won_e8, won_f4, won_ncg, in_title, won_ff
        Each value is (n_sims, n_teams) array of counts.
    """
    forced = forced_picks or {}
    has_ff = bool(first_four_games)
    n = len(team_names)
    name_to_idx = {name: i for i, name in enumerate(team_names)}

    sigma_per_team = np.array([_latent_sigma_per_team(teams[name], latent_sigma) for name in team_names])
    latent = rng.normal(0, sigma_per_team, size=(n_sims, n))

    won_r64 = np.zeros((n_sims, n), dtype=np.int32)
    won_r32 = np.zeros((n_sims, n), dtype=np.int32)
    won_s16 = np.zeros((n_sims, n), dtype=np.int32)
    won_e8 = np.zeros((n_sims, n), dtype=np.int32)
    won_f4 = np.zeros((n_sims, n), dtype=np.int32)
    won_ncg = np.zeros((n_sims, n), dtype=np.int32)
    in_title = np.zeros((n_sims, n), dtype=np.int32)
    won_ff = np.zeros((n_sims, n), dtype=np.int32)

    ff_winners: Dict[str, np.ndarray] = {}
    if has_ff and first_four_games:
        for gi, game in enumerate(first_four_games):
            slot = game["slot"]
            ia = name_to_idx.get(game["team_a"], -1)
            ib = name_to_idx.get(game["team_b"], -1)
            if ia < 0 or ib < 0:
                continue
            base_p = prob_matrix[ia, ib]
            la = latent[:, ia]
            lb = latent[:, ib]
            probs = _apply_latent_to_probs(np.full(n_sims, base_p), la, lb)
            forced_w = forced.get(f"{game.get('region', 'FF')}:0:{gi}")
            if forced_w == team_names[ia]:
                winners = np.ones(n_sims, dtype=np.int32)   # 1 = team_a wins
            elif forced_w == team_names[ib]:
                winners = np.zeros(n_sims, dtype=np.int32)
            else:
                winners = (rng.random(n_sims) < probs).astype(np.int32)
            ff_winners[slot] = np.where(winners == 1, ia, ib)
            for s in range(n_sims):
                won_ff[s, ff_winners[slot][s]] += 1

    ff_slot_map: Dict[str, Tuple[str, int]] = {}
    if has_ff and first_four_games:
        for game in first_four_games:
            slot = game["slot"]
            region = game["region"]
            seed = game["seed"]
            for idx, (s1, s2) in enumerate(R64_ORDER):
                if seed == s1:
                    ff_slot_map[slot] = (region, 2 * idx)
                    break
                if seed == s2:
                    ff_slot_map[slot] = (region, 2 * idx + 1)
                    break

    region_brackets: Dict[str, np.ndarray] = {}
    for region in REGIONS:
        br = _build_region_bracket_indices(teams, team_names, name_to_idx, region)
        region_brackets[region] = br

    if ff_winners and ff_slot_map:
        for slot, (region, pos) in ff_slot_map.items():
            if slot in ff_winners:
                br = region_brackets[region]
                br_expanded = np.broadcast_to(br, (n_sims, 16)).copy()
                br_expanded[:, pos] = ff_winners[slot]
                region_brackets[region] = br_expanded

    region_champs_all = np.zeros((n_sims, 4), dtype=np.int32)
    for ri, region in enumerate(REGIONS):
        br = region_brackets[region]
        if br.ndim == 1:
            cur = np.broadcast_to(br, (n_sims, 16)).copy()
        else:
            cur = br.copy()

        for round_i in range(4):
            n_games = 8 >> round_i
            winners_team, _ = _simulate_region_round(
                cur, prob_matrix, latent, rng, forced, region, round_i,
                name_to_idx, n_sims,
            )
            for s in range(n_sims):
                for g in range(n_games):
                    w = winners_team[s, g]
                    if round_i == 0:
                        won_r64[s, w] += 1
                    elif round_i == 1:
                        won_r32[s, w] += 1
                    elif round_i == 2:
                        won_s16[s, w] += 1
                    else:
                        won_e8[s, w] += 1
            if n_games == 1:
                region_champs_all[:, ri] = winners_team[:, 0]
                break
            cur = winners_team

    # REGIONS = ["East","West","Midwest","South"] → idx 0=East, 1=West, 2=Midwest, 3=South
    # SF1: East vs Midwest; SF2: West vs South
    e_idx, w_idx, m_idx, s_idx = 0, 1, 2, 3
    sf1_a = region_champs_all[:, e_idx]   # East
    sf1_b = region_champs_all[:, m_idx]  # Midwest
    sf2_a = region_champs_all[:, w_idx]  # West
    sf2_b = region_champs_all[:, s_idx]  # South

    for s in range(n_sims):
        won_f4[s, region_champs_all[s, e_idx]] += 1
        won_f4[s, region_champs_all[s, m_idx]] += 1
        won_f4[s, region_champs_all[s, w_idx]] += 1
        won_f4[s, region_champs_all[s, s_idx]] += 1

    base_p1 = prob_matrix[sf1_a, sf1_b]
    probs1 = _apply_latent_to_probs(
        base_p1,
        latent[np.arange(n_sims), sf1_a],
        latent[np.arange(n_sims), sf1_b],
    )
    forced_sf1 = forced.get("FinalFour:0:0")
    if forced_sf1 is not None:
        idx_f = name_to_idx.get(forced_sf1, -1)
        if idx_f >= 0:
            mask_a = (sf1_a == idx_f)
            mask_b = (sf1_b == idx_f)
            rand1 = (rng.random(n_sims) < probs1).astype(np.int32)
            winners_sf1 = np.where(mask_a, 1, np.where(mask_b, 0, rand1))
        else:
            winners_sf1 = (rng.random(n_sims) < probs1).astype(np.int32)
    else:
        winners_sf1 = (rng.random(n_sims) < probs1).astype(np.int32)
    sf1_winner = np.where(winners_sf1 == 1, sf1_a, sf1_b)

    base_p2 = prob_matrix[sf2_a, sf2_b]
    probs2 = _apply_latent_to_probs(
        base_p2,
        latent[np.arange(n_sims), sf2_a],
        latent[np.arange(n_sims), sf2_b],
    )
    forced_sf2 = forced.get("FinalFour:0:1")
    if forced_sf2 is not None:
        idx_f = name_to_idx.get(forced_sf2, -1)
        if idx_f >= 0:
            mask_a = (sf2_a == idx_f)
            mask_b = (sf2_b == idx_f)
            rand2 = (rng.random(n_sims) < probs2).astype(np.int32)
            winners_sf2 = np.where(mask_a, 1, np.where(mask_b, 0, rand2))
        else:
            winners_sf2 = (rng.random(n_sims) < probs2).astype(np.int32)
    else:
        winners_sf2 = (rng.random(n_sims) < probs2).astype(np.int32)
    sf2_winner = np.where(winners_sf2 == 1, sf2_a, sf2_b)

    in_title[np.arange(n_sims), sf1_winner] += 1
    in_title[np.arange(n_sims), sf2_winner] += 1

    base_p3 = prob_matrix[sf1_winner, sf2_winner]
    probs3 = _apply_latent_to_probs(
        base_p3,
        latent[np.arange(n_sims), sf1_winner],
        latent[np.arange(n_sims), sf2_winner],
    )
    forced_final = forced.get("FinalFour:1:0")
    if forced_final is not None:
        idx_f = name_to_idx.get(forced_final, -1)
        if idx_f >= 0:
            mask_f1 = (sf1_winner == idx_f)
            mask_f2 = (sf2_winner == idx_f)
            rand3 = (rng.random(n_sims) < probs3).astype(np.int32)
            winners_final = np.where(mask_f1, 1, np.where(mask_f2, 0, rand3))
        else:
            winners_final = (rng.random(n_sims) < probs3).astype(np.int32)
    else:
        winners_final = (rng.random(n_sims) < probs3).astype(np.int32)
    champion = np.where(winners_final == 1, sf1_winner, sf2_winner)
    won_ncg[np.arange(n_sims), champion] += 1

    return {
        "won_r64": won_r64,
        "won_r32": won_r32,
        "won_s16": won_s16,
        "won_e8": won_e8,
        "won_f4": won_f4,
        "won_ncg": won_ncg,
        "in_title": in_title,
        "won_ff": won_ff,
    }
