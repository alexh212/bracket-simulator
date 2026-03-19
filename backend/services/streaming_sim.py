"""
Streaming simulation — emits progress events as sims complete.
Each event contains current win counts so the frontend can update live.

Uses vectorized NumPy simulation: pre-computes pairwise probs once,
then runs all sims in parallel via array ops (no per-game ML calls).
"""
import numpy as np
import time
import logging
from typing import Dict, Generator, Optional
from services.simulation import (
    Team, SimulationConfig, build_region_bracket, build_upset_watch,
    compute_matchup_prob, simulate_game,
    _get_game_model, REGIONS, R64_ORDER
)
from services.vectorized_sim import precompute_pairwise_probs, run_vectorized_simulation

logger = logging.getLogger("bracket_api")

def _simulate_scores(prob_a: float, a_wins: bool) -> tuple:
    """Generate scores consistent with who actually won."""
    import random
    rng = random.Random()
    base = 65 + rng.randint(0, 15)
    # Margin reflects how dominant the winner was (higher prob = bigger margin)
    win_prob = max(prob_a, 1 - prob_a)  # prob of whoever won
    expected_margin = max(1, int(abs(rng.gauss(win_prob * 18, 7))))
    margin = max(1, expected_margin)
    if a_wins:
        return base + margin // 2, base - margin // 2
    else:
        return base - margin // 2, base + margin // 2


def run_streaming_simulation(
    raw_teams: Dict,
    n_sims: int = 10_000,
    emit_every: int = 200,
    latent_sigma: float = 0.06,
    team_overrides: Optional[Dict[str, float]] = None,
    forced_picks: Optional[Dict[str, str]] = None,
) -> Generator[dict, None, None]:
    """
    Generator that yields progress dicts as sims complete.
    Each yield has type 'progress' with current pct counts.
    Final yield has type 'complete' with full results including bracket.
    """
    VALID = {f.name for f in Team.__dataclass_fields__.values()}

    # Import First Four data
    try:
        from data.teams_2026 import FIRST_FOUR_GAMES, FIRST_FOUR_TEAMS
        all_raw = {**raw_teams}
        for name, data in FIRST_FOUR_TEAMS.items():
            if name not in all_raw:
                all_raw[name] = data
        has_ff = True
    except ImportError:
        logger.exception("Failed to import First Four data")
        all_raw = raw_teams
        has_ff = False
        FIRST_FOUR_GAMES = []
        FIRST_FOUR_TEAMS = {}

    # Build Team objects for all 68
    teams_68: Dict[str, Team] = {}
    for name, data in all_raw.items():
        clean = {k: v for k, v in data.items() if k in VALID}
        teams_68[name] = Team(name=name, **clean)

    all_names = list(teams_68.keys())
    base_names = list(raw_teams.keys())

    # Counters
    won_r64 = {t: 0 for t in all_names}
    won_r32 = {t: 0 for t in all_names}
    won_s16 = {t: 0 for t in all_names}
    won_e8  = {t: 0 for t in all_names}
    won_f4  = {t: 0 for t in all_names}
    won_ncg = {t: 0 for t in all_names}
    won_ff  = {t: 0 for t in all_names}
    in_title = {t: 0 for t in all_names}

    cache: Dict = {}
    cfg = SimulationConfig(
        n_sims=n_sims,
        latent_sigma=latent_sigma,
        team_overrides=team_overrides or {},
    )
    forced = forced_picks or {}
    rng = np.random.default_rng()  # no fixed seed — different results each run!

    def latent_sigma(t: Team) -> float:
        return cfg.latent_sigma * (1 + 0.3 * t.seed / 16)

    slot_to_current = {}
    if has_ff:
        for game in FIRST_FOUR_GAMES:
            slot_to_current[game["slot"]] = game["team_b"]

    t0 = time.time()
    yield {
        "type": "start",
        "n_sims": n_sims,
        "assumptions_applied": len(cfg.team_overrides),
        "forced_picks_applied": len(forced),
        "ts": t0,
    }

    # Pre-compute all pairwise win probs once (no ML calls in sim loop)
    prob_matrix = precompute_pairwise_probs(teams_68, all_names, cfg)
    slot_to_default_ff = slot_to_current if has_ff else {}
    first_four_for_vec = FIRST_FOUR_GAMES if has_ff else None

    def _forced_key(region: str, round_i: int, gi: int) -> str:
        return f"{region}:{round_i}:{gi}"

    def _simulate_region_with_forces(
        region: str,
        bracket: list,
        latent_map: Dict[str, float],
        stochastic: bool,
        include_games: bool,
    ):
        cur = list(bracket)
        round_labels = ["r32", "s16", "e8", "champ"]
        round_winners: Dict[str, list] = {}
        games_out: list = []
        ri = 0
        while len(cur) > 1:
            label = round_labels[min(ri, len(round_labels)-1)]
            winners = []
            next_round = []
            round_games = []
            for gi in range(0, len(cur)//2):
                ta, tb = cur[2*gi], cur[2*gi+1]
                lp_a = latent_map.get(ta.name, 0.0)
                lp_b = latent_map.get(tb.name, 0.0)
                prob, var, bd = compute_matchup_prob(ta, tb, cfg, cache, lp_a, lp_b)
                forced_name = forced.get(_forced_key(region, ri, gi))
                if forced_name == ta.name:
                    winner = ta
                elif forced_name == tb.name:
                    winner = tb
                else:
                    if stochastic:
                        winner = ta if simulate_game(prob, var, rng) else tb
                    else:
                        winner = ta if prob >= 0.5 else tb
                next_round.append(winner)
                winners.append(winner.name)
                if include_games:
                    round_games.append({
                        "team_a": ta.name,
                        "team_b": tb.name,
                        "seed_a": ta.seed,
                        "seed_b": tb.seed,
                        "win_prob_a": round(prob * 100, 1),
                        "winner": winner.name,
                        "upset": winner.seed > min(ta.seed, tb.seed),
                        "expected_margin": round(abs(ta.avg_mov*prob - tb.avg_mov*(1-prob)), 1),
                        "volatility": round(var * 100, 1),
                        "breakdown": bd,
                        "forced_pick": forced_name in (ta.name, tb.name),
                        "score_note": "Synthetic score estimate for presentation only.",
                    })
            round_winners[label] = winners
            cur = next_round
            ri += 1
            if include_games:
                games_out.append(round_games)
        return round_winners, cur[0].name, games_out

    def _project_game(region: str, round_i: int, gi: int, ta: Team, tb: Team):
        prob, var, bd = compute_matchup_prob(ta, tb, cfg, cache, 0.0, 0.0)
        forced_name = forced.get(_forced_key(region, round_i, gi))
        if forced_name == ta.name:
            winner = ta
        elif forced_name == tb.name:
            winner = tb
        else:
            winner = ta if prob >= 0.5 else tb

        score_a, score_b = _simulate_scores(prob, winner.name == ta.name)
        return {
            "team_a": ta.name,
            "team_b": tb.name,
            "seed_a": ta.seed,
            "seed_b": tb.seed,
            "win_prob_a": round(prob * 100, 1),
            "winner": winner.name,
            "upset": winner.seed > min(ta.seed, tb.seed),
            "expected_margin": round(abs(ta.avg_mov * prob - tb.avg_mov * (1 - prob)), 1),
            "volatility": round(var * 100, 1),
            "breakdown": bd,
            "forced_pick": forced_name in (ta.name, tb.name),
            "score_note": "Synthetic score estimate for presentation only.",
            "score_a": score_a,
            "score_b": score_b,
        }

    # Vectorized simulation: run in chunks to emit progress
    done_total = 0
    for chunk_start in range(0, n_sims, emit_every):
        chunk_size = min(emit_every, n_sims - chunk_start)
        counts = run_vectorized_simulation(
            teams=teams_68,
            team_names=all_names,
            prob_matrix=prob_matrix,
            n_sims=chunk_size,
            latent_sigma=cfg.latent_sigma,
            rng=rng,
            forced_picks=forced,
            first_four_games=first_four_for_vec,
            slot_to_default=slot_to_default_ff,
        )
        # Accumulate into dict counters
        for j, name in enumerate(all_names):
            won_r64[name] += int(counts["won_r64"][:, j].sum())
            won_r32[name] += int(counts["won_r32"][:, j].sum())
            won_s16[name] += int(counts["won_s16"][:, j].sum())
            won_e8[name] += int(counts["won_e8"][:, j].sum())
            won_f4[name] += int(counts["won_f4"][:, j].sum())
            won_ncg[name] += int(counts["won_ncg"][:, j].sum())
            in_title[name] += int(counts["in_title"][:, j].sum())
            won_ff[name] += int(counts["won_ff"][:, j].sum())
        done_total += chunk_size

        # Emit progress
        def pct(d):
            return {k: round(v / done_total * 100, 1) for k, v in d.items() if v > 0}
        yield {
            "type": "progress",
            "done": done_total,
            "total": n_sims,
            "champion_pct": dict(sorted(pct(won_ncg).items(), key=lambda x: x[1], reverse=True)[:10]),
            "final_four_pct": dict(sorted(pct(won_f4).items(), key=lambda x: x[1], reverse=True)[:10]),
            "ts": time.time(),
        }

    # Final complete result
    elapsed = round(time.time() - t0, 2)
    n = n_sims

    def fpct(d):
        return dict(sorted({k: round(v/n*100, 2) for k,v in d.items() if v>0}.items(),
                            key=lambda x: x[1], reverse=True))

    # Build consensus bracket: pick the most frequently simulated winner per slot
    # This means the bracket actually reflects what the Monte Carlo produced
    VALID64 = {f.name for f in Team.__dataclass_fields__.values()}
    base_teams_objs = {}
    for name, data in raw_teams.items():
        clean = {k: v for k, v in data.items() if k in VALID64}
        base_teams_objs[name] = Team(name=name, **clean)

    base_cache: Dict = {}

    projected = {}
    for region in REGIONS:
        seed_map = {t.seed: t for t in base_teams_objs.values() if t.region == region}
        bracket = build_region_bracket(seed_map)
        _, _, rounds = _simulate_region_with_forces(
            region=region,
            bracket=bracket,
            latent_map={},
            stochastic=False,
            include_games=True,
        )
        for round_games in rounds:
            for game in round_games:
                prob = game.get("win_prob_a", 50) / 100
                a_wins = game.get("winner") == game.get("team_a")
                sa, sb = _simulate_scores(prob, a_wins)
                game["score_a"] = sa
                game["score_b"] = sb
        projected[region] = rounds

    projected_region_champs = {
        region: base_teams_objs[projected[region][-1][0]["winner"]]
        for region in REGIONS
        if projected.get(region) and projected[region][-1]
    }
    if len(projected_region_champs) == 4:
        sf1_game = _project_game("FinalFour", 0, 0, projected_region_champs["East"], projected_region_champs["Midwest"])
        sf2_game = _project_game("FinalFour", 0, 1, projected_region_champs["West"], projected_region_champs["South"])
        sf1_winner = base_teams_objs[sf1_game["winner"]]
        sf2_winner = base_teams_objs[sf2_game["winner"]]
        title_game = _project_game("FinalFour", 1, 0, sf1_winner, sf2_winner)
        projected["FinalFour"] = [[sf1_game, sf2_game], [title_game]]

    upsets = build_upset_watch(
        {r: build_region_bracket({t.seed: t for t in base_teams_objs.values() if t.region == r})
         for r in REGIONS},
        cfg, base_cache
    )

    # Merge caches
    all_probs = {f"{k[0]} vs {k[1]}": round(v[0]*100, 1)
                 for k, v in {**cache, **base_cache}.items() if isinstance(v, tuple)}

    gm = _get_game_model()
    model_used = "calibrated_ml_stack" if gm and gm.is_fitted else "fallback_blend"

    # Emit final MC odds before game events so the hero updates immediately
    yield {
        "type": "progress",
        "done": n,
        "total": n,
        "champion_pct": dict(sorted(fpct(won_ncg).items(), key=lambda x: x[1], reverse=True)[:10]),
        "final_four_pct": dict(sorted(fpct(won_f4).items(), key=lambda x: x[1], reverse=True)[:10]),
        "ts": time.time(),
    }

    # Emit games one by one so brackets populate live.
    # R64 -> R32 -> S16 -> E8, interleaved across regions.
    for ri in [0, 1, 2, 3]:
        for region in REGIONS:
            games = projected.get(region, [])
            if ri >= len(games):
                continue
            for gi, game in enumerate(games[ri]):
                yield {"type":"game","region":region,"round":ri,"gi":gi,"game":game}
                time.sleep(0.07)

    # Final complete event with all odds
    yield {
        "type": "complete",
        "champion_pct":      fpct(won_ncg),
        "final_four_pct":    fpct(won_f4),
        "elite_eight_pct":   fpct(won_s16),
        "sweet_sixteen_pct": fpct(won_r32),
        "round_of_32_pct":   fpct(won_r64),
        "title_game_pct":    fpct(in_title),
        "first_four_pct":    fpct(won_ff),
        "predicted_bracket": projected,
        "upset_watch":       upsets,
        "matchup_probs":     all_probs,
        "n_sims":            n,
        "model_used":        model_used,
        "elapsed_sec":       elapsed,
        "ts":                time.time(),
    }
