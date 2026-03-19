"""Shared fixtures for bracket-simulator tests."""

import numpy as np
import pytest

from services.simulation import Team, SimulationConfig, REGIONS, R64_ORDER


def _make_team(name: str, seed: int, region: str, **overrides) -> Team:
    """Build a Team with sensible defaults; override any field via kwargs."""
    defaults = dict(
        conference="TestConf",
        kenpom_adj_off=110.0,
        kenpom_adj_def=95.0,
        kenpom_tempo=68.0,
        kenpom_luck=0.0,
        torvik_rating=25.0,
        sagarin_rating=85.0,
        ncaa_net_rank=seed * 4,
        elo_current=1700 - seed * 30,
        elo_preseason=1700 - seed * 30,
        elo_change_last10=0.0,
        elo_volatility=50.0,
        efg_pct=52.0,
        def_efg_pct=48.0,
        ts_pct=56.0,
        ppp=1.10,
        def_ppp=0.95,
        three_pt_rate=0.38,
        three_pt_pct=0.35,
        three_pt_variance=0.04,
        rim_rate=0.30,
        ft_rate=0.30,
        ft_pct=0.75,
        orb_rate=30.0,
        drb_rate=70.0,
        reb_margin=3.0,
        tov_rate=16.0,
        forced_tov_rate=18.0,
        tov_variance=3.0,
        possessions_per_game=68.0,
        foul_rate=18.0,
        opp_ft_rate=0.28,
        avg_mov=8.0 - seed * 0.5,
        close_game_record=0.55,
        sos=5.0,
        nonconf_sos=3.0,
        q1_record_pct=0.60,
        q2_record_pct=0.75,
        last10_net=5.0,
        last10_off=110.0,
        last10_def=95.0,
        last10_3pt=0.36,
        last10_tov=15.0,
        last10_reb=3.0,
        last10_mov=6.0,
        form_score=0.7,
        win_streak=3,
        experience=2.5,
        upperclassmen_pct=0.45,
        bench_pct=0.30,
        avg_height=78.0,
        nba_prospects=1,
        coach_tourney_wins=8,
        coach_upset_pct=0.10,
        moneyline_prob=0.5 - seed * 0.02,
        championship_odds_pct=max(0.1, 20.0 - seed * 1.5),
        injury_factor=1.0,
        distance_bucket=2,
        conference_power=0.70,
        historical_seed_win_pct=max(0.01, 1.0 - seed * 0.06),
    )
    defaults.update(overrides)
    return Team(name=name, seed=seed, region=region, **defaults)


@pytest.fixture
def mini_teams():
    """64-team catalog (16 per region) with seed-based strength gradient."""
    teams = {}
    for region in REGIONS:
        for seed in range(1, 17):
            name = f"{region}_{seed}"
            teams[name] = _make_team(name, seed, region)
    return teams


@pytest.fixture
def mini_teams_dict(mini_teams):
    """Raw dict version (as received by the API)."""
    from dataclasses import asdict
    return {name: asdict(t) for name, t in mini_teams.items()}


@pytest.fixture
def cfg():
    return SimulationConfig(n_sims=100, latent_sigma=0.04)
