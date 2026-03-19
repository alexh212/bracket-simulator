"""Tests for probability calculations — fallback blend, injury, shrinkage."""

import numpy as np
import pytest
from unittest.mock import patch

from services.simulation import (
    _fallback_prob,
    compute_matchup_prob,
    SimulationConfig,
)
from tests.conftest import _make_team


# ---------------------------------------------------------------------------
# Fallback probability
# ---------------------------------------------------------------------------
class TestFallbackProb:
    def test_equal_teams_near_half(self):
        t = _make_team("A", 8, "East")
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        p = _fallback_prob(t, t, cfg)
        assert 0.40 <= p <= 0.60

    def test_strong_beats_weak(self):
        strong = _make_team("Strong", 1, "East", elo_current=1900)
        weak = _make_team("Weak", 16, "East", elo_current=1200)
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        p = _fallback_prob(strong, weak, cfg)
        assert p > 0.7

    def test_bounded_output(self):
        a = _make_team("A", 1, "East", elo_current=2200, moneyline_prob=0.99)
        b = _make_team("B", 16, "East", elo_current=800, moneyline_prob=0.01)
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        p = _fallback_prob(a, b, cfg)
        assert 0.02 <= p <= 0.98

    def test_injury_hurts_team_a(self):
        healthy = _make_team("A", 5, "East", injury_factor=1.0)
        injured = _make_team("A_inj", 5, "East", injury_factor=0.7)
        opponent = _make_team("B", 5, "East")
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        p_healthy = _fallback_prob(healthy, opponent, cfg)
        p_injured = _fallback_prob(injured, opponent, cfg)
        assert p_injured < p_healthy

    def test_opponent_injury_helps_team_a(self):
        a = _make_team("A", 5, "East")
        b_healthy = _make_team("B", 5, "East", injury_factor=1.0)
        b_injured = _make_team("B_inj", 5, "East", injury_factor=0.7)
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        p_vs_healthy = _fallback_prob(a, b_healthy, cfg)
        p_vs_injured = _fallback_prob(a, b_injured, cfg)
        assert p_vs_injured > p_vs_healthy


# ---------------------------------------------------------------------------
# Shrinkage / latent in compute_matchup_prob
# ---------------------------------------------------------------------------
class TestComputeMatchupProb:
    @patch("services.simulation._get_game_model", return_value=None)
    def test_shrinkage_moves_toward_half(self, _mock):
        strong = _make_team("Strong", 1, "East", elo_current=1900)
        weak = _make_team("Weak", 16, "East", elo_current=1200)
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.0)
        cache = {}
        prob, _, _ = compute_matchup_prob(strong, weak, cfg, cache)
        raw = _fallback_prob(strong, weak, cfg)
        shrunk = 0.5 + (raw - 0.5) * 0.78
        assert abs(prob - shrunk) < 0.02

    @patch("services.simulation._get_game_model", return_value=None)
    def test_positive_latent_helps_team_a(self, _mock):
        a = _make_team("A", 5, "East")
        b = _make_team("B", 5, "East")
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.06)
        no_lat, _, _ = compute_matchup_prob(a, b, cfg, {}, 0.0, 0.0)
        pos_lat, _, _ = compute_matchup_prob(a, b, cfg, {}, 0.1, 0.0)
        assert pos_lat > no_lat

    @patch("services.simulation._get_game_model", return_value=None)
    def test_symmetric_latent(self, _mock):
        """P(A beats B | la, lb) + P(B beats A | lb, la) ≈ 1.0."""
        a = _make_team("A", 5, "East")
        b = _make_team("B", 5, "East")
        cfg = SimulationConfig(n_sims=10, latent_sigma=0.06)
        p_ab, _, _ = compute_matchup_prob(a, b, cfg, {}, 0.05, -0.05)
        p_ba, _, _ = compute_matchup_prob(b, a, cfg, {}, -0.05, 0.05)
        assert abs(p_ab + p_ba - 1.0) < 0.05
