"""Tests for the vectorized NumPy simulation engine."""

import numpy as np
import pytest
from unittest.mock import patch

from services.simulation import SimulationConfig, REGIONS, R64_ORDER
from services.vectorized_sim import (
    precompute_pairwise_probs,
    run_vectorized_simulation,
    _apply_latent_to_probs,
    _latent_sigma_per_team,
    SHRINK_ALPHA,
    LATENT_SCALE,
)


# ---------------------------------------------------------------------------
# Unit: logit shift
# ---------------------------------------------------------------------------
class TestApplyLatentToProbs:
    def test_zero_delta_preserves_base(self):
        base = np.array([0.6, 0.4, 0.5])
        result = _apply_latent_to_probs(base, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(result, base, atol=1e-6)

    def test_positive_delta_increases_prob(self):
        base = np.array([0.5])
        result = _apply_latent_to_probs(base, np.array([0.1]), np.array([0.0]))
        assert result[0] > 0.5

    def test_negative_delta_decreases_prob(self):
        base = np.array([0.5])
        result = _apply_latent_to_probs(base, np.array([0.0]), np.array([0.1]))
        assert result[0] < 0.5

    def test_output_clipped_to_bounds(self):
        base = np.array([0.99, 0.01])
        result = _apply_latent_to_probs(
            base, np.array([10.0, -10.0]), np.array([0.0, 0.0])
        )
        assert np.all(result >= 0.02)
        assert np.all(result <= 0.98)

    def test_symmetric_delta(self):
        """Shifting A up by x should mirror shifting B up by x."""
        base = np.array([0.5])
        up = _apply_latent_to_probs(base, np.array([0.1]), np.array([0.0]))
        down = _apply_latent_to_probs(base, np.array([0.0]), np.array([0.1]))
        np.testing.assert_allclose(up + down, [1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Unit: latent sigma scaling
# ---------------------------------------------------------------------------
class TestLatentSigma:
    def test_seed_1_gets_base_sigma_plus_small_boost(self, mini_teams):
        t = mini_teams["East_1"]
        sigma = _latent_sigma_per_team(t, 0.06)
        expected = 0.06 * (1 + 0.3 * 1 / 16)
        assert abs(sigma - expected) < 1e-9

    def test_higher_seed_gets_more_variance(self, mini_teams):
        s1 = _latent_sigma_per_team(mini_teams["East_1"], 0.06)
        s16 = _latent_sigma_per_team(mini_teams["East_16"], 0.06)
        assert s16 > s1


# ---------------------------------------------------------------------------
# Unit: pairwise prob matrix
# ---------------------------------------------------------------------------
class TestPairwiseProbs:
    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_matrix_shape(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        n = len(names)
        assert P.shape == (n, n)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_diagonal_is_half(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        np.testing.assert_allclose(np.diag(P), 0.5, atol=1e-9)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_symmetry(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        np.testing.assert_allclose(P + P.T, 1.0, atol=1e-9)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_probs_within_bounds(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        np.fill_diagonal(P, 0.5)
        assert np.all(P >= 0.02)
        assert np.all(P <= 0.98)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_one_seed_favored_over_sixteen(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        name_to_idx = {n: i for i, n in enumerate(names)}
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        i1 = name_to_idx["East_1"]
        i16 = name_to_idx["East_16"]
        assert P[i1, i16] > 0.5


# ---------------------------------------------------------------------------
# Integration: full vectorized sim
# ---------------------------------------------------------------------------
class TestRunVectorizedSimulation:
    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_output_keys(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=50, latent_sigma=0.04, rng=rng,
        )
        expected_keys = {"won_r64", "won_r32", "won_s16", "won_e8",
                         "won_f4", "won_ncg", "in_title", "won_ff"}
        assert set(counts.keys()) == expected_keys

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_shapes(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        n_sims = 50
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=n_sims, latent_sigma=0.04, rng=rng,
        )
        n = len(names)
        for key in counts:
            assert counts[key].shape == (n_sims, n)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_exactly_one_champion_per_sim(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        n_sims = 200
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=n_sims, latent_sigma=0.04, rng=rng,
        )
        champs_per_sim = counts["won_ncg"].sum(axis=1)
        np.testing.assert_array_equal(champs_per_sim, 1)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_exactly_four_final_four_per_sim(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        n_sims = 200
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=n_sims, latent_sigma=0.04, rng=rng,
        )
        f4_per_sim = counts["won_f4"].sum(axis=1)
        np.testing.assert_array_equal(f4_per_sim, 4)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_exactly_two_title_game_per_sim(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        n_sims = 200
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=n_sims, latent_sigma=0.04, rng=rng,
        )
        title_per_sim = counts["in_title"].sum(axis=1)
        np.testing.assert_array_equal(title_per_sim, 2)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_r64_winners_total_per_sim(self, _mock, mini_teams, cfg):
        """32 R64 games → 32 winners per sim."""
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=100, latent_sigma=0.04, rng=rng,
        )
        r64_per_sim = counts["won_r64"].sum(axis=1)
        np.testing.assert_array_equal(r64_per_sim, 32)

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_round_monotonicity(self, _mock, mini_teams, cfg):
        """For each team: R64 wins >= R32 wins >= ... >= championship wins."""
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=500, latent_sigma=0.04, rng=rng,
        )
        totals = {k: v.sum(axis=0) for k, v in counts.items()}
        for j in range(len(names)):
            assert totals["won_r64"][j] >= totals["won_r32"][j]
            assert totals["won_r32"][j] >= totals["won_s16"][j]
            assert totals["won_s16"][j] >= totals["won_e8"][j]
            assert totals["won_f4"][j] >= totals["in_title"][j]
            assert totals["in_title"][j] >= totals["won_ncg"][j]

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_one_seed_wins_more_than_sixteen(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        name_to_idx = {n: i for i, n in enumerate(names)}
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=1000, latent_sigma=0.04, rng=rng,
        )
        champ_totals = counts["won_ncg"].sum(axis=0)
        i1 = name_to_idx["East_1"]
        i16 = name_to_idx["East_16"]
        assert champ_totals[i1] > champ_totals[i16]

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_deterministic_with_seed(self, _mock, mini_teams, cfg):
        names = list(mini_teams.keys())
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        r1 = run_vectorized_simulation(
            mini_teams, names, P, n_sims=100, latent_sigma=0.04,
            rng=np.random.default_rng(99),
        )
        r2 = run_vectorized_simulation(
            mini_teams, names, P, n_sims=100, latent_sigma=0.04,
            rng=np.random.default_rng(99),
        )
        np.testing.assert_array_equal(r1["won_ncg"], r2["won_ncg"])


# ---------------------------------------------------------------------------
# Integration: forced picks
# ---------------------------------------------------------------------------
class TestForcedPicks:
    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_forced_r64_always_wins(self, _mock, mini_teams, cfg):
        """Locking East 1-seed game (East:0:0) to the 16-seed should make
        the 16-seed win that game 100% of the time."""
        names = list(mini_teams.keys())
        name_to_idx = {n: i for i, n in enumerate(names)}
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        forced = {"East:0:0": "East_16"}
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=200, latent_sigma=0.04, rng=rng,
            forced_picks=forced,
        )
        i16 = name_to_idx["East_16"]
        i1 = name_to_idx["East_1"]
        wins_16 = counts["won_r64"][:, i16].sum()
        wins_1 = counts["won_r64"][:, i1].sum()
        assert wins_16 == 200
        assert wins_1 == 0

    @patch("services.vectorized_sim._get_game_model", return_value=None)
    def test_forced_championship_winner(self, _mock, mini_teams, cfg):
        """Forcing the title game should guarantee that team wins it all."""
        names = list(mini_teams.keys())
        name_to_idx = {n: i for i, n in enumerate(names)}
        P = precompute_pairwise_probs(mini_teams, names, cfg)
        forced = {"FinalFour:1:0": "East_1"}
        rng = np.random.default_rng(42)
        counts = run_vectorized_simulation(
            mini_teams, names, P, n_sims=200, latent_sigma=0.04, rng=rng,
            forced_picks=forced,
        )
        i1 = name_to_idx["East_1"]
        # East_1 should be champion whenever it reaches the title game
        title_appearances = counts["in_title"][:, i1]
        championships = counts["won_ncg"][:, i1]
        np.testing.assert_array_equal(
            championships[title_appearances > 0],
            1,
        )
