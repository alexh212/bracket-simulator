import unittest
from unittest.mock import patch

from data.teams_2026 import TEAMS_2026
from services.simulation import SimulationConfig, run_simulation


def fake_compute_matchup_prob(a, b, cfg, cache, latent_a=0.0, latent_b=0.0):
    if a.seed < b.seed:
        return 0.9, 0.0, {}
    if a.seed > b.seed:
        return 0.1, 0.0, {}
    return 0.5, 0.0, {}


def fake_simulate_game(prob, variance, rng):
    return prob >= 0.5


class SimulationInvariantTestCase(unittest.TestCase):
    @patch("services.simulation.compute_matchup_prob", side_effect=fake_compute_matchup_prob)
    @patch("services.simulation.simulate_game", side_effect=fake_simulate_game)
    @patch("services.simulation._get_game_model", return_value=None)
    def test_round_probabilities_are_monotonic(self, _mock_model, _mock_sampler, _mock_matchup_prob):
        result = run_simulation(TEAMS_2026, SimulationConfig(n_sims=4, latent_sigma=0.0))

        teams = set(TEAMS_2026.keys())
        for team in teams:
            r32 = result.round_of_32_pct.get(team, 0.0)
            s16 = result.sweet_sixteen_pct.get(team, 0.0)
            e8 = result.elite_eight_pct.get(team, 0.0)
            f4 = result.final_four_pct.get(team, 0.0)
            title = result.title_game_pct.get(team, 0.0)
            champ = result.champion_pct.get(team, 0.0)
            self.assertGreaterEqual(r32, s16)
            self.assertGreaterEqual(s16, e8)
            self.assertGreaterEqual(e8, f4)
            self.assertGreaterEqual(f4, title)
            self.assertGreaterEqual(title, champ)


if __name__ == "__main__":
    unittest.main()
