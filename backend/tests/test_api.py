import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from main import CLEAN, app


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.team_a, cls.team_b = list(CLEAN.keys())[:2]

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["teams"], len(CLEAN))

    def test_teams_endpoint_returns_catalog(self):
        response = self.client.get("/teams")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(self.team_a, payload)
        self.assertIn("seed", payload[self.team_a])

    def test_model_info_endpoint_returns_metadata(self):
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["inference_data"]["dataset"], "TEAMS_2026")
        self.assertIn("stack", payload["model_details"])

    @patch("pipeline.calibrated_game_model.get_game_model")
    def test_matchup_returns_score_note(self, mock_get_game_model):
        mock_model = Mock()
        mock_model.predict_with_breakdown.return_value = {
            "win_prob_a": 62.5,
            "win_prob_b": 37.5,
            "expected_margin": 6.2,
            "upset": False,
            "volatility": 11.4,
            "breakdown": {"elo": 60.0},
            "model_confidence": 71.0,
        }
        mock_get_game_model.return_value = mock_model

        response = self.client.post("/matchup", json={"team_a": self.team_a, "team_b": self.team_b})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("score_note", payload)
        self.assertIn("win_prob_a", payload)

    def test_whatif_rejects_unknown_fields(self):
        response = self.client.post(
            "/whatif",
            json={
                "team_a": self.team_a,
                "team_b": self.team_b,
                "scenarios": [{"label": "Bad", "target": "a", "mystery_stat": 1.0}],
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_simulate_stream_rejects_invalid_override_json(self):
        response = self.client.get("/simulate/stream?team_overrides=not-json")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
