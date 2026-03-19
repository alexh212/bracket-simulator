"""API endpoint tests — contract validation for all routes."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from main import CLEAN, app

client = TestClient(app)
TEAM_A, TEAM_B = list(CLEAN.keys())[:2]


# ---------------------------------------------------------------------------
# Health & catalog
# ---------------------------------------------------------------------------
class TestHealthAndCatalog:
    def test_health_endpoint(self):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["teams"] == len(CLEAN)

    def test_teams_endpoint_returns_catalog(self):
        r = client.get("/teams")
        assert r.status_code == 200
        body = r.json()
        assert TEAM_A in body
        assert "seed" in body[TEAM_A]
        assert "region" in body[TEAM_A]

    def test_model_info_endpoint(self):
        r = client.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        assert "model_details" in body
        assert body["inference_data"]["dataset"] == "TEAMS_2026"

    def test_seed_history(self):
        r = client.get("/seed-history")
        assert r.status_code == 200
        body = r.json()
        assert "1" in body or 1 in body

    def test_region_stats(self):
        r = client.get("/region-stats")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 4
        regions = {item["region"] for item in body}
        assert regions == {"East", "West", "Midwest", "South"}

    def test_results_endpoint(self):
        r = client.get("/results")
        assert r.status_code == 200
        body = r.json()
        assert "games" in body
        assert isinstance(body["games"], list)


# ---------------------------------------------------------------------------
# Matchup
# ---------------------------------------------------------------------------
class TestMatchup:
    @patch("pipeline.calibrated_game_model.get_game_model")
    def test_matchup_returns_required_fields(self, mock_gm):
        mock_model = Mock()
        mock_model.predict_with_breakdown.return_value = {
            "win_prob_a": 62.5, "win_prob_b": 37.5,
            "expected_margin": 6.2, "upset": False,
            "volatility": 11.4, "breakdown": {"elo": 60.0},
            "model_confidence": 71.0,
        }
        mock_gm.return_value = mock_model
        r = client.post("/matchup", json={"team_a": TEAM_A, "team_b": TEAM_B})
        assert r.status_code == 200
        body = r.json()
        assert "win_prob_a" in body
        assert "score_note" in body

    @patch("pipeline.calibrated_game_model.get_game_model")
    def test_matchup_second_pair(self, mock_gm):
        """Verify matchup works for a different team pair."""
        mock_model = Mock()
        mock_model.predict_with_breakdown.return_value = {
            "win_prob_a": 55.0, "win_prob_b": 45.0,
            "expected_margin": 3.1, "upset": False,
            "volatility": 14.0, "breakdown": {"elo": 53.0},
            "model_confidence": 62.0,
        }
        mock_gm.return_value = mock_model
        teams = list(CLEAN.keys())
        r = client.post("/matchup", json={"team_a": teams[2], "team_b": teams[3]})
        assert r.status_code == 200
        body = r.json()
        assert "win_prob_a" in body


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------
class TestWhatIf:
    def test_rejects_unknown_fields(self):
        r = client.post("/whatif", json={
            "team_a": TEAM_A, "team_b": TEAM_B,
            "scenarios": [{"label": "Bad", "target": "a", "mystery_stat": 1.0}],
        })
        assert r.status_code == 422

    @patch("pipeline.calibrated_game_model.get_game_model")
    def test_accepts_valid_scenario(self, mock_gm):
        mock_model = Mock()
        mock_model.predict.return_value = 0.6
        mock_model.predict_with_breakdown.return_value = {
            "win_prob_a": 60.0, "win_prob_b": 40.0,
            "expected_margin": 5.0, "upset": False,
            "volatility": 12.0, "breakdown": {"elo": 58.0},
            "model_confidence": 68.0,
        }
        mock_gm.return_value = mock_model
        r = client.post("/whatif", json={
            "team_a": TEAM_A, "team_b": TEAM_B,
            "scenarios": [{"label": "Boost", "target": "a", "delta_elo_current": 50}],
        })
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Simulation stream
# ---------------------------------------------------------------------------
class TestSimulateStream:
    def test_rejects_invalid_override_json(self):
        r = client.get("/simulate/stream?team_overrides=not-json")
        assert r.status_code == 400

    def test_clamps_excessive_sims(self):
        """API clamps n_sims to MAX_N_SIMS rather than rejecting."""
        r = client.get("/simulate/stream?n_sims=999999")
        assert r.status_code == 200

    def test_rejects_invalid_forced_picks_key(self):
        picks = json.dumps({"BadKey": "Duke"})
        r = client.get(f"/simulate/stream?forced_picks={picks}")
        assert r.status_code == 400
