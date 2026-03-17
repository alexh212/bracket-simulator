from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from main import CLEAN, app

client = TestClient(app)
TEAM_A, TEAM_B = list(CLEAN.keys())[:2]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["teams"] == len(CLEAN)


def test_teams_endpoint_returns_catalog():
    response = client.get("/teams")
    assert response.status_code == 200
    payload = response.json()
    assert TEAM_A in payload
    assert "seed" in payload[TEAM_A]


def test_model_info_endpoint_returns_metadata():
    response = client.get("/model-info")
    assert response.status_code == 200
    payload = response.json()
    assert payload["inference_data"]["dataset"] == "TEAMS_2026"
    assert "stack" in payload["model_details"]


@patch("pipeline.calibrated_game_model.get_game_model")
def test_matchup_returns_score_note(mock_get_game_model):
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

    response = client.post("/matchup", json={"team_a": TEAM_A, "team_b": TEAM_B})
    assert response.status_code == 200
    payload = response.json()
    assert "score_note" in payload
    assert "win_prob_a" in payload


def test_whatif_rejects_unknown_fields():
    response = client.post(
        "/whatif",
        json={
            "team_a": TEAM_A,
            "team_b": TEAM_B,
            "scenarios": [{"label": "Bad", "target": "a", "mystery_stat": 1.0}],
        },
    )
    assert response.status_code == 422


def test_simulate_stream_rejects_invalid_override_json():
    response = client.get("/simulate/stream?team_overrides=not-json")
    assert response.status_code == 400
