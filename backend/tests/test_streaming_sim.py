"""Tests for the SSE streaming simulation generator."""

import pytest
from unittest.mock import patch

from data.teams_2026 import TEAMS_2026
from services.streaming_sim import run_streaming_simulation


@patch("services.simulation._get_game_model", return_value=None)
@patch("services.vectorized_sim._get_game_model", return_value=None)
class TestStreamingSimEvents:
    """Validate the event sequence and data contracts of the streaming sim."""

    def test_event_sequence(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        types = [e["type"] for e in events]
        assert types[0] == "start"
        assert "progress" in types
        assert types[-1] == "complete"

    def test_start_event_has_metadata(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        start = events[0]
        assert start["n_sims"] == 250
        assert "ts" in start

    def test_progress_event_has_champion_pct(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        progress_events = [e for e in events if e["type"] == "progress"]
        assert len(progress_events) >= 1
        last_progress = progress_events[-1]
        assert "champion_pct" in last_progress
        assert "final_four_pct" in last_progress
        assert last_progress["done"] == last_progress["total"]

    def test_complete_event_has_full_payload(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        complete = events[-1]
        required_keys = [
            "champion_pct", "final_four_pct", "elite_eight_pct",
            "sweet_sixteen_pct", "round_of_32_pct", "title_game_pct",
            "predicted_bracket", "n_sims", "model_used", "elapsed_sec",
        ]
        for key in required_keys:
            assert key in complete, f"Missing key: {key}"

    def test_champion_pcts_sum_near_100(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=500, emit_every=250))
        complete = events[-1]
        total = sum(complete["champion_pct"].values())
        assert 95 < total < 105

    def test_bracket_has_all_regions(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        complete = events[-1]
        bracket = complete["predicted_bracket"]
        for region in ["East", "West", "Midwest", "South"]:
            assert region in bracket
        assert "FinalFour" in bracket

    def test_game_events_have_required_fields(self, _m1, _m2):
        events = list(run_streaming_simulation(TEAMS_2026, n_sims=250, emit_every=250))
        game_events = [e for e in events if e["type"] == "game"]
        assert len(game_events) > 0
        for ge in game_events[:5]:
            assert "region" in ge
            assert "round" in ge
            assert "game" in ge
            game = ge["game"]
            assert "team_a" in game
            assert "team_b" in game
            assert "winner" in game
            assert "win_prob_a" in game

    def test_forced_picks_applied(self, _m1, _m2):
        forced = {"East:0:0": "Siena"}
        events = list(run_streaming_simulation(
            TEAMS_2026, n_sims=250, emit_every=250, forced_picks=forced,
        ))
        complete = events[-1]
        east_r64 = complete["predicted_bracket"]["East"][0]
        first_game = east_r64[0]
        assert first_game["winner"] == "Siena"
        assert first_game["forced_pick"] is True
