"""Tests for ESPN merge / team matching."""
import json

import pytest

from services.live_scores import EspnGame, merge_static_with_espn, _team_matches


def test_team_matches_basic():
    aliases = {}
    assert _team_matches("Duke", "Duke Blue Devils", aliases)
    assert _team_matches("Houston", "Houston Cougars", aliases)
    assert not _team_matches("Duke", "North Carolina Tar Heels", aliases)


def test_team_matches_aliases():
    aliases = {"Ohio St.": ["Ohio State"]}
    assert _team_matches("Ohio St.", "Ohio State Buckeyes", aliases)


def test_merge_overlays_scores():
    static = {
        "last_updated": "2026-01-01T00:00:00Z",
        "tournament_status": "Test",
        "games": [
            {
                "region": "East",
                "round": 0,
                "game_index": 0,
                "team_a": "Duke",
                "team_b": "Siena",
                "seed_a": 1,
                "seed_b": 16,
                "score_a": 0,
                "score_b": 0,
                "winner": "",
                "status": "upcoming",
            }
        ],
    }
    espn = [
        EspnGame(
            team_a="Duke Blue Devils",
            team_b="Siena Saints",
            score_a=82,
            score_b=58,
            status="final",
            winner="Duke Blue Devils",
            espn_status_raw="STATUS_FINAL",
            detail="Final",
        )
    ]
    out = merge_static_with_espn(static, espn)
    g = out["games"][0]
    assert g["score_a"] == 82
    assert g["score_b"] == 58
    assert g["status"] == "final"
    assert g["winner"] == "Duke"
    assert out["live_scores_matched"] == 1


def test_merge_live_no_winner():
    static = {
        "last_updated": None,
        "tournament_status": "Live",
        "games": [
            {
                "region": "West",
                "round": 0,
                "game_index": 0,
                "team_a": "Arizona",
                "team_b": "LIU",
                "seed_a": 1,
                "seed_b": 16,
                "score_a": 0,
                "score_b": 0,
                "winner": "",
                "status": "upcoming",
            }
        ],
    }
    espn = [
        EspnGame(
            team_a="Arizona Wildcats",
            team_b="Long Island University Sharks",
            score_a=45,
            score_b=40,
            status="live",
            winner=None,
            espn_status_raw="STATUS_SECOND_HALF",
            detail="2nd 12:00",
        )
    ]
    out = merge_static_with_espn(static, espn)
    g = out["games"][0]
    assert g["status"] == "live"
    assert g["score_a"] == 45
    assert g["score_b"] == 40
    assert g.get("winner") in ("", None) or g["winner"] == ""
