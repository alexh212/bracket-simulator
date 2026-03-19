"""Tests for bracket structure, seeding, and region integrity."""

import pytest

from services.simulation import R64_ORDER, REGIONS, build_region_bracket
from data.teams_2026 import TEAMS_2026
from tests.conftest import _make_team


class TestR64Order:
    def test_eight_matchups(self):
        assert len(R64_ORDER) == 8

    def test_all_seeds_present(self):
        seeds = set()
        for s1, s2 in R64_ORDER:
            seeds.add(s1)
            seeds.add(s2)
        assert seeds == set(range(1, 17))

    def test_matchup_seeds_sum_to_17(self):
        for s1, s2 in R64_ORDER:
            assert s1 + s2 == 17


class TestRegions:
    def test_four_regions(self):
        assert len(REGIONS) == 4
        assert set(REGIONS) == {"East", "West", "Midwest", "South"}


class TestBuildRegionBracket:
    def test_bracket_has_16_teams(self):
        seed_map = {}
        for seed in range(1, 17):
            seed_map[seed] = _make_team(f"T{seed}", seed, "East")
        bracket = build_region_bracket(seed_map)
        assert len(bracket) == 16

    def test_bracket_first_game_is_1v16(self):
        seed_map = {}
        for seed in range(1, 17):
            seed_map[seed] = _make_team(f"T{seed}", seed, "East")
        bracket = build_region_bracket(seed_map)
        assert bracket[0].seed == 1
        assert bracket[1].seed == 16


class TestTeamsCatalog:
    def test_64_teams(self):
        assert len(TEAMS_2026) == 64

    def test_16_per_region(self):
        for region in REGIONS:
            count = sum(1 for t in TEAMS_2026.values() if t.get("region") == region)
            assert count == 16, f"{region} has {count} teams, expected 16"

    def test_all_seeds_present_per_region(self):
        for region in REGIONS:
            seeds = {t["seed"] for t in TEAMS_2026.values() if t.get("region") == region}
            assert seeds == set(range(1, 17)), f"{region} missing seeds"

    def test_required_fields_present(self):
        required = ["seed", "region", "conference", "elo_current", "kenpom_adj_off"]
        for name, data in TEAMS_2026.items():
            for field in required:
                assert field in data, f"{name} missing {field}"
