"""
Fetch NCAA men's scores from ESPN's public scoreboard API and match to bracket slots.

Enable with ENABLE_LIVE_SCORES=1. This is best-effort: team-name matching can miss edge cases;
extend data/espn_team_aliases.json when ESPN strings don't align with TEAMS_2026 names.

ESPN ToS may restrict automated use — suitable for personal / low-traffic apps; use a licensed
feed for production at scale.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bracket_api")

ESPN_SCOREBOARD = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
)

ROOT_DIR = Path(__file__).resolve().parent.parent
ALIASES_PATH = ROOT_DIR / "data" / "espn_team_aliases.json"

_STATUS_MAP = {
    "STATUS_FINAL": "final",
    "STATUS_IN_PROGRESS": "live",
    "STATUS_HALFTIME": "live",
    "STATUS_END_PERIOD": "live",
    "STATUS_FIRST_HALF": "live",
    "STATUS_SECOND_HALF": "live",
    "STATUS_OVERTIME": "live",
    "STATUS_SCHEDULED": "upcoming",
    "STATUS_POSTPONED": "upcoming",
    "STATUS_DELAYED": "live",
}


_ROUND_NAMES = {0: "Round of 64", 1: "Round of 32", 2: "Sweet 16", 3: "Elite 8", 4: "Final Four", 5: "Championship"}


@dataclass
class EspnGame:
    team_a: str
    team_b: str
    score_a: int
    score_b: int
    status: str
    winner: Optional[str]
    espn_status_raw: str
    detail: str
    game_date: Optional[str] = None


_cache_lock = threading.Lock()
_cache_events: Optional[Tuple[float, List[EspnGame]]] = None


def _load_aliases() -> Dict[str, List[str]]:
    if not ALIASES_PATH.exists():
        return {}
    try:
        raw = json.loads(ALIASES_PATH.read_text())
        return {str(k): list(v) for k, v in raw.items() if isinstance(v, list)}
    except (OSError, json.JSONDecodeError, TypeError):
        logger.exception("Failed to read espn_team_aliases.json")
        return {}


def _fold(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _team_matches(ours: str, espn_display: str, aliases: Dict[str, List[str]]) -> bool:
    """Return True if our canonical team name plausibly matches ESPN display name."""
    o = ours.strip()
    e = espn_display.strip()
    if not o or not e:
        return False
    for pat in aliases.get(o, []):
        if pat.lower() in e.lower():
            return True
    ol, el = o.lower(), e.lower()
    if ol in el or el.startswith(ol):
        return True
    fo, fe = _fold(o), _fold(e)
    if len(fo) >= 4 and fo in fe:
        return True
    if len(fo) >= 5 and fe.startswith(fo):
        return True
    # First token (e.g. "Duke" ~ "Duke Blue Devils")
    o0 = ol.replace(".", "").split()[0]
    e0 = el.replace(".", "").split()[0]
    if len(o0) >= 3 and o0 == e0:
        return True
    return False


def _parse_event(event: Dict[str, Any]) -> Optional[EspnGame]:
    comps = event.get("competitions") or []
    if not comps:
        return None
    c = comps[0]
    status = c.get("status") or {}
    stype = (status.get("type") or {}).get("name") or ""
    detail = (status.get("type") or {}).get("shortDetail") or ""
    mapped = _STATUS_MAP.get(stype, "upcoming")
    competitors = c.get("competitors") or []
    if len(competitors) != 2:
        return None

    game_date: Optional[str] = None
    raw_date = event.get("date") or c.get("date")
    if raw_date:
        try:
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            game_date = dt.strftime("%b %d")
        except (ValueError, AttributeError):
            pass

    rows = []
    winner_name = None
    for x in competitors:
        team = x.get("team") or {}
        name = team.get("displayName") or team.get("shortDisplayName") or ""
        try:
            sc = int(x.get("score", "0") or 0)
        except (TypeError, ValueError):
            sc = 0
        rows.append((name, sc, x.get("winner") is True))
        if x.get("winner") is True:
            winner_name = name
    (n0, s0, _), (n1, s1, _) = rows
    return EspnGame(
        team_a=n0,
        team_b=n1,
        score_a=s0,
        score_b=s1,
        status=mapped,
        winner=winner_name if mapped == "final" else None,
        espn_status_raw=stype,
        detail=detail,
        game_date=game_date,
    )


def fetch_espn_games_for_dates(dates_yyyymmdd: List[str], timeout_sec: float = 12.0) -> List[EspnGame]:
    out: List[EspnGame] = []
    seen: set[str] = set()
    for d in dates_yyyymmdd:
        url = f"{ESPN_SCOREBOARD}?dates={d}&limit=400"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bracket-simulator/1.0"})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            logger.warning("ESPN scoreboard fetch failed for %s: %s", d, e)
            continue
        for ev in payload.get("events") or []:
            eid = str(ev.get("id", ""))
            if eid in seen:
                continue
            seen.add(eid)
            g = _parse_event(ev)
            if g:
                out.append(g)
    return out


def fetch_espn_near_today(ttl_sec: float = 45.0) -> List[EspnGame]:
    """Cached list of ESPN games spanning the tournament window (5 days back, 1 forward)."""
    global _cache_events
    now = time.time()
    with _cache_lock:
        if _cache_events is not None and now - _cache_events[0] < ttl_sec:
            return _cache_events[1]
    utc = datetime.now(timezone.utc)
    dates = [(utc + timedelta(days=i)).strftime("%Y%m%d") for i in range(-5, 2)]
    games = fetch_espn_games_for_dates(dates)
    with _cache_lock:
        _cache_events = (time.time(), games)
    return games


def _find_matching_espn(
    team_a: str,
    team_b: str,
    espn_games: List[EspnGame],
    aliases: Dict[str, List[str]],
) -> Optional[EspnGame]:
    for g in espn_games:
        pairs = [
            (_team_matches(team_a, g.team_a, aliases) and _team_matches(team_b, g.team_b, aliases)),
            (_team_matches(team_a, g.team_b, aliases) and _team_matches(team_b, g.team_a, aliases)),
        ]
        if any(pairs):
            return g
    return None


def _canonical_round(game: Dict[str, Any]) -> int:
    """Map game to a canonical round order (0=R64 .. 5=Championship) across all regions."""
    region = game.get("region", "")
    rnd = game.get("round", 0)
    if region == _FF_REGION:
        return 4 + rnd  # FF semis=4, championship=5
    return rnd


def _compute_tournament_status(games: List[Dict[str, Any]]) -> str:
    """Derive a human-readable tournament status from actual game data."""
    if not games:
        return "Not started"

    live_count = sum(1 for g in games if g.get("status") == "live")
    final_count = sum(1 for g in games if g.get("status") == "final")
    upcoming_count = sum(1 for g in games if g.get("status") == "upcoming")

    rounds_with_games: Dict[int, Dict[str, int]] = {}
    for g in games:
        cr = _canonical_round(g)
        if cr not in rounds_with_games:
            rounds_with_games[cr] = {"live": 0, "final": 0, "upcoming": 0}
        st = g.get("status", "upcoming")
        if st in rounds_with_games[cr]:
            rounds_with_games[cr][st] += 1

    active_round = max(rounds_with_games.keys()) if rounds_with_games else 0
    for r in sorted(rounds_with_games.keys()):
        counts = rounds_with_games[r]
        if counts["live"] > 0 or counts["upcoming"] > 0:
            active_round = r
            break

    round_name = _ROUND_NAMES.get(active_round, f"Round {active_round}")
    counts = rounds_with_games.get(active_round, {})

    if live_count > 0:
        return f"{round_name} — {live_count} game{'s' if live_count != 1 else ''} live"

    if counts.get("final", 0) > 0 and counts.get("upcoming", 0) > 0:
        done = counts["final"]
        total = done + counts["upcoming"]
        return f"{round_name} — {done}/{total} complete"

    if counts.get("upcoming", 0) > 0 and counts.get("final", 0) == 0:
        return f"{round_name} — games upcoming"

    if final_count > 0 and upcoming_count == 0 and live_count == 0:
        return f"{round_name} — complete"

    return round_name


_FF_REGION = "FinalFour"


def _overlay_espn(
    games: List[Dict[str, Any]],
    espn_games: List[EspnGame],
    aliases: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Overlay ESPN scores onto a list of game dicts. Returns (merged_games, match_count)."""
    merged = []
    hits = 0
    for row in games:
        if not isinstance(row, dict):
            merged.append(row)
            continue
        ta = str(row.get("team_a", ""))
        tb = str(row.get("team_b", ""))
        eg = _find_matching_espn(ta, tb, espn_games, aliases)
        if not eg:
            merged.append(row)
            continue
        hits += 1
        a_on_espn_a = _team_matches(ta, eg.team_a, aliases)
        b_on_espn_b = _team_matches(tb, eg.team_b, aliases)
        if a_on_espn_a and b_on_espn_b:
            sa, sb = eg.score_a, eg.score_b
        elif _team_matches(ta, eg.team_b, aliases) and _team_matches(tb, eg.team_a, aliases):
            sa, sb = eg.score_b, eg.score_a
        else:
            merged.append(row)
            continue

        new_row = {**row, "score_a": sa, "score_b": sb, "status": eg.status}
        if eg.status == "final":
            if eg.winner:
                if _team_matches(ta, eg.winner, aliases) or ta.lower() in eg.winner.lower():
                    new_row["winner"] = ta
                elif _team_matches(tb, eg.winner, aliases) or tb.lower() in eg.winner.lower():
                    new_row["winner"] = tb
                else:
                    new_row["winner"] = ta if sa >= sb else tb
            else:
                new_row["winner"] = ta if sa > sb else tb if sb > sa else row.get("winner", "")
        elif eg.status == "live":
            new_row["winner"] = row.get("winner", "") or ""
        else:
            new_row["winner"] = row.get("winner", "")
        if eg.detail:
            new_row["status_detail"] = eg.detail
        if eg.game_date:
            new_row["game_date"] = eg.game_date
        merged.append(new_row)
    return merged, hits


def _winner_info(game: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Return (winner_name, winner_seed) if game is final, else None."""
    if game.get("status") != "final" or not game.get("winner"):
        return None
    w = game["winner"]
    if w == game.get("team_a"):
        return w, game.get("seed_a", 0)
    if w == game.get("team_b"):
        return w, game.get("seed_b", 0)
    return w, 0


def _auto_advance(
    games: List[Dict[str, Any]],
    espn_games: List[EspnGame],
    aliases: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Auto-generate next-round entries from completed game pairs.
    Within each region: pairs are (0,1), (2,3), (4,5), (6,7) → next round indices 0-3.
    Rounds 0-3 are intra-region. Round 4 = Final Four (East vs Midwest, West vs South).
    Round 5 = Championship.
    """
    extra_hits = 0

    for _ in range(6):
        existing = {(g["region"], g["round"], g["game_index"]) for g in games if isinstance(g, dict)}
        lookup: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        for g in games:
            if isinstance(g, dict):
                lookup[(g["region"], g["round"], g["game_index"])] = g

        new_entries: List[Dict[str, Any]] = []

        regions = {g["region"] for g in games if isinstance(g, dict) and g.get("region")}
        for region in regions:
            if region == _FF_REGION:
                continue
            for rnd in range(4):
                n_games = 8 >> rnd
                for pair_start in range(0, n_games, 2):
                    next_rnd = rnd + 1
                    next_gi = pair_start // 2
                    if next_rnd <= 3 and (region, next_rnd, next_gi) in existing:
                        continue
                    g1 = lookup.get((region, rnd, pair_start))
                    g2 = lookup.get((region, rnd, pair_start + 1))
                    if not g1 or not g2:
                        continue
                    w1 = _winner_info(g1)
                    w2 = _winner_info(g2)
                    if not w1 or not w2:
                        continue

                    if next_rnd <= 3:
                        new_entries.append({
                            "region": region,
                            "round": next_rnd,
                            "game_index": next_gi,
                            "team_a": w1[0],
                            "team_b": w2[0],
                            "seed_a": w1[1],
                            "seed_b": w2[1],
                            "score_a": 0,
                            "score_b": 0,
                            "winner": "",
                            "status": "upcoming",
                        })

        # Final Four semis: round=0 gi=0 (East vs Midwest), round=0 gi=1 (West vs South)
        # Matches vectorized_sim forced key format: FinalFour:0:0, FinalFour:0:1
        for gi, (r1, r2) in enumerate([("East", "Midwest"), ("West", "South")]):
            if (_FF_REGION, 0, gi) in existing:
                continue
            c1 = lookup.get((r1, 3, 0))
            c2 = lookup.get((r2, 3, 0))
            if not c1 or not c2:
                continue
            w1 = _winner_info(c1)
            w2 = _winner_info(c2)
            if not w1 or not w2:
                continue
            new_entries.append({
                "region": _FF_REGION,
                "round": 0,
                "game_index": gi,
                "team_a": w1[0],
                "team_b": w2[0],
                "seed_a": w1[1],
                "seed_b": w2[1],
                "score_a": 0,
                "score_b": 0,
                "winner": "",
                "status": "upcoming",
            })

        # Championship: round=1 gi=0 — matches FinalFour:1:0
        if (_FF_REGION, 1, 0) not in existing:
            sf1 = lookup.get((_FF_REGION, 0, 0))
            sf2 = lookup.get((_FF_REGION, 0, 1))
            if sf1 and sf2:
                w1 = _winner_info(sf1)
                w2 = _winner_info(sf2)
                if w1 and w2:
                    new_entries.append({
                        "region": _FF_REGION,
                        "round": 1,
                        "game_index": 0,
                        "team_a": w1[0],
                        "team_b": w2[0],
                        "seed_a": w1[1],
                        "seed_b": w2[1],
                        "score_a": 0,
                        "score_b": 0,
                        "winner": "",
                        "status": "upcoming",
                    })

        if not new_entries:
            break

        overlaid, hits = _overlay_espn(new_entries, espn_games, aliases)
        extra_hits += hits
        games = games + overlaid

    return games, extra_hits


def merge_static_with_espn(static_payload: Dict[str, Any], espn_games: List[EspnGame]) -> Dict[str, Any]:
    """
    Overlay live/final scores from ESPN onto static bracket rows when both teams match.
    Then auto-generate next-round entries from completed pairs and overlay those too.
    """
    aliases = _load_aliases()
    games = static_payload.get("games")
    if not isinstance(games, list):
        return static_payload

    merged, live_hits = _overlay_espn(games, espn_games, aliases)
    merged, extra_hits = _auto_advance(merged, espn_games, aliases)
    live_hits += extra_hits

    out = {**static_payload, "games": merged}
    out["tournament_status"] = _compute_tournament_status(merged)
    out["last_updated"] = datetime.now(timezone.utc).isoformat()
    out["live_scores_source"] = "espn"
    out["live_scores_matched"] = live_hits
    return out
