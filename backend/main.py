"""Bracket Simulation — FastAPI backend with SSE streaming."""
import sys, os, time, json, random
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from data.teams_2026 import TEAMS_2026, FIRST_FOUR_GAMES, FIRST_FOUR_TEAMS
from services.simulation import SimulationConfig, Team

app = FastAPI(title="Bracket Simulation API", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_VALID = {f.name for f in Team.__dataclass_fields__.values()}
CLEAN  = {k:{fk:fv for fk,fv in d.items() if fk in _VALID} for k,d in TEAMS_2026.items()}

_MODEL_LOG = None


class MatchupRequest(BaseModel):
    team_a: str
    team_b: str
    overrides: Dict[str, float] = Field(default_factory=dict)

class WhatIfRequest(BaseModel):
    team_a: str
    team_b: str
    scenarios: List[Dict]


@app.get("/health")
def health():
    return {"status": "ok", "teams": len(CLEAN)}

@app.get("/teams")
def teams():
    return CLEAN

@app.get("/model-info")
def model_info():
    """
    Lightweight provenance endpoint for frontend transparency.
    """
    from data.historical.tournament_games import load_dataframe
    from pipeline.model_pipeline import CORE_FEATURES, MARKET_FEATURES

    hist = load_dataframe()
    seasons = sorted(hist["season"].unique().tolist())
    max_s = float(hist["season"].max()) if len(hist) else 0.0
    age = max_s - hist["season"].astype(float).values if len(hist) else []
    weights = [max(0.25, min(1.0, float(0.5 ** (a / 4.0)))) for a in age]
    by_season = (
        hist.assign(_w=weights)
        .groupby("season")["_w"]
        .mean()
        .round(3)
        .to_dict()
    )
    return {
        "inference_data": {
            "dataset": "TEAMS_2026",
            "season": 2026,
            "selection_sunday_freeze": "2026-03-15",
            "teams_in_bracket": len(CLEAN),
        },
        "training_data": {
            "dataset": "historical.tournament_games",
            "season_min": int(min(seasons)) if seasons else None,
            "season_max": int(max(seasons)) if seasons else None,
            "seasons": seasons,
            "rows": int(len(hist)),
            "recency_weighting": "half-life 4 seasons, clipped [0.25,1.00]",
            "season_weight_summary": {str(int(k)): float(v) for k, v in by_season.items()},
        },
        "model_details": {
            "stack": ["logistic_regression", "xgboost", "lightgbm", "meta_logistic"],
            "calibration": "isotonic",
            "core_feature_count": len(CORE_FEATURES),
            "market_feature_count": len(MARKET_FEATURES),
            "market_usage": "used in blend/meta features; recency weighted season training",
            "variance_modeling": "latent team strength draws per simulation",
        },
    }


# ── SSE streaming simulation ───────────────────────────────────────────────────
@app.get("/simulate/stream")
def simulate_stream(
    n_sims: int = 10_000,
    emit_every: int = 250,
    latent_sigma: float = 0.06,
    team_overrides: str = Query(default="{}", description="JSON map of {team_name: elo_delta}"),
    forced_picks: str = Query(default="{}", description="JSON map of {'Region:round:game': winner_name}"),
):
    """
    Server-Sent Events stream.
    Emits: start → progress (every emit_every sims) → complete
    Each run uses a random seed so results differ between runs.
    """
    from services.streaming_sim import run_streaming_simulation

    parsed_overrides: Dict[str, float] = {}
    try:
        raw = json.loads(team_overrides) if team_overrides else {}
        if isinstance(raw, dict):
            for k, v in raw.items():
                if k in CLEAN:
                    try:
                        # Clamp to a sane tournament adjustment range.
                        parsed_overrides[k] = max(-250.0, min(250.0, float(v)))
                    except Exception:
                        continue
    except Exception:
        parsed_overrides = {}

    parsed_forced_picks: Dict[str, str] = {}
    try:
        raw_forced = json.loads(forced_picks) if forced_picks else {}
        if isinstance(raw_forced, dict):
            for k, v in raw_forced.items():
                if isinstance(k, str) and isinstance(v, str):
                    parsed_forced_picks[k] = v
    except Exception:
        parsed_forced_picks = {}

    def event_stream():
        try:
            for event in run_streaming_simulation(
                CLEAN,
                n_sims=n_sims,
                emit_every=emit_every,
                latent_sigma=latent_sigma,
                team_overrides=parsed_overrides,
                forced_picks=parsed_forced_picks,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


# ── Matchup / WhatIf (still synchronous — fast) ────────────────────────────────
@app.post("/matchup")
def matchup(req: MatchupRequest):
    if req.team_a not in CLEAN: return {"error": f"Unknown: {req.team_a}"}
    if req.team_b not in CLEAN: return {"error": f"Unknown: {req.team_b}"}
    from pipeline.calibrated_game_model import get_game_model
    gm = get_game_model()
    ta = {**CLEAN[req.team_a], "name": req.team_a, **req.overrides}
    tb = {**CLEAN[req.team_b], "name": req.team_b}
    result = gm.predict_with_breakdown(ta, tb)
    prob = result["win_prob_a"] / 100
    a_favored = prob >= 0.5
    win_prob = max(prob, 1 - prob)
    rng = random.Random()
    base = 65 + rng.randint(0, 15)
    margin = max(1, int(abs(rng.gauss(win_prob * 18, 7))))
    if a_favored:
        result["score_a"] = base + margin // 2
        result["score_b"] = base - margin // 2
    else:
        result["score_a"] = base - margin // 2
        result["score_b"] = base + margin // 2
    return {"team_a": req.team_a, "team_b": req.team_b, **result}

@app.post("/whatif")
def whatif(req: WhatIfRequest):
    if req.team_a not in CLEAN: return {"error": f"Unknown: {req.team_a}"}
    if req.team_b not in CLEAN: return {"error": f"Unknown: {req.team_b}"}
    from pipeline.phases_10_12 import what_if_analysis
    ta = {**CLEAN[req.team_a], "name": req.team_a}
    tb = {**CLEAN[req.team_b], "name": req.team_b}
    return {"results": what_if_analysis(ta, tb, req.scenarios)}

@app.get("/comps/{team_a}/{team_b}")
def historical_comps(team_a: str, team_b: str):
    if team_a not in CLEAN: return {"error": f"Unknown: {team_a}"}
    if team_b not in CLEAN: return {"error": f"Unknown: {team_b}"}
    from pipeline.phases_10_12 import find_historical_comps
    ta = {**CLEAN[team_a], "name": team_a}
    tb = {**CLEAN[team_b], "name": team_b}
    return find_historical_comps(ta, tb, n_comps=8)

@app.get("/disagreement/{team_a}/{team_b}")
def disagreement(team_a: str, team_b: str):
    if team_a not in CLEAN: return {"error": f"Unknown: {team_a}"}
    if team_b not in CLEAN: return {"error": f"Unknown: {team_b}"}
    from pipeline.phases_10_12 import ensemble_disagreement
    ta = {**CLEAN[team_a], "name": team_a}
    tb = {**CLEAN[team_b], "name": team_b}
    return ensemble_disagreement(ta, tb)

@app.get("/first-four")
def first_four():
    return {"games": FIRST_FOUR_GAMES, "teams": {k: v for k, v in FIRST_FOUR_TEAMS.items()}}


@app.get("/seed-history")
def seed_history():
    from data.teams_2026 import _SEED_WIN
    return _SEED_WIN


@app.get("/region-stats")
def region_stats():
    from services.simulation import REGIONS
    stats = []
    for region in REGIONS:
        region_teams = {k: v for k, v in TEAMS_2026.items() if v.get("region") == region}
        seeds = [v["seed"] for v in region_teams.values()]
        avg_seed = sum(seeds) / len(seeds) if seeds else 8.5
        total_em = sum(v.get("kenpom_adj_off", 110) - v.get("kenpom_adj_def", 103) for v in region_teams.values())
        avg_em = total_em / len(region_teams) if region_teams else 0
        top_elo = sorted(region_teams.values(), key=lambda t: t.get("elo_current", 0), reverse=True)
        top4 = [{"name": k, "seed": v["seed"], "elo": v.get("elo_current", 0),
                  "championship_odds_pct": v.get("championship_odds_pct", 0)}
                 for k, v in region_teams.items() if v["seed"] <= 4]
        top4.sort(key=lambda x: x["seed"])
        champ_sum = sum(v.get("championship_odds_pct", 0) for v in region_teams.values())
        stats.append({
            "region": region,
            "avg_em": round(avg_em, 1),
            "top_seeds": top4,
            "total_championship_pct": round(champ_sum, 1),
            "num_teams": len(region_teams),
        })
    return stats


@app.get("/model-log")
def model_log():
    global _MODEL_LOG
    if _MODEL_LOG is None:
        try:
            import io, contextlib
            from pipeline.model_pipeline import run_model_pipeline
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_model_pipeline(save=False, verbose=True)
            _MODEL_LOG = buf.getvalue()
        except Exception as e:
            _MODEL_LOG = f"Error: {e}"
    return {"log": _MODEL_LOG}
