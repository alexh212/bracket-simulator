"""Bracket Simulation — FastAPI backend with SSE streaming."""
import sys, os, time, json, random, threading, logging, math, re
from collections import defaultdict, deque
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from typing import Deque, DefaultDict, Dict, List, Optional

from data.teams_2026 import TEAMS_2026, FIRST_FOUR_GAMES, FIRST_FOUR_TEAMS
from services.simulation import SimulationConfig, Team

logger = logging.getLogger("bracket_api")

ROOT_DIR = Path(__file__).resolve().parent
REPORT_PATH = ROOT_DIR / "reports" / "model_pipeline_results.json"

MAX_N_SIMS = 50_000
MAX_JSON_PARAM_LEN = 10_000
MAX_FORCED_PICKS = 64
MAX_OVERRIDES = 68
SIM_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("SIM_RATE_LIMIT_MAX_REQUESTS", "6"))
SIM_RATE_LIMIT_WINDOW_SEC = int(os.getenv("SIM_RATE_LIMIT_WINDOW_SEC", "60"))
MODEL_LOG_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("MODEL_LOG_RATE_LIMIT_MAX_REQUESTS", "2"))
MODEL_LOG_RATE_LIMIT_WINDOW_SEC = int(os.getenv("MODEL_LOG_RATE_LIMIT_WINDOW_SEC", "300"))
ALLOW_MODEL_LOG_REBUILD = os.getenv("ENABLE_MODEL_LOG_REBUILD", "").lower() in {"1", "true", "yes"}
_FORCED_PICK_KEY_RE = re.compile(r"^(East|West|Midwest|South|FinalFour):\d+:\d+$")

app = FastAPI(title="Bracket Simulation API", version="4.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bracket-simulator.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https://bracket-simulator(?:-[a-z0-9]+)?\.onrender\.com$",
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_VALID = {f.name for f in Team.__dataclass_fields__.values()}
CLEAN  = {k:{fk:fv for fk,fv in d.items() if fk in _VALID} for k,d in TEAMS_2026.items()}
_ALL_TEAM_NAMES = set(CLEAN) | set(FIRST_FOUR_TEAMS)
_NUMERIC_SCENARIO_FIELDS = {
    name for name, field_info in Team.__dataclass_fields__.items()
    if field_info.type in {int, float}
}

_model_log_lock = threading.Lock()
_model_log_cache: Optional[str] = None
_rate_limit_lock = threading.Lock()
_rate_limit_buckets: Dict[str, DefaultDict[str, Deque[float]]] = {
    "simulate_stream": defaultdict(deque),
    "model_log": defaultdict(deque),
}


def _normalize_team_name(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("Team name required")
    if len(v) > 100:
        raise ValueError("Team name too long")
    return v


def _client_key(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(
    request: Request,
    bucket: str,
    max_requests: int,
    window_sec: int,
) -> None:
    client = _client_key(request)
    if client in {"127.0.0.1", "::1", "localhost"}:
        return

    now = time.time()
    with _rate_limit_lock:
        hits = _rate_limit_buckets[bucket][client]
        while hits and now - hits[0] > window_sec:
            hits.popleft()
        if len(hits) >= max_requests:
            raise HTTPException(status_code=429, detail="Rate limit exceeded, please retry shortly")
        hits.append(now)


def _parse_team_overrides(raw_value: str) -> Dict[str, float]:
    try:
        raw = json.loads(raw_value) if raw_value else {}
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="team_overrides must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="team_overrides must be an object")
    if len(raw) > MAX_OVERRIDES:
        raise HTTPException(status_code=400, detail=f"Too many team overrides (max {MAX_OVERRIDES})")

    parsed: Dict[str, float] = {}
    for team_name, value in raw.items():
        if team_name not in CLEAN:
            raise HTTPException(status_code=400, detail=f"Unknown override team: {team_name}")
        if isinstance(value, bool):
            raise HTTPException(status_code=400, detail=f"Invalid override for {team_name}")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Override for {team_name} must be numeric") from exc
        if not math.isfinite(numeric_value):
            raise HTTPException(status_code=400, detail=f"Override for {team_name} must be finite")
        parsed[team_name] = max(-250.0, min(250.0, numeric_value))
    return parsed


def _parse_forced_picks(raw_value: str) -> Dict[str, str]:
    try:
        raw = json.loads(raw_value) if raw_value else {}
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="forced_picks must be valid JSON") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="forced_picks must be an object")
    if len(raw) > MAX_FORCED_PICKS:
        raise HTTPException(status_code=400, detail=f"Too many forced picks (max {MAX_FORCED_PICKS})")

    parsed: Dict[str, str] = {}
    for pick_key, winner_name in raw.items():
        if not isinstance(pick_key, str) or not _FORCED_PICK_KEY_RE.match(pick_key):
            raise HTTPException(status_code=400, detail=f"Invalid forced pick key: {pick_key}")
        if not isinstance(winner_name, str) or len(winner_name) > 100:
            raise HTTPException(status_code=400, detail=f"Invalid forced pick winner for {pick_key}")
        if winner_name not in _ALL_TEAM_NAMES:
            raise HTTPException(status_code=400, detail=f"Unknown forced pick team: {winner_name}")
        parsed[pick_key] = winner_name
    return parsed


def _format_model_report_log(report: Dict) -> str:
    lines = [
        "Precomputed model summary",
        "Generated from backend/reports/model_pipeline_results.json",
        "",
        f"model_version: {report.get('model_version', 'unknown')}",
    ]

    final_metrics = report.get("final_metrics", {})
    if isinstance(final_metrics, dict) and final_metrics:
        lines.append("")
        lines.append("Final metrics")
        for model_name, metrics in sorted(final_metrics.items()):
            if not isinstance(metrics, dict):
                continue
            lines.append(
                f"- {model_name}: "
                f"log_loss={metrics.get('log_loss', 'n/a')} "
                f"brier={metrics.get('brier', 'n/a')} "
                f"accuracy={metrics.get('accuracy', 'n/a')} "
                f"ece={metrics.get('ece', 'n/a')}"
            )

    benchmarks = report.get("benchmarks", {})
    if isinstance(benchmarks, dict) and benchmarks:
        lines.append("")
        lines.append("Benchmarks")
        for key, value in sorted(benchmarks.items()):
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)


def _load_model_log_cache() -> str:
    if REPORT_PATH.exists():
        try:
            with REPORT_PATH.open() as f:
                report = json.load(f)
            return _format_model_report_log(report)
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to read precomputed model report")

    if ALLOW_MODEL_LOG_REBUILD:
        try:
            import io, contextlib
            from pipeline.model_pipeline import run_model_pipeline

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_model_pipeline(save=False, verbose=True)
            return buf.getvalue()
        except Exception:
            logger.exception("Model pipeline failed")
            return "Model training failed. Check server logs."

    return (
        "Model log rebuild is disabled in this deployment.\n"
        "Run the training pipeline locally to regenerate reports."
    )


class MatchupRequest(BaseModel):
    team_a: str
    team_b: str
    overrides: Dict[str, float] = Field(default_factory=dict)

    @field_validator("team_a", "team_b")
    @classmethod
    def team_must_be_short(cls, v: str) -> str:
        return _normalize_team_name(v)

    @field_validator("overrides")
    @classmethod
    def validate_overrides(cls, v: Dict[str, float]) -> Dict[str, float]:
        if len(v) > MAX_OVERRIDES:
            raise ValueError(f"Too many overrides (max {MAX_OVERRIDES})")
        cleaned: Dict[str, float] = {}
        for team_name, value in v.items():
            if team_name not in CLEAN:
                raise ValueError(f"Unknown override team: {team_name}")
            if isinstance(value, bool):
                raise ValueError(f"Invalid override for {team_name}")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Override for {team_name} must be finite")
            cleaned[team_name] = max(-250.0, min(250.0, numeric_value))
        return cleaned


class WhatIfRequest(BaseModel):
    team_a: str
    team_b: str
    scenarios: List[Dict]

    @field_validator("team_a", "team_b")
    @classmethod
    def normalize_teams(cls, v: str) -> str:
        return _normalize_team_name(v)

    @field_validator("scenarios")
    @classmethod
    def limit_scenarios(cls, v: list) -> list:
        if len(v) > 20:
            raise ValueError("Too many scenarios (max 20)")
        for scenario in v:
            if not isinstance(scenario, dict):
                raise ValueError("Each scenario must be an object")
            target = scenario.get("target", "a")
            label = scenario.get("label", "")
            if target not in {"a", "b", "both"}:
                raise ValueError("Scenario target must be one of: a, b, both")
            if label and (not isinstance(label, str) or len(label) > 80):
                raise ValueError("Scenario label must be a short string")

            unknown_fields = set(scenario) - {"label", "target"} - _NUMERIC_SCENARIO_FIELDS
            if unknown_fields:
                bad = ", ".join(sorted(unknown_fields))
                raise ValueError(f"Unsupported scenario fields: {bad}")

            update_fields = [k for k in scenario.keys() if k not in {"label", "target"}]
            if not update_fields:
                raise ValueError("Each scenario must modify at least one field")
            for field_name in update_fields:
                value = scenario[field_name]
                if isinstance(value, bool):
                    raise ValueError(f"Scenario field {field_name} must be numeric")
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Scenario field {field_name} must be numeric") from exc
                if not math.isfinite(numeric_value):
                    raise ValueError(f"Scenario field {field_name} must be finite")
        return v


@app.get("/health")
def health():
    return {"status": "ok", "teams": len(CLEAN)}

@app.get("/teams")
def teams():
    return CLEAN

@app.get("/model-info")
def model_info():
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
    request: Request,
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
    _enforce_rate_limit(request, "simulate_stream", SIM_RATE_LIMIT_MAX_REQUESTS, SIM_RATE_LIMIT_WINDOW_SEC)
    n_sims = max(100, min(n_sims, MAX_N_SIMS))
    emit_every = max(50, min(emit_every, n_sims))
    latent_sigma = max(0.0, min(latent_sigma, 0.20))

    if len(team_overrides) > MAX_JSON_PARAM_LEN:
        raise HTTPException(status_code=400, detail="team_overrides payload too large")
    if len(forced_picks) > MAX_JSON_PARAM_LEN:
        raise HTTPException(status_code=400, detail="forced_picks payload too large")

    from services.streaming_sim import run_streaming_simulation

    parsed_overrides = _parse_team_overrides(team_overrides)
    parsed_forced_picks = _parse_forced_picks(forced_picks)

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
        except Exception:
            logger.exception("Simulation stream error")
            yield f"data: {json.dumps({'type': 'error', 'message': 'Simulation failed'})}\n\n"

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
    return {
        "team_a": req.team_a,
        "team_b": req.team_b,
        "score_note": "Synthetic score estimate for presentation only; use win probability as the model output.",
        **result,
    }

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
        total_em = sum(v.get("kenpom_adj_off", 110) - v.get("kenpom_adj_def", 103) for v in region_teams.values())
        avg_em = total_em / len(region_teams) if region_teams else 0
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
def model_log(request: Request):
    _enforce_rate_limit(request, "model_log", MODEL_LOG_RATE_LIMIT_MAX_REQUESTS, MODEL_LOG_RATE_LIMIT_WINDOW_SEC)
    global _model_log_cache
    with _model_log_lock:
        if _model_log_cache is not None:
            return {"log": _model_log_cache}

    with _model_log_lock:
        if _model_log_cache is not None:
            return {"log": _model_log_cache}
        _model_log_cache = _load_model_log_cache()
    return {
        "log": _model_log_cache,
        "rebuild_enabled": ALLOW_MODEL_LOG_REBUILD,
        "source": "precomputed_report" if REPORT_PATH.exists() else "runtime_message",
    }
