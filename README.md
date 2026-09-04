# bracket-simulator

A FastAPI + Next.js NCAA bracket simulator that runs a vectorized Monte Carlo engine over hand-authored 2026 bracket data.

**Status:** prototype
**API is live:** https://bracketedge-api.onrender.com (root, `/docs`, `/health`, `/simulate/stream`, `/results`, and other read endpoints returned 200 in testing).
**Frontend is down:** https://bracket-simulator-vq00.onrender.com returns 503 — the Render free-tier service is suspended. There is currently no working demo of the UI, only the API.

## The problem

Simulating a 64-team bracket by looping over games in Python doesn't scale to tens of thousands of runs. The part of this repo worth looking at is `services/vectorized_sim.py`: it runs up to 50,000 full brackets as batched NumPy array operations instead of a per-game loop, applies a correlated logit shift per simulation so variance stays realistic across rounds, and streams round-by-round progress to the frontend over SSE without blocking. The matchup-probability layer is walled off behind `pipeline/calibrated_game_model.py` so the simulation code never imports sklearn, XGBoost, or LightGBM directly — a clean seam, even though (see below) what sits behind that seam in production isn't what it claims to be.

## How it works

`backend/main.py` exposes `GET /simulate/stream` (SSE). It rate-limits requests, validates any team overrides or forced picks, then calls `services/streaming_sim.py:run_streaming_simulation()`, which calls `services/vectorized_sim.py` to run the batched simulation. Pairwise win probabilities come from `services/simulation.py:compute_matchup_prob()`, which tries `pipeline/calibrated_game_model.py:get_game_model()` and, if that raises, falls back to `_fallback_prob()` — a hand-tuned blend of KenPom-style efficiency margin, Elo, market odds, and seed prior. In the current deployment, that fallback is what actually runs every simulation: the trained model artifact (`backend/models/model_v1_best.pkl`) and its training report are gitignored and never produced by `render.yaml`'s build command or the CI workflow, so `get_game_model()` always raises `FileNotFoundError`. Team data comes from `backend/data/teams_2026.py`.

Four other endpoints — `/matchup`, `/whatif`, `/comps/{a}/{b}`, `/disagreement/{a}/{b}` — call `get_game_model()` directly instead of through the guarded wrapper in `services/simulation.py`, and have no try/except of their own. These are the endpoints the frontend's `insights.tsx` panel calls in parallel on load.

`GET /results` reads `backend/data/real_results.json`, a synthetic "completed" 2026 bracket, and merges in live ESPN scores when `ENABLE_LIVE_SCORES` is on. With no real games in season right now, that merge is a no-op and the endpoint just republishes the fake bracket with a fresh timestamp.

The frontend (`frontend/components/App.tsx`, 1,296 lines) renders sim controls, a pick/override panel, the bracket grid, an advancement table, and charts. `frontend/components/app/useSimulation.ts` owns the `EventSource` client against `/simulate/stream`.

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
# macOS only: brew install libomp
# without it, `import xgboost` fails at import time with a missing
# libomp.dylib error — hits both pytest and uvicorn
uvicorn main:app --reload --port 8001

# separate terminal
cd frontend
npm install && npm run dev
```

| Env var | Purpose |
|---|---|
| `ENABLE_LIVE_SCORES` | backend, default on. Set to `0` to disable the ESPN live-score merge in `/results` and serve only the static bracket. |
| `SIM_RATE_LIMIT_MAX_REQUESTS` / `SIM_RATE_LIMIT_WINDOW_SEC` | backend, default 6 / 60. Rate limit on `/simulate/stream`. |
| `MODEL_LOG_RATE_LIMIT_MAX_REQUESTS` / `MODEL_LOG_RATE_LIMIT_WINDOW_SEC` | backend, default 2 / 300. Rate limit on `/model-log`. |
| `ENABLE_MODEL_LOG_REBUILD` | backend, default off. If on and no precomputed report exists, `/model-log` tries to retrain the pipeline live. |
| `MODEL_SHA256` | backend, optional. Verifies the model artifact's checksum before loading — moot today since no artifact ships. |
| `NEXT_PUBLIC_API_URL` | frontend, default `http://localhost:8001`. `render.yaml` overrides this to the real API URL at deploy time; `frontend/.env.production` still ships a stale placeholder, so a local production build points nowhere real unless you set this yourself. |

## Tests

```bash
cd backend && pytest
cd frontend && npm test
```

CI actually runs `pytest tests/ --ignore=tests/test_simulation.py`, silently skipping one test file for no documented reason. Running the full suite locally, including that file (after `brew install libomp`), all 67 backend tests pass. The frontend suite is 2 test files that mock the network/SSE layer entirely — nothing renders the real `App.tsx` or exercises `insights.tsx`, which is the panel affected by the bug below.

## Known limitations

- The "ensemble ML model" (logistic regression + XGBoost + LightGBM, isotonic-calibrated) does not run in production. Its artifact and training report are gitignored and nothing in CI or the Render build produces them. Every simulation served today runs on the heuristic fallback in `services/simulation.py`, not the described ensemble. `GET /model-log` on the live API confirms this: "Model log rebuild is disabled in this deployment."
- `POST /matchup` throws an unhandled `FileNotFoundError` and returns a 500 on the first call after any process start or restart, reproduced live and locally. `get_game_model()` sets its module-level singleton before calling `.load()`, so the failed load leaves a permanently unfitted instance in place — every call after the first one silently succeeds on the fallback instead of erroring. The bug only ever surfaces on the one request most likely to be a demo or health check. `/whatif`, `/comps`, and `/disagreement` share the same unguarded call and the same exposure.
- The historical training dataset (`backend/data/historical/tournament_games.py`) is 87 hand-typed rows covering 2005–2025, not an ingested KenPom/Elo/market feed. It includes placeholder duplicate names ("Robert Morris2", "Omaha2") added to dodge dict-key collisions, and those placeholders leak into live output — `GET /comps/Duke/Siena` returns "Omaha2" as a historical comp.
- `backend/data/teams_2026.py`'s docstring claims real Selection Sunday data from KenPom, but AdjO/AdjD are computed from a rank-to-value formula, not measured efficiency numbers, and the cited source URL isn't KenPom's actual domain.
- The entire 2026 tournament this app is built around is fabricated: `real_results.json` is a synthetic "completed" bracket. `GET /results` currently republishes it with a fresh timestamp since there are no real games to merge right now (off-season).
- Frontend test coverage doesn't reach the code path with the confirmed bug above.
- `frontend/.env.production` still points at a placeholder URL, not the real API.
- `frontend/package.json` pins `next@14.2.5`, which npm flags with a known security vulnerability; `package-lock.json` hasn't been bumped past it.
- The frontend deployment is suspended (503, Render free-tier spin-down). Only the API is currently reachable.

## What I'd build next

- Make the model pipeline reproducible in the deployed environment — commit a trained artifact or have the build actually run training — and make `get_game_model()` catch its own `FileNotFoundError` so `/matchup`, `/whatif`, `/comps`, and `/disagreement` never 500 on cold start.
- Add a startup event that calls `get_game_model()` once at boot, so a cold-start failure is a logged event at deploy time instead of a client-facing 500 on whichever request happens to land first.
- Replace the 87-row hand-typed dataset with a real ingested history before making any accuracy or "trained since 2005" claims.
- Write tests that render `App.tsx` / `insights.tsx` against a mocked API, covering the matchup insights panel.
