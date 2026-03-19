# Bracket Simulator

Monte Carlo bracket simulator for the NCAA Tournament. Lock picks, import real results, run thousands of simulations, and explore win probabilities, upset candidates, and bracket outcomes — all with live streaming updates.

## Features

- **Monte Carlo simulation** — Run 1k–50k full-bracket simulations (R64 through champion) with live SSE streaming and real-time hero updates
- **Vectorized simulation engine** — NumPy-accelerated simulation with latent team-strength perturbations in log-odds space for fast, accurate results
- **Live results ticker** — ESPN-style scrolling banner showing real tournament scores with upset highlighting. One-click import locks completed games as forced picks so you can simulate the remaining bracket from reality.
- **Lock picks** — Click any team to lock a game winner; the sim respects your picks and simulates everything else. Up to 64 forced picks supported.
- **Team strength adjustments** — Elo deltas (±250) per team to model injuries, hot streaks, or gut feelings
- **Variance control** — Slider from "favorites dominate" to "upsets galore" to tune how much game-day randomness each run includes
- **First Four** — Play-in games simulated before the main bracket
- **Bracket value picks** — Mid-seeds and underdogs whose sim odds significantly beat their historical seed average
- **Upset watch** — R64 upset candidates ranked by probability with historical context
- **Head-to-head analysis** — Pick any two teams for win probability, signal breakdown, historical comps, and what-if scenarios (injuries, pace, hot streaks)
- **Model vs Vegas** — Side-by-side comparison of sim championship odds vs pre-tournament betting market futures
- **Visualizations** — Championship odds chart, round-by-round advancement heatmap, region difficulty, full advancement table with hover tooltips
- **Sim phase tracking** — Elapsed timer freezes when sims finish; status shows "Presenting results…" during bracket animation

## Data

- **Teams 2026** — Real 2026 NCAA Tournament field from Selection Sunday (KenPom rankings, betting odds/point spreads, NET rankings, records, conferences, coaching history, roster metrics)

- **Historical games (2005–2025)** — NCAA Tournament game logs with efficiency stats, Elo ratings, closing market probabilities, and derived features

- **Real results** — `backend/data/real_results.json` stores completed tournament games. Update this file as the tournament progresses; the ticker and import feature read from it automatically.

- **Modeling** — Features frozen at Selection Sunday (no look-ahead bias). 3-model ensemble (Logistic Regression + XGBoost + LightGBM) with isotonic calibration, trained on historical data via rolling-origin cross-validation.

## Architecture

- **Backend** — FastAPI + SSE streaming. ML pipeline (scikit-learn, XGBoost, LightGBM) for calibrated game probabilities. Vectorized NumPy simulation engine for speed. Rate limiting on simulation and model-log endpoints.

- **Frontend** — Next.js 14 + React 18. Live results ticker, bracket pick UI, real-time simulation progress, charts, and collapsible insights panels. Phase-aware elapsed timer.

- **Simulation** — Monte Carlo runs with latent team-strength draws (logit-space perturbation via `expit`), matchup caching, and forced picks. Outputs predicted bracket, champion/Final Four/advancement odds, and upset detection.

## Tech Stack

**Backend**
- FastAPI
- Python 3.12
- uvicorn
- scipy, numpy, pandas
- scikit-learn, xgboost, lightgbm
- joblib

**Frontend**
- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Vitest

**Deployment**
- Render (frontend and backend)

## Live Demo

- Frontend: https://bracket-simulator-vq00.onrender.com
- API: https://bracketedge-api.onrender.com

## Running Locally

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn main:app --reload --port 8001
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The frontend connects to `http://localhost:8001` by default (configured in `frontend/.env.local`).

## Updating Real Results

As the tournament progresses, edit `backend/data/real_results.json` with completed game data:

```json
{
  "region": "East", "round": 0, "game_index": 0,
  "team_a": "Duke", "team_b": "Siena",
  "seed_a": 1, "seed_b": 16,
  "score_a": 82, "score_b": 58,
  "winner": "Duke", "status": "final"
}
```

The ticker will display them automatically. Click "Lock results as picks" to import all completed games as forced picks, then re-run the simulation to project the remaining bracket from real data.

## Testing

**Backend**

```bash
cd backend
pytest
```

**Frontend**

```bash
cd frontend
npm test
```

## CI/CD

- GitHub Actions runs backend `pytest`, frontend `vitest`, and a production frontend build on pull requests and pushes to `main`.
- Render auto-deploys from the connected GitHub repository after `main` is updated.

## Environment Variables

**Backend**
- `SIM_RATE_LIMIT_MAX_REQUESTS` — Max sim requests per window (default: 6)
- `SIM_RATE_LIMIT_WINDOW_SEC` — Sim rate limit window in seconds (default: 60)
- `MODEL_LOG_RATE_LIMIT_MAX_REQUESTS` — Max model-log requests per window (default: 2)
- `MODEL_LOG_RATE_LIMIT_WINDOW_SEC` — Model-log rate limit window in seconds (default: 300)
- `ENABLE_MODEL_LOG_REBUILD` — Set to `1`, `true`, or `yes` to allow model log rebuild at runtime

**Frontend**
- `NEXT_PUBLIC_API_URL` — API base URL (default: `http://localhost:8001`)
