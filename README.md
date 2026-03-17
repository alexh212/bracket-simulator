# Bracket Simulator

March Madness–style bracket simulator: lock picks, run Monte Carlo simulations, adjust team strength assumptions, and explore win probabilities, upset candidates, and bracket outcomes.

## Features

- **Monte Carlo simulation** — Run 1k–50k full-bracket simulations (R64 through champion) with live SSE streaming
- **Chalk vs chaos** — Variance slider to tune how predictable or chaotic results are
- **Lock picks** — Click to lock any game winner; supports up to 64 forced picks
- **Team assumptions** — Elo deltas (±250) per team to model injuries, form, etc.
- **First Four** — Play-in games simulated before the main bracket
- **Upset watch** — R64 upset candidates with probabilities and reasons
- **Visualizations** — Championship odds, advancement heatmap, upset chart, per-team advancement table
- **Insights** — Head-to-head analysis, model vs market, region difficulty, seed history, what-if scenarios

## Data

- **Teams 2026** — Real 2026 NCAA Tournament field built from Selection Sunday data (KenPom rankings, betting odds/point spreads, NET rankings, records, conferences)

- **Historical Games (2005–2024)** — NCAA Tournament game logs with efficiency stats, Elo ratings, closing market probabilities, and derived features

- **Modeling** — Features frozen at Selection Sunday (no look-ahead bias). ML stack (LR, XGBoost, LightGBM, meta-LR) with isotonic calibration, trained on historical data and applied to the current bracket

## Architecture

- **Backend** — FastAPI + SSE streaming. ML pipeline (scikit-learn, XGBoost, LightGBM) for calibrated game probabilities. Rate limiting on simulation and model-log endpoints.

- **Frontend** — Next.js 14 + React 18. Bracket UI, lock picks, live simulation progress, charts, and insights panels.

- **Simulation** — Monte Carlo runs with latent team strength draws, matchup caching, and forced picks. Outputs predicted bracket, champion/Final Four odds, and upset detection.

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

- Frontend: https://bracket-simulator.onrender.com
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

The frontend connects to `http://localhost:8001` by default.

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
- Render is expected to auto-deploy from the connected GitHub repository after `main` is updated.

## Environment Variables

**Backend**
- `SIM_RATE_LIMIT_MAX_REQUESTS` — Max sim requests per window (default: 6)
- `SIM_RATE_LIMIT_WINDOW_SEC` — Sim rate limit window in seconds (default: 60)
- `MODEL_LOG_RATE_LIMIT_MAX_REQUESTS` — Max model-log requests per window (default: 2)
- `MODEL_LOG_RATE_LIMIT_WINDOW_SEC` — Model-log rate limit window in seconds (default: 300)
- `ENABLE_MODEL_LOG_REBUILD` — Set to `1`, `true`, or `yes` to allow model log rebuild at runtime

**Frontend**
- `NEXT_PUBLIC_API_URL` — API base URL (default: `http://localhost:8001`)

## Notes

- `backend/models/model_v1_best.pkl` is the runtime model artifact used for inference.
- `backend/reports/model_pipeline_results.json` is a generated report; not required to run the app.
- Render deployment is defined in `render.yaml`.
