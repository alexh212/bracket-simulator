# Bracket Simulator
 
Monte Carlo bracket simulator for the NCAA Tournament with a live results ticker and ensemble ML model.
 
## How it works
 
Lock in your picks, adjust the variance, and watch championship odds shift in real time. Run up to 50,000 full-bracket simulations — each one adds per-team strength noise so results vary realistically across runs. The percentages you see are how often each outcome actually happened across all the simulations. Import completed games from the ticker and simulate only what's left in the tournament.
 
## Architecture
 
- **Simulation engine** — vectorized NumPy simulation with latent team-strength perturbations in log-odds space, up to 50k full brackets per run
- **ML model** — Logistic Regression, XGBoost, and LightGBM ensemble with isotonic calibration, trained on tournament data back to 2005
- **Features** — KenPom efficiency, Elo ratings, betting lines, tempo, shooting splits, and 30+ engineered features, all frozen at selection time
- **Streaming** — FastAPI SSE streams simulation progress to the frontend in real time
- **Results ticker** — reads `/results` (static JSON and/or live merge); import final games as locked picks to simulate the remaining bracket
 
## Tech stack
 
FastAPI, Uvicorn, Pydantic, NumPy, Pandas, SciPy, scikit-learn, XGBoost, LightGBM, Joblib, Next.js, React, TypeScript, Tailwind, Vitest, SSE, GitHub Actions, Render
 
## Live demo
 
https://bracket-simulator-vq00.onrender.com
 
API: https://bracketedge-api.onrender.com
 
## Run locally
 
```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
uvicorn main:app --reload --port 8001
 
# Frontend (separate terminal)
cd frontend
npm install && npm run dev
```
 
Set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` if the frontend can't reach the API (default is `http://localhost:8001`).

### Live scores (demo-friendly default)

**`real_results.json`** still defines *which* matchups exist in your bracket. By default the API **merges in today’s (and adjacent days’) scores** from ESPN’s public scoreboard so the ticker can show **live and final** games during the tournament — no extra env on Render for a typical demo.

- **Turn off** outbound fetches / merge (static file only): `ENABLE_LIVE_SCORES=0`
- Matching uses **both team names** per row; extend **`backend/data/espn_team_aliases.json`** if a name doesn’t line up with ESPN.
- The site **polls `/results` every 60s** while the tab is open.

This is **best-effort**, not a licensed feed — fine for a personal demo; use a commercial provider if you need guarantees.
 
## Tests
 
```bash
cd backend && pytest
cd frontend && npm test
```
