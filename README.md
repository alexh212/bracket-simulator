# Bracket Simulator

Simulate the NCAA tournament thousands of times and see win odds. Pick a team to lock a game, slide variance up or down, and watch the numbers change.

**Model:** LR + XGBoost + LightGBM ensemble with isotonic calibration, trained on ~20 years of NCAA Tournament games. Inputs include KenPom efficiency, Elo ratings, betting market odds, tempo, shooting splits, and ~30 engineered features — all frozen at selection time (no look-ahead).

**Sim:** Up to 50k full brackets per run, vectorized with NumPy for speed. Each run adds per-team strength noise so results vary realistically across sims.

**Insights after a run:** championship odds, round-by-round advancement, upset watch, model vs Vegas, region difficulty, bracket value picks (mid-seeds beating their historical seed average), head-to-head matchup analysis, and what-if scenarios.

**Built with:** Python (FastAPI) + Next.js. Streams progress to the UI while sims run.

**Scores:** The ticker reads `backend/data/real_results.json`. Update it when games finish, refresh, then import as locked picks to sim only what's left.

**Model file:** `backend/models/model_v1_best.pkl` is not in git. Without it the app falls back to a simpler blend (still works).

## Try it

- https://bracket-simulator-vq00.onrender.com  
- API: https://bracketedge-api.onrender.com  

## Run on your machine

```bash
cd backend && python -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt && uvicorn main:app --reload --port 8001

cd frontend && npm install && npm run dev
```

If the frontend can’t reach the API, set `NEXT_PUBLIC_API_URL` in `frontend/.env.local` (default is `http://localhost:8001`).

## Tests

Backend: `cd backend && pytest`  
Frontend: `cd frontend && npm test`  
Pushes to `main` also run tests and a production build on GitHub Actions.
