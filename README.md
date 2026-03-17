# Bracket Simulator

March Madness–style bracket simulator: lock first-round picks, run Monte Carlo simulations, and see win probabilities and bracket outcomes.

## Architecture

- **Backend** — FastAPI + SSE streaming; ML pipeline (scikit-learn, XGBoost, etc.) for game probabilities
- **Frontend** — Next.js + React; bracket UI, lock picks, live sim progress
- **Simulation** — Runs many brackets in parallel; predicted bracket, champion/F4 odds, upset watch

## Tech Stack

- **Backend:** FastAPI, Python 3.12, uvicorn, scipy, pandas, scikit-learn, xgboost, lightgbm
- **Frontend:** Next.js 14, React, TypeScript, Recharts, Tailwind CSS
- Deployed on Render

## Live Demo

- **Frontend:** https://bracket-simulator.onrender.com  
- **API:** https://bracketedge-api.onrender.com  

## Running Locally

**Backend**

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Frontend talks to `http://localhost:8001` by default. For a different API URL, set `NEXT_PUBLIC_API_URL`.

## Environment Variables

- **Backend:** none required for local run
- **Frontend (production):** `NEXT_PUBLIC_API_URL` — API base URL (e.g. `https://bracketedge-api.onrender.com`)
