# Bracket Simulator

March Madness–style bracket simulator: lock first-round picks, run Monte Carlo simulations, and see win probabilities and bracket outcomes.

Built in ~24 hours. There are some style quirks here and there, but I'm happy with how it came out.

I just wanted to make a good bracket despite not knowing a lot about basketball.

## Data

- Teams 2026  
  Real 2026 NCAA Tournament field built from Selection Sunday data  
  (KenPom rankings, betting odds/point spreads, NET rankings, records, conferences)

- Historical Games (2005–2024)  
  NCAA Tournament game logs with:
  - Efficiency stats  
  - Elo ratings  
  - Closing market probabilities  
  - Derived features  

- Modeling  
  - Features frozen at Selection Sunday (no look-ahead bias)  
  - Probabilities learned from historical data and applied to current bracket  

## Architecture

- Backend  
  FastAPI + SSE streaming  
  ML pipeline (scikit-learn, XGBoost, LightGBM) for game probabilities  

- Frontend  
  Next.js + React  
  Bracket UI, lock picks, live simulation progress  

- Simulation  
  - Runs large-scale Monte Carlo simulations  
  - Outputs:
    - Predicted bracket  
    - Champion / Final Four odds  
    - Upset detection  

## Tech Stack

Backend
- FastAPI
- Python 3.12
- uvicorn
- scipy
- pandas
- scikit-learn
- xgboost
- lightgbm

Frontend
- Next.js 14
- React
- TypeScript
- Tailwind CSS (used for global styles/build pipeline)

Deployment
- Render for both frontend and backend

## Live Demo

Frontend: https://bracket-simulator.onrender.com
API: https://bracketedge-api.onrender.com

## Running Locally

Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend connects to `http://localhost:8001` by default.
To override it, set `NEXT_PUBLIC_API_URL`.

## Environment Variables

Backend
- None required for local development

Frontend
- `NEXT_PUBLIC_API_URL` for pointing the UI at the API

## Notes

- `backend/models/model_v1_best.pkl` is a runtime artifact used for inference.
- `backend/reports/*.json` are generated reports and are not required to run the app.
- Render deployment is defined in `render.yaml`.
