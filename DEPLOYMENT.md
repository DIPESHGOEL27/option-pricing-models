# Deployment Guide

## Render (primary, free)

This is the current recommended free deployment target (`render.yaml`,
`requirements.txt`, full feature set including ML pricing — Render's native
Python runtime has no serverless size cap, unlike Vercel).

1. Go to [Render](https://render.com) and sign in with GitHub.
2. **New → Blueprint** → select this repository. Render auto-detects
   `render.yaml` and configures the service (build: `pip install -r
   requirements.txt`; start: `gunicorn api.app:app`; health check:
   `/api/status`) with no manual field-filling required.
3. Select the **Free** plan when prompted and click **Apply**.
4. No environment variables are required; `PORT` is set automatically by
   Render.

The free plan spins the service down after ~15 minutes of inactivity; the
next request wakes it, which takes roughly 30-50 seconds. This is a real
cold-start cost, not a bug — acceptable for a demo/portfolio deployment, not
for a low-latency production use case.

Once deployed, verify with:

```bash
curl https://<your-app>.onrender.com/api/status
```

## Railway

`railway.toml`, `nixpacks.toml`, and `Procfile` are also configured for
Railway and this project has previously run there. Railway's free tier is
usage-credit based and may be exhausted or unavailable depending on your
account's current credit balance — check [Railway.app](https://railway.app)
directly rather than assuming free capacity is available.

1. Go to [Railway.app](https://railway.app) and sign in with GitHub.
2. **New Project → Deploy from GitHub repo** → select this repository.
3. Railway builds via Nixpacks (Python 3.12.7, see `nixpacks.toml`) and
   starts the app with the command in `railway.toml`:
   `python main.py`, which reads `$PORT` and runs the Flask app via
   `api.app`.
4. No environment variables are required; `PORT` is set automatically by
   Railway.

Once deployed, verify with:

```bash
curl https://<your-app>.up.railway.app/api/status
```

## Docker

```bash
docker build -t option-pricing-platform .
docker run -p 8080:8080 option-pricing-platform
```

The `Dockerfile` installs `requirements.txt`, exposes port `8080`, and runs
`gunicorn api.app:app`.

## Vercel (limited support)

`vercel.json` and `requirements-vercel.txt` exist for a lightweight Vercel
deployment, but `requirements-vercel.txt` intentionally excludes
`scikit-learn`/`xgboost`/`joblib` to stay under Vercel's 250MB serverless
function limit. That means **the ML pricing endpoints
(`/api/ml/*`) will not work on Vercel** — only Black-Scholes, Binomial, and
Monte Carlo pricing are available there. Use Railway or Docker for the full
feature set, including ML pricing.

India/NSE market data (`jugaad-data`) is light enough to include in the
Vercel build and is expected to work there, but this has not been verified
with an actual Vercel deploy — treat it as untested until confirmed. Note
also that `NSELive`'s session/cookie bootstrap against nseindia.com adds
real latency on a cold serverless start; expect the first India-market
request after an idle period to be noticeably slower than subsequent ones.

`api/app.py` previously had an unconditional `import matplotlib.pyplot`
that was never actually called (the app renders all charts through Plotly)
but was not in `requirements-vercel.txt` — this would have crashed the app
at import time on Vercel. That dead import has been removed.

```bash
npm i -g vercel
vercel
```

## Local development

```bash
pip install -r requirements.txt   # or: python toggle_features.py full
python main.py                    # serves on http://localhost:8000
```

`toggle_features.py [minimal|light|full]` regenerates `requirements.txt` for
different deployment sizes — `full` is required for ML pricing
(`scikit-learn`, `xgboost`) and the other heavy dependencies.
