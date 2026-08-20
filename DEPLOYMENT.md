# Deployment Guide

## Railway (primary)

This is the platform this project actually deploys to (`railway.toml`,
`nixpacks.toml`, `Procfile`).

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

### Alternative: Render / Heroku-style platforms

`Procfile` (`web: gunicorn api.app:app --bind 0.0.0.0:$PORT --workers 1
--timeout 120 --preload`) targets Procfile-based platforms like Render or
Heroku, as an alternative to Railway's own `railway.toml` start command.

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
