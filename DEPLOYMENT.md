# Deployment Guide

## Render (primary, free)

This is the current recommended free deployment target: `render.yaml`
configures the full feature set from `requirements.txt` with no manual
setup.

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

## Vercel

`vercel.json` configures a `@vercel/python` build from `api/app.py`, which
auto-detects `requirements.txt` (there is no separate, reduced dependency
file — the full dependency set is light enough for Vercel's serverless
size limit now that ML pricing has been removed). This has not been
verified against an actual Vercel deployment; treat it as untested until
confirmed.

`NSELive`'s session/cookie bootstrap against nseindia.com adds real
latency on a cold serverless start; expect the first India-market request
after an idle period to be noticeably slower than subsequent ones.

```bash
npm i -g vercel
vercel
```

## Local development

```bash
pip install -r requirements.txt   # or: python toggle_features.py full
python main.py                    # serves on http://localhost:8000
```

`toggle_features.py [minimal|full]` regenerates `requirements.txt` --
`minimal` is core pricing math only (no market data, no charts), `full`
adds plotting and market data (yfinance, jugaad-data).

To run the test suite:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
```
