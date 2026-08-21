# NSE Options Terminal

[![CI](https://github.com/DIPESHGOEL27/option-pricing-models/actions/workflows/ci.yml/badge.svg)](https://github.com/DIPESHGOEL27/option-pricing-models/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3%2B-green)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A live NSE (National Stock Exchange of India) options analytics terminal.
It pulls a real F&O option chain, solves implied volatility independently
from each strike's actual bid/ask quotes, and compares that against both
NSE's own published figure and a fitted volatility skew — so a strike that
looks mispriced relative to its neighbors is visible directly, not buried
in a raw quote table.

## Problem Statement

Retail options traders and students working with Indian market data have
no free, self-hostable tool that goes past raw quotes. NSE's own site
shows a chain with NSE's own implied volatility; it does not show whether
that IV is internally consistent (whether the market's own put-call parity
implies a different discount rate than the naive risk-free rate),
independently re-solve IV from the actual bid/ask to cross-check it, or
flag which strikes are priced rich or cheap relative to the rest of the
day's smile. Institutional tools that do this are not free, and general
options-pricing side projects are almost universally built around US
markets and yfinance, leaving NSE/BSE underserved.

## Use Cases

- **Reading the live NSE chain with more than a raw quote table**: see
  change-in-open-interest, put/call ratio, and max pain alongside the
  chain itself, for NIFTY, BANKNIFTY, FINNIFTY, and individual F&O stocks.
- **Cross-checking implied volatility**: this terminal solves IV
  independently (Newton-Raphson, falling back to Brent) from real bid/ask
  quotes and shows it next to NSE's own published IV for the same strike —
  agreement is a sanity check; disagreement is itself informative.
- **Spotting relative mispricing**: a quadratic curve is fit through the
  chain's solved IVs and each strike is flagged rich or cheap relative to
  that curve — a standard volatility relative-value technique, applied to
  real, live data.
- **Learning how option pricing infrastructure is actually built**: the
  codebase is a worked example of the difference between a solver that
  "runs without crashing" and one that's actually correct — see
  [Engineering Notes](#engineering-notes) below for two real bugs a naive
  implementation would have shipped silently.
- **API integration**: every feature is a JSON REST endpoint, independent
  of the bundled dashboard, for use from another application or a
  notebook.

## How It Works

For a chosen instrument and expiry:

1. Fetch the real NSE F&O chain (via [jugaad-data](https://github.com/jugaad-py/jugaad-data), an unofficial NSE scraper — the same class of dependency yfinance is for US data).
2. For each strike's out-of-the-money leg (the liquid side — see [Engineering Notes](#engineering-notes)), solve implied volatility from the real bid-ask mid.
3. Derive the risk-free rate the market is actually implying from put-call parity on real near-the-money quotes, rather than assuming the underlying pays no dividend.
4. Fit a quadratic curve through the solved IVs (log-moneyness vs. IV) and flag each strike's deviation from it as rich, cheap, or normal.
5. Report solved IV, NSE's own IV, and the delta between them, per strike — never picking one and hiding the other.

## Core Features

- **Live NSE F&O option chain**: NIFTY, BANKNIFTY, FINNIFTY, and
  individual stocks, with real change-in-open-interest, put/call ratio,
  and max pain.
- **Implied volatility engine** (`api/iv_engine.py`): per-strike IV solved
  from real quotes, with explicit convergence verification (neither
  underlying solver raises on failure to converge, so this wrapper
  re-prices at the solved value and confirms it), a moneyness/liquidity
  guard, and a vega-based low-confidence flag for strikes where the
  price-to-IV mapping is too flat to trust (deep OTM, little time left).
- **Put-call-parity rate derivation**: the effective risk-free rate is
  derived from the chain's own near-ATM quotes rather than a single
  external number, so the user never has to type or guess one.
- **Volatility skew**: a fitted curve across the chain's liquid strikes,
  with each strike flagged rich/cheap relative to it, and a term-structure
  view (`/api/india/skew/<symbol>`) across multiple real expiries.
- **Position quick-look**: click a strike to see its payoff diagram and
  Greeks (delta, gamma, theta, vega, rho — computed at that strike's
  solved IV, with units labeled).
- **Standard pricing models** for reference and comparison: Black-Scholes
  (with full, correctly put/call-branched Greeks), Binomial Tree, and
  Monte Carlo (GBM with antithetic variates, plus Heston stochastic
  volatility and Merton jump-diffusion).

## Engineering Notes

Two real bugs, found and fixed during this project's development, are
worth calling out because a naive implementation would have shipped either
silently:

- **Missing dividend/carry adjustment.** Solving IV independently for
  calls and puts at the same strike, using the raw risk-free rate,
  produced solved IVs that systematically disagreed by 1.5-2.5 volatility
  points — NIFTY has a real dividend/carry adjustment that a plain
  Black-Scholes formula (which assumes zero dividends) doesn't account
  for. The fix derives the market's own implied rate from put-call parity
  on real quotes instead of assuming one; call/put agreement at the same
  strike improved to within ~0.1 vol points, verified against the live
  chain.
- **Illiquid legs producing confident-looking noise.** Deep in-the-money
  legs are far less liquid than their out-of-the-money counterpart at the
  same strike (a real observed case: 1 contract of open interest on one
  side, ~1,500 on the other) — solving IV from the illiquid side produced
  numbers like 140% IV that looked like valid solver output but were
  actually bid-ask noise amplified by a tiny vega. The fix uses only the
  OTM leg for the skew fit (the standard market convention) and adds an
  explicit vega-based confidence check, rather than trusting every
  numerically-convergent solve.

The project's earlier history includes catching and removing a fabricated
validation metric (a hardcoded R² floor presented as a measured result) —
documented in full, including the real root cause and the genuine
measured numbers, in [MODEL_VALIDATION_GUIDE.md](MODEL_VALIDATION_GUIDE.md).

## Technology Stack

**Backend**: Python 3.12, Flask, NumPy, SciPy, pandas, Plotly (server-side
figure generation)

**Market data**: [jugaad-data](https://github.com/jugaad-py/jugaad-data)
(real NSE F&O option chains and RBI policy/G-Sec rates), yfinance (used
only where NSE data doesn't cover a need, e.g. historical return series
for risk calculations)

**Frontend**: vanilla ES modules (`api/static/js/`) loaded via
`<script type="module">`, no build step, no framework — no Node.js
toolchain or `package.json` required to run or deploy this project. Plotly.js
for charts.

**Testing**: pytest, 109 tests, zero network dependency (network calls are
mocked against a recorded live chain fixture) — see [Testing](#testing).

**Deployment**: Render (primary free target), Railway, Docker/Gunicorn.

## API

The backend exposes 15 REST endpoints under `/api/`:

- **Pricing**: `/api/calculate_black_scholes`, `/api/calculate_binomial`, `/api/monte_carlo` (model=gbm/heston/jump_diffusion), `/api/model_validation`
- **NSE market data**: `/api/india/option_chain/<symbol>` (the core endpoint — solved IV, skew flags, PCR, max pain, OI chart), `/api/india/skew/<symbol>` (IV term structure across real expiries), `/api/india_market_sentiment`, `/api/india/risk_free_rate`
- **US market data**: `/api/market_data/<symbol>`, `/api/volatility_smile`
- **Risk**: `/api/risk_metrics`, `/api/risk/portfolio_risk`, `/api/risk/dynamic_hedging`, `/api/stress_test`
- **Analytics**: `/api/plot_payoff`
- **Platform**: `/api/status`

Every endpoint returns JSON and is usable independently of the dashboard.

## Getting Started

### Prerequisites

- Python 3.12
- Git

### Installation

```bash
git clone https://github.com/DIPESHGOEL27/option-pricing-models.git
cd option-pricing-models

python toggle_features.py full
pip install -r requirements.txt

python main.py
```

The dashboard is then available at `http://localhost:8000`.

For a lighter install (Black-Scholes, Binomial, and Monte Carlo pricing
only, no market data), use `python toggle_features.py minimal` instead.
See [DEPLOYMENT.md](DEPLOYMENT.md) for Render, Railway, Docker, and Vercel
deployment instructions.

## Testing

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest tests/ -v
ruff check api/ tests/ toggle_features.py main.py
```

109 tests across pricing math (reference values, put-call parity, every
Greek checked against its own finite difference), the Monte Carlo engine
(convergence, variance reduction), the IV engine (round-trip recovery
against both synthetic and real recorded chain data), chain analytics
(PCR, max pain, OI buildup), and the API layer (status codes, the error
contract, that no response ever contains invalid JSON). None of it depends
on live network access — see `tests/conftest.py` and
`tests/fixtures/nifty_chain.json`. Runs in CI on every push
(`.github/workflows/ci.yml`).

## Known Limitations

- `jugaad-data` scrapes an unofficial NSE endpoint; it can be slower or
  intermittently unavailable compared to a paid market-data feed, and the
  first request after an idle period is slower due to session bootstrap.
- Off-hours or on a stale connection, the chain reflects the last
  available NSE snapshot, not a live tick — this is disclosed in the UI
  rather than implied to be real-time.
- BSE has no free, working options-chain data source; this is shown
  explicitly rather than returning empty or fabricated data.
- The rich/cheap skew flag is a relative-value signal (deviation from a
  fitted curve through the day's own chain), not a prediction or an
  absolute mispricing claim.
- The hedging endpoint's transaction cost is a disclosed, fixed assumption
  (0.2% of hedge notional), not a live bid-ask spread.
- The bundled Flask development server (`python main.py`) is not a
  production WSGI server; deployments use Gunicorn (see
  [DEPLOYMENT.md](DEPLOYMENT.md)).

## Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) — Render, Railway, Docker, and Vercel deployment
- [MODEL_VALIDATION_GUIDE.md](MODEL_VALIDATION_GUIDE.md) — validation approach, and the historical record of a fabricated metric found and removed
- [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) — project layout

## License

MIT License — see [LICENSE](LICENSE).

## Author

Dipesh Goel — [LinkedIn](https://www.linkedin.com/in/dipeshgoel27/) —
[Portfolio](https://dipeshgoel.vercel.app/)
