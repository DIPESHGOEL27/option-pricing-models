# Advanced Option Pricing Platform

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3%2B-green)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Flask-based option pricing and risk analysis platform covering the US and
Indian (NSE/BSE) equity derivatives markets. It implements standard and
advanced pricing models (Black-Scholes, Binomial Trees, Monte Carlo,
Heston, Jump-Diffusion), a machine-learning pricing ensemble, a portfolio
risk suite (VaR, Expected Shortfall, stress testing, delta hedging), and
live market data integration, all exposed through a REST API and an
interactive web dashboard.

## Problem Statement

Pricing and risk tools for options are either locked behind expensive
institutional platforms (Bloomberg, Refinitiv) or scattered across
disconnected scripts and notebooks that compute a single price with no
surrounding risk context. Retail traders, students, and engineers evaluating
quantitative finance workflows generally have no single, free, self-hostable
place to:

- price the same option under multiple models and compare the results,
- see how a machine-learning pricer performs against closed-form and
  simulation-based benchmarks,
- evaluate portfolio-level risk (VaR, Expected Shortfall, stress scenarios,
  hedging) using real historical data rather than assumed constants, and
- do all of the above for Indian market instruments (NSE/BSE), where free
  tooling is considerably sparser than for US markets.

This project addresses that gap: one Flask application, backed by real
market data where real data is available, with every model's assumptions
and data source documented rather than hidden behind a number.

## Use Cases

- **Learning and teaching quantitative finance**: compare Black-Scholes,
  Binomial, and Monte Carlo prices for the same contract, inspect Greeks,
  and see Monte Carlo convergence as simulation count increases.
- **Model prototyping**: use the ML pricing ensemble (neural network,
  random forest, XGBoost) and volatility forecasting as a reference
  implementation, including a documented example of a real accuracy bug
  (see [Model Validation](MODEL_VALIDATION_GUIDE.md)) and how it was
  diagnosed and fixed.
- **Portfolio risk analysis**: enter option positions, compute historical
  and parametric VaR, Expected Shortfall, and stress-test results derived
  from real historical returns for the underlying symbols.
- **Indian market analysis**: pull real NSE F&O option chains, Put/Call
  Ratio, max pain, and RBI policy/G-Sec rates for NIFTY, BANKNIFTY, and
  individual NSE-listed stocks.
- **API integration**: every feature is available as a JSON REST endpoint,
  independent of the bundled dashboard, for use from another application or
  a notebook.

## Core Features

### Pricing Models

- Black-Scholes-Merton with full Greeks (Delta, Gamma, Theta, Vega, Rho)
- Binomial Tree pricing for American and European options
- Monte Carlo simulation (GBM) with antithetic variates for variance
  reduction, plus Heston stochastic volatility and Merton jump-diffusion
  models via `api/advanced_models.py`
- Interactive Monte Carlo convergence analysis (real simulations at
  increasing path counts, plotted live)

### Machine Learning Pricing

- An ensemble of a neural network (`MLPRegressor`), random forest, and
  XGBoost, trained on synthetically generated, Black-Scholes-priced data
  (never presented as market data)
- Feature importance computed with real permutation importance
  (`sklearn.inspection.permutation_importance`), not a static or invented
  ranking
- Documented, reproducible validation results, including a real accuracy
  bug found and fixed during development of this project — see
  [Model Validation](MODEL_VALIDATION_GUIDE.md) for the full account,
  including what was wrong, why, and the measured before/after numbers

### Risk Management

- Value at Risk (historical and parametric) and Expected Shortfall,
  computed from real historical returns of the position symbols supplied,
  not simulated placeholder returns
- Stress testing against the empirical worst days observed in the
  lookback window (1st/5th percentile, worst single day, max drawdown)
- Delta hedging with formula-derived hedge effectiveness and a disclosed,
  labeled transaction-cost assumption (no live bid-ask feed is used to
  estimate execution cost)
- Portfolio-level position tracking with Black-Scholes-computed market
  values and P&L

### Market Data & Analytics

- Live US market data via `yfinance`: quotes, option chains, historical
  volatility, implied-volatility term structure and smile, and sentiment
  indicators (VIX level, put/call ratio, fear-greed index) computed from
  that data
- Interactive Plotly visualizations: payoff diagrams, Greeks sensitivity,
  volatility smile and term structure, Monte Carlo convergence

### India Market (NSE/BSE) Support

- NSE/BSE equity and index quotes via `yfinance` (`.NS` / `.BO` suffixes)
- Real NSE F&O option chains for indices (NIFTY, BANKNIFTY, FINNIFTY,
  MIDCPNIFTY, NIFTYNXT50) and individual stocks, via
  [jugaad-data](https://github.com/jugaad-py/jugaad-data)
- Put/Call Ratio, max pain, and open-interest buildup analytics with an
  interactive chart
- RBI policy repo rate and G-Sec yields as an India risk-free-rate proxy
- A market switcher (US / India-NSE / India-BSE) that swaps the dashboard,
  currency formatting, and symbol handling; BSE has no free options data
  source and is shown as unavailable rather than silently failing

## Screenshots

### Black-Scholes Option Pricing

![Black-Scholes Pricing](Screenshots/Black_Scholes.png)
Real-time Black-Scholes pricing with Greeks and sensitivity analysis.

### Binomial Tree Model

![Binomial Model](Screenshots/Binomial.png)
Multi-step binomial tree pricing for American and European options.

### Risk Management Dashboard

![Risk Management](Screenshots/Risk_management.png)
VaR, Expected Shortfall, and stress-test results computed from real
historical data.

### Greeks Visualization

![Greeks Plot](Screenshots/greeks_plot.png)
Interactive visualization of option sensitivities.

### Monte Carlo Convergence Analysis

![Convergence Plot](Screenshots/conv_plot.png)
Live convergence of the Monte Carlo price estimate as simulation count
increases.

## Model Validation

The ML pricing ensemble's validation R² is measured on a held-out split of
synthetically generated, Black-Scholes-priced data — it demonstrates that
the pipeline correctly learns the pricing function it was trained on, and
is not a claim about predicting real market prices. An earlier version of
this project reported a fabricated R² of 0.94 by clamping the computed
value; that clamp has been removed. The real, measured validation R² after
fixing the underlying feature bug is documented in full, including
methodology and per-model numbers, in
[MODEL_VALIDATION_GUIDE.md](MODEL_VALIDATION_GUIDE.md).

## Technology Stack

**Backend**: Python 3.8+, Flask, NumPy, SciPy, pandas, scikit-learn,
XGBoost, Plotly (server-side figure generation)

**Market data**: yfinance (US and NSE/BSE quotes and option chains via
`.NS`/`.BO` suffixes), jugaad-data (real NSE F&O option chains and RBI
rates)

**Frontend**: server-rendered Jinja2 template, vanilla JavaScript and
jQuery, Bootstrap 5, Plotly.js, and Font Awesome, all loaded from CDN — no
Node.js build step or `package.json` is required to run this project

**Deployment**: Railway (primary target, via Nixpacks), Docker/Gunicorn,
and a reduced-dependency configuration for Vercel

## API

The backend exposes over 20 REST endpoints under `/api/`, covering
pricing (`/api/calculate_black_scholes`, `/api/calculate_binomial`,
`/api/monte_carlo`), ML pricing (`/api/ml/ensemble_price`,
`/api/ml/volatility_forecast`), risk (`/api/risk_metrics`,
`/api/risk/portfolio_risk`, `/api/risk/dynamic_hedging`,
`/api/stress_test`), market data and analytics
(`/api/market_data/<symbol>`, `/api/option_chain/<symbol>`,
`/api/volatility_surface/<symbol>`, `/api/market/volatility_term_structure`,
`/api/market/sentiment`), India market data
(`/api/india/option_chain/<symbol>`, `/api/india_market_sentiment`,
`/api/india/risk_free_rate`), and platform status (`/api/status`). Every
endpoint returns JSON and is usable independently of the dashboard.

## Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
git clone https://github.com/DIPESHGOEL27/option-pricing-models.git
cd option-pricing-models

# Full feature set (ML pricing, India market data, plotting):
python toggle_features.py full
pip install -r requirements.txt

python main.py
```

The dashboard is then available at `http://localhost:8000`.

For a lighter install without ML dependencies (Black-Scholes, Binomial, and
Monte Carlo pricing only), use `python toggle_features.py light` instead.
See [DEPLOYMENT.md](DEPLOYMENT.md) for Railway, Docker, and Vercel
deployment instructions, including which features are available in each
environment.

## Known Limitations

- The ML pricing ensemble is trained on synthetically generated data and
  is not fit to real market option prices; it is a modeling and validation
  exercise, not a trading signal.
- The Vercel deployment configuration excludes `scikit-learn` and
  `xgboost` to stay under the serverless size limit, so ML pricing
  endpoints are unavailable there — use Railway or Docker for the full
  feature set.
- `jugaad-data` scrapes an unofficial NSE endpoint; it can be slower or
  intermittently unavailable compared to a paid market-data feed, and its
  first request after an idle period is slower due to session bootstrap.
- BSE has no free, working options-chain data source; the dashboard shows
  this explicitly instead of returning empty or fabricated data.
- The hedging endpoint's transaction cost is a disclosed, fixed assumption
  (0.2% of hedge notional), not a live bid-ask spread, since no live quote
  feed is queried for the hedge instrument.
- The bundled Flask development server (`python main.py`) is not a
  production WSGI server; deployments use Gunicorn (see
  [DEPLOYMENT.md](DEPLOYMENT.md)).

## Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) — Railway, Docker, and Vercel deployment
- [MODEL_VALIDATION_GUIDE.md](MODEL_VALIDATION_GUIDE.md) — validation
  methodology and real, measured results
- [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) — project layout

## License

MIT License — see [LICENSE](LICENSE).

## Author

Dipesh Goel — [LinkedIn](https://www.linkedin.com/in/dipeshgoel27/) —
[Portfolio](https://dipeshgoel.vercel.app/)
