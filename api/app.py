from flask import Flask, request, jsonify, render_template
from flask.json.provider import DefaultJSONProvider
import numpy as np
import scipy.stats as si
from scipy import stats
import pandas as pd
import math
from datetime import datetime, timedelta
import json
import io
import base64
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Import market data provider
try:
    from .market_data import MarketDataProvider, VolatilityEstimator
    BASIC_MARKET_DATA_AVAILABLE = True
except ImportError:
    try:
        from market_data import MarketDataProvider, VolatilityEstimator
        BASIC_MARKET_DATA_AVAILABLE = True
    except ImportError:
        BASIC_MARKET_DATA_AVAILABLE = False

# Import India (NSE/BSE) market data provider
try:
    from .india_market_data import (
        IndiaMarketDataProvider, IndiaRiskFreeRateProvider,
        calculate_put_call_ratio, calculate_max_pain, summarize_oi_buildup
    )
    INDIA_MARKET_DATA_AVAILABLE = True
except ImportError:
    try:
        from india_market_data import (
            IndiaMarketDataProvider, IndiaRiskFreeRateProvider,
            calculate_put_call_ratio, calculate_max_pain, summarize_oi_buildup
        )
        INDIA_MARKET_DATA_AVAILABLE = True
    except ImportError:
        INDIA_MARKET_DATA_AVAILABLE = False

# Import our advanced modules
try:
    from .advanced_models import MonteCarloEngine, RiskMetrics, ModelValidation
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    try:
        from advanced_models import MonteCarloEngine, RiskMetrics, ModelValidation
        MONTE_CARLO_AVAILABLE = True
    except ImportError:
        MONTE_CARLO_AVAILABLE = False

try:
    from .advanced_models import MonteCarloEngine, HestonCalibration
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    try:
        from advanced_models import MonteCarloEngine, HestonCalibration
        ADVANCED_PRICING_AVAILABLE = True
    except ImportError:
        ADVANCED_PRICING_AVAILABLE = False

try:
    from .option_pricing import AdvancedOptionPricer
except ImportError:
    from option_pricing import AdvancedOptionPricer

try:
    from . import iv_engine
except ImportError:
    import iv_engine

# Check overall advanced features availability
ADVANCED_FEATURES_AVAILABLE = any([
    MONTE_CARLO_AVAILABLE, BASIC_MARKET_DATA_AVAILABLE,
    ADVANCED_PRICING_AVAILABLE, INDIA_MARKET_DATA_AVAILABLE
])

# Ensure Python can find the modules in the current directory
import sys
import os
# Add the current directory to the Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

app = Flask(__name__, template_folder='templates', static_folder='static')


class NumpyJSONProvider(DefaultJSONProvider):
    """Extends Flask's JSON provider to serialize numpy/pandas scalar types.

    Routes across this app build response dicts directly from pandas/yfinance
    data (e.g. hist['Volume'].iloc[-1] is a numpy.int64), which the stdlib
    json module cannot serialize on its own — every route doing this would
    otherwise fail with "Object of type int64/float64 is not JSON
    serializable" on every request.
    """

    @staticmethod
    def default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return DefaultJSONProvider.default(obj)


app.json = NumpyJSONProvider(app)

# Process-wide singletons so each provider's own instance-level cache
# (MarketDataProvider.cache, IndiaMarketDataProvider.cache) actually
# accumulates across requests -- constructing a fresh provider per request,
# as every route used to, means the cache dict starts empty every time and
# never hits, so every page load was a cold network fetch regardless of the
# cache_duration each class declares.
_option_pricer = AdvancedOptionPricer()
_market_data_provider = MarketDataProvider() if BASIC_MARKET_DATA_AVAILABLE else None
_india_market_data_provider = IndiaMarketDataProvider() if INDIA_MARKET_DATA_AVAILABLE else None
_india_risk_free_rate_provider = IndiaRiskFreeRateProvider() if INDIA_MARKET_DATA_AVAILABLE else None


@app.route('/')
def index():
    return render_template('index.html')


def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes price and Greeks, delegating to the tested, correctly
    put/call-branched implementation in option_pricing.py rather than
    duplicating the math here. Theta is per day, vega/rho are per 1%
    change -- the standard trader convention.
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    if S <= 0 or K <= 0:
        raise ValueError('S and K must be positive')
    if T < 0:
        raise ValueError('T must be non-negative')
    if sigma < 0 or (sigma == 0 and T > 0):
        raise ValueError('sigma must be positive for T > 0')

    price = _option_pricer.black_scholes(S, K, T, r, sigma, option_type)
    greeks = _option_pricer.calculate_greeks(S, K, T, r, sigma, option_type)
    return price, greeks['delta'], greeks['gamma'], greeks['theta'], greeks['vega'], greeks['rho']

@app.route('/api/calculate_black_scholes', methods=['POST'])
def calculate_black_scholes():
    data = request.json
    
    # Validate required numeric inputs
    required_fields = ['S', 'K', 'T', 'r', 'sigma']
    validated_data, error = validate_numeric_inputs(data, required_fields)
    if error:
        return jsonify(error), 400
        
    # Extract validated values
    S = validated_data['S']
    K = validated_data['K']
    T = validated_data['T']
    r = validated_data['r']
    sigma = validated_data['sigma']
    
    # Get option type with fallbacks
    option_type = data.get('option_type', data.get('optionType', 'call'))
    
    try:
        price, delta, gamma, theta, vega, rho = black_scholes(S, K, T, r, sigma, option_type)
        return jsonify({
            'option_price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def binomial_tree(S, K, T, r, sigma, steps, option_type='call'):
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    if steps <= 0:
        raise ValueError('steps must be a positive integer')
    if S <= 0 or K <= 0:
        raise ValueError('S and K must be positive')
    if T <= 0:
        raise ValueError('T must be positive')
    if sigma <= 0:
        raise ValueError('sigma must be positive')

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros(steps + 1)
    asset_prices[0] = S * (d ** steps)
    for i in range(1, steps + 1):
        asset_prices[i] = asset_prices[i - 1] * u / d
    
    # Initialize option values at maturity
    option_values = np.zeros(steps + 1)
    if option_type == 'call':
        option_values = np.maximum(0, asset_prices - K)
    elif option_type == 'put':
        option_values = np.maximum(0, K - asset_prices)
    
    # Backward induction
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = (p * option_values[i + 1] + (1 - p) * option_values[i]) * disc
    
    return option_values[0]

@app.route('/api/calculate_binomial', methods=['POST'])
def calculate_binomial():
    data = request.json
    required_fields = ['S', 'K', 'T', 'r', 'sigma']
    validated_data, error = validate_numeric_inputs(data, required_fields)
    if error:
        return jsonify(error), 400

    S = validated_data['S']
    K = validated_data['K']
    T = validated_data['T']
    r = validated_data['r']
    sigma = validated_data['sigma']
    option_type = data.get('option_type', data.get('optionType', 'call'))

    try:
        steps = int(data.get('steps', 100))
    except (ValueError, TypeError):
        return jsonify({'error': f"Invalid numeric input for steps: {data.get('steps')!r}"}), 400

    try:
        price = binomial_tree(S, K, T, r, sigma, steps, option_type)
        return jsonify({'option_price': price})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Advanced API endpoints

# Hard simulation caps to bound peak memory on a 512MB instance.
# GBM only needs the terminal price (see below), so its per-path memory is
# negligible and it can run far more paths safely. Heston and jump-diffusion
# genuinely simulate a full multi-step path (Heston's variance process is
# path-dependent; the jump-diffusion implementation compounds jumps step by
# step), so each one allocates several full (n_simulations x n_steps)
# arrays -- that combination is what was previously capable of allocating
# roughly 800MB-1GB at the old 100,000-simulation x 252-step default.
MAX_MC_SIMULATIONS_TERMINAL = 200000
MAX_MC_SIMULATIONS_PATH = 20000


@app.route('/api/monte_carlo', methods=['POST'])
def calculate_monte_carlo():
    """Advanced Monte Carlo pricing with multiple models"""
    try:
        data = request.json
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data['optionType']
        model_type = data.get('model', 'gbm')  # gbm, heston, jump_diffusion
        n_simulations = int(data.get('simulations', 100000))
        if n_simulations < 1000:
            return jsonify({'error': 'simulations must be at least 1000'}), 400

        if model_type == 'gbm':
            # A vanilla European payoff only depends on the terminal price,
            # and GBM's terminal distribution is identical whether it's
            # reached via one large step or many small ones -- so a single
            # step is exact here, not an approximation, and is roughly
            # 250x cheaper in memory and time than simulating a full
            # 252-step path for no benefit.
            n_simulations = min(n_simulations, MAX_MC_SIMULATIONS_TERMINAL)
            mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=1)
            paths = mc_engine.geometric_brownian_motion(S, T, r, sigma)
        elif model_type == 'heston':
            n_simulations = min(n_simulations, MAX_MC_SIMULATIONS_PATH)
            mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=252)
            kappa = float(data.get('kappa', 2.0))
            theta = float(data.get('theta', 0.04))
            sigma_v = float(data.get('sigma_v', 0.3))
            rho = float(data.get('rho', -0.5))
            v0 = float(data.get('v0', 0.04))
            paths, vol_paths = mc_engine.heston_model(S, T, r, v0, kappa, theta, sigma_v, rho)
        elif model_type == 'jump_diffusion':
            n_simulations = min(n_simulations, MAX_MC_SIMULATIONS_PATH)
            mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=252)
            lam = float(data.get('lambda', 0.1))
            mu_j = float(data.get('mu_j', -0.1))
            sigma_j = float(data.get('sigma_j', 0.2))
            paths = mc_engine.jump_diffusion_merton(S, T, r, sigma, lam, mu_j, sigma_j)
        else:
            return jsonify({'error': 'Invalid model type'}), 400

        result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)

        # Calculate Greeks using Monte Carlo (independently bounded -- it
        # runs several bumped simulations of its own, see calculate_greeks_mc).
        greeks = RiskMetrics.calculate_greeks_mc(S, K, T, r, sigma, option_type, max(n_simulations // 10, 1000))

        return jsonify({
            'option_price': result['price'],
            'std_error': result['std_error'],
            'confidence_interval': result['confidence_interval'],
            'model_type': model_type,
            'simulations': n_simulations,
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'vega': greeks['vega'],
            'theta': greeks['theta'],
            'rho': greeks['rho']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/market_data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get real-time market data"""
    try:
        market_data = _market_data_provider
        stock_data = market_data.get_stock_price(symbol.upper())
        
        if 'error' in stock_data:
            return jsonify(stock_data)
        
        # Get historical volatility
        hist_data = market_data.get_historical_data(symbol.upper(), period="1y")
        if not hist_data.empty:
            current_vol = VolatilityEstimator.historical_volatility(hist_data['Close']).iloc[-1]
            stock_data['implied_volatility'] = current_vol
        
        return jsonify(stock_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/risk_metrics', methods=['POST'])
def calculate_risk_metrics():
    """Calculate portfolio risk metrics"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        positions = data['positions']  # List of position dictionaries
        
        # Simple portfolio risk calculation without OptionPortfolio class
        total_value = 0
        total_pnl = 0
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0

        position_details = []

        for pos in positions:
            S = float(pos['underlying_price'])
            K = float(pos['strike'])
            T = max(0.01, (datetime.strptime(pos['expiry'], '%Y-%m-%d') - datetime.now()).days / 365.0)
            r = float(pos['risk_free_rate'])
            sigma = float(pos['volatility'])
            option_type = pos['option_type']
            quantity = int(pos['quantity'])
            premium_paid = float(pos.get('premium_paid', 0))

            # Calculate Black-Scholes price and Greeks
            price, delta, gamma, theta, vega, rho = black_scholes(S, K, T, r, sigma, option_type)

            position_value = price * quantity * 100  # 100 shares per contract
            position_pnl = position_value - premium_paid * quantity * 100
            total_value += position_value
            total_pnl += position_pnl
            total_delta += delta * quantity * 100
            total_gamma += gamma * quantity * 100
            total_theta += theta * quantity * 100
            total_vega += vega * quantity * 100

            position_details.append({
                'symbol': pos['symbol'],
                'option_type': option_type,
                'strike': K,
                'quantity': quantity,
                'market_value': float(position_value),
                'pnl': float(position_pnl),
                'delta': float(delta * quantity * 100),
                'gamma': float(gamma * quantity * 100),
                'theta': float(theta * quantity * 100),
                'vega': float(vega * quantity * 100)
            })

        # Portfolio summary
        summary = {
            'total_positions': len(positions),
            'portfolio_value': float(total_value),
            'total_pnl': float(total_pnl),
            'portfolio_greeks': {
                'delta': float(total_delta),
                'gamma': float(total_gamma),
                'theta': float(total_theta),
                'vega': float(total_vega)
            },
            'position_details': position_details
        }
        
        # Simple risk report
        risk_report = {
            'delta_risk': abs(total_delta),
            'gamma_risk': abs(total_gamma), 
            'theta_decay': abs(total_theta),
            'vega_risk': abs(total_vega),
            'concentration_risk': 'Low' if len(positions) > 3 else 'High',
            'max_loss_estimate': float(abs(total_value * 0.2))  # 20% max loss estimate
        }
        
        # Delta hedge recommendation
        hedge_required = abs(total_delta) > 50
        shares_to_hedge = int(-total_delta / 100) if abs(total_delta) > 50 else 0
        hedge_direction = 'buy' if total_delta < 0 else 'sell'
        
        hedge_rec = {
            'hedge_required': bool(hedge_required),
            'shares_to_hedge': int(shares_to_hedge),
            'hedge_direction': str(hedge_direction),
            'hedge_cost_estimate': float(abs(total_delta) * 0.01)  # 1 cent per delta
        }
        
        return jsonify({
            'portfolio_summary': summary,
            'risk_report': risk_report,
            'hedge_recommendations': hedge_rec
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/stress_test', methods=['POST'])
def perform_stress_test():
    """Perform stress testing"""
    try:
        data = request.json
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data['optionType']
        
        # Custom scenarios or use defaults
        scenarios = data.get('scenarios', None)

        stress_results = RiskMetrics.stress_test(S, K, T, r, sigma, option_type, scenarios)

        # A single named scenario (one of the dedicated stress-test buttons)
        # narrows the response to just that scenario instead of all of them.
        scenario = data.get('scenario')
        if scenario and scenario in stress_results:
            stress_results = {
                'base_price': stress_results['base_price'],
                scenario: stress_results[scenario]
            }

        return jsonify(stress_results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/model_validation', methods=['POST'])
def validate_models():
    """Validate pricing models"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data['optionType']
        
        if not MONTE_CARLO_AVAILABLE:
            # Fallback validation using basic comparison
            bs_price, delta, gamma, theta, vega, rho = black_scholes(S, K, T, r, sigma, option_type)
            
            # Simple Monte Carlo simulation for comparison
            np.random.seed(42)
            dt = T / 252
            n_simulations = 10000
            
            # Generate random paths
            random_shocks = np.random.normal(0, 1, (n_simulations, int(T * 252)))
            price_paths = np.zeros((n_simulations, int(T * 252) + 1))
            price_paths[:, 0] = S
            
            for i in range(1, int(T * 252) + 1):
                price_paths[:, i] = price_paths[:, i-1] * np.exp(
                    (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks[:, i-1]
                )
            
            final_prices = price_paths[:, -1]
            if option_type == 'call':
                payoffs = np.maximum(final_prices - K, 0)
            else:
                payoffs = np.maximum(K - final_prices, 0)
            
            mc_price = np.exp(-r * T) * np.mean(payoffs)
            mc_std = np.std(payoffs) / np.sqrt(n_simulations)
            
            # Create validation results with explicit Python types
            price_diff = abs(bs_price - mc_price)
            rel_error = price_diff / bs_price * 100
            validation_passed = price_diff / bs_price < 0.05
            
            validation = {
                'black_scholes_price': float(bs_price),
                'monte_carlo_price': float(mc_price),
                'price_difference': float(price_diff),
                'relative_error': float(rel_error),
                'monte_carlo_std_error': float(mc_std),
                'confidence_interval_95': [float(mc_price - 1.96 * mc_std), float(mc_price + 1.96 * mc_std)],
                'validation_passed': 1 if validation_passed else 0
            }
            
            # Simple convergence analysis with explicit Python types
            convergence_steps = [1000, 2500, 5000, 7500, 10000]
            convergence_prices = []
            
            for n_sims in convergence_steps:
                subset_payoffs = payoffs[:n_sims]
                conv_price = np.exp(-r * T) * np.mean(subset_payoffs)
                convergence_prices.append(float(conv_price))
            
            # Check convergence with explicit type conversion
            converged = False
            if len(convergence_prices) >= 2:
                last_diff = abs(convergence_prices[-1] - convergence_prices[-2])
                converged = last_diff < 0.01
            
            convergence = {
                'simulation_counts': convergence_steps,
                'prices': convergence_prices,
                'final_price': float(mc_price),
                'converged': 1 if converged else 0
            }
            
        else:
            # Use advanced model validation if available - but wrap in try/catch
            try:
                validation = ModelValidation.validate_black_scholes_vs_mc(S, K, T, r, sigma, option_type)
                convergence = ModelValidation.convergence_analysis(S, K, T, r, sigma, option_type)
                
                # Ensure all values are JSON serializable
                def make_json_safe(obj):
                    if isinstance(obj, dict):
                        return {k: make_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [make_json_safe(item) for item in obj]
                    elif isinstance(obj, np.bool_):
                        return bool(obj)
                    elif isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    elif hasattr(obj, 'item'):
                        return obj.item()
                    else:
                        return obj
                
                validation = make_json_safe(validation)
                convergence = make_json_safe(convergence)
                
            except Exception:
                # Fall back to simple validation if advanced fails
                validation = {'error': 'Advanced validation not available'}
                convergence = {'error': 'Advanced convergence analysis not available'}
        
        return jsonify({
            'validation': validation,
            'convergence': convergence
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== ADVANCED RISK MANAGEMENT API ENDPOINTS ===================

def _historical_expected_shortfall(returns, confidence_level=0.95):
    """Expected Shortfall (CVaR): mean loss in the tail beyond historical VaR.

    Returns a positive number representing a loss, matching the sign
    convention of every other risk figure in this route.
    """
    if len(returns) == 0:
        return 0.0
    var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) == 0:
        return max(-var_threshold, 0.0)
    return max(-float(np.mean(tail_returns)), 0.0)


@app.route('/api/risk/portfolio_risk', methods=['POST'])
def calculate_portfolio_risk():
    """Calculate comprehensive portfolio risk metrics"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})

        positions = data['positions']  # List of positions with weights and symbols
        confidence_level = float(data.get('confidence_level', 0.95)) if data else 0.95
        time_horizon = int(data.get('time_horizon', 1)) if data else 1

        market_data = _market_data_provider

        # Pull real daily returns for every position's symbol and build a
        # value-weighted portfolio return series -- no synthetic returns.
        symbol_returns = {}
        total_value = 0.0
        for pos in positions:
            symbol = pos.get('symbol') or 'SPY'
            value = float(pos.get('value', 100000))
            total_value += value
            hist = market_data.get_historical_data(symbol, period="2y")
            if hist is not None and not hist.empty and 'Returns' in hist.columns:
                returns = hist['Returns'].dropna()
                if len(returns) > 0:
                    symbol_returns[symbol] = (returns, value)

        if not symbol_returns:
            return jsonify({'error': 'No historical market data available for the provided position symbols'})

        returns_df = pd.DataFrame({sym: r for sym, (r, _) in symbol_returns.items()}).dropna()
        if returns_df.empty or len(returns_df) < 30:
            return jsonify({'error': 'Insufficient overlapping historical data for the provided positions'})

        weights_arr = np.array([symbol_returns[sym][1] / total_value for sym in returns_df.columns])
        portfolio_returns = (returns_df.values @ weights_arr)
        portfolio_value = total_value

        # Calculate VaR and Expected Shortfall from the real return series,
        # scaled to the requested time horizon (standard square-root-of-time rule).
        # Both are reported as positive numbers representing a loss magnitude.
        horizon_scale = np.sqrt(time_horizon)
        es_result = _historical_expected_shortfall(portfolio_returns, confidence_level) * horizon_scale

        historical_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100) * horizon_scale
        parametric_var = -stats.norm.ppf(1 - confidence_level) * np.std(portfolio_returns) * horizon_scale

        volatility = np.std(portfolio_returns) * np.sqrt(252)
        skewness = float(stats.skew(portfolio_returns))
        kurtosis = float(stats.kurtosis(portfolio_returns))

        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdown))

        # Historical-simulation stress scenarios: empirical statistics of the
        # position's own real return history, not hardcoded assumptions.
        sorted_returns = np.sort(portfolio_returns)
        stress_results = {
            'worst_1pct_day': {
                'loss': float(np.percentile(portfolio_returns, 1)),
                'basis': '1st percentile of real daily returns over the lookback window'
            },
            'worst_5pct_day': {
                'loss': float(np.percentile(portfolio_returns, 5)),
                'basis': '5th percentile of real daily returns over the lookback window'
            },
            'worst_observed_day': {
                'loss': float(sorted_returns[0]),
                'basis': 'single worst observed daily return in the lookback window'
            },
            'max_drawdown_period': {
                'loss': max_drawdown,
                'basis': 'largest peak-to-trough decline observed in the lookback window'
            }
        }

        return jsonify({
            'portfolio_value': portfolio_value,
            'symbols_used': list(returns_df.columns),
            'observations': int(len(returns_df)),
            'var': {
                'historical': float(historical_var * portfolio_value),
                'parametric': float(parametric_var * portfolio_value),
                'confidence_level': confidence_level
            },
            'expected_shortfall': float(es_result * portfolio_value),
            'stress_test_results': stress_results,
            'risk_metrics': {
                'volatility': volatility,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': (np.mean(portfolio_returns) * 252 - 0.02) / volatility
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/risk/dynamic_hedging', methods=['POST'])
def calculate_dynamic_hedging():
    """Calculate dynamic hedging strategy"""
    try:
        data = request.json
        
        # Validate required fields
        validated_data, error = validate_numeric_inputs(data, ['portfolio_delta'])
        if error:
            return jsonify(error), 400
            
        portfolio_delta = validated_data['portfolio_delta']
        
        # Parse optional parameters with defaults
        try:
            target_delta = float(data.get('target_delta', 0))
        except (ValueError, TypeError):
            target_delta = 0
            
        try:
            hedge_ratio = float(data.get('hedge_ratio', 1.0))
        except (ValueError, TypeError):
            hedge_ratio = 1.0
        
        # Real, directly-derived hedging math -- for a linear hedge sized as
        # hedge_quantity = -delta_exposure * hedge_ratio, the fraction of
        # delta risk eliminated is exactly hedge_ratio (capped at 100%);
        # nothing here is assumed or hardcoded except the disclosed
        # transaction-cost rate, since no live bid-ask/instrument data is
        # available from this endpoint's inputs (delta values only).
        TRANSACTION_COST_RATE = 0.002  # 0.2% of hedge notional -- a disclosed assumption, not measured

        delta_exposure = portfolio_delta - target_delta
        hedge_quantity = -delta_exposure * hedge_ratio
        hedge_effectiveness = min(abs(hedge_ratio), 1.0)
        residual_delta_exposure = delta_exposure * (1 - hedge_effectiveness)
        delta_risk_eliminated = abs(delta_exposure) * hedge_effectiveness

        return jsonify({
            'current_delta': portfolio_delta,
            'target_delta': target_delta,
            'delta_exposure': delta_exposure,
            'hedge_quantity': hedge_quantity,
            'delta_risk_eliminated': delta_risk_eliminated,
            'residual_delta_exposure': residual_delta_exposure,
            'hedge_effectiveness': hedge_effectiveness,
            'recommendation': 'buy' if hedge_quantity > 0 else 'sell',
            'hedge_cost': abs(hedge_quantity) * TRANSACTION_COST_RATE,
            'transaction_cost_rate': TRANSACTION_COST_RATE,
            'transaction_cost_note': 'Assumed rate (0.2% of hedge notional) -- no live bid-ask data available without an instrument symbol.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== PLOTLY ANALYTICS API ENDPOINTS ===================

@app.route('/api/plot_payoff', methods=['POST'])
def plot_payoff_diagram():
    """Generate interactive payoff diagram for portfolio positions"""
    try:
        data = request.json
        if not data or 'positions' not in data:
            return jsonify({'error': 'No positions provided'})
        
        positions = data['positions']

        # Derive the spot-price grid from the actual strikes (and underlying
        # prices, where supplied) in this portfolio, rather than a hardcoded
        # 80-120 range -- a fixed range built for ~$100 US equities produces
        # a flat, meaningless diagram for instruments priced in the
        # thousands or tens of thousands (e.g. NIFTY around 24,000).
        strikes = [float(p.get('strike', 100)) for p in positions]
        reference_points = strikes + [
            float(p['underlying_price']) for p in positions
            if p.get('underlying_price')
        ]
        low_ref, high_ref = min(reference_points), max(reference_points)
        center = (low_ref + high_ref) / 2
        margin = max(high_ref - low_ref, center * 0.15)
        spot_low = max(0.01, low_ref - margin)
        spot_high = high_ref + margin
        spot_range = np.linspace(spot_low, spot_high, 200)
        total_payoff = np.zeros_like(spot_range)
        
        position_traces = []
        
        for i, position in enumerate(positions):
            symbol = position.get('symbol', 'OPTION')
            option_type = position.get('option_type', 'call')
            strike = float(position.get('strike', 100))
            quantity = int(position.get('quantity', 1))
            premium = float(position.get('premium_paid', 5))
            
            # Calculate payoff for this position
            if option_type.lower() == 'call':
                payoff = np.maximum(spot_range - strike, 0) * quantity - premium * quantity
            else:  # put
                payoff = np.maximum(strike - spot_range, 0) * quantity - premium * quantity
            
            total_payoff += payoff
            
            # Create trace for individual position
            position_traces.append(go.Scatter(
                x=spot_range.tolist(),
                y=payoff.tolist(),
                mode='lines',
                name=f'{symbol} {option_type.upper()} {strike}',
                line=dict(width=2, dash='dot'),
                opacity=0.7
            ))
        
        # Create total payoff trace
        traces = position_traces + [go.Scatter(
            x=spot_range.tolist(),
            y=total_payoff.tolist(),
            mode='lines',
            name='Total Portfolio',
            line=dict(width=4, color='yellow'),
            fill='tonexty' if len(position_traces) == 1 else None
        )]
        
        # Add break-even line
        traces.append(go.Scatter(
            x=[spot_range.min(), spot_range.max()],
            y=[0, 0],
            mode='lines',
            name='Break-even',
            line=dict(width=2, color='red', dash='dash')
        ))
        
        # Create layout
        layout = go.Layout(
            title='Portfolio Payoff Diagram',
            xaxis=dict(title='Underlying Price at Expiration'),
            yaxis=dict(title='Profit/Loss'),
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True
        )
        
        fig_dict = {
            'data': traces,
            'layout': layout
        }
        
        return jsonify({
            'plot': json.dumps(fig_dict, cls=PlotlyJSONEncoder),
            'analysis': {
                # max_profit/max_loss are the payoff at the edges of the
                # plotted spot range, not a true analytical bound -- for a
                # position with unbounded upside or downside (e.g. a naked
                # long/short option), the real max/min lies outside any
                # finite grid. No profit_probability is reported: doing so
                # correctly requires a stated distributional assumption
                # (volatility, time to expiry) per position, which this
                # endpoint does not currently receive.
                'max_profit': float(np.max(total_payoff)),
                'max_loss': float(np.min(total_payoff)),
                'break_even_points': _calculate_break_even_points(spot_range, total_payoff),
                'spot_range': {'low': float(spot_low), 'high': float(spot_high)}
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/volatility_smile', methods=['POST'])
def plot_volatility_smile():
    """Generate an implied volatility smile chart from the live option chain."""
    try:
        data = request.json
        if not data or not data.get('symbol'):
            return jsonify({'error': 'No symbol provided'})

        if not BASIC_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'Market data features are not available'})

        symbol = data['symbol'].upper()
        market_data = _market_data_provider
        option_chain = market_data.get_option_chain(symbol)

        if 'error' in option_chain:
            return jsonify(option_chain)

        calls = option_chain['calls']
        puts = option_chain['puts']
        calls_iv = calls[calls['impliedVolatility'] > 0]
        puts_iv = puts[puts['impliedVolatility'] > 0]

        if calls_iv.empty and puts_iv.empty:
            return jsonify({'error': f'No implied volatility data available for {symbol}'})

        traces = []
        if not calls_iv.empty:
            traces.append(go.Scatter(
                x=calls_iv['strike'].tolist(),
                y=(calls_iv['impliedVolatility'] * 100).tolist(),
                mode='markers+lines',
                name='Calls',
                marker=dict(size=6)
            ))
        if not puts_iv.empty:
            traces.append(go.Scatter(
                x=puts_iv['strike'].tolist(),
                y=(puts_iv['impliedVolatility'] * 100).tolist(),
                mode='markers+lines',
                name='Puts',
                marker=dict(size=6)
            ))

        max_iv_pct = max(
            calls_iv['impliedVolatility'].max() if not calls_iv.empty else 0,
            puts_iv['impliedVolatility'].max() if not puts_iv.empty else 0
        ) * 100
        traces.append(go.Scatter(
            x=[option_chain['underlying_price']] * 2,
            y=[0, max_iv_pct],
            mode='lines',
            name='Spot Price',
            line=dict(width=1, color='gray', dash='dot')
        ))

        layout = go.Layout(
            title=f"{symbol} Implied Volatility Smile ({option_chain['expiry']})",
            xaxis=dict(title='Strike Price'),
            yaxis=dict(title='Implied Volatility (%)'),
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True
        )

        return jsonify({
            'symbol': symbol,
            'expiry': option_chain['expiry'],
            'underlying_price': option_chain['underlying_price'],
            'plot': json.dumps({'data': traces, 'layout': layout}, cls=PlotlyJSONEncoder)
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# =================== INDIA (NSE/BSE) MARKET DATA API ENDPOINTS ===================
# Basic NSE/BSE equity price/history already works through the existing
# /api/market_data/<symbol> etc. routes via yfinance's .NS/.BO suffixes --
# these routes cover what yfinance does not provide for Indian markets:
# real F&O option chains (via jugaad-data) and RBI-sourced risk-free rates.

@app.route('/api/india/option_chain/<symbol>', methods=['GET'])
def get_india_option_chain(symbol):
    """Real NSE F&O option chain with PCR, max pain, an OI buildup chart,
    and per-strike model-vs-market analysis: implied volatility solved from
    real bid/ask quotes, cross-checked against NSE's own published IV, and
    flagged rich/cheap relative to a fitted skew curve."""
    try:
        if not INDIA_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'India market data features are not available'})

        expiry = request.args.get('expiry')
        chain = _india_market_data_provider.get_option_chain(symbol.upper(), expiry)

        if 'error' in chain:
            return jsonify(chain)

        rows = chain['rows']
        pcr = calculate_put_call_ratio(rows)
        max_pain = calculate_max_pain(rows)
        buildup = summarize_oi_buildup(rows)

        strikes = buildup['strikes']
        max_pain_strike = max_pain.get('strike')
        underlying = chain.get('underlying_value')

        expiry_info = iv_engine.time_to_expiry(chain['expiry'])
        fallback_rate = (
            _india_risk_free_rate_provider.interpolate_rate(expiry_info['calendar_years'])
            if expiry_info and _india_risk_free_rate_provider else None
        )
        analysis = iv_engine.analyze_chain(rows, underlying, chain['expiry'], fallback_rate)

        skew = None
        if 'error' not in analysis:
            skew = iv_engine.fit_skew(analysis['strikes'], underlying)
            if 'error' not in skew:
                flags_by_strike = {}
                for point in skew['points']:
                    flags_by_strike.setdefault(point['strike'], {})[point['option_type']] = point
                for entry in analysis['strikes']:
                    for leg_key, option_type in (('ce', 'call'), ('pe', 'put')):
                        leg = entry.get(leg_key)
                        point = flags_by_strike.get(entry['strike'], {}).get(option_type)
                        if leg and point:
                            leg['fitted_iv'] = point['fitted_iv']
                            leg['skew_deviation_sigma'] = point['deviation_sigma']
                            leg['skew_flag'] = point['flag']

        traces = [
            go.Bar(x=strikes, y=buildup['call_oi'], name='Call OI', marker=dict(color='#dc3545')),
            go.Bar(x=strikes, y=buildup['put_oi'], name='Put OI', marker=dict(color='#198754'))
        ]

        if underlying:
            max_oi = max(buildup['call_oi'] + buildup['put_oi'], default=0)
            traces.append(go.Scatter(
                x=[underlying, underlying], y=[0, max_oi],
                mode='lines', name='Spot Price',
                line=dict(width=1, color='gray', dash='dot')
            ))
        if max_pain_strike is not None:
            max_oi = max(buildup['call_oi'] + buildup['put_oi'], default=0)
            traces.append(go.Scatter(
                x=[max_pain_strike, max_pain_strike], y=[0, max_oi],
                mode='lines', name='Max Pain',
                line=dict(width=2, color='yellow', dash='dash')
            ))

        layout = go.Layout(
            title=f"{chain['symbol']} Open Interest Buildup ({chain['expiry']})",
            xaxis=dict(title='Strike Price'),
            yaxis=dict(title='Open Interest'),
            barmode='group',
            template='plotly_dark',
            hovermode='x unified',
            showlegend=True
        )

        return jsonify({
            'symbol': chain['symbol'],
            'underlying_value': underlying,
            'expiry': chain['expiry'],
            'expiry_dates': chain['expiry_dates'],
            'strikes': analysis.get('strikes', []) if 'error' not in analysis else [],
            'pcr': pcr,
            'max_pain': max_pain,
            'plot': json.dumps({'data': traces, 'layout': layout}, cls=PlotlyJSONEncoder),
            'iv_analysis': {
                'time_to_expiry': analysis.get('time_to_expiry'),
                'risk_free_rate': analysis.get('risk_free_rate'),
                'rate_source': analysis.get('rate_source'),
                'implied_forward': analysis.get('implied_forward'),
                'summary': analysis.get('summary'),
                'error': analysis.get('error')
            },
            'skew': skew
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/india/skew/<symbol>', methods=['GET'])
def get_india_skew_term_structure(symbol):
    """Solved-IV skew curve and ATM-IV term structure across real NSE
    expiries -- e.g. is the market pricing more uncertainty into the
    monthly expiry than next week's, and how does the smile shape change
    along the way.

    The underlying NSE fetch is cached per-symbol (not per-expiry) by
    IndiaMarketDataProvider, so looping over several expiries here costs
    one network round-trip, not one per expiry.
    """
    try:
        if not INDIA_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'India market data features are not available'})

        max_expiries = min(int(request.args.get('max_expiries', 4)), 8)

        first_chain = _india_market_data_provider.get_option_chain(symbol.upper())
        if 'error' in first_chain:
            return jsonify(first_chain)

        underlying = first_chain['underlying_value']
        expiry_dates = first_chain['expiry_dates'][:max_expiries]

        term_structure = []
        for expiry_str in expiry_dates:
            chain = _india_market_data_provider.get_option_chain(symbol.upper(), expiry_str)
            if 'error' in chain:
                term_structure.append({'expiry': expiry_str, 'error': chain['error']})
                continue

            expiry_info = iv_engine.time_to_expiry(expiry_str)
            fallback_rate = (
                _india_risk_free_rate_provider.interpolate_rate(expiry_info['calendar_years'])
                if expiry_info and _india_risk_free_rate_provider else None
            )
            analysis = iv_engine.analyze_chain(chain['rows'], underlying, expiry_str, fallback_rate)
            if 'error' in analysis:
                term_structure.append({'expiry': expiry_str, 'error': analysis['error']})
                continue

            skew = iv_engine.fit_skew(analysis['strikes'], underlying)
            term_structure.append({
                'expiry': expiry_str,
                'time_to_expiry': analysis['time_to_expiry'],
                'rate_source': analysis['rate_source'],
                'risk_free_rate': analysis['risk_free_rate'],
                'atm_iv': skew.get('atm_iv'),
                'curve_fit': skew.get('curve_fit'),
                'residual_std': skew.get('residual_std'),
                'points_used': len(skew.get('points', [])),
                'error': skew.get('error')
            })

        return jsonify({
            'symbol': symbol.upper(),
            'underlying_value': underlying,
            'term_structure': term_structure
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/india_market_sentiment', methods=['GET'])
def get_india_market_sentiment():
    """India equivalent of /api/market_sentiment -- feeds the dashboard widget in India mode."""
    try:
        if not INDIA_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'India market data features are not available'})

        symbol = request.args.get('symbol', 'NIFTY')
        india_provider = _india_market_data_provider

        india_vix = _market_data_provider.get_stock_price('^INDIAVIX')
        nifty = _market_data_provider.get_stock_price('^NSEI')
        rates = _india_risk_free_rate_provider.get_rates()
        market_status = india_provider.get_market_status()

        pcr = {'error': 'Option chain unavailable'}
        chain = india_provider.get_option_chain(symbol)
        if 'error' not in chain:
            pcr = calculate_put_call_ratio(chain['rows'])

        capital_market_status = None
        if 'error' not in market_status:
            for segment in market_status.get('marketState', []):
                if segment.get('market') == 'Capital Market':
                    capital_market_status = segment
                    break

        vix_level = india_vix.get('price') if 'error' not in india_vix else None
        fear_greed_score = max(0, min(100, 100 - (vix_level - 10) * 2)) if vix_level is not None else None

        return jsonify({
            'india_vix': {
                'vix_level': vix_level,
                'fear_greed_score': fear_greed_score
            } if vix_level is not None else india_vix,
            'pcr': pcr,
            'risk_free_rate': rates,
            'nifty': nifty if 'error' not in nifty else {'error': nifty.get('error')},
            'market_status': capital_market_status or market_status
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/india/risk_free_rate', methods=['GET'])
def get_india_risk_free_rate():
    """RBI policy rate + G-Sec yields, used as an India risk-free-rate proxy."""
    try:
        if not INDIA_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'India market data features are not available'})

        return jsonify(_india_risk_free_rate_provider.get_rates())

    except Exception as e:
        return jsonify({'error': str(e)})

def _calculate_break_even_points(spot_range, payoff):
    """Calculate break-even points where payoff crosses zero"""
    break_even_points = []
    for i in range(len(payoff) - 1):
        if (payoff[i] <= 0 and payoff[i+1] > 0) or (payoff[i] >= 0 and payoff[i+1] < 0):
            # Linear interpolation to find exact break-even point
            be_point = spot_range[i] + (spot_range[i+1] - spot_range[i]) * (-payoff[i] / (payoff[i+1] - payoff[i]))
            break_even_points.append(round(be_point, 2))
    return break_even_points

@app.route('/api/status')
def deployment_status():
    """Check deployment status and feature availability"""
    import sys
    import platform
    
    status = {
        'deployment': 'success',
        'python_version': sys.version,
        'platform': platform.platform(),
        'features': {
            'monte_carlo': MONTE_CARLO_AVAILABLE,
            'market_data': BASIC_MARKET_DATA_AVAILABLE,
            'advanced_pricing': ADVANCED_PRICING_AVAILABLE,
            'india_market_data': INDIA_MARKET_DATA_AVAILABLE,
            'overall_advanced': ADVANCED_FEATURES_AVAILABLE
        },
        'core_libraries': {}
    }
    
    # Test core library versions
    try:
        import numpy as np
        status['core_libraries']['numpy'] = np.__version__
    except ImportError:
        status['core_libraries']['numpy'] = 'not available'
    
    try:
        import scipy
        status['core_libraries']['scipy'] = scipy.__version__
    except ImportError:
        status['core_libraries']['scipy'] = 'not available'
    
    try:
        import pandas as pd
        status['core_libraries']['pandas'] = pd.__version__
    except ImportError:
        status['core_libraries']['pandas'] = 'not available'
    
    try:
        import matplotlib
        status['core_libraries']['matplotlib'] = matplotlib.__version__
    except ImportError:
        status['core_libraries']['matplotlib'] = 'not available'
    
    return jsonify(status)

# =================== UTILITY FUNCTIONS ===================

def validate_numeric_inputs(data, required_fields):
    """Validate that required fields exist and are finite numbers.

    Returns (validated_values, None) on success, or (None, error_dict) on
    failure. Callers do `values, error = validate_numeric_inputs(...)` then
    `if error: return jsonify(error), 400` -- the error slot must hold the
    JSON-able error body, not a bare status code.

    This only rejects NaN/inf and non-numeric input; it does not enforce
    sign (e.g. S > 0), since some callers validate fields like
    portfolio_delta that are legitimately negative or zero. Domain-specific
    bounds belong in the caller.
    """
    if not data:
        return None, {'error': 'No data provided'}

    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        return None, {'error': f'Missing required fields: {", ".join(missing_fields)}'}

    validated_values = {}
    for field in required_fields:
        try:
            value = float(data[field])
        except (ValueError, TypeError):
            return None, {'error': f'Invalid numeric input for {field}: {data[field]!r}'}
        if not math.isfinite(value):
            return None, {'error': f'{field} must be a finite number, got {value}'}
        validated_values[field] = value

    return validated_values, None
