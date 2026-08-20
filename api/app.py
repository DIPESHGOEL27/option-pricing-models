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
    from .market_data import MarketDataProvider, VolatilityEstimator, MarketSentimentIndicators, RiskFreeRateProvider
    BASIC_MARKET_DATA_AVAILABLE = True
except ImportError:
    try:
        from market_data import MarketDataProvider, VolatilityEstimator, MarketSentimentIndicators, RiskFreeRateProvider
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
    from advanced_risk import AdvancedRiskManager, RiskMetrics as RiskMetricsAdvanced, StressTestScenario
    RISK_FEATURES_AVAILABLE = True
except ImportError:
    RISK_FEATURES_AVAILABLE = False

# Try to import ML modules
try:
    from .ml_pricing import NeuralNetworkPricer, EnsembleOptionPricer, VolatilityPredictor, create_sample_data
    print("Successfully imported ML modules from .ml_pricing")
    ML_FEATURES_AVAILABLE = True
except ImportError:
    try:
        # Try with the api prefix
        from api.ml_pricing import NeuralNetworkPricer, EnsembleOptionPricer, VolatilityPredictor, create_sample_data
        print("Successfully imported ML modules from api.ml_pricing")
        ML_FEATURES_AVAILABLE = True
    except ImportError:
        try:
            # Try without any prefix
            from ml_pricing import NeuralNetworkPricer, EnsembleOptionPricer, VolatilityPredictor, create_sample_data
            print("Successfully imported ML modules from ml_pricing")
            ML_FEATURES_AVAILABLE = True
        except ImportError:
            print("Failed to import ML modules from any location")
            ML_FEATURES_AVAILABLE = False

try:
    from model_validation import ModelValidator, BacktestResults
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

try:
    from advanced_models import MonteCarloEngine, HestonCalibration
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    ADVANCED_PRICING_AVAILABLE = False

# Check overall advanced features availability
ADVANCED_FEATURES_AVAILABLE = any([
    MONTE_CARLO_AVAILABLE, RISK_FEATURES_AVAILABLE, BASIC_MARKET_DATA_AVAILABLE,
    ML_FEATURES_AVAILABLE, VALIDATION_AVAILABLE, ADVANCED_PRICING_AVAILABLE,
    INDIA_MARKET_DATA_AVAILABLE
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

_ensemble_pricer_cache = {}


def get_cached_ensemble_pricer():
    """Return a lazily-trained, process-wide EnsembleOptionPricer.

    Interactive pricing endpoints need a fast response; retraining all three
    ensemble models from scratch on every request (as this used to do) adds
    several seconds of latency to every click for no benefit, since nothing
    about the training data depends on the request.
    """
    if 'pricer' not in _ensemble_pricer_cache:
        training_data = create_sample_data(5000)
        pricer = EnsembleOptionPricer()
        pricer.train(training_data)
        _ensemble_pricer_cache['pricer'] = pricer
    return _ensemble_pricer_cache['pricer']


@app.route('/')
def index():
    return render_template('index.html')

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
        delta = si.norm.cdf(d1, 0.0, 1.0)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
        delta = -si.norm.cdf(-d1, 0.0, 1.0)
    
    gamma = si.norm.pdf(d1, 0.0, 1.0) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0.0, 1.0) * np.sqrt(T)
    theta = (-S * si.norm.pdf(d1, 0.0, 1.0) * sigma / (2 * np.sqrt(T)) 
             - r * K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0))
    rho = K * T * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)

    return price, delta, gamma, theta, vega, rho

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
    S = float(data['S'])
    K = float(data['K'])
    T = float(data['T'])
    r = float(data['r'])
    sigma = float(data['sigma'])
    steps = int(data['steps'])
    option_type = data['optionType']
    price = binomial_tree(S, K, T, r, sigma, steps, option_type)
    return jsonify({'option_price': price})

# Advanced API endpoints

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
        n_simulations = int(data.get('simulations', 100000))
        model_type = data.get('model', 'gbm')  # gbm, heston, jump_diffusion
        
        mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=252)
        
        if model_type == 'gbm':
            paths = mc_engine.geometric_brownian_motion(S, T, r, sigma)
        elif model_type == 'heston':
            # Heston parameters (could be user inputs)
            kappa = float(data.get('kappa', 2.0))
            theta = float(data.get('theta', 0.04))
            sigma_v = float(data.get('sigma_v', 0.3))
            rho = float(data.get('rho', -0.5))
            v0 = float(data.get('v0', 0.04))
            paths, vol_paths = mc_engine.heston_model(S, T, r, v0, kappa, theta, sigma_v, rho)
        elif model_type == 'jump_diffusion':
            # Jump-diffusion parameters
            lam = float(data.get('lambda', 0.1))
            mu_j = float(data.get('mu_j', -0.1))
            sigma_j = float(data.get('sigma_j', 0.2))
            paths = mc_engine.jump_diffusion_merton(S, T, r, sigma, lam, mu_j, sigma_j)
        else:
            return jsonify({'error': 'Invalid model type'})
        
        result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)
        
        # Calculate Greeks using Monte Carlo
        greeks = RiskMetrics.calculate_greeks_mc(S, K, T, r, sigma, option_type, n_simulations//10)
        
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
        return jsonify({'error': str(e)})

@app.route('/api/market_data/<symbol>', methods=['GET'])
def get_market_data(symbol):
    """Get real-time market data"""
    try:
        market_data = MarketDataProvider()
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

@app.route('/api/option_chain/<symbol>', methods=['GET'])
def get_option_chain(symbol):
    """Get option chain data"""
    try:
        market_data = MarketDataProvider()
        expiry = request.args.get('expiry')
        option_data = market_data.get_option_chain(symbol.upper(), expiry)
        
        if 'error' in option_data:
            return jsonify(option_data)
        
        # Convert DataFrames to dictionaries for JSON serialization
        result = {
            'calls': option_data['calls'].to_dict('records'),
            'puts': option_data['puts'].to_dict('records'),
            'expiry': option_data['expiry'],
            'underlying_price': option_data['underlying_price']
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/volatility_surface/<symbol>', methods=['GET'])
def get_volatility_surface(symbol):
    """Get implied volatility surface"""
    try:
        market_data = MarketDataProvider()
        vol_surface = market_data.get_volatility_surface(symbol.upper())
        
        if 'error' in vol_surface:
            return jsonify(vol_surface)
        
        # Convert DataFrame to records for JSON
        result = {
            'volatility_surface': vol_surface['volatility_surface'].to_dict('records'),
            'underlying_price': vol_surface['underlying_price'],
            'surface_summary': vol_surface['surface_summary']
        }
        
        return jsonify(result)
        
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

# =================== MACHINE LEARNING API ENDPOINTS ===================

@app.route('/api/ml/train_neural_network', methods=['POST'])
def train_neural_network_pricer():
    """Train neural network option pricing model"""
    try:
        if not ML_FEATURES_AVAILABLE:
            return jsonify({
                'error': 'ML features not available in this deployment',
                'fallback': 'Using standard Black-Scholes pricing'
            }), 503
            
        data = request.json
        
        # Create sample training data using the utility function
        n_samples = int(data.get('n_samples', 10000))
        
        # Import the sample data creation function
        from ml_pricing import create_sample_data
        training_data = create_sample_data(n_samples)
        
        # Create and train the neural network
        nn_pricer = NeuralNetworkPricer()
        performance = nn_pricer.train(training_data)
        
        # Save the model
        model_id = f"nn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        nn_pricer.save_model(f"models/{model_id}.joblib")
        
        return jsonify({
            'model_id': model_id,
            'performance': performance,
            'training_samples': n_samples,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml/ensemble_price', methods=['POST'])
def calculate_ensemble_price():
    """Price options using ensemble ML models"""
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

        try:
            ensemble_pricer = get_cached_ensemble_pricer()
        except Exception as e:
            return jsonify({'error': f"ML Pricing Error: {str(e)}"})

        # Build the prediction row with the exact column names/types the
        # models were trained on (S/K/T/r/sigma, option_type as 'call'/'put'
        # string, and a matching intrinsic_value) -- using different names
        # here silently produces an all-zero/garbage feature vector at
        # predict time (see NeuralNetworkPricer.prepare_features and
        # EnsembleOptionPricer._prepare_tree_features in ml_pricing.py).
        intrinsic_value = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        prediction_data = pd.DataFrame({
            'S': [S],
            'K': [K],
            'T': [T],
            'r': [r],
            'sigma': [sigma],
            'option_type': [option_type],
            'intrinsic_value': [intrinsic_value]
        })

        # Get ensemble prediction
        ml_prices = ensemble_pricer.predict(prediction_data)
        ml_price = ml_prices[0]
        
        # Compare with Black-Scholes
        bs_price, delta, gamma, theta, vega, rho = black_scholes(S, K, T, r, sigma, option_type)
        
        return jsonify({
            'ml_price': float(ml_price),
            'black_scholes_price': bs_price,
            'price_difference': float(ml_price) - bs_price,
            'relative_difference': (float(ml_price) - bs_price) / bs_price * 100,
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ml/volatility_forecast', methods=['POST'])
def forecast_volatility():
    """Forecast volatility from real historical price data using EWMA."""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})

        symbol = data.get('symbol', 'AAPL') if data else 'AAPL'
        horizon_days = int(data.get('horizon_days', 30)) if data else 30

        market_data = MarketDataProvider()
        hist = market_data.get_historical_data(symbol, period="1y")

        if hist is None or hist.empty or 'Close' not in hist.columns:
            return jsonify({'error': f'No historical data available for {symbol}'})

        close = hist['Close'].dropna()
        if len(close) < 60:
            return jsonify({'error': f'Insufficient historical data for {symbol} to estimate volatility'})

        simple_vol = VolatilityEstimator.historical_volatility(close, window=30, method='simple').dropna()
        ewma_vol = VolatilityEstimator.historical_volatility(close, window=30, method='ewma').dropna()

        if simple_vol.empty or ewma_vol.empty:
            return jsonify({'error': f'Unable to compute volatility for {symbol}'})

        current_vol = float(simple_vol.iloc[-1])
        # EWMA weights recent observations more heavily -- a standard,
        # genuinely forward-leaning volatility estimate, not a random guess.
        forecast_vol = float(ewma_vol.iloc[-1])

        # Uncertainty band from the real dispersion of the rolling vol
        # series itself, not an arbitrary +/-20%.
        recent_vol = simple_vol.tail(60) if len(simple_vol) >= 60 else simple_vol
        vol_std = float(recent_vol.std())

        return jsonify({
            'symbol': symbol,
            'current_volatility': current_vol,
            'forecasted_volatility': forecast_vol,
            'forecast_horizon_days': horizon_days,
            'confidence_interval_lower': max(0.0, forecast_vol - vol_std),
            'confidence_interval_upper': forecast_vol + vol_std,
            'method': 'EWMA of 30-day realized volatility, computed from real historical prices',
            'observations': int(len(close))
        })

    except Exception as e:
        return jsonify({'error': str(e)})

# =================== ADVANCED RISK MANAGEMENT API ENDPOINTS ===================

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

        risk_manager = AdvancedRiskManager()
        market_data = MarketDataProvider()

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
        horizon_scale = np.sqrt(time_horizon)
        es_result = risk_manager.calculate_expected_shortfall(
            portfolio_returns, confidence_level
        ) * horizon_scale

        historical_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100) * horizon_scale
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

# =================== ADVANCED MARKET DATA API ENDPOINTS ===================

@app.route('/api/market/sentiment', methods=['GET'])
def get_market_sentiment():
    """Get real market sentiment indicators (VIX + put/call ratio) for US or India."""
    try:
        market = request.args.get('market', 'us').lower()

        if market == 'india':
            if not INDIA_MARKET_DATA_AVAILABLE:
                return jsonify({'error': 'India market data features are not available'})

            symbol = request.args.get('symbol', 'NIFTY')
            india_data = MarketDataProvider().get_stock_price('^INDIAVIX')
            if 'error' in india_data:
                return jsonify(india_data)
            vix_level = india_data['price']

            chain = IndiaMarketDataProvider().get_option_chain(symbol)
            if 'error' in chain:
                return jsonify(chain)
            pcr_data = calculate_put_call_ratio(chain['rows'])

            fear_greed_index = max(0, min(100, 100 - (vix_level - 10) * 2))
            overall_sentiment = 'greedy' if fear_greed_index > 60 else 'fearful' if fear_greed_index < 40 else 'neutral'

            return jsonify({
                'symbol': symbol,
                'market': 'india',
                'sentiment_indicators': {
                    'fear_greed_index': fear_greed_index,
                    'put_call_ratio': pcr_data['oi_pcr'],
                    'vix_level': vix_level
                },
                'overall_sentiment': overall_sentiment,
                'market_regime': 'high_volatility' if vix_level > 25 else 'low_volatility' if vix_level < 15 else 'normal'
            })

        symbol = request.args.get('symbol', 'SPY')
        sentiment = MarketSentimentIndicators()

        vix_data = sentiment.get_vix_data()
        if not vix_data or 'error' in vix_data:
            return jsonify(vix_data or {'error': 'Failed to fetch VIX data'})

        pcr_data = sentiment.get_put_call_ratio(symbol)
        put_call_ratio = pcr_data.get('put_call_ratio', 0) if pcr_data and 'error' not in pcr_data else 0

        overall_sentiment = 'greedy' if vix_data['fear_greed_score'] > 60 else 'fearful' if vix_data['fear_greed_score'] < 40 else 'neutral'

        return jsonify({
            'symbol': symbol,
            'market': 'us',
            'sentiment_indicators': {
                'fear_greed_index': vix_data['fear_greed_score'],
                'put_call_ratio': put_call_ratio,
                'vix_level': vix_data['vix_level']
            },
            'overall_sentiment': overall_sentiment,
            'market_regime': 'high_volatility' if vix_data['vix_level'] > 25 else 'low_volatility' if vix_data['vix_level'] < 15 else 'normal'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market_sentiment', methods=['GET'])
def get_market_sentiment_simple():
    """Get market sentiment indicators for dashboard"""
    try:
        # Initialize response data with fallback values
        response_data = {
            'vix': {
                'vix_level': 20.5,
                'sentiment': "Moderate Fear",
                'fear_greed_score': 55
            },
            'put_call_ratio': {
                'put_call_ratio': 1.05,
                'sentiment': "Neutral"
            },
            'treasury_rates': {
                '10Y': 0.045  # 4.5%
            }
        }
        
        # Try to get real market data
        try:
            market_data = MarketDataProvider()
            
            # Get VIX data
            try:
                vix_data = market_data.get_stock_price('^VIX')
                if 'error' not in vix_data and 'price' in vix_data:
                    vix_level = vix_data['price']
                    if vix_level < 20:
                        sentiment = "Low Fear"
                        fear_greed_score = 70 + (20 - vix_level) * 1.5  # Higher score for low VIX
                    elif vix_level < 30:
                        sentiment = "Moderate Fear"
                        fear_greed_score = 50 + (25 - vix_level) * 2
                    else:
                        sentiment = "High Fear"
                        fear_greed_score = 30 - (vix_level - 30) * 1.5  # Lower score for high VIX
                        
                    response_data['vix'] = {
                        'vix_level': vix_level,
                        'sentiment': sentiment,
                        'fear_greed_score': max(0, min(100, fear_greed_score))
                    }
            except Exception as e:
                print(f"VIX data error: {e}")
                # Keep fallback VIX data
                pass
            
            # Put/Call Ratio from the real option chain
            pcr_result = MarketSentimentIndicators().get_put_call_ratio('SPY')
            put_call_ratio = pcr_result.get('put_call_ratio', 1.0) if pcr_result and 'error' not in pcr_result else 1.0
            response_data['put_call_ratio'] = {
                'put_call_ratio': put_call_ratio,
                'sentiment': "Bearish" if put_call_ratio > 1.1 else "Bullish" if put_call_ratio < 0.9 else "Neutral"
            }

            # Get Treasury rates
            try:
                treasury_data = market_data.get_stock_price('^TNX')
                if 'error' not in treasury_data and 'price' in treasury_data:
                    response_data['treasury_rates'] = {
                        '10Y': treasury_data['price'] / 100  # Convert percentage to decimal
                    }
            except Exception as e:
                print(f"Treasury data error: {e}")
                # Keep fallback treasury data
                pass
                
        except Exception as e:
            print(f"MarketDataProvider error: {e}")
            # response_data already holds the static fallback values defined
            # above -- leave them as-is rather than fabricating random noise
            # that would look like live market movement.
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Market sentiment endpoint error: {e}")
        # Return absolute fallback
        return jsonify({
            'vix': {
                'vix_level': 20.0,
                'sentiment': "Moderate Fear",
                'fear_greed_score': 50
            },
            'put_call_ratio': {
                'put_call_ratio': 1.0,
                'sentiment': "Neutral"
            },
            'treasury_rates': {
                '10Y': 0.045
            }
        })

@app.route('/api/market/volatility_term_structure', methods=['GET'])
def get_volatility_term_structure():
    """Get implied volatility term structure from real option chain data."""
    try:
        symbol = request.args.get('symbol', 'SPY')

        market_data = MarketDataProvider()
        vol_data = market_data.get_volatility_surface(symbol)

        if 'error' in vol_data:
            return jsonify(vol_data)

        df = vol_data.get('volatility_surface')
        if df is None or df.empty:
            return jsonify({'error': f'No option chain data available for {symbol}'})

        # One term-structure point per real listed expiry -- open-interest
        # weighted average implied vol across strikes for that expiry.
        term_structure = {}
        days_by_expiry = {}
        for _, group in df.groupby('expiry'):
            valid = group.dropna(subset=['implied_vol'])
            valid = valid[valid['implied_vol'] > 0]
            if valid.empty:
                continue
            days = max(0, int(round(valid['tte'].iloc[0] * 365)))
            weights = valid['open_interest'].fillna(0).replace(0, 1)
            iv = float(np.average(valid['implied_vol'], weights=weights))
            key = f"{days}d"
            term_structure[key] = iv
            days_by_expiry[days] = key

        if not term_structure:
            return jsonify({'error': f'Unable to build a volatility term structure for {symbol}'})

        sorted_days = sorted(days_by_expiry.keys())
        shortest_key = days_by_expiry[sorted_days[0]]
        longest_key = days_by_expiry[sorted_days[-1]]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sorted_days,
            y=[term_structure[days_by_expiry[d]] for d in sorted_days],
            mode='markers+lines',
            name='Implied Volatility',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f'Volatility Term Structure - {symbol}',
            xaxis_title='Days to Expiration',
            yaxis_title='Implied Volatility',
            template='plotly_dark'
        )

        graphJSON = json.dumps(fig, cls=PlotlyJSONEncoder)

        return jsonify({
            'symbol': symbol,
            'term_structure': term_structure,
            'plot': graphJSON,
            'analysis': {
                'contango': term_structure[longest_key] > term_structure[shortest_key],
                'backwardation': term_structure[shortest_key] > term_structure[longest_key],
                'short_term_vol': term_structure[shortest_key],
                'long_term_vol': term_structure[longest_key]
            }
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
        
        # Generate spot price range
        spot_range = np.linspace(80, 120, 100)
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
                'max_profit': float(np.max(total_payoff)),
                'max_loss': float(np.min(total_payoff)),
                'break_even_points': _calculate_break_even_points(spot_range, total_payoff),
                'profit_probability': float(np.mean(total_payoff > 0))
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
        market_data = MarketDataProvider()
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
    """Real NSE F&O option chain with PCR, max pain, and an OI buildup chart."""
    try:
        if not INDIA_MARKET_DATA_AVAILABLE:
            return jsonify({'error': 'India market data features are not available'})

        expiry = request.args.get('expiry')
        chain = IndiaMarketDataProvider().get_option_chain(symbol.upper(), expiry)

        if 'error' in chain:
            return jsonify(chain)

        rows = chain['rows']
        pcr = calculate_put_call_ratio(rows)
        max_pain = calculate_max_pain(rows)
        buildup = summarize_oi_buildup(rows)

        strikes = buildup['strikes']
        max_pain_strike = max_pain.get('strike')
        underlying = chain.get('underlying_value')

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
            'strikes': rows,
            'pcr': pcr,
            'max_pain': max_pain,
            'plot': json.dumps({'data': traces, 'layout': layout}, cls=PlotlyJSONEncoder)
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
        india_provider = IndiaMarketDataProvider()

        india_vix = MarketDataProvider().get_stock_price('^INDIAVIX')
        nifty = MarketDataProvider().get_stock_price('^NSEI')
        rates = IndiaRiskFreeRateProvider().get_rates()
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

        return jsonify(IndiaRiskFreeRateProvider().get_rates())

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

# =================== PERFORMANCE ANALYTICS API ENDPOINTS ===================

@app.route('/api/analytics/performance_attribution', methods=['POST'])
def analyze_performance_attribution():
    """Analyze portfolio performance attribution"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        portfolio_returns = data['portfolio_returns']  # List of returns
        benchmark_returns = data.get('benchmark_returns', []) if data else []
        benchmark_symbol = data.get('benchmark_symbol', 'SPY') if data else 'SPY'

        if not benchmark_returns:
            # Default to a real benchmark (SPY daily returns) rather than
            # synthetic noise -- matches how real performance-attribution
            # tools default to a broad market index when none is supplied.
            hist = MarketDataProvider().get_historical_data(benchmark_symbol, period="2y")
            if hist is None or hist.empty or 'Returns' not in hist.columns:
                return jsonify({'error': f'No benchmark data available for {benchmark_symbol} and none was provided'})
            real_returns = hist['Returns'].dropna().tail(len(portfolio_returns))
            if len(real_returns) < len(portfolio_returns):
                return jsonify({'error': f'Not enough {benchmark_symbol} history to match the portfolio_returns length'})
            benchmark_returns = real_returns.tolist()

        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Calculate performance metrics
        portfolio_total_return = float((1 + portfolio_returns).prod() - 1)
        benchmark_total_return = float((1 + benchmark_returns).prod() - 1)
        excess_return = portfolio_total_return - benchmark_total_return
        
        # Risk metrics
        portfolio_vol = float(np.std(portfolio_returns) * np.sqrt(252))
        benchmark_vol = float(np.std(benchmark_returns) * np.sqrt(252))
        tracking_error = float(np.std(portfolio_returns - benchmark_returns) * np.sqrt(252))
        
        # Sharpe ratios (assuming risk-free rate of 2%)
        rf_rate = 0.02
        portfolio_sharpe = float((np.mean(portfolio_returns) * 252 - rf_rate) / portfolio_vol)
        benchmark_sharpe = float((np.mean(benchmark_returns) * 252 - rf_rate) / benchmark_vol)
        
        # Information ratio
        information_ratio = float((np.mean(portfolio_returns - benchmark_returns) * 252) / tracking_error)
        
        # Beta calculation
        covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
        beta = float(covariance_matrix[0, 1] / np.var(benchmark_returns))
        correlation = float(np.corrcoef(portfolio_returns, benchmark_returns)[0, 1])
        
        return jsonify({
            'performance_metrics': {
                'portfolio_return': portfolio_total_return,
                'benchmark_return': benchmark_total_return,
                'excess_return': excess_return,
                'portfolio_volatility': portfolio_vol,
                'benchmark_volatility': benchmark_vol,
                'tracking_error': tracking_error,
                'portfolio_sharpe': portfolio_sharpe,
                'benchmark_sharpe': benchmark_sharpe,
                'information_ratio': information_ratio
            },
            'attribution_analysis': {
                'alpha': excess_return,
                'beta': beta,
                'correlation': correlation
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

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
            'risk_features': RISK_FEATURES_AVAILABLE,
            'market_data': BASIC_MARKET_DATA_AVAILABLE,
            'ml_features': ML_FEATURES_AVAILABLE,
            'validation': VALIDATION_AVAILABLE,
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

@app.route('/api/ml/benchmark', methods=['POST'])
def ml_model_benchmark():
    """Benchmark ML models on 50,000+ synthetic records and report real validation R²."""
    try:
        # Generate large training dataset
        print("Generating 50,000+ training records...")
        training_data = create_sample_data(50000)
        
        # Train Neural Network
        nn_pricer = NeuralNetworkPricer(
            hidden_layers=(200, 100, 50, 25),  # Deeper network
            activation='relu',
            solver='adam',
            learning_rate=0.001,
            max_iter=2000
        )
        
        print("Training neural network on 50,000+ records...")
        nn_metrics = nn_pricer.train(training_data)
        
        # Train Ensemble Model
        try:
            ensemble_pricer = EnsembleOptionPricer(['neural_network', 'xgboost', 'random_forest'])
            ensemble_metrics = ensemble_pricer.train(training_data)
        except Exception as e:
            print(f"Error initializing ensemble pricer: {str(e)}")
            # Provide default metrics if ensemble fails
            ensemble_metrics = {'neural_network': {'val_r2': 0}, 'xgboost': {'val_r2': 0}, 'random_forest': {'val_r2': 0}}

        # Calculate performance metrics
        best_r2 = max(nn_metrics.get('val_r2', 0),
                     max(model_metrics.get('val_r2', 0) for model_metrics in ensemble_metrics.values()))

        benchmark_results = {
            'dataset_size': len(training_data),
            'neural_network_metrics': nn_metrics,
            'ensemble_metrics': ensemble_metrics,
            'best_validation_r2': best_r2,
            'training_features': len(training_data.columns) - 1,
            'performance_summary': {
                'achieved_r2': best_r2,
                'training_records': len(training_data),
                'model_complexity': 'Neural Network + Random Forest + XGBoost Ensemble'
            }
        }
        
        return jsonify(benchmark_results)
        
    except Exception as e:
        return jsonify({'error': f'ML benchmark error: {str(e)}'})

# =================== UTILITY FUNCTIONS ===================

def validate_numeric_inputs(data, required_fields):
    """Validate that required numeric fields exist and contain valid numbers"""
    if not data:
        return {'error': 'No data provided'}, 400
        
    # Check for required fields
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    if missing_fields:
        return {'error': f'Missing required fields: {", ".join(missing_fields)}'}, 400
        
    # Convert and validate numeric fields
    validated_values = {}
    try:
        for field in required_fields:
            value = float(data[field])
            if math.isnan(value):
                return {'error': f'Invalid numeric value for {field}'}, 400
            validated_values[field] = value
    except (ValueError, TypeError) as e:
        return {'error': f'Invalid numeric input for {field}: {str(e)}'}, 400
        
    return validated_values, None
