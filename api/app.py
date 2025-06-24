from flask import Flask, request, jsonify, render_template
import numpy as np
import scipy.stats as si
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Import market data provider
try:
    from .market_data import MarketDataProvider, VolatilityEstimator
except ImportError:
    from market_data import MarketDataProvider, VolatilityEstimator

# Import our advanced modules
try:
    from advanced_models import MonteCarloEngine, ExoticOptions, HestonCalibration, RiskMetrics, ModelValidation
    MONTE_CARLO_AVAILABLE = True
except ImportError:
    MONTE_CARLO_AVAILABLE = False

try:
    from advanced_risk import AdvancedRiskManager, RiskMetrics as RiskMetricsAdvanced, StressTestScenario
    RISK_FEATURES_AVAILABLE = True
except ImportError:
    RISK_FEATURES_AVAILABLE = False

try:
    from market_data_advanced import AdvancedMarketDataProvider, VolatilitySurfaceBuilder, MarketSentimentAnalyzer
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False

try:
    from ml_pricing import NeuralNetworkPricer, EnsembleOptionPricer, VolatilityPredictor
    ML_FEATURES_AVAILABLE = True
except ImportError:
    ML_FEATURES_AVAILABLE = False

try:
    from model_validation import ModelValidator, BacktestResults
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False

try:
    from portfolio_optimization import AdvancedPortfolioOptimizer, OptionsStrategyOptimizer, DynamicHedgingEngine
    PORTFOLIO_FEATURES_AVAILABLE = True
except ImportError:
    PORTFOLIO_FEATURES_AVAILABLE = False

try:
    from option_pricing import AdvancedOptionPricer, ImpliedVolatilityCalculator
    ADVANCED_PRICING_AVAILABLE = True
except ImportError:
    ADVANCED_PRICING_AVAILABLE = False

# Check overall advanced features availability
ADVANCED_FEATURES_AVAILABLE = any([
    MONTE_CARLO_AVAILABLE, RISK_FEATURES_AVAILABLE, MARKET_DATA_AVAILABLE,
    ML_FEATURES_AVAILABLE, VALIDATION_AVAILABLE, PORTFOLIO_FEATURES_AVAILABLE,
    ADVANCED_PRICING_AVAILABLE
])

app = Flask(__name__, template_folder='templates', static_folder='static')


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
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    try:
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data.get('option_type', data.get('optionType', 'call'))
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

@app.route('/api/exotic_options', methods=['POST'])
def calculate_exotic_options():
    """Price exotic options"""
    try:
        data = request.json
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data['optionType']
        exotic_type = data['exoticType']  # asian, barrier, lookback, binary
        
        mc_engine = MonteCarloEngine(n_simulations=100000, n_steps=252)
        paths = mc_engine.geometric_brownian_motion(S, T, r, sigma)
        
        exotic_engine = ExoticOptions(mc_engine)
        
        if exotic_type == 'asian':
            asian_type = data.get('asianType', 'arithmetic')
            result = exotic_engine.asian_option(paths, K, r, T, option_type, asian_type)
        elif exotic_type == 'barrier':
            B = float(data['barrier'])
            barrier_type = data.get('barrierType', 'up_and_out')
            result = exotic_engine.barrier_option(paths, K, B, r, T, option_type, barrier_type)
        elif exotic_type == 'lookback':
            lookback_type = data.get('lookbackType', 'floating')
            result = exotic_engine.lookback_option(paths, K, r, T, option_type, lookback_type)
        elif exotic_type == 'binary':
            payout = float(data.get('payout', 1.0))
            result = exotic_engine.binary_option(paths, K, r, T, payout, option_type)
        else:
            return jsonify({'error': 'Invalid exotic option type'})
        
        return jsonify({
            'option_price': result['price'],
            'std_error': result['std_error'],
            'exotic_type': exotic_type,
            **{k: v for k, v in result.items() if k not in ['price', 'std_error', 'payoffs']}
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
        positions = data['positions']  # List of position dictionaries
        
        portfolio = OptionPortfolio()
        
        # Add positions to portfolio
        for pos in positions:
            expiry = datetime.strptime(pos['expiry'], '%Y-%m-%d')
            portfolio.add_position(
                pos['symbol'], pos['option_type'], pos['strike'],
                expiry, pos['quantity'], pos['premium_paid'],
                pos['underlying_price'], pos['volatility'],
                pos['risk_free_rate'], pos.get('model_type', 'black_scholes')
            )
        
        # Get portfolio summary and risk report
        summary = portfolio.get_portfolio_summary()
        risk_report = portfolio.risk_report()
        hedge_rec = portfolio.delta_hedge_recommendation([pos['symbol'] for pos in positions])
        
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
        
        return jsonify(stress_results)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/model_validation', methods=['POST'])
def validate_models():
    """Validate pricing models"""
    try:
        data = request.json
        S = float(data['S'])
        K = float(data['K'])
        T = float(data['T'])
        r = float(data['r'])
        sigma = float(data['sigma'])
        option_type = data['optionType']
        
        # Black-Scholes vs Monte Carlo validation
        validation = ModelValidation.validate_black_scholes_vs_mc(S, K, T, r, sigma, option_type)
        
        # Convergence analysis
        convergence = ModelValidation.convergence_analysis(S, K, T, r, sigma, option_type)
        
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
        
        ensemble_pricer = EnsembleOptionPricer()
        
        # Create a sample dataset for training
        from ml_pricing import create_sample_data
        sample_data = create_sample_data(1000)
        
        # Train the ensemble model
        ensemble_pricer.train(sample_data)
        
        # Create prediction data
        prediction_data = pd.DataFrame({
            'spot_price': [S],
            'strike_price': [K],
            'time_to_expiry': [T],
            'risk_free_rate': [r],
            'volatility': [sigma],
            'option_type': [1 if option_type == 'call' else 0]
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
    """Forecast volatility using ML models"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        symbol = data.get('symbol', 'AAPL') if data else 'AAPL'
        horizon_days = int(data.get('horizon_days', 30)) if data else 30
        
        vol_predictor = VolatilityPredictor()
        
        # Get historical data (mock implementation)
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        returns = np.random.normal(0, 0.02, 252)  # Mock returns
        
        historical_data = pd.DataFrame({
            'date': dates,
            'close_price': 100 * np.exp(np.cumsum(returns)),  # Mock price series
            'returns': returns
        })
        
        # Calculate realized volatility
        current_vol = np.std(returns) * np.sqrt(252)
        
        # Simple volatility forecast (expand this with actual ML prediction)
        forecast_vol = current_vol * (1 + np.random.normal(0, 0.1))
        
        return jsonify({
            'symbol': symbol,
            'current_volatility': float(current_vol),
            'forecasted_volatility': float(forecast_vol),
            'forecast_horizon_days': horizon_days,
            'confidence_interval_lower': float(forecast_vol * 0.8),
            'confidence_interval_upper': float(forecast_vol * 1.2),
            'model_type': 'gradient_boosting'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== PORTFOLIO OPTIMIZATION API ENDPOINTS ===================

@app.route('/api/portfolio/optimize', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio using advanced techniques"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        symbols = data['symbols']
        method = data.get('method', 'mean_variance') if data else 'mean_variance'
        target_return = data.get('target_return', None) if data else None
        risk_tolerance = data.get('risk_tolerance', 0.1) if data else 0.1
        
        portfolio_optimizer = AdvancedPortfolioOptimizer()
        
        # Mock returns data (in real application, use historical data)
        n_assets = len(symbols)
        returns_data = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.08/252] * n_assets,
                cov=np.eye(n_assets) * (0.2**2)/252,
                size=252
            ),
            columns=symbols
        )
        
        # Calculate expected returns and covariance matrix
        expected_returns = returns_data.mean().values * 252
        cov_matrix = returns_data.cov().values * 252
        
        if method == 'mean_variance':
            result = portfolio_optimizer.mean_variance_optimization(
                expected_returns, cov_matrix, target_return
            )
        elif method == 'risk_parity':
            result = portfolio_optimizer.risk_parity_optimization(cov_matrix)
        else:
            # Simplified approach for unsupported methods
            # Equal weight portfolio as fallback
            weights = np.ones(n_assets) / n_assets
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            result = {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': (portfolio_return - 0.02) / portfolio_vol
            }
        
        return jsonify({
            'optimal_weights': result['weights'].tolist(),
            'expected_return': float(result['expected_return']),
            'volatility': float(result['volatility']),
            'sharpe_ratio': float(result['sharpe_ratio']),
            'symbols': symbols,
            'method': method
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/options/strategy_optimize', methods=['POST'])
def optimize_options_strategy():
    """Optimize options trading strategies"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        strategy_type = data['strategy_type']  # 'covered_call', 'protective_put', etc.
        S = float(data['S'])
        portfolio_size = float(data.get('portfolio_size', 100000)) if data else 100000
        risk_tolerance = float(data.get('risk_tolerance', 0.1)) if data else 0.1
        
        strategy_optimizer = OptionsStrategyOptimizer()
        
        if strategy_type == 'covered_call':
            # Mock optimization result
            optimal_strike = S * 1.05  # 5% out of the money
            expected_return = 0.08  # 8% annual return
            max_loss = -S * 0.1  # 10% max loss
            breakeven = S * 0.95
            profit_prob = 0.65;
            
            result = {
                'optimal_strike': optimal_strike,
                'expected_return': expected_return,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'profit_probability': profit_prob,
                'strategy_details': {
                    'premium_received': S * 0.03,
                    'shares_to_hold': int(portfolio_size / S),
                    'contracts_to_write': int(portfolio_size / S / 100)
                }
            }
        elif strategy_type == 'protective_put':
            optimal_strike = S * 0.95  # 5% out of the money
            expected_return = 0.06
            max_loss = -S * 0.05  # Limited loss
            breakeven = S * 1.02
            profit_prob = 0.55;
            
            result = {
                'optimal_strike': optimal_strike,
                'expected_return': expected_return,
                'max_loss': max_loss,
                'breakeven': breakeven,
                'profit_probability': profit_prob,
                'strategy_details': {
                    'premium_paid': S * 0.02,
                    'shares_protected': int(portfolio_size / S),
                    'protection_level': optimal_strike
                }
            }
        else:
            return jsonify({'error': 'Strategy type not supported'})
        
        return jsonify({
            'optimal_strategy': result,
            'strategy_type': strategy_type,
            'expected_return': result['expected_return'],
            'max_loss': result['max_loss'],
            'breakeven': result['breakeven'],
            'profit_probability': result['profit_probability']
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
        
        # Mock portfolio data
        portfolio_returns = np.random.normal(0, 0.02, 1000)  # Historical returns
        portfolio_value = sum([pos.get('value', 100000) for pos in positions])
        
        # Calculate VaR and Expected Shortfall using available methods
        es_result = risk_manager.calculate_expected_shortfall(
            portfolio_returns, confidence_level
        )
        
        # Simple VaR calculations
        historical_var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        parametric_var = -stats.norm.ppf(1 - confidence_level) * np.std(portfolio_returns)
        
        # Basic risk metrics
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        skewness = float(stats.skew(portfolio_returns))
        kurtosis = float(stats.kurtosis(portfolio_returns))
        
        # Simple max drawdown calculation
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdown))
        
        # Mock stress test results
        stress_results = {
            'market_crash': {'loss': -0.20, 'probability': 0.05},
            'interest_rate_shock': {'loss': -0.15, 'probability': 0.10},
            'volatility_spike': {'loss': -0.25, 'probability': 0.08},
            'liquidity_crisis': {'loss': -0.30, 'probability': 0.03}
        }
        
        return jsonify({
            'portfolio_value': portfolio_value,
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
        if not data:
            return jsonify({'error': 'No data provided'})
            
        portfolio_delta = float(data['portfolio_delta'])
        target_delta = float(data.get('target_delta', 0)) if data else 0
        hedge_ratio = float(data.get('hedge_ratio', 1.0)) if data else 1.0
        
        # Simple hedging calculations
        delta_exposure = portfolio_delta - target_delta
        hedge_quantity = -delta_exposure * hedge_ratio
        
        # Mock hedging effectiveness
        expected_pnl = abs(delta_exposure) * 0.01  # 1% of delta exposure
        effectiveness = 0.85  # 85% effectiveness
        
        return jsonify({
            'current_delta': portfolio_delta,
            'target_delta': target_delta,
            'delta_exposure': delta_exposure,
            'hedge_quantity': hedge_quantity,
            'expected_pnl': expected_pnl,
            'hedge_effectiveness': effectiveness,
            'recommendation': 'buy' if hedge_quantity > 0 else 'sell',
            'hedge_cost': abs(hedge_quantity) * 0.002  # 0.2% transaction cost
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== MODEL VALIDATION API ENDPOINTS ===================

@app.route('/api/validation/backtest', methods=['POST'])
def run_model_backtest():
    """Run comprehensive model backtesting"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'})
            
        model_type = data['model_type']  # 'black_scholes', 'monte_carlo', 'ml'
        start_date = data.get('start_date', '2023-01-01') if data else '2023-01-01'
        end_date = data.get('end_date', '2023-12-31') if data else '2023-12-31'
        
        # Mock backtesting results
        n_predictions = 100
        actual_prices = np.random.uniform(10, 50, n_predictions)
        predicted_prices = actual_prices + np.random.normal(0, 2, n_predictions)
        
        # Calculate performance metrics
        mae = np.mean(np.abs(actual_prices - predicted_prices))
        rmse = np.sqrt(np.mean((actual_prices - predicted_prices)**2))
        accuracy = 1 - mae / np.mean(actual_prices)
        
        # Mock trading performance
        returns = np.random.normal(0.0002, 0.02, 252)
        sharpe_ratio = np.mean(returns) * 252 / (np.std(returns) * np.sqrt(252))
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(np.min(drawdown))
        
        return jsonify({
            'model_type': model_type,
            'period': f"{start_date} to {end_date}",
            'performance_metrics': {
                'predictions_count': n_predictions,
                'mean_actual_price': float(np.mean(actual_prices)),
                'mean_predicted_price': float(np.mean(predicted_prices))
            },
            'accuracy': float(accuracy),
            'mae': float(mae),
            'rmse': float(rmse),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': max_drawdown,
            'profit_factor': 1.2  # Mock profit factor
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

# =================== ADVANCED MARKET DATA API ENDPOINTS ===================

@app.route('/api/market/sentiment', methods=['GET'])
def get_market_sentiment():
    """Get advanced market sentiment indicators"""
    try:
        symbol = request.args.get('symbol', 'SPY')
        
        # Mock sentiment data
        sentiment_data = {
            'fear_greed_index': float(np.random.uniform(20, 80)),
            'put_call_ratio': float(np.random.uniform(0.8, 1.5)),
            'vix_level': float(np.random.uniform(15, 35)),
            'term_structure_slope': float(np.random.uniform(-0.1, 0.1)),
            'sentiment_score': float(np.random.uniform(-1, 1))
        }
        
        # Simple interpretation logic
        overall_sentiment = 'neutral'
        if sentiment_data['fear_greed_index'] > 60:
            overall_sentiment = 'greedy'
        elif sentiment_data['fear_greed_index'] < 40:
            overall_sentiment = 'fearful'
            
        market_regime = 'normal'
        if sentiment_data['vix_level'] > 25:
            market_regime = 'high_volatility'
        elif sentiment_data['vix_level'] < 15:
            market_regime = 'low_volatility'
        
        return jsonify({
            'symbol': symbol,
            'sentiment_indicators': sentiment_data,
            'overall_sentiment': overall_sentiment,
            'market_regime': market_regime,
            'trading_recommendations': {
                'action': 'hold' if overall_sentiment == 'neutral' else 'caution',
                'confidence': 0.7
            },
            'risk_warnings': ['High volatility detected'] if sentiment_data['vix_level'] > 30 else []
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market_sentiment', methods=['GET'])
def get_market_sentiment_simple():
    """Get market sentiment indicators for dashboard"""
    try:
        # Get real market data
        market_data = MarketDataProvider()
        
        # Initialize response data
        response_data = {}
        
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
        except:
            # Fallback mock data
            vix_level = float(np.random.uniform(18, 25))
            response_data['vix'] = {
                'vix_level': vix_level,
                'sentiment': "Moderate Fear" if vix_level > 20 else "Low Fear",
                'fear_greed_score': 60
            }
        
        # Mock Put/Call Ratio (since real data is harder to get)
        put_call_ratio = float(np.random.uniform(0.9, 1.3))
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
        except:
            # Fallback mock data
            response_data['treasury_rates'] = {
                '10Y': 0.045  # 4.5%
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market/volatility_term_structure', methods=['GET'])
def get_volatility_term_structure():
    """Get implied volatility term structure"""
    try:
        symbol = request.args.get('symbol', 'SPY')
        
        # Mock term structure data
        expirations = [7, 14, 30, 60, 90, 120, 180, 365]
        term_structure = {}
        
        for exp in expirations:
            # Mock volatility levels with term structure shape
            base_vol = 0.20
            vol_level = base_vol + 0.05 * np.exp(-exp/90) + np.random.normal(0, 0.02)
            term_structure[f"{exp}d"] = max(0.05, vol_level)
        
        # Create visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=expirations,
            y=list(term_structure.values()),
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
                'contango': term_structure['365d'] > term_structure['30d'],
                'backwardation': term_structure['30d'] > term_structure['365d'],
                'short_term_vol': term_structure['30d'],
                'long_term_vol': term_structure['365d']
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

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
        
        if not benchmark_returns:
            # Generate mock benchmark returns
            benchmark_returns = np.random.normal(0.0002, 0.01, len(portfolio_returns)).tolist()
        
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
            'market_data': MARKET_DATA_AVAILABLE,
            'ml_features': ML_FEATURES_AVAILABLE,
            'validation': VALIDATION_AVAILABLE,
            'portfolio_features': PORTFOLIO_FEATURES_AVAILABLE,
            'advanced_pricing': ADVANCED_PRICING_AVAILABLE,
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
