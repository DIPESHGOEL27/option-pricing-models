"""
Comprehensive Test Suite for Advanced Option Pricing Platform

Tests cover:
- Basic option pricing models
- Advanced Monte Carlo simulations
- Exotic options pricing
- Portfolio management
- Risk metrics
- Model validation
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the api directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from advanced_models import MonteCarloEngine, ExoticOptions, RiskMetrics, ModelValidation
from portfolio_management import OptionPortfolio
from market_data import MarketDataProvider, VolatilityEstimator


class TestBlackScholesModel(unittest.TestCase):
    """Test basic Black-Scholes pricing"""
    
    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_call_option_pricing(self):
        """Test call option pricing accuracy"""
        from api.app import black_scholes
        
        price, delta, gamma, theta, vega, rho = black_scholes(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        # Known analytical result for ATM call
        expected_price = 5.987
        self.assertAlmostEqual(price, expected_price, places=2)
        
        # Delta should be around 0.5 for ATM call
        self.assertGreater(delta, 0.4)
        self.assertLess(delta, 0.6)
        
        # Gamma should be positive
        self.assertGreater(gamma, 0)
        
        # Vega should be positive
        self.assertGreater(vega, 0)
    
    def test_put_option_pricing(self):
        """Test put option pricing accuracy"""
        from api.app import black_scholes
        
        price, delta, gamma, theta, vega, rho = black_scholes(
            self.S, self.K, self.T, self.r, self.sigma, 'put'
        )
        
        # Put delta should be negative
        self.assertLess(delta, 0)
        
        # Put price should be positive
        self.assertGreater(price, 0)
    
    def test_put_call_parity(self):
        """Test put-call parity relationship"""
        from api.app import black_scholes
        
        call_price, _, _, _, _, _ = black_scholes(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        put_price, _, _, _, _, _ = black_scholes(
            self.S, self.K, self.T, self.r, self.sigma, 'put'
        )
        
        # Put-call parity: C - P = S - K*e^(-rT)
        parity_left = call_price - put_price
        parity_right = self.S - self.K * np.exp(-self.r * self.T)
        
        self.assertAlmostEqual(parity_left, parity_right, places=6)


class TestMonteCarloEngine(unittest.TestCase):
    """Test Monte Carlo simulation engine"""
    
    def setUp(self):
        self.mc_engine = MonteCarloEngine(n_simulations=10000, n_steps=50, random_seed=42)
        self.S = 100.0
        self.K = 100.0
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_gbm_path_generation(self):
        """Test geometric Brownian motion path generation"""
        paths = self.mc_engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        
        # Check dimensions
        self.assertEqual(paths.shape[0], self.mc_engine.n_simulations)
        self.assertEqual(paths.shape[1], self.mc_engine.n_steps + 1)
        
        # Initial values should be S
        np.testing.assert_array_almost_equal(paths[:, 0], self.S, decimal=10)
        
        # Final values should be log-normally distributed
        final_prices = paths[:, -1]
        log_returns = np.log(final_prices / self.S)
        
        # Mean of log returns should approximate (r - 0.5*sigma^2)*T
        expected_mean = (self.r - 0.5 * self.sigma**2) * self.T
        actual_mean = np.mean(log_returns)
        self.assertAlmostEqual(actual_mean, expected_mean, places=1)
    
    def test_vanilla_option_pricing(self):
        """Test vanilla option pricing convergence to Black-Scholes"""
        paths = self.mc_engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        result = self.mc_engine.price_vanilla_option(paths, self.K, self.r, self.T, 'call')
        
        # Compare with Black-Scholes
        from api.app import black_scholes
        bs_price, _, _, _, _, _ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'call')
        
        # Monte Carlo should be within 2 standard errors of Black-Scholes
        mc_price = result['price']
        std_error = result['std_error']
        
        self.assertLess(abs(mc_price - bs_price), 2 * std_error)
    
    def test_heston_model_paths(self):
        """Test Heston stochastic volatility model"""
        kappa, theta, sigma_v, rho, v0 = 2.0, 0.04, 0.3, -0.5, 0.04
        
        paths, vol_paths = self.mc_engine.heston_model(
            self.S, self.T, self.r, v0, kappa, theta, sigma_v, rho
        )
        
        # Check dimensions
        self.assertEqual(paths.shape[0], self.mc_engine.n_simulations)
        self.assertEqual(vol_paths.shape[0], self.mc_engine.n_simulations)
        
        # Volatility should be positive
        self.assertTrue(np.all(vol_paths >= 0))
        
        # Initial conditions
        np.testing.assert_array_almost_equal(paths[:, 0], self.S, decimal=10)
        np.testing.assert_array_almost_equal(vol_paths[:, 0], v0, decimal=10)


class TestExoticOptions(unittest.TestCase):
    """Test exotic options pricing"""
    
    def setUp(self):
        self.mc_engine = MonteCarloEngine(n_simulations=10000, n_steps=100, random_seed=42)
        self.exotic_engine = ExoticOptions(self.mc_engine)
        self.S = 100.0
        self.K = 100.0
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_asian_option_pricing(self):
        """Test Asian option pricing"""
        paths = self.mc_engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        
        # Arithmetic Asian call
        result = self.exotic_engine.asian_option(paths, self.K, self.r, self.T, 'call', 'arithmetic')
        
        # Asian option should be cheaper than vanilla (reduced volatility)
        vanilla_result = self.mc_engine.price_vanilla_option(paths, self.K, self.r, self.T, 'call')
        
        self.assertLess(result['price'], vanilla_result['price'])
        self.assertGreater(result['price'], 0)
    
    def test_barrier_option_pricing(self):
        """Test barrier option pricing"""
        paths = self.mc_engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        
        # Up-and-out barrier call
        B = 120.0  # Barrier level
        result = self.exotic_engine.barrier_option(
            paths, self.K, B, self.r, self.T, 'call', 'up_and_out'
        )
        
        # Barrier option should be cheaper than vanilla
        vanilla_result = self.mc_engine.price_vanilla_option(paths, self.K, self.r, self.T, 'call')
        
        self.assertLess(result['price'], vanilla_result['price'])
        self.assertGreater(result['price'], 0)
        self.assertIn('barrier_hit_prob', result)
    
    def test_binary_option_pricing(self):
        """Test binary option pricing"""
        paths = self.mc_engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        
        payout = 10.0
        result = self.exotic_engine.binary_option(paths, self.K, self.r, self.T, payout, 'call')
        
        # Binary option price should be between 0 and discounted payout
        max_price = payout * np.exp(-self.r * self.T)
        
        self.assertGreater(result['price'], 0)
        self.assertLess(result['price'], max_price)
        self.assertIn('hit_probability', result)


class TestPortfolioManagement(unittest.TestCase):
    """Test portfolio management functionality"""
    
    def setUp(self):
        self.portfolio = OptionPortfolio()
        self.expiry = datetime.now() + timedelta(days=30)
    
    def test_add_position(self):
        """Test adding positions to portfolio"""
        initial_count = len(self.portfolio.positions)
        
        self.portfolio.add_position(
            'AAPL', 'call', 150.0, self.expiry, 10, 5.0, 155.0, 0.25, 0.05
        )
        
        self.assertEqual(len(self.portfolio.positions), initial_count + 1)
        self.assertGreater(self.portfolio.portfolio_value, 0)
    
    def test_portfolio_greeks(self):
        """Test portfolio Greeks calculation"""
        # Add multiple positions
        self.portfolio.add_position('AAPL', 'call', 150.0, self.expiry, 10, 5.0, 155.0, 0.25, 0.05)
        self.portfolio.add_position('AAPL', 'put', 140.0, self.expiry, -5, 3.0, 155.0, 0.25, 0.05)
        
        greeks = self.portfolio.portfolio_greeks
        
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('vega', greeks)
        self.assertIn('theta', greeks)
        
        # Greeks should be finite numbers
        for greek_name, greek_value in greeks.items():
            self.assertTrue(np.isfinite(greek_value))
    
    def test_hedge_recommendations(self):
        """Test delta hedging recommendations"""
        self.portfolio.add_position('AAPL', 'call', 150.0, self.expiry, 10, 5.0, 155.0, 0.25, 0.05)
        
        hedge_rec = self.portfolio.delta_hedge_recommendation(['AAPL'])
        
        self.assertIn('total_portfolio_delta', hedge_rec)
        self.assertIn('hedge_recommendations', hedge_rec)


class TestRiskMetrics(unittest.TestCase):
    """Test risk management calculations"""
    
    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_stress_testing(self):
        """Test stress testing functionality"""
        stress_results = RiskMetrics.stress_test(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        self.assertIn('base_price', stress_results)
        self.assertIn('market_crash', stress_results)
        self.assertIn('vol_spike', stress_results)
        
        # Stress scenarios should show different impacts
        base_price = stress_results['base_price']
        crash_impact = stress_results['market_crash']['pnl_impact']
        
        # Market crash should negatively impact call options
        self.assertLess(crash_impact, 0)
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)  # Daily returns
        
        var_results = RiskMetrics.calculate_var(returns, confidence_level=0.05)
        
        self.assertIn('VaR', var_results)
        self.assertIn('Expected_Shortfall', var_results)
        
        # VaR should be negative (loss)
        self.assertLess(var_results['VaR'], 0)
        
        # Expected Shortfall should be worse than VaR
        self.assertLess(var_results['Expected_Shortfall'], var_results['VaR'])


class TestModelValidation(unittest.TestCase):
    """Test model validation framework"""
    
    def setUp(self):
        self.S = 100.0
        self.K = 100.0
        self.T = 0.25
        self.r = 0.05
        self.sigma = 0.2
    
    def test_bs_vs_mc_validation(self):
        """Test Black-Scholes vs Monte Carlo validation"""
        validation = ModelValidation.validate_black_scholes_vs_mc(
            self.S, self.K, self.T, self.r, self.sigma, 'call', n_simulations=50000
        )
        
        self.assertIn('black_scholes_price', validation)
        self.assertIn('monte_carlo_price', validation)
        self.assertIn('relative_error', validation)
        self.assertIn('is_within_confidence', validation)
        
        # Relative error should be small for large number of simulations
        self.assertLess(validation['relative_error'], 0.05)  # Less than 5%
    
    def test_convergence_analysis(self):
        """Test Monte Carlo convergence analysis"""
        convergence = ModelValidation.convergence_analysis(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        self.assertIn('simulation_sizes', convergence)
        self.assertIn('prices', convergence)
        self.assertIn('std_errors', convergence)
        
        # Standard errors should decrease with more simulations
        std_errors = convergence['std_errors']
        self.assertGreater(std_errors[0], std_errors[-1])


class TestMarketDataIntegration(unittest.TestCase):
    """Test market data integration (when available)"""
    
    def setUp(self):
        self.market_data = MarketDataProvider()
    
    def test_volatility_estimation(self):
        """Test volatility estimation methods"""
        # Create sample price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 100))), index=dates)
        
        # Test different volatility methods
        simple_vol = VolatilityEstimator.historical_volatility(prices, window=30, method='simple')
        ewma_vol = VolatilityEstimator.historical_volatility(prices, window=30, method='ewma')
        
        # Volatilities should be positive and finite
        self.assertTrue(np.all(simple_vol.dropna() > 0))
        self.assertTrue(np.all(ewma_vol.dropna() > 0))
        self.assertTrue(np.all(np.isfinite(simple_vol.dropna())))
        self.assertTrue(np.all(np.isfinite(ewma_vol.dropna())))


class TestAPIEndpoints(unittest.TestCase):
    """Test Flask API endpoints"""
    
    def setUp(self):
        # Import app here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
        
        from app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_black_scholes_endpoint(self):
        """Test Black-Scholes API endpoint"""
        data = {
            'S': 100,
            'K': 100,
            'T': 0.25,
            'r': 0.05,
            'sigma': 0.2,
            'optionType': 'call'
        }
        
        response = self.app.post('/api/calculate_black_scholes', 
                               json=data, 
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('option_price', result)
        self.assertIn('delta', result)
        self.assertGreater(result['option_price'], 0)
    
    def test_monte_carlo_endpoint(self):
        """Test Monte Carlo API endpoint"""
        data = {
            'S': 100,
            'K': 100,
            'T': 0.25,
            'r': 0.05,
            'sigma': 0.2,
            'optionType': 'call',
            'model': 'gbm',
            'simulations': 10000
        }
        
        response = self.app.post('/api/monte_carlo',
                               json=data,
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        
        result = response.get_json()
        self.assertIn('option_price', result)
        self.assertIn('std_error', result)
        self.assertGreater(result['option_price'], 0)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_classes = [
        TestBlackScholesModel,
        TestMonteCarloEngine,
        TestExoticOptions,
        TestPortfolioManagement,
        TestRiskMetrics,
        TestModelValidation,
        TestMarketDataIntegration,
        TestAPIEndpoints
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    print(f"\nTests {'PASSED' if success else 'FAILED'}")
    exit(0 if success else 1)
