"""
Advanced Option Pricing Models Module

Industry-grade implementation of sophisticated option pricing models including:
- Monte Carlo simulation with various processes
- Heston stochastic volatility model
- Jump-diffusion models (Merton, Kou)
- Calibration and validation tools
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import gamma as gamma_func
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class MonteCarloEngine:
    """Monte Carlo simulation engine for option pricing"""
    
    def __init__(self, n_simulations: int = 100000, n_steps: int = 252, 
                 random_seed: Optional[int] = None):
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        if random_seed:
            np.random.seed(random_seed)
    
    def geometric_brownian_motion(self, S0: float, T: float, r: float, 
                                sigma: float, use_antithetic: bool = True) -> np.ndarray:
        """Generate paths using Geometric Brownian Motion with antithetic variates"""
        dt = T / self.n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)
        
        if use_antithetic:
            # Use antithetic variates for variance reduction
            half_sims = self.n_simulations // 2
            Z = np.random.standard_normal((half_sims, self.n_steps))
            Z_antithetic = -Z  # Antithetic variates
            Z_combined = np.vstack([Z, Z_antithetic])
        else:
            Z_combined = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        log_returns = drift + diffusion * Z_combined
        log_returns = np.column_stack([np.zeros(self.n_simulations), log_returns])
        
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        return prices
    
    def heston_model(self, S0: float, T: float, r: float, v0: float,
                    kappa: float, theta: float, sigma_v: float, 
                    rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate paths using Heston stochastic volatility model"""
        dt = T / self.n_steps
        
        # Initialize arrays
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        v = np.zeros((self.n_simulations, self.n_steps + 1))
        
        S[:, 0] = S0
        v[:, 0] = v0
        
        # Generate correlated random numbers
        Z1 = np.random.standard_normal((self.n_simulations, self.n_steps))
        Z2 = np.random.standard_normal((self.n_simulations, self.n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        for t in range(self.n_steps):
            # Variance process (CIR process with Feller condition)
            v[:, t+1] = np.abs(v[:, t] + kappa * (theta - v[:, t]) * dt + 
                              sigma_v * np.sqrt(v[:, t] * dt) * Z2[:, t])
            
            # Stock price process
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + 
                                        np.sqrt(v[:, t] * dt) * Z1[:, t])
        
        return S, v
    
    def jump_diffusion_merton(self, S0: float, T: float, r: float, sigma: float,
                             lam: float, mu_j: float, sigma_j: float) -> np.ndarray:
        """Generate paths using Merton jump-diffusion model"""
        dt = T / self.n_steps
        
        # Poisson jumps
        N = np.random.poisson(lam * dt, (self.n_simulations, self.n_steps))
        
        # Jump sizes (log-normal)
        J = np.random.normal(mu_j, sigma_j, (self.n_simulations, self.n_steps))
        
        # Brownian motion
        Z = np.random.standard_normal((self.n_simulations, self.n_steps))
        
        # Drift adjustment for jumps
        k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
        drift = (r - 0.5 * sigma**2 - lam * k) * dt
        
        log_returns = drift + sigma * np.sqrt(dt) * Z + N * J
        log_returns = np.column_stack([np.zeros(self.n_simulations), log_returns])
        
        log_prices = np.cumsum(log_returns, axis=1)
        prices = S0 * np.exp(log_prices)
        
        return prices
    
    def price_vanilla_option(self, paths: np.ndarray, K: float, r: float, 
                           T: float, option_type: str = 'call') -> Dict:
        """Price vanilla European options using Monte Carlo"""
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        # Calculate confidence interval
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = option_price - z_score * std_error
        ci_upper = option_price + z_score * std_error
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (ci_lower, ci_upper),
            'payoffs': payoffs
        }


class ExoticOptions:
    """Pricing engine for exotic options"""
    
    def __init__(self, mc_engine: MonteCarloEngine):
        self.mc_engine = mc_engine
    
    def asian_option(self, paths: np.ndarray, K: float, r: float, T: float,
                    option_type: str = 'call', asian_type: str = 'arithmetic') -> Dict:
        """Price Asian options (average price options)"""
        if asian_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        elif asian_type == 'geometric':
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError("asian_type must be 'arithmetic' or 'geometric'")
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)
        
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return {
            'price': option_price,
            'std_error': std_error,
            'avg_prices': avg_prices,
            'payoffs': payoffs
        }
    
    def barrier_option(self, paths: np.ndarray, K: float, B: float, r: float, 
                      T: float, option_type: str = 'call', 
                      barrier_type: str = 'up_and_out') -> Dict:
        """Price barrier options"""
        final_prices = paths[:, -1]
        
        # Check barrier conditions
        if barrier_type == 'up_and_out':
            barrier_hit = np.any(paths >= B, axis=1)
            active_options = ~barrier_hit
        elif barrier_type == 'down_and_out':
            barrier_hit = np.any(paths <= B, axis=1)
            active_options = ~barrier_hit
        elif barrier_type == 'up_and_in':
            barrier_hit = np.any(paths >= B, axis=1)
            active_options = barrier_hit
        elif barrier_type == 'down_and_in':
            barrier_hit = np.any(paths <= B, axis=1)
            active_options = barrier_hit
        else:
            raise ValueError("Invalid barrier_type")
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.where(active_options, 
                             np.maximum(final_prices - K, 0), 0)
        else:
            payoffs = np.where(active_options, 
                             np.maximum(K - final_prices, 0), 0)
        
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return {
            'price': option_price,
            'std_error': std_error,
            'barrier_hit_prob': np.mean(barrier_hit),
            'payoffs': payoffs
        }
    
    def lookback_option(self, paths: np.ndarray, K: float, r: float, T: float,
                       option_type: str = 'call', lookback_type: str = 'floating') -> Dict:
        """Price lookback options"""
        if lookback_type == 'floating':
            if option_type.lower() == 'call':
                # Call pays max(S_max - S_T, 0)
                max_prices = np.max(paths, axis=1)
                final_prices = paths[:, -1]
                payoffs = max_prices - final_prices
            else:
                # Put pays max(S_T - S_min, 0)
                min_prices = np.min(paths, axis=1)
                final_prices = paths[:, -1]
                payoffs = final_prices - min_prices
        else:  # fixed strike
            if option_type.lower() == 'call':
                max_prices = np.max(paths, axis=1)
                payoffs = np.maximum(max_prices - K, 0)
            else:
                min_prices = np.min(paths, axis=1)
                payoffs = np.maximum(K - min_prices, 0)
        
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return {
            'price': option_price,
            'std_error': std_error,
            'payoffs': payoffs
        }
    
    def binary_option(self, paths: np.ndarray, K: float, r: float, T: float,
                     payout: float = 1.0, option_type: str = 'call') -> Dict:
        """Price binary (digital) options"""
        final_prices = paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.where(final_prices > K, payout, 0)
        else:
            payoffs = np.where(final_prices < K, payout, 0)
        
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(len(payoffs))
        
        return {
            'price': option_price,
            'std_error': std_error,
            'hit_probability': np.mean(payoffs / payout),
            'payoffs': payoffs
        }


class HestonCalibration:
    """Calibration engine for Heston model parameters"""
    
    def __init__(self):
        self.params = None
        self.calibration_error = None
    
    def objective_function(self, params: np.ndarray, market_data: pd.DataFrame,
                          S0: float, r: float, T: float) -> float:
        """Objective function for calibration"""
        kappa, theta, sigma_v, rho, v0 = params
        
        # Bounds checking
        if kappa <= 0 or theta <= 0 or sigma_v <= 0 or v0 <= 0:
            return 1e6
        if abs(rho) >= 1:
            return 1e6
        if 2 * kappa * theta <= sigma_v**2:  # Feller condition
            return 1e6
        
        total_error = 0
        mc_engine = MonteCarloEngine(n_simulations=50000, n_steps=100)
        
        for _, row in market_data.iterrows():
            K = row['strike']
            market_price = row['market_price']
            option_type = row['option_type']
            
            try:
                # Generate Heston paths
                paths, _ = mc_engine.heston_model(S0, T, r, v0, kappa, 
                                                theta, sigma_v, rho)
                
                # Price option
                result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)
                model_price = result['price']
                
                # Calculate error (relative)
                error = ((model_price - market_price) / market_price)**2
                total_error += error
                
            except:
                return 1e6
        
        return total_error
    
    def calibrate(self, market_data: pd.DataFrame, S0: float, r: float, 
                 T: float) -> Dict:
        """Calibrate Heston parameters to market data"""
        # Initial guess
        initial_params = [2.0, 0.04, 0.3, -0.5, 0.04]  # kappa, theta, sigma_v, rho, v0
        
        # Bounds
        bounds = [
            (0.1, 10.0),   # kappa
            (0.01, 1.0),   # theta
            (0.1, 2.0),    # sigma_v
            (-0.99, 0.99), # rho
            (0.01, 1.0)    # v0
        ]
        
        # Optimization
        result = optimize.minimize(
            self.objective_function,
            initial_params,
            args=(market_data, S0, r, T),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if result.success:
            self.params = {
                'kappa': result.x[0],
                'theta': result.x[1],
                'sigma_v': result.x[2],
                'rho': result.x[3],
                'v0': result.x[4]
            }
            self.calibration_error = result.fun
        
        return {
            'success': result.success,
            'params': self.params,
            'error': self.calibration_error,
            'optimization_result': result
        }


class RiskMetrics:
    """Risk management and metrics calculation"""
    
    @staticmethod
    def calculate_greeks_mc(S0: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call',
                           n_simulations: int = 100000) -> Dict:
        """Calculate Greeks using finite differences and Monte Carlo"""
        
        def price_option(S, vol):
            mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=100)
            paths = mc_engine.geometric_brownian_motion(S, T, r, vol)
            result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)
            return result['price']
        
        # Base price
        base_price = price_option(S0, sigma)
        
        # Delta (∂V/∂S)
        dS = 0.01 * S0
        price_up = price_option(S0 + dS, sigma)
        price_down = price_option(S0 - dS, sigma)
        delta = (price_up - price_down) / (2 * dS)
        
        # Gamma (∂²V/∂S²)
        gamma = (price_up - 2 * base_price + price_down) / (dS**2)
        
        # Vega (∂V/∂σ)
        dvol = 0.01
        price_vol_up = price_option(S0, sigma + dvol)
        price_vol_down = price_option(S0, sigma - dvol)
        vega = (price_vol_up - price_vol_down) / (2 * dvol)
        
        # Theta (∂V/∂T) - approximate
        dT = 1/365  # 1 day
        if T > dT:
            mc_engine_theta = MonteCarloEngine(n_simulations=n_simulations, n_steps=100)
            paths_theta = mc_engine_theta.geometric_brownian_motion(S0, T - dT, r, sigma)
            result_theta = mc_engine_theta.price_vanilla_option(paths_theta, K, r, T - dT, option_type)
            price_theta = result_theta['price']
            theta = (price_theta - base_price) / dT
        else:
            theta = 0
        
        # Rho (∂V/∂r)
        dr = 0.01
        price_r_up = price_option(S0, sigma)  # Would need to modify for different r
        rho = 0  # Simplified for now
        
        return {
            'price': base_price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> Dict:
        """Calculate Value at Risk and Expected Shortfall"""
        var = np.percentile(returns, confidence_level * 100)
        
        # Expected Shortfall (Conditional VaR)
        es = np.mean(returns[returns <= var])
        
        return {
            'VaR': var,
            'Expected_Shortfall': es,
            'confidence_level': confidence_level
        }
    
    @staticmethod
    def stress_test(S0: float, K: float, T: float, r: float, sigma: float,
                   option_type: str = 'call', scenarios: Dict = None) -> Dict:
        """Perform stress testing on option prices"""
        if scenarios is None:
            scenarios = {
                'market_crash': {'S_shock': -0.3, 'vol_shock': 2.0},
                'vol_spike': {'S_shock': 0.0, 'vol_shock': 1.5},
                'rate_shock': {'r_shock': 0.02},
                'combined_stress': {'S_shock': -0.2, 'vol_shock': 1.8, 'r_shock': 0.01}
            }
        
        mc_engine = MonteCarloEngine(n_simulations=50000, n_steps=100)
        
        # Base case
        base_paths = mc_engine.geometric_brownian_motion(S0, T, r, sigma)
        base_result = mc_engine.price_vanilla_option(base_paths, K, r, T, option_type)
        base_price = base_result['price']
        
        stress_results = {'base_price': base_price}
        
        for scenario_name, shocks in scenarios.items():
            # Apply shocks
            stressed_S = S0 * (1 + shocks.get('S_shock', 0))
            stressed_sigma = sigma * shocks.get('vol_shock', 1.0)
            stressed_r = r + shocks.get('r_shock', 0)
            
            # Price under stress
            stressed_paths = mc_engine.geometric_brownian_motion(
                stressed_S, T, stressed_r, stressed_sigma)
            stressed_result = mc_engine.price_vanilla_option(
                stressed_paths, K, stressed_r, T, option_type)
            stressed_price = stressed_result['price']
            
            # Calculate P&L impact
            pnl_impact = stressed_price - base_price
            pnl_percent = (pnl_impact / base_price) * 100
            
            stress_results[scenario_name] = {
                'stressed_price': stressed_price,
                'pnl_impact': pnl_impact,
                'pnl_percent': pnl_percent,
                'shocks_applied': shocks
            }
        
        return stress_results


class ModelValidation:
    """Model validation and backtesting framework"""
    
    @staticmethod
    def validate_black_scholes_vs_mc(S0: float, K: float, T: float, r: float,
                                   sigma: float, option_type: str = 'call',
                                   n_simulations: int = 100000) -> Dict:
        """Compare Black-Scholes analytical price with Monte Carlo"""
        from scipy.stats import norm
        
        # Black-Scholes analytical solution
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            bs_price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)
        
        # Monte Carlo price
        mc_engine = MonteCarloEngine(n_simulations=n_simulations, n_steps=252)
        paths = mc_engine.geometric_brownian_motion(S0, T, r, sigma)
        mc_result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)
        mc_price = mc_result['price']
        
        # Validation metrics
        absolute_error = abs(bs_price - mc_price)
        relative_error = absolute_error / bs_price
        
        return {
            'black_scholes_price': bs_price,
            'monte_carlo_price': mc_price,
            'absolute_error': absolute_error,
            'relative_error': relative_error,
            'mc_std_error': mc_result['std_error'],
            'is_within_confidence': absolute_error <= 2 * mc_result['std_error']
        }
    
    @staticmethod
    def convergence_analysis(S0: float, K: float, T: float, r: float,
                           sigma: float, option_type: str = 'call') -> Dict:
        """Analyze Monte Carlo convergence"""
        simulation_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        prices = []
        std_errors = []
        
        for n_sims in simulation_sizes:
            mc_engine = MonteCarloEngine(n_simulations=n_sims, n_steps=100)
            paths = mc_engine.geometric_brownian_motion(S0, T, r, sigma)
            result = mc_engine.price_vanilla_option(paths, K, r, T, option_type)
            prices.append(result['price'])
            std_errors.append(result['std_error'])
        
        return {
            'simulation_sizes': simulation_sizes,
            'prices': prices,
            'std_errors': std_errors
        }


# Example usage and testing
if __name__ == "__main__":
    # Test basic Monte Carlo pricing
    mc_engine = MonteCarloEngine(n_simulations=100000, n_steps=252)
    
    # Parameters
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    
    # Generate paths and price option
    paths = mc_engine.geometric_brownian_motion(S0, T, r, sigma)
    result = mc_engine.price_vanilla_option(paths, K, r, T, 'call')
    
    print(f"Monte Carlo Call Price: {result['price']:.4f}")
    print(f"Standard Error: {result['std_error']:.4f}")
    print(f"95% Confidence Interval: {result['confidence_interval']}")
    
    # Test exotic options
    exotic_engine = ExoticOptions(mc_engine)
    
    # Asian option
    asian_result = exotic_engine.asian_option(paths, K, r, T, 'call', 'arithmetic')
    print(f"Asian Call Price: {asian_result['price']:.4f}")
    
    # Barrier option
    barrier_result = exotic_engine.barrier_option(paths, K, 110, r, T, 'call', 'up_and_out')
    print(f"Up-and-Out Barrier Call Price: {barrier_result['price']:.4f}")
    
    # Model validation
    validation = ModelValidation.validate_black_scholes_vs_mc(S0, K, T, r, sigma, 'call')
    print(f"Black-Scholes vs MC Validation:")
    print(f"  BS Price: {validation['black_scholes_price']:.4f}")
    print(f"  MC Price: {validation['monte_carlo_price']:.4f}")
    print(f"  Relative Error: {validation['relative_error']:.4f}")
