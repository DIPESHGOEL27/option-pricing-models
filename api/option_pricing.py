"""
Advanced Option Pricing Module

Industry-grade option pricing library with sophisticated models and analytics.
Features include Black-Scholes, Heston, Jump-Diffusion, Implied Volatility Calibration,
Machine Learning models, and comprehensive risk analytics.

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm
from scipy.optimize import minimize, brentq
from scipy.special import erfc
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings('ignore')

class AdvancedOptionPricer:
    """
    Industry-grade option pricing engine with multiple models and advanced analytics.
    """
    
    def __init__(self, cache_size: int = 128, parallel: bool = True):
        """
        Initialize the advanced option pricer.
        
        Args:
            cache_size: Size of LRU cache for price computations
            parallel: Enable parallel processing for portfolio calculations
        """
        self.cache_size = cache_size
        self.parallel = parallel
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup caching for frequently called functions."""
        self.black_scholes = lru_cache(maxsize=self.cache_size)(self._black_scholes_core)
        self.calculate_greeks = lru_cache(maxsize=self.cache_size)(self._calculate_greeks_core)
    
    def _black_scholes_core(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> float:
        """Core Black-Scholes pricing function (cached)."""
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type.lower() == 'call':
            option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type.lower() == 'put':
            option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'")
        
        return option_price
    
    def _calculate_greeks_core(self, S: float, K: float, T: float, r: float, 
                              sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """Core Greeks calculation function (cached)."""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1
        
        # Gamma
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        # Vega (per 1% change in volatility)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        # Theta (per day)
        if option_type.lower() == 'call':
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        # Rho (per 1% change in interest rate)
        if option_type.lower() == 'call':
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

class ImpliedVolatilityCalculator:
    """
    Advanced implied volatility calculation with multiple methods and market conventions.
    """
    
    @staticmethod
    def newton_raphson_iv(market_price: float, S: float, K: float, T: float, 
                         r: float, option_type: str = 'call', 
                         initial_guess: float = 0.2, max_iterations: int = 100,
                         tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        This is more efficient than Brent's method for well-behaved functions.
        """
        sigma = initial_guess
        pricer = AdvancedOptionPricer()
        
        for i in range(max_iterations):
            # Calculate price and vega
            price = pricer.black_scholes(S, K, T, r, sigma, option_type)
            greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
            vega = greeks['vega'] * 100  # Convert back to per unit change
            
            # Newton-Raphson update
            price_diff = price - market_price
            
            if abs(price_diff) < tolerance:
                return sigma
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            sigma = sigma - price_diff / vega
            
            # Ensure sigma stays positive
            sigma = max(sigma, 0.001)
            
        return sigma
    
    @staticmethod
    def brent_method_iv(market_price: float, S: float, K: float, T: float, 
                       r: float, option_type: str = 'call',
                       vol_min: float = 0.001, vol_max: float = 5.0) -> float:
        """
        Calculate implied volatility using Brent's method.
        
        More robust but slower than Newton-Raphson.
        """
        pricer = AdvancedOptionPricer()
        
        def objective(sigma):
            return pricer.black_scholes(S, K, T, r, sigma, option_type) - market_price
        
        try:
            return brentq(objective, vol_min, vol_max, xtol=1e-6)
        except ValueError:
            # If Brent's method fails, fall back to Newton-Raphson
            return ImpliedVolatilityCalculator.newton_raphson_iv(
                market_price, S, K, T, r, option_type)

class AdvancedGreeksCalculator:
    """
    Advanced Greeks calculation including higher-order Greeks and cross-derivatives.
    """
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks including higher-order ones.
        """
        pricer = AdvancedOptionPricer()
        
        # First-order Greeks
        greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
        
        # Calculate higher-order Greeks using finite differences
        ds = S * 0.01  # 1% of spot price
        dt = 1/365    # 1 day
        dsigma = 0.01 # 1% volatility
        dr = 0.0001   # 1 basis point
        
        # Speed (third derivative w.r.t. S)
        gamma_up = pricer.calculate_greeks(S + ds, K, T, r, sigma, option_type)['gamma']
        gamma_down = pricer.calculate_greeks(S - ds, K, T, r, sigma, option_type)['gamma']
        speed = (gamma_up - gamma_down) / (2 * ds)
        
        # Color (second derivative of gamma w.r.t. time)
        if T > dt:
            gamma_t_down = pricer.calculate_greeks(S, K, T - dt, r, sigma, option_type)['gamma']
            color = (greeks['gamma'] - gamma_t_down) / dt
        else:
            color = 0
        
        # Vanna (cross-derivative delta/vega)
        delta_vol_up = pricer.calculate_greeks(S, K, T, r, sigma + dsigma, option_type)['delta']
        delta_vol_down = pricer.calculate_greeks(S, K, T, r, sigma - dsigma, option_type)['delta']
        vanna = (delta_vol_up - delta_vol_down) / (2 * dsigma)
        
        # Volga (second derivative w.r.t. volatility)
        vega_vol_up = pricer.calculate_greeks(S, K, T, r, sigma + dsigma, option_type)['vega']
        vega_vol_down = pricer.calculate_greeks(S, K, T, r, sigma - dsigma, option_type)['vega']
        volga = (vega_vol_up - vega_vol_down) / (2 * dsigma)
        
        # Add higher-order Greeks
        greeks.update({
            'speed': speed,
            'color': color,
            'vanna': vanna,
            'volga': volga
        })
        
        return greeks

class ModelCalibration:
    """
    Advanced model calibration for various option pricing models.
    """
    
    @staticmethod
    def calibrate_heston_model(option_data: pd.DataFrame, 
                              S0: float, r: float) -> Dict[str, float]:
        """
        Calibrate Heston model parameters to market option prices.
        
        Args:
            option_data: DataFrame with columns ['strike', 'expiry', 'option_type', 'market_price']
            S0: Current underlying price
            r: Risk-free rate
        
        Returns:
            Dictionary with calibrated Heston parameters
        """
        def heston_price_fft(S0, K, T, r, kappa, theta, sigma_v, rho, v0):
            """Heston price using FFT (simplified implementation)."""
            # This is a simplified version - real implementation would use full FFT
            # For now, we'll use a Monte Carlo approximation
            from .advanced_models import MonteCarloEngine, HestonCalibration
            
            mc_engine = MonteCarloEngine(n_simulations=50000, n_steps=100)
            paths, _ = mc_engine.heston_model(S0, T, r, v0, kappa, theta, sigma_v, rho)
            result = mc_engine.price_vanilla_option(paths, K, r, T, 'call')
            return result['price']
        
        def objective_function(params):
            kappa, theta, sigma_v, rho, v0 = params
            
            # Parameter constraints
            if kappa <= 0 or theta <= 0 or sigma_v <= 0 or v0 <= 0:
                return 1e6
            if abs(rho) >= 1:
                return 1e6
            if 2 * kappa * theta <= sigma_v**2:  # Feller condition
                return 1e6
            
            total_error = 0
            for _, row in option_data.iterrows():
                try:
                    model_price = heston_price_fft(
                        S0, row['strike'], row['expiry'], r,
                        kappa, theta, sigma_v, rho, v0
                    )
                    market_price = row['market_price']
                    total_error += ((model_price - market_price) / market_price) ** 2
                except:
                    total_error += 1e6
            
            return total_error
        
        # Initial parameter guess
        initial_params = [2.0, 0.04, 0.3, -0.5, 0.04]
        
        # Parameter bounds
        bounds = [
            (0.1, 10.0),    # kappa
            (0.01, 1.0),    # theta
            (0.01, 2.0),    # sigma_v
            (-0.99, 0.99),  # rho
            (0.01, 1.0)     # v0
        ]
        
        result = minimize(objective_function, initial_params, bounds=bounds, 
                         method='L-BFGS-B')
        
        if result.success:
            kappa, theta, sigma_v, rho, v0 = result.x
            return {
                'kappa': kappa,
                'theta': theta, 
                'sigma_v': sigma_v,
                'rho': rho,
                'v0': v0,
                'rmse': np.sqrt(result.fun / len(option_data))
            }
        else:
            raise ValueError("Calibration failed to converge")

# Backwards compatibility
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """Wrapper for backwards compatibility."""
    pricer = AdvancedOptionPricer()
    return pricer.black_scholes(S, K, T, r, sigma, option_type)

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Wrapper for backwards compatibility."""
    pricer = AdvancedOptionPricer()
    return pricer.calculate_greeks(S, K, T, r, sigma, option_type)
