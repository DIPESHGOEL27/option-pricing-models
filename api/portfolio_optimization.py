"""
Advanced Portfolio Optimization and Strategy Module

Sophisticated portfolio optimization framework for option strategies including:
- Modern Portfolio Theory (MPT) optimization
- Risk parity and factor-based allocation
- Black-Litterman model implementation
- Options strategy optimization
- Dynamic hedging strategies
- Multi-objective optimization
- Alternative risk measures (CVaR, etc.)
- Real-time portfolio rebalancing

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy import stats
import cvxpy as cp
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of portfolio optimization."""
    MEAN_VARIANCE = "mean_variance"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_SHARPE = "maximum_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    CVAR_OPTIMIZATION = "cvar_optimization"
    MULTI_OBJECTIVE = "multi_objective"

class RiskMeasure(Enum):
    """Risk measures for optimization."""
    VARIANCE = "variance"
    CVAR = "cvar"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    TRACKING_ERROR = "tracking_error"

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints."""
    min_weights: Optional[np.ndarray] = None
    max_weights: Optional[np.ndarray] = None
    sum_weights: float = 1.0
    max_turnover: Optional[float] = None
    sector_limits: Optional[Dict[str, Tuple[float, float]]] = None
    risk_budget: Optional[float] = None
    
@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    information_ratio: Optional[float] = None

class AdvancedPortfolioOptimizer:
    """
    Advanced portfolio optimization engine with multiple methodologies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.risk_free_rate = risk_free_rate
        self.optimization_history = []
        
    def mean_variance_optimization(self, 
                                 expected_returns: np.ndarray,
                                 covariance_matrix: np.ndarray,
                                 target_return: Optional[float] = None,
                                 constraints: Optional[OptimizationConstraints] = None) -> Dict[str, Any]:
        """
        Perform mean-variance optimization (Markowitz).
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            target_return: Target portfolio return (if None, maximizes Sharpe ratio)
            constraints: Portfolio constraints
            
        Returns:
            Optimization results dictionary
        """
        n_assets = len(expected_returns)
        
        if constraints is None:
            constraints = OptimizationConstraints()
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Objective function
        portfolio_variance = cp.quad_form(weights, covariance_matrix)
        portfolio_return = expected_returns.T @ weights
        
        if target_return is not None:
            # Minimize variance subject to target return
            objective = cp.Minimize(portfolio_variance)
            constraints_list = [portfolio_return == target_return]
        else:
            # Maximize Sharpe ratio (equivalent to maximizing excess return / volatility)
            # Using the transformation from Markowitz
            risk_free_return = self.risk_free_rate
            excess_returns = expected_returns - risk_free_return
            
            # Maximize (excess_return) / sqrt(variance)
            # Equivalent to minimizing variance - 2*lambda*excess_return
            # We'll use a grid search over lambda to find the maximum Sharpe ratio
            objective = cp.Minimize(portfolio_variance - 2 * 1.0 * (excess_returns.T @ weights))
        
        # Basic constraints
        constraints_list = [cp.sum(weights) == constraints.sum_weights]
        
        # Weight bounds
        if constraints.min_weights is not None:
            constraints_list.append(weights >= constraints.min_weights)
        else:
            constraints_list.append(weights >= 0)  # Long-only by default
            
        if constraints.max_weights is not None:
            constraints_list.append(weights <= constraints.max_weights)
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints_list)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Optimization failed with status: {problem.status}")
            return {'error': f'Optimization failed: {problem.status}'}
        
        optimal_weights = weights.value
        
        # Calculate portfolio metrics
        portfolio_return_opt = expected_returns.T @ optimal_weights
        portfolio_variance_opt = optimal_weights.T @ covariance_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance_opt)
        sharpe_ratio = (portfolio_return_opt - self.risk_free_rate) / portfolio_volatility
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': portfolio_return_opt,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'optimization_status': problem.status,
            'optimization_type': OptimizationType.MEAN_VARIANCE.value
        }
    
    def risk_parity_optimization(self, 
                               covariance_matrix: np.ndarray,
                               risk_budgets: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform risk parity optimization.
        
        Args:
            covariance_matrix: Covariance matrix of asset returns
            risk_budgets: Target risk contribution for each asset (defaults to equal)
            
        Returns:
            Risk parity optimization results
        """
        n_assets = covariance_matrix.shape[0]
        
        if risk_budgets is None:
            risk_budgets = np.ones(n_assets) / n_assets
        
        # Initial guess (equal weights)
        x0 = np.ones(n_assets) / n_assets
        
        def risk_contribution(weights, cov_matrix):
            """Calculate risk contribution of each asset."""
            portfolio_variance = weights.T @ cov_matrix @ weights
            marginal_risk = cov_matrix @ weights
            risk_contrib = weights * marginal_risk / portfolio_variance
            return risk_contrib
        
        def objective_function(weights):
            """Objective function to minimize (sum of squared deviations from target)."""
            risk_contrib = risk_contribution(weights, covariance_matrix)
            return np.sum((risk_contrib - risk_budgets) ** 2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # Weights sum to 1
        ]
        
        # Bounds (long-only)
        bounds = [(0.001, 1.0) for _ in range(n_assets)]  # Small minimum to avoid numerical issues
        
        # Solve optimization
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'disp': False}
        )
        
        if not result.success:
            logger.warning(f"Risk parity optimization failed: {result.message}")
            return {'error': f'Optimization failed: {result.message}'}
        
        optimal_weights = result.x
        
        # Calculate final risk contributions
        final_risk_contrib = risk_contribution(optimal_weights, covariance_matrix)
        portfolio_variance = optimal_weights.T @ covariance_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        return {
            'optimal_weights': optimal_weights,
            'risk_contributions': final_risk_contrib,
            'target_risk_budgets': risk_budgets,
            'volatility': portfolio_volatility,
            'optimization_status': 'success' if result.success else 'failed',
            'optimization_type': OptimizationType.RISK_PARITY.value
        }
    
    def black_litterman_optimization(self,
                                   prior_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   market_cap_weights: np.ndarray,
                                   views_matrix: np.ndarray,
                                   view_returns: np.ndarray,
                                   view_uncertainty: np.ndarray,
                                   risk_aversion: float = 3.0) -> Dict[str, Any]:
        """
        Implement Black-Litterman model for portfolio optimization.
        
        Args:
            prior_returns: Prior expected returns (usually from CAPM)
            covariance_matrix: Historical covariance matrix
            market_cap_weights: Market capitalization weights
            views_matrix: Matrix encoding investor views (P matrix)
            view_returns: Expected returns from views (Q vector)
            view_uncertainty: Uncertainty matrix for views (Omega matrix)
            risk_aversion: Risk aversion parameter
            
        Returns:
            Black-Litterman optimization results
        """
        n_assets = len(prior_returns)
        
        # Calculate implied returns (reverse optimization)
        implied_returns = risk_aversion * covariance_matrix @ market_cap_weights
        
        # Calculate tau (scaling factor)
        tau = 1 / len(prior_returns)  # Common choice
        
        # Black-Litterman formula
        # New expected returns
        M1 = np.linalg.inv(tau * covariance_matrix)
        M2 = views_matrix.T @ np.linalg.inv(view_uncertainty) @ views_matrix
        M3 = np.linalg.inv(tau * covariance_matrix) @ implied_returns
        M4 = views_matrix.T @ np.linalg.inv(view_uncertainty) @ view_returns
        
        bl_returns = np.linalg.inv(M1 + M2) @ (M3 + M4)
        
        # New covariance matrix
        bl_covariance = np.linalg.inv(M1 + M2)
        
        # Optimize with Black-Litterman inputs
        bl_result = self.mean_variance_optimization(bl_returns, bl_covariance)
        
        if 'error' in bl_result:
            return bl_result
        
        bl_result.update({
            'black_litterman_returns': bl_returns,
            'black_litterman_covariance': bl_covariance,
            'implied_returns': implied_returns,
            'views_matrix': views_matrix,
            'view_returns': view_returns,
            'optimization_type': OptimizationType.BLACK_LITTERMAN.value
        })
        
        return bl_result
    
    def cvar_optimization(self,
                         returns_scenarios: np.ndarray,
                         confidence_level: float = 0.95,
                         target_return: Optional[float] = None) -> Dict[str, Any]:
        """
        Optimize portfolio using Conditional Value at Risk (CVaR).
        
        Args:
            returns_scenarios: Matrix of return scenarios (scenarios x assets)
            confidence_level: Confidence level for CVaR
            target_return: Target portfolio return
            
        Returns:
            CVaR optimization results
        """
        n_scenarios, n_assets = returns_scenarios.shape
        alpha = 1 - confidence_level
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        var = cp.Variable()  # Value at Risk
        u = cp.Variable(n_scenarios)  # Auxiliary variables for CVaR
        
        # Portfolio returns for each scenario
        portfolio_returns = returns_scenarios @ weights
        
        # CVaR constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0,  # Long-only
            u >= 0,  # Auxiliary variables non-negative
            u >= -(portfolio_returns - var)  # CVaR constraint
        ]
        
        # Target return constraint (if specified)
        if target_return is not None:
            expected_portfolio_return = cp.sum(portfolio_returns) / n_scenarios
            constraints.append(expected_portfolio_return >= target_return)
        
        # Objective: minimize CVaR
        cvar = var + (1 / (alpha * n_scenarios)) * cp.sum(u)
        objective = cp.Minimize(cvar)
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"CVaR optimization failed with status: {problem.status}")
            return {'error': f'Optimization failed: {problem.status}'}
        
        optimal_weights = weights.value
        optimal_var = var.value
        optimal_cvar = cvar.value
        
        # Calculate portfolio metrics
        portfolio_returns_opt = returns_scenarios @ optimal_weights
        expected_return = np.mean(portfolio_returns_opt)
        volatility = np.std(portfolio_returns_opt)
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': -optimal_var,
            'cvar_95': -optimal_cvar,
            'optimization_status': problem.status,
            'optimization_type': OptimizationType.CVAR_OPTIMIZATION.value
        }
    
    def multi_objective_optimization(self,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   objectives: List[str],
                                   weights_objectives: List[float]) -> Dict[str, Any]:
        """
        Multi-objective portfolio optimization.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix
            objectives: List of objectives ('return', 'risk', 'skewness', 'kurtosis')
            weights_objectives: Weights for each objective
            
        Returns:
            Multi-objective optimization results
        """
        n_assets = len(expected_returns)
        
        # Define optimization variables
        weights = cp.Variable(n_assets)
        
        # Initialize objective function
        objective_terms = []
        
        for obj, weight in zip(objectives, weights_objectives):
            if obj == 'return':
                # Maximize return (minimize negative return)
                obj_term = -weight * (expected_returns.T @ weights)
            elif obj == 'risk':
                # Minimize risk (variance)
                obj_term = weight * cp.quad_form(weights, covariance_matrix)
            elif obj == 'concentration':
                # Minimize concentration (maximize diversification)
                obj_term = weight * cp.sum(cp.square(weights))
            else:
                logger.warning(f"Unknown objective: {obj}")
                continue
            
            objective_terms.append(obj_term)
        
        # Combined objective
        if objective_terms:
            objective = cp.Minimize(cp.sum(objective_terms))
        else:
            return {'error': 'No valid objectives specified'}
        
        # Constraints
        constraints = [
            cp.sum(weights) == 1,  # Weights sum to 1
            weights >= 0  # Long-only
        ]
        
        # Solve problem
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        if problem.status != cp.OPTIMAL:
            logger.warning(f"Multi-objective optimization failed: {problem.status}")
            return {'error': f'Optimization failed: {problem.status}'}
        
        optimal_weights = weights.value
        
        # Calculate portfolio metrics
        portfolio_return = expected_returns.T @ optimal_weights
        portfolio_variance = optimal_weights.T @ covariance_matrix @ optimal_weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            'optimal_weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'objectives': objectives,
            'objective_weights': weights_objectives,
            'optimization_status': problem.status,
            'optimization_type': OptimizationType.MULTI_OBJECTIVE.value
        }

class OptionsStrategyOptimizer:
    """
    Specialized optimizer for options strategies.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize options strategy optimizer.
        
        Args:
            risk_free_rate: Risk-free rate
        """
        self.risk_free_rate = risk_free_rate
    
    def optimize_covered_call_strategy(self,
                                     stock_price: float,
                                     stock_volatility: float,
                                     strike_prices: np.ndarray,
                                     time_to_expiry: float,
                                     target_return: float = 0.12) -> Dict[str, Any]:
        """
        Optimize covered call strategy parameters.
        
        Args:
            stock_price: Current stock price
            stock_volatility: Stock volatility
            strike_prices: Array of available strike prices
            time_to_expiry: Time to option expiry
            target_return: Target annual return
            
        Returns:
            Optimal covered call strategy
        """
        from .option_pricing import black_scholes, AdvancedOptionPricer
        
        pricer = AdvancedOptionPricer()
        results = []
        
        for strike in strike_prices:
            try:
                # Calculate call option price
                call_price = black_scholes(
                    stock_price, strike, time_to_expiry, 
                    self.risk_free_rate, stock_volatility, 'call'
                )
                
                # Calculate Greeks
                greeks = pricer.calculate_greeks(
                    stock_price, strike, time_to_expiry,
                    self.risk_free_rate, stock_volatility, 'call'
                )
                
                # Strategy metrics
                initial_investment = stock_price - call_price  # Net cost
                max_profit = strike - initial_investment
                max_profit_return = max_profit / initial_investment
                
                # Breakeven point
                breakeven = stock_price - call_price
                
                # Probability of profit (approximation)
                # Stock price needs to be above breakeven at expiry
                d2 = (np.log(stock_price / breakeven) + 
                     (self.risk_free_rate - 0.5 * stock_volatility**2) * time_to_expiry) / \
                     (stock_volatility * np.sqrt(time_to_expiry))
                prob_profit = stats.norm.cdf(d2)
                
                # Risk-adjusted return
                risk_adjusted_return = max_profit_return * prob_profit
                
                results.append({
                    'strike': strike,
                    'call_price': call_price,
                    'initial_investment': initial_investment,
                    'max_profit': max_profit,
                    'max_profit_return': max_profit_return,
                    'breakeven': breakeven,
                    'prob_profit': prob_profit,
                    'risk_adjusted_return': risk_adjusted_return,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega']
                })
                
            except Exception as e:
                logger.warning(f"Error calculating strategy for strike {strike}: {e}")
                continue
        
        if not results:
            return {'error': 'No valid strategies found'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Find optimal strategy based on risk-adjusted return
        optimal_idx = df['risk_adjusted_return'].idxmax()
        optimal_strategy = df.iloc[optimal_idx].to_dict()
        
        # Alternative selections
        highest_return_idx = df['max_profit_return'].idxmax()
        highest_prob_idx = df['prob_profit'].idxmax()
        
        return {
            'optimal_strategy': optimal_strategy,
            'highest_return_strategy': df.iloc[highest_return_idx].to_dict(),
            'highest_probability_strategy': df.iloc[highest_prob_idx].to_dict(),
            'all_strategies': results,
            'recommendation': self._generate_strategy_recommendation(optimal_strategy, target_return)
        }
    
    def optimize_protective_put_strategy(self,
                                       stock_price: float,
                                       stock_volatility: float,
                                       strike_prices: np.ndarray,
                                       time_to_expiry: float,
                                       max_loss_tolerance: float = 0.10) -> Dict[str, Any]:
        """
        Optimize protective put strategy.
        
        Args:
            stock_price: Current stock price
            stock_volatility: Stock volatility
            strike_prices: Array of available strike prices
            time_to_expiry: Time to option expiry
            max_loss_tolerance: Maximum acceptable loss (as fraction)
            
        Returns:
            Optimal protective put strategy
        """
        from .option_pricing import black_scholes, AdvancedOptionPricer
        
        pricer = AdvancedOptionPricer()
        results = []
        
        for strike in strike_prices:
            if strike >= stock_price:  # Only consider puts below current price
                continue
                
            try:
                # Calculate put option price
                put_price = black_scholes(
                    stock_price, strike, time_to_expiry,
                    self.risk_free_rate, stock_volatility, 'put'
                )
                
                # Calculate Greeks
                greeks = pricer.calculate_greeks(
                    stock_price, strike, time_to_expiry,
                    self.risk_free_rate, stock_volatility, 'put'
                )
                
                # Strategy metrics
                total_cost = stock_price + put_price
                max_loss = total_cost - strike
                max_loss_pct = max_loss / stock_price
                
                # Insurance cost as percentage
                insurance_cost_pct = put_price / stock_price
                
                # Breakeven point
                breakeven = stock_price + put_price
                
                # Cost efficiency (protection per dollar spent)
                protection_efficiency = (stock_price - strike) / put_price
                
                results.append({
                    'strike': strike,
                    'put_price': put_price,
                    'total_cost': total_cost,
                    'max_loss': max_loss,
                    'max_loss_pct': max_loss_pct,
                    'insurance_cost_pct': insurance_cost_pct,
                    'breakeven': breakeven,
                    'protection_efficiency': protection_efficiency,
                    'delta': greeks['delta'],
                    'gamma': greeks['gamma'],
                    'theta': greeks['theta'],
                    'vega': greeks['vega']
                })
                
            except Exception as e:
                logger.warning(f"Error calculating protective put for strike {strike}: {e}")
                continue
        
        if not results:
            return {'error': 'No valid protective put strategies found'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Filter by loss tolerance
        acceptable_strategies = df[df['max_loss_pct'] <= max_loss_tolerance]
        
        if acceptable_strategies.empty:
            # If no strategies meet tolerance, find the one with minimum loss
            optimal_idx = df['max_loss_pct'].idxmin()
            optimal_strategy = df.iloc[optimal_idx].to_dict()
            warning = "No strategies meet loss tolerance; showing minimum loss option"
        else:
            # Among acceptable strategies, choose the most cost-efficient
            optimal_idx = acceptable_strategies['protection_efficiency'].idxmax()
            optimal_strategy = acceptable_strategies.iloc[optimal_idx].to_dict()
            warning = None
        
        # Alternative selections
        cheapest_idx = df['insurance_cost_pct'].idxmin()
        most_protection_idx = df['protection_efficiency'].idxmax()
        
        result = {
            'optimal_strategy': optimal_strategy,
            'cheapest_strategy': df.iloc[cheapest_idx].to_dict(),
            'most_protection_strategy': df.iloc[most_protection_idx].to_dict(),
            'all_strategies': results,
            'recommendation': self._generate_protection_recommendation(optimal_strategy, max_loss_tolerance)
        }
        
        if warning:
            result['warning'] = warning
            
        return result
    
    def _generate_strategy_recommendation(self, strategy: Dict, target_return: float) -> str:
        """Generate recommendation text for covered call strategy."""
        recommendations = []
        
        if strategy['max_profit_return'] >= target_return:
            recommendations.append(f"Strategy meets target return of {target_return:.1%}")
        else:
            recommendations.append(f"Strategy falls short of target return ({strategy['max_profit_return']:.1%} vs {target_return:.1%})")
        
        if strategy['prob_profit'] > 0.7:
            recommendations.append(f"High probability of profit ({strategy['prob_profit']:.1%})")
        elif strategy['prob_profit'] > 0.5:
            recommendations.append(f"Moderate probability of profit ({strategy['prob_profit']:.1%})")
        else:
            recommendations.append(f"Low probability of profit ({strategy['prob_profit']:.1%})")
        
        if abs(strategy['delta']) < 0.3:
            recommendations.append("Low directional risk (delta)")
        elif abs(strategy['delta']) > 0.7:
            recommendations.append("High directional risk (delta)")
        
        if strategy['theta'] > 0.01:
            recommendations.append("Benefits from time decay")
        
        return " | ".join(recommendations)
    
    def _generate_protection_recommendation(self, strategy: Dict, max_loss_tolerance: float) -> str:
        """Generate recommendation text for protective put strategy."""
        recommendations = []
        
        if strategy['max_loss_pct'] <= max_loss_tolerance:
            recommendations.append(f"Meets loss tolerance ({strategy['max_loss_pct']:.1%} ≤ {max_loss_tolerance:.1%})")
        else:
            recommendations.append(f"Exceeds loss tolerance ({strategy['max_loss_pct']:.1%} > {max_loss_tolerance:.1%})")
        
        if strategy['insurance_cost_pct'] < 0.02:
            recommendations.append("Low cost insurance (<2%)")
        elif strategy['insurance_cost_pct'] < 0.05:
            recommendations.append("Moderate cost insurance (2-5%)")
        else:
            recommendations.append("High cost insurance (>5%)")
        
        if strategy['protection_efficiency'] > 5:
            recommendations.append("Highly efficient protection")
        elif strategy['protection_efficiency'] > 3:
            recommendations.append("Moderately efficient protection")
        else:
            recommendations.append("Low efficiency protection")
        
        return " | ".join(recommendations)

class DynamicHedgingEngine:
    """
    Dynamic hedging engine for options portfolios.
    """
    
    def __init__(self, rebalance_threshold: float = 0.1):
        """
        Initialize dynamic hedging engine.
        
        Args:
            rebalance_threshold: Delta threshold for rebalancing
        """
        self.rebalance_threshold = rebalance_threshold
        self.hedging_history = []
    
    def calculate_hedge_ratio(self, portfolio_delta: float, 
                            hedge_instrument_delta: float = 1.0) -> float:
        """
        Calculate optimal hedge ratio.
        
        Args:
            portfolio_delta: Portfolio delta exposure
            hedge_instrument_delta: Delta of hedge instrument (1.0 for stock)
            
        Returns:
            Optimal hedge ratio
        """
        return -portfolio_delta / hedge_instrument_delta
    
    def delta_hedging_strategy(self, 
                             portfolio_positions: List[Dict],
                             market_data: Dict,
                             current_hedge_position: float = 0.0) -> Dict[str, Any]:
        """
        Calculate delta hedging requirements.
        
        Args:
            portfolio_positions: List of portfolio positions
            market_data: Current market data
            current_hedge_position: Current hedge position
            
        Returns:
            Hedging recommendations
        """
        from .option_pricing import AdvancedOptionPricer
        
        pricer = AdvancedOptionPricer()
        
        # Calculate total portfolio delta
        total_delta = 0.0
        position_deltas = []
        
        for position in portfolio_positions:
            if position.get('type') == 'option':
                try:
                    S = market_data.get('spot_price', 100)
                    K = position.get('strike', 100)
                    T = position.get('time_to_expiry', 0.25)
                    r = market_data.get('risk_free_rate', 0.02)
                    sigma = market_data.get('volatility', 0.2)
                    option_type = position.get('option_type', 'call')
                    quantity = position.get('quantity', 0)
                    
                    greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
                    position_delta = greeks['delta'] * quantity
                    total_delta += position_delta
                    
                    position_deltas.append({
                        'instrument': position.get('instrument', 'Unknown'),
                        'delta': position_delta,
                        'quantity': quantity
                    })
                    
                except Exception as e:
                    logger.warning(f"Error calculating delta for position: {e}")
            
            elif position.get('type') == 'stock':
                # Stock has delta of 1
                quantity = position.get('quantity', 0)
                total_delta += quantity
                
                position_deltas.append({
                    'instrument': position.get('instrument', 'Stock'),
                    'delta': quantity,
                    'quantity': quantity
                })
        
        # Calculate required hedge
        required_hedge = self.calculate_hedge_ratio(total_delta)
        hedge_adjustment = required_hedge - current_hedge_position
        
        # Check if rebalancing is needed
        needs_rebalancing = abs(hedge_adjustment) > self.rebalance_threshold
        
        # Calculate hedging costs
        spot_price = market_data.get('spot_price', 100)
        transaction_cost_rate = 0.001  # 0.1% transaction cost
        hedging_cost = abs(hedge_adjustment) * spot_price * transaction_cost_rate
        
        hedge_recommendation = {
            'total_portfolio_delta': total_delta,
            'current_hedge_position': current_hedge_position,
            'required_hedge_position': required_hedge,
            'hedge_adjustment': hedge_adjustment,
            'needs_rebalancing': needs_rebalancing,
            'rebalance_threshold': self.rebalance_threshold,
            'estimated_hedging_cost': hedging_cost,
            'position_deltas': position_deltas,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in history
        self.hedging_history.append(hedge_recommendation)
        
        return hedge_recommendation
    
    def gamma_hedging_strategy(self,
                             portfolio_positions: List[Dict],
                             market_data: Dict,
                             hedge_options: List[Dict]) -> Dict[str, Any]:
        """
        Calculate gamma hedging strategy using options.
        
        Args:
            portfolio_positions: Portfolio positions
            market_data: Market data
            hedge_options: Available options for hedging
            
        Returns:
            Gamma hedging recommendations
        """
        from .option_pricing import AdvancedOptionPricer
        
        pricer = AdvancedOptionPricer()
        
        # Calculate portfolio gamma and delta
        total_delta = 0.0
        total_gamma = 0.0
        
        for position in portfolio_positions:
            if position.get('type') == 'option':
                try:
                    S = market_data.get('spot_price', 100)
                    K = position.get('strike', 100)
                    T = position.get('time_to_expiry', 0.25)
                    r = market_data.get('risk_free_rate', 0.02)
                    sigma = market_data.get('volatility', 0.2)
                    option_type = position.get('option_type', 'call')
                    quantity = position.get('quantity', 0)
                    
                    greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
                    total_delta += greeks['delta'] * quantity
                    total_gamma += greeks['gamma'] * quantity
                    
                except Exception as e:
                    logger.warning(f"Error calculating Greeks for position: {e}")
        
        # Find best hedge option (highest gamma per dollar)
        best_hedge = None
        best_efficiency = 0
        
        for hedge_option in hedge_options:
            try:
                S = market_data.get('spot_price', 100)
                K = hedge_option.get('strike', 100)
                T = hedge_option.get('time_to_expiry', 0.25)
                r = market_data.get('risk_free_rate', 0.02)
                sigma = market_data.get('volatility', 0.2)
                option_type = hedge_option.get('option_type', 'call')
                
                from .option_pricing import black_scholes
                option_price = black_scholes(S, K, T, r, sigma, option_type)
                greeks = pricer.calculate_greeks(S, K, T, r, sigma, option_type)
                
                # Efficiency: gamma per dollar
                efficiency = abs(greeks['gamma']) / option_price
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_hedge = {
                        'option': hedge_option,
                        'price': option_price,
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'efficiency': efficiency
                    }
                    
            except Exception as e:
                logger.warning(f"Error evaluating hedge option: {e}")
        
        if best_hedge is None:
            return {'error': 'No suitable hedge options found'}
        
        # Calculate hedge quantities
        # Solve system: hedge_delta * n_hedge = -total_delta (delta neutral)
        #               hedge_gamma * n_hedge = -total_gamma (gamma neutral)
        
        hedge_gamma = best_hedge['gamma']
        hedge_delta = best_hedge['delta']
        
        if abs(hedge_gamma) < 1e-6:
            return {'error': 'Hedge option has insufficient gamma'}
        
        # Number of options needed for gamma neutrality
        n_hedge_gamma = -total_gamma / hedge_gamma
        
        # Remaining delta after gamma hedge
        remaining_delta = total_delta + n_hedge_gamma * hedge_delta
        
        # Stock position needed for delta neutrality
        stock_hedge = -remaining_delta
        
        hedge_cost = (abs(n_hedge_gamma) * best_hedge['price'] + 
                     abs(stock_hedge) * market_data.get('spot_price', 100)) * 0.001
        
        return {
            'portfolio_delta': total_delta,
            'portfolio_gamma': total_gamma,
            'best_hedge_option': best_hedge,
            'hedge_option_quantity': n_hedge_gamma,
            'stock_hedge_quantity': stock_hedge,
            'estimated_hedge_cost': hedge_cost,
            'hedging_efficiency': best_efficiency,
            'timestamp': datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced Portfolio Optimization Framework...")
    
    # Create sample data
    np.random.seed(42)
    n_assets = 5
    
    # Generate sample expected returns and covariance matrix
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    correlation_matrix = np.random.uniform(0.3, 0.8, (n_assets, n_assets))
    np.fill_diagonal(correlation_matrix, 1.0)
    volatilities = np.random.uniform(0.15, 0.30, n_assets)
    covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
    
    print(f"Sample portfolio: {n_assets} assets")
    print(f"Expected returns: {expected_returns}")
    print(f"Volatilities: {volatilities}")
    
    # Initialize optimizer
    optimizer = AdvancedPortfolioOptimizer(risk_free_rate=0.02)
    
    # Test mean-variance optimization
    print("\n1. Testing Mean-Variance Optimization...")
    mv_result = optimizer.mean_variance_optimization(expected_returns, covariance_matrix)
    
    if 'error' not in mv_result:
        print("Mean-Variance Optimization Results:")
        print(f"  Expected Return: {mv_result['expected_return']:.2%}")
        print(f"  Volatility: {mv_result['volatility']:.2%}")
        print(f"  Sharpe Ratio: {mv_result['sharpe_ratio']:.3f}")
        print(f"  Optimal Weights: {mv_result['optimal_weights']}")
    
    # Test risk parity optimization
    print("\n2. Testing Risk Parity Optimization...")
    rp_result = optimizer.risk_parity_optimization(covariance_matrix)
    
    if 'error' not in rp_result:
        print("Risk Parity Optimization Results:")
        print(f"  Volatility: {rp_result['volatility']:.2%}")
        print(f"  Optimal Weights: {rp_result['optimal_weights']}")
        print(f"  Risk Contributions: {rp_result['risk_contributions']}")
    
    # Test CVaR optimization
    print("\n3. Testing CVaR Optimization...")
    
    # Generate return scenarios
    n_scenarios = 1000
    return_scenarios = np.random.multivariate_normal(
        expected_returns / 252, covariance_matrix / 252, n_scenarios
    )
    
    cvar_result = optimizer.cvar_optimization(return_scenarios)
    
    if 'error' not in cvar_result:
        print("CVaR Optimization Results:")
        print(f"  Expected Return: {cvar_result['expected_return']:.2%}")
        print(f"  Volatility: {cvar_result['volatility']:.2%}")
        print(f"  VaR (95%): {cvar_result['var_95']:.2%}")
        print(f"  CVaR (95%): {cvar_result['cvar_95']:.2%}")
        print(f"  Optimal Weights: {cvar_result['optimal_weights']}")
    
    # Test options strategy optimization
    print("\n4. Testing Options Strategy Optimization...")
    
    options_optimizer = OptionsStrategyOptimizer()
    
    # Test covered call optimization
    stock_price = 100
    stock_vol = 0.25
    strikes = np.arange(100, 120, 2.5)
    time_to_exp = 30/365
    
    cc_result = options_optimizer.optimize_covered_call_strategy(
        stock_price, stock_vol, strikes, time_to_exp, target_return=0.12
    )
    
    if 'error' not in cc_result:
        optimal = cc_result['optimal_strategy']
        print("Optimal Covered Call Strategy:")
        print(f"  Strike: ${optimal['strike']:.2f}")
        print(f"  Call Price: ${optimal['call_price']:.2f}")
        print(f"  Max Return: {optimal['max_profit_return']:.2%}")
        print(f"  Probability of Profit: {optimal['prob_profit']:.1%}")
        print(f"  Recommendation: {cc_result['recommendation']}")
    
    # Test protective put optimization
    put_strikes = np.arange(85, 100, 2.5)
    pp_result = options_optimizer.optimize_protective_put_strategy(
        stock_price, stock_vol, put_strikes, time_to_exp, max_loss_tolerance=0.10
    )
    
    if 'error' not in pp_result:
        optimal = pp_result['optimal_strategy']
        print("\nOptimal Protective Put Strategy:")
        print(f"  Strike: ${optimal['strike']:.2f}")
        print(f"  Put Price: ${optimal['put_price']:.2f}")
        print(f"  Max Loss: {optimal['max_loss_pct']:.1%}")
        print(f"  Insurance Cost: {optimal['insurance_cost_pct']:.1%}")
        print(f"  Recommendation: {pp_result['recommendation']}")
    
    print("\nPortfolio optimization framework testing completed!")
