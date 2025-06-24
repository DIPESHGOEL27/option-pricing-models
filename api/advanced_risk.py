"""
Advanced Risk Management Module

Comprehensive risk management framework for option portfolios including:
- Value at Risk (VaR) and Expected Shortfall (ES)
- Stress testing and scenario analysis
- Monte Carlo risk simulations
- Dynamic hedging recommendations
- Real-time risk monitoring
- Regulatory capital calculations

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class RiskMeasureType(Enum):
    """Enumeration of available risk measures."""
    VAR_HISTORICAL = "var_historical"
    VAR_PARAMETRIC = "var_parametric" 
    VAR_MONTE_CARLO = "var_monte_carlo"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "maximum_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    SHARPE_RATIO = "sharpe_ratio"

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var_95: float
    var_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    volatility: float
    maximum_drawdown: float
    sharpe_ratio: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for JSON serialization."""
        return {
            'var_95': self.var_95,
            'var_99': self.var_99,
            'expected_shortfall_95': self.expected_shortfall_95,
            'expected_shortfall_99': self.expected_shortfall_99,
            'volatility': self.volatility,
            'maximum_drawdown': self.maximum_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'beta': self.beta,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio
        }

@dataclass
class StressTestScenario:
    """Stress test scenario definition."""
    name: str
    description: str
    equity_shock: float  # Percentage change in equity price
    volatility_shock: float  # Percentage change in volatility
    rate_shock: float  # Absolute change in interest rates
    correlation_shock: float  # Change in correlation
    probability: Optional[float] = None

class AdvancedRiskManager:
    """
    Advanced risk management system for option portfolios.
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99],
                 risk_free_rate: float = 0.02):
        """
        Initialize advanced risk manager.
        
        Args:
            confidence_levels: List of confidence levels for VaR calculations
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
        """
        self.confidence_levels = confidence_levels
        self.risk_free_rate = risk_free_rate
        self.portfolio_history = []
        self.stress_scenarios = self._create_default_stress_scenarios()
    
    def _create_default_stress_scenarios(self) -> List[StressTestScenario]:
        """Create default stress test scenarios."""
        scenarios = [
            StressTestScenario(
                name="Market Crash",
                description="Severe equity market decline with volatility spike",
                equity_shock=-0.30,
                volatility_shock=1.50,
                rate_shock=-0.02,
                correlation_shock=0.20,
                probability=0.05
            ),
            StressTestScenario(
                name="Volatility Spike",
                description="Sudden increase in market volatility",
                equity_shock=-0.10,
                volatility_shock=1.00,
                rate_shock=0.00,
                correlation_shock=0.15,
                probability=0.10
            ),
            StressTestScenario(
                name="Interest Rate Shock",
                description="Sudden increase in interest rates",
                equity_shock=-0.05,
                volatility_shock=0.20,
                rate_shock=0.02,
                correlation_shock=0.05,
                probability=0.15
            ),
            StressTestScenario(
                name="Flight to Quality",
                description="Risk-off market environment",
                equity_shock=-0.15,
                volatility_shock=0.50,
                rate_shock=-0.01,
                correlation_shock=0.30,
                probability=0.20
            ),
            StressTestScenario(
                name="Black Swan",
                description="Extreme tail event",
                equity_shock=-0.50,
                volatility_shock=2.00,
                rate_shock=-0.03,
                correlation_shock=0.50,
                probability=0.01
            )
        ]
        return scenarios
    
    def calculate_var_historical(self, returns: np.ndarray, 
                               confidence_level: float = 0.95) -> float:
        """
        Calculate Value at Risk using historical simulation method.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns) == 0:
            return 0.0
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR as the percentile
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(sorted_returns, percentile)
        
        return max(var, 0.0)  # VaR should be positive (representing loss)
    
    def calculate_var_parametric(self, returns: np.ndarray,
                                confidence_level: float = 0.95,
                                distribution: str = 'normal') -> float:
        """
        Calculate parametric VaR assuming a specific distribution.
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            distribution: Distribution assumption ('normal', 't', 'skewed_t')
            
        Returns:
            Parametric VaR value
        """
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if distribution == 'normal':
            # Normal distribution VaR
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mean_return + z_score * std_return)
        
        elif distribution == 't':
            # Student's t-distribution VaR
            df = len(returns) - 1
            t_score = stats.t.ppf(1 - confidence_level, df)
            var = -(mean_return + t_score * std_return)
        
        elif distribution == 'skewed_t':
            # Skewed t-distribution (simplified)
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            # Cornish-Fisher expansion for skewness and kurtosis adjustment
            z = stats.norm.ppf(1 - confidence_level)
            z_adjusted = (z + 
                         (z**2 - 1) * skewness / 6 +
                         (z**3 - 3*z) * kurtosis / 24 -
                         (2*z**3 - 5*z) * skewness**2 / 36)
            
            var = -(mean_return + z_adjusted * std_return)
        
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return max(var, 0.0)
    
    def calculate_var_monte_carlo(self, portfolio_value: float,
                                portfolio_weights: np.ndarray,
                                expected_returns: np.ndarray,
                                covariance_matrix: np.ndarray,
                                confidence_level: float = 0.95,
                                n_simulations: int = 10000,
                                time_horizon: int = 1) -> float:
        """
        Calculate VaR using Monte Carlo simulation.
        
        Args:
            portfolio_value: Current portfolio value
            portfolio_weights: Weight of each asset in portfolio
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            confidence_level: Confidence level
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
            
        Returns:
            Monte Carlo VaR value
        """
        # Generate correlated random returns
        random_returns = np.random.multivariate_normal(
            expected_returns * time_horizon,
            covariance_matrix * time_horizon,
            n_simulations
        )
        
        # Calculate portfolio returns for each simulation
        portfolio_returns = np.dot(random_returns, portfolio_weights)
        
        # Calculate portfolio values
        portfolio_values = portfolio_value * (1 + portfolio_returns)
        
        # Calculate P&L
        pnl = portfolio_values - portfolio_value
        
        # Calculate VaR
        percentile = (1 - confidence_level) * 100
        var = -np.percentile(pnl, percentile)
        
        return max(var, 0.0)
    
    def calculate_expected_shortfall(self, returns: np.ndarray,
                                   confidence_level: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Array of historical returns
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        if len(returns) == 0:
            return 0.0
        
        # Calculate VaR first
        var = self.calculate_var_historical(returns, confidence_level)
        
        # Find returns worse than VaR
        var_threshold = -var  # Convert back to negative return
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var
        
        # Expected Shortfall is the mean of tail losses
        expected_shortfall = -np.mean(tail_returns)
        
        return max(expected_shortfall, 0.0)
    
    def calculate_maximum_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown from portfolio values.
        
        Args:
            portfolio_values: Array of portfolio values over time
            
        Returns:
            Maximum drawdown as a percentage
        """
        if len(portfolio_values) == 0:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdown at each point
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Return maximum drawdown (positive number)
        max_dd = -np.min(drawdowns)
        
        return max(max_dd, 0.0)
    
    def calculate_portfolio_risk_metrics(self, returns: np.ndarray,
                                       portfolio_values: Optional[np.ndarray] = None,
                                       benchmark_returns: Optional[np.ndarray] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.
        
        Args:
            returns: Portfolio returns
            portfolio_values: Portfolio values over time
            benchmark_returns: Benchmark returns for relative metrics
            
        Returns:
            RiskMetrics object with calculated metrics
        """
        if len(returns) == 0:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # VaR calculations
        var_95 = self.calculate_var_historical(returns, 0.95)
        var_99 = self.calculate_var_historical(returns, 0.99)
        
        # Expected Shortfall
        es_95 = self.calculate_expected_shortfall(returns, 0.95)
        es_99 = self.calculate_expected_shortfall(returns, 0.99)
        
        # Volatility (annualized)
        volatility = np.std(returns, ddof=1) * np.sqrt(252)
        
        # Maximum Drawdown
        if portfolio_values is not None:
            max_dd = self.calculate_maximum_drawdown(portfolio_values)
        else:
            # Approximate from cumulative returns
            cumulative_returns = np.cumprod(1 + returns)
            max_dd = self.calculate_maximum_drawdown(cumulative_returns)
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / 252
        if np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Beta and relative metrics
        beta = None
        tracking_error = None
        information_ratio = None
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # Beta calculation
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns, ddof=1)
            if benchmark_variance > 0:
                beta = covariance / benchmark_variance
            
            # Tracking Error
            active_returns = returns - benchmark_returns
            tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)
            
            # Information Ratio
            if tracking_error > 0:
                information_ratio = np.mean(active_returns) / np.std(active_returns, ddof=1) * np.sqrt(252)
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall_95=es_95,
            expected_shortfall_99=es_99,
            volatility=volatility,
            maximum_drawdown=max_dd,
            sharpe_ratio=sharpe_ratio,
            beta=beta,
            tracking_error=tracking_error,
            information_ratio=information_ratio
        )
    
    def stress_test_portfolio(self, portfolio_positions: List[Dict],
                            current_market_data: Dict,
                            scenarios: Optional[List[StressTestScenario]] = None) -> Dict[str, Dict]:
        """
        Perform stress testing on portfolio under various scenarios.
        
        Args:
            portfolio_positions: List of position dictionaries
            current_market_data: Current market data
            scenarios: List of stress test scenarios (uses default if None)
            
        Returns:
            Dictionary of stress test results by scenario
        """
        if scenarios is None:
            scenarios = self.stress_scenarios
        
        results = {}
        
        for scenario in scenarios:
            scenario_results = self._apply_stress_scenario(
                portfolio_positions, current_market_data, scenario
            )
            results[scenario.name] = scenario_results
        
        return results
    
    def _apply_stress_scenario(self, portfolio_positions: List[Dict],
                              current_market_data: Dict,
                              scenario: StressTestScenario) -> Dict:
        """
        Apply a specific stress scenario to the portfolio.
        
        Args:
            portfolio_positions: Portfolio positions
            current_market_data: Current market data
            scenario: Stress test scenario
            
        Returns:
            Scenario results dictionary
        """
        # Create stressed market data
        stressed_data = current_market_data.copy()
        
        # Apply equity shock
        if 'spot_price' in stressed_data:
            stressed_data['spot_price'] *= (1 + scenario.equity_shock)
        
        # Apply volatility shock
        if 'volatility' in stressed_data:
            stressed_data['volatility'] *= (1 + scenario.volatility_shock)
        
        # Apply rate shock
        if 'risk_free_rate' in stressed_data:
            stressed_data['risk_free_rate'] += scenario.rate_shock
        
        # Calculate portfolio P&L under stress
        total_pnl = 0.0
        position_pnls = []
        
        for position in portfolio_positions:
            try:
                # Calculate position P&L under stress
                current_value = self._calculate_position_value(position, current_market_data)
                stressed_value = self._calculate_position_value(position, stressed_data)
                position_pnl = stressed_value - current_value
                
                total_pnl += position_pnl
                position_pnls.append({
                    'instrument': position.get('instrument', 'Unknown'),
                    'current_value': current_value,
                    'stressed_value': stressed_value,
                    'pnl': position_pnl,
                    'pnl_percent': (position_pnl / current_value * 100) if current_value != 0 else 0
                })
                
            except Exception as e:
                print(f"Error calculating stress P&L for position: {e}")
                position_pnls.append({
                    'instrument': position.get('instrument', 'Unknown'),
                    'error': str(e)
                })
        
        return {
            'scenario': scenario.name,
            'description': scenario.description,
            'total_pnl': total_pnl,
            'total_pnl_percent': (total_pnl / sum(p.get('current_value', 0) for p in position_pnls) * 100) 
                                if sum(p.get('current_value', 0) for p in position_pnls) != 0 else 0,
            'position_details': position_pnls,
            'stressed_market_data': stressed_data
        }
    
    def _calculate_position_value(self, position: Dict, market_data: Dict) -> float:
        """
        Calculate the value of a position given market data.
        
        Args:
            position: Position dictionary
            market_data: Market data dictionary
            
        Returns:
            Position value
        """
        # This is a simplified implementation
        # In practice, this would use sophisticated pricing models
        
        instrument_type = position.get('type', 'option')
        quantity = position.get('quantity', 0)
        
        if instrument_type == 'option':
            # Use Black-Scholes pricing
            from .option_pricing import black_scholes
            
            S = market_data.get('spot_price', 100)
            K = position.get('strike', 100)
            T = position.get('time_to_expiry', 0.25)
            r = market_data.get('risk_free_rate', 0.02)
            sigma = market_data.get('volatility', 0.2)
            option_type = position.get('option_type', 'call')
            
            try:
                option_price = black_scholes(S, K, T, r, sigma, option_type)
                return quantity * option_price
            except:
                return 0.0
        
        elif instrument_type == 'stock':
            # Simple stock valuation
            price = market_data.get('spot_price', 100)
            return quantity * price
        
        else:
            # Unknown instrument type
            return 0.0
    
    def calculate_dynamic_hedge_ratio(self, portfolio_positions: List[Dict],
                                    market_data: Dict,
                                    hedge_instrument: str = 'delta_hedge') -> Dict:
        """
        Calculate dynamic hedge ratios for portfolio risk management.
        
        Args:
            portfolio_positions: Portfolio positions
            market_data: Current market data
            hedge_instrument: Type of hedge instrument
            
        Returns:
            Hedge ratio recommendations
        """
        from .option_pricing import AdvancedOptionPricer
        
        pricer = AdvancedOptionPricer()
        
        # Calculate portfolio Greeks
        total_delta = 0.0
        total_gamma = 0.0
        total_vega = 0.0
        total_theta = 0.0
        
        position_greeks = []
        
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
                    position_gamma = greeks['gamma'] * quantity
                    position_vega = greeks['vega'] * quantity
                    position_theta = greeks['theta'] * quantity
                    
                    total_delta += position_delta
                    total_gamma += position_gamma
                    total_vega += position_vega
                    total_theta += position_theta
                    
                    position_greeks.append({
                        'instrument': position.get('instrument', 'Unknown'),
                        'delta': position_delta,
                        'gamma': position_gamma,
                        'vega': position_vega,
                        'theta': position_theta
                    })
                    
                except Exception as e:
                    print(f"Error calculating Greeks for position: {e}")
        
        # Calculate hedge recommendations
        hedge_recommendations = {}
        
        if hedge_instrument == 'delta_hedge':
            # Simple delta hedge with underlying
            hedge_recommendations['underlying_shares'] = -total_delta
            hedge_recommendations['hedge_cost'] = abs(total_delta) * market_data.get('spot_price', 100)
        
        elif hedge_instrument == 'delta_gamma_hedge':
            # Delta-gamma neutral hedge (requires options)
            # This is a simplified implementation
            hedge_recommendations['underlying_shares'] = -total_delta
            hedge_recommendations['hedge_options'] = -total_gamma / 0.1  # Simplified
            hedge_recommendations['residual_gamma'] = total_gamma
        
        return {
            'portfolio_greeks': {
                'total_delta': total_delta,
                'total_gamma': total_gamma,
                'total_vega': total_vega,
                'total_theta': total_theta
            },
            'position_greeks': position_greeks,
            'hedge_recommendations': hedge_recommendations,
            'hedge_effectiveness': self._calculate_hedge_effectiveness(total_delta, total_gamma)
        }
    
    def _calculate_hedge_effectiveness(self, delta: float, gamma: float) -> float:
        """
        Calculate hedge effectiveness score.
        
        Args:
            delta: Portfolio delta
            gamma: Portfolio gamma
            
        Returns:
            Hedge effectiveness score (0-1)
        """
        # Simplified hedge effectiveness calculation
        # In practice, this would be more sophisticated
        
        delta_risk = abs(delta)
        gamma_risk = abs(gamma) * 0.01  # Assume 1% move
        
        total_risk = delta_risk + gamma_risk
        
        if total_risk == 0:
            return 1.0
        
        # Higher risk = lower effectiveness
        effectiveness = max(0, 1 - total_risk / 100)
        
        return min(effectiveness, 1.0)
    
    def generate_risk_report(self, portfolio_positions: List[Dict],
                           market_data: Dict,
                           historical_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Generate comprehensive risk report for portfolio.
        
        Args:
            portfolio_positions: Portfolio positions
            market_data: Current market data
            historical_returns: Historical portfolio returns
            
        Returns:
            Comprehensive risk report dictionary
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_summary': {
                'total_positions': len(portfolio_positions),
                'total_value': sum(self._calculate_position_value(pos, market_data) 
                                 for pos in portfolio_positions)
            }
        }
        
        # Risk metrics
        if historical_returns is not None and len(historical_returns) > 0:
            risk_metrics = self.calculate_portfolio_risk_metrics(historical_returns)
            report['risk_metrics'] = risk_metrics.to_dict()
        
        # Stress testing
        stress_results = self.stress_test_portfolio(portfolio_positions, market_data)
        report['stress_test_results'] = stress_results
        
        # Hedge analysis
        hedge_analysis = self.calculate_dynamic_hedge_ratio(portfolio_positions, market_data)
        report['hedge_analysis'] = hedge_analysis
        
        # Risk warnings
        report['risk_warnings'] = self._generate_risk_warnings(
            report.get('risk_metrics', {}),
            stress_results,
            hedge_analysis
        )
        
        return report
    
    def _generate_risk_warnings(self, risk_metrics: Dict,
                               stress_results: Dict,
                               hedge_analysis: Dict) -> List[str]:
        """
        Generate risk warnings based on analysis results.
        
        Args:
            risk_metrics: Risk metrics dictionary
            stress_results: Stress test results
            hedge_analysis: Hedge analysis results
            
        Returns:
            List of risk warning messages
        """
        warnings = []
        
        # VaR warnings
        var_95 = risk_metrics.get('var_95', 0)
        if var_95 > 0.05:  # 5% of portfolio
            warnings.append(f"High VaR (95%): {var_95:.2%} of portfolio value")
        
        # Volatility warnings
        volatility = risk_metrics.get('volatility', 0)
        if volatility > 0.30:  # 30% annual volatility
            warnings.append(f"High portfolio volatility: {volatility:.1%} annualized")
        
        # Maximum drawdown warnings
        max_dd = risk_metrics.get('maximum_drawdown', 0)
        if max_dd > 0.20:  # 20% drawdown
            warnings.append(f"High maximum drawdown: {max_dd:.1%}")
        
        # Stress test warnings
        for scenario_name, results in stress_results.items():
            pnl_percent = results.get('total_pnl_percent', 0)
            if pnl_percent < -20:  # 20% loss in stress scenario
                warnings.append(f"Severe stress loss in {scenario_name}: {pnl_percent:.1f}%")
        
        # Hedge warnings
        portfolio_greeks = hedge_analysis.get('portfolio_greeks', {})
        total_delta = abs(portfolio_greeks.get('total_delta', 0))
        if total_delta > 100:  # High delta exposure
            warnings.append(f"High delta exposure: {total_delta:.0f}")
        
        total_vega = abs(portfolio_greeks.get('total_vega', 0))
        if total_vega > 50:  # High vega exposure
            warnings.append(f"High vega exposure: {total_vega:.0f}")
        
        return warnings

# Utility functions for risk management
def create_sample_portfolio() -> List[Dict]:
    """Create a sample portfolio for testing."""
    portfolio = [
        {
            'instrument': 'SPY_Call_400_30d',
            'type': 'option',
            'option_type': 'call',
            'strike': 400,
            'time_to_expiry': 30/365,
            'quantity': 10
        },
        {
            'instrument': 'SPY_Put_390_30d', 
            'type': 'option',
            'option_type': 'put',
            'strike': 390,
            'time_to_expiry': 30/365,
            'quantity': -5
        },
        {
            'instrument': 'SPY_Stock',
            'type': 'stock',
            'quantity': 100
        }
    ]
    return portfolio

def create_sample_market_data() -> Dict:
    """Create sample market data for testing."""
    return {
        'spot_price': 395.50,
        'volatility': 0.22,
        'risk_free_rate': 0.025,
        'dividend_yield': 0.018
    }

if __name__ == "__main__":
    # Test the risk management system
    print("Testing Advanced Risk Management System...")
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Create sample data
    portfolio = create_sample_portfolio()
    market_data = create_sample_market_data()
    
    print(f"\nSample Portfolio: {len(portfolio)} positions")
    print(f"Current Spot Price: ${market_data['spot_price']}")
    
    # Generate sample returns for risk metrics
    np.random.seed(42)
    sample_returns = np.random.normal(0.0005, 0.02, 252)  # Daily returns for 1 year
    
    # Calculate risk metrics
    print("\n1. Calculating Risk Metrics...")
    risk_metrics = risk_manager.calculate_portfolio_risk_metrics(sample_returns)
    
    print("Risk Metrics:")
    for metric, value in risk_metrics.to_dict().items():
        if value is not None:
            if 'ratio' in metric.lower():
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value:.2%}")
    
    # Stress testing
    print("\n2. Performing Stress Tests...")
    stress_results = risk_manager.stress_test_portfolio(portfolio, market_data)
    
    print("Stress Test Results:")
    for scenario, results in stress_results.items():
        pnl_pct = results['total_pnl_percent']
        print(f"  {scenario}: {pnl_pct:.1f}% P&L")
    
    # Hedge analysis
    print("\n3. Calculating Hedge Recommendations...")
    hedge_analysis = risk_manager.calculate_dynamic_hedge_ratio(portfolio, market_data)
    
    portfolio_greeks = hedge_analysis['portfolio_greeks']
    print("Portfolio Greeks:")
    for greek, value in portfolio_greeks.items():
        print(f"  {greek}: {value:.2f}")
    
    hedge_recs = hedge_analysis['hedge_recommendations']
    print("\nHedge Recommendations:")
    for rec, value in hedge_recs.items():
        print(f"  {rec}: {value:.2f}")
    
    # Generate comprehensive risk report
    print("\n4. Generating Risk Report...")
    risk_report = risk_manager.generate_risk_report(portfolio, market_data, sample_returns)
    
    print(f"Risk Report Generated at: {risk_report['timestamp']}")
    print(f"Portfolio Value: ${risk_report['portfolio_summary']['total_value']:.2f}")
    
    warnings = risk_report['risk_warnings']
    if warnings:
        print("\nRisk Warnings:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("\n✅ No significant risk warnings detected")
