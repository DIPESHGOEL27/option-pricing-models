# This module is now deprecated. Portfolio management features have been removed from the platform.

"""
Risk Analytics Module

Professional-grade risk analytics tools including:
- Greek hedging and risk management
- Correlation analysis and portfolio optimization
- Real-time risk monitoring
- Performance attribution
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CorrelationAnalysis:
    """Correlation and covariance analysis for multi-asset portfolios"""
    
    def __init__(self):
        self.price_data = pd.DataFrame()
        self.return_data = pd.DataFrame()
        self.correlation_matrix = pd.DataFrame()
        self.covariance_matrix = pd.DataFrame()
    
    def load_price_data(self, price_data: pd.DataFrame) -> None:
        """Load historical price data"""
        self.price_data = price_data.copy()
        self.return_data = price_data.pct_change().dropna()
        self._calculate_correlation_matrices()
    
    def _calculate_correlation_matrices(self) -> None:
        """Calculate correlation and covariance matrices"""
        self.correlation_matrix = self.return_data.corr()
        self.covariance_matrix = self.return_data.cov()
    
    def portfolio_correlation_impact(self, weights: Dict[str, float]) -> Dict:
        """Analyze correlation impact on portfolio risk"""
        symbols = list(weights.keys())
        weight_vector = np.array([weights[s] for s in symbols])
        
        # Portfolio variance with correlation
        relevant_cov = self.covariance_matrix.loc[symbols, symbols]
        portfolio_variance = np.dot(weight_vector.T, np.dot(relevant_cov, weight_vector))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio variance without correlation (assumes zero correlation)
        individual_variances = np.diagonal(relevant_cov)
        uncorrelated_variance = np.dot(weight_vector**2, individual_variances)
        uncorrelated_volatility = np.sqrt(uncorrelated_variance)
        
        # Diversification ratio
        diversification_ratio = uncorrelated_volatility / portfolio_volatility
        
        return {
            'portfolio_volatility': portfolio_volatility,
            'uncorrelated_volatility': uncorrelated_volatility,
            'diversification_ratio': diversification_ratio,
            'correlation_benefit': uncorrelated_volatility - portfolio_volatility,
            'correlation_matrix': self.correlation_matrix.loc[symbols, symbols]
        }
    
    def risk_attribution(self, weights: Dict[str, float]) -> Dict:
        """Decompose portfolio risk by asset and correlation"""
        symbols = list(weights.keys())
        weight_vector = np.array([weights[s] for s in symbols])
        
        relevant_cov = self.covariance_matrix.loc[symbols, symbols]
        portfolio_variance = np.dot(weight_vector.T, np.dot(relevant_cov, weight_vector))
        
        # Marginal contribution to risk
        marginal_contrib = np.dot(relevant_cov, weight_vector) / np.sqrt(portfolio_variance)
        
        # Component contribution to risk
        component_contrib = weight_vector * marginal_contrib
        
        # Percentage contribution
        pct_contrib = component_contrib / np.sqrt(portfolio_variance)
        
        risk_attribution = {}
        for i, symbol in enumerate(symbols):
            risk_attribution[symbol] = {
                'weight': weights[symbol],
                'marginal_risk': marginal_contrib[i],
                'component_risk': component_contrib[i],
                'risk_percentage': pct_contrib[i] * 100
            }
        
        return risk_attribution
    
    def correlation_stress_test(self, base_correlations: pd.DataFrame,
                              stress_scenarios: Dict) -> Dict:
        """Test portfolio under different correlation scenarios"""
        results = {}
        
        for scenario_name, correlation_changes in stress_scenarios.items():
            # Apply correlation stress
            stressed_corr = base_correlations.copy()
            
            for (asset1, asset2), new_corr in correlation_changes.items():
                stressed_corr.loc[asset1, asset2] = new_corr
                stressed_corr.loc[asset2, asset1] = new_corr
            
            # Convert correlation to covariance
            std_devs = np.sqrt(np.diagonal(self.covariance_matrix))
            stressed_cov = np.outer(std_devs, std_devs) * stressed_corr
            
            results[scenario_name] = {
                'stressed_correlation': stressed_corr,
                'stressed_covariance': pd.DataFrame(stressed_cov, 
                                                  index=stressed_corr.index,
                                                  columns=stressed_corr.columns)
            }
        
        return results


class PerformanceAttribution:
    """Performance attribution and analytics"""
    
    def __init__(self):
        self.performance_history = pd.DataFrame()
        self.benchmark_returns = pd.Series()
    
    def load_performance_data(self, portfolio_returns: pd.Series,
                            benchmark_returns: pd.Series = None) -> None:
        """Load performance data"""
        self.performance_history = portfolio_returns.copy()
        if benchmark_returns is not None:
            self.benchmark_returns = benchmark_returns.copy()
    
    def calculate_performance_metrics(self, risk_free_rate: float = 0.02) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = self.performance_history
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean())**252 - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Value at Risk
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        # Expected Shortfall
        es_95 = returns[returns <= var_95].mean()
        es_99 = returns[returns <= var_99].mean()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99
        }
        
        # Benchmark comparison if available
        if len(self.benchmark_returns) > 0:
            aligned_returns = returns.align(self.benchmark_returns, join='inner')
            portfolio_aligned = aligned_returns[0]
            benchmark_aligned = aligned_returns[1]
            
            # Tracking error
            tracking_error = (portfolio_aligned - benchmark_aligned).std() * np.sqrt(252)
            
            # Information ratio
            excess_return = portfolio_aligned.mean() - benchmark_aligned.mean()
            information_ratio = (excess_return * 252) / tracking_error if tracking_error > 0 else 0
            
            # Beta
            covariance = np.cov(portfolio_aligned, benchmark_aligned)[0, 1]
            benchmark_variance = np.var(benchmark_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            # Alpha
            benchmark_annual_return = (1 + benchmark_aligned.mean())**252 - 1
            alpha = annualized_return - (risk_free_rate + beta * (benchmark_annual_return - risk_free_rate))
            
            metrics.update({
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'beta': beta,
                'alpha': alpha,
                'benchmark_annual_return': benchmark_annual_return
            })
        
        return metrics
    
    def attribution_analysis(self, sector_returns: pd.DataFrame,
                           portfolio_weights: pd.DataFrame) -> Dict:
        """Perform return attribution analysis"""
        # Ensure alignment
        aligned_data = sector_returns.align(portfolio_weights, join='inner')
        returns_aligned = aligned_data[0]
        weights_aligned = aligned_data[1]
        
        # Calculate contribution
        contributions = returns_aligned.multiply(weights_aligned, axis=0)
        total_contribution = contributions.sum(axis=1)
        
        # Attribution by sector
        sector_attribution = {}
        for sector in contributions.columns:
            sector_attribution[sector] = {
                'total_contribution': contributions[sector].sum(),
                'average_weight': weights_aligned[sector].mean(),
                'average_return': returns_aligned[sector].mean()
            }
        
        return {
            'total_portfolio_return': total_contribution.sum(),
            'sector_attribution': sector_attribution,
            'daily_contributions': contributions
        }


class RiskBudgeting:
    """Risk budgeting and allocation optimization"""
    
    def __init__(self, covariance_matrix: pd.DataFrame):
        self.cov_matrix = covariance_matrix
        self.assets = covariance_matrix.index.tolist()
    
    def equal_risk_contribution(self) -> Dict:
        """Calculate equal risk contribution portfolio"""
        n = len(self.assets)
        
        def objective(weights):
            # Calculate risk contributions
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Target equal risk contribution (1/n for each asset)
            target_contrib = np.ones(n) / n
            
            # Minimize sum of squared deviations from target
            return np.sum((risk_contrib - target_contrib)**2)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Bounds (no short selling)
        bounds = [(0, 1) for _ in range(n)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n) / n
        
        # Optimize
        result = optimize.minimize(objective, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, 
                                         np.dot(self.cov_matrix, optimal_weights)))
            marginal_contrib = np.dot(self.cov_matrix, optimal_weights) / portfolio_vol
            risk_contrib = optimal_weights * marginal_contrib / portfolio_vol
            
            return {
                'weights': dict(zip(self.assets, optimal_weights)),
                'risk_contributions': dict(zip(self.assets, risk_contrib)),
                'portfolio_volatility': portfolio_vol,
                'optimization_success': True
            }
        else:
            return {'optimization_success': False, 'message': result.message}
    
    def risk_budgeted_portfolio(self, risk_budgets: Dict[str, float]) -> Dict:
        """Create portfolio with specified risk budgets"""
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib / portfolio_vol
            
            # Calculate deviations from target risk budgets
            total_deviation = 0
            for i, asset in enumerate(self.assets):
                target_risk = risk_budgets.get(asset, 1/len(self.assets))
                total_deviation += (risk_contrib[i] - target_risk)**2
            
            return total_deviation
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1) for _ in range(len(self.assets))]
        x0 = np.ones(len(self.assets)) / len(self.assets)
        
        result = optimize.minimize(objective, x0, method='SLSQP',
                                 bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            portfolio_vol = np.sqrt(np.dot(optimal_weights.T, 
                                         np.dot(self.cov_matrix, optimal_weights)))
            marginal_contrib = np.dot(self.covariance_matrix, optimal_weights) / portfolio_vol
            actual_risk_contrib = optimal_weights * marginal_contrib / portfolio_vol
            
            return {
                'weights': dict(zip(self.assets, optimal_weights)),
                'actual_risk_contributions': dict(zip(self.assets, actual_risk_contrib)),
                'target_risk_budgets': risk_budgets,
                'portfolio_volatility': portfolio_vol,
                'optimization_success': True
            }
        else:
            return {'optimization_success': False, 'message': result.message}


# Example usage
if __name__ == "__main__":
    print("This module is deprecated. Portfolio management features have been removed.")
