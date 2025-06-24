"""
Portfolio Management and Risk Analytics Module

Professional-grade portfolio management tools including:
- Multi-asset option portfolios
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


class OptionPortfolio:
    """Option portfolio management and analytics"""
    
    def __init__(self):
        self.positions = pd.DataFrame(columns=[
            'symbol', 'option_type', 'strike', 'expiry', 'quantity', 
            'premium_paid', 'current_price', 'underlying_price', 
            'volatility', 'risk_free_rate', 'model_type'
        ])
        self.portfolio_greeks = {}
        self.portfolio_value = 0.0
        self.total_pnl = 0.0
    
    def add_position(self, symbol: str, option_type: str, strike: float,
                    expiry: datetime, quantity: int, premium_paid: float,
                    underlying_price: float, volatility: float,
                    risk_free_rate: float, model_type: str = 'black_scholes') -> None:
        """Add an option position to the portfolio"""
        
        # Calculate current option price
        from .advanced_models import MonteCarloEngine
        
        T = (expiry - datetime.now()).days / 365.0
        
        if model_type == 'black_scholes':
            current_price = self._black_scholes_price(
                underlying_price, strike, T, risk_free_rate, volatility, option_type)
        else:  # Monte Carlo
            mc_engine = MonteCarloEngine(n_simulations=50000, n_steps=100)
            paths = mc_engine.geometric_brownian_motion(
                underlying_price, T, risk_free_rate, volatility)
            result = mc_engine.price_vanilla_option(paths, strike, risk_free_rate, T, option_type)
            current_price = result['price']
        
        new_position = {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'expiry': expiry,
            'quantity': quantity,
            'premium_paid': premium_paid,
            'current_price': current_price,
            'underlying_price': underlying_price,
            'volatility': volatility,
            'risk_free_rate': risk_free_rate,
            'model_type': model_type
        }
        
        self.positions = pd.concat([self.positions, pd.DataFrame([new_position])], 
                                 ignore_index=True)
        self._update_portfolio_metrics()
    
    def _black_scholes_price(self, S: float, K: float, T: float, r: float,
                           sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        from scipy.stats import norm
        
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
    def _calculate_greeks(self, S: float, K: float, T: float, r: float,
                         sigma: float, option_type: str) -> Dict:
        """Calculate option Greeks"""
        from scipy.stats import norm
        
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        else:
            delta = norm.cdf(d1) - 1
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                r*K*np.exp(-r*T)*norm.cdf(d2 if option_type.lower() == 'call' else -d2))
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def _update_portfolio_metrics(self) -> None:
        """Update portfolio-level metrics"""
        if len(self.positions) == 0:
            return
        
        # Calculate portfolio value and P&L
        position_values = self.positions['current_price'] * self.positions['quantity']
        premium_paid_total = self.positions['premium_paid'] * self.positions['quantity']
        
        self.portfolio_value = position_values.sum()
        self.total_pnl = self.portfolio_value - premium_paid_total.sum()
        
        # Calculate portfolio Greeks
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        total_rho = 0
        
        for _, position in self.positions.iterrows():
            T = (position['expiry'] - datetime.now()).days / 365.0
            greeks = self._calculate_greeks(
                position['underlying_price'], position['strike'], T,
                position['risk_free_rate'], position['volatility'], 
                position['option_type']
            )
            
            total_delta += greeks['delta'] * position['quantity']
            total_gamma += greeks['gamma'] * position['quantity']
            total_theta += greeks['theta'] * position['quantity']
            total_vega += greeks['vega'] * position['quantity']
            total_rho += greeks['rho'] * position['quantity']
        
        self.portfolio_greeks = {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega,
            'rho': total_rho
        }
    
    def get_portfolio_summary(self) -> Dict:
        """Get comprehensive portfolio summary"""
        if len(self.positions) == 0:
            return {"message": "No positions in portfolio"}
        
        # Position-level analytics
        position_analytics = []
        for _, position in self.positions.iterrows():
            T = (position['expiry'] - datetime.now()).days / 365.0
            greeks = self._calculate_greeks(
                position['underlying_price'], position['strike'], T,
                position['risk_free_rate'], position['volatility'], 
                position['option_type']
            )
            
            position_value = position['current_price'] * position['quantity']
            position_pnl = position_value - position['premium_paid'] * position['quantity']
            
            position_analytics.append({
                'symbol': position['symbol'],
                'position_value': position_value,
                'position_pnl': position_pnl,
                'delta': greeks['delta'] * position['quantity'],
                'gamma': greeks['gamma'] * position['quantity'],
                'theta': greeks['theta'] * position['quantity'],
                'vega': greeks['vega'] * position['quantity'],
                'days_to_expiry': T * 365
            })
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'portfolio_greeks': self.portfolio_greeks,
            'position_count': len(self.positions),
            'position_analytics': position_analytics,
            'positions_df': self.positions
        }
    
    def delta_hedge_recommendation(self, underlying_symbols: List[str]) -> Dict:
        """Recommend delta hedging strategy"""
        if len(self.positions) == 0:
            return {"message": "No positions to hedge"}
        
        # Group delta by underlying symbol
        symbol_deltas = {}
        for _, position in self.positions.iterrows():
            symbol = position['symbol']
            T = (position['expiry'] - datetime.now()).days / 365.0
            greeks = self._calculate_greeks(
                position['underlying_price'], position['strike'], T,
                position['risk_free_rate'], position['volatility'], 
                position['option_type']
            )
            
            if symbol not in symbol_deltas:
                symbol_deltas[symbol] = 0
            symbol_deltas[symbol] += greeks['delta'] * position['quantity']
        
        # Hedge recommendations
        hedge_recommendations = {}
        for symbol, net_delta in symbol_deltas.items():
            if abs(net_delta) > 0.01:  # Only hedge if meaningful exposure
                hedge_recommendations[symbol] = {
                    'net_delta': net_delta,
                    'hedge_shares': -net_delta,  # Opposite position
                    'hedge_direction': 'short' if net_delta > 0 else 'long'
                }
        
        return {
            'total_portfolio_delta': self.portfolio_greeks['delta'],
            'symbol_deltas': symbol_deltas,
            'hedge_recommendations': hedge_recommendations
        }
    
    def risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        if len(self.positions) == 0:
            return {"message": "No positions for risk analysis"}
        
        # Time decay analysis
        theta_1day = self.portfolio_greeks['theta']
        theta_7day = theta_1day * 7
        theta_30day = theta_1day * 30
        
        # Volatility exposure
        vega_exposure = self.portfolio_greeks['vega']
        vol_1pct_impact = vega_exposure * 0.01
        
        # Gamma exposure (convexity)
        gamma_exposure = self.portfolio_greeks['gamma']
        
        # Maximum position concentration
        position_values = self.positions['current_price'] * self.positions['quantity']
        max_position_pct = (position_values.max() / self.portfolio_value) * 100
        
        # Days to expiry analysis
        current_date = datetime.now()
        days_to_expiry = [(exp - current_date).days for exp in self.positions['expiry']]
        
        return {
            'portfolio_value': self.portfolio_value,
            'total_pnl': self.total_pnl,
            'time_decay': {
                '1_day_theta': theta_1day,
                '7_day_theta': theta_7day,
                '30_day_theta': theta_30day
            },
            'volatility_risk': {
                'vega_exposure': vega_exposure,
                '1pct_vol_impact': vol_1pct_impact
            },
            'gamma_exposure': gamma_exposure,
            'concentration_risk': {
                'max_position_percentage': max_position_pct
            },
            'expiry_analysis': {
                'min_days_to_expiry': min(days_to_expiry) if days_to_expiry else 0,
                'max_days_to_expiry': max(days_to_expiry) if days_to_expiry else 0,
                'avg_days_to_expiry': np.mean(days_to_expiry) if days_to_expiry else 0
            }
        }


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
            marginal_contrib = np.dot(self.cov_matrix, optimal_weights) / portfolio_vol
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
    # Create sample portfolio
    portfolio = OptionPortfolio()
    
    # Add some positions
    expiry = datetime.now() + timedelta(days=30)
    portfolio.add_position('AAPL', 'call', 150, expiry, 10, 5.0, 155, 0.25, 0.05)
    portfolio.add_position('AAPL', 'put', 140, expiry, -5, 3.0, 155, 0.25, 0.05)
    
    # Get portfolio summary
    summary = portfolio.get_portfolio_summary()
    print("Portfolio Summary:")
    print(f"Portfolio Value: ${summary['portfolio_value']:.2f}")
    print(f"Total P&L: ${summary['total_pnl']:.2f}")
    print(f"Portfolio Delta: {summary['portfolio_greeks']['delta']:.4f}")
    
    # Get delta hedge recommendation
    hedge_rec = portfolio.delta_hedge_recommendation(['AAPL'])
    print("\nDelta Hedge Recommendation:")
    print(hedge_rec)
    
    # Generate risk report
    risk_report = portfolio.risk_report()
    print("\nRisk Report:")
    print(f"1-day Theta: ${risk_report['time_decay']['1_day_theta']:.2f}")
    print(f"Vega Exposure: {risk_report['volatility_risk']['vega_exposure']:.2f}")
