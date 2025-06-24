"""
Advanced Backtesting and Model Validation Framework

Comprehensive framework for testing and validating option pricing models including:
- Historical backtesting with multiple time periods
- Cross-validation and walk-forward analysis
- Model performance benchmarking
- Statistical significance testing
- Risk-adjusted performance metrics
- Model comparison and ranking
- Overfitting detection
- Production readiness assessment

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestType(Enum):
    """Types of backtesting strategies."""
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

class ModelType(Enum):
    """Supported model types for validation."""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    MONTE_CARLO = "monte_carlo"
    HESTON = "heston"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class BacktestResults:
    """Container for backtest results."""
    model_name: str
    test_period: Tuple[datetime, datetime]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    expected_shortfall: float
    pricing_errors: Dict[str, float]
    statistical_tests: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_name': self.model_name,
            'test_period_start': self.test_period[0].isoformat(),
            'test_period_end': self.test_period[1].isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'var_95': self.var_95,
            'expected_shortfall': self.expected_shortfall,
            'pricing_errors': self.pricing_errors,
            'statistical_tests': self.statistical_tests
        }

class ModelValidator:
    """
    Advanced model validation and testing framework.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize model validator.
        
        Args:
            risk_free_rate: Risk-free rate for performance calculations
        """
        self.risk_free_rate = risk_free_rate
        self.validation_results = {}
        self.benchmark_models = {}
    
    def validate_pricing_accuracy(self, model_prices: np.ndarray, 
                                market_prices: np.ndarray,
                                model_name: str = "Model") -> Dict[str, float]:
        """
        Validate pricing accuracy against market prices.
        
        Args:
            model_prices: Array of model-predicted prices
            market_prices: Array of actual market prices
            model_name: Name of the model being tested
            
        Returns:
            Dictionary of pricing accuracy metrics
        """
        # Remove any invalid prices
        valid_mask = ~(np.isnan(model_prices) | np.isnan(market_prices) | 
                      (market_prices <= 0) | (model_prices <= 0))
        
        model_clean = model_prices[valid_mask]
        market_clean = market_prices[valid_mask]
        
        if len(model_clean) == 0:
            return {'error': 'No valid price pairs for validation'}
        
        # Calculate pricing errors
        absolute_errors = np.abs(model_clean - market_clean)
        relative_errors = absolute_errors / market_clean
        
        # Mean pricing errors
        mae = np.mean(absolute_errors)
        mape = np.mean(relative_errors) * 100  # Mean Absolute Percentage Error
        rmse = np.sqrt(np.mean((model_clean - market_clean)**2))
        
        # Relative RMSE
        rmse_relative = rmse / np.mean(market_clean) * 100
        
        # R-squared
        r2 = r2_score(market_clean, model_clean)
        
        # Directional accuracy (for relative moves)
        if len(model_clean) > 1:
            market_returns = np.diff(market_clean) / market_clean[:-1]
            model_returns = np.diff(model_clean) / model_clean[:-1]
            
            # Calculate correlation of returns
            return_correlation = np.corrcoef(market_returns, model_returns)[0, 1]
            
            # Directional accuracy
            directional_accuracy = np.mean(np.sign(market_returns) == np.sign(model_returns))
        else:
            return_correlation = np.nan
            directional_accuracy = np.nan
        
        # Statistical tests
        
        # Normality test of errors (Shapiro-Wilk)
        if len(relative_errors) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(relative_errors)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Test for bias (t-test that mean error = 0)
        if len(relative_errors) >= 3:
            bias_tstat, bias_p = stats.ttest_1samp(relative_errors, 0)
        else:
            bias_tstat, bias_p = np.nan, np.nan
        
        # Ljung-Box test for autocorrelation in errors
        if len(relative_errors) >= 10:
            # Simplified autocorrelation test
            errors_lag1 = relative_errors[1:]
            errors_lag0 = relative_errors[:-1]
            autocorr = np.corrcoef(errors_lag0, errors_lag1)[0, 1]
        else:
            autocorr = np.nan
        
        return {
            'mae': mae,
            'mape': mape,
            'rmse': rmse,
            'rmse_relative': rmse_relative,
            'r_squared': r2,
            'return_correlation': return_correlation,
            'directional_accuracy': directional_accuracy,
            'bias_test_stat': bias_tstat,
            'bias_test_p_value': bias_p,
            'normality_test_stat': shapiro_stat,
            'normality_test_p_value': shapiro_p,
            'error_autocorrelation': autocorr,
            'sample_size': len(model_clean)
        }
    
    def cross_validate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                           cv_folds: int = 5, 
                           scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model object with fit/predict methods
            X: Feature matrix
            y: Target values
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            # Use time series split for financial data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring)
            
            return {
                'cv_mean_score': np.mean(cv_scores),
                'cv_std_score': np.std(cv_scores),
                'cv_min_score': np.min(cv_scores),
                'cv_max_score': np.max(cv_scores),
                'cv_scores': cv_scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Cross-validation error: {e}")
            return {'error': str(e)}
    
    def walk_forward_analysis(self, price_data: pd.DataFrame,
                            model_func: Callable,
                            window_size: int = 252,
                            step_size: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward analysis of a pricing model.
        
        Args:
            price_data: DataFrame with price and option data
            model_func: Function that takes data and returns predictions
            window_size: Size of training window (in days)
            step_size: Step size for moving window (in days)
            
        Returns:
            Walk-forward analysis results
        """
        results = []
        all_predictions = []
        all_actuals = []
        
        # Ensure data is sorted by date
        price_data = price_data.sort_index()
        
        start_idx = window_size
        end_idx = len(price_data)
        
        for i in range(start_idx, end_idx, step_size):
            try:
                # Define training and test periods
                train_start = max(0, i - window_size)
                train_end = i
                test_start = i
                test_end = min(i + step_size, end_idx)
                
                # Split data
                train_data = price_data.iloc[train_start:train_end]
                test_data = price_data.iloc[test_start:test_end]
                
                if len(train_data) < 30 or len(test_data) == 0:
                    continue
                
                # Apply model
                predictions = model_func(train_data, test_data)
                
                if predictions is None or len(predictions) == 0:
                    continue
                
                # Get actual values
                actual_prices = test_data['option_price'].values
                
                # Calculate period metrics
                if len(predictions) == len(actual_prices):
                    period_metrics = self.validate_pricing_accuracy(
                        predictions, actual_prices, f"Period_{i}"
                    )
                    
                    period_metrics['period_start'] = test_data.index[0]
                    period_metrics['period_end'] = test_data.index[-1]
                    period_metrics['train_size'] = len(train_data)
                    period_metrics['test_size'] = len(test_data)
                    
                    results.append(period_metrics)
                    all_predictions.extend(predictions)
                    all_actuals.extend(actual_prices)
                
            except Exception as e:
                logger.warning(f"Error in walk-forward period {i}: {e}")
                continue
        
        if not results:
            return {'error': 'No valid walk-forward periods'}
        
        # Aggregate results
        aggregate_metrics = {}
        numeric_metrics = ['mae', 'mape', 'rmse', 'r_squared', 'directional_accuracy']
        
        for metric in numeric_metrics:
            values = [r.get(metric, np.nan) for r in results if metric in r]
            if values:
                aggregate_metrics[f'{metric}_mean'] = np.nanmean(values)
                aggregate_metrics[f'{metric}_std'] = np.nanstd(values)
                aggregate_metrics[f'{metric}_trend'] = self._calculate_trend(values)
        
        # Overall performance
        if all_predictions and all_actuals:
            overall_metrics = self.validate_pricing_accuracy(
                np.array(all_predictions), np.array(all_actuals), "Overall"
            )
            aggregate_metrics.update(overall_metrics)
        
        return {
            'period_results': results,
            'aggregate_metrics': aggregate_metrics,
            'total_periods': len(results),
            'total_predictions': len(all_predictions)
        }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values."""
        if len(values) < 3:
            return 0.0
        
        x = np.arange(len(values))
        valid_mask = ~np.isnan(values)
        
        if np.sum(valid_mask) < 3:
            return 0.0
        
        try:
            slope, _, _, p_value, _ = stats.linregress(x[valid_mask], np.array(values)[valid_mask])
            return slope if p_value < 0.05 else 0.0
        except:
            return 0.0
    
    def benchmark_model_performance(self, models: Dict[str, Any],
                                  test_data: pd.DataFrame,
                                  benchmark_model: str = 'black_scholes') -> Dict[str, Any]:
        """
        Benchmark multiple models against each other.
        
        Args:
            models: Dictionary of model names and objects
            test_data: Test dataset
            benchmark_model: Name of benchmark model
            
        Returns:
            Benchmarking results
        """
        results = {}
        benchmark_metrics = None
        
        for model_name, model in models.items():
            try:
                # Generate predictions (this would be model-specific)
                if hasattr(model, 'predict'):
                    predictions = model.predict(test_data)
                else:
                    # Assume it's a function
                    predictions = model(test_data)
                
                actual_prices = test_data['option_price'].values
                
                # Calculate metrics
                metrics = self.validate_pricing_accuracy(predictions, actual_prices, model_name)
                
                # Add model-specific metrics
                metrics['model_name'] = model_name
                metrics['is_benchmark'] = (model_name == benchmark_model)
                
                if model_name == benchmark_model:
                    benchmark_metrics = metrics
                
                results[model_name] = metrics
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        # Calculate relative performance
        if benchmark_metrics and 'mae' in benchmark_metrics:
            for model_name, metrics in results.items():
                if 'mae' in metrics and model_name != benchmark_model:
                    metrics['mae_improvement'] = (benchmark_metrics['mae'] - metrics['mae']) / benchmark_metrics['mae']
                    metrics['rmse_improvement'] = (benchmark_metrics['rmse'] - metrics['rmse']) / benchmark_metrics['rmse']
                    metrics['r2_improvement'] = metrics['r_squared'] - benchmark_metrics['r_squared']
        
        # Rank models by performance
        valid_models = {k: v for k, v in results.items() if 'mae' in v}
        
        if valid_models:
            # Rank by multiple criteria
            rankings = {}
            
            # Rank by MAE (lower is better)
            mae_ranking = sorted(valid_models.items(), key=lambda x: x[1]['mae'])
            rankings['mae'] = [(name, rank+1) for rank, (name, _) in enumerate(mae_ranking)]
            
            # Rank by R² (higher is better)
            r2_ranking = sorted(valid_models.items(), key=lambda x: x[1]['r_squared'], reverse=True)
            rankings['r_squared'] = [(name, rank+1) for rank, (name, _) in enumerate(r2_ranking)]
            
            # Combined ranking (simple average)
            combined_ranks = {}
            for name in valid_models.keys():
                mae_rank = next(rank for n, rank in rankings['mae'] if n == name)
                r2_rank = next(rank for n, rank in rankings['r_squared'] if n == name)
                combined_ranks[name] = (mae_rank + r2_rank) / 2
            
            overall_ranking = sorted(combined_ranks.items(), key=lambda x: x[1])
            rankings['overall'] = [(name, rank+1) for rank, (name, _) in enumerate(overall_ranking)]
        else:
            rankings = {}
        
        return {
            'model_results': results,
            'rankings': rankings,
            'benchmark_model': benchmark_model,
            'test_period': (test_data.index[0], test_data.index[-1]),
            'total_observations': len(test_data)
        }
    
    def detect_overfitting(self, train_metrics: Dict[str, float],
                          validation_metrics: Dict[str, float],
                          test_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Detect overfitting by comparing train/validation/test performance.
        
        Args:
            train_metrics: Training set metrics
            validation_metrics: Validation set metrics
            test_metrics: Test set metrics
            
        Returns:
            Overfitting analysis results
        """
        analysis = {}
        
        # Performance degradation from train to validation
        if 'r_squared' in train_metrics and 'r_squared' in validation_metrics:
            r2_degradation = train_metrics['r_squared'] - validation_metrics['r_squared']
            analysis['r2_train_val_gap'] = r2_degradation
            analysis['r2_overfitting_risk'] = 'High' if r2_degradation > 0.1 else 'Medium' if r2_degradation > 0.05 else 'Low'
        
        # Performance degradation from validation to test
        if 'r_squared' in validation_metrics and 'r_squared' in test_metrics:
            r2_val_test_gap = validation_metrics['r_squared'] - test_metrics['r_squared']
            analysis['r2_val_test_gap'] = r2_val_test_gap
            analysis['r2_generalization_risk'] = 'High' if r2_val_test_gap > 0.1 else 'Medium' if r2_val_test_gap > 0.05 else 'Low'
        
        # Error increase from train to test
        if 'mae' in train_metrics and 'mae' in test_metrics:
            mae_increase = (test_metrics['mae'] - train_metrics['mae']) / train_metrics['mae']
            analysis['mae_increase_pct'] = mae_increase * 100
            analysis['mae_overfitting_risk'] = 'High' if mae_increase > 0.5 else 'Medium' if mae_increase > 0.2 else 'Low'
        
        # Overall overfitting assessment
        risk_indicators = [
            analysis.get('r2_overfitting_risk', 'Low'),
            analysis.get('r2_generalization_risk', 'Low'),
            analysis.get('mae_overfitting_risk', 'Low')
        ]
        
        high_risk_count = risk_indicators.count('High')
        medium_risk_count = risk_indicators.count('Medium')
        
        if high_risk_count >= 2:
            analysis['overall_overfitting_risk'] = 'High'
        elif high_risk_count >= 1 or medium_risk_count >= 2:
            analysis['overall_overfitting_risk'] = 'Medium'
        else:
            analysis['overall_overfitting_risk'] = 'Low'
        
        # Recommendations
        recommendations = []
        
        if analysis.get('overall_overfitting_risk') == 'High':
            recommendations.extend([
                "Consider regularization techniques",
                "Reduce model complexity",
                "Increase training data size",
                "Use cross-validation for hyperparameter tuning"
            ])
        elif analysis.get('overall_overfitting_risk') == 'Medium':
            recommendations.extend([
                "Monitor model performance on new data",
                "Consider ensemble methods",
                "Validate feature selection"
            ])
        else:
            recommendations.append("Model shows good generalization")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def stress_test_model(self, model: Any, base_data: pd.DataFrame,
                         stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Stress test a model under various market scenarios.
        
        Args:
            model: Model to test
            base_data: Base market data
            stress_scenarios: List of stress scenario dictionaries
            
        Returns:
            Stress test results
        """
        results = {}
        base_predictions = None
        
        try:
            # Get base case predictions
            if hasattr(model, 'predict'):
                base_predictions = model.predict(base_data)
            else:
                base_predictions = model(base_data)
            
            results['base_case'] = {
                'mean_price': np.mean(base_predictions),
                'std_price': np.std(base_predictions),
                'min_price': np.min(base_predictions),
                'max_price': np.max(base_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error in base case prediction: {e}")
            return {'error': 'Failed to generate base predictions'}
        
        # Apply stress scenarios
        for i, scenario in enumerate(stress_scenarios):
            try:
                # Create stressed data
                stressed_data = base_data.copy()
                
                # Apply shocks
                for param, shock in scenario.items():
                    if param in stressed_data.columns:
                        if param == 'volatility':
                            stressed_data[param] *= (1 + shock)
                        elif param == 'spot_price':
                            stressed_data[param] *= (1 + shock)
                        elif param == 'interest_rate':
                            stressed_data[param] += shock
                        else:
                            stressed_data[param] *= (1 + shock)
                
                # Generate stressed predictions
                if hasattr(model, 'predict'):
                    stressed_predictions = model.predict(stressed_data)
                else:
                    stressed_predictions = model(stressed_data)
                
                # Calculate impact
                price_change = np.mean(stressed_predictions) - np.mean(base_predictions)
                price_change_pct = price_change / np.mean(base_predictions) * 100
                
                results[f'scenario_{i+1}'] = {
                    'scenario': scenario,
                    'mean_price': np.mean(stressed_predictions),
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'std_price': np.std(stressed_predictions),
                    'worst_case': np.min(stressed_predictions),
                    'best_case': np.max(stressed_predictions)
                }
                
            except Exception as e:
                logger.error(f"Error in stress scenario {i+1}: {e}")
                results[f'scenario_{i+1}'] = {'error': str(e)}
        
        return results
    
    def generate_validation_report(self, model_name: str,
                                 validation_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive validation report.
        
        Args:
            model_name: Name of the model
            validation_results: Results from various validation tests
            
        Returns:
            Formatted validation report string
        """
        report_lines = []
        report_lines.append(f"Model Validation Report: {model_name}")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Pricing Accuracy
        if 'pricing_accuracy' in validation_results:
            pa = validation_results['pricing_accuracy']
            report_lines.append("PRICING ACCURACY")
            report_lines.append("-" * 20)
            report_lines.append(f"Mean Absolute Error: {pa.get('mae', 'N/A'):.4f}")
            report_lines.append(f"Mean Absolute Percentage Error: {pa.get('mape', 'N/A'):.2f}%")
            report_lines.append(f"Root Mean Square Error: {pa.get('rmse', 'N/A'):.4f}")
            report_lines.append(f"R-squared: {pa.get('r_squared', 'N/A'):.4f}")
            report_lines.append(f"Directional Accuracy: {pa.get('directional_accuracy', 'N/A'):.2%}")
            report_lines.append("")
        
        # Cross-validation
        if 'cross_validation' in validation_results:
            cv = validation_results['cross_validation']
            report_lines.append("CROSS-VALIDATION")
            report_lines.append("-" * 20)
            report_lines.append(f"Mean CV Score: {cv.get('cv_mean_score', 'N/A'):.4f}")
            report_lines.append(f"CV Standard Deviation: {cv.get('cv_std_score', 'N/A'):.4f}")
            report_lines.append("")
        
        # Overfitting Analysis
        if 'overfitting' in validation_results:
            of = validation_results['overfitting']
            report_lines.append("OVERFITTING ANALYSIS")
            report_lines.append("-" * 20)
            report_lines.append(f"Overall Risk: {of.get('overall_overfitting_risk', 'N/A')}")
            report_lines.append(f"R² Train-Val Gap: {of.get('r2_train_val_gap', 'N/A'):.4f}")
            report_lines.append(f"MAE Increase: {of.get('mae_increase_pct', 'N/A'):.2f}%")
            
            if 'recommendations' in of:
                report_lines.append("Recommendations:")
                for rec in of['recommendations']:
                    report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        # Statistical Tests
        if 'pricing_accuracy' in validation_results:
            pa = validation_results['pricing_accuracy']
            report_lines.append("STATISTICAL TESTS")
            report_lines.append("-" * 20)
            
            bias_p = pa.get('bias_test_p_value')
            if bias_p is not None:
                bias_significant = "Yes" if bias_p < 0.05 else "No"
                report_lines.append(f"Bias Test p-value: {bias_p:.4f} (Significant: {bias_significant})")
            
            norm_p = pa.get('normality_test_p_value')
            if norm_p is not None:
                norm_significant = "Yes" if norm_p < 0.05 else "No"
                report_lines.append(f"Normality Test p-value: {norm_p:.4f} (Non-normal: {norm_significant})")
            
            autocorr = pa.get('error_autocorrelation')
            if autocorr is not None:
                report_lines.append(f"Error Autocorrelation: {autocorr:.4f}")
            report_lines.append("")
        
        # Production Readiness
        report_lines.append("PRODUCTION READINESS ASSESSMENT")
        report_lines.append("-" * 30)
        
        readiness_score = self._calculate_readiness_score(validation_results)
        report_lines.append(f"Readiness Score: {readiness_score}/100")
        
        if readiness_score >= 80:
            report_lines.append("Status: READY FOR PRODUCTION")
        elif readiness_score >= 60:
            report_lines.append("Status: READY WITH MONITORING")
        else:
            report_lines.append("Status: NOT READY FOR PRODUCTION")
        
        return "\n".join(report_lines)
    
    def _calculate_readiness_score(self, validation_results: Dict[str, Any]) -> int:
        """Calculate production readiness score (0-100)."""
        score = 0
        
        # Pricing accuracy (40 points)
        if 'pricing_accuracy' in validation_results:
            pa = validation_results['pricing_accuracy']
            
            # MAPE score (20 points)
            mape = pa.get('mape', 100)
            if mape < 5:
                score += 20
            elif mape < 10:
                score += 15
            elif mape < 20:
                score += 10
            elif mape < 30:
                score += 5
            
            # R² score (20 points)
            r2 = pa.get('r_squared', 0)
            if r2 > 0.9:
                score += 20
            elif r2 > 0.8:
                score += 15
            elif r2 > 0.7:
                score += 10
            elif r2 > 0.5:
                score += 5
        
        # Overfitting (30 points)
        if 'overfitting' in validation_results:
            of = validation_results['overfitting']
            risk = of.get('overall_overfitting_risk', 'High')
            
            if risk == 'Low':
                score += 30
            elif risk == 'Medium':
                score += 20
            else:
                score += 10
        
        # Statistical validity (20 points)
        if 'pricing_accuracy' in validation_results:
            pa = validation_results['pricing_accuracy']
            
            # No significant bias (10 points)
            bias_p = pa.get('bias_test_p_value')
            if bias_p is not None and bias_p > 0.05:
                score += 10
            
            # Low autocorrelation (10 points)
            autocorr = pa.get('error_autocorrelation')
            if autocorr is not None and abs(autocorr) < 0.2:
                score += 10
        
        # Cross-validation stability (10 points)
        if 'cross_validation' in validation_results:
            cv = validation_results['cross_validation']
            cv_std = cv.get('cv_std_score', 1.0)
            
            if cv_std < 0.1:
                score += 10
            elif cv_std < 0.2:
                score += 5
        
        return min(score, 100)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced Model Validation Framework...")
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic option data
    test_data = pd.DataFrame({
        'spot_price': np.random.uniform(90, 110, n_samples),
        'strike': np.random.uniform(85, 115, n_samples),
        'time_to_expiry': np.random.uniform(0.1, 1.0, n_samples),
        'risk_free_rate': np.random.uniform(0.01, 0.05, n_samples),
        'volatility': np.random.uniform(0.15, 0.35, n_samples),
    })
    
    # Generate synthetic option prices (with noise)
    from api.option_pricing import black_scholes
    
    true_prices = []
    for _, row in test_data.iterrows():
        try:
            price = black_scholes(
                row['spot_price'], row['strike'], row['time_to_expiry'],
                row['risk_free_rate'], row['volatility'], 'call'
            )
            # Add noise to simulate market imperfections
            noise = np.random.normal(0, 0.05 * price)
            true_prices.append(price + noise)
        except:
            true_prices.append(np.nan)
    
    test_data['option_price'] = true_prices
    test_data = test_data.dropna()
    
    print(f"Generated {len(test_data)} synthetic option contracts")
    
    # Initialize validator
    validator = ModelValidator()
    
    # Test pricing accuracy validation
    print("\n1. Testing Pricing Accuracy Validation...")
    
    # Create model predictions (Black-Scholes with slightly different parameters)
    model_prices = []
    for _, row in test_data.iterrows():
        # Simulate model bias by adjusting volatility slightly
        adjusted_vol = row['volatility'] * 1.02  # 2% vol bias
        price = black_scholes(
            row['spot_price'], row['strike'], row['time_to_expiry'],
            row['risk_free_rate'], adjusted_vol, 'call'
        )
        model_prices.append(price)
    
    pricing_metrics = validator.validate_pricing_accuracy(
        np.array(model_prices), test_data['option_price'].values, "Test Model"
    )
    
    print("Pricing Accuracy Metrics:")
    for metric, value in pricing_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Test overfitting detection
    print("\n2. Testing Overfitting Detection...")
    
    # Simulate train/validation/test metrics
    train_metrics = {'r_squared': 0.95, 'mae': 0.10}
    val_metrics = {'r_squared': 0.88, 'mae': 0.15}
    test_metrics = {'r_squared': 0.85, 'mae': 0.18}
    
    overfitting_analysis = validator.detect_overfitting(
        train_metrics, val_metrics, test_metrics
    )
    
    print("Overfitting Analysis:")
    for key, value in overfitting_analysis.items():
        if key != 'recommendations':
            print(f"  {key}: {value}")
    
    print("Recommendations:")
    for rec in overfitting_analysis.get('recommendations', []):
        print(f"  • {rec}")
    
    # Test stress testing
    print("\n3. Testing Model Stress Testing...")
    
    # Define a simple model function for testing
    def simple_bs_model(data):
        prices = []
        for _, row in data.iterrows():
            price = black_scholes(
                row['spot_price'], row['strike'], row['time_to_expiry'],
                row['risk_free_rate'], row['volatility'], 'call'
            )
            prices.append(price)
        return np.array(prices)
    
    # Define stress scenarios
    stress_scenarios = [
        {'spot_price': -0.20, 'volatility': 0.50},  # Market crash
        {'volatility': 1.00},  # Volatility spike
        {'interest_rate': 0.02},  # Rate increase
        {'spot_price': -0.10, 'volatility': 0.30, 'interest_rate': 0.01}  # Combined stress
    ]
    
    stress_results = validator.stress_test_model(
        simple_bs_model, test_data.head(100), stress_scenarios
    )
    
    print("Stress Test Results:")
    for scenario, results in stress_results.items():
        if 'error' not in results:
            if scenario == 'base_case':
                print(f"  {scenario}: Mean Price = {results['mean_price']:.4f}")
            else:
                print(f"  {scenario}: Price Change = {results.get('price_change_pct', 0):.2f}%")
    
    # Generate validation report
    print("\n4. Generating Validation Report...")
    
    validation_results = {
        'pricing_accuracy': pricing_metrics,
        'overfitting': overfitting_analysis,
        'stress_test': stress_results
    }
    
    report = validator.generate_validation_report("Black-Scholes Test Model", validation_results)
    
    print("\nValidation Report:")
    print(report)
    
    print("\nModel validation framework testing completed!")
