"""
Machine Learning-Based Option Pricing Module

Advanced neural network and machine learning models for option pricing,
volatility prediction, and risk assessment. Includes deep learning models,
ensemble methods, and real-time calibration capabilities.

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

warnings.filterwarnings('ignore')

class NeuralNetworkPricer:
    """
    Deep neural network for option pricing with market features.
    """
    
    def __init__(self, hidden_layers: Tuple[int, ...] = (100, 50, 25),
                 activation: str = 'relu', solver: str = 'adam',
                 learning_rate: float = 0.001, max_iter: int = 1000):
        """
        Initialize neural network option pricer.
        
        Args:
            hidden_layers: Tuple of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Solver for weight optimization
            learning_rate: Learning rate for weight updates
            max_iter: Maximum iterations for training
        """
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        
    def prepare_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for neural network training.
        
        Args:
            market_data: DataFrame with market data and option parameters
            
        Returns:
            Feature matrix for ML model
        """
        features = []
        
        # Basic option parameters
        basic_features = ['S', 'K', 'T', 'r', 'sigma']
        for feature in basic_features:
            if feature in market_data.columns:
                features.append(market_data[feature].values)
        
        # Derived features
        if all(col in market_data.columns for col in ['S', 'K']):
            # Moneyness
            moneyness = market_data['S'] / market_data['K']
            features.append(moneyness.values)
            
            # Log moneyness
            log_moneyness = np.log(moneyness)
            features.append(log_moneyness.values)
        
        # Time-based features
        if 'T' in market_data.columns:
            # Time to expiry categories
            time_to_exp = market_data['T'].values
            features.append(time_to_exp)
            features.append(np.sqrt(time_to_exp))  # Square root of time
            features.append(1 / (1 + time_to_exp))  # Inverse time decay
        
        # Volatility features
        if 'sigma' in market_data.columns:
            vol = market_data['sigma'].values
            features.append(vol)
            features.append(vol**2)  # Variance
            features.append(np.log(vol))  # Log volatility
        
        # Market regime features (if available)
        if 'vix' in market_data.columns:
            features.append(market_data['vix'].values)
        
        if 'term_structure_slope' in market_data.columns:
            features.append(market_data['term_structure_slope'].values)
        
        # Technical indicators (if available)
        technical_features = ['rsi', 'macd', 'bollinger_position', 'volume_ratio']
        for feature in technical_features:
            if feature in market_data.columns:
                features.append(market_data[feature].values)
        
        # Stack all features
        if features:
            feature_matrix = np.column_stack(features)
            
            # Store feature names for reference
            self.feature_names = (basic_features + 
                                ['moneyness', 'log_moneyness', 'time_to_exp', 
                                 'sqrt_time', 'inv_time', 'volatility', 'variance', 
                                 'log_vol'] + 
                                [f for f in ['vix', 'term_structure_slope'] + technical_features 
                                 if f in market_data.columns])
            
            return feature_matrix
        else:
            raise ValueError("No valid features found in market data")
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'option_price',
              validation_split: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the neural network on market data.
        
        Args:
            training_data: DataFrame with features and target prices
            target_column: Name of the target price column
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Training metrics dictionary
        """
        # Prepare features and target
        X = self.prepare_features(training_data)
        y = training_data[target_column].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=random_state
        )
        
        # Scale features and target
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        
        # Initialize and train model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation=self.activation,
            solver=self.solver,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20
        )
        
        # Train model
        self.model.fit(X_train_scaled, y_train_scaled)
        
        # Make predictions
        y_train_pred_scaled = self.model.predict(X_train_scaled)
        y_val_pred_scaled = self.model.predict(X_val_scaled)
        
        # Inverse transform predictions
        y_train_pred = self.scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()
        y_val_pred = self.scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()
        
        # Calculate metrics
        train_metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred),
            'training_loss': self.model.loss_,
            'n_iterations': self.model.n_iter_
        }
        
        self.is_trained = True
        return train_metrics
    
    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Predict option prices using trained neural network.
        
        Args:
            market_data: DataFrame with market features
            
        Returns:
            Array of predicted option prices
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X = self.prepare_features(market_data)
        X_scaled = self.scaler_X.transform(X)
        
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        return y_pred
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (approximated using permutation importance).
        """
        if not self.is_trained or self.feature_names is None:
            raise ValueError("Model must be trained to get feature importance")
        
        # Note: MLPRegressor doesn't have built-in feature importance
        # This is a simplified placeholder - real implementation would use
        # permutation importance or SHAP values
        
        # For demonstration, return uniform importance
        n_features = len(self.feature_names)
        importance = {name: 1.0/n_features for name in self.feature_names}
        
        return importance
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'hidden_layers': self.hidden_layers,
                'activation': self.activation,
                'solver': self.solver,
                'learning_rate': self.learning_rate,
                'max_iter': self.max_iter
            }
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        self.feature_names = model_data['feature_names']
        
        # Load hyperparameters
        hyperparams = model_data['hyperparameters']
        self.hidden_layers = hyperparams['hidden_layers']
        self.activation = hyperparams['activation']
        self.solver = hyperparams['solver']
        self.learning_rate = hyperparams['learning_rate']
        self.max_iter = hyperparams['max_iter']
        
        self.is_trained = True

class EnsembleOptionPricer:
    """
    Ensemble model combining multiple ML algorithms for robust option pricing.
    """
    
    def __init__(self, models: Optional[List[str]] = None):
        """
        Initialize ensemble option pricer.
        
        Args:
            models: List of model types to include in ensemble
        """
        if models is None:
            models = ['neural_network', 'random_forest', 'gradient_boosting']
        
        self.models = {}
        self.scalers = {}
        self.weights = {}
        self.is_trained = False
        
        # Initialize models
        for model_name in models:
            if model_name == 'neural_network':
                self.models[model_name] = NeuralNetworkPricer()
            elif model_name == 'random_forest':
                self.models[model_name] = RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=42
                )
                self.scalers[model_name] = StandardScaler()
            elif model_name == 'gradient_boosting':
                self.models[model_name] = GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
                )
                self.scalers[model_name] = StandardScaler()
    
    def train(self, training_data: pd.DataFrame, target_column: str = 'option_price',
              validation_split: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.
        
        Args:
            training_data: DataFrame with features and target prices
            target_column: Name of the target price column
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training metrics for each model
        """
        metrics = {}
        
        # Split data once for all models
        if 'neural_network' in self.models:
            # Use neural network feature preparation
            X = self.models['neural_network'].prepare_features(training_data)
        else:
            # Basic feature preparation for tree-based models
            basic_features = ['S', 'K', 'T', 'r', 'sigma']
            X = training_data[basic_features].values
        
        y = training_data[target_column].values
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Train each model
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network':
                    # Neural network has its own training method
                    model_metrics = model.train(training_data, target_column, validation_split)
                    metrics[model_name] = model_metrics
                    
                    # Get validation predictions for weight calculation
                    val_data = training_data.iloc[X_val.shape[0]:]  # Approximate
                    val_pred = model.predict(val_data)
                    
                else:
                    # Sklearn models
                    scaler = self.scalers[model_name]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train_scaled)
                    val_pred = model.predict(X_val_scaled)
                    
                    # Calculate metrics
                    model_metrics = {
                        'train_mse': mean_squared_error(y_train, y_train_pred),
                        'train_mae': mean_absolute_error(y_train, y_train_pred),
                        'train_r2': r2_score(y_train, y_train_pred),
                        'val_mse': mean_squared_error(y_val, val_pred),
                        'val_mae': mean_absolute_error(y_val, val_pred),
                        'val_r2': r2_score(y_val, val_pred)
                    }
                    metrics[model_name] = model_metrics
                
                # Calculate model weight based on validation R²
                val_r2 = model_metrics.get('val_r2', 0)
                self.weights[model_name] = max(val_r2, 0.1)  # Minimum weight of 0.1
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                self.weights[model_name] = 0.1
                metrics[model_name] = {'error': str(e)}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.is_trained = True
        return metrics
    
    def predict(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        Predict option prices using ensemble of models.
        
        Args:
            market_data: DataFrame with market features
            
        Returns:
            Array of predicted option prices
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                if model_name == 'neural_network':
                    pred = model.predict(market_data)
                else:
                    # Sklearn models
                    if model_name == 'neural_network':
                        X = model.prepare_features(market_data)
                    else:
                        basic_features = ['S', 'K', 'T', 'r', 'sigma']
                        X = market_data[basic_features].values
                    
                    scaler = self.scalers[model_name]
                    X_scaled = scaler.transform(X)
                    pred = model.predict(X_scaled)
                
                predictions[model_name] = pred
                
            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
                # Use zero prediction if model fails
                predictions[model_name] = np.zeros(len(market_data))
        
        # Weighted ensemble prediction
        final_prediction = np.zeros(len(market_data))
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            final_prediction += weight * pred
        
        return final_prediction

class VolatilityPredictor:
    """
    Machine learning model for volatility prediction and forecasting.
    """
    
    def __init__(self, model_type: str = 'gradient_boosting'):
        """
        Initialize volatility predictor.
        
        Args:
            model_type: Type of ML model ('gradient_boosting', 'random_forest', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=42
            )
        elif model_type == 'neural_network':
            self.model = MLPRegressor(
                hidden_layer_sizes=(50, 25), activation='relu', 
                solver='adam', max_iter=1000, random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_volatility_features(self, price_data: pd.DataFrame, 
                                  lookback_periods: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """
        Prepare features for volatility prediction.
        
        Args:
            price_data: DataFrame with price data (must have 'close' column)
            lookback_periods: List of lookback periods for feature calculation
            
        Returns:
            DataFrame with volatility prediction features
        """
        features_df = price_data.copy()
        
        # Calculate returns
        features_df['returns'] = price_data['close'].pct_change()
        
        # Historical volatility features
        for period in lookback_periods:
            features_df[f'vol_{period}d'] = features_df['returns'].rolling(period).std() * np.sqrt(252)
            features_df[f'return_mean_{period}d'] = features_df['returns'].rolling(period).mean()
            features_df[f'return_skew_{period}d'] = features_df['returns'].rolling(period).skew()
            features_df[f'return_kurt_{period}d'] = features_df['returns'].rolling(period).kurt()
        
        # Range-based volatility estimators
        if all(col in price_data.columns for col in ['high', 'low', 'open']):
            # Parkinson volatility
            features_df['parkinson_vol'] = np.sqrt(
                252 * 0.25 * np.log(price_data['high'] / price_data['low'])**2
            )
            
            # Garman-Klass volatility
            features_df['gk_vol'] = np.sqrt(
                252 * (0.5 * np.log(price_data['high'] / price_data['low'])**2 - 
                       (2*np.log(2) - 1) * np.log(price_data['close'] / price_data['open'])**2)
            )
        
        # Technical indicators
        if 'volume' in price_data.columns:
            # Volume-based features
            for period in [10, 20]:
                features_df[f'volume_ratio_{period}d'] = (
                    price_data['volume'] / price_data['volume'].rolling(period).mean()
                )
        
        # Market microstructure features (if available)
        if 'bid' in price_data.columns and 'ask' in price_data.columns:
            features_df['bid_ask_spread'] = (price_data['ask'] - price_data['bid']) / price_data['close']
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        return features_df
    
    def train(self, price_data: pd.DataFrame, target_column: str = 'realized_vol',
              validation_split: float = 0.2) -> Dict[str, float]:
        """
        Train volatility prediction model.
        
        Args:
            price_data: DataFrame with price data and target volatility
            target_column: Name of target volatility column
            validation_split: Fraction of data for validation
            
        Returns:
            Training metrics
        """
        # Prepare features
        features_df = self.prepare_volatility_features(price_data)
        
        # Select feature columns (exclude non-feature columns)
        exclude_cols = ['close', 'high', 'low', 'open', 'volume', 'returns', 'bid', 'ask', target_column]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        y = features_df[target_column].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_train_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'val_mse': mean_squared_error(y_val, y_val_pred),
            'val_mae': mean_absolute_error(y_val, y_val_pred),
            'val_r2': r2_score(y_val, y_val_pred)
        }
        
        self.is_trained = True
        return metrics
    
    def predict(self, price_data: pd.DataFrame) -> np.ndarray:
        """
        Predict future volatility.
        
        Args:
            price_data: DataFrame with recent price data
            
        Returns:
            Array of predicted volatilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        features_df = self.prepare_volatility_features(price_data)
        
        # Select feature columns
        exclude_cols = ['close', 'high', 'low', 'open', 'volume', 'returns', 'bid', 'ask']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        X = features_df[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

# Example usage and testing functions
def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create sample option market data for testing ML models.
    """
    np.random.seed(42)
    
    # Generate basic parameters
    S = np.random.uniform(80, 120, n_samples)
    K = np.random.uniform(85, 115, n_samples)
    T = np.random.uniform(0.02, 2.0, n_samples)  # 1 week to 2 years
    r = np.random.uniform(0.01, 0.06, n_samples)
    sigma = np.random.uniform(0.1, 0.5, n_samples)
    
    # Calculate theoretical Black-Scholes prices with some noise
    from .option_pricing import black_scholes
    
    theoretical_prices = []
    for i in range(n_samples):
        try:
            price = black_scholes(S[i], K[i], T[i], r[i], sigma[i], 'call')
            # Add some market noise
            noise = np.random.normal(0, 0.02 * price)
            theoretical_prices.append(price + noise)
        except:
            theoretical_prices.append(np.nan)
    
    # Create additional market features
    vix = np.random.uniform(15, 40, n_samples)
    term_structure_slope = np.random.uniform(-0.02, 0.03, n_samples)
    
    data = pd.DataFrame({
        'S': S,
        'K': K, 
        'T': T,
        'r': r,
        'sigma': sigma,
        'option_price': theoretical_prices,
        'vix': vix,
        'term_structure_slope': term_structure_slope
    })
    
    return data.dropna()

if __name__ == "__main__":
    # Test neural network pricer
    print("Testing Machine Learning Option Pricing Models...")
    
    # Create sample data
    sample_data = create_sample_data(5000)
    print(f"Created sample dataset with {len(sample_data)} options")
    
    # Test Neural Network Pricer
    print("\n1. Testing Neural Network Pricer...")
    nn_pricer = NeuralNetworkPricer()
    nn_metrics = nn_pricer.train(sample_data)
    
    print("Neural Network Training Metrics:")
    for metric, value in nn_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test Ensemble Pricer
    print("\n2. Testing Ensemble Pricer...")
    ensemble_pricer = EnsembleOptionPricer()
    ensemble_metrics = ensemble_pricer.train(sample_data)
    
    print("Ensemble Training Metrics:")
    for model_name, metrics in ensemble_metrics.items():
        print(f"  {model_name}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")
    
    print(f"\nEnsemble Weights: {ensemble_pricer.weights}")
    
    # Test predictions
    test_sample = sample_data.head(10)
    nn_predictions = nn_pricer.predict(test_sample)
    ensemble_predictions = ensemble_pricer.predict(test_sample)
    
    print("\nSample Predictions Comparison:")
    print("Actual\t\tNeural Net\tEnsemble")
    for i in range(len(test_sample)):
        actual = test_sample.iloc[i]['option_price']
        nn_pred = nn_predictions[i]
        ens_pred = ensemble_predictions[i]
        print(f"{actual:.4f}\t\t{nn_pred:.4f}\t\t{ens_pred:.4f}")
