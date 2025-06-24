"""
Market Data Integration Module

Real-time and historical market data integration for:
- Stock prices and option chains
- Volatility surfaces and term structures
- Risk-free rates and dividend yields
- Market sentiment indicators
- Data validation and quality checks
"""

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class MarketDataProvider:
    """Unified market data interface"""
    
    def __init__(self, provider: str = 'yahoo', api_key: Optional[str] = None):
        self.provider = provider.lower()
        self.api_key = api_key
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def get_stock_price(self, symbol: str, real_time: bool = True) -> Dict:
        """Get current stock price and basic metrics"""
        cache_key = f"stock_{symbol}"
        
        if not real_time and cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_duration:
                return data
        
        try:
            if self.provider == 'yahoo':
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d", interval="1m")
                
                if len(hist) > 0:
                    current_price = hist['Close'].iloc[-1]
                    volume = hist['Volume'].iloc[-1]
                    day_change = current_price - hist['Open'].iloc[0]
                    day_change_pct = (day_change / hist['Open'].iloc[0]) * 100
                else:
                    current_price = info.get('currentPrice', 0)
                    volume = info.get('volume', 0)
                    day_change = info.get('regularMarketChange', 0)
                    day_change_pct = info.get('regularMarketChangePercent', 0)
                
                data = {
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume,
                    'day_change': day_change,
                    'day_change_pct': day_change_pct,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 1.0),
                    'timestamp': datetime.now()
                }
                
                self.cache[cache_key] = (time.time(), data)
                return data
                
        except Exception as e:
            return {'error': f"Failed to fetch data for {symbol}: {str(e)}"}
    
    def get_historical_data(self, symbol: str, period: str = "1y",
                          interval: str = "1d") -> pd.DataFrame:
        """Get historical price data"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            # Add calculated fields
            hist['Returns'] = hist['Close'].pct_change()
            hist['Log_Returns'] = np.log(hist['Close'] / hist['Close'].shift(1))
            hist['Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)
            
            return hist
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_option_chain(self, symbol: str, expiry_date: Optional[str] = None) -> Dict:
        """Get option chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            if expiry_date:
                options = ticker.option_chain(expiry_date)
            else:
                # Get nearest expiry
                expiry_dates = ticker.options
                if expiry_dates:
                    options = ticker.option_chain(expiry_dates[0])
                else:
                    return {'error': 'No option data available'}
            
            calls = options.calls
            puts = options.puts
            
            # Add derived metrics
            calls['mid_price'] = (calls['bid'] + calls['ask']) / 2
            puts['mid_price'] = (puts['bid'] + puts['ask']) / 2
            calls['bid_ask_spread'] = calls['ask'] - calls['bid']
            puts['bid_ask_spread'] = puts['ask'] - puts['bid']
            
            return {
                'calls': calls,
                'puts': puts,
                'expiry': expiry_date or expiry_dates[0],
                'underlying_price': self.get_stock_price(symbol)['price']
            }
            
        except Exception as e:
            return {'error': f"Failed to fetch option chain for {symbol}: {str(e)}"}
    
    def get_volatility_surface(self, symbol: str) -> Dict:
        """Construct implied volatility surface"""
        try:
            ticker = yf.Ticker(symbol)
            underlying_price = self.get_stock_price(symbol)['price']
            
            vol_surface = []
            
            for expiry in ticker.options[:6]:  # Get first 6 expiries
                option_chain = ticker.option_chain(expiry)
                calls = option_chain.calls
                puts = option_chain.puts
                
                # Calculate time to expiry
                expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
                tte = (expiry_date - datetime.now()).days / 365.0
                
                # Process calls
                for _, call in calls.iterrows():
                    if call['impliedVolatility'] > 0 and call['volume'] > 0:
                        moneyness = call['strike'] / underlying_price
                        vol_surface.append({
                            'expiry': expiry,
                            'tte': tte,
                            'strike': call['strike'],
                            'moneyness': moneyness,
                            'option_type': 'call',
                            'implied_vol': call['impliedVolatility'],
                            'volume': call['volume'],
                            'open_interest': call['openInterest']
                        })
                
                # Process puts
                for _, put in puts.iterrows():
                    if put['impliedVolatility'] > 0 and put['volume'] > 0:
                        moneyness = put['strike'] / underlying_price
                        vol_surface.append({
                            'expiry': expiry,
                            'tte': tte,
                            'strike': put['strike'],
                            'moneyness': moneyness,
                            'option_type': 'put',
                            'implied_vol': put['impliedVolatility'],
                            'volume': put['volume'],
                            'open_interest': put['openInterest']
                        })
            
            vol_df = pd.DataFrame(vol_surface)
            
            return {
                'volatility_surface': vol_df,
                'underlying_price': underlying_price,
                'surface_summary': {
                    'min_vol': vol_df['implied_vol'].min(),
                    'max_vol': vol_df['implied_vol'].max(),
                    'avg_vol': vol_df['implied_vol'].mean(),
                    'total_points': len(vol_df)
                }
            }
            
        except Exception as e:
            return {'error': f"Failed to construct volatility surface for {symbol}: {str(e)}"}


class VolatilityEstimator:
    """Advanced volatility estimation methods"""
    
    @staticmethod
    def historical_volatility(prices: pd.Series, window: int = 30,
                            method: str = 'simple') -> pd.Series:
        """Calculate historical volatility using various methods"""
        returns = prices.pct_change().dropna()
        
        if method == 'simple':
            # Simple rolling standard deviation
            vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        elif method == 'ewma':
            # Exponentially weighted moving average
            vol = returns.ewm(span=window).std() * np.sqrt(252)
        
        elif method == 'garch':
            # GARCH(1,1) - simplified implementation
            try:
                from arch import arch_model
                model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                fitted = model.fit(disp='off')
                vol = fitted.conditional_volatility / 100 * np.sqrt(252)
            except:
                # Fallback to simple method
                vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        else:
            raise ValueError("Method must be 'simple', 'ewma', or 'garch'")
        
        return vol
    
    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series,
                           window: int = 30) -> pd.Series:
        """Parkinson volatility estimator using high-low prices"""
        log_hl = np.log(high / low)
        parkinson_var = log_hl.rolling(window=window).apply(
            lambda x: np.sum(x**2) / (4 * np.log(2) * len(x))
        )
        return np.sqrt(parkinson_var * 252)
    
    @staticmethod
    def garman_klass_volatility(high: pd.Series, low: pd.Series,
                              open_price: pd.Series, close: pd.Series,
                              window: int = 30) -> pd.Series:
        """Garman-Klass volatility estimator"""
        log_hl = (np.log(high) - np.log(low))**2
        log_cc = (np.log(close) - np.log(open_price))**2
        
        gk_var = log_hl.rolling(window=window).mean() - \
                 (2 * np.log(2) - 1) * log_cc.rolling(window=window).mean()
        
        return np.sqrt(gk_var * 252)
    
    @staticmethod
    def realized_volatility(returns: pd.Series, frequency: str = 'daily') -> float:
        """Calculate realized volatility"""
        if frequency == 'daily':
            scale = 252
        elif frequency == 'hourly':
            scale = 252 * 24
        elif frequency == 'minute':
            scale = 252 * 24 * 60
        else:
            scale = 252
        
        return np.sqrt(np.sum(returns**2) * scale)


class RiskFreeRateProvider:
    """Risk-free rate data provider"""
    
    def __init__(self):
        self.fed_rates_url = "https://api.stlouisfed.org/fred/series/observations"
        self.treasury_symbols = {
            '1M': '^IRX',
            '3M': '^IRX', 
            '6M': '^IRX',
            '1Y': '^TNX',
            '2Y': '^TNX',
            '5Y': '^FVX',
            '10Y': '^TNX',
            '30Y': '^TYX'
        }
    
    def get_treasury_rates(self) -> Dict:
        """Get current US Treasury rates"""
        rates = {}
        
        try:
            # Use Yahoo Finance for treasury rates
            for period, symbol in self.treasury_symbols.items():
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                if len(hist) > 0:
                    current_rate = hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
                    rates[period] = current_rate
        except Exception as e:
            print(f"Error fetching treasury rates: {str(e)}")
            # Fallback rates
            rates = {
                '1M': 0.05,
                '3M': 0.052,
                '6M': 0.054,
                '1Y': 0.056,
                '2Y': 0.058,
                '5Y': 0.060,
                '10Y': 0.062,
                '30Y': 0.064
            }
        
        return rates
    
    def interpolate_rate(self, time_to_maturity: float) -> float:
        """Interpolate risk-free rate for specific maturity"""
        rates = self.get_treasury_rates()
        
        # Define maturity points in years
        maturities = {
            '1M': 1/12,
            '3M': 3/12,
            '6M': 6/12,
            '1Y': 1,
            '2Y': 2,
            '5Y': 5,
            '10Y': 10,
            '30Y': 30
        }
        
        # Create arrays for interpolation
        times = []
        rate_values = []
        
        for period, rate in rates.items():
            if period in maturities:
                times.append(maturities[period])
                rate_values.append(rate)
        
        # Linear interpolation
        if time_to_maturity <= min(times):
            return rate_values[0]
        elif time_to_maturity >= max(times):
            return rate_values[-1]
        else:
            return np.interp(time_to_maturity, times, rate_values)


class MarketSentimentIndicators:
    """Market sentiment and fear/greed indicators"""
    
    def __init__(self):
        self.vix_symbol = '^VIX'
        self.put_call_symbols = ['SPY']
    
    def get_vix_data(self) -> Dict:
        """Get VIX (volatility index) data"""
        try:
            vix = yf.Ticker(self.vix_symbol)
            hist = vix.history(period="1d")
            
            if len(hist) > 0:
                current_vix = hist['Close'].iloc[-1]
                
                # VIX interpretation
                if current_vix < 12:
                    sentiment = "Very Low Volatility (Complacency)"
                elif current_vix < 20:
                    sentiment = "Low Volatility (Calm)"
                elif current_vix < 30:
                    sentiment = "Normal Volatility"
                elif current_vix < 40:
                    sentiment = "High Volatility (Fear)"
                else:
                    sentiment = "Very High Volatility (Panic)"
                
                return {
                    'vix_level': current_vix,
                    'sentiment': sentiment,
                    'fear_greed_score': max(0, min(100, 100 - (current_vix - 10) * 2))
                }
            
        except Exception as e:
            return {'error': f"Failed to fetch VIX data: {str(e)}"}
    
    def get_put_call_ratio(self, symbol: str = 'SPY') -> Dict:
        """Calculate put/call ratio from option chain"""
        try:
            market_data = MarketDataProvider()
            option_chain = market_data.get_option_chain(symbol)
            
            if 'error' not in option_chain:
                calls = option_chain['calls']
                puts = option_chain['puts']
                
                call_volume = calls['volume'].sum()
                put_volume = puts['volume'].sum()
                
                put_call_ratio = put_volume / call_volume if call_volume > 0 else 0
                
                # Interpretation
                if put_call_ratio > 1.0:
                    sentiment = "Bearish (More put buying)"
                elif put_call_ratio > 0.7:
                    sentiment = "Neutral to Bearish"
                elif put_call_ratio > 0.5:
                    sentiment = "Neutral"
                else:
                    sentiment = "Bullish (More call buying)"
                
                return {
                    'put_call_ratio': put_call_ratio,
                    'sentiment': sentiment,
                    'call_volume': call_volume,
                    'put_volume': put_volume
                }
            
        except Exception as e:
            return {'error': f"Failed to calculate put/call ratio: {str(e)}"}


class DataQualityChecker:
    """Data quality and validation tools"""
    
    @staticmethod
    def check_price_data_quality(data: pd.DataFrame) -> Dict:
        """Check quality of price data"""
        quality_report = {
            'total_observations': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'zero_values': (data == 0).sum().to_dict(),
            'negative_prices': (data < 0).sum().to_dict() if 'Close' in data.columns else {},
            'outliers': {},
            'data_gaps': [],
            'quality_score': 100
        }
        
        # Check for outliers (prices more than 3 standard deviations)
        if 'Close' in data.columns:
            returns = data['Close'].pct_change().dropna()
            outlier_threshold = 3 * returns.std()
            outliers = returns[abs(returns) > outlier_threshold]
            quality_report['outliers'] = {
                'count': len(outliers),
                'dates': outliers.index.tolist(),
                'values': outliers.tolist()
            }
        
        # Check for data gaps (missing trading days)
        if not data.empty and hasattr(data.index, 'date'):
            date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
            missing_dates = date_range.difference(data.index)
            quality_report['data_gaps'] = missing_dates.tolist()
        
        # Calculate quality score
        missing_penalty = sum(quality_report['missing_values'].values()) * 2
        outlier_penalty = quality_report['outliers'].get('count', 0) * 5
        gap_penalty = len(quality_report['data_gaps']) * 1
        
        quality_report['quality_score'] = max(0, 100 - missing_penalty - outlier_penalty - gap_penalty)
        
        return quality_report
    
    @staticmethod
    def validate_option_data(option_chain: pd.DataFrame) -> Dict:
        """Validate option chain data"""
        validation_report = {
            'total_contracts': len(option_chain),
            'valid_prices': 0,
            'valid_volumes': 0,
            'valid_iv': 0,
            'arbitrage_violations': [],
            'suspicious_spreads': [],
            'data_quality': 'Good'
        }
        
        if len(option_chain) == 0:
            validation_report['data_quality'] = 'No Data'
            return validation_report
        
        # Check for valid prices
        valid_prices = (option_chain['bid'] >= 0) & (option_chain['ask'] >= option_chain['bid'])
        validation_report['valid_prices'] = valid_prices.sum()
        
        # Check for valid volumes
        if 'volume' in option_chain.columns:
            validation_report['valid_volumes'] = (option_chain['volume'] >= 0).sum()
        
        # Check for valid implied volatility
        if 'impliedVolatility' in option_chain.columns:
            valid_iv = (option_chain['impliedVolatility'] > 0) & (option_chain['impliedVolatility'] < 5)
            validation_report['valid_iv'] = valid_iv.sum()
        
        # Check for arbitrage violations (bid > theoretical value)
        # Simplified check: bid should not be greater than intrinsic value for ITM options
        
        # Check for suspicious bid-ask spreads
        if 'bid' in option_chain.columns and 'ask' in option_chain.columns:
            spreads = option_chain['ask'] - option_chain['bid']
            mid_prices = (option_chain['ask'] + option_chain['bid']) / 2
            relative_spreads = spreads / mid_prices
            
            suspicious = relative_spreads > 0.5  # Spread > 50% of mid price
            validation_report['suspicious_spreads'] = option_chain[suspicious].index.tolist()
        
        # Overall quality assessment
        valid_ratio = validation_report['valid_prices'] / len(option_chain)
        if valid_ratio > 0.95:
            validation_report['data_quality'] = 'Excellent'
        elif valid_ratio > 0.85:
            validation_report['data_quality'] = 'Good'
        elif valid_ratio > 0.7:
            validation_report['data_quality'] = 'Fair'
        else:
            validation_report['data_quality'] = 'Poor'
        
        return validation_report


# Example usage
if __name__ == "__main__":
    # Initialize market data provider
    market_data = MarketDataProvider()
    
    # Get stock price
    stock_data = market_data.get_stock_price('AAPL')
    print(f"AAPL Price: ${stock_data['price']:.2f}")
    
    # Get option chain
    options = market_data.get_option_chain('AAPL')
    if 'error' not in options:
        print(f"Found {len(options['calls'])} call options and {len(options['puts'])} put options")
    
    # Get volatility surface
    vol_surface = market_data.get_volatility_surface('AAPL')
    if 'error' not in vol_surface:
        print(f"Volatility surface has {vol_surface['surface_summary']['total_points']} data points")
    
    # Get VIX sentiment
    sentiment = MarketSentimentIndicators()
    vix_data = sentiment.get_vix_data()
    if 'error' not in vix_data:
        print(f"VIX: {vix_data['vix_level']:.2f} - {vix_data['sentiment']}")
    
    # Get risk-free rates
    rates = RiskFreeRateProvider()
    treasury_rates = rates.get_treasury_rates()
    print(f"10Y Treasury Rate: {treasury_rates.get('10Y', 0)*100:.2f}%")
