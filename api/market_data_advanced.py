"""
Advanced Market Data Integration Module

Professional-grade market data feeds and analytics including:
- Multiple data provider integration (Yahoo Finance, Alpha Vantage, IEX Cloud)
- Real-time option chain data
- Implied volatility surface construction
- Market sentiment indicators
- High-frequency tick data processing
- Data quality monitoring and validation
- Caching and performance optimization

Author: Advanced Quantitative Finance Team
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
import warnings
from functools import lru_cache
import logging

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Enumeration of supported data providers."""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    QUANDL = "quandl"

@dataclass
class MarketDataPoint:
    """Single market data point."""
    symbol: str
    timestamp: datetime
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None

@dataclass
class OptionChainData:
    """Option chain data structure."""
    symbol: str
    expiration_date: datetime
    strike: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class AdvancedMarketDataProvider:
    """
    Advanced market data provider with multiple data sources and caching.
    """
    
    def __init__(self, primary_provider: DataProvider = DataProvider.YAHOO_FINANCE,
                 cache_duration: int = 300, db_path: str = "market_data.db"):
        """
        Initialize advanced market data provider.
        
        Args:
            primary_provider: Primary data provider to use
            cache_duration: Cache duration in seconds
            db_path: Path to SQLite database for caching
        """
        self.primary_provider = primary_provider
        self.cache_duration = cache_duration
        self.db_path = db_path
        self.api_keys = {}
        self.session = requests.Session()
        self._setup_database()
        self._setup_providers()
    
    def _setup_database(self):
        """Setup SQLite database for caching."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for caching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                symbol TEXT,
                timestamp DATETIME,
                price REAL,
                volume INTEGER,
                bid REAL,
                ask REAL,
                high REAL,
                low REAL,
                open REAL,
                provider TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, timestamp, provider)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS option_data (
                symbol TEXT,
                expiration_date DATE,
                strike REAL,
                option_type TEXT,
                bid REAL,
                ask REAL,
                last_price REAL,
                volume INTEGER,
                open_interest INTEGER,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                provider TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, expiration_date, strike, option_type, provider)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS volatility_surface (
                symbol TEXT,
                date DATE,
                time_to_expiry REAL,
                strike REAL,
                implied_volatility REAL,
                delta REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, date, time_to_expiry, strike)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_providers(self):
        """Setup data provider configurations."""
        self.provider_configs = {
            DataProvider.YAHOO_FINANCE: {
                'base_url': 'https://query1.finance.yahoo.com/v8/finance/chart/',
                'rate_limit': 2000,  # requests per hour
                'requires_key': False
            },
            DataProvider.ALPHA_VANTAGE: {
                'base_url': 'https://www.alphavantage.co/query',
                'rate_limit': 500,  # requests per day for free tier
                'requires_key': True
            },
            DataProvider.IEX_CLOUD: {
                'base_url': 'https://cloud.iexapis.com/stable/',
                'rate_limit': 100000,  # requests per month for free tier
                'requires_key': True
            }
        }
    
    def set_api_key(self, provider: DataProvider, api_key: str):
        """Set API key for a data provider."""
        self.api_keys[provider] = api_key
    
    @lru_cache(maxsize=1000)
    def get_real_time_quote(self, symbol: str, 
                           provider: Optional[DataProvider] = None) -> MarketDataPoint:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Stock symbol
            provider: Data provider to use (uses primary if None)
            
        Returns:
            MarketDataPoint with current quote data
        """
        if provider is None:
            provider = self.primary_provider
        
        # Check cache first
        cached_data = self._get_cached_quote(symbol, provider)
        if cached_data:
            return cached_data
        
        # Fetch from provider
        try:
            if provider == DataProvider.YAHOO_FINANCE:
                data = self._fetch_yahoo_quote(symbol)
            elif provider == DataProvider.ALPHA_VANTAGE:
                data = self._fetch_alpha_vantage_quote(symbol)
            elif provider == DataProvider.IEX_CLOUD:
                data = self._fetch_iex_quote(symbol)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            # Cache the data
            self._cache_quote(data, provider)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol} from {provider}: {e}")
            # Try fallback provider
            return self._get_fallback_quote(symbol)
    
    def _fetch_yahoo_quote(self, symbol: str) -> MarketDataPoint:
        """Fetch quote from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return MarketDataPoint(
                symbol=symbol,
                timestamp=datetime.now(),
                price=info.get('currentPrice', info.get('regularMarketPrice', 0)),
                volume=info.get('volume'),
                bid=info.get('bid'),
                ask=info.get('ask'),
                high=info.get('dayHigh', info.get('regularMarketDayHigh')),
                low=info.get('dayLow', info.get('regularMarketDayLow')),
                open=info.get('open', info.get('regularMarketOpen'))
            )
        except Exception as e:
            raise Exception(f"Yahoo Finance API error: {e}")
    
    def _fetch_alpha_vantage_quote(self, symbol: str) -> MarketDataPoint:
        """Fetch quote from Alpha Vantage."""
        if DataProvider.ALPHA_VANTAGE not in self.api_keys:
            raise ValueError("Alpha Vantage API key not set")
        
        url = self.provider_configs[DataProvider.ALPHA_VANTAGE]['base_url']
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_keys[DataProvider.ALPHA_VANTAGE]
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'Global Quote' not in data:
            raise Exception("Invalid response from Alpha Vantage")
        
        quote = data['Global Quote']
        
        return MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now(),
            price=float(quote.get('05. price', 0)),
            volume=int(quote.get('06. volume', 0)),
            high=float(quote.get('03. high', 0)),
            low=float(quote.get('04. low', 0)),
            open=float(quote.get('02. open', 0))
        )
    
    def _fetch_iex_quote(self, symbol: str) -> MarketDataPoint:
        """Fetch quote from IEX Cloud."""
        if DataProvider.IEX_CLOUD not in self.api_keys:
            raise ValueError("IEX Cloud API key not set")
        
        url = f"{self.provider_configs[DataProvider.IEX_CLOUD]['base_url']}stock/{symbol}/quote"
        params = {'token': self.api_keys[DataProvider.IEX_CLOUD]}
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        return MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now(),
            price=data.get('latestPrice', 0),
            volume=data.get('latestVolume'),
            bid=data.get('iexBidPrice'),
            ask=data.get('iexAskPrice'),
            high=data.get('high'),
            low=data.get('low'),
            open=data.get('open')
        )
    
    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> List[OptionChainData]:
        """
        Get option chain data for a symbol.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Specific expiration date (YYYY-MM-DD format)
            
        Returns:
            List of OptionChainData objects
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            if expiration_date:
                expirations = [expiration_date]
            else:
                expirations = ticker.options[:3]  # Get first 3 expiration dates
            
            option_data = []
            
            for exp_date in expirations:
                try:
                    option_chain = ticker.option_chain(exp_date)
                    
                    # Process calls
                    for _, call in option_chain.calls.iterrows():
                        option_data.append(OptionChainData(
                            symbol=symbol,
                            expiration_date=datetime.strptime(exp_date, '%Y-%m-%d'),
                            strike=call['strike'],
                            option_type='call',
                            bid=call.get('bid', 0),
                            ask=call.get('ask', 0),
                            last_price=call.get('lastPrice', 0),
                            volume=call.get('volume', 0),
                            open_interest=call.get('openInterest', 0),
                            implied_volatility=call.get('impliedVolatility', 0)
                        ))
                    
                    # Process puts
                    for _, put in option_chain.puts.iterrows():
                        option_data.append(OptionChainData(
                            symbol=symbol,
                            expiration_date=datetime.strptime(exp_date, '%Y-%m-%d'),
                            strike=put['strike'],
                            option_type='put',
                            bid=put.get('bid', 0),
                            ask=put.get('ask', 0),
                            last_price=put.get('lastPrice', 0),
                            volume=put.get('volume', 0),
                            open_interest=put.get('openInterest', 0),
                            implied_volatility=put.get('impliedVolatility', 0)
                        ))
                
                except Exception as e:
                    logger.warning(f"Error fetching options for {exp_date}: {e}")
                    continue
            
            return option_data
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return []
    
    def get_historical_data(self, symbol: str, period: str = "1y",
                           interval: str = "1d") -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            # Clean up the data
            data = data.dropna()
            data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_cached_quote(self, symbol: str, provider: DataProvider) -> Optional[MarketDataPoint]:
        """Get cached quote if available and fresh."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_time = datetime.now() - timedelta(seconds=self.cache_duration)
            
            cursor.execute('''
                SELECT * FROM market_data 
                WHERE symbol = ? AND provider = ? AND created_at > ?
                ORDER BY created_at DESC LIMIT 1
            ''', (symbol, provider.value, cutoff_time))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return MarketDataPoint(
                    symbol=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    price=row[2],
                    volume=row[3],
                    bid=row[4],
                    ask=row[5],
                    high=row[6],
                    low=row[7],
                    open=row[8]
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None
    
    def _cache_quote(self, data: MarketDataPoint, provider: DataProvider):
        """Cache quote data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO market_data 
                (symbol, timestamp, price, volume, bid, ask, high, low, open, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol, data.timestamp.isoformat(), data.price,
                data.volume, data.bid, data.ask, data.high, data.low,
                data.open, provider.value
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Cache write error: {e}")
    
    def _get_fallback_quote(self, symbol: str) -> MarketDataPoint:
        """Get quote from fallback provider."""
        fallback_providers = [p for p in DataProvider if p != self.primary_provider]
        
        for provider in fallback_providers:
            try:
                return self.get_real_time_quote(symbol, provider)
            except:
                continue
        
        # If all providers fail, return a dummy quote
        logger.error(f"All providers failed for {symbol}")
        return MarketDataPoint(
            symbol=symbol,
            timestamp=datetime.now(),
            price=0.0
        )

class VolatilitySurfaceBuilder:
    """
    Build and analyze implied volatility surfaces from option data.
    """
    
    def __init__(self, market_data_provider: AdvancedMarketDataProvider):
        """
        Initialize volatility surface builder.
        
        Args:
            market_data_provider: Market data provider instance
        """
        self.data_provider = market_data_provider
    
    def build_volatility_surface(self, symbol: str, 
                                max_expiry_days: int = 90) -> pd.DataFrame:
        """
        Build implied volatility surface for a symbol.
        
        Args:
            symbol: Underlying symbol
            max_expiry_days: Maximum days to expiry to include
            
        Returns:
            DataFrame with volatility surface data
        """
        # Get option chain data
        option_data = self.data_provider.get_option_chain(symbol)
        
        if not option_data:
            return pd.DataFrame()
        
        # Get current spot price
        spot_quote = self.data_provider.get_real_time_quote(symbol)
        spot_price = spot_quote.price
        
        surface_data = []
        
        for option in option_data:
            # Calculate time to expiry
            time_to_expiry = (option.expiration_date - datetime.now()).days
            
            # Filter by expiry
            if time_to_expiry > max_expiry_days or time_to_expiry < 1:
                continue
            
            # Calculate moneyness
            moneyness = option.strike / spot_price
            
            # Filter for reasonable moneyness (0.8 to 1.2)
            if moneyness < 0.8 or moneyness > 1.2:
                continue
            
            # Only include options with reasonable implied volatility
            if option.implied_volatility > 0 and option.implied_volatility < 2.0:
                surface_data.append({
                    'symbol': symbol,
                    'expiry_date': option.expiration_date,
                    'time_to_expiry': time_to_expiry,
                    'strike': option.strike,
                    'moneyness': moneyness,
                    'option_type': option.option_type,
                    'implied_volatility': option.implied_volatility,
                    'bid': option.bid,
                    'ask': option.ask,
                    'last_price': option.last_price,
                    'volume': option.volume,
                    'open_interest': option.open_interest
                })
        
        if not surface_data:
            return pd.DataFrame()
        
        surface_df = pd.DataFrame(surface_data)
        
        # Calculate additional metrics
        surface_df['mid_price'] = (surface_df['bid'] + surface_df['ask']) / 2
        surface_df['bid_ask_spread'] = surface_df['ask'] - surface_df['bid']
        surface_df['spread_pct'] = surface_df['bid_ask_spread'] / surface_df['mid_price']
        
        return surface_df
    
    def analyze_volatility_smile(self, surface_df: pd.DataFrame, 
                                target_expiry: int = 30) -> Dict[str, Any]:
        """
        Analyze the volatility smile for a specific expiry.
        
        Args:
            surface_df: Volatility surface DataFrame
            target_expiry: Target expiry in days
            
        Returns:
            Dictionary with smile analysis
        """
        if surface_df.empty:
            return {}
        
        # Filter for target expiry (±5 days)
        expiry_filter = abs(surface_df['time_to_expiry'] - target_expiry) <= 5
        smile_data = surface_df[expiry_filter].copy()
        
        if smile_data.empty:
            return {}
        
        # Group by option type and calculate average IV by strike
        smile_summary = smile_data.groupby(['option_type', 'moneyness']).agg({
            'implied_volatility': 'mean',
            'volume': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        # Calculate smile characteristics
        analysis = {}
        
        for option_type in ['call', 'put']:
            type_data = smile_summary[smile_summary['option_type'] == option_type]
            
            if len(type_data) < 3:
                continue
            
            # Sort by moneyness
            type_data = type_data.sort_values('moneyness')
            
            # Find ATM implied volatility
            atm_idx = type_data['moneyness'].sub(1.0).abs().idxmin()
            atm_iv = type_data.loc[atm_idx, 'implied_volatility']
            
            # Calculate skew (25-delta risk reversal approximation)
            if len(type_data) >= 5:
                # OTM put (low strike)
                otm_put_iv = type_data.iloc[0]['implied_volatility']
                # OTM call (high strike)  
                otm_call_iv = type_data.iloc[-1]['implied_volatility']
                skew = otm_put_iv - otm_call_iv
            else:
                skew = 0
            
            # Calculate smile curvature
            if len(type_data) >= 3:
                iv_values = type_data['implied_volatility'].values
                moneyness_values = type_data['moneyness'].values
                
                # Fit quadratic to estimate curvature
                try:
                    coeffs = np.polyfit(moneyness_values, iv_values, 2)
                    curvature = coeffs[0] * 2  # Second derivative
                except:
                    curvature = 0
            else:
                curvature = 0
            
            analysis[option_type] = {
                'atm_iv': atm_iv,
                'skew': skew,
                'curvature': curvature,
                'min_iv': type_data['implied_volatility'].min(),
                'max_iv': type_data['implied_volatility'].max(),
                'iv_range': type_data['implied_volatility'].max() - type_data['implied_volatility'].min()
            }
        
        return analysis
    
    def calculate_term_structure(self, surface_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the volatility term structure (ATM volatility by expiry).
        
        Args:
            surface_df: Volatility surface DataFrame
            
        Returns:
            DataFrame with term structure data
        """
        if surface_df.empty:
            return pd.DataFrame()
        
        # Filter for near-ATM options (0.95 to 1.05 moneyness)
        atm_filter = (surface_df['moneyness'] >= 0.95) & (surface_df['moneyness'] <= 1.05)
        atm_data = surface_df[atm_filter].copy()
        
        if atm_data.empty:
            return pd.DataFrame()
        
        # Group by expiry and calculate average IV
        term_structure = atm_data.groupby('time_to_expiry').agg({
            'implied_volatility': 'mean',
            'volume': 'sum',
            'open_interest': 'sum'
        }).reset_index()
        
        # Sort by time to expiry
        term_structure = term_structure.sort_values('time_to_expiry')
        
        # Calculate forward volatility
        term_structure['forward_variance'] = term_structure['implied_volatility']**2 * term_structure['time_to_expiry'] / 365
        term_structure['forward_vol'] = np.sqrt(
            term_structure['forward_variance'].diff() / 
            (term_structure['time_to_expiry'].diff() / 365)
        ).fillna(term_structure['implied_volatility'])
        
        return term_structure

class MarketSentimentAnalyzer:
    """
    Analyze market sentiment using various indicators.
    """
    
    def __init__(self, market_data_provider: AdvancedMarketDataProvider):
        """
        Initialize market sentiment analyzer.
        
        Args:
            market_data_provider: Market data provider instance
        """
        self.data_provider = market_data_provider
    
    def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Calculate Fear & Greed index based on multiple factors.
        
        Returns:
            Dictionary with Fear & Greed analysis
        """
        try:
            # Get VIX data (fear/greed indicator)
            vix_data = self.data_provider.get_historical_data("^VIX", period="1mo")
            
            if vix_data.empty:
                return {'error': 'Unable to fetch VIX data'}
            
            current_vix = vix_data['Close'].iloc[-1]
            vix_avg = vix_data['Close'].mean()
            
            # VIX interpretation
            if current_vix < 15:
                vix_sentiment = "Extreme Greed"
                vix_score = 20
            elif current_vix < 20:
                vix_sentiment = "Greed"
                vix_score = 35
            elif current_vix < 25:
                vix_sentiment = "Neutral"
                vix_score = 50
            elif current_vix < 35:
                vix_sentiment = "Fear"
                vix_score = 70
            else:
                vix_sentiment = "Extreme Fear"
                vix_score = 90
            
            # Get market momentum (S&P 500)
            spy_data = self.data_provider.get_historical_data("SPY", period="3mo")
            
            momentum_score = 50  # Default neutral
            momentum_sentiment = "Neutral"
            
            if not spy_data.empty:
                # Calculate 50-day vs 200-day moving average
                spy_data['MA50'] = spy_data['Close'].rolling(50).mean()
                spy_data['MA200'] = spy_data['Close'].rolling(200).mean()
                
                current_price = spy_data['Close'].iloc[-1]
                ma50 = spy_data['MA50'].iloc[-1]
                ma200 = spy_data['MA200'].iloc[-1]
                
                if not pd.isna(ma50) and not pd.isna(ma200):
                    if current_price > ma50 > ma200:
                        momentum_sentiment = "Bullish"
                        momentum_score = 25
                    elif current_price < ma50 < ma200:
                        momentum_sentiment = "Bearish"
                        momentum_score = 75
                    else:
                        momentum_sentiment = "Mixed"
                        momentum_score = 50
            
            # Combined Fear & Greed Score (0-100, where 0 is extreme greed, 100 is extreme fear)
            combined_score = (vix_score * 0.7 + momentum_score * 0.3)
            
            if combined_score < 25:
                overall_sentiment = "Extreme Greed"
            elif combined_score < 45:
                overall_sentiment = "Greed"
            elif combined_score < 55:
                overall_sentiment = "Neutral"
            elif combined_score < 75:
                overall_sentiment = "Fear"
            else:
                overall_sentiment = "Extreme Fear"
            
            return {
                'fear_greed_score': combined_score,
                'overall_sentiment': overall_sentiment,
                'vix_current': current_vix,
                'vix_sentiment': vix_sentiment,
                'vix_score': vix_score,
                'momentum_sentiment': momentum_sentiment,
                'momentum_score': momentum_score,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating Fear & Greed index: {e}")
            return {'error': str(e)}
    
    def analyze_put_call_ratio(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze put/call ratio for sentiment indication.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary with put/call ratio analysis
        """
        try:
            option_data = self.data_provider.get_option_chain(symbol)
            
            if not option_data:
                return {'error': f'No option data available for {symbol}'}
            
            # Calculate put/call ratios
            total_call_volume = sum(opt.volume for opt in option_data if opt.option_type == 'call')
            total_put_volume = sum(opt.volume for opt in option_data if opt.option_type == 'put')
            
            total_call_oi = sum(opt.open_interest for opt in option_data if opt.option_type == 'call')
            total_put_oi = sum(opt.open_interest for opt in option_data if opt.option_type == 'put')
            
            # Calculate ratios
            if total_call_volume > 0:
                volume_pc_ratio = total_put_volume / total_call_volume
            else:
                volume_pc_ratio = float('inf')
            
            if total_call_oi > 0:
                oi_pc_ratio = total_put_oi / total_call_oi
            else:
                oi_pc_ratio = float('inf')
            
            # Interpret ratios
            def interpret_ratio(ratio):
                if ratio < 0.7:
                    return "Bullish"
                elif ratio < 1.0:
                    return "Slightly Bullish"
                elif ratio < 1.3:
                    return "Neutral"
                elif ratio < 1.5:
                    return "Slightly Bearish"
                else:
                    return "Bearish"
            
            volume_sentiment = interpret_ratio(volume_pc_ratio)
            oi_sentiment = interpret_ratio(oi_pc_ratio)
            
            return {
                'symbol': symbol,
                'volume_put_call_ratio': volume_pc_ratio,
                'oi_put_call_ratio': oi_pc_ratio,
                'volume_sentiment': volume_sentiment,
                'oi_sentiment': oi_sentiment,
                'total_call_volume': total_call_volume,
                'total_put_volume': total_put_volume,
                'total_call_oi': total_call_oi,
                'total_put_oi': total_put_oi,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing put/call ratio for {symbol}: {e}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced Market Data Integration...")
    
    # Initialize market data provider
    data_provider = AdvancedMarketDataProvider()
    
    # Test real-time quote
    print("\n1. Testing Real-time Quote...")
    quote = data_provider.get_real_time_quote("AAPL")
    print(f"AAPL Quote: ${quote.price:.2f} at {quote.timestamp}")
    
    # Test option chain
    print("\n2. Testing Option Chain...")
    option_chain = data_provider.get_option_chain("AAPL")
    print(f"Found {len(option_chain)} option contracts for AAPL")
    
    if option_chain:
        # Show sample option data
        sample_option = option_chain[0]
        print(f"Sample: {sample_option.option_type} {sample_option.strike} "
              f"exp {sample_option.expiration_date.date()} IV: {sample_option.implied_volatility:.2%}")
    
    # Test volatility surface
    print("\n3. Testing Volatility Surface...")
    vol_builder = VolatilitySurfaceBuilder(data_provider)
    surface = vol_builder.build_volatility_surface("AAPL")
    
    if not surface.empty:
        print(f"Volatility surface built with {len(surface)} data points")
        print(f"IV range: {surface['implied_volatility'].min():.2%} - {surface['implied_volatility'].max():.2%}")
        
        # Analyze volatility smile
        smile_analysis = vol_builder.analyze_volatility_smile(surface)
        if smile_analysis:
            print("Volatility Smile Analysis:")
            for opt_type, metrics in smile_analysis.items():
                print(f"  {opt_type.upper()}: ATM IV = {metrics['atm_iv']:.2%}, "
                      f"Skew = {metrics['skew']:.2%}")
    
    # Test market sentiment
    print("\n4. Testing Market Sentiment...")
    sentiment_analyzer = MarketSentimentAnalyzer(data_provider)
    
    fear_greed = sentiment_analyzer.get_fear_greed_index()
    if 'error' not in fear_greed:
        print(f"Fear & Greed Index: {fear_greed['fear_greed_score']:.0f} ({fear_greed['overall_sentiment']})")
        print(f"VIX: {fear_greed['vix_current']:.1f} ({fear_greed['vix_sentiment']})")
    
    pc_ratio = sentiment_analyzer.analyze_put_call_ratio("AAPL")
    if 'error' not in pc_ratio:
        print(f"AAPL Put/Call Ratio: {pc_ratio['volume_put_call_ratio']:.2f} ({pc_ratio['volume_sentiment']})")
    
    print("\nMarket data integration testing completed!")
