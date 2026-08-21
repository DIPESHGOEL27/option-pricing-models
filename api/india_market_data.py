"""
India Market Data Integration Module

Free NSE/BSE market data via jugaad-data (an unofficial NSE/RBI scraper --
same class of unofficial-API risk as yfinance, which the rest of this app
already relies on): live quotes, F&O option chains, index snapshots, and
RBI policy/G-Sec rates used as a risk-free-rate proxy.

Basic equity price/history for NSE/BSE symbols (RELIANCE.NS, RELIANCE.BO,
^NSEI, etc.) is already served by the existing yfinance-based
MarketDataProvider in market_data.py -- this module only covers what
yfinance does not provide for Indian markets: option chains and RBI rates.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from jugaad_data.nse import NSELive
from jugaad_data.rbi import RBI

# Symbols that need index_option_chain() rather than equities_option_chain().
INDEX_SYMBOLS = {'NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'NIFTYNXT50'}

_nse_live_singleton = None


def get_nse_live() -> NSELive:
    """Lazily create the shared NSELive session (bootstraps cookies against
    nseindia.com, which is comparatively expensive to redo per request)."""
    global _nse_live_singleton
    if _nse_live_singleton is None:
        _nse_live_singleton = NSELive()
    return _nse_live_singleton


def _reset_nse_live():
    global _nse_live_singleton
    _nse_live_singleton = None


def _call_with_retry(fn):
    """NSE's session/cookie bootstrap occasionally fails silently on first
    use (observed during testing); retrying once with a fresh session
    resolves it without surfacing a false negative to the caller."""
    try:
        return fn()
    except Exception:
        _reset_nse_live()
        return fn()


class IndiaMarketDataProvider:
    """NSE quotes, F&O option chains, index snapshot, and market status."""

    def __init__(self):
        self.cache = {}
        self.cache_duration = 90  # seconds -- NSE is an unofficial scrape target

    def _cached(self, key, fetch_fn):
        if key in self.cache:
            cached_time, data = self.cache[key]
            if time.time() - cached_time < self.cache_duration:
                return data
        data = fetch_fn()
        self.cache[key] = (time.time(), data)
        return data

    def get_quote(self, symbol: str) -> Dict:
        """Current NSE quote for a bare symbol, e.g. 'RELIANCE'."""
        symbol = symbol.upper()

        def fetch():
            q = _call_with_retry(lambda: get_nse_live().stock_quote(symbol))

            trade_info = q.get('tradeInfo', {}) or {}
            price_info = q.get('priceInfo', {}) or {}
            meta = q.get('metaData', {}) or {}
            last_price = trade_info.get('lastPrice') or meta.get('closePrice')

            if not last_price:
                return {'error': f'No quote data available for {symbol}'}

            prev_close = meta.get('previousClose') or trade_info.get('basePrice') or last_price
            day_change = last_price - prev_close
            day_change_pct = (day_change / prev_close * 100) if prev_close else 0

            return {
                'symbol': symbol,
                'price': last_price,
                'day_change': day_change,
                'day_change_pct': day_change_pct,
                'volume': trade_info.get('totalTradedVolume', 0),
                'year_high': price_info.get('yearHigh'),
                'year_low': price_info.get('yearLow'),
                'last_update': q.get('lastUpdateTime'),
                'currency': 'INR'
            }

        try:
            return self._cached(f'quote_{symbol}', fetch)
        except Exception as e:
            return {'error': f"Failed to fetch NSE quote for {symbol}: {str(e)}"}

    def get_indices_snapshot(self) -> Dict:
        """Bundle of NIFTY 50 / NIFTY BANK / INDIA VIX / etc in one call."""
        def fetch():
            data = _call_with_retry(lambda: get_nse_live().all_indices())
            return {row.get('index'): row for row in data.get('data', [])}

        try:
            return self._cached('indices_snapshot', fetch)
        except Exception as e:
            return {'error': f"Failed to fetch indices snapshot: {str(e)}"}

    def get_market_status(self) -> Dict:
        try:
            return _call_with_retry(lambda: get_nse_live().market_status())
        except Exception as e:
            return {'error': f"Failed to fetch market status: {str(e)}"}

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict:
        """Real NSE F&O option chain for an index or individual stock.

        Each expiry is a genuinely separate NSE request: jugaad-data's
        index_option_chain/equities_option_chain fetch only ONE expiry's
        contracts per call (whichever `expiry` names, or the nearest one if
        omitted) even though the response's top-level `expiryDates` lists
        every future expiry as metadata -- that field describes what
        expiries exist, not what's included in `rows`. Passing `expiry`
        through to the underlying call (rather than only using it to filter
        client-side, which silently returned nothing for any non-nearest
        expiry) is what actually fetches that expiry's contracts. Cached
        per (symbol, expiry) accordingly.
        """
        symbol = symbol.upper()
        is_index = symbol in INDEX_SYMBOLS

        def fetch_chain():
            nse = get_nse_live()
            if is_index:
                return nse.index_option_chain(symbol, expiry=expiry)
            return nse.equities_option_chain(symbol, expiry=expiry)

        def fetch_with_fallback():
            try:
                return fetch_chain()
            except Exception:
                # Symbol classification can be wrong (e.g. a new index not yet
                # in INDEX_SYMBOLS) -- try the other endpoint before giving up.
                nse = get_nse_live()
                return (nse.equities_option_chain(symbol, expiry=expiry) if is_index
                        else nse.index_option_chain(symbol, expiry=expiry))

        cache_key = f'chain_{symbol}_{expiry or "nearest"}'
        try:
            raw = self._cached(cache_key, lambda: _call_with_retry(fetch_with_fallback))
        except Exception as e:
            return {'error': f"Failed to fetch option chain for {symbol}: {str(e)}"}

        records = raw.get('records', {}) or {}
        rows = records.get('data', []) or []
        expiry_dates = records.get('expiryDates', []) or []
        underlying_value = records.get('underlyingValue')

        chosen_expiry = expiry or (expiry_dates[0] if expiry_dates else None)
        if chosen_expiry:
            # NSE's per-row key is (confusingly) "expiryDates" (plural) but
            # holds a single date string for that row.
            rows = [r for r in rows if r.get('expiryDates') == chosen_expiry]

        if not rows:
            detail = f' at expiry {expiry}' if expiry else ''
            return {'error': f'No option chain data available for {symbol}{detail}'}

        return {
            'symbol': symbol,
            'underlying_value': underlying_value,
            'expiry': chosen_expiry,
            'expiry_dates': expiry_dates,
            'rows': rows,
            'currency': 'INR'
        }


class IndiaRiskFreeRateProvider:
    """RBI policy rate + G-Sec/T-bill yields, used as an India risk-free-rate
    proxy the same way ^TNX is used as a blunt US-10Y proxy elsewhere in
    this app."""

    # Matches labels like "6.94% GS 2036" -> captures the maturity year.
    _GSEC_LABEL_RE = re.compile(r'GS\s*(\d{4})', re.IGNORECASE)
    # Matches labels like "91 day T-bills" -> captures the day count.
    _TBILL_LABEL_RE = re.compile(r'(\d+)\s*day\s*T-?bills?', re.IGNORECASE)

    def __init__(self):
        self._cache = None  # (fetched_at, raw_dict)
        self._cache_duration = 6 * 3600  # seconds -- RBI rates move slowly

    def _fetch_raw(self) -> Optional[Dict]:
        if self._cache is not None:
            fetched_at, raw = self._cache
            if time.time() - fetched_at < self._cache_duration:
                return raw
        try:
            raw = RBI().current_rates()
        except Exception:
            return None
        if raw:
            self._cache = (time.time(), raw)
        return raw

    def get_rates(self) -> Dict:
        raw = self._fetch_raw()
        if not raw:
            return {'error': 'Failed to fetch RBI rates'}

        ten_year_proxy, ten_year_label = self._pick_ten_year_gsec(raw)

        return {
            'repo_rate': self._parse_pct(raw.get('Policy Repo Rate')),
            'ten_year_proxy': ten_year_proxy,
            'ten_year_proxy_label': ten_year_label,
            'raw': raw
        }

    def interpolate_rate(self, time_to_maturity: float) -> Optional[float]:
        """Interpolate a risk-free rate for a specific maturity (in years)
        from real T-bill/G-Sec yields.

        NSE options typically expire in days to a few months -- using the
        blunt 10-year G-Sec proxy for those would badly misstate the
        relevant rate, so this interpolates across every T-bill/G-Sec
        point RBI publishes rather than picking a single fixed maturity.
        Returns None if no real rate data is available -- never a
        fabricated fallback.
        """
        raw = self._fetch_raw()
        if not raw:
            return None

        points = self._parse_maturity_points(raw)
        if not points:
            return None

        times = [p[0] for p in points]
        rates = [p[1] for p in points]
        if time_to_maturity <= times[0]:
            return rates[0]
        if time_to_maturity >= times[-1]:
            return rates[-1]
        return float(np.interp(time_to_maturity, times, rates))

    def _parse_maturity_points(self, raw: Dict) -> List[Tuple[float, float]]:
        """Parse every T-bill/G-Sec label into (years_to_maturity, rate)."""
        from datetime import datetime
        current_year = datetime.now().year
        points = []

        for label, value in raw.items():
            pct = self._parse_pct(value)
            if pct is None:
                continue

            tbill_match = self._TBILL_LABEL_RE.search(label)
            if tbill_match:
                points.append((int(tbill_match.group(1)) / 365.0, pct))
                continue

            gsec_match = self._GSEC_LABEL_RE.search(label)
            if gsec_match:
                years = int(gsec_match.group(1)) - current_year
                if years > 0:
                    points.append((float(years), pct))

        return sorted(points)

    def _pick_ten_year_gsec(self, raw: Dict):
        """Pick the G-Sec whose maturity year is closest to 10 years out.

        This is a heuristic (nearest-maturity label matching), not a
        guaranteed correct instrument match -- documented approximation,
        same caveat class as using ^TNX as a blunt US-10Y proxy.
        """
        from datetime import datetime
        current_year = datetime.now().year
        best_label, best_pct, best_diff = None, None, None

        for label, value in raw.items():
            match = self._GSEC_LABEL_RE.search(label)
            if not match:
                continue
            maturity_year = int(match.group(1))
            diff = abs((maturity_year - current_year) - 10)
            pct = self._parse_pct(value)
            if pct is None:
                continue
            if best_diff is None or diff < best_diff:
                best_label, best_pct, best_diff = label, pct, diff

        return best_pct, best_label

    @staticmethod
    def _parse_pct(value) -> Optional[float]:
        if value is None:
            return None
        match = re.search(r'[-+]?\d*\.?\d+', str(value))
        return float(match.group()) / 100 if match else None


def calculate_put_call_ratio(rows: List[Dict]) -> Dict:
    """Open-interest-based put/call ratio -- the NSE-standard PCR metric.

    Interpretation bands match MarketSentimentIndicators.get_put_call_ratio
    in market_data.py for consistency with the US side of the app.
    """
    call_oi = sum((row.get('CE') or {}).get('openInterest', 0) for row in rows)
    put_oi = sum((row.get('PE') or {}).get('openInterest', 0) for row in rows)

    oi_pcr = put_oi / call_oi if call_oi > 0 else 0

    if oi_pcr > 1.0:
        sentiment = "Bearish (More put buying)"
    elif oi_pcr > 0.7:
        sentiment = "Neutral to Bearish"
    elif oi_pcr > 0.5:
        sentiment = "Neutral"
    else:
        sentiment = "Bullish (More call buying)"

    return {
        'oi_pcr': oi_pcr,
        'sentiment': sentiment,
        'call_oi': call_oi,
        'put_oi': put_oi
    }


def calculate_max_pain(rows: List[Dict]) -> Dict:
    """Max pain: the strike minimizing total option-writer payout.

    payout(K) = sum(CE_OI(Ki) * max(K - Ki, 0)) + sum(PE_OI(Ki) * max(Ki - K, 0))
    O(n^2) over the ~40-120 strikes NSE typically lists -- no optimization needed.
    """
    strikes = sorted({row['strikePrice'] for row in rows if 'strikePrice' in row})
    if not strikes:
        return {'error': 'No strikes available for max pain calculation'}

    oi_by_strike = {}
    for row in rows:
        k = row.get('strikePrice')
        if k is None:
            continue
        oi_by_strike[k] = {
            'call_oi': (row.get('CE') or {}).get('openInterest', 0),
            'put_oi': (row.get('PE') or {}).get('openInterest', 0)
        }

    best_strike, best_payout = None, None
    for candidate in strikes:
        payout = 0
        for strike, oi in oi_by_strike.items():
            payout += oi['call_oi'] * max(candidate - strike, 0)
            payout += oi['put_oi'] * max(strike - candidate, 0)
        if best_payout is None or payout < best_payout:
            best_strike, best_payout = candidate, payout

    return {'strike': best_strike, 'total_payout': best_payout}


def summarize_oi_buildup(rows: List[Dict]) -> Dict:
    """Per-strike CE/PE open interest and change-in-OI, for charting."""
    strikes, call_oi, put_oi, call_oi_change, put_oi_change = [], [], [], [], []

    for row in sorted(rows, key=lambda r: r.get('strikePrice', 0)):
        strike = row.get('strikePrice')
        if strike is None:
            continue
        ce = row.get('CE') or {}
        pe = row.get('PE') or {}
        strikes.append(strike)
        call_oi.append(ce.get('openInterest', 0))
        put_oi.append(pe.get('openInterest', 0))
        call_oi_change.append(ce.get('changeinOpenInterest', 0))
        put_oi_change.append(pe.get('changeinOpenInterest', 0))

    return {
        'strikes': strikes,
        'call_oi': call_oi,
        'put_oi': put_oi,
        'call_oi_change': call_oi_change,
        'put_oi_change': put_oi_change
    }
