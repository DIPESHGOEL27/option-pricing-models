"""
Implied volatility engine for the NSE options terminal.

Wraps the Newton-Raphson / Brent solvers in option_pricing.py with the
domain rules a real NSE option chain needs: which quotes can even support a
stable solve, how to turn an NSE expiry-date string into a proper time to
expiry, and how to verify a solve actually converged (neither underlying
solver raises on failure -- see _iv_converges below).
"""

from datetime import datetime
import math
from typing import Dict, List, Optional

import numpy as np

try:
    from .option_pricing import AdvancedOptionPricer, ImpliedVolatilityCalculator
except ImportError:
    from option_pricing import AdvancedOptionPricer, ImpliedVolatilityCalculator


# Strikes far from spot are usually too illiquid for a stable IV solve --
# the bid-ask spread dominates any real volatility signal at that point.
MIN_MONEYNESS = 0.5
MAX_MONEYNESS = 2.0
# Reject a quote whose spread is wide enough that the mid price is not a
# meaningful estimate of fair value.
MAX_RELATIVE_SPREAD = 0.5
MIN_IV, MAX_IV = 0.01, 5.0

_pricer = AdvancedOptionPricer()


def parse_nse_expiry(expiry_str: str) -> Optional[datetime]:
    """Parse an NSE expiry date string. Two formats appear in the API:
    the top-level chain uses '25-Aug-2026'; per-leg CE/PE rows use
    '25-08-2026'."""
    if not expiry_str:
        return None
    for fmt in ('%d-%b-%Y', '%d-%m-%Y'):
        try:
            return datetime.strptime(expiry_str, fmt)
        except ValueError:
            continue
    return None


def time_to_expiry(expiry_str: str, now: Optional[datetime] = None) -> Optional[Dict]:
    """Time to expiry in year fractions, from a real NSE expiry date string.

    Reports both a calendar-day figure (365/yr -- what this app's
    Black-Scholes implementation and every other T input already use, so
    it's what actually gets priced with) and a trading-day figure (252/yr,
    approximated as calendar_days * 252/365 in the absence of a real NSE
    holiday calendar) for comparison. Returns None if the date can't be
    parsed, or a zero-time dict if the option has already expired.
    """
    expiry_dt = parse_nse_expiry(expiry_str)
    if expiry_dt is None:
        return None

    now = now or datetime.now()
    expiry_dt = expiry_dt.replace(hour=15, minute=30)  # NSE F&O expires intraday
    calendar_days = (expiry_dt - now).total_seconds() / 86400

    if calendar_days <= 0:
        return {'calendar_years': 0.0, 'trading_years': 0.0, 'calendar_days': 0.0}

    calendar_years = calendar_days / 365.0
    return {
        'calendar_years': calendar_years,
        'trading_years': calendar_years * (252 / 365),
        'calendar_days': calendar_days
    }


def _iv_converges(sigma: Optional[float], market_price: float, S: float, K: float,
                   T: float, r: float, option_type: str, tolerance: float = 1e-4) -> bool:
    """Neither newton_raphson_iv nor brent_method_iv raises on failure to
    converge -- Newton returns whatever sigma it last reached after
    max_iterations even if price_diff never got small, and Brent silently
    falls back to Newton if it can't bracket a root. So this wrapper is the
    actual convergence check: re-price at the solved sigma and confirm it
    reproduces the market price.
    """
    if sigma is None or not (MIN_IV <= sigma <= MAX_IV):
        return False
    repriced = _pricer.black_scholes(S, K, T, r, sigma, option_type)
    return abs(repriced - market_price) < max(tolerance, market_price * 0.01)


def solve_leg_iv(option_type: str, market_price: Optional[float], S: float,
                  K: float, T: float, r: float) -> Dict:
    """Solve IV for one option leg (a single CE or PE quote).

    Returns {'iv': float, 'method': 'newton'|'brent'} on success, or
    {'skip_reason': str} if the quote can't support a stable solve --
    callers should surface the skip reason rather than silently dropping
    the strike, per this project's standing rule against papering over
    missing data.
    """
    if T <= 0:
        return {'skip_reason': 'expired'}
    if market_price is None or market_price <= 0:
        return {'skip_reason': 'no_trade_or_quote'}
    if S <= 0 or K <= 0:
        return {'skip_reason': 'invalid_spot_or_strike'}

    disc_k = K * math.exp(-r * T)
    intrinsic = max(S - disc_k, 0.0) if option_type == 'call' else max(disc_k - S, 0.0)
    if market_price < intrinsic - 1e-6:
        return {'skip_reason': 'price_below_intrinsic'}

    moneyness = S / K
    if not (MIN_MONEYNESS <= moneyness <= MAX_MONEYNESS):
        return {'skip_reason': 'outside_moneyness_band'}

    sigma = ImpliedVolatilityCalculator.newton_raphson_iv(market_price, S, K, T, r, option_type)
    method = 'newton'
    if not _iv_converges(sigma, market_price, S, K, T, r, option_type):
        sigma = ImpliedVolatilityCalculator.brent_method_iv(market_price, S, K, T, r, option_type)
        method = 'brent'
        if not _iv_converges(sigma, market_price, S, K, T, r, option_type):
            return {'skip_reason': 'solver_did_not_converge'}

    # Vega at the solved IV measures how much price moves per unit of vol
    # -- when it's tiny (deep OTM, very little time left), the price-to-IV
    # mapping is nearly flat, so ordinary bid-ask noise implies wildly
    # exaggerated IV swings even though the solve itself "converged". Flag
    # this rather than silently reporting a number that looks precise but
    # isn't: a real chain observed during testing had a strike ~12% OTM
    # trading at 0.35-0.45 rupees where this produced a "solved" 33-35%
    # IV against a ~12% ATM IV -- numerically valid, practically noise.
    vega = _pricer.calculate_greeks(S, K, T, r, sigma, option_type)['vega']
    low_confidence = abs(vega) < 0.5  # rupees of price move per 1% vol

    return {'iv': float(sigma), 'method': method, 'low_confidence': low_confidence}


def _leg_market_price(leg: Dict, S: float, K: float, T: float, r: float,
                       option_type: str) -> Optional[float]:
    """Prefer the bid-ask mid (a real, tradeable fair-value estimate) over
    last traded price (which can be stale in an illiquid strike); fall back
    to LTP when there's no two-sided quote.

    The spread is judged against the leg's extrinsic (time) value, not its
    total price. For a deep ITM/OTM leg, intrinsic value dominates the
    price, so a spread that looks tight relative to the total price can
    still swamp the extrinsic value an IV solve actually depends on. A real
    NIFTY chain observed during testing had a strike 14% ITM with
    open_interest=1, bid=2994.3, ask=3809.6 -- spread only 24% of the
    3401.95 mid (comfortably inside this cap measured against total price)
    but 223% of the leg's ~366 extrinsic value, and solved to 153% IV
    against NSE's own published 60.4% for that same leg.
    """
    bid = leg.get('buyPrice1')
    ask = leg.get('sellPrice1')
    if bid and ask and bid > 0 and ask > 0:
        spread = ask - bid
        mid = (bid + ask) / 2
        disc_k = K * math.exp(-r * T)
        intrinsic = max(S - disc_k, 0.0) if option_type == 'call' else max(disc_k - S, 0.0)
        extrinsic = mid - intrinsic
        if extrinsic > 0 and spread / extrinsic <= MAX_RELATIVE_SPREAD:
            return mid
    ltp = leg.get('lastPrice')
    return ltp if ltp and ltp > 0 else None


def implied_forward_rate(rows: List[Dict], S: float, T: float,
                          moneyness_band: float = 0.02) -> Optional[Dict]:
    """Derive the market-implied (r - dividend yield) directly from
    put-call parity on real near-the-money quotes, instead of assuming the
    underlying pays no dividend (as the plain Black-Scholes formula used
    throughout this app does).

    Put-call parity: C - P = (F - K) * e^(-rT), so F ~= K + (C - P) for
    the small T typical of NSE weekly/monthly expiries. Averaging that
    across several near-ATM strikes (where parity holds most reliably --
    wide spreads further out make it noisy) gives a real, data-derived
    forward, from which the implied rate is ln(F/S) / T.

    This was added after observing calls and puts at the same strike
    solving to systematically different IV (~1-2 vol points apart) when
    using the raw risk-free rate: NIFTY is an index with a real, unknown
    (to this app) dividend/carry adjustment, and the raw risk-free rate
    silently assumes there is none. Using the parity-implied rate instead
    makes call and put solves at the same strike agree with each other --
    a real no-arbitrage property -- which naive use of the risk-free rate
    does not. It will not necessarily match NSE's own displayed per-leg
    IV (NSE's calculation methodology isn't published); that divergence is
    reported via iv_vs_nse_delta, not hidden.

    Returns None if too few strikes have a usable two-sided quote to
    derive a stable estimate.
    """
    if T <= 0 or S <= 0:
        return None

    forwards = []
    for row in rows:
        K = row.get('strikePrice')
        if K is None or abs(K - S) > S * moneyness_band:
            continue
        ce, pe = row.get('CE') or {}, row.get('PE') or {}
        c_bid, c_ask = ce.get('buyPrice1'), ce.get('sellPrice1')
        p_bid, p_ask = pe.get('buyPrice1'), pe.get('sellPrice1')
        if not (c_bid and c_ask and p_bid and p_ask):
            continue
        C, P = (c_bid + c_ask) / 2, (p_bid + p_ask) / 2
        forwards.append(K + (C - P))

    if len(forwards) < 3:
        return None

    F = float(np.median(forwards))
    rate = float(np.log(F / S) / T)
    return {'forward': F, 'rate': rate, 'strikes_used': len(forwards)}


def analyze_chain(rows: List[Dict], underlying_value: float, expiry_str: str,
                   fallback_risk_free_rate: Optional[float]) -> Dict:
    """Solve IV for every strike/leg in an NSE option chain.

    Uses the put-call-parity-implied rate (see implied_forward_rate) when
    enough liquid near-ATM quotes are available to derive one reliably,
    falling back to fallback_risk_free_rate (the RBI-derived proxy)
    otherwise. Returns per-strike results plus a summary of how many legs
    were priced vs skipped and why -- skipped strikes are reported, never
    silently dropped.
    """
    expiry = time_to_expiry(expiry_str)
    if expiry is None:
        return {'error': f'Could not parse expiry date: {expiry_str!r}'}

    T = expiry['calendar_years']

    forward_info = implied_forward_rate(rows, underlying_value, T)
    if forward_info is not None:
        risk_free_rate = forward_info['rate']
        rate_source = 'put_call_parity'
    elif fallback_risk_free_rate is not None:
        risk_free_rate = fallback_risk_free_rate
        rate_source = 'rbi_fallback'
    else:
        return {'error': 'No risk-free rate available: too few liquid strikes to '
                          'derive a put-call-parity rate, and the RBI rate fetch failed'}

    results = []
    skip_counts: Dict[str, int] = {}

    for row in rows:
        strike = row.get('strikePrice')
        if strike is None:
            continue

        entry = {'strike': strike}
        for leg_key, option_type in (('CE', 'call'), ('PE', 'put')):
            leg = row.get(leg_key) or {}
            if not leg:
                continue

            market_price = _leg_market_price(leg, underlying_value, strike, T, risk_free_rate, option_type)
            solved = solve_leg_iv(option_type, market_price, underlying_value, strike, T, risk_free_rate)

            nse_iv = leg.get('impliedVolatility')
            nse_iv = nse_iv / 100 if nse_iv else None

            leg_result = {
                'market_price': market_price,
                'last_price': leg.get('lastPrice'),
                'bid': leg.get('buyPrice1'),
                'ask': leg.get('sellPrice1'),
                'open_interest': leg.get('openInterest'),
                'oi_change': leg.get('changeinOpenInterest'),
                'nse_iv': nse_iv,
            }
            if 'iv' in solved:
                leg_result['solved_iv'] = solved['iv']
                leg_result['solve_method'] = solved['method']
                leg_result['low_confidence'] = solved['low_confidence']
                # Re-derived price at the solved IV -- an audit/sanity field
                # confirming the solve: this should match market_price
                # closely by construction (that agreement is the solver's
                # own convergence criterion), not an independent estimate.
                leg_result['model_price'] = float(
                    _pricer.black_scholes(underlying_value, strike, T, risk_free_rate, solved['iv'], option_type)
                )
                if nse_iv:
                    leg_result['iv_vs_nse_delta'] = solved['iv'] - nse_iv
            else:
                leg_result['skip_reason'] = solved['skip_reason']
                skip_counts[solved['skip_reason']] = skip_counts.get(solved['skip_reason'], 0) + 1

            entry[leg_key.lower()] = leg_result

        results.append(entry)

    priced = sum(
        1 for r in results for k in ('ce', 'pe')
        if k in r and 'solved_iv' in r[k]
    )
    total_legs = sum(1 for r in results for k in ('ce', 'pe') if k in r)

    return {
        'expiry': expiry_str,
        'time_to_expiry': expiry,
        'risk_free_rate': risk_free_rate,
        'rate_source': rate_source,
        'implied_forward': forward_info,
        'strikes': results,
        'summary': {
            'legs_priced': priced,
            'legs_skipped': total_legs - priced,
            'skip_reasons': skip_counts
        }
    }


def fit_skew(strike_results: List[Dict], underlying_value: float) -> Dict:
    """Fit a simple quadratic curve to solved IV vs log-moneyness, and flag
    each strike's deviation from that fitted curve.

    This is a relative-value view, not an absolute mispricing signal: a
    strike sitting above the fitted curve has a higher IV than its
    neighbors after accounting for the normal smile/skew shape, which is
    the standard way volatility relative value is screened -- it says
    "priced rich relative to the rest of the curve today", not "will fall".
    Requires at least 5 solved points to fit a stable quadratic; returns an
    explicit error otherwise rather than fitting noise.
    """
    points = []  # (log_moneyness, iv, strike, option_type)
    for entry in strike_results:
        strike = entry['strike']
        # Use only the out-of-the-money leg at each strike: a call above
        # spot or a put below spot. ITM options are systematically far less
        # liquid than their OTM counterpart at the same strike (traders
        # prefer the cheaper, more capital-efficient OTM side), which shows
        # up directly in NSE's own open-interest figures -- e.g. a real
        # chain observed during testing had CE open interest of 1 contract
        # at a strike ~12% ITM, against ~1,500 contracts on the PE side of
        # that same strike. Solving IV from a near-untraded ITM quote
        # produces IV noise (~140% on that example) that a liquid OTM quote
        # at the identical strike does not have. This is standard market
        # convention for building a skew, not an arbitrary exclusion.
        is_otm_call = strike >= underlying_value
        leg_key, option_type = ('ce', 'call') if is_otm_call else ('pe', 'put')
        leg = entry.get(leg_key)
        if leg and 'solved_iv' in leg and not leg.get('low_confidence'):
            log_m = float(np.log(underlying_value / strike))
            points.append((log_m, leg['solved_iv'], strike, option_type))

    if len(points) < 5:
        return {'error': f'Only {len(points)} OTM, non-low-confidence solved IV points available; need at least 5 to fit a skew curve'}

    log_moneyness = np.array([p[0] for p in points])
    ivs = np.array([p[1] for p in points])

    # Fit, then one robust refit pass excluding points whose residual is a
    # large multiple of the median absolute deviation -- a safety net for
    # any remaining anomaly not caught by the OTM-leg/low-confidence
    # filtering above (a robust bound rather than a fixed vol-point
    # threshold, so it adapts to how volatile the instrument actually is).
    coeffs = np.polyfit(log_moneyness, ivs, deg=2)
    first_pass_resid = ivs - np.polyval(coeffs, log_moneyness)
    resid_mad = float(np.median(np.abs(first_pass_resid - np.median(first_pass_resid))))
    if resid_mad > 1e-8:
        keep = np.abs(first_pass_resid - np.median(first_pass_resid)) <= 4 * 1.4826 * resid_mad
        if keep.sum() >= 5:
            points = [p for p, k in zip(points, keep) if k]
            log_moneyness = log_moneyness[keep]
            ivs = ivs[keep]
            coeffs = np.polyfit(log_moneyness, ivs, deg=2)

    # Quadratic fit: iv = a*x^2 + b*x + c, the standard simple parametric
    # smile shape (captures curvature/skew without overfitting to noise
    # with only a few dozen strikes).
    fitted = np.polyval(coeffs, log_moneyness)
    residuals = ivs - fitted
    residual_std = float(np.std(residuals)) if len(residuals) > 1 else 0.0

    per_point = []
    for (log_m, iv, strike, option_type), fit_val, resid in zip(points, fitted, residuals):
        deviation_sigma = (resid / residual_std) if residual_std > 1e-8 else 0.0
        per_point.append({
            'strike': strike,
            'option_type': option_type,
            'solved_iv': iv,
            'fitted_iv': float(fit_val),
            'deviation': float(resid),
            'deviation_sigma': float(deviation_sigma),
            'flag': 'rich' if deviation_sigma > 1.0 else 'cheap' if deviation_sigma < -1.0 else 'normal'
        })

    atm_log_m = 0.0
    atm_iv = float(np.polyval(coeffs, atm_log_m))

    return {
        'curve_fit': {'a': float(coeffs[0]), 'b': float(coeffs[1]), 'c': float(coeffs[2])},
        'atm_iv': atm_iv,
        'residual_std': residual_std,
        'points': per_point
    }
