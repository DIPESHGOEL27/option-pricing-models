"""Tests for api.iv_engine: the IV solver wrapper, put-call-parity rate
derivation, and skew fitting.
"""

import pytest

from api import iv_engine
from api.app import black_scholes


class TestSolveLegIv:
    """Round-trip: price at a known vol, solve back, and confirm recovery
    -- the core correctness property of the IV engine.
    """

    @pytest.mark.parametrize('S,K,T,r,true_vol,option_type', [
        (100, 100, 0.25, 0.05, 0.20, 'call'),
        (100, 100, 0.25, 0.05, 0.20, 'put'),
        (100, 110, 0.5, 0.03, 0.35, 'call'),
        (100, 90, 0.5, 0.03, 0.35, 'put'),
        (24232, 24250, 0.0192, 0.055, 0.1017, 'call'),  # NIFTY weekly scale
        (24232, 24250, 0.0192, 0.055, 0.1017, 'put'),
    ])
    def test_round_trip_recovers_known_vol(self, S, K, T, r, true_vol, option_type):
        price, *_ = black_scholes(S, K, T, r, true_vol, option_type)
        result = iv_engine.solve_leg_iv(option_type, price, S, K, T, r)
        assert 'iv' in result, f"solve failed: {result.get('skip_reason')}"
        assert result['iv'] == pytest.approx(true_vol, abs=1e-4)

    def test_expired_option_is_skipped(self):
        result = iv_engine.solve_leg_iv('call', 5.0, 100, 100, 0.0, 0.05)
        assert result['skip_reason'] == 'expired'

    def test_missing_price_is_skipped(self):
        result = iv_engine.solve_leg_iv('call', None, 100, 100, 0.25, 0.05)
        assert result['skip_reason'] == 'no_trade_or_quote'

    def test_zero_price_is_skipped(self):
        result = iv_engine.solve_leg_iv('call', 0, 100, 100, 0.25, 0.05)
        assert result['skip_reason'] == 'no_trade_or_quote'

    def test_price_below_intrinsic_is_rejected(self):
        # A call at S=110, K=100 has discounted intrinsic value ~11.23 -- a quoted price
        # of 5 is arbitrage-violating and must not produce a "solved" IV.
        result = iv_engine.solve_leg_iv('call', 5.0, 110, 100, 0.25, 0.05)
        assert result['skip_reason'] == 'price_below_intrinsic'

    def test_discounted_intrinsic_allows_valid_high_rate_quote(self):
        # A put at S=24217, K=24350, T=0.0192 (1 week), r=0.182 (observed live carry rate).
        # Naive intrinsic: K - S = 133.
        # Discounted intrinsic: K*e^(-rT) - S = 24350*e^(-0.182*0.0192) - 24217 ~= 48.0.
        # A market quote of 129.73 is below naive intrinsic (133) but above discounted
        # intrinsic (48.0), and corresponds to a valid implied vol (~16%).
        result = iv_engine.solve_leg_iv('put', 129.73, 24217, 24350, 0.0192, 0.182)
        assert 'iv' in result, f"solve failed: {result.get('skip_reason')}"
        assert 0.05 < result['iv'] < 0.50

    def test_far_outside_moneyness_band_is_skipped(self):
        result = iv_engine.solve_leg_iv('call', 1.0, 100, 1000, 0.25, 0.05)
        assert result['skip_reason'] == 'outside_moneyness_band'

    def test_low_confidence_flag_on_tiny_vega(self):
        # Deep OTM, very little time to expiry: real case observed against
        # the live NIFTY chain where a strike ~12% OTM trading near its tick
        # size solved to an implausible IV.
        price, *_ = black_scholes(24232, 27000, 0.012, 0.05, 0.35, 'call')
        result = iv_engine.solve_leg_iv('call', price, 24232, 27000, 0.012, 0.05)
        if 'iv' in result:
            assert bool(result['low_confidence']) is True


class TestLegMarketPrice:
    """_leg_market_price decides whether a leg's bid-ask mid is a
    trustworthy market price -- the spread must be judged against
    extrinsic (time) value, not total price, or a wide-but-proportionally-
    small spread on a deep ITM/OTM leg slips through and produces an
    implausible solved IV.
    """

    def test_deep_itm_wide_spread_leg_rejects_mid(self):
        # Real NIFTY fixture case (tests/fixtures/nifty_chain.json, strike
        # 21250 CE): open_interest=1, bid=2994.3, ask=3809.6. Spread is only
        # 24% of the 3401.95 mid (well inside MAX_RELATIVE_SPREAD measured
        # against total price) but 223% of the leg's ~366 extrinsic value --
        # this used to solve to 153% IV against NSE's own published 60.4%.
        S, K, T, r = 24231.85, 21250, 0.01002, 0.2542
        leg = {'buyPrice1': 2994.3, 'sellPrice1': 3809.6}
        assert iv_engine._leg_market_price(leg, S, K, T, r, 'call') is None

    def test_low_oi_but_extrinsic_tight_spread_leg_still_solves(self):
        # Contrast case: a genuinely low-OI leg (thin market) whose spread
        # is tight relative to its extrinsic value should still be trusted
        # -- low OI alone must not gate the price, only a spread that
        # swamps the leg's actual time value should.
        S, K, T, r = 100.0, 105.0, 0.25, 0.05
        leg = {'buyPrice1': 4.8, 'sellPrice1': 5.2, 'openInterest': 3}
        assert iv_engine._leg_market_price(leg, S, K, T, r, 'call') == pytest.approx(5.0)


class TestImpliedForwardRate:
    def test_returns_none_with_too_few_liquid_strikes(self):
        assert iv_engine.implied_forward_rate([], 100, 0.1) is None

    def test_recovers_known_rate_from_synthetic_parity_quotes(self):
        S, T, true_rate = 24232.0, 0.02, 0.07
        rows = []
        for K in [24000, 24100, 24200, 24300, 24400]:
            call_price, *_ = black_scholes(S, K, T, true_rate, 0.12, 'call')
            put_price, *_ = black_scholes(S, K, T, true_rate, 0.12, 'put')
            # Reconstruct bid/ask as a tight two-sided quote around the
            # theoretical mid so implied_forward_rate has real inputs to
            # parse, not the theoretical prices directly.
            rows.append({
                'strikePrice': K,
                'CE': {'buyPrice1': call_price - 0.05, 'sellPrice1': call_price + 0.05},
                'PE': {'buyPrice1': put_price - 0.05, 'sellPrice1': put_price + 0.05},
            })
        result = iv_engine.implied_forward_rate(rows, S, T)
        assert result is not None
        assert result['rate'] == pytest.approx(true_rate, abs=5e-3)


class TestTimeToExpiry:
    def test_parses_nse_top_level_format(self):
        from datetime import datetime, timedelta
        future = datetime.now() + timedelta(days=10)
        expiry_str = future.strftime('%d-%b-%Y')
        result = iv_engine.time_to_expiry(expiry_str)
        assert result is not None
        assert 9 / 365 < result['calendar_years'] < 11 / 365

    def test_parses_nse_leg_level_format(self):
        from datetime import datetime, timedelta
        future = datetime.now() + timedelta(days=10)
        expiry_str = future.strftime('%d-%m-%Y')
        result = iv_engine.time_to_expiry(expiry_str)
        assert result is not None

    def test_unparseable_string_returns_none(self):
        assert iv_engine.time_to_expiry('not-a-date') is None

    def test_past_expiry_returns_zero(self):
        result = iv_engine.time_to_expiry('01-Jan-2020')
        assert result['calendar_years'] == 0.0


class TestFitSkew:
    def test_too_few_points_returns_error(self):
        result = iv_engine.fit_skew([], 100)
        assert 'error' in result

    def test_fits_a_flat_curve_when_vol_is_constant(self):
        underlying = 100.0
        strikes = [
            {'strike': 95, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 98, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 100, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 102, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 105, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 110, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
        ]
        result = iv_engine.fit_skew(strikes, underlying)
        assert 'error' not in result
        assert result['atm_iv'] == pytest.approx(0.20, abs=1e-6)
        assert result['residual_std'] == pytest.approx(0.0, abs=1e-6)

    def test_low_confidence_points_excluded_from_fit(self):
        underlying = 100.0
        # Note: strike == underlying_value is treated as the OTM-call side
        # by fit_skew's `strike >= underlying_value` convention, so strike
        # 100 below carries a 'ce' leg, not 'pe'.
        strikes = [
            {'strike': 92, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 95, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 98, 'pe': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 100, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 102, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 105, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            # An implausible, low-confidence outlier that should not pull the fit.
            {'strike': 150, 'ce': {'solved_iv': 3.5, 'low_confidence': True}},
        ]
        result = iv_engine.fit_skew(strikes, underlying)
        assert 'error' not in result
        strikes_used = {p['strike'] for p in result['points']}
        assert 150 not in strikes_used

    def test_itm_leg_ignored_in_favor_of_otm_leg(self):
        underlying = 100.0
        # At K=90 (below spot), the OTM leg is the put -- if a call is also
        # present it must be ignored for the fit even if it has a solved IV.
        strikes = [
            {'strike': 90, 'pe': {'solved_iv': 0.22, 'low_confidence': False}, 'ce': {'solved_iv': 0.55, 'low_confidence': False}},
            {'strike': 92, 'pe': {'solved_iv': 0.215, 'low_confidence': False}},
            {'strike': 95, 'pe': {'solved_iv': 0.21, 'low_confidence': False}},
            {'strike': 100, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 105, 'ce': {'solved_iv': 0.20, 'low_confidence': False}},
            {'strike': 110, 'ce': {'solved_iv': 0.21, 'low_confidence': False}},
        ]
        result = iv_engine.fit_skew(strikes, underlying)
        assert 'error' not in result
        used_types = {(p['strike'], p['option_type']) for p in result['points']}
        assert (90, 'call') not in used_types
        assert (90, 'put') in used_types


class TestAnalyzeChainAgainstRealFixture:
    """Exercises the whole pipeline against a real, recorded NIFTY chain
    (tests/fixtures/nifty_chain.json) rather than synthetic data."""

    def test_produces_priced_and_skipped_legs(self, nifty_chain_raw):
        expiry = iv_engine.time_to_expiry(nifty_chain_raw['expiry'])
        assert expiry is not None and expiry['calendar_years'] > 0

        # Use a plausible rate directly (avoids depending on RBI network access).
        analysis = iv_engine.analyze_chain(
            nifty_chain_raw['rows'], nifty_chain_raw['underlying_value'],
            nifty_chain_raw['expiry'], fallback_risk_free_rate=0.055,
        )
        assert 'error' not in analysis
        assert analysis['summary']['legs_priced'] > 0
        assert analysis['rate_source'] in ('put_call_parity', 'rbi_fallback')
        assert analysis['time_to_expiry']['calendar_years'] == pytest.approx(expiry['calendar_years'], abs=1e-6)

    def test_solved_ivs_are_plausible(self, nifty_chain_raw):
        analysis = iv_engine.analyze_chain(
            nifty_chain_raw['rows'], nifty_chain_raw['underlying_value'],
            nifty_chain_raw['expiry'], fallback_risk_free_rate=0.055,
        )
        solved = [
            leg['solved_iv']
            for entry in analysis['strikes']
            for key in ('ce', 'pe')
            if (leg := entry.get(key)) and 'solved_iv' in leg and not leg.get('low_confidence')
        ]
        assert len(solved) > 5
        # Every non-low-confidence solved IV on a real NIFTY chain should be
        # within a broad but not unlimited band -- catches wholesale solver
        # breakage without being a brittle exact-value assertion.
        assert all(0.01 < iv < 1.5 for iv in solved)

    def test_wide_spread_deep_itm_leg_is_not_wildly_mispriced(self, nifty_chain_raw):
        # Regression for strike 21250 CE: open_interest=1, bid=2994.3,
        # ask=3809.6 -- a spread that's only 24% of the mid (passed the old
        # total-price-relative check easily) but 223% of the leg's own
        # extrinsic value, and used to solve to 153% IV against NSE's own
        # published 60.4% for the same leg. It must now either be skipped
        # outright or, if solved, land in a plausible band.
        analysis = iv_engine.analyze_chain(
            nifty_chain_raw['rows'], nifty_chain_raw['underlying_value'],
            nifty_chain_raw['expiry'], fallback_risk_free_rate=0.055,
        )
        leg = next(e['ce'] for e in analysis['strikes'] if e['strike'] == 21250)
        if 'solved_iv' in leg:
            assert 0.01 < leg['solved_iv'] < 1.5
        else:
            assert leg['skip_reason']

    def test_skew_fit_produces_tight_residuals(self, nifty_chain_raw):
        analysis = iv_engine.analyze_chain(
            nifty_chain_raw['rows'], nifty_chain_raw['underlying_value'],
            nifty_chain_raw['expiry'], fallback_risk_free_rate=0.055,
        )
        skew = iv_engine.fit_skew(analysis['strikes'], nifty_chain_raw['underlying_value'])
        assert 'error' not in skew
        # Regression bound for the noise-filtering pipeline: an earlier,
        # unfiltered version of this fit had a residual_std of ~13.5 vol
        # points on this exact fixture.
        assert skew['residual_std'] < 0.03
