"""Tests for api.india_market_data's pure analytics functions: PCR, max
pain, and OI buildup summarization -- checked against hand-computed
fixtures, then sanity-checked against the real recorded chain.
"""

import pytest

from api.india_market_data import calculate_max_pain, calculate_put_call_ratio, summarize_oi_buildup


def make_rows(strike_oi_pairs):
    """strike_oi_pairs: list of (strike, call_oi, put_oi, call_oi_chg, put_oi_chg)"""
    rows = []
    for strike, call_oi, put_oi, *rest in strike_oi_pairs:
        call_chg, put_chg = (rest + [0, 0])[:2]
        rows.append({
            'strikePrice': strike,
            'CE': {'openInterest': call_oi, 'changeinOpenInterest': call_chg},
            'PE': {'openInterest': put_oi, 'changeinOpenInterest': put_chg},
        })
    return rows


class TestPutCallRatio:
    def test_hand_computed_ratio(self):
        rows = make_rows([(100, 1000, 500), (110, 2000, 1500)])
        result = calculate_put_call_ratio(rows)
        assert result['call_oi'] == 3000
        assert result['put_oi'] == 2000
        assert result['oi_pcr'] == pytest.approx(2000 / 3000)

    def test_zero_call_oi_does_not_divide_by_zero(self):
        rows = make_rows([(100, 0, 500)])
        result = calculate_put_call_ratio(rows)
        assert result['oi_pcr'] == 0

    @pytest.mark.parametrize('pcr,expected_sentiment', [
        (1.5, 'Bearish (More put buying)'),
        (0.8, 'Neutral to Bearish'),
        (0.6, 'Neutral'),
        (0.3, 'Bullish (More call buying)'),
    ])
    def test_sentiment_bands(self, pcr, expected_sentiment):
        put_oi = int(pcr * 1000)
        rows = make_rows([(100, 1000, put_oi)])
        result = calculate_put_call_ratio(rows)
        assert result['sentiment'] == expected_sentiment


class TestMaxPain:
    def test_hand_computed_max_pain(self):
        # All open interest concentrated at strike 100 on both sides -- the
        # writer's payout is minimized exactly at 100 (zero payout there;
        # positive everywhere else), so max pain must be 100.
        rows = make_rows([
            (90, 0, 0),
            (100, 1000, 1000),
            (110, 0, 0),
        ])
        result = calculate_max_pain(rows)
        assert result['strike'] == 100
        assert result['total_payout'] == 0

    def test_max_pain_with_asymmetric_oi(self):
        # Heavy call OI at 110 pulls the writer-payout-minimizing strike
        # down relative to a pure put/call symmetric case, since above 110
        # each point costs 1000x more than below it.
        rows = make_rows([
            (90, 0, 0),
            (100, 0, 0),
            (110, 10_000, 100),
            (120, 0, 0),
        ])
        result = calculate_max_pain(rows)
        assert result['strike'] <= 110

    def test_no_strikes_returns_error(self):
        result = calculate_max_pain([])
        assert 'error' in result


class TestOiBuildup:
    def test_sorted_by_strike_with_changes(self):
        rows = make_rows([
            (110, 200, 100, -10, 5),
            (100, 500, 300, 20, -15),
        ])
        result = summarize_oi_buildup(rows)
        assert result['strikes'] == [100, 110]
        assert result['call_oi'] == [500, 200]
        assert result['put_oi'] == [300, 100]
        assert result['call_oi_change'] == [20, -10]
        assert result['put_oi_change'] == [-15, 5]

    def test_missing_strike_price_skipped(self):
        rows = [{'CE': {'openInterest': 1}, 'PE': {'openInterest': 1}}]
        result = summarize_oi_buildup(rows)
        assert result['strikes'] == []


class TestAgainstRealFixture:
    """Sanity checks against the recorded live NIFTY chain -- values aren't
    pinned exactly (the fixture reflects one live snapshot), but their
    basic structural properties must hold.
    """

    def test_pcr_is_a_plausible_ratio(self, nifty_chain_raw):
        result = calculate_put_call_ratio(nifty_chain_raw['rows'])
        assert result['call_oi'] > 0
        assert result['put_oi'] > 0
        assert 0 < result['oi_pcr'] < 10

    def test_max_pain_strike_is_one_of_the_real_strikes(self, nifty_chain_raw):
        result = calculate_max_pain(nifty_chain_raw['rows'])
        real_strikes = {r['strikePrice'] for r in nifty_chain_raw['rows']}
        assert result['strike'] in real_strikes

    def test_oi_buildup_covers_every_strike(self, nifty_chain_raw):
        result = summarize_oi_buildup(nifty_chain_raw['rows'])
        assert len(result['strikes']) == len(nifty_chain_raw['rows'])
        assert result['strikes'] == sorted(result['strikes'])
