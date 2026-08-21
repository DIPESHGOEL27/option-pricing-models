"""Tests for Black-Scholes pricing and Greeks (api.app.black_scholes, which
delegates to api.option_pricing.AdvancedOptionPricer).

Includes explicit regression tests for a real bug found this session: the
previous local implementation in app.py applied the call formula for theta
and rho to puts too, so a put's rho came back with the wrong sign.
"""

import math

import pytest

from api.app import black_scholes


def bs_reference(S, K, T, r, sigma, option_type):
    """An independent, minimal Black-Scholes implementation (scipy norm
    directly, no shared code with the module under test) used as a ground
    truth for reference-value tests.
    """
    from scipy.stats import norm

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


class TestReferenceValues:
    @pytest.mark.parametrize('S,K,T,r,sigma,option_type', [
        (100, 100, 0.25, 0.05, 0.2, 'call'),
        (100, 100, 0.25, 0.05, 0.2, 'put'),
        (100, 110, 1.0, 0.05, 0.3, 'call'),
        (100, 90, 1.0, 0.05, 0.3, 'put'),
        (24232, 24250, 0.05, 0.06, 0.12, 'call'),  # NIFTY-scale
        (24232, 24250, 0.05, 0.06, 0.12, 'put'),
    ])
    def test_price_matches_independent_reference(self, S, K, T, r, sigma, option_type):
        price, *_ = black_scholes(S, K, T, r, sigma, option_type)
        expected = bs_reference(S, K, T, r, sigma, option_type)
        assert price == pytest.approx(expected, rel=1e-9)


class TestPutCallParity:
    @pytest.mark.parametrize('S,K,T,r,sigma', [
        (100, 100, 0.25, 0.05, 0.2),
        (100, 120, 0.5, 0.03, 0.35),
        (24232, 24250, 0.05, 0.06, 0.12),
        (50, 45, 2.0, 0.08, 0.5),
    ])
    def test_call_minus_put_equals_forward_minus_strike(self, S, K, T, r, sigma):
        call_price, *_ = black_scholes(S, K, T, r, sigma, 'call')
        put_price, *_ = black_scholes(S, K, T, r, sigma, 'put')
        expected = S - K * math.exp(-r * T)
        assert (call_price - put_price) == pytest.approx(expected, rel=1e-9)


class TestGreeksVsFiniteDifference:
    """Every analytic Greek should match a central finite difference of the
    price itself -- a Greek that's internally inconsistent with its own
    price function is wrong regardless of what closed-form formula produced
    it."""

    S, K, T, r, sigma = 100.0, 105.0, 0.4, 0.04, 0.25

    @pytest.mark.parametrize('option_type', ['call', 'put'])
    def test_delta(self, option_type):
        price, delta, *_ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, option_type)
        dS = self.S * 1e-4
        up, *_ = black_scholes(self.S + dS, self.K, self.T, self.r, self.sigma, option_type)
        down, *_ = black_scholes(self.S - dS, self.K, self.T, self.r, self.sigma, option_type)
        fd_delta = (up - down) / (2 * dS)
        assert delta == pytest.approx(fd_delta, abs=1e-4)

    @pytest.mark.parametrize('option_type', ['call', 'put'])
    def test_gamma(self, option_type):
        price, delta, gamma, *_ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, option_type)
        dS = self.S * 1e-3
        up, *_ = black_scholes(self.S + dS, self.K, self.T, self.r, self.sigma, option_type)
        down, *_ = black_scholes(self.S - dS, self.K, self.T, self.r, self.sigma, option_type)
        fd_gamma = (up - 2 * price + down) / (dS ** 2)
        assert gamma == pytest.approx(fd_gamma, rel=1e-2)

    @pytest.mark.parametrize('option_type', ['call', 'put'])
    def test_vega_per_1pct(self, option_type):
        _, _, _, _, vega, _ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, option_type)
        dvol = 1e-4
        up, *_ = black_scholes(self.S, self.K, self.T, self.r, self.sigma + dvol, option_type)
        down, *_ = black_scholes(self.S, self.K, self.T, self.r, self.sigma - dvol, option_type)
        fd_vega_per_1pct = (up - down) / (2 * dvol) / 100
        assert vega == pytest.approx(fd_vega_per_1pct, rel=1e-3)

    @pytest.mark.parametrize('option_type', ['call', 'put'])
    def test_theta_per_day(self, option_type):
        price, *_, theta, _, _ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, option_type)
        dT = 1 / 365
        price_tomorrow, *_ = black_scholes(self.S, self.K, self.T - dT, self.r, self.sigma, option_type)
        fd_theta_per_day = price_tomorrow - price
        assert theta == pytest.approx(fd_theta_per_day, abs=5e-3)

    @pytest.mark.parametrize('option_type', ['call', 'put'])
    def test_rho_per_1pct(self, option_type):
        *_, rho = black_scholes(self.S, self.K, self.T, self.r, self.sigma, option_type)
        dr = 1e-5
        up, *_ = black_scholes(self.S, self.K, self.T, self.r + dr, self.sigma, option_type)
        down, *_ = black_scholes(self.S, self.K, self.T, self.r - dr, self.sigma, option_type)
        fd_rho_per_1pct = (up - down) / (2 * dr) / 100
        assert rho == pytest.approx(fd_rho_per_1pct, rel=1e-3)


class TestPutGreeksRegression:
    """Regression tests for a real bug: the pre-fix local black_scholes()
    in app.py used the call formula for theta and rho on puts too. A put's
    rho must be negative (put value falls as rates rise) and its theta must
    not equal the call's theta at the same strike.
    """

    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2

    def test_put_rho_is_negative(self):
        *_, put_rho = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'put')
        assert put_rho < 0

    def test_put_rho_matches_known_reference(self):
        # Independently computed: -K*T*exp(-rT)*N(-d2)/100
        from scipy.stats import norm
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        expected = -self.K * self.T * math.exp(-self.r * self.T) * norm.cdf(-d2) / 100
        *_, put_rho = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'put')
        assert put_rho == pytest.approx(expected, rel=1e-6)

    def test_call_and_put_theta_differ(self):
        _, _, _, call_theta, _, _ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'call')
        _, _, _, put_theta, _, _ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'put')
        assert call_theta != pytest.approx(put_theta, rel=1e-6)

    def test_call_and_put_rho_have_opposite_signs(self):
        *_, call_rho = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'call')
        *_, put_rho = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'put')
        assert call_rho > 0
        assert put_rho < 0


class TestBoundaryBehavior:
    def test_at_expiry_call_returns_intrinsic_value(self):
        price, delta, gamma, theta, vega, rho = black_scholes(110, 100, 0.0, 0.05, 0.2, 'call')
        assert price == pytest.approx(10.0)
        assert delta == 0
        assert gamma == 0

    def test_at_expiry_put_returns_intrinsic_value(self):
        price, *_ = black_scholes(90, 100, 0.0, 0.05, 0.2, 'put')
        assert price == pytest.approx(10.0)

    def test_at_expiry_otm_call_is_worthless(self):
        price, *_ = black_scholes(90, 100, 0.0, 0.05, 0.2, 'call')
        assert price == pytest.approx(0.0)

    def test_negative_time_rejected(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, -0.1, 0.05, 0.2, 'call')

    def test_zero_or_negative_spot_rejected(self):
        with pytest.raises(ValueError):
            black_scholes(0, 100, 0.25, 0.05, 0.2, 'call')
        with pytest.raises(ValueError):
            black_scholes(-10, 100, 0.25, 0.05, 0.2, 'call')

    def test_zero_or_negative_strike_rejected(self):
        with pytest.raises(ValueError):
            black_scholes(100, 0, 0.25, 0.05, 0.2, 'call')

    def test_zero_vol_with_positive_time_rejected(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, 0.25, 0.05, 0, 'call')

    def test_invalid_option_type_rejected(self):
        with pytest.raises(ValueError):
            black_scholes(100, 100, 0.25, 0.05, 0.2, 'straddle')

    def test_deep_itm_call_price_approaches_intrinsic(self):
        price, delta, *_ = black_scholes(200, 100, 0.1, 0.05, 0.15, 'call')
        intrinsic = 200 - 100 * math.exp(-0.05 * 0.1)
        assert price == pytest.approx(intrinsic, rel=1e-3)
        assert delta == pytest.approx(1.0, abs=1e-3)

    def test_deep_otm_call_price_near_zero(self):
        price, delta, *_ = black_scholes(50, 200, 0.1, 0.05, 0.15, 'call')
        assert price < 1e-6
        assert delta == pytest.approx(0.0, abs=1e-4)
