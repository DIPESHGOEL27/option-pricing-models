"""Tests for api.advanced_models: Monte Carlo pricing and Greeks.

Includes a regression test for a real bug found this session: MC Greeks
used independent random draws per bumped valuation, so gamma (a
second-order, small-signal Greek) came back with the wrong sign and order
of magnitude, and rho was hardcoded to 0.
"""

import numpy as np
import pytest

from api.advanced_models import MonteCarloEngine, RiskMetrics
from api.app import black_scholes


class TestGbmConvergence:
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2

    def test_price_converges_to_analytic(self):
        analytic_price, *_ = black_scholes(self.S, self.K, self.T, self.r, self.sigma, 'call')
        np.random.seed(42)
        engine = MonteCarloEngine(n_simulations=200_000, n_steps=1)
        paths = engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
        result = engine.price_vanilla_option(paths, self.K, self.r, self.T, 'call')
        # 200k paths gives a std error on the order of a few cents for this
        # scenario; allow a generous but real statistical tolerance rather
        # than pinning an exact value that would be flaky by construction.
        assert result['price'] == pytest.approx(analytic_price, abs=0.15)

    def test_single_step_matches_multi_step_terminal_distribution(self):
        # A vanilla payoff only depends on the terminal price, and GBM's
        # terminal distribution is identical whether it's reached in one
        # step or many small ones -- this is the basis for the n_steps=1
        # optimization in /api/monte_carlo (see app.py).
        np.random.seed(7)
        engine_1 = MonteCarloEngine(n_simulations=100_000, n_steps=1)
        price_1 = engine_1.price_vanilla_option(
            engine_1.geometric_brownian_motion(self.S, self.T, self.r, self.sigma),
            self.K, self.r, self.T, 'call',
        )['price']

        np.random.seed(7)
        engine_252 = MonteCarloEngine(n_simulations=100_000, n_steps=252)
        price_252 = engine_252.price_vanilla_option(
            engine_252.geometric_brownian_motion(self.S, self.T, self.r, self.sigma),
            self.K, self.r, self.T, 'call',
        )['price']

        assert price_1 == pytest.approx(price_252, abs=0.25)

    def test_std_error_shrinks_with_more_simulations(self):
        errors = []
        for n in (2_000, 20_000, 200_000):
            np.random.seed(1)
            engine = MonteCarloEngine(n_simulations=n, n_steps=1)
            paths = engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma)
            result = engine.price_vanilla_option(paths, self.K, self.r, self.T, 'call')
            errors.append(result['std_error'])
        assert errors[0] > errors[1] > errors[2]
        # Standard error should shrink roughly as 1/sqrt(n): a 10x increase
        # in simulations should shrink it by roughly sqrt(10) ~ 3.16x.
        ratio = errors[0] / errors[1]
        assert 2.0 < ratio < 4.5

    def test_antithetic_variates_reduce_variance(self):
        np.random.seed(3)
        engine = MonteCarloEngine(n_simulations=50_000, n_steps=1)
        paths_anti = engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma, use_antithetic=True)
        err_anti = engine.price_vanilla_option(paths_anti, self.K, self.r, self.T, 'call')['std_error']

        np.random.seed(3)
        paths_plain = engine.geometric_brownian_motion(self.S, self.T, self.r, self.sigma, use_antithetic=False)
        err_plain = engine.price_vanilla_option(paths_plain, self.K, self.r, self.T, 'call')['std_error']

        assert err_anti < err_plain


class TestMonteCarloGreeksRegression:
    """Regression tests for a real bug: calculate_greeks_mc used independent
    MC draws per bumped valuation, so gamma had the wrong sign/magnitude
    (observed live: -0.526 instead of the analytic +0.039) and rho was
    hardcoded to 0.
    """

    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.05, 0.2

    def test_greeks_converge_to_analytic(self):
        analytic_price, a_delta, a_gamma, a_theta, a_vega, a_rho = black_scholes(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        mc = RiskMetrics.calculate_greeks_mc(
            self.S, self.K, self.T, self.r, self.sigma, 'call', n_simulations=300_000
        )
        assert mc['delta'] == pytest.approx(a_delta, abs=0.02)
        assert mc['gamma'] == pytest.approx(a_gamma, abs=0.01)
        assert mc['gamma'] > 0  # the regressed bug produced a negative gamma
        assert mc['vega'] == pytest.approx(a_vega, abs=1.0)
        assert mc['rho'] == pytest.approx(a_rho, abs=1.0)
        assert mc['rho'] != 0  # the regressed bug hardcoded rho to 0

    def test_put_rho_is_negative(self):
        mc = RiskMetrics.calculate_greeks_mc(
            self.S, self.K, self.T, self.r, self.sigma, 'put', n_simulations=300_000
        )
        assert mc['rho'] < 0

    def test_common_random_numbers_make_gamma_stable_across_calls(self):
        # Before the fix, each bumped valuation drew fresh random numbers,
        # so gamma (a small second-order signal) was dominated by
        # independent sampling noise and varied wildly between otherwise
        # identical calls. With a fixed random_seed reused across every
        # bump, repeated calls should agree closely.
        results = [
            RiskMetrics.calculate_greeks_mc(self.S, self.K, self.T, self.r, self.sigma, 'call', n_simulations=100_000)
            for _ in range(3)
        ]
        gammas = [r['gamma'] for r in results]
        assert max(gammas) - min(gammas) < 0.005
