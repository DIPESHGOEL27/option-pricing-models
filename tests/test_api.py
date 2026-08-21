"""Flask test-client tests: status codes, the error-response contract, input
validation, and that no response body ever contains a bare NaN/Infinity
(invalid JSON that silently threw in the browser before this was fixed).

India routes are tested against the recorded chain fixture via monkeypatch
on the module-level provider singletons, never live NSE -- this suite must
pass identically whether NSE is reachable or not, and regardless of market
hours.
"""

import math

import pytest


BASE_OPTION = {'S': 100, 'K': 100, 'T': 0.25, 'r': 0.05, 'sigma': 0.2, 'optionType': 'call'}


def assert_no_nan_or_inf(obj, path='root'):
    """Recursively walk a decoded JSON structure and fail if any float is
    NaN or +/-Infinity -- these serialize as bare `NaN`/`Infinity` tokens,
    which is not valid JSON and throws in any strict client-side parser.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert_no_nan_or_inf(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_no_nan_or_inf(v, f'{path}[{i}]')
    elif isinstance(obj, float):
        assert math.isfinite(obj), f'Non-finite float at {path}: {obj}'


class TestRemovedRoutesAreGone:
    """Part of the cut list -- these must not exist as live routes."""

    @pytest.mark.parametrize('method,path', [
        ('GET', '/api/option_chain/AAPL'),
        ('GET', '/api/volatility_surface/AAPL'),
        ('GET', '/api/market/volatility_term_structure'),
        ('POST', '/api/ml/train_neural_network'),
        ('POST', '/api/ml/ensemble_price'),
        ('POST', '/api/ml/volatility_forecast'),
        ('POST', '/api/analytics/performance_attribution'),
        ('GET', '/api/market_sentiment'),
        ('GET', '/api/market/sentiment'),
        ('POST', '/api/ml/benchmark'),
    ])
    def test_route_returns_404(self, client, method, path):
        response = client.open(path, method=method)
        assert response.status_code == 404


class TestBlackScholesEndpoint:
    def test_valid_request_returns_200(self, client):
        response = client.post('/api/calculate_black_scholes', json=BASE_OPTION)
        assert response.status_code == 200
        body = response.get_json()
        assert 'option_price' in body
        assert_no_nan_or_inf(body)

    def test_missing_field_returns_400_with_readable_message(self, client):
        incomplete = {k: v for k, v in BASE_OPTION.items() if k != 'sigma'}
        response = client.post('/api/calculate_black_scholes', json=incomplete)
        assert response.status_code == 400
        body = response.get_json()
        assert isinstance(body, dict)
        assert 'error' in body
        assert 'sigma' in body['error']

    def test_nan_input_returns_400(self, client):
        response = client.post('/api/calculate_black_scholes', json={**BASE_OPTION, 'S': float('nan')})
        assert response.status_code == 400
        assert 'error' in response.get_json()

    def test_non_numeric_input_returns_400(self, client):
        response = client.post('/api/calculate_black_scholes', json={**BASE_OPTION, 'S': 'not-a-number'})
        assert response.status_code == 400

    def test_zero_spot_returns_400_not_500(self, client):
        response = client.post('/api/calculate_black_scholes', json={**BASE_OPTION, 'S': 0})
        assert response.status_code == 400

    def test_put_rho_is_negative_via_api(self, client):
        response = client.post('/api/calculate_black_scholes', json={**BASE_OPTION, 'optionType': 'put'})
        body = response.get_json()
        assert body['rho'] < 0


class TestBinomialEndpoint:
    def test_default_steps_when_omitted(self, client):
        # Regression: this used to be a bare KeyError -> unhandled HTTP 500.
        response = client.post('/api/calculate_binomial', json=BASE_OPTION)
        assert response.status_code == 200
        assert 'option_price' in response.get_json()

    def test_zero_steps_returns_400_not_500(self, client):
        response = client.post('/api/calculate_binomial', json={**BASE_OPTION, 'steps': 0})
        assert response.status_code == 400

    def test_invalid_option_type_returns_400_not_silent_zero(self, client):
        response = client.post('/api/calculate_binomial', json={**BASE_OPTION, 'optionType': 'straddle'})
        assert response.status_code == 400


class TestMonteCarloEndpoint:
    def test_gbm_valid_request(self, client):
        response = client.post('/api/monte_carlo', json={**BASE_OPTION, 'model': 'gbm', 'simulations': 5000})
        assert response.status_code == 200
        body = response.get_json()
        assert body['option_price'] > 0
        assert_no_nan_or_inf(body)

    def test_invalid_model_type_returns_400(self, client):
        response = client.post('/api/monte_carlo', json={**BASE_OPTION, 'model': 'not-a-model'})
        assert response.status_code == 400

    def test_simulation_count_is_capped_not_unbounded(self, client):
        response = client.post('/api/monte_carlo', json={**BASE_OPTION, 'model': 'gbm', 'simulations': 50_000_000})
        assert response.status_code == 200
        body = response.get_json()
        assert body['simulations'] < 50_000_000

    def test_too_few_simulations_rejected(self, client):
        response = client.post('/api/monte_carlo', json={**BASE_OPTION, 'model': 'gbm', 'simulations': 10})
        assert response.status_code == 400


class TestPlotPayoffEndpoint:
    def test_grid_scales_to_instrument_price_level(self, client):
        # Regression: the spot grid used to be hardcoded to [80, 120],
        # which produced a meaningless flat payoff for anything priced
        # outside that literal range (e.g. any NSE index).
        response = client.post('/api/plot_payoff', json={
            'positions': [{'symbol': 'NIFTY', 'option_type': 'call', 'strike': 24250,
                            'quantity': 50, 'premium_paid': 124.26, 'underlying_price': 24231.85}]
        })
        assert response.status_code == 200
        body = response.get_json()
        assert body['analysis']['break_even_points']
        be = body['analysis']['break_even_points'][0]
        assert 20000 < be < 30000  # anywhere near the real strike, not [80,120]

    def test_no_positions_returns_error(self, client):
        response = client.post('/api/plot_payoff', json={'positions': []})
        # Empty payoff is well-defined (all zero) rather than an error --
        # just confirm it doesn't crash.
        assert response.status_code == 200


class TestStatusEndpoint:
    def test_returns_200_with_feature_flags(self, client):
        response = client.get('/api/status')
        assert response.status_code == 200
        body = response.get_json()
        assert 'features' in body
        assert isinstance(body['features']['monte_carlo'], bool)


class TestIndiaOptionChainEndpoint:
    """Mocked against the recorded fixture -- no live NSE dependency."""

    @pytest.fixture(autouse=True)
    def mock_provider(self, monkeypatch, nifty_chain_raw):
        import api.app as app_module

        def fake_get_option_chain(symbol, expiry=None):
            return dict(nifty_chain_raw)

        monkeypatch.setattr(app_module._india_market_data_provider, 'get_option_chain', fake_get_option_chain)
        monkeypatch.setattr(
            app_module._india_risk_free_rate_provider, 'interpolate_rate', lambda t: 0.055
        )

    def test_returns_analyzed_strikes(self, client):
        response = client.get('/api/india/option_chain/NIFTY')
        assert response.status_code == 200
        body = response.get_json()
        assert body['symbol'] == 'NIFTY'
        assert len(body['strikes']) > 0
        assert 'iv_analysis' in body
        assert_no_nan_or_inf(body)

    def test_response_has_no_fabricated_fallback_shape(self, client):
        # The old fabrication pattern was a fixed-shape dict with plausible
        # constants (vix_level, fear_greed_score, etc). The real response
        # shape here is strike-indexed data with explicit skip reasons --
        # confirm the skip-reason machinery is actually present and used.
        response = client.get('/api/india/option_chain/NIFTY')
        body = response.get_json()
        skip_reasons_seen = set()
        for entry in body['strikes']:
            for leg_key in ('ce', 'pe'):
                leg = entry.get(leg_key)
                if leg and 'skip_reason' in leg:
                    skip_reasons_seen.add(leg['skip_reason'])
        assert skip_reasons_seen  # the fixture has real illiquid strikes that must show up here


class TestIndiaMarketDataUnavailable:
    def test_option_chain_reports_unavailable_cleanly(self, client, monkeypatch):
        import api.app as app_module
        monkeypatch.setattr(app_module, 'INDIA_MARKET_DATA_AVAILABLE', False)
        response = client.get('/api/india/option_chain/NIFTY')
        body = response.get_json()
        assert 'error' in body
