"""Shared pytest fixtures.

Network access (NSE, yfinance, RBI) is never required to run this suite --
tests either exercise pure math directly, or use the recorded chain fixture
in tests/fixtures/nifty_chain.json in place of a live NSE call. This is
deliberate: the suite must pass identically whether NSE is up, whether
markets are open, and on any day the option chain in the fixture has since
expired.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')


@pytest.fixture
def app():
    import api.app as app_module
    app_module.app.config.update(TESTING=True)
    return app_module.app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def nifty_chain_raw():
    """The recorded live NIFTY chain: real strikes, real bid/ask/OI/IV.

    Includes both liquid near-the-money strikes and a handful of deep
    strikes, so tests can exercise the low-liquidity/skip-reason code paths
    against real (not synthetic) noisy data.
    """
    with open(os.path.join(FIXTURES_DIR, 'nifty_chain.json'), encoding='utf-8') as f:
        return json.load(f)
