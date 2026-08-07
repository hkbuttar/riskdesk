"""Hypothetical stress scenario tests. Repricing is checked against
constructed positions with known Greeks/market values so the exact
expected P&L (including the delta+gamma+vega Taylor expansion) can be
hand-computed. One end-to-end test runs every real scenario on the real
book and asserts structural invariants only.
"""

from __future__ import annotations

import datetime as dt

import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from stress.hypothetical import HYPOTHETICAL_SCENARIOS, reprice_position, run_scenario

TODAY = dt.date(2026, 8, 7)

TEST_SCENARIO = {
    "equity_pct": -0.20, "crypto_pct": -0.40, "vol_shock_pct": 0.50,
    "sector_overrides": {"Energy": -0.30},
}


def _equity(asset, market_value):
    return Position(
        strategy="a", asset=asset, quantity=1.0, market_value=market_value,
        asset_class=AssetClass.EQUITY, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def _crypto(market_value):
    return Position(
        strategy="b", asset="BTC-USD", quantity=1.0, market_value=market_value,
        asset_class=AssetClass.CRYPTO, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def _option(quantity, delta, gamma, vega, entry_iv, spot):
    return Position(
        strategy="v", asset="SPY C500", quantity=quantity, market_value=100.0, price=spot,
        asset_class=AssetClass.OPTION, counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN, as_of=TODAY,
        greeks={"delta": delta, "gamma": gamma, "vega": vega, "theta": 0.0, "rho": 0.0},
        extra={"underlying": "SPY", "entry_iv": entry_iv},
    )


def _synthetic():
    return Position(
        strategy="c", asset="SIM", quantity=1.0, market_value=500.0,
        asset_class=AssetClass.SYNTHETIC, counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN, as_of=TODAY,
    )


def test_reprice_equity_uses_broad_shock_when_no_sector_override():
    p = _equity("WFC", 1000.0)  # Financials, no override in TEST_SCENARIO
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == linear == pytest.approx(1000.0 * -0.20)


def test_reprice_equity_uses_sector_override_when_present():
    p = _equity("CVX", 1000.0)  # Energy, TEST_SCENARIO overrides to -30%
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == linear == pytest.approx(1000.0 * -0.30)


def test_reprice_crypto_uses_crypto_shock():
    p = _crypto(1000.0)
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == linear == pytest.approx(1000.0 * -0.40)


def test_reprice_synthetic_position_is_zero():
    p = _synthetic()
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == 0.0
    assert linear == 0.0


def test_reprice_option_matches_hand_computed_taylor_expansion():
    quantity, delta, gamma, vega, entry_iv, spot = 2.0, 0.5, 0.02, 10.0, 0.20, 500.0
    p = _option(quantity, delta, gamma, vega, entry_iv, spot)
    full, linear = reprice_position(p, TEST_SCENARIO)

    d_s = spot * TEST_SCENARIO["equity_pct"]
    d_sigma = entry_iv * TEST_SCENARIO["vol_shock_pct"]
    expected_linear = quantity * delta * d_s
    expected_full = expected_linear + quantity * (0.5 * gamma * d_s**2 + vega * d_sigma)

    assert linear == pytest.approx(expected_linear)
    assert full == pytest.approx(expected_full)
    # Gamma is positive and vol shock is positive -- convexity should add value here.
    assert full > linear


def test_reprice_option_with_missing_greeks_or_price_is_zero():
    p = Position(
        strategy="v", asset="SPY C500", quantity=1.0, market_value=100.0, price=None,
        asset_class=AssetClass.OPTION, counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN, as_of=TODAY, greeks=None,
    )
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == 0.0 and linear == 0.0


def test_reprice_unpriced_position_is_zero():
    p = _equity("AAPL", None)
    full, linear = reprice_position(p, TEST_SCENARIO)
    assert full == 0.0 and linear == 0.0


def test_run_scenario_aggregates_across_positions():
    positions = [_equity("WFC", 1000.0), _crypto(500.0), _synthetic()]
    result = run_scenario("crypto_specific_crash", positions)
    scenario = HYPOTHETICAL_SCENARIOS["crypto_specific_crash"]
    expected = 1000.0 * scenario["equity_pct"] + 500.0 * scenario["crypto_pct"]
    assert result.total_pnl == pytest.approx(expected)
    assert result.linear_only_pnl == pytest.approx(expected)  # no options -> no convexity correction
    assert result.convexity_correction == pytest.approx(0.0, abs=1e-9)
    assert len(result.by_position) == 3


def test_all_scenario_names_are_runnable_and_produce_finite_pnl():
    positions = [
        _equity("CVX", 1000.0), _crypto(500.0),
        _option(1.0, 0.4, 0.01, 5.0, 0.25, 500.0),
    ]
    for name in HYPOTHETICAL_SCENARIOS:
        result = run_scenario(name, positions)
        assert result.total_pnl == result.total_pnl  # not NaN
        assert result.scenario_name == name


def test_end_to_end_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    for name in HYPOTHETICAL_SCENARIOS:
        result = run_scenario(name, valued.positions)
        assert result.total_pnl == result.total_pnl
        assert result.linear_only_pnl == result.linear_only_pnl
        assert len(result.by_position) == len(valued.positions)
