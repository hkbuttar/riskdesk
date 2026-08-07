"""Liquidity/concentration tests. Impact cost is checked against the exact
square-root-law formula by hand, including the textbook property that cost
scales with the square root (not linearly) of position size. Concentration
is checked against constructed positions with known exposures, including
the deliberate net-vs-gross difference from credit/concentration.py. One
end-to-end test runs the real pipeline for structural invariants.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from liquidity.concentration import check_by_name, check_by_sector, check_by_strategy
from liquidity.impact import SQRT_LAW_Y, estimate_liquidation_cost, liquidity_adjusted_var

TODAY = dt.date(2026, 8, 7)


def test_estimate_liquidation_cost_matches_formula():
    market_value, sigma, dollar_volume = 10000.0, 0.02, 1_000_000.0
    cost = estimate_liquidation_cost(market_value, sigma, dollar_volume)
    expected_participation = market_value / dollar_volume
    expected_fraction = SQRT_LAW_Y * sigma * np.sqrt(expected_participation)
    assert cost.participation_rate == pytest.approx(expected_participation)
    assert cost.cost_fraction == pytest.approx(expected_fraction)
    assert cost.dollar_cost == pytest.approx(market_value * expected_fraction)


def test_estimate_liquidation_cost_scales_with_sqrt_of_position_size():
    small = estimate_liquidation_cost(10000.0, 0.02, 1_000_000.0)
    large = estimate_liquidation_cost(40000.0, 0.02, 1_000_000.0)  # 4x the size
    # cost_fraction ~ sqrt(participation_rate) ~ sqrt(size) -> 4x size -> 2x cost_fraction.
    assert large.cost_fraction == pytest.approx(small.cost_fraction * 2, rel=1e-6)


def test_estimate_liquidation_cost_returns_none_for_zero_volume():
    assert estimate_liquidation_cost(10000.0, 0.02, 0.0) is None


def test_estimate_liquidation_cost_uses_absolute_market_value():
    long_cost = estimate_liquidation_cost(10000.0, 0.02, 1_000_000.0)
    short_cost = estimate_liquidation_cost(-10000.0, 0.02, 1_000_000.0)
    assert long_cost.dollar_cost == pytest.approx(short_cost.dollar_cost)


def _position(strategy, asset, market_value, asset_class=AssetClass.EQUITY, extra=None):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=asset_class, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY, extra=extra or {},
    )


def test_liquidity_adjusted_var_adds_costs_to_base():
    positions = [_position("a", "AAPL", 10000.0), _position("b", "MSFT", 5000.0)]
    vol = {"AAPL": 0.02, "MSFT": 0.015}
    dollar_vol = {"AAPL": 1_000_000.0, "MSFT": 2_000_000.0}

    adjusted, costs, notes = liquidity_adjusted_var(1000.0, positions, vol, dollar_vol)
    expected_total_cost = sum(c.dollar_cost for c in costs.values())
    assert adjusted == pytest.approx(1000.0 + expected_total_cost)
    assert len(costs) == 2
    assert notes == [] or all("excluded" not in n for n in notes)


def test_liquidity_adjusted_var_excludes_synthetic_positions():
    positions = [_position("a", "SIM", 5000.0, asset_class=AssetClass.SYNTHETIC)]
    adjusted, costs, notes = liquidity_adjusted_var(1000.0, positions, {}, {})
    assert adjusted == pytest.approx(1000.0)
    assert costs == {}


def test_liquidity_adjusted_var_discloses_missing_data():
    positions = [_position("a", "UNKNOWN_TICKER", 5000.0)]
    adjusted, costs, notes = liquidity_adjusted_var(1000.0, positions, {}, {})
    assert adjusted == pytest.approx(1000.0)
    assert any("no volume/volatility data" in n for n in notes)


def test_liquidity_adjusted_var_resolves_option_underlying():
    positions = [
        _position("v", "SPY C500", 100.0, asset_class=AssetClass.OPTION, extra={"underlying": "SPY"})
    ]
    vol = {"SPY": 0.015}
    dollar_vol = {"SPY": 5_000_000_000.0}
    adjusted, costs, notes = liquidity_adjusted_var(1000.0, positions, vol, dollar_vol)
    assert "v/SPY C500" in costs


def test_check_by_name_flags_dominant_ticker():
    positions = [_position("a", "AXP", 9000.0), _position("a", "WFC", 1000.0)]
    result = check_by_name(positions, threshold=0.5)
    assert "AXP" in result.flagged
    assert result.shares["AXP"] == pytest.approx(0.9)


def test_check_by_strategy_uses_gross_not_net():
    # Two offsetting positions in the same strategy -- gross should sum
    # both magnitudes, not net them to near-zero (contrast with
    # credit/concentration.py's deliberate NET choice for counterparty risk).
    positions = [_position("a", "X", 10000.0), _position("a", "Y", -9000.0)]
    result = check_by_strategy(positions, threshold=0.5)
    assert result.exposures["a"] == pytest.approx(19000.0)


def test_check_by_sector_labels_unrecognized_tickers_unclassified():
    positions = [_position("a", "NOT_A_REAL_TICKER", 1000.0), _position("b", "CVX", 1000.0)]
    result = check_by_sector(positions)
    assert "unclassified" in result.exposures
    assert "Energy" in result.exposures


def test_concentration_check_empty_for_no_exposure():
    positions = [_position("a", "X", None)]
    result = check_by_name(positions)
    assert result.total_exposure == 0.0
    assert result.flagged == []


def test_end_to_end_liquidity_and_concentration_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from liquidity.impact import fetch_avg_daily_dollar_volume
    from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor
    from risk_measures.var import historical_simulation

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    daily_vol = returns_df.std().to_dict()
    dollar_volume = fetch_avg_daily_dollar_volume(tickers)

    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)
    base_var = historical_simulation(pnl_series).var_dollar

    adjusted, costs, _ = liquidity_adjusted_var(base_var, valued.positions, daily_vol, dollar_volume)
    assert adjusted >= base_var  # unwind cost should never be negative

    for check_fn in (check_by_name, check_by_strategy, check_by_sector):
        result = check_fn(valued.positions)
        if result.total_exposure > 0:
            assert abs(sum(result.shares.values()) - 1.0) < 1e-6
