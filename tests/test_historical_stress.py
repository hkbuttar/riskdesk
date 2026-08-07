"""Historical scenario replay tests. Replay math and diversification-ratio
math are checked against constructed positions/price series with hand-
computable expected values. VaR breach counting is checked against a
synthetic pre-window/window split with an injected, known number of
exceedances. One end-to-end test runs the real book against the real
disclosed historical windows and asserts structural invariants only.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from stress.historical import (
    check_diversification_erosion,
    replay_window,
    validate_regime_conditional_var,
)

TODAY = dt.date(2026, 8, 7)


def _position(strategy, asset, market_value):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=AssetClass.EQUITY, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_replay_window_computes_weighted_pnl_exactly():
    positions = [_position("a", "AAPL", 1000.0), _position("b", "MSFT", -500.0)]
    dates = pd.bdate_range("2024-01-01", periods=4)
    prices = pd.DataFrame({"AAPL": [100, 101, 99, 102], "MSFT": [200, 202, 198, 204]}, index=dates)

    result = replay_window(positions, prices)
    returns = prices.pct_change().dropna()
    expected_daily = returns["AAPL"] * 1000.0 + returns["MSFT"] * -500.0
    pd.testing.assert_series_equal(result.daily_pnl, expected_daily, check_names=False)
    assert result.total_pnl == pytest.approx(expected_daily.sum())


def test_replay_window_returns_none_for_empty_price_history():
    positions = [_position("a", "AAPL", 1000.0)]
    empty = pd.DataFrame({"AAPL": []})
    assert replay_window(positions, empty) is None


def test_replay_window_excludes_unpriced_positions():
    positions = [_position("a", "AAPL", None), _position("b", "MSFT", 1000.0)]
    dates = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({"AAPL": [100, 105, 110], "MSFT": [200, 201, 199]}, index=dates)
    result = replay_window(positions, prices)
    returns = prices.pct_change().dropna()
    expected = returns["MSFT"] * 1000.0
    pd.testing.assert_series_equal(result.daily_pnl, expected, check_names=False)


def test_diversification_erosion_ratio_for_perfectly_correlated_strategies():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(0)
    base = pd.Series(rng.normal(0, 100, 100), index=dates)
    # Two strategies with identical P&L -- perfectly correlated.
    strategy_pnl = {"a": base, "b": base}
    result = check_diversification_erosion(strategy_pnl)
    assert result.diversification_erosion_ratio == pytest.approx(np.sqrt(2), rel=0.05)


def test_diversification_erosion_ratio_for_perfectly_offsetting_strategies():
    dates = pd.bdate_range("2024-01-01", periods=100)
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 100, 100), index=dates)
    strategy_pnl = {"a": base, "b": -base}
    result = check_diversification_erosion(strategy_pnl)
    assert result.realized_portfolio_vol == pytest.approx(0.0, abs=1e-6)
    assert result.diversification_erosion_ratio == pytest.approx(0.0, abs=1e-6)


def test_diversification_erosion_returns_none_with_fewer_than_two_strategies():
    dates = pd.bdate_range("2024-01-01", periods=10)
    result = check_diversification_erosion({"a": pd.Series(range(10), index=dates)})
    assert result is None


def test_validate_regime_conditional_var_skips_when_too_few_volatile_days():
    dates = pd.bdate_range("2024-01-01", periods=50)
    pre_pnl = pd.Series(np.random.default_rng(2).normal(0, 100, 50), index=dates)
    pre_spy = pd.Series(np.linspace(100, 110, 50), index=dates)  # smooth, low-vol -> few "volatile" days
    window_pnl = pd.Series(np.random.default_rng(3).normal(0, 100, 10), index=pd.bdate_range("2024-03-15", periods=10))

    result = validate_regime_conditional_var(pre_pnl, pre_spy, window_pnl)
    assert result.conditional_var is None
    assert "too few" in result.conditional_label


def test_validate_regime_conditional_var_counts_breaches_correctly():
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2020-01-01", periods=600)
    pre_pnl = pd.Series(rng.normal(0, 100, 600), index=dates)
    # Give SPY genuine volatility variation so terciles produce >=30 "volatile" days.
    pre_spy = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 600))), index=dates)

    window_dates = pd.bdate_range("2021-08-01", periods=20)
    # Inject exactly 3 days far beyond any plausible VaR bound.
    window_pnl = pd.Series(rng.normal(0, 50, 20), index=window_dates)
    window_pnl.iloc[:3] = -100_000.0

    result = validate_regime_conditional_var(pre_pnl, pre_spy, window_pnl)
    assert result.pooled_breaches >= 3
    assert result.n_window_days == 20


def test_end_to_end_on_real_book_and_real_windows():
    from aggregation.valuation import value_positions
    from aggregation.pricing import resolve_symbol
    from connectors.registry import fetch_all
    from risk_measures.returns import position_risk_factor
    from stress.historical import HISTORICAL_WINDOWS, fetch_price_history

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    risk_factors = {position_risk_factor(p) for p in valued.positions if position_risk_factor(p)}
    tickers = sorted({resolve_symbol(f) for f in risk_factors} | {"SPY"})

    window = HISTORICAL_WINDOWS["ftx_collapse"]  # shortest window -- fastest test
    history = fetch_price_history(tickers, window["pre_start"], window["end"])
    window_prices = history.loc[window["start"] : window["end"]]

    result = replay_window(valued.positions, window_prices)
    assert result is not None
    assert result.total_pnl == result.total_pnl  # not NaN

    pre_prices = history.loc[window["pre_start"] : window["start"]]
    pre_result = replay_window(valued.positions, pre_prices)
    assert pre_result is not None
    comparison = validate_regime_conditional_var(pre_result.daily_pnl, pre_prices["SPY"], result.daily_pnl)
    assert comparison.pooled_breaches >= 0
