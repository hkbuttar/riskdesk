"""P&L attribution tests. Factor attribution is checked for exact
reconciliation (the defining property a real attribution must have) and
against hand-computable single-factor contributions. Strategy attribution
is checked for additivity (per-strategy P&L must sum back to the whole
portfolio's, since P&L is linear by construction). One end-to-end test
runs the real pipeline and checks reconciliation on the real book.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from attribution.pnl import attribute_by_factor
from attribution.strategy import attribute_by_strategy
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position

TODAY = dt.date(2026, 8, 7)


def test_attribute_by_factor_matches_hand_computed_single_factor_contribution():
    dates = pd.bdate_range("2024-01-01", periods=5)
    factor_returns = pd.DataFrame({"A": [0.01, -0.02, 0.005, 0.03, -0.01]}, index=dates)
    loadings = {"A": 1000.0}
    alpha = 5.0
    pnl_series = factor_returns["A"] * 1000.0 + alpha + pd.Series([1, -1, 2, -2, 0], index=dates)

    result = attribute_by_factor(pnl_series, factor_returns, loadings, alpha)
    expected_contribution = float((factor_returns["A"] * 1000.0).sum())
    assert result.cumulative_by_factor["A"] == pytest.approx(expected_contribution)
    assert result.cumulative_alpha == pytest.approx(alpha * 5)


def test_attribute_by_factor_reconciles_exactly():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=200)
    factor_returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 200), "B": rng.normal(0, 0.02, 200)}, index=dates
    )
    loadings = {"A": 5000.0, "B": -2000.0}
    alpha = 10.0
    noise = pd.Series(rng.normal(0, 50, 200), index=dates)
    pnl_series = factor_returns["A"] * loadings["A"] + factor_returns["B"] * loadings["B"] + alpha + noise

    result = attribute_by_factor(pnl_series, factor_returns, loadings, alpha)
    assert result.reconciles()
    assert result.cumulative_total == pytest.approx(float(pnl_series.sum()))


def test_attribute_by_factor_residual_captures_noise_exactly_for_known_construction():
    dates = pd.bdate_range("2024-01-01", periods=50)
    factor_returns = pd.DataFrame({"A": np.linspace(-0.01, 0.01, 50)}, index=dates)
    loadings = {"A": 100.0}
    alpha = 0.0
    injected_noise = pd.Series(np.arange(50) * 0.1, index=dates)
    pnl_series = factor_returns["A"] * 100.0 + injected_noise

    result = attribute_by_factor(pnl_series, factor_returns, loadings, alpha)
    pd.testing.assert_series_equal(result.residual, injected_noise, check_names=False, check_exact=False)


def _position(strategy, asset, market_value):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=AssetClass.EQUITY, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_attribute_by_strategy_sums_match_whole_portfolio():
    from risk_measures.returns import build_portfolio_pnl_series

    positions = [
        _position("a", "AAPL", 1000.0),
        _position("a", "MSFT", -500.0),
        _position("b", "GOOG", 2000.0),
    ]
    dates = pd.bdate_range("2024-01-01", periods=10)
    rng = np.random.default_rng(1)
    returns_df = pd.DataFrame(
        {"AAPL": rng.normal(0, 0.01, 10), "MSFT": rng.normal(0, 0.01, 10), "GOOG": rng.normal(0, 0.01, 10)},
        index=dates,
    )

    by_strategy = attribute_by_strategy(positions, returns_df)
    whole_portfolio_pnl, _, _ = build_portfolio_pnl_series(positions, returns_df)

    summed = sum(by_strategy.values())
    pd.testing.assert_series_equal(summed, whole_portfolio_pnl, check_names=False)


def test_attribute_by_strategy_excludes_strategies_with_no_mappable_positions():
    positions = [_position("a", "AAPL", 1000.0), _position("b", "NO_SUCH_TICKER", 500.0)]
    dates = pd.bdate_range("2024-01-01", periods=5)
    returns_df = pd.DataFrame({"AAPL": np.random.default_rng(2).normal(0, 0.01, 5)}, index=dates)

    by_strategy = attribute_by_strategy(positions, returns_df)
    assert "a" in by_strategy
    assert "b" not in by_strategy


def test_end_to_end_attribution_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from factor_model.factors import fetch_factor_returns, sector_of
    from factor_model.regression import fit_factor_regression
    from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)

    recent = pnl_series.iloc[-63:]
    result = attribute_by_factor(recent, factor_returns, fit.loadings, fit.alpha)
    assert result.reconciles()

    by_strategy = attribute_by_strategy(valued.positions, returns_df)
    assert len(by_strategy) > 0
    for series in by_strategy.values():
        assert len(series) == len(pnl_series)
