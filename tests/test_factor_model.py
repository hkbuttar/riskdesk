"""Factor model tests. Regression and PCA are checked against synthetic
data with known/injected structure (a specific loading, a dominant
correlated cluster). Sector tagging is checked against a few tickers known
to be in alpha-signal-lab's own SECTOR_TICKERS. Vega aggregation is checked
against constructed positions. One end-to-end test runs the real pipeline
and asserts structural invariants only.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from factor_model.factors import sector_of
from factor_model.pca import fit_pca, top_loadings
from factor_model.regression import fit_factor_regression, significant_factors
from factor_model.vega import aggregate_vega

TODAY = dt.date(2026, 8, 7)


def test_sector_of_known_tickers():
    assert sector_of("NVDA") == "Technology"
    assert sector_of("CVX") == "Energy"
    assert sector_of("WFC") == "Financials"
    assert sector_of("PFE") == "Healthcare"


def test_sector_of_unknown_ticker_is_none():
    assert sector_of("NOT_A_REAL_TICKER") is None


def test_fit_factor_regression_recovers_injected_loading():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=500)
    factor_return = rng.normal(0, 0.01, 500)
    noise = rng.normal(0, 10, 500)
    pnl = 5000.0 * factor_return + noise  # true loading = 5000

    factor_returns = pd.DataFrame({"FACTOR": factor_return}, index=dates)
    pnl_series = pd.Series(pnl, index=dates)

    result = fit_factor_regression("test", pnl_series, factor_returns)
    assert result.loadings["FACTOR"] == pytest.approx(5000.0, rel=0.05)
    assert result.r_squared > 0.9
    assert "FACTOR" in significant_factors(result)


def test_fit_factor_regression_low_r_squared_for_unrelated_factor():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=500)
    pnl_series = pd.Series(rng.normal(0, 1000, 500), index=dates)
    unrelated_factor = pd.DataFrame({"FACTOR": rng.normal(0, 0.01, 500)}, index=dates)

    result = fit_factor_regression("test", pnl_series, unrelated_factor)
    assert result.r_squared < 0.05
    assert "FACTOR" not in significant_factors(result)


def test_fit_factor_regression_perfect_linear_relationship_gives_r_squared_near_one():
    # Mirrors the real bookmaker/voledge case: a single-underlying position's
    # proxied P&L is an exact multiple of its own risk factor by construction.
    dates = pd.bdate_range("2024-01-01", periods=200)
    factor_return = np.linspace(-0.02, 0.02, 200)
    pnl_series = pd.Series(1234.0 * factor_return, index=dates)
    factor_returns = pd.DataFrame({"FACTOR": factor_return}, index=dates)

    result = fit_factor_regression("degenerate", pnl_series, factor_returns)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)
    assert result.loadings["FACTOR"] == pytest.approx(1234.0, rel=1e-3)


def test_pca_explained_variance_sums_to_one():
    rng = np.random.default_rng(2)
    returns_df = pd.DataFrame(rng.normal(0, 0.01, size=(300, 6)), columns=list("abcdef"))
    result = fit_pca(returns_df)
    assert result.explained_variance_ratio.sum() == pytest.approx(1.0, abs=1e-6)
    assert 1 <= result.n_components_for_90pct <= 6


def test_pca_first_component_dominates_for_highly_correlated_cluster():
    rng = np.random.default_rng(3)
    common_factor = rng.normal(0, 0.01, 500)
    returns_df = pd.DataFrame(
        {name: common_factor + rng.normal(0, 0.001, 500) for name in ["a", "b", "c", "d", "e"]}
    )
    result = fit_pca(returns_df)
    assert result.explained_variance_ratio["PC1"] > 0.8
    assert result.n_components_for_90pct <= 2
    loadings = top_loadings(result, "PC1", n=5)
    assert set(loadings.index) == {"a", "b", "c", "d", "e"}


def _option_position(strategy, asset, quantity, vega):
    return Position(
        strategy=strategy, asset=asset, quantity=quantity, market_value=100.0,
        asset_class=AssetClass.OPTION, counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN, as_of=TODAY,
        greeks={"delta": 0.5, "gamma": 0.01, "vega": vega, "theta": -1.0, "rho": 0.1},
        extra={"underlying": "SPY"},
    )


def _equity_position(strategy, asset):
    return Position(
        strategy=strategy, asset=asset, quantity=10.0, market_value=1000.0,
        asset_class=AssetClass.EQUITY, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_aggregate_vega_sums_quantity_times_vega():
    positions = [
        _option_position("voledge", "SPY C500", 2.0, vega=10.0),
        _option_position("voledge", "SPY P500", -1.0, vega=5.0),
    ]
    result = aggregate_vega(positions)
    assert result.net_vega == pytest.approx(2.0 * 10.0 + -1.0 * 5.0)
    assert result.n_option_positions == 2


def test_aggregate_vega_ignores_non_option_positions():
    positions = [_option_position("voledge", "SPY C500", 1.0, vega=10.0), _equity_position("a", "AAPL")]
    result = aggregate_vega(positions)
    assert result.n_option_positions == 1
    assert result.net_vega == pytest.approx(10.0)


def test_end_to_end_factor_decomposition_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from factor_model.factors import fetch_factor_returns
    from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    result = fit_factor_regression("portfolio", pnl_series, factor_returns)
    assert 0.0 <= result.r_squared <= 1.0 + 1e-9
    assert result.n_obs > 0

    vega = aggregate_vega(valued.positions)
    assert vega.n_option_positions >= 0
