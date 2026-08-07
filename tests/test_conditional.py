"""Regime-conditional VaR/correlation tests. Alignment and partitioning
logic is checked exactly (deterministic). The VaR/correlation shift tests
inject a real, controllable regime-dependent structure (different variance
or correlation per labeled segment) and check the conditional estimates
recover the correct direction, since real market data wouldn't give
predictable expected values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.conditional import (
    MIN_DAYS_FOR_CONDITIONAL_VAR,
    align_regime_labels,
    compare_pooled_vs_conditional,
    regime_conditional_correlation,
    regime_conditional_var,
    split_by_regime,
)


def test_align_regime_labels_does_not_forward_fill_missing_dates():
    pnl_index = pd.bdate_range("2024-01-01", periods=5)
    regime_labels = pd.Series(
        ["calm", "calm", "volatile"], index=pd.bdate_range("2024-01-01", periods=3)
    )
    aligned = align_regime_labels(pnl_index, regime_labels)
    assert aligned.iloc[3:].isna().all()  # dates 4-5 have no label -- must stay NaN, not ffilled
    assert list(aligned.iloc[:3]) == ["calm", "calm", "volatile"]


def test_split_by_regime_partitions_correctly():
    dates = pd.bdate_range("2024-01-01", periods=6)
    series = pd.Series(range(6), index=dates)
    labels = pd.Series(["calm", "calm", "volatile", "volatile", "normal", np.nan], index=dates)
    result = split_by_regime(series, labels)
    assert list(result["calm"]) == [0, 1]
    assert list(result["volatile"]) == [2, 3]
    assert list(result["normal"]) == [4]
    assert "nan" not in result  # NaN-labeled days are dropped, not their own bucket


def test_regime_conditional_var_skips_regimes_with_too_few_days():
    rng = np.random.default_rng(0)
    n_ok, n_short = MIN_DAYS_FOR_CONDITIONAL_VAR + 10, MIN_DAYS_FOR_CONDITIONAL_VAR - 5
    dates = pd.bdate_range("2024-01-01", periods=n_ok + n_short)
    pnl = pd.Series(rng.normal(0, 100, n_ok + n_short), index=dates)
    labels = pd.Series(["calm"] * n_ok + ["volatile"] * n_short, index=dates)

    results, day_counts, notes = regime_conditional_var(pnl, labels)
    assert "calm" in results
    assert "volatile" not in results
    assert day_counts["volatile"] == n_short
    assert any("volatile" in n and "too few" in n for n in notes)


def test_regime_conditional_var_reflects_injected_variance_difference():
    rng = np.random.default_rng(1)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=2 * n)
    calm_pnl = rng.normal(0, 100, n)
    volatile_pnl = rng.normal(0, 1000, n)
    pnl = pd.Series(np.concatenate([calm_pnl, volatile_pnl]), index=dates)
    labels = pd.Series(["calm"] * n + ["volatile"] * n, index=dates)

    results, _, _ = regime_conditional_var(pnl, labels, confidence=0.95)
    calm_var = results["calm"]["historical_simulation"].var_dollar
    volatile_var = results["volatile"]["historical_simulation"].var_dollar
    assert volatile_var > calm_var * 3  # ~10x sigma ratio should show up clearly


def test_regime_conditional_correlation_recovers_injected_shift():
    rng = np.random.default_rng(2)
    n = 200
    calm = rng.multivariate_normal([0, 0], [[1, 0.1], [0.1, 1]], size=n)
    volatile = rng.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], size=n)
    dates = pd.bdate_range("2024-01-01", periods=2 * n)
    returns_df = pd.DataFrame(np.vstack([calm, volatile]), index=dates, columns=["A", "B"])
    labels = pd.Series(["calm"] * n + ["volatile"] * n, index=dates)

    results, notes = regime_conditional_correlation(returns_df, labels)
    assert results["calm"].loc["A", "B"] == pytest.approx(0.1, abs=0.1)
    assert results["volatile"].loc["A", "B"] == pytest.approx(0.8, abs=0.1)
    assert results["volatile"].loc["A", "B"] > results["calm"].loc["A", "B"]


def test_compare_pooled_vs_conditional_structural_invariants():
    rng = np.random.default_rng(3)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=2 * n)
    pnl = pd.Series(np.concatenate([rng.normal(0, 100, n), rng.normal(0, 500, n)]), index=dates)
    returns_df = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 2 * n), "B": rng.normal(0, 0.01, 2 * n)}, index=dates
    )
    labels = pd.Series(["calm"] * n + ["volatile"] * n, index=dates)

    result = compare_pooled_vs_conditional(pnl, returns_df, labels)
    assert set(result.pooled_var.keys()) == {"historical_simulation", "parametric_normal", "cornish_fisher"}
    assert set(result.conditional_var.keys()) == {"calm", "volatile"}
    assert result.pooled_correlation.shape == (2, 2)
    assert set(result.conditional_correlation.keys()) == {"calm", "volatile"}
    assert result.regime_day_counts == {"calm": n, "volatile": n}


def test_end_to_end_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.alpaca_market_data import fetch_history
    from connectors.registry import fetch_all
    from regime.volatility_tercile import classify_regimes, rolling_realized_vol
    from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    spy_close = fetch_history(["SPY"], period="2y", field="close")["SPY"]
    regime_labels = classify_regimes(rolling_realized_vol(spy_close)).labels

    result = compare_pooled_vs_conditional(pnl_series, returns_df, regime_labels)
    assert len(result.pooled_var) == 3
    for methods in result.conditional_var.values():
        for r in methods.values():
            assert r.var_dollar == r.var_dollar  # not NaN
    for corr in result.conditional_correlation.values():
        assert corr.shape == result.pooled_correlation.shape
