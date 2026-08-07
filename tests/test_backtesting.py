"""VaR backtesting tests. Kupiec and Christoffersen are checked against
constructed breach series with known statistical properties: an exactly-
calibrated series should not reject, a badly miscalibrated one should, a
deliberately clustered series should fail independence while a scattered
one with the same total breach count should not. One end-to-end test runs
the real pipeline for structural invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_measures.backtesting import (
    backtest_breach_series,
    christoffersen_independence_test,
    kupiec_pof_test,
    pooled_breach_series,
    regime_conditional_breach_series,
)


def test_kupiec_pof_near_zero_lr_for_exact_calibration():
    lr, p_value = kupiec_pof_test(n_obs=1000, n_breaches=50, confidence=0.95)
    assert lr == pytest.approx(0.0, abs=0.01)
    assert p_value > 0.9


def test_kupiec_pof_detects_gross_miscalibration():
    lr, p_value = kupiec_pof_test(n_obs=1000, n_breaches=200, confidence=0.95)  # 20% vs 5% expected
    assert lr > 10.0
    assert p_value < 0.01


def test_kupiec_pof_handles_zero_breaches_without_error():
    lr, p_value = kupiec_pof_test(n_obs=200, n_breaches=0, confidence=0.95)
    assert lr == lr  # not NaN
    assert 0.0 <= p_value <= 1.0


def test_kupiec_pof_handles_all_breaches_without_error():
    lr, p_value = kupiec_pof_test(n_obs=200, n_breaches=200, confidence=0.95)
    assert lr == lr
    assert 0.0 <= p_value <= 1.0


def test_christoffersen_detects_clustering():
    n = 1000
    # Clustered: 50 consecutive breaches, then none -- same total count as scattered.
    clustered = np.zeros(n, dtype=bool)
    clustered[:50] = True

    rng = np.random.default_rng(0)
    scattered_idx = rng.choice(n, size=50, replace=False)
    scattered = np.zeros(n, dtype=bool)
    scattered[scattered_idx] = True

    clustered_lr, clustered_p = christoffersen_independence_test(clustered)
    scattered_lr, scattered_p = christoffersen_independence_test(scattered)

    assert clustered_lr > scattered_lr
    assert clustered_p < 0.01
    assert scattered_p > 0.05


def test_christoffersen_fails_to_reject_for_genuinely_iid_series():
    rng = np.random.default_rng(1)
    breach_series = rng.random(2000) < 0.05
    _, p_value = christoffersen_independence_test(breach_series)
    assert p_value > 0.05


def test_backtest_breach_series_combined_lr_is_sum_of_kupiec_and_christoffersen():
    rng = np.random.default_rng(2)
    series = pd.Series(rng.random(500) < 0.05)
    result = backtest_breach_series(series, confidence=0.95, model_label="test")
    assert result.combined_lr == pytest.approx(result.kupiec_lr + result.christoffersen_lr)
    assert result.n_breaches == int(series.sum())


def test_pooled_breach_series_rate_matches_confidence_by_construction():
    # Documents the near-tautological property: historical_simulation's VaR
    # IS the empirical quantile of this same series, so the in-sample
    # breach rate is mechanically close to (1 - confidence), not an
    # independent finding about calibration -- see risk_measures/run_backtest.py.
    rng = np.random.default_rng(3)
    pnl = pd.Series(rng.normal(0, 1000, 1000))
    breaches = pooled_breach_series(pnl, confidence=0.95)
    assert breaches.mean() == pytest.approx(0.05, abs=0.01)


def test_regime_conditional_breach_series_falls_back_for_insufficient_regime():
    rng = np.random.default_rng(4)
    n_calm, n_thin = 300, 10
    dates = pd.bdate_range("2024-01-01", periods=n_calm + n_thin)
    pnl = pd.Series(rng.normal(0, 100, n_calm + n_thin), index=dates)
    labels = pd.Series(["calm"] * n_calm + ["volatile"] * n_thin, index=dates)

    breaches, notes = regime_conditional_breach_series(pnl, labels, confidence=0.95, min_days=30)
    assert any("volatile" in n and "falling back" in n for n in notes)
    assert len(breaches) == len(pnl)


def test_end_to_end_backtest_on_real_book():
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

    pooled = backtest_breach_series(pooled_breach_series(pnl_series, 0.95), 0.95, "pooled")
    conditional_breaches, _ = regime_conditional_breach_series(pnl_series, regime_labels, 0.95)
    conditional = backtest_breach_series(conditional_breaches, 0.95, "conditional")

    for result in (pooled, conditional):
        assert result.n_obs == len(pnl_series)
        assert 0.0 <= result.combined_p_value <= 1.0
