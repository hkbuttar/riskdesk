"""Extreme value / tail risk tests. GPD fitting is checked against
synthetic data with injected, known GPD parameters (recoverable within
statistical noise) and against hand-computable edge cases (the xi~=0
exponential-tail formula). Regime-conditional fitting is checked against
constructed buckets with a controlled, known exceedance count so the
MIN_EXCEEDANCES cutoff can be triggered deterministically. One end-to-end
test runs the real pipeline at both thresholds used in extreme_value/run.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from extreme_value.gpd import MIN_EXCEEDANCES, fit_gpd_tail, gpd_var_cvar
from extreme_value.tail_risk import compare_tail_shape, fit_gpd_by_regime
from risk_measures.var import evt_pot


def test_fit_gpd_tail_returns_none_below_min_exceedances():
    rng = np.random.default_rng(0)
    loss = rng.normal(0, 100, 50)  # 90% threshold -> ~5 exceedances, well under MIN_EXCEEDANCES
    fit = fit_gpd_tail(loss, threshold_quantile=0.90)
    assert fit is None


def test_fit_gpd_tail_recovers_injected_gpd_parameters():
    rng = np.random.default_rng(1)
    true_xi, true_beta, threshold_val = 0.3, 200.0, 1000.0
    # Injected tail is 500/2500 = 20% of the data; requesting the 90th
    # percentile (top 10%) keeps the empirical threshold safely INSIDE the
    # true GPD-distributed tail rather than at the body/tail boundary --
    # GPD's threshold-stability property (excess over a higher threshold
    # within a GPD tail is again GPD with the same shape) is what makes
    # this self-consistent; landing exactly at the boundary is not.
    body = np.clip(rng.normal(0, 50, 2000), None, threshold_val - 1)
    exceedances = stats.genpareto.rvs(true_xi, scale=true_beta, size=500, random_state=rng)
    loss = np.concatenate([body, threshold_val + exceedances])

    fit = fit_gpd_tail(loss, threshold_quantile=0.90)
    assert fit is not None
    assert fit.xi == pytest.approx(true_xi, abs=0.2)
    assert fit.n_exceedances >= MIN_EXCEEDANCES


def test_gpd_var_cvar_returns_nan_below_threshold_coverage():
    from extreme_value.gpd import GPDFit

    fit = GPDFit(threshold=100.0, threshold_quantile=0.90, xi=0.2, beta=50.0, n_exceedances=50, n_total=500)
    # coverage = 50/500 = 0.10; requesting confidence=0.85 -> tail_prob=0.15 >= coverage -> NaN.
    var, cvar = gpd_var_cvar(fit, confidence=0.85)
    assert var != var and cvar != cvar


def test_gpd_var_cvar_matches_hand_computed_exponential_case():
    from extreme_value.gpd import GPDFit

    # xi ~ 0 -> exponential tail: VaR = u - beta*ln(tail_prob/coverage), CVaR = VaR + beta.
    fit = GPDFit(threshold=1000.0, threshold_quantile=0.90, xi=1e-9, beta=200.0, n_exceedances=50, n_total=500)
    confidence = 0.98
    tail_prob = 1 - confidence
    coverage = 50 / 500
    expected_var = 1000.0 - 200.0 * np.log(tail_prob / coverage)
    var, cvar = gpd_var_cvar(fit, confidence)
    assert var == pytest.approx(expected_var)
    assert cvar == pytest.approx(expected_var + 200.0)


def test_evt_pot_refactor_matches_direct_gpd_fit():
    # risk_measures.var.evt_pot now delegates to extreme_value.gpd -- check
    # its VaR/CVaR match a direct fit_gpd_tail + gpd_var_cvar call exactly.
    rng = np.random.default_rng(2)
    pnl = pd.Series(rng.normal(0, 100, 1000))
    result = evt_pot(pnl, confidence=0.95, threshold_quantile=0.90)

    loss = (-pnl).to_numpy()
    fit = fit_gpd_tail(loss, 0.90)
    assert fit is not None
    var, cvar = gpd_var_cvar(fit, 0.95)
    assert result.var_dollar == pytest.approx(var)
    assert result.cvar_dollar == pytest.approx(cvar)


def _regime_pnl_and_labels(n_per_regime: int, vols: dict[str, float], seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_per_regime * len(vols))
    values, labels = [], []
    for regime, vol in vols.items():
        values.append(rng.normal(0, vol, n_per_regime))
        labels.extend([regime] * n_per_regime)
    pnl = pd.Series(np.concatenate(values), index=dates)
    regime_labels = pd.Series(labels, index=dates)
    return pnl, regime_labels


def test_fit_gpd_by_regime_skips_insufficient_regimes():
    pnl, labels = _regime_pnl_and_labels(50, {"calm": 100.0, "volatile": 500.0}, seed=3)
    comparison = fit_gpd_by_regime(pnl, labels, threshold_quantile=0.90)
    # 50 days * 10% = 5 exceedances per regime, well under MIN_EXCEEDANCES.
    assert all(fit is None for fit in comparison.regime_fits.values())
    assert len(comparison.notes) == 2


def test_fit_gpd_by_regime_fits_with_sufficient_data():
    pnl, labels = _regime_pnl_and_labels(400, {"calm": 100.0, "volatile": 800.0}, seed=4)
    comparison = fit_gpd_by_regime(pnl, labels, threshold_quantile=0.80)
    # 400 days * 20% = 80 exceedances per regime, comfortably above MIN_EXCEEDANCES.
    assert comparison.regime_fits["calm"] is not None
    assert comparison.regime_fits["volatile"] is not None
    # Volatile regime's much larger injected sigma should show up as a larger beta (scale).
    assert comparison.regime_fits["volatile"].beta > comparison.regime_fits["calm"].beta


def test_compare_tail_shape_table_includes_pooled_and_regime_rows():
    pnl, labels = _regime_pnl_and_labels(400, {"calm": 100.0, "volatile": 800.0}, seed=5)
    comparison = fit_gpd_by_regime(pnl, labels, threshold_quantile=0.80)
    table = compare_tail_shape(comparison)
    assert "pooled" in table.index
    assert "calm" in table.index
    assert "volatile" in table.index
    assert {"xi", "beta", "n_exceedances", "threshold"} <= set(table.columns)


def test_end_to_end_on_real_book_both_thresholds():
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

    for threshold in (0.90, 0.80):
        comparison = fit_gpd_by_regime(pnl_series, regime_labels, threshold_quantile=threshold)
        assert comparison.pooled_fit is not None  # pooled sample (501 days) always has enough
        table = compare_tail_shape(comparison)
        assert "pooled" in table.index
