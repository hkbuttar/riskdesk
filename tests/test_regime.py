"""Regime classification tests. Tercile logic is checked against synthetic
series with known/controllable structure (deterministic). The HMM is
checked on a synthetic two-regime series with a clean variance separation,
where recovering the injected regime is realistic to assert; the real-SPY
end-to-end test only checks structural invariants, since exact regime
counts on live market data would be flaky.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from regime.hmm_regime import fit_hmm_regimes
from regime.volatility_tercile import classify_regimes, rolling_realized_vol


def _price_series_from_log_returns(log_returns: np.ndarray, start: float = 100.0) -> pd.Series:
    prices = start * np.exp(np.cumsum(log_returns))
    dates = pd.bdate_range("2024-01-01", periods=len(prices))
    return pd.Series(prices, index=dates)


def test_rolling_realized_vol_is_nan_before_window_fills():
    rng = np.random.default_rng(0)
    close = _price_series_from_log_returns(rng.normal(0, 0.01, 100))
    vol = rolling_realized_vol(close, window=21)
    assert vol.iloc[:21].isna().all()
    assert vol.iloc[21:].notna().all()


def test_rolling_realized_vol_scales_with_input_sigma():
    rng = np.random.default_rng(1)
    low_vol = _price_series_from_log_returns(rng.normal(0, 0.005, 300))
    high_vol = _price_series_from_log_returns(rng.normal(0, 0.03, 300))
    low = rolling_realized_vol(low_vol, window=21).dropna().mean()
    high = rolling_realized_vol(high_vol, window=21).dropna().mean()
    assert high > low * 3  # roughly proportional to the 6x sigma ratio


def test_classify_regimes_terciles_are_roughly_balanced():
    rng = np.random.default_rng(2)
    vol = pd.Series(rng.uniform(0.05, 0.5, 900))
    result = classify_regimes(vol)
    counts = result.value_counts()
    fractions = counts / len(vol)
    for label in ("calm", "normal", "volatile"):
        assert 0.20 <= fractions[label] <= 0.45


def test_classify_regimes_labels_known_calm_and_volatile_segments():
    vol = pd.Series([0.05] * 100 + [0.20] * 100 + [0.50] * 100)
    result = classify_regimes(vol)
    assert (result.labels.iloc[:100] == "calm").all()
    assert (result.labels.iloc[100:200] == "normal").all()
    assert (result.labels.iloc[200:] == "volatile").all()


def test_classify_regimes_thresholds_match_quantiles():
    vol = pd.Series(np.arange(1, 301, dtype=float))
    result = classify_regimes(vol)
    assert result.thresholds["low_threshold"] == pytest.approx(vol.quantile(1 / 3))
    assert result.thresholds["high_threshold"] == pytest.approx(vol.quantile(2 / 3))


def test_hmm_recovers_injected_two_regime_structure():
    rng = np.random.default_rng(3)
    n_half = 150
    calm = rng.normal(0.0005, 0.004, n_half)
    volatile = rng.normal(-0.002, 0.03, n_half)
    log_returns = np.concatenate([calm, volatile, calm])
    close = _price_series_from_log_returns(log_returns)

    result = fit_hmm_regimes(close, k_regimes=2)
    assert set(result.probabilities.columns) <= {"calm", "normal", "volatile"}
    # Probabilities must be a valid distribution over regimes each day.
    row_sums = result.probabilities.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    # Middle third was injected as the volatile regime -- majority of days
    # there should be labeled with the higher-variance regime.
    middle_labels = result.hard_labels.iloc[n_half : 2 * n_half]
    non_calm_fraction = (middle_labels != "calm").mean()
    assert non_calm_fraction > 0.6


def test_hmm_regime_params_ordered_by_ascending_variance():
    rng = np.random.default_rng(4)
    log_returns = np.concatenate(
        [rng.normal(0, 0.004, 150), rng.normal(0, 0.015, 150), rng.normal(0, 0.04, 150)]
    )
    close = _price_series_from_log_returns(log_returns)
    result = fit_hmm_regimes(close, k_regimes=3)
    variances = [result.regime_params[label]["variance"] for label in ("calm", "normal", "volatile")]
    assert variances == sorted(variances)


def test_end_to_end_regime_classification_on_real_spy():
    import yfinance as yf

    close = yf.Ticker("SPY").history(period="1y")["Close"]
    close.index = close.index.tz_localize(None)

    tercile = classify_regimes(rolling_realized_vol(close))
    assert set(tercile.labels.dropna().unique()) <= {"calm", "normal", "volatile"}

    hmm = fit_hmm_regimes(close)
    assert len(hmm.probabilities) == len(close) - 1  # one fewer, due to the return diff
    np.testing.assert_allclose(hmm.probabilities.sum(axis=1), 1.0, atol=1e-6)
