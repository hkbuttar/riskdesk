"""Tests for DCC-GARCH-during-stress-windows comparison. Checked against a
synthetic price series with a known, injected correlation spike exactly
inside one of the real HISTORICAL_WINDOWS date ranges, so the "did DCC
correctly detect the spike" assertion has a verifiable ground truth. One
end-to-end test runs the real pipeline on real data for structural
invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stress.dcc_during_stress import compare_during_stress_windows
from stress.historical import HISTORICAL_WINDOWS


def test_compare_during_stress_windows_detects_injected_spike():
    rng = np.random.default_rng(0)
    window = HISTORICAL_WINDOWS["covid_crash"]
    dates = pd.bdate_range("2019-06-01", "2020-06-01")
    window_mask = (dates >= window["start"]) & (dates <= window["end"])

    calm_corr, spike_corr = 0.1, 0.9
    returns = np.zeros((len(dates), 2))
    for i in range(len(dates)):
        corr = spike_corr if window_mask[i] else calm_corr
        cov = [[1, corr], [corr, 1]]
        returns[i] = rng.multivariate_normal([0, 0], cov) * 0.01

    prices = pd.DataFrame(100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=["A", "B"])
    results = compare_during_stress_windows(prices, [("A", "B")])

    covid_result = next(r for r in results if r.window_name == "covid_crash")
    assert covid_result.dcc_mean_during_window > covid_result.dcc_pre_window_mean


def test_compare_during_stress_windows_skips_missing_tickers():
    dates = pd.bdate_range("2019-06-01", "2020-06-01")
    rng = np.random.default_rng(1)
    # fit_dcc_garch needs >= 2 real assets to fit at all -- "NONEXISTENT"
    # simply isn't one of the columns present, which is what's under test.
    returns = rng.normal(0, 0.01, size=(len(dates), 2))
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=["A", "B"]
    )
    results = compare_during_stress_windows(prices, [("A", "NONEXISTENT"), ("A", "B")])
    # Synthetic dates only span through 2020-06-01, covering only the COVID
    # window -- 2022/FTX correctly produce no rows (empty window), not a bug.
    assert all(r.pair == ("A", "B") for r in results)
    assert any(r.window_name == "covid_crash" for r in results)


def test_end_to_end_on_real_historical_windows():
    from stress.historical import fetch_price_history

    earliest = min(w["pre_start"] for w in HISTORICAL_WINDOWS.values())
    latest = max(w["end"] for w in HISTORICAL_WINDOWS.values())
    prices = fetch_price_history(["SPY", "BTC-USD"], earliest, latest)

    results = compare_during_stress_windows(prices, [("SPY", "BTC-USD")])
    assert len(results) == len(HISTORICAL_WINDOWS)
    for r in results:
        assert -1.0 <= r.static_corr <= 1.0
        assert -1.0 <= r.dcc_mean_during_window <= 1.0
