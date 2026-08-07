"""Static vs. DCC-GARCH correlation, specifically DURING the three real
historical crisis windows already replayed in stress/historical.py --
closing the gap between correlation/run.py (static vs. DCC-GARCH compared
over a generic 2-year window) and regime/conditional.py's and
correlation.py's own stated "future work: validating against a real
historical stress window." This is that validation.

Method: fit DCC-GARCH once on a long history spanning all three crisis
windows (reusing correlation/dcc_garch.py directly), then read off the
fitted dynamic correlation specifically on the days inside each crisis
window, compared against the static (same long-window) correlation for
the same pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from correlation.dcc_garch import fit_dcc_garch
from correlation.static import static_correlation
from stress.historical import HISTORICAL_WINDOWS


@dataclass
class StressWindowCorrelation:
    window_name: str
    pair: tuple[str, str]
    static_corr: float
    dcc_mean_during_window: float
    dcc_max_during_window: float
    dcc_pre_window_mean: float  # DCC correlation in the 60 days before the window, for contrast


def compare_during_stress_windows(
    price_history: pd.DataFrame, pairs: list[tuple[str, str]]
) -> list[StressWindowCorrelation]:
    returns = price_history.pct_change().dropna(how="any")
    static_corr = static_correlation(returns)
    dcc = fit_dcc_garch(returns)

    results = []
    for window_name, window in HISTORICAL_WINDOWS.items():
        window_mask = (dcc.dates >= window["start"]) & (dcc.dates <= window["end"])
        pre_window_start = pd.Timestamp(window["start"]) - pd.Timedelta(days=90)
        pre_mask = (dcc.dates >= pre_window_start) & (dcc.dates < window["start"])

        for ticker_a, ticker_b in pairs:
            if ticker_a not in dcc.tickers or ticker_b not in dcc.tickers:
                continue
            series = dcc.correlation_series(ticker_a, ticker_b)
            during = series[window_mask]
            pre = series[pre_mask]
            if during.empty:
                continue
            results.append(StressWindowCorrelation(
                window_name=window_name, pair=(ticker_a, ticker_b),
                static_corr=float(static_corr.loc[ticker_a, ticker_b]),
                dcc_mean_during_window=float(during.mean()),
                dcc_max_during_window=float(during.max()),
                dcc_pre_window_mean=float(pre.mean()) if not pre.empty else float("nan"),
            ))
    return results
