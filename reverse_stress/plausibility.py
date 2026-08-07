"""Plausibility assessment for a solved reverse-stress scenario: "here's a
scenario that breaks the portfolio" is not useful risk management on its
own without an honest answer to "how likely is that, really" -- this
module answers it two ways, reusing work already built elsewhere in this
project rather than re-deriving either:

1. **Regime-conditional distance** (correlation/, regime/): the same
   solved scenario's Mahalanobis distance recomputed under the volatile
   regime's own factor covariance instead of the pooled one. If regimes
   genuinely matter (as regime/conditional.py's own findings were mixed
   on), a scenario that looks extreme (many standard deviations) under a
   calm-period-dominated pooled covariance may look far less extreme once
   measured against how much factors actually co-move during real stress.
2. **Historical comparison** (stress/historical.py): the solved scenario's
   per-factor shock size, set directly next to what each named factor
   ACTUALLY did, factor by factor, during the three real crisis windows
   already replayed in this project -- "has anything like this happened
   before" answered with real numbers, not a probability model's assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from correlation.shrinkage import ledoit_wolf_covariance
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from reverse_stress.optimization import DEFAULT_HORIZON_DAYS, to_horizon_returns
from stress.historical import HISTORICAL_WINDOWS, fetch_price_history


@dataclass
class RegimeConditionalDistance:
    pooled_distance: float
    pooled_probability: float
    volatile_distance: float | None
    volatile_probability: float | None
    n_volatile_days: int
    notes: str


def mahalanobis_distance(shock: dict[str, float], cov: pd.DataFrame) -> tuple[float, float]:
    factors = list(shock.keys())
    x = np.array([shock[f] for f in factors])
    sigma = cov.loc[factors, factors].to_numpy()
    sigma_inv = np.linalg.inv(sigma)
    d_sq = float(x @ sigma_inv @ x)
    return float(np.sqrt(d_sq)), float(stats.chi2.sf(d_sq, df=len(factors)))


def compare_pooled_vs_volatile_regime(
    shock: dict[str, float],
    factor_returns: pd.DataFrame,
    pooled_distance: float,
    pooled_probability: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> RegimeConditionalDistance:
    """`factor_returns` is DAILY returns; regime labels are classified from
    the original daily SPY series (a rolling-vol classifier needs daily
    granularity to work at all), then aligned to the horizon-return series'
    index -- each horizon-window's regime label is the one on its LAST day,
    a disclosed choice ("was the market already turbulent by the time this
    month-long window ended"), not the window's start or an average.
    """
    if "SPY" not in factor_returns.columns:
        return RegimeConditionalDistance(
            pooled_distance, pooled_probability, None, None, 0, "No SPY in factor set -- cannot classify regime."
        )

    tercile = classify_regimes(rolling_realized_vol((1 + factor_returns["SPY"]).cumprod()))
    horizon_returns = to_horizon_returns(factor_returns[list(shock.keys())], horizon_days)
    aligned = tercile.labels.reindex(horizon_returns.index)
    volatile_returns = horizon_returns[aligned == "volatile"]

    if len(volatile_returns) < 30:
        return RegimeConditionalDistance(
            pooled_distance, pooled_probability, None, None, len(volatile_returns),
            f"Only {len(volatile_returns)} volatile-regime horizon-windows -- too few for a stable "
            "conditional covariance.",
        )

    shrinkage = ledoit_wolf_covariance(volatile_returns)
    volatile_distance, volatile_probability = mahalanobis_distance(shock, shrinkage.covariance)

    return RegimeConditionalDistance(
        pooled_distance, pooled_probability, volatile_distance, volatile_probability,
        len(volatile_returns),
        f"Volatile-regime covariance fit on {len(volatile_returns)} {horizon_days}-day horizon-windows "
        "(heavily overlapping, since only ~500 daily observations are available).",
    )


def compare_to_historical_windows(shock: dict[str, float]) -> pd.DataFrame:
    """For each real crisis window already replayed in stress/historical.py,
    the actual realized total return of every factor the solved scenario
    shocks -- set directly next to the solved shock for comparison.
    """
    factors = sorted(shock.keys())
    rows = {}
    for name, window in HISTORICAL_WINDOWS.items():
        prices = fetch_price_history(factors, window["start"], window["end"])
        available = [f for f in factors if f in prices.columns and not prices[f].empty]
        if not available:
            continue
        total_return = prices[available].iloc[-1] / prices[available].iloc[0] - 1
        rows[name] = total_return.reindex(factors)

    df = pd.DataFrame(rows).T
    df.loc["solved_scenario"] = pd.Series(shock)
    return df
