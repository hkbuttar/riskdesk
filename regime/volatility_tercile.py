"""Calm/normal/volatile regime classification via rolling realized
volatility terciles.

This reuses execedge's own methodology (`data/regimes.py` in that repo)
directly, for methodological continuity across the portfolio rather than
inventing a new regime definition: the same rolling-window log-return
volatility estimator, the same static (computed-once-per-run) tercile
split via `Series.quantile(1/3)` / `Series.quantile(2/3)` over the whole
available history (not `pd.qcut`, not a per-bar expanding recompute), and
the same label rule (`<=` low -> calm, `>=` high -> volatile, else normal).

Two adaptations, both disclosed: (1) execedge's default window (24 bars)
assumes hourly crypto bars; this project works with daily equity/crypto
data, so the window and annualization factor are re-derived for daily bars
(`window=21` ~ one trading month, `periods_per_year=252`) rather than reused
verbatim. (2) execedge's third label is `"volatile"`, not `"stressed"` --
kept as `"volatile"` here specifically to preserve continuity with the
methodology being reused, even though the plan's own prose says "stressed";
renaming it would be a cosmetic change dressed up as continuity.

Regime choice of reference series: classified on a broad market proxy
(SPY) rather than this project's own aggregated portfolio P&L, deliberately
-- a regime label is meant to describe the market condition the book is
exposed to, not be circularly defined by the book's own returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REGIME_LABELS = ("calm", "normal", "volatile")

DEFAULT_WINDOW_DAYS = 21
DEFAULT_PERIODS_PER_YEAR = 252


def rolling_realized_vol(
    close: pd.Series, window: int = DEFAULT_WINDOW_DAYS, periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
) -> pd.Series:
    log_returns = np.log(close / close.shift(1))
    rolling_std = log_returns.rolling(window=window, min_periods=window).std()
    return rolling_std * np.sqrt(periods_per_year)


@dataclass
class TercileResult:
    labels: pd.Series
    thresholds: dict[str, float]
    vol: pd.Series

    def value_counts(self) -> pd.Series:
        return self.labels.value_counts()

    def current(self) -> str | None:
        valid = self.labels.dropna()
        return valid.iloc[-1] if len(valid) else None


def classify_regimes(vol: pd.Series, low_q: float = 1 / 3, high_q: float = 2 / 3) -> TercileResult:
    valid = vol.dropna()
    low_thresh = float(valid.quantile(low_q))
    high_thresh = float(valid.quantile(high_q))

    def label(v: float) -> str | None:
        if pd.isna(v):
            return None
        if v <= low_thresh:
            return "calm"
        if v >= high_thresh:
            return "volatile"
        return "normal"

    labels = vol.apply(label)
    thresholds = {
        "low_quantile": low_q, "high_quantile": high_q,
        "low_threshold": low_thresh, "high_threshold": high_thresh,
    }
    return TercileResult(labels=labels, thresholds=thresholds, vol=vol)
