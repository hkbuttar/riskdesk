"""Static (whole-window) historical correlation and covariance.

This is the same estimation approach behind alpha-signal-lab's kill-switch:
a single correlation matrix computed once over a historical window and
assumed to hold going forward. It's the baseline every other estimator in
this module (Ledoit-Wolf shrinkage, DCC-GARCH) is compared against, not
because it's wrong to compute, but because assuming it's STABLE is exactly
the failure mode that exposed alpha-signal-lab's zero-correlation
vol-targeting assumption during the 2020 COVID window: correlations that
look low in a calm sample can spike sharply once markets actually move
together under stress, and a static matrix has no way to see that coming.
"""

from __future__ import annotations

import pandas as pd


def static_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
    return returns_df.corr()


def static_covariance(returns_df: pd.DataFrame, annualize: bool = False) -> pd.DataFrame:
    cov = returns_df.cov()
    return cov * 252 if annualize else cov
