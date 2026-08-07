"""Ledoit-Wolf shrinkage: a well-conditioned covariance estimate.

The raw sample covariance matrix (static.py) is a noisy estimator once the
number of assets approaches the number of observations -- eigenvalues get
pulled apart (largest inflated, smallest deflated), which is exactly the
instability that later makes Monte Carlo VaR and reverse stress testing's
inverse optimization numerically unreliable. Ledoit-Wolf shrinkage
(Ledoit & Wolf, 2004) shrinks the sample covariance toward a structured
target (`sklearn`'s default: a scaled identity matrix) by a data-driven
shrinkage constant, trading a small amount of bias for a large reduction in
estimation variance -- reported here via the condition number improvement,
a direct, quantified before/after rather than an assertion that it helps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class ShrinkageResult:
    covariance: pd.DataFrame
    shrinkage: float
    sample_condition_number: float
    shrunk_condition_number: float


def ledoit_wolf_covariance(returns_df: pd.DataFrame) -> ShrinkageResult:
    lw = LedoitWolf().fit(returns_df.to_numpy())
    shrunk_cov = pd.DataFrame(lw.covariance_, index=returns_df.columns, columns=returns_df.columns)
    sample_cov = returns_df.cov()

    return ShrinkageResult(
        covariance=shrunk_cov,
        shrinkage=float(lw.shrinkage_),
        sample_condition_number=float(np.linalg.cond(sample_cov.to_numpy())),
        shrunk_condition_number=float(np.linalg.cond(shrunk_cov.to_numpy())),
    )
