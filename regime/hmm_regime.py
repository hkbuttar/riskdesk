"""Probabilistic regime classification via a 3-state Markov-switching
model (`statsmodels.tsa.regime_switching.markov_regression`), the plan's
"and/or a simple hidden Markov model for a probabilistic regime label
instead of a hard cutoff" alternative to volatility_tercile.py's static
quantile cutoffs.

A regime-switching mean/variance model is fit directly on daily log
returns (not on the already-smoothed rolling-vol series terciles.py uses --
feeding the HMM a rolling statistic would double-smooth and understate how
fast it can actually detect a regime change). `statsmodels` doesn't order
its regimes by any particular meaning (regime index 0/1/2 are arbitrary
labels from the optimizer), so the three fitted regimes are re-labeled
calm/normal/volatile here by sorting on fitted variance ascending -- the
same ordering convention volatility_tercile.py's terciles use, so the two
methods' labels are directly comparable.

Output is a full smoothed probability distribution over the three regimes
for every day, not just a hard label -- e.g. "62% volatile, 35% normal, 3%
calm" on a given day, which a tercile cutoff cannot express.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from regime.volatility_tercile import REGIME_LABELS


@dataclass
class HMMRegimeResult:
    probabilities: pd.DataFrame  # columns calm/normal/volatile, index = dates
    hard_labels: pd.Series
    regime_params: dict[str, dict[str, float]]  # per-label fitted mean/variance
    notes: str

    def current_probabilities(self) -> pd.Series:
        return self.probabilities.iloc[-1]


def fit_hmm_regimes(close: pd.Series, k_regimes: int = 3) -> HMMRegimeResult:
    if not 2 <= k_regimes <= len(REGIME_LABELS):
        raise ValueError(f"k_regimes must be between 2 and {len(REGIME_LABELS)}, got {k_regimes}")
    labels = REGIME_LABELS[:k_regimes]  # e.g. k_regimes=2 -> ("calm", "normal")

    log_returns = np.log(close / close.shift(1)).dropna()

    model = MarkovRegression(log_returns, k_regimes=k_regimes, switching_variance=True)
    # A single default-start fit frequently lands in a poor local optimum on
    # this kind of data (observed: one regime absorbing almost no days) --
    # search_reps retries from multiple random starting points and keeps the
    # best, which is what actually gets a converged, well-identified fit
    # rather than a fit() call that merely returns without raising.
    res = model.fit(search_reps=50, maxiter=1000)

    variances = [res.params[f"sigma2[{i}]"] for i in range(k_regimes)]
    means = [res.params[f"const[{i}]"] for i in range(k_regimes)]
    order = np.argsort(variances)  # ascending variance -> calm, normal, [volatile]
    label_by_regime_index = {int(order[i]): labels[i] for i in range(k_regimes)}

    probs = res.smoothed_marginal_probabilities.rename(columns=label_by_regime_index)
    probs = probs[list(labels)]
    hard_labels = probs.idxmax(axis=1)

    regime_params = {
        label_by_regime_index[i]: {"mean": float(means[i]), "variance": float(variances[i])}
        for i in range(k_regimes)
    }

    return HMMRegimeResult(
        probabilities=probs,
        hard_labels=hard_labels,
        regime_params=regime_params,
        notes=f"MarkovRegression(k_regimes={k_regimes}, switching_variance=True), n={len(log_returns)}",
    )
