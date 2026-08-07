"""Shared Generalized Pareto tail-fitting core (Peaks-Over-Threshold), used
by both `risk_measures/var.py::evt_pot` (the pooled EVT VaR method) and
this module's regime-conditional extension below -- extracted here rather
than duplicated, since both need the exact same fit, just applied to
different slices of the loss series.

`fit_gpd_tail` returns the fitted shape/scale parameters directly
(structured, not embedded in a notes string) specifically so tail SHAPE
(xi) can be compared numerically across regimes -- "is the volatile
regime's tail fatter, or just wider" is a different, more granular
question than "is the volatile regime's VaR higher," and answering it
requires xi itself, not just the resulting dollar VaR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

MIN_EXCEEDANCES = 20


@dataclass
class GPDFit:
    threshold: float
    threshold_quantile: float
    xi: float  # shape parameter -- higher = fatter tail
    beta: float  # scale parameter
    n_exceedances: int
    n_total: int

    @property
    def coverage(self) -> float:
        return self.n_exceedances / self.n_total


def fit_gpd_tail(loss: np.ndarray, threshold_quantile: float = 0.90) -> GPDFit | None:
    """Fits a GPD to exceedances of `loss` (already loss-sign-convention,
    i.e. positive = loss) over its own `threshold_quantile` quantile.
    Returns None if there are too few exceedances to fit reliably (see
    MIN_EXCEEDANCES) -- disclosed via None, not a fabricated fit.
    """
    n = len(loss)
    u = float(np.quantile(loss, threshold_quantile))
    exceedances = loss[loss > u] - u
    n_u = len(exceedances)

    if n_u < MIN_EXCEEDANCES:
        return None

    xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0)
    return GPDFit(
        threshold=u, threshold_quantile=threshold_quantile, xi=float(xi), beta=float(beta),
        n_exceedances=n_u, n_total=n,
    )


def gpd_var_cvar(fit: GPDFit, confidence: float) -> tuple[float, float]:
    """VaR/CVaR at `confidence` implied by a fitted GPD tail (McNeil-Frey
    POT formula). Returns (nan, nan) if `confidence` is below the
    threshold's own coverage -- extrapolating a tail fit to a confidence
    level *less* extreme than the threshold itself used to fit it isn't
    meaningful.
    """
    tail_prob = 1 - confidence
    if tail_prob >= fit.coverage:
        return float("nan"), float("nan")

    if abs(fit.xi) < 1e-6:
        var = fit.threshold - fit.beta * np.log(tail_prob / fit.coverage)
        cvar = var + fit.beta
    else:
        var = fit.threshold + (fit.beta / fit.xi) * ((tail_prob / fit.coverage) ** (-fit.xi) - 1)
        cvar = (var + fit.beta - fit.xi * fit.threshold) / (1 - fit.xi) if fit.xi < 1 else float("inf")

    return float(var), float(cvar)
