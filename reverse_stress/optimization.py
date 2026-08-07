"""Reverse stress testing: the inverse of every other stress module in this
project. Historical/hypothetical stress (stress/) asks "apply scenario X,
measure loss Y." This asks the opposite, harder question: "what combination
of factor moves produces a specific target loss Y, and how plausible is
that combination, really?"

Method (the plan's own framing, "minimum-distance-to-default-scenario in
factor space"): the portfolio's P&L is modeled as linear in the named
factors (factor_model/regression.py's own fitted loadings -- reusing that
fit directly rather than re-deriving it). Given that linear model, the set
of factor-move combinations that produce exactly the target loss is a
hyperplane in factor space. Of all the points on that hyperplane, the
"most plausible" one -- under a multivariate-normal assumption on factor
returns -- is the one at minimum Mahalanobis distance from zero, i.e. the
smallest, least-improbable combined move that still produces the target
loss. This is solved as a quadratic program via `cvxpy`:

    minimize    x^T Sigma^-1 x
    subject to  loadings^T x + alpha == -target_loss

`Sigma` is Ledoit-Wolf-shrunk (correlation/shrinkage.py, reused directly)
rather than the raw sample covariance of only a handful of named factors --
an unshrunk, poorly-conditioned Sigma would make Sigma^-1 numerically
unstable and bias the "minimum distance" solution toward whichever
direction the raw sample covariance happens to be noisiest in.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy import stats

from correlation.shrinkage import ledoit_wolf_covariance

DEFAULT_HORIZON_DAYS = 21  # ~1 trading month -- see module docstring on why daily-scale was wrong


def to_horizon_returns(factor_returns: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """Overlapping `horizon_days`-window cumulative returns, approximated as
    a rolling sum of daily simple returns (a linear approximation of true
    compounding -- accurate for the small daily moves typical here, and
    consistent with the linear factor model already used throughout this
    project; the error grows with return magnitude and horizon length, a
    disclosed simplification, not treated as exact compounding).
    """
    return factor_returns.rolling(horizon_days).sum().dropna()


@dataclass
class ReverseStressResult:
    target_loss: float
    horizon_days: int
    factor_shocks: dict[str, float]
    implied_pnl: float
    mahalanobis_distance: float  # combined "standard deviations" away from zero
    implied_annual_probability: float  # P(a move this extreme or more), under the fitted Sigma
    shrinkage_used: float


def solve_reverse_stress(
    loadings: dict[str, float],
    alpha: float,
    factor_returns: pd.DataFrame,
    target_loss: float,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
) -> ReverseStressResult:
    """`factor_returns` is DAILY returns (as fetched by factor_model/factors.py);
    solved internally over `horizon_days` (default ~1 trading month), not a
    single day -- a reverse-stress scenario expressed as "SPY +33% tomorrow"
    is not a useful answer to "what could plausibly break this portfolio";
    a ~month-scale cumulative move is directly comparable to the real
    historical crisis windows in stress/historical.py (COVID: 23 days,
    2022: 195 days, FTX: 4 days -- still not an exact horizon match to any
    of the three, but far closer than daily-vs-multi-week).

    `alpha` (the regression's daily unexplained-P&L intercept) is scaled by
    `horizon_days` for the same reason the returns are summed -- the same
    linear, disclosed approximation applied consistently to both terms of
    the P&L model.
    """
    factors = list(loadings.keys())
    loading_vec = np.array([loadings[f] for f in factors])
    horizon_alpha = alpha * horizon_days

    horizon_returns = to_horizon_returns(factor_returns[factors], horizon_days)
    shrinkage = ledoit_wolf_covariance(horizon_returns)
    sigma = shrinkage.covariance.to_numpy()
    sigma_inv = np.linalg.inv(sigma)

    x = cp.Variable(len(factors))
    objective = cp.Minimize(cp.quad_form(x, sigma_inv))
    # Portfolio P&L = loadings^T x + alpha; a "loss of target_loss" means P&L == -target_loss.
    constraints = [loading_vec @ x + horizon_alpha == -target_loss]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != "optimal":
        raise RuntimeError(f"Reverse stress optimization did not converge: status={problem.status}")

    x_star = x.value
    mahalanobis_sq = float(x_star @ sigma_inv @ x_star)
    mahalanobis_distance = float(np.sqrt(mahalanobis_sq))
    # Mahalanobis distance^2 for a multivariate normal ~ chi-squared(df=n_factors);
    # survival function gives P(a draw at least this far from the mean).
    implied_probability = float(stats.chi2.sf(mahalanobis_sq, df=len(factors)))

    return ReverseStressResult(
        target_loss=target_loss,
        horizon_days=horizon_days,
        factor_shocks=dict(zip(factors, x_star)),
        implied_pnl=float(loading_vec @ x_star + horizon_alpha),
        mahalanobis_distance=mahalanobis_distance,
        implied_annual_probability=implied_probability,
        shrinkage_used=shrinkage.shrinkage,
    )
