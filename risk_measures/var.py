"""Five VaR/CVaR methods, run on the same portfolio P&L series
(risk_measures/returns.py::build_portfolio_pnl_series) so the comparison in
run.py is apples-to-apples: same book, same historical window, same
confidence level, different distributional assumption.

All five work in LOSS space (loss = -pnl) so VaR/CVaR are reported as
positive dollar loss numbers at the given confidence, e.g. "95% 1-day VaR
= $1,200" means a 5% chance of losing more than $1,200 over one day, given
today's book applied to this historical window.

- **Historical simulation**: the empirical quantile of the realized loss
  series -- no distributional assumption, but only as good as the historical
  window actually covering the relevant tail behavior.
- **Parametric (variance-covariance)**: assumes losses are normally
  distributed -- fast, closed-form, and the textbook source of VaR's
  well-known tail-risk understatement whenever real returns are fat-tailed.
- **Monte Carlo**: simulates from a fitted multivariate normal over the
  underlying risk factors' returns (not the portfolio series directly), so
  it captures the portfolio's actual correlation structure, but still
  inherits the normal assumption at the factor level.
- **Cornish-Fisher**: parametric, but corrects the normal quantile for the
  loss series's own sample skewness/kurtosis -- meant to fix parametric's
  fat-tail blind spot without fully abandoning a closed-form-ish approach.
- **EVT (Peaks-Over-Threshold)**: fits a Generalized Pareto Distribution to
  the losses that exceed a high threshold (scipy.stats.genpareto), the
  approach built specifically for tail estimation rather than
  whole-distribution fitting. Threshold choice is a disclosed judgment call
  (default: 90th percentile of losses) -- too low blends in non-tail
  behavior, too high leaves too few exceedances to fit reliably.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class VaRResult:
    method: str
    confidence: float
    var_dollar: float
    cvar_dollar: float
    notes: str = ""


def _to_loss(pnl_series: pd.Series) -> pd.Series:
    return -pnl_series


def historical_simulation(pnl_series: pd.Series, confidence: float = 0.95) -> VaRResult:
    loss = _to_loss(pnl_series)
    var = float(np.quantile(loss, confidence))
    tail = loss[loss >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return VaRResult("historical_simulation", confidence, var, cvar, f"n={len(loss)}, tail n={len(tail)}")


def parametric_variance_covariance(pnl_series: pd.Series, confidence: float = 0.95) -> VaRResult:
    loss = _to_loss(pnl_series)
    mu, sigma = float(loss.mean()), float(loss.std(ddof=1))
    z = stats.norm.ppf(confidence)
    var = mu + z * sigma
    cvar = mu + sigma * stats.norm.pdf(z) / (1 - confidence)
    return VaRResult("parametric_normal", confidence, var, cvar, f"mu={mu:.2f}, sigma={sigma:.2f}")


def monte_carlo(
    returns_df: pd.DataFrame,
    weights: dict[str, float],
    confidence: float = 0.95,
    n_sims: int = 20_000,
    seed: int = 42,
) -> VaRResult:
    tickers = list(weights.keys())
    factor_returns = returns_df[tickers]
    mean_vec = factor_returns.mean().to_numpy()
    cov = factor_returns.cov().to_numpy()
    weight_vec = np.array([weights[t] for t in tickers])

    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mean_vec, cov, size=n_sims)
    sim_pnl = sims @ weight_vec
    sim_loss = -sim_pnl

    var = float(np.quantile(sim_loss, confidence))
    tail = sim_loss[sim_loss >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return VaRResult("monte_carlo", confidence, var, cvar, f"n_sims={n_sims}, n_factors={len(tickers)}")


def cornish_fisher(
    pnl_series: pd.Series, confidence: float = 0.95, n_samples: int = 200_000, seed: int = 42
) -> VaRResult:
    loss = _to_loss(pnl_series)
    mu, sigma = float(loss.mean()), float(loss.std(ddof=1))
    skew = float(stats.skew(loss, bias=False))
    kurt = float(stats.kurtosis(loss, fisher=True, bias=False))  # excess kurtosis

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_samples)
    z_cf = (
        z
        + (z**2 - 1) * skew / 6
        + (z**3 - 3 * z) * kurt / 24
        - (2 * z**3 - 5 * z) * skew**2 / 36
    )
    cf_loss = mu + sigma * z_cf

    var = float(np.quantile(cf_loss, confidence))
    tail = cf_loss[cf_loss >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return VaRResult(
        "cornish_fisher", confidence, var, cvar, f"skew={skew:.3f}, excess_kurtosis={kurt:.3f}"
    )


def evt_pot(pnl_series: pd.Series, confidence: float = 0.95, threshold_quantile: float = 0.90) -> VaRResult:
    loss = _to_loss(pnl_series).to_numpy()
    n = len(loss)
    u = float(np.quantile(loss, threshold_quantile))
    exceedances = loss[loss > u] - u
    n_u = len(exceedances)

    if n_u < 20:
        return VaRResult(
            "evt_pot", confidence, float("nan"), float("nan"),
            f"Only {n_u} exceedances above the {threshold_quantile:.0%} threshold -- too few to fit a GPD reliably.",
        )

    xi, _loc, beta = stats.genpareto.fit(exceedances, floc=0)
    tail_prob = 1 - confidence
    coverage = n_u / n

    if tail_prob >= coverage:
        return VaRResult(
            "evt_pot", confidence, float("nan"), float("nan"),
            f"Requested confidence {confidence:.2%} is below the threshold's own coverage "
            f"({1 - coverage:.2%}) -- EVT extrapolation not meaningful here; lower threshold_quantile "
            "or raise confidence.",
        )

    if abs(xi) < 1e-6:
        var = u - beta * np.log(tail_prob / coverage)
        cvar = var + beta
    else:
        var = u + (beta / xi) * ((tail_prob / coverage) ** (-xi) - 1)
        cvar = (var + beta - xi * u) / (1 - xi) if xi < 1 else float("inf")

    return VaRResult(
        "evt_pot", confidence, float(var), float(cvar),
        f"threshold_quantile={threshold_quantile:.0%}, xi={xi:.3f}, beta={beta:.2f}, n_exceedances={n_u}",
    )


ALL_METHODS = {
    "historical_simulation": historical_simulation,
    "parametric_normal": parametric_variance_covariance,
    "cornish_fisher": cornish_fisher,
    "evt_pot": evt_pot,
}  # monte_carlo excluded -- different signature (needs returns_df + weights, not just pnl_series)
