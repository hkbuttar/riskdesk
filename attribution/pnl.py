"""P&L attribution: decomposes a P&L series, day by day, into the portion
explained by each named factor, an alpha (intercept) term, and a residual
-- reusing factor_model/regression.py's own fitted loadings directly rather
than re-deriving them. Unlike factor_model/regression.py itself (which only
reports static coefficients), this walks the loadings back across a real
window to show cumulative dollar contribution per factor, which is what
"attribution" means in practice: not just "the book has X exposure to
SPY," but "SPY moves accounted for $Y of this window's actual P&L."

The residual is defined as whatever's left over (`pnl - factor
contributions - alpha`), which makes the reconciliation
(`sum(factor contributions) + alpha + residual == total pnl`) hold
EXACTLY, by construction, not approximately -- a real attribution has to
tie out to the total P&L to be trustworthy, and this one is checked to,
not merely asserted to.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class FactorAttribution:
    factor_contributions: pd.DataFrame  # date x factor, $ P&L attributable to each factor that day
    alpha_contribution: pd.Series
    residual: pd.Series
    cumulative_by_factor: dict[str, float]
    cumulative_alpha: float
    cumulative_residual: float
    cumulative_total: float

    def reconciles(self, tol: float = 1e-6) -> bool:
        total_explained = sum(self.cumulative_by_factor.values()) + self.cumulative_alpha + self.cumulative_residual
        return abs(total_explained - self.cumulative_total) < tol


def attribute_by_factor(
    pnl_series: pd.Series, factor_returns: pd.DataFrame, loadings: dict[str, float], alpha: float
) -> FactorAttribution:
    aligned = factor_returns.join(pnl_series.rename("pnl"), how="inner").dropna()
    factors = list(loadings.keys())

    contributions = aligned[factors].mul(pd.Series(loadings), axis=1)
    alpha_series = pd.Series(alpha, index=aligned.index)
    explained = contributions.sum(axis=1) + alpha_series
    residual = aligned["pnl"] - explained

    return FactorAttribution(
        factor_contributions=contributions,
        alpha_contribution=alpha_series,
        residual=residual,
        cumulative_by_factor={f: float(contributions[f].sum()) for f in factors},
        cumulative_alpha=float(alpha_series.sum()),
        cumulative_residual=float(residual.sum()),
        cumulative_total=float(aligned["pnl"].sum()),
    )
