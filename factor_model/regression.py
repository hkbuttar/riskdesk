"""Named-factor decomposition: OLS regression of a P&L series (portfolio-
or strategy-level) on the named factor returns (factors.py), giving a
dollar factor loading per factor, an intercept ("alpha" -- P&L not
explained by any named factor), and R² (how much of the P&L variance the
named factors explain jointly).

This is what makes "does PairTrade Lab's market-neutral book carry hidden
market beta once aggregated" a directly answerable, quantified question:
run this same regression on that one strategy's own P&L series and read
off its SPY coefficient and t-stat, rather than asserting an answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class FactorRegressionResult:
    label: str
    loadings: dict[str, float]  # $ P&L per unit factor return
    t_stats: dict[str, float]
    p_values: dict[str, float]
    alpha: float  # intercept: $ P&L per day not explained by named factors
    alpha_p_value: float
    r_squared: float
    n_obs: int


def fit_factor_regression(
    label: str, pnl_series: pd.Series, factor_returns: pd.DataFrame
) -> FactorRegressionResult:
    aligned = factor_returns.join(pnl_series.rename("pnl"), how="inner").dropna()
    y = aligned["pnl"]
    x = sm.add_constant(aligned.drop(columns="pnl"))

    model = sm.OLS(y, x).fit()

    factor_names = [c for c in x.columns if c != "const"]
    return FactorRegressionResult(
        label=label,
        loadings={f: float(model.params[f]) for f in factor_names},
        t_stats={f: float(model.tvalues[f]) for f in factor_names},
        p_values={f: float(model.pvalues[f]) for f in factor_names},
        alpha=float(model.params["const"]),
        alpha_p_value=float(model.pvalues["const"]),
        r_squared=float(model.rsquared),
        n_obs=int(model.nobs),
    )


def significant_factors(result: FactorRegressionResult, alpha: float = 0.05) -> list[str]:
    return [f for f, p in result.p_values.items() if p < alpha]
