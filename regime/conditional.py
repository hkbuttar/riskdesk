"""Regime-conditional VaR and correlation: re-estimate risk_measures/var.py
and correlation/static.py separately per regime, rather than pooling every
historical day into one static model regardless of market condition.

Regime partition choice: this module partitions history using
volatility_tercile.py's labels, not hmm_regime.py's -- a disclosed choice,
not an oversight. The HMM's data-driven "volatile" regime had only 13 days
over the 2-year window (see regime/README section in the project README),
nowhere near enough to fit a stable per-regime VaR/correlation estimate;
terciles guarantee a usable sample size (~160 days) in every bucket by
construction, which is what per-regime estimation actually needs here.

At live-monitoring time (see regime/run_conditional.py), whichever regime
the classifier assigns to the most recent trading day determines which of
these three conditional models is "active" -- a genuinely adaptive risk
view, not a single fixed model applied regardless of current conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from correlation.static import static_correlation
from risk_measures.var import VaRResult, cornish_fisher, historical_simulation, parametric_variance_covariance

CONDITIONAL_METHODS = {
    "historical_simulation": historical_simulation,
    "parametric_normal": parametric_variance_covariance,
    "cornish_fisher": cornish_fisher,
}

MIN_DAYS_FOR_CONDITIONAL_VAR = 30


@dataclass
class RegimeConditionalResult:
    pooled_var: dict[str, VaRResult]
    conditional_var: dict[str, dict[str, VaRResult]]  # {regime: {method: VaRResult}}
    pooled_correlation: pd.DataFrame
    conditional_correlation: dict[str, pd.DataFrame]  # {regime: corr_df}
    regime_day_counts: dict[str, int]
    notes: list[str]


def align_regime_labels(pnl_index: pd.DatetimeIndex, regime_labels: pd.Series) -> pd.Series:
    """Regime labels are fit on SPY's own date index; the portfolio P&L
    series' index (risk_measures/returns.py) may not match exactly (e.g.
    if any single-ticker gap dropped a day elsewhere). Aligns via a plain
    reindex -- no forward-fill: a date the regime series has no label for
    (e.g. inside the vol window's warm-up) stays unlabeled (NaN) rather
    than silently inheriting a neighboring day's regime.
    """
    return regime_labels.reindex(pnl_index)


def split_by_regime(series: pd.Series, aligned_labels: pd.Series) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for regime in aligned_labels.dropna().unique():
        mask = aligned_labels == regime
        out[regime] = series[mask]
    return out


def regime_conditional_var(
    pnl_series: pd.Series, aligned_labels: pd.Series, confidence: float = 0.95
) -> tuple[dict[str, dict[str, VaRResult]], dict[str, int], list[str]]:
    notes: list[str] = []
    day_counts: dict[str, int] = {}
    results: dict[str, dict[str, VaRResult]] = {}

    by_regime = split_by_regime(pnl_series, aligned_labels)
    for regime, series in by_regime.items():
        day_counts[regime] = len(series)
        if len(series) < MIN_DAYS_FOR_CONDITIONAL_VAR:
            notes.append(
                f"{regime}: only {len(series)} days (< {MIN_DAYS_FOR_CONDITIONAL_VAR}) -- "
                "conditional VaR skipped, too few observations to be meaningful."
            )
            continue
        results[regime] = {
            name: method(series, confidence) for name, method in CONDITIONAL_METHODS.items()
        }
    return results, day_counts, notes


def regime_conditional_correlation(
    returns_df: pd.DataFrame, aligned_labels: pd.Series
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    notes: list[str] = []
    results: dict[str, pd.DataFrame] = {}
    by_regime = split_by_regime(returns_df, aligned_labels)
    for regime, sub_df in by_regime.items():
        if len(sub_df) < MIN_DAYS_FOR_CONDITIONAL_VAR:
            notes.append(f"{regime}: only {len(sub_df)} days -- conditional correlation skipped.")
            continue
        results[regime] = static_correlation(sub_df)
    return results, notes


def compare_pooled_vs_conditional(
    pnl_series: pd.Series, returns_df: pd.DataFrame, regime_labels: pd.Series, confidence: float = 0.95
) -> RegimeConditionalResult:
    aligned_labels = align_regime_labels(pnl_series.index, regime_labels)
    unlabeled = aligned_labels.isna().sum()

    pooled_var = {name: method(pnl_series, confidence) for name, method in CONDITIONAL_METHODS.items()}
    pooled_corr = static_correlation(returns_df)

    conditional_var, day_counts, var_notes = regime_conditional_var(pnl_series, aligned_labels, confidence)
    conditional_corr, corr_notes = regime_conditional_correlation(returns_df, aligned_labels)

    notes = [f"{unlabeled} of {len(pnl_series)} days unlabeled (regime warm-up window)."]
    notes += var_notes + corr_notes

    return RegimeConditionalResult(
        pooled_var=pooled_var,
        conditional_var=conditional_var,
        pooled_correlation=pooled_corr,
        conditional_correlation=conditional_corr,
        regime_day_counts=day_counts,
        notes=notes,
    )
