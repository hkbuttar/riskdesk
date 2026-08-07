"""Regime-conditional tail risk: does the shape of the loss distribution's
tail itself differ by regime, not just its VaR level (already checked in
regime/conditional.py -- see that module's own note explaining why it
deliberately excludes EVT from its regime-conditional methods: a ~160-day
regime bucket gives only ~16 exceedances at the usual 90% threshold,
under `extreme_value/gpd.py::MIN_EXCEEDANCES` (20). This module exists
specifically to work through that data-sufficiency problem properly rather
than skip it, using regime/conditional.py's own partitioning helpers
(`align_regime_labels`, `split_by_regime`) directly rather than
re-implementing them.

The genuinely new question here, beyond what regime/conditional.py already
answered: is a higher volatile-regime VaR driven by a fatter tail SHAPE
(xi genuinely larger -- extreme losses are disproportionately more likely
relative to moderate ones) or just a wider SCALE (beta larger, same
relative tail shape, everything just bigger)? These have different
practical implications: a shape change means the historical calm-period
tail fit is qualitatively wrong for the volatile regime, not just
mis-scaled.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from extreme_value.gpd import GPDFit, fit_gpd_tail, gpd_var_cvar
from regime.conditional import align_regime_labels, split_by_regime


@dataclass
class RegimeGPDComparison:
    pooled_fit: GPDFit | None
    regime_fits: dict[str, GPDFit | None]
    notes: list[str]


def fit_gpd_by_regime(
    pnl_series: pd.Series, regime_labels: pd.Series, threshold_quantile: float = 0.90
) -> RegimeGPDComparison:
    loss = (-pnl_series).to_numpy()
    pooled_fit = fit_gpd_tail(loss, threshold_quantile)

    aligned = align_regime_labels(pnl_series.index, regime_labels)
    by_regime = split_by_regime(pnl_series, aligned)

    notes: list[str] = []
    regime_fits: dict[str, GPDFit | None] = {}
    for regime, series in by_regime.items():
        regime_loss = (-series).to_numpy()
        fit = fit_gpd_tail(regime_loss, threshold_quantile)
        regime_fits[regime] = fit
        if fit is None:
            n_exceed_est = int(len(regime_loss) * (1 - threshold_quantile))
            notes.append(
                f"{regime}: ~{n_exceed_est} exceedances at {threshold_quantile:.0%} threshold on "
                f"{len(regime_loss)} days -- too few to fit a GPD reliably at this threshold."
            )

    return RegimeGPDComparison(pooled_fit=pooled_fit, regime_fits=regime_fits, notes=notes)


def compare_tail_shape(comparison: RegimeGPDComparison) -> pd.DataFrame:
    """A table of xi (shape), beta (scale), and n_exceedances per regime
    plus the pooled fit, for direct numerical comparison -- not embedded in
    a notes string, so shape can be compared across regimes programmatically.
    """
    rows = {}
    if comparison.pooled_fit is not None:
        f = comparison.pooled_fit
        rows["pooled"] = {"xi": f.xi, "beta": f.beta, "n_exceedances": f.n_exceedances, "threshold": f.threshold}
    for regime, fit in comparison.regime_fits.items():
        if fit is not None:
            rows[regime] = {
                "xi": fit.xi, "beta": fit.beta, "n_exceedances": fit.n_exceedances, "threshold": fit.threshold,
            }
    return pd.DataFrame(rows).T
