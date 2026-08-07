"""VaR backtesting: Kupiec's Proportion-of-Failures (POF) test and
Christoffersen's independence test, the standard statistical-rigor pair
for asking "is this VaR model actually calibrated" with an actual
hypothesis test and p-value, not just eyeballing a breach count.

- **Kupiec POF**: tests whether the observed breach RATE matches the
  expected rate (`1 - confidence`) under a binomial model -- e.g. a 95%
  VaR should be breached on ~5% of days; POF asks whether the actual
  breach frequency is statistically distinguishable from 5%, too high
  (VaR too tight) or too low (VaR too loose, wasting capital).
- **Christoffersen independence**: even a VaR with exactly the right
  breach RATE can be badly calibrated if breaches cluster together in
  time (e.g. five breaches in one bad week, then none for a year) rather
  than being independently scattered -- this tests specifically for that
  clustering via a first-order Markov chain on the breach indicator series.
- **Combined (Christoffersen's full test)**: `LR_pof + LR_ind`, chi-squared
  with 2 degrees of freedom -- the joint test for "correct rate AND no
  clustering," the real bar a well-calibrated VaR model needs to clear.

Applied here to both the pooled model (one static VaR for every day) and a
regime-conditional model (whichever regime's VaR applies on that specific
day, the same day-by-day model-selection logic monitor/live.py uses live)
over the SAME evaluation period -- an apples-to-apples comparison of
whether regime-conditioning actually improves calibration, with a real
p-value attached to the answer instead of an eyeballed breach count.

Disclosed scope: this is an IN-SAMPLE calibration check -- VaR is
estimated once over the same 2-year window the breaches are counted
against, not a rolling/expanding walk-forward backtest (which would need
a fresh re-fit for every single day, computationally far more expensive).
stress/historical.py's VaR-breach validation is this project's genuinely
out-of-sample complement (fit on pre-window data only, tested against a
real crisis window); this module is the formal, full-sample statistical
test the plan calls for, not a replacement for that out-of-sample check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlogy

from regime.conditional import align_regime_labels, split_by_regime
from risk_measures.var import historical_simulation


@dataclass
class BacktestResult:
    model_label: str
    n_obs: int
    n_breaches: int
    expected_breach_rate: float
    observed_breach_rate: float
    kupiec_lr: float
    kupiec_p_value: float
    christoffersen_lr: float
    christoffersen_p_value: float
    combined_lr: float
    combined_p_value: float

    @property
    def well_calibrated(self) -> bool:
        """Fails to reject the null (correct rate, no clustering) at the
        conventional 5% significance level -- "well calibrated" in the
        specific, limited sense a hypothesis test can support: no
        statistically detectable problem, not proof of correctness.
        """
        return self.combined_p_value > 0.05


def kupiec_pof_test(n_obs: int, n_breaches: int, confidence: float) -> tuple[float, float]:
    p = 1 - confidence
    pi_hat = n_breaches / n_obs
    log_l0 = xlogy(n_obs - n_breaches, 1 - p) + xlogy(n_breaches, p)
    log_l1 = xlogy(n_obs - n_breaches, 1 - pi_hat) + xlogy(n_breaches, pi_hat)
    lr = -2 * (log_l0 - log_l1)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def christoffersen_independence_test(breach_indicator: np.ndarray) -> tuple[float, float]:
    prev, curr = breach_indicator[:-1], breach_indicator[1:]
    n00 = int(np.sum((~prev) & (~curr)))
    n01 = int(np.sum((~prev) & curr))
    n10 = int(np.sum(prev & (~curr)))
    n11 = int(np.sum(prev & curr))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi2 = (n01 + n11) / (n00 + n01 + n10 + n11)

    log_l0 = xlogy(n00 + n10, 1 - pi2) + xlogy(n01 + n11, pi2)
    log_l1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    lr = -2 * (log_l0 - log_l1)
    return float(lr), float(stats.chi2.sf(lr, df=1))


def backtest_breach_series(breach_indicator: pd.Series, confidence: float, model_label: str) -> BacktestResult:
    indicator = breach_indicator.to_numpy().astype(bool)
    n_obs, n_breaches = len(indicator), int(indicator.sum())

    kupiec_lr, kupiec_p = kupiec_pof_test(n_obs, n_breaches, confidence)
    christoffersen_lr, christoffersen_p = christoffersen_independence_test(indicator)
    combined_lr = kupiec_lr + christoffersen_lr
    combined_p = float(stats.chi2.sf(combined_lr, df=2))

    return BacktestResult(
        model_label=model_label, n_obs=n_obs, n_breaches=n_breaches,
        expected_breach_rate=1 - confidence, observed_breach_rate=n_breaches / n_obs,
        kupiec_lr=kupiec_lr, kupiec_p_value=kupiec_p,
        christoffersen_lr=christoffersen_lr, christoffersen_p_value=christoffersen_p,
        combined_lr=combined_lr, combined_p_value=combined_p,
    )


def pooled_breach_series(pnl_series: pd.Series, confidence: float) -> pd.Series:
    var_dollar = historical_simulation(pnl_series, confidence).var_dollar
    loss = -pnl_series
    return loss > var_dollar


def regime_conditional_breach_series(
    pnl_series: pd.Series, regime_labels: pd.Series, confidence: float, min_days: int = 30
) -> tuple[pd.Series, list[str]]:
    """Breach indicator using whichever regime's own VaR applies on each
    day -- the same day-by-day model-selection logic monitor/live.py uses
    live. A regime with too few days to fit its own VaR falls back to the
    pooled VaR for its days (disclosed via notes), matching
    monitor/live.py's honest fallback rather than silently reusing another
    regime's estimate.
    """
    notes: list[str] = []
    aligned = align_regime_labels(pnl_series.index, regime_labels)
    by_regime = split_by_regime(pnl_series, aligned)

    regime_var: dict[str, float] = {}
    for regime, series in by_regime.items():
        if len(series) < min_days:
            notes.append(f"{regime}: only {len(series)} days -- falling back to pooled VaR for these days.")
            continue
        regime_var[regime] = historical_simulation(series, confidence).var_dollar

    pooled_var = historical_simulation(pnl_series, confidence).var_dollar
    applicable_var = aligned.map(lambda r: regime_var.get(r, pooled_var))

    loss = -pnl_series
    return (loss > applicable_var), notes
