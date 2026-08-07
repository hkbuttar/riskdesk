"""Limit checks feeding the kill-switch: market-risk (VaR as a fraction of
gross exposure) and credit-risk (counterparty concentration, reusing
credit/concentration.py directly) -- independently triggerable, per the
plan's own requirement.
"""

from __future__ import annotations

from dataclasses import dataclass

from connectors.schema import Position
from credit.concentration import check_concentration

VAR_LIMIT_FRACTION = 0.05  # disclosed: 1-day 95% VaR should not exceed 5% of gross exposure


@dataclass
class LimitCheckResult:
    name: str
    breached: bool
    detail: str


def check_var_limit(
    var_dollar: float, gross_exposure: float, limit_fraction: float = VAR_LIMIT_FRACTION
) -> LimitCheckResult:
    if gross_exposure <= 0:
        return LimitCheckResult("var_limit", False, "No gross exposure -- limit not applicable.")

    var_fraction = var_dollar / gross_exposure
    breached = var_fraction > limit_fraction
    detail = (
        f"VaR is {var_fraction:.2%} of gross exposure (limit {limit_fraction:.0%})"
        + (" -- BREACHED" if breached else "")
    )
    return LimitCheckResult("var_limit", breached, detail)


def check_credit_concentration_limit(positions: list[Position]) -> LimitCheckResult:
    concentration = check_concentration(positions)
    breached = len(concentration.flagged) > 0
    if not concentration.exposure_by_counterparty:
        detail = "No real-venue exposure -- limit not applicable."
    elif breached:
        detail = f"Counterparty concentration breached: {concentration.flagged} " \
                  f"(HHI={concentration.herfindahl_index:.3f})"
    else:
        detail = f"No counterparty concentration breach (HHI={concentration.herfindahl_index:.3f})"
    return LimitCheckResult("credit_concentration_limit", breached, detail)
