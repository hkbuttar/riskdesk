"""Vega factor exposure: the plan's fourth named factor ("a vol factor from
VolEdge's vega"). Vega risk (sensitivity to implied-vol moves) is not a
price-return factor like the others in this module -- it doesn't belong in
the same OLS regression, since IV moves aren't spanned by the underlying's
own return series -- so it's reported separately here as a direct Greek
aggregation, the same way Step 8's Greek aggregation will eventually work.
"""

from __future__ import annotations

from dataclasses import dataclass

from connectors.schema import Position


@dataclass
class VegaExposure:
    net_vega: float
    n_option_positions: int
    by_position: dict[str, float]


def aggregate_vega(positions: list[Position]) -> VegaExposure:
    by_position: dict[str, float] = {}
    for p in positions:
        if p.greeks and "vega" in p.greeks:
            by_position[f"{p.strategy}/{p.asset}"] = p.quantity * p.greeks["vega"]

    return VegaExposure(
        net_vega=sum(by_position.values()),
        n_option_positions=len(by_position),
        by_position=by_position,
    )
