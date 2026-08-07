"""Concentration limits by name, strategy, and factor/sector -- with
disclosed thresholds, the same HHI + threshold-flag pattern
`credit/concentration.py` already established for counterparty risk,
generalized here to any rollup dimension via `aggregation/rollup.py`'s own
`rollup_by`.

Exposure measure, deliberately different from credit/concentration.py's
choice: **gross**, not net, for all three dimensions here. Credit's
counterparty check used net exposure because a custodial default nets to
one account value (see that module's docstring); a name/strategy/sector
concentration limit is about how much capital and risk is actually
deployed to that name/strategy/sector, where aggregation.py's own earlier
finding applies directly -- "a portfolio that's net-zero can still carry
large gross (offsetting long/short) risk," so gross is the right measure
for a risk *limit*, not a custodial-loss estimate. Both choices are
disclosed explicitly so they don't read as an inconsistency.
"""

from __future__ import annotations

from dataclasses import dataclass

from aggregation.rollup import rollup_by
from connectors.schema import Position
from factor_model.factors import sector_of

DEFAULT_THRESHOLDS = {
    "name": 0.20,      # no single ticker should exceed 20% of gross exposure
    "strategy": 0.50,  # no single strategy should exceed 50%
    "sector": 0.40,    # no single sector should exceed 40%
}


@dataclass
class ConcentrationCheck:
    dimension: str
    exposures: dict[str, float]  # gross market value per key
    total_exposure: float
    shares: dict[str, float]
    herfindahl_index: float
    threshold: float
    flagged: list[str]


def _check(positions: list[Position], key_fn, dimension: str, threshold: float) -> ConcentrationCheck:
    buckets = rollup_by(positions, key_fn)
    exposures = {k: b.gross_market_value for k, b in buckets.items()}
    total = sum(exposures.values())

    if total == 0:
        return ConcentrationCheck(dimension, exposures, 0.0, {}, 0.0, threshold, [])

    shares = {k: v / total for k, v in exposures.items()}
    hhi = sum(s**2 for s in shares.values())
    flagged = [k for k, s in shares.items() if s > threshold]

    return ConcentrationCheck(dimension, exposures, total, shares, hhi, threshold, flagged)


def check_by_name(positions: list[Position], threshold: float = DEFAULT_THRESHOLDS["name"]) -> ConcentrationCheck:
    return _check(positions, lambda p: p.asset, "name", threshold)


def check_by_strategy(
    positions: list[Position], threshold: float = DEFAULT_THRESHOLDS["strategy"]
) -> ConcentrationCheck:
    return _check(positions, lambda p: p.strategy, "strategy", threshold)


def check_by_sector(
    positions: list[Position], threshold: float = DEFAULT_THRESHOLDS["sector"]
) -> ConcentrationCheck:
    return _check(positions, lambda p: sector_of(p.asset) or "unclassified", "sector", threshold)
