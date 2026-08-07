"""Counterparty concentration risk: is too much exposure sitting with a
single venue -- the real, disclosed lesson from FTX's 2022 collapse
(already replayed as a real historical window in stress/historical.py),
directly relevant here given streamalpha and bookmaker's crypto-exchange
exposure.

Concentration is computed only among counterparties with a real,
non-zero default probability (Counterparty.NONE -- positions with no live
venue, e.g. yfinance-sourced backtest data -- is excluded): there is no
real venue to be "concentrated" in for a position that isn't actually held
anywhere, and including that bucket would make concentration figures
meaningless (it currently holds the majority of this book's dollar
exposure purely because most connectors' stand-in snapshots have no real
counterparty, not because of any real concentration risk).
"""

from __future__ import annotations

from dataclasses import dataclass

from aggregation.rollup import by_counterparty
from connectors.schema import Counterparty, Position

DEFAULT_CONCENTRATION_THRESHOLD = 0.50  # flag if one counterparty exceeds 50% of real-venue exposure


@dataclass
class ConcentrationResult:
    exposure_by_counterparty: dict[str, float]  # real venues only, |net exposure|
    total_real_venue_exposure: float
    shares: dict[str, float]  # each counterparty's fraction of total
    herfindahl_index: float  # sum of squared shares; 1/n_venues (even split) .. 1.0 (single venue)
    flagged: list[str]  # counterparties exceeding the threshold


def check_concentration(
    positions: list[Position], threshold: float = DEFAULT_CONCENTRATION_THRESHOLD
) -> ConcentrationResult:
    buckets = by_counterparty(positions)
    real_venues = {
        key: abs(bucket.net_market_value)
        for key, bucket in buckets.items()
        if Counterparty(key) != Counterparty.NONE
    }
    total = sum(real_venues.values())

    if total == 0:
        return ConcentrationResult(real_venues, 0.0, {}, 0.0, [])

    shares = {k: v / total for k, v in real_venues.items()}
    hhi = sum(s**2 for s in shares.values())
    flagged = [k for k, s in shares.items() if s > threshold]

    return ConcentrationResult(
        exposure_by_counterparty=real_venues,
        total_real_venue_exposure=total,
        shares=shares,
        herfindahl_index=hhi,
        flagged=flagged,
    )
