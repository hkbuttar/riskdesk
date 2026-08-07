"""Rolls up valued positions to portfolio, strategy, asset-class, and
counterparty level.

Net market value sums signed exposure (shorts subtract); gross sums
absolute exposure -- the standard pair, since a portfolio that's net-zero
can still carry large gross (offsetting long/short) risk that net alone
would hide. Positions with no market_value (unpriced -- see valuation.py)
are excluded from both sums but counted separately in `n_unpriced`, so a
rollup total never silently understates exposure without saying so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from connectors.schema import Position


@dataclass
class RollupBucket:
    key: str
    net_market_value: float = 0.0
    gross_market_value: float = 0.0
    n_positions: int = 0
    n_priced: int = 0
    n_unpriced: int = 0
    unpriced_assets: list[str] = field(default_factory=list)

    def add(self, p: Position) -> None:
        self.n_positions += 1
        if p.market_value is None:
            self.n_unpriced += 1
            self.unpriced_assets.append(f"{p.strategy}/{p.asset}")
            return
        self.n_priced += 1
        self.net_market_value += p.market_value
        self.gross_market_value += abs(p.market_value)


def rollup_by(positions: list[Position], key_fn: Callable[[Position], str]) -> dict[str, RollupBucket]:
    buckets: dict[str, RollupBucket] = {}
    for p in positions:
        key = key_fn(p)
        bucket = buckets.setdefault(key, RollupBucket(key=key))
        bucket.add(p)
    return buckets


def by_strategy(positions: list[Position]) -> dict[str, RollupBucket]:
    return rollup_by(positions, lambda p: p.strategy)


def by_asset_class(positions: list[Position]) -> dict[str, RollupBucket]:
    return rollup_by(positions, lambda p: p.asset_class.value)


def by_counterparty(positions: list[Position]) -> dict[str, RollupBucket]:
    return rollup_by(positions, lambda p: p.counterparty.value)


def portfolio_total(positions: list[Position]) -> RollupBucket:
    bucket = RollupBucket(key="portfolio")
    for p in positions:
        bucket.add(p)
    return bucket
