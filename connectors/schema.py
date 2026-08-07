"""Common position/exposure schema that every connector adapter normalizes into.

Design note: the six source repos have wildly different native shapes (Postgres
JSONB snapshots, SQLite inventory rows, flat JSON backtest results, no position
concept at all). Rather than propagate that heterogeneity, every adapter in this
package maps its source data down to `Position` and `SourceMeta`, defined here.
Fields that a source repo genuinely cannot supply (e.g. entry price is not
tracked anywhere in alpha-signal-lab's snapshot schema) are left `None`, not
guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class AssetClass(str, Enum):
    EQUITY = "equity"
    OPTION = "option"
    CRYPTO = "crypto"
    SYNTHETIC = "synthetic"  # bookmaker's simulated single-asset LOB


class Counterparty(str, Enum):
    ALPACA = "alpaca"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    NONE = "none"  # no live venue (e.g. market-data-sourced backtest positions)


class DataProvenance(str, Enum):
    """Was this position pulled from a live/paper trading feed, or backfilled
    from a repo's most recent saved backtest as a disclosed stand-in? Where a
    prior project never ran live, its most recent backtest's simulated
    positions are used as a disclosed stand-in.
    """

    LIVE_PAPER = "live_paper"
    BACKTEST_STANDIN = "backtest_standin"


@dataclass
class Position:
    strategy: str  # source repo name, e.g. "alpha-signal-lab"
    asset: str  # ticker / symbol, e.g. "AAPL", "BTC-USD"
    quantity: float  # signed; short positions negative
    market_value: float  # quantity * price, in USD
    asset_class: AssetClass
    counterparty: Counterparty
    provenance: DataProvenance
    as_of: date
    price: float | None = None
    entry_price: float | None = None
    strategy_tag: str | None = None  # sub-strategy, e.g. algorithm name in execedge
    greeks: dict[str, float] | None = None  # delta/gamma/vega/theta, options only
    extra: dict = field(default_factory=dict)  # source-specific fields kept for audit


@dataclass
class SourceMeta:
    """Describes exactly what a connector read: what's read and any schema
    reconciliation, documented so nothing is silently assumed.
    """

    repo: str
    read_from: str  # table name, file path, or file glob actually queried
    provenance: DataProvenance
    as_of: datetime
    n_positions: int
    notes: str = ""
