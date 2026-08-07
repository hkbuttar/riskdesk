"""Connector for alpha-signal-lab.

Reads: the latest row of `portfolio_snapshots` (columns: date, equity, cash,
positions_json JSONB) from alpha-signal-lab's own Postgres, via the
DATABASE_URL in alpha-signal-lab/.env (a Render-managed instance; this repo
is the only one of the six that actually runs live, via a daily paper-trading
GitHub Actions scheduler — see live/scheduler.py in that repo).

Schema reconciliation: positions_json is a flat {ticker: signed_shares} dict.
It carries no per-ticker price, so `Position.market_value`/`price` are left
None here — the aggregation layer is responsible for attaching a
point-in-time price and computing market value, since pricing is a valuation
concern shared across all connectors, not something to duplicate per-adapter.
"""

from __future__ import annotations

import datetime as dt

import psycopg

from connectors._env import load_repo_env
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position, SourceMeta

REPO = "alpha-signal-lab"


def fetch_positions() -> tuple[list[Position], SourceMeta]:
    env = load_repo_env(REPO)
    database_url = env.get("DATABASE_URL")
    if not database_url:
        return [], SourceMeta(
            repo=REPO,
            read_from="portfolio_snapshots (DATABASE_URL not set)",
            provenance=DataProvenance.LIVE_PAPER,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes="No DATABASE_URL found in alpha-signal-lab/.env; cannot connect.",
        )

    try:
        with psycopg.connect(database_url, connect_timeout=8) as con, con.cursor() as cur:
            cur.execute(
                "SELECT date, equity, cash, positions_json FROM portfolio_snapshots "
                "ORDER BY date DESC LIMIT 1"
            )
            row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001 - connector boundary, report don't crash
        return [], SourceMeta(
            repo=REPO,
            read_from="portfolio_snapshots",
            provenance=DataProvenance.LIVE_PAPER,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes=f"Connection/query failed: {type(exc).__name__}: {exc}",
        )

    if row is None:
        return [], SourceMeta(
            repo=REPO,
            read_from="portfolio_snapshots",
            provenance=DataProvenance.LIVE_PAPER,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes="Table is empty; scheduler may not have run yet.",
        )

    snapshot_date, equity, cash, positions_json = row
    positions = [
        Position(
            strategy=REPO,
            asset=ticker,
            quantity=float(shares),
            market_value=None,
            asset_class=AssetClass.EQUITY,
            counterparty=Counterparty.ALPACA,
            provenance=DataProvenance.LIVE_PAPER,
            as_of=snapshot_date,
            extra={"snapshot_equity": equity, "snapshot_cash": cash},
        )
        for ticker, shares in (positions_json or {}).items()
    ]
    return positions, SourceMeta(
        repo=REPO,
        read_from="portfolio_snapshots",
        provenance=DataProvenance.LIVE_PAPER,
        as_of=dt.datetime.combine(snapshot_date, dt.time.min, tzinfo=dt.timezone.utc),
        n_positions=len(positions),
        notes=f"Latest snapshot {snapshot_date}: equity={equity}, cash={cash}.",
    )
