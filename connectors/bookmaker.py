"""Connector for bookmaker.

bookmaker is a market-making simulator (synthetic limit-order-book, plus a
real-data harness validated against captured Binance BTCUSDT L1 snapshots
and trade prints under data/binance_sample/), backtest-only. It persists to
a local SQLite file (`bookmaker.db`, SQLAlchemy-managed) with tables
`simulation_runs` (final_inventory, final_pnl, strategy_name, data_source,
...) and `book_snapshots` (run_id, time, inventory, cash, equity). There is
one instrument per run, so a "position" here is the final inventory of the
most recent completed simulation_run.

This connector prefers the most recent `data_source = 'binance_real'` run
(strategy quoted against real captured Binance BTCUSDT data, filled against
the real trade tape -- see backtest/binance_backtest.py) over a
`data_source = 'synthetic'` run, since real market microstructure is a
better disclosed stand-in than a purely synthetic LOB. bookmaker's own
FastAPI `/simulate` endpoint only exposes `data_source="synthetic"`
(`backend/schemas.py: DATA_SOURCES = ("synthetic",)`, with a comment noting
Binance simulate-on-demand isn't wired into the API); the binance_real run
was produced by calling `backtest/binance_backtest.py::run_binance_backtest`
directly against the captured sample and inserting a `SimulationRun` row the
same way the API does, as this project's own tests exercise that same
real-data path. A binance_real position is tagged AssetClass.CRYPTO /
Counterparty.BINANCE (it reflects real BTCUSDT market data, even though no
live order was ever placed at Binance); a synthetic-run position is tagged
AssetClass.SYNTHETIC / Counterparty.NONE, since it reflects no real venue at
all.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

from connectors._env import repo_path
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position, SourceMeta

REPO = "bookmaker"
DB_FILENAME = "bookmaker.db"


def fetch_positions() -> tuple[list[Position], SourceMeta]:
    db_path = repo_path(REPO) / DB_FILENAME
    if not db_path.exists():
        return [], SourceMeta(
            repo=REPO,
            read_from=str(db_path),
            provenance=DataProvenance.BACKTEST_STANDIN,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes=f"{DB_FILENAME} not found at {db_path}.",
        )

    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(
            "SELECT id, strategy_name, data_source, final_inventory, final_pnl, created_at "
            "FROM simulation_runs WHERE status = 'completed' "
            "ORDER BY (data_source = 'binance_real') DESC, created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
    except Exception as exc:  # noqa: BLE001
        return [], SourceMeta(
            repo=REPO,
            read_from=str(db_path),
            provenance=DataProvenance.BACKTEST_STANDIN,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes=f"Query failed: {type(exc).__name__}: {exc}",
        )
    finally:
        con.close()

    if row is None:
        return [], SourceMeta(
            repo=REPO,
            read_from=f"{db_path}::simulation_runs",
            provenance=DataProvenance.BACKTEST_STANDIN,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes="No completed simulation_runs rows yet; DB is provisioned but unpopulated.",
        )

    run_id, strategy_name, data_source, final_inventory, final_pnl, created_at = row
    is_real = data_source == "binance_real"
    position = Position(
        strategy=REPO,
        asset="BTCUSDT" if is_real else "BOOKMAKER-SIM-LOB",
        quantity=float(final_inventory or 0.0),
        market_value=None,
        asset_class=AssetClass.CRYPTO if is_real else AssetClass.SYNTHETIC,
        counterparty=Counterparty.BINANCE if is_real else Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=dt.date.today(),
        strategy_tag=strategy_name,
        extra={
            "run_id": run_id,
            "data_source": data_source,
            "final_pnl": final_pnl,
            "run_created_at": str(created_at),
        },
    )
    return [position], SourceMeta(
        repo=REPO,
        read_from=f"{db_path}::simulation_runs(id={run_id}, data_source={data_source})",
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=dt.datetime.now(dt.timezone.utc),
        n_positions=1,
        notes=f"Most recent completed run: strategy={strategy_name}, "
        f"data_source={data_source}, final_pnl={final_pnl}.",
    )
