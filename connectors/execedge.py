"""Connector for execedge.

execedge is an execution-cost benchmark (TWAP/VWAP/AC/RL child-order
scheduling measured against arrival price, across Binance/Coinbase/Kraken
synthetic order flow). It has no Position/Portfolio class anywhere in its
codebase -- its unit of output is a single parent-order `BacktestResponse`
(executed_quantity, executed_cost, opportunity_cost, total_cost_bps), not an
open position that persists over time. It therefore contributes zero rows
to position aggregation, by design, same as streamalpha (see that
connector's docstring) but for a different reason: this repo measures the
cost of *getting into* a position, not the position itself.

execedge also has no results persisted to disk at the time this connector
was written (`backtest/results/` does not exist; only `book_snapshots` in
Postgres, and its docker DB is not running locally) -- disclosed via
SourceMeta.notes rather than silently returning nothing.
"""

from __future__ import annotations

import datetime as dt

from connectors._env import repo_path
from connectors.schema import DataProvenance, Position, SourceMeta

REPO = "execedge"


def fetch_positions() -> tuple[list[Position], SourceMeta]:
    results_dir = repo_path(REPO) / "backtest" / "results"
    notes = (
        "execedge has no position/portfolio concept (execution-cost benchmark only); "
        "excluded from position aggregation by design."
    )
    if not results_dir.exists():
        notes += f" Additionally, {results_dir} does not exist (no persisted backtest runs)."

    return [], SourceMeta(
        repo=REPO,
        read_from="n/a",
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=dt.datetime.now(dt.timezone.utc),
        n_positions=0,
        notes=notes,
    )
