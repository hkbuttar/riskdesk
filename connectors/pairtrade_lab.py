"""Connector for pairtrade-lab-1.

pairtrade-lab-1 is backtest/monitoring-only (no order execution). Its
`backtest/portfolio.py::Portfolio` records a full per-date position history
in memory during a backtest run, but `run_backtest.py` / `run_comparison.py`
never persist that history to disk -- only aggregate performance metrics get
written to `backtest/results/comparison_2018-01-01_2025-01-01.json`.

Where a prior project never ran live, its most recent backtest's simulated
positions are used as a disclosed stand-in: this connector reads a one-off
stand-in snapshot at `connectors/standins/pairtrade_lab_1_positions.json`,
produced by actually re-running pairtrade-lab-1's own, unmodified
`backtest.engine.run_backtest()` (its own venv, its own cached prices) over
2023-01-03..2024-12-31 and capturing the final portfolio state via a
non-invasive monkeypatch (no source file in pairtrade-lab-1 was edited) --
see that JSON file's "_disclosure" field for the exact method and caveat:
the strategy was flat at the window's end, so the snapshot is dated to the
last day within the window it actually held a position, not "now" and not
the window end. Regenerate by re-running the same script if a fresher
stand-in is wanted.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position, SourceMeta

REPO = "pairtrade-lab-1"
STANDIN_FILE = Path(__file__).parent / "standins" / "pairtrade_lab_1_positions.json"


def fetch_positions() -> tuple[list[Position], SourceMeta]:
    if not STANDIN_FILE.exists():
        return [], SourceMeta(
            repo=REPO,
            read_from=str(STANDIN_FILE),
            provenance=DataProvenance.BACKTEST_STANDIN,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes=f"{STANDIN_FILE} not found; no stand-in snapshot has been generated yet.",
        )

    with STANDIN_FILE.open() as f:
        data = json.load(f)

    as_of_date = dt.datetime.fromisoformat(data["as_of"]).date()
    positions = [
        Position(
            strategy=REPO,
            asset=row["ticker"],
            quantity=float(row["shares"]),
            market_value=None,
            asset_class=AssetClass.EQUITY,
            counterparty=Counterparty.NONE,
            provenance=DataProvenance.BACKTEST_STANDIN,
            as_of=as_of_date,
            extra={"backtest_window": data["window"], "backtest_cash": data["cash"]},
        )
        for row in data["positions"]
    ]
    return positions, SourceMeta(
        repo=REPO,
        read_from=str(STANDIN_FILE),
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=dt.datetime.combine(as_of_date, dt.time.min, tzinfo=dt.timezone.utc),
        n_positions=len(positions),
        notes=(
            f"Disclosed backtest stand-in (not live): last non-flat snapshot within "
            f"backtest window {data['window']['start']}..{data['window']['end']} was "
            f"{data['as_of']}; sharpe={data['metrics'].get('sharpe_ratio')}, "
            f"n_trades={data['metrics'].get('n_trades')}."
        ),
    )
