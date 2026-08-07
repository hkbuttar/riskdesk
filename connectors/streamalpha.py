"""Connector for streamalpha.

streamalpha is a real-time anomaly-detection pipeline (Alpaca trade/quote
WebSocket -> Kafka -> streaming detectors -> Postgres), not a trading
strategy: it has no Position/Trade/Portfolio concept anywhere in its
codebase. It therefore contributes zero rows to RiskDesk's position
aggregation by design, not by connector failure.

What IS useful from this repo: `anomalies` (ticker, window_start,
anomaly_type IN ('volume_anomaly','regime_change'), details JSONB,
detected_at), read from streamalpha/.env's DATABASE_URL. This is exposed
as a risk-factor/context signal (e.g. a live regime_change flag on a name
already held elsewhere in the aggregated book) rather than a position, for
later work (regime classification, live monitoring) to optionally consume.
"""

from __future__ import annotations

import datetime as dt

import psycopg

from connectors._env import load_repo_env
from connectors.schema import DataProvenance, Position, SourceMeta

REPO = "streamalpha"


def fetch_positions() -> tuple[list[Position], SourceMeta]:
    """Always returns an empty position list -- see module docstring."""
    return [], SourceMeta(
        repo=REPO,
        read_from="n/a",
        provenance=DataProvenance.LIVE_PAPER,
        as_of=dt.datetime.now(dt.timezone.utc),
        n_positions=0,
        notes="streamalpha has no position/portfolio concept (anomaly-detection "
        "pipeline only); excluded from position aggregation by design.",
    )


def fetch_recent_anomalies(limit: int = 50) -> tuple[list[dict], SourceMeta]:
    env = load_repo_env(REPO)
    database_url = env.get("DATABASE_URL")
    if not database_url:
        return [], SourceMeta(
            repo=REPO,
            read_from="anomalies (DATABASE_URL not set)",
            provenance=DataProvenance.LIVE_PAPER,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes="No DATABASE_URL found in streamalpha/.env; cannot connect.",
        )

    try:
        with psycopg.connect(database_url, connect_timeout=8) as con, con.cursor() as cur:
            cur.execute(
                "SELECT ticker, window_start, anomaly_type, details, detected_at "
                "FROM anomalies ORDER BY detected_at DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001
        return [], SourceMeta(
            repo=REPO,
            read_from="anomalies",
            provenance=DataProvenance.LIVE_PAPER,
            as_of=dt.datetime.now(dt.timezone.utc),
            n_positions=0,
            notes=f"Connection/query failed: {type(exc).__name__}: {exc}",
        )

    anomalies = [
        {
            "ticker": ticker,
            "window_start": window_start,
            "anomaly_type": anomaly_type,
            "details": details,
            "detected_at": detected_at,
        }
        for ticker, window_start, anomaly_type, details, detected_at in rows
    ]
    return anomalies, SourceMeta(
        repo=REPO,
        read_from="anomalies",
        provenance=DataProvenance.LIVE_PAPER,
        as_of=dt.datetime.now(dt.timezone.utc),
        n_positions=len(anomalies),
        notes="n_positions here counts anomaly rows, not positions (this repo has none).",
    )
