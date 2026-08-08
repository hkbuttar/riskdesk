"""Tests for connectors/bookmaker.py's deployment fallback: bookmaker's
local SQLite file is not remotely reachable at all (unlike the Postgres-
backed connectors, there's no env-var override that can fix this -- a
local file just doesn't exist anywhere else), so this connector falls back
to a committed stand-in snapshot instead, the same disclosed pattern
already used for pairtrade-lab-1 and voledge.
"""

from __future__ import annotations

from connectors.bookmaker import fetch_positions
from connectors.schema import AssetClass, Counterparty, DataProvenance


def test_falls_back_to_standin_when_db_unreachable(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))  # no bookmaker.db anywhere here

    positions, meta = fetch_positions()

    assert len(positions) == 1
    assert positions[0].strategy == "bookmaker"
    assert positions[0].provenance == DataProvenance.BACKTEST_STANDIN
    assert "stand-in" in meta.notes.lower()
    assert "standins" in meta.read_from


def test_standin_position_matches_real_binance_real_run_shape(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKDESK_SIBLINGS_ROOT", str(tmp_path))

    positions, _ = fetch_positions()
    position = positions[0]

    assert position.asset == "BTCUSDT"
    assert position.asset_class == AssetClass.CRYPTO
    assert position.counterparty == Counterparty.BINANCE
    assert position.extra["data_source"] == "binance_real"


def test_prefers_real_local_db_over_standin_when_reachable():
    # No RISKDESK_SIBLINGS_ROOT override here -- uses the real local
    # bookmaker.db on this machine, confirming the fallback doesn't
    # shadow real local data when it IS actually available.
    positions, meta = fetch_positions()
    assert len(positions) == 1
    assert "bookmaker.db" in meta.read_from
    assert "stand-in" not in meta.notes.lower() or "standins" not in meta.read_from
