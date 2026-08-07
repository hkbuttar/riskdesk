"""Deterministic tests for the shared Alpaca market-data adapter."""

from __future__ import annotations

import pandas as pd
import pytest

import connectors.alpaca_market_data as market_data


def test_missing_credentials_fail_with_actionable_message(monkeypatch):
    monkeypatch.setattr(market_data, "dotenv_values", lambda *_: {})
    monkeypatch.setattr(market_data, "load_repo_env", lambda *_: {})
    for name in ("APCA_API_KEY_ID", "ALPACA_API_KEY", "APCA_API_SECRET_KEY", "ALPACA_SECRET_KEY"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(market_data.AlpacaMarketDataError, match="APCA_API_KEY_ID"):
        market_data._credentials()


def test_fetch_daily_bars_splits_stocks_and_crypto_and_normalizes(monkeypatch):
    calls = []

    def fake_request(path, params):
        calls.append((path, params))
        if "crypto" in path:
            return {"BTC/USD": [{"t": "2026-08-06T00:00:00Z", "o": 9, "h": 12, "l": 8, "c": 11, "v": 3}]}
        return {"SPY": [{"t": "2026-08-06T04:00:00Z", "o": 99, "h": 102, "l": 98, "c": 101, "v": 10}]}

    monkeypatch.setattr(market_data, "_request_bars", fake_request)
    result = market_data.fetch_daily_bars(["SPY", "BTC-USD"], period="1m")

    assert [path for path, _ in calls] == ["/v2/stocks/bars", "/v1beta3/crypto/us/bars"]
    assert calls[1][1]["symbols"] == "BTC/USD"
    assert set(result) == {"SPY", "BTC-USD"}
    assert result["SPY"].loc[pd.Timestamp("2026-08-06"), "close"] == 101
    assert result["BTC-USD"].iloc[0]["volume"] == 3
    assert result["SPY"].index.tz is None


def test_fetch_history_returns_canonical_columns(monkeypatch):
    index = pd.DatetimeIndex(["2026-08-05", "2026-08-06"])
    bars = {
        "SPY": pd.DataFrame({"close": [100.0, 101.0]}, index=index),
        "BTC-USD": pd.DataFrame({"close": [10.0, 11.0]}, index=index),
    }
    monkeypatch.setattr(market_data, "fetch_daily_bars", lambda *args, **kwargs: bars)

    result = market_data.fetch_history(["SPY", "BTC-USD"], period="1m")

    assert list(result.columns) == ["SPY", "BTC-USD"]
    assert result.loc[pd.Timestamp("2026-08-06"), "BTC-USD"] == 11.0


def test_month_period_alias_is_supported():
    start, end = market_data._date_range("3mo", None, None)
    assert (pd.Timestamp(end) - pd.Timestamp(start)).days == 93
