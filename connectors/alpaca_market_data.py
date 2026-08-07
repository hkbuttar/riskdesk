"""Alpaca historical market-data adapter shared by every analytics module.

Stocks and crypto use different Alpaca endpoints and symbol conventions.
This module hides that split and returns ordinary pandas frames whose
columns use RiskDesk's canonical symbols (for example ``BTC-USD``).
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Iterable

import httpx
import pandas as pd
from dotenv import dotenv_values

from connectors._env import load_repo_env

DATA_URL = os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets").rstrip("/")
CRYPTO_SYMBOLS = {"BTC-USD": "BTC/USD"}
REQUEST_TIMEOUT_SECONDS = 20.0


class AlpacaMarketDataError(RuntimeError):
    """Raised when Alpaca credentials or a market-data request are invalid."""


def _credentials() -> tuple[str, str]:
    local = dotenv_values(".env")
    sibling = load_repo_env("alpha-signal-lab")
    key = (
        os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_API_KEY")
        or local.get("APCA_API_KEY_ID")
        or local.get("ALPACA_API_KEY")
        or sibling.get("APCA_API_KEY_ID")
        or sibling.get("ALPACA_API_KEY")
    )
    secret = (
        os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET_KEY")
        or local.get("APCA_API_SECRET_KEY")
        or local.get("ALPACA_SECRET_KEY")
        or sibling.get("APCA_API_SECRET_KEY")
        or sibling.get("ALPACA_SECRET_KEY")
    )
    if not key or not secret:
        raise AlpacaMarketDataError(
            "Alpaca credentials are missing; set APCA_API_KEY_ID and APCA_API_SECRET_KEY."
        )
    return str(key), str(secret)


def _date_range(
    period: str | None, start: str | dt.date | None, end: str | dt.date | None
) -> tuple[str, str]:
    today = dt.datetime.now(dt.timezone.utc).date()
    if end is None:
        end_date = today + dt.timedelta(days=1)
    else:
        end_date = dt.date.fromisoformat(str(end)) if not isinstance(end, dt.date) else end
    if start is not None:
        start_date = dt.date.fromisoformat(str(start)) if not isinstance(start, dt.date) else start
    elif period:
        normalized = period.lower()
        units = {"d": 1, "mo": 31, "m": 31, "y": 366}
        unit = next((candidate for candidate in ("mo", "d", "m", "y") if normalized.endswith(candidate)), None)
        if unit is None:
            raise ValueError(f"Unsupported period {period!r}; use values such as '30d', '3mo', or '2y'.")
        amount = int(normalized[: -len(unit)])
        days = amount * units[unit]
        start_date = end_date - dt.timedelta(days=days)
    else:
        raise ValueError("Either period or start must be supplied.")
    return start_date.isoformat(), end_date.isoformat()


def _request_bars(path: str, params: dict[str, str]) -> dict:
    key, secret = _credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    bars: dict[str, list[dict]] = {}
    page_token: str | None = None
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        while True:
            page_params = dict(params)
            if page_token:
                page_params["page_token"] = page_token
            response = client.get(f"{DATA_URL}{path}", params=page_params, headers=headers)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:300]
                raise AlpacaMarketDataError(
                    f"Alpaca market-data request failed ({response.status_code}): {detail}"
                ) from exc
            payload = response.json()
            for symbol, rows in payload.get("bars", {}).items():
                bars.setdefault(symbol, []).extend(rows)
            page_token = payload.get("next_page_token")
            if not page_token:
                return bars


def fetch_daily_bars(
    symbols: Iterable[str], *, period: str | None = None,
    start: str | dt.date | None = None, end: str | dt.date | None = None,
) -> dict[str, pd.DataFrame]:
    """Return daily OHLCV bars keyed by canonical RiskDesk symbol."""
    requested = sorted(set(symbols))
    if not requested:
        return {}
    start_date, end_date = _date_range(period, start, end)
    stock_symbols = [symbol for symbol in requested if symbol not in CRYPTO_SYMBOLS]
    crypto_symbols = [symbol for symbol in requested if symbol in CRYPTO_SYMBOLS]
    raw: dict[str, list[dict]] = {}

    if stock_symbols:
        raw.update(_request_bars("/v2/stocks/bars", {
            "symbols": ",".join(stock_symbols), "timeframe": "1Day",
            "start": start_date, "end": end_date, "adjustment": "all",
            "feed": os.environ.get("ALPACA_STOCK_FEED", "iex"), "limit": "10000",
        }))
    if crypto_symbols:
        alpaca_crypto = [CRYPTO_SYMBOLS[symbol] for symbol in crypto_symbols]
        crypto_raw = _request_bars("/v1beta3/crypto/us/bars", {
            "symbols": ",".join(alpaca_crypto), "timeframe": "1Day",
            "start": start_date, "end": end_date, "limit": "10000",
        })
        reverse = {value: key for key, value in CRYPTO_SYMBOLS.items()}
        raw.update({reverse.get(symbol, symbol): rows for symbol, rows in crypto_raw.items()})

    result: dict[str, pd.DataFrame] = {}
    for symbol in requested:
        rows = raw.get(symbol, [])
        if not rows:
            result[symbol] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            continue
        frame = pd.DataFrame(rows).rename(columns={
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "timestamp",
        })
        timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop("timestamp"), utc=True))
        # Alpaca timestamps stock bars at the session boundary and crypto at
        # UTC midnight. RiskDesk models daily observations, so normalize both
        # onto the calendar date before aligning cross-asset histories.
        frame.index = timestamps.tz_localize(None).normalize()
        result[symbol] = frame[["open", "high", "low", "close", "volume"]].sort_index()
    return result


def fetch_history(
    symbols: Iterable[str], *, period: str | None = None,
    start: str | dt.date | None = None, end: str | dt.date | None = None,
    field: str = "close",
) -> pd.DataFrame:
    """Return one aligned date-indexed column per requested symbol."""
    bars = fetch_daily_bars(symbols, period=period, start=start, end=end)
    series = {symbol: frame[field].rename(symbol) for symbol, frame in bars.items() if not frame.empty}
    return pd.DataFrame(series).sort_index()
