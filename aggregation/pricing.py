"""Point-in-time pricing for the aggregation layer.

Connectors deliberately leave `Position.price`/`market_value` unset (see
each connector's docstring) -- pricing is a valuation concern shared across
every source repo, so it lives here once rather than duplicated per-adapter.

Each position is priced as of its OWN `as_of` date, not "today" -- blending
a 2024-04-25 backtest stand-in with today's price would silently conflate
two different points in time into one number. For live positions `as_of`
is today (or the latest trading day), so this collapses to "current price"
naturally; for backtest stand-ins it's a real historical lookup.

Symbol translation: internal asset tickers occasionally don't match Alpaca's
canonical convention (e.g. bookmaker's "BTCUSDT" -> "BTC-USD").
`SYMBOL_MAP` is the disclosed, explicit translation table -- deliberately
not a guessing heuristic.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from connectors.alpaca_market_data import fetch_daily_bars

SYMBOL_MAP: dict[str, str] = {
    "BTCUSDT": "BTC-USD",
}

_HISTORY_LOOKBACK_DAYS = 7  # window before as_of to find the most recent prior close
_HISTORY_LOOKAHEAD_DAYS = 2  # small buffer past as_of so an exact-date match isn't missed

_cache: dict[tuple[str, dt.date], "PriceResult"] = {}


@dataclass
class PriceResult:
    price: float | None
    source: str
    note: str = ""


def resolve_symbol(asset: str) -> str:
    return SYMBOL_MAP.get(asset, asset)


def fetch_price_asof(asset: str, as_of: dt.date) -> PriceResult:
    """Most recent close on or before `as_of`, via Alpaca.

    Never raises: network/lookup failures come back as a PriceResult with
    price=None and a note explaining why, matching the rest of this
    project's "disclose, don't fabricate or crash" pattern for missing data.
    """
    cache_key = (asset, as_of)
    if cache_key in _cache:
        return _cache[cache_key]

    symbol = resolve_symbol(asset)
    start = as_of - dt.timedelta(days=_HISTORY_LOOKBACK_DAYS)
    end = as_of + dt.timedelta(days=_HISTORY_LOOKAHEAD_DAYS)

    try:
        hist = fetch_daily_bars([symbol], start=start, end=end).get(symbol)
    except Exception as exc:  # noqa: BLE001 - pricing boundary, disclose don't crash
        result = PriceResult(price=None, source=symbol, note=f"Alpaca error: {type(exc).__name__}: {exc}")
        _cache[cache_key] = result
        return result

    if hist.empty:
        result = PriceResult(price=None, source=symbol, note=f"No price history returned for {symbol!r}.")
        _cache[cache_key] = result
        return result

    hist = hist.tz_localize(None) if hist.index.tz is not None else hist
    on_or_before = hist[hist.index.date <= as_of]
    row = on_or_before.iloc[-1] if not on_or_before.empty else hist.iloc[0]
    used_date = row.name.date()

    note = "" if used_date == as_of else f"No close on {as_of}; used nearest available ({used_date})."
    result = PriceResult(price=float(row["close"]), source=f"alpaca:{symbol}", note=note)
    _cache[cache_key] = result
    return result
