"""Named risk factors: market beta, sector, crypto -- plus the sector
tagging reused directly from alpha-signal-lab's own universe config, not
regenerated, per the plan's "sector exposures from alpha-signal-lab's
tagging."

`SECTOR_TICKERS` below is copied verbatim from
alpha-signal-lab/config/universe.py (four sectors: Technology, Healthcare,
Financials, Energy) -- every equity ticker this project's book actually
holds falls inside one of these four sectors already, so no extension was
needed. Sector *factor returns* come from the standard SPDR sector ETFs
(XLK/XLV/XLF/XLE) rather than from an average of this project's own held
tickers -- using the book's own returns as its own factor would be
circular (a position would partly explain itself); real sector ETFs are
the standard, independent proxy.
"""

from __future__ import annotations

import pandas as pd
from connectors.alpaca_market_data import fetch_history

# Copied verbatim from alpha-signal-lab/config/universe.py::SECTOR_TICKERS.
SECTOR_TICKERS: dict[str, list[str]] = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "AMD", "INTC", "TXN"],
    "Healthcare": ["JNJ", "UNH", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "BLK", "SPGI", "PNC", "USB", "TFC"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB", "KMI", "DVN"],
}

TICKER_SECTOR: dict[str, str] = {
    ticker: sector for sector, tickers in SECTOR_TICKERS.items() for ticker in tickers
}

SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Energy": "XLE",
}

MARKET_FACTOR = "SPY"
CRYPTO_FACTOR = "BTC-USD"


def sector_of(ticker: str) -> str | None:
    return TICKER_SECTOR.get(ticker)


def fetch_factor_returns(sectors_held: set[str], period: str = "2y") -> tuple[pd.DataFrame, list[str]]:
    """Daily returns for the market, crypto, and every sector ETF actually
    relevant to `sectors_held` (skip fetching ETFs for sectors nothing in
    the book belongs to).
    """
    notes: list[str] = []
    etf_tickers = [SECTOR_ETF[s] for s in sectors_held if s in SECTOR_ETF]
    all_tickers = sorted({MARKET_FACTOR, CRYPTO_FACTOR, *etf_tickers})

    raw = fetch_history(all_tickers, period=period, field="close")

    missing = [t for t in all_tickers if t not in raw.columns or raw[t].dropna().empty]
    for t in missing:
        notes.append(f"{t}: no price history -- excluded from the factor set.")
    present = [t for t in all_tickers if t not in missing]

    prices = raw[present].dropna(how="any")
    returns = prices.pct_change().dropna(how="any")
    notes.append(f"Factor returns: {len(returns)} aligned days across {len(present)} factors.")
    return returns, notes
