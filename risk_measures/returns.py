"""Historical daily return series for every underlying actually held, and
the resulting hypothetical portfolio P&L series VaR/CVaR is computed from.

Methodology: for each position, its RISK FACTOR is either the asset itself
(equity/crypto) or its underlying (options, via `extra["underlying"]` --
voledge's positions are already carried as delta-equivalent dollar exposure,
see aggregation/valuation.py, so `market_value * underlying_return`
approximates the position's daily P&L to first order, i.e. delta-only,
ignoring gamma/theta -- a disclosed simplification consistent with the
delta-equivalent approach already used for valuation, not a new one
introduced here).

The portfolio's hypothetical historical daily P&L on day t is:

    pnl_t = sum_i market_value_i * return_i_t

using TODAY's market_value (current position sizing) applied to each
historical day's return -- the standard "historical simulation" convention:
what would the portfolio have made/lost on a given historical day, holding
today's book. This one P&L series is reused by all five VaR/CVaR methods in
var.py for a consistent, apples-to-apples comparison.

Equities trade on fewer calendar days than crypto (no weekends/holidays);
rows where any held equity/underlying has no price are dropped rather than
forward-filled, so no method is fed a fabricated "no-move" day -- a
disclosed reduction in sample size, not a gap-filling assumption.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from aggregation.pricing import resolve_symbol
from connectors.schema import AssetClass, Position

DEFAULT_LOOKBACK_PERIOD = "2y"


def position_risk_factor(p: Position) -> str | None:
    """The ticker whose historical returns proxy this position's daily P&L."""
    if p.asset_class == AssetClass.SYNTHETIC:
        return None
    if p.asset_class == AssetClass.OPTION:
        return p.extra.get("underlying")
    return p.asset


def fetch_return_history(
    tickers: list[str], period: str = DEFAULT_LOOKBACK_PERIOD
) -> tuple[pd.DataFrame, list[str]]:
    """Daily simple returns for `tickers` (deduplicated, symbol-resolved),
    aligned to the intersection of dates where every ticker has a price.

    Returns (returns_df, notes) -- notes discloses any ticker that came back
    empty rather than silently omitting it.
    """
    notes: list[str] = []
    resolved = sorted({resolve_symbol(t) for t in tickers})
    if not resolved:
        return pd.DataFrame(), ["No tickers requested."]

    raw = yf.download(resolved, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.Series):  # yfinance collapses to a Series for a single ticker
        raw = raw.to_frame(resolved[0])

    missing = [t for t in resolved if t not in raw.columns or raw[t].dropna().empty]
    for t in missing:
        notes.append(f"{t}: no price history returned; excluded from the risk factor set.")
    present = [t for t in resolved if t not in missing]

    prices = raw[present].dropna(how="any")
    returns = prices.pct_change().dropna(how="any")
    notes.append(
        f"Return history: {len(returns)} aligned trading days across {len(present)} risk factors "
        f"(period={period})."
    )
    return returns, notes


def build_portfolio_pnl_series(
    positions: list[Position], returns_df: pd.DataFrame
) -> tuple[pd.Series, dict[str, float], list[str]]:
    """Hypothetical daily portfolio $ P&L, plus the {ticker: dollar_weight}
    map actually used (needed by var.py's parametric/Monte Carlo methods),
    plus notes on any position excluded and why.
    """
    notes: list[str] = []
    weights: dict[str, float] = {}

    for p in positions:
        if p.market_value is None:
            notes.append(f"{p.strategy}/{p.asset}: unpriced -- excluded from VaR.")
            continue
        factor = position_risk_factor(p)
        if factor is None:
            notes.append(f"{p.strategy}/{p.asset}: synthetic, no risk factor -- excluded from VaR.")
            continue
        resolved = resolve_symbol(factor)
        if resolved not in returns_df.columns:
            notes.append(
                f"{p.strategy}/{p.asset}: no return history for {resolved} -- excluded from VaR."
            )
            continue
        weights[resolved] = weights.get(resolved, 0.0) + p.market_value

    if not weights:
        return pd.Series(dtype=float), weights, notes + ["No positions could be mapped to a risk factor."]

    used_returns = returns_df[list(weights.keys())]
    weight_vector = pd.Series(weights)
    pnl_series = used_returns.mul(weight_vector, axis=1).sum(axis=1)
    pnl_series.name = "portfolio_pnl"
    return pnl_series, weights, notes
