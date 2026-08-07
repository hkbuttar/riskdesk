"""Liquidity-adjusted VaR: adds an estimated unwind (market-impact) cost to
mark-to-market VaR, using execedge's own square-root-law market-impact
model directly (`algos/impact_calibration.py` in that sibling repo), not a
reinvented formula.

execedge's own module docstring is explicit about a real, disclosed gap:
neither the Almgren & Chriss (2000) nor Almgren-Thum-Hauptmann-Li (2005)
papers' exact fitted linear-impact coefficients could be extracted in that
project's environment (no PDF-to-text tooling available, two attempts
failed) -- so execedge itself falls back to the **square-root-law
functional form**, `cost_fraction = Y * sigma * sqrt(participation_rate)`,
with `Y` (an order-1 constant, roughly 0.5-1.5 across independent studies
per that module's own citation) having "no default: pass whatever value
you can source, or 1.0 as the explicit textbook order-of-magnitude
convention." This module reuses that exact formula and that exact Y=1.0
disclosed convention -- the same honest gap, not silently resolved here
with a more confident-looking number.

Participation rate, here, is computed in DOLLAR terms uniformly across
asset classes (`|market_value| / average_daily_dollar_volume`), not
share/coin counts. Alpaca provides base-unit volume, so this module converts
it to approximate dollar volume as close times volume for both asset classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from aggregation.pricing import resolve_symbol
from connectors.alpaca_market_data import fetch_daily_bars
from connectors.schema import AssetClass, Position

SQRT_LAW_Y = 1.0  # execedge's own disclosed "textbook order-of-magnitude" convention


@dataclass
class LiquidationCost:
    cost_fraction: float  # Y * sigma * sqrt(participation_rate)
    dollar_cost: float
    participation_rate: float
    avg_daily_dollar_volume: float


def fetch_avg_daily_dollar_volume(tickers: list[str], period: str = "3mo") -> dict[str, float]:
    resolved = sorted({resolve_symbol(t) for t in tickers})
    bars = fetch_daily_bars(resolved, period=period)
    return {
        symbol: float((frame["close"] * frame["volume"]).mean())
        for symbol, frame in bars.items() if not frame.empty
    }


def estimate_liquidation_cost(
    market_value: float, daily_realized_vol: float, avg_daily_dollar_volume: float, y: float = SQRT_LAW_Y
) -> LiquidationCost | None:
    if avg_daily_dollar_volume <= 0 or daily_realized_vol != daily_realized_vol:  # NaN check
        return None

    participation_rate = abs(market_value) / avg_daily_dollar_volume
    cost_fraction = y * daily_realized_vol * np.sqrt(participation_rate)
    return LiquidationCost(
        cost_fraction=cost_fraction,
        dollar_cost=abs(market_value) * cost_fraction,
        participation_rate=participation_rate,
        avg_daily_dollar_volume=avg_daily_dollar_volume,
    )


def liquidity_adjusted_var(
    base_var: float, positions: list[Position], daily_vol_by_asset: dict[str, float],
    dollar_volume_by_asset: dict[str, float],
) -> tuple[float, dict[str, LiquidationCost], list[str]]:
    """`base_var` is a mark-to-market VaR (e.g. from risk_measures/var.py) --
    this adds an estimated cost of actually unwinding today's book on top of
    it, not a replacement for it. Positions with no volume/vol data (e.g.
    synthetic instruments, or a fetch failure) are excluded, disclosed via
    the returned notes, not silently zeroed.
    """
    notes: list[str] = []
    costs: dict[str, LiquidationCost] = {}
    total_cost = 0.0

    for p in positions:
        if p.market_value is None or p.asset_class == AssetClass.SYNTHETIC:
            continue
        factor = resolve_symbol(p.extra.get("underlying", p.asset)) if p.asset_class == AssetClass.OPTION else resolve_symbol(p.asset)
        vol = daily_vol_by_asset.get(factor)
        dollar_vol = dollar_volume_by_asset.get(factor)

        if vol is None or dollar_vol is None:
            notes.append(f"{p.strategy}/{p.asset}: no volume/volatility data for {factor} -- excluded.")
            continue

        cost = estimate_liquidation_cost(p.market_value, vol, dollar_vol)
        if cost is None:
            notes.append(f"{p.strategy}/{p.asset}: liquidation cost could not be estimated for {factor}.")
            continue

        costs[f"{p.strategy}/{p.asset}"] = cost
        total_cost += cost.dollar_cost

    return base_var + total_cost, costs, notes
