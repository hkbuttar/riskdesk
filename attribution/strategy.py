"""P&L attribution by strategy: the same hypothetical-historical-P&L
methodology used throughout this project (risk_measures/returns.py), split
per strategy over a recent window rather than the full 2-year lookback --
"which strategy actually drove this window's P&L," reusing
build_portfolio_pnl_series per-strategy exactly as stress/historical.py and
factor_model/run.py already do inline, centralized here as the canonical
attribution entry point.
"""

from __future__ import annotations

import pandas as pd

from connectors.schema import Position
from risk_measures.returns import build_portfolio_pnl_series


def attribute_by_strategy(positions: list[Position], returns_df: pd.DataFrame) -> dict[str, pd.Series]:
    strategies = sorted({p.strategy for p in positions})
    result: dict[str, pd.Series] = {}
    for strategy in strategies:
        strat_positions = [p for p in positions if p.strategy == strategy]
        pnl_series, weights, _ = build_portfolio_pnl_series(strat_positions, returns_df)
        if weights:
            result[strategy] = pnl_series
    return result
