"""Historical scenario replay: apply real historical return windows to
TODAY's actual position sizes, same "hypothetical historical P&L"
methodology risk_measures/returns.py already uses for VaR, just narrowed to
three specific, disclosed crisis windows instead of a rolling 2-year
lookback.

Window choice, disclosed judgment calls, not universally agreed exact
dates:
  - COVID crash: 2020-02-19 (S&P 500 pre-crash closing high) to 2020-03-23
    (the closing low) -- the most commonly cited peak-to-trough dates for
    this crash.
  - 2022 rate-hike bear market: 2022-01-03 (S&P 500's 2022 opening high) to
    2022-10-12 (the 2022 closing low) -- a slower, longer drawdown than
    COVID's, deliberately included as a contrasting shape of stress.
  - FTX collapse: 2022-11-06 (the CoinDesk report that triggered the run)
    to 2022-11-14 (a week later, past the immediate acute phase) -- a
    crypto-specific, not broad-equity-market, crisis.

Each window has a `pre_start` date: the replay's regime-conditional
validation (validate_regime_conditional_var below) fits its models ONLY on
data strictly before the window starts, to avoid lookahead bias -- "would
this model, estimated from what was known before the crisis, have bounded
the crisis's actual losses" is the honest question, not "does a model fit
on the crisis itself predict the crisis."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from aggregation.pricing import resolve_symbol
from connectors.alpaca_market_data import fetch_history
from connectors.schema import Position
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from risk_measures.returns import position_risk_factor
from risk_measures.var import historical_simulation

HISTORICAL_WINDOWS: dict[str, dict[str, str]] = {
    "covid_crash": {
        "start": "2020-02-19", "end": "2020-03-23", "pre_start": "2018-06-01",
        "description": "COVID crash: S&P 500 pre-crash high to closing low.",
    },
    "2022_rate_hike": {
        "start": "2022-01-03", "end": "2022-10-12", "pre_start": "2019-06-01",
        "description": "2022 rate-hike bear market: S&P 500 2022 opening high to closing low.",
    },
    "ftx_collapse": {
        "start": "2022-11-06", "end": "2022-11-14", "pre_start": "2019-06-01",
        "description": "FTX collapse: CoinDesk report to one week later.",
    },
}


def fetch_price_history(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = fetch_history(sorted(set(tickers)), start=start, end=end, field="close")
    # Preserve partial histories. Alpaca's crypto archive does not extend to
    # every equity stress window (notably COVID), so forcing a complete-case
    # intersection here would erase otherwise valid SPY/equity observations.
    return raw.dropna(how="all")


@dataclass
class ReplayResult:
    window_name: str
    total_pnl: float
    max_daily_loss: float
    daily_pnl: pd.Series
    portfolio_return_pct: float  # total_pnl / starting portfolio market value


def replay_window(positions: list[Position], price_history: pd.DataFrame) -> ReplayResult | None:
    """Applies each position's own risk-factor return, day by day within
    the window, to TODAY's market_value -- unpriced/unmappable positions
    are excluded (consistent with risk_measures/returns.py).
    """
    available_prices = price_history.dropna(axis=1, how="all")
    returns = available_prices.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        return None

    weights: dict[str, float] = {}
    for p in positions:
        if p.market_value is None:
            continue
        factor = position_risk_factor(p)
        if factor is None:
            continue
        resolved = resolve_symbol(factor)
        if resolved not in returns.columns:
            continue
        weights[resolved] = weights.get(resolved, 0.0) + p.market_value

    if not weights:
        return None

    weight_vector = pd.Series(weights)
    daily_pnl = returns[list(weights.keys())].mul(weight_vector, axis=1).sum(axis=1)
    starting_value = sum(abs(w) for w in weights.values())

    return ReplayResult(
        window_name="",
        total_pnl=float(daily_pnl.sum()),
        max_daily_loss=float(daily_pnl.min()),
        daily_pnl=daily_pnl,
        portfolio_return_pct=float(daily_pnl.sum() / starting_value) if starting_value else float("nan"),
    )


@dataclass
class DiversificationCheck:
    realized_portfolio_vol: float
    uncorrelated_estimate_vol: float
    diversification_erosion_ratio: float  # realized / uncorrelated_estimate -- >1 means correlation hurt


def check_diversification_erosion(strategy_daily_pnl: dict[str, pd.Series]) -> DiversificationCheck | None:
    """Compares the AGGREGATE portfolio's realized daily P&L volatility
    during the window against what volatility would be if each strategy's
    own P&L moved independently (uncorrelated) -- sqrt(sum of each
    strategy's own variance). The ratio is the concrete answer to "does
    cross-strategy correlation amplify risk beyond what isolated
    strategy-level views would suggest": ratio > 1 means the real,
    correlated portfolio was riskier than a naive independence assumption
    would predict; ratio < 1 means diversification actually helped.
    """
    series_list = [s for s in strategy_daily_pnl.values() if len(s) > 1]
    if len(series_list) < 2:
        return None

    combined = pd.concat(series_list, axis=1).dropna(how="any")
    if combined.empty or len(combined) < 2:
        return None

    realized_vol = float(combined.sum(axis=1).std())
    uncorrelated_vol = float(np.sqrt((combined.std() ** 2).sum()))

    return DiversificationCheck(
        realized_portfolio_vol=realized_vol,
        uncorrelated_estimate_vol=uncorrelated_vol,
        diversification_erosion_ratio=realized_vol / uncorrelated_vol if uncorrelated_vol else float("nan"),
    )


@dataclass
class VaRBreachComparison:
    n_window_days: int
    pooled_var: float
    pooled_breaches: int
    conditional_var: float | None
    conditional_breaches: int | None
    conditional_label: str


def validate_regime_conditional_var(
    pre_window_pnl: pd.Series,
    pre_window_spy_close: pd.Series,
    window_daily_pnl: pd.Series,
    confidence: float = 0.95,
) -> VaRBreachComparison:
    """Fits a pooled VaR and a volatile-regime-conditional VaR using ONLY
    data strictly before the stress window (see module docstring on
    lookahead), then counts how many of the window's actual daily losses
    breach each bound. If regime-conditioning genuinely helps, the
    volatile-conditional VaR should be breached less often (or not more
    often) than the pooled VaR, since it was fit on the kind of turbulent
    days most analogous to what's coming.
    """
    pooled = historical_simulation(pre_window_pnl, confidence)
    window_losses = -window_daily_pnl
    pooled_breaches = int((window_losses > pooled.var_dollar).sum())

    vol = rolling_realized_vol(pre_window_spy_close)
    tercile = classify_regimes(vol)
    aligned_labels = tercile.labels.reindex(pre_window_pnl.index)
    volatile_pnl = pre_window_pnl[aligned_labels == "volatile"]

    if len(volatile_pnl) < 30:
        return VaRBreachComparison(
            n_window_days=len(window_daily_pnl), pooled_var=pooled.var_dollar,
            pooled_breaches=pooled_breaches, conditional_var=None, conditional_breaches=None,
            conditional_label=f"volatile regime had only {len(volatile_pnl)} pre-window days -- too few to fit",
        )

    conditional = historical_simulation(volatile_pnl, confidence)
    conditional_breaches = int((window_losses > conditional.var_dollar).sum())

    return VaRBreachComparison(
        n_window_days=len(window_daily_pnl), pooled_var=pooled.var_dollar,
        pooled_breaches=pooled_breaches, conditional_var=conditional.var_dollar,
        conditional_breaches=conditional_breaches,
        conditional_label=f"volatile-regime-conditional (fit on {len(volatile_pnl)} pre-window volatile days)",
    )
