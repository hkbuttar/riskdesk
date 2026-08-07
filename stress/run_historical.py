"""Historical scenario replay on the real book: COVID crash, 2022 rate-hike
bear market, and the FTX collapse.

    python -m stress.run_historical
"""

from __future__ import annotations

from aggregation.pricing import resolve_symbol
from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from risk_measures.returns import position_risk_factor
from stress.historical import (
    HISTORICAL_WINDOWS,
    check_diversification_erosion,
    fetch_price_history,
    replay_window,
    validate_regime_conditional_var,
)


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    risk_factors = {position_risk_factor(p) for p in valued.positions if position_risk_factor(p)}
    tickers = sorted({resolve_symbol(f) for f in risk_factors} | {"SPY"})
    strategies = sorted({p.strategy for p in valued.positions})

    earliest_pre_start = min(w["pre_start"] for w in HISTORICAL_WINDOWS.values())
    latest_end = max(w["end"] for w in HISTORICAL_WINDOWS.values())
    print(f"Fetching {tickers} from {earliest_pre_start} to {latest_end}...\n")
    full_history = fetch_price_history(tickers, earliest_pre_start, latest_end)

    for name, window in HISTORICAL_WINDOWS.items():
        print(f"=== {name}: {window['description']} ===")
        print(f"    window: {window['start']} .. {window['end']}")

        window_prices = full_history.loc[window["start"] : window["end"]]
        result = replay_window(valued.positions, window_prices)
        if result is None:
            print("    No data / no mappable positions for this window -- skipped.\n")
            continue

        print(f"    total hypothetical P&L: ${result.total_pnl:,.2f} ({result.portfolio_return_pct:+.1%} "
              f"of gross exposure)")
        print(f"    worst single day: ${result.max_daily_loss:,.2f}")

        strategy_daily_pnl = {}
        for strategy in strategies:
            strat_positions = [p for p in valued.positions if p.strategy == strategy]
            strat_result = replay_window(strat_positions, window_prices)
            if strat_result is not None:
                strategy_daily_pnl[strategy] = strat_result.daily_pnl
                print(f"      {strategy:20s} P&L: ${strat_result.total_pnl:>12,.2f}")

        diversification = check_diversification_erosion(strategy_daily_pnl)
        if diversification:
            print(f"    diversification erosion ratio (realized vol / uncorrelated-estimate vol): "
                  f"{diversification.diversification_erosion_ratio:.2f}")
            if diversification.diversification_erosion_ratio > 1.05:
                print("      -> cross-strategy correlation made this window WORSE than an independence "
                      "assumption would have predicted.")
            elif diversification.diversification_erosion_ratio < 0.95:
                print("      -> strategies partly offset each other during this window (real diversification).")
            else:
                print("      -> roughly what independence would have predicted.")

        pre_window_prices = full_history.loc[window["pre_start"] : window["start"]]
        pre_window_result = replay_window(valued.positions, pre_window_prices)
        if pre_window_result is not None and "SPY" in pre_window_prices.columns:
            comparison = validate_regime_conditional_var(
                pre_window_result.daily_pnl, pre_window_prices["SPY"], result.daily_pnl
            )
            print(f"    VaR breach check ({comparison.n_window_days} window days, pre-window-fit models, "
                  "95% confidence):")
            expected_breaches = comparison.n_window_days * 0.05
            print(f"      pooled VaR=${comparison.pooled_var:,.2f}: {comparison.pooled_breaches} breaches "
                  f"(expected ~{expected_breaches:.1f} if well-calibrated)")
            if comparison.conditional_var is not None:
                print(f"      {comparison.conditional_label}: VaR=${comparison.conditional_var:,.2f}, "
                      f"{comparison.conditional_breaches} breaches")
            else:
                print(f"      conditional model skipped: {comparison.conditional_label}")
        print()


if __name__ == "__main__":
    main()
