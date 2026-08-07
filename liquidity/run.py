"""Liquidity-adjusted VaR and concentration limits (name/strategy/sector)
on the real book.

    python -m liquidity.run
"""

from __future__ import annotations

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from liquidity.concentration import check_by_name, check_by_sector, check_by_strategy
from liquidity.impact import fetch_avg_daily_dollar_volume, liquidity_adjusted_var
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor
from risk_measures.var import historical_simulation


def _print_check(check) -> None:
    print(f"  -- by {check.dimension} (threshold={check.threshold:.0%}) --")
    if check.total_exposure == 0:
        print("     no exposure -- nothing to check.")
        return
    for key, share in sorted(check.shares.items(), key=lambda kv: -kv[1]):
        marker = " *** FLAGGED ***" if key in check.flagged else ""
        print(f"     {key:20s} ${check.exposures[key]:>12,.2f}  ({share:.1%}){marker}")
    print(f"     Herfindahl index: {check.herfindahl_index:.3f}")


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, notes = fetch_return_history(tickers)
    for note in notes:
        print(f"  {note}")
    daily_vol_by_asset = returns_df.std().to_dict()

    print("\nFetching average daily dollar volume (3mo)...")
    dollar_volume_by_asset = fetch_avg_daily_dollar_volume(tickers)

    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)
    base_result = historical_simulation(pnl_series)
    print(f"\nBase (mark-to-market) 95% 1-day VaR: ${base_result.var_dollar:,.2f}")

    adjusted_var, costs, cost_notes = liquidity_adjusted_var(
        base_result.var_dollar, valued.positions, daily_vol_by_asset, dollar_volume_by_asset
    )
    for note in cost_notes:
        print(f"  {note}")
    print(f"Liquidity-adjusted VaR (base + estimated unwind cost): ${adjusted_var:,.2f}")
    print(f"  -> estimated unwind cost adds ${adjusted_var - base_result.var_dollar:,.2f} "
          f"({(adjusted_var / base_result.var_dollar - 1):+.1%} of base VaR)")

    print("\n  Largest estimated liquidation costs by position:")
    for key, cost in sorted(costs.items(), key=lambda kv: -kv[1].dollar_cost)[:5]:
        print(f"    {key:30s} participation_rate={cost.participation_rate:.4%}  "
              f"cost_fraction={cost.cost_fraction:.4%}  ${cost.dollar_cost:,.2f}")

    print("\n=== Concentration limits ===")
    _print_check(check_by_name(valued.positions))
    _print_check(check_by_strategy(valued.positions))
    _print_check(check_by_sector(valued.positions))


if __name__ == "__main__":
    main()
