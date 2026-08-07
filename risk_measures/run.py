"""VaR/CVaR: five methods, compared on the same real aggregated portfolio.

    python -m risk_measures.run
"""

from __future__ import annotations

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor
from risk_measures.var import (
    cornish_fisher,
    evt_pot,
    historical_simulation,
    monte_carlo,
    parametric_variance_covariance,
)

CONFIDENCE_LEVELS = (0.95, 0.99)


def main() -> None:
    raw_positions, _ = fetch_all()
    valuation = value_positions(raw_positions)
    print(f"{valuation.n_priced} priced positions ({valuation.n_unpriced} unpriced, excluded).\n")

    tickers = sorted({position_risk_factor(p) for p in valuation.positions if position_risk_factor(p)})
    returns_df, notes = fetch_return_history(tickers)
    for note in notes:
        print(f"  {note}")
    print()

    pnl_series, weights, pnl_notes = build_portfolio_pnl_series(valuation.positions, returns_df)
    for note in pnl_notes:
        print(f"  {note}")
    print(f"\nPortfolio hypothetical daily P&L series: n={len(pnl_series)}, "
          f"mean=${pnl_series.mean():,.2f}, std=${pnl_series.std():,.2f}\n")

    for confidence in CONFIDENCE_LEVELS:
        print(f"=== {confidence:.0%} 1-day VaR/CVaR ===")
        results = [
            historical_simulation(pnl_series, confidence),
            parametric_variance_covariance(pnl_series, confidence),
            monte_carlo(returns_df, weights, confidence),
            cornish_fisher(pnl_series, confidence),
            evt_pot(pnl_series, confidence),
        ]
        for r in results:
            var_str = f"${r.var_dollar:,.2f}" if r.var_dollar == r.var_dollar else "n/a"
            cvar_str = f"${r.cvar_dollar:,.2f}" if r.cvar_dollar == r.cvar_dollar else "n/a"
            print(f"  {r.method:22s} VaR={var_str:>14s}  CVaR={cvar_str:>14s}  {r.notes}")

        priced_vars = [r.var_dollar for r in results if r.var_dollar == r.var_dollar]
        if priced_vars:
            spread = max(priced_vars) - min(priced_vars)
            print(f"  -> spread across methods: ${spread:,.2f} "
                  f"({spread / min(priced_vars):.0%} of the lowest estimate)")
        print()


if __name__ == "__main__":
    main()
