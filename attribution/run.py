"""P&L attribution by strategy and factor over a recent window, on the
real book.

    python -m attribution.run
"""

from __future__ import annotations

from aggregation.valuation import value_positions
from attribution.pnl import attribute_by_factor
from attribution.strategy import attribute_by_strategy
from connectors.registry import fetch_all
from factor_model.factors import fetch_factor_returns, sector_of
from factor_model.regression import fit_factor_regression
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

RECENT_WINDOW_DAYS = 63  # ~3 trading months


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)

    # Fit on the FULL 2-year window (stable loadings), attribute over just
    # the recent window -- a standard practice: a longer estimation period
    # for the factor model, applied to a shorter recent realized period.
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)
    print(f"Factor model (fit on full window): R²={fit.r_squared:.3f}\n")

    recent_pnl = pnl_series.iloc[-RECENT_WINDOW_DAYS:]
    print(f"=== Attribution window: last {len(recent_pnl)} trading days "
          f"({recent_pnl.index[0].date()} .. {recent_pnl.index[-1].date()}) ===")
    print(f"Total hypothetical P&L over window: ${recent_pnl.sum():,.2f}\n")

    print("--- By strategy ---")
    by_strategy = attribute_by_strategy(valued.positions, returns_df)
    for strategy, series in by_strategy.items():
        recent = series.reindex(recent_pnl.index).dropna()
        print(f"  {strategy:20s} ${recent.sum():>12,.2f}")

    print("\n--- By factor (loadings fit on full window, applied to recent window) ---")
    attribution = attribute_by_factor(recent_pnl, factor_returns, fit.loadings, fit.alpha)
    for factor, contribution in attribution.cumulative_by_factor.items():
        print(f"  {factor:16s} ${contribution:>12,.2f}")
    print(f"  {'alpha':16s} ${attribution.cumulative_alpha:>12,.2f}")
    print(f"  {'residual':16s} ${attribution.cumulative_residual:>12,.2f}  "
          f"(idiosyncratic -- not explained by any named factor)")
    print(f"  {'TOTAL':16s} ${attribution.cumulative_total:>12,.2f}")
    print(f"  reconciles exactly to total P&L: {attribution.reconciles()}")

    print("\n--- Strategy x factor (each strategy's own fitted loadings, applied to recent window) ---")
    for strategy in by_strategy:
        strat_positions = [p for p in valued.positions if p.strategy == strategy]
        strat_pnl, strat_weights, _ = build_portfolio_pnl_series(strat_positions, returns_df)
        if not strat_weights:
            continue
        strat_fit = fit_factor_regression(strategy, strat_pnl, factor_returns)
        strat_recent = strat_pnl.iloc[-RECENT_WINDOW_DAYS:]
        strat_attribution = attribute_by_factor(strat_recent, factor_returns, strat_fit.loadings, strat_fit.alpha)
        print(f"  {strategy} (R²={strat_fit.r_squared:.3f}):")
        for factor, contribution in strat_attribution.cumulative_by_factor.items():
            print(f"    {factor:16s} ${contribution:>10,.2f}")
        print(f"    {'alpha+residual':16s} ${strat_attribution.cumulative_alpha + strat_attribution.cumulative_residual:>10,.2f}")
        print(f"    {'total':16s} ${strat_attribution.cumulative_total:>10,.2f}")


if __name__ == "__main__":
    main()
