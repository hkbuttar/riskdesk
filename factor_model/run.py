"""Factor risk decomposition on the real aggregated book: named factors
(market, sector, crypto) via regression, PCA as a model-free complement,
and vega exposure reported separately.

    python -m factor_model.run
"""

from __future__ import annotations

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from factor_model.factors import SECTOR_ETF, fetch_factor_returns, sector_of
from factor_model.pca import fit_pca, top_loadings
from factor_model.regression import fit_factor_regression, significant_factors
from factor_model.vega import aggregate_vega
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, notes = fetch_return_history(tickers)
    for note in notes:
        print(f"  {note}")

    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    print(f"\n  Sectors held (via alpha-signal-lab's tagging): {sectors_held}")
    factor_returns, factor_notes = fetch_factor_returns(sectors_held)
    for note in factor_notes:
        print(f"  {note}")

    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)
    etf_to_sector = {etf: sector for sector, etf in SECTOR_ETF.items()}
    factor_returns = factor_returns.rename(
        columns={etf: f"{sector}_{etf}" for etf, sector in etf_to_sector.items() if etf in factor_returns.columns}
    )

    print("\n=== Portfolio-level factor decomposition ===")
    portfolio_result = fit_factor_regression("portfolio", pnl_series, factor_returns)
    _print_regression(portfolio_result)

    print("\n=== Per-strategy factor decomposition (does 'market-neutral' pairtrade-lab-1 carry hidden beta?) ===")
    strategies = sorted({p.strategy for p in valued.positions})
    for strategy in strategies:
        strat_positions = [p for p in valued.positions if p.strategy == strategy]
        strat_pnl, strat_weights, _ = build_portfolio_pnl_series(strat_positions, returns_df)
        if strat_pnl.empty or not strat_weights:
            print(f"  {strategy}: no priced, mappable positions -- skipped.")
            continue
        result = fit_factor_regression(strategy, strat_pnl, factor_returns)
        _print_regression(result)

    print("\n=== PCA (model-free statistical factors) ===")
    pca_result = fit_pca(returns_df)
    print(f"  components needed for 90% of variance: {pca_result.n_components_for_90pct} of {len(returns_df.columns)}")
    print(f"  explained variance ratio (first 5): {pca_result.explained_variance_ratio.head(5).round(3).to_dict()}")
    for pc in pca_result.explained_variance_ratio.index[:3]:
        print(f"  {pc} top loadings: {top_loadings(pca_result, pc).round(3).to_dict()}")

    print("\n=== Vega exposure (VolEdge's vega, reported separately -- not a return factor) ===")
    vega = aggregate_vega(valued.positions)
    print(f"  net portfolio vega: {vega.net_vega:,.2f} (across {vega.n_option_positions} option positions)")


def _print_regression(result) -> None:
    sig = significant_factors(result)
    print(f"  -- {result.label} (n={result.n_obs}, R²={result.r_squared:.3f}) --")
    for factor, loading in result.loadings.items():
        marker = "*" if factor in sig else " "
        print(f"    {factor:12s} loading={loading:>10.2f}  t={result.t_stats[factor]:>6.2f}  "
              f"p={result.p_values[factor]:.3f} {marker}")
    print(f"    alpha (unexplained)={result.alpha:>10.2f}  p={result.alpha_p_value:.3f}")


if __name__ == "__main__":
    main()
