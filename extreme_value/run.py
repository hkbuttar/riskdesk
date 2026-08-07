"""Regime-conditional tail risk on the real book: does the GPD tail SHAPE
(not just VaR level) differ by regime, and how much does the data-
sufficiency problem (small regime buckets -> few tail exceedances) actually
bite in practice.

    python -m extreme_value.run
"""

from __future__ import annotations

from connectors.alpaca_market_data import fetch_history

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from extreme_value.gpd import gpd_var_cvar
from extreme_value.tail_risk import compare_tail_shape, fit_gpd_by_regime
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

CONFIDENCE = 0.95


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    spy_close = fetch_history(["SPY"], period="2y", field="close")["SPY"]
    regime_labels = classify_regimes(rolling_realized_vol(spy_close)).labels

    for threshold in (0.90, 0.80):
        print(f"=== threshold_quantile={threshold:.0%} ===")
        comparison = fit_gpd_by_regime(pnl_series, regime_labels, threshold_quantile=threshold)
        for note in comparison.notes:
            print(f"  {note}")

        table = compare_tail_shape(comparison)
        print(table.round(3).to_string().replace("\n", "\n    "))

        print(f"\n  {CONFIDENCE:.0%} VaR/CVaR implied by each successfully-fit GPD:")
        if comparison.pooled_fit is not None:
            var, cvar = gpd_var_cvar(comparison.pooled_fit, CONFIDENCE)
            print(f"    pooled       VaR=${var:>10,.2f}  CVaR=${cvar:>10,.2f}")
        for regime, fit in comparison.regime_fits.items():
            if fit is not None:
                var, cvar = gpd_var_cvar(fit, CONFIDENCE)
                print(f"    {regime:12s} VaR=${var:>10,.2f}  CVaR=${cvar:>10,.2f}")
        print()


if __name__ == "__main__":
    main()
