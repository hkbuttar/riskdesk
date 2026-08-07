"""Reverse stress testing on the real book: for a target loss, solve the
minimum-Mahalanobis-distance combination of named-factor moves that
produces it, then assess how plausible that scenario actually is.

    python -m reverse_stress.run
"""

from __future__ import annotations

from aggregation.rollup import portfolio_total
from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from factor_model.factors import fetch_factor_returns, sector_of
from factor_model.regression import fit_factor_regression
from reverse_stress.optimization import solve_reverse_stress
from reverse_stress.plausibility import compare_pooled_vs_volatile_regime, compare_to_historical_windows
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

TARGET_LOSS_FRACTIONS = (0.10, 0.25, 0.50)


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    gross_exposure = portfolio_total(valued.positions).gross_market_value
    print(f"Portfolio gross exposure: ${gross_exposure:,.2f}\n")

    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, notes = fetch_factor_returns(sectors_held)
    for note in notes:
        print(f"  {note}")

    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)
    print(f"\nFactor model: R²={fit.r_squared:.3f}, loadings={ {k: round(v, 0) for k, v in fit.loadings.items()} }\n")

    for fraction in TARGET_LOSS_FRACTIONS:
        target_loss = gross_exposure * fraction
        print(f"=== Target loss: ${target_loss:,.2f} ({fraction:.0%} of gross exposure) ===")

        result = solve_reverse_stress(fit.loadings, fit.alpha, factor_returns, target_loss)
        print(f"  solved factor shocks (Ledoit-Wolf shrinkage={result.shrinkage_used:.3f}):")
        for factor, shock in result.factor_shocks.items():
            print(f"    {factor:16s} {shock:+.1%}")
        print(f"  implied P&L: ${result.implied_pnl:,.2f} (should match -target)")

        print(f"  pooled Mahalanobis distance: {result.mahalanobis_distance:.2f} SD "
              f"(implied probability of a move this extreme or worse: {result.implied_annual_probability:.2%})")

        regime_comparison = compare_pooled_vs_volatile_regime(
            result.factor_shocks, factor_returns, result.mahalanobis_distance,
            result.implied_annual_probability, horizon_days=result.horizon_days,
        )
        if regime_comparison.volatile_distance is not None:
            print(f"  volatile-regime-conditional distance: {regime_comparison.volatile_distance:.2f} SD "
                  f"(probability: {regime_comparison.volatile_probability:.2%}) -- {regime_comparison.notes}")
        else:
            print(f"  volatile-regime-conditional distance: skipped -- {regime_comparison.notes}")

        historical = compare_to_historical_windows(result.factor_shocks)
        print("  vs. real historical windows (realized total return per factor):")
        print(historical.round(3).to_string().replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    main()
