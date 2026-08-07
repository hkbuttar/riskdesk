"""VaR backtesting on the real book: Kupiec POF + Christoffersen
independence tests, pooled vs. regime-conditional.

    python -m risk_measures.run_backtest
"""

from __future__ import annotations

from connectors.alpaca_market_data import fetch_history

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from risk_measures.backtesting import backtest_breach_series, pooled_breach_series, regime_conditional_breach_series
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

CONFIDENCE = 0.95


def _print_result(result) -> None:
    print(f"  -- {result.model_label} --")
    print(f"    n={result.n_obs}, breaches={result.n_breaches} "
          f"(observed {result.observed_breach_rate:.2%} vs. expected {result.expected_breach_rate:.2%})")
    print(f"    Kupiec POF:        LR={result.kupiec_lr:.2f}  p={result.kupiec_p_value:.3f}")
    print(f"    Christoffersen:    LR={result.christoffersen_lr:.2f}  p={result.christoffersen_p_value:.3f}")
    print(f"    Combined (df=2):   LR={result.combined_lr:.2f}  p={result.combined_p_value:.3f}  "
          f"-> {'well-calibrated' if result.well_calibrated else 'NOT well-calibrated'} "
          f"(fails to reject H0 at 5%: {result.well_calibrated})")


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    spy_close = fetch_history(["SPY"], period="2y", field="close")["SPY"]
    regime_labels = classify_regimes(rolling_realized_vol(spy_close)).labels

    print(f"=== VaR backtest: {CONFIDENCE:.0%} confidence, {len(pnl_series)} days ===\n")

    pooled_breaches = pooled_breach_series(pnl_series, CONFIDENCE)
    pooled_result = backtest_breach_series(pooled_breaches, CONFIDENCE, "pooled")
    _print_result(pooled_result)

    conditional_breaches, notes = regime_conditional_breach_series(pnl_series, regime_labels, CONFIDENCE)
    for note in notes:
        print(f"  {note}")
    conditional_result = backtest_breach_series(conditional_breaches, CONFIDENCE, "regime-conditional")
    print()
    _print_result(conditional_result)

    print("\n=== Does regime-conditioning actually improve calibration? ===")
    print(f"  pooled combined p-value:              {pooled_result.combined_p_value:.4f}")
    print(f"  regime-conditional combined p-value:  {conditional_result.combined_p_value:.4f}")
    if conditional_result.combined_p_value > pooled_result.combined_p_value:
        print("  -> regime-conditioning improved calibration (higher p-value = less evidence of miscalibration).")
    else:
        print("  -> regime-conditioning did NOT improve calibration by this test.")


if __name__ == "__main__":
    main()
