"""Regime-conditional VaR and correlation vs. the pooled (regime-agnostic)
model, on the real aggregated book.

    python -m regime.run_conditional
"""

from __future__ import annotations

import numpy as np
from connectors.alpaca_market_data import fetch_history

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from regime.conditional import compare_pooled_vs_conditional
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

REFERENCE_TICKER = "SPY"
CONFIDENCE = 0.95


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, notes = fetch_return_history(tickers)
    for note in notes:
        print(f"  {note}")

    pnl_series, _weights, pnl_notes = build_portfolio_pnl_series(valued.positions, returns_df)
    for note in pnl_notes:
        print(f"  {note}")

    spy_close = fetch_history([REFERENCE_TICKER], period="2y", field="close")[REFERENCE_TICKER]
    regime_labels = classify_regimes(rolling_realized_vol(spy_close)).labels
    print(f"\nRegime reference: {REFERENCE_TICKER} volatility terciles.\n")

    result = compare_pooled_vs_conditional(pnl_series, returns_df, regime_labels, CONFIDENCE)
    for note in result.notes:
        print(f"  {note}")

    print(f"\n=== Pooled (regime-agnostic) {CONFIDENCE:.0%} VaR ===")
    for name, r in result.pooled_var.items():
        print(f"  {name:22s} VaR=${r.var_dollar:>12,.2f}  CVaR=${r.cvar_dollar:>12,.2f}")

    print(f"\n=== Regime-conditional {CONFIDENCE:.0%} VaR (n days per regime: {result.regime_day_counts}) ===")
    for regime, methods in result.conditional_var.items():
        print(f"  -- {regime} --")
        for name, r in methods.items():
            print(f"    {name:22s} VaR=${r.var_dollar:>12,.2f}  CVaR=${r.cvar_dollar:>12,.2f}")

    print("\n=== Does calm-regime VaR understate what the volatile regime actually produces? ===")
    if "calm" in result.conditional_var and "volatile" in result.conditional_var:
        for name in CONDITIONAL_METHODS_NAMES:
            calm_var = result.conditional_var["calm"][name].var_dollar
            volatile_var = result.conditional_var["volatile"][name].var_dollar
            pooled_var = result.pooled_var[name].var_dollar
            ratio = volatile_var / calm_var if calm_var else float("nan")
            print(
                f"  {name:22s} calm=${calm_var:>10,.2f}  volatile=${volatile_var:>10,.2f}  "
                f"pooled=${pooled_var:>10,.2f}  volatile/calm={ratio:.1f}x"
            )
    else:
        print("  Skipped -- one of calm/volatile had too few days for a conditional estimate (see notes above).")

    print("\n=== Does correlation structure genuinely shift by regime? ===")
    for regime, corr in result.conditional_correlation.items():
        diff = (corr - result.pooled_correlation).abs()
        diff_values = diff.to_numpy().copy()
        np.fill_diagonal(diff_values, 0.0)
        print(f"  {regime:10s} mean |conditional - pooled| correlation: {diff_values.mean():.3f}, "
              f"max: {diff_values.max():.3f}")

    print(f"\n=== Live monitoring readiness: which model is 'active' right now? ===")
    current_regime = regime_labels.dropna().iloc[-1]
    print(f"  Current classified regime ({REFERENCE_TICKER}, most recent trading day): {current_regime}")
    if current_regime in result.conditional_var:
        active_var = result.conditional_var[current_regime]["historical_simulation"].var_dollar
        print(f"  -> active model: {current_regime}-conditional historical-simulation VaR = ${active_var:,.2f} "
              f"(vs. pooled ${result.pooled_var['historical_simulation'].var_dollar:,.2f})")
    else:
        print(f"  -> {current_regime} regime has too few historical days for its own conditional model; "
              "falling back to the pooled model is the honest choice here, not silently reusing another regime's estimate.")


CONDITIONAL_METHODS_NAMES = ("historical_simulation", "parametric_normal", "cornish_fisher")


if __name__ == "__main__":
    main()
