"""Live risk monitoring, regime-aware: pulls live positions, recomputes
VaR using whichever regime-conditional model currently applies (falling
back to the pooled model honestly when the current regime doesn't have
enough data for its own, exactly as regime/conditional.py's own design
anticipates), checks market-risk and credit-risk limits independently, and
feeds both into a persistent, manual-reset-only kill-switch.

    python -m monitor.live            # run one live check
    python -m monitor.live --reset    # manually reset the kill-switch after review
"""

from __future__ import annotations

import sys

from aggregation.rollup import portfolio_total
from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from connectors.alpaca_market_data import fetch_history
from monitor.kill_switch import load_state, save_state
from monitor.limits import check_credit_concentration_limit, check_var_limit
from regime.conditional import compare_pooled_vs_conditional
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

REFERENCE_TICKER = "SPY"
CONFIDENCE = 0.95


def run_live_check() -> bool:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    gross_exposure = portfolio_total(valued.positions).gross_market_value
    print(f"Live positions pulled: {len(valued.positions)} ({valued.n_priced} priced). "
          f"Gross exposure: ${gross_exposure:,.2f}")

    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    spy_close = fetch_history([REFERENCE_TICKER], period="2y", field="close")[REFERENCE_TICKER]
    regime_labels = classify_regimes(rolling_realized_vol(spy_close)).labels
    current_regime = regime_labels.dropna().iloc[-1]
    print(f"Current regime ({REFERENCE_TICKER}, most recent trading day): {current_regime}")

    conditional = compare_pooled_vs_conditional(pnl_series, returns_df, regime_labels, CONFIDENCE)
    if current_regime in conditional.conditional_var:
        active_var = conditional.conditional_var[current_regime]["historical_simulation"].var_dollar
        model_label = f"{current_regime}-conditional"
    else:
        active_var = conditional.pooled_var["historical_simulation"].var_dollar
        model_label = "pooled (current regime has too few historical days for its own model)"
    print(f"Active VaR model: {model_label} -> ${active_var:,.2f}")

    var_check = check_var_limit(active_var, gross_exposure)
    credit_check = check_credit_concentration_limit(valued.positions)
    print(f"  {var_check.detail}")
    print(f"  {credit_check.detail}")

    kill_switch = load_state()
    was_already_triggered = kill_switch.triggered
    kill_switch.check({var_check.name: var_check.detail if var_check.breached else False,
                        credit_check.name: credit_check.detail if credit_check.breached else False})
    save_state(kill_switch)

    if kill_switch.triggered:
        status = "ALREADY TRIGGERED (unchanged)" if was_already_triggered else "TRIGGERED NOW"
        print(f"\n*** KILL-SWITCH: {status} ***")
        for reason in kill_switch.trigger_reasons:
            print(f"  - {reason}")
        print("  Manual reset required: python -m monitor.live --reset")
    else:
        print("\nKill-switch: not triggered.")

    return kill_switch.triggered


def main() -> None:
    if "--reset" in sys.argv:
        kill_switch = load_state()
        kill_switch.reset()
        save_state(kill_switch)
        print("Kill-switch manually reset.")
        return
    run_live_check()


if __name__ == "__main__":
    main()
