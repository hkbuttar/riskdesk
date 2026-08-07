"""Portfolio-level Greek aggregation and options-book convexity, on the
real book.

    python -m aggregation.run_greeks
"""

from __future__ import annotations

from aggregation.greeks import aggregate_greeks, gamma_convexity_table
from aggregation.valuation import value_positions
from connectors.registry import fetch_all


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    greeks = aggregate_greeks(valued.positions)
    print(f"=== Portfolio Greeks ({greeks.n_option_positions} option positions) ===")
    print(f"  net delta: {greeks.net_delta_shares:>10.2f} shares-equivalent  "
          f"(${greeks.net_delta_dollars:>12,.2f})")
    print(f"  net gamma: {greeks.net_gamma_shares:>10.4f} shares-equivalent per $1 underlying move")
    print(f"  net vega:  ${greeks.net_vega:>12,.2f}  (P&L per 1.00/100-vol-point IV move; "
          f"${greeks.net_vega / 100:,.2f} per single vol point)")
    print(f"  net theta: ${greeks.net_theta:>12,.2f}  (P&L per day, time decay)")
    print(f"  net rho:   ${greeks.net_rho:>12,.2f}")

    print("\n=== Convexity: delta-only (linear) vs. delta+gamma (quadratic) P&L, by hypothetical SPY move ===")
    print("  (delta-only is the approximation risk_measures/returns.py and factor_model both use)")
    for row in gamma_convexity_table(valued.positions):
        pct_str = f"{row.pct_understatement:+.1%}" if row.pct_understatement == row.pct_understatement else "n/a"
        print(
            f"  move={row.move_pct:+.0%}  linear=${row.linear_pnl:>10,.2f}  "
            f"quadratic=${row.quadratic_pnl:>10,.2f}  gamma_correction=${row.gamma_correction:>10,.2f}  "
            f"({pct_str} of linear)"
        )

    print(
        "\n  -> the gamma correction grows with the square of the move size (a hallmark of convexity): "
        "small at +-2%, materially larger at +-10%. Every VaR/factor-regression number reported elsewhere "
        "in this project for the options book uses the delta-only (linear) approximation -- this table "
        "quantifies exactly how much that approximation understates or overstates P&L once the underlying "
        "moves far enough, rather than leaving it as an unquantified caveat."
    )


if __name__ == "__main__":
    main()
