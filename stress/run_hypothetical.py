"""Hypothetical multi-factor stress scenarios on the real book, with full
options repricing (delta+gamma+vega) compared against the linear-only
proxy used everywhere else in this project.

    python -m stress.run_hypothetical
"""

from __future__ import annotations

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from stress.hypothetical import HYPOTHETICAL_SCENARIOS, run_scenario


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    for name, scenario in HYPOTHETICAL_SCENARIOS.items():
        print(f"=== {name} ===")
        print(f"    {scenario['description']}")
        print(f"    equity={scenario['equity_pct']:+.0%}  crypto={scenario['crypto_pct']:+.0%}  "
              f"vol_shock={scenario['vol_shock_pct']:+.0%} (relative)  "
              f"sector_overrides={scenario['sector_overrides']}")

        result = run_scenario(name, valued.positions)
        print(f"    total P&L (full repricing): ${result.total_pnl:,.2f}")
        print(f"    linear-only P&L (this project's usual proxy): ${result.linear_only_pnl:,.2f}")
        print(f"    convexity correction (gamma + vega, options only): ${result.convexity_correction:,.2f}")

        by_strategy: dict[str, float] = {}
        for key, pnl in result.by_position.items():
            strategy = key.split("/", 1)[0]
            by_strategy[strategy] = by_strategy.get(strategy, 0.0) + pnl
        for strategy, pnl in sorted(by_strategy.items()):
            print(f"      {strategy:20s} ${pnl:>12,.2f}")
        print()


if __name__ == "__main__":
    main()
