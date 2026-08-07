"""Counterparty & credit risk on the real book: exposure by venue, a
simplified CVA, and concentration risk. This is explicitly a second,
genuinely distinct risk category from everything else in this project --
every other module answers "what if the market moves against me"; this
one answers "what if the entity holding my assets disappears."

    python -m credit.run
"""

from __future__ import annotations

from aggregation.rollup import by_counterparty
from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from connectors.schema import Counterparty
from credit.concentration import check_concentration
from credit.counterparty import COUNTERPARTY_PD
from credit.cva import compute_cva


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    print("=== Exposure by counterparty ===")
    for key, bucket in by_counterparty(valued.positions).items():
        tier = COUNTERPARTY_PD[Counterparty(key)]
        print(f"  {key:12s} net=${bucket.net_market_value:>12,.2f}  gross=${bucket.gross_market_value:>12,.2f}  "
              f"(PD tier: {tier.tier}, annualized PD={tier.annualized_pd:.2%})")

    print("\n=== Simplified CVA (custodial/settlement counterparty risk, see module docstring) ===")
    cva = compute_cva(valued.positions)
    for key, detail in cva.by_counterparty.items():
        print(f"  {key:12s} |net exposure|=${abs(detail['net_exposure']):>12,.2f}  "
              f"PD={detail['annualized_pd']:.2%}  CVA=${detail['cva']:,.2f}")
    print(f"  TOTAL CVA (LGD={cva.lgd_used:.0%}): ${cva.total_cva:,.2f}")

    print("\n=== Concentration risk (real venues only; see FTX 2022 -- already replayed in stress/historical.py) ===")
    concentration = check_concentration(valued.positions)
    if concentration.total_real_venue_exposure == 0:
        print("  No exposure at any real venue currently -- nothing to concentrate.")
    else:
        for key, share in concentration.shares.items():
            marker = " *** FLAGGED ***" if key in concentration.flagged else ""
            print(f"  {key:12s} ${concentration.exposure_by_counterparty[key]:>12,.2f}  "
                  f"({share:.1%} of real-venue exposure){marker}")
        print(f"  Herfindahl index: {concentration.herfindahl_index:.3f} "
              f"(1.0 = single venue, 1/n = evenly split across n venues)")
        if concentration.flagged:
            from credit.concentration import DEFAULT_CONCENTRATION_THRESHOLD
            print(f"  -> {concentration.flagged} exceed the "
                  f"{DEFAULT_CONCENTRATION_THRESHOLD:.0%} concentration threshold.")
        else:
            print("  No counterparty currently exceeds the concentration threshold.")


if __name__ == "__main__":
    main()
