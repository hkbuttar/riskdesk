"""Position/exposure aggregation: fetches every connector's positions,
attaches point-in-time prices (delta-equivalent for options), and rolls up
to portfolio, strategy, asset-class, and counterparty level.

    python -m aggregation.run
"""

from __future__ import annotations

from aggregation.rollup import RollupBucket, by_asset_class, by_counterparty, by_strategy, portfolio_total
from aggregation.valuation import value_positions
from connectors.registry import fetch_all


def _print_bucket(bucket: RollupBucket) -> None:
    print(
        f"  {bucket.key:20s} net={bucket.net_market_value:>14,.2f}  "
        f"gross={bucket.gross_market_value:>14,.2f}  "
        f"priced={bucket.n_priced}/{bucket.n_positions}"
    )
    for asset in bucket.unpriced_assets:
        print(f"      (unpriced: {asset})")


def main() -> None:
    raw_positions, source_meta = fetch_all()
    print(f"Fetched {len(raw_positions)} raw positions from {len(source_meta)} connectors.\n")

    result = value_positions(raw_positions)
    print(f"Valuation: {result.n_priced} priced, {result.n_unpriced} unpriced.")
    for note in result.notes:
        print(f"  - {note}")
    print()

    print("=== Portfolio total ===")
    _print_bucket(portfolio_total(result.positions))

    print("\n=== By strategy ===")
    for bucket in by_strategy(result.positions).values():
        _print_bucket(bucket)

    print("\n=== By asset class ===")
    for bucket in by_asset_class(result.positions).values():
        _print_bucket(bucket)

    print("\n=== By counterparty ===")
    for bucket in by_counterparty(result.positions).values():
        _print_bucket(bucket)


if __name__ == "__main__":
    main()
