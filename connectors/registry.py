"""Runs every connector and returns the combined, normalized position set."""

from __future__ import annotations

from connectors import alpha_signal_lab, bookmaker, execedge, pairtrade_lab, streamalpha, voledge
from connectors.schema import Position, SourceMeta

CONNECTORS = {
    "alpha-signal-lab": alpha_signal_lab,
    "streamalpha": streamalpha,
    "bookmaker": bookmaker,
    "execedge": execedge,
    "pairtrade-lab-1": pairtrade_lab,
    "voledge": voledge,
}


def fetch_all() -> tuple[list[Position], list[SourceMeta]]:
    all_positions: list[Position] = []
    all_meta: list[SourceMeta] = []
    for module in CONNECTORS.values():
        positions, meta = module.fetch_positions()
        all_positions.extend(positions)
        all_meta.append(meta)
    return all_positions, all_meta


if __name__ == "__main__":
    positions, meta = fetch_all()
    for m in meta:
        print(f"[{m.repo}] n_positions={m.n_positions} read_from={m.read_from!r}")
        print(f"    {m.notes}")
    print(f"\nTotal positions aggregated: {len(positions)}")
    for p in positions:
        print(f"  {p.strategy:20s} {p.asset:20s} qty={p.quantity:>12.4f} "
              f"class={p.asset_class.value:10s} counterparty={p.counterparty.value}")
