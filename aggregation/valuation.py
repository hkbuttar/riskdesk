"""Attaches point-in-time price/market_value to Position objects.

Three valuation paths, one per asset class actually seen from the
connectors (see connectors/schema.py::AssetClass):

- EQUITY / CRYPTO: market_value = quantity * price, price looked up as of
  the position's own `as_of` date (aggregation/pricing.py).
- OPTION: no live/historical dollar price is available from voledge's
  connector (it only carries the vol-surface edge, not a bid/ask); the plan
  calls for delta-equivalent exposure for options anyway, so market_value =
  quantity * delta * underlying_spot (spot priced as of `as_of`, same
  pricing path as everything else). Full Greeks stay on the position
  unchanged, for later Greek aggregation.
- SYNTHETIC: no real venue, so no real price exists -- market_value stays
  None, disclosed via ValuationResult.notes rather than guessed.

Positions that can't be priced (network failure, no history) are returned
unchanged (market_value stays None) with the reason recorded, never
silently dropped or defaulted to zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from aggregation.pricing import fetch_price_asof
from connectors.schema import AssetClass, Position


@dataclass
class ValuationResult:
    positions: list[Position]
    n_priced: int
    n_unpriced: int
    notes: list[str]


def _value_equity_or_crypto(p: Position) -> tuple[Position, str | None]:
    result = fetch_price_asof(p.asset, p.as_of)
    if result.price is None:
        return p, f"{p.strategy}/{p.asset}: no price available ({result.note or 'unknown reason'})"
    priced = replace(p, price=result.price, market_value=p.quantity * result.price)
    note = f"{p.strategy}/{p.asset}: priced at {result.price:.4f} ({result.source})" + (
        f" -- {result.note}" if result.note else ""
    )
    return priced, note


def _value_option(p: Position) -> tuple[Position, str | None]:
    if not p.greeks or "delta" not in p.greeks:
        return p, f"{p.strategy}/{p.asset}: no delta available; cannot compute delta-equivalent exposure"
    underlying = p.extra.get("underlying")
    if not underlying:
        return p, f"{p.strategy}/{p.asset}: no underlying recorded; cannot price"

    result = fetch_price_asof(underlying, p.as_of)
    if result.price is None:
        return p, f"{p.strategy}/{p.asset}: no underlying price for {underlying} ({result.note})"

    delta_equivalent = p.quantity * p.greeks["delta"] * result.price
    priced = replace(p, price=result.price, market_value=delta_equivalent)
    return priced, (
        f"{p.strategy}/{p.asset}: delta-equivalent exposure = {p.quantity:g} * "
        f"{p.greeks['delta']:.4f} * {result.price:.2f} = {delta_equivalent:.2f}"
    )


def value_position(p: Position) -> tuple[Position, str | None]:
    if p.asset_class == AssetClass.SYNTHETIC:
        return p, f"{p.strategy}/{p.asset}: synthetic instrument, no real market -- left unpriced"
    if p.asset_class == AssetClass.OPTION:
        return _value_option(p)
    return _value_equity_or_crypto(p)


def value_positions(positions: list[Position]) -> ValuationResult:
    valued: list[Position] = []
    notes: list[str] = []
    n_priced = 0
    for p in positions:
        priced, note = value_position(p)
        valued.append(priced)
        if note:
            notes.append(note)
        if priced.market_value is not None:
            n_priced += 1
    return ValuationResult(
        positions=valued, n_priced=n_priced, n_unpriced=len(valued) - n_priced, notes=notes
    )
