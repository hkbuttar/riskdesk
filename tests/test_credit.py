"""Credit risk tests. CVA and concentration are checked against constructed
positions with known market values/counterparties so exact expected
figures (net-not-gross exposure, HHI, threshold flagging) can be hand
computed. One end-to-end test runs the real pipeline for structural
invariants.
"""

from __future__ import annotations

import datetime as dt

import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from credit.concentration import check_concentration
from credit.counterparty import COUNTERPARTY_PD
from credit.cva import compute_cva

TODAY = dt.date(2026, 8, 7)


def _position(strategy, asset, market_value, counterparty):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=AssetClass.EQUITY, counterparty=counterparty,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_compute_cva_uses_net_not_gross_exposure():
    positions = [
        _position("a", "X", 10000.0, Counterparty.ALPACA),
        _position("a", "Y", -9000.0, Counterparty.ALPACA),  # mostly offsetting
    ]
    result = compute_cva(positions)
    expected_net = abs(10000.0 - 9000.0)
    pd_tier = COUNTERPARTY_PD[Counterparty.ALPACA]
    expected_cva = expected_net * pd_tier.annualized_pd * result.lgd_used
    assert result.by_counterparty["alpaca"]["cva"] == pytest.approx(expected_cva)
    # Sanity: gross would have been 19000, materially larger than net 1000.
    assert result.by_counterparty["alpaca"]["cva"] < 19000.0 * pd_tier.annualized_pd * result.lgd_used


def test_compute_cva_is_zero_for_no_venue_counterparty():
    positions = [_position("a", "X", 50000.0, Counterparty.NONE)]
    result = compute_cva(positions)
    assert result.by_counterparty["none"]["cva"] == pytest.approx(0.0)
    assert result.total_cva == pytest.approx(0.0)


def test_compute_cva_scales_linearly_with_lgd():
    positions = [_position("a", "X", 10000.0, Counterparty.BINANCE)]
    low = compute_cva(positions, lgd=0.30)
    high = compute_cva(positions, lgd=0.60)
    assert high.total_cva == pytest.approx(low.total_cva * 2, rel=1e-6)


def test_compute_cva_matches_hand_calculation():
    positions = [_position("a", "X", 5000.0, Counterparty.COINBASE)]
    result = compute_cva(positions, lgd=0.5)
    pd_tier = COUNTERPARTY_PD[Counterparty.COINBASE]
    expected = 5000.0 * pd_tier.annualized_pd * 0.5
    assert result.total_cva == pytest.approx(expected)


def test_check_concentration_excludes_none_counterparty():
    positions = [
        _position("a", "X", 500000.0, Counterparty.NONE),  # huge but no real venue
        _position("b", "Y", 1000.0, Counterparty.ALPACA),
    ]
    result = check_concentration(positions)
    assert "none" not in result.exposure_by_counterparty
    assert result.total_real_venue_exposure == pytest.approx(1000.0)
    assert result.shares["alpaca"] == pytest.approx(1.0)


def test_check_concentration_flags_single_dominant_venue():
    positions = [
        _position("a", "X", 9000.0, Counterparty.BINANCE),
        _position("b", "Y", 1000.0, Counterparty.ALPACA),
    ]
    result = check_concentration(positions, threshold=0.5)
    assert "binance" in result.flagged
    assert "alpaca" not in result.flagged
    assert result.shares["binance"] == pytest.approx(0.9)


def test_check_concentration_hhi_for_even_split():
    positions = [
        _position("a", "X", 5000.0, Counterparty.BINANCE),
        _position("b", "Y", 5000.0, Counterparty.ALPACA),
    ]
    result = check_concentration(positions)
    assert result.herfindahl_index == pytest.approx(0.5)  # 0.5^2 + 0.5^2
    assert result.flagged == []


def test_check_concentration_returns_zero_when_no_real_venue_exposure():
    positions = [_position("a", "X", 10000.0, Counterparty.NONE)]
    result = check_concentration(positions)
    assert result.total_real_venue_exposure == 0.0
    assert result.flagged == []
    assert result.herfindahl_index == 0.0


def test_end_to_end_credit_risk_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)

    cva = compute_cva(valued.positions)
    assert cva.total_cva >= 0.0
    assert cva.by_counterparty["none"]["cva"] == pytest.approx(0.0)

    concentration = check_concentration(valued.positions)
    assert 0.0 <= concentration.herfindahl_index <= 1.0
    for flagged in concentration.flagged:
        assert concentration.shares[flagged] > 0.50
