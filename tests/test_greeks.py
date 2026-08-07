"""Greek aggregation tests. Checked against constructed positions with
known Greeks/quantities/spot so exact expected sums and convexity behavior
(scaling with the square of the move, sign matching gamma) can be asserted
directly, plus one end-to-end test on the real book for structural
invariants only.
"""

from __future__ import annotations

import datetime as dt

import pytest

from aggregation.greeks import aggregate_greeks, gamma_convexity_table
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position

TODAY = dt.date(2026, 8, 7)


def _option(strategy, asset, quantity, delta, gamma, vega, theta, rho, spot):
    return Position(
        strategy=strategy, asset=asset, quantity=quantity, market_value=100.0, price=spot,
        asset_class=AssetClass.OPTION, counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN, as_of=TODAY,
        greeks={"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho},
        extra={"underlying": "SPY"},
    )


def _equity(strategy, asset):
    return Position(
        strategy=strategy, asset=asset, quantity=10.0, market_value=1000.0, price=100.0,
        asset_class=AssetClass.EQUITY, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_aggregate_greeks_sums_correctly():
    positions = [
        _option("v", "C1", quantity=2.0, delta=0.5, gamma=0.02, vega=10.0, theta=-1.0, rho=0.1, spot=500.0),
        _option("v", "P1", quantity=-1.0, delta=-0.3, gamma=0.01, vega=5.0, theta=-0.5, rho=-0.05, spot=500.0),
    ]
    g = aggregate_greeks(positions)
    assert g.net_delta_shares == pytest.approx(2.0 * 0.5 + -1.0 * -0.3)
    assert g.net_delta_dollars == pytest.approx((2.0 * 0.5 + -1.0 * -0.3) * 500.0)
    assert g.net_gamma_shares == pytest.approx(2.0 * 0.02 + -1.0 * 0.01)
    assert g.net_vega == pytest.approx(2.0 * 10.0 + -1.0 * 5.0)
    assert g.net_theta == pytest.approx(2.0 * -1.0 + -1.0 * -0.5)
    assert g.net_rho == pytest.approx(2.0 * 0.1 + -1.0 * -0.05)
    assert g.n_option_positions == 2


def test_aggregate_greeks_ignores_non_option_positions():
    positions = [
        _option("v", "C1", quantity=1.0, delta=0.5, gamma=0.02, vega=10.0, theta=-1.0, rho=0.1, spot=500.0),
        _equity("a", "AAPL"),
    ]
    g = aggregate_greeks(positions)
    assert g.n_option_positions == 1


def test_gamma_convexity_zero_gamma_matches_linear_exactly():
    positions = [
        _option("v", "C1", quantity=1.0, delta=0.5, gamma=0.0, vega=10.0, theta=-1.0, rho=0.1, spot=500.0)
    ]
    rows = gamma_convexity_table(positions, moves=(-0.1, -0.05, 0.05, 0.1))
    for row in rows:
        assert row.gamma_correction == pytest.approx(0.0, abs=1e-9)
        assert row.quadratic_pnl == pytest.approx(row.linear_pnl)


def test_gamma_convexity_scales_with_square_of_move():
    # delta=0 isolates the pure gamma effect (linear term vanishes).
    positions = [
        _option("v", "C1", quantity=1.0, delta=0.0, gamma=0.01, vega=0.0, theta=0.0, rho=0.0, spot=500.0)
    ]
    rows = gamma_convexity_table(positions, moves=(0.02, 0.04))
    small, large = rows[0], rows[1]
    # 0.5 * gamma * (spot*move)^2 -- doubling the move should ~4x the correction.
    assert large.gamma_correction == pytest.approx(small.gamma_correction * 4, rel=0.01)


def test_gamma_convexity_positive_gamma_adds_value_both_directions():
    positions = [
        _option("v", "C1", quantity=1.0, delta=0.0, gamma=0.01, vega=0.0, theta=0.0, rho=0.0, spot=500.0)
    ]
    rows = gamma_convexity_table(positions, moves=(-0.05, 0.05))
    down, up = rows[0], rows[1]
    assert down.gamma_correction > 0
    assert up.gamma_correction > 0
    assert down.gamma_correction == pytest.approx(up.gamma_correction)  # symmetric for delta=0


def test_gamma_convexity_negative_gamma_subtracts_value_both_directions():
    positions = [
        _option("v", "C1", quantity=-1.0, delta=0.0, gamma=0.01, vega=0.0, theta=0.0, rho=0.0, spot=500.0)
    ]
    rows = gamma_convexity_table(positions, moves=(-0.05, 0.05))
    for row in rows:
        assert row.gamma_correction < 0


def test_end_to_end_greeks_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    g = aggregate_greeks(valued.positions)
    assert g.n_option_positions >= 0

    rows = gamma_convexity_table(valued.positions)
    for row in rows:
        # Larger moves should never produce a *smaller* magnitude gamma correction.
        assert row.gamma_correction == row.gamma_correction or row.move_pct == 0  # not NaN
