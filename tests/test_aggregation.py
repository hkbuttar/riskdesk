"""Aggregation-layer tests. Rollup/valuation math is tested against
constructed Position objects with a stubbed price fetch, for determinism
(unlike tests/test_connectors.py, real market prices change every day so
hardcoding expected numbers here would be flaky). One end-to-end test does
hit real connectors + real pricing, matching this project's established
"test against real state, not mocks" preference, but only asserts
structural invariants, not specific values.
"""

from __future__ import annotations

import datetime as dt

import pytest

from aggregation import pricing
from aggregation.rollup import by_asset_class, by_counterparty, by_strategy, portfolio_total
from aggregation.valuation import value_position, value_positions
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position

TODAY = dt.date(2026, 8, 7)


def _equity_position(strategy="strat-a", asset="AAPL", quantity=10.0) -> Position:
    return Position(
        strategy=strategy,
        asset=asset,
        quantity=quantity,
        market_value=None,
        asset_class=AssetClass.EQUITY,
        counterparty=Counterparty.ALPACA,
        provenance=DataProvenance.LIVE_PAPER,
        as_of=TODAY,
    )


def _option_position(quantity=1.0, delta=0.5) -> Position:
    return Position(
        strategy="voledge",
        asset="SPY 2026-08-11 C500",
        quantity=quantity,
        market_value=None,
        asset_class=AssetClass.OPTION,
        counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=TODAY,
        greeks={"delta": delta, "gamma": 0.01, "vega": 1.0, "theta": -1.0, "rho": 0.1},
        extra={"underlying": "SPY"},
    )


def _synthetic_position() -> Position:
    return Position(
        strategy="bookmaker",
        asset="BOOKMAKER-SIM-LOB",
        quantity=5.0,
        market_value=None,
        asset_class=AssetClass.SYNTHETIC,
        counterparty=Counterparty.NONE,
        provenance=DataProvenance.BACKTEST_STANDIN,
        as_of=TODAY,
    )


@pytest.fixture(autouse=True)
def stub_pricing(monkeypatch):
    prices = {"AAPL": 200.0, "SPY": 500.0}

    def fake_fetch(asset, as_of):
        resolved = pricing.resolve_symbol(asset)
        if resolved in prices:
            return pricing.PriceResult(price=prices[resolved], source=resolved)
        return pricing.PriceResult(price=None, source=resolved, note="no stub price")

    monkeypatch.setattr(pricing, "fetch_price_asof", fake_fetch)
    # valuation.py imported fetch_price_asof by name, so patch its local binding too.
    import aggregation.valuation as valuation_mod

    monkeypatch.setattr(valuation_mod, "fetch_price_asof", fake_fetch)
    yield


def test_equity_market_value_is_quantity_times_price():
    p = _equity_position(quantity=10.0, asset="AAPL")
    priced, note = value_position(p)
    assert priced.price == 200.0
    assert priced.market_value == 2000.0
    assert note is not None


def test_option_market_value_is_delta_equivalent_exposure():
    p = _option_position(quantity=2.0, delta=0.5)
    priced, note = value_position(p)
    assert priced.market_value == pytest.approx(2.0 * 0.5 * 500.0)
    assert priced.greeks["delta"] == 0.5  # untouched


def test_synthetic_position_left_unpriced():
    p = _synthetic_position()
    priced, note = value_position(p)
    assert priced.market_value is None
    assert "synthetic" in note


def test_missing_price_leaves_position_unpriced_not_zero():
    p = _equity_position(asset="NO_SUCH_TICKER")
    priced, note = value_position(p)
    assert priced.market_value is None
    assert priced.quantity == p.quantity  # position itself untouched
    assert "no stub price" in note


def test_value_positions_counts_priced_and_unpriced():
    positions = [_equity_position(asset="AAPL"), _equity_position(asset="NOPE"), _synthetic_position()]
    result = value_positions(positions)
    assert result.n_priced == 1
    assert result.n_unpriced == 2
    assert len(result.notes) == 3


def test_rollup_net_vs_gross_with_offsetting_longs_and_shorts():
    long_pos = _equity_position(strategy="a", asset="AAPL", quantity=10.0)
    short_pos = _equity_position(strategy="b", asset="AAPL", quantity=-10.0)
    result = value_positions([long_pos, short_pos])
    total = portfolio_total(result.positions)
    assert total.net_market_value == pytest.approx(0.0)
    assert total.gross_market_value == pytest.approx(4000.0)  # 2000 + 2000, not netted


def test_rollup_by_strategy_asset_class_counterparty_partition_correctly():
    positions = [
        _equity_position(strategy="a", asset="AAPL"),
        _option_position(),
        _synthetic_position(),
    ]
    result = value_positions(positions)
    strat = by_strategy(result.positions)
    assert set(strat) == {"a", "voledge", "bookmaker"}
    asset_class = by_asset_class(result.positions)
    assert set(asset_class) == {"equity", "option", "synthetic"}
    counterparty = by_counterparty(result.positions)
    assert set(counterparty) == {"alpaca", "none"}
    # Synthetic position has no market_value -- excluded from sums, present in counts.
    assert asset_class["synthetic"].n_unpriced == 1
    assert asset_class["synthetic"].net_market_value == 0.0


def test_end_to_end_against_real_connectors_and_real_prices():
    from connectors.registry import fetch_all

    raw_positions, _ = fetch_all()
    result = value_positions(raw_positions)
    assert result.n_priced + result.n_unpriced == len(raw_positions)
    total = portfolio_total(result.positions)
    assert total.n_priced == result.n_priced
    assert total.gross_market_value >= abs(total.net_market_value)


def test_strategy_rollups_reconcile_exactly_to_portfolio_total():
    # Aggregation correctness invariant: summing every strategy-level bucket
    # must reproduce the portfolio-level total exactly -- net and gross both
    # -- since a rollup is just a partition of the same underlying positions.
    positions = [
        _equity_position(strategy="a", asset="AAPL", quantity=10.0),
        _equity_position(strategy="a", asset="MSFT", quantity=-5.0),
        _equity_position(strategy="b", asset="AAPL", quantity=3.0),
        _option_position(quantity=2.0),
        _synthetic_position(),
    ]
    result = value_positions(positions)
    total = portfolio_total(result.positions)
    strategy_buckets = by_strategy(result.positions)

    assert sum(b.net_market_value for b in strategy_buckets.values()) == pytest.approx(total.net_market_value)
    assert sum(b.gross_market_value for b in strategy_buckets.values()) == pytest.approx(total.gross_market_value)
    assert sum(b.n_positions for b in strategy_buckets.values()) == total.n_positions
    assert sum(b.n_priced for b in strategy_buckets.values()) == total.n_priced
    assert sum(b.n_unpriced for b in strategy_buckets.values()) == total.n_unpriced


def test_asset_class_and_counterparty_rollups_also_reconcile():
    positions = [
        _equity_position(strategy="a", asset="AAPL", quantity=10.0),
        _option_position(quantity=1.0),
        _synthetic_position(),
    ]
    result = value_positions(positions)
    total = portfolio_total(result.positions)

    for rollup_fn in (by_asset_class, by_counterparty):
        buckets = rollup_fn(result.positions)
        assert sum(b.net_market_value for b in buckets.values()) == pytest.approx(total.net_market_value)
        assert sum(b.gross_market_value for b in buckets.values()) == pytest.approx(total.gross_market_value)
        assert sum(b.n_positions for b in buckets.values()) == total.n_positions


def test_empty_position_list_produces_zeroed_portfolio_total_not_an_error():
    result = value_positions([])
    assert result.n_priced == 0
    assert result.n_unpriced == 0
    total = portfolio_total(result.positions)
    assert total.net_market_value == 0.0
    assert total.gross_market_value == 0.0
    assert total.n_positions == 0
    assert by_strategy(result.positions) == {}
