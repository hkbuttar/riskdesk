"""Live monitoring tests. Kill-switch state machine and persistence are
checked directly (manual-reset-only semantics, survival across separate
load/save cycles simulating separate process invocations). Limit checks
are checked against hand-computable cases. One end-to-end test runs the
real live-check pipeline with kill-switch persistence isolated to a temp
path, so running the test suite never mutates the real monitor/state.json.
"""

from __future__ import annotations

import datetime as dt

import pytest

from connectors.schema import AssetClass, Counterparty, DataProvenance, Position
from monitor.kill_switch import KillSwitch, load_state, save_state
from monitor.limits import check_credit_concentration_limit, check_var_limit

TODAY = dt.date(2026, 8, 7)


def test_kill_switch_triggers_on_any_breach():
    ks = KillSwitch()
    ks.check({"a": False, "b": True})
    assert ks.triggered
    assert "b" in ks.trigger_reasons


def test_kill_switch_stays_triggered_once_breach_clears():
    ks = KillSwitch()
    ks.check({"a": True})
    assert ks.triggered
    ks.check({"a": False})  # breach cleared on this check
    assert ks.triggered  # but switch does NOT auto-clear


def test_kill_switch_reset_clears_state():
    ks = KillSwitch()
    ks.check({"a": True})
    ks.reset()
    assert not ks.triggered
    assert ks.trigger_reasons == []


def test_kill_switch_does_not_duplicate_reasons_across_checks():
    ks = KillSwitch()
    ks.check({"a": "reason A"})
    ks.check({"a": "reason A"})
    assert ks.trigger_reasons.count("reason A") == 1


def test_kill_switch_uses_string_detail_as_reason():
    ks = KillSwitch()
    ks.check({"var_limit": "VaR is 12% of gross exposure (limit 5%) -- BREACHED"})
    assert ks.trigger_reasons == ["VaR is 12% of gross exposure (limit 5%) -- BREACHED"]


def test_kill_switch_persistence_survives_separate_load_save_cycles(tmp_path):
    path = tmp_path / "state.json"
    ks = load_state(path)
    assert not ks.triggered  # no file yet -> fresh switch

    ks.check({"a": True})
    save_state(ks, path)

    # Simulate a separate process invocation: fresh KillSwitch object, reloaded from disk.
    reloaded = load_state(path)
    assert reloaded.triggered
    assert reloaded.trigger_reasons == ["a"]

    reloaded.reset()
    save_state(reloaded, path)
    assert not load_state(path).triggered


def _position(strategy, asset, market_value, counterparty=Counterparty.NONE):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=AssetClass.EQUITY, counterparty=counterparty,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY,
    )


def test_check_var_limit_breaches_above_threshold():
    result = check_var_limit(var_dollar=10000.0, gross_exposure=100000.0, limit_fraction=0.05)
    assert result.breached  # 10% > 5% limit
    assert "BREACHED" in result.detail


def test_check_var_limit_does_not_breach_below_threshold():
    result = check_var_limit(var_dollar=2000.0, gross_exposure=100000.0, limit_fraction=0.05)
    assert not result.breached  # 2% < 5% limit


def test_check_var_limit_not_applicable_for_zero_exposure():
    result = check_var_limit(var_dollar=1000.0, gross_exposure=0.0)
    assert not result.breached
    assert "not applicable" in result.detail


def test_check_credit_concentration_limit_breaches_on_dominant_venue():
    positions = [
        _position("a", "X", 9000.0, Counterparty.BINANCE),
        _position("b", "Y", 1000.0, Counterparty.ALPACA),
    ]
    result = check_credit_concentration_limit(positions)
    assert result.breached
    assert "binance" in result.detail.lower()


def test_check_credit_concentration_limit_not_breached_for_even_split():
    positions = [
        _position("a", "X", 5000.0, Counterparty.BINANCE),
        _position("b", "Y", 5000.0, Counterparty.ALPACA),
    ]
    result = check_credit_concentration_limit(positions)
    assert not result.breached


def test_end_to_end_live_check_with_isolated_state(tmp_path, monkeypatch):
    import monitor.live as live_module
    from monitor.kill_switch import KillSwitch as KS

    isolated_path = tmp_path / "state.json"
    monkeypatch.setattr(live_module, "load_state", lambda: __import__("monitor.kill_switch", fromlist=["load_state"]).load_state(isolated_path))
    monkeypatch.setattr(live_module, "save_state", lambda ks: __import__("monitor.kill_switch", fromlist=["save_state"]).save_state(ks, isolated_path))

    triggered = live_module.run_live_check()
    assert isinstance(triggered, bool)
    assert isolated_path.exists()  # state was persisted, not left only in memory
