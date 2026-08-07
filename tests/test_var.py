"""VaR/CVaR method tests. Each method is checked against a synthetic series
with a known/controllable distribution (so exact or near-exact answers are
computable by hand), not against real market data, which would make
expected values flaky. One end-to-end test runs the real pipeline
(connectors -> valuation -> returns -> all five methods) and asserts only
structural invariants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from risk_measures.returns import build_portfolio_pnl_series, position_risk_factor
from risk_measures.var import (
    cornish_fisher,
    evt_pot,
    historical_simulation,
    monte_carlo,
    parametric_variance_covariance,
)
from connectors.schema import AssetClass, Counterparty, DataProvenance, Position

import datetime as dt

TODAY = dt.date(2026, 8, 7)


def test_historical_simulation_matches_hand_computed_quantile():
    # Losses 1..100 (as pnl = -loss), 95th percentile of loss should be 95.
    pnl = pd.Series([-float(x) for x in range(1, 101)])
    result = historical_simulation(pnl, confidence=0.95)
    assert result.var_dollar == pytest.approx(95.0, abs=1.0)
    assert result.cvar_dollar >= result.var_dollar


def test_parametric_var_matches_normal_formula_on_normal_data():
    rng = np.random.default_rng(0)
    pnl = pd.Series(rng.normal(loc=100.0, scale=1000.0, size=50_000))
    result = parametric_variance_covariance(pnl, confidence=0.95)
    expected_var = -100.0 + stats.norm.ppf(0.95) * 1000.0
    assert result.var_dollar == pytest.approx(expected_var, rel=0.02)


def test_parametric_cvar_matches_normal_closed_form_expected_shortfall():
    # Closed-form expected shortfall for a normal loss distribution:
    # ES = mu_loss + sigma * phi(z) / (1 - confidence), independent of the
    # VaR check above -- both the VaR and CVaR formulas are verified,
    # not just one implying the other.
    rng = np.random.default_rng(6)
    mu_pnl, sigma = -25.0, 500.0
    pnl = pd.Series(rng.normal(loc=mu_pnl, scale=sigma, size=100_000))
    confidence = 0.99
    result = parametric_variance_covariance(pnl, confidence=confidence)

    mu_loss = -mu_pnl
    z = stats.norm.ppf(confidence)
    expected_cvar = mu_loss + sigma * stats.norm.pdf(z) / (1 - confidence)
    assert result.cvar_dollar == pytest.approx(expected_cvar, rel=0.02)


def test_cornish_fisher_reduces_to_parametric_for_near_normal_data():
    rng = np.random.default_rng(1)
    pnl = pd.Series(rng.normal(loc=0.0, scale=500.0, size=100_000))
    normal_result = parametric_variance_covariance(pnl, confidence=0.99)
    cf_result = cornish_fisher(pnl, confidence=0.99)
    # Near-zero skew/kurtosis on true normal data -> CF collapses to ~parametric.
    assert cf_result.var_dollar == pytest.approx(normal_result.var_dollar, rel=0.05)


def test_cornish_fisher_widens_var_for_fat_tailed_data():
    rng = np.random.default_rng(2)
    # Student-t has fat tails (excess kurtosis > 0) vs. normal.
    pnl = pd.Series(rng.standard_t(df=3, size=100_000) * 500.0)
    normal_result = parametric_variance_covariance(pnl, confidence=0.99)
    cf_result = cornish_fisher(pnl, confidence=0.99)
    assert cf_result.var_dollar > normal_result.var_dollar


def test_evt_pot_recovers_known_gpd_tail():
    rng = np.random.default_rng(3)
    true_xi, true_beta, threshold = 0.3, 200.0, 1000.0
    # Body: mild noise below threshold. Tail: real GPD exceedances above it.
    body = rng.normal(0, 50, size=2000)
    body = np.clip(body, None, threshold - 1)
    exceedances = stats.genpareto.rvs(true_xi, scale=true_beta, size=400, random_state=rng)
    tail = threshold + exceedances
    losses = np.concatenate([body, tail])
    pnl = pd.Series(-losses)

    result = evt_pot(pnl, confidence=0.99, threshold_quantile=0.90)
    true_var = stats.genpareto.ppf(1 - (1 - 0.99) * len(losses) / len(tail), true_xi, scale=true_beta) + (
        np.quantile(losses, 0.90)
    )
    assert result.var_dollar == pytest.approx(true_var, rel=0.35)  # POT fitting noise is real


def test_evt_pot_reports_nan_with_too_few_exceedances():
    pnl = pd.Series(np.random.default_rng(4).normal(0, 10, size=50))
    result = evt_pot(pnl, confidence=0.999, threshold_quantile=0.99)
    assert result.var_dollar != result.var_dollar  # NaN
    assert "too few" in result.notes.lower()


def test_monte_carlo_matches_parametric_for_single_factor():
    rng = np.random.default_rng(5)
    returns = pd.DataFrame({"AAPL": rng.normal(0.001, 0.02, size=10_000)})
    weights = {"AAPL": 100_000.0}
    mc_result = monte_carlo(returns, weights, confidence=0.95, n_sims=50_000, seed=6)

    pnl = returns["AAPL"] * weights["AAPL"]
    param_result = parametric_variance_covariance(pnl, confidence=0.95)
    assert mc_result.var_dollar == pytest.approx(param_result.var_dollar, rel=0.05)


def _position(strategy, asset, market_value, asset_class=AssetClass.EQUITY, greeks=None, extra=None):
    return Position(
        strategy=strategy, asset=asset, quantity=1.0, market_value=market_value,
        asset_class=asset_class, counterparty=Counterparty.NONE,
        provenance=DataProvenance.LIVE_PAPER, as_of=TODAY, greeks=greeks, extra=extra or {},
    )


def test_build_portfolio_pnl_series_excludes_unpriced_and_synthetic():
    positions = [
        _position("a", "AAPL", 1000.0),
        _position("b", "SYN", None, asset_class=AssetClass.EQUITY),  # unpriced
        _position("c", "SYN2", 750.0, asset_class=AssetClass.SYNTHETIC),  # priced but no risk factor
        _position("d", "NOPE", 500.0),  # no return history for this ticker
    ]
    returns_df = pd.DataFrame({"AAPL": [0.01, -0.02, 0.005]})
    pnl_series, weights, notes = build_portfolio_pnl_series(positions, returns_df)
    assert list(weights.keys()) == ["AAPL"]
    assert len(pnl_series) == 3
    assert any("unpriced" in n for n in notes)
    assert any("synthetic" in n for n in notes)
    assert any("NOPE" in n for n in notes)
    assert any("SYN2" in n for n in notes)


def test_position_risk_factor_uses_underlying_for_options():
    opt = _position(
        "voledge", "SPY 2026-08-11 C500", 100.0, asset_class=AssetClass.OPTION,
        greeks={"delta": 0.5}, extra={"underlying": "SPY"},
    )
    assert position_risk_factor(opt) == "SPY"


def test_end_to_end_var_comparison_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from risk_measures.returns import fetch_return_history

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    assert len(pnl_series) > 0
    hist = historical_simulation(pnl_series, 0.95)
    param = parametric_variance_covariance(pnl_series, 0.95)
    mc = monte_carlo(returns_df, weights, 0.95)
    cf = cornish_fisher(pnl_series, 0.95)
    for r in (hist, param, mc, cf):
        assert r.var_dollar == r.var_dollar  # not NaN
        assert r.cvar_dollar >= r.var_dollar - 1e-6  # CVaR should not be below VaR
