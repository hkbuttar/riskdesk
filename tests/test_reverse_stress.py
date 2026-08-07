"""Reverse stress testing tests. The optimizer is checked against a
hand-computable closed-form solution (minimum-Mahalanobis-distance-subject-
to-linear-constraint has a known closed form via Lagrange multipliers,
independent of cvxpy) and against constraint satisfaction directly. The
plausibility helpers are checked against a diagonal covariance where the
Mahalanobis distance is hand-computable exactly. One end-to-end test runs
the full pipeline on the real book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reverse_stress.optimization import solve_reverse_stress, to_horizon_returns
from reverse_stress.plausibility import mahalanobis_distance


def test_to_horizon_returns_sums_correctly():
    dates = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"A": [0.01, 0.02, -0.01, 0.03, 0.01]}, index=dates)
    result = to_horizon_returns(df, horizon_days=3)
    # Rows: sum of each rolling 3-day window, first 2 rows dropped (incomplete window).
    assert len(result) == 3
    assert result["A"].iloc[0] == pytest.approx(0.01 + 0.02 + -0.01)
    assert result["A"].iloc[1] == pytest.approx(0.02 + -0.01 + 0.03)
    assert result["A"].iloc[2] == pytest.approx(-0.01 + 0.03 + 0.01)


def test_solve_reverse_stress_satisfies_constraint_exactly():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=600)
    factor_returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 600), "B": rng.normal(0, 0.02, 600)}, index=dates
    )
    loadings = {"A": 10000.0, "B": -5000.0}
    alpha = 100.0
    target_loss = 50000.0

    result = solve_reverse_stress(loadings, alpha, factor_returns, target_loss, horizon_days=21)
    assert result.implied_pnl == pytest.approx(-target_loss, rel=1e-6)


def test_solve_reverse_stress_matches_closed_form_minimum_distance_solution():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=600)
    factor_returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 600), "B": rng.normal(0, 0.015, 600), "C": rng.normal(0, 0.008, 600)},
        index=dates,
    )
    loadings = {"A": 8000.0, "B": -3000.0, "C": 4000.0}
    alpha = 0.0
    horizon_days = 21
    target_loss = 30000.0

    result = solve_reverse_stress(loadings, alpha, factor_returns, target_loss, horizon_days=horizon_days)

    # Independently reproduce the closed-form Lagrangian solution using the
    # SAME horizon-transform + Ledoit-Wolf covariance the function uses
    # internally, to check cvxpy's numerical solve against known math.
    from correlation.shrinkage import ledoit_wolf_covariance

    horizon_returns = to_horizon_returns(factor_returns[list(loadings.keys())], horizon_days)
    sigma = ledoit_wolf_covariance(horizon_returns).covariance.to_numpy()
    loading_vec = np.array([loadings[f] for f in loadings])
    closed_form = sigma @ loading_vec * (-target_loss) / (loading_vec @ sigma @ loading_vec)

    solved = np.array([result.factor_shocks[f] for f in loadings])
    np.testing.assert_allclose(solved, closed_form, rtol=1e-3)


def test_mahalanobis_distance_grows_with_target_loss():
    rng = np.random.default_rng(2)
    dates = pd.bdate_range("2020-01-01", periods=600)
    factor_returns = pd.DataFrame({"A": rng.normal(0, 0.01, 600)}, index=dates)
    loadings = {"A": 10000.0}

    small = solve_reverse_stress(loadings, 0.0, factor_returns, target_loss=10000.0)
    large = solve_reverse_stress(loadings, 0.0, factor_returns, target_loss=100000.0)
    assert large.mahalanobis_distance > small.mahalanobis_distance
    assert large.implied_annual_probability <= small.implied_annual_probability


def test_mahalanobis_distance_matches_hand_computation_for_diagonal_covariance():
    cov = pd.DataFrame({"A": [4.0, 0.0], "B": [0.0, 9.0]}, index=["A", "B"])
    shock = {"A": 2.0, "B": 3.0}
    distance, probability = mahalanobis_distance(shock, cov)
    # sqrt((2^2/4) + (3^2/9)) = sqrt(1 + 1) = sqrt(2)
    assert distance == pytest.approx(np.sqrt(2))
    assert 0.0 <= probability <= 1.0


def test_end_to_end_reverse_stress_on_real_book():
    from aggregation.rollup import portfolio_total
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from factor_model.factors import fetch_factor_returns, sector_of
    from factor_model.regression import fit_factor_regression
    from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    gross = portfolio_total(valued.positions).gross_market_value

    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, _weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)

    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)

    result = solve_reverse_stress(fit.loadings, fit.alpha, factor_returns, target_loss=gross * 0.10)
    assert result.implied_pnl == pytest.approx(-gross * 0.10, rel=1e-4)
    assert result.mahalanobis_distance > 0
    assert 0.0 <= result.implied_annual_probability <= 1.0
