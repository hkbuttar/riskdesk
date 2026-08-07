"""Correlation module tests. Static correlation and Ledoit-Wolf shrinkage
are checked against synthetic data with known structure. DCC-GARCH is
checked for basic well-formedness (valid correlation matrices, stationarity
constraint honored) against both synthetic and real data -- fitting exact
DCC parameters from a short synthetic series is too noisy to assert a
precise number against, so the synthetic test instead checks that DCC
recovers *the direction* of a deliberately time-varying correlation
(materially higher in a correlated regime than a calm one), which a static
estimate by construction cannot distinguish.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlation.dcc_garch import fit_dcc_garch
from correlation.shrinkage import ledoit_wolf_covariance
from correlation.static import static_correlation, static_covariance


def test_static_correlation_recovers_known_correlation():
    rng = np.random.default_rng(0)
    n = 5000
    x = rng.normal(0, 1, n)
    y = 0.7 * x + rng.normal(0, np.sqrt(1 - 0.7**2), n)  # corr(x,y) ~ 0.7 by construction
    df = pd.DataFrame({"x": x, "y": y})
    corr = static_correlation(df)
    assert corr.loc["x", "y"] == pytest.approx(0.7, abs=0.03)
    assert corr.loc["x", "x"] == pytest.approx(1.0)


def test_static_covariance_diagonal_is_variance():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 2, 2000)})
    cov = static_covariance(df)
    assert cov.loc["x", "x"] == pytest.approx(4.0, rel=0.1)


def test_ledoit_wolf_improves_condition_number():
    rng = np.random.default_rng(2)
    # Few observations relative to dimensions -> genuinely ill-conditioned sample covariance.
    n_obs, n_assets = 40, 15
    data = rng.normal(0, 1, size=(n_obs, n_assets))
    df = pd.DataFrame(data, columns=[f"a{i}" for i in range(n_assets)])
    result = ledoit_wolf_covariance(df)
    assert result.shrunk_condition_number < result.sample_condition_number
    assert 0.0 <= result.shrinkage <= 1.0


def test_ledoit_wolf_covariance_is_symmetric_positive_semidefinite():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(rng.normal(0, 1, size=(200, 6)), columns=list("abcdef"))
    result = ledoit_wolf_covariance(df)
    cov = result.covariance.to_numpy()
    assert np.allclose(cov, cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert (eigenvalues >= -1e-8).all()


@pytest.fixture(scope="module")
def dcc_regime_switch_returns():
    """Two assets, calm (low corr) for the first half of the window, then a
    sharply correlated regime for the second half -- a static estimate over
    the whole window necessarily blends these into one number; DCC-GARCH
    should show materially higher correlation by the end of the window.
    """
    rng = np.random.default_rng(4)
    n_half = 300
    calm = rng.multivariate_normal([0, 0], [[1, 0.05], [0.05, 1]], size=n_half)
    stressed = rng.multivariate_normal([0, 0], [[1, 0.85], [0.85, 1]], size=n_half)
    data = np.vstack([calm, stressed]) * 0.01  # return-scale, not raw normal units
    dates = pd.bdate_range("2024-01-01", periods=len(data))
    return pd.DataFrame(data, index=dates, columns=["A", "B"])


def test_dcc_garch_produces_valid_correlation_matrices(dcc_regime_switch_returns):
    result = fit_dcc_garch(dcc_regime_switch_returns)
    assert result.R.shape == (len(dcc_regime_switch_returns), 2, 2)
    # Every R_t must be a valid correlation matrix: unit diagonal, values in [-1, 1].
    diag = result.R[:, 0, 0], result.R[:, 1, 1]
    assert np.allclose(diag[0], 1.0, atol=1e-6)
    assert np.allclose(diag[1], 1.0, atol=1e-6)
    assert (result.R >= -1 - 1e-8).all() and (result.R <= 1 + 1e-8).all()


def test_dcc_garch_honors_stationarity_constraint(dcc_regime_switch_returns):
    result = fit_dcc_garch(dcc_regime_switch_returns)
    assert result.a >= 0
    assert result.b >= 0
    assert result.a + result.b < 1.0


def test_dcc_garch_correlation_series_moves_toward_the_later_regime(dcc_regime_switch_returns):
    result = fit_dcc_garch(dcc_regime_switch_returns)
    series = result.correlation_series("A", "B")
    early_avg = series.iloc[:100].mean()
    late_avg = series.iloc[-100:].mean()
    # The injected regime switch is calm -> highly correlated; DCC should
    # track that direction even if it doesn't exactly recover 0.05 / 0.85.
    assert late_avg > early_avg


def test_dcc_garch_end_to_end_on_real_book():
    from aggregation.valuation import value_positions
    from connectors.registry import fetch_all
    from risk_measures.returns import fetch_return_history, position_risk_factor

    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)

    result = fit_dcc_garch(returns_df)
    latest = result.latest_correlation()
    assert latest.shape == (len(tickers), len(tickers))
    np.testing.assert_allclose(np.diag(latest.to_numpy()), 1.0, atol=1e-6)
