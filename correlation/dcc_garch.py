"""DCC-GARCH (Engle 2002): time-varying correlation, the direct answer to
static.py's limitation -- correlation that can actually move when markets
move together under stress, rather than a single number assumed constant
over the whole window.

`arch` (the package used here for the univariate stage) does not ship a
ready-made multivariate DCC class, so this is a standard two-stage
estimation implemented directly on top of it:

  Stage 1 (per-asset): fit a univariate GARCH(1,1) to each risk factor's
  returns (`arch.univariate.arch_model`), extract standardized residuals
  z_i,t = r_i,t / sigma_i,t. If z_t is well-specified, it should have unit
  variance and no autocorrelation in its second moment -- the GARCH stage
  is what makes the DCC stage's inputs comparable across assets with very
  different absolute volatility (e.g. BTC-USD vs. a defensive equity).

  Stage 2 (DCC): the standardized residuals' *correlation* structure is
  itself allowed to vary over time via

      Q_t = (1 - a - b) * Qbar + a * z_{t-1} z_{t-1}' + b * Q_{t-1}
      R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

  where Qbar is the unconditional covariance of the standardized residuals
  and (a, b) are fit by maximizing the concentrated Gaussian
  quasi-log-likelihood of the correlation stage only (standard two-stage
  DCC estimation -- the volatility stage's likelihood is already accounted
  for in Stage 1, so this stage's likelihood is conditional on it, not a
  full joint MLE). a, b >= 0 and a + b < 1 are the model's own stationarity
  constraints.

Disclosed simplification: GARCH(1,1) with a normal conditional distribution
for every asset, not asset-specific model/distribution selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize


@dataclass
class DCCGARCHResult:
    tickers: list[str]
    dates: pd.DatetimeIndex
    a: float
    b: float
    garch_params: dict[str, dict[str, float]]
    R: np.ndarray  # shape (T, N, N) time-varying correlation matrices
    notes: str

    def latest_correlation(self) -> pd.DataFrame:
        return pd.DataFrame(self.R[-1], index=self.tickers, columns=self.tickers)

    def correlation_series(self, ticker_a: str, ticker_b: str) -> pd.Series:
        i, j = self.tickers.index(ticker_a), self.tickers.index(ticker_b)
        return pd.Series(self.R[:, i, j], index=self.dates, name=f"{ticker_a}-{ticker_b}")


def _fit_univariate_garch(returns_pct: pd.Series) -> tuple[np.ndarray, dict[str, float]]:
    am = arch_model(returns_pct.to_numpy(), mean="Zero", vol="GARCH", p=1, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)
    std_resid = res.std_resid
    params = {"omega": res.params["omega"], "alpha": res.params["alpha[1]"], "beta": res.params["beta[1]"]}
    return std_resid, params


def _dcc_neg_loglik(theta: np.ndarray, z: np.ndarray, qbar: np.ndarray) -> float:
    a, b = theta
    if a < 0 or b < 0 or a + b >= 1:
        return 1e10

    t_obs, n = z.shape
    q_prev = qbar.copy()
    nll = 0.0
    for t in range(t_obs):
        d_inv = np.diag(1.0 / np.sqrt(np.diag(q_prev)))
        r_t = d_inv @ q_prev @ d_inv
        try:
            sign, logdet = np.linalg.slogdet(r_t)
            r_inv = np.linalg.inv(r_t)
        except np.linalg.LinAlgError:
            return 1e10
        if sign <= 0:
            return 1e10
        z_t = z[t]
        nll += 0.5 * (logdet + z_t @ r_inv @ z_t - z_t @ z_t)

        outer = np.outer(z_t, z_t)
        q_prev = (1 - a - b) * qbar + a * outer + b * q_prev

    return nll


def fit_dcc_garch(returns_df: pd.DataFrame) -> DCCGARCHResult:
    """`returns_df` holds simple daily returns (as produced by
    risk_measures/returns.py::fetch_return_history); rescaled to percent
    internally, per `arch`'s own recommendation for numerically stable
    GARCH fitting.
    """
    tickers = list(returns_df.columns)
    returns_pct = returns_df * 100.0

    garch_params: dict[str, dict[str, float]] = {}
    std_resid_cols = {}
    for ticker in tickers:
        std_resid, params = _fit_univariate_garch(returns_pct[ticker])
        garch_params[ticker] = params
        std_resid_cols[ticker] = std_resid

    z = pd.DataFrame(std_resid_cols, index=returns_df.index).to_numpy()
    qbar = np.cov(z, rowvar=False)

    fit = minimize(
        _dcc_neg_loglik, x0=np.array([0.03, 0.90]), args=(z, qbar),
        method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 500},
    )
    a, b = float(fit.x[0]), float(fit.x[1])
    a, b = max(a, 0.0), max(b, 0.0)
    if a + b >= 1:
        scale = 0.999 / (a + b)
        a, b = a * scale, b * scale

    t_obs, n = z.shape
    r_series = np.zeros((t_obs, n, n))
    q_prev = qbar.copy()
    for t in range(t_obs):
        d_inv = np.diag(1.0 / np.sqrt(np.diag(q_prev)))
        r_series[t] = d_inv @ q_prev @ d_inv
        outer = np.outer(z[t], z[t])
        q_prev = (1 - a - b) * qbar + a * outer + b * q_prev

    return DCCGARCHResult(
        tickers=tickers,
        dates=returns_df.index,
        a=a,
        b=b,
        garch_params=garch_params,
        R=r_series,
        notes=(
            f"DCC params: a={a:.4f}, b={b:.4f} (persistence a+b={a + b:.4f}); "
            f"{t_obs} days x {n} assets; optimizer: {fit.message}"
        ),
    )
