"""RiskDesk API: one endpoint per risk-analytics module built in this
project, each a thin wrapper over that module's own functions -- no
computation lives here that doesn't already live in the module it exposes.

Endpoints recompute everything fresh on every request (fetching live data
from connectors, Alpaca, etc.) -- there is deliberately no caching
layer. That's a real, disclosed limitation for production use (several
endpoints, e.g. DCC-GARCH, the HMM, reverse stress, take multiple seconds),
not an oversight; adding caching is real future work, not implemented here
to keep this module's job (routing) separate from a caching layer's job.

CORS follows the exact same pattern already established across this
portfolio's other backends (alpha-signal-lab/backend/main.py):
ALLOWED_ORIGINS env var, comma-separated, defaults wide open ("*") for
initial setup.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from aggregation.greeks import aggregate_greeks, gamma_convexity_table
from aggregation.rollup import by_asset_class, by_counterparty, by_strategy, portfolio_total
from aggregation.valuation import value_positions
from attribution.pnl import attribute_by_factor
from attribution.strategy import attribute_by_strategy
from backend.serialize import to_jsonable
from connectors.registry import fetch_all
from connectors.alpaca_market_data import fetch_history
from correlation.dcc_garch import fit_dcc_garch
from correlation.shrinkage import ledoit_wolf_covariance
from correlation.static import static_correlation
from credit.concentration import check_concentration
from credit.cva import compute_cva
from credit.counterparty import COUNTERPARTY_PD
from extreme_value.tail_risk import compare_tail_shape, fit_gpd_by_regime
from factor_model.factors import fetch_factor_returns, sector_of
from factor_model.pca import fit_pca
from factor_model.regression import fit_factor_regression
from factor_model.vega import aggregate_vega
from liquidity.concentration import check_by_name, check_by_sector, check_by_strategy
from liquidity.impact import fetch_avg_daily_dollar_volume, liquidity_adjusted_var
from monitor.kill_switch import load_state
from monitor.limits import check_credit_concentration_limit, check_var_limit
from regime.conditional import compare_pooled_vs_conditional
from regime.hmm_regime import fit_hmm_regimes
from regime.volatility_tercile import classify_regimes, rolling_realized_vol
from reverse_stress.optimization import solve_reverse_stress
from reverse_stress.plausibility import compare_pooled_vs_volatile_regime, compare_to_historical_windows
from risk_measures.returns import build_portfolio_pnl_series, fetch_return_history, position_risk_factor
from risk_measures.var import (
    cornish_fisher,
    evt_pot,
    historical_simulation,
    monte_carlo,
    parametric_variance_covariance,
)
from stress.hypothetical import HYPOTHETICAL_SCENARIOS, run_scenario
from stress.historical import HISTORICAL_WINDOWS, fetch_price_history, replay_window

app = FastAPI(
    title="RiskDesk API",
    description="Portfolio risk aggregation, stress testing, credit risk, and live monitoring.",
    version="1.0.0",
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_origins = [origin.strip() for origin in _allowed_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _allowed_origins.strip() == "*" else _origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _valued_positions():
    raw_positions, _ = fetch_all()
    return value_positions(raw_positions)


def _risk_factor_setup():
    """The book, its risk-factor return history, and its hypothetical P&L
    series -- the common input every risk-measure/factor/regime endpoint
    below needs, assembled once per request rather than duplicated per route.
    """
    valued = _valued_positions()
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, _ = fetch_return_history(tickers)
    pnl_series, weights, _ = build_portfolio_pnl_series(valued.positions, returns_df)
    return valued, tickers, returns_df, pnl_series, weights


def _spy_regime_labels():
    spy_close = fetch_history(["SPY"], period="2y", field="close")["SPY"]
    return spy_close, classify_regimes(rolling_realized_vol(spy_close)).labels


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "riskdesk", "version": app.version}


@app.get("/api/positions")
def positions() -> dict:
    valued = _valued_positions()
    return to_jsonable({
        "positions": valued.positions,
        "n_priced": valued.n_priced,
        "n_unpriced": valued.n_unpriced,
        "notes": valued.notes,
    })


@app.get("/api/exposure")
def exposure() -> dict:
    valued = _valued_positions()
    return to_jsonable({
        "portfolio": portfolio_total(valued.positions),
        "by_strategy": by_strategy(valued.positions),
        "by_asset_class": by_asset_class(valued.positions),
        "by_counterparty": by_counterparty(valued.positions),
    })


@app.get("/api/var")
def var(confidence: float = Query(0.95, gt=0.5, lt=1.0)) -> dict:
    _, _, returns_df, pnl_series, weights = _risk_factor_setup()
    return to_jsonable({
        "historical_simulation": historical_simulation(pnl_series, confidence),
        "parametric_normal": parametric_variance_covariance(pnl_series, confidence),
        "monte_carlo": monte_carlo(returns_df, weights, confidence),
        "cornish_fisher": cornish_fisher(pnl_series, confidence),
        "evt_pot": evt_pot(pnl_series, confidence),
    })


@app.get("/api/var/regime-conditional")
def var_regime_conditional(confidence: float = Query(0.95, gt=0.5, lt=1.0)) -> dict:
    _, _, returns_df, pnl_series, _ = _risk_factor_setup()
    _, regime_labels = _spy_regime_labels()
    result = compare_pooled_vs_conditional(pnl_series, returns_df, regime_labels, confidence)
    return to_jsonable(result)


@app.get("/api/correlation/static")
def correlation_static() -> dict:
    _, _, returns_df, _, _ = _risk_factor_setup()
    return to_jsonable({
        "correlation": static_correlation(returns_df),
        "ledoit_wolf": ledoit_wolf_covariance(returns_df),
    })


@app.get("/api/correlation/dcc-garch")
def correlation_dcc_garch() -> dict:
    _, _, returns_df, _, _ = _risk_factor_setup()
    result = fit_dcc_garch(returns_df)
    return to_jsonable({
        "a": result.a, "b": result.b, "notes": result.notes,
        "latest_correlation": result.latest_correlation(),
        "tickers": result.tickers,
        "dates": result.dates,
        "correlation_history": result.R,
    })


@app.get("/api/regime")
def regime() -> dict:
    close, tercile_labels = _spy_regime_labels()
    tercile = classify_regimes(rolling_realized_vol(close))
    hmm = fit_hmm_regimes(close)
    return to_jsonable({
        "tercile": {
            "current_regime": tercile.current(),
            "thresholds": tercile.thresholds,
            "value_counts": tercile.value_counts(),
        },
        "hmm": {
            "regime_params": hmm.regime_params,
            "current_probabilities": hmm.current_probabilities(),
        },
    })


@app.get("/api/factors")
def factors() -> dict:
    valued, tickers, returns_df, pnl_series, _ = _risk_factor_setup()
    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)

    by_strategy_fit = {}
    for strategy in sorted({p.strategy for p in valued.positions}):
        strat_positions = [p for p in valued.positions if p.strategy == strategy]
        strat_pnl, strat_weights, _ = build_portfolio_pnl_series(strat_positions, returns_df)
        if strat_weights:
            by_strategy_fit[strategy] = fit_factor_regression(strategy, strat_pnl, factor_returns)

    pca = fit_pca(returns_df)
    vega = aggregate_vega(valued.positions)
    return to_jsonable({
        "portfolio": fit, "by_strategy": by_strategy_fit,
        "pca": {
            "explained_variance_ratio": pca.explained_variance_ratio,
            "n_components_for_90pct": pca.n_components_for_90pct,
        },
        "vega": vega,
    })


@app.get("/api/greeks")
def greeks() -> dict:
    valued = _valued_positions()
    return to_jsonable({
        "portfolio_greeks": aggregate_greeks(valued.positions),
        "convexity_table": gamma_convexity_table(valued.positions),
    })


@app.get("/api/stress/historical")
def stress_historical() -> dict:
    valued = _valued_positions()
    risk_factors = {position_risk_factor(p) for p in valued.positions if position_risk_factor(p)}
    from aggregation.pricing import resolve_symbol
    tickers = sorted({resolve_symbol(f) for f in risk_factors} | {"SPY"})
    earliest = min(w["pre_start"] for w in HISTORICAL_WINDOWS.values())
    latest = max(w["end"] for w in HISTORICAL_WINDOWS.values())
    full_history = fetch_price_history(tickers, earliest, latest)

    results = {}
    for name, window in HISTORICAL_WINDOWS.items():
        window_prices = full_history.loc[window["start"] : window["end"]]
        result = replay_window(valued.positions, window_prices)
        results[name] = {"window": window, "result": result}
    return to_jsonable(results)


@app.get("/api/stress/hypothetical")
def stress_hypothetical() -> dict:
    valued = _valued_positions()
    return to_jsonable({
        name: run_scenario(name, valued.positions) for name in HYPOTHETICAL_SCENARIOS
    })


@app.get("/api/reverse-stress")
def reverse_stress(target_fraction: float = Query(0.25, gt=0.0, lt=1.0)) -> dict:
    valued, tickers, returns_df, pnl_series, _ = _risk_factor_setup()
    gross = portfolio_total(valued.positions).gross_market_value
    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)

    result = solve_reverse_stress(fit.loadings, fit.alpha, factor_returns, gross * target_fraction)
    regime_comparison = compare_pooled_vs_volatile_regime(
        result.factor_shocks, factor_returns, result.mahalanobis_distance,
        result.implied_annual_probability, horizon_days=result.horizon_days,
    )
    historical = compare_to_historical_windows(result.factor_shocks)
    return to_jsonable({
        "target_loss": result.target_loss, "solved_scenario": result,
        "regime_conditional_distance": regime_comparison, "vs_historical_windows": historical,
    })


@app.get("/api/credit")
def credit() -> dict:
    valued = _valued_positions()
    return to_jsonable({
        "pd_tiers": {cp.value: tier for cp, tier in COUNTERPARTY_PD.items()},
        "cva": compute_cva(valued.positions),
        "concentration": check_concentration(valued.positions),
    })


@app.get("/api/extreme-value")
def extreme_value(threshold_quantile: float = Query(0.80, gt=0.0, lt=1.0)) -> dict:
    _, _, returns_df, pnl_series, _ = _risk_factor_setup()
    _, regime_labels = _spy_regime_labels()
    comparison = fit_gpd_by_regime(pnl_series, regime_labels, threshold_quantile)
    return to_jsonable({
        "notes": comparison.notes,
        "fits": comparison,
        "tail_shape_by_regime": compare_tail_shape(comparison),
    })


@app.get("/api/liquidity")
def liquidity() -> dict:
    valued, tickers, returns_df, pnl_series, _ = _risk_factor_setup()
    daily_vol = returns_df.std().to_dict()
    dollar_volume = fetch_avg_daily_dollar_volume(tickers)
    base_var = historical_simulation(pnl_series).var_dollar
    adjusted_var, costs, notes = liquidity_adjusted_var(base_var, valued.positions, daily_vol, dollar_volume)
    return to_jsonable({
        "base_var": base_var, "liquidity_adjusted_var": adjusted_var,
        "liquidation_costs": costs, "notes": notes,
        "concentration_by_name": check_by_name(valued.positions),
        "concentration_by_strategy": check_by_strategy(valued.positions),
        "concentration_by_sector": check_by_sector(valued.positions),
    })


@app.get("/api/attribution")
def attribution(window_days: int = Query(63, gt=0, le=500)) -> dict:
    valued, tickers, returns_df, pnl_series, _ = _risk_factor_setup()
    sectors_held = {sector_of(t) for t in tickers if sector_of(t)}
    factor_returns, _ = fetch_factor_returns(sectors_held)
    fit = fit_factor_regression("portfolio", pnl_series, factor_returns)

    recent_pnl = pnl_series.iloc[-window_days:]
    factor_attribution = attribute_by_factor(recent_pnl, factor_returns, fit.loadings, fit.alpha)
    by_strategy_pnl = attribute_by_strategy(valued.positions, returns_df)
    by_strategy_recent = {k: float(v.reindex(recent_pnl.index).dropna().sum()) for k, v in by_strategy_pnl.items()}

    return to_jsonable({
        "window_days": window_days,
        "by_strategy": by_strategy_recent,
        "by_factor": factor_attribution,
    })


@app.get("/api/monitor")
def monitor() -> dict:
    """Read-only live status -- the kill-switch can only be reset via the
    CLI (`python -m monitor.live --reset`), a deliberate choice: resetting
    a kill-switch is a highly consequential action this project has no
    authentication system to gate behind an HTTP endpoint, so it isn't
    exposed as one.
    """
    valued, _, returns_df, pnl_series, _ = _risk_factor_setup()
    gross = portfolio_total(valued.positions).gross_market_value
    _, regime_labels = _spy_regime_labels()
    current_regime = regime_labels.dropna().iloc[-1]

    conditional = compare_pooled_vs_conditional(pnl_series, returns_df, regime_labels)
    if current_regime in conditional.conditional_var:
        active_var = conditional.conditional_var[current_regime]["historical_simulation"].var_dollar
        model_label = f"{current_regime}-conditional"
    else:
        active_var = conditional.pooled_var["historical_simulation"].var_dollar
        model_label = "pooled"

    var_check = check_var_limit(active_var, gross)
    credit_check = check_credit_concentration_limit(valued.positions)
    kill_switch = load_state()

    return to_jsonable({
        "current_regime": current_regime, "active_model": model_label, "active_var": active_var,
        "gross_exposure": gross, "var_check": var_check, "credit_check": credit_check,
        "kill_switch_triggered": kill_switch.triggered, "kill_switch_reasons": kill_switch.trigger_reasons,
    })
