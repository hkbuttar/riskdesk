"""Network-independent contract tests for the FastAPI layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from fastapi.testclient import TestClient

import backend.main as api


client = TestClient(api.app)


REQUIRED_ROUTES = {
    "/health",
    "/api/positions",
    "/api/exposure",
    "/api/var",
    "/api/var/regime-conditional",
    "/api/correlation/static",
    "/api/correlation/dcc-garch",
    "/api/regime",
    "/api/factors",
    "/api/greeks",
    "/api/stress/historical",
    "/api/stress/hypothetical",
    "/api/reverse-stress",
    "/api/credit",
    "/api/extreme-value",
    "/api/liquidity",
    "/api/attribution",
    "/api/monitor",
}


@dataclass
class _Valued:
    positions: list
    n_priced: int = 0
    n_unpriced: int = 0
    notes: list[str] | None = None


def test_openapi_exposes_every_step_20_route():
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert REQUIRED_ROUTES <= set(schema.json()["paths"])
    assert schema.json()["info"]["version"] == "1.0.0"


def test_health_is_a_stable_machine_readable_probe():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "riskdesk", "version": "1.0.0"}


def test_positions_route_serializes_module_output(monkeypatch):
    position = {"asset": "SPY", "market_value": 1000.0}
    monkeypatch.setattr(api, "_valued_positions", lambda: _Valued([position], 1, 0, ["test source"]))

    response = client.get("/api/positions")

    assert response.status_code == 200
    assert response.json() == {
        "positions": [position],
        "n_priced": 1,
        "n_unpriced": 0,
        "notes": ["test source"],
    }


def test_var_route_exposes_all_five_methods_and_forwards_confidence(monkeypatch):
    pnl = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    returns = pd.DataFrame({"SPY": [0.01, -0.01, 0.0, 0.02, -0.02]})
    monkeypatch.setattr(api, "_risk_factor_setup", lambda: (_Valued([]), ["SPY"], returns, pnl, {"SPY": 100.0}))

    response = client.get("/api/var?confidence=0.975")

    assert response.status_code == 200
    assert set(response.json()) == {
        "historical_simulation",
        "parametric_normal",
        "monte_carlo",
        "cornish_fisher",
        "evt_pot",
    }
    assert {result["confidence"] for result in response.json().values()} == {0.975}


def test_query_constraints_return_422_before_analytics_run():
    assert client.get("/api/var?confidence=1").status_code == 422
    assert client.get("/api/reverse-stress?target_fraction=0").status_code == 422
    assert client.get("/api/extreme-value?threshold_quantile=1").status_code == 422
    assert client.get("/api/attribution?window_days=0").status_code == 422


def test_cors_origins_are_trimmed():
    middleware = next(item for item in api.app.user_middleware if item.cls.__name__ == "CORSMiddleware")
    assert all(origin == origin.strip() for origin in middleware.kwargs["allow_origins"])
