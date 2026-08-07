"""Hypothetical multi-factor stress scenarios: construct a shock (equities
down X%, crypto down Y%, implied vol up Z%, optional sector overrides), then
fully reprice every position -- linear for equity/crypto, delta+gamma+vega
Taylor expansion for options (extending aggregation/greeks.py's gamma-only
convexity check to also include the vega leg, now that these scenarios
define an explicit vol shock to apply it against).

Scenario construction is a disclosed judgment call, not calibrated to any
specific model -- these are round, illustrative numbers meant to span a
few different shapes of stress (broad equity crash, crypto-specific,
sector-specific, rate-driven), not a claim about any one of them being
more likely than another. See HYPOTHETICAL_SCENARIOS docstrings for the
reasoning behind each.

Vega units, precisely: voledge's raw Black-Scholes vega (greeks/analytical.py)
is dV/dσ with σ in decimal -- "vega per 1.00 = 100 vol points" per that
module's own docstring. A scenario's `vol_shock_pct` here is a RELATIVE
shock to each option's own entry IV (e.g. 0.50 means IV rises 50% relative
to whatever that specific contract's IV already was, not a flat +50-point
move), converted to an absolute decimal delta (`d_sigma = entry_iv *
vol_shock_pct`) before multiplying by the raw vega -- getting this
unit conversion right is exactly what caught and fixed a real vega-units
error in aggregation/greeks.py's earlier documentation (see that module's
now-corrected comment).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from connectors.schema import AssetClass, Position
from factor_model.factors import sector_of

HYPOTHETICAL_SCENARIOS: dict[str, dict] = {
    "broad_equity_selloff": {
        "description": "2008/2020-style broad risk-off: equities and crypto both sell off hard, "
        "vol spikes. Round numbers, not calibrated to any specific historical event.",
        "equity_pct": -0.20, "crypto_pct": -0.40, "vol_shock_pct": 0.50, "sector_overrides": {},
    },
    "crypto_specific_crash": {
        "description": "FTX-style contained crypto event: crypto craters, broad equities barely move, "
        "vol rises only modestly (a crypto-specific, not systemic, shock).",
        "equity_pct": -0.05, "crypto_pct": -0.50, "vol_shock_pct": 0.20, "sector_overrides": {},
    },
    "energy_shock": {
        "description": "Oil-price shock: Energy sector hit hard, broader market only modestly, "
        "vol rises moderately. Tests whether the book's real Energy concentration "
        "(alpha-signal-lab's CVX/EOG/MPC/PSX/VLO tilt) is actually a risk driver.",
        "equity_pct": -0.05, "crypto_pct": -0.05, "vol_shock_pct": 0.15,
        "sector_overrides": {"Energy": -0.30},
    },
    "rate_shock_financials": {
        "description": "Sharp rate move stressing Financials specifically (credit/duration concerns), "
        "broad equities down moderately, vol up meaningfully.",
        "equity_pct": -0.10, "crypto_pct": -0.10, "vol_shock_pct": 0.30,
        "sector_overrides": {"Financials": -0.20},
    },
}


@dataclass
class ScenarioPnL:
    scenario_name: str
    total_pnl: float
    linear_only_pnl: float  # what the rest of this project's delta-only proxy would have estimated
    convexity_correction: float  # total - linear_only, isolates gamma+vega contribution
    by_position: dict[str, float] = field(default_factory=dict)


def _equity_shock(p: Position, scenario: dict) -> float:
    sector = sector_of(p.asset)
    overrides = scenario.get("sector_overrides", {})
    return overrides.get(sector, scenario["equity_pct"])


def reprice_position(p: Position, scenario: dict) -> tuple[float, float]:
    """Returns (full_pnl, linear_only_pnl) for one position under `scenario`."""
    if p.market_value is None:
        return 0.0, 0.0

    if p.asset_class == AssetClass.CRYPTO:
        pnl = p.market_value * scenario["crypto_pct"]
        return pnl, pnl

    if p.asset_class == AssetClass.EQUITY:
        pnl = p.market_value * _equity_shock(p, scenario)
        return pnl, pnl

    if p.asset_class == AssetClass.OPTION:
        if not p.greeks or p.price is None:
            return 0.0, 0.0
        spot = p.price
        delta = p.greeks.get("delta", 0.0)
        gamma = p.greeks.get("gamma", 0.0)
        vega = p.greeks.get("vega", 0.0)
        entry_iv = p.extra.get("entry_iv", 0.0)

        d_s = spot * scenario["equity_pct"]  # SPY, an equity index -- no sector shock applies
        d_sigma = entry_iv * scenario["vol_shock_pct"]  # relative IV shock, in decimal vol

        linear_pnl = p.quantity * delta * d_s
        full_pnl = linear_pnl + p.quantity * (0.5 * gamma * d_s**2 + vega * d_sigma)
        return full_pnl, linear_pnl

    return 0.0, 0.0  # SYNTHETIC: no real market, no scenario applies


def run_scenario(scenario_name: str, positions: list[Position]) -> ScenarioPnL:
    scenario = HYPOTHETICAL_SCENARIOS[scenario_name]
    by_position: dict[str, float] = {}
    total = 0.0
    linear_total = 0.0

    for p in positions:
        full_pnl, linear_pnl = reprice_position(p, scenario)
        by_position[f"{p.strategy}/{p.asset}"] = full_pnl
        total += full_pnl
        linear_total += linear_pnl

    return ScenarioPnL(
        scenario_name=scenario_name, total_pnl=total, linear_only_pnl=linear_total,
        convexity_correction=total - linear_total, by_position=by_position,
    )
