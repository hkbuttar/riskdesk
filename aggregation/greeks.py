"""Portfolio-level Greek aggregation, and a focused, honest demonstration
of what the linear delta-equivalent treatment used everywhere else in this
project (valuation.py's market_value, risk_measures/returns.py's P&L proxy,
factor_model's regression) actually leaves out: gamma convexity.

Scope boundary, deliberate: this module aggregates the OPTIONS BOOK's own
Greeks and quantifies its own convexity correction against a hypothetical
underlying move. It does not reprice the whole portfolio (equities + crypto
+ options together) under a shock -- that is full stress testing, a
separate, larger piece of future work. What's here is self-contained and
honest on its own: "how wrong is the delta-only P&L estimate for the
options book specifically, once the underlying moves far enough for
convexity to matter."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from connectors.schema import AssetClass, Position

DEFAULT_STRESS_MOVES = (-0.10, -0.05, -0.02, 0.02, 0.05, 0.10)


@dataclass
class PortfolioGreeks:
    net_delta_shares: float  # sum(quantity * delta) -- share-equivalent exposure
    net_delta_dollars: float  # sum(quantity * delta * spot) -- matches valuation.py's option market_value
    net_gamma_shares: float  # sum(quantity * gamma)
    net_vega: float  # sum(quantity * vega) -- $ P&L per 1.00 (100-vol-point) IV move,
    # matching voledge's own raw vega convention (greeks/analytical.py's docstring:
    # "vega per 1.00 = 100 vol points"); divide by 100 for $ per single vol point.
    net_theta: float  # sum(quantity * theta) -- $ P&L per day, time decay
    net_rho: float
    n_option_positions: int
    by_position: dict[str, dict[str, float]] = field(default_factory=dict)


def aggregate_greeks(positions: list[Position]) -> PortfolioGreeks:
    net_delta_shares = net_delta_dollars = net_gamma_shares = net_vega = net_theta = net_rho = 0.0
    by_position: dict[str, dict[str, float]] = {}
    n = 0

    for p in positions:
        if p.asset_class != AssetClass.OPTION or not p.greeks:
            continue
        n += 1
        spot = p.price or 0.0
        delta = p.greeks.get("delta", 0.0)
        gamma = p.greeks.get("gamma", 0.0)
        vega = p.greeks.get("vega", 0.0)
        theta = p.greeks.get("theta", 0.0)
        rho = p.greeks.get("rho", 0.0)

        pos_delta_shares = p.quantity * delta
        pos_delta_dollars = p.quantity * delta * spot
        pos_gamma_shares = p.quantity * gamma
        pos_vega = p.quantity * vega
        pos_theta = p.quantity * theta
        pos_rho = p.quantity * rho

        net_delta_shares += pos_delta_shares
        net_delta_dollars += pos_delta_dollars
        net_gamma_shares += pos_gamma_shares
        net_vega += pos_vega
        net_theta += pos_theta
        net_rho += pos_rho

        by_position[f"{p.strategy}/{p.asset}"] = {
            "delta_shares": pos_delta_shares, "delta_dollars": pos_delta_dollars,
            "gamma_shares": pos_gamma_shares, "vega": pos_vega, "theta": pos_theta, "rho": pos_rho,
        }

    return PortfolioGreeks(
        net_delta_shares=net_delta_shares, net_delta_dollars=net_delta_dollars,
        net_gamma_shares=net_gamma_shares, net_vega=net_vega, net_theta=net_theta, net_rho=net_rho,
        n_option_positions=n, by_position=by_position,
    )


@dataclass
class ConvexityRow:
    move_pct: float
    linear_pnl: float  # delta-only estimate (same approximation used elsewhere in this project)
    quadratic_pnl: float  # delta + 0.5 * gamma * dS^2 (second-order Taylor expansion)
    gamma_correction: float  # quadratic - linear -- the convexity this project's other modules miss
    pct_understatement: float  # gamma_correction / |linear_pnl|, when linear_pnl != 0


def gamma_convexity_table(
    positions: list[Position], moves: tuple[float, ...] = DEFAULT_STRESS_MOVES
) -> list[ConvexityRow]:
    """For each hypothetical underlying % move, compares the delta-only P&L
    estimate (what risk_measures/returns.py and factor_model both implicitly
    use for options) against a delta+gamma second-order estimate, isolating
    the gamma contribution. Positions across different underlyings are
    summed together per move -- this book's options are all SPY, but the
    logic generalizes: dS is computed per-position from its own spot.
    """
    option_positions = [p for p in positions if p.asset_class == AssetClass.OPTION and p.greeks]
    rows: list[ConvexityRow] = []

    for move in moves:
        linear_pnl = 0.0
        quadratic_pnl = 0.0
        for p in option_positions:
            spot = p.price or 0.0
            delta = p.greeks.get("delta", 0.0)
            gamma = p.greeks.get("gamma", 0.0)
            d_s = spot * move

            pos_linear = p.quantity * delta * d_s
            pos_quadratic = pos_linear + p.quantity * 0.5 * gamma * d_s**2

            linear_pnl += pos_linear
            quadratic_pnl += pos_quadratic

        gamma_correction = quadratic_pnl - linear_pnl
        pct = gamma_correction / abs(linear_pnl) if linear_pnl != 0 else float("nan")
        rows.append(ConvexityRow(move, linear_pnl, quadratic_pnl, gamma_correction, pct))

    return rows
