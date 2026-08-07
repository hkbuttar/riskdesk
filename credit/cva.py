"""Simplified Credit Valuation Adjustment (CVA): expected loss from
counterparty default, weighted by exposure and the disclosed PD tiers in
credit/counterparty.py.

Scoping distinction, stated precisely rather than left implicit: "CVA" in
derivatives pricing usually refers to a bilateral swap counterparty's
time-varying expected positive exposure, discounted over the life of the
trade. What's actually relevant here is closer to CUSTODIAL/SETTLEMENT
counterparty risk -- the risk that a broker or exchange HOLDING this
project's assets (Alpaca, Binance, Coinbase, Kraken) becomes insolvent and
those assets are lost or frozen, the FTX scenario. This module reuses the
term "CVA" because that's the plan's own framing and the underlying
expected-loss formula is the same (Exposure x PD x LGD), but the exposure
measure is a static current net position, not a discounted expected
exposure profile.

    CVA_c = |net_exposure_c| * PD_c * LGD

`|net_exposure_c|` (not gross): in a custodial default, what's actually at
risk is the net asset value held at that venue -- an offsetting long/short
pair at the same broker doesn't put 2x the gross notional at risk in an
insolvency, since the account nets to one value. This is a disclosed
simplification (real custody arrangements can be more complex --
segregation, margin posted elsewhere, partial netting rules that vary by
jurisdiction and venue) not a claim about actual legal recovery mechanics.

`LGD` (loss given default, i.e. 1 - recovery rate): no real recovery data
exists for any of these counterparties either, so a single disclosed
assumption (default 60%, a common simplified Basel-style figure for
unsecured exposure) is used uniformly across all counterparties -- not
because 60% is precisely right for a crypto exchange vs. a regulated
broker-dealer, but because inventing a per-counterparty recovery rate with
no supporting data would be a more confident-sounding but no more accurate
number.
"""

from __future__ import annotations

from dataclasses import dataclass

from aggregation.rollup import RollupBucket, by_counterparty
from connectors.schema import Counterparty, Position
from credit.counterparty import COUNTERPARTY_PD

DEFAULT_LGD = 0.60


@dataclass
class CVAResult:
    total_cva: float
    lgd_used: float
    by_counterparty: dict[str, dict[str, float]]  # {counterparty: {exposure, pd, cva}}


def compute_cva(positions: list[Position], lgd: float = DEFAULT_LGD) -> CVAResult:
    buckets: dict[str, RollupBucket] = by_counterparty(positions)
    detail: dict[str, dict[str, float]] = {}
    total = 0.0

    for key, bucket in buckets.items():
        counterparty = Counterparty(key)
        pd_tier = COUNTERPARTY_PD[counterparty]
        exposure = abs(bucket.net_market_value)
        cva = exposure * pd_tier.annualized_pd * lgd
        detail[key] = {
            "net_exposure": bucket.net_market_value,
            "annualized_pd": pd_tier.annualized_pd,
            "cva": cva,
        }
        total += cva

    return CVAResult(total_cva=total, lgd_used=lgd, by_counterparty=detail)
