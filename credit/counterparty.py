"""Disclosed counterparty default-probability assumptions.

Public credit spread / CDS-proxy data is not freely available for Alpaca,
Binance, Coinbase, or Kraken specifically. Rather than pretend otherwise,
RiskDesk assigns each counterparty a literature-informed probability-of-
default (PD) *tier*, stated here as an explicit modeling assumption -- not
derived from any live credit market. This is a stated limitation (see
README), and future credit-risk/CVA work will build directly on these
numbers, so any change here should be reasoned about the same way a change
to a VaR confidence level would be: a disclosed judgment call, not a
calibration.

Tiering rationale (annualized PD, illustrative order of magnitude):
  - Alpaca: US SEC/FINRA-registered broker-dealer, SIPC member, regulated
    custody of cash/securities. Treated as investment-grade-adjacent
    counterparty risk -- comparable in spirit to a regulated prime broker.
  - Binance, Coinbase, Kraken: centralized crypto exchanges. Materially
    higher tier than a regulated broker-dealer given the sector's default
    history (FTX 2022 being the reference event this project's planned
    historical-scenario replay will cover). Coinbase is a US publicly-listed,
    more heavily regulated exchange than Binance/Kraken, so it is placed one
    notch better within the "crypto exchange" band.
"""

from __future__ import annotations

from dataclasses import dataclass

from connectors.schema import Counterparty


@dataclass(frozen=True)
class PDTier:
    tier: str
    annualized_pd: float  # illustrative, disclosed assumption -- not market-implied
    rationale: str


COUNTERPARTY_PD: dict[Counterparty, PDTier] = {
    Counterparty.ALPACA: PDTier(
        tier="regulated_broker_dealer",
        annualized_pd=0.0010,
        rationale="SEC/FINRA-registered US broker-dealer, SIPC member.",
    ),
    Counterparty.COINBASE: PDTier(
        tier="regulated_exchange_crypto",
        annualized_pd=0.0100,
        rationale="US publicly-listed, more heavily regulated crypto exchange.",
    ),
    Counterparty.KRAKEN: PDTier(
        tier="offshore_exchange_crypto",
        annualized_pd=0.0200,
        rationale="Established but less regulated than Coinbase; no US public listing.",
    ),
    Counterparty.BINANCE: PDTier(
        tier="offshore_exchange_crypto",
        annualized_pd=0.0200,
        rationale="Established but less regulated than Coinbase; no US public listing.",
    ),
    Counterparty.NONE: PDTier(
        tier="no_venue",
        annualized_pd=0.0,
        rationale="No live counterparty (e.g. market-data-sourced backtest position, not traded).",
    ),
}
