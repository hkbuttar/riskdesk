from connectors.schema import Counterparty
from credit.counterparty import COUNTERPARTY_PD


def test_every_counterparty_has_a_disclosed_pd_tier():
    for cp in Counterparty:
        assert cp in COUNTERPARTY_PD
        tier = COUNTERPARTY_PD[cp]
        assert 0.0 <= tier.annualized_pd <= 1.0
        assert tier.rationale


def test_crypto_exchanges_rank_riskier_than_regulated_broker():
    broker = COUNTERPARTY_PD[Counterparty.ALPACA].annualized_pd
    for cp in (Counterparty.BINANCE, Counterparty.COINBASE, Counterparty.KRAKEN):
        assert COUNTERPARTY_PD[cp].annualized_pd > broker
