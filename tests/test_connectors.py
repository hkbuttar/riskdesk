"""Connector tests. These hit real sibling-repo state (live Postgres for
alpha-signal-lab/streamalpha, local files for the rest) rather than mocks:
the whole point of the connector layer is correctly reading actual external
schemas, so a mocked test would validate nothing about schema drift. Each
test therefore only asserts invariants that hold regardless of what data
happens to be present right now (types, required disclosure fields), not
specific values.
"""

from __future__ import annotations

from connectors import registry
from connectors.schema import DataProvenance, Position, SourceMeta


def test_all_connectors_return_position_and_meta_lists():
    for repo, module in registry.CONNECTORS.items():
        positions, meta = module.fetch_positions()
        assert isinstance(positions, list), repo
        assert isinstance(meta, SourceMeta), repo
        assert meta.repo == repo
        assert isinstance(meta.notes, str) and meta.notes, f"{repo} must disclose notes"
        for p in positions:
            assert isinstance(p, Position)
            assert p.strategy == repo
            assert isinstance(p.quantity, float)
            assert isinstance(p.provenance, DataProvenance)


def test_streamalpha_and_execedge_never_contribute_positions():
    # Both repos have no position/portfolio concept by design (see their
    # connector docstrings) -- this must hold regardless of DB/file state.
    from connectors import execedge, streamalpha

    positions, _ = streamalpha.fetch_positions()
    assert positions == []
    positions, _ = execedge.fetch_positions()
    assert positions == []


def test_fetch_all_aggregates_across_connectors():
    positions, meta = registry.fetch_all()
    assert len(meta) == len(registry.CONNECTORS)
    assert isinstance(positions, list)
    # Every position must be attributable to one of the six known repos.
    for p in positions:
        assert p.strategy in registry.CONNECTORS


def test_alpha_signal_lab_positions_have_no_fabricated_market_value():
    # Connectors deliberately leave market_value/price as None --
    # see alpha_signal_lab.py docstring: pricing is the aggregation layer's job.
    from connectors import alpha_signal_lab

    positions, _ = alpha_signal_lab.fetch_positions()
    for p in positions:
        assert p.market_value is None
        assert p.price is None
