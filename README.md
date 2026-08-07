# RiskDesk — Portfolio Risk & Stress-Testing Capstone
Portfolio risk aggregation across six strategies. VaR/CVaR, DCC-GARCH correlation, factor decomposition, regime-conditional models, historical + reverse stress testing, EVT tail risk, counterparty credit risk, and live monitoring. Observable Framework dashboard. The capstone risk layer.

## Status

**Environment & data acquisition — done.** Position/exposure aggregation, risk measures, correlation, regime models, factor decomposition, stress testing, credit risk, tail risk, liquidity, attribution, live monitoring, backend, and frontend are not yet started.

## Environment & Data Acquisition

### Environment

Python 3.13 (not 3.14 — chosen deliberately: `arch` and `cvxpy` wheel availability is a real risk on a Python release this new, and 3.13 is what the sibling repos' own environments implicitly target). Dependencies in `requirements.txt`: numpy, pandas, scipy, statsmodels, arch (GARCH/DCC-GARCH), scikit-learn, cvxpy, fastapi, psycopg, pyarrow. All installed and import-clean in `.venv`.

### Connector layer (`connectors/`)

Six adapters, one per sibling repo, each normalizing to a common `Position`/`SourceMeta` schema (`connectors/schema.py`). Every connector reads real, currently-live external state — no source repo was modified, and no connector fabricates data it doesn't have. Credentials are read from each sibling repo's *own* `.env` (via `connectors/_env.py`, `RISKDESK_SIBLINGS_ROOT`), not duplicated into RiskDesk's environment.

Run `python -m connectors.registry` for a live summary. As of this writing:

| Repo | What's actually read | Positions contributed | Why |
|---|---|---|---|
| **alpha-signal-lab** | `portfolio_snapshots` (Postgres, Render-hosted, live) | **10** (real, live paper positions) | The only sibling repo that actually trades live (paper), via a daily GitHub Actions scheduler. |
| **streamalpha** | `anomalies` table (Postgres) | 0, by design | Real-time anomaly-detection pipeline — has no Position/Portfolio concept anywhere in its codebase. Exposed separately via `fetch_recent_anomalies()` as a risk-factor signal for later steps (regime classification, live monitoring), not as a position. |
| **bookmaker** | `bookmaker.db::simulation_runs` (SQLite) | **1** (real, `data_source="binance_real"`) | Ran `backtest/binance_backtest.py::run_binance_backtest` (bookmaker's own, unmodified code) against its committed captured Binance BTCUSDT L1 sample and persisted the result — the same real-data path bookmaker's own tests exercise, but not wired into its `/simulate` API (`DATA_SOURCES = ("synthetic",)` there). Preferred over a synthetic run when both exist. |
| **execedge** | n/a — no results persisted | 0, by design | Execution-cost benchmark (TWAP/VWAP/AC/RL), not a portfolio — its unit of output is a single parent-order cost report, not a standing position. |
| **pairtrade-lab-1** | `connectors/standins/pairtrade_lab_1_positions.json` (generated stand-in) | **2** (disclosed backtest stand-in) | Re-ran pairtrade-lab-1's own, unmodified `backtest.engine.run_backtest()` over 2023-01-03..2024-12-31 (its own venv, its own cached prices) and captured the final portfolio state via a non-invasive monkeypatch — no source file was edited. The strategy was flat at the window's end; the snapshot is dated to the last day it actually held a position (2024-04-25), disclosed as such. Note: `/Users/hkbuttar/pairtrade-lab` (no `-1`) is a dead, empty, non-git scaffold — not the real repo. |
| **voledge** | `connectors/standins/voledge_positions.json` (generated stand-in) | **20** (disclosed stand-in, top 20 of 548 by signal edge) | voledge has no "position held over time" concept at all — its own strategy is point-in-time (which contracts look rich/cheap right now). Ran voledge's own, unmodified `strategy.signal.compute_signal()` against the REAL, current SPY options chain (live yfinance/Alpaca) and its own fitted vol surface, then computed each flagged contract's real Black-Scholes Greeks via `greeks/analytical.py`. |

**Honest finding:** all six connectors are fully wired and every non-empty one reads real data — no fabricated numbers anywhere. Two repos (streamalpha, execedge) correctly contribute zero positions by design (no position concept in their own codebases). The other four (alpha-signal-lab, bookmaker, pairtrade-lab-1, voledge) each contribute real positions, three of them via a disclosed backtest/signal stand-in rather than a live feed — where a prior project never ran live, its most recent backtest's simulated positions are used as a disclosed stand-in. Regenerate a stand-in by re-running the corresponding script in that repo's own venv (see each connector module's docstring) — this was a one-off snapshot, not a live pipeline.

Where a connector can't produce a position (no data, no concept, connection failure), it returns an empty list plus a `SourceMeta.notes` string that says exactly why — never a silent empty result.

**Deliberate scope boundary:** connectors return `quantity` but leave `price`/`market_value` as `None`. Positions carry no per-ticker price in their source schemas (e.g. alpha-signal-lab's `positions_json` is just `{ticker: shares}`); attaching a point-in-time price is a valuation concern that belongs to the aggregation layer, shared across all six adapters, not duplicated per-connector.

### Counterparty credit data (`credit/counterparty.py`)

Public credit-spread/CDS-proxy data isn't freely available for Alpaca, Binance, Coinbase, or Kraken specifically. Each is assigned a disclosed, literature-informed probability-of-default tier (Alpaca as a regulated US broker-dealer at the low end; Binance/Kraken as less-regulated offshore exchanges at the high end; Coinbase in between as a regulated, US-listed exchange) — stated explicitly as a modeling assumption, not market-implied. This feeds the future CVA (credit valuation adjustment) calculation.

### Limitations

- The three stand-in snapshots (bookmaker's binance_real run, pairtrade-lab-1's and voledge's `connectors/standins/*.json`) are frozen at generation time, not live — they need to be regenerated by re-running each source script to stay current.
- pairtrade-lab-1's stand-in reflects a 2023-01-03..2024-12-31 backtest window (cached price data doesn't extend past 2024-12-31), not "today."
- voledge's stand-in keeps only the top 20 of 548 signal-flagged contracts by edge magnitude, a disclosed cap for aggregation practicality.
- Counterparty PD tiers are assumption-based, not market-implied (see above).
- `price`/`market_value` are not yet populated on any `Position` — the aggregation layer's job.

### Tests

`pytest` (`tests/test_connectors.py`, `tests/test_counterparty.py`) — 6 passing. Tests assert schema invariants and the "no positions from streamalpha/execedge" design constraint, not specific live values, since the underlying data (real Postgres/files) changes over time.
