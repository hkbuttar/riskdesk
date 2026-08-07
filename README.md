# RiskDesk — Portfolio Risk & Stress-Testing Capstone
Portfolio risk aggregation across six strategies. VaR/CVaR, DCC-GARCH correlation, factor decomposition, regime-conditional models, historical + reverse stress testing, EVT tail risk, counterparty credit risk, and live monitoring. Observable Framework dashboard. The capstone risk layer.

## Status

**Environment & data acquisition — done. Position & exposure aggregation — done.** Risk measures, correlation, regime models, factor decomposition, stress testing, credit risk, tail risk, liquidity, attribution, live monitoring, backend, and frontend are not yet started.

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

## Position & Exposure Aggregation (`aggregation/`)

Three modules, run end-to-end via `python -m aggregation.run`:

- **`pricing.py`** — attaches a point-in-time price to each position via Yahoo Finance, looked up as of the position's *own* `as_of` date, not "today." This matters concretely: pairtrade-lab-1's stand-in is dated 2024-04-25, so its AXP/WFC shares are priced at their 2024-04-25 close, not blended with today's price. Live positions (alpha-signal-lab, bookmaker's binance_real run, voledge's stand-in) have `as_of = today`, so this collapses to "current price" for them. `SYMBOL_MAP` explicitly translates internal tickers that don't match Yahoo's convention (`BTCUSDT` → `BTC-USD`); failures return `price=None` with a disclosed reason, never a guess.
- **`valuation.py`** — computes `market_value`. Equity/crypto: `quantity * price`. Options (voledge): no dollar price is available from that connector (it only carries the vol-surface edge, not a bid/ask), so market_value is the plan's own prescription — **delta-equivalent exposure** = `quantity * delta * underlying_spot` — with full Greeks left untouched on the position for later Greek-level aggregation. Synthetic positions (bookmaker, when a synthetic rather than binance_real run is picked up) have no real venue and are left unpriced, disclosed, not zeroed.
- **`rollup.py`** — rolls up to portfolio, strategy, asset-class, and counterparty level, both **net** (signed, shorts subtract) and **gross** (absolute value summed) exposure. Unpriced positions are excluded from both sums but counted and named separately (`n_unpriced`, `unpriced_assets`) so a rollup total can never silently understate exposure without saying so.

As of this writing, all 33 positions price successfully (`python -m aggregation.run`):

| Level | Net | Gross |
|---|---|---|
| **Portfolio** | -2,670 | 706,602 |
| alpha-signal-lab | 1,117 | 48,911 |
| bookmaker | 2,849 | 2,849 |
| pairtrade-lab-1 | -8,839 | 652,639 |
| voledge | 2,202 | 2,202 |

**Honest finding:** pairtrade-lab-1's gross exposure (~$653k) dwarfs every other strategy's and dominates the whole portfolio total. This isn't a bug — it's a real consequence of each source repo starting from a wildly different capital base (pairtrade-lab-1's backtest starts with $1M cash; alpha-signal-lab's live paper account has ~$100k equity) that nothing in the six source repos normalizes for. A true cross-strategy risk view needs either capital-normalized position sizing or an explicit disclosed per-strategy allocation weight before net/gross numbers are comparable — flagged here as a limitation, not fixed by inventing a weighting scheme unprompted.

### Limitations

- Aggregation blends position sizes across strategies with different starting capital, unnormalized (see finding above) — net/gross totals are directionally informative, not yet apples-to-apples across strategies.
- Delta-equivalent option exposure ignores gamma/vega/theta risk entirely (by design, per the plan) — full Greek aggregation is separate future work.
- Pricing depends on Yahoo Finance availability; a network failure or delisted/unusual ticker leaves a position unpriced rather than fabricating a value, but that means portfolio totals can shift based on data availability, not just market moves.

### Tests

`pytest` (`tests/test_aggregation.py`) — unit tests use constructed positions with a stubbed price fetch for determinism (real market prices change daily); one end-to-end test runs against real connectors and real Yahoo Finance prices, asserting structural invariants only. 14 tests passing total across the project.
