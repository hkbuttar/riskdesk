# RiskDesk — Portfolio Risk & Stress-Testing Capstone
Portfolio risk aggregation across six strategies. VaR/CVaR, DCC-GARCH correlation, factor decomposition, regime-conditional models, historical + reverse stress testing, EVT tail risk, counterparty credit risk, and live monitoring. Observable Framework dashboard. The capstone risk layer.

## Status

**Steps 1–21 are implemented** — the full risk-analytics layer, tested FastAPI backend, and Observable Framework frontend. Production deployment is the remaining step.

**`notebooks/research.ipynb`** is an interactive companion covering the same key findings below — every cell calls this project's real modules against real, live-fetched data (no separate, simplified reimplementation), and is executed end-to-end (`jupyter nbconvert --execute`) so its saved outputs are genuine current results, not placeholder code. Re-running it will regenerate fresh numbers, since the underlying data (live positions, current prices, current regime) changes day to day.

**This is the point in the project where scope explicitly widens beyond pure market risk.** Every module up to here answers some version of "what if the market moves against this book." Counterparty & credit risk, below, answers a genuinely different question: "what if the entity holding this book's assets — Alpaca, Binance, Coinbase, Kraken — disappears," independent of what the market does. Worth stating plainly since it's a different risk category, not a variation on the same one.

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
| **voledge** | `connectors/standins/voledge_positions.json` (generated stand-in) | **20** (disclosed stand-in, top 20 of 548 by signal edge) | voledge has no "position held over time" concept at all — its own strategy is point-in-time (which contracts look rich/cheap right now). Ran voledge's own, unmodified `strategy.signal.compute_signal()` against the REAL, current SPY options chain (live Alpaca) and its own fitted vol surface, then computed each flagged contract's real Black-Scholes Greeks via `greeks/analytical.py`. |

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

- **`pricing.py`** — attaches a point-in-time price to each position via Alpaca Market Data, looked up as of the position's *own* `as_of` date, not "today." This matters concretely: pairtrade-lab-1's stand-in is dated 2024-04-25, so its AXP/WFC shares are priced at their 2024-04-25 close, not blended with today's price. Live positions (alpha-signal-lab, bookmaker's binance_real run, voledge's stand-in) have `as_of = today`, so this collapses to "current price" for them. `SYMBOL_MAP` explicitly translates internal tickers that don't match Alpaca's canonical convention (`BTCUSDT` → `BTC-USD`); failures return `price=None` with a disclosed reason, never a guess.
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
- Pricing depends on Alpaca Market Data availability; a network failure or delisted/unusual ticker leaves a position unpriced rather than fabricating a value, but that means portfolio totals can shift based on data availability, not just market moves.
- Alpaca's available BTC/USD archive does not extend back to the 2020 COVID window. COVID replay therefore uses the equity factors Alpaca actually returns, and SPY/BTC DCC comparison is reported only for the 2022 and FTX windows; RiskDesk does not backfill the missing crypto history from a second provider.

### Tests

`pytest` (`tests/test_aggregation.py`) — unit tests use constructed positions with a stubbed price fetch for determinism (real market prices change daily); one end-to-end test runs against real connectors and real Alpaca Market Data prices, asserting structural invariants only.

## VaR/CVaR — Five Methods, Compared (`risk_measures/`)

- **`returns.py`** — for every position, its *risk factor* is the asset itself (equity/crypto) or, for voledge's options, `extra["underlying"]` (SPY) — since options are already carried as delta-equivalent dollar exposure (see aggregation/valuation.py), `market_value * underlying_return` approximates each option position's daily P&L to first order (delta-only, ignoring gamma/theta — a disclosed simplification, not a new one introduced here). Two years of daily history is pulled for every distinct risk factor via Alpaca Market Data; days where any held risk factor lacks a price (mostly equity-only weekends/holidays vs. crypto trading every day) are dropped rather than forward-filled. This produces one hypothetical daily portfolio $ P&L series — today's position sizes applied to each historical day's return — reused identically by all five VaR methods so the comparison is apples-to-apples.
- **`var.py`** — five methods, all working in loss space (positive $ = loss) at a given confidence:
  - **Historical simulation** — empirical quantile of the realized P&L series. No distributional assumption; only as good as the window's actual tail coverage.
  - **Parametric (variance-covariance)** — assumes normally distributed losses; fast, closed-form, and the textbook source of VaR's fat-tail blind spot.
  - **Monte Carlo** — simulates from a fitted multivariate normal over the *underlying risk factors'* returns (captures the real correlation structure across factors), then aggregates by today's dollar weights.
  - **Cornish-Fisher** — parametric, but corrects the normal quantile using the loss series's own sample skewness/kurtosis.
  - **EVT (Peaks-Over-Threshold)** — fits a Generalized Pareto Distribution (`scipy.stats.genpareto`) to losses exceeding a high threshold (default: 90th percentile of losses, a disclosed judgment call), the standard approach built specifically for tail estimation.

As of this writing (`python -m risk_measures.run`, 501 aligned trading days across 14 risk factors):

| Method | 95% VaR | 95% CVaR | 99% VaR | 99% CVaR |
|---|---|---|---|---|
| Historical simulation | 7,592 | 11,563 | 14,075 | 19,003 |
| Parametric (normal) | 8,119 | 10,199 | 11,511 | 13,197 |
| Monte Carlo | 8,063 | 10,176 | 11,528 | 13,263 |
| Cornish-Fisher | 8,483 | 13,986 | 17,133 | 23,775 |
| EVT (POT) | 7,249 | 12,576 | 13,721 | 25,934 |

**Honest finding — exactly where and why the five methods diverge:** at 95% confidence all five roughly agree (a ~17% spread). At 99% they diverge sharply (a ~49% spread): parametric-normal and Monte Carlo — both assuming normally distributed returns — bottom out around $11.5k, while historical simulation, Cornish-Fisher, and EVT all land well above $13.7k, up to $17.1k for Cornish-Fisher. The book's realized loss series has positive skew (0.48) and excess kurtosis (3.45) — fatter and more lopsided tails than a normal distribution — so the two methods that assume normality systematically understate deep-tail risk exactly where it matters most, a textbook, now-quantified illustration of parametric VaR's well-known weakness rather than an assumed one.

### Limitations

- The portfolio P&L series is a hypothetical historical replay (today's position sizes × historical factor returns), not a realized track record — it says nothing about how the book was actually sized on any past day.
- Delta-only options P&L (see above) misses gamma/theta effects that matter most exactly during the large moves VaR cares about.
- EVT's threshold choice (90th percentile) is a disclosed judgment call with real sensitivity — too low blends in non-tail behavior, too high leaves too few exceedances (this book's window gives 50) to fit a stable tail shape.
- All five methods share one two-year lookback window; none of them are regime-conditional yet — a calm-vs-stressed-period distinction is future work.

### Tests

`pytest` (`tests/test_var.py`) — each method is checked against synthetic data with a known/controllable distribution (e.g. recovering a hand-computed quantile, recovering injected Generalized Pareto tail parameters), not real market data, since hardcoded expected values against live prices would be flaky. One end-to-end test runs the full real pipeline and asserts structural invariants (CVaR ≥ VaR, no NaNs).

## Correlation & Covariance Estimation (`correlation/`)

- **`static.py`** — one Pearson correlation/covariance matrix over the whole lookback window. This is explicitly the same estimation approach behind alpha-signal-lab's kill-switch: a single number assumed to hold going forward, exposed as a real failure mode during the 2020 COVID window when correlations spiked well past what a calm-period sample implied. It's kept here as the deliberate baseline every other estimator is compared against, not as a recommended approach on its own.
- **`shrinkage.py`** — Ledoit-Wolf shrinkage (`sklearn.covariance.LedoitWolf`) toward a scaled-identity target, reported as a condition-number improvement (a direct, quantified before/after) rather than an assertion that shrinkage helps.
- **`dcc_garch.py`** — time-varying correlation via Engle's DCC-GARCH (2002). `arch` (the package used for the univariate GARCH stage) doesn't ship a ready multivariate DCC class, so this is a from-scratch two-stage estimator built on top of it: (1) fit a univariate GARCH(1,1) per risk factor via `arch.univariate.arch_model`, extract standardized residuals; (2) fit the DCC recursion `Q_t = (1-a-b)*Qbar + a*z_{t-1}z_{t-1}' + b*Q_{t-1}` by maximizing the concentrated Gaussian quasi-log-likelihood of the correlation stage (`scipy.optimize.minimize`), subject to the model's own stationarity constraint `a + b < 1`.

As of this writing (`python -m correlation.run`, same 14 risk factors, 501-day window):

- **Ledoit-Wolf**: shrinkage constant 0.048, improves the covariance matrix's condition number from 157 to 62 (2.5x better conditioned) — meaningful given 14 assets over ~500 observations is not a large ratio.
- **DCC-GARCH**: fitted persistence `a + b ≈ 0.93` (high persistence, typical for daily equity/crypto data) — see the top-5 divergence table below.

| Pair | Static | DCC (latest) | DCC (min .. max over window) |
|---|---|---|---|
| CVX–SPY | +0.24 | +0.02 | -0.04 .. +0.34 |
| EOG–SPY | +0.23 | +0.02 | -0.03 .. +0.37 |
| AXP–EOG | +0.18 | +0.02 | +0.02 .. +0.37 |
| AXP–CVX | +0.22 | +0.06 | +0.05 .. +0.37 |
| PSX–SPY | +0.36 | +0.21 | +0.13 .. +0.50 |

**Honest finding:** static correlation is, by construction, one fixed number for the whole window — it cannot distinguish "this pair moved together in March and decoupled in July" from "this pair had a stable, moderate correlation the whole time." DCC-GARCH's min..max range on every one of the top-5 divergent pairs above is wide enough (e.g. CVX–SPY spans -0.04 to +0.34) that the static estimate is a genuinely misleading summary of the relationship — not just imprecise, but capturing a blended average that no single day in the window actually looked like.

### Limitations

- Full validation of DCC-GARCH's value ("does it capture a correlation spike static estimation misses") needs a real historical stress window (e.g. a sharp drawdown) to check against — that's future work once the historical-scenario-replay module exists. What's shown here (real time-variation over a calm-to-moderate 2-year window) is suggestive, not a stress-period validation.
- Every asset is fit with the same GARCH(1,1) + normal-distribution specification — no per-asset model selection (e.g. a fatter-tailed innovation distribution, higher-order GARCH terms).
- The two-stage DCC estimator (GARCH stage, then a concentrated-likelihood correlation stage) is the standard practical approach, not a full joint MLE — it's the right trade-off for a 14-asset system but is a disclosed approximation, not the theoretically optimal estimator.
- Ledoit-Wolf's shrinkage target is `sklearn`'s default (scaled identity); other structured targets (e.g. a single-factor market model) are a possible refinement, not implemented here.

### Tests

`pytest` (`tests/test_correlation.py`) — static correlation and Ledoit-Wolf are checked against synthetic data with known structure (a specific injected correlation, a deliberately ill-conditioned small sample). DCC-GARCH is checked for well-formedness (valid correlation matrices, the stationarity constraint honored) and, via a synthetic calm-then-correlated regime switch, for correctly tracking the *direction* of a real correlation change that a static estimate cannot see. One end-to-end test runs DCC-GARCH on the real book.

## Regime Classification (`regime/`)

Two independent methods, both classifying on SPY (a broad market proxy, not this project's own portfolio P&L — a regime label should describe market conditions the book is exposed to, not be circularly defined by the book's own returns):

- **`volatility_tercile.py`** — reuses execedge's own methodology directly (`data/regimes.py` in that sibling repo) rather than inventing a new regime definition: rolling realized volatility of log returns, then a **static** tercile split (`Series.quantile(1/3)` / `Series.quantile(2/3)` computed once over the whole available history — not `pd.qcut`, not a per-bar expanding recompute) labels every day `calm` / `normal` / `volatile`. Two disclosed adaptations from the original: execedge's window (24 hourly bars) and label name ("volatile", not the plan's own "stressed") are both kept as-is for genuine continuity; only the window/annualization (`window=21` trading days, `periods_per_year=252`) changed, since this project uses daily bars where execedge used hourly.
- **`hmm_regime.py`** — the plan's "and/or a simple hidden Markov model for a probabilistic regime label instead of a hard cutoff": a 3-state Markov-switching mean/variance model (`statsmodels.tsa.regime_switching.markov_regression.MarkovRegression`) fit directly on daily log returns (not on the tercile method's already-smoothed rolling-vol series, which would double-smooth and blunt how fast a real regime change could be detected). `statsmodels` doesn't order its regimes by meaning, so the three fitted regimes are relabeled calm/normal/volatile by sorting on fitted variance ascending, matching the tercile method's ordering convention so the two are directly comparable. Output is a full daily probability distribution over all three regimes, not just a hard label. A single default-start MLE fit converged poorly on real SPY data (one regime absorbing almost no days); `search_reps=50` (multiple random restarts, keep the best) fixed this. That introduced a second, real issue: `statsmodels`' random restarts have no seed control and occasionally land on a degenerate starting point themselves, raising a `numpy.linalg.LinAlgError` from inside the EM step rather than just fitting badly — hit directly during test runs on real SPY data, non-deterministically. Fixed with a retry ladder (`search_reps=50` → `10` → `0`, catching `LinAlgError` at each step) rather than letting one unlucky restart crash the whole classification — both issues disclosed as real fitting problems encountered and resolved, not hypothetical caveats.

As of this writing (`python -m regime.run`, SPY, 2-year daily window):

| | calm | normal | volatile |
|---|---|---|---|
| Tercile (by construction) | 161 (32%) | 159 (32%) | 161 (32%) |
| HMM (data-driven) | 317 (63%) | 171 (34%) | 13 (3%) |

**Honest finding:** the two methods agree on only 43.2% of days. This isn't a bug in either — it's a real, structural difference in what each method is built to do. Terciles force an exactly-balanced three-way split by construction, regardless of whether the underlying volatility distribution actually clusters that way. The HMM has no such constraint: it found volatility on SPY over this window is genuinely dominated by a low-variance regime (63% of days), with true high-variance days rare (3%) and sharply distinguished (fitted volatile-regime variance is ~13x the calm regime's, with a negative mean return, a classic risk-off signature) rather than evenly spread across a forced middle third. Which one is "right" depends on what the classification is for — terciles guarantee enough samples in every bucket for later per-regime statistics; the HMM better reflects the market's actual clustering, at the cost of a genuinely rare "volatile" bucket that may not have enough days for reliable regime-conditional statistics downstream.

### Limitations

- Validating either method against a real, known stress period (e.g. a sharp historical drawdown) is future work once historical-scenario-replay exists — right now both are fit on a 2-year window that never contained a genuine crisis, so "volatile" here means "volatile relative to a fairly calm two years," not "crisis-level."
- The HMM's rare volatile regime (13 of 501 days) may not carry enough observations for stable regime-conditional risk estimates later — a real, disclosed tension between the two methods' designs, not resolved by picking one over the other here.
- Both methods classify the market via SPY alone; a book with material crypto or options exposure (as this one has) may experience regime shifts SPY doesn't capture — a single reference series is a disclosed simplification.

### Tests

`pytest` (`tests/test_regime.py`) — tercile logic is checked against synthetic series with known/controllable structure (a series with clean calm/normal/volatile segments by construction). The HMM is checked on a synthetic two-regime series with a clean variance separation, where recovering the injected regime is realistic to assert exactly; the real-SPY end-to-end test only checks structural invariants (valid probability distributions, label sets), since exact regime counts on live market data would be flaky.

## Regime-Conditional Risk Models (`regime/conditional.py`)

Re-estimates `risk_measures/var.py` (historical simulation, parametric, Cornish-Fisher) and `correlation/static.py` separately for each regime, instead of pooling every historical day into one static model regardless of market condition — the direct, quantified version of the question correlation.py's static-vs-DCC comparison raised only qualitatively.

**Regime partition choice, disclosed:** this uses `volatility_tercile.py`'s labels, not the HMM's. The HMM's data-driven "volatile" regime had only 13 days over the 2-year window — nowhere near enough to fit a stable per-regime VaR or correlation estimate. Terciles guarantee a usable sample (~160 days) in every bucket by construction, which per-regime estimation actually needs; a regime with fewer than 30 days is skipped entirely (disclosed via a note), not forced through a method that would just be overfitting noise.

As of this writing (`python -m regime.run_conditional`, 95% confidence, same 14-risk-factor book):

| Method | Pooled | Calm | Normal | Volatile |
|---|---|---|---|---|
| Historical simulation | 7,592 | 7,964 | 6,111 | 7,601 |
| Parametric (normal) | 8,120 | 7,432 | 8,439 | 8,605 |
| Cornish-Fisher | 8,483 | 7,345 | 8,652 | 9,669 |

**Honest finding — this book's risk is not cleanly separated by SPY's own volatility regime.** The calm-vs-volatile VaR ratio is only 1.0x–1.3x across methods, far short of a dramatic split. This is a real, useful negative result, not a null one: it means this specific aggregated book (energy-sector equities, a pairs-trade book, BTC, and SPY options) isn't dominated by broad-market beta the way SPY's own regime would predict — consistent with the aggregation-layer finding that pairtrade-lab-1's idiosyncratic pairs exposure dwarfs everything else in the book. Where the regime split shows up more clearly is in the **tail**: historical-simulation CVaR is $9,625 in the calm regime vs. $13,158 in the volatile regime (a 1.4x gap, larger than the VaR gap) — the calm regime understates deep-tail risk more than it understates the VaR threshold itself.

**Correlation shift by regime** also doesn't move in the textbook direction: mean |conditional − pooled| correlation is *larger* in the calm regime (0.154) than the volatile regime (0.095). This is reported as-is rather than reinterpreted to fit the "stress increases correlation" prior — with only three ~160-day buckets from one 2-year window that never contained a real crisis, this is as plausibly sampling noise in a relatively idiosyncratic book as it is a genuine effect, and is flagged as exactly that rather than oversold as a finding.

**Live-monitoring readiness:** the module also demonstrates the mechanism Step 16-equivalent live monitoring will need — classify the most recent trading day's regime, then select that regime's conditional model as "active" (falling back to the pooled model, explicitly, if the current regime doesn't have enough historical days for its own estimate, rather than silently borrowing another regime's numbers).

### Limitations

- This book's weak regime separation (see finding above) may be specific to its current composition (energy/financials/pairs/crypto/options), not a general statement that regime-conditioning doesn't matter — a more SPY-beta-heavy book would likely show a starker split.
- Three ~160-day buckets from a single 2-year window is a small sample for correlation *differences* specifically (individual correlations are noisy even with ~160 points); the correlation-shift finding above is reported with that caveat attached, not smoothed over.
- No real historical stress window is in this data yet (same limitation as correlation.py and regime/README's tercile section) — "volatile" here still means "volatile relative to a fairly calm two years."

### Tests

`pytest` (`tests/test_conditional.py`) — alignment/partitioning logic (no forward-fill across unlabeled dates, correct bucket membership, the min-days skip rule) is checked exactly. VaR and correlation shift-detection are checked by injecting a real, controllable regime-dependent variance/correlation difference into synthetic data and confirming the conditional estimates recover the correct direction. One end-to-end test runs the full pipeline on the real book.

## Factor Risk Decomposition (`factor_model/`)

- **`factors.py`** — named factors: market (SPY), crypto (BTC-USD), and four sector factors. Sector *membership* (`SECTOR_TICKERS`) is copied verbatim from alpha-signal-lab's `config/universe.py`, per the plan's "sector exposures from alpha-signal-lab's tagging" — every equity ticker this book actually holds already falls inside one of alpha-signal-lab's four defined sectors (Technology, Healthcare, Financials, Energy), so no extension was needed. Sector *factor returns*, deliberately, come from real SPDR sector ETFs (XLK/XLV/XLF/XLE) rather than an average of this book's own held tickers — using the book's own returns as its own factor would be circular, a position partly "explaining" itself.
- **`regression.py`** — OLS regression (`statsmodels`) of a P&L series (portfolio- or strategy-level) on the named factor returns: a dollar loading per factor, t-stats/p-values, an intercept ("alpha" — P&L unexplained by any named factor), and R². Run once at the portfolio level and once per strategy — the per-strategy run is what makes "does pairtrade-lab-1's market-neutral book carry hidden market beta" a directly answerable, quantified question rather than an assertion.
- **`pca.py`** — PCA on the same 14 risk-factor returns, a model-free complement: it makes no hypothesis about what drives risk, just finds the actual directions of maximum variance, useful for checking whether the named-factor model is missing something.
- **`vega.py`** — net portfolio vega (voledge's options), reported separately from the return-based regression above since IV moves aren't spanned by the underlying's own price-return series — it doesn't belong in the same OLS model as the price factors.

As of this writing (`python -m factor_model.run`, 500 aligned trading days):

| | R² | SPY | Energy (XLE) | Financials (XLF) | Technology (XLK) | Healthcare (XLV) | BTC-USD |
|---|---|---|---|---|---|---|---|
| **Portfolio** | 0.087 | **-258,594 (p=0.003)** | 31,442 (p=0.05) | **116,113 (p=0.001)** | 30,802 | -22,370 | 13,030 |
| alpha-signal-lab | 0.716 | -3,417 | **26,036 (p<0.001)** | **6,497 (p=0.003)** | **-18,800 (p<0.001)** | **-10,679 (p<0.001)** | -303 |
| pairtrade-lab-1 | 0.062 | **-257,380 (p=0.003)** | 5,406 | **109,616 (p=0.002)** | 49,602 | -11,691 | 10,484 |
| bookmaker | 1.000* | ~0 | ~0 | ~0 | ~0 | ~0 | **2,849 (p<0.001)** |
| voledge | 1.000* | **2,203 (p<0.001)** | ~0 | ~0 | ~0 | ~0 | ~0 |

*(bold = statistically significant at p<0.05; bookmaker/voledge's R²=1.000 is a mechanical artifact, not a discovery — see below)*

**Honest finding — this is the capstone's central question, directly answered:** yes, pairtrade-lab-1's "market-neutral" pairs book carries substantial, statistically significant hidden market beta once aggregated with real position sizing — a highly significant **negative** SPY loading (t=-2.98, p=0.003) and a large, significant Financials sector loading (t=3.06, p=0.002). This is in fact the single largest driver of the whole *portfolio's* SPY exposure (-258,594) and Financials exposure (116,113) — pairtrade-lab-1 alone accounts for essentially all of it. A caveat that belongs right next to this finding, not hidden in a footnote below: pairtrade-lab-1's current book is only two positions (a short AXP / long WFC pair, per aggregation.py's stand-in snapshot), both in Financials — so this "hidden beta" is real and large for *this specific two-leg snapshot*, not evidence that pairs trading in general, or a fully diversified multi-pair version of this strategy, necessarily carries hidden beta. It's a genuine finding about the book as currently aggregated, not a general indictment of the strategy.

alpha-signal-lab, by contrast, shows a well-explained (R²=0.716), diversified factor profile matching its actual holdings directly: a large positive Energy loading (its biggest sector tilt: CVX/EOG/MPC/PSX/VLO), and negative Technology/Healthcare loadings that correctly reflect its short positions in those sectors (AVGO/INTC/NVDA short, AMGN short outweighing PFE long).

bookmaker and voledge's R²=1.000 is expected, not a finding: each is a single-underlying position whose proxied P&L is *defined* as `market_value × factor_return` (see risk_measures/returns.py) — an exact algebraic identity with its own risk factor, not a statistical discovery. Reported plainly rather than presented as a result on par with the other two.

**PCA**: 8 of 14 components are needed to explain 90% of variance — a fairly diffuse risk structure, consistent with a book spanning four sectors plus crypto plus options, not a low-dimensional one. PC1 (35% of variance) loads heavily on the Energy names plus SPY — a broad market/energy composite. PC3 (11%) loads almost entirely on AMGN and PFE — a clean, interpretable Healthcare-specific factor the sector-ETF regression's own Healthcare loading (not significant at the portfolio level) didn't surface as clearly, a concrete example of PCA catching something the named-factor model's aggregate view smoothed over.

**Vega**: net portfolio vega is 242.24 across voledge's 20 option positions — a modest net long-vol tilt, not currently large enough to be a dominant risk driver next to the equity-factor exposures above, but tracked separately for when Greek-level aggregation (Step 8-equivalent) builds on it.

### Limitations

- pairtrade-lab-1's "hidden beta" finding reflects its current 2-position stand-in snapshot specifically (see caveat above) — it should be re-checked once/if a fuller, multi-pair stand-in is generated.
- Sector factor returns (real ETFs) and sector *membership* (alpha-signal-lab's tagging) are both real and independent of this book's own positions, but the regression's R² at the portfolio level (0.087) is still fairly low — four sector ETFs plus SPY plus BTC-USD don't explain most of this specific book's day-to-day P&L variance, meaning a meaningful share of this book's risk is idiosyncratic / not captured by these named factors at all (PCA's 8-components-for-90% finding is the same fact from a different angle).
- bookmaker and voledge's perfect R² is a structural artifact of this project's single-underlying delta-equivalent P&L proxy (risk_measures/returns.py), not evidence those strategies are actually risk-free or fully explained — disclosed explicitly above, not silently included in a headline "everything is well-explained" claim.

### Tests

`pytest` (`tests/test_factor_model.py`) — regression and PCA are checked against synthetic data with a known/injected structure (a specific loading recovered from noisy data, a dominant correlated cluster recovered as PC1, and the exact bookmaker/voledge-style perfect-collinearity case reproduced deliberately). Sector tagging is checked against tickers known to be in alpha-signal-lab's own mapping. Vega aggregation is checked against constructed option/equity positions. One end-to-end test runs the full pipeline on the real book.

## Greeks & Options Risk Aggregation (`aggregation/greeks.py`)

**Scope boundary, deliberate:** this aggregates the options book's own Greeks and quantifies its own convexity against a hypothetical underlying move. It does not reprice the whole portfolio (equities + crypto + options together) under a shock — that's full stress testing, separate future work. What's here is self-contained: how wrong is the delta-only P&L estimate for the options book specifically, once the underlying moves far enough for gamma to matter.

- **`aggregate_greeks()`** — net delta (share- and dollar-equivalent), gamma, vega, theta, rho across every option position, plus a per-position breakdown.
- **`gamma_convexity_table()`** — for a range of hypothetical SPY moves (±2%, ±5%, ±10%), compares the **delta-only (linear)** P&L estimate — the exact approximation `risk_measures/returns.py` and `factor_model` both use for options everywhere else in this project — against a **delta+gamma (second-order Taylor)** estimate, isolating the pure gamma contribution.

As of this writing (`python -m aggregation.run_greeks`, 20 real option positions):

| | net delta | net gamma | net vega | net theta | net rho |
|---|---|---|---|---|---|
| Portfolio | 2.85 shares (\$2,203) | 0.485 | \$242 | -\$1,018/day | \$26 |

*(net vega is voledge's raw Black-Scholes vega, `dV/dσ` with σ in decimal — i.e. \$ P&L per 1.00 / 100-vol-point IV move, matching voledge's own documented convention; a 1-point (0.01) IV move is \$2.42, not \$242 — a units mix-up caught and fixed while building the stress-scenario module below, which needed to get this exactly right to size a vega shock correctly.)*

| SPY move | Linear (delta-only) | Quadratic (delta+gamma) | Gamma correction | % of linear |
|---|---|---|---|---|
| -10% | -\$220 | \$1,229 | +\$1,449 | +658% |
| -5% | -\$110 | \$252 | +\$362 | +329% |
| -2% | -\$44 | \$14 | +\$58 | +132% |
| +2% | \$44 | \$102 | +\$58 | +132% |
| +5% | \$110 | \$472 | +\$362 | +329% |
| +10% | \$220 | \$1,670 | +\$1,449 | +658% |

**Honest finding — this is a material, quantified problem with an approximation used throughout the rest of this project, not a minor footnote.** This options book is nearly delta-neutral (net delta of only 2.85 shares / \$2,203) but carries substantial positive net gamma (0.485). Because the linear term is close to zero by construction, the gamma correction *dominates* the total P&L at every move size shown — at a 10% SPY move, the delta-only estimate used everywhere else in this project (VaR, factor regression, the pooled/regime-conditional models) is off by nearly 7x, understating the true P&L. The correction scales with the square of the move, exactly as gamma convexity should — small and mostly ignorable at ±2%, dominant at ±10%. This directly quantifies a limitation flagged only qualitatively in earlier sections ("delta-only P&L misses gamma/theta effects that matter most exactly during the large moves VaR cares about," from the VaR section) — now with an actual number attached.

### Limitations

- This convexity check covers gamma only; theta (time decay) and vega (IV-move) effects are aggregated but not folded into the stress table above — a full options-aware stress scenario needs all three, future work alongside the broader stress-testing module.
- The gamma correction is computed per-position and summed; it does not account for correlation between the options book's convexity and the rest of the portfolio's linear moves under the same shock (e.g. does gamma P&L help or hurt exactly when the rest of the book is also losing money) — that cross-asset interaction is explicitly deferred to full stress testing, not attempted here.
- All 20 option positions share one underlying (SPY) in this book currently, so this table's "move" is unambiguous; a book with multiple option underlyings would need the convexity table broken out per underlying rather than summed together (the code supports it — each position's own spot is used — but the current book doesn't exercise that path).

### Tests

`pytest` (`tests/test_greeks.py`) — Greek aggregation is checked against constructed positions with known values (exact hand-computable sums). Convexity is checked for the textbook properties directly: zero gamma collapses quadratic to linear exactly, the correction scales with the square of the move size, and positive/negative gamma add/subtract value symmetrically in both directions. One end-to-end test runs on the real book.

## Historical Scenario Replay (`stress/historical.py`)

Applies real historical return windows to TODAY's actual position sizes — the same hypothetical-historical-P&L methodology `risk_measures/returns.py` already uses for VaR, narrowed to three specific, disclosed crisis windows instead of a rolling 2-year lookback. This is also where two "future work" callouts from earlier sections finally get resolved: correlation.py's and regime/conditional.py's own limitations both explicitly said "validating this against a real historical stress window is future work" — this module is that validation.

**Windows, disclosed judgment calls:**
- **COVID crash**: 2020-02-19 (S&P 500 pre-crash high) to 2020-03-23 (closing low).
- **2022 rate-hike bear market**: 2022-01-03 (2022 opening high) to 2022-10-12 (closing low) — a slower, longer drawdown, deliberately contrasted against COVID's sharp shock.
- **FTX collapse**: 2022-11-06 (the CoinDesk report that triggered the run) to 2022-11-14 — crypto-specific, not broad-equity.

**Lookahead discipline**: the VaR-breach validation below fits both the pooled and volatile-regime-conditional models *only* on data strictly before each window starts (`pre_start` dates going back to 2018/2019) — "would a model estimated from what was known before the crisis have bounded the crisis's actual losses," not "does a model fit on the crisis predict the crisis."

As of this writing (`python -m stress.run_historical`):

| Window | Total P&L | Worst day | Diversification erosion ratio | Pooled VaR breaches | Conditional VaR breaches |
|---|---|---|---|---|---|
| COVID crash (23 days) | +\$9,185 (+1.3%) | -\$19,301 | 0.97 | 6 (expected ~1.2) | 6 (expected ~1.2) |
| 2022 rate-hike (195 days) | +\$14,087 (+2.0%) | -\$26,722 | 1.00 | 8 (expected ~9.8) | 2 (expected ~9.8) |
| FTX collapse (4 days) | -\$15,977 (-2.3%) | -\$7,198 | 1.00 | 0 (expected ~0.2) | 0 (expected ~0.2) |

**Honest finding #1 — the portfolio's realized P&L is not a simple story, and it's driven by whichever strategy happens to have offsetting exposure, not genuine hedging design.** The book made money in both the COVID crash and the 2022 bear market — during COVID because pairtrade-lab-1's AXP/WFC pair gained \$24,228 while alpha-signal-lab lost \$13,426 (mostly offsetting, by coincidence of current position sizing, not by any strategy being designed to hedge another); during 2022, alpha-signal-lab actually profited (+\$19,858), plausibly because its live book currently holds short positions in AVGO/INTC/NVDA, and 2022 was a brutal year for exactly those names. The one window where the book actually lost money — FTX — is dominated almost entirely by pairtrade-lab-1 (-\$14,470 of the -\$15,977 total), not by bookmaker or any other crypto-touching strategy, because pairtrade-lab-1's stand-in position sizing happens to be far larger than everything else in the book (the same capital-base mismatch flagged back in the aggregation-layer README section).

**Honest finding #2 — regime-conditioning does NOT uniformly help, and this module reports that plainly rather than only showcasing wins.** Three different outcomes across three windows: in the COVID window, the volatile-regime-conditional VaR was breached exactly as often as the pooled model (6 of 23 days each) — a regime classifier fit on pre-2020 "volatile" days had never seen anything like COVID, so conditioning on it provided no edge for a genuinely novel shock. In the 2022 window, the pooled model was already reasonably well-calibrated (8 breaches vs. ~9.8 expected) while the conditional model was *overly* conservative (a much higher VaR, only 2 breaches — under-breaching, meaning it overstated risk for that window). Only the FTX window is too short (4 days) to draw any real conclusion either way. The honest summary: regime-conditioning is not a free upgrade over pooled VaR — its value depends on whether the crisis at hand resembles the "volatile" days the regime model was actually trained on, which is a real, disclosed limitation of the whole regime-conditional approach, not something the pooled-vs-conditional comparison in the regime/ section could show on its own without this out-of-sample test.

**Diversification erosion** stayed close to 1.0 in every window (0.97–1.00) — cross-strategy correlation neither meaningfully amplified nor reduced portfolio risk relative to a naive independence assumption during these three specific historical windows, for this specific book. This is a real, if modest, finding: it does not confirm the "diversification breaks down under stress" narrative that motivated the DCC-GARCH work — at least not for this book, in these three windows.

### Limitations

- Three windows, one small book — none of these findings generalize to "regime-conditioning helps/doesn't help" as a universal statement; they're specific, honest results for this specific aggregated book's current position sizing.
- The FTX window (4 trading days) is too short for any of the statistical comparisons (breach counts, diversification ratio) to be meaningful — reported anyway for completeness, with the limitation stated directly next to the numbers rather than omitted.
- "Hypothetical P&L" here means today's position sizes applied to historical returns — it is explicitly not a claim about what any of these strategies actually earned or lost in real history, several of which (bookmaker, pairtrade-lab-1, voledge) weren't even running as currently configured back in 2020–2022.
- The regime classifier used for the VaR-breach validation is refit fresh on each window's own pre-window data (not reusing the "current" 2-year classifier from the regime/ section) — a deliberately different, out-of-sample-correct classifier instance, disclosed so the two aren't confused for the same fitted model.

### Tests

`pytest` (`tests/test_historical_stress.py`) — replay math and the diversification-erosion ratio are checked against constructed positions/price series with exact hand-computable expected values (including the textbook sqrt(2) erosion ratio for two perfectly correlated equal-vol strategies, and a ratio of exactly 0 for perfectly offsetting ones). VaR breach counting is checked against a synthetic window with an injected, known number of extreme-loss days. One end-to-end test runs the real book against a real disclosed historical window.

## Hypothetical Stress Scenarios (`stress/hypothetical.py`)

Four disclosed, illustrative multi-factor shocks (equity %, crypto %, a *relative* IV shock, optional sector overrides) — round numbers meant to span different shapes of stress, not calibrated to or claiming to predict any specific future event. Every position is fully repriced: linear for equity/crypto, a **delta+gamma+vega second-order Taylor expansion** for options — extending the gamma-only convexity check from the Greeks section to also include the vega leg, now that these scenarios define an explicit vol shock to apply it against.

**A real units bug was caught and fixed building this module.** voledge's raw Black-Scholes vega is `dV/dσ` with σ in decimal (its own `greeks/analytical.py` docstring: "vega per 1.00 = 100 vol points"). Applying a scenario's vol shock correctly required converting it to an absolute decimal `d_sigma = entry_iv * vol_shock_pct` before multiplying by vega — getting this right surfaced that the Greeks section's earlier text had mislabeled `$242` as "P&L per 1-vol-point IV move," when it's actually per 100 points (\$2.42 per single point). Both the code comment and the README are now corrected, flagged inline rather than silently fixed.

As of this writing (`python -m stress.run_hypothetical`):

| Scenario | Full repricing P&L | Linear-only P&L (this project's usual proxy) | Convexity correction |
|---|---|---|---|
| Broad equity selloff (equity -20%, crypto -40%, vol +50%) | **+\$5,767** | -\$36 | +\$5,804 |
| Crypto-specific crash (crypto -50%, equity -5%, vol +20%) | -\$784 | -\$1,149 | +\$365 |
| Energy shock (Energy -30%, else -5%, vol +15%) | -\$5,778 | -\$6,142 | +\$364 |
| Rate shock / Financials (Financials -20%, else -10%, vol +30%) | +\$2,604 | +\$1,151 | +\$1,453 |

**Honest finding — this is the single most consequential number this project has produced, and it comes directly from combining the Greeks work with this module.** In the broad equity selloff scenario, the linear-only P&L estimate — the exact approximation `risk_measures/returns.py`'s VaR, `factor_model`'s regression, and every regime-conditional number in this project implicitly uses for the options book — predicts essentially **flat P&L (-\$36)** for a -20%/-40% crash. Full repricing shows **+\$5,767**, a swing of over \$5,800 driven almost entirely by voledge's options book (+\$5,363 of the total), which is net long gamma and long vega. That means every VaR and factor-regression number reported elsewhere in this project for a scenario of this magnitude would have gotten not just the *size* but plausibly the *sign* of the options book's contribution wrong. Stated precisely, not oversold: this \$5,800 swing is modest next to the portfolio's ~\$707k gross exposure (aggregation-layer section) — it does not mean the whole book is "saved" by its options — but it is large *relative to what the linear proxy predicted for this specific scenario*, which is exactly the point: the linear approximation isn't uniformly a little bit wrong, it can be wrong by an order of magnitude and in a specific direction depending on the scenario and the options book's actual convexity profile.

The other three scenarios show smaller, still real convexity effects (all positive here, since this options book happens to be net long gamma and long vega — a book with net short gamma would show the opposite sign, larger losses than the linear proxy predicts, which is the more dangerous case in practice and worth remembering isn't what's being shown here). The energy shock scenario also does what it was built to test: alpha-signal-lab loses \$6,331 of the portfolio's \$5,778 total loss, directly confirming its real, disclosed Energy sector concentration (already flagged in the factor-decomposition section) is a genuine risk driver, not a paper concern.

### Limitations

- These are illustrative, disclosed judgment-call scenarios, not calibrated to any model of scenario likelihood or historical frequency — "how likely is this, really" is exactly reverse stress testing's job, future work.
- The options book's convexity happened to be favorable (net long gamma/vega) in every scenario shown here — that is a fact about this specific book's current composition, not a general property of options books; a net-short-gamma book would show the opposite, more dangerous pattern, and nothing here should be read as "options always help in a crash."
- Sector overrides only apply to equities (via alpha-signal-lab's sector tagging); the options book's sole underlying (SPY) always uses the broad equity shock, since SPY itself has no single sector.
- The vol shock is relative to each option's own entry IV, not an absolute vol-point move — a disclosed modeling choice; results would differ for e.g. AMGN's very-low-baseline-vol contracts vs. one with a much higher one, since the same relative shock produces a different absolute `d_sigma` for each.

### Tests

`pytest` (`tests/test_hypothetical_stress.py`) — repricing is checked against constructed positions with known Greeks/market values, including hand-computing the full delta+gamma+vega Taylor expansion and confirming it matches exactly. Sector-override precedence, synthetic/unpriced-position handling, and aggregation across mixed asset classes are all checked directly. One end-to-end test runs every real scenario on the real book.

## Reverse Stress Testing (`reverse_stress/`)

The plan's own framing for this module: "the single most differentiated addition to the whole capstone." Every other stress module in this project asks "apply scenario X, measure loss Y." This asks the opposite: "what combination of factor moves produces a specific target loss, and how plausible is that, really?"

- **`optimization.py`** — models portfolio P&L as linear in the named factors (`factor_model/regression.py`'s own fitted loadings, reused directly, not re-derived). The set of factor-move combinations producing exactly a target loss is a hyperplane in factor space; of all points on it, the most plausible under a multivariate-normal assumption is the one at **minimum Mahalanobis distance** from zero — solved as a quadratic program via `cvxpy` (`minimize x^T Σ⁻¹x subject to loadings^T x + α == -target_loss`), using the Ledoit-Wolf-shrunk factor covariance (`correlation/shrinkage.py`, reused directly) rather than the raw sample covariance, for the same numerical-stability reason shrinkage was built in the first place.
- **A real scale bug was caught and fixed building this.** The first version solved the QP against a *daily* factor covariance, producing scenarios like "SPY +33% tomorrow" — nonsensical as a single-day move and impossible to compare against the multi-week historical crisis windows already in this project. Fixed by solving over a disclosed ~1-trading-month horizon (`to_horizon_returns`: overlapping 21-day rolling sums of daily returns, a linear approximation of compounding — and the regression's own daily alpha scaled by the same 21 days for consistency) instead of a single day.
- **`plausibility.py`** — two honest answers to "how likely is that, really," per the plan's own requirement: (1) the same solved scenario's Mahalanobis distance recomputed under the **volatile-regime-conditional** factor covariance instead of the pooled one (reusing `regime/volatility_tercile.py` directly); (2) the solved scenario's per-factor shock set next to what each factor **actually did**, factor by factor, during the three real historical windows already replayed in `stress/historical.py`.

As of this writing (`python -m reverse_stress.run`, portfolio factor model R²=0.087):

| Target loss | SPY | XLK | XLE | BTC-USD | Mahalanobis SD (pooled) | Implied probability |
|---|---|---|---|---|---|---|
| 10% (\$70,678) | +32.8% | +54.9% | -11.7% | -11.6% | 11.9 | ~0% |
| 25% (\$176,695) | +80.0% | +133.9% | -28.4% | -28.3% | 29.0 | ~0% |
| 50% (\$353,389) | +158.7% | +265.4% | -56.4% | -56.2% | 57.6 | ~0% |

**Honest finding #1 — the single most important, and most uncomfortable, result this project has produced: the "most plausible" way to break this specific book does not look like a historical crash at all, it looks like a rally.** The solver consistently finds that SPY and XLK **rising** (not falling) is the minimum-distance path to a large loss — a direct, mechanical consequence of the factor decomposition's own earlier finding: this book carries a large, significant *negative* SPY loading (pairtrade-lab-1's hidden short-market beta) and negative Technology/Healthcare loadings (alpha-signal-lab's short AVGO/INTC/NVDA/AMGN positions). None of the three real historical crisis windows in `stress/historical.py` look anything like this — COVID, 2022, and FTX all show SPY, XLK, and BTC-USD moving *down together*, not up. This is not a bug in the optimizer; it is reverse stress testing doing exactly its job: finding the scenario the portfolio is actually exposed to, which is not the scenario intuition (or a standard "equity crash" hypothetical, like the one in `stress/hypothetical.py`) would suggest.

**Honest finding #2 — every solved scenario is a many-standard-deviation, near-zero-probability event under both the pooled and volatile-regime-conditional covariances, and the two barely differ.** Even the smallest target (a 10% portfolio loss) requires an ~11.9 SD combined move (pooled) vs. ~12.0 SD (volatile-regime-conditional) — essentially the same, unlike the historical-replay section's finding where regime-conditioning materially changed the answer. This is itself informative: it means the *direction* of the required move (a rally, not a crash) is so far outside what either covariance considers typical that regime-conditioning on "how volatile were things" barely moves the needle — the problem isn't magnitude, it's that this book's risk is concentrated in a direction (rising SPY + rising tech + falling energy/crypto simultaneously) that essentially never happens historically, calm or stressed.

**A real limitation that has to be stated prominently, not buried:** the portfolio's own named-factor model has R²=0.087 (from the factor-decomposition section) — the six named factors barely explain this book's actual daily P&L variance, because pairtrade-lab-1's idiosyncratic AXP/WFC spread (the book's largest position by far) isn't well spanned by any of them. A reverse-stress scenario solved against a weak factor model is correspondingly less trustworthy as "the" plausible path to a loss — it may be finding an artifact of which six factors happen to load significantly, rather than a genuine description of this book's real risk. This isn't a caveat added after the fact to excuse a strange-looking result; it's the honest reason the result looks strange, and it is exactly the kind of finding reverse stress testing is supposed to produce: not just "here's a scenario," but "here's what that scenario reveals about whether your risk model actually describes your risk."

### Limitations

- The linear factor-model approximation (already used throughout this project) is least trustworthy exactly where reverse stress testing pushes it hardest — the solved scenarios (SPY up 33%–159% over a month) are far outside the range the model was ever fit on, an extrapolation, not an interpolation.
- R²=0.087 (see finding above) means these solved scenarios should be read as "what a 6-factor model implies," not as this project's confident answer to "what would actually break this portfolio" — a genuinely low-R² book needs either more/better factors or should lean more heavily on the historical-replay and hypothetical-scenario modules instead.
- The historical-window comparison table still isn't a perfect horizon match (COVID: 23 days, FTX: 4 days, 2022: 195 days, vs. the solved scenario's 21-day horizon) — closer than the original daily-vs-multi-week mismatch, but not exact.
- Options convexity (gamma/vega, `aggregation/greeks.py`) is not folded into the reverse-stress solve — the linear factor loadings used here already embed voledge's delta-only proxy, the same simplification flagged in the hypothetical-scenarios section.

### Tests

`pytest` (`tests/test_reverse_stress.py`) — the optimizer's `cvxpy` solution is checked against an independently-computed closed-form solution (the minimum-Mahalanobis-distance-subject-to-a-linear-constraint problem has a known Lagrangian closed form) and against exact constraint satisfaction. Mahalanobis distance is checked against a hand-computable diagonal-covariance case. One end-to-end test runs the full pipeline on the real book.

## Counterparty & Credit Risk (`credit/`)

PD tiers (`credit/counterparty.py`) were built back in the environment/data-acquisition phase — disclosed, literature-informed default-probability tiers per counterparty (Alpaca as a regulated broker-dealer at the low end; Binance/Kraken as less-regulated offshore exchanges at the high end; Coinbase in between), since no live credit-spread data exists for any of these. This section adds the rest: exposure by venue (reusing `aggregation/rollup.py::by_counterparty` directly), a simplified CVA, and concentration flagging.

**Scoping distinction, stated precisely:** "CVA" in derivatives pricing usually means a bilateral swap counterparty's time-varying expected exposure, discounted over a trade's life. What's actually relevant here is closer to **custodial/settlement counterparty risk** — the risk that a broker or exchange holding this project's assets becomes insolvent (the FTX scenario, already replayed as a real historical window). This module reuses the term "CVA" because that's the plan's own framing and the underlying formula is the same shape (`Exposure × PD × LGD`), but the exposure measure is a static current position, not a discounted expected-exposure profile — disclosed rather than left to look more sophisticated than it is.

- **Exposure measure**: `|net exposure|` per counterparty, not gross — in a custodial default, an offsetting long/short pair at the same broker nets to one account value, so gross would overstate what's actually at risk. A disclosed simplification (real custody/margin arrangements can be more complex), not a claim about actual legal recovery mechanics.
- **LGD (loss given default)**: a single disclosed assumption, 60% (a common simplified Basel-style figure for unsecured exposure), applied uniformly — not because 60% is precisely right for a crypto exchange vs. a regulated broker-dealer specifically, but because inventing a per-counterparty recovery rate with no supporting data would sound more precise without being more accurate.
- **Concentration**: computed only among counterparties with a real, non-zero PD — `Counterparty.NONE` (positions with no live venue, e.g. market-data-sourced backtest data) is excluded, since there's no real venue to be concentrated in for a position that isn't actually held anywhere.

As of this writing (`python -m credit.run`):

| Counterparty | Net exposure | PD tier | CVA | Share of real-venue exposure |
|---|---|---|---|---|
| Alpaca | \$1,115 | regulated broker-dealer, 0.10% | \$0.67 | 28.1% |
| Binance | \$2,849 | offshore exchange, 2.00% | \$34.18 | **71.9% — flagged** |
| (no live venue) | \$6,636 | n/a, PD=0 | \$0.00 | excluded |

Total CVA: \$34.85. Herfindahl index (real venues only): 0.596.

**Honest finding:** Binance is flagged at 71.9% of real-venue exposure — technically correct and a real concentration signal (HHI 0.596, well above the 0.5 threshold), but it needs its own caveat stated immediately, not left for a reader to discover: **this concentration finding currently applies to only \$3,964 of real-venue exposure**, against a book with \$654,842 of gross exposure tagged `Counterparty.NONE` (no live venue). That's not a gap in the concentration math — it's an accurate reflection of where this project actually stands: only alpha-signal-lab (Alpaca) and bookmaker's `binance_real` run (Binance) currently carry a real counterparty tag; pairtrade-lab-1 and voledge's disclosed stand-ins were generated from data sources (cached prices, a live options chain) with no live trading venue attached, so `Counterparty.NONE` is the correct, honest tag for them — not a placeholder that should be "fixed." The concentration finding is real but narrow in scope right now, and both facts are stated together rather than one implying the other.

### Limitations

- The concentration/CVA figures only cover the ~\$3,964 of currently-tagged real-venue exposure (see finding above) — they will become more representative of the whole book's true counterparty risk as more connectors' stand-ins are regenerated against live-venue data, not as a fix to this module.
- LGD is a single uniform assumption across very different counterparty types (a regulated US broker-dealer vs. an offshore exchange) — a more refined model would assign a different, still-disclosed LGD per counterparty type, not implemented here.
- Net (not gross) exposure as the CVA/concentration base is a disclosed simplification of real custody/margin mechanics, which can differ meaningfully by jurisdiction and by venue (e.g. per-asset segregation rules) — not modeled here.
- PD tiers themselves remain assumption-based, not market-implied (see the environment/data-acquisition section) — CVA numbers here inherit that same limitation directly.

### Tests

`pytest` (`tests/test_credit.py`) — CVA is checked against hand-computable cases, including confirming it uses net (not gross) exposure for an offsetting long/short pair, scales linearly with LGD, and is exactly zero for `Counterparty.NONE`. Concentration is checked for HHI correctness on an even split, correct threshold flagging, and correct exclusion of the no-live-venue bucket. One end-to-end test runs the real pipeline.

## Extreme Value Theory: Regime-Conditional Tail Risk (`extreme_value/`)

The pooled EVT/POT method (Generalized Pareto tail fit) was already built as one of the five VaR methods, back in the risk-measures section. This step's actual new work, per the plan, is checking whether **tail behavior itself** differs by regime — a more granular question than regime/conditional.py already answered (that module checked whether regime-conditional VaR *levels* differ; this checks whether the underlying tail *shape* does).

**A real refactor, not just an addition**: rather than duplicate the GPD-fitting logic to get structured shape/scale parameters (needed to compare xi across regimes numerically, not just read off a VaR dollar figure), the fitting core was extracted from `risk_measures/var.py::evt_pot` into a shared `extreme_value/gpd.py::fit_gpd_tail()` / `gpd_var_cvar()`, and `evt_pot` was refactored to delegate to it. The existing VaR-method tests were re-run unchanged after the refactor to confirm identical behavior (they pass without modification).

**The data-sufficiency problem regime/conditional.py's own comment anticipated is real, and this module works through it rather than around it.** A ~160-day regime bucket gives only ~16 exceedances at the usual 90% POT threshold — under `MIN_EXCEEDANCES` (20). At the default threshold, **all three regimes fail to fit** on this book's real data:

```
=== threshold_quantile=90% ===
  volatile: ~15 exceedances on 160 days -- too few to fit a GPD reliably.
  normal:   ~15 exceedances on 160 days -- too few to fit a GPD reliably.
  calm:     ~15 exceedances on 160 days -- too few to fit a GPD reliably.
```

Lowering the threshold to 80% (more exceedances, but blending in more non-tail behavior — a real, disclosed trade-off, not a free fix) makes all three fit:

| | ξ (shape) | β (scale) | n exceedances | 95% VaR | 95% CVaR |
|---|---|---|---|---|---|
| Pooled | 0.071 | 3,325 | 100 | \$7,976 | \$11,924 |
| Volatile | 0.063 | 4,134 | 32 | \$9,030 | \$13,843 |
| Normal | 0.185 | 2,674 | 32 | \$7,445 | \$11,684 |
| Calm | -0.315 | 4,510 | 32 | \$7,953 | \$10,171 |

**Honest finding — a genuinely non-obvious answer to the question this module set out to answer.** The volatile regime does have the highest CVaR (\$13,843, as expected), but that's driven almost entirely by **scale (β=4,134, the largest of the three)**, not by tail **shape** — its ξ (0.063) is actually the *second-lowest* of the three regimes. The regime with the fattest tail shape is, surprisingly, **normal** (ξ=0.185) — meaning extreme losses are disproportionately more likely there, relative to that regime's own more moderate scale, than in either calm or volatile. Naive intuition ("stress = fatter tails") would predict volatile has the highest ξ; the real fitted data says otherwise. Also notable: calm's ξ is *negative* (-0.315), implying a GPD with a genuine finite upper bound on losses in that regime — consistent with "calm" being a real, qualitatively different loss regime, not just a scaled-down version of the others.

**A second honest finding, about the method itself, not just this book:** the pooled fit's own ξ changes from 0.515 (at the 90% threshold, `risk_measures/run.py`'s original result) to 0.071 (at 80%) on the *exact same* underlying loss series — a well-known GPD methodological issue (threshold-selection instability), now directly demonstrated on real data rather than only mentioned as a theoretical caveat. Any single reported ξ should be read together with the threshold it was fit at, not as a fixed property of the return series.

### Limitations

- The 80% threshold was chosen specifically because 90% left every regime bucket unfittable — a data-driven necessity, not an independently principled choice; the resulting fits blend more non-tail behavior into the tail estimate than a stricter threshold would.
- Threshold-selection instability (see finding above) means these ξ/β estimates, regime-conditional or pooled, should be treated as threshold-dependent snapshots, not precise, unique properties of the loss distribution.
- Each regime bucket's 32 exceedances (at 80%) is barely above `MIN_EXCEEDANCES` (20) — comfortably fittable, but still a small sample for a shape-parameter estimate, which is known to have high variance even with 100+ exceedances in the literature.
- This shares the regime-conditional VaR work's own limitation: SPY-defined regimes, applied to a book whose risk (per the factor-decomposition section) isn't SPY-beta-dominated.

### Tests

`pytest` (`tests/test_extreme_value.py`) — GPD fitting is checked against synthetic data with injected, known shape/scale parameters (recovered within statistical noise, using the same threshold-placement care the plan's own POT construction requires) and hand-computable edge cases (the ξ≈0 exponential-tail formula, the below-threshold-coverage NaN case). The `evt_pot` refactor is checked to produce numerically identical output to a direct `fit_gpd_tail` call. Regime-conditional fitting is checked against constructed buckets with a controlled, known exceedance count to deterministically trigger the `MIN_EXCEEDANCES` cutoff. One end-to-end test runs both thresholds on the real book.

## Liquidity-Adjusted Risk & Concentration (`liquidity/`)

- **`impact.py`** — adds an estimated unwind cost on top of mark-to-market VaR, reusing **execedge's own square-root-law market-impact model directly** (`algos/impact_calibration.py` in that sibling repo), not a reinvented formula: `cost_fraction = Y × σ × √(participation_rate)`. execedge's own module docstring discloses a real gap — neither the Almgren & Chriss (2000) nor the Almgren-Thum-Hauptmann-Li (2005) papers' exact fitted coefficients could be extracted in that project's environment (no PDF-to-text tooling, two attempts failed) — so it falls back to the square-root-law functional form with `Y=1.0` as an explicit "textbook order-of-magnitude convention," not a verified fitted number. This module reuses that exact formula and that exact disclosed gap, rather than silently resolving it with a more confident-looking constant of its own.
- **Participation rate is computed in dollar terms** (`|market_value| / average_daily_dollar_volume`), not share/coin counts. Alpaca supplies base-unit volume; RiskDesk converts it consistently to approximate dollar volume as daily close × volume before averaging. Dollar participation rate is standard practice in real execution work and keeps equity and crypto liquidity estimates in comparable units.
- **`concentration.py`** — the same HHI + threshold-flag pattern `credit/concentration.py` already established for counterparty risk, generalized to name/strategy/sector via `aggregation/rollup.py`'s own `rollup_by`. Disclosed thresholds: 20% for a single name, 50% for a single strategy, 40% for a single sector. Deliberately uses **gross** exposure (not net, unlike the counterparty check) — a name/strategy/sector limit is about how much capital and risk is actually deployed there, where the aggregation-layer section's own finding applies directly ("a portfolio that's net-zero can still carry large gross risk"), whereas counterparty risk is about custodial netting at one account. Both choices are stated so they don't read as an inconsistency.

As of this writing (`python -m liquidity.run`):

- Base 95% 1-day VaR: \$7,594. Liquidity-adjusted: \$7,800 (+2.7%, +\$206).
- Concentration: **AXP** (46.8%) and **WFC** (45.5%) both flagged by name (20% threshold); **pairtrade-lab-1** flagged by strategy (92.3% vs. 50% threshold); **Financials** flagged by sector (92.3% vs. 40% threshold).

**Honest finding #1 — liquidity risk is genuinely small for this book, and the number says so plainly rather than being dressed up as more significant.** All participation rates come out under 0.04% (this book's positions are tiny relative to real market volume in AXP, WFC, the energy names, and BTC), so the unwind-cost add-on is modest — \$206 on a \$7,594 base VaR. This is a real, useful negative result: for this specific book's current size, market-impact cost is not where the risk is, and the liquidity-adjusted number should not be read as "the real VaR is meaningfully higher than reported" — it isn't, here.

**Honest finding #2 — this is the *third* independent lens in this project to land on the exact same root cause.** AXP/WFC's name concentration, pairtrade-lab-1's strategy concentration, and Financials' sector concentration are not three separate findings — they're the same fact (pairtrade-lab-1's oversized two-leg stand-in position) viewed through three different, independently-built modules: the factor-decomposition section found it as hidden SPY/Financials beta (t-stats, p-values), the aggregation-layer section found it as a capital-base mismatch (gross exposure dollars), and this section finds it as a concentration-limit breach (HHI, threshold flags). Three different methodologies converging on one answer is a stronger, more trustworthy signal than any one of them alone — and a direct, repeated illustration of why the pairtrade-lab-1 stand-in snapshot's small size (2 positions) is the single most consequential open data-quality issue in this project, flagged again here rather than left to only the earlier sections.

### Limitations

- The square-root-law's `Y=1.0` is execedge's own disclosed placeholder, not a verified fitted constant for these specific assets (equities/BTC/SPY options) — it was originally validated in that project's context against crypto order-book data, not directly against this book's holdings.
- Liquidation cost is estimated per-position independently and summed — it does not model correlated liquidity stress (e.g. every position becoming harder to unwind simultaneously during a real crisis), which is exactly when liquidity risk usually matters most and is precisely least captured by a business-as-usual 3-month average daily volume.
- Concentration thresholds (20%/50%/40%) are disclosed, round judgment calls, not derived from any formal risk-limit framework or this book's own actual risk tolerance.
- The name/strategy/sector concentration finding is, again, driven almost entirely by one position (see finding #2) — it will look very different once pairtrade-lab-1's stand-in is regenerated with a fuller, multi-pair book.

### Tests

`pytest` (`tests/test_liquidity.py`) — impact cost is checked against the exact square-root-law formula by hand, including confirming the textbook property that cost scales with the square root (not linearly) of position size, and that it's symmetric for long/short positions of equal magnitude. Concentration is checked against constructed positions with known exposures, including a direct test of the deliberate gross-vs-net difference from credit/concentration.py. One end-to-end test runs the full pipeline on the real book.

## P&L Attribution (`attribution/`)

Decomposes a P&L series -- day by day -- into the portion explained by each named factor, an alpha (intercept) term, and a residual, reusing `factor_model/regression.py`'s own fitted loadings directly rather than re-deriving them. This is a different deliverable from the factor-decomposition section itself: that section reports static coefficients ("the book has X exposure to SPY"); this one walks those loadings back across a real window to show cumulative *dollar* contribution ("SPY moves accounted for $Y of this window's actual P&L").

- **`pnl.py::attribute_by_factor`** — the residual is defined as whatever's left over after factor contributions and alpha are subtracted from actual P&L, which makes `sum(factor contributions) + alpha + residual == total P&L` hold **exactly**, not approximately — checked directly (`.reconciles()`), not merely asserted. A real attribution has to tie out to the total to be trustworthy.
- **`strategy.py::attribute_by_strategy`** — the same hypothetical-historical-P&L split already used inline in `stress/historical.py` and `factor_model/run.py`, centralized here as the canonical attribution entry point rather than left duplicated a third time.
- **Methodology, disclosed**: factor loadings are fit on the full 2-year window (for stable coefficients) but *attributed* over just the most recent ~3 months (63 trading days) — a standard practice (longer estimation window, shorter attribution window), not a new kind of claim; this is still the project's established hypothetical-historical-P&L framing (today's positions × historical returns), not a real trading track record.

As of this writing (`python -m attribution.run`, last 63 trading days):

| | Total | SPY | XLF | XLK | XLE | XLV | BTC-USD | alpha | residual |
|---|---|---|---|---|---|---|---|---|---|
| **Portfolio** | \$11,226 | -\$15,623 | \$13,558 | \$3,682 | \$1,290 | -\$3,214 | -\$2,516 | \$9,089 | \$4,959 |
| alpha-signal-lab | \$1,664 | -\$207 | \$760 | -\$2,247 | \$1,069 | -\$1,538 | \$59 | \$3,769* | — |
| pairtrade-lab-1 | \$9,979 | -\$15,550 | \$12,799 | \$5,929 | \$221 | -\$1,676 | -\$2,024 | \$10,279* | — |
| bookmaker | -\$550 | ~0 | ~0 | ~0 | ~0 | ~0 | -\$550 | ~0* | — |
| voledge | \$133 | \$133 | ~0 | ~0 | ~0 | ~0 | ~0 | ~0* | — |

*(per-strategy alpha and residual are combined in the table above; reconciles exactly to each strategy's total, same as the portfolio-level row)*

**Honest finding — this is the fourth independent module in this project to converge on the same root cause, now expressed as a dollar attribution rather than a statistical test.** \$9,979 of this window's \$11,226 total P&L (89%) came from pairtrade-lab-1 alone, and of that, \$10,279 is alpha+residual — i.e. *more than its entire realized P&L* is unexplained by any named factor (offset by a small negative net factor contribution). This is the same fact the factor-decomposition section found as a low R² (0.062) and a significant SPY/Financials loading, the aggregation section found as a capital-base mismatch, and the liquidity section found as a concentration-limit breach — now shown as an actual dollar figure: the book's realized P&L this window was overwhelmingly driven by one strategy's idiosyncratic, factor-unexplained risk, not by any of the six named market factors this project built.

### Limitations

- Same disclosed limitation as risk_measures/returns.py throughout this project: this is a hypothetical replay (today's position sizes × historical returns), not a real trading track record — several strategies (bookmaker, pairtrade-lab-1, voledge) weren't running as currently configured over this actual calendar window.
- The 63-day attribution window is a disclosed round-number choice (~1 quarter), not derived from any particular reporting cadence requirement.
- Per-strategy attribution uses each strategy's own separately-fit factor loadings (matching factor_model/run.py's own per-strategy regressions) rather than the portfolio-level loadings applied strategy-by-strategy — a deliberate choice (each strategy's own factor exposure is what actually explains its own P&L) stated so the portfolio-level and per-strategy factor columns in the table above aren't mistaken for using the same coefficients.

### Tests

`pytest` (`tests/test_attribution.py`) — factor attribution is checked for exact reconciliation (constructed cases with known injected noise, confirming the residual recovers that exact noise series) and against hand-computable single-factor contributions. Strategy attribution is checked for additivity — per-strategy P&L series must sum back to the whole portfolio's, since P&L is linear by construction — and for correctly excluding strategies with no mappable positions. One end-to-end test runs the full pipeline and checks reconciliation on the real book.

## Live Risk Monitoring (`monitor/`)

The last risk-analytics module. Pulls live positions, recomputes VaR using **whichever regime-conditional model currently applies** (falling back to the pooled model, explicitly, when the current regime doesn't have enough historical days for its own — the exact mechanism `regime/run_conditional.py`'s "live monitoring readiness" section already demonstrated, now built into an actual live-check entry point), checks market-risk and credit-risk limits independently, and feeds both into a kill-switch.

**The kill-switch reuses alpha-signal-lab's own established pattern directly, not a new design.** alpha-signal-lab's `risk/kill_switch.py::KillSwitch` (and pairtrade-lab-1's own kill-switch, same shape) is manual-reset-only: once triggered, `.triggered` stays `True` until `.reset()` is called explicitly, so a kill-switch doesn't silently re-arm itself just because the breach that caused it happens to clear on the next check. This module's `monitor/kill_switch.py::KillSwitch` matches that exact shape (`.triggered` / `.check()` / `.reset()`), extended per the plan's own requirement to trigger on multiple independent conditions — market-risk (VaR vs. a disclosed 5%-of-gross-exposure limit) or credit-risk (`credit/concentration.py`'s counterparty concentration check, reused directly) — rather than the sibling repos' single-condition (equity drawdown) switches.

**A real design difference from the sibling repos' switches, worth stating explicitly:** their kill-switches only need to survive across bars *within* one backtest run — an in-memory object is enough, since the whole run is one process. This project's monitor is meant to be invoked repeatedly (e.g. on a schedule) as a genuinely live, recurring check; for "stays triggered until manually reset" to mean anything *across separate process invocations*, state has to be written to disk and reloaded, not just held in memory. `monitor/kill_switch.py::load_state`/`save_state` persist to `monitor/state.json` (gitignored — it's runtime state, not source) for exactly this reason.

As of this writing (`python -m monitor.live`):

```
Live positions pulled: 33 (33 priced). Gross exposure: $706,779.08
Current regime (SPY, most recent trading day): normal
Active VaR model: normal-conditional -> $6,110.21
  VaR is 0.86% of gross exposure (limit 5%)
  Counterparty concentration breached: ['binance'] (HHI=0.596)

*** KILL-SWITCH: TRIGGERED NOW ***
  - Counterparty concentration breached: ['binance'] (HHI=0.596)
  Manual reset required: python -m monitor.live --reset
```

Reset persists correctly across separate invocations (verified directly, not just asserted): `python -m monitor.live --reset`, then a fresh `python -m monitor.live` re-triggers cleanly from the same real breach, not from stale in-memory state.

**Honest finding:** the market-risk (VaR) limit is nowhere close to breached (0.86% vs. a 5% limit) — consistent with every VaR figure reported throughout this project for this book's actual current size. The kill-switch trips purely on the **credit-risk** side: the same Binance concentration finding from the counterparty & credit risk section. This is a genuinely useful demonstration of the plan's own requirement that market-risk and credit-risk limits be "independently triggerable" — here, only one of the two categories is actually live-triggering, and the live monitor reports that distinction plainly (which limit, not just that some limit) rather than collapsing both into one undifferentiated "risk is high" signal.

### Limitations

- This is a one-shot script re-run on demand (or externally scheduled, e.g. via cron), not a persistent running service with its own event loop — appropriate for this project's scope, but "near-real-time" here means "as fresh as the last invocation," not a continuously streaming feed.
- The 5% VaR/gross-exposure limit is a disclosed, round-number judgment call, not derived from this book's actual risk tolerance or any formal limit-setting framework (the same caveat already stated for liquidity/concentration.py's thresholds).
- The kill-switch currently checks two limit categories (VaR, credit concentration); it does not yet also check the name/strategy/sector concentration limits already built in liquidity/concentration.py — a natural, straightforward extension not implemented here to keep the live-check's output focused on the two categories the plan explicitly calls out.
- `monitor/state.json` is a single local file, not a shared/durable store — fine for this project's scope, but a real production deployment would need the kill-switch state itself to survive the monitor process's own host disappearing, which a local file does not guarantee.

### Tests

`pytest` (`tests/test_monitor.py`) — the kill-switch state machine is checked directly for manual-reset-only semantics (stays triggered after the breach clears, does not duplicate reasons across repeated checks) and for persistence surviving separate load/save cycles (simulating separate process invocations, not just an in-memory round-trip). Limit checks are checked against hand-computable breach/no-breach cases. One end-to-end test runs the real live-check pipeline with kill-switch persistence isolated to a temp path, so the test suite never mutates the real `monitor/state.json`.

## VaR Backtesting / Statistical Rigor (`risk_measures/backtesting.py`)

Kupiec's Proportion-of-Failures (POF) test and Christoffersen's independence test — the standard pair for asking "is this VaR model actually calibrated" with an actual hypothesis test and p-value attached, not an eyeballed breach count. Run for both the pooled model and a regime-conditional model (whichever regime's VaR applies on a given day — the same day-by-day selection logic `monitor/live.py` uses live) over the *same* evaluation period, so "does regime-conditioning improve calibration" gets a direct, quantified answer.

- **Kupiec POF**: does the observed breach *rate* match the expected rate (`1 - confidence`)?
- **Christoffersen independence**: even a correctly-rated VaR can be badly calibrated if breaches *cluster* in time rather than scattering independently — this tests specifically for that, via a first-order Markov chain on the breach indicator series.
- **Combined test**: `LR_pof + LR_ind`, chi-squared with 2 degrees of freedom — the real bar a well-calibrated model needs to clear (correct rate *and* no clustering).

**A real methodological nuance surfaced running this, disclosed rather than smoothed over:** the pooled model's Kupiec result is close to **tautological** in-sample. `historical_simulation`'s VaR *is* the empirical quantile of the same series the breach rate is then measured against — so of course a 95th-percentile threshold breaches ~5% of that same sample; that's what a quantile *is*, not independent evidence of good calibration. The regime-conditional model's breach rate is a more genuine test, since it isn't reading off one static quantile — it's a day-by-day mix of different regime-specific VaRs that has no guarantee of landing near 5% just by construction. The Christoffersen independence test is meaningful for both models regardless (clustering isn't guaranteed away by either construction), and is where the real comparison lives.

As of this writing (`python -m risk_measures.run_backtest`, 95% confidence, 500 days):

| | Breaches | Kupiec p | Christoffersen p | Combined p | Well-calibrated? |
|---|---|---|---|---|---|
| Pooled | 25 (5.00%, near-tautological — see above) | 1.000 | 0.514 | 0.808 | Yes |
| Regime-conditional | 24 (4.80%) | 0.836 | 0.449 | 0.735 | Yes |

**Honest finding — a third, now formally statistically-tested confirmation of a theme already threaded through this project.** Both models pass (fail to reject the null of correct calibration), but the regime-conditional model's combined p-value (0.735) is actually *lower* than the pooled model's (0.808) — regime-conditioning did not improve calibration by this test, on this book, over this window. This lines up directly with the historical-replay section's own finding (regime-conditioning provided no edge in the COVID window, and was overly conservative in the 2022 window) and the regime-conditional-VaR section's finding (this book's risk isn't cleanly separated by SPY's own volatility regime) — three independent methodologies now agreeing that regime-conditioning's value, for this specific book, is genuinely mixed rather than a reliable improvement, not a result to be talked around.

### Limitations

- This is an **in-sample** calibration check — VaR is estimated once over the same 2-year window the breaches are counted against, not a rolling/expanding walk-forward backtest (which would need a fresh re-fit every single day, computationally far more expensive). `stress/historical.py`'s VaR-breach validation is this project's genuinely out-of-sample complement (fit on pre-window data only, tested against a real crisis window); this module is the formal statistical test the plan calls for, not a replacement for that check.
- The pooled model's Kupiec result specifically should be read with the tautology caveat above front and center — it is expected to look well-calibrated by construction, not because the underlying VaR methodology is validated.
- 500 days gives only ~25 breaches at 95% confidence — a small sample for a hypothesis test; the combined test's power to actually detect real miscalibration at this sample size is limited, a real statistical constraint, not a modeling choice.

### Tests

`pytest` (`tests/test_backtesting.py`) — Kupiec is checked against an exactly-calibrated synthetic series (near-zero LR) and a grossly miscalibrated one (large LR, reject), plus zero- and all-breach edge cases that would otherwise hit `log(0)`. Christoffersen is checked against a deliberately clustered breach series (rejects independence) vs. a scattered one with the identical total breach count (does not reject), and against a genuinely IID series (does not reject). The pooled breach series' near-tautological property is checked and documented directly as a test, not just prose.

## Testing & Validation — audit

This project has been test-driven throughout — every module above already shipped with its own tests before moving to the next step, not as an afterthought. This section is the explicit audit the plan calls for: going back through five specific validation requirements and checking, item by item, whether they were genuinely covered or only appeared to be. Two were already fully satisfied by existing tests; three had real, identifiable gaps that got filled here rather than assumed away.

| Requirement | Status | Where |
|---|---|---|
| Aggregation correctness tests | **Gap found & filled** — existing tests covered valuation and single-case rollups, but no test verified that per-dimension rollups (strategy/asset-class/counterparty) actually *reconcile* to the portfolio total, the core aggregation-correctness invariant. Added, plus an empty-position-list edge case. | `tests/test_aggregation.py::test_strategy_rollups_reconcile_exactly_to_portfolio_total`, `test_asset_class_and_counterparty_rollups_also_reconcile`, `test_empty_position_list_produces_zeroed_portfolio_total_not_an_error` |
| VaR method sanity checks against closed-form cases | **Gap found & filled** — VaR itself was checked against the closed-form normal formula, but CVaR (expected shortfall) wasn't checked against its own separate closed-form formula. Added. | `tests/test_var.py::test_parametric_cvar_matches_normal_closed_form_expected_shortfall` (plus the pre-existing historical/parametric/EVT/Cornish-Fisher/Monte-Carlo checks) |
| DCC-GARCH / Ledoit-Wolf sanity checks | **Gap found & filled** — both were checked for validity (PSD, stationarity, condition-number improvement) but not for their own *degenerate-case* correctness. Added: DCC-GARCH should detect the *absence* of real dynamics (fitted persistence near zero, dynamic correlation staying close to static) on genuinely constant-correlation data; Ledoit-Wolf's shrinkage should *decrease* with sample size for the same underlying structure — the actual mechanism a first attempt at this test got wrong (a "well-conditioned implies low shrinkage" premise, verified false against real `sklearn` behavior before writing a test around it). | `tests/test_correlation.py::test_dcc_garch_collapses_toward_static_correlation_with_no_true_dynamics`, `test_ledoit_wolf_shrinkage_decreases_with_sample_size_for_same_structure` |
| Regime classifier tests against known synthetic regime-switching series | **Already covered**, no gap — both tercile and HMM classifiers were already checked against synthetic series with known, injected calm/normal/volatile segments. | `tests/test_regime.py::test_classify_regimes_labels_known_calm_and_volatile_segments`, `test_hmm_recovers_injected_two_regime_structure` |
| Reverse-stress optimization: recovers a known injected scenario | **Already substantively covered, made explicit** — the existing closed-form-match test proves the same thing mathematically; added a version framed as the plan literally describes it (construct a known scenario, compute the loss it produces forward, solve backward from that loss, check the exact scenario comes back), for direct traceability rather than relying on the reader to connect the two. | `tests/test_reverse_stress.py::test_solve_reverse_stress_recovers_a_known_injected_scenario` (see also `test_solve_reverse_stress_matches_closed_form_minimum_distance_solution`) |
| CVA calculation sanity check against a known simple case | **Already covered**, no gap. | `tests/test_credit.py::test_compute_cva_matches_hand_calculation` |

**Honest note on the Ledoit-Wolf gap specifically, since it's a real example of this project's own working method, not just its conclusion:** the first version of that test assumed "a well-conditioned, large sample should shrink less" and was about to be written around that premise — checking it directly against real `sklearn.covariance.LedoitWolf` behavior first showed that assumption is false (a large, *genuinely near-independent* sample shrinks heavily, close to 1.0, since there's no real off-diagonal structure to estimate regardless of sample size). The test that actually shipped checks the correct, verified mechanism instead (shrinkage vs. sample size *holding the true structure fixed*), a small, direct example of the verify-before-assert discipline applied throughout this project, not just claimed by it.

158 tests passing total across the project (155 above, plus 3 more from the static-vs-DCC-GARCH-during-stress analysis below, built specifically to answer this section's second question).

## Results & Honest Comparison

Every finding below was already established, tested, and documented in its own section above — this section's job is synthesis: pulling six specific, plan-mandated questions together into direct answers, with one piece of genuinely new analysis (static vs. DCC-GARCH correlation *during* the three real historical crisis windows specifically, `stress/dcc_during_stress.py`) built because it didn't already exist in exactly that form.

### 1. VaR method × confidence level × regime, with backtest calibration

| | 95% VaR | 99% VaR | Calm | Normal | Volatile | Backtest (95%, combined p-value) |
|---|---|---|---|---|---|---|
| Historical simulation | 7,592 | 14,075 | 7,964 | 6,111 | 7,601 | pooled 0.808 |
| Parametric (normal) | 8,119 | 11,511 | 7,432 | 8,439 | 8,605 | — |
| Monte Carlo | 8,063 | 11,528 | — | — | — | — |
| Cornish-Fisher | 8,483 | 17,133 | 7,345 | 8,652 | 9,669 | — |
| EVT (POT) | 7,249 | 13,721 | — (needs 80% threshold; see EVT section) | — | — | — |
| Regime-conditional (mixed methods) | — | — | — | — | — | 0.735 |

The five methods **agree closely at 95%** (~17% spread) but **diverge sharply at 99%** (~49% spread) — parametric and Monte Carlo (both assume normal returns) understate the tail relative to historical, Cornish-Fisher, and EVT, because the book's real loss series has positive skew (0.48) and excess kurtosis (3.45). Regime-conditioning barely separates calm from volatile (1.0–1.3x ratio) — this book's risk isn't cleanly SPY-regime-driven. And formally: **regime-conditioning did not improve backtest calibration** (combined p-value 0.735 vs. pooled's 0.808 — lower, not higher), consistent with the regime section's own finding, not an isolated result.

### 2. Static vs. DCC-GARCH correlation during real stress windows

New analysis (`python -m stress.run_dcc_during_stress`), fitting DCC-GARCH once on a long history spanning all three crisis windows, then reading its fitted correlation specifically *during* each window vs. the same period's static correlation:

| Window | Pair | Static | DCC (pre-window) | DCC (during) | DCC (max during) |
|---|---|---|---|---|---|
| COVID | CVX–SPY | +0.63 | +0.49 | +0.64 | **+0.69** |
| COVID | AXP–WFC | +0.74 | +0.54 | +0.63 | +0.72 |
| COVID | SPY–BTC-USD | +0.29 | +0.16 | +0.23 | +0.35 |
| 2022 rate-hike | CVX–SPY | +0.63 | +0.52 | +0.43 | +0.55 |
| 2022 rate-hike | AXP–WFC | +0.74 | +0.58 | +0.65 | +0.70 |
| FTX collapse | CVX–SPY | +0.63 | +0.49 | +0.54 | +0.55 |

**Honest finding — the "correlations spike under stress" story holds cleanly for COVID and is genuinely mixed for 2022, not a uniform law.** Every single pair tested (5 of 5) showed DCC-GARCH correlation rising during COVID relative to the pre-window level, with CVX–SPY rising the most (+0.15, 0.49→0.64, peaking at 0.69) — a textbook systemic-stress correlation spike. **2022 tells a different, real story**: CVX–SPY actually *fell* during that window (0.52→0.43) — because energy was one of the only sectors that outperformed while broad equities sold off through 2022, breaking the "everything correlates in a crash" pattern that held for COVID. FTX's window is too short (4 days) for the moves to be meaningful either way. This is a more honest, differentiated answer than "DCC-GARCH catches spikes static correlation misses" as a blanket claim — it depends entirely on what *kind* of crisis is happening, and only a real out-of-sample check against actual historical data (not just a generic 2-year window) can show that.

### 3. Regime-conditional vs. pooled: quantified improvement, or lack of it

No single answer — three independent tests, three different outcomes, all real:

- **Formal calibration (Kupiec + Christoffersen, in-sample)**: regime-conditional combined p-value (0.735) is *lower* than pooled's (0.808) — no improvement.
- **Historical scenario prediction (out-of-sample, real crisis windows)**: COVID — regime-conditional VaR was breached exactly as often as pooled (6 of 23 days each; a regime classifier trained on pre-2020 "volatile" days had never seen anything like COVID, so conditioning provided no edge for a genuinely novel shock). 2022 — pooled was already reasonably well-calibrated (8 breaches vs. ~9.8 expected) while regime-conditional was *overly conservative* (2 breaches — too high a VaR, wasting capital). FTX — too short (4 days) to conclude either way.
- **Tail shape (EVT, per-regime)**: the volatile regime's higher CVaR is driven by *scale* (β=4,134), not *shape* (ξ=0.063, actually the second-lowest of the three regimes) — the *normal* regime has the fattest tail shape (ξ=0.185), the opposite of naive intuition.

**The honest, unifying answer**: regime-conditioning's value for this book, over this window, is genuinely mixed — not a reliable improvement, not worthless, dependent on whether the regime classifier has actually seen anything like the shock in question before. A capstone that only reported the cases where regime-conditioning helped would be cherry-picking; this one reports all three outcomes because that's what a rigorous comparison actually produced.

### 4. Reverse stress testing: the most plausible breaking scenario, and how extreme it really is

**The most plausible way to break this book is a rally, not a crash.** The solver consistently finds SPY and Technology *rising* (not falling) as the minimum-distance path to a large loss — SPY +32.8%/XLK +54.9% for a 10% portfolio loss, scaling up to SPY +158.7%/XLK +265.4% for a 50% loss (all over a ~1-month horizon). This is a direct, mechanical consequence of pairtrade-lab-1's hidden negative SPY loading and alpha-signal-lab's short Technology/Healthcare positions (both findings from the factor-decomposition section) — not an artifact of the optimizer.

**How extreme, really**: every solved scenario is a many-standard-deviation, near-zero-probability event — 11.9 SD (10% loss) up to 57.6 SD (50% loss) under the pooled factor covariance, and the volatile-regime-conditional distance is barely different (12.0 SD vs. 11.9 SD at the 10% level) — unlike the historical-replay section, regime-conditioning doesn't meaningfully change the plausibility verdict here, because the *direction* of the required move (a simultaneous rally + tech surge + energy/crypto decline) is so far outside what either covariance considers typical that "how volatile were things" barely matters.

**The caveat that has to travel with this finding, not trail after it**: the portfolio's own named-factor model has R²=0.087 — the six named factors barely explain this book's actual P&L variance, because pairtrade-lab-1's idiosyncratic AXP/WFC spread (the book's largest position) isn't well spanned by any of them. This scenario should be read as "what a weak 6-factor model implies," not this project's confident answer to what would actually break the portfolio.

### 5. Counterparty concentration and CVA findings

Total CVA: **\$34.85** — small in absolute terms, reflecting both low disclosed PD tiers (0.10%–2.00% annualized) and small current net exposures. **Binance is flagged at 71.9% of real-venue exposure** (Herfindahl index 0.596, well above the 0.5 concentration threshold) — a real signal, but one that needs its own scope caveat stated immediately: it currently applies to only **\$3,964** of real-venue exposure, against a book with \$654,842 of gross exposure tagged "no live venue" (pairtrade-lab-1 and voledge's disclosed stand-ins have no live trading venue attached — a correct tag, not a gap to fix). The live monitor (`monitor/live.py`) confirms this split concretely: the kill-switch trips on the **credit-risk** side specifically, while the market-risk (VaR) limit sits comfortably inside its threshold (0.86% vs. 5%) — the two risk categories the plan asks to be independently triggerable are, in this book's current state, genuinely giving different answers.

### 6. The capstone's central question: is this six-project book actually as diversified as each project's own backtest implied?

**No — and the evidence is not one finding, it's four independent methodologies, built at different points in this project with no knowledge of each other's conclusions, all converging on the exact same root cause:**

1. **Factor decomposition** found pairtrade-lab-1's "market-neutral" pairs book carries statistically significant hidden market beta (SPY loading t=-2.98, p=0.003) and Financials beta (t=3.06, p=0.002) once aggregated with real position sizing — the opposite of what "market-neutral" implies about its own risk.
2. **Position aggregation** found pairtrade-lab-1's gross exposure (\$652,639) dwarfs the rest of the book combined (~\$54k) — a capital-base mismatch invisible to any strategy's own isolated backtest, since each one only ever sees its own book.
3. **Concentration limits** (liquidity section) independently flag the same thing three ways: AXP and WFC each ~46% of name concentration (20% limit), pairtrade-lab-1 92.3% of strategy concentration (50% limit), Financials 92.3% of sector concentration (40% limit).
4. **P&L attribution** found 89% of a recent window's total P&L came from pairtrade-lab-1 alone, and *more than its entire realized gain* was unexplained residual — not attributable to any of the six named market factors this project built.

And **reverse stress testing** reveals the consequence directly: the scenario that would actually break this aggregated book — a rally, not a crash — is not one any individual strategy's own risk process would ever have surfaced, because none of the six source repos model cross-strategy aggregation at all. That is the whole reason this capstone exists.

**The necessary nuance, stated plainly rather than buried in a footnote**: this finding's *magnitude* is inflated by a real, disclosed data-quality limitation from the very first section of this project — pairtrade-lab-1's stand-in is a thin, 2-leg snapshot (short AXP / long WFC), not a fully diversified multi-pair book, because that's the most recent position pairtrade-lab-1's own backtest happened to hold when its snapshot was generated. A fuller multi-pair stand-in would likely show a smaller, though almost certainly still nonzero, version of this same effect. What doesn't depend on that snapshot's specific size is the *methodology's* answer: alpha-signal-lab, examined in isolation, shows a well-explained, sensible factor profile (R²=0.716, loadings matching its actual sector tilts) — the hidden risk found here is concentrated in how *one* strategy's current sizing interacts with the aggregate, not evidence that all six strategies are secretly correlated. Aggregation revealed a specific, real, and fixable risk — not a general indictment of the whole portfolio's diversification story, and the difference between those two conclusions is exactly what a risk desk needs a tool like this to tell it apart.

## Backend (`backend/`)

The FastAPI app in `backend/main.py` exposes positions and exposure, all five VaR methods, regime-conditional risk, static/Ledoit-Wolf and DCC-GARCH correlation, regime classification, factors, Greeks, historical/hypothetical/reverse stress, credit/CVA, EVT, liquidity, attribution, and the read-only live monitor. `backend/serialize.py` converts module-native dataclasses, pandas objects, and NumPy values into strict JSON-safe responses without coupling analytics return types to HTTP.

Run locally with:

```bash
uvicorn backend.main:app --reload
```

Interactive documentation is available at `/docs`, the OpenAPI contract at `/openapi.json`, and the deployment health probe at `/health`. CORS uses a comma-separated `ALLOWED_ORIGINS` environment variable; it defaults to `*` for local setup. The monitor route is intentionally read-only: resetting the persistent kill-switch remains a manual CLI action because the API has no authentication layer.

## Frontend (`frontend/`)

The Observable Framework site implements all ten Step 21 views: portfolio overview, VaR comparison with regime control, static/DCC and regime-conditional correlation, factors/PCA, historical and hypothetical stress, interactive reverse stress, counterparty/CVA, EVT and liquidity, attribution, and the live risk monitor. Slow analytics are captured by a build-time FastAPI data loader; live monitoring and non-default reverse-stress targets use runtime requests. If the API is unavailable during a static build, the site builds with an explicit unavailable-data state rather than publishing fabricated values.

```bash
cd frontend
npm install
npm run dev
```

Set `RISKDESK_API_URL` to the FastAPI origin. For production, set it to the Render backend URL in Vercel and set backend `ALLOWED_ORIGINS` to the Vercel site URL. `frontend/vercel.json` declares the static `dist` output; deployment itself is Step 22.
