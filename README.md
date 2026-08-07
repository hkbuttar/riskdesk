# RiskDesk — Portfolio Risk & Stress-Testing Capstone
Portfolio risk aggregation across six strategies. VaR/CVaR, DCC-GARCH correlation, factor decomposition, regime-conditional models, historical + reverse stress testing, EVT tail risk, counterparty credit risk, and live monitoring. Observable Framework dashboard. The capstone risk layer.

## Status

**Environment & data acquisition — done. Position & exposure aggregation — done. VaR/CVaR risk measures — done. Correlation & covariance estimation — done. Regime classification — done. Regime-conditional risk models — done.** Factor decomposition, stress testing, credit risk, tail risk, liquidity, attribution, live monitoring, backend, and frontend are not yet started.

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

`pytest` (`tests/test_aggregation.py`) — unit tests use constructed positions with a stubbed price fetch for determinism (real market prices change daily); one end-to-end test runs against real connectors and real Yahoo Finance prices, asserting structural invariants only.

## VaR/CVaR — Five Methods, Compared (`risk_measures/`)

- **`returns.py`** — for every position, its *risk factor* is the asset itself (equity/crypto) or, for voledge's options, `extra["underlying"]` (SPY) — since options are already carried as delta-equivalent dollar exposure (see aggregation/valuation.py), `market_value * underlying_return` approximates each option position's daily P&L to first order (delta-only, ignoring gamma/theta — a disclosed simplification, not a new one introduced here). Two years of daily history is pulled for every distinct risk factor via Yahoo Finance; days where any held risk factor lacks a price (mostly equity-only weekends/holidays vs. crypto trading every day) are dropped rather than forward-filled. This produces one hypothetical daily portfolio $ P&L series — today's position sizes applied to each historical day's return — reused identically by all five VaR methods so the comparison is apples-to-apples.
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

`pytest` (`tests/test_conditional.py`) — alignment/partitioning logic (no forward-fill across unlabeled dates, correct bucket membership, the min-days skip rule) is checked exactly. VaR and correlation shift-detection are checked by injecting a real, controllable regime-dependent variance/correlation difference into synthetic data and confirming the conditional estimates recover the correct direction. One end-to-end test runs the full pipeline on the real book. 47 tests passing total across the project.
