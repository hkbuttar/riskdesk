# RiskDesk — Portfolio Risk & Stress-Testing Capstone
Portfolio risk aggregation across six strategies. VaR/CVaR, DCC-GARCH correlation, factor decomposition, regime-conditional models, historical + reverse stress testing, EVT tail risk, counterparty credit risk, and live monitoring. Observable Framework dashboard. The capstone risk layer.

## Status

**Environment & data acquisition — done. Position & exposure aggregation — done. VaR/CVaR risk measures — done. Correlation & covariance estimation — done. Regime classification — done. Regime-conditional risk models — done. Factor risk decomposition — done. Greeks & options risk aggregation — done. Historical scenario replay — done.** Hypothetical stress scenarios, reverse stress testing, credit risk, tail risk, liquidity, attribution, live monitoring, backend, and frontend are not yet started.

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

`pytest` (`tests/test_historical_stress.py`) — replay math and the diversification-erosion ratio are checked against constructed positions/price series with exact hand-computable expected values (including the textbook sqrt(2) erosion ratio for two perfectly correlated equal-vol strategies, and a ratio of exactly 0 for perfectly offsetting ones). VaR breach counting is checked against a synthetic window with an injected, known number of extreme-loss days. One end-to-end test runs the real book against a real disclosed historical window. 73 tests passing total across the project.
