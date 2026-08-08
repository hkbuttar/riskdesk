---
title: Portfolio overview
---

```js
import {card, money, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
const exposure = snapshot.exposure ?? {};
const portfolio = exposure.portfolio ?? {};
const regime = snapshot.regime?.tercile?.current_regime ?? "unknown";
const credit = snapshot.credit?.cva ?? {};
```

<div class="hero">
  <div class="eyebrow">Aggregate risk / six strategies</div>
  <h1>Portfolio overview</h1>
  <p class="lede">One consolidated view of market, factor, liquidity and counterparty risk. Figures are generated from the same module-native calculations exposed by FastAPI.</p>
</div>

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
display(html`<div class="section-grid">
  ${card("Gross exposure", money(portfolio.gross_market_value), `${portfolio.n_positions ?? 0} positions`)}
  ${card("Net exposure", money(portfolio.net_market_value))}
  ${card("Current regime", regime, "SPY volatility tercile", regime === "volatile" ? "status-breach" : "status-ok")}
  ${card("Portfolio CVA", money(credit.total_cva), "Assumption-based PD tiers")}
</div>`);
```

## Strategy exposure

```js
const strategyRows = Object.entries(exposure.by_strategy ?? {}).map(([strategy, value]) => ({strategy, gross: value.gross_market_value, net: value.net_market_value}));
display(Plot.plot({height: 320, marginLeft: 130, x: {label: "Gross market value ($)", grid: true}, y: {label: null}, marks: [Plot.barX(strategyRows, {y: "strategy", x: "gross", fill: "#40d9c4", sort: {y: "-x"}}), Plot.ruleX([0])]}));
```

<span class="risk-note">Snapshot generated ${snapshot.generatedAt ?? "—"}</span>
