---
title: Counterparty & credit risk
---

```js
import {money, pct, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
const credit = snapshot.credit ?? {};
const cva = credit.cva ?? {};
const entries = Object.entries(cva.by_counterparty ?? {}).map(([counterparty, value]) => ({counterparty, ...value}));
const concentration = credit.concentration ?? {};
const largestShare = Math.max(0, ...Object.values(concentration.shares ?? {}));
```

<div class="eyebrow">Who holds the assets</div>
# Counterparty & credit risk

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
display(html`<div class="section-grid"><div class="risk-card"><div class="risk-label">Total CVA</div><div class="risk-value">${money(cva.total_cva)}</div></div><div class="risk-card"><div class="risk-label">Largest venue share</div><div class="risk-value ${concentration.flagged?.length ? "status-breach" : "status-ok"}">${pct(largestShare)}</div></div><div class="risk-card"><div class="risk-label">Concentration status</div><div class="risk-value ${concentration.flagged?.length ? "status-breach" : "status-ok"}">${concentration.flagged?.length ? "BREACH" : "WITHIN LIMIT"}</div></div></div>`);
```

```js
Plot.plot({height: 320, marginLeft: 100, x: {grid: true, label: "Expected loss / CVA ($)"}, marks: [Plot.barX(entries, {y: "counterparty", x: "cva", fill: "#ffbd59", tip: true})]})
```

```js
Inputs.table(entries.map(d => ({counterparty: d.counterparty, net_exposure: money(d.net_exposure), PD: pct(d.annualized_pd), LGD: pct(cva.lgd_used), CVA: money(d.cva)})))
```
