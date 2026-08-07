---
title: Stress testing
---

```js
import {money, warning} from "./components.js";
const snapshot = FileAttachment("data/riskdesk.json").json();
const historical = Object.entries(snapshot.historicalStress ?? {}).map(([scenario, value]) => ({scenario, ...(value.result ?? {})}));
const hypothetical = Object.entries(snapshot.hypotheticalStress ?? {}).map(([scenario, value]) => ({scenario, ...value}));
```

<div class="eyebrow">Forward scenarios</div>
# Historical + hypothetical stress

```js
display(warning(snapshot));
```

## Historical replay

```js
Plot.plot({height: 320, marginLeft: 145, x: {grid: true, label: "Total P&L ($)"}, marks: [Plot.barX(historical, {y: "scenario", x: "total_pnl", fill: d => d.total_pnl < 0 ? "#ff667a" : "#40d9c4", tip: true}), Plot.ruleX([0])]})
```

## Hypothetical multi-factor shocks

```js
Inputs.table(hypothetical.map(d => ({scenario: d.scenario, total_pnl: money(d.total_pnl), portfolio_return: d.portfolio_return_pct == null ? "—" : `${(100 * d.portfolio_return_pct).toFixed(2)}%`, notes: d.notes})))
```
