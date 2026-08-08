---
title: P&L attribution
---

```js
import {money, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
const attribution = snapshot.attribution ?? {};
const strategy = Object.entries(attribution.by_strategy ?? {}).map(([source, pnl]) => ({source, pnl, type: "Strategy"}));
const byFactor = attribution.by_factor?.cumulative_by_factor ?? {};
const factor = Object.entries(byFactor).map(([source, pnl]) => ({source, pnl, type: "Factor"}));
```

<div class="eyebrow">Realized drivers</div>
# P&L attribution

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
display(Plot.plot({height: 430, marginLeft: 130, x: {grid: true, label: "Cumulative P&L ($)"}, facet: {data: [...strategy, ...factor], y: "type"}, marks: [Plot.barX([...strategy, ...factor], {y: "source", x: "pnl", fy: "type", fill: d => d.pnl < 0 ? "#ff667a" : "#40d9c4", tip: true}), Plot.ruleX([0])]}));
```

```js
Inputs.table([...strategy, ...factor].map(d => ({view: d.type, source: d.source, cumulative_pnl: money(d.pnl)})))
```
