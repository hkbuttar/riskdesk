---
title: VaR comparison
---

```js
import {money, rows, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
```

```js
const regimeChoice = view(Inputs.select(["pooled", ...Object.keys(snapshot.conditional?.conditional_var ?? {})], {label: "Risk regime", value: "pooled"}));
```

```js
const selected = regimeChoice === "pooled" ? snapshot.conditional?.pooled_var : snapshot.conditional?.conditional_var?.[regimeChoice];
const results = rows(selected ?? snapshot.var, "method");
```

<div class="eyebrow">Distributional risk</div>
# VaR comparison

Compare model assumptions directly and switch between the pooled history and regime-conditioned estimates.

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
display(Plot.plot({height: 360, marginLeft: 145, x: {grid: true, label: "1-day loss ($)"}, y: {label: null}, color: {legend: true}, marks: [Plot.barX(results.flatMap(d => [{...d, metric: "VaR", value: d.var_dollar}, {...d, metric: "CVaR", value: d.cvar_dollar}]), {y: "method", x: "value", fill: "metric", fx: "metric", tip: true})]}));
```

```js
Inputs.table(results.map(d => ({method: d.method, confidence: d.confidence, VaR: money(d.var_dollar), CVaR: money(d.cvar_dollar), notes: d.notes})), {columns: ["method", "confidence", "VaR", "CVaR", "notes"]})
```
