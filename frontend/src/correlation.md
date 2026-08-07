---
title: Correlation
---

```js
import {matrixRows, warning} from "./components.js";
const snapshot = FileAttachment("data/riskdesk.json").json();
const model = view(Inputs.radio(["Static", "DCC-GARCH latest"], {label: "Estimator", value: "Static"}));
const matrix = model === "Static" ? snapshot.staticCorrelation?.correlation : snapshot.dcc?.latest_correlation;
const cells = matrixRows(matrix);
```

<div class="eyebrow">Dependence structure</div>
# Correlation

Static history versus the latest time-varying DCC-GARCH estimate.

```js
display(warning(snapshot));
Plot.plot({height: 520, marginLeft: 90, x: {label: null}, y: {label: null}, color: {type: "diverging", domain: [-1, 1], scheme: "RdBu", legend: true}, marks: [Plot.cell(cells, {x: "column", y: "row", fill: "value", inset: 1, tip: true}), Plot.text(cells, {x: "column", y: "row", text: d => d.value?.toFixed(2), fill: d => Math.abs(d.value) > .55 ? "white" : "currentColor"})]})
```

## Regime-conditional matrices

```js
const conditionalRegime = view(Inputs.select(Object.keys(snapshot.conditional?.conditional_correlation ?? {}), {label: "Regime"}));
const conditionalCells = matrixRows(snapshot.conditional?.conditional_correlation?.[conditionalRegime]);
Plot.plot({height: 420, marginLeft: 90, color: {type: "diverging", domain: [-1, 1], scheme: "RdBu"}, marks: [Plot.cell(conditionalCells, {x: "column", y: "row", fill: "value", tip: true})]})
```
