---
title: Correlation
---

```js
import {matrixRows, warning} from "./components.js";
const snapshot = FileAttachment("data/riskdesk.json").json();
const history = snapshot.dcc?.correlation_history ?? [];
const dates = snapshot.dcc?.dates ?? [];
const tickers = snapshot.dcc?.tickers ?? [];
const lastIndex = Math.max(0, history.length - 1);
const model = view(Inputs.radio(["Static", "DCC-GARCH through time"], {label: "Estimator", value: "Static"}));
const timeIndex = view(Inputs.range([0, lastIndex], {label: "DCC date / stress-window slider", value: lastIndex, step: 1}));
const historyMatrix = history[timeIndex]
  ? Object.fromEntries(tickers.map((row, i) => [row, Object.fromEntries(tickers.map((column, j) => [column, history[timeIndex][i][j]]))]))
  : snapshot.dcc?.latest_correlation;
const matrix = model === "Static" ? snapshot.staticCorrelation?.correlation : historyMatrix;
const cells = matrixRows(matrix);
const selectedDate = dates[timeIndex]?.slice(0, 10) ?? "latest";
const stressWindow = selectedDate >= "2020-02-19" && selectedDate <= "2020-03-23" ? "COVID crash"
  : selectedDate >= "2022-01-03" && selectedDate <= "2022-10-12" ? "2022 rate-hike shock"
  : selectedDate >= "2022-11-06" && selectedDate <= "2022-11-14" ? "FTX collapse" : "Outside named stress windows";
```

<div class="eyebrow">Dependence structure</div>
# Correlation

Static history versus the time-varying DCC-GARCH estimate. Move the slider across the historical sample to inspect correlation during named stress windows.

```js
display(html`<div class="risk-card"><div class="risk-label">Selected DCC observation</div><div class="risk-value">${selectedDate}</div><div class="risk-note">${stressWindow}</div></div>`);
```

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
