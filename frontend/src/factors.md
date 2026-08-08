---
title: Factor decomposition
---

```js
import {number, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
const portfolio = snapshot.factors?.portfolio ?? {};
const loadings = Object.entries(portfolio.loadings ?? {}).map(([factor, loading]) => ({factor, loading}));
const pca = Object.entries(snapshot.factors?.pca?.explained_variance_ratio ?? {}).map(([component, variance]) => ({component, variance}));
```

<div class="eyebrow">What drives the book</div>
# Factor decomposition

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
display(html`<div class="section-grid"><div class="risk-card"><div class="risk-label">Named-factor R²</div><div class="risk-value">${number(portfolio.r_squared, 3)}</div></div><div class="risk-card"><div class="risk-label">PCs explaining 90%</div><div class="risk-value">${snapshot.factors?.pca?.n_components_for_90pct ?? "—"}</div></div></div>`);
```

## Named-factor loadings

```js
Plot.plot({height: 330, marginLeft: 110, x: {grid: true}, color: {type: "diverging", domain: [-1, 1], scheme: "RdBu"}, marks: [Plot.barX(loadings, {y: "factor", x: "loading", fill: "loading", tip: true}), Plot.ruleX([0])]})
```

## Statistical factors

```js
Plot.plot({height: 280, y: {percent: true, grid: true, label: "Explained variance"}, marks: [Plot.barY(pca, {x: "component", y: "variance", fill: "#40d9c4", tip: true}), Plot.ruleY([0])]})
```
