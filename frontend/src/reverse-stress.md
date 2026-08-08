---
title: Reverse stress testing
---

```js
import {money, number, warning} from "./components.js";
const snapshot = await FileAttachment("data/riskdesk.json").json();
```

```js
// The input widget lives in its own cell -- Observable's reactive graph
// only correctly tracks a view() binding's changes (and has its resolved
// value ready synchronously for downstream cells) when it isn't mixed
// into a larger cell with other unrelated logic. Combining them caused
// `target` to read as undefined (Number(undefined) = NaN) on first run.
const target = view(Inputs.range([0.05, 0.50], {label: "Target portfolio loss (as % of gross exposure)", value: 0.25, step: 0.05}));
```

```js
const apiBase = snapshot.apiBase ?? "http://127.0.0.1:8000";
const targetFraction = Number(target);
const result = Number.isFinite(targetFraction) && Math.abs(targetFraction - 0.25) < 1e-9 && snapshot.reverseStress
  ? snapshot.reverseStress
  : await fetch(`${apiBase}/api/reverse-stress?target_fraction=${targetFraction}`).then(r => {if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json();}).catch(error => ({error: error.message}));
const solved = result.solved_scenario ?? {};
const shocks = Object.entries(solved.factor_shocks ?? {}).map(([factor, shock]) => ({factor, shock}));
```

<div class="eyebrow">Solve backward from failure</div>
# Reverse stress testing

Choose a catastrophic loss target. RiskDesk solves for the minimum-covariance-distance factor scenario that reaches it, then assesses plausibility.

```js
const snapshotWarning = warning(snapshot);
if (snapshotWarning) display(snapshotWarning);
if (result.error) display(html`<div class="data-warning">Runtime request failed: ${result.error}</div>`);
display(html`<div class="section-grid"><div class="risk-card"><div class="risk-label">Target loss</div><div class="risk-value">${money(solved.target_loss ?? result.target_loss)}</div></div><div class="risk-card"><div class="risk-label">Mahalanobis distance</div><div class="risk-value">${number(solved.mahalanobis_distance, 2)}σ</div></div><div class="risk-card"><div class="risk-label">Horizon</div><div class="risk-value">${solved.horizon_days ?? "—"} days</div></div></div>`);
```

```js
Plot.plot({height: 340, marginLeft: 110, x: {percent: true, grid: true, label: "Solved factor shock"}, marks: [Plot.barX(shocks, {y: "factor", x: "shock", fill: d => d.shock < 0 ? "#ff667a" : "#40d9c4", tip: true}), Plot.ruleX([0])]})
```
