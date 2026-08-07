---
title: Reverse stress testing
---

```js
import {money, number, warning} from "./components.js";
const snapshot = FileAttachment("data/riskdesk.json").json();
const target = view(Inputs.range([0.05, 0.50], {label: "Target portfolio loss", value: 0.25, step: 0.05, transform: x => `${Math.round(x * 100)}%`}));
const apiBase = snapshot.apiBase ?? "http://127.0.0.1:8000";
const result = target === 0.25 && snapshot.reverseStress
  ? snapshot.reverseStress
  : await fetch(`${apiBase}/api/reverse-stress?target_fraction=${target}`).then(r => {if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json();}).catch(error => ({error: error.message}));
const solved = result.solved_scenario ?? {};
const shocks = Object.entries(solved.factor_shocks ?? {}).map(([factor, shock]) => ({factor, shock}));
```

<div class="eyebrow">Solve backward from failure</div>
# Reverse stress testing

Choose a catastrophic loss target. RiskDesk solves for the minimum-covariance-distance factor scenario that reaches it, then assesses plausibility.

```js
display(warning(snapshot));
if (result.error) display(html`<div class="data-warning">Runtime request failed: ${result.error}</div>`);
display(html`<div class="section-grid"><div class="risk-card"><div class="risk-label">Target loss</div><div class="risk-value">${money(solved.target_loss ?? result.target_loss)}</div></div><div class="risk-card"><div class="risk-label">Mahalanobis distance</div><div class="risk-value">${number(solved.mahalanobis_distance, 2)}σ</div></div><div class="risk-card"><div class="risk-label">Horizon</div><div class="risk-value">${solved.horizon_days ?? "—"} days</div></div></div>`);
```

```js
Plot.plot({height: 340, marginLeft: 110, x: {percent: true, grid: true, label: "Solved factor shock"}, marks: [Plot.barX(shocks, {y: "factor", x: "shock", fill: d => d.shock < 0 ? "#ff667a" : "#40d9c4", tip: true}), Plot.ruleX([0])]})
```
