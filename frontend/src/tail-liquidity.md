---
title: Tail risk & liquidity
---

```js
import {money, number, warning} from "./components.js";
const snapshot = FileAttachment("data/riskdesk.json").json();
const liquidity = snapshot.liquidity ?? {};
const tails = Object.entries(snapshot.extremeValue?.tail_shape_by_regime ?? {}).map(([regime, value]) => ({regime, ...value}));
```

<div class="eyebrow">Beyond ordinary volatility</div>
# Tail risk & liquidity

```js
display(warning(snapshot));
display(html`<div class="section-grid"><div class="risk-card"><div class="risk-label">Base VaR</div><div class="risk-value">${money(liquidity.base_var)}</div></div><div class="risk-card"><div class="risk-label">Liquidity-adjusted VaR</div><div class="risk-value">${money(liquidity.liquidity_adjusted_var)}</div></div><div class="risk-card"><div class="risk-label">Unwind premium</div><div class="risk-value">${money((liquidity.liquidity_adjusted_var ?? 0) - (liquidity.base_var ?? 0))}</div></div></div>`);
```

## EVT tail shape by regime

```js
Plot.plot({height: 320, y: {grid: true, label: "GPD shape ξ"}, marks: [Plot.barY(tails, {x: "regime", y: "xi", fill: "#ffbd59", tip: true}), Plot.ruleY([0])]})
```

```js
Inputs.table(tails.map(d => ({regime: d.regime, shape_xi: number(d.xi, 3), scale_beta: money(d.beta), exceedances: d.n_exceedances, threshold: money(d.threshold)})))
```
