---
title: Live risk monitor
---

```js
import {card, money} from "./components.js";
const fallbackBase = "http://127.0.0.1:8000";
const apiBase = globalThis.RISKDESK_API_URL ?? fallbackBase;
const refresh = view(Inputs.button("Refresh live risk", {value: 0, reduce: value => value + 1}));
const status = await fetch(`${apiBase}/api/monitor?refresh=${refresh}`, {cache: "no-store"})
  .then(r => {if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json();})
  .catch(error => ({error: error.message}));
```

<div class="eyebrow">Runtime data / no static snapshot</div>
# Live risk monitor

This page requests FastAPI directly on every refresh. The kill-switch is read-only here and still requires a deliberate CLI reset.

```js
if (status.error) display(html`<div class="data-warning">Live backend unavailable at ${apiBase}: ${status.error}</div>`);
display(html`<div class="section-grid">
  ${card("Current regime", status.current_regime ?? "—", status.active_model ?? "", status.current_regime === "volatile" ? "status-breach" : "status-ok")}
  ${card("Active VaR", money(status.active_var), `Gross ${money(status.gross_exposure)}`)}
  ${card("Market-risk limit", status.var_check?.breached ? "BREACH" : "OK", status.var_check?.detail ?? "", status.var_check?.breached ? "status-breach" : "status-ok")}
  ${card("Credit-risk limit", status.credit_check?.breached ? "BREACH" : "OK", status.credit_check?.detail ?? "", status.credit_check?.breached ? "status-breach" : "status-ok")}
  ${card("Kill-switch", status.kill_switch_triggered ? "TRIGGERED" : "ARMED", (status.kill_switch_reasons ?? []).join(" · "), status.kill_switch_triggered ? "status-breach" : "status-ok")}
</div>`);
```
