const base = (process.env.RISKDESK_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const endpoints = {
  positions: "/api/positions", exposure: "/api/exposure", var: "/api/var",
  conditional: "/api/var/regime-conditional", staticCorrelation: "/api/correlation/static",
  dcc: "/api/correlation/dcc-garch", regime: "/api/regime", factors: "/api/factors",
  greeks: "/api/greeks", historicalStress: "/api/stress/historical",
  hypotheticalStress: "/api/stress/hypothetical", reverseStress: "/api/reverse-stress",
  credit: "/api/credit", extremeValue: "/api/extreme-value", liquidity: "/api/liquidity",
  attribution: "/api/attribution"
};

async function load([name, path]) {
  const response = await fetch(`${base}${path}`, {signal: AbortSignal.timeout(120_000)});
  if (!response.ok) throw new Error(`${name}: HTTP ${response.status}`);
  return [name, await response.json()];
}

const requested = Object.entries(endpoints);
const settled = await Promise.allSettled(requested.map(load));
const values = {};
const errors = {};
settled.forEach((result, index) => {
  const name = requested[index][0];
  if (result.status === "fulfilled") values[name] = result.value[1];
  else errors[name] = result.reason?.message || String(result.reason);
});
const available = Object.keys(values).length > 0;
process.stdout.write(JSON.stringify({
  available, generatedAt: new Date().toISOString(), apiBase: base,
  loaded: Object.keys(values), errors,
  error: available ? null : "RiskDesk API unavailable; no snapshot endpoints could be loaded.",
  ...values
}));
