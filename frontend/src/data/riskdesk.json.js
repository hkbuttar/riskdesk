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

try {
  const entries = await Promise.all(Object.entries(endpoints).map(load));
  process.stdout.write(JSON.stringify({available: true, generatedAt: new Date().toISOString(), apiBase: base, ...Object.fromEntries(entries)}));
} catch (error) {
  process.stdout.write(JSON.stringify({available: false, generatedAt: new Date().toISOString(), apiBase: base, error: `RiskDesk API snapshot unavailable: ${error.message}`}));
}
