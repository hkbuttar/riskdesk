export const money = (value) => value == null || !Number.isFinite(+value)
  ? "—" : new Intl.NumberFormat("en-US", {style: "currency", currency: "USD", maximumFractionDigits: 0}).format(value);

export const pct = (value, digits = 1) => value == null || !Number.isFinite(+value)
  ? "—" : `${(+value * 100).toFixed(digits)}%`;

export const number = (value, digits = 2) => value == null || !Number.isFinite(+value)
  ? "—" : new Intl.NumberFormat("en-US", {maximumFractionDigits: digits}).format(value);

export function rows(record = {}, key = "name") {
  return Object.entries(record || {}).map(([name, value]) => ({[key]: name, ...(value || {})}));
}

export function card(label, value, note = "", status = "") {
  const node = document.createElement("div");
  node.className = "risk-card";
  node.innerHTML = `<div class="risk-label">${label}</div><div class="risk-value ${status}">${value}</div>${note ? `<div class="risk-note">${note}</div>` : ""}`;
  return node;
}

export function warning(snapshot) {
  const failures = Object.entries(snapshot?.errors ?? {});
  if (snapshot?.available && failures.length === 0) return null;
  const node = document.createElement("div");
  node.className = "data-warning";
  node.textContent = snapshot?.error || `Partial snapshot: ${failures.map(([name, error]) => `${name} (${error})`).join("; ")}`;
  return node;
}

export function matrixRows(matrix = {}) {
  return Object.entries(matrix || {}).flatMap(([row, values]) =>
    Object.entries(values || {}).map(([column, value]) => ({row, column, value}))
  );
}
