const apiUrl = (process.env.RISKDESK_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const serializedApiUrl = JSON.stringify(apiUrl).replaceAll("<", "\\u003c");

export default {
  title: "RiskDesk",
  root: "src",
  output: "dist",
  theme: ["dashboard", "wide"],
  style: "style.css",
  pager: false,
  pages: [
    {name: "Portfolio", path: "/"},
    {name: "VaR comparison", path: "/var"},
    {name: "Correlation", path: "/correlation"},
    {name: "Factors", path: "/factors"},
    {name: "Stress testing", path: "/stress"},
    {name: "Reverse stress", path: "/reverse-stress"},
    {name: "Credit risk", path: "/credit"},
    {name: "Tail & liquidity", path: "/tail-liquidity"},
    {name: "Attribution", path: "/attribution"},
    {name: "Live monitor", path: "/monitor"}
  ],
  header: `<div class="risk-header"><span>RISKDESK</span><span id="api-connection" class="api-connection">API CHECKING</span></div>`,
  head: `<meta name="description" content="Aggregate portfolio market and credit risk across six strategies.">
    <script>
      globalThis.RISKDESK_API_URL=${serializedApiUrl};
      addEventListener("DOMContentLoaded", async () => {
        const indicator = document.querySelector("#api-connection");
        if (!indicator) return;
        try {
          const response = await fetch(globalThis.RISKDESK_API_URL + "/health", {cache: "no-store"});
          if (!response.ok) throw new Error("HTTP " + response.status);
          const health = await response.json();
          indicator.textContent = health.status === "ok" ? "API CONNECTED" : "API DEGRADED";
          indicator.dataset.state = health.status === "ok" ? "ok" : "error";
        } catch (error) {
          indicator.textContent = "API OFFLINE";
          indicator.dataset.state = "error";
          indicator.title = error.message;
        }
      });
    </script>`
};
