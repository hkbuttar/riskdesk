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
  head: `<meta name="description" content="Aggregate portfolio market and credit risk across six strategies.">
    <script>globalThis.RISKDESK_API_URL=${JSON.stringify(process.env.RISKDESK_API_URL || "http://127.0.0.1:8000")};</script>`
};
