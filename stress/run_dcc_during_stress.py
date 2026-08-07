"""Static vs. DCC-GARCH correlation during the three real historical crisis
windows -- closes the "future work" gap flagged in both correlation.py and
regime/conditional.py's README sections.

    python -m stress.run_dcc_during_stress
"""

from __future__ import annotations

from aggregation.pricing import resolve_symbol
from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from risk_measures.returns import position_risk_factor
from stress.dcc_during_stress import compare_during_stress_windows
from stress.historical import HISTORICAL_WINDOWS, fetch_price_history

PAIRS = [("CVX", "PSX"), ("CVX", "VLO"), ("AXP", "WFC"), ("SPY", "BTC-USD"), ("CVX", "SPY")]


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    risk_factors = {position_risk_factor(p) for p in valued.positions if position_risk_factor(p)}
    tickers = sorted({resolve_symbol(f) for f in risk_factors} | {"SPY"})

    earliest = min(w["pre_start"] for w in HISTORICAL_WINDOWS.values())
    latest = max(w["end"] for w in HISTORICAL_WINDOWS.values())
    print(f"Fetching {tickers} from {earliest} to {latest} (this covers all three crisis windows)...\n")
    price_history = fetch_price_history(tickers, earliest, latest)

    results = compare_during_stress_windows(price_history, PAIRS)
    for r in results:
        spike = r.dcc_mean_during_window - r.dcc_pre_window_mean
        print(f"{r.window_name:16s} {r.pair[0]:8s}-{r.pair[1]:8s}  "
              f"static={r.static_corr:+.2f}  DCC(pre-window)={r.dcc_pre_window_mean:+.2f}  "
              f"DCC(during)={r.dcc_mean_during_window:+.2f}  DCC(max during)={r.dcc_max_during_window:+.2f}  "
              f"spike={spike:+.2f}")


if __name__ == "__main__":
    main()
