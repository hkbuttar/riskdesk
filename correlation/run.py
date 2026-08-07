"""Correlation & covariance estimation: static vs. Ledoit-Wolf-shrunk vs.
DCC-GARCH, compared on the real book's risk factors.

    python -m correlation.run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aggregation.valuation import value_positions
from connectors.registry import fetch_all
from correlation.dcc_garch import fit_dcc_garch
from correlation.shrinkage import ledoit_wolf_covariance
from correlation.static import static_correlation
from risk_measures.returns import fetch_return_history, position_risk_factor


def main() -> None:
    raw_positions, _ = fetch_all()
    valued = value_positions(raw_positions)
    tickers = sorted({position_risk_factor(p) for p in valued.positions if position_risk_factor(p)})
    returns_df, notes = fetch_return_history(tickers)
    for note in notes:
        print(f"  {note}")
    print()

    static_corr = static_correlation(returns_df)

    print("=== Static (whole-window) correlation ===")
    print(static_corr.round(2))

    print("\n=== Ledoit-Wolf shrinkage vs. raw sample covariance ===")
    lw = ledoit_wolf_covariance(returns_df)
    print(f"  shrinkage constant: {lw.shrinkage:.4f}")
    print(f"  sample covariance condition number: {lw.sample_condition_number:,.1f}")
    print(f"  shrunk covariance condition number:  {lw.shrunk_condition_number:,.1f}")
    print(f"  -> {lw.sample_condition_number / lw.shrunk_condition_number:.1f}x better conditioned")

    print("\n=== DCC-GARCH ===")
    dcc = fit_dcc_garch(returns_df)
    print(f"  {dcc.notes}")
    latest_dcc = dcc.latest_correlation()
    print("\n  Latest (most recent day) DCC correlation:")
    print(latest_dcc.round(2))

    print("\n=== Static vs. DCC-GARCH: where do they disagree most? ===")
    diff = (latest_dcc - static_corr).abs()
    diff_values = diff.to_numpy().copy()
    np.fill_diagonal(diff_values, 0.0)
    diff = pd.DataFrame(diff_values, index=diff.index, columns=diff.columns)
    pairs = []
    for i, ticker_a in enumerate(dcc.tickers):
        for ticker_b in dcc.tickers[i + 1 :]:
            pairs.append((ticker_a, ticker_b, diff.loc[ticker_a, ticker_b]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    print("  Top 5 pairs by |latest DCC corr - static corr|:")
    for ticker_a, ticker_b, d in pairs[:5]:
        series = dcc.correlation_series(ticker_a, ticker_b)
        print(
            f"    {ticker_a:8s}-{ticker_b:8s}  static={static_corr.loc[ticker_a, ticker_b]:+.2f}  "
            f"DCC(latest)={series.iloc[-1]:+.2f}  |diff|={d:.2f}  DCC(min..max over window)="
            f"{series.min():+.2f}..{series.max():+.2f}"
        )

    print(
        "\n  -> static correlation is one fixed number for the whole window; DCC-GARCH's "
        "min..max range above is exactly the correlation movement a static estimate cannot see. "
        "Validating this against a real historical stress window (e.g. a sharp drawdown) is "
        "future work once the historical-scenario-replay module exists -- see README."
    )


if __name__ == "__main__":
    main()
