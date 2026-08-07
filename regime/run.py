"""Regime classification on real SPY data: volatility terciles (hard
labels, reusing execedge's methodology) vs. a Markov-switching model
(probabilistic labels).

    python -m regime.run
"""

from __future__ import annotations

import yfinance as yf

from regime.hmm_regime import fit_hmm_regimes
from regime.volatility_tercile import classify_regimes, rolling_realized_vol

REFERENCE_TICKER = "SPY"


def main() -> None:
    close = yf.Ticker(REFERENCE_TICKER).history(period="2y")["Close"]
    close.index = close.index.tz_localize(None)
    print(f"Reference series: {REFERENCE_TICKER}, {len(close)} daily closes.\n")

    print("=== Volatility-tercile regimes (execedge methodology, daily adaptation) ===")
    vol = rolling_realized_vol(close)
    tercile = classify_regimes(vol)
    print(f"  thresholds: {tercile.thresholds}")
    print(f"  distribution:\n{tercile.value_counts()}")
    print(f"  current regime: {tercile.current()}\n")

    print("=== Markov-switching (HMM) regimes ===")
    hmm = fit_hmm_regimes(close)
    print(f"  {hmm.notes}")
    print(f"  fitted regime params: {hmm.regime_params}")
    print(f"  hard-label distribution:\n{hmm.hard_labels.value_counts()}")
    print(f"  current probabilities:\n{hmm.current_probabilities()}\n")

    print("=== Agreement between the two methods ===")
    aligned = tercile.labels.dropna().to_frame("tercile").join(
        hmm.hard_labels.to_frame("hmm"), how="inner"
    )
    agree = (aligned["tercile"] == aligned["hmm"]).mean()
    print(f"  {len(aligned)} overlapping days, {agree:.1%} label agreement.")
    print(f"  disagreement crosstab:\n{aligned.groupby(['tercile', 'hmm']).size().unstack(fill_value=0)}")


if __name__ == "__main__":
    main()
