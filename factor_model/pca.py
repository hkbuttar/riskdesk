"""PCA-based statistical factors: a model-free complement to the named
factor regression. Named factors (SPY, sector ETFs, BTC-USD) are a
hypothesis about what drives this book's risk; PCA makes no such
hypothesis -- it just finds the directions of maximum variance in the
actual held-asset returns, so it's a useful check for whether the named
model is missing something (a PC that explains real variance but doesn't
load cleanly on any named factor) or is a redundant echo of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    explained_variance_ratio: pd.Series  # per component
    cumulative_variance_ratio: pd.Series
    loadings: pd.DataFrame  # assets x components
    n_components_for_90pct: int


def fit_pca(returns_df: pd.DataFrame) -> PCAResult:
    standardized = (returns_df - returns_df.mean()) / returns_df.std()
    pca = PCA()
    pca.fit(standardized)

    n_components = len(returns_df.columns)
    component_names = [f"PC{i + 1}" for i in range(n_components)]
    explained = pd.Series(pca.explained_variance_ratio_, index=component_names)
    cumulative = explained.cumsum()
    n_for_90 = int((cumulative < 0.90).sum() + 1)

    loadings = pd.DataFrame(pca.components_.T, index=returns_df.columns, columns=component_names)

    return PCAResult(
        explained_variance_ratio=explained,
        cumulative_variance_ratio=cumulative,
        loadings=loadings,
        n_components_for_90pct=n_for_90,
    )


def top_loadings(pca_result: PCAResult, component: str, n: int = 5) -> pd.Series:
    return pca_result.loadings[component].abs().sort_values(ascending=False).head(n)
