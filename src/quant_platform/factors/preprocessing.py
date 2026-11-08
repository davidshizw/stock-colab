"""Cross-sectional factor preprocessing: winsorisation, Z-score normalisation, and sector neutralisation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorise_series(s: pd.Series, lower_q: float = 0.025, upper_q: float = 0.975) -> pd.Series:
    """Clip extreme values of a cross-sectional series at specified quantiles."""
    valid = s.dropna()
    if len(valid) < 3:
        return s.copy()
    low_val = float(valid.quantile(lower_q))
    high_val = float(valid.quantile(upper_q))
    return s.clip(lower=low_val, upper=high_val)


def zscore_series(s: pd.Series, winsorise: bool = True) -> pd.Series:
    """Compute cross-sectional Z-score after optional winsorisation."""
    work = winsorise_series(s) if winsorise else s.copy()
    valid = work.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=s.index)
    std = float(valid.std(ddof=1))
    if std < 1e-12 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    mean = float(valid.mean())
    return (work - mean) / std


def sector_neutralise_series(
    values: pd.Series, sectors: pd.Series, rescale: bool = True
) -> pd.Series:
    """Demean factor values within each GICS sector and optionally re-standardise."""
    df = pd.DataFrame({"val": winsorise_series(values), "sector": sectors})
    sector_means = df.groupby("sector")["val"].transform("mean")
    demeaned = df["val"] - sector_means
    if not rescale:
        return demeaned
    return zscore_series(demeaned, winsorise=False)


def compute_factor_correlation_matrix(
    factor_df: pd.DataFrame, factor_cols: list[str]
) -> pd.DataFrame:
    """Compute Spearman rank correlation matrix across specified factor columns."""
    available = [c for c in factor_cols if c in factor_df.columns]
    if not available:
        return pd.DataFrame()
    return factor_df[available].corr(method="spearman").round(4)
