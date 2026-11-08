"""Unit tests for the Quantitative Factor Engine, preprocessing, screener, and IC evaluation."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FACTOR_CATEGORIES, FactorEngine
from quant_platform.factors.evaluation import FactorEvaluator
from quant_platform.factors.preprocessing import (
    sector_neutralise_series,
    winsorise_series,
    zscore_series,
)
from quant_platform.screener.ranker import EquityScreener


def test_winsorise_zscore_and_sector_neutralise() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    w = winsorise_series(s, lower_q=0.1, upper_q=0.8)
    assert w.max() < 100.0

    z = zscore_series(s)
    assert abs(float(z.mean())) < 1e-6
    assert abs(float(z.std(ddof=1)) - 1.0) < 1e-6

    sectors = pd.Series(["Tech", "Tech", "Bank", "Bank", "Bank"])
    sn = sector_neutralise_series(pd.Series([10.0, 20.0, 1.0, 2.0, 3.0]), sectors, rescale=False)
    assert abs(float(sn.iloc[:2].mean())) < 1e-6
    assert abs(float(sn.iloc[2:].mean())) < 1e-6


def test_factor_engine_all_categories(populated_store: MarketDataStore) -> None:
    engine = FactorEngine(populated_store)
    df = engine.compute_factors_as_of(as_of_date=date(2026, 11, 6), market="ALL")
    assert not df.empty
    assert len(df) >= 15

    for cat, cols in FACTOR_CATEGORIES.items():
        for col in cols:
            assert col in df.columns, f"Missing factor {col} from category {cat}"

    # Verify ASX Franking Credit gross-up yield > raw dividend yield for 100% franked ASX stocks
    bhp_row = df[df["symbol"] == "BHP.AX"].iloc[0]
    assert bhp_row["franking_grossed_yield"] > bhp_row["dividend_yield"]

    # Verify US Dividend Withholding Tax reduces net yield below raw dividend yield
    jpm_row = df[df["symbol"] == "JPM"].iloc[0]
    assert jpm_row["franking_grossed_yield"] < jpm_row["dividend_yield"]


def test_screener_and_factor_evaluator(populated_store: MarketDataStore) -> None:
    screener = EquityScreener(populated_store)
    asx_rows = screener.screen(as_of_date=date(2026, 11, 6), market="ASX", min_franking_rate=0.5)
    assert len(asx_rows) >= 5
    assert all(r["market"] == "ASX" and r["franking_rate"] >= 0.5 for r in asx_rows)

    evaluator = FactorEvaluator(populated_store)
    ev = evaluator.evaluate_factors(
        start_date=date(2025, 11, 1),
        end_date=date(2026, 11, 6),
        market="ALL",
        forward_horizon_days=21,
    )
    assert "factor_summary" in ev
    assert len(ev["factor_summary"]) >= 4
    assert "quantile_annualized_returns" in ev
