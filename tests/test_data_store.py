"""Unit tests for DuckDB/Parquet store, Point-in-Time queries, survivorship bias, and data quality."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quant_platform.data.calendar import get_market_close_utc
from quant_platform.data.store import MarketDataStore
from quant_platform.schemas import (
    CurrencyCode,
    FundamentalSnapshotPIT,
    MarketCode,
    UniverseMembership,
)


def test_store_quality_and_parquet_export(populated_store: MarketDataStore) -> None:
    report = populated_store.check_data_quality()
    assert report.total_instruments >= 20
    assert report.total_price_rows > 2000
    assert report.ohlc_violations == 0
    assert report.status in ("HEALTHY", "DEGRADED")
    if populated_store.parquet_dir is not None:
        assert (populated_store.parquet_dir / "daily_prices.parquet").exists()


def test_pit_fundamentals_prevents_lookahead(tmp_path: Path) -> None:
    store = MarketDataStore(db_path=tmp_path / "pit.duckdb")
    # Q1 ends 2026-03-31, filed 2026-05-05; Q2 ends 2026-06-30, filed 2026-08-10
    snaps = [
        FundamentalSnapshotPIT(
            symbol="BHP.AX",
            market=MarketCode.ASX,
            currency=CurrencyCode.AUD,
            period_end_date=date(2026, 3, 31),
            available_at_utc=get_market_close_utc(date(2026, 5, 5), MarketCode.ASX),
            market_cap=150e9,
            pe_ratio=11.0,
            roe=0.25,
        ),
        FundamentalSnapshotPIT(
            symbol="BHP.AX",
            market=MarketCode.ASX,
            currency=CurrencyCode.AUD,
            period_end_date=date(2026, 6, 30),
            available_at_utc=get_market_close_utc(date(2026, 8, 10), MarketCode.ASX),
            market_cap=165e9,
            pe_ratio=12.5,
            roe=0.30,
        ),
    ]
    store.upsert_fundamentals_pit(snaps)

    # Query on 2026-07-15 (AFTER Q2 period_end_date 2026-06-30, but BEFORE filing date 2026-08-10)
    df_july = store.get_fundamentals_as_of(date(2026, 7, 15))
    assert len(df_july) == 1
    assert pd.to_datetime(df_july.iloc[0]["period_end_date"]).date() == date(2026, 3, 31)
    assert float(df_july.iloc[0]["roe"]) == 0.25

    # Query on 2026-08-15 (AFTER Q2 filing date)
    df_aug = store.get_fundamentals_as_of(date(2026, 8, 15))
    assert pd.to_datetime(df_aug.iloc[0]["period_end_date"]).date() == date(2026, 6, 30)
    assert float(df_aug.iloc[0]["roe"]) == 0.30


def test_universe_membership_survivorship_filter(tmp_path: Path) -> None:
    store = MarketDataStore(db_path=tmp_path / "univ.duckdb")
    store.upsert_universe_memberships(
        [
            UniverseMembership(
                universe_name="ASX200",
                symbol="DELIST.AX",
                effective_from=date(2024, 1, 1),
                effective_to=date(2025, 12, 31),
            ),
            UniverseMembership(
                universe_name="ASX200",
                symbol="BHP.AX",
                effective_from=date(2024, 1, 1),
                effective_to=None,
            ),
        ]
    )
    members_2025 = store.get_active_universe_as_of(date(2025, 6, 15), "ASX200")
    assert "DELIST.AX" in members_2025
    assert "BHP.AX" in members_2025

    members_2026 = store.get_active_universe_as_of(date(2026, 6, 15), "ASX200")
    assert "DELIST.AX" not in members_2026
    assert "BHP.AX" in members_2026

