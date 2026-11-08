"""Shared pytest fixtures for offline deterministic testing of the Quant Research Platform."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant_platform.data.ingestion import DataIngestionPipeline
from quant_platform.data.providers import FixtureDataProvider
from quant_platform.data.store import MarketDataStore


@pytest.fixture(scope="session")
def populated_store(tmp_path_factory: pytest.TempPathFactory) -> MarketDataStore:
    """Create a populated DuckDB + Parquet store with 14 months of US & ASX data."""
    base_dir: Path = tmp_path_factory.mktemp("quant_test_store")
    db_path = base_dir / "test_quant.duckdb"
    parquet_dir = base_dir / "parquet"

    store = MarketDataStore(db_path=db_path, parquet_dir=parquet_dir)
    provider = FixtureDataProvider(seed=42)
    pipeline = DataIngestionPipeline(store=store, provider=provider)
    pipeline.run_ingestion(
        start_date=date(2025, 9, 1),
        end_date=date(2026, 11, 6),
        incremental=False,
    )
    return store
