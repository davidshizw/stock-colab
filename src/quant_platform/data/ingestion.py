"""Incremental market data ingestion pipeline with provenance tracking and quality checks."""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta
from typing import Any

from quant_platform.data.providers import (
    FixtureDataProvider,
    MarketDataProvider,
    YFinanceDataProvider,
)
from quant_platform.data.store import MarketDataStore

logger = logging.getLogger("quant_platform.ingestion")


class DataIngestionPipeline:
    """Orchestrates incremental market data ingestion into DuckDB and Parquet storage."""

    def __init__(self, store: MarketDataStore, provider: MarketDataProvider | None = None) -> None:
        self.store = store
        self.provider: MarketDataProvider = provider or FixtureDataProvider()

    def run_ingestion(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        symbols: list[str] | None = None,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """Execute end-to-end ingestion of instruments, universes, prices, fundamentals, and actions."""
        run_id = f"ing-{uuid.uuid4().hex[:8]}"
        end_d = end_date or date(2026, 11, 6)
        start_d = start_date or (end_d - timedelta(days=540))

        instruments = self.provider.fetch_instruments()
        if symbols:
            target_set = set(symbols) | {"SPY", "QQQ", "STW.AX", "AUDUSD=X", "HG=F"}
            instruments = [i for i in instruments if i.symbol in target_set]

        inst_count = self.store.upsert_instruments(instruments)
        self.store.log_ingestion(run_id, type(self.provider).__name__, "instruments", inst_count)

        memberships = self.provider.fetch_universe_memberships(start_d, end_d)
        mem_count = self.store.upsert_universe_memberships(memberships)
        self.store.log_ingestion(run_id, type(self.provider).__name__, "universe_membership", mem_count)

        all_syms = [i.symbol for i in instruments]

        # Determine incremental start date if enabled
        fetch_start = start_d
        if incremental:
            row = self.store.conn.execute("SELECT MAX(trade_date) FROM daily_prices").fetchone()
            if row and row[0] is not None:
                last_dt = row[0]
                if isinstance(last_dt, date) and last_dt > start_d and last_dt < end_d:
                    fetch_start = last_dt - timedelta(days=3)

        bars = self.provider.fetch_daily_prices(all_syms, fetch_start, end_d)
        price_count = self.store.upsert_daily_prices(bars)
        self.store.log_ingestion(run_id, type(self.provider).__name__, "daily_prices", price_count)

        fundamentals = self.provider.fetch_fundamentals_pit(all_syms, start_d, end_d)
        fund_count = self.store.upsert_fundamentals_pit(fundamentals)
        self.store.log_ingestion(run_id, type(self.provider).__name__, "fundamentals_pit", fund_count)

        actions = self.provider.fetch_corporate_actions(all_syms, start_d, end_d)
        act_count = self.store.upsert_corporate_actions(actions)
        self.store.log_ingestion(run_id, type(self.provider).__name__, "corporate_actions", act_count)

        quality = self.store.check_data_quality()
        logger.info(
            "Ingestion %s completed: %d instruments, %d price bars, %d PIT fundamentals (status=%s)",
            run_id,
            inst_count,
            price_count,
            fund_count,
            quality.status,
        )
        return {
            "run_id": run_id,
            "instruments_upserted": inst_count,
            "prices_upserted": price_count,
            "fundamentals_upserted": fund_count,
            "corporate_actions_upserted": act_count,
            "quality_report": quality.model_dump(mode="json"),
        }


def build_default_pipeline(
    store: MarketDataStore, mode: str = "fixture"
) -> DataIngestionPipeline:
    """Factory for creating a fixture or live yfinance ingestion pipeline."""
    if mode.lower() == "yfinance":
        return DataIngestionPipeline(store=store, provider=YFinanceDataProvider())
    return DataIngestionPipeline(store=store, provider=FixtureDataProvider())
