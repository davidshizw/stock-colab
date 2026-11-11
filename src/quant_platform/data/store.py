"""DuckDB + Parquet analytical storage layer with Point-in-Time queries and data quality checks."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from quant_platform.data.calendar import get_market_close_utc
from quant_platform.schemas import (
    CorporateAction,
    DailyPriceBar,
    DataQualityReport,
    FundamentalSnapshotPIT,
    Instrument,
    MarketCode,
    UniverseMembership,
)


class MarketDataStore:
    """Unified local analytical data store backed by DuckDB and Parquet files."""

    def __init__(self, db_path: Path | str = ":memory:", parquet_dir: Path | None = None) -> None:
        self.db_path = str(db_path)
        self.parquet_dir = parquet_dir
        if self.parquet_dir is not None:
            self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS instruments (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                market VARCHAR,
                exchange VARCHAR,
                currency VARCHAR,
                asset_type VARCHAR,
                sector VARCHAR,
                industry VARCHAR,
                benchmark_index VARCHAR,
                is_active BOOLEAN
            );

            CREATE TABLE IF NOT EXISTS universe_membership (
                universe_name VARCHAR,
                symbol VARCHAR,
                effective_from DATE,
                effective_to DATE,
                weight_hint DOUBLE,
                PRIMARY KEY (universe_name, symbol, effective_from)
            );

            CREATE TABLE IF NOT EXISTS daily_prices (
                symbol VARCHAR,
                trade_date DATE,
                market VARCHAR,
                currency VARCHAR,
                market_close_utc TIMESTAMPTZ,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                turnover_local DOUBLE,
                bid_ask_spread_bps DOUBLE,
                source VARCHAR,
                PRIMARY KEY (symbol, trade_date)
            );

            CREATE TABLE IF NOT EXISTS corporate_actions (
                symbol VARCHAR,
                ex_date DATE,
                pay_date DATE,
                action_type VARCHAR,
                currency VARCHAR,
                cash_amount DOUBLE,
                franking_rate DOUBLE,
                split_ratio DOUBLE,
                PRIMARY KEY (symbol, ex_date, action_type)
            );

            CREATE TABLE IF NOT EXISTS fundamentals_pit (
                symbol VARCHAR,
                market VARCHAR,
                currency VARCHAR,
                period_end_date DATE,
                available_at_utc TIMESTAMPTZ,
                market_cap DOUBLE,
                pe_ratio DOUBLE,
                pb_ratio DOUBLE,
                ev_ebitda DOUBLE,
                fcf_yield DOUBLE,
                roe DOUBLE,
                roic DOUBLE,
                profit_margin DOUBLE,
                earnings_stability DOUBLE,
                debt_to_equity DOUBLE,
                dividend_yield DOUBLE,
                franking_rate DOUBLE,
                source VARCHAR,
                PRIMARY KEY (symbol, period_end_date, available_at_utc)
            );

            CREATE TABLE IF NOT EXISTS factor_values (
                symbol VARCHAR,
                trade_date DATE,
                market VARCHAR,
                sector VARCHAR,
                factor_name VARCHAR,
                raw_value DOUBLE,
                zscore_value DOUBLE,
                sector_neutral_value DOUBLE,
                PRIMARY KEY (symbol, trade_date, factor_name)
            );

            CREATE TABLE IF NOT EXISTS ingestion_audit (
                run_id VARCHAR,
                ingested_at_utc TIMESTAMPTZ,
                source VARCHAR,
                table_name VARCHAR,
                rows_affected INTEGER,
                status VARCHAR
            );
        """)

    def upsert_instruments(self, instruments: list[Instrument]) -> int:
        if not instruments:
            return 0
        df = pd.DataFrame([i.model_dump(mode="json") for i in instruments])
        self.conn.register("tmp_instruments", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO instruments
            SELECT symbol, name, market, exchange, currency, asset_type,
                   sector, industry, benchmark_index, is_active
            FROM tmp_instruments
        """)
        self.conn.unregister("tmp_instruments")
        self._export_parquet("instruments")
        return len(df)

    def upsert_universe_memberships(self, memberships: list[UniverseMembership]) -> int:
        if not memberships:
            return 0
        df = pd.DataFrame([m.model_dump(mode="json") for m in memberships])
        self.conn.register("tmp_memberships", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO universe_membership
            SELECT universe_name, symbol, CAST(effective_from AS DATE),
                   CAST(effective_to AS DATE), weight_hint
            FROM tmp_memberships
        """)
        self.conn.unregister("tmp_memberships")
        self._export_parquet("universe_membership")
        return len(df)

    def upsert_daily_prices(self, bars: list[DailyPriceBar] | pd.DataFrame) -> int:
        if isinstance(bars, list):
            if not bars:
                return 0
            df = pd.DataFrame([b.model_dump(mode="json") for b in bars])
        else:
            if bars.empty:
                return 0
            df = bars.copy()

        self.conn.register("tmp_prices", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO daily_prices
            SELECT symbol,
                   CAST(trade_date AS DATE),
                   market,
                   currency,
                   CAST(market_close_utc AS TIMESTAMPTZ),
                   open, high, low, close, adj_close,
                   volume, turnover_local, bid_ask_spread_bps, source
            FROM tmp_prices
        """)
        self.conn.unregister("tmp_prices")
        self._export_parquet("daily_prices")
        return len(df)

    def upsert_corporate_actions(self, actions: list[CorporateAction]) -> int:
        if not actions:
            return 0
        df = pd.DataFrame([a.model_dump(mode="json") for a in actions])
        self.conn.register("tmp_actions", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO corporate_actions
            SELECT symbol, CAST(ex_date AS DATE), CAST(pay_date AS DATE),
                   action_type, currency, cash_amount, franking_rate, split_ratio
            FROM tmp_actions
        """)
        self.conn.unregister("tmp_actions")
        self._export_parquet("corporate_actions")
        return len(df)

    def upsert_fundamentals_pit(self, snapshots: list[FundamentalSnapshotPIT] | pd.DataFrame) -> int:
        if isinstance(snapshots, list):
            if not snapshots:
                return 0
            df = pd.DataFrame([s.model_dump(mode="json") for s in snapshots])
        else:
            if snapshots.empty:
                return 0
            df = snapshots.copy()

        self.conn.register("tmp_fundamentals", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO fundamentals_pit
            SELECT symbol, market, currency,
                   CAST(period_end_date AS DATE),
                   CAST(available_at_utc AS TIMESTAMPTZ),
                   market_cap, pe_ratio, pb_ratio, ev_ebitda, fcf_yield,
                   roe, roic, profit_margin, earnings_stability,
                   debt_to_equity, dividend_yield, franking_rate, source
            FROM tmp_fundamentals
        """)
        self.conn.unregister("tmp_fundamentals")
        self._export_parquet("fundamentals_pit")
        return len(df)

    def save_factor_values(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        self.conn.register("tmp_factors", df)
        self.conn.execute("""
            INSERT OR REPLACE INTO factor_values
            SELECT symbol, CAST(trade_date AS DATE), market, sector,
                   factor_name, raw_value, zscore_value, sector_neutral_value
            FROM tmp_factors
        """)
        self.conn.unregister("tmp_factors")
        self._export_parquet("factor_values")
        return len(df)

    def log_ingestion(self, run_id: str, source: str, table_name: str, rows: int, status: str = "SUCCESS") -> None:
        self.conn.execute(
            """
            INSERT INTO ingestion_audit VALUES (?, ?, ?, ?, ?, ?)
            """,
            [run_id, datetime.now(UTC), source, table_name, rows, status],
        )

    def _export_parquet(self, table_name: str) -> None:
        if self.parquet_dir is None:
            return
        out_file = self.parquet_dir / f"{table_name}.parquet"
        self.conn.execute(f"COPY {table_name} TO '{out_file.as_posix()}' (FORMAT PARQUET)")

    def get_instruments(
        self,
        market: str | None = None,
        asset_type: str | None = "EQUITY",
        active_only: bool = False,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[object] = []
        if market and market != "ALL":
            clauses.append("market = ?")
            params.append(market)
        if asset_type and asset_type != "ALL":
            clauses.append("asset_type = ?")
            params.append(asset_type)
        if active_only:
            clauses.append("is_active = TRUE")
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self.conn.cursor().execute(f"SELECT * FROM instruments {where_sql} ORDER BY symbol", params).df()

    def get_active_universe_as_of(self, as_of_date: date, universe_name: str | None = None) -> list[str]:
        """Return symbols that were members of the universe on `as_of_date` (prevents survivorship bias)."""
        cur = self.conn.cursor()
        if universe_name:
            rows = cur.execute(
                """
                SELECT DISTINCT symbol
                FROM universe_membership
                WHERE universe_name = ?
                  AND effective_from <= ?
                  AND (effective_to IS NULL OR effective_to >= ?)
                ORDER BY symbol
                """,
                [universe_name, as_of_date, as_of_date],
            ).fetchall()
        else:
            rows = cur.execute(
                """
                SELECT DISTINCT symbol
                FROM universe_membership
                WHERE effective_from <= ?
                  AND (effective_to IS NULL OR effective_to >= ?)
                ORDER BY symbol
                """,
                [as_of_date, as_of_date],
            ).fetchall()
        return [r[0] for r in rows]

    def get_prices(
        self,
        symbols: list[str] | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        market: str | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[object] = []
        if symbols:
            placeholders = ", ".join(["?"] * len(symbols))
            clauses.append(f"p.symbol IN ({placeholders})")
            params.extend(symbols)
        if start_date:
            clauses.append("p.trade_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("p.trade_date <= ?")
            params.append(end_date)
        if market and market != "ALL":
            clauses.append("p.market = ?")
            params.append(market)

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT p.*, i.sector, i.asset_type, i.name
            FROM daily_prices p
            LEFT JOIN instruments i ON p.symbol = i.symbol
            {where_sql}
            ORDER BY p.trade_date, p.symbol
        """
        df = self.conn.cursor().execute(query, params).df()
        if not df.empty:
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        return df

    def get_fundamentals_as_of(
        self,
        as_of_date: date,
        market: str | None = None,
    ) -> pd.DataFrame:
        """Retrieve latest fundamental snapshot strictly available at or before market close on `as_of_date`.

        Guarantees zero look-ahead bias by filtering on `available_at_utc <= cutoff_utc`.
        """
        # Use US close as conservative global end-of-day cutoff or market-specific close
        cutoff_market = MarketCode.ASX if market == "ASX" else MarketCode.US
        cutoff_utc = get_market_close_utc(as_of_date, cutoff_market)

        market_clause = "AND market = ?" if (market and market != "ALL") else ""
        params: list[object] = [cutoff_utc]
        if market and market != "ALL":
            params.append(market)

        query = f"""
            WITH ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY symbol
                           ORDER BY period_end_date DESC, available_at_utc DESC
                       ) AS rn
                FROM fundamentals_pit
                WHERE available_at_utc <= ?
                {market_clause}
            )
            SELECT * EXCLUDE (rn)
            FROM ranked
            WHERE rn = 1
            ORDER BY symbol
        """
        return self.conn.cursor().execute(query, params).df()

    def get_corporate_actions(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[object] = []
        if start_date:
            clauses.append("ex_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("ex_date <= ?")
            params.append(end_date)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        df = self.conn.cursor().execute(f"SELECT * FROM corporate_actions {where_sql} ORDER BY ex_date, symbol", params).df()
        if not df.empty:
            df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.date
        return df

    def check_data_quality(self) -> DataQualityReport:
        """Inspect local store for missing data gaps, OHLC violations, staleness, and provenance."""
        cur = self.conn.cursor()
        total_inst_row = cur.execute("SELECT COUNT(*) FROM instruments").fetchone()
        total_inst = int(total_inst_row[0]) if total_inst_row else 0

        total_price_row = cur.execute("SELECT COUNT(*) FROM daily_prices").fetchone()
        total_prices = int(total_price_row[0]) if total_price_row else 0

        total_fund_row = cur.execute("SELECT COUNT(*) FROM fundamentals_pit").fetchone()
        total_fund = int(total_fund_row[0]) if total_fund_row else 0

        ohlc_row = cur.execute("""
            SELECT COUNT(*) FROM daily_prices
            WHERE high < low OR high < open OR high < close OR low > open OR low > close OR close <= 0
        """).fetchone()
        ohlc_violations = int(ohlc_row[0]) if ohlc_row else 0

        # Check date gaps > 7 calendar days within active series
        gap_row = cur.execute("""
            WITH lagged AS (
                SELECT symbol, trade_date,
                       LAG(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_date
                FROM daily_prices
            )
            SELECT COUNT(*) FROM lagged
            WHERE DATEDIFF('day', prev_date, trade_date) > 7
        """).fetchone()
        missing_gaps = int(gap_row[0]) if gap_row else 0

        # Check stale symbols relative to max trade_date in store
        stale_rows = cur.execute("""
            WITH max_global AS (
                SELECT market, MAX(trade_date) AS market_max_date
                FROM daily_prices
                GROUP BY market
            ),
            sym_max AS (
                SELECT p.symbol, p.market, MAX(p.trade_date) AS last_date, i.is_active
                FROM daily_prices p
                JOIN instruments i ON p.symbol = i.symbol
                GROUP BY p.symbol, p.market, i.is_active
            )
            SELECT s.symbol
            FROM sym_max s
            JOIN max_global m ON s.market = m.market
            WHERE s.is_active = TRUE AND DATEDIFF('day', s.last_date, m.market_max_date) > 5
        """).fetchall()
        stale_symbols = [r[0] for r in stale_rows]

        prov_rows = cur.execute("""
            SELECT source, COUNT(*) FROM daily_prices GROUP BY source
        """).fetchall()
        provenance = {str(r[0]): int(r[1]) for r in prov_rows}

        status = "HEALTHY"
        if total_prices == 0 or ohlc_violations > 0:
            status = "CRITICAL"
        elif missing_gaps > 5 or len(stale_symbols) > 0:
            status = "DEGRADED"

        return DataQualityReport(
            checked_at_utc=datetime.now(UTC),
            total_instruments=total_inst,
            total_price_rows=total_prices,
            total_fundamental_rows=total_fund,
            missing_date_gaps=missing_gaps,
            stale_symbols=stale_symbols,
            ohlc_violations=ohlc_violations,
            provenance_sources=provenance,
            status=status,
        )
