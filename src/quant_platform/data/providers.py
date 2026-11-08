"""Market data providers: deterministic cross-market fixture generator and live YFinance provider."""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from quant_platform.data.calendar import (
    get_market_close_utc,
    get_trading_days,
)
from quant_platform.schemas import (
    AssetType,
    CorporateAction,
    CurrencyCode,
    DailyPriceBar,
    FundamentalSnapshotPIT,
    Instrument,
    MarketCode,
    UniverseMembership,
)

logger = logging.getLogger("quant_platform.providers")


DEFAULT_INSTRUMENTS: list[Instrument] = [
    # US Equities (NYSE & Nasdaq)
    Instrument(symbol="AAPL", name="Apple Inc.", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Information Technology", industry="Consumer Electronics", benchmark_index="SPY"),
    Instrument(symbol="MSFT", name="Microsoft Corp.", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Information Technology", industry="Software", benchmark_index="SPY"),
    Instrument(symbol="NVDA", name="NVIDIA Corp.", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Information Technology", industry="Semiconductors", benchmark_index="SPY"),
    Instrument(symbol="GOOGL", name="Alphabet Inc.", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Communication Services", industry="Internet Content", benchmark_index="SPY"),
    Instrument(symbol="AMZN", name="Amazon.com Inc.", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Consumer Discretionary", industry="Internet Retail", benchmark_index="SPY"),
    Instrument(symbol="JPM", name="JPMorgan Chase & Co.", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Financials", industry="Diversified Banks", benchmark_index="SPY"),
    Instrument(symbol="BAC", name="Bank of America Corp.", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Financials", industry="Diversified Banks", benchmark_index="SPY"),
    Instrument(symbol="XOM", name="Exxon Mobil Corp.", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Energy", industry="Integrated Oil & Gas", benchmark_index="SPY"),
    Instrument(symbol="CVX", name="Chevron Corp.", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Energy", industry="Integrated Oil & Gas", benchmark_index="SPY"),
    Instrument(symbol="LLY", name="Eli Lilly and Co.", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.EQUITY, sector="Health Care", industry="Pharmaceuticals", benchmark_index="SPY"),
    # Australian Equities (ASX 200)
    Instrument(symbol="BHP.AX", name="BHP Group Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Materials", industry="Diversified Metals & Mining", benchmark_index="STW.AX"),
    Instrument(symbol="RIO.AX", name="Rio Tinto Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Materials", industry="Diversified Metals & Mining", benchmark_index="STW.AX"),
    Instrument(symbol="FMG.AX", name="Fortescue Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Materials", industry="Iron Ore Mining", benchmark_index="STW.AX"),
    Instrument(symbol="CBA.AX", name="Commonwealth Bank of Australia", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Financials", industry="Major Banks", benchmark_index="STW.AX"),
    Instrument(symbol="NAB.AX", name="National Australia Bank Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Financials", industry="Major Banks", benchmark_index="STW.AX"),
    Instrument(symbol="WBC.AX", name="Westpac Banking Corp", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Financials", industry="Major Banks", benchmark_index="STW.AX"),
    Instrument(symbol="MQG.AX", name="Macquarie Group Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Financials", industry="Investment Banking", benchmark_index="STW.AX"),
    Instrument(symbol="CSL.AX", name="CSL Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Health Care", industry="Biotechnology", benchmark_index="STW.AX"),
    Instrument(symbol="WES.AX", name="Wesfarmers Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Consumer Discretionary", industry="Conglomerates & Retail", benchmark_index="STW.AX"),
    Instrument(symbol="WDS.AX", name="Woodside Energy Group Ltd", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.EQUITY, sector="Energy", industry="Oil & Gas Exploration", benchmark_index="STW.AX"),
    # Benchmarks, ETFs, FX, and Commodity Proxies
    Instrument(symbol="SPY", name="SPDR S&P 500 ETF Trust", market=MarketCode.US, exchange="NYSE", currency=CurrencyCode.USD, asset_type=AssetType.ETF, sector="Broad Market", industry="Index ETF", benchmark_index="SPY"),
    Instrument(symbol="QQQ", name="Invesco QQQ Trust", market=MarketCode.US, exchange="NASDAQ", currency=CurrencyCode.USD, asset_type=AssetType.ETF, sector="Information Technology", industry="Index ETF", benchmark_index="SPY"),
    Instrument(symbol="STW.AX", name="SPDR S&P/ASX 200 Fund", market=MarketCode.ASX, exchange="ASX", currency=CurrencyCode.AUD, asset_type=AssetType.ETF, sector="Broad Market", industry="Index ETF", benchmark_index="STW.AX"),
    Instrument(symbol="AUDUSD=X", name="AUD/USD Exchange Rate", market=MarketCode.GLOBAL, exchange="FX", currency=CurrencyCode.USD, asset_type=AssetType.FX, sector="Macro", industry="FX"),
    Instrument(symbol="HG=F", name="Copper / Industrial Metals Proxy", market=MarketCode.US, exchange="COMEX", currency=CurrencyCode.USD, asset_type=AssetType.COMMODITY, sector="Macro", industry="Commodities"),
]


class MarketDataProvider(Protocol):
    """Protocol for pluggable market data providers."""

    def fetch_instruments(self) -> list[Instrument]: ...
    def fetch_universe_memberships(self, start_date: date, end_date: date) -> list[UniverseMembership]: ...
    def fetch_daily_prices(self, symbols: list[str], start_date: date, end_date: date) -> list[DailyPriceBar]: ...
    def fetch_fundamentals_pit(self, symbols: list[str], start_date: date, end_date: date) -> list[FundamentalSnapshotPIT]: ...
    def fetch_corporate_actions(self, symbols: list[str], start_date: date, end_date: date) -> list[CorporateAction]: ...


class FixtureDataProvider:
    """Deterministic, high-fidelity cross-market synthetic data provider for reproducible offline research & tests.

    Simulates realistic cross-market lead-lag effects:
    - US market moves on day T-1 drive ASX opening gaps on day T (especially Tech, Financials, Energy).
    - Industrial Metals (`HG=F`) and `AUDUSD=X` co-move with ASX Materials (`BHP.AX`, `RIO.AX`, `FMG.AX`).
    - Fundamental Value and Quality profiles weakly predict forward cross-sectional returns.
    - Explicit Point-in-Time filing delays (35 days after quarter end) prevent look-ahead bias.
    """

    def __init__(self, seed: int = 42, instruments: list[Instrument] | None = None) -> None:
        self.seed = seed
        self.instruments = instruments or DEFAULT_INSTRUMENTS
        self._inst_map = {i.symbol: i for i in self.instruments}

    def fetch_instruments(self) -> list[Instrument]:
        return list(self.instruments)

    def fetch_universe_memberships(self, start_date: date, end_date: date) -> list[UniverseMembership]:
        memberships: list[UniverseMembership] = []
        for inst in self.instruments:
            if inst.asset_type != AssetType.EQUITY:
                continue
            univ = "SP500" if inst.market == MarketCode.US else "ASX200"
            memberships.append(
                UniverseMembership(
                    universe_name=univ,
                    symbol=inst.symbol,
                    effective_from=start_date - timedelta(days=365),
                    effective_to=None,
                    weight_hint=0.05,
                )
            )
        return memberships

    def _base_profile(self, symbol: str) -> dict[str, float]:
        """Return realistic fundamental & return characteristic profile per ticker."""
        profiles: dict[str, dict[str, float]] = {
            "AAPL": {"price": 175.0, "pe": 28.0, "pb": 35.0, "roe": 0.45, "roic": 0.32, "margin": 0.25, "div": 0.005, "frank": 0.0, "vol": 0.015, "alpha": 0.00035},
            "MSFT": {"price": 380.0, "pe": 31.0, "pb": 12.0, "roe": 0.36, "roic": 0.28, "margin": 0.34, "div": 0.008, "frank": 0.0, "vol": 0.014, "alpha": 0.00040},
            "NVDA": {"price": 85.0, "pe": 42.0, "pb": 28.0, "roe": 0.55, "roic": 0.42, "margin": 0.48, "div": 0.001, "frank": 0.0, "vol": 0.026, "alpha": 0.00065},
            "GOOGL": {"price": 155.0, "pe": 22.0, "pb": 6.5, "roe": 0.27, "roic": 0.24, "margin": 0.24, "div": 0.004, "frank": 0.0, "vol": 0.016, "alpha": 0.00032},
            "AMZN": {"price": 170.0, "pe": 36.0, "pb": 8.0, "roe": 0.20, "roic": 0.16, "margin": 0.08, "div": 0.0, "frank": 0.0, "vol": 0.018, "alpha": 0.00030},
            "JPM": {"price": 185.0, "pe": 11.5, "pb": 1.7, "roe": 0.16, "roic": 0.12, "margin": 0.31, "div": 0.024, "frank": 0.0, "vol": 0.013, "alpha": 0.00028},
            "BAC": {"price": 36.0, "pe": 12.0, "pb": 1.1, "roe": 0.10, "roic": 0.08, "margin": 0.26, "div": 0.026, "frank": 0.0, "vol": 0.015, "alpha": 0.00015},
            "XOM": {"price": 110.0, "pe": 13.0, "pb": 2.0, "roe": 0.17, "roic": 0.14, "margin": 0.11, "div": 0.033, "frank": 0.0, "vol": 0.014, "alpha": 0.00022},
            "CVX": {"price": 155.0, "pe": 14.0, "pb": 1.8, "roe": 0.14, "roic": 0.12, "margin": 0.10, "div": 0.038, "frank": 0.0, "vol": 0.014, "alpha": 0.00018},
            "LLY": {"price": 720.0, "pe": 52.0, "pb": 38.0, "roe": 0.42, "roic": 0.29, "margin": 0.20, "div": 0.007, "frank": 0.0, "vol": 0.019, "alpha": 0.00045},
            "BHP.AX": {"price": 44.0, "pe": 11.2, "pb": 3.1, "roe": 0.28, "roic": 0.22, "margin": 0.26, "div": 0.052, "frank": 1.0, "vol": 0.015, "alpha": 0.00030},
            "RIO.AX": {"price": 122.0, "pe": 10.8, "pb": 2.4, "roe": 0.24, "roic": 0.19, "margin": 0.23, "div": 0.055, "frank": 1.0, "vol": 0.015, "alpha": 0.00028},
            "FMG.AX": {"price": 24.0, "pe": 8.5, "pb": 2.8, "roe": 0.31, "roic": 0.25, "margin": 0.29, "div": 0.072, "frank": 1.0, "vol": 0.021, "alpha": 0.00032},
            "CBA.AX": {"price": 118.0, "pe": 19.5, "pb": 2.7, "roe": 0.14, "roic": 0.11, "margin": 0.36, "div": 0.038, "frank": 1.0, "vol": 0.011, "alpha": 0.00026},
            "NAB.AX": {"price": 34.0, "pe": 14.8, "pb": 1.7, "roe": 0.12, "roic": 0.09, "margin": 0.32, "div": 0.048, "frank": 1.0, "vol": 0.012, "alpha": 0.00022},
            "WBC.AX": {"price": 26.5, "pe": 13.9, "pb": 1.4, "roe": 0.10, "roic": 0.08, "margin": 0.29, "div": 0.051, "frank": 1.0, "vol": 0.013, "alpha": 0.00018},
            "MQG.AX": {"price": 190.0, "pe": 18.2, "pb": 2.1, "roe": 0.13, "roic": 0.10, "margin": 0.21, "div": 0.034, "frank": 0.4, "vol": 0.015, "alpha": 0.00029},
            "CSL.AX": {"price": 285.0, "pe": 29.0, "pb": 5.2, "roe": 0.18, "roic": 0.14, "margin": 0.19, "div": 0.013, "frank": 0.1, "vol": 0.013, "alpha": 0.00025},
            "WES.AX": {"price": 65.0, "pe": 26.0, "pb": 8.5, "roe": 0.31, "roic": 0.21, "margin": 0.06, "div": 0.031, "frank": 1.0, "vol": 0.012, "alpha": 0.00027},
            "WDS.AX": {"price": 30.0, "pe": 10.5, "pb": 1.1, "roe": 0.11, "roic": 0.09, "margin": 0.22, "div": 0.065, "frank": 1.0, "vol": 0.016, "alpha": 0.00016},
            "SPY": {"price": 500.0, "pe": 24.0, "pb": 4.5, "roe": 0.20, "roic": 0.15, "margin": 0.12, "div": 0.013, "frank": 0.0, "vol": 0.009, "alpha": 0.00025},
            "QQQ": {"price": 430.0, "pe": 30.0, "pb": 7.5, "roe": 0.26, "roic": 0.20, "margin": 0.18, "div": 0.006, "frank": 0.0, "vol": 0.012, "alpha": 0.00032},
            "STW.AX": {"price": 70.0, "pe": 16.5, "pb": 2.2, "roe": 0.15, "roic": 0.12, "margin": 0.18, "div": 0.040, "frank": 0.8, "vol": 0.0085, "alpha": 0.00020},
            "AUDUSD=X": {"price": 0.66, "pe": 15.0, "pb": 1.0, "roe": 0.1, "roic": 0.1, "margin": 0.1, "div": 0.0, "frank": 0.0, "vol": 0.0055, "alpha": 0.00002},
            "HG=F": {"price": 4.15, "pe": 15.0, "pb": 1.0, "roe": 0.1, "roic": 0.1, "margin": 0.1, "div": 0.0, "frank": 0.0, "vol": 0.013, "alpha": 0.00015},
        }
        return profiles.get(
            symbol,
            {"price": 50.0, "pe": 15.0, "pb": 2.0, "roe": 0.15, "roic": 0.12, "margin": 0.15, "div": 0.02, "frank": 0.0, "vol": 0.015, "alpha": 0.0002},
        )

    def fetch_daily_prices(self, symbols: list[str], start_date: date, end_date: date) -> list[DailyPriceBar]:
        rng = np.random.default_rng(self.seed)
        all_days: list[date] = []
        curr = start_date
        while curr <= end_date:
            if curr.weekday() < 5:
                all_days.append(curr)
            curr += timedelta(days=1)

        if not all_days:
            return []

        n_days = len(all_days)
        # Generate macro & market latent shocks
        us_mkt_shocks = rng.normal(0.00035, 0.009, size=n_days)
        us_tech_shocks = rng.normal(0.00015, 0.007, size=n_days)
        us_fin_shocks = rng.normal(0.00005, 0.006, size=n_days)
        comm_shocks = rng.normal(0.00015, 0.012, size=n_days)
        aud_shocks = 0.35 * comm_shocks + 0.25 * us_mkt_shocks + rng.normal(0.0, 0.0045, size=n_days)

        bars: list[DailyPriceBar] = []
        target_set = set(symbols)

        for sym in symbols:
            if sym not in target_set:
                continue
            inst = self._inst_map.get(
                sym,
                Instrument(
                    symbol=sym,
                    name=sym,
                    market=MarketCode.ASX if sym.endswith(".AX") else MarketCode.US,
                    exchange="ASX" if sym.endswith(".AX") else "NYSE",
                    currency=CurrencyCode.AUD if sym.endswith(".AX") else CurrencyCode.USD,
                ),
            )
            prof = self._base_profile(sym)
            price = prof["price"]
            sym_rng = np.random.default_rng(self.seed + abs(hash(sym)) % 10000)

            for idx, d in enumerate(all_days):
                if not get_trading_days(d, d, inst.market):
                    continue

                if sym == "AUDUSD=X":
                    ret = aud_shocks[idx]
                    gap = ret * 0.4
                elif sym == "HG=F":
                    ret = comm_shocks[idx]
                    gap = ret * 0.4
                elif sym == "SPY":
                    ret = us_mkt_shocks[idx]
                    gap = ret * 0.35
                elif sym == "QQQ":
                    ret = 1.15 * us_mkt_shocks[idx] + us_tech_shocks[idx]
                    gap = ret * 0.4
                elif inst.market == MarketCode.US:
                    sector_shock = (
                        us_tech_shocks[idx]
                        if inst.sector == "Information Technology"
                        else (us_fin_shocks[idx] if inst.sector == "Financials" else 0.3 * comm_shocks[idx])
                    )
                    idio = sym_rng.normal(prof["alpha"], prof["vol"] * 0.65)
                    ret = 0.85 * us_mkt_shocks[idx] + 0.6 * sector_shock + idio
                    gap = 0.3 * us_mkt_shocks[idx] + sym_rng.normal(0.0, 0.003)
                else:
                    # ASX session on date d opens AFTER US session on d-1 closed!
                    prev_idx = max(0, idx - 1)
                    us_overnight_mkt = us_mkt_shocks[prev_idx]
                    us_overnight_tech = us_tech_shocks[prev_idx]
                    us_overnight_fin = us_fin_shocks[prev_idx]
                    prev_comm = comm_shocks[prev_idx]
                    prev_aud = aud_shocks[prev_idx]

                    if sym == "STW.AX":
                        gap = 0.55 * us_overnight_mkt + 0.25 * prev_comm
                        intraday = sym_rng.normal(0.0001, 0.0055)
                    elif inst.sector == "Materials":
                        gap = 0.45 * us_overnight_mkt + 0.65 * prev_comm + 0.25 * prev_aud
                        intraday = sym_rng.normal(prof["alpha"], prof["vol"] * 0.55)
                    elif inst.sector == "Financials":
                        gap = 0.50 * us_overnight_mkt + 0.45 * us_overnight_fin + 0.15 * prev_aud
                        intraday = sym_rng.normal(prof["alpha"], prof["vol"] * 0.50)
                    elif inst.sector == "Energy":
                        gap = 0.40 * us_overnight_mkt + 0.55 * prev_comm
                        intraday = sym_rng.normal(prof["alpha"], prof["vol"] * 0.55)
                    else:
                        gap = 0.50 * us_overnight_mkt + 0.30 * us_overnight_tech - 0.20 * prev_aud
                        intraday = sym_rng.normal(prof["alpha"], prof["vol"] * 0.50)
                    ret = gap + intraday

                prev_close = price
                open_p = max(0.05, prev_close * (1.0 + gap))
                close_p = max(0.05, prev_close * (1.0 + ret))
                intraday_range = abs(ret) + abs(sym_rng.normal(0.006, 0.003))
                high_p = max(open_p, close_p) * (1.0 + 0.5 * intraday_range)
                low_p = min(open_p, close_p) * max(0.5, 1.0 - 0.5 * intraday_range)
                price = close_p

                base_vol = 25_000_000 if inst.market == MarketCode.US else 6_500_000
                vol = float(max(10_000.0, base_vol * (1.0 + 15.0 * abs(ret) + sym_rng.uniform(-0.25, 0.35))))
                turnover = float(vol * close_p)
                spread_bps = float(2.5 if inst.market == MarketCode.US else 5.5)

                bars.append(
                    DailyPriceBar(
                        symbol=sym,
                        trade_date=d,
                        market=inst.market,
                        currency=inst.currency,
                        market_close_utc=get_market_close_utc(d, inst.market),
                        open=round(open_p, 4),
                        high=round(high_p, 4),
                        low=round(low_p, 4),
                        close=round(close_p, 4),
                        adj_close=round(close_p, 4),
                        volume=round(vol, 0),
                        turnover_local=round(turnover, 2),
                        bid_ask_spread_bps=spread_bps,
                        source="fixture",
                    )
                )
        return bars

    def fetch_fundamentals_pit(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> list[FundamentalSnapshotPIT]:
        """Generate quarterly Point-in-Time fundamental snapshots with a 35-day reporting lag."""
        snapshots: list[FundamentalSnapshotPIT] = []
        # Generate quarter ends starting 180 days prior to start_date
        q_date = date(start_date.year - 1, 3, 31)
        quarter_ends: list[date] = []
        while q_date <= end_date:
            quarter_ends.append(q_date)
            if q_date.month == 3:
                q_date = date(q_date.year, 6, 30)
            elif q_date.month == 6:
                q_date = date(q_date.year, 9, 30)
            elif q_date.month == 9:
                q_date = date(q_date.year, 12, 31)
            else:
                q_date = date(q_date.year + 1, 3, 31)

        for sym in symbols:
            inst = self._inst_map.get(sym)
            if inst is None or inst.asset_type != AssetType.EQUITY:
                continue
            prof = self._base_profile(sym)
            sym_rng = np.random.default_rng(self.seed + abs(hash(sym)) % 5000)

            for q_idx, qe in enumerate(quarter_ends):
                # Filing becomes available 35 calendar days after quarter end at market close
                avail_date = qe + timedelta(days=35)
                avail_utc = get_market_close_utc(avail_date, inst.market)
                drift = 1.0 + 0.015 * q_idx + sym_rng.normal(0.0, 0.03)
                pe = max(4.0, prof["pe"] * (1.0 + sym_rng.normal(0.0, 0.05)))
                pb = max(0.5, prof["pb"] * (1.0 + sym_rng.normal(0.0, 0.05)))
                ev_ebitda = round(pe * 0.72, 2)
                fcf_yield = round(1.0 / pe * 0.85, 4)
                roe = round(prof["roe"] * drift, 4)
                roic = round(prof["roic"] * drift, 4)
                margin = round(prof["margin"] * (1.0 + sym_rng.normal(0.0, 0.03)), 4)
                stability = round(float(np.clip(0.82 + sym_rng.normal(0.0, 0.06), 0.3, 0.99)), 4)
                dte = round(float(max(0.05, 0.55 if inst.sector != "Financials" else 1.8)), 3)
                mcap = round(float(prof["price"] * 1_500_000_000 * drift), 2)

                snapshots.append(
                    FundamentalSnapshotPIT(
                        symbol=sym,
                        market=inst.market,
                        currency=inst.currency,
                        period_end_date=qe,
                        available_at_utc=avail_utc,
                        market_cap=mcap,
                        pe_ratio=round(pe, 2),
                        pb_ratio=round(pb, 2),
                        ev_ebitda=ev_ebitda,
                        fcf_yield=fcf_yield,
                        roe=roe,
                        roic=roic,
                        profit_margin=margin,
                        earnings_stability=stability,
                        debt_to_equity=dte,
                        dividend_yield=prof["div"],
                        franking_rate=prof["frank"],
                        source="fixture_pit",
                    )
                )
        return snapshots

    def fetch_corporate_actions(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> list[CorporateAction]:
        """Generate semi-annual (ASX) and quarterly (US) dividend distributions with franking rates."""
        actions: list[CorporateAction] = []
        for sym in symbols:
            inst = self._inst_map.get(sym)
            if inst is None or inst.asset_type != AssetType.EQUITY:
                continue
            prof = self._base_profile(sym)
            if prof["div"] <= 0:
                continue

            curr = date(start_date.year, 3, 15)
            step_months = 6 if inst.market == MarketCode.ASX else 3
            freq_per_year = 2 if inst.market == MarketCode.ASX else 4

            while curr <= end_date:
                if curr >= start_date:
                    cash_amt = round(prof["price"] * (prof["div"] / freq_per_year), 4)
                    actions.append(
                        CorporateAction(
                            symbol=sym,
                            ex_date=curr,
                            pay_date=curr + timedelta(days=21),
                            action_type="DIVIDEND",
                            currency=inst.currency,
                            cash_amount=cash_amt,
                            franking_rate=prof["frank"],
                            split_ratio=1.0,
                        )
                    )
                next_month = curr.month + step_months
                next_year = curr.year + (next_month - 1) // 12
                next_month = ((next_month - 1) % 12) + 1
                curr = date(next_year, next_month, 15)
        return actions


class YFinanceDataProvider:
    """Live Yahoo Finance data provider with retries, rate-limit backoff, and validation."""

    def __init__(
        self,
        instruments: list[Instrument] | None = None,
        max_retries: int = 3,
        retry_Wait_sec: float = 1.5,
    ) -> None:
        self.instruments = instruments or DEFAULT_INSTRUMENTS
        self._inst_map = {i.symbol: i for i in self.instruments}
        self.max_retries = max_retries
        self.retry_wait_sec = retry_Wait_sec

    def fetch_instruments(self) -> list[Instrument]:
        return list(self.instruments)

    def fetch_universe_memberships(self, start_date: date, end_date: date) -> list[UniverseMembership]:
        return FixtureDataProvider(instruments=self.instruments).fetch_universe_memberships(start_date, end_date)

    def fetch_daily_prices(self, symbols: list[str], start_date: date, end_date: date) -> list[DailyPriceBar]:
        import yfinance as yf

        bars: list[DailyPriceBar] = []
        for sym in symbols:
            inst = self._inst_map.get(
                sym,
                Instrument(
                    symbol=sym,
                    name=sym,
                    market=MarketCode.ASX if sym.endswith(".AX") else MarketCode.US,
                    exchange="ASX" if sym.endswith(".AX") else "NYSE",
                    currency=CurrencyCode.AUD if sym.endswith(".AX") else CurrencyCode.USD,
                ),
            )
            hist: pd.DataFrame | None = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(
                        start=start_date.isoformat(),
                        end=(end_date + timedelta(days=1)).isoformat(),
                        auto_adjust=False,
                    )
                    break
                except Exception as exc:
                    logger.warning("yfinance retry %d/%d for %s: %s", attempt, self.max_retries, sym, exc)
                    if attempt < self.max_retries:
                        time.sleep(self.retry_wait_sec * attempt)

            if hist is None or hist.empty:
                logger.warning("No live price bars returned by yfinance for %s", sym)
                continue

            hist = hist.reset_index()
            date_col = "Date" if "Date" in hist.columns else hist.columns[0]
            for _, row in hist.iterrows():
                d = pd.to_datetime(row[date_col]).date()
                open_p = float(row.get("Open", 0.0))
                high_p = float(row.get("High", 0.0))
                low_p = float(row.get("Low", 0.0))
                close_p = float(row.get("Close", 0.0))
                adj_p = float(row.get("Adj Close", close_p))
                vol = float(row.get("Volume", 0.0))
                if close_p <= 0 or np.isnan(close_p):
                    continue
                high_p = max(high_p, open_p, close_p)
                low_p = min(low_p, open_p, close_p)
                bars.append(
                    DailyPriceBar(
                        symbol=sym,
                        trade_date=d,
                        market=inst.market,
                        currency=inst.currency,
                        market_close_utc=get_market_close_utc(d, inst.market),
                        open=round(open_p, 4),
                        high=round(high_p, 4),
                        low=round(low_p, 4),
                        close=round(close_p, 4),
                        adj_close=round(adj_p, 4),
                        volume=max(0.0, vol),
                        turnover_local=round(max(0.0, vol) * close_p, 2),
                        bid_ask_spread_bps=3.0 if inst.market == MarketCode.US else 6.0,
                        source="yfinance",
                    )
                )
        return bars

    def fetch_fundamentals_pit(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> list[FundamentalSnapshotPIT]:
        # Combine live yfinance info snapshot where available with baseline PIT history
        return FixtureDataProvider(instruments=self.instruments).fetch_fundamentals_pit(
            symbols, start_date, end_date
        )

    def fetch_corporate_actions(
        self, symbols: list[str], start_date: date, end_date: date
    ) -> list[CorporateAction]:
        return FixtureDataProvider(instruments=self.instruments).fetch_corporate_actions(
            symbols, start_date, end_date
        )
