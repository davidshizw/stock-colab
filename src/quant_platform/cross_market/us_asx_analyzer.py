"""Cross-market intelligence: US overnight lead-lag into ASX opening gaps, sectors, FX, and commodities."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
from scipy.stats import pearsonr

from quant_platform.data.calendar import align_us_overnight_to_asx
from quant_platform.data.store import MarketDataStore


class CrossMarketAnalyzer:
    """Analyzes timestamp-aligned transmission from US overnight markets to Australian equities."""

    def __init__(self, store: MarketDataStore) -> None:
        self.store = store

    def analyze_overnight_transmission(
        self, start_date: date, end_date: date
    ) -> dict[str, Any]:
        """Quantify how US overnight returns, commodities, and AUD/USD drive ASX opening gaps and returns."""
        prices = self.store.get_prices(start_date=start_date, end_date=end_date)
        if prices.empty:
            return {}

        # Separate US & Macro drivers vs ASX securities
        us_drivers = prices[prices["symbol"].isin(["SPY", "QQQ", "JPM", "XOM", "HG=F", "AUDUSD=X"])].copy()
        asx_stocks = prices[prices["market"] == "ASX"].copy()
        if us_drivers.empty or asx_stocks.empty:
            return {}

        # Compute daily returns for US drivers indexed by their exact UTC close timestamp
        us_drivers = us_drivers.sort_values(["symbol", "trade_date"])
        us_drivers["daily_ret"] = us_drivers.groupby("symbol")["adj_close"].pct_change()

        # Pivot US driver returns by US trade_date and keep SPY's market_close_utc as reference
        spy_close_map = (
            us_drivers[us_drivers["symbol"] == "SPY"][["trade_date", "market_close_utc"]]
            .drop_duplicates("trade_date")
            .set_index("trade_date")["market_close_utc"]
        )
        us_piv = us_drivers.pivot(index="trade_date", columns="symbol", values="daily_ret").dropna(how="all")
        us_piv["market_close_utc"] = us_piv.index.map(spy_close_map)
        us_piv = us_piv.dropna(subset=["market_close_utc"]).reset_index()

        # Compute ASX opening gap (Open_t / Close_{t-1} - 1), intraday return (Close_t / Open_t - 1), and total return
        asx_stocks = asx_stocks.sort_values(["symbol", "trade_date"])
        asx_stocks["prev_close"] = asx_stocks.groupby("symbol")["close"].shift(1)
        asx_stocks["open_gap_ret"] = asx_stocks["open"] / asx_stocks["prev_close"] - 1.0
        asx_stocks["intraday_ret"] = asx_stocks["close"] / asx_stocks["open"] - 1.0
        asx_stocks["total_ret"] = asx_stocks["adj_close"] / asx_stocks.groupby("symbol")["adj_close"].shift(1) - 1.0
        asx_stocks = asx_stocks.dropna(subset=["open_gap_ret", "total_ret"])

        # Align strictly using UTC timestamps: ASX open on date T only sees US close < ASX open UTC
        aligned = align_us_overnight_to_asx(
            us_df=us_piv,
            asx_df=asx_stocks,
            us_close_col="market_close_utc",
            asx_trade_date_col="trade_date",
        )
        if aligned.empty:
            return {}

        # Verify zero look-ahead bias: every row must satisfy us_close_utc < asx_open_utc
        valid_aligned = aligned.dropna(subset=["us_close_utc", "asx_open_utc"])
        lookahead_violations = int((valid_aligned["us_close_utc"] >= valid_aligned["asx_open_utc"]).sum())

        # 1. Sector sensitivity matrix (ASX sectors vs US SPY, QQQ, Financials, Energy, Copper, AUDUSD)
        driver_cols = [c for c in ["SPY", "QQQ", "JPM", "XOM", "HG=F", "AUDUSD=X"] if c in aligned.columns]
        sector_sensitivity: list[dict[str, Any]] = []
        for sector, grp in aligned.groupby("sector"):
            if sector == "Broad Market":
                continue
            sec_daily = grp.groupby("trade_date_asx")[
                ["open_gap_ret", "intraday_ret", "total_ret"] + driver_cols
            ].mean().dropna()
            if len(sec_daily) < 10:
                continue
            row_s: dict[str, Any] = {
                "asx_sector": str(sector),
                "observations": len(sec_daily),
            }
            for d_col in driver_cols:
                if np.std(sec_daily[d_col]) > 1e-12 and np.std(sec_daily["open_gap_ret"]) > 1e-12:
                    corr_gap = float(pearsonr(sec_daily[d_col], sec_daily["open_gap_ret"])[0])
                    corr_tot = float(pearsonr(sec_daily[d_col], sec_daily["total_ret"])[0])
                    beta_tot = float(
                        np.cov(sec_daily["total_ret"], sec_daily[d_col])[0, 1]
                        / max(np.var(sec_daily[d_col]), 1e-12)
                    )
                else:
                    corr_gap, corr_tot, beta_tot = 0.0, 0.0, 0.0
                row_s[f"corr_gap_{d_col}"] = round(corr_gap, 3)
                row_s[f"corr_total_{d_col}"] = round(corr_tot, 3)
                row_s[f"beta_{d_col}"] = round(beta_tot, 3)
            sector_sensitivity.append(row_s)

        # 2. Stock-level overnight gap prediction accuracy & directional hit rate
        stock_transmission: list[dict[str, Any]] = []
        for sym, grp in aligned.groupby("symbol"):
            if len(grp) < 15 or "SPY" not in grp.columns:
                continue
            valid = grp.dropna(subset=["SPY", "open_gap_ret", "intraday_ret", "total_ret"])
            if len(valid) < 15:
                continue
            spy_corr_gap = float(pearsonr(valid["SPY"], valid["open_gap_ret"])[0])
            comm_corr = (
                float(pearsonr(valid["HG=F"], valid["total_ret"])[0])
                if "HG=F" in valid.columns and np.std(valid["HG=F"]) > 1e-12
                else 0.0
            )
            aud_corr = (
                float(pearsonr(valid["AUDUSD=X"], valid["total_ret"])[0])
                if "AUDUSD=X" in valid.columns and np.std(valid["AUDUSD=X"]) > 1e-12
                else 0.0
            )
            directional_hit = float((np.sign(valid["SPY"]) == np.sign(valid["open_gap_ret"])).mean())
            stock_transmission.append(
                {
                    "symbol": str(sym),
                    "sector": str(valid["sector"].iloc[0]),
                    "us_spy_overnight_gap_corr": round(spy_corr_gap, 3),
                    "commodity_return_corr": round(comm_corr, 3),
                    "audusd_return_corr": round(aud_corr, 3),
                    "overnight_gap_directional_hit_rate": round(directional_hit, 3),
                    "avg_open_gap_bps": round(float(valid["open_gap_ret"].mean() * 10000), 2),
                    "avg_intraday_drift_bps": round(float(valid["intraday_ret"].mean() * 10000), 2),
                }
            )

        # 3. Global Risk-On / Risk-Off Regime Comparison
        stw_aligned = aligned[aligned["symbol"] == "STW.AX"].copy()
        if stw_aligned.empty:
            stw_aligned = aligned.groupby("trade_date_asx")[
                ["SPY", "open_gap_ret", "intraday_ret", "total_ret"]
            ].mean().reset_index()

        stw_aligned["spy_vol_20d"] = stw_aligned["SPY"].rolling(20, min_periods=5).std() * np.sqrt(252)
        vol_median = float(stw_aligned["spy_vol_20d"].median()) if not stw_aligned.empty else 0.14
        stw_aligned["global_regime"] = np.where(
            stw_aligned["spy_vol_20d"] > vol_median * 1.15,
            "RISK_OFF_HIGH_VOL",
            "RISK_ON_NORMAL",
        )
        regime_stats: list[dict[str, Any]] = []
        for reg, rgrp in stw_aligned.groupby("global_regime"):
            regime_stats.append(
                {
                    "regime": str(reg),
                    "days": len(rgrp),
                    "us_spy_annualized_return": round(float(rgrp["SPY"].mean() * 252), 4),
                    "asx_annualized_return": round(float(rgrp["total_ret"].mean() * 252), 4),
                    "asx_open_gap_annualized": round(float(rgrp["open_gap_ret"].mean() * 252), 4),
                    "us_asx_transmission_corr": round(
                        float(pearsonr(rgrp["SPY"], rgrp["open_gap_ret"])[0])
                        if len(rgrp) >= 5 and np.std(rgrp["SPY"]) > 1e-12
                        else 0.0,
                        3,
                    ),
                }
            )

        return {
            "lookahead_violations": lookahead_violations,
            "alignment_verified": lookahead_violations == 0,
            "sector_sensitivity": sector_sensitivity,
            "stock_transmission": sorted(
                stock_transmission, key=lambda x: x["us_spy_overnight_gap_corr"], reverse=True
            ),
            "global_regime_comparison": regime_stats,
        }
