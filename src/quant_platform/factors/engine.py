"""Modular Quantitative Factor Engine for US & Australian Equities."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from quant_platform.data.calendar import align_us_overnight_to_asx
from quant_platform.data.store import MarketDataStore
from quant_platform.factors.preprocessing import (
    sector_neutralise_series,
    zscore_series,
)

FACTOR_CATEGORIES: dict[str, list[str]] = {
    "value": ["earnings_yield", "book_to_price", "ev_ebitda_inv", "fcf_yield"],
    "quality": ["roe", "roic", "profit_margin", "earnings_stability", "balance_sheet_strength"],
    "momentum": ["mom_1m", "mom_3m", "mom_6m", "mom_12m", "risk_adj_momentum"],
    "risk": ["volatility_20d", "downside_volatility", "beta_60d", "max_drawdown_120d"],
    "liquidity": ["adv_20d", "turnover_ratio", "spread_tightness"],
    "market_specific": [
        "franking_grossed_yield",
        "commodity_sensitivity",
        "audusd_sensitivity",
        "us_overnight_signal",
    ],
}


class FactorEngine:
    """Computes fundamental, technical, risk, liquidity, and cross-market factors."""

    def __init__(
        self,
        store: MarketDataStore,
        au_corporate_tax_rate: float = 0.30,
        us_withholding_tax_rate: float = 0.15,
    ) -> None:
        self.store = store
        self.au_tax = au_corporate_tax_rate
        self.us_wht = us_withholding_tax_rate

    def compute_factors_as_of(
        self,
        as_of_date: date,
        market: str = "ALL",
        sector_neutral: bool = True,
        weights: dict[str, float] | None = None,
    ) -> pd.DataFrame:
        """Compute cross-sectional factor table for all active equities on `as_of_date`."""
        inst_df = self.store.get_instruments(market=market, asset_type="EQUITY", active_only=True)
        if inst_df.empty:
            return pd.DataFrame()

        eq_symbols = inst_df["symbol"].tolist()
        macro_symbols = ["SPY", "QQQ", "STW.AX", "AUDUSD=X", "HG=F"]
        all_query_symbols = list(set(eq_symbols + macro_symbols))

        # Pull ~380 calendar days of price history up to as_of_date
        start_lookback = pd.Timestamp(as_of_date) - pd.Timedelta(days=400)
        prices_df = self.store.get_prices(
            symbols=all_query_symbols,
            start_date=start_lookback.date(),
            end_date=as_of_date,
        )
        if prices_df.empty:
            return pd.DataFrame()

        # Pull Point-in-Time fundamentals strictly available as of as_of_date
        fund_df = self.store.get_fundamentals_as_of(as_of_date=as_of_date, market=market)

        # Pivot daily close returns for equities and macro benchmarks
        prices_df["trade_date_dt"] = pd.to_datetime(prices_df["trade_date"])
        close_piv = prices_df.pivot(index="trade_date_dt", columns="symbol", values="adj_close").sort_index()
        ret_piv = close_piv.pct_change(fill_method=None)

        # Extract macro series returns
        spy_ret = ret_piv["SPY"] if "SPY" in ret_piv.columns else pd.Series(dtype=float)
        stw_ret = ret_piv["STW.AX"] if "STW.AX" in ret_piv.columns else pd.Series(dtype=float)
        comm_ret = ret_piv["HG=F"] if "HG=F" in ret_piv.columns else pd.Series(dtype=float)
        aud_ret = ret_piv["AUDUSD=X"] if "AUDUSD=X" in ret_piv.columns else pd.Series(dtype=float)

        # Compute PIT-aligned US overnight return available prior to ASX open on as_of_date
        us_overnight_ret = 0.0
        if "SPY" in prices_df["symbol"].values and "STW.AX" in prices_df["symbol"].values:
            spy_bars = prices_df[prices_df["symbol"] == "SPY"][
                ["trade_date", "market_close_utc", "adj_close"]
            ].copy()
            spy_bars["spy_ret"] = spy_bars["adj_close"].pct_change()
            asx_bars = prices_df[prices_df["symbol"] == "STW.AX"][["trade_date"]].drop_duplicates()
            aligned = align_us_overnight_to_asx(spy_bars, asx_bars)
            if not aligned.empty and "spy_ret" in aligned.columns:
                val = aligned.iloc[-1]["spy_ret"]
                if pd.notna(val):
                    us_overnight_ret = float(val)

        rows: list[dict[str, object]] = []
        for _, inst in inst_df.iterrows():
            sym = str(inst["symbol"])
            if sym not in close_piv.columns:
                continue
            sym_close = close_piv[sym].dropna()
            if len(sym_close) < 22:
                continue

            sym_rets = sym_close.pct_change().dropna()
            sym_bars = prices_df[prices_df["symbol"] == sym].sort_values("trade_date")
            latest_bar = sym_bars.iloc[-1]
            latest_price = float(latest_bar["adj_close"])

            # 1. Momentum Factors
            def _TrailingRet(days: int, s: pd.Series = sym_close) -> float:
                if len(s) <= days:
                    return float(s.iloc[-1] / s.iloc[0] - 1.0)
                return float(s.iloc[-1] / s.iloc[-1 - days] - 1.0)

            mom_1m = _TrailingRet(21)
            mom_3m = _TrailingRet(63)
            mom_6m = _TrailingRet(126)
            mom_12m = _TrailingRet(min(252, len(sym_close) - 1))

            vol_6m = float(sym_rets.tail(126).std(ddof=1) * np.sqrt(252)) if len(sym_rets) >= 20 else 0.20
            risk_adj_mom = mom_6m / max(vol_6m, 0.05)

            # 2. Risk Factors
            vol_20d = float(sym_rets.tail(20).std(ddof=1) * np.sqrt(252))
            neg_rets = sym_rets.tail(60)[sym_rets.tail(60) < 0]
            downside_vol = float(neg_rets.std(ddof=1) * np.sqrt(252)) if len(neg_rets) >= 3 else vol_20d * 0.7

            # Rolling 60d Beta against local benchmark
            bench_s = stw_ret if inst["market"] == "ASX" else spy_ret
            beta_60d = 1.0
            if not bench_s.empty:
                joined_b = pd.concat([sym_rets.tail(60), bench_s], axis=1, join="inner").dropna()
                if len(joined_b) >= 15:
                    cov = np.cov(joined_b.iloc[:, 0], joined_b.iloc[:, 1])
                    if cov[1, 1] > 1e-10:
                        beta_60d = float(cov[0, 1] / cov[1, 1])

            # Max Drawdown (120d)
            window_close = sym_close.tail(120)
            roll_peak = window_close.cummax()
            dd_series = window_close / roll_peak - 1.0
            max_dd_120d = float(abs(dd_series.min()))

            # 3. Liquidity Factors
            recent_bars = sym_bars.tail(20)
            adv_20d = float(recent_bars["turnover_local"].mean())
            spread_bps = float(recent_bars["bid_ask_spread_bps"].mean()) if "bid_ask_spread_bps" in recent_bars else 4.0

            # 4. Fundamental Factors (from PIT table)
            f_row = fund_df[fund_df["symbol"] == sym] if not fund_df.empty else pd.DataFrame()
            if not f_row.empty:
                fr = f_row.iloc[0]
                pe = float(fr["pe_ratio"]) if pd.notna(fr["pe_ratio"]) and fr["pe_ratio"] > 0 else 20.0
                pb = float(fr["pb_ratio"]) if pd.notna(fr["pb_ratio"]) and fr["pb_ratio"] > 0 else 2.5
                ev_ebitda = float(fr["ev_ebitda"]) if pd.notna(fr["ev_ebitda"]) and fr["ev_ebitda"] > 0 else 14.0
                fcf_yield = float(fr["fcf_yield"]) if pd.notna(fr["fcf_yield"]) else 0.04
                roe = float(fr["roe"]) if pd.notna(fr["roe"]) else 0.15
                roic = float(fr["roic"]) if pd.notna(fr["roic"]) else 0.12
                profit_margin = float(fr["profit_margin"]) if pd.notna(fr["profit_margin"]) else 0.15
                earnings_stability = float(fr["earnings_stability"]) if pd.notna(fr["earnings_stability"]) else 0.75
                dte = float(fr["debt_to_equity"]) if pd.notna(fr["debt_to_equity"]) and fr["debt_to_equity"] > 0 else 0.5
                div_yield = float(fr["dividend_yield"]) if pd.notna(fr["dividend_yield"]) else 0.02
                franking_rate = float(fr["franking_rate"]) if pd.notna(fr["franking_rate"]) else 0.0
                mcap = float(fr["market_cap"]) if pd.notna(fr["market_cap"]) and fr["market_cap"] > 0 else adv_20d * 250
            else:
                pe, pb, ev_ebitda, fcf_yield = 20.0, 2.5, 14.0, 0.04
                roe, roic, profit_margin, earnings_stability, dte = 0.15, 0.12, 0.15, 0.75, 0.5
                div_yield, franking_rate, mcap = 0.02, 0.0, adv_20d * 250

            turnover_ratio = adv_20d / max(mcap, 1.0)

            # 5. Market-Specific Factors (ASX Franking Gross-up / US WHT / Commodity & FX Sensitivity / Overnight Lead)
            if inst["market"] == "ASX":
                # Grossed-up dividend yield including Australian franking credits: Yield * (1 + f * tau / (1 - tau))
                franking_multiplier = 1.0 + franking_rate * (self.au_tax / (1.0 - self.au_tax))
                franking_grossed_yield = div_yield * franking_multiplier
            else:
                # Net dividend yield after US dividend withholding tax for non-US resident
                franking_grossed_yield = div_yield * (1.0 - self.us_wht)

            comm_sens = 0.0
            if not comm_ret.empty:
                jc = pd.concat([sym_rets.tail(60), comm_ret], axis=1, join="inner").dropna()
                if len(jc) >= 15:
                    cov_c = np.cov(jc.iloc[:, 0], jc.iloc[:, 1])
                    if cov_c[1, 1] > 1e-10:
                        comm_sens = float(cov_c[0, 1] / cov_c[1, 1])

            aud_sens = 0.0
            if not aud_ret.empty:
                ja = pd.concat([sym_rets.tail(60), aud_ret], axis=1, join="inner").dropna()
                if len(ja) >= 15:
                    cov_a = np.cov(ja.iloc[:, 0], ja.iloc[:, 1])
                    if cov_a[1, 1] > 1e-10:
                        aud_sens = float(cov_a[0, 1] / cov_a[1, 1])

            overnight_sig = (
                beta_60d * us_overnight_ret if inst["market"] == "ASX" else mom_1m * 0.1
            )

            rows.append(
                {
                    "symbol": sym,
                    "name": inst["name"],
                    "trade_date": as_of_date,
                    "market": inst["market"],
                    "currency": inst["currency"],
                    "sector": inst["sector"],
                    "close": round(latest_price, 4),
                    "market_cap": mcap,
                    # Raw display metrics
                    "pe_ratio": round(pe, 2),
                    "pb_ratio": round(pb, 2),
                    "ev_ebitda": round(ev_ebitda, 2),
                    "dividend_yield": round(div_yield, 4),
                    "franking_rate": round(franking_rate, 2),
                    # Value Factors (higher = better value)
                    "earnings_yield": 1.0 / pe,
                    "book_to_price": 1.0 / pb,
                    "ev_ebitda_inv": 1.0 / ev_ebitda,
                    "fcf_yield": fcf_yield,
                    # Quality Factors (higher = higher quality)
                    "roe": roe,
                    "roic": roic,
                    "profit_margin": profit_margin,
                    "earnings_stability": earnings_stability,
                    "balance_sheet_strength": 1.0 / (1.0 + dte),
                    # Momentum Factors
                    "mom_1m": mom_1m,
                    "mom_3m": mom_3m,
                    "mom_6m": mom_6m,
                    "mom_12m": mom_12m,
                    "risk_adj_momentum": risk_adj_mom,
                    # Risk Factors (raw: lower is safer; we invert when building low_risk_score)
                    "volatility_20d": vol_20d,
                    "downside_volatility": downside_vol,
                    "beta_60d": beta_60d,
                    "max_drawdown_120d": max_dd_120d,
                    # Liquidity Factors
                    "adv_20d": np.log1p(adv_20d),
                    "turnover_ratio": turnover_ratio,
                    "spread_tightness": -spread_bps,
                    # Market-Specific Factors
                    "franking_grossed_yield": franking_grossed_yield,
                    "commodity_sensitivity": comm_sens,
                    "audusd_sensitivity": aud_sens,
                    "us_overnight_signal": overnight_sig,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Standardise and optionally sector-neutralise all factor columns
        all_factor_cols = [f for cols in FACTOR_CATEGORIES.values() for f in cols]
        for col in all_factor_cols:
            if col not in df.columns:
                continue
            z_col = f"{col}_z"
            sn_col = f"{col}_sn"
            df[z_col] = zscore_series(df[col], winsorise=True)
            df[sn_col] = sector_neutralise_series(df[col], df["sector"], rescale=True)

        suffix = "_sn" if sector_neutral else "_z"

        # Category composite scores
        df["value_score"] = df[[f"{c}{suffix}" for c in FACTOR_CATEGORIES["value"]]].mean(axis=1)
        df["quality_score"] = df[[f"{c}{suffix}" for c in FACTOR_CATEGORIES["quality"]]].mean(axis=1)
        df["momentum_score"] = df[[f"{c}{suffix}" for c in FACTOR_CATEGORIES["momentum"]]].mean(axis=1)
        # For risk, lower raw risk = higher low_risk_score
        df["low_risk_score"] = -1.0 * df[[f"{c}{suffix}" for c in FACTOR_CATEGORIES["risk"]]].mean(axis=1)
        df["liquidity_score"] = df[[f"{c}{suffix}" for c in FACTOR_CATEGORIES["liquidity"]]].mean(axis=1)
        df["market_specific_score"] = df[
            ["franking_grossed_yield" + suffix, "us_overnight_signal" + suffix]
        ].mean(axis=1)

        w = weights or {
            "value_score": 0.25,
            "quality_score": 0.30,
            "momentum_score": 0.25,
            "low_risk_score": 0.10,
            "market_specific_score": 0.10,
        }
        total_w = sum(w.values()) or 1.0
        df["composite_score"] = sum(
            df[k] * (wt / total_w) for k, wt in w.items() if k in df.columns
        )
        df["composite_rank"] = df["composite_score"].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        return df

    def persist_factors_for_date(self, as_of_date: date, market: str = "ALL") -> pd.DataFrame:
        """Compute factors and persist normalized records to DuckDB & Parquet."""
        df = self.compute_factors_as_of(as_of_date=as_of_date, market=market)
        if df.empty:
            return df

        long_rows: list[dict[str, object]] = []
        all_factor_cols = [f for cols in FACTOR_CATEGORIES.values() for f in cols] + [
            "value_score",
            "quality_score",
            "momentum_score",
            "low_risk_score",
            "liquidity_score",
            "market_specific_score",
            "composite_score",
        ]
        for _, r in df.iterrows():
            for f_name in all_factor_cols:
                if f_name not in r:
                    continue
                raw_v = float(r[f_name])
                z_v = float(r.get(f"{f_name}_z", raw_v))
                sn_v = float(r.get(f"{f_name}_sn", raw_v))
                long_rows.append(
                    {
                        "symbol": r["symbol"],
                        "trade_date": as_of_date,
                        "market": r["market"],
                        "sector": r["sector"],
                        "factor_name": f_name,
                        "raw_value": raw_v,
                        "zscore_value": z_v,
                        "sector_neutral_value": sn_v,
                    }
                )
        self.store.save_factor_values(pd.DataFrame(long_rows))
        return df
