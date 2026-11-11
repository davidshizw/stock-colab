"""Full-Spectrum Single-Stock Quantitative & Market Deep-Dive Analyzer.

Given ANY stock ticker (e.g. ``EML``, ``EML.AX``, ``AU.EML``, ``NVDA``, ``BHP.AX``),
this module performs a comprehensive 6-dimension institutional analysis:
1. Valuation & Market Profile (PE, Forward PE, PB, PS, EV/EBITDA, Dividend/Franking, Ownership)
2. Multi-Year Financial Statements & Balance Sheet Health (Income, Cash Flow, Debt/Cash, Margins)
3. Full Technical & Microstructure Suite (MA20/50/200, RSI-14, MACD, Stoch, Bollinger, ATR, VaR/CVaR, Volume Surge, Support/Resistance)
4. Cross-Market Macro Beta & Correlation (^AXJO, SPY, QQQ, AUDUSD=X, Gold, Copper, US10Y)
5. Direct Sector/Industry Peer Benchmarking (live peer group comparison table)
6. Analyst Consensus + Grok News Sentiment + 5-Pillar Quant Scorecard & Moomoo Execution Plan
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from quant_platform.agents.grok_bridge import GrokColabBridge

# Curated peer groups by market & sector/sub-industry for instant benchmarking
ASX_PEER_GROUPS: dict[str, list[str]] = {
    "FINTECH_PAYMENTS": ["EML.AX", "TYR.AX", "ZIP.AX", "OFX.AX", "HUM.AX", "SQ2.AX"],
    "Technology": ["WTC.AX", "XRO.AX", "TNE.AX", "MP1.AX", "NXT.AX", "CDA.AX"],
    "Basic Materials": ["BHP.AX", "RIO.AX", "FMG.AX", "SFR.AX", "LYC.AX", "PLS.AX"],
    "Healthcare": ["CSL.AX", "PME.AX", "RMD.AX", "COH.AX", "FPH.AX", "TLX.AX"],
    "Financial Services": ["CBA.AX", "NAB.AX", "WBC.AX", "MQG.AX", "SUN.AX", "HUB.AX"],
    "Energy": ["WDS.AX", "STO.AX", "YAL.AX", "WHC.AX", "PDN.AX", "BOE.AX"],
    "Consumer": ["WES.AX", "WOW.AX", "COL.AX", "JBH.AX", "DMP.AX", "QAN.AX"],
}

US_PEER_GROUPS: dict[str, list[str]] = {
    "Technology": ["NVDA", "AVGO", "MSFT", "AAPL", "AMD", "MU", "PLTR", "CRM"],
    "Industrials": ["EML", "CAT", "DE", "ETN", "PH", "HON", "GE"],
    "Financial Services": ["JPM", "GS", "MS", "V", "MA", "AXP", "HOOD"],
    "Healthcare": ["LLY", "UNH", "ABBV", "ISRG", "VRTX", "REGN"],
    "Consumer": ["AMZN", "TSLA", "COST", "WMT", "META", "NFLX"],
}

MACRO_BENCHMARKS = ["^AXJO", "SPY", "QQQ", "AUDUSD=X", "GC=F", "HG=F", "^TNX"]


def _safe_float(val: Any, default: float | None = None) -> float | None:
    if val is None:
        return default
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _round_or_none(val: Any, digits: int = 2) -> float | None:
    f = _safe_float(val, None)
    return round(f, digits) if f is not None else None


class StockDeepDiveAnalyzer:
    """Comprehensive single-stock full-metric + cross-market + peer + sentiment analyzer."""

    def __init__(self, grok_bridge: GrokColabBridge | None = None) -> None:
        self.grok = grok_bridge or GrokColabBridge()

    @staticmethod
    def resolve_ticker(raw_symbol: str, prefer_market: str = "AUTO") -> dict[str, Any]:
        """Resolve user input (e.g. 'EML', 'EML.AX', 'AU.EML', 'US.EML') to canonical Yahoo ticker(s)."""
        s = raw_symbol.strip().upper()
        if s.startswith("AU."):
            primary = f"{s[3:]}.AX"
            return {"input": raw_symbol, "primary_symbol": primary, "alternate_symbols": []}
        if s.startswith("US."):
            primary = s[3:]
            return {"input": raw_symbol, "primary_symbol": primary, "alternate_symbols": []}
        if s.endswith(".AX"):
            base = s[:-3]
            return {"input": raw_symbol, "primary_symbol": s, "alternate_symbols": [base]}

        # Bare ticker like "EML" -> check both ASX (EML.AX) and US (EML)
        asx_sym = f"{s}.AX"
        us_sym = s

        has_asx = False
        has_us = False
        asx_name = None
        us_name = None

        try:
            asx_info = yf.Ticker(asx_sym).info or {}
            if asx_info.get("regularMarketPrice") or asx_info.get("currentPrice"):
                has_asx = True
                asx_name = asx_info.get("longName") or asx_info.get("shortName")
        except Exception:
            pass

        try:
            us_info = yf.Ticker(us_sym).info or {}
            if us_info.get("regularMarketPrice") or us_info.get("currentPrice"):
                has_us = True
                us_name = us_info.get("longName") or us_info.get("shortName")
        except Exception:
            pass

        if prefer_market.upper() == "US" and has_us:
            return {
                "input": raw_symbol,
                "primary_symbol": us_sym,
                "alternate_symbols": [asx_sym] if has_asx else [],
                "matches": {"US": {"symbol": us_sym, "name": us_name}, "ASX": {"symbol": asx_sym, "name": asx_name} if has_asx else None},
            }

        # Default priority: if ASX exists and prefer_market is AUTO or ASX, use ASX as primary when user trades ASX finTech like EML, or if only ASX exists
        if has_asx and (prefer_market.upper() in ("AUTO", "ASX") or not has_us):
            return {
                "input": raw_symbol,
                "primary_symbol": asx_sym,
                "alternate_symbols": [us_sym] if has_us else [],
                "matches": {
                    "ASX": {"symbol": asx_sym, "name": asx_name},
                    "US": {"symbol": us_sym, "name": us_name} if has_us else None,
                },
            }

        return {
            "input": raw_symbol,
            "primary_symbol": us_sym if has_us else asx_sym,
            "alternate_symbols": [],
            "matches": {
                "US": {"symbol": us_sym, "name": us_name} if has_us else None,
                "ASX": {"symbol": asx_sym, "name": asx_name} if has_asx else None,
            },
        }

    def analyze(self, raw_symbol: str, prefer_market: str = "AUTO", include_alternates: bool = True) -> dict[str, Any]:
        """Run full-metric + market analysis on the target stock (and optional homonym summary)."""
        resolution = self.resolve_ticker(raw_symbol, prefer_market=prefer_market)
        sym = resolution["primary_symbol"]
        report = self._analyze_single_canonical(sym)
        report["resolution"] = resolution

        # If there is an alternate listing/homonym (e.g. US: EML vs ASX: EML.AX), include a compact snapshot
        alt_snapshots: list[dict[str, Any]] = []
        if include_alternates:
            for alt_sym in resolution.get("alternate_symbols", []):
                try:
                    alt_t = yf.Ticker(alt_sym)
                    alt_info = alt_t.info or {}
                    price = _safe_float(alt_info.get("currentPrice") or alt_info.get("regularMarketPrice"))
                    if price:
                        alt_snapshots.append({
                            "symbol": alt_sym,
                            "moomoo_code": f"AU.{alt_sym[:-3]}" if alt_sym.endswith(".AX") else f"US.{alt_sym}",
                            "name": alt_info.get("longName") or alt_info.get("shortName") or alt_sym,
                            "exchange": alt_info.get("exchange", ""),
                            "currency": alt_info.get("currency", "USD"),
                            "sector": alt_info.get("sector", "Unknown"),
                            "industry": alt_info.get("industry", "Unknown"),
                            "price": round(price, 4),
                            "market_cap": alt_info.get("marketCap"),
                            "trailing_pe": _round_or_none(alt_info.get("trailingPE")),
                            "forward_pe": _round_or_none(alt_info.get("forwardPE")),
                            "price_to_book": _round_or_none(alt_info.get("priceToBook")),
                            "roe_pct": _round_or_none((_safe_float(alt_info.get("returnOnEquity"), 0.0) or 0.0) * 100),
                            "52w_low": _round_or_none(alt_info.get("fiftyTwoWeekLow"), 4),
                            "52w_high": _round_or_none(alt_info.get("fiftyTwoWeekHigh"), 4),
                        })
                except Exception:
                    pass
        report["alternate_listings"] = alt_snapshots
        return report

    def _analyze_single_canonical(self, sym: str) -> dict[str, Any]:
        is_asx = sym.endswith(".AX")
        market = "ASX" if is_asx else "US"
        moomoo_code = f"AU.{sym[:-3]}" if is_asx else f"US.{sym}"

        ticker = yf.Ticker(sym)
        info = ticker.info or {}

        # 1. Fetch 1-Year Daily OHLCV + Macro Benchmarks in one batch
        dl_symbols = [sym] + MACRO_BENCHMARKS
        raw_df = yf.download(
            tickers=dl_symbols,
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
        )

        stock_df = raw_df[sym].dropna(how="all").copy() if sym in raw_df.columns.get_level_values(0) else pd.DataFrame()
        if stock_df.empty:
            raise ValueError(f"No historical price data found for symbol {sym}")

        # Technicals & Price Series
        tech = self._compute_technicals(stock_df)

        # Cross-Market Beta & Correlations
        cross_market = self._compute_cross_market_sensitivity(sym, raw_df)

        # Valuation & Company Profile
        profile_valuation = self._extract_profile_and_valuation(sym, market, moomoo_code, info, tech["close"])

        # Multi-Year Financials & Balance Sheet
        financials_health = self._extract_financial_statements(ticker, info, profile_valuation["market_cap"])

        # Peer Comparison
        peers = self._compute_peer_comparison(sym, market, info)

        # Analyst Consensus & Grok News Sentiment
        sentiment_catalysts = self._compute_sentiment_and_consensus(sym, ticker, info, tech["close"])

        # 5-Pillar Scorecard & Execution Plan
        scorecard = self._compute_scorecard_and_plan(
            sym=sym,
            market=market,
            moomoo_code=moomoo_code,
            profile=profile_valuation,
            fin=financials_health,
            tech=tech,
            sentiment=sentiment_catalysts,
        )

        # Recent 120-day chart series for UI visualization
        recent_df = stock_df.tail(120).copy()
        close_s = recent_df["Close"].astype(float)
        sma20_s = close_s.rolling(20, min_periods=1).mean()
        sma50_s = close_s.rolling(50, min_periods=1).mean()
        chart_series = [
            {
                "date": pd.Timestamp(idx).strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"] or 0),
                "sma20": round(float(sma20_s.loc[idx]), 4),
                "sma50": round(float(sma50_s.loc[idx]), 4),
            }
            for idx, row in recent_df.iterrows()
        ]

        return {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "symbol": sym,
            "moomoo_code": moomoo_code,
            "market": market,
            "profile_and_valuation": profile_valuation,
            "financial_health": financials_health,
            "technicals": tech,
            "cross_market_sensitivity": cross_market,
            "peer_comparison": peers,
            "sentiment_and_catalysts": sentiment_catalysts,
            "scorecard_and_execution": scorecard,
            "chart_series_120d": chart_series,
        }

    def _extract_profile_and_valuation(
        self, sym: str, market: str, moomoo_code: str, info: dict[str, Any], latest_close: float
    ) -> dict[str, Any]:
        div_yield = _safe_float(info.get("dividendYield"), 0.0) or 0.0
        if div_yield > 0.25:
            div_yield = div_yield / 100.0
        franking_rate = 0.85 if (market == "ASX" and div_yield > 0) else 0.0
        grossed_up_yield = div_yield * (1.0 + franking_rate * (0.30 / 0.70)) if market == "ASX" else div_yield * 0.85

        return {
            "symbol": sym,
            "moomoo_code": moomoo_code,
            "name": info.get("longName") or info.get("shortName") or sym,
            "market": market,
            "exchange": info.get("exchange", "ASX" if market == "ASX" else "NMS"),
            "currency": info.get("currency", "AUD" if market == "ASX" else "USD"),
            "sector": info.get("sector", "Technology"),
            "industry": info.get("industry", "Software - Infrastructure"),
            "website": info.get("website", ""),
            "employees": info.get("fullTimeEmployees"),
            "business_summary": (info.get("longBusinessSummary") or "")[:600],
            "price": round(latest_close, 4),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "trailing_pe": _round_or_none(info.get("trailingPE")),
            "forward_pe": _round_or_none(info.get("forwardPE")),
            "peg_ratio": _round_or_none(info.get("pegRatio")),
            "price_to_book": _round_or_none(info.get("priceToBook")),
            "price_to_sales": _round_or_none(info.get("priceToSalesTrailing12Months")),
            "ev_to_revenue": _round_or_none(info.get("enterpriseToRevenue")),
            "ev_to_ebitda": _round_or_none(info.get("enterpriseToEbitda")),
            "dividend_yield_pct": round(div_yield * 100, 2),
            "franking_rate_pct": round(franking_rate * 100, 0),
            "after_tax_grossed_yield_pct": round(grossed_up_yield * 100, 2),
            "beta_yahoo": _round_or_none(info.get("beta")),
            "held_pct_insiders": _round_or_none((_safe_float(info.get("heldPercentInsiders"), 0.0) or 0.0) * 100),
            "held_pct_institutions": _round_or_none((_safe_float(info.get("heldPercentInstitutions"), 0.0) or 0.0) * 100),
        }

    def _extract_financial_statements(
        self, ticker: yf.Ticker, info: dict[str, Any], market_cap: int | None
    ) -> dict[str, Any]:
        gross_margin = (_safe_float(info.get("grossMargins"), 0.0) or 0.0) * 100
        ebitda_margin = (_safe_float(info.get("ebitdaMargins"), 0.0) or 0.0) * 100
        oper_margin = (_safe_float(info.get("operatingMargins"), 0.0) or 0.0) * 100
        profit_margin = (_safe_float(info.get("profitMargins"), 0.0) or 0.0) * 100
        roe = (_safe_float(info.get("returnOnEquity"), 0.0) or 0.0) * 100
        roa = (_safe_float(info.get("returnOnAssets"), 0.0) or 0.0) * 100
        rev_growth = _safe_float(info.get("revenueGrowth"))
        earn_growth = _safe_float(info.get("earningsGrowth"))

        total_cash = _safe_float(info.get("totalCash"))
        total_debt = _safe_float(info.get("totalDebt"))
        fcf = _safe_float(info.get("freeCashflow"))
        ocf = _safe_float(info.get("operatingCashflow"))
        dte = _safe_float(info.get("debtToEquity"))
        curr_ratio = _safe_float(info.get("currentRatio"))
        quick_ratio = _safe_float(info.get("quickRatio"))

        # Multi-year income statement & balance sheet history
        annual_history: list[dict[str, Any]] = []
        try:
            inc = ticker.financials
            bs = ticker.balance_sheet
            cf = ticker.cashflow
            if inc is not None and not inc.empty:
                for col in list(inc.columns)[:3]:
                    period_str = pd.Timestamp(col).strftime("%Y-%m-%d")
                    rev = _safe_float(inc.loc["Total Revenue", col]) if "Total Revenue" in inc.index else None
                    gp = _safe_float(inc.loc["Gross Profit", col]) if "Gross Profit" in inc.index else None
                    norm_ebitda = (
                        _safe_float(inc.loc["Normalized EBITDA", col])
                        if "Normalized EBITDA" in inc.index
                        else (_safe_float(inc.loc["EBITDA", col]) if "EBITDA" in inc.index else None)
                    )
                    net_int = (
                        _safe_float(inc.loc["Net Interest Income", col])
                        if "Net Interest Income" in inc.index
                        else None
                    )
                    net_inc = _safe_float(inc.loc["Net Income", col]) if "Net Income" in inc.index else None
                    eps = _safe_float(inc.loc["Diluted EPS", col]) if "Diluted EPS" in inc.index else None

                    equity = None
                    tangible_bv = None
                    net_debt_yr = None
                    if bs is not None and not bs.empty and col in bs.columns:
                        if "Stockholders Equity" in bs.index:
                            equity = _safe_float(bs.loc["Stockholders Equity", col])
                        if "Tangible Book Value" in bs.index:
                            tangible_bv = _safe_float(bs.loc["Tangible Book Value", col])
                        if "Net Debt" in bs.index:
                            net_debt_yr = _safe_float(bs.loc["Net Debt", col])

                    fcf_yr = None
                    if cf is not None and not cf.empty and col in cf.columns and "Free Cash Flow" in cf.index:
                        fcf_yr = _safe_float(cf.loc["Free Cash Flow", col])

                    annual_history.append({
                        "period_ending": period_str,
                        "total_revenue": rev,
                        "gross_profit": gp,
                        "normalized_ebitda": norm_ebitda,
                        "net_interest_income": net_int,
                        "net_income": net_inc,
                        "diluted_eps": eps,
                        "stockholders_equity": equity,
                        "tangible_book_value": tangible_bv,
                        "net_debt": net_debt_yr,
                        "free_cash_flow": fcf_yr,
                    })
        except Exception:
            pass

        # Compute YoY Revenue change from annual_history if Yahoo's revenueGrowth is None
        if rev_growth is None and len(annual_history) >= 2:
            r0 = annual_history[0].get("total_revenue")
            r1 = annual_history[1].get("total_revenue")
            if r0 and r1 and r1 != 0:
                rev_growth = (r0 - r1) / abs(r1)

        net_debt = (total_debt or 0.0) - (total_cash or 0.0) if (total_debt is not None or total_cash is not None) else None
        cash_to_mcap_pct = (total_cash / market_cap * 100) if (total_cash and market_cap) else None

        return {
            "gross_margin_pct": round(gross_margin, 2),
            "ebitda_margin_pct": round(ebitda_margin, 2),
            "operating_margin_pct": round(oper_margin, 2),
            "net_profit_margin_pct": round(profit_margin, 2),
            "roe_pct": round(roe, 2),
            "roa_pct": round(roa, 2),
            "revenue_growth_yoy_pct": _round_or_none(rev_growth * 100 if rev_growth is not None else None),
            "earnings_growth_yoy_pct": _round_or_none(earn_growth * 100 if earn_growth is not None else None),
            "total_cash": total_cash,
            "total_debt": total_debt,
            "net_debt": net_debt,
            "cash_to_market_cap_pct": _round_or_none(cash_to_mcap_pct),
            "free_cash_flow": fcf,
            "operating_cash_flow": ocf,
            "debt_to_equity_pct": _round_or_none(dte),
            "current_ratio": _round_or_none(curr_ratio),
            "quick_ratio": _round_or_none(quick_ratio),
            "annual_statements": annual_history,
        }

    def _compute_technicals(self, df: pd.DataFrame) -> dict[str, Any]:
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        vol = df["Volume"].fillna(0).astype(float)

        curr = float(close.iloc[-1])

        def _ret(days: int) -> float:
            if len(close) > days:
                return float(curr / close.iloc[-1 - days] - 1.0)
            return float(curr / close.iloc[0] - 1.0)

        ret_1d = _ret(1)
        ret_5d = _ret(5)
        ret_1m = _ret(21)
        ret_3m = _ret(63)
        ret_6m = _ret(126)
        ret_12m = _ret(min(251, len(close) - 1))

        # Moving Averages
        sma20 = float(close.tail(20).mean())
        sma50 = float(close.tail(50).mean()) if len(close) >= 50 else sma20
        sma200 = float(close.tail(200).mean()) if len(close) >= 200 else sma50
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])

        # RSI-14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_series = 100.0 - (100.0 / (1.0 + rs))
        rsi_14 = float(rsi_series.iloc[-1]) if not np.isnan(rsi_series.iloc[-1]) else 50.0

        # MACD (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_sig

        # Stochastic (14, 3)
        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        stoch_k = ((close - low14) / (high14 - low14).replace(0, np.nan)) * 100.0
        stoch_d = stoch_k.rolling(3).mean()

        # Bollinger Bands (20, 2)
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_up = float((bb_mid + 2.0 * bb_std).iloc[-1])
        bb_low = float((bb_mid - 2.0 * bb_std).iloc[-1])
        bb_pct_b = (curr - bb_low) / (bb_up - bb_low) if (bb_up > bb_low) else 0.5
        bb_width = (bb_up - bb_low) / sma20 if sma20 > 0 else 0.0

        # ATR-14
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr_14 = float(tr.tail(14).mean())
        atr_pct = (atr_14 / curr * 100.0) if curr > 0 else 0.0

        # Volatility & Tail Risk
        daily_rets = close.pct_change().dropna()
        vol_20d = float(daily_rets.tail(20).std() * np.sqrt(252))
        vol_60d = float(daily_rets.tail(60).std() * np.sqrt(252))
        vol_252d = float(daily_rets.std() * np.sqrt(252))

        cum = (1.0 + daily_rets).cumprod()
        peak = cum.cummax()
        dd = (cum / peak) - 1.0
        max_dd_1y = float(dd.min())

        sharpe_1y = float((daily_rets.mean() * 252) / (vol_252d + 1e-9))
        downside_rets = daily_rets[daily_rets < 0]
        downside_vol = float(downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 2 else vol_252d
        sortino_1y = float((daily_rets.mean() * 252) / (downside_vol + 1e-9))

        var_95 = float(np.percentile(daily_rets, 5))
        cvar_95 = float(daily_rets[daily_rets <= var_95].mean()) if any(daily_rets <= var_95) else var_95

        # Volume & Support / Resistance
        vol_5d = float(vol.tail(5).mean())
        vol_20d_avg = float(vol.tail(20).mean()) + 1.0
        vol_surge = vol_5d / vol_20d_avg
        dollar_turnover_20d = vol_20d_avg * curr

        high_52w = float(high.max())
        low_52w = float(low.min())
        recent_20d_low = float(low.tail(20).min())
        recent_20d_high = float(high.tail(20).max())
        recent_60d_low = float(low.tail(60).min())
        recent_60d_high = float(high.tail(60).max())

        if curr > sma20 > sma50 > sma200:
            trend_regime = "多头排列 (BULLISH_ALIGNMENT)"
        elif curr > sma20 and curr > sma50 and curr < sma200:
            trend_regime = "底部反转修复 (BOTTOMING_RECOVERY)"
        elif curr < sma20 < sma50 < sma200:
            trend_regime = "空头寻底趋势 (BEARISH_DOWNTREND)"
        else:
            trend_regime = "区间震荡筑底 (RANGE_CONSOLIDATION)"

        return {
            "close": round(curr, 4),
            "ret_1d_pct": round(ret_1d * 100, 2),
            "ret_5d_pct": round(ret_5d * 100, 2),
            "ret_1m_pct": round(ret_1m * 100, 2),
            "ret_3m_pct": round(ret_3m * 100, 2),
            "ret_6m_pct": round(ret_6m * 100, 2),
            "ret_12m_pct": round(ret_12m * 100, 2),
            "sma_20": round(sma20, 4),
            "sma_50": round(sma50, 4),
            "sma_200": round(sma200, 4),
            "ema_20": round(ema20, 4),
            "price_vs_sma20_pct": round((curr / sma20 - 1.0) * 100, 2),
            "price_vs_sma50_pct": round((curr / sma50 - 1.0) * 100, 2),
            "price_vs_sma200_pct": round((curr / sma200 - 1.0) * 100, 2),
            "trend_regime": trend_regime,
            "rsi_14": round(rsi_14, 2),
            "macd_line": round(float(macd_line.iloc[-1]), 4),
            "macd_signal": round(float(macd_sig.iloc[-1]), 4),
            "macd_histogram": round(float(macd_hist.iloc[-1]), 4),
            "stoch_k": round(float(stoch_k.iloc[-1]) if not np.isnan(stoch_k.iloc[-1]) else 50.0, 2),
            "stoch_d": round(float(stoch_d.iloc[-1]) if not np.isnan(stoch_d.iloc[-1]) else 50.0, 2),
            "bollinger_upper": round(bb_up, 4),
            "bollinger_lower": round(bb_low, 4),
            "bollinger_pct_b": round(bb_pct_b, 2),
            "bollinger_bandwidth_pct": round(bb_width * 100, 2),
            "atr_14": round(atr_14, 4),
            "atr_pct": round(atr_pct, 2),
            "vol_20d_ann_pct": round(vol_20d * 100, 2),
            "vol_60d_ann_pct": round(vol_60d * 100, 2),
            "vol_252d_ann_pct": round(vol_252d * 100, 2),
            "max_drawdown_1y_pct": round(max_dd_1y * 100, 2),
            "sharpe_1y": round(sharpe_1y, 2),
            "sortino_1y": round(sortino_1y, 2),
            "var_95_daily_pct": round(var_95 * 100, 2),
            "cvar_95_daily_pct": round(cvar_95 * 100, 2),
            "avg_vol_20d": int(vol_20d_avg),
            "avg_dollar_turnover_20d": round(dollar_turnover_20d, 0),
            "volume_surge_5d_vs_20d": round(vol_surge, 2),
            "high_52w": round(high_52w, 4),
            "low_52w": round(low_52w, 4),
            "pct_from_52w_high": round((curr / high_52w - 1.0) * 100, 2),
            "pct_from_52w_low": round((curr / low_52w - 1.0) * 100, 2),
            "support_1_20d_low": round(recent_20d_low, 4),
            "support_2_60d_low": round(recent_60d_low, 4),
            "resistance_1_20d_high": round(recent_20d_high, 4),
            "resistance_2_60d_high": round(recent_60d_high, 4),
        }

    def _compute_cross_market_sensitivity(self, sym: str, raw_df: pd.DataFrame) -> list[dict[str, Any]]:
        labels = {
            "^AXJO": "澳洲 ASX 200 指数",
            "SPY": "美国标普 500 (SPY)",
            "QQQ": "美国纳斯达克 100 (QQQ)",
            "AUDUSD=X": "澳元/美元汇率 (AUD/USD)",
            "GC=F": "COMEX 黄金期货 (Gold)",
            "HG=F": "COMEX 铜期货 (Copper)",
            "^TNX": "美债 10 年期收益率 (US 10Y)",
        }
        top_cols = set(raw_df.columns.get_level_values(0))
        if sym not in top_cols:
            return []

        s_rets = raw_df[sym]["Close"].astype(float).pct_change().dropna()
        results: list[dict[str, Any]] = []

        for b_sym in MACRO_BENCHMARKS:
            if b_sym not in top_cols:
                continue
            b_rets = raw_df[b_sym]["Close"].astype(float).pct_change().dropna()
            aligned = pd.concat([s_rets.rename("s"), b_rets.rename("b")], axis=1).dropna()
            if len(aligned) < 20:
                continue
            corr = float(aligned["s"].corr(aligned["b"]))
            cov = float(aligned["s"].cov(aligned["b"]))
            var_b = float(aligned["b"].var())
            beta = cov / var_b if var_b > 1e-12 else 0.0
            results.append({
                "benchmark": b_sym,
                "name": labels.get(b_sym, b_sym),
                "correlation_1y": round(corr, 3),
                "beta_1y": round(beta, 2),
                "relationship": (
                    "正相关共振" if corr >= 0.25 else ("负相关对冲" if corr <= -0.20 else "低相关独立行情")
                ),
            })
        return results

    def _compute_peer_comparison(
        self, sym: str, market: str, info: dict[str, Any]
    ) -> list[dict[str, Any]]:
        sector = info.get("sector", "Technology")
        if sym in ("EML.AX", "TYR.AX", "ZIP.AX", "OFX.AX", "HUM.AX"):
            peer_list = ASX_PEER_GROUPS["FINTECH_PAYMENTS"]
        elif market == "ASX":
            peer_list = ASX_PEER_GROUPS.get(sector, ASX_PEER_GROUPS["Technology"])
        else:
            peer_list = US_PEER_GROUPS.get(sector, US_PEER_GROUPS["Technology"])

        peers = [sym] + [p for p in peer_list if p != sym][:5]
        rows: list[dict[str, Any]] = []

        for p_sym in peers:
            try:
                p_info = yf.Ticker(p_sym).info or {}
                price = _safe_float(p_info.get("currentPrice") or p_info.get("regularMarketPrice"))
                if not price:
                    continue
                target_mean = _safe_float(p_info.get("targetMeanPrice"))
                upside_pct = ((target_mean / price - 1.0) * 100.0) if (target_mean and price > 0) else None
                rows.append({
                    "symbol": p_sym,
                    "is_target": p_sym == sym,
                    "name": (p_info.get("shortName") or p_info.get("longName") or p_sym)[:26],
                    "price": round(price, 3),
                    "market_cap_m": round((_safe_float(p_info.get("marketCap"), 0.0) or 0.0) / 1e6, 1),
                    "forward_pe": _round_or_none(p_info.get("forwardPE"), 1),
                    "price_to_book": _round_or_none(p_info.get("priceToBook"), 2),
                    "gross_margin_pct": _round_or_none((_safe_float(p_info.get("grossMargins"), 0.0) or 0.0) * 100, 1),
                    "roe_pct": _round_or_none((_safe_float(p_info.get("returnOnEquity"), 0.0) or 0.0) * 100, 1),
                    "52w_change_pct": _round_or_none((_safe_float(p_info.get("52WeekChange"), 0.0) or 0.0) * 100, 1),
                    "analyst_upside_pct": _round_or_none(upside_pct, 1),
                })
            except Exception:
                continue
        return rows

    def _compute_sentiment_and_consensus(
        self, sym: str, ticker: yf.Ticker, info: dict[str, Any], current_price: float
    ) -> dict[str, Any]:
        target_mean = _safe_float(info.get("targetMeanPrice"))
        target_high = _safe_float(info.get("targetHighPrice"))
        target_low = _safe_float(info.get("targetLowPrice"))
        target_median = _safe_float(info.get("targetMedianPrice"))
        num_analysts = info.get("numberOfAnalystOpinions") or 0
        rec_key = info.get("recommendationKey") or "none"

        upside_mean_pct = ((target_mean / current_price - 1.0) * 100.0) if (target_mean and current_price > 0) else None
        upside_high_pct = ((target_high / current_price - 1.0) * 100.0) if (target_high and current_price > 0) else None
        upside_low_pct = ((target_low / current_price - 1.0) * 100.0) if (target_low and current_price > 0) else None

        articles = self.grok.fetch_live_news_for_symbol(sym, max_items=6)
        lex_score, top_hl = self.grok._score_news_lexicon(articles)
        # Convert [-1, 1] lexicon score + analyst upside into [0, 100] sentiment score
        upside_adj = max(-15.0, min(20.0, (upside_mean_pct or 0.0) * 0.15))
        grok_score_100 = float(np.clip(50.0 + lex_score * 35.0 + upside_adj, 10.0, 95.0))
        if grok_score_100 >= 65.0:
            sent_label = "BULLISH (偏多/催化积极)"
        elif grok_score_100 <= 40.0:
            sent_label = "CAUTIOUS (谨慎/修复期)"
        else:
            sent_label = "NEUTRAL_TURNAROUND (中性/转型观察)"

        return {
            "analyst_count": num_analysts,
            "recommendation_key": rec_key.upper(),
            "target_low": _round_or_none(target_low, 4),
            "target_mean": _round_or_none(target_mean, 4),
            "target_median": _round_or_none(target_median, 4),
            "target_high": _round_or_none(target_high, 4),
            "implied_upside_mean_pct": _round_or_none(upside_mean_pct, 1),
            "implied_upside_high_pct": _round_or_none(upside_high_pct, 1),
            "implied_upside_low_pct": _round_or_none(upside_low_pct, 1),
            "grok_sentiment_score": round(grok_score_100, 1),
            "sentiment_label": sent_label,
            "catalyst_summary": top_hl,
            "recent_news": articles,
        }

    def _compute_scorecard_and_plan(
        self,
        sym: str,
        market: str,
        moomoo_code: str,
        profile: dict[str, Any],
        fin: dict[str, Any],
        tech: dict[str, Any],
        sentiment: dict[str, Any],
    ) -> dict[str, Any]:
        # 1. Value Score (0-100): Forward PE, Price-to-Book, Analyst Upside
        f_pe = profile.get("forward_pe")
        pb = profile.get("price_to_book")
        upside = sentiment.get("implied_upside_mean_pct") or 0.0

        val_pts = 50.0
        if f_pe is not None and 0 < f_pe <= 35:
            val_pts += max(-20.0, min(25.0, (20.0 - f_pe) * 1.5))
        if pb is not None and pb > 0:
            val_pts += max(-15.0, min(20.0, (2.0 - pb) * 12.0))
        val_pts += max(-15.0, min(20.0, upside * 0.25))
        value_score = round(float(np.clip(val_pts, 10.0, 98.0)), 1)

        # 2. Quality & Financial Health Score (0-100)
        qual_pts = 50.0
        gm = fin.get("gross_margin_pct") or 0.0
        roe = fin.get("roe_pct") or 0.0
        fcf = fin.get("free_cash_flow")
        cr = fin.get("current_ratio") or 1.0

        qual_pts += max(-15.0, min(20.0, (gm - 30.0) * 0.4))
        qual_pts += max(-25.0, min(25.0, roe * 0.8))
        if fcf is not None:
            qual_pts += 10.0 if fcf > 0 else -12.0
        if cr < 1.0:
            qual_pts -= 8.0
        quality_score = round(float(np.clip(qual_pts, 10.0, 98.0)), 1)

        # 3. Momentum Score (0-100)
        mom_pts = 50.0
        r1m = tech.get("ret_1m_pct") or 0.0
        r3m = tech.get("ret_3m_pct") or 0.0
        r6m = tech.get("ret_6m_pct") or 0.0
        mom_pts += max(-35.0, min(35.0, r1m * 0.5 + r3m * 0.35 + r6m * 0.25))
        momentum_score = round(float(np.clip(mom_pts, 5.0, 98.0)), 1)

        # 4. Technical Structure Score (0-100)
        tech_pts = 50.0
        rsi = tech.get("rsi_14") or 50.0
        vs_sma20 = tech.get("price_vs_sma20_pct") or 0.0
        vs_sma50 = tech.get("price_vs_sma50_pct") or 0.0
        vol_surge = tech.get("volume_surge_5d_vs_20d") or 1.0

        if vs_sma20 > 0:
            tech_pts += 10.0
        else:
            tech_pts -= 8.0
        if vs_sma50 > 0:
            tech_pts += 10.0
        else:
            tech_pts -= 8.0
        if 30 <= rsi <= 65:
            tech_pts += 8.0
        elif rsi < 30:
            tech_pts += 5.0  # oversold bounce potential
        if vol_surge >= 1.15 and vs_sma20 >= 0:
            tech_pts += 10.0
        technical_score = round(float(np.clip(tech_pts, 10.0, 98.0)), 1)

        # 5. Sentiment & Catalyst Score (0-100)
        catalyst_score = float(sentiment.get("grok_sentiment_score") or 50.0)

        # Composite Score
        composite_score = round(
            0.25 * value_score
            + 0.25 * quality_score
            + 0.20 * momentum_score
            + 0.15 * technical_score
            + 0.15 * catalyst_score,
            1,
        )

        # Action Verdict
        if composite_score >= 72.0:
            verdict = "STRONG_BUY (主升浪/基本面共振强买)"
        elif composite_score >= 58.0:
            verdict = "ACCUMULATE (逢低分批建仓)"
        elif composite_score >= 45.0 or (value_score >= 70.0 and quality_score < 45.0):
            verdict = "SPECULATIVE_TURNAROUND / WATCH (困境反转博弈 · 严控仓位与止损)"
        else:
            verdict = "AVOID / WAIT_FOR_TREND (趋势偏弱 · 观望等待右侧突破)"

        close = tech["close"]
        atr = tech["atr_14"]
        decimals = 4 if close < 2.0 else 2

        entry_low = round(max(0.001, close - 0.8 * atr), decimals)
        entry_high = round(close + 0.3 * atr, decimals)
        stop_loss = round(max(0.001, close - 2.5 * atr), decimals)
        right_side_breakout = round(max(tech["sma_50"], tech["resistance_1_20d_high"]), decimals)

        target_1 = round(max(close * 1.18, tech["resistance_2_60d_high"]), decimals)
        target_2 = round(sentiment["target_mean"] or (close * 1.35), decimals)

        risk_per_share = max(1e-4, close - stop_loss)
        reward_per_share = max(0.0, target_2 - close)
        rr_ratio = round(reward_per_share / risk_per_share, 2)

        return {
            "value_score": value_score,
            "quality_score": quality_score,
            "momentum_score": momentum_score,
            "technical_score": technical_score,
            "catalyst_score": catalyst_score,
            "composite_score": composite_score,
            "verdict": verdict,
            "current_price": close,
            "entry_zone_low": entry_low,
            "entry_zone_high": entry_high,
            "right_side_breakout_trigger": right_side_breakout,
            "stop_loss_2_5_atr": stop_loss,
            "target_1_technical": target_1,
            "target_2_consensus": target_2,
            "risk_reward_ratio": rr_ratio,
        }
