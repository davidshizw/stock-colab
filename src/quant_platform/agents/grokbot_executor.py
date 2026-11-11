"""Grok Bot + Moomoo MCP Execution & Paper-Trading Signal Generator.

Bridges the Colab/Local Quantitative + ML + Grok Sentiment Engine with Grok Bot (`~/.grokbot`)
and Moomoo OpenAPI MCP (`https://mcp.moomoo.com/mcp`).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from quant_platform.agents.grok_bridge import GrokColabBridge
from quant_platform.screener.discovery import MarketWidePotentialScanner


def to_moomoo_code(symbol: str) -> str:
    """Convert Yahoo Finance symbol (e.g. 'BHP.AX', 'NVDA') to Moomoo OpenAPI code ('AU.BHP', 'US.NVDA')."""
    if symbol.endswith(".AX"):
        return f"AU.{symbol[:-3]}"
    return f"US.{symbol}"


class GrokBotSignalGenerator:
    """Produces structured Watchlist, Entry/Stop/Target levels, and Paper-Trading Order Plans for Grok Bot + Moomoo MCP."""

    def __init__(
        self,
        capital_aud: float = 100_000.0,
        audusd_rate: float = 0.69,
        output_path: Path | str = "./data_store/grokbot_latest_signals.json",
    ) -> None:
        self.capital_aud = capital_aud
        self.audusd_rate = audusd_rate
        self.output_path = Path(output_path)

    def generate_signal_packet(
        self,
        top_us_n: int = 5,
        top_asx_n: int = 5,
        xai_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Run full-market scan + Grok sentiment fusion and build Moomoo-compatible paper-trading order sheet."""
        scanner = MarketWidePotentialScanner(max_workers=16)
        scan_res = scanner.scan_market(period="1y", top_k_per_market=max(top_us_n, top_asx_n) + 3)

        candidates = (
            scan_res["us_potential_leaders"][:top_us_n]
            + scan_res["asx_potential_leaders"][:top_asx_n]
        )
        bridge = GrokColabBridge(api_key=xai_api_key)
        enriched = bridge.enrich_candidates_with_grok(candidates)
        rows = enriched["enriched_stocks"]

        # Compute conviction-weighted target weights across positive-signal stocks
        raw_scores = np.array([max(0.15, float(r["grok_fused_score"]) + 0.5) for r in rows], dtype=float)
        # Penalize stocks with negative breaking news sentiment
        for idx, r in enumerate(rows):
            if float(r.get("grok_sentiment_score", 0.0)) < -0.05:
                raw_scores[idx] *= 0.55
        weights = raw_scores / max(np.sum(raw_scores), 1e-8)

        order_plans: list[dict[str, Any]] = []
        moomoo_watchlist: list[str] = []

        for r, wt in zip(rows, weights, strict=False):
            sym = str(r["symbol"])
            mm_code = to_moomoo_code(sym)
            moomoo_watchlist.append(mm_code)

            close_p = float(r["close"])
            vol_ann = float(r.get("vol_20d", 0.28))
            daily_atr_est = close_p * (vol_ann / np.sqrt(252.0)) * 1.25

            sentiment = float(r.get("grok_sentiment_score", 0.0))
            exp_6m = float(r.get("grok_adjusted_6m_return", 0.12))

            # Entry, Stop-Loss (2.5x daily ATR), and 6-Month Take-Profit Target
            entry_low = round(close_p * 0.985, 2)
            entry_high = round(close_p * 1.005, 2)
            stop_loss = round(max(0.01, close_p - 2.5 * daily_atr_est), 2)
            target_6m = round(close_p * (1.0 + max(0.06, exp_6m)), 2)
            risk_reward = round((target_6m - close_p) / max(close_p - stop_loss, 0.01), 2)

            alloc_aud = round(self.capital_aud * float(wt), 2)
            price_in_aud = close_p if sym.endswith(".AX") else (close_p / self.audusd_rate)
            est_shares = int(max(1, alloc_aud // max(price_in_aud, 0.01)))

            if sentiment < -0.05:
                signal_action = "WAIT_FOR_DIP (短期舆情扰动·等回调分批)"
            elif float(r.get("mom_6m", 0.0)) >= 0.05 and float(r.get("grok_fused_score", 0.0)) >= 0.6:
                signal_action = "STRONG_BUY_TREND (量价舆情共振·右侧建仓)"
            else:
                signal_action = "BUY_GARP_VALUE (低估高成长·逢低吸纳)"

            q_score_100 = round(float(np.clip(55.0 + float(r.get("final_potential_score", 0.5)) * 22.0, 20.0, 98.0)), 1)
            g_score_100 = round(float(np.clip(50.0 + sentiment * 45.0, 15.0, 96.0)), 1)
            conv_100 = round(0.70 * q_score_100 + 0.30 * g_score_100, 1)
            sent_label = "BULLISH" if sentiment >= 0.08 else ("CAUTIOUS" if sentiment <= -0.05 else "NEUTRAL+")

            order_plans.append(
                {
                    "fused_rank": int(r["fused_rank"]),
                    "symbol": sym,
                    "moomoo_code": mm_code,
                    "name": r.get("name", sym),
                    "market": r.get("market", "US"),
                    "sector": r.get("sector", "Technology"),
                    "currency": "AUD" if sym.endswith(".AX") else "USD",
                    "signal_action": signal_action,
                    "action": signal_action,
                    "quant_potential_score": q_score_100,
                    "grok_sentiment_score": g_score_100,
                    "sentiment_label": sent_label,
                    "combined_conviction": conv_100,
                    "target_weight_pct": round(float(wt) * 100.0, 2),
                    "target_weight": round(float(wt), 4),
                    "target_alloc_aud": alloc_aud,
                    "allocated_aud": alloc_aud,
                    "suggested_paper_shares": est_shares,
                    "suggested_shares": est_shares,
                    "reference_close": close_p,
                    "current_price": close_p,
                    "entry_zone": f"{entry_low} ~ {entry_high}",
                    "entry_zone_low": entry_low,
                    "entry_zone_high": entry_high,
                    "stop_loss_price": stop_loss,
                    "target_6m_price": target_6m,
                    "target_price_6m": target_6m,
                    "expected_6m_return_pct": round(exp_6m * 100.0, 2),
                    "expected_upside_6m_pct": round(exp_6m * 100.0, 1),
                    "risk_reward_ratio": risk_reward,
                    "forward_pe": r.get("forward_pe"),
                    "revenue_growth_pct": round(float(r.get("revenue_growth", 0.0)) * 100.0, 1),
                    "catalyst_summary": r.get("grok_catalyst_summary", ""),
                    "latest_headline": r.get("latest_headline", ""),
                    "top_headline": r.get("latest_headline", ""),
                    "risk_note": r.get("grok_risk_warning", ""),
                }
            )

        packet = {
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "generated_at_aest": datetime.now().strftime("%Y-%m-%d %H:%M:%S AEST"),
            "capital_base_aud": self.capital_aud,
            "capital_aud": self.capital_aud,
            "audusd_rate": self.audusd_rate,
            "total_market_scanned": scan_res["total_scanned"],
            "universe_scanned": scan_res["total_scanned"],
            "moomoo_mcp_watchlist": moomoo_watchlist,
            "moomoo_safety_policy": "READ_ONLY_QUOTES_AND_PORTFOLIO_CHECK_plus_PAPER_TRADING_PLAN",
            "grokbot_instructions": (
                "Grok Bot 执行步骤：1) 使用 moomoo MCP 查询 moomoo_mcp_watchlist 中标的的实时盘口 (bid/ask/last_price) 与当前账户持仓；"
                "2) 对比 reference_close 与 entry_zone，检查实时价格是否触发 stop_loss_price 或进入建仓区间；"
                "3) 输出今日盯盘信号与模拟盘调仓清单（遵守只读安全约束，不自动触发实盘下单）。"
            ),
            "signals": order_plans,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(packet, indent=2, ensure_ascii=False), encoding="utf-8")
        return packet
