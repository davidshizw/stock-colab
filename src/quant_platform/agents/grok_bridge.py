"""Colab <-> Grok (xAI) Intelligence Bridge: Combines Colab quantitative/ML stock screening with Grok live X (Twitter) & real-time financial news sentiment analysis."""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger("quant_platform.grok_bridge")

POSITIVE_FIN_TERMS = {
    "beat", "beats", "surge", "surges", "soar", "soars", "record", "upgrade", "upgrades", "bullish",
    "outperform", "buy", "growth", "profit", "strong", "breakthrough", "rally", "rallies", "dividend",
    "expansion", "ai", "demand", "partner", "partnership", "contract", "approval", "accelerate", "top",
    "gain", "gains", "upside", "positive", "exceed", "exceeds", "boom", "leader", "winning",
}

NEGATIVE_FIN_TERMS = {
    "miss", "misses", "plunge", "plunges", "drop", "drops", "downgrade", "downgrades", "bearish",
    "underperform", "sell", "loss", "weak", "investigation", "probe", "lawsuit", "delay", "cut", "cuts",
    "warning", "risk", "decline", "declines", "slump", "tariff", "recall", "fine", "penalty", "concern",
    "deficit", "headwind", "crash", "fall", "falls",
}


def resolve_xai_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve xAI Grok API key from explicit arg, environment variable, or Google Colab Secrets."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for env_var in ("XAI_API_KEY", "GROK_API_KEY"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    # Try Google Colab Secrets (userdata)
    try:
        from google.colab import userdata  # type: ignore

        for secret_name in ("XAI_API_KEY", "GROK_API_KEY"):
            try:
                sec = userdata.get(secret_name)
                if sec and str(sec).strip():
                    return str(sec).strip()
            except Exception:
                continue
    except Exception:
        pass
    return None


class GrokColabBridge:
    """Connects Google Colab quantitative stock candidates with xAI Grok Live Search (X/Twitter & News) and real-time Yahoo Finance news streams."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-3-mini",
        base_url: str = "https://api.x.ai/v1",
        enable_live_x_search: bool = True,
        timeout_sec: float = 35.0,
    ) -> None:
        self.api_key = resolve_xai_api_key(api_key)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.enable_live_x_search = enable_live_x_search
        self.timeout_sec = timeout_sec

    def fetch_live_news_for_symbol(self, symbol: str, max_items: int = 5) -> list[dict[str, str]]:
        """Fetch real-time news headlines & summaries for a symbol via Yahoo Finance."""
        import yfinance as yf

        articles: list[dict[str, str]] = []
        try:
            raw_news = yf.Ticker(symbol).news or []
            for item in raw_news[:max_items]:
                content = item.get("content", item) if isinstance(item, dict) else {}
                title = str(content.get("title") or item.get("title") or "").strip()
                summary = str(content.get("summary") or content.get("description") or "").strip()
                pub_date = str(content.get("pubDate") or item.get("providerPublishTime") or "")[:19]
                if title:
                    articles.append(
                        {
                            "title": title,
                            "summary": summary[:240],
                            "pub_date": pub_date,
                        }
                    )
        except Exception as exc:
            logger.debug("News fetch failed for %s: %s", symbol, exc)
        return articles

    def _score_news_lexicon(self, articles: list[dict[str, str]]) -> tuple[float, str]:
        """Deterministic financial NLP sentiment scoring on live headlines & summaries."""
        if not articles:
            return 0.0, "暂无近期突发新闻，舆情中性"
        pos_hits = 0
        neg_hits = 0
        for art in articles:
            text = f"{art.get('title', '')} {art.get('summary', '')}".lower()
            tokens = re.findall(r"[a-z]+", text)
            for t in tokens:
                if t in POSITIVE_FIN_TERMS:
                    pos_hits += 1
                elif t in NEGATIVE_FIN_TERMS:
                    neg_hits += 1
        total = pos_hits + neg_hits
        score = float(np.clip((pos_hits - neg_hits) / max(total + 2, 4), -1.0, 1.0))
        top_headline = articles[0]["title"][:75]
        return round(score, 3), top_headline

    def _call_grok_api(self, candidates_payload: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Call xAI Grok API (`https://api.x.ai/v1/chat/completions`) with optional Live X/Web Search."""
        if not self.api_key:
            return {}

        system_prompt = (
            "你是资深跨市场（美股与澳股 ASX）量化与事件驱动投研专家 Grok。"
            "你将收到来自 Google Colab 量化引擎筛选出的高潜力股票及其硬核财务指标与最新实时新闻标题。"
            "请结合实时新闻与 X (Twitter) 社区情绪，对每只股票输出严格合法的 JSON 对象，格式为：\n"
            '{"stocks": {"SYMBOL": {"grok_sentiment": 0.65, "x_buzz_level": "HIGH", "key_catalyst_zh": "一句话核心催化剂与多空逻辑", "risk_flag_zh": "主要潜在风险点"}}}\n'
            "其中 grok_sentiment 取值范围为 -1.0（极度看空/暴雷）到 +1.0（极度看多/强催化剂）。只输出 JSON，不要输出多余文本。"
        )
        user_prompt = json.dumps(candidates_payload, ensure_ascii=False)

        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        if self.enable_live_x_search:
            body["search_parameters"] = {
                "mode": "auto",
                "return_citations": True,
                "sources": [{"type": "x"}, {"type": "news"}, {"type": "web"}],
            }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self.timeout_sec) as client:
                resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=body)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                # Extract JSON block if wrapped in markdown
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                parsed = json.loads(match.group(0) if match else content)
                return parsed.get("stocks", parsed) if isinstance(parsed, dict) else {}
        except Exception as exc:
            logger.warning("Grok API call failed, falling back to live news NLP engine: %s", exc)
            return {}

    def enrich_candidates_with_grok(
        self,
        candidates: list[dict[str, Any]],
        quant_weight: float = 0.70,
        grok_weight: float = 0.30,
    ) -> dict[str, Any]:
        """Enrich Colab top stock candidates with live news & Grok X sentiment, producing fused rankings."""
        if not candidates:
            return {"mode": "empty", "stocks": []}

        symbols = [c["symbol"] for c in candidates]
        with ThreadPoolExecutor(max_workers=min(12, len(symbols))) as ex:
            news_list = list(ex.map(self.fetch_live_news_for_symbol, symbols))
        news_map = dict(zip(symbols, news_list, strict=False))

        compact_payload = []
        lexicon_map: dict[str, tuple[float, str]] = {}
        for c in candidates:
            sym = c["symbol"]
            arts = news_map.get(sym, [])
            lex_score, top_hl = self._score_news_lexicon(arts)
            lexicon_map[sym] = (lex_score, top_hl)
            compact_payload.append(
                {
                    "symbol": sym,
                    "name": c.get("name", sym),
                    "market": c.get("market", "US"),
                    "sector": c.get("sector", ""),
                    "forward_pe": c.get("forward_pe"),
                    "revenue_growth": c.get("revenue_growth"),
                    "earnings_growth": c.get("earnings_growth"),
                    "mom_6m": c.get("mom_6m"),
                    "analyst_target_upside": c.get("analyst_target_upside"),
                    "latest_headlines": [a["title"] for a in arts[:3]],
                }
            )

        grok_results = self._call_grok_api(compact_payload) if self.api_key else {}
        active_mode = (
            f"Grok Live API ({self.model} + X/Web Search)"
            if grok_results
            else "Live Yahoo Finance News NLP + Quant Catalyst Engine (Ready for XAI_API_KEY)"
        )

        enriched_rows: list[dict[str, Any]] = []
        for c in candidates:
            sym = c["symbol"]
            lex_score, top_hl = lexicon_map[sym]
            g_item = grok_results.get(sym, {}) if isinstance(grok_results, dict) else {}

            if g_item and "grok_sentiment" in g_item:
                sentiment_score = float(np.clip(float(g_item.get("grok_sentiment", 0.0)), -1.0, 1.0))
                buzz = str(g_item.get("x_buzz_level", "MEDIUM"))
                catalyst_zh = str(g_item.get("key_catalyst_zh", top_hl))
                risk_zh = str(g_item.get("risk_flag_zh", "关注财报兑现度与宏观波动"))
            else:
                # Combine live news lexicon score with fundamental/analyst momentum signal
                upside_sig = float(np.clip(float(c.get("analyst_target_upside", 0.0)) * 0.8, -0.5, 0.6))
                growth_sig = float(np.clip(float(c.get("revenue_growth", 0.0)) * 0.4, -0.3, 0.5))
                sentiment_score = round(float(np.clip(0.55 * lex_score + 0.25 * upside_sig + 0.20 * growth_sig, -1.0, 1.0)), 3)
                n_arts = len(news_map.get(sym, []))
                buzz = "HIGH 🔥" if n_arts >= 4 else ("MEDIUM" if n_arts >= 2 else "LOW")
                catalyst_zh = f"[{c.get('catalyst_tag', '核心催化')}] 最新动态: {top_hl}"
                risk_zh = (
                    "短期存在负面舆情扰动，建议分批建仓"
                    if lex_score < -0.15
                    else ("估值偏高，需紧盯下季指引" if float(c.get("forward_pe", 20)) > 40 else "基本面与舆情共振良好")
                )

            quant_score = float(c.get("final_potential_score", c.get("composite_score", 0.0)))
            fused_score = round(quant_weight * quant_score + grok_weight * (sentiment_score * 2.0), 4)

            # Adjust expected 6-month return mildly (+/- up to 2.5%) based on real-time sentiment/catalyst
            base_6m = float(c.get("expected_6m_total_return", 0.10))
            grok_adj_6m = round(base_6m + 0.025 * sentiment_score, 4)

            row = dict(c)
            row.update(
                {
                    "live_news_count": len(news_map.get(sym, [])),
                    "latest_headline": top_hl,
                    "grok_sentiment_score": sentiment_score,
                    "x_buzz_level": buzz,
                    "grok_catalyst_summary": catalyst_zh,
                    "grok_risk_warning": risk_zh,
                    "grok_fused_score": fused_score,
                    "grok_adjusted_6m_return": grok_adj_6m,
                }
            )
            enriched_rows.append(row)

        df_out = pd.DataFrame(enriched_rows).sort_values("grok_fused_score", ascending=False).reset_index(drop=True)
        df_out["fused_rank"] = df_out.index + 1

        return {
            "engine_mode": active_mode,
            "api_key_configured": bool(self.api_key),
            "quant_weight": quant_weight,
            "grok_weight": grok_weight,
            "enriched_stocks": df_out.to_dict(orient="records"),
            "enriched_df": df_out,
        }
