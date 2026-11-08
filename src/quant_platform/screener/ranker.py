"""Multi-factor stock screener and ranker across US and Australian equities."""

from __future__ import annotations

from datetime import date
from typing import Any

from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine


class EquityScreener:
    """Screens and ranks US and ASX equities using multi-factor composite scores and hard filters."""

    def __init__(self, store: MarketDataStore, factor_engine: FactorEngine | None = None) -> None:
        self.store = store
        self.factor_engine = factor_engine or FactorEngine(store)

    def screen(
        self,
        as_of_date: date,
        market: str = "ALL",
        sector: str | None = None,
        max_pe: float | None = None,
        min_roe: float | None = None,
        min_dividend_yield: float | None = None,
        min_franking_rate: float | None = None,
        sector_neutral: bool = True,
        weights: dict[str, float] | None = None,
        top_n: int = 50,
    ) -> list[dict[str, Any]]:
        """Execute cross-sectional factor screening and return ranked equity records."""
        df = self.factor_engine.compute_factors_as_of(
            as_of_date=as_of_date,
            market=market,
            sector_neutral=sector_neutral,
            weights=weights,
        )
        if df.empty:
            return []

        if sector and sector != "ALL":
            df = df[df["sector"] == sector]
        if max_pe is not None:
            df = df[df["pe_ratio"] <= max_pe]
        if min_roe is not None:
            df = df[df["roe"] >= min_roe]
        if min_dividend_yield is not None:
            df = df[df["dividend_yield"] >= min_dividend_yield]
        if min_franking_rate is not None:
            df = df[df["franking_rate"] >= min_franking_rate]

        if df.empty:
            return []

        df = df.sort_values("composite_score", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1

        display_cols = [
            "rank",
            "symbol",
            "name",
            "market",
            "currency",
            "sector",
            "close",
            "market_cap",
            "pe_ratio",
            "pb_ratio",
            "ev_ebitda",
            "roe",
            "roic",
            "profit_margin",
            "dividend_yield",
            "franking_rate",
            "franking_grossed_yield",
            "mom_1m",
            "mom_3m",
            "mom_6m",
            "mom_12m",
            "volatility_20d",
            "beta_60d",
            "max_drawdown_120d",
            "value_score",
            "quality_score",
            "momentum_score",
            "low_risk_score",
            "liquidity_score",
            "market_specific_score",
            "composite_score",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        out_df = df[available_cols].copy()
        for col in out_df.select_dtypes(include=["float64", "float32"]).columns:
            out_df[col] = out_df[col].round(4)
        return out_df.to_dict(orient="records")
