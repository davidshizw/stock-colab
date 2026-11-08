"""Unified cross-market data models and schemas for US and ASX equities."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class MarketCode(StrEnum):
    US = "US"
    ASX = "ASX"
    GLOBAL = "GLOBAL"


class CurrencyCode(StrEnum):
    USD = "USD"
    AUD = "AUD"


class AssetType(StrEnum):
    EQUITY = "EQUITY"
    ETF = "ETF"
    INDEX = "INDEX"
    FX = "FX"
    COMMODITY = "COMMODITY"


class Instrument(BaseModel):
    """Master instrument definition across US (NYSE/Nasdaq) and Australian (ASX) markets."""

    symbol: str = Field(..., description="Canonical ticker symbol, e.g. AAPL or BHP.AX")
    name: str
    market: MarketCode
    exchange: str = Field(..., description="NYSE, NASDAQ, ASX, FX, NYMEX, etc.")
    currency: CurrencyCode
    asset_type: AssetType = AssetType.EQUITY
    sector: str = Field(default="Unknown", description="GICS sector classification")
    industry: str = Field(default="Unknown")
    benchmark_index: str = Field(default="SPY", description="Primary index e.g. S&P500, ASX200")
    is_active: bool = True


class UniverseMembership(BaseModel):
    """Historical index membership record to prevent survivorship bias."""

    universe_name: str = Field(..., description="e.g. SP500, NASDAQ100, ASX200, ASX300")
    symbol: str
    effective_from: date
    effective_to: date | None = None
    weight_hint: float | None = None


class DailyPriceBar(BaseModel):
    """Unified daily OHLCV bar with exact market close UTC timestamp for PIT alignment."""

    symbol: str
    trade_date: date
    market: MarketCode
    currency: CurrencyCode
    market_close_utc: datetime = Field(
        ..., description="Exact UTC timestamp when this session closed"
    )
    open: float = Field(..., gt=0)
    high: float = Field(..., gt=0)
    low: float = Field(..., gt=0)
    close: float = Field(..., gt=0)
    adj_close: float = Field(..., gt=0)
    volume: float = Field(default=0.0, ge=0)
    turnover_local: float = Field(default=0.0, ge=0)
    bid_ask_spread_bps: float | None = Field(default=None, ge=0)
    source: str = Field(default="fixture")

    @model_validator(mode="after")
    def validate_ohlc(self) -> DailyPriceBar:
        if self.high < max(self.open, self.close, self.low) - 1e-6:
            raise ValueError(f"Invalid high price for {self.symbol} on {self.trade_date}")
        if self.low > min(self.open, self.close, self.high) + 1e-6:
            raise ValueError(f"Invalid low price for {self.symbol} on {self.trade_date}")
        return self


class CorporateAction(BaseModel):
    """Dividend, franking credit, or stock split event."""

    symbol: str
    ex_date: date
    pay_date: date | None = None
    action_type: str = Field(..., description="DIVIDEND | SPLIT")
    currency: CurrencyCode
    cash_amount: float = Field(default=0.0, ge=0.0)
    franking_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Proportion of dividend franked (ASX specific, 0.0 to 1.0)",
    )
    split_ratio: float = Field(default=1.0, gt=0.0)


class FundamentalSnapshotPIT(BaseModel):
    """Point-in-Time fundamental snapshot with explicit availability timestamp."""

    symbol: str
    market: MarketCode
    currency: CurrencyCode
    period_end_date: date = Field(..., description="Fiscal period end date")
    available_at_utc: datetime = Field(
        ..., description="Exact UTC timestamp when the filing became public"
    )
    market_cap: float = Field(..., gt=0)
    pe_ratio: float | None = None
    pb_ratio: float | None = None
    ev_ebitda: float | None = None
    fcf_yield: float | None = None
    roe: float | None = None
    roic: float | None = None
    profit_margin: float | None = None
    earnings_stability: float | None = None
    debt_to_equity: float | None = None
    dividend_yield: float = 0.0
    franking_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="fixture")


class DataQualityReport(BaseModel):
    """Data health and quality validation report."""

    checked_at_utc: datetime
    total_instruments: int
    total_price_rows: int
    total_fundamental_rows: int
    missing_date_gaps: int
    stale_symbols: list[str]
    ohlc_violations: int
    provenance_sources: dict[str, int]
    status: str = Field(..., description="HEALTHY | DEGRADED | CRITICAL")


class BacktestConfig(BaseModel):
    """Configuration for strategy backtesting."""

    strategy_name: str = "Multi-Factor Composite"
    market_filter: str = "ALL"  # "US", "ASX", or "ALL"
    base_currency: CurrencyCode = CurrencyCode.AUD
    start_date: date
    end_date: date
    rebalance_frequency: str = "M"  # "W" (Weekly), "M" (Monthly)
    top_n: int = Field(default=8, ge=1)
    weighting_scheme: str = Field(
        default="factor_weight",
        description="equal_weight | market_cap | factor_weight | risk_parity | min_variance | mean_variance",
    )
    long_short: bool = False
    max_position_weight: float = Field(default=0.25, gt=0.0, le=1.0)
    max_sector_weight: float = Field(default=0.45, gt=0.0, le=1.0)
    commission_bps: float = Field(default=8.0, ge=0.0)
    slippage_bps: float = Field(default=5.0, ge=0.0)
    include_franking_credits: bool = True
    apply_us_withholding_tax: bool = True
    benchmark_symbol: str = "SPY"


class PerformanceMetrics(BaseModel):
    """Standardized quantitative backtest performance metrics."""

    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    alpha_annualized: float
    beta: float
    information_ratio: float
    avg_turnover: float
    total_return: float
    benchmark_total_return: float
    win_rate_days: float
    total_tax_and_franking_adjustment_bps: float


class BacktestResult(BaseModel):
    """Complete output of a strategy backtest run."""

    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: list[dict[str, Any]]
    holdings_history: list[dict[str, Any]]
    regime_breakdown: list[dict[str, Any]]
    split_summary: dict[str, Any]


class AgentEvidenceItem(BaseModel):
    """Single verifiable piece of numerical or factual evidence from an agent tool."""

    agent_role: str
    metric_name: str
    value: float | str | dict[str, Any]
    methodology: str
    as_of_utc: str
    source_table: str


class ResearchReport(BaseModel):
    """Structured, evidence-backed AI research report."""

    report_id: str
    generated_at_utc: str
    target_scope: str
    executive_summary: str
    hypothesis_tested: str
    key_findings: list[str]
    risk_warnings: list[str]
    uncertainty_assessment: str
    evidence_items: list[AgentEvidenceItem]
    agent_contributions: dict[str, str]
