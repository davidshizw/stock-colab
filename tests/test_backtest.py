"""Unit tests for BacktestEngine, Purged Walk-Forward splits, Cross-Market analysis, and AI Agents."""

from __future__ import annotations

from datetime import date, timedelta

from quant_platform.agents.coordinator import ResearchCoordinatorAgent, black_scholes_analytics
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.backtest.validation import (
    create_purged_chronological_split,
    generate_walk_forward_windows,
)
from quant_platform.cross_market.us_asx_analyzer import CrossMarketAnalyzer
from quant_platform.data.store import MarketDataStore
from quant_platform.schemas import BacktestConfig, CurrencyCode


def test_purged_chronological_split_and_walk_forward() -> None:
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(250)]
    split = create_purged_chronological_split(dates, purge_days=5, embargo_days=5)
    assert max(split.train_dates) < min(split.validation_dates)
    assert max(split.validation_dates) < min(split.test_dates)
    assert (min(split.validation_dates) - max(split.train_dates)).days >= 10

    wf = generate_walk_forward_windows(dates, train_window_days=100, test_window_days=30, purge_days=5)
    assert len(wf) >= 3
    for w in wf:
        assert w["train_end"] < w["test_start"]


def test_backtest_engine_and_franking_effect(populated_store: MarketDataStore) -> None:
    bt = BacktestEngine(populated_store)
    cfg_with_frank = BacktestConfig(
        strategy_name="ASX Franking Test",
        market_filter="ASX",
        base_currency=CurrencyCode.AUD,
        start_date=date(2025, 11, 1),
        end_date=date(2026, 11, 6),
        top_n=5,
        weighting_scheme="equal_weight",
        include_franking_credits=True,
    )
    res_with = bt.run_backtest(cfg_with_frank)

    cfg_no_frank = cfg_with_frank.model_copy(update={"include_franking_credits": False})
    res_no = bt.run_backtest(cfg_no_frank)

    assert len(res_with.equity_curve) > 100
    assert res_with.metrics.total_return >= res_no.metrics.total_return
    assert "in_sample" in res_with.split_summary
    assert "out_of_sample" in res_with.split_summary


def test_cross_market_and_research_agents(populated_store: MarketDataStore) -> None:
    cm = CrossMarketAnalyzer(populated_store).analyze_overnight_transmission(
        start_date=date(2025, 11, 1), end_date=date(2026, 11, 6)
    )
    assert cm["alignment_verified"] is True
    assert cm["lookahead_violations"] == 0
    assert len(cm["sector_sensitivity"]) >= 3

    bs = black_scholes_analytics(
        spot=45.0, strike=45.0, time_to_expiry_years=30 / 365, risk_free_rate=0.04, volatility=0.25
    )
    assert bs["theoretical_price"] > 0
    assert 0.35 < bs["delta"] < 0.70

    coordinator = ResearchCoordinatorAgent(populated_store)
    rep = coordinator.generate_research_report(
        as_of_date=date(2026, 11, 6), market_scope="ALL", focus_symbol="BHP.AX"
    )
    assert len(rep.evidence_items) >= 5
    assert "Market Data Agent" in rep.agent_contributions
    assert "Options Analysis Agent" in rep.agent_contributions
