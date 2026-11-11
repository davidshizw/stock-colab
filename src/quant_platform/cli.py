"""Command-line interface for the US & ASX Quantitative Research Platform."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import uvicorn

from quant_platform.agents.coordinator import ResearchCoordinatorAgent
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.config import configure_logging, get_settings
from quant_platform.cross_market.us_asx_analyzer import CrossMarketAnalyzer
from quant_platform.data.ingestion import build_default_pipeline
from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine
from quant_platform.factors.evaluation import FactorEvaluator
from quant_platform.schemas import BacktestConfig, CurrencyCode
from quant_platform.screener.ranker import EquityScreener


def run_end_to_end_demo(mode: str = "fixture") -> dict[str, object]:
    """Execute end-to-end demonstration: Ingest -> Factors -> Screen -> Cross-Market -> Backtest -> AI Report."""
    settings = get_settings()
    configure_logging(settings.log_level)
    store = MarketDataStore(
        db_path=settings.duckdb_path,
        parquet_dir=settings.data_dir / "parquet",
    )

    end_d = date.today() if mode == "yfinance" else date(2026, 11, 6)
    start_d = end_d - timedelta(days=500)
    if mode == "yfinance":
        store.conn.cursor().execute("DELETE FROM daily_prices WHERE source = 'fixture'")
        store.conn.cursor().execute("DELETE FROM fundamentals_pit WHERE source = 'fixture'")
        store.conn.cursor().execute("DELETE FROM factor_values")

    print("=" * 88)
    print("1. [DATA INGESTION] Ingesting US (NYSE/Nasdaq) & Australian (ASX 200) Market Data...")
    pipeline = build_default_pipeline(store, mode=mode)
    ing_summary = pipeline.run_ingestion(start_date=start_d, end_date=end_d, incremental=False)
    print(
        f"   -> Upserted {ing_summary['instruments_upserted']} instruments, "
        f"{ing_summary['prices_upserted']} OHLCV bars, "
        f"{ing_summary['fundamentals_upserted']} PIT fundamentals (Quality: {ing_summary['quality_report']['status']})"
    )

    print("\n2. [FACTOR ENGINE & SCREENER] Computing 6 Factor Categories & Ranking US + ASX Equities...")
    factor_engine = FactorEngine(store)
    factor_engine.persist_factors_for_date(as_of_date=end_d, market="ALL")
    screener = EquityScreener(store, factor_engine)
    us_top = screener.screen(as_of_date=end_d, market="US", top_n=5)
    asx_top = screener.screen(as_of_date=end_d, market="ASX", top_n=5)

    print("   -> Top 5 US Equities (Sector-Neutral Multi-Factor Composite):")
    for r in us_top:
        print(
            f"      #{r['rank']} {r['symbol']:<6} | {r['sector']:<24} | Close=${r['close']:<7.2f} "
            f"| PE={r['pe_ratio']:<5.1f} | ROE={r['roe']*100:<5.1f}% | Score={r['composite_score']:+.3f}"
        )

    print("   -> Top 5 ASX Equities (with Franking Credit Gross-Up Yield):")
    for r in asx_top:
        print(
            f"      #{r['rank']} {r['symbol']:<6} | {r['sector']:<24} | Close=A${r['close']:<6.2f} "
            f"| CashDiv={r['dividend_yield']*100:<4.2f}% | GrossedDiv={r['franking_grossed_yield']*100:<4.2f}% "
            f"| Score={r['composite_score']:+.3f}"
        )

    print("\n3. [FACTOR EVALUATION] Evaluating IC, Rank IC, Quantile Spread, and Decay...")
    evaluator = FactorEvaluator(store, factor_engine)
    eval_res = evaluator.evaluate_factors(start_date=start_d, end_date=end_d, market="ALL")
    for fs in eval_res.get("factor_summary", []):
        print(
            f"   -> {fs['factor']:<22} | Mean IC={fs['mean_ic']:+.4f} | Rank IC={fs['mean_rank_ic']:+.4f} "
            f"| IC_IR={fs['ic_ir']:+.2f} | Turnover={fs['avg_rank_turnover']*100:.1f}%"
        )

    print("\n4. [CROSS-MARKET INTELLIGENCE] Analyzing US Overnight -> ASX Session Lead-Lag...")
    cm = CrossMarketAnalyzer(store).analyze_overnight_transmission(start_date=start_d, end_date=end_d)
    print(f"   -> Zero Look-Ahead Alignment Verified: {cm.get('alignment_verified')} (Violations={cm.get('lookahead_violations')})")
    for s in cm.get("sector_sensitivity", [])[:4]:
        print(
            f"   -> ASX Sector: {s['asx_sector']:<24} | Corr(Gap, US SPY)={s.get('corr_gap_SPY', 0):+.3f} "
            f"| Corr(Total, Copper)={s.get('corr_total_HG=F', 0):+.3f}"
        )

    print("\n5. [PORTFOLIO BACKTESTING] Running Multi-Factor Strategy Backtest (AUD Base, Franking & WHT Enabled)...")
    bt_engine = BacktestEngine(store, factor_engine)
    bt_cfg = BacktestConfig(
        strategy_name="US+ASX Multi-Factor Composite",
        market_filter="ALL",
        base_currency=CurrencyCode.AUD,
        start_date=start_d + timedelta(days=60),
        end_date=end_d,
        rebalance_frequency="M",
        top_n=6,
        weighting_scheme="factor_weight",
        include_franking_credits=True,
        apply_us_withholding_tax=True,
    )
    bt_res = bt_engine.run_backtest(bt_cfg)
    m = bt_res.metrics
    print(
        f"   -> Total Return={m.total_return*100:+.2f}% (Benchmark={m.benchmark_total_return*100:+.2f}%) "
        f"| CAGR={m.cagr*100:+.2f}% | Sharpe={m.sharpe_ratio:.2f} | Sortino={m.sortino_ratio:.2f}"
    )
    print(
        f"   -> Max Drawdown=-{m.max_drawdown*100:.2f}% | Calmar={m.calmar_ratio:.2f} "
        f"| Alpha={m.alpha_annualized*100:+.2f}% | Beta={m.beta:.2f} "
        f"| Tax & Franking Impact={m.total_tax_and_franking_adjustment_bps:+.1f} bps"
    )
    print(
        f"   -> Purged Split Sharpe: In-Sample={bt_res.split_summary['in_sample']['sharpe']:.2f} "
        f"| Validation={bt_res.split_summary['validation']['sharpe']:.2f} "
        f"| Out-of-Sample={bt_res.split_summary['out_of_sample']['sharpe']:.2f}"
    )

    print("\n6. [AI RESEARCH COORDINATOR] Generating Verifiable Multi-Agent Research Report...")
    agent_report = ResearchCoordinatorAgent(store).generate_research_report(
        as_of_date=end_d, market_scope="ALL", focus_symbol="BHP.AX"
    )
    print(f"   -> Report ID: {agent_report.report_id}")
    print(f"   -> Summary:   {agent_report.executive_summary}")
    print("=" * 88)

    return {
        "ingestion": ing_summary,
        "us_top": us_top,
        "asx_top": asx_top,
        "factor_evaluation": eval_res,
        "cross_market": cm,
        "backtest_metrics": m.model_dump(mode="json"),
        "agent_report_id": agent_report.report_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="US & ASX AI Quantitative Research Platform CLI")
    sub = parser.add_subparsers(dest="command")

    demo_p = sub.add_parser("demo", help="Run end-to-end ingestion, factor ranking, and backtest pipeline")
    demo_p.add_argument("--mode", default="fixture", choices=["fixture", "yfinance"])

    serve_p = sub.add_parser("serve", help="Start FastAPI server and interactive dashboard")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)

    screen_p = sub.add_parser("screen", help="Run equity screener for US, ASX, or ALL")
    screen_p.add_argument("--market", default="ALL", choices=["US", "ASX", "ALL"])
    screen_p.add_argument("--top-n", type=int, default=10)

    gb_p = sub.add_parser("grokbot", help="Generate Grok Bot + Moomoo MCP watchlist & paper-trading signal packet")
    gb_p.add_argument("--capital-aud", type=float, default=100000.0)
    gb_p.add_argument("--top-us", type=int, default=5)
    gb_p.add_argument("--top-asx", type=int, default=5)

    an_p = sub.add_parser("analyze", help="Run full-metric + cross-market + peer + sentiment deep-dive on any stock ticker")
    an_p.add_argument("symbol", help="Ticker symbol (e.g. EML, EML.AX, AU.EML, NVDA, BHP.AX)")
    an_p.add_argument("--market", default="AUTO", choices=["AUTO", "ASX", "US"])

    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("quant_platform.api.app:app", host=args.host, port=args.port, reload=False)
    elif args.command == "screen":
        settings = get_settings()
        store = MarketDataStore(db_path=settings.duckdb_path, parquet_dir=settings.data_dir / "parquet")
        screener = EquityScreener(store)
        rows = screener.screen(as_of_date=date(2026, 11, 6), market=args.market, top_n=args.top_n)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.command == "grokbot":
        from quant_platform.agents.grokbot_executor import GrokBotSignalGenerator

        gen = GrokBotSignalGenerator(capital_aud=args.capital_aud)
        packet = gen.generate_signal_packet(top_us_n=args.top_us, top_asx_n=args.top_asx)
        print(json.dumps(packet, indent=2, ensure_ascii=False))
    elif args.command == "analyze":
        from pathlib import Path

        from quant_platform.screener.deep_dive import StockDeepDiveAnalyzer

        analyzer = StockDeepDiveAnalyzer()
        rep = analyzer.analyze(args.symbol, prefer_market=args.market)
        clean_tag = args.symbol.strip().upper().replace(".", "_")
        out_path = Path(f"./data_store/deep_dive_{clean_tag}.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        mode = getattr(args, "mode", "fixture")
        run_end_to_end_demo(mode=mode)


if __name__ == "__main__":
    main()
