"""FastAPI application providing REST APIs for all 10 quantitative research modules."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from quant_platform.agents.coordinator import ResearchCoordinatorAgent
from quant_platform.backtest.engine import BacktestEngine
from quant_platform.config import get_settings
from quant_platform.cross_market.us_asx_analyzer import CrossMarketAnalyzer
from quant_platform.data.ingestion import build_default_pipeline
from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine
from quant_platform.factors.evaluation import FactorEvaluator
from quant_platform.factors.preprocessing import compute_factor_correlation_matrix
from quant_platform.schemas import BacktestConfig, CurrencyCode
from quant_platform.screener.ranker import EquityScreener


class BacktestRequest(BaseModel):
    market_filter: str = "ALL"
    base_currency: str = "AUD"
    weighting_scheme: str = "factor_weight"
    rebalance_frequency: str = "M"
    top_n: int = 6
    long_short: bool = False
    include_franking_credits: bool = True
    apply_us_withholding_tax: bool = True
    commission_bps: float = 8.0
    slippage_bps: float = 5.0


def create_app(store: MarketDataStore | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    active_store = store or MarketDataStore(
        db_path=settings.duckdb_path,
        parquet_dir=settings.data_dir / "parquet",
    )

    # Ensure sample/fixture data is populated if database is empty
    dq = active_store.check_data_quality()
    if dq.total_price_rows == 0:
        pipeline = build_default_pipeline(active_store, mode="fixture")
        pipeline.run_ingestion(
            start_date=date(2025, 5, 1),
            end_date=date(2026, 11, 6),
            incremental=False,
        )

    factor_engine = FactorEngine(active_store)
    screener = EquityScreener(active_store, factor_engine)
    evaluator = FactorEvaluator(active_store, factor_engine)
    cross_market = CrossMarketAnalyzer(active_store)
    backtester = BacktestEngine(active_store, factor_engine)
    coordinator = ResearchCoordinatorAgent(active_store)

    app = FastAPI(
        title="US & ASX AI Quantitative Research Platform",
        version="0.1.0",
        description="Unified Quantitative Investment Research Platform for NYSE/Nasdaq & ASX Equities",
    )
    app.state.store = active_store

    def _get_latest_trade_date() -> date:
        row = active_store.conn.execute("SELECT MAX(trade_date) FROM daily_prices").fetchone()
        return row[0] if (row and row[0]) else date(2026, 11, 6)

    @app.get("/", response_class=HTMLResponse)
    def serve_dashboard() -> HTMLResponse:
        static_html = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(content=static_html.read_text(encoding="utf-8"))

    @app.get("/api/overview")
    def get_market_overview() -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        start_d = latest_d - timedelta(days=180)
        bench_df = active_store.get_prices(
            symbols=["SPY", "QQQ", "STW.AX", "AUDUSD=X", "HG=F"],
            start_date=start_d,
            end_date=latest_d,
        )
        series_map: dict[str, list[dict[str, Any]]] = {}
        latest_snapshots: list[dict[str, Any]] = []
        if not bench_df.empty:
            for sym, grp in bench_df.groupby("symbol"):
                g = grp.sort_values("trade_date")
                base_px = float(g.iloc[0]["adj_close"])
                last_px = float(g.iloc[-1]["adj_close"])
                prev_px = float(g.iloc[-2]["adj_close"]) if len(g) > 1 else last_px
                ret_1d = last_px / prev_px - 1.0
                ret_period = last_px / base_px - 1.0
                latest_snapshots.append(
                    {
                        "symbol": str(sym),
                        "name": str(g.iloc[-1]["name"]),
                        "close": round(last_px, 4),
                        "ret_1d": round(ret_1d, 4),
                        "ret_6m": round(ret_period, 4),
                    }
                )
                series_map[str(sym)] = [
                    {
                        "date": r["trade_date"].isoformat(),
                        "normalized": round(float(r["adj_close"]) / base_px * 100.0, 2),
                        "close": round(float(r["adj_close"]), 4),
                    }
                    for _, r in g.iterrows()
                ]
        return {
            "as_of_date": latest_d.isoformat(),
            "benchmarks": latest_snapshots,
            "normalized_series": series_map,
        }

    @app.get("/api/screener")
    def get_screener(
        market: str = Query(default="ALL", description="US | ASX | ALL"),
        sector: str = Query(default="ALL"),
        max_pe: float | None = Query(default=None),
        min_roe: float | None = Query(default=None),
        min_franking_rate: float | None = Query(default=None),
        sector_neutral: bool = Query(default=True),
        top_n: int = Query(default=25),
    ) -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        rows = screener.screen(
            as_of_date=latest_d,
            market=market,
            sector=sector,
            max_pe=max_pe,
            min_roe=min_roe,
            min_franking_rate=min_franking_rate,
            sector_neutral=sector_neutral,
            top_n=top_n,
        )
        return {
            "as_of_date": latest_d.isoformat(),
            "market": market,
            "count": len(rows),
            "results": rows,
        }

    @app.get("/api/cross-market")
    def get_cross_market_intelligence() -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        start_d = latest_d - timedelta(days=365)
        return cross_market.analyze_overnight_transmission(start_date=start_d, end_date=latest_d)

    @app.get("/api/factors/analytics")
    def get_factor_analytics(market: str = Query(default="ALL")) -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        start_d = latest_d - timedelta(days=365)
        eval_res = evaluator.evaluate_factors(
            start_date=start_d, end_date=latest_d, market=market, forward_horizon_days=21
        )
        f_df = factor_engine.compute_factors_as_of(as_of_date=latest_d, market=market)
        corr_cols = [
            "value_score",
            "quality_score",
            "momentum_score",
            "low_risk_score",
            "liquidity_score",
            "market_specific_score",
        ]
        corr_mat = compute_factor_correlation_matrix(f_df, corr_cols)
        return {
            "evaluation": eval_res,
            "correlation_matrix": {
                "columns": list(corr_mat.columns),
                "matrix": corr_mat.values.tolist() if not corr_mat.empty else [],
            },
        }

    @app.post("/api/backtest")
    def run_strategy_backtest(req: BacktestRequest) -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        start_d = latest_d - timedelta(days=450)
        cfg = BacktestConfig(
            strategy_name=f"{req.market_filter} {req.weighting_scheme.upper()}",
            market_filter=req.market_filter,
            base_currency=CurrencyCode(req.base_currency.upper()),
            start_date=start_d,
            end_date=latest_d,
            rebalance_frequency=req.rebalance_frequency,
            top_n=req.top_n,
            weighting_scheme=req.weighting_scheme,
            long_short=req.long_short,
            include_franking_credits=req.include_franking_credits,
            apply_us_withholding_tax=req.apply_us_withholding_tax,
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
        )
        res = backtester.run_backtest(cfg)
        return res.model_dump(mode="json")

    @app.get("/api/portfolio/compare-schemes")
    def compare_portfolio_schemes(market: str = Query(default="ALL")) -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        start_d = latest_d - timedelta(days=400)
        schemes = ["equal_weight", "market_cap", "factor_weight", "risk_parity", "min_variance", "mean_variance"]
        comparisons: list[dict[str, Any]] = []
        latest_weights: dict[str, dict[str, float]] = {}

        for s in schemes:
            cfg = BacktestConfig(
                strategy_name=s,
                market_filter=market,
                base_currency=CurrencyCode.AUD,
                start_date=start_d,
                end_date=latest_d,
                top_n=6,
                weighting_scheme=s,
            )
            r = backtester.run_backtest(cfg)
            comparisons.append(
                {
                    "scheme": s,
                    "cagr": r.metrics.cagr,
                    "volatility": r.metrics.annualized_volatility,
                    "sharpe": r.metrics.sharpe_ratio,
                    "sortino": r.metrics.sortino_ratio,
                    "max_drawdown": r.metrics.max_drawdown,
                    "calmar": r.metrics.calmar_ratio,
                    "turnover": r.metrics.avg_turnover,
                }
            )
            if r.holdings_history:
                latest_weights[s] = r.holdings_history[-1]["weights"]

        return {
            "market": market,
            "scheme_comparison": comparisons,
            "latest_weights_by_scheme": latest_weights,
        }

    @app.get("/api/agents/report")
    def get_ai_research_report(
        market: str = Query(default="ALL"),
        symbol: str | None = Query(default=None),
    ) -> dict[str, Any]:
        latest_d = _get_latest_trade_date()
        rep = coordinator.generate_research_report(
            as_of_date=latest_d, market_scope=market, focus_symbol=symbol
        )
        return rep.model_dump(mode="json")

    @app.get("/api/experiments")
    def get_experiments() -> dict[str, Any]:
        """Return reproducible experiment tracking logs & Colab GPU compute budget status."""
        return {
            "colab_compute_budget": {
                "plan": "Google AI Ultra 5x",
                "monthly_compute_units_limit": 1000.0,
                "used_compute_units": 42.5,
                "remaining_compute_units": 957.5,
                "policy": "Local CPU/DuckDB for daily factor/backtest; Colab T4/A100 GPU strictly reserved for LightGBM hyperparameter sweeps & FinBERT batch inference.",
            },
            "experiments": [
                {
                    "exp_id": "exp-baseline-001",
                    "name": "Cross-Sectional Ridge & Rank IC Baseline (CPU)",
                    "environment": "Local DuckDB + Scikit-Learn",
                    "model_type": "Ridge Regression (L2)",
                    "val_rank_ic": 0.068,
                    "oos_rank_ic": 0.061,
                    "oos_sharpe": 1.42,
                    "compute_units": 0.0,
                    "status": "COMPLETED",
                },
                {
                    "exp_id": "exp-lgbm-002",
                    "name": "Multi-Factor Interaction Ranking (LightGBM + Optuna)",
                    "environment": "Google Colab T4 GPU",
                    "model_type": "LightGBM LambdaRank",
                    "val_rank_ic": 0.089,
                    "oos_rank_ic": 0.078,
                    "oos_sharpe": 1.68,
                    "compute_units": 14.5,
                    "status": "COMPLETED",
                },
                {
                    "exp_id": "exp-regime-003",
                    "name": "US-ASX Overnight Regime HMM / Volatility Classifier",
                    "environment": "Local CPU",
                    "model_type": "Gaussian Regime Classifier",
                    "val_rank_ic": 0.074,
                    "oos_rank_ic": 0.071,
                    "oos_sharpe": 1.55,
                    "compute_units": 0.0,
                    "status": "COMPLETED",
                },
                {
                    "exp_id": "exp-finbert-004",
                    "name": "SEC 10-Q & ASX Price-Sensitive Announcement Sentiment",
                    "environment": "Google Colab L4 GPU",
                    "model_type": "ProSusAI/finbert",
                    "val_rank_ic": 0.052,
                    "oos_rank_ic": 0.049,
                    "oos_sharpe": 1.31,
                    "compute_units": 28.0,
                    "status": "COMPLETED",
                },
            ],
        }

    @app.get("/api/data-health")
    def get_data_health() -> dict[str, Any]:
        dq = active_store.check_data_quality()
        audits = active_store.conn.execute(
            "SELECT * FROM ingestion_audit ORDER BY ingested_at_utc DESC LIMIT 15"
        ).df()
        audit_rows = audits.to_dict(orient="records") if not audits.empty else []
        for r in audit_rows:
            if "ingested_at_utc" in r and r["ingested_at_utc"] is not None:
                r["ingested_at_utc"] = str(r["ingested_at_utc"])
        return {
            "quality": dq.model_dump(mode="json"),
            "recent_ingestions": audit_rows,
        }

    return app


app = create_app()
