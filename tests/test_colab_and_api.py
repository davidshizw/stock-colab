"""Integration tests for Google Colab runtime/ML experiments and FastAPI dashboard endpoints."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from quant_platform.api.app import create_app
from quant_platform.colab.ml_experiments import ColabMLExperimentRunner
from quant_platform.colab.runtime import ColabSessionManager
from quant_platform.data.store import MarketDataStore


def test_colab_session_and_ml_experiments(populated_store: MarketDataStore, tmp_path: Path) -> None:
    colab_mgr = ColabSessionManager(local_fallback_dir=tmp_path / "colab_store")
    cu_status = colab_mgr.get_compute_unit_status()
    assert cu_status["monthly_cu_budget"] == 1000.0
    assert cu_status["remaining_cu"] <= 1000.0

    runner = ColabMLExperimentRunner(store=populated_store, colab_session=colab_mgr, seed=42)
    rank_res = runner.run_stock_ranking_experiments(
        start_date=date(2025, 11, 1), end_date=date(2026, 11, 6), market="ALL"
    )
    assert "ridge_baseline" in rank_res
    assert "gbdt_interaction_model" in rank_res
    assert (colab_mgr.checkpoints_dir / "ml_stock_ranking_experiment.json").exists()

    vol_res = runner.run_volatility_and_regime_experiments(
        start_date=date(2025, 11, 1), end_date=date(2026, 11, 6), market="ALL"
    )
    assert "volatility_forecast_oos_mae" in vol_res
    assert "regime_classifier_oos_accuracy" in vol_res


def test_fastapi_all_dashboard_endpoints(populated_store: MarketDataStore) -> None:
    app = create_app(store=populated_store)
    client = TestClient(app)

    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "US & ASX Quantitative Investment Research Platform" in r_home.text

    r_ov = client.get("/api/overview")
    assert r_ov.status_code == 200
    assert len(r_ov.json()["benchmarks"]) >= 4

    r_us = client.get("/api/screener?market=US")
    assert r_us.status_code == 200
    assert r_us.json()["count"] >= 5

    r_asx = client.get("/api/screener?market=ASX")
    assert r_asx.status_code == 200
    assert r_asx.json()["count"] >= 5

    r_cm = client.get("/api/cross-market")
    assert r_cm.status_code == 200
    assert r_cm.json()["alignment_verified"] is True

    r_fa = client.get("/api/factors/analytics")
    assert r_fa.status_code == 200
    assert "evaluation" in r_fa.json()

    r_bt = client.post(
        "/api/backtest",
        json={
            "market_filter": "ALL",
            "base_currency": "AUD",
            "weighting_scheme": "factor_weight",
            "top_n": 5,
        },
    )
    assert r_bt.status_code == 200
    assert "metrics" in r_bt.json()

    r_pf = client.get("/api/portfolio/compare-schemes")
    assert r_pf.status_code == 200
    assert len(r_pf.json()["scheme_comparison"]) == 6

    r_ag = client.get("/api/agents/report?symbol=BHP.AX")
    assert r_ag.status_code == 200
    assert len(r_ag.json()["evidence_items"]) >= 5

    r_ex = client.get("/api/experiments")
    assert r_ex.status_code == 200
    assert r_ex.json()["colab_compute_budget"]["monthly_compute_units_limit"] == 1000.0

    r_dh = client.get("/api/data-health")
    assert r_dh.status_code == 200
    assert r_dh.json()["quality"]["ohlc_violations"] == 0
