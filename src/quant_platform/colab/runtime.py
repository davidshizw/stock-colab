"""Google Colab environment manager: Drive mounting, GPU/CU profiling, checkpointing, and inline Colab UI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Approximate Colab Compute Unit (CU) burn rates per hour by accelerator tier
COLAB_CU_RATES_PER_HOUR: dict[str, float] = {
    "CPU": 0.08,
    "T4": 1.96,
    "L4": 4.82,
    "A100": 13.10,
    "V100": 5.36,
}


def is_colab_runtime() -> bool:
    """Return True if executing inside a Google Colab kernel."""
    return "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None


def detect_accelerator() -> dict[str, Any]:
    """Detect available GPU/CPU accelerator and estimate Colab Compute Unit (CU) burn rate."""
    gpu_name = "CPU"
    vram_gb = 0.0
    device = "cpu"

    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            line = res.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            gpu_name = parts[0]
            vram_gb = round(float(parts[1]) / 1024.0, 2) if len(parts) > 1 else 0.0
            device = "cuda"
    except Exception:
        pass

    tier = "CPU"
    upper_name = gpu_name.upper()
    for t_key in ("A100", "L4", "T4", "V100"):
        if t_key in upper_name:
            tier = t_key
            break

    return {
        "in_colab": is_colab_runtime(),
        "device": device,
        "accelerator_name": gpu_name,
        "tier": tier,
        "vram_gb": vram_gb,
        "estimated_cu_per_hour": COLAB_CU_RATES_PER_HOUR.get(tier, 0.08),
    }


class ColabSessionManager:
    """Manages Google Drive persistence, Compute Unit budget tracking, and Colab notebook helpers."""

    def __init__(
        self,
        drive_folder_name: str = "stock_colab_store",
        monthly_cu_budget: float = 1000.0,
        local_fallback_dir: Path | str = "./data_store",
    ) -> None:
        self.drive_folder_name = drive_folder_name
        self.monthly_cu_budget = monthly_cu_budget
        self.local_fallback_dir = Path(local_fallback_dir)
        self.session_start_ts = time.time()
        self.hw_info = detect_accelerator()
        self.storage_root = self._resolve_storage_root()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        (self.storage_root / "parquet").mkdir(parents=True, exist_ok=True)
        (self.storage_root / "checkpoints").mkdir(parents=True, exist_ok=True)

    def _resolve_storage_root(self) -> Path:
        if is_colab_runtime():
            drive_mount = Path("/content/drive")
            if not (drive_mount / "MyDrive").exists():
                try:
                    from google.colab import drive  # type: ignore[import-not-found]

                    drive.mount(str(drive_mount), force_remount=False)
                except Exception:
                    return self.local_fallback_dir
            return drive_mount / "MyDrive" / self.drive_folder_name
        return self.local_fallback_dir

    @property
    def duckdb_path(self) -> Path:
        return self.storage_root / "quant_research.duckdb"

    @property
    def parquet_dir(self) -> Path:
        return self.storage_root / "parquet"

    @property
    def checkpoints_dir(self) -> Path:
        return self.storage_root / "checkpoints"

    def get_compute_unit_status(self) -> dict[str, Any]:
        """Calculate elapsed session time and estimated Google Colab Compute Units consumed."""
        elapsed_hours = max(0.0, (time.time() - self.session_start_ts) / 3600.0)
        rate = float(self.hw_info["estimated_cu_per_hour"])
        session_cu = round(elapsed_hours * rate, 4)

        ledger_file = self.storage_root / "cu_usage_ledger.json"
        historical_cu = 0.0
        if ledger_file.exists():
            try:
                data = json.loads(ledger_file.read_text(encoding="utf-8"))
                historical_cu = float(data.get("cumulative_cu", 0.0))
            except Exception:
                historical_cu = 0.0

        total_used = round(historical_cu + session_cu, 3)
        return {
            "accelerator": self.hw_info["accelerator_name"],
            "tier": self.hw_info["tier"],
            "in_colab": self.hw_info["in_colab"],
            "storage_root": str(self.storage_root),
            "cu_rate_per_hour": rate,
            "session_elapsed_minutes": round(elapsed_hours * 60.0, 2),
            "session_cu_used": session_cu,
            "monthly_cu_budget": self.monthly_cu_budget,
            "total_cu_used": total_used,
            "remaining_cu": round(max(0.0, self.monthly_cu_budget - total_used), 3),
        }

    def save_checkpoint(self, experiment_id: str, payload: dict[str, Any]) -> Path:
        """Persist experiment metrics, parameters, and state to Google Drive / local checkpoint dir."""
        cp_path = self.checkpoints_dir / f"{experiment_id}.json"
        record = {
            "experiment_id": experiment_id,
            "saved_at_utc": datetime.now(UTC).isoformat(),
            "compute_status": self.get_compute_unit_status(),
            "payload": payload,
        }
        cp_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        return cp_path

    def render_inline_colab_dashboard(
        self,
        us_top: list[dict[str, Any]],
        asx_top: list[dict[str, Any]],
        backtest_result: dict[str, Any],
        cross_market: dict[str, Any],
    ) -> str:
        """Generate a self-contained HTML widget with Apache ECharts that renders directly inside a Google Colab cell."""
        eq_curve = backtest_result.get("equity_curve", [])
        metrics = backtest_result.get("metrics", {})
        dates_json = json.dumps([r["date"] for r in eq_curve])
        p_nav_json = json.dumps([r["portfolio_nav"] for r in eq_curve])
        b_nav_json = json.dumps([r["benchmark_nav"] for r in eq_curve])
        dd_json = json.dumps([r["drawdown"] for r in eq_curve])

        us_rows = "".join(
            f"<tr><td>#{r['rank']}</td><td><b>{r['symbol']}</b></td><td>{r['sector']}</td>"
            f"<td>${r['close']:.2f}</td><td>{r['pe_ratio']:.1f}x</td><td>{r['roe']*100:.1f}%</td>"
            f"<td style='color:#10b981;font-weight:bold'>{r['composite_score']:+.3f}</td></tr>"
            for r in us_top[:6]
        )
        asx_rows = "".join(
            f"<tr><td>#{r['rank']}</td><td><b>{r['symbol']}</b></td><td>{r['sector']}</td>"
            f"<td>A${r['close']:.2f}</td><td>{r['dividend_yield']*100:.2f}%</td>"
            f"<td style='color:#34d399;font-weight:bold'>{r['franking_grossed_yield']*100:.2f}%</td>"
            f"<td style='color:#10b981;font-weight:bold'>{r['composite_score']:+.3f}</td></tr>"
            for r in asx_top[:6]
        )

        html = f"""
        <div style="background:#0f172a;color:#f8fafc;padding:18px;border-radius:10px;font-family:sans-serif;">
          <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #334155;padding-bottom:10px;margin-bottom:14px;">
            <div>
              <h2 style="margin:0;font-size:18px;color:#fff;">US & ASX Quantitative Research Dashboard (Colab Inline View)</h2>
              <p style="margin:2px 0 0;font-size:12px;color:#94a3b8;">
                Storage: {self.storage_root} | Hardware: {self.hw_info['accelerator_name']} | Zero Look-Ahead Verified: {cross_market.get('alignment_verified', True)}
              </p>
            </div>
            <div style="text-align:right;font-size:12px;">
              <span style="background:#065f46;color:#a7f3d0;padding:4px 8px;border-radius:4px;">
                CAGR: {metrics.get('cagr', 0)*100:+.2f}% | Sharpe: {metrics.get('sharpe_ratio', 0):.2f} | MaxDD: -{metrics.get('max_drawdown', 0)*100:.2f}%
              </span>
            </div>
          </div>

          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px;">
            <div style="background:#1e293b;padding:12px;border-radius:8px;">
              <h4 style="margin:0 0 8px;font-size:13px;color:#60a5fa;">Top Ranked US Equities (NYSE / Nasdaq)</h4>
              <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:left;">
                <thead><tr style="border-bottom:1px solid #334155;color:#94a3b8;">
                  <th>Rank</th><th>Symbol</th><th>Sector</th><th>Price</th><th>PE</th><th>ROE</th><th>Score</th>
                </tr></thead>
                <tbody>{us_rows}</tbody>
              </table>
            </div>
            <div style="background:#1e293b;padding:12px;border-radius:8px;">
              <h4 style="margin:0 0 8px;font-size:13px;color:#fbbf24;">Top Ranked ASX Equities (Franking Credit Grossed-Up)</h4>
              <table style="width:100%;font-size:12px;border-collapse:collapse;text-align:left;">
                <thead><tr style="border-bottom:1px solid #334155;color:#94a3b8;">
                  <th>Rank</th><th>Symbol</th><th>Sector</th><th>Price</th><th>CashDiv</th><th>GrossDiv</th><th>Score</th>
                </tr></thead>
                <tbody>{asx_rows}</tbody>
              </table>
            </div>
          </div>

          <div style="background:#1e293b;padding:12px;border-radius:8px;">
            <h4 style="margin:0 0 8px;font-size:13px;color:#e2e8f0;">Multi-Factor Strategy Cumulative NAV vs Benchmark & Drawdown (AUD Base)</h4>
            <div id="colab-echarts-nav" style="width:100%;height:320px;"></div>
          </div>

          <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
          <script>
            (function() {{
              var initChart = function() {{
                if (typeof echarts === 'undefined') {{ setTimeout(initChart, 200); return; }}
                var dom = document.getElementById('colab-echarts-nav');
                if (!dom) return;
                var chart = echarts.init(dom, 'dark');
                chart.setOption({{
                  backgroundColor: 'transparent',
                  tooltip: {{ trigger: 'axis' }},
                  legend: {{ data: ['Portfolio NAV', 'Benchmark NAV', 'Drawdown'], top: 0 }},
                  grid: {{ left: 50, right: 50, top: 35, bottom: 25 }},
                  xAxis: {{ type: 'category', data: {dates_json} }},
                  yAxis: [
                    {{ type: 'value', name: 'NAV', scale: true }},
                    {{ type: 'value', name: 'Drawdown' }}
                  ],
                  series: [
                    {{ name: 'Portfolio NAV', type: 'line', showSymbol: false, data: {p_nav_json} }},
                    {{ name: 'Benchmark NAV', type: 'line', showSymbol: false, data: {b_nav_json} }},
                    {{ name: 'Drawdown', type: 'line', yAxisIndex: 1, showSymbol: false, areaStyle: {{ opacity: 0.15 }}, data: {dd_json} }}
                  ]
                }});
              }};
              initChart();
            }})();
          </script>
        </div>
        """
        return html
