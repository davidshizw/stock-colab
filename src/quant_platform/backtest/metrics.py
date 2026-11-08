"""Quantitative performance metrics for strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.schemas import PerformanceMetrics


def compute_performance_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    avg_turnover: float = 0.0,
    risk_free_rate_annual: float = 0.04,
    tax_and_franking_bps: float = 0.0,
) -> PerformanceMetrics:
    """Compute comprehensive annualized risk-adjusted performance metrics."""
    clean_p = portfolio_returns.dropna()
    clean_b = benchmark_returns.reindex(clean_p.index).fillna(0.0)

    n_days = len(clean_p)
    if n_days < 2:
        return PerformanceMetrics(
            cagr=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            calmar_ratio=0.0,
            alpha_annualized=0.0,
            beta=1.0,
            information_ratio=0.0,
            avg_turnover=avg_turnover,
            total_return=0.0,
            benchmark_total_return=0.0,
            win_rate_days=0.0,
            total_tax_and_franking_adjustment_bps=tax_and_franking_bps,
        )

    cum_p = (1.0 + clean_p).cumprod()
    cum_b = (1.0 + clean_b).cumprod()
    total_ret = float(cum_p.iloc[-1] - 1.0)
    bench_tot_ret = float(cum_b.iloc[-1] - 1.0)

    years = max(n_days / 252.0, 1.0 / 252.0)
    cagr = float((max(cum_p.iloc[-1], 1e-6)) ** (1.0 / years) - 1.0)

    ann_vol = float(clean_p.std(ddof=1) * np.sqrt(252.0))
    rf_daily = risk_free_rate_annual / 252.0
    excess_p = clean_p - rf_daily
    sharpe = float((excess_p.mean() * 252.0) / max(ann_vol, 1e-6))

    downside = clean_p[clean_p < rf_daily] - rf_daily
    downside_vol = (
        float(np.sqrt((downside**2).mean()) * np.sqrt(252.0))
        if len(downside) > 0
        else ann_vol * 0.7
    )
    sortino = float((excess_p.mean() * 252.0) / max(downside_vol, 1e-6))

    peak = cum_p.cummax()
    dd = cum_p / peak - 1.0
    max_dd = float(abs(dd.min()))
    calmar = float(cagr / max(max_dd, 1e-4))

    # Alpha, Beta, and Information Ratio relative to benchmark
    cov_mat = np.cov(clean_p.to_numpy(dtype=float), clean_b.to_numpy(dtype=float))
    beta = float(cov_mat[0, 1] / max(cov_mat[1, 1], 1e-10)) if cov_mat.shape == (2, 2) else 1.0
    bench_cagr = float((max(cum_b.iloc[-1], 1e-6)) ** (1.0 / years) - 1.0)
    alpha_ann = float(cagr - (risk_free_rate_annual + beta * (bench_cagr - risk_free_rate_annual)))

    active_ret = clean_p - clean_b
    tracking_error = float(active_ret.std(ddof=1) * np.sqrt(252.0))
    info_ratio = float((active_ret.mean() * 252.0) / max(tracking_error, 1e-6))

    win_rate = float((clean_p > 0).mean())

    return PerformanceMetrics(
        cagr=round(cagr, 4),
        annualized_volatility=round(ann_vol, 4),
        sharpe_ratio=round(sharpe, 3),
        sortino_ratio=round(sortino, 3),
        max_drawdown=round(max_dd, 4),
        calmar_ratio=round(calmar, 3),
        alpha_annualized=round(alpha_ann, 4),
        beta=round(beta, 3),
        information_ratio=round(info_ratio, 3),
        avg_turnover=round(avg_turnover, 4),
        total_return=round(total_ret, 4),
        benchmark_total_return=round(bench_tot_ret, 4),
        win_rate_days=round(win_rate, 4),
        total_tax_and_franking_adjustment_bps=round(tax_and_franking_bps, 2),
    )
