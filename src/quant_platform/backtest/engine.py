"""Hybrid vectorised & rebalance-stepping backtesting engine for US & Australian equities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_platform.backtest.metrics import compute_performance_metrics
from quant_platform.backtest.validation import create_purged_chronological_split
from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine
from quant_platform.portfolio.optimizer import PortfolioOptimizer
from quant_platform.portfolio.tax_fx import (
    compute_after_tax_dividend_amount,
    translate_returns_to_base_currency,
)
from quant_platform.schemas import BacktestConfig, BacktestResult


class BacktestEngine:
    """Executes multi-factor and event-aware portfolio backtests across US and ASX markets."""

    def __init__(
        self,
        store: MarketDataStore,
        factor_engine: FactorEngine | None = None,
        optimizer: PortfolioOptimizer | None = None,
    ) -> None:
        self.store = store
        self.factor_engine = factor_engine or FactorEngine(store)
        self.optimizer = optimizer or PortfolioOptimizer()

    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """Execute a complete backtest with transaction costs, FX translation, and franking/tax adjustments."""
        inst_df = self.store.get_instruments(
            market=config.market_filter, asset_type="EQUITY", active_only=False
        )
        eq_symbols = inst_df["symbol"].tolist()
        bench_sym = "STW.AX" if config.market_filter == "ASX" else config.benchmark_symbol
        all_syms = list(set(eq_symbols + [bench_sym, "AUDUSD=X", "SPY", "STW.AX"]))

        prices_df = self.store.get_prices(
            symbols=all_syms,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        if prices_df.empty:
            raise ValueError("No price data found for the requested backtest window.")

        prices_df["trade_date_dt"] = pd.to_datetime(prices_df["trade_date"])
        close_piv = (
            prices_df.pivot(index="trade_date_dt", columns="symbol", values="close")
            .sort_index()
            .ffill()
        )
        ret_piv = close_piv.pct_change().fillna(0.0)

        # Map currencies and markets per instrument
        all_inst_df = self.store.get_instruments(asset_type="ALL")
        curr_map = dict(zip(all_inst_df["symbol"], all_inst_df["currency"], strict=False))
        mkt_map = dict(zip(all_inst_df["symbol"], all_inst_df["market"], strict=False))
        sec_map = dict(zip(all_inst_df["symbol"], all_inst_df["sector"], strict=False))

        audusd_ret = (
            ret_piv["AUDUSD=X"]
            if "AUDUSD=X" in ret_piv.columns
            else pd.Series(0.0, index=ret_piv.index)
        )

        # Translate all price returns into base_currency (AUD or USD)
        base_ret_piv = translate_returns_to_base_currency(
            asset_returns=ret_piv,
            asset_currencies=curr_map,
            audusd_returns=audusd_ret,
            base_currency=config.base_currency,
        )

        # Incorporate dividend yields + Franking Credit / US Withholding Tax adjustments on ex-dates
        actions_df = self.store.get_corporate_actions(
            start_date=config.start_date, end_date=config.end_date
        )
        tax_franking_bonus_piv = pd.DataFrame(0.0, index=base_ret_piv.index, columns=base_ret_piv.columns)

        if not actions_df.empty:
            for _, act in actions_df.iterrows():
                if act["action_type"] != "DIVIDEND":
                    continue
                sym = str(act["symbol"])
                ex_dt = pd.Timestamp(act["ex_date"])
                if sym not in close_piv.columns or ex_dt not in close_piv.index:
                    continue
                px = float(close_piv.loc[ex_dt, sym])
                if px <= 0 or np.isnan(px):
                    continue
                cash_amt = float(act["cash_amount"])
                eff_amt = compute_after_tax_dividend_amount(
                    cash_amount=cash_amt,
                    market=mkt_map.get(sym, "US"),
                    franking_rate=float(act.get("franking_rate", 0.0)),
                    include_franking_credits=config.include_franking_credits,
                    apply_us_withholding_tax=config.apply_us_withholding_tax,
                )
                div_ret = eff_amt / px
                base_ret_piv.loc[ex_dt, sym] += div_ret
                tax_franking_bonus_piv.loc[ex_dt, sym] += (eff_amt - cash_amt) / px

        # Determine rebalance dates
        all_dt_index = base_ret_piv.index
        step = 5 if config.rebalance_frequency.upper().startswith("W") else 21
        start_offset = min(25, max(1, len(all_dt_index) // 5))
        rebalance_indices = set(range(start_offset, len(all_dt_index), step))
        if not rebalance_indices:
            rebalance_indices = {0}

        avail_eq = [s for s in eq_symbols if s in base_ret_piv.columns]
        current_weights = pd.Series(0.0, index=avail_eq)
        port_daily_rets = pd.Series(0.0, index=all_dt_index)
        tax_daily_effect = pd.Series(0.0, index=all_dt_index)
        turnovers: list[float] = []
        holdings_history: list[dict[str, Any]] = []

        cost_rate = (config.commission_bps + config.slippage_bps) / 10000.0

        for idx, ts in enumerate(all_dt_index):
            d = ts.date()
            if idx in rebalance_indices:
                # Filter by historical active universe as of date d (prevents survivorship bias)
                active_univ = set(self.store.get_active_universe_as_of(d))
                f_df = self.factor_engine.compute_factors_as_of(
                    as_of_date=d, market=config.market_filter
                )
                if not f_df.empty:
                    if active_univ:
                        f_df = f_df[f_df["symbol"].isin(active_univ)]
                    f_df = f_df[f_df["symbol"].isin(avail_eq)].sort_values(
                        "composite_score", ascending=False
                    )

                if not f_df.empty:
                    top_df = f_df.head(config.top_n)
                    sel_syms = top_df["symbol"].tolist()
                    f_scores = top_df.set_index("symbol")["composite_score"]
                    m_caps = top_df.set_index("symbol")["market_cap"]
                    trail_rets = base_ret_piv.iloc[max(0, idx - 60) : max(1, idx)][sel_syms]

                    long_w = self.optimizer.optimize_weights(
                        symbols=sel_syms,
                        scheme=config.weighting_scheme,
                        factor_scores=f_scores,
                        market_caps=m_caps,
                        trailing_returns=trail_rets,
                        sectors=sec_map,
                        max_position_weight=config.max_position_weight,
                        max_sector_weight=config.max_sector_weight,
                    )
                    new_weights = pd.Series(0.0, index=avail_eq)
                    new_weights.loc[long_w.index] = long_w.values

                    if config.long_short and len(f_df) >= config.top_n * 2:
                        bot_syms = f_df.tail(config.top_n)["symbol"].tolist()
                        short_w = pd.Series(-0.5 / len(bot_syms), index=bot_syms)
                        new_weights.loc[long_w.index] = long_w.values * 0.5
                        new_weights.loc[short_w.index] = short_w.values

                    one_way_turnover = float(np.abs(new_weights - current_weights).sum() / 2.0)
                    turnovers.append(one_way_turnover)
                    rebal_cost = one_way_turnover * 2.0 * cost_rate
                    current_weights = new_weights

                    holdings_history.append(
                        {
                            "rebalance_date": d.isoformat(),
                            "turnover": round(one_way_turnover, 4),
                            "cost_bps": round(rebal_cost * 10000, 2),
                            "weights": {
                                k: round(float(v), 4)
                                for k, v in current_weights[current_weights.abs() > 1e-4].items()
                            },
                        }
                    )
                else:
                    rebal_cost = 0.0
            else:
                rebal_cost = 0.0

            day_ret = float((base_ret_piv.loc[ts, avail_eq] * current_weights).sum()) - rebal_cost
            port_daily_rets.iloc[idx] = day_ret
            tax_daily_effect.iloc[idx] = float(
                (tax_franking_bonus_piv.loc[ts, avail_eq] * current_weights).sum()
            )

        # Trim initial lookback warm-up days before first rebalance
        active_slice = slice(start_offset, len(all_dt_index))
        p_rets = port_daily_rets.iloc[active_slice]
        b_rets = (
            base_ret_piv[bench_sym].iloc[active_slice]
            if bench_sym in base_ret_piv.columns
            else base_ret_piv[avail_eq].mean(axis=1).iloc[active_slice]
        )
        avg_to = float(np.mean(turnovers)) if turnovers else 0.0
        total_tax_bps = float(tax_daily_effect.iloc[active_slice].sum() * 10000.0)

        metrics = compute_performance_metrics(
            portfolio_returns=p_rets,
            benchmark_returns=b_rets,
            avg_turnover=avg_to,
            tax_and_franking_bps=total_tax_bps,
        )

        # Build equity & drawdown curves
        cum_p = (1.0 + p_rets).cumprod()
        cum_b = (1.0 + b_rets).cumprod()
        dd_p = cum_p / cum_p.cummax() - 1.0

        # Chronological Purged Split (In-Sample / Validation / Out-of-Sample)
        active_dates = [d.date() for d in p_rets.index]
        split = create_purged_chronological_split(active_dates)
        train_set = set(split.train_dates)
        val_set = set(split.validation_dates)
        test_set = set(split.test_dates)

        equity_curve: list[dict[str, Any]] = []
        for ts, p_val in cum_p.items():
            dt_val = ts.date()
            phase = (
                "IN_SAMPLE"
                if dt_val in train_set
                else ("VALIDATION" if dt_val in val_set else ("OUT_OF_SAMPLE" if dt_val in test_set else "PURGE_EMBARGO"))
            )
            equity_curve.append(
                {
                    "date": dt_val.isoformat(),
                    "portfolio_nav": round(float(p_val), 4),
                    "benchmark_nav": round(float(cum_b.loc[ts]), 4),
                    "drawdown": round(float(dd_p.loc[ts]), 4),
                    "daily_return": round(float(p_rets.loc[ts]), 5),
                    "phase": phase,
                }
            )

        # Compute metrics per chronological split phase
        def _phase_summary(dates_subset: list[Any]) -> dict[str, float]:
            if not dates_subset:
                return {"cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
            mask = [ts.date() in set(dates_subset) for ts in p_rets.index]
            sub_p = p_rets[mask]
            sub_b = b_rets[mask]
            m = compute_performance_metrics(sub_p, sub_b)
            return {
                "days": len(sub_p),
                "cagr": m.cagr,
                "sharpe": m.sharpe_ratio,
                "max_drawdown": m.max_drawdown,
            }

        split_summary = {
            "in_sample": _phase_summary(split.train_dates),
            "validation": _phase_summary(split.validation_dates),
            "out_of_sample": _phase_summary(split.test_dates),
            "purge_days": split.purge_days,
            "embargo_days": split.embargo_days,
        }

        # Regime breakdown (Low-Vol vs High-Vol benchmark periods)
        roll_vol = b_rets.rolling(20, min_periods=5).std() * np.sqrt(252)
        med_vol = float(roll_vol.median()) if not roll_vol.empty else 0.15
        high_vol_mask = roll_vol > med_vol
        regime_breakdown = [
            {
                "regime": "LOW_VOLATILITY",
                "days": int((~high_vol_mask).sum()),
                "annualized_return": round(float(p_rets[~high_vol_mask].mean() * 252), 4),
                "sharpe": compute_performance_metrics(p_rets[~high_vol_mask], b_rets[~high_vol_mask]).sharpe_ratio,
            },
            {
                "regime": "HIGH_VOLATILITY",
                "days": int(high_vol_mask.sum()),
                "annualized_return": round(float(p_rets[high_vol_mask].mean() * 252), 4),
                "sharpe": compute_performance_metrics(p_rets[high_vol_mask], b_rets[high_vol_mask]).sharpe_ratio,
            },
        ]

        return BacktestResult(
            config=config,
            metrics=metrics,
            equity_curve=equity_curve,
            holdings_history=holdings_history,
            regime_breakdown=regime_breakdown,
            split_summary=split_summary,
        )
