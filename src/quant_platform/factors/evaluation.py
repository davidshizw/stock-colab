"""Statistical evaluation of quantitative factors: IC, Rank IC, Quantiles, Decay, and Turnover."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine


class FactorEvaluator:
    """Evaluates predictive power, decay, quantile spread, and regime stability of factors."""

    def __init__(self, store: MarketDataStore, factor_engine: FactorEngine | None = None) -> None:
        self.store = store
        self.factor_engine = factor_engine or FactorEngine(store)

    def evaluate_factors(
        self,
        start_date: date,
        end_date: date,
        market: str = "ALL",
        factor_names: list[str] | None = None,
        forward_horizon_days: int = 21,
        n_quantiles: int = 5,
    ) -> dict[str, Any]:
        """Run multi-period cross-sectional factor evaluation."""
        target_factors = factor_names or [
            "composite_score",
            "value_score",
            "quality_score",
            "momentum_score",
            "low_risk_score",
            "market_specific_score",
        ]

        inst_df = self.store.get_instruments(market=market, asset_type="EQUITY", active_only=True)
        if inst_df.empty:
            return {}

        prices_df = self.store.get_prices(
            symbols=inst_df["symbol"].tolist(),
            start_date=start_date,
            end_date=end_date,
        )
        if prices_df.empty:
            return {}

        prices_df["trade_date_dt"] = pd.to_datetime(prices_df["trade_date"])
        close_piv = prices_df.pivot(index="trade_date_dt", columns="symbol", values="adj_close").sort_index()
        if len(close_piv) <= forward_horizon_days + 25:
            forward_horizon_days = max(5, len(close_piv) // 6)

        # Sample evaluation dates every `forward_horizon_days` to avoid overlapping return autocorrelation
        eval_indices = list(range(25, len(close_piv) - max(forward_horizon_days, 21), forward_horizon_days))
        if not eval_indices:
            eval_indices = [max(0, len(close_piv) // 2 - 5)]

        ic_records: list[dict[str, Any]] = []
        quantile_records: list[dict[str, Any]] = []
        decay_records: dict[str, dict[int, list[float]]] = {
            f: {5: [], 10: [], 21: []} for f in target_factors
        }
        prev_ranks: dict[str, pd.Series] = {}
        turnover_records: dict[str, list[float]] = {f: [] for f in target_factors}

        for idx in eval_indices:
            eval_dt = close_piv.index[idx].date()
            f_df = self.factor_engine.compute_factors_as_of(as_of_date=eval_dt, market=market)
            if f_df.empty:
                continue
            f_df = f_df.set_index("symbol")

            # Compute forward return over `forward_horizon_days`
            future_idx = min(idx + forward_horizon_days, len(close_piv) - 1)
            fwd_ret = (close_piv.iloc[future_idx] / close_piv.iloc[idx] - 1.0).dropna()

            # Determine market regime at `idx` based on trailing 20d cross-sectional average return & volatility
            trail_idx = max(0, idx - 20)
            mkt_trail_ret = float((close_piv.iloc[idx] / close_piv.iloc[trail_idx] - 1.0).mean())
            regime = "BULL" if mkt_trail_ret >= 0.0 else "BEAR"

            for f_name in target_factors:
                if f_name not in f_df.columns:
                    continue
                joined = pd.concat([f_df[f_name], fwd_ret.rename("fwd_ret")], axis=1, join="inner").dropna()
                if len(joined) < 4:
                    continue

                x = joined[f_name].to_numpy(dtype=float)
                y = joined["fwd_ret"].to_numpy(dtype=float)

                ic_val = float(pearsonr(x, y)[0]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0
                rank_ic_val = float(spearmanr(x, y)[0]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0

                ic_records.append(
                    {
                        "eval_date": eval_dt.isoformat(),
                        "factor": f_name,
                        "ic": round(ic_val, 4),
                        "rank_ic": round(rank_ic_val, 4),
                        "regime": regime,
                    }
                )

                # Factor turnover (1 - rank autocorrelation with previous evaluation date)
                curr_rank = joined[f_name].rank(pct=True)
                if f_name in prev_ranks:
                    common_r = pd.concat([prev_ranks[f_name], curr_rank], axis=1, join="inner").dropna()
                    if len(common_r) >= 4:
                        autocorr = float(spearmanr(common_r.iloc[:, 0], common_r.iloc[:, 1])[0])
                        turnover_records[f_name].append(float(np.clip(1.0 - autocorr, 0.0, 2.0) / 2.0))
                prev_ranks[f_name] = curr_rank

                # Quantile bucket returns (Q1 = lowest factor score, Q5 = highest factor score)
                if f_name == "composite_score":
                    eff_q = min(n_quantiles, len(joined))
                    joined["quantile"] = pd.qcut(
                        joined[f_name].rank(method="first"),
                        q=eff_q,
                        labels=[f"Q{i}" for i in range(1, eff_q + 1)],
                    )
                    q_means = joined.groupby("quantile", observed=False)["fwd_ret"].mean()
                    row_q: dict[str, Any] = {"eval_date": eval_dt.isoformat()}
                    for q_label, val in q_means.items():
                        row_q[str(q_label)] = round(float(val), 5)
                    quantile_records.append(row_q)

                # Decay horizons (5d, 10d, 21d)
                for h in (5, 10, 21):
                    h_idx = min(idx + h, len(close_piv) - 1)
                    if h_idx > idx:
                        h_ret = (close_piv.iloc[h_idx] / close_piv.iloc[idx] - 1.0).dropna()
                        jh = pd.concat([f_df[f_name], h_ret.rename("h_ret")], axis=1, join="inner").dropna()
                        if len(jh) >= 4 and np.std(jh[f_name]) > 1e-12 and np.std(jh["h_ret"]) > 1e-12:
                            decay_records[f_name][h].append(float(spearmanr(jh[f_name], jh["h_ret"])[0]))

        ic_df = pd.DataFrame(ic_records)
        summary_rows: list[dict[str, Any]] = []
        if not ic_df.empty:
            for f_name, grp in ic_df.groupby("factor"):
                mean_ic = float(grp["ic"].mean())
                mean_ric = float(grp["rank_ic"].mean())
                std_ric = float(grp["rank_ic"].std(ddof=1)) if len(grp) > 1 else 0.1
                ic_ir = mean_ric / max(std_ric, 1e-6)
                hit_rate = float((grp["rank_ic"] > 0).mean())
                bull_ric = float(grp[grp["regime"] == "BULL"]["rank_ic"].mean()) if (grp["regime"] == "BULL").any() else mean_ric
                bear_ric = float(grp[grp["regime"] == "BEAR"]["rank_ic"].mean()) if (grp["regime"] == "BEAR").any() else mean_ric
                avg_to = float(np.mean(turnover_records.get(str(f_name), [0.25]))) if turnover_records.get(str(f_name)) else 0.25

                summary_rows.append(
                    {
                        "factor": str(f_name),
                        "mean_ic": round(mean_ic, 4),
                        "mean_rank_ic": round(mean_ric, 4),
                        "std_rank_ic": round(std_ric, 4),
                        "ic_ir": round(ic_ir, 3),
                        "positive_ic_rate": round(hit_rate, 3),
                        "bull_regime_rank_ic": round(bull_ric, 4),
                        "bear_regime_rank_ic": round(bear_ric, 4),
                        "avg_rank_turnover": round(avg_to, 3),
                        "decay_5d_ric": round(float(np.mean(decay_records[str(f_name)][5])) if decay_records[str(f_name)][5] else 0.0, 4),
                        "decay_10d_ric": round(float(np.mean(decay_records[str(f_name)][10])) if decay_records[str(f_name)][10] else 0.0, 4),
                        "decay_21d_ric": round(float(np.mean(decay_records[str(f_name)][21])) if decay_records[str(f_name)][21] else 0.0, 4),
                    }
                )

        # Aggregate quantile average returns
        quantile_summary: dict[str, float] = {}
        if quantile_records:
            q_df = pd.DataFrame(quantile_records).drop(columns=["eval_date"], errors="ignore")
            for col in q_df.columns:
                quantile_summary[col] = round(float(q_df[col].mean()) * (252.0 / forward_horizon_days), 4)
            if "Q5" in quantile_summary and "Q1" in quantile_summary:
                quantile_summary["Long_Short_Spread_Annualized"] = round(
                    quantile_summary["Q5"] - quantile_summary["Q1"], 4
                )

        return {
            "horizon_days": forward_horizon_days,
            "market": market,
            "factor_summary": summary_rows,
            "quantile_annualized_returns": quantile_summary,
            "ic_timeseries": ic_records,
            "quantile_timeseries": quantile_records,
        }
