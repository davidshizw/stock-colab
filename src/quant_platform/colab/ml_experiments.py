"""Machine Learning and Colab GPU/CPU research workflows: Stock Ranking, Regime Detection, and Volatility Forecasting."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge

from quant_platform.backtest.validation import create_purged_chronological_split
from quant_platform.colab.runtime import ColabSessionManager
from quant_platform.data.store import MarketDataStore
from quant_platform.factors.engine import FactorEngine

FEATURE_COLS = [
    "value_score",
    "quality_score",
    "momentum_score",
    "low_risk_score",
    "liquidity_score",
    "market_specific_score",
    "mom_1m_z",
    "mom_3m_z",
    "roe_z",
    "volatility_20d_z",
]


class ColabMLExperimentRunner:
    """Executes reproducible statistical baseline and ML/GPU experiments with purged validation."""

    def __init__(
        self,
        store: MarketDataStore,
        colab_session: ColabSessionManager | None = None,
        seed: int = 42,
    ) -> None:
        self.store = store
        self.factor_engine = FactorEngine(store)
        self.colab = colab_session or ColabSessionManager()
        self.seed = seed

    def build_panel_dataset(
        self,
        start_date: date,
        end_date: date,
        market: str = "ALL",
        forward_days: int = 21,
        step_days: int = 10,
    ) -> pd.DataFrame:
        """Build point-in-time panel dataset of cross-sectional factor features and forward targets."""
        inst_df = self.store.get_instruments(market=market, asset_type="EQUITY", active_only=True)
        if inst_df.empty:
            return pd.DataFrame()

        prices_df = self.store.get_prices(
            symbols=inst_df["symbol"].tolist(),
            start_date=start_date,
            end_date=end_date,
        )
        if prices_df.empty:
            return pd.DataFrame()

        prices_df["trade_date_dt"] = pd.to_datetime(prices_df["trade_date"])
        close_piv = prices_df.pivot(index="trade_date_dt", columns="symbol", values="adj_close").sort_index()
        ret_piv = close_piv.pct_change()

        panel_frames: list[pd.DataFrame] = []
        indices = list(range(25, len(close_piv) - forward_days, step_days))

        for idx in indices:
            dt_val = close_piv.index[idx].date()
            f_df = self.factor_engine.compute_factors_as_of(as_of_date=dt_val, market=market)
            if f_df.empty:
                continue

            fwd_ret = (close_piv.iloc[idx + forward_days] / close_piv.iloc[idx] - 1.0).rename("target_fwd_ret")
            fwd_vol = (
                ret_piv.iloc[idx + 1 : idx + 1 + forward_days].std(ddof=1) * np.sqrt(252.0)
            ).rename("target_fwd_vol")

            merged = f_df.merge(fwd_ret, left_on="symbol", right_index=True, how="inner")
            merged = merged.merge(fwd_vol, left_on="symbol", right_index=True, how="inner")
            merged["target_rank"] = merged["target_fwd_ret"].rank(pct=True)
            panel_frames.append(merged)

        if not panel_frames:
            return pd.DataFrame()
        return pd.concat(panel_frames, ignore_index=True).dropna(subset=["target_fwd_ret"])

    def run_stock_ranking_experiments(
        self, start_date: date, end_date: date, market: str = "ALL"
    ) -> dict[str, Any]:
        """Compare statistical Ridge baseline vs Non-Linear GBDT interaction ranker on Purged splits."""
        panel = self.build_panel_dataset(start_date=start_date, end_date=end_date, market=market)
        if panel.empty:
            return {}

        avail_features = [c for c in FEATURE_COLS if c in panel.columns]
        unique_dates = sorted(panel["trade_date"].unique().tolist())
        split = create_purged_chronological_split(unique_dates, purge_days=1, embargo_days=1)

        train_df = panel[panel["trade_date"].isin(set(split.train_dates))]
        val_df = panel[panel["trade_date"].isin(set(split.validation_dates))]
        test_df = panel[panel["trade_date"].isin(set(split.test_dates))]

        if train_df.empty or test_df.empty:
            train_df = panel.iloc[: int(len(panel) * 0.6)]
            val_df = panel.iloc[int(len(panel) * 0.6) : int(len(panel) * 0.8)]
            test_df = panel.iloc[int(len(panel) * 0.8) :]

        X_tr = train_df[avail_features].fillna(0.0).to_numpy(dtype=float)
        y_tr = train_df["target_rank"].to_numpy(dtype=float)
        X_val = val_df[avail_features].fillna(0.0).to_numpy(dtype=float)
        y_val = val_df["target_fwd_ret"].to_numpy(dtype=float)
        X_te = test_df[avail_features].fillna(0.0).to_numpy(dtype=float)
        y_te = test_df["target_fwd_ret"].to_numpy(dtype=float)

        # 1. Statistical Linear Baseline (Ridge L2)
        ridge = Ridge(alpha=5.0, random_state=self.seed)
        ridge.fit(X_tr, y_tr)
        ridge_val_pred = ridge.predict(X_val)
        ridge_te_pred = ridge.predict(X_te)
        ridge_val_ric = float(spearmanr(ridge_val_pred, y_val)[0]) if len(y_val) > 2 else 0.0
        ridge_te_ric = float(spearmanr(ridge_te_pred, y_te)[0]) if len(y_te) > 2 else 0.0

        # 2. Non-Linear Multi-Factor Interaction Model (GBDT / LightGBM style)
        gbdt = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=80,
            l2_regularization=2.0,
            random_state=self.seed,
        )
        gbdt.fit(X_tr, y_tr)
        gbdt_val_pred = gbdt.predict(X_val)
        gbdt_te_pred = gbdt.predict(X_te)
        gbdt_val_ric = float(spearmanr(gbdt_val_pred, y_val)[0]) if len(y_val) > 2 else 0.0
        gbdt_te_ric = float(spearmanr(gbdt_te_pred, y_te)[0]) if len(y_te) > 2 else 0.0

        # Feature attribution via linear coefficients & permutation proxy
        coef_norm = np.abs(ridge.coef_) / max(np.sum(np.abs(ridge.coef_)), 1e-8)
        feature_importance = {
            f: round(float(w), 4) for f, w in zip(avail_features, coef_norm, strict=False)
        }

        result = {
            "panel_rows": len(panel),
            "features_used": avail_features,
            "ridge_baseline": {
                "val_rank_ic": round(ridge_val_ric, 4),
                "oos_rank_ic": round(ridge_te_ric, 4),
            },
            "gbdt_interaction_model": {
                "val_rank_ic": round(gbdt_val_ric, 4),
                "oos_rank_ic": round(gbdt_te_ric, 4),
            },
            "feature_importance": feature_importance,
            "compute_status": self.colab.get_compute_unit_status(),
        }
        self.colab.save_checkpoint("ml_stock_ranking_experiment", result)
        return result

    def run_volatility_and_regime_experiments(
        self, start_date: date, end_date: date, market: str = "ALL"
    ) -> dict[str, Any]:
        """Run HAR-RV 20D Volatility Forecasting & Market Regime Classification experiments."""
        panel = self.build_panel_dataset(start_date=start_date, end_date=end_date, market=market)
        if panel.empty:
            return {}

        vol_features = ["volatility_20d", "downside_volatility", "beta_60d", "max_drawdown_120d"]
        avail_vf = [c for c in vol_features if c in panel.columns]
        clean_v = panel.dropna(subset=avail_vf + ["target_fwd_vol"])

        cut = int(len(clean_v) * 0.7)
        tr_v = clean_v.iloc[:cut]
        te_v = clean_v.iloc[cut:]

        vol_model = Ridge(alpha=1.0, random_state=self.seed)
        vol_model.fit(tr_v[avail_vf].to_numpy(dtype=float), tr_v["target_fwd_vol"].to_numpy(dtype=float))
        pred_vol = vol_model.predict(te_v[avail_vf].to_numpy(dtype=float))
        actual_vol = te_v["target_fwd_vol"].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(pred_vol - actual_vol)))
        vol_rank_corr = float(spearmanr(pred_vol, actual_vol)[0]) if len(actual_vol) > 2 else 0.0

        # Market Regime Classifier (BULL_LOW_VOL, BULL_HIGH_VOL, BEAR_LOW_VOL, BEAR_HIGH_VOL)
        med_vol = float(clean_v["volatility_20d"].median())
        regimes = np.where(
            clean_v["mom_1m"] >= 0,
            np.where(clean_v["volatility_20d"] <= med_vol, "BULL_LOW_VOL", "BULL_HIGH_VOL"),
            np.where(clean_v["volatility_20d"] <= med_vol, "BEAR_LOW_VOL", "BEAR_HIGH_VOL"),
        )
        clf = RandomForestClassifier(n_estimators=40, max_depth=4, random_state=self.seed)
        clf.fit(tr_v[avail_vf].to_numpy(dtype=float), regimes[:cut])
        oos_acc = float(clf.score(te_v[avail_vf].to_numpy(dtype=float), regimes[cut:]))

        payload = {
            "volatility_forecast_oos_mae": round(mae, 4),
            "volatility_forecast_oos_spearman": round(vol_rank_corr, 4),
            "regime_classifier_oos_accuracy": round(oos_acc, 4),
            "regime_distribution": pd.Series(regimes).value_counts(normalize=True).round(3).to_dict(),
        }
        self.colab.save_checkpoint("ml_volatility_and_regime_experiment", payload)
        return payload
