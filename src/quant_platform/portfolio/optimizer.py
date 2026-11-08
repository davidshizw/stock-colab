"""Portfolio construction and convex risk/return optimisation with position and sector constraints."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def estimate_shrunk_covariance(returns_df: pd.DataFrame, shrinkage: float = 0.25) -> np.ndarray:
    """Compute annualised covariance matrix with linear shrinkage toward diagonal variance."""
    clean = returns_df.fillna(0.0)
    n_assets = clean.shape[1]
    if len(clean) < 5 or n_assets == 0:
        return np.eye(max(1, n_assets)) * 0.04
    sample_cov = np.cov(clean.to_numpy(dtype=float), rowvar=False) * 252.0
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]])
    diag_target = np.diag(np.diag(sample_cov))
    shrunk = (1.0 - shrinkage) * sample_cov + shrinkage * diag_target
    # Ensure positive definiteness
    shrunk += np.eye(n_assets) * 1e-6
    return shrunk


def enforce_position_and_sector_caps(
    weights: pd.Series,
    sectors: dict[str, str] | None = None,
    max_position_weight: float = 0.25,
    max_sector_weight: float = 0.45,
) -> pd.Series:
    """Iteratively project long-only weights onto position and sector cap constraints and re-normalise."""
    if weights.empty:
        return weights
    w = weights.clip(lower=0.0).copy()
    if w.sum() <= 0:
        w[:] = 1.0 / len(w)
    else:
        w /= w.sum()

    eff_pos_cap = max(max_position_weight, 1.0 / len(w))
    for _ in range(10):
        # 1. Position cap
        over_pos = w > eff_pos_cap
        if over_pos.any():
            excess = (w[over_pos] - eff_pos_cap).sum()
            w[over_pos] = eff_pos_cap
            under_pos = ~over_pos
            if under_pos.any() and w[under_pos].sum() > 0:
                w[under_pos] += excess * (w[under_pos] / w[under_pos].sum())
            elif under_pos.any():
                w[under_pos] += excess / under_pos.sum()

        # 2. Sector cap
        if sectors:
            sec_series = pd.Series({sym: sectors.get(sym, "Unknown") for sym in w.index})
            sec_totals = w.groupby(sec_series).sum()
            n_sectors = len(sec_totals)
            eff_sec_cap = max(max_sector_weight, 1.0 / max(1, n_sectors))
            over_secs = sec_totals[sec_totals > eff_sec_cap]
            if not over_secs.empty:
                for sec_name, sec_tot in over_secs.items():
                    mask = sec_series == sec_name
                    w[mask] *= eff_sec_cap / sec_tot
                rem = 1.0 - w.sum()
                under_mask = ~sec_series.isin(over_secs.index)
                if rem > 1e-8 and under_mask.any():
                    if w[under_mask].sum() > 0:
                        w[under_mask] += rem * (w[under_mask] / w[under_mask].sum())
                    else:
                        w[under_mask] += rem / under_mask.sum()

    total = float(w.sum())
    if total > 0:
        w /= total
    return w


class PortfolioOptimizer:
    """Constructs optimal portfolio weights across 6 weighting schemes."""

    def optimize_weights(
        self,
        symbols: list[str],
        scheme: str = "factor_weight",
        factor_scores: pd.Series | None = None,
        market_caps: pd.Series | None = None,
        trailing_returns: pd.DataFrame | None = None,
        sectors: dict[str, str] | None = None,
        max_position_weight: float = 0.25,
        max_sector_weight: float = 0.45,
        risk_aversion: float = 3.0,
    ) -> pd.Series:
        """Compute normalized portfolio weights for `symbols`."""
        n = len(symbols)
        if n == 0:
            return pd.Series(dtype=float)
        if n == 1:
            return pd.Series([1.0], index=symbols)

        scheme_clean = scheme.lower().strip()
        raw_w = pd.Series(1.0 / n, index=symbols)

        if scheme_clean == "equal_weight":
            raw_w = pd.Series(1.0 / n, index=symbols)

        elif scheme_clean == "market_cap" and market_caps is not None:
            mc = market_caps.reindex(symbols).fillna(1.0).clip(lower=1.0)
            raw_w = mc / mc.sum()

        elif scheme_clean == "factor_weight" and factor_scores is not None:
            fs = factor_scores.reindex(symbols).fillna(0.0)
            # Softmax / shifted positive score weighting
            shifted = fs - fs.min() + 0.25
            raw_w = shifted / shifted.sum()

        elif scheme_clean == "risk_parity" and trailing_returns is not None:
            cov = estimate_shrunk_covariance(trailing_returns[symbols])
            vols = np.sqrt(np.diag(cov))
            inv_vol = 1.0 / np.clip(vols, 1e-4, None)
            w_init = inv_vol / inv_vol.sum()

            # Equal Risk Contribution (ERC) optimization
            def _erc_objective(w: np.ndarray) -> float:
                port_var = float(w @ cov @ w)
                if port_var <= 1e-12:
                    return 0.0
                mrc = cov @ w
                rc = w * mrc / port_var
                target_rc = 1.0 / n
                return float(np.sum((rc - target_rc) ** 2))

            bounds = [(0.01 / n, max(max_position_weight, 1.0 / n)) for _ in range(n)]
            res = minimize(
                _erc_objective,
                w_init,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            )
            raw_w = pd.Series(res.x if res.success else w_init, index=symbols)

        elif scheme_clean in ("min_variance", "mean_variance") and trailing_returns is not None:
            cov = estimate_shrunk_covariance(trailing_returns[symbols])
            w_init = np.ones(n) / n
            if factor_scores is not None and scheme_clean == "mean_variance":
                fs = factor_scores.reindex(symbols).fillna(0.0).to_numpy(dtype=float)
                # Expected return tilt scaled from standardized factor scores
                mu = 0.08 + 0.04 * (fs / max(np.std(fs), 1e-4))
            elif trailing_returns is not None and len(trailing_returns) >= 10:
                mu = trailing_returns[symbols].mean().to_numpy(dtype=float) * 252.0
            else:
                mu = np.zeros(n)

            def _obj(w: np.ndarray) -> float:
                var = float(w @ cov @ w)
                if scheme_clean == "min_variance":
                    return var
                return float(0.5 * risk_aversion * var - float(w @ mu))

            bounds = [(0.0, max(max_position_weight, 1.0 / n)) for _ in range(n)]
            res = minimize(
                _obj,
                w_init,
                method="SLSQP",
                bounds=bounds,
                constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
            )
            raw_w = pd.Series(res.x if res.success else w_init, index=symbols)

        return enforce_position_and_sector_caps(
            raw_w,
            sectors=sectors,
            max_position_weight=max_position_weight,
            max_sector_weight=max_sector_weight,
        )
