"""Unit tests for Australian Franking Credits, US Withholding Tax, FX conversion, and Portfolio Optimization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.portfolio.optimizer import PortfolioOptimizer
from quant_platform.portfolio.tax_fx import (
    compute_after_tax_dividend_amount,
    translate_returns_to_base_currency,
)
from quant_platform.schemas import CurrencyCode, MarketCode


def test_franking_credits_and_us_withholding_tax() -> None:
    # 100% franked A$1.00 dividend at 30% AU corporate tax rate -> A$1.42857 grossed up
    asx_gross = compute_after_tax_dividend_amount(
        cash_amount=1.0,
        market=MarketCode.ASX,
        franking_rate=1.0,
        au_corporate_tax_rate=0.30,
        include_franking_credits=True,
    )
    assert abs(asx_gross - (1.0 / 0.70)) < 1e-5

    # US $1.00 dividend with 15% W-8BEN withholding tax -> $0.85 net
    us_net = compute_after_tax_dividend_amount(
        cash_amount=1.0,
        market=MarketCode.US,
        us_withholding_tax_rate=0.15,
        apply_us_withholding_tax=True,
    )
    assert abs(us_net - 0.85) < 1e-6


def test_fx_return_translation_aud_and_usd() -> None:
    rets = pd.DataFrame({"AAPL": [0.02], "BHP.AX": [0.01]})
    currs = {"AAPL": "USD", "BHP.AX": "AUD"}
    # Suppose AUD appreciates by +1% against USD (AUDUSD=X goes up 1%)
    audusd_ret = pd.Series([0.01])

    in_aud = translate_returns_to_base_currency(rets, currs, audusd_ret, base_currency=CurrencyCode.AUD)
    # BHP.AX is already in AUD, so unchanged
    assert abs(float(in_aud["BHP.AX"].iloc[0]) - 0.01) < 1e-8
    # AAPL in AUD: (1.02 / 1.01) - 1 = +0.9901%
    assert abs(float(in_aud["AAPL"].iloc[0]) - (1.02 / 1.01 - 1.0)) < 1e-6


def test_portfolio_optimizer_all_six_schemes() -> None:
    rng = np.random.default_rng(7)
    syms = ["AAPL", "MSFT", "JPM", "BHP.AX", "CBA.AX", "CSL.AX"]
    rets = pd.DataFrame(rng.normal(0.0005, 0.015, size=(60, 6)), columns=syms)
    f_scores = pd.Series([1.2, 0.9, 0.4, 1.1, 0.5, 0.8], index=syms)
    m_caps = pd.Series([3000.0, 2800.0, 600.0, 150.0, 200.0, 140.0], index=syms)
    sectors = {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "JPM": "Financials",
        "BHP.AX": "Materials",
        "CBA.AX": "Financials",
        "CSL.AX": "Health Care",
    }

    opt = PortfolioOptimizer()
    for scheme in ["equal_weight", "market_cap", "factor_weight", "risk_parity", "min_variance", "mean_variance"]:
        w = opt.optimize_weights(
            symbols=syms,
            scheme=scheme,
            factor_scores=f_scores,
            market_caps=m_caps,
            trailing_returns=rets,
            sectors=sectors,
            max_position_weight=0.30,
            max_sector_weight=0.50,
        )
        assert abs(float(w.sum()) - 1.0) < 1e-5, f"Weights did not sum to 1 for {scheme}"
        assert (w >= -1e-6).all(), f"Negative weights in long-only {scheme}"
        assert float(w.max()) <= 0.30 + 1e-4, f"Position cap violated in {scheme}"
