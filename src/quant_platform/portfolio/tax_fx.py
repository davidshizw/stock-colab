"""Currency translation (AUD/USD), Australian Franking Credits, and US Withholding Tax models."""

from __future__ import annotations

import pandas as pd

from quant_platform.schemas import CurrencyCode, MarketCode


def compute_after_tax_dividend_amount(
    cash_amount: float,
    market: MarketCode | str,
    franking_rate: float = 0.0,
    au_corporate_tax_rate: float = 0.30,
    us_withholding_tax_rate: float = 0.15,
    include_franking_credits: bool = True,
    apply_us_withholding_tax: bool = True,
) -> float:
    """Compute effective dividend per share including Australian franking credits or US withholding tax.

    For ASX equities (when `include_franking_credits=True`):
        Grossed-up dividend = Cash * (1 + franking_rate * (tau / (1 - tau)))
        At 30% corporate tax rate and 100% franked (franking_rate=1.0), $1.00 cash dividend
        carries $0.4286 in franking credit (total grossed-up value $1.4286).

    For US equities (when `apply_us_withholding_tax=True`):
        Net dividend = Cash * (1 - us_withholding_tax_rate)
        Under the AU-US tax treaty (W-8BEN), default withholding tax is 15%.
    """
    if cash_amount <= 0:
        return 0.0
    m = MarketCode(market) if isinstance(market, str) else market
    if m == MarketCode.ASX:
        if not include_franking_credits:
            return cash_amount
        credit_mult = franking_rate * (au_corporate_tax_rate / (1.0 - au_corporate_tax_rate))
        return cash_amount * (1.0 + credit_mult)
    if m == MarketCode.US and apply_us_withholding_tax:
        return cash_amount * (1.0 - us_withholding_tax_rate)
    return cash_amount


def translate_returns_to_base_currency(
    asset_returns: pd.DataFrame,
    asset_currencies: dict[str, str | CurrencyCode],
    audusd_returns: pd.Series,
    base_currency: str | CurrencyCode = CurrencyCode.AUD,
) -> pd.DataFrame:
    """Translate local-currency daily asset returns into `base_currency` (AUD or USD).

    Note on convention:
    `AUDUSD=X` is quoted as USD per 1 AUD.
    Let $S_t$ be USD per 1 AUD, and $r_{\text{fx}, t} = S_t / S_{t-1} - 1$.
    - If `base_currency == 'AUD'` and asset is in `USD`:
      $1 + R_{\text{AUD}, t} = \frac{1 + R_{\text{USD}, t}}{1 + r_{\text{fx}, t}}$
    - If `base_currency == 'USD'` and asset is in `AUD`:
      $1 + R_{\text{USD}, t} = (1 + R_{\text{AUD}, t})(1 + r_{\text{fx}, t})$
    """
    base = str(base_currency).upper()
    out = asset_returns.copy()
    fx_aligned = audusd_returns.reindex(out.index).fillna(0.0)

    for sym in out.columns:
        curr = str(asset_currencies.get(sym, base)).upper()
        if curr == base:
            continue
        if base == "AUD" and curr == "USD":
            out[sym] = (1.0 + out[sym]) / (1.0 + fx_aligned).clip(lower=0.1) - 1.0
        elif base == "USD" and curr == "AUD":
            out[sym] = (1.0 + out[sym]) * (1.0 + fx_aligned) - 1.0
    return out
