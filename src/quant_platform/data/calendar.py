"""Market trading calendars, timezone conversions, and Point-in-Time alignment for US & ASX."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from quant_platform.schemas import MarketCode

NY_TZ = ZoneInfo("America/New_York")
SYDNEY_TZ = ZoneInfo("Australia/Sydney")

# Major fixed and observed holidays for US (NYSE/Nasdaq) and Australia (ASX)
US_FIXED_HOLIDAYS_MD = {(1, 1), (6, 19), (7, 4), (12, 25)}
ASX_FIXED_HOLIDAYS_MD = {(1, 1), (1, 26), (4, 25), (12, 25), (12, 26)}


def is_trading_day(d: date, market: MarketCode | str) -> bool:
    """Return True if date `d` is a valid trading session for the given market."""
    if d.weekday() >= 5:
        return False
    m = MarketCode(market) if isinstance(market, str) else market
    md = (d.month, d.day)
    if m == MarketCode.US and md in US_FIXED_HOLIDAYS_MD:
        return False
    if m == MarketCode.ASX and md in ASX_FIXED_HOLIDAYS_MD:
        return False
    return True


def get_trading_days(start_date: date, end_date: date, market: MarketCode | str) -> list[date]:
    """Generate sorted list of trading days in [start_date, end_date] for a market."""
    days: list[date] = []
    curr = start_date
    while curr <= end_date:
        if is_trading_day(curr, market):
            days.append(curr)
        curr += timedelta(days=1)
    return days


def get_market_open_utc(trade_date: date, market: MarketCode | str) -> datetime:
    """Return exact UTC datetime when the regular market session opens on `trade_date`.

    - US (NYSE/Nasdaq): 09:30 America/New_York
    - ASX (Sydney): 10:00 Australia/Sydney
    """
    m = MarketCode(market) if isinstance(market, str) else market
    if m == MarketCode.US:
        local_dt = datetime.combine(trade_date, time(9, 30), tzinfo=NY_TZ)
    elif m == MarketCode.ASX:
        local_dt = datetime.combine(trade_date, time(10, 0), tzinfo=SYDNEY_TZ)
    else:
        local_dt = datetime.combine(trade_date, time(0, 0), tzinfo=UTC)
    return local_dt.astimezone(UTC)


def get_market_close_utc(trade_date: date, market: MarketCode | str) -> datetime:
    """Return exact UTC datetime when the regular market session closes on `trade_date`.

    - US (NYSE/Nasdaq): 16:00 America/New_York (typically 20:00 or 21:00 UTC same day)
    - ASX (Sydney): 16:10 Australia/Sydney (typically 05:10 or 06:10 UTC same day)
    """
    m = MarketCode(market) if isinstance(market, str) else market
    if m == MarketCode.US:
        local_dt = datetime.combine(trade_date, time(16, 0), tzinfo=NY_TZ)
    elif m == MarketCode.ASX:
        local_dt = datetime.combine(trade_date, time(16, 10), tzinfo=SYDNEY_TZ)
    else:
        local_dt = datetime.combine(trade_date, time(23, 59), tzinfo=UTC)
    return local_dt.astimezone(UTC)


def align_us_overnight_to_asx(
    us_df: pd.DataFrame,
    asx_df: pd.DataFrame,
    us_close_col: str = "market_close_utc",
    asx_trade_date_col: str = "trade_date",
) -> pd.DataFrame:
    """Align the most recent completed US trading session to each ASX trading session
    strictly before ASX market open (10:00 Sydney time), preventing any look-ahead bias.

    Why this matters:
    On calendar date T (e.g. 2026-03-10), ASX closes at ~05:10 UTC, while the US market
    for 2026-03-10 does not close until ~20:00 UTC (which is 07:00 Sydney time on 2026-03-11).
    Joining naively on `trade_date == trade_date` leaks 15 hours of future US returns into ASX!
    This function uses `pd.merge_asof` on UTC timestamps so ASX date T only sees US session T-1.
    """
    if us_df.empty or asx_df.empty:
        return pd.DataFrame()

    us_clean = us_df.copy()
    asx_clean = asx_df.copy()

    us_clean["us_close_utc"] = pd.to_datetime(us_clean[us_close_col], utc=True)
    asx_clean["asx_open_utc"] = asx_clean[asx_trade_date_col].apply(
        lambda d: get_market_open_utc(pd.to_datetime(d).date(), MarketCode.ASX)
    )
    asx_clean["asx_open_utc"] = pd.to_datetime(asx_clean["asx_open_utc"], utc=True)

    us_sorted = us_clean.sort_values("us_close_utc").reset_index(drop=True)
    asx_sorted = asx_clean.sort_values("asx_open_utc").reset_index(drop=True)

    merged = pd.merge_asof(
        asx_sorted,
        us_sorted,
        left_on="asx_open_utc",
        right_on="us_close_utc",
        direction="backward",
        suffixes=("_asx", "_us_overnight"),
    )
    return merged
