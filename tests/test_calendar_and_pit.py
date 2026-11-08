"""Unit tests for US & ASX trading calendars, DST UTC conversion, and anti-look-ahead alignment."""

from __future__ import annotations

from datetime import date

import pandas as pd

from quant_platform.data.calendar import (
    align_us_overnight_to_asx,
    get_market_close_utc,
    get_market_open_utc,
    is_trading_day,
)
from quant_platform.schemas import MarketCode


def test_trading_holidays_and_weekends() -> None:
    # Weekend check
    assert not is_trading_day(date(2026, 1, 3), MarketCode.US)
    assert not is_trading_day(date(2026, 1, 4), MarketCode.ASX)
    # US Independence Day (July 4) vs ASX normal day (if weekday)
    assert not is_trading_day(date(2025, 7, 4), MarketCode.US)
    assert is_trading_day(date(2025, 7, 4), MarketCode.ASX)
    # Australia Day (Jan 26, 2026 is a Monday)
    assert not is_trading_day(date(2026, 1, 26), MarketCode.ASX)
    assert is_trading_day(date(2026, 1, 26), MarketCode.US)


def test_dst_aware_utc_timestamps() -> None:
    # Same calendar date: ASX closes ~15 hours BEFORE US closes!
    d = date(2026, 3, 10)
    asx_open = get_market_open_utc(d, MarketCode.ASX)
    asx_close = get_market_close_utc(d, MarketCode.ASX)
    us_open = get_market_open_utc(d, MarketCode.US)
    us_close = get_market_close_utc(d, MarketCode.US)

    assert asx_open < asx_close < us_open < us_close


def test_align_us_overnight_to_asx_prevents_lookahead_bias() -> None:
    # Create 3 days of US closes and ASX sessions
    dates = [date(2026, 3, 9), date(2026, 3, 10), date(2026, 3, 11)]
    us_rows = [
        {
            "us_date": d,
            "market_close_utc": get_market_close_utc(d, MarketCode.US),
            "spy_ret": 0.01 * (i + 1),
        }
        for i, d in enumerate(dates)
    ]
    asx_rows = [{"trade_date": d, "bhp_open_gap": 0.005 * (i + 1)} for i, d in enumerate(dates)]

    aligned = align_us_overnight_to_asx(pd.DataFrame(us_rows), pd.DataFrame(asx_rows))
    # On ASX date 2026-03-10 (opens 2026-03-09 23:00 UTC), the latest completed US session is 2026-03-09 (closed 2026-03-09 20:00 UTC)
    row_mar10 = aligned[aligned["trade_date"] == date(2026, 3, 10)].iloc[0]
    assert row_mar10["us_date"] == date(2026, 3, 9)
    assert row_mar10["market_close_utc"] < row_mar10["asx_open_utc"]
