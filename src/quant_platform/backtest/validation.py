"""Chronological train/validation/test splits with purging, embargo, and walk-forward validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class ChronologicalSplit:
    """Represents a purged chronological Train / Validation / Out-of-Sample split."""

    train_dates: list[date]
    validation_dates: list[date]
    test_dates: list[date]
    purge_days: int
    embargo_days: int


def create_purged_chronological_split(
    trading_dates: list[date],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    purge_days: int = 5,
    embargo_days: int = 5,
) -> ChronologicalSplit:
    """Split sorted trading dates into In-Sample (Train), Validation, and Out-of-Sample (Test)
    with explicit purge and embargo gaps between splits to prevent overlapping return leakage.
    """
    dates = sorted(trading_dates)
    n = len(dates)
    gap = purge_days + embargo_days
    if n < 30:
        c1 = max(1, int(n * train_ratio))
        c2 = max(c1 + 1, int(n * (train_ratio + val_ratio)))
        return ChronologicalSplit(
            train_dates=dates[:c1],
            validation_dates=dates[c1:c2],
            test_dates=dates[c2:],
            purge_days=0,
            embargo_days=0,
        )

    train_end = int(n * train_ratio)
    val_start = min(n - 2, train_end + gap)
    val_end = min(n - 1, val_start + int(n * val_ratio))
    test_start = min(n - 1, val_end + gap)

    return ChronologicalSplit(
        train_dates=dates[:train_end],
        validation_dates=dates[val_start:val_end],
        test_dates=dates[test_start:],
        purge_days=purge_days,
        embargo_days=embargo_days,
    )


def generate_walk_forward_windows(
    trading_dates: list[date],
    train_window_days: int = 126,
    test_window_days: int = 42,
    purge_days: int = 5,
) -> list[dict[str, Any]]:
    """Generate rolling Walk-Forward train and out-of-sample test windows with purging."""
    dates = sorted(trading_dates)
    n = len(dates)
    windows: list[dict[str, Any]] = []
    start_idx = 0
    fold = 1

    while start_idx + train_window_days + purge_days + test_window_days <= n:
        tr_end = start_idx + train_window_days
        te_start = tr_end + purge_days
        te_end = te_start + test_window_days

        windows.append(
            {
                "fold": fold,
                "train_start": dates[start_idx],
                "train_end": dates[tr_end - 1],
                "test_start": dates[te_start],
                "test_end": dates[te_end - 1],
                "purge_days": purge_days,
            }
        )
        start_idx += test_window_days
        fold += 1

    return windows
