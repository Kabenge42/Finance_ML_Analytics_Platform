"""
Utilities for feature engineering tests (Phase 9.3 - Week 1 Infra)

- assert_no_inf: ensure no +/- inf present
- assert_nan_ratio_below: ensure NaN ratio below threshold for specified columns
- assert_within_range: feature values within [min_value, max_value] (ignoring NaNs)
- time_block: context manager to assert execution within time budget
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Iterable, Optional

import numpy as np
import pandas as pd


def assert_no_inf(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> None:
    """Assert that no +/- infinity values exist in DataFrame for given columns.

    Converts non-numeric columns to NaN (ignored). Raises AssertionError if any
    checked column contains +/- inf.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        # Coerce to numeric; non-numeric values become NaN and are ignored for inf check
        numeric = pd.to_numeric(series, errors="coerce")
        if np.isinf(numeric.to_numpy(copy=False)).any():
            raise AssertionError(f"Column '{c}' contains +/- inf values")


def assert_nan_ratio_below(df: pd.DataFrame, *, columns: Iterable[str], max_ratio: float) -> None:
    """Assert the NaN ratio for each column is <= max_ratio.

    Args:
        df: DataFrame
        columns: columns to check
        max_ratio: maximum allowed NaN ratio in [0, 1]
    """
    if not (0.0 <= max_ratio <= 1.0):
        raise ValueError("max_ratio must be within [0, 1]")
    for c in columns:
        if c not in df.columns:
            continue  # skip missing columns for flexibility in tests
        s = df[c]
        ratio = float(s.isna().mean())
        if ratio > max_ratio:
            raise AssertionError(
                f"NaN ratio {ratio:.3f} for column '{c}' exceeds allowed {max_ratio:.3f}"
            )


def assert_within_range(
    df: pd.DataFrame, *, column: str, min_value: float = -math.inf, max_value: float = math.inf
) -> None:
    """Assert all finite values in the column lie within [min_value, max_value]."""
    if column not in df.columns:
        raise AssertionError(f"Column '{column}' not found in DataFrame")
    s = df[column].dropna().astype(float)
    if not s.empty:
        if (s < min_value).any() or (s > max_value).any():
            raise AssertionError(f"Values in '{column}' out of range [{min_value}, {max_value}]")


@contextmanager
def time_block(max_seconds: float):
    """Assert a code block completes within max_seconds.

    Usage:
        with time_block(0.5):
            run_something()
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if elapsed > max_seconds:
        raise AssertionError(f"Execution exceeded {max_seconds:.3f}s (elapsed {elapsed:.3f}s)")
