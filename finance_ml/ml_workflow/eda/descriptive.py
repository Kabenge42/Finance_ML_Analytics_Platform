"""Descriptive EDA helpers (facade).

Provides a target-architecture import path for core descriptive EDA utilities.
Re-exports implementations from ``finance_ml.ml_workflow.eda.eda`` to avoid
behavior changes during restructuring.
"""

from __future__ import annotations

from typing import Optional, List

import pandas as pd

from finance_ml.ml_workflow.eda.eda import (  # noqa: E402
    eda_summary,
    sector_distribution_summary,
    correlation_analysis,
)


def basic_describe(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Return basic descriptive statistics for selected columns.

    Thin wrapper around ``DataFrame.describe`` to provide a stable API in the
    new module. If ``columns`` is None, uses numeric columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=["number"]).columns.tolist()
    return df[columns].describe().T


__all__ = [
    "eda_summary",
    "sector_distribution_summary",
    "correlation_analysis",
    "basic_describe",
]
