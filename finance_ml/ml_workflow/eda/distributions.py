"""Distribution analysis helpers (facade).

Aligns EDA distribution utilities with the target architecture. Re-exports
existing implementations and provides a couple of thin convenience wrappers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.eda.eda import distribution_summary  # noqa: E402


def histogram(
    df: pd.DataFrame, column: str, bins: int = 10, range: Optional[tuple[float, float]] = None
) -> pd.DataFrame:
    """Return histogram bin edges and counts as a DataFrame.

    This is a lightweight helper suitable for tests and simple analytics where
    returning a table is preferred over plotting.
    """
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    counts, bin_edges = np.histogram(series.values, bins=bins, range=range)
    return pd.DataFrame(
        {
            "bin_left": bin_edges[:-1],
            "bin_right": bin_edges[1:],
            "count": counts,
        }
    )


def empirical_cdf(
    df: pd.DataFrame, column: str, values: Optional[Sequence[float]] = None
) -> pd.DataFrame:
    """Compute an empirical CDF table for a numeric column.

    If ``values`` is provided, returns the CDF evaluated at those points;
    otherwise returns it at the sorted unique values of the column.
    """
    series = pd.to_numeric(df[column], errors="coerce").dropna().sort_values()
    n = len(series)
    if n == 0:
        return pd.DataFrame({"value": [], "cdf": []})
    if values is None:
        x = series.values
    else:
        x = np.asarray(list(values), dtype=float)
    # For each x, fraction of data <= x
    cdf_vals = np.searchsorted(series.values, x, side="right") / n
    return pd.DataFrame({"value": x, "cdf": cdf_vals})


__all__ = [
    "distribution_summary",
    "histogram",
    "empirical_cdf",
]
