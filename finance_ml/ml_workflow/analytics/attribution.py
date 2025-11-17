"""Performance attribution helpers (Phase 5 of the portfolio plan).

This module implements a minimal Brinson–Fachler attribution function to
support the backtesting framework tests. The focus is on correctness for
single‑period, sector‑level attribution, which is sufficient for the
unit tests derived from the enhancement plan.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def calculate_performance_attribution(
    portfolio_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_weights: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
) -> Dict[str, float]:
    """Compute Brinson–Fachler performance attribution.

    Parameters
    ----------
    portfolio_weights, benchmark_weights:
        DataFrames of weights with the same columns (sectors or asset
        groups). Rows represent time periods; the current implementation
        aggregates across all rows.
    portfolio_returns, benchmark_returns:
        DataFrames of realised returns for the same columns and index.

    Returns
    -------
    dict
        Dictionary with keys ``allocation_effect``, ``selection_effect``
        and ``interaction_effect`` (all floats). The sum of these three
        effects equals the total excess return of the portfolio relative
        to the benchmark (up to numerical precision).
    """

    # Align on index/columns and convert to NumPy for speed and clarity.
    pw, pr = portfolio_weights.align(portfolio_returns, join="inner", axis=1)
    bw, br = benchmark_weights.align(benchmark_returns, join="inner", axis=1)

    # Use all periods available; attribution tests currently use a
    # single‑period example so summing over rows is sufficient.
    pw_values = pw.to_numpy(dtype=float)
    pr_values = pr.to_numpy(dtype=float)
    bw_values = bw.to_numpy(dtype=float)
    br_values = br.to_numpy(dtype=float)

    # Brinson–Fachler sector‑level effects per period and sector.
    allocation = (pw_values - bw_values) * br_values
    selection = bw_values * (pr_values - br_values)
    interaction = (pw_values - bw_values) * (pr_values - br_values)

    # Aggregate over sectors and periods.
    allocation_effect = float(allocation.sum())
    selection_effect = float(selection.sum())
    interaction_effect = float(interaction.sum())

    return {
        "allocation_effect": allocation_effect,
        "selection_effect": selection_effect,
        "interaction_effect": interaction_effect,
    }


__all__ = ["calculate_performance_attribution"]
