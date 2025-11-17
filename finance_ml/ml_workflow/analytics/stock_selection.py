"""Stock selection helpers for portfolio optimization workflows.

This module implements Phase 1 utilities from
docs/improvement_plan/portfolio_optimization_enhancement_plan.md:

- rank_stocks_multi_metric: composite scoring across multiple metrics
- rank_stocks_balanced: sector‑balanced ranking with max sector share
- select_portfolio_candidates: notebook‑friendly wrapper that combines
  filtering and ranking into a single call.

The functions are intentionally lightweight, rely only on pandas/numpy,
and are easy to unit test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .eval import filter_stocks_by_criteria


def _validate_weights(metrics: Sequence[str], weights: Sequence[float]) -> np.ndarray:
    """Validate and normalize metric weights.

    The enhancement plan expresses weights as plain Python lists; this helper
    keeps the public APIs simple while providing robust validation that is
    easy to test in isolation.
    """

    if len(metrics) != len(weights):  # pragma: no cover - defensive branch
        raise ValueError("metrics and weights must have the same length")

    weights_arr = np.asarray(weights, dtype=float)
    if weights_arr.ndim != 1 or weights_arr.size == 0:
        raise ValueError("weights must be a non‑empty 1D sequence")

    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")

    return weights_arr / total


def rank_stocks_multi_metric(
    df: pd.DataFrame,
    metrics: Sequence[str],
    weights: Sequence[float],
    top_n: Optional[int] = None,
    descending: bool = True,
    composite_col: str = "composite_score",
) -> pd.DataFrame:
    """Rank stocks using a weighted combination of multiple metrics.

    Parameters
    ----------
    df:
        Input DataFrame containing the metric columns.
    metrics:
        Column names to use in the composite score.
    weights:
        Weights associated with ``metrics``; they are normalized to sum to 1.
    top_n:
        Optional cap on the number of rows to return. If ``None`` all rows
        are returned sorted by the composite score.
    descending:
        If True (default), higher scores are considered better.
    composite_col:
        Name of the column that will store the composite score.
    """

    if not metrics:
        raise ValueError("metrics list must not be empty")

    weights_arr = _validate_weights(metrics, weights)

    missing_cols = [m for m in metrics if m not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing metric columns: {missing_cols}")

    # Use values to avoid alignment surprises and keep it fast.
    metric_values = df[list(metrics)].to_numpy(dtype=float)
    composite = metric_values @ weights_arr

    ranked = df.copy()
    ranked[composite_col] = composite
    ranked = ranked.sort_values(composite_col, ascending=not descending)

    if top_n is not None:
        ranked = ranked.head(int(top_n))

    return ranked


def rank_stocks_balanced(
    df: pd.DataFrame,
    top_n: int,
    max_sector_weight: float,
    ranking_col: str = "mispricing_score",
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Rank stocks with a maximum share per sector.

    This helper implements the "sector balance" requirement from the
    enhancement plan: no sector should represent more than
    ``max_sector_weight`` fraction of the selected universe.
    """

    if sector_col not in df.columns:
        raise KeyError(f"Column '{sector_col}' not found in DataFrame")
    if ranking_col not in df.columns:
        raise KeyError(f"Column '{ranking_col}' not found in DataFrame")
    if not (0 < max_sector_weight <= 1):
        raise ValueError("max_sector_weight must be in (0, 1]")

    # Sort by ranking column descending (higher is better) and select top_n
    ranked = df.sort_values(ranking_col, ascending=False).copy()
    ranked = ranked.head(int(top_n))

    total = len(ranked)
    if total == 0:
        return ranked

    max_count = int(np.floor(max_sector_weight * total))
    if max_count <= 0:
        # Fall back to at least one per sector if top_n is small.
        max_count = 1

    def _cap_group(group: pd.DataFrame) -> pd.DataFrame:
        return group.head(max_count)

    balanced = ranked.groupby(sector_col, group_keys=False).apply(_cap_group)

    # The grouped cap may reduce the total number of rows below top_n.
    return balanced.reset_index(drop=True)


@dataclass
class PortfolioCandidateConfig:
    """Configuration for ``select_portfolio_candidates`` helper.

    This dataclass mainly documents the parameters used in the notebook; the
    function below also exposes them directly for convenience.
    """

    min_market_cap: float = 1.0
    cap_unit: str = "B"
    top_n: int = 50
    max_sector_weight: float = 0.25


def select_portfolio_candidates(
    df: pd.DataFrame,
    min_market_cap: float = 1.0,
    top_n: int = 50,
    max_sector_weight: float = 0.25,
    cap_unit: str = "B",
) -> pd.DataFrame:
    """Pipeline helper used by the notebook's portfolio section.

    The implementation mirrors the pseudo‑code in the enhancement plan:

    1. Apply ``filter_stocks_by_criteria`` using market‑cap thresholds.
    2. Rank candidates using a composite score built from expected_return,
       1‑year historical return, and mispricing score.
    3. Apply sector balance constraints via ``rank_stocks_balanced``.
    """

    filtered = filter_stocks_by_criteria(
        df,
        sectors=None,
        regions=None,
        min_market_cap=min_market_cap,
        cap_unit=cap_unit,
    )

    # If any of the ranking metrics are missing, raise an explicit error so
    # notebook and tests fail loudly rather than silently degrading.
    required_cols = ["expected_return", "return_1y", "mispricing_score"]
    missing = [c for c in required_cols if c not in filtered.columns]
    if missing:
        available_cols = list(filtered.columns[:20])
        raise KeyError(
            f"select_portfolio_candidates: missing required columns {missing}\n"
            f"\n"
            f"Required columns: {required_cols}\n"
            f"Available columns (first 20): {available_cols}...\n"
            f"\n"
            f"💡 Solution: Call ensure_portfolio_metrics() BEFORE select_portfolio_candidates():\n"
            f"\n"
            f"    from finance_ml.ml_workflow.analytics.portfolio_metrics import ensure_portfolio_metrics\n"
            f"    df_with_metrics = ensure_portfolio_metrics(your_dataframe)\n"
            f"    portfolio_candidates = select_portfolio_candidates(df_with_metrics, ...)\n"
            f"\n"
            f"See: docs/improvement_plan/portfolio_optimization_enhancement_plan.md\n"
        )

    ranked = rank_stocks_multi_metric(
        filtered,
        metrics=["expected_return", "return_1y", "mispricing_score"],
        weights=[0.5, 0.3, 0.2],
        top_n=top_n,
    )

    balanced = rank_stocks_balanced(
        ranked,
        top_n=top_n,
        max_sector_weight=max_sector_weight,
        ranking_col="composite_score",
    )

    return balanced
