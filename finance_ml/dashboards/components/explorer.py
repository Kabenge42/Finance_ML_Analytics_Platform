from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from finance_ml.ml_workflow.preprocessing.column_semantics import classify_columns


def build_explorer_column_options(
    df: Optional[pd.DataFrame],
    categories: Optional[Iterable[str]] = None,
    base_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[dict], List[str]]:
    """Build explorer dropdown options using canonical column classification.

    Args:
        df: DataFrame backing the explorer. If None/empty, returns empty options.
        categories: Optional semantic categories to prioritize (price, ratio, etc.).
        base_columns: Base columns to include by default (e.g., DEFAULT_EXPLORER_COLUMNS).

    Returns:
        options: List of dash-friendly option dicts.
        defaults: Default selected values drawn from ``base_columns`` when available.
    """

    if df is None or df.empty:
        return [], []

    available_cols = list(df.columns)
    categories_list = list(categories or [])
    base_cols = list(base_columns or [])

    classification = classify_columns(available_cols)

    selected: List[str] = []

    # Always include base columns if present
    for col in base_cols:
        if col in available_cols and col not in selected:
            selected.append(col)

    # Add classified columns based on requested categories
    for category in categories_list:
        for col in sorted(classification.get(category, [])):
            if col in available_cols and col not in selected:
                selected.append(col)

    # Fallback: if no categories requested, include a handful of price/ratio columns
    if not categories_list:
        for fallback_cat in ("price", "ratio", "market_value"):
            for col in sorted(classification.get(fallback_cat, [])):
                if col in available_cols and col not in selected:
                    selected.append(col)

    options = [
        {"label": col.replace("_", " ").title(), "value": col} for col in selected
    ]
    defaults = [col for col in base_cols if col in selected]

    return options, defaults


__all__ = ["build_explorer_column_options"]
