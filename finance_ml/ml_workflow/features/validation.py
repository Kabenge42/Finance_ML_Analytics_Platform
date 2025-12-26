"""
Phase 9.3 feature engineering validation and pruning utilities.

This lightweight module provides helpers to:
- Validate that a sufficient number of engineered features are present
- Prune low-importance features based on an importance DataFrame

Designed for minimal coupling and easy use from notebooks and pipelines.

UPDATED: 2025-12-27
- Aligned with schema.py as Single Source of Truth
- validate_feature_coverage now derives expected count from PHASE93_FEATURE_CATEGORIES
- Added get_schema_expected_feature_count() helper
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Union

import pandas as pd

from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES


def get_schema_expected_feature_count() -> int:
    """
    Get expected feature count from PHASE93_FEATURE_CATEGORIES (Single Source of Truth).

    Returns:
        Total number of unique features defined across all Phase 9.3 categories
    """
    all_features = set()
    for features in PHASE93_FEATURE_CATEGORIES.values():
        all_features.update(features)
    return len(all_features)


def validate_feature_coverage(
    X: pd.DataFrame,
    expected: Optional[Union[Iterable[str], int]] = None,
    strict: bool = False,
) -> Tuple[bool, dict]:
    """
    Validate engineered feature coverage.

    Args:
        X: Feature matrix used for modeling (columns are features).
        expected: Either an integer expected count, an iterable of expected feature names,
                  or None to use the schema-derived count from PHASE93_FEATURE_CATEGORIES.
        strict: If True and names provided, fail when any expected feature is missing.

    Returns:
        (ok, report) where report contains counts and lists of missing/extra features.
    """
    report: dict = {
        "feature_count": int(X.shape[1]),
        "expected_count": None,
        "missing": [],
        "extra": [],
        "coverage_pct": 0.0,
    }

    # Default to schema-derived count if not specified
    if expected is None:
        expected = get_schema_expected_feature_count()

    if isinstance(expected, int):
        report["expected_count"] = expected
        report["coverage_pct"] = (X.shape[1] / expected * 100) if expected > 0 else 0.0
        ok = X.shape[1] >= expected
        return ok, report

    # Treat as list of names
    exp_list = list(expected)
    report["expected_count"] = len(exp_list)
    missing = [c for c in exp_list if c not in X.columns]
    extra = [c for c in X.columns if c not in exp_list]
    report["missing"] = missing
    report["extra"] = extra
    report["coverage_pct"] = ((len(exp_list) - len(missing)) / len(exp_list) * 100) if exp_list else 0.0
    ok = len(missing) == 0 if strict else X.shape[1] >= len(exp_list) * 0.9
    return ok, report


def validate_feature_coverage_by_category(
    X: pd.DataFrame,
    min_coverage_pct: float = 90.0,
) -> Tuple[bool, dict]:
    """
    Validate feature coverage against PHASE93_FEATURE_CATEGORIES from the schema.

    This function checks coverage by category, providing detailed diagnostics
    about which categories are well-covered vs. lacking features.

    Args:
        X: Feature matrix used for modeling (columns are features).
        min_coverage_pct: Minimum overall percentage coverage required (default 90%).

    Returns:
        (ok, report) where report contains:
            - total_expected: Total unique features in schema
            - total_found: Total matching features in X
            - coverage_pct: Overall coverage percentage
            - by_category: Dict with expected/found counts per category
            - missing_by_category: Dict mapping category to list of missing features
    """
    all_expected = set()
    for features in PHASE93_FEATURE_CATEGORIES.values():
        all_expected.update(features)

    # Track by category
    by_category = {}
    missing_by_category = {}
    total_found = 0

    for category, features in PHASE93_FEATURE_CATEGORIES.items():
        present = [f for f in features if f in X.columns]
        missing = [f for f in features if f not in X.columns]
        by_category[category] = {
            "expected": len(features),
            "found": len(present),
            "coverage_pct": (len(present) / len(features) * 100) if features else 0.0,
        }
        if missing:
            missing_by_category[category] = missing
        total_found += len(present)

    total_expected = len(all_expected)
    coverage_pct = (total_found / total_expected * 100) if total_expected > 0 else 0.0

    report = {
        "total_expected": total_expected,
        "total_found": total_found,
        "coverage_pct": coverage_pct,
        "by_category": by_category,
        "missing_by_category": missing_by_category,
    }

    ok = coverage_pct >= min_coverage_pct
    return ok, report


def prune_low_importance_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame],
    importance_df: pd.DataFrame,
    threshold: float = 0.01,
    keep_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Drop features whose normalized importance is strictly below the given threshold.

    Args:
        X_train: Training feature matrix.
        X_test: Optional test feature matrix aligned by column names.
        importance_df: DataFrame with at least columns ['feature','importance'].
        threshold: Minimum importance to keep a feature (e.g., 0.01 for 1%).
        keep_cols: Columns to always keep regardless of importance.

    Returns:
        (X_train_pruned, X_test_pruned, kept_importances)
    """
    if keep_cols is None:
        keep_cols = []

    if not {"feature", "importance"}.issubset(importance_df.columns):
        raise ValueError("importance_df must contain 'feature' and 'importance' columns")

    imp = importance_df.copy()
    # Normalize importance if not already normalized
    total = imp["importance"].sum()
    if total > 0:
        imp["importance_norm"] = imp["importance"] / total
    else:
        imp["importance_norm"] = imp["importance"]

    # Determine features to keep
    keep_mask = (imp["importance_norm"] >= threshold) | (imp["feature"].isin(keep_cols))
    kept = imp.loc[keep_mask, ["feature", "importance", "importance_norm"]].copy()
    keep_set = set(kept["feature"]) | set(keep_cols)

    # Intersect with existing X columns to avoid KeyErrors
    keep_cols_final = [c for c in X_train.columns if c in keep_set]
    if not keep_cols_final:
        # If nothing passes threshold, do not prune; return originals
        return X_train, X_test, imp.sort_values("importance", ascending=False)

    X_train_pruned = X_train[keep_cols_final].copy()
    X_test_pruned = X_test[keep_cols_final].copy() if X_test is not None else None

    return (
        X_train_pruned,
        X_test_pruned,
        kept.sort_values("importance", ascending=False),
    )


def save_feature_list(features: Iterable[str], path: Path) -> None:
    """Persist list of features to a text file (one per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for feat in features:
            f.write(str(feat) + "\n")
