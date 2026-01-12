"""
Data split and leakage policy validation (Phase 9.6).

Functions for validating cross-validation splits and detecting leakage:
- Fold overlap analysis
- Grouped CV balance metrics
- Time-aware leakage detection
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Helper Functions (Phase 9.6)
# =============================================================================


def validate_fold_assignments(
    fold_assignments: Optional[pd.DataFrame],
    required_columns: Optional[List[str]] = None,
) -> bool:
    """
    Validate fold_assignments DataFrame structure.

    Replaces unsafe 'fold_assignments' in dir() checks with explicit validation.

    Parameters
    ----------
    fold_assignments : Optional[pd.DataFrame]
        DataFrame to validate
    required_columns : Optional[List[str]]
        List of required column names (default: ['fold'])

    Returns
    -------
    bool
        True if valid, False otherwise

    Examples
    --------
    >>> df = pd.DataFrame({'ticker': ['A', 'B'], 'fold': [0, 1]})
    >>> validate_fold_assignments(df, ['ticker', 'fold'])
    True
    >>> validate_fold_assignments(None)
    False
    """
    if fold_assignments is None:
        return False
    if not isinstance(fold_assignments, pd.DataFrame):
        return False
    if fold_assignments.empty:
        return False

    required_columns = required_columns or ["fold"]
    return all(col in fold_assignments.columns for col in required_columns)


def validate_temporal_data(
    df: Optional[pd.DataFrame],
    date_col: str = "snapshot_date",
) -> bool:
    """
    Validate temporal column availability for leakage checks.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        DataFrame to validate
    date_col : str
        Name of the date/timestamp column

    Returns
    -------
    bool
        True if date column exists and is valid, False otherwise

    Examples
    --------
    >>> df = pd.DataFrame({'snapshot_date': pd.date_range('2024-01-01', periods=10)})
    >>> validate_temporal_data(df, 'snapshot_date')
    True
    >>> validate_temporal_data(None, 'snapshot_date')
    False
    """
    if df is None or df.empty:
        return False
    if date_col not in df.columns:
        return False

    # Check if column has valid datetime-like values
    try:
        pd.to_datetime(df[date_col])
        return True
    except Exception:
        return False


def run_fold_overlap_analysis(
    fold_assignments: Optional[pd.DataFrame],
    output_dir: Union[str, Path],
    group_col: str = "ticker",
) -> Dict[str, Any]:
    """
    Wrapper function for fold overlap analysis with validation.

    Encapsulates validation + compute_fold_overlap() to eliminate unsafe
    dir() checks in notebook cells.

    Parameters
    ----------
    fold_assignments : Optional[pd.DataFrame]
        DataFrame with group_col and 'fold' columns
    output_dir : Union[str, Path]
        Directory to save artifacts
    group_col : str
        Column name for grouping (default: 'ticker')

    Returns
    -------
    Dict[str, Any]
        Overlap statistics or skip metadata

    Examples
    --------
    >>> df = pd.DataFrame({'ticker': ['A', 'B', 'C'], 'fold': [0, 1, 0]})
    >>> result = run_fold_overlap_analysis(df, 'outputs/splits')
    >>> 'n_folds' in result
    True
    """
    if not validate_fold_assignments(fold_assignments, [group_col, "fold"]):
        logger.warning("⚠️ fold_assignments validation failed. Skipping overlap analysis.")
        return {"skipped": True, "reason": "invalid_fold_assignments"}

    logger.info("\n🔍 Computing fold overlap...")
    overlap_dict = compute_fold_overlap(
        fold_assignments=fold_assignments,
        output_dir=output_dir,
        group_col=group_col,
    )

    logger.info("✓ Fold overlap analysis complete")
    logger.info(f"  Zero overlap validated: {overlap_dict.get('zero_overlap_validated', False)}")

    return overlap_dict


# =============================================================================
# Core Functions (Phase 9.6)
# =============================================================================


def compute_fold_overlap(
    fold_assignments: pd.DataFrame,
    output_dir: Union[str, Path],
    group_col: str = "ticker",
) -> Dict[str, Any]:
    """
    Compute overlap statistics across CV folds.

    Creates:
    - fold_overlap_heatmap.html (visualization of overlaps)
    - Overlap summary statistics

    Parameters
    ----------
    fold_assignments : pd.DataFrame
        DataFrame with group_col and 'fold' columns indicating fold assignments
    output_dir : Union[str, Path]
        Directory to save artifacts
    group_col : str
        Column name for grouping (e.g., 'ticker', 'sector')

    Returns
    -------
    Dict[str, Any]
        Overlap statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing fold overlap for {len(fold_assignments)} samples")

    if "fold" not in fold_assignments.columns:
        logger.warning("'fold' column not found in fold_assignments")
        return {"error": "fold column missing"}

    # Get unique folds
    unique_folds = sorted(fold_assignments["fold"].unique())
    n_folds = len(unique_folds)

    # Compute overlap between train/val splits
    overlap_matrix = np.zeros((n_folds, n_folds))

    for i, fold_i in enumerate(unique_folds):
        val_i = set(fold_assignments[fold_assignments["fold"] == fold_i][group_col].unique())
        train_i = set(fold_assignments[fold_assignments["fold"] != fold_i][group_col].unique())

        for j, fold_j in enumerate(unique_folds):
            if i == j:
                overlap_matrix[i, j] = 0  # Same fold
            else:
                val_j = set(
                    fold_assignments[fold_assignments["fold"] == fold_j][group_col].unique()
                )
                # Check if validation set i overlaps with validation set j
                overlap = len(val_i & val_j)
                overlap_matrix[i, j] = overlap

    summary = {
        "n_folds": n_folds,
        "total_groups": int(fold_assignments[group_col].nunique()),
        "max_overlap": int(overlap_matrix.max()),
        "mean_overlap": float(overlap_matrix.mean()),
        "zero_overlap_pairs": int((overlap_matrix == 0).sum() - n_folds),  # Exclude diagonal
    }

    # Create heatmap HTML
    html_path = output_dir / "fold_overlap_heatmap.html"

    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=go.Heatmap(
                z=overlap_matrix,
                x=[f"Fold {i}" for i in unique_folds],
                y=[f"Fold {i}" for i in unique_folds],
                colorscale="Reds",
                text=overlap_matrix.astype(int),
                texttemplate="%{text}",
                textfont={"size": 12},
            )
        )

        fig.update_layout(
            title=f"Fold Overlap Matrix ({group_col})",
            xaxis_title="Fold",
            yaxis_title="Fold",
            height=600,
            width=700,
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved fold overlap heatmap to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Fold Overlap Analysis</h1>")
            f.write(f'<p>Max overlap: {summary["max_overlap"]}</p>')
            f.write("</body></html>")

    return summary


def summarize_grouped_cv_balance(
    fold_assignments: pd.DataFrame,
    output_dir: Union[str, Path],
    group_col: str = "ticker",
    stratify_col: Optional[str] = "sector",
) -> Dict[str, Any]:
    """
    Summarize balance of grouped CV splits.

    Creates:
    - grouped_cv_balance_metrics.json (per-fold statistics)

    Parameters
    ----------
    fold_assignments : pd.DataFrame
        DataFrame with fold assignments and grouping/stratification columns
    output_dir : Union[str, Path]
        Directory to save artifacts
    group_col : str
        Column name for grouping
    stratify_col : Optional[str]
        Column name for stratification (e.g., 'sector')

    Returns
    -------
    Dict[str, Any]
        Balance metrics per fold
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Summarizing CV balance for {len(fold_assignments)} samples")

    if "fold" not in fold_assignments.columns:
        logger.warning("'fold' column not found")
        return {"error": "fold column missing"}

    balance_metrics = {}
    unique_folds = sorted(fold_assignments["fold"].unique())

    for fold in unique_folds:
        fold_data = fold_assignments[fold_assignments["fold"] == fold]

        metrics = {
            "n_samples": int(len(fold_data)),
            "n_groups": (
                int(fold_data[group_col].nunique()) if group_col in fold_data.columns else 0
            ),
        }

        # Stratification balance
        if stratify_col and stratify_col in fold_data.columns:
            strat_counts = fold_data[stratify_col].value_counts().to_dict()
            metrics["stratification"] = {str(k): int(v) for k, v in strat_counts.items()}

        balance_metrics[f"fold_{fold}"] = metrics

    # Overall balance check
    if stratify_col and stratify_col in fold_assignments.columns:
        overall_dist = fold_assignments[stratify_col].value_counts(normalize=True).to_dict()
        balance_metrics["overall_distribution"] = {
            str(k): float(v) for k, v in overall_dist.items()
        }

    # Save JSON
    json_path = output_dir / "grouped_cv_balance_metrics.json"
    with open(json_path, "w") as f:
        json.dump(balance_metrics, f, indent=2)
    logger.info(f"Saved CV balance metrics to {json_path}")

    return balance_metrics


def time_leakage_checks(
    fold_assignments: pd.DataFrame,
    output_dir: Union[str, Path],
    date_col: str = "snapshot_date",
) -> Dict[str, Any]:
    """
    Check for time-based leakage in CV splits.

    Creates:
    - leakage_report.json (leakage detection results)

    Parameters
    ----------
    fold_assignments : pd.DataFrame
        DataFrame with fold assignments and date column
    output_dir : Union[str, Path]
        Directory to save artifacts
    date_col : str
        Column name for date/timestamp

    Returns
    -------
    Dict[str, Any]
        Leakage detection report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checking time-based leakage for {len(fold_assignments)} samples")

    leakage_report = {
        "date_column": date_col,
        "leakage_detected": False,
        "violations": [],
        "summary": "No time-based leakage policy checks performed (date column missing or not applicable)",
    }

    if date_col not in fold_assignments.columns:
        logger.warning(f"Date column '{date_col}' not found, skipping time leakage checks")
    elif "fold" not in fold_assignments.columns:
        logger.warning("'fold' column not found, skipping time leakage checks")
    else:
        # Convert to datetime if needed
        try:
            dates = pd.to_datetime(fold_assignments[date_col])
            fold_assignments = fold_assignments.copy()
            fold_assignments["_date_parsed"] = dates

            unique_folds = sorted(fold_assignments["fold"].unique())
            violations = []

            for fold in unique_folds:
                val_dates = fold_assignments[fold_assignments["fold"] == fold]["_date_parsed"]
                train_dates = fold_assignments[fold_assignments["fold"] != fold]["_date_parsed"]

                if len(val_dates) > 0 and len(train_dates) > 0:
                    val_min = val_dates.min()
                    train_max = train_dates.max()

                    # Check if training data has dates >= validation dates (leakage)
                    if train_max >= val_min:
                        violations.append(
                            {
                                "fold": int(fold),
                                "val_min_date": str(val_min),
                                "train_max_date": str(train_max),
                                "severity": "WARNING",
                            }
                        )

            leakage_report["leakage_detected"] = len(violations) > 0
            leakage_report["violations"] = violations
            leakage_report["summary"] = (
                f"Checked {len(unique_folds)} folds, found {len(violations)} potential time leakage violations"
            )

        except Exception as e:
            logger.warning(f"Error parsing dates: {e}")
            leakage_report["error"] = str(e)

    # Save JSON
    json_path = output_dir / "leakage_report.json"
    with open(json_path, "w") as f:
        json.dump(leakage_report, f, indent=2)
    logger.info(f"Saved leakage report to {json_path}")

    return leakage_report
