"""Base utilities for dashboard widgets."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union


def resolve_output_path(
    output_path: Optional[Union[str, Path]],
    default_filename: str,
) -> Optional[Path]:
    """Resolve output path to a complete file path for HTML artifacts.

    Handles three input cases consistently:
    1. None → Returns None (no file output)
    2. Directory path → Appends default_filename
    3. File path (with .html) → Uses as-is

    This ensures callers can pass either a directory OR a complete file path,
    following the principle of least surprise.

    Args:
        output_path: Directory path, file path, or None.
        default_filename: Filename to use if output_path is a directory.
            Must end with '.html'.

    Returns:
        Resolved Path object or None if output_path is None.

    Raises:
        ValueError: If default_filename doesn't end with '.html'.

    Examples:
        >>> resolve_output_path(None, "dashboard.html")
        None
        >>> resolve_output_path(Path("outputs/eda"), "dashboard.html")
        PosixPath('outputs/eda/dashboard.html')
        >>> resolve_output_path("outputs/eda/my_dashboard.html", "dashboard.html")
        PosixPath('outputs/eda/my_dashboard.html')
    """
    if output_path is None:
        return None

    if not default_filename.endswith(".html"):
        raise ValueError(f"default_filename must end with '.html': {default_filename}")

    path = Path(output_path)

    # Case 1: Already a complete file path (has .html extension)
    if path.suffix.lower() == ".html":
        return path

    # Case 2: Directory path - append default filename
    # Also handles edge case where path exists as a file without extension
    if path.is_dir() or not path.suffix:
        return path / default_filename

    # Case 3: Has a different extension - treat as directory and append
    # This is defensive; callers should use .html or no extension
    return path / default_filename


def _write_html_artifact(
    fig,
    output_path: Optional[Union[str, Path]],
    default_filename: Optional[str] = None,
) -> Optional[Path]:
    """Write Plotly figure to HTML file with robust path handling.

    Args:
        fig: Plotly figure to save.
        output_path: Output path (directory or file).
        default_filename: Default filename if output_path is a directory.
            If None, output_path must be a complete file path or None.

    Returns:
        Path where file was saved, or None if output_path was None.

    Raises:
        ValueError: If output_path is a directory but no default_filename provided.
    """
    if output_path is None:
        return None

    # Resolve the path
    if default_filename:
        resolved_path = resolve_output_path(output_path, default_filename)
    else:
        resolved_path = Path(output_path)
        # Validate it looks like a file path
        if not resolved_path.suffix:
            raise ValueError(
                f"output_path appears to be a directory but no default_filename provided: {output_path}"
            )

    if resolved_path is None:
        return None

    # Create parent directories (not the file itself)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the HTML
    fig.write_html(str(resolved_path))
    logger.info(f"Widget artifact saved to {resolved_path}")

    return resolved_path


import pandas as pd

from finance_ml.core.constants import (
    DATE_DISPLAY_FORMAT,
)

logger = logging.getLogger(__name__)

# Valid mode options
EarningsMode = Literal[
    "all",
    "earnings",
    "dividends",
    "valuation",
    "quality_risk",
    "technical",
    "forecasts",
    "momentum",
    "profitability",
    "growth",
    "cash_flow",
    "employment",
    "earnings_quality",
]

def resolve_reference_date(
    df: Optional[pd.DataFrame], reference_date: Optional[pd.Timestamp]
) -> pd.Timestamp:
    """Resolve reference_date using dataset provenance with a stable fallback."""
    if reference_date is not None:
        return pd.Timestamp(reference_date).normalize()

    if df is not None:
        for candidate in ["_reference_date", "reference_date"]:
            if candidate in df.columns:
                ref_series = pd.to_datetime(df[candidate], errors="coerce")
                if ref_series.notna().any():
                    return ref_series.dropna().max().normalize()

    return pd.Timestamp.now().normalize()


def add_formatted_date_columns(df: pd.DataFrame, date_columns: List[str]) -> List[str]:
    """Add `*_formatted` companions using the canonical date display format."""
    cols_to_process = list(date_columns)
    # Ensure fy_end_date and next_fy_end_date are processed if they exist
    for extra_col in ["fy_end_date", "next_fy_end_date"]:
        if extra_col in df.columns and extra_col not in cols_to_process:
            cols_to_process.append(extra_col)

    formatted_cols: List[str] = []
    for col in cols_to_process:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], errors="coerce")
        formatted_col = f"{col}_formatted"
        df[formatted_col] = series.dt.strftime(DATE_DISPLAY_FORMAT).where(series.notna(), pd.NA)
        formatted_cols.append(formatted_col)
    return formatted_cols


def _build_format_dict(
    df: pd.DataFrame,
    date_columns: list[str] | None = None,
    date_format: str = "%Y-%m-%d",
) -> dict:
    """Build schema-aware format dictionary for Pandas Styler.

    Applies consistent formatting per code_guidelines.md §17.3:
    - Dates: YYYY-MM-DD format
    - Currency: $1,234.56
    - Percentages: 12.34%
    - Numbers: 2 decimal places
    - Integers: no decimals

    Args:
        df: DataFrame to format.
        date_columns: List of date column names.
        date_format: strftime format for dates.

    Returns:
        Dict mapping column names to format strings or callables.
    """
    format_dict: dict = {}
    date_columns = date_columns or []

    # Column name patterns for classification
    pct_patterns = (
        "_pct",
        "pct_",
        "margin",
        "yield",
        "return",
        "growth",
        "surprise",
        "roe",
        "roa",
        "roic",
        "beat_",
        "miss_",
    )
    currency_patterns = (
        "price",
        "target",
        "market_cap",
        "enterprise_value",
        "revenue",
        "ebitda",
        "ebit",
        "income",
        "eps",
        "dividend",
        "debt",
        "equity",
        "assets",
        "cash",
        "capex",
    )
    ratio_patterns = (
        "ratio",
        "p_e",
        "p_b",
        "p_s",
        "ev_",
        "multiple",
        "score",
        "z_score",
        "f_score",
        "beta",
    )
    count_patterns = ("count", "num_", "days_", "streak", "analysts")

    for col in df.columns:
        col_lower = col.lower()

        # Date columns
        if col in date_columns:
            format_dict[col] = lambda x, fmt=date_format: x.strftime(fmt) if pd.notnull(x) else "—"
            continue

        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        # Percentage columns: 12.34%
        if any(p in col_lower for p in pct_patterns):
            format_dict[col] = "{:.2f}%"
            continue

        # Currency/financial columns: $1,234.56 or plain with commas for large values
        if any(p in col_lower for p in currency_patterns):
            # Check if values are large (market cap, revenue) vs small (EPS, price)
            max_val = df[col].abs().max() if df[col].notna().any() else 0
            if max_val > 1_000_000:
                format_dict[col] = "{:,.0f}"  # No decimals for large values
            else:
                format_dict[col] = "{:,.2f}"  # 2 decimals for prices/EPS
            continue

        # Ratio/score columns: 2 decimals
        if any(p in col_lower for p in ratio_patterns):
            format_dict[col] = "{:.2f}"
            continue

        # Count/integer columns: no decimals
        if any(p in col_lower for p in count_patterns):
            format_dict[col] = "{:,.0f}"
            continue

        # Default numeric: 2 decimals
        format_dict[col] = "{:.2f}"

    return format_dict


def _ensure_schema_dtypes(df: pd.DataFrame, date_columns: list[str] | None = None) -> pd.DataFrame:
    """Ensure dataframe columns have appropriate dtypes for visualization."""
    df_copy = df.copy()
    if date_columns:
        for col in date_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors="coerce")
    return df_copy

@dataclass
class EarningsAlertConfig:
    """Configuration for earnings quality alerts."""
    eps_surprise_miss_threshold_pct: float = 5.0
    analyst_downgrade_threshold_pct: float = 10.0
    analyst_downgrade_min_periods: int = 2
    target_spread_threshold_pct: float = 25.0
    pre_earnings_window_days: int = 14
    pre_earnings_volatility_quantile: float = 0.80
    max_tickers_per_alert: int = 10
    gaap_adj_diff_threshold_pct: float = 10.0
    negative_profitability_warning: bool = True
    high_valuation_warning_pe: float = 50.0
    low_quality_score_threshold: float = 0.4
