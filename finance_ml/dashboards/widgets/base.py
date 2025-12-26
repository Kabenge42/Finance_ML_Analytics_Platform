"""Base utilities for dashboard widgets."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Union

import pandas as pd
import plotly.graph_objects as go

from finance_ml.core.constants import (
    DATE_DISPLAY_FORMAT,
    DEFAULT_REFERENCE_DATE,
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

    return DEFAULT_REFERENCE_DATE

def add_formatted_date_columns(df: pd.DataFrame, date_columns: List[str]) -> List[str]:
    """Add `*_formatted` companions using the canonical date display format."""
    formatted_cols: List[str] = []
    for col in date_columns:
        if col not in df.columns:
            continue
        series = pd.to_datetime(df[col], errors="coerce")
        formatted_col = f"{col}_formatted"
        df[formatted_col] = series.dt.strftime(DATE_DISPLAY_FORMAT).where(series.notna(), pd.NA)
        formatted_cols.append(formatted_col)
    return formatted_cols

def _write_html_artifact(fig: go.Figure, output_path: Optional[Union[str, Path]]) -> None:
    """Write a Plotly figure to HTML when an output path is provided."""
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(out))
        logger.info(f"Widget artifact saved to {out}")

def _build_format_dict(df: pd.DataFrame, date_columns: list[str] | None = None, date_format: str = DATE_DISPLAY_FORMAT) -> dict:
    """Build a column formatting dictionary for pandas Styler."""
    format_dict = {}
    if date_columns:
        for col in date_columns:
            if col in df.columns:
                format_dict[col] = lambda x: x.strftime(date_format) if pd.notnull(x) else ""
    
    # Currency formatting for price-like columns
    for col in df.columns:
        if any(p in col.lower() for p in ["price", "market_cap", "enterprise_value"]):
            if "pct" not in col.lower() and "ratio" not in col.lower():
                format_dict[col] = "${:,.2f}"
        elif "pct" in col.lower() or "growth" in col.lower() or "margin" in col.lower():
            format_dict[col] = "{:.2f}%"
            
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
