from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


def _safe_options(df: pd.DataFrame, col: str) -> List[Dict[str, str]]:
    """Get unique sorted options from a column for a dropdown."""
    if df is None or df.empty or col not in df.columns:
        return []
    values = sorted([v for v in df[col].dropna().astype(str).unique().tolist()])
    return [{"label": v, "value": v} for v in values]


def apply_filters(
    df: pd.DataFrame,
    *,
    sectors: Optional[Iterable[str]] = None,
    regions: Optional[Iterable[str]] = None,
    countries: Optional[Iterable[str]] = None,
    trading_countries: Optional[Iterable[str]] = None,
    industries: Optional[Iterable[str]] = None,
    exchanges: Optional[Iterable[str]] = None,
    style_classes: Optional[Iterable[str]] = None,
    size_classes: Optional[Iterable[str]] = None,
    fiscal_quarters: Optional[Iterable[str]] = None,
    fiscal_years: Optional[Iterable[str]] = None,
    earnings_statuses: Optional[Iterable[str]] = None,
    earnings_reports: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Filter helper with graceful missing-column behavior."""

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else [])

    filtered = df
    filters: List[Tuple[str, Optional[Iterable[str]]]] = [
        ("sector", sectors),
        ("region", regions),
        ("country", countries),
        ("trading_country", trading_countries),
        ("industry", industries),
        ("exchange", exchanges),
        ("style_class", style_classes),
        ("size_class", size_classes),
        ("fiscal_quarter", fiscal_quarters),
        ("fiscal_year", fiscal_years),
        ("next_earnings_status", earnings_statuses),
        ("next_earnings_report", earnings_reports),
    ]

    for col, values in filters:
        values_list = list(values) if values is not None else []
        if not values_list:
            continue
        if col not in filtered.columns:
            continue
        filtered = filtered[filtered[col].isin(values_list)]

    return filtered
