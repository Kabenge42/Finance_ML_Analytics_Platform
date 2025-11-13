"""
Sector-specific feature engineering (Priority 3: Improve Sector-Specific Modeling).

This lightweight module provides helpers to augment the dataframe with
sector-tailored features, addressing large performance variance across sectors.

Design goals:
- Safe: guard for missing columns and division-by-zero; produces finite values
- Minimal dependencies and small surface area for easy integration
- Non-destructive: returns a copy by default to avoid side-effects in callers
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(
    numer: pd.Series | float, denom: pd.Series | float, default: float = 0.0
) -> pd.Series:
    """Element-wise safe division with zero/NaN protection, returns finite series.

    Accepts scalars or Series for convenience; always returns a Series aligned
    with input lengths when Series provided, or a scalar series if scalars used.
    """
    if not isinstance(numer, pd.Series):
        numer = pd.Series([numer])
    if not isinstance(denom, pd.Series):
        denom = pd.Series([denom])

    numer = pd.to_numeric(numer, errors="coerce")
    denom = pd.to_numeric(denom, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numer / denom
    out = out.replace([np.inf, -np.inf], np.nan).fillna(default)
    return out.astype(float)


def engineer_sector_features(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Create sector-specific features for a subset dataframe of a given sector.

    The input df should already be filtered to rows for the specified sector.
    Columns are expected in normalized form per code_guidelines.md. Where columns
    are missing, fallback to zeros to avoid raising errors.
    """
    result = df.copy()

    # Financials: book value, ROE, leverage
    if sector == "Financials":
        mc = result.get("market_cap", 0)
        tbv = result.get("tangible_book_value", result.get("total_equity", 1))
        ni = result.get("net_income", result.get("net_income_is_ltm", 0))
        eq = result.get("shareholders_equity", result.get("total_equity", 1))
        debt = result.get("total_debt", result.get("total_debt_ltm", 0))

        result["p_tbv"] = _safe_div(mc, tbv)
        result["roe"] = _safe_div(ni, eq)
        result["leverage_ratio"] = _safe_div(debt, eq)

    # Industrials: margins, asset turnover, operating leverage
    elif sector == "Industrials":
        rev = result.get(
            "revenue", result.get("total_revenues_ltm", pd.Series(0, index=result.index))
        )
        assets = result.get(
            "total_assets", result.get("total_assets_ltm", pd.Series(1, index=result.index))
        )
        op_income = result.get(
            "operating_income", result.get("operating_income_ltm", pd.Series(0, index=result.index))
        )

        result["asset_turnover"] = _safe_div(rev, assets)
        result["operating_leverage"] = _safe_div(op_income, rev)

    # Information Technology: growth, R&D intensity, gross margins
    elif sector == "Information Technology":
        rd = result.get("r_d_expense", result.get("r_d_expenses", pd.Series(0, index=result.index)))
        rev = result.get(
            "revenue", result.get("total_revenues_ltm", pd.Series(1, index=result.index))
        )
        gp = result.get(
            "gross_profit", result.get("gross_profit_ltm", pd.Series(0, index=result.index))
        )

        result["rd_intensity"] = _safe_div(rd, rev)
        result["gross_margin"] = _safe_div(gp, rev)

    # Real Estate: property-specific proxies when available (Issue 3.2 hint)
    elif sector == "Real Estate":
        # Optional features; default safe computations
        ffo = result.get("ffo", pd.Series(0, index=result.index))
        affo = result.get("affo", pd.Series(0, index=result.index))
        noi = result.get("noi", pd.Series(0, index=result.index))
        mcap = result.get("market_cap", pd.Series(1, index=result.index))
        result["ffo_yield"] = _safe_div(ffo, mcap)
        result["affo_yield"] = _safe_div(affo, mcap)
        result["noi_yield"] = _safe_div(noi, mcap)

    # Health Care: pipeline/regulatory proxies when available (Issue 3.2 hint)
    elif sector == "Health Care":
        r_and_d = result.get(
            "r_d_expense", result.get("r_d_expenses", pd.Series(0, index=result.index))
        )
        rev = result.get(
            "revenue", result.get("total_revenues_ltm", pd.Series(1, index=result.index))
        )
        result["rd_intensity"] = _safe_div(r_and_d, rev)

    return result


def engineer_features_by_sector(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """Apply sector-specific features across entire dataframe by sector.

    Returns a copy with new columns added where applicable.
    """
    if sector_col not in df.columns:
        return df.copy()

    result = df.copy()

    # Track columns that may be created to ensure consistent schema across sectors
    potential_cols = {
        "Financials": ["p_tbv", "roe", "leverage_ratio"],
        "Industrials": ["asset_turnover", "operating_leverage"],
        "Information Technology": ["rd_intensity", "gross_margin"],
        "Real Estate": ["ffo_yield", "affo_yield", "noi_yield"],
        "Health Care": ["rd_intensity"],
    }

    # Pre-create columns with default zeros to avoid assignment issues on masked frames
    for cols in potential_cols.values():
        for col in cols:
            if col not in result.columns:
                result[col] = 0.0

    for sector_value in result[sector_col].dropna().unique():
        mask = result[sector_col] == sector_value
        if mask.any():
            engineered = engineer_sector_features(result.loc[mask], str(sector_value))
            # Ensure all engineered columns are present in result
            new_cols = [c for c in engineered.columns if c not in result.columns]
            for c in new_cols:
                result[c] = 0.0
            result.loc[mask, engineered.columns] = engineered.values
    return result
