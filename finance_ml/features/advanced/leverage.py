"""Leverage, liquidity, and efficiency feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_leverage_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer leverage and solvency ratios.

    Ratios computed:
    - Debt to Equity
    - Net Debt to EBITDA
    - Interest Coverage
    - Debt to Assets
    - Equity Ratio

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with leverage ratios added
    """
    result = df.copy()

    # Debt to Equity
    if "total_debt" in df.columns and "total_equity" in df.columns:
        result["debt_to_equity"] = _safe_div(df["total_debt"], df["total_equity"])

    # Net Debt to EBITDA
    if "net_debt" in df.columns and "ebitda" in df.columns:
        result["net_debt_to_ebitda"] = _safe_div(df["net_debt"], df["ebitda"])

    # Interest Coverage (EBIT / Interest Expense)
    if "ebit" in df.columns and "interest_expense" in df.columns:
        result["interest_coverage"] = _safe_div(df["ebit"], df["interest_expense"])

    # Debt to Assets
    if "total_debt" in df.columns and "total_assets" in df.columns:
        result["debt_to_assets"] = _safe_div(df["total_debt"], df["total_assets"])

    # Equity Ratio
    if "total_equity" in df.columns and "total_assets" in df.columns:
        result["equity_ratio"] = _safe_div(df["total_equity"], df["total_assets"])

    logger.info("Engineered leverage ratios")
    return result

def engineer_liquidity_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer liquidity ratios.

    Ratios computed:
    - Current Ratio
    - Quick Ratio (Acid Test)
    - Cash Ratio
    - Working Capital to Sales

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with liquidity ratios added
    """
    result = df.copy()

    # Current Ratio
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        result["current_ratio"] = _safe_div(df["current_assets"], df["current_liabilities"])

    # Quick Ratio (Current Assets - Inventory) / Current Liabilities
    if (
        "current_assets" in df.columns
        and "inventory" in df.columns
        and "current_liabilities" in df.columns
    ):
        quick_assets = df["current_assets"] - df["inventory"].fillna(0)
        result["quick_ratio"] = _safe_div(quick_assets, df["current_liabilities"])

    # Cash Ratio
    if "cash_and_equivalents" in df.columns and "current_liabilities" in df.columns:
        result["cash_ratio"] = _safe_div(df["cash_and_equivalents"], df["current_liabilities"])

    # Working Capital to Sales
    if "working_capital" in df.columns and "revenue" in df.columns:
        result["working_capital_to_sales"] = _safe_div(df["working_capital"], df["revenue"])

    logger.info("Engineered liquidity ratios")
    return result

def engineer_efficiency_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer efficiency and activity ratios.

    Ratios computed:
    - Asset Turnover
    - Inventory Turnover
    - Receivables Turnover
    - Revenue per Employee

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with efficiency ratios added
    """
    result = df.copy()

    # Asset Turnover
    if "revenue" in df.columns and "total_assets" in df.columns:
        result["asset_turnover"] = _safe_div(df["revenue"], df["total_assets"])

    # Inventory Turnover (COGS / Average Inventory)
    if "cost_of_revenues_ltm" in df.columns and "inventory" in df.columns:
        result["inventory_turnover"] = _safe_div(df["cost_of_revenues_ltm"], df["inventory"])

    # Receivables Turnover (Revenue / Accounts Receivable)
    if "revenue" in df.columns and "accounts_receivable_total_fy" in df.columns:
        result["receivables_turnover"] = _safe_div(
            df["revenue"], df["accounts_receivable_total_fy"]
        )

    # Revenue per Employee
    if "revenue" in df.columns and "full_time_employees_fy" in df.columns:
        result["revenue_per_employee"] = _safe_div(df["revenue"], df["full_time_employees_fy"])

    logger.info("Engineered efficiency ratios")
    return result

def engineer_balance_sheet_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer balance sheet growth and liquidity trends.

    Features computed (when inputs exist):
    - debt_growth_rate, equity_growth_rate, asset_growth_rate
    - balance_sheet_expansion: mean of available growth rates
    - current_ratio_trend: current_ratio_ltm - current_ratio_fy
    - cash_ratio: cash_and_equivalents / current_liabilities
    - working_capital_ratio: working_capital_ltm / total_assets_ltm
    - retained_earnings_growth: (retained_earnings_ltm - retained_earnings_fy) / total_equity_ltm
    - earnings_retention_rate: (retained_earnings_ltm - retained_earnings_fy) / net_income_ltm
    """
    result = df.copy()

    # Growth rates
    if all(c in df.columns for c in ("total_debt_ltm", "total_debt_fy")):
        result["debt_growth_rate"] = _safe_div(
            df["total_debt_ltm"].astype(float) - df["total_debt_fy"].astype(float),
            df["total_debt_fy"].astype(float),
        )
    if all(c in df.columns for c in ("total_equity_ltm", "total_equity_fy")):
        result["equity_growth_rate"] = _safe_div(
            df["total_equity_ltm"].astype(float) - df["total_equity_fy"].astype(float),
            df["total_equity_fy"].astype(float),
        )
    if all(c in df.columns for c in ("total_assets_ltm", "total_assets_fy")):
        result["asset_growth_rate"] = _safe_div(
            df["total_assets_ltm"].astype(float) - df["total_assets_fy"].astype(float),
            df["total_assets_fy"].astype(float),
        )

    # Composite expansion = mean of available growth rates
    growth_cols = [
        c
        for c in ("debt_growth_rate", "equity_growth_rate", "asset_growth_rate")
        if c in result.columns
    ]
    if growth_cols:
        result["balance_sheet_expansion"] = result[growth_cols].mean(axis=1, skipna=True)

    # Liquidity trends
    if all(c in df.columns for c in ("current_ratio_ltm", "current_ratio_fy")):
        result["current_ratio_trend"] = df["current_ratio_ltm"].astype(float) - df[
            "current_ratio_fy"
        ].astype(float)
    if all(c in df.columns for c in ("cash_and_equivalents", "current_liabilities")):
        result["cash_ratio"] = _safe_div(
            df["cash_and_equivalents"].astype(float),
            df["current_liabilities"].astype(float),
        )
    if all(c in df.columns for c in ("working_capital_ltm", "total_assets_ltm")):
        result["working_capital_ratio"] = _safe_div(
            df["working_capital_ltm"].astype(float),
            df["total_assets_ltm"].astype(float),
        )

    # Retained earnings patterns
    if all(
        c in df.columns
        for c in ("retained_earnings_ltm", "retained_earnings_fy", "total_equity_ltm")
    ):
        delta_re = df["retained_earnings_ltm"].astype(float) - df["retained_earnings_fy"].astype(
            float
        )
        result["retained_earnings_growth"] = _safe_div(
            delta_re, df["total_equity_ltm"].astype(float)
        )
    if all(
        c in df.columns for c in ("retained_earnings_ltm", "retained_earnings_fy", "net_income_ltm")
    ):
        delta_re = df["retained_earnings_ltm"].astype(float) - df["retained_earnings_fy"].astype(
            float
        )
        result["earnings_retention_rate"] = _safe_div(delta_re, df["net_income_ltm"].astype(float))

    logger.info("Engineered balance sheet growth & liquidity trends")
    return result
