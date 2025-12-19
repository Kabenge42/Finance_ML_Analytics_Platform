"""
finance_ml.ml_workflow.features.advanced - Advanced feature engineering

This module implements sophisticated feature engineering techniques:
- Comprehensive financial ratios (valuation, profitability, leverage, liquidity, efficiency, growth)
- Sector-specific features for major sectors (Financials, Energy, Tech, Healthcare, etc.)
- Temporal features (earnings dates, time-based patterns)
- Market microstructure features (spreads, relative positioning)
- Nonlinear transforms (log, sqrt, inverse)
- Feature interactions and polynomial features
- Relative value features (sector-normalized metrics)
- Analyst quality features (coverage, target spread, rating consensus)
- Accounting quality features (exceptional items, write-downs)
- Employee productivity features (revenue/employee, assets/employee)

Phase 9.3 refactor: Extracted from advanced_features.py for better modularity.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "engineer_valuation_ratios",
    "engineer_profitability_ratios",
    "engineer_leverage_ratios",
    "engineer_liquidity_ratios",
    "engineer_efficiency_ratios",
    "engineer_growth_metrics",
    "engineer_sector_specific_features",
    "engineer_temporal_features",
    "engineer_market_microstructure_features",
    "engineer_nonlinear_transforms",
    "create_feature_interactions",
    "create_relative_value_features",
    "engineer_analyst_quality_features",
    "engineer_market_sentiment_features",
    "engineer_accounting_quality_features",
    "engineer_financial_distress_features",
    "engineer_cash_flow_quality_features",
    "engineer_capital_allocation_features",
    "engineer_margin_trends",
    "engineer_balance_sheet_trends",
    "engineer_composite_scores",
    "engineer_sector_relative_interactions",
    "engineer_employee_productivity_features",
    "engineer_estimated_vs_actual_analytics",
    "engineer_gaap_vs_adjusted_analytics",
    "build_comprehensive_features",
]


def _safe_div(numer: pd.Series | float | int, denom: pd.Series) -> pd.Series:
    """Safely divide two Series (or scalar and Series), replacing inf/NaN with appropriate values.

    Args:
        numer: Numerator Series or scalar (float/int)
        denom: Denominator Series

    Returns:
        Result Series with inf/NaN handled
    """
    # Handle scalar numerator by converting to Series
    if isinstance(numer, (float, int)):
        numer = pd.Series(numer, index=denom.index)

    result = numer.astype(float) / denom.astype(float).replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def _ensure_float_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Ensure a column exists and is float64 dtype to prevent TypeError on masked assignment.

    This helper prevents TypeError when assigning float values to StringDtype or other incompatible
    columns during sector-specific feature engineering with boolean masks.

    Args:
        df: DataFrame to modify
        col_name: Column name to ensure as float64

    Returns:
        Modified DataFrame with column guaranteed to be float64

    Example:
        >>> df = _ensure_float_column(df, "efficiency_ratio")
        >>> df.loc[mask, "efficiency_ratio"] = values.loc[mask]  # No TypeError
    """
    if col_name not in df.columns:
        df[col_name] = pd.Series(np.nan, index=df.index, dtype="float64")
    else:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    return df


def engineer_valuation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive valuation ratios.

    Ratios computed:
    - P/E (Price/Earnings)
    - P/B (Price/Book)
    - P/S (Price/Sales)
    - EV/EBITDA
    - EV/Sales
    - PEG (P/E to Growth)
    - Dividend Yield

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with valuation ratios added
    """
    result = df.copy()

    # Book Value per Share
    if "total_equity" in df.columns and "shares_outstanding" in df.columns:
        result["book_value_per_share"] = _safe_div(df["total_equity"], df["shares_outstanding"])

    # P/E ratio
    if "last_price" in df.columns and "eps" in df.columns:
        result["p_e_ratio"] = _safe_div(df["last_price"], df["eps"])

    # P/B ratio
    if "last_price" in df.columns and "book_value_per_share" in df.columns:
        result["p_b_ratio"] = _safe_div(df["last_price"], df["book_value_per_share"])

    # P/S ratio (Price to Sales per share)
    if (
        "last_price" in df.columns
        and "revenue" in df.columns
        and "shares_outstanding" in df.columns
    ):
        sales_per_share = _safe_div(df["revenue"], df["shares_outstanding"])
        result["p_s_ratio"] = _safe_div(df["last_price"], sales_per_share)

    # EV/EBITDA
    if "enterprise_value" in df.columns and "ebitda" in df.columns:
        result["ev_ebitda_ratio"] = _safe_div(df["enterprise_value"], df["ebitda"])

    # EV/Sales
    if "enterprise_value" in df.columns and "revenue" in df.columns:
        result["ev_sales_ratio"] = _safe_div(df["enterprise_value"], df["revenue"])

    # PEG ratio (P/E to Growth)
    if "p_e" in result.columns and "earnings_growth" in df.columns:
        result["peg_ratio"] = _safe_div(result["p_e"], df["earnings_growth"])

    # Dividend Yield
    if "dividend_per_share" in df.columns and "last_price" in df.columns:
        result["dividend_yield"] = _safe_div(df["dividend_per_share"], df["last_price"]) * 100

    logger.info("Engineered valuation ratios")
    return result


def engineer_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer profitability ratios.

    Ratios computed:
    - ROE (Return on Equity)
    - ROA (Return on Assets)
    - ROIC (Return on Invested Capital)
    - Gross Margin %
    - Operating Margin %
    - Net Margin %
    - EBITDA/EBIT adjustment ratios (adj/LTM)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with profitability ratios added
    """
    result = df.copy()

    # ROE
    if "net_income" in df.columns and "total_equity" in df.columns:
        result["roe"] = _safe_div(df["net_income"], df["total_equity"]) * 100

    # ROA
    if "net_income" in df.columns and "total_assets" in df.columns:
        result["roa"] = _safe_div(df["net_income"], df["total_assets"]) * 100

    # ROIC (simplified: Net Income / (Total Equity + Total Debt))
    if "net_income" in df.columns and "total_equity" in df.columns and "total_debt" in df.columns:
        invested_capital = df["total_equity"] + df["total_debt"]
        result["roic"] = _safe_div(df["net_income"], invested_capital) * 100

    # Gross Margin %
    if "gross_profit" in df.columns and "revenue" in df.columns:
        result["gross_margin_pct"] = _safe_div(df["gross_profit"], df["revenue"]) * 100

    # Operating Margin %
    if "operating_income" in df.columns and "revenue" in df.columns:
        result["operating_margin_pct"] = _safe_div(df["operating_income"], df["revenue"]) * 100

    # Net Margin %
    if "net_income" in df.columns and "revenue" in df.columns:
        result["net_margin_pct"] = _safe_div(df["net_income"], df["revenue"]) * 100

    # Adjustment ratios (adj/LTM/FY) as robustness/quality proxies
    if "ebitda_adj_ltm" in df.columns and "ebitda_ltm" in df.columns:
        result["ebitda_adjustment_ratio_ltm"] = _safe_div(
            df["ebitda_adj_ltm"].abs(), df["ebitda_ltm"].abs()
        )

    if "ebitda_adj_fy" in df.columns and "ebitda_fy" in df.columns:
        result["ebitda_adjustment_ratio_fy"] = _safe_div(
            df["ebitda_adj_fy"].abs(), df["ebitda_fy"].abs()
        )

    if "ebit_adj_ltm" in df.columns and "ebit_ltm" in df.columns:
        result["ebit_adjustment_ratio_ltm"] = _safe_div(
            df["ebit_adj_ltm"].abs(), df["ebit_ltm"].abs()
        )

    if "ebit_adj_fy" in df.columns and "ebit_fy" in df.columns:
        result["ebit_adjustment_ratio_fy"] = _safe_div(df["ebit_adj_fy"].abs(), df["ebit_fy"].abs())

    if "net_income_adj_ltm" in df.columns and "net_income_is_ltm" in df.columns:
        result["net_income_adjustment_ratio_ltm"] = _safe_div(
            df["normalized_net_income_fq"].abs(), df["net_income_is_ltm"].abs()
        )

    if "net_income_adj_fy" in df.columns and "net_income_is_fy" in df.columns:
        result["net_income_adjustment_ratio_fy"] = _safe_div(
            df["net_income_adj_fy"].abs(), df["net_income_is_fy"].abs()
        )

    if "net_income_adj_1fy" in df.columns and "net_income_is_1fy" in df.columns:
        result["net_income_adjustment_ratio_fy"] = _safe_div(
            df["net_income_adj_1fy"].abs(), df["net_income_is_1fy"].abs()
        )

    logger.info("Engineered profitability ratios")
    return result


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
    if "revenue" in df.columns and "accounts_receivable_fy" in df.columns:
        result["receivables_turnover"] = _safe_div(df["revenue_fy"], df["accounts_receivable_fy"])

    # Revenue per Employee
    if "revenue" in df.columns and "full_time_employees_fy" in df.columns:
        result["revenue_per_employee"] = _safe_div(df["revenue"], df["full_time_employees_fy"])

    logger.info("Engineered efficiency ratios")
    return result


def engineer_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer growth metrics.

    Metrics computed:
    - Revenue CAGR (if multi-year data available)
    - EPS Growth %
    - EBITDA Growth %
    - Book Value Growth %

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with growth metrics added
    """
    result = df.copy()

    # Revenue Growth (YoY if available)
    if "revenue" in df.columns and "revenue_previous_year" in df.columns:
        result["revenue_growth_yoy"] = (
            _safe_div(
                (df["revenue"] - df["revenue_previous_year"]),
                df["revenue_previous_year"],
            )
            * 100
        )
        # Backward-compatible alias for event labels
        result["revenue_growth"] = result["revenue_growth_yoy"]

    # EPS Growth
    if "eps" in df.columns and "eps_previous_year" in df.columns:
        result["eps_growth_yoy"] = (
            _safe_div((df["eps"] - df["eps_previous_year"]), df["eps_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["earnings_growth"] = result["eps_growth_yoy"]

    # EBITDA Growth
    if "ebitda" in df.columns and "ebitda_previous_year" in df.columns:
        result["ebitda_growth_yoy"] = (
            _safe_div((df["ebitda"] - df["ebitda_previous_year"]), df["ebitda_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["ebitda_growth"] = result["ebitda_growth_yoy"]

    logger.info("Engineered growth metrics")
    return result


def engineer_sector_specific_features(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """
    Add sector-specific engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with normalized columns.
    sector_col : str, default "sector"
        Column name indicating sector classification.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional sector-specific features.
    """
    result = df.copy()

    # Basic sector masks
    financials_mask = result[sector_col] == "Financials"
    # Energy/Materials sector features
    energy_mask = result[sector_col].str.contains("Energy|Materials", case=False, na=False)
    # Technology sector features
    tech_mask = result[sector_col].str.contains("Technology|Information", case=False, na=False)
    # Healthcare sector features
    health_mask = result[sector_col].str.contains("Health", case=False, na=False)
    # Consumer sector features (Consumer Discretionary, Consumer Staples)
    consumer_mask = result[sector_col].str.contains("Consumer", case=False, na=False)
    # Industrials sector features
    industrials_mask = result[sector_col].str.contains("Industrial", case=False, na=False)
    # Utilities sector features
    utilities_mask = result[sector_col].str.contains("Utilities", case=False, na=False)

    if financials_mask.any():
        # Add Tangible Book Value features if applicable
        if "total_equity" in df.columns and "intangible_assets" in df.columns:
            # Ensure numeric source columns (protect against unexpected dtypes)
            total_equity = pd.to_numeric(df["total_equity"], errors="coerce")
            intangible_assets = pd.to_numeric(df["intangible_assets"], errors="coerce").fillna(0)

            tangible_book_value = total_equity - intangible_assets

            # Ensure the target column is a floating dtype, not StringDtype
            if "tangible_book_value" in result.columns:
                result["tangible_book_value"] = pd.to_numeric(
                    result["tangible_book_value"], errors="coerce"
                )
            else:
                # Create an all-NaN float column if it doesn't exist yet
                result["tangible_book_value"] = pd.Series(
                    np.nan, index=result.index, dtype="float64"
                )

            # Now safe to assign numeric values into this column
            result.loc[financials_mask, "tangible_book_value"] = tangible_book_value.loc[
                financials_mask
            ]

        # Price to Tangible Book Value ratio
        if (
            "last_price" in df.columns
            and "shares_outstanding" in df.columns
            and "tangible_book_value" in result.columns
        ):
            # Ensure numeric types for ratio calculation
            last_price = pd.to_numeric(df["last_price"], errors="coerce")
            shares_outstanding = pd.to_numeric(df["shares_outstanding"], errors="coerce")
            tbv = pd.to_numeric(result["tangible_book_value"], errors="coerce")

            market_cap_tbv = last_price * shares_outstanding

            # Create / ensure numeric ratio column
            if "p_tbv_ratio" in result.columns:
                result["p_tbv_ratio"] = pd.to_numeric(result["p_tbv_ratio"], errors="coerce")
            else:
                result["p_tbv_ratio"] = pd.Series(np.nan, index=result.index, dtype="float64")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = market_cap_tbv / tbv.replace(0, np.nan)

            result.loc[financials_mask, "p_tbv_ratio"] = ratio.loc[financials_mask]

        # Net Interest Margin
        if all(
            col in df.columns for col in ["interest_income", "interest_expense", "earning_assets"]
        ):
            result = _ensure_float_column(result, "net_interest_margin")
            net_interest_income = df["interest_income"] - df["interest_expense"]
            result.loc[financials_mask, "net_interest_margin"] = (
                _safe_div(net_interest_income, df["earning_assets"]) * 100
            ).loc[financials_mask]

        # Efficiency Ratio
        if "operating_expenses" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "efficiency_ratio")
            result.loc[financials_mask, "efficiency_ratio"] = (
                _safe_div(df["operating_expenses"], df["revenue"]) * 100
            ).loc[financials_mask]

    # Energy/Materials sector features
    if energy_mask.any():
        # CAPEX Intensity
        if "capex" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "capex_intensity")
            result.loc[energy_mask, "capex_intensity"] = (
                _safe_div(df["capex"], df["revenue"]) * 100
            ).loc[energy_mask]

        # Asset Turnover
        if "revenue" in df.columns and "total_assets" in df.columns:
            result = _ensure_float_column(result, "asset_turnover")
            result.loc[energy_mask, "asset_turnover"] = _safe_div(
                df["revenue"], df["total_assets"]
            ).loc[energy_mask]

    # Technology sector features
    if tech_mask.any():
        # R&D Intensity
        if "r_d_expenses" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "r_d_intensity")
            result.loc[tech_mask, "r_d_intensity"] = (
                _safe_div(df["r_d_expenses"], df["revenue"]) * 100
            ).loc[tech_mask]

        # SG&A Efficiency
        if "sga_expenses" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "sga_efficiency")
            result.loc[tech_mask, "sga_efficiency"] = (
                _safe_div(df["sga_expenses"], df["revenue"]) * 100
            ).loc[tech_mask]

        # Rule of 40 (Growth + Margin)
        if "revenue_growth_yoy" in df.columns and "operating_margin_pct" in df.columns:
            result = _ensure_float_column(result, "rule_of_40")
            result.loc[tech_mask, "rule_of_40"] = (
                df["revenue_growth_yoy"] + df["operating_margin_pct"]
            ).loc[tech_mask]

        # Cash Burn Rate
        if "operating_cash_flow" in df.columns and "capex" in df.columns:
            result = _ensure_float_column(result, "cash_burn_rate")
            result.loc[tech_mask, "cash_burn_rate"] = (df["operating_cash_flow"] - df["capex"]).loc[
                tech_mask
            ]

    # Healthcare sector features
    if health_mask.any():
        # R&D intensity for healthcare
        if "r_d_expenses" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "r_d_intensity")
            result.loc[health_mask, "r_d_intensity"] = (
                _safe_div(df["r_d_expenses"], df["revenue"]) * 100
            ).loc[health_mask]

    # Consumer sector features (Consumer Discretionary, Consumer Staples)
    if consumer_mask.any():
        # Inventory Days
        if "inventory" in df.columns and "cost_of_goods_sold" in df.columns:
            result = _ensure_float_column(result, "inventory_days")
            result.loc[consumer_mask, "inventory_days"] = (
                _safe_div(df["inventory"], df["cost_of_goods_sold"]) * 365
            ).loc[consumer_mask]

        # Marketing Efficiency
        if "marketing_expenses" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "marketing_efficiency")
            result.loc[consumer_mask, "marketing_efficiency"] = (
                _safe_div(df["marketing_expenses"], df["revenue"]) * 100
            ).loc[consumer_mask]

    # Industrials sector features
    if industrials_mask.any():
        # CAPEX Intensity
        if "capex" in df.columns and "revenue" in df.columns:
            result = _ensure_float_column(result, "capex_intensity")
            result.loc[industrials_mask, "capex_intensity"] = (
                _safe_div(df["capex"], df["revenue"]) * 100
            ).loc[industrials_mask]

        # CAPEX to Depreciation ratio
        if "capex" in df.columns and "depreciation_amortization" in df.columns:
            result = _ensure_float_column(result, "capex_to_depreciation")
            result.loc[industrials_mask, "capex_to_depreciation"] = _safe_div(
                df["capex"],
                df["depreciation_amortization"],
            ).loc[industrials_mask]

        # Working Capital Efficiency
        if all(col in df.columns for col in ["current_assets", "current_liabilities", "revenue"]):
            result = _ensure_float_column(result, "working_capital_efficiency")
            working_capital = df["current_assets"] - df["current_liabilities"]
            result.loc[industrials_mask, "working_capital_efficiency"] = (
                _safe_div(working_capital, df["revenue"]) * 100
            ).loc[industrials_mask]

    # Utilities sector features
    if utilities_mask.any():
        # Dividend Payout Ratio
        if "dividends_paid" in df.columns and "net_income" in df.columns:
            result = _ensure_float_column(result, "dividend_payout_ratio")
            result.loc[utilities_mask, "dividend_payout_ratio"] = (
                _safe_div(df["dividends_paid"], df["net_income"]) * 100
            ).loc[utilities_mask]

    logger.info(f"Engineered sector-specific features")
    return result


def engineer_temporal_features(
    df: pd.DataFrame,
    date_col: str = "last_updated",
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Engineer temporal and seasonality features using consistent reference_date.

    Per code_guidelines.md Section 9.3.0: All temporal calculations MUST use
    reference_date instead of last_updated for consistency and reproducibility.

    Adds:
    - fiscal_quarter, month, year from date_col
    - _reference_date column for auditing
    - days_to_earnings: (next_earnings - reference_date).days
    - earnings_report_recency: (reference_date - income_statement_report_date).days
    - reporting_lag: (next_earnings - income_statement_report_date).days
    - ltm_vs_5yavg_revenue: (total_revenues_1fy - 5Y avg)/5Y avg
    - fq_vs_5yavg_ebitda: (ebitda_fq - ebitda_5yavgfq)/ebitda_5yavgfq
    - quarterly_volatility_score: coefficient of variation across quarterly EBITDA

    Args:
        df: Input DataFrame with date columns.
        date_col: Column name for fiscal timing (quarter, month, year).
        reference_date: Reference date for temporal calculations.
                       Defaults to pd.Timestamp.now().normalize() if not provided.

    Returns:
        DataFrame with temporal features added.
    """
    result = df.copy()

    # Use reference_date per code_guidelines.md Section 9.3.0
    if reference_date is None:
        effective_ref_date = pd.Timestamp.now().normalize()
    else:
        effective_ref_date = pd.Timestamp(reference_date).normalize()

    # Store reference date for auditing/reproducibility
    result["_reference_date"] = effective_ref_date

    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found, skipping some temporal features")
    else:
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
            try:
                result[date_col] = pd.to_datetime(result[date_col])
            except Exception as e:
                logger.warning(f"Could not convert {date_col} to datetime: {e}")

        if (
            pd.api.types.is_datetime64_any_dtype(result[date_col])
            and not result[date_col].isna().all()
        ):
            # Extract fiscal timing features
            result["fiscal_quarter"] = result[date_col].dt.quarter
            result["month"] = result[date_col].dt.month
            result["year"] = result[date_col].dt.year

            # Days since reference date (only if explicitly requested via parameter)
            if reference_date is not None:
                result["days_since_reference"] = (result[date_col] - effective_ref_date).dt.days

    # Calculate days_to_earnings using reference_date (NOT last_updated)
    if "next_earnings" in result.columns:
        result["next_earnings"] = pd.to_datetime(result["next_earnings"], errors="coerce")
        result["days_to_earnings"] = (result["next_earnings"] - effective_ref_date).dt.days

        logger.debug(
            f"Calculated days_to_earnings using reference_date={effective_ref_date}: "
            f"{(result['days_to_earnings'].abs() <= 30).sum()} events within ±30 days"
        )

    # Calculate days_to_dividend using reference_date
    if "dividend_record_ex_date" in result.columns:
        result["dividend_record_ex_date"] = pd.to_datetime(
            result["dividend_record_ex_date"], errors="coerce"
        )
        result["days_to_dividend"] = (
            result["dividend_record_ex_date"] - effective_ref_date
        ).dt.days

    # Calculate earnings_report_recency using reference_date
    if "income_statement_report_date" in result.columns:
        result["income_statement_report_date"] = pd.to_datetime(
            result["income_statement_report_date"], errors="coerce"
        )
        result["earnings_report_recency"] = (
            effective_ref_date - result["income_statement_report_date"]
        ).dt.days

    if "income_statement_report_date" in result.columns and "next_earnings" in result.columns:
        isrd = pd.to_datetime(result["income_statement_report_date"], errors="coerce")
        ne = pd.to_datetime(result["next_earnings"], errors="coerce")
        result["reporting_lag"] = (ne - isrd).dt.days

    # Seasonality vs 5Y averages
    rev_5y_cols = [
        c
        for c in (
            "total_revenues_5yavg",
            "total_revenues_5yavgfq",
            "revenue_5yavg",
            "total_revenues_ltm",
        )
        if c in result.columns
    ]
    if "total_revenues_1fy" in result.columns and rev_5y_cols:
        base = result[rev_5y_cols[0]].astype(float)
        result["ltm_vs_5yavg_revenue"] = _safe_div(
            result["total_revenues_1fy"].astype(float) - base, base
        )
    elif "total_revenues_ltm" in result.columns and "total_revenues_5yavg" in result.columns:
        base = result["total_revenues_5yavg"].astype(float)
        result["ltm_vs_5yavg_revenue"] = _safe_div(
            result["total_revenues_ltm"].astype(float) - base, base
        )

    if "ebitda_fq" in result.columns and "ebitda_5yavgfq" in result.columns:
        base = result["ebitda_5yavgfq"].astype(float)
        result["fq_vs_5yavg_ebitda"] = _safe_div(result["ebitda_fq"].astype(float) - base, base)

    # Quarterly volatility score
    quarterly_cols = [c for c in result.columns if c.startswith("ebitda_fq")]
    if quarterly_cols:
        qmat = pd.concat([result[c].astype(float) for c in quarterly_cols], axis=1)
        mean = qmat.mean(axis=1)
        std = qmat.std(axis=1, ddof=0)
        result["quarterly_volatility_score"] = _safe_div(std, mean)

    logger.info(f"Engineered temporal features using reference_date={effective_ref_date}")
    return result


def engineer_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer price momentum and technical indicators.

    Features (added when sufficient columns are available):
    - price_momentum_1m, 3m, 6m, 1y: Percent change vs price_Nm_ago columns
    - price_acceleration_3m: mom_3m - mom_1m (rate-of-change proxy)
    - rsi_14d: 14-day RSI computed from last_price and price_{1..14}d_ago columns
    - rsi_30d: 30-day RSI if 30-day history is present
    - ma_crossover_signal: 1 if MA20>MA50 and price>MA50, -1 if MA20<MA50 and price<MA50, else 0
    - price_distance_from_ma: % distance of last_price from MA50
    - return_stability_score: total_return_1y_pct / volatility_1y_pct
    - sharpe_proxy: (total_return_1y_pct - risk_free_rate_pct) / volatility_1y_pct

    Notes:
    - All percentage features are expressed in percent (not decimals).
    - Missing inputs result in NaN for the affected features; no exceptions raised.
    """
    result = df.copy()

    def pct_change(cur: pd.Series, prev: pd.Series) -> pd.Series:
        """Calculate percentage change between current and previous values."""
        return _safe_div(cur - prev, prev) * 100

    # Basic momentum windows
    if "last_price" in df.columns and "price_20d_ago" in df.columns:
        result["momentum_20d"] = pct_change(df["last_price"], df["price_20d_ago"])
    if "last_price" in df.columns and "price_1m_ago" in df.columns:
        result["price_momentum_1m"] = pct_change(df["last_price"], df["price_1m_ago"])
    if "last_price" in df.columns and "price_3m_ago" in df.columns:
        result["price_momentum_3m"] = pct_change(df["last_price"], df["price_3m_ago"])
    if "last_price" in df.columns and "price_6m_ago" in df.columns:
        result["price_momentum_6m"] = pct_change(df["last_price"], df["price_6m_ago"])
    if "last_price" in df.columns and "price_1y_ago" in df.columns:
        result["price_momentum_1y"] = pct_change(df["last_price"], df["price_1y_ago"])

    # Acceleration vs 1m
    if "price_momentum_3m" in result.columns and "price_momentum_1m" in result.columns:
        result["price_acceleration_3m"] = result["price_momentum_3m"] - result["price_momentum_1m"]

    # RSI helper (row-wise due to per-row wide history columns)
    def compute_rsi_row(row: pd.Series, period: int) -> float:
        """Compute RSI (Relative Strength Index) for a single row over specified period."""
        # Build sequence oldest->newest using daily columns if present
        prices = []
        # Include historical days period back to 1 day
        for d in range(period, 0, -1):
            col = f"price_{d}d_ago"
            prices.append(row.get(col, np.nan))
        prices.append(row.get("last_price", np.nan))
        arr = np.asarray(prices, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        deltas = np.diff(arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss == 0 and avg_gain == 0:
            return 50.0  # flat
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(rsi)

    # RSI 14d
    have_14 = (
        all(f"price_{d}d_ago" in df.columns for d in range(14, 0, -1))
        and "last_price" in df.columns
    )
    if have_14:
        result["rsi_14d"] = df.apply(lambda r: compute_rsi_row(r, 14), axis=1)

    # RSI 30d
    have_30 = (
        all(f"price_{d}d_ago" in df.columns for d in range(30, 0, -1))
        and "last_price" in df.columns
    )
    if have_30:
        result["rsi_30d"] = df.apply(lambda r: compute_rsi_row(r, 30), axis=1)

    # Use existing EMA columns from database instead of computing MAs from non-existent daily history
    # The database provides ema_20d, ema_50d, ema_100d, ema_250d which are already calculated
    # EMAs provide equivalent (arguably better) technical signals than simple moving averages
    if "ema_20d" in df.columns:
        result["ma_20d_simple"] = df["ema_20d"]
    if "ema_50d" in df.columns:
        result["ma_50d_simple"] = df["ema_50d"]

    if "last_price" in df.columns:
        # price distance from MA50
        if "ma_50d_simple" in result.columns:
            result["price_distance_from_ma"] = (
                _safe_div(df["last_price"] - result["ma_50d_simple"], result["ma_50d_simple"]) * 100
            )
        # crossover signal
        if "ma_20d_simple" in result.columns and "ma_50d_simple" in result.columns:
            cond_up = (result["ma_20d_simple"] > result["ma_50d_simple"]) & (
                df["last_price"] > result["ma_50d_simple"]
            )
            cond_down = (result["ma_20d_simple"] < result["ma_50d_simple"]) & (
                df["last_price"] < result["ma_50d_simple"]
            )
            signal = pd.Series(0, index=df.index, dtype=float)
            signal[cond_up] = 1.0
            signal[cond_down] = -1.0
            result["ma_crossover_signal"] = signal

    # Return stability and Sharpe proxy
    if "last_price" in df.columns and "price_1y_ago" in df.columns:
        total_return_pct = pct_change(df["last_price"], df["price_1y_ago"]).rename(
            "total_return_1y_pct"
        )
        result["total_return_1y_pct"] = total_return_pct
        if "volatility_1y_pct" in df.columns:
            vol = df["volatility_1y_pct"].astype(float)
            result["return_stability_score"] = _safe_div(total_return_pct, vol)
            try:
                rf = float(os.getenv("RISK_FREE_RATE_PCT", "0.0"))
            except (ValueError, TypeError):
                rf = 0.0
            excess = total_return_pct - rf
            result["sharpe_proxy"] = _safe_div(excess, vol)

    logger.info("Engineered momentum & technical features")
    return result


def engineer_market_microstructure_features(
    df: pd.DataFrame,
    price_col: str = "last_price",
    high_col: str = "52w_high_adj",
    low_col: str = "52w_low_adj",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Engineer market microstructure features (volatility, momentum, moving averages).

    Args:
        df: Input DataFrame
        price_col: Name of price column
        high_col: Name of high price column (for range calculation)
        low_col: Name of low price column (for range calculation)
        group_col: Optional grouping column (e.g., ticker) for time-series features

    Returns:
        DataFrame with market microstructure features added
    """
    result = df.copy()

    if price_col not in df.columns:
        logger.warning(
            f"Price column '{price_col}' not found, skipping market microstructure features"
        )
        return result

    # Price range indicator (requires high and low prices)
    if high_col in df.columns and low_col in df.columns:
        price_range = df[high_col] - df[low_col]
        result["price_range_pct"] = _safe_div(price_range, df[price_col]) * 100

    # Time-series features (volatility, momentum, moving averages)
    if group_col and group_col in df.columns:
        # Historical volatility (30, 60, 90 day rolling windows)
        for window in [30, 60, 90]:
            result[f"volatility_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.pct_change()
                .rolling(window=window, min_periods=max(1, window // 2))
                .std()
                * 100
            )

        # Momentum (rate of change over 20 days)
        result["momentum_20d"] = df.groupby(group_col)[price_col].transform(
            lambda x: x.pct_change(periods=20) * 100
        )

        # Moving averages (20, 50 day)
        for window in [20, 50]:
            result[f"ma_{window}d"] = df.groupby(group_col)[price_col].transform(
                lambda x: x.rolling(window=window, min_periods=max(1, window // 2)).mean()
            )
    else:
        # Without grouping, calculate simple rolling features if enough data
        if len(df) >= 30:
            for window in [30, 60, 90]:
                if len(df) >= window:
                    result[f"volatility_{window}d"] = (
                        df[price_col]
                        .pct_change()
                        .rolling(window=window, min_periods=window // 2)
                        .std()
                        * 100
                    )

            if len(df) >= 20:
                result["momentum_20d"] = df[price_col].pct_change(periods=20) * 100

            for window in [20, 50]:
                if len(df) >= window:
                    result[f"ma_{window}d"] = (
                        df[price_col].rolling(window=window, min_periods=window // 2).mean()
                    )

    logger.info("Engineered market microstructure features")
    return result


def engineer_nonlinear_transforms(
    df: pd.DataFrame,
    log_features: Optional[List[str]] = None,
    sqrt_features: Optional[List[str]] = None,
    inverse_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply non-linear transformations to features.

    Args:
        df: Input DataFrame
        log_features: Features to apply natural log transformation (for skewed distributions)
        sqrt_features: Features to apply square root transformation
        inverse_features: Features to apply inverse transformation (1/x)

    Returns:
        DataFrame with non-linear transformed features added
    """
    result = df.copy()

    # Log transformation (natural log)
    if log_features:
        for feature in log_features:
            if feature in df.columns:
                # Only apply log to positive values
                result[f"log_{feature}"] = df[feature].apply(
                    lambda x: np.log(x) if x > 0 else np.nan
                )

    # Square root transformation
    if sqrt_features:
        for feature in sqrt_features:
            if feature in df.columns:
                # Only apply sqrt to non-negative values
                result[f"sqrt_{feature}"] = df[feature].apply(
                    lambda x: np.sqrt(x) if x >= 0 else np.nan
                )

    # Inverse transformation (1/x)
    if inverse_features:
        for feature in inverse_features:
            if feature in df.columns:
                result[f"inv_{feature}"] = _safe_div(pd.Series([1.0] * len(df)), df[feature])

    logger.info(f"Applied non-linear transforms")
    return result


def create_feature_interactions(
    df: pd.DataFrame, features: Optional[List[str]] = None, max_degree: int = 2
) -> pd.DataFrame:
    """Create polynomial and interaction features.

    Args:
        df: Input DataFrame
        features: Features to create interactions for (default: key financial metrics)
        max_degree: Maximum polynomial degree (default: 2)

    Returns:
        DataFrame with interaction features added
    """
    result = df.copy()

    if features is None:
        # Default key features for interactions
        features = [
            "market_cap",
            "p_e_ratio",
            "roe",
            "debt_to_equity",
            "revenue_growth_yoy",
        ]
        features = [f for f in features if f in df.columns]

    if len(features) == 0:
        logger.warning("No features available for interactions")
        return result

    # Create pairwise interactions (requires at least 2 features)
    if len(features) >= 2:
        for i, feat1 in enumerate(features):
            for feat2 in features[i + 1 :]:
                interaction_name = f"{feat1}_x_{feat2}"
                result[interaction_name] = df[feat1] * df[feat2]
    else:
        logger.warning("Not enough features for pairwise interactions (need 2+)")

    # Create polynomial features if degree > 1
    if max_degree >= 2:
        for feat in features:
            result[f"{feat}_squared"] = df[feat] ** 2

    logger.info(f"Created {len(result.columns) - len(df.columns)} interaction features")
    return result


def create_relative_value_features(
    df: pd.DataFrame, sector_col: str = "sector", metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create relative value features (deviations from sector median).

    Args:
        df: Input DataFrame
        sector_col: Name of sector column
        metrics: Metrics to compute relative values for

    Returns:
        DataFrame with relative value features added
    """
    result = df.copy()

    if sector_col not in df.columns:
        logger.warning(f"Sector column '{sector_col}' not found")
        return result

    if metrics is None:
        metrics = ["p_e_ratio", "p_b_ratio", "roe", "net_margin_pct", "debt_to_equity"]
        metrics = [m for m in metrics if m in df.columns]

    for metric in metrics:
        if metric not in df.columns:
            continue

        # Calculate sector median
        sector_median = df.groupby(sector_col)[metric].transform("median")

        # Z-score relative to sector
        sector_mean = df.groupby(sector_col)[metric].transform("mean")
        sector_std = df.groupby(sector_col)[metric].transform("std")

        result[f"{metric}_vs_sector_median"] = df[metric] - sector_median
        result[f"{metric}_sector_zscore"] = _safe_div(df[metric] - sector_mean, sector_std)

        # Percentile rank within sector
        result[f"{metric}_sector_percentile"] = df.groupby(sector_col)[metric].rank(pct=True) * 100

    logger.info(f"Created relative value features for {len(metrics)} metrics")
    return result


def engineer_analyst_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer analyst quality, consensus, and price target features.

    Features computed (when inputs exist):
    - Analyst consensus: analyst_bullish_pct, analyst_bearish_pct, analyst_conviction (abs diff in pct points)
    - Price target metrics: price_target_spread_pct, price_target_range (alias), consensus_strength (100-spread),
      upside_potential ((median-last)/last * 100), price_target_revision ((median - ytd_ago)/ytd_ago)
    - Coverage quality: analyst_coverage_quality = (# analysts) / log1p(market_cap)
    - Backward-compatibility: target_price_upside_pct alias retained if last_price + price_target present

    Column naming (normalized expected; legacy tolerated where possible):
    - Ratings: strong_buy_ratings, buy_ratings, hold_ratings, sell_ratings, strong_sell_ratings
    - Targets: price_target_median, price_target_high, price_target_low, price_target_ytd_ago, price_target_number
    - Other: last_price, market_cap
    """
    result = df.copy()

    # --- Price target spread and consensus strength ---
    if all(
        c in df.columns for c in ("price_target_high", "price_target_low", "price_target_median")
    ):
        target_range = df["price_target_high"].astype(float) - df["price_target_low"].astype(float)
        spread_pct = _safe_div(target_range, df["price_target_median"].astype(float)) * 100
        result["price_target_spread_pct"] = spread_pct
        # Alias used by tests/plan
        result["price_target_range"] = spread_pct
        result["consensus_strength"] = 100 - spread_pct.clip(upper=100)

    # --- Analyst ratings distribution & consensus ---
    # Support normalized names primarily; allow legacy names with leading underscores if present
    cols_norm = [
        "num_strong_buys_ratings",
        "num_buys_ratings",
        "num_hold_ratings",
        "num_sell_ratings",
        "num_strong_sell_ratings",
    ]
    cols_legacy = [
        "_strong_buy_ratings",
        "_buy_ratings",
        "_hold_ratings",
        "_sell_ratings",
        "_strong_sell_ratings",
    ]
    use_cols = None
    if all(c in df.columns for c in cols_norm):
        use_cols = cols_norm
    elif all(c in df.columns for c in cols_legacy):
        use_cols = cols_legacy
    if use_cols is not None:
        sb, b, h, s, ss = [df[c].astype(float).fillna(0) for c in use_cols]
        total = sb + b + h + s + ss
        bullish = sb + b
        bearish = s + ss
        result["analyst_bullish_pct"] = _safe_div(bullish, total) * 100
        result["analyst_bearish_pct"] = _safe_div(bearish, total) * 100
        # Conviction: absolute difference in percentage points
        if "analyst_bullish_pct" in result.columns and "analyst_bearish_pct" in result.columns:
            result["analyst_conviction"] = (
                result["analyst_bullish_pct"] - result["analyst_bearish_pct"]
            ).abs()

    # --- Upside potential and revisions ---
    if all(c in df.columns for c in ("price_target_median", "last_price")):
        upside = (
            _safe_div(
                df["price_target_median"].astype(float) - df["last_price"].astype(float),
                df["last_price"].astype(float),
            )
            * 100
        )
        result["upside_potential"] = upside
        # Backward-compatible alias
        result["target_price_upside_pct"] = upside
    if all(c in df.columns for c in ("price_target_median", "price_target_ytd_ago")):
        result["price_target_revision"] = _safe_div(
            df["price_target_median"].astype(float) - df["price_target_ytd_ago"].astype(float),
            df["price_target_ytd_ago"].astype(float),
        )

    # --- Coverage quality (#analysts scaled by firm size) ---
    if "price_target_number" in df.columns and "market_cap" in df.columns:
        # log1p(market_cap) in denominator; safe-div guards zero/negatives (log1p of negative is NaN)
        denom = pd.Series(np.log1p(df["market_cap"].astype(float)), index=df.index)
        result["analyst_coverage_quality"] = _safe_div(
            df["price_target_number"].astype(float), denom
        )

    logger.info("Engineered analyst quality & consensus features")
    return result


def engineer_accounting_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer accounting quality and red flag features.

    Features computed:
    - Exceptional items flags and aggregation (goodwill impairment, asset writedowns, restructuring)
    - Exceptional items scaling ratios (to EBITDA/Net Income) and trend (YoY if available)
    - Goodwill to assets ratio (+ change rate), Intangibles intensity
    - Restructuring intensity to total assets
    - Composite accounting quality score (0-100, higher is better)

    Args:
        df: Input DataFrame (normalized column names expected)

    Returns:
        DataFrame with accounting quality features added
    """
    result = df.copy()

    # Goodwill impairment flag (red flag if present)
    if "impairment_of_goodwill_ltm" in df.columns:
        result["has_goodwill_impairment"] = (
            df["impairment_of_goodwill_ltm"].fillna(0) != 0
        ).astype(int)
        # Alias for compatibility with tests/plan wording
        result["goodwill_impairment_flag"] = result["has_goodwill_impairment"]

    # Asset writedown flag
    if "asset_writedown_ltm" in df.columns:
        result["has_asset_writedown"] = (df["asset_writedown_ltm"].fillna(0) != 0).astype(int)

    # Restructuring charges flag
    if "restructuring_charges_ltm" in df.columns:
        result["has_restructuring"] = (df["restructuring_charges_ltm"].fillna(0) != 0).astype(int)

    # Aggregate exceptional items (LTM)
    if all(
        c in df.columns
        for c in [
            "impairment_of_goodwill_ltm",
            "asset_writedown_ltm",
            "restructuring_charges_ltm",
        ]
    ):
        exceptional_items_ltm = (
            df["impairment_of_goodwill_ltm"].fillna(0).abs()
            + df["asset_writedown_ltm"].fillna(0).abs()
            + df["restructuring_charges_ltm"].fillna(0).abs()
        )
        result["total_exceptional_items_ltm"] = exceptional_items_ltm

        # Scale to EBITDA if available
        if "ebitda_ltm" in df.columns:
            result["exceptional_items_to_ebitda"] = _safe_div(
                exceptional_items_ltm, df["ebitda_ltm"].abs()
            )

        # Backward compatible ratio to Net Income (percent)
        if "net_income_ltm" in df.columns:
            result["exceptional_items_to_ni_pct"] = (
                _safe_div(exceptional_items_ltm, df["net_income_ltm"].abs()) * 100
            )

    # Exceptional items trend YoY if -1FY columns exist
    if all(
        c in df.columns
        for c in [
            "impairment_of_goodwill_1fy",
            "asset_writedown_1fy",
            "restructuring_charges_1fy",
        ]
    ):
        exceptional_items_1fy = (
            df["impairment_of_goodwill_1fy"].fillna(0).abs()
            + df["asset_writedown_1fy"].fillna(0).abs()
            + df["restructuring_charges_1fy"].fillna(0).abs()
        )
        if "total_exceptional_items_ltm" in result.columns:
            result["exceptional_items_trend"] = _safe_div(
                result["total_exceptional_items_ltm"] - exceptional_items_1fy,
                exceptional_items_1fy,
            )

    # Goodwill to total assets ratio (high ratio can be risky)
    if "goodwill_ltm" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["goodwill_ltm"], df["total_assets_ltm"])
        result["goodwill_to_assets_pct"] = ratio * 100
        # Aliases (fractional forms)
        result["goodwill_to_assets"] = ratio

    # Intangibles intensity
    if "intangible_assets" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["intangible_assets"], df["total_assets_ltm"])
        result["intangibles_to_assets_pct"] = ratio * 100
        result["intangible_intensity"] = ratio

    # Goodwill change rate (YoY)
    if "goodwill_ltm" in df.columns and "goodwill_1fy" in df.columns:
        result["goodwill_change_rate"] = _safe_div(
            df["goodwill_ltm"] - df["goodwill_1fy"], df["goodwill_1fy"]
        )

    # Restructuring intensity to total assets
    if "restructuring_charges_ltm" in df.columns and "total_assets_ltm" in df.columns:
        result["restructuring_intensity"] = _safe_div(
            df["restructuring_charges_ltm"], df["total_assets_ltm"]
        )

    # Accounting quality score (lower is better, 0-100 scale)
    # High exceptional items, high goodwill, presence of impairments = lower quality
    quality_components = []
    if "has_goodwill_impairment" in result.columns:
        quality_components.append(result["has_goodwill_impairment"] * 30)  # Major red flag
    if "has_asset_writedown" in result.columns:
        quality_components.append(result["has_asset_writedown"] * 20)
    if "has_restructuring" in result.columns:
        quality_components.append(result["has_restructuring"] * 15)
    if "goodwill_to_assets_pct" in result.columns:
        # Penalize if goodwill > 20% of assets
        quality_components.append((result["goodwill_to_assets_pct"] > 20).astype(int) * 20)

    if quality_components:
        total_penalties = sum(quality_components)
        result["accounting_quality_score"] = (100 - total_penalties).clip(lower=0, upper=100)

    logger.info("Engineered accounting quality features")
    return result


def engineer_financial_distress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer financial distress indicators using Altman Z-Score variants.

    Features:
    - altman_z_trend: FY vs 1FY (fallback to FQ vs LTM if FY/1FY missing)
    - distress_risk_score: Composite 0–100 using available z-scores (higher = healthier)
    - z_score_volatility: Std deviation across available z-score periods (FQ, FY, LTM)

    Notes:
    - For Financials sector (sector == 'Financials'), returns NaN for all features due to sector-specific model caveats.
    - Missing inputs yield NaNs; function is robust to absent columns.
    """
    result = df.copy()

    # Mask out Financials sector for these metrics
    fin_mask = (
        (result.get("sector").astype(str).str.lower() == "financials")
        if "sector" in result.columns
        else pd.Series(False, index=result.index)
    )

    # Columns for different periods
    z_fy = result.get("altman_z_score_fy")
    z_1fy = result.get("altman_z_score_1fy")
    z_fq = result.get("altman_z_score_fq")
    z_ltm = result.get("altman_z_score_ltm")

    # Trend: prefer FY vs 1FY; fallback to FQ vs LTM
    trend = pd.Series(np.nan, index=result.index, dtype=float)
    if z_fy is not None and z_1fy is not None:
        trend = z_fy.astype(float) - z_1fy.astype(float)
    elif z_fq is not None and z_ltm is not None:
        trend = z_fq.astype(float) - z_ltm.astype(float)
    result["altman_z_trend"] = trend

    # Volatility across available periods
    z_stack = []
    for s in (z_fq, z_fy, z_ltm):
        if s is not None:
            z_stack.append(s.astype(float))
    if z_stack:
        z_mat = np.vstack([s.to_numpy(copy=False) for s in z_stack]).astype(float)
        # Std dev across rows (axis=0)
        z_vol = np.nanstd(z_mat, axis=0)
        result["z_score_volatility"] = pd.Series(z_vol, index=result.index)
    else:
        result["z_score_volatility"] = np.nan

    # Distress risk score: map average z-score to 0–100
    # Use simple clipping: z<=1.8 -> 0; z>=3.0 -> 100; linear in between
    z_components = []
    for s in (z_fq, z_fy, z_ltm):
        if s is not None:
            z_components.append(s.astype(float))
    if z_components:
        z_mean = pd.concat(z_components, axis=1).mean(axis=1)
        score = ((z_mean - 1.8) / (3.0 - 1.8) * 100.0).clip(lower=0.0, upper=100.0)
        result["distress_risk_score"] = score
    else:
        result["distress_risk_score"] = np.nan

    # Apply Financials mask -> NaN
    if fin_mask.any():
        for col in ["altman_z_trend", "z_score_volatility", "distress_risk_score"]:
            result.loc[fin_mask, col] = np.nan

    logger.info("Engineered financial distress features (Altman Z-Score)")
    return result


def engineer_cash_flow_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer cash flow quality and conversion metrics.

    Features computed (added when inputs exist):
    - cfo_to_net_income: CFO / Net Income (accruals quality)
    - fcf_to_net_income: FCF / Net Income
    - fcf_margin: FCF / Total Revenues (LTM)
    - cfo_growth_yoy: (CFO_LTM - CFO_1FY) / CFO_1FY
    - fcf_stability: Std deviation of available FCF periods (ltm, fy, 1fy)

    Notes:
    - Uses normalized column names (e.g., cfo_ltm, net_income_ltm, fcf_ltm, total_revenues_1fy)
    - Safe divisions via _safe_div; returns NaNs when inputs missing.
    """
    result = df.copy()

    # Core ratios
    if "cfo_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["cfo_to_net_income"] = _safe_div(df["cfo_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["fcf_to_net_income"] = _safe_div(df["fcf_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "total_revenues_1fy" in df.columns:
        result["fcf_margin"] = _safe_div(df["fcf_ltm"], df["total_revenues_1fy"])

    # Growth YoY
    if "cfo_ltm" in df.columns and "cfo_1fy" in df.columns:
        result["cfo_growth_yoy"] = _safe_div(df["cfo_ltm"] - df["cfo_1fy"], df["cfo_1fy"])

    # Stability of FCF across available periods
    fcf_cols = [c for c in ("fcf_ltm", "fcf_fy", "fcf_1fy") if c in df.columns]
    if fcf_cols:
        fcf_mat = pd.concat([df[c].astype(float) for c in fcf_cols], axis=1)
        result["fcf_stability"] = fcf_mat.std(axis=1, ddof=0)

    logger.info("Engineered cash flow quality features")
    return result


def engineer_capital_allocation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer capital allocation efficiency and working capital metrics.

    Features computed (added when inputs exist):
    - capex_intensity: CapEx / Total Revenues (LTM)
    - capex_to_depreciation: CapEx / Depreciation & Amortization (LTM)
    - capex_growth_rate: (CapEx_LTM - CapEx_1FY) / CapEx_1FY
    - capex_volatility: Coefficient of variation of CapEx across periods (if enough data)
    - total_shareholder_return_yield: Dividend Yield + Buyback Yield (percent units)
    - payout_ratio: (Dividends + Share Repurchases) / Net Income (LTM)
    - reinvestment_rate: (CapEx + Cash Acquisitions) / CFO (LTM)
    - acquisition_intensity: Cash Acquisitions / Total Assets (LTM)
    - working_capital_efficiency: Revenues / Working Capital (LTM)
    - working_capital_trend: (WC_LTM - WC_1FY) / Revenues_LTM
    """
    result = df.copy()

    # Capital intensity & efficiency
    if "capital_expenditure_ltm" in df.columns and "total_revenues_1fy" in df.columns:
        result["capex_intensity"] = _safe_div(
            df["capital_expenditure_ltm"], df["total_revenues_1fy"]
        )
    if "capital_expenditure_ltm" in df.columns and "depreciation_amortization_ltm" in df.columns:
        result["capex_to_depreciation"] = _safe_div(
            df["capital_expenditure_ltm"], df["depreciation_amortization_ltm"]
        )
    if "capital_expenditure_ltm" in df.columns and "capital_expenditure_1fy" in df.columns:
        result["capex_growth_rate"] = _safe_div(
            df["capital_expenditure_ltm"] - df["capital_expenditure_1fy"],
            df["capital_expenditure_1fy"],
        )

    # CapEx volatility (coefficient of variation if at least 2 periods)
    capex_cols = [
        c
        for c in (
            "capital_expenditure_ltm",
            "capital_expenditure_fy",
            "capital_expenditure_1fy",
        )
        if c in df.columns
    ]
    if len(capex_cols) >= 2:
        capex_mat = pd.concat([df[c].astype(float) for c in capex_cols], axis=1)
        mean = capex_mat.mean(axis=1)
        std = capex_mat.std(axis=1, ddof=0)
        result["capex_volatility"] = _safe_div(std, mean)

    # Shareholder yield (percent inputs expected)
    if "div_yield_ltm" in df.columns and "buyback_yield_ltm" in df.columns:
        result["total_shareholder_return_yield"] = df["div_yield_ltm"].astype(float).fillna(0) + df[
            "buyback_yield_ltm"
        ].astype(float).fillna(0)

    # Payout ratio and reinvestment
    if all(
        c in df.columns for c in ["dividends_paid_ltm", "share_repurchases_ltm", "net_income_ltm"]
    ):
        payout = df["dividends_paid_ltm"].fillna(0) + df["share_repurchases_ltm"].fillna(0)
        result["payout_ratio"] = _safe_div(payout, df["net_income_ltm"].abs())
    if all(
        c in df.columns for c in ["capital_expenditure_ltm", "cash_acquisitions_ltm", "cfo_ltm"]
    ):
        reinvest = df["capital_expenditure_ltm"].fillna(0) + df["cash_acquisitions_ltm"].fillna(0)
        result["reinvestment_rate"] = _safe_div(reinvest, df["cfo_ltm"].abs())

    # Acquisition intensity
    if "cash_acquisitions_ltm" in df.columns and "total_assets_ltm" in df.columns:
        result["acquisition_intensity"] = _safe_div(
            df["cash_acquisitions_ltm"], df["total_assets_ltm"]
        )

    # Working capital metrics
    if "total_revenues_1fy" in df.columns and "working_capital_ltm" in df.columns:
        result["working_capital_efficiency"] = _safe_div(
            df["total_revenues_1fy"], df["working_capital_ltm"]
        )
    if all(
        c in df.columns
        for c in ["working_capital_ltm", "working_capital_1fy", "total_revenues_1fy"]
    ):
        result["working_capital_trend"] = _safe_div(
            df["working_capital_ltm"] - df["working_capital_1fy"],
            df["total_revenues_1fy"],
        )

    logger.info("Engineered capital allocation & working capital features")
    return result


def engineer_employee_productivity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer employee productivity and efficiency features.

    Features computed:
    - Revenue per employee
    - Profit per employee
    - Assets per employee
    - EBITDA per employee
    - Employee growth trends (1Y, 2Y, 3Y using Full Time Employees data)
    - Workforce volatility metrics
    - Employee CAGR (compound annual growth rate)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with employee productivity features added
    """
    result = df.copy()

    # Check for employee data - prefer full_time_employees, fallback to avg_employees
    employee_col = None
    for col in [
        "full_time_employees_fy",
        "full_time_employees_fq",
        "full_time_employees_fq",
        "full_time_employees_1fy",
        "employees",
    ]:
        if col in df.columns:
            employee_col = col
            break

    if employee_col is None:
        logger.warning("No employee data found, skipping employee productivity features")
        return result

    employees = df[employee_col]

    # Revenue per employee
    if "total_revenues_1fy" in df.columns:
        result["revenue_per_employee"] = _safe_div(df["total_revenues_1fy"], employees)

    # Profit per employee
    if "net_income_ltm" in df.columns:
        result["profit_per_employee"] = _safe_div(df["net_income_ltm"], employees)

    # Assets per employee (capital intensity)
    if "total_assets_ltm" in df.columns:
        result["assets_per_employee"] = _safe_div(df["total_assets_ltm"], employees)

    # EBITDA per employee
    if "ebitda_ltm" in df.columns:
        result["ebitda_per_employee"] = _safe_div(df["ebitda_ltm"], employees)

    # Operating income per employee
    if "operating_income_ltm" in df.columns:
        result["operating_income_per_employee"] = _safe_div(df["operating_income_ltm"], employees)

    # =========================================================================
    # Full Time Employees Growth Trends (using historical FY data)
    # =========================================================================

    # 1-Year employee growth using Full Time Employees
    if "full_time_employees_fy" in df.columns and "full_time_employees_1fy" in df.columns:
        result["fte_growth_1y_pct"] = (
            _safe_div(
                (df["full_time_employees_fy"] - df["full_time_employees_1fy"]),
                df["full_time_employees_1fy"],
            )
            * 100
        )

    # 2-Year employee growth using Full Time Employees
    if "full_time_employees_fy" in df.columns and "full_time_employees_2fy" in df.columns:
        result["fte_growth_2y_pct"] = (
            _safe_div(
                (df["full_time_employees_fy"] - df["full_time_employees_2fy"]),
                df["full_time_employees_2fy"],
            )
            * 100
        )

    # 3-Year employee growth using Full Time Employees
    if "full_time_employees_fy" in df.columns and "full_time_employees_3fy" in df.columns:
        result["fte_growth_3y_pct"] = (
            _safe_div(
                (df["full_time_employees_fy"] - df["full_time_employees_3fy"]),
                df["full_time_employees_3fy"],
            )
            * 100
        )

    # 3-Year employee CAGR (Compound Annual Growth Rate)
    if "full_time_employees_fy" in df.columns and "full_time_employees_3fy" in df.columns:
        # CAGR = (end/start)^(1/n) - 1
        fte_fy = df["full_time_employees_fy"].astype(float)
        fte_3fy = df["full_time_employees_3fy"].astype(float)
        # Only compute where both values are positive
        valid_mask = (fte_fy > 0) & (fte_3fy > 0)
        cagr = pd.Series(np.nan, index=df.index)
        cagr[valid_mask] = (np.power(fte_fy[valid_mask] / fte_3fy[valid_mask], 1 / 3) - 1) * 100
        result["fte_cagr_3y_pct"] = cagr

    # =========================================================================
    # Workforce Volatility Metrics
    # =========================================================================

    # Workforce volatility (std dev of year-over-year changes)
    fte_cols = [
        "full_time_employees_fy",
        "full_time_employees_1fy",
        "full_time_employees_2fy",
        "full_time_employees_3fy",
    ]
    available_fte_cols = [c for c in fte_cols if c in df.columns]

    if len(available_fte_cols) >= 3:
        # Compute YoY changes
        fte_data = df[available_fte_cols].astype(float)
        yoy_changes = []
        for i in range(len(available_fte_cols) - 1):
            curr_col = available_fte_cols[i]
            prev_col = available_fte_cols[i + 1]
            yoy_pct = _safe_div((fte_data[curr_col] - fte_data[prev_col]), fte_data[prev_col]) * 100
            yoy_changes.append(yoy_pct)

        if yoy_changes:
            yoy_df = pd.concat(yoy_changes, axis=1)
            result["fte_volatility"] = yoy_df.std(axis=1, skipna=True)

    # Quarterly vs Annual employee comparison (hiring momentum)
    if "full_time_employees_fq" in df.columns and "full_time_employees_fy" in df.columns:
        result["fte_quarterly_momentum"] = (
            _safe_div(
                (df["full_time_employees_fq"] - df["full_time_employees_fy"]),
                df["full_time_employees_fy"],
            )
            * 100
        )

    # =========================================================================
    # Legacy employee growth (using avg_employees for backward compatibility)
    # =========================================================================

    # Employee growth (if historical data available)
    if "full_time_employees_fq" in df.columns and "full_time_employees_1fy" in df.columns:
        result["employee_growth_yoy_pct"] = (
            _safe_div(
                (df["full_time_employees_fq"] - df["full_time_employees_1fy"]),
                df["full_time_employees_1fy"],
            )
            * 100
        )

    # Productivity trend (revenue per employee vs 5Y average)
    if (
        "revenue_per_employee" in result.columns
        and "total_revenues_5yavgfq" in df.columns
        and "avg_employees_5yavgfy" in df.columns
    ):
        avg_5y_rev_per_emp = _safe_div(df["total_revenues_5yavgfq"], df["avg_employees_5yavgfy"])
        result["revenue_per_employee_vs_5y_pct"] = (
            _safe_div(
                (result["revenue_per_employee"] - avg_5y_rev_per_emp),
                avg_5y_rev_per_emp,
            )
            * 100
        )

    logger.info("Engineered employee productivity features (including FTE growth trends)")
    return result


def engineer_margin_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer profitability margin trends and quality metrics.

    Features computed (when inputs exist):
    - ebitda_margin_trend: (ebitda_ltm/total_revenues_1fy) - (ebitda_1fy/total_revenues_1fy)
    - gross_margin_trend: (gross_profit_ltm/total_revenues_1fy) - (gross_profit_fy/revenue_fy)
    - operating_leverage: (%ΔEBIT) / (%ΔRevenue) using ltm vs 1fy
    - margin_stability_5y: optional std of margins if 5Y averages exist (not required by tests)
    - earnings_quality_score: 0–100 from adjustment ratios: 100 - 50*ebitda_adj_ratio - 30*ebit_adj_ratio

    Notes:
    - Uses normalized columns where available; falls back gracefully if missing.
    - All divisions go through _safe_div to prevent inf.
    """
    result = df.copy()

    # EBITDA margin trend
    if all(
        c in df.columns
        for c in (
            "ebitda_ltm",
            "total_revenues_1fy",
            "ebitda_1fy",
            "total_revenues_1fy",
        )
    ):
        cur = _safe_div(df["ebitda_ltm"].astype(float), df["total_revenues_ltm"].astype(float))
        prev = _safe_div(df["ebitda_fy"].astype(float), df["total_revenues_fy"].astype(float))
        result["ebitda_margin_trend"] = cur - prev

    # Gross margin trend (FY reference for previous)
    if all(
        c in df.columns
        for c in (
            "gross_profit_ltm",
            "total_revenues_1fy",
            "gross_profit_fy",
            "revenue_fy",
        )
    ):
        cur = _safe_div(
            df["gross_profit_ltm"].astype(float), df["total_revenues_ltm"].astype(float)
        )
        prev = _safe_div(df["gross_profit_fy"].astype(float), df["revenue_fy"].astype(float))
        result["gross_margin_trend"] = cur - prev

    # Operating leverage = (%ΔEBIT)/(%ΔRevenue)
    if all(
        c in df.columns
        for c in ("ebit_ltm", "ebit_1fy", "total_revenues_1fy", "total_revenues_1fy")
    ):
        delta_ebit = _safe_div(
            df["ebit_fy"].astype(float) - df["ebit_1fy"].astype(float),
            df["ebit_1fy"].astype(float),
        )
        delta_rev = _safe_div(
            df["total_revenues_fy"].astype(float) - df["total_revenues_1fy"].astype(float),
            df["total_revenues_1fy"].astype(float),
        )
        result["operating_leverage"] = _safe_div(delta_ebit, delta_rev)

    # Earnings quality score based on adjustment ratios (compute if ratios not present)
    ebitda_adj_ratio = None
    ebit_adj_ratio = None
    if "ebitda_adjustment_ratio" in df.columns:
        ebitda_adj_ratio = df["ebitda_adjustment_ratio"].astype(float)
    elif all(c in df.columns for c in ("ebitda_adj_ltm", "ebitda_ltm")):
        ebitda_adj_ratio = _safe_div(df["ebitda_adj_ltm"].abs(), df["ebitda_ltm"].abs())

    if "ebit_adjustment_ratio" in df.columns:
        ebit_adj_ratio = df["ebit_adjustment_ratio"].astype(float)
    elif all(c in df.columns for c in ("ebit_adj_ltm", "ebit_ltm")):
        ebit_adj_ratio = _safe_div(df["ebit_adj_ltm"].abs(), df["ebit_ltm"].abs())

    if ebitda_adj_ratio is not None or ebit_adj_ratio is not None:
        a = (
            ebitda_adj_ratio
            if ebitda_adj_ratio is not None
            else pd.Series(np.nan, index=result.index)
        )
        b = ebit_adj_ratio if ebit_adj_ratio is not None else pd.Series(np.nan, index=result.index)
        score = 100.0 - 50.0 * a - 30.0 * b
        result["earnings_quality_score"] = score.clip(lower=0.0, upper=100.0)

    logger.info("Engineered margin trend & profitability quality features")
    return result


def build_comprehensive_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_relative_values: bool = True,
    sector_col: str = "sector",
    preset: Optional[str] = None,
) -> pd.DataFrame:
    """Build feature sets by applying advanced feature engineering functions.

    Supports presets for Phase 9 integration:
    - preset=None or "comprehensive": full pipeline (backward compatible default)
    - preset="momentum": only momentum & technical indicators
    - preset="quality": accounting quality + financial distress (+ analyst quality)

    This orchestrator applies feature groups in sequence for the comprehensive preset:
    1. Valuation ratios
    2. Profitability ratios
    3. Leverage ratios
    4. Liquidity ratios
    5. Efficiency ratios
    6. Growth metrics
    7. Sector-specific features
    8. Analyst quality features
    9. Accounting quality features
    10. Employee productivity features
    11. Temporal features (if date columns available)
    12. Non-linear transforms
    13. Feature interactions (optional)
    14. Relative value features (optional)

    Args:
        df: Input DataFrame with financial data
        include_interactions: Whether to create polynomial/interaction features (default: True)
        include_relative_values: Whether to create sector-relative features (default: True)
        sector_col: Name of sector column (default: "sector")
        preset: Optional preset name {None,"comprehensive","momentum","quality"}

    Returns:
        DataFrame with engineered features

    Example:
        >>> from finance_ml.ml_workflow.features.advanced import build_comprehensive_features
        >>> features_df = build_comprehensive_features(
        ...     raw_data,
        ...     include_interactions=True,
        ...     include_relative_values=True,
        ...     sector_col="sector"
        ... )
    """
    # Handle presets first (momentum/quality). None means comprehensive (BC)
    preset_norm = (
        (preset or "comprehensive").lower()
        if isinstance(preset, str) or preset is None
        else "comprehensive"
    )
    if preset_norm == "momentum":
        result = engineer_momentum_features(df.copy())
        return result.replace([np.inf, -np.inf], np.nan)
    if preset_norm == "quality":
        result = df.copy()
        result = engineer_accounting_quality_features(result)
        result = engineer_financial_distress_features(result)
        result = engineer_analyst_quality_features(result)
        return result.replace([np.inf, -np.inf], np.nan)

    # Default comprehensive path
    result = df.copy()

    # Apply all feature engineering functions in sequence
    result = engineer_valuation_ratios(result)
    result = engineer_profitability_ratios(result)
    # Phase 6: margins trends and leverage dynamics
    result = engineer_margin_trends(result)
    result = engineer_leverage_ratios(result)
    result = engineer_liquidity_ratios(result)
    result = engineer_efficiency_ratios(result)
    result = engineer_growth_metrics(result)
    # Momentum & technical features (Phase 9.3 Week 2)
    result = engineer_momentum_features(result)
    result = engineer_sector_specific_features(result, sector_col=sector_col)
    # Analyst and market sentiment (Phase 5)
    result = engineer_analyst_quality_features(result)
    result = engineer_market_sentiment_features(result)
    # Market microstructure features (time-series price patterns)
    # Note: Requires historical price data; gracefully skips if unavailable
    result = engineer_market_microstructure_features(result)
    # Accounting and distress
    result = engineer_accounting_quality_features(result)
    # Financial distress features (Altman Z trends & composite)
    result = engineer_financial_distress_features(result)
    # Phase 4: Cash flow & capital allocation
    result = engineer_cash_flow_quality_features(result)
    result = engineer_capital_allocation_features(result)
    result = engineer_employee_productivity_features(result)
    # Phase 7: Balance sheet trends
    result = engineer_balance_sheet_trends(result)

    # Phase 9.3 Schema Version 1.3: New feature categories
    result = engineer_technical_analysis_features(result)
    result = engineer_valuation_timeseries_features(result)
    result = engineer_revenue_forecast_features(result)
    result = engineer_dividend_reliability_features(result)
    result = engineer_employment_dynamics_features(result)

    # Temporal features (if any date column exists)
    # Try multiple date columns in priority order
    date_col_candidates = [
        "next_earnings",
        "last_updated",
        "income_statement_report_date",
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
    ]
    date_col = next((c for c in date_col_candidates if c in result.columns), None)
    if date_col:
        result = engineer_temporal_features(result, date_col=date_col)

    # Non-linear transforms on key features
    log_features = ["market_cap", "revenue", "total_assets"]
    log_features = [f for f in log_features if f in result.columns]
    if log_features:
        result = engineer_nonlinear_transforms(result, log_features=log_features)

    # Optional: Create feature interactions
    if include_interactions:
        result = create_feature_interactions(result)

    # Optional: Create relative value features
    if include_relative_values and sector_col in result.columns:
        result = create_relative_value_features(result, sector_col=sector_col)
        # Additional sector-relative interactions (Phase 8)
        result = engineer_sector_relative_interactions(result, sector_col=sector_col)

    # Composite scores (Phase 8) — safe to compute regardless of flags
    result = engineer_composite_scores(result)

    # Final numeric hygiene: replace any infinities with NaN to avoid downstream issues
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(
        f"Built comprehensive features: {len(result.columns)} total features "
        f"({len(result.columns) - len(df.columns)} new features added)"
    )
    return result


def engineer_market_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer market sentiment features from short interest and betas.

    Features computed (when inputs exist):
    - one_day_chg: Pass-through of short_int_pct (already percent units)
    - beta_stability: Population variance (ddof=0) across available betas (beta_1y, beta_2y, beta_5y)
    - systematic_risk_trend: beta_1y - beta_5y (risk profile change)

    Args:
        df: Input DataFrame with normalized column names

    Returns:
        DataFrame with market sentiment features added
    """
    result = df.copy()

    # One-day price change (percent already)
    if "one_day_pct" in df.columns:
        result["one_day_chg"] = df["one_day_pct"].astype(float)

    # Beta metrics
    beta_cols = [c for c in ("beta_1y", "beta_2y", "beta_5y") if c in df.columns]
    if beta_cols:
        beta_mat = df[beta_cols].astype(float)
        # Population variance across the provided beta horizons
        result["beta_stability"] = beta_mat.var(axis=1, ddof=0)

    if "beta_1y" in df.columns and "beta_5y" in df.columns:
        result["systematic_risk_trend"] = df["beta_1y"].astype(float) - df["beta_5y"].astype(float)

    # Short interest ratio
    if "short_interest" in df.columns and "volume_shrs" in df.columns:
        result["short_interest_ratio"] = _safe_div(
            df["short_interest"].astype(float), df["volume_shrs"].astype(float)
        )
    elif "short_interest" in df.columns and "shares_outstanding" in df.columns:
        result["short_interest_ratio"] = _safe_div(
            df["short_interest"].astype(float), df["shares_outstanding"].astype(float)
        )

    logger.info("Engineered market sentiment features (short interest, betas)")
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


def engineer_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer composite scores (quality, value, momentum) and keep within [0,100].

    Composite scores computed:
    - piotroski_f_score: 9-point fundamental strength score (0-9)
    - altman_z_score: Bankruptcy prediction score (higher is better)
    - beneish_m_score: Earnings manipulation detection (< -1.78 unlikely manipulator)
    - composite_quality_score: mean of available {distress_risk_score, accounting_quality_score}
    - momentum_score: normalized return_stability_score scaled to 0-100

    Phase 9.3 Enhancement: Added Piotroski F-Score, Altman Z-Score, Beneish M-Score
    for composite_event label method support.
    """
    result = df.copy()

    # Piotroski F-Score (0-9): 9 binary signals for fundamental strength
    # Profitability (4 points): ROA > 0, CFO > 0, ΔROAdelta > 0, Accruals (CFO > NI)
    # Leverage (3 points): ΔLEV < 0, ΔLiquid > 0, No equity issuance
    # Operating Efficiency (2 points): ΔMargin > 0, ΔTurnover > 0
    f_score_components = []

    # F1: Positive ROA
    if "roa" in df.columns:
        f_score_components.append((df["roa"].fillna(0) > 0).astype(int))

    # F2: Positive Operating Cash Flow
    if "cfo_ltm" in df.columns:
        f_score_components.append((df["cfo_ltm"].fillna(0) > 0).astype(int))

    # F3: Change in ROA (positive)
    if "roa" in df.columns and "roa_previous_year" in df.columns:
        delta_roa = df["roa"].fillna(0) - df["roa_previous_year"].fillna(0)
        f_score_components.append((delta_roa > 0).astype(int))

    # F4: Quality of Earnings (CFO > Net Income)
    if "cfo_ltm" in df.columns and "net_income" in df.columns:
        f_score_components.append(
            (df["cfo_ltm"].fillna(0) > df["net_income"].fillna(0)).astype(int)
        )

    # F5: Decrease in Leverage (Long-term debt ratio)
    if "debt_to_equity" in df.columns:
        if "debt_to_equity_previous_year" in df.columns:
            delta_lev = df["debt_to_equity"].fillna(0) - df["debt_to_equity_previous_year"].fillna(
                0
            )
            f_score_components.append((delta_lev < 0).astype(int))
        else:
            # Fallback: low leverage is good
            f_score_components.append((df["debt_to_equity"].fillna(0) < 1.0).astype(int))

    # F6: Increase in Liquidity (Current Ratio)
    if "current_ratio" in df.columns:
        if "current_ratio_previous_year" in df.columns:
            delta_liq = df["current_ratio"].fillna(0) - df["current_ratio_previous_year"].fillna(0)
            f_score_components.append((delta_liq > 0).astype(int))
        else:
            # Fallback: healthy liquidity
            f_score_components.append((df["current_ratio"].fillna(0) > 1.5).astype(int))

    # F7: No new equity issuance (shares outstanding decreased or stable)
    if "shares_outstanding" in df.columns and "shares_outstanding_previous_year" in df.columns:
        delta_shares = df["shares_outstanding"].fillna(0) - df[
            "shares_outstanding_previous_year"
        ].fillna(0)
        f_score_components.append((delta_shares <= 0).astype(int))

    # F8: Increase in Gross Margin
    if "gross_margin_pct" in df.columns:
        if "gross_margin_pct_previous_year" in df.columns:
            delta_margin = df["gross_margin_pct"].fillna(0) - df[
                "gross_margin_pct_previous_year"
            ].fillna(0)
            f_score_components.append((delta_margin > 0).astype(int))
        else:
            # Fallback: healthy margin
            f_score_components.append((df["gross_margin_pct"].fillna(0) > 30).astype(int))

    # F9: Increase in Asset Turnover
    if "asset_turnover" in df.columns:
        if "asset_turnover_previous_year" in df.columns:
            delta_turn = df["asset_turnover"].fillna(0) - df["asset_turnover_previous_year"].fillna(
                0
            )
            f_score_components.append((delta_turn > 0).astype(int))
        else:
            # Fallback: efficient asset use
            f_score_components.append((df["asset_turnover"].fillna(0) > 0.5).astype(int))

    if f_score_components:
        result["piotroski_f_score"] = pd.concat(f_score_components, axis=1).sum(axis=1)

    # Altman Z-Score: Bankruptcy prediction model
    # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    # X1 = Working Capital / Total Assets
    # X2 = Retained Earnings / Total Assets
    # X3 = EBIT / Total Assets
    # X4 = Market Value of Equity / Total Liabilities
    # X5 = Sales / Total Assets
    # Z > 2.99: Safe zone, 1.81-2.99: Grey zone, < 1.81: Distress zone
    z_components = {}

    if "working_capital" in df.columns and "total_assets" in df.columns:
        z_components["x1"] = _safe_div(df["working_capital"], df["total_assets"]) * 1.2

    if "retained_earnings" in df.columns and "total_assets" in df.columns:
        z_components["x2"] = _safe_div(df["retained_earnings"], df["total_assets"]) * 1.4

    if "ebit" in df.columns and "total_assets" in df.columns:
        z_components["x3"] = _safe_div(df["ebit"], df["total_assets"]) * 3.3

    if "market_cap" in df.columns and "total_liabilities" in df.columns:
        z_components["x4"] = _safe_div(df["market_cap"], df["total_liabilities"]) * 0.6

    if "revenue" in df.columns and "total_assets" in df.columns:
        z_components["x5"] = _safe_div(df["revenue"], df["total_assets"]) * 1.0

    if z_components:
        z_df = pd.DataFrame(z_components).fillna(0)
        result["altman_z_score"] = z_df.sum(axis=1)

    # Beneish M-Score: Earnings manipulation detection
    # M = -4.84 + 0.92*DSRI + 0.528*GMI + 0.404*AQI + 0.892*SGI + 0.115*DEPI
    #     - 0.172*SGAI + 4.679*TATA - 0.327*LVGI
    # M < -1.78: Unlikely manipulator, M > -1.78: Possible manipulator
    # Simplified version using available features
    m_components = []
    m_weights = []

    # DSRI (Days Sales Receivable Index): (AR_t/Sales_t) / (AR_t-1/Sales_t-1)
    if all(
        c in df.columns
        for c in [
            "accounts_receivable",
            "revenue",
            "accounts_receivable_previous_year",
            "revenue_previous_year",
        ]
    ):
        dsr_current = _safe_div(df["accounts_receivable"], df["revenue"])
        dsr_prior = _safe_div(df["accounts_receivable_previous_year"], df["revenue_previous_year"])
        dsri = _safe_div(dsr_current, dsr_prior).fillna(1.0)
        m_components.append(dsri)
        m_weights.append(0.92)

    # GMI (Gross Margin Index): GM_t-1 / GM_t (deteriorating margin = higher GMI = bad)
    if "gross_margin_pct" in df.columns and "gross_margin_pct_previous_year" in df.columns:
        gmi = _safe_div(df["gross_margin_pct_previous_year"], df["gross_margin_pct"]).fillna(1.0)
        m_components.append(gmi)
        m_weights.append(0.528)

    # SGI (Sales Growth Index): Sales_t / Sales_t-1
    if "revenue" in df.columns and "revenue_previous_year" in df.columns:
        sgi = _safe_div(df["revenue"], df["revenue_previous_year"]).fillna(1.0)
        m_components.append(sgi)
        m_weights.append(0.892)

    # LVGI (Leverage Index): Leverage_t / Leverage_t-1
    if "debt_to_equity" in df.columns and "debt_to_equity_previous_year" in df.columns:
        lvgi = _safe_div(df["debt_to_equity"], df["debt_to_equity_previous_year"]).fillna(1.0)
        m_components.append(lvgi)
        m_weights.append(-0.327)

    if m_components:
        m_score = pd.Series(-4.84, index=df.index)
        for comp, weight in zip(m_components, m_weights):
            m_score += comp * weight
        result["beneish_m_score"] = m_score

    # Original composite_quality_score
    components = []
    if "distress_risk_score" in df.columns:
        components.append(df["distress_risk_score"].astype(float))
    if "accounting_quality_score" in df.columns:
        components.append(df["accounting_quality_score"].astype(float))
    if components:
        comp = pd.concat(components, axis=1).mean(axis=1)
        result["composite_quality_score"] = comp.clip(lower=0.0, upper=100.0)

    # Simple momentum score (0-100) if available: normalize return_stability_score to 0-100 by 2*atan scaling
    if "return_stability_score" in df.columns:
        rss = df["return_stability_score"].astype(float)
        # map real line to (0,100) via arctan, center at 50
        result["momentum_score"] = (np.arctan(rss) / (np.pi / 2) * 50.0 + 50.0).clip(0.0, 100.0)

    logger.info("Engineered composite scores (Piotroski F-Score, Altman Z-Score, Beneish M-Score)")
    return result


def engineer_sector_relative_interactions(
    df: pd.DataFrame, sector_col: str = "sector"
) -> pd.DataFrame:
    """Create sector-relative interaction features for key metrics.

    For each metric present among a small default set, compute:
    - metric_vs_sector_median (if not already present)
    - metric_vs_sector_top_quartile (metric - 75th percentile by sector)
    """
    result = df.copy()
    if sector_col not in df.columns:
        return result

    metrics = [
        m for m in ("p_e_ratio", "roe", "net_margin_pct", "ev_ebitda_ratio") if m in df.columns
    ]
    if not metrics:
        return result

    grouped = df.groupby(sector_col)
    for m in metrics:
        if f"{m}_vs_sector_median" not in result.columns:
            sector_median = grouped[m].transform("median")
            result[f"{m}_vs_sector_median"] = df[m] - sector_median
        sector_q3 = grouped[m].transform(lambda s: s.quantile(0.75))
        result[f"{m}_vs_sector_top_quartile"] = df[m] - sector_q3

    logger.info("Engineered sector-relative interaction features")
    return result


def engineer_technical_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer technical analysis features using EMA and 52-week data.

    Phase 9.3 Schema Version 1.3: Leverages new technical indicator columns
    (EMA 20D/50D/100D/250D, 52W High/Low, Rel. Volume).

    Features created:
    - EMA crossover signals (20D/50D, 50D/250D)
    - Price vs EMA deviations
    - EMA slope and trend consistency
    - 52-week range position indicators
    - Volume momentum composite

    Args:
        df: Input DataFrame with technical indicator columns

    Returns:
        DataFrame with technical analysis features added

    Example:
        >>> df_tech = engineer_technical_analysis_features(stocks_df)
        >>> print(df_tech[['ema_crossover_20_50', 'pct_off_52w_high']].head())
    """
    result = df.copy()

    # 1. EMA-Based Signals
    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        # EMA crossover: 1 if 20D > 50D (bullish), -1 if 20D < 50D (bearish), 0 if equal/missing
        result["ema_crossover_20_50"] = np.where(
            df["ema_20d"] > df["ema_50d"],
            1,
            np.where(df["ema_20d"] < df["ema_50d"], -1, 0),
        )

    if "ema_50d" in df.columns and "ema_250d" in df.columns:
        result["ema_crossover_50_250"] = np.where(
            df["ema_50d"] > df["ema_250d"],
            1,
            np.where(df["ema_50d"] < df["ema_250d"], -1, 0),
        )

    if "last_price" in df.columns and "ema_20d" in df.columns:
        result["price_vs_ema_20d"] = _safe_div(df["last_price"] - df["ema_20d"], df["ema_20d"])

    if "last_price" in df.columns and "ema_250d" in df.columns:
        result["price_vs_ema_250d"] = _safe_div(df["last_price"] - df["ema_250d"], df["ema_250d"])

    # EMA slope (approximate using 20D vs 50D as proxy for slope)
    if "ema_20d" in df.columns and "ema_50d" in df.columns:
        result["ema_slope_20d"] = _safe_div(df["ema_20d"] - df["ema_50d"], df["ema_50d"])

    # EMA trend consistency: check if EMAs are aligned (all ascending or descending)
    if all(c in df.columns for c in ["ema_20d", "ema_50d", "ema_100d", "ema_250d"]):
        bullish = (
            (df["ema_20d"] > df["ema_50d"])
            & (df["ema_50d"] > df["ema_100d"])
            & (df["ema_100d"] > df["ema_250d"])
        )
        bearish = (
            (df["ema_20d"] < df["ema_50d"])
            & (df["ema_50d"] < df["ema_100d"])
            & (df["ema_100d"] < df["ema_250d"])
        )
        result["ema_trend_consistency"] = np.where(bullish, 1, np.where(bearish, -1, 0))

    # 2. 52-Week Position Features
    if "52w_high_adj" in df.columns and "last_price" in df.columns:
        result["pct_off_52w_high"] = _safe_div(
            df["52w_high_adj"] - df["last_price"], df["52w_high_adj"]
        )

    if "52w_low_adj" in df.columns and "last_price" in df.columns:
        result["pct_above_52w_low"] = _safe_div(
            df["last_price"] - df["52w_low_adj"], df["52w_low_adj"]
        )

    if all(c in df.columns for c in ["52w_high_adj", "52w_low_adj", "last_price"]):
        # 52W range position: 0 at low, 1 at high
        range_width = df["52w_high_adj"] - df["52w_low_adj"]
        result["52w_range_position"] = _safe_div(
            df["last_price"] - df["52w_low_adj"], range_width
        ).clip(0, 1)

        # Near 52W high/low flags
        result["near_52w_high_flag"] = (result["pct_off_52w_high"] <= 0.05).astype(int)
        result["near_52w_low_flag"] = (result["pct_above_52w_low"] <= 0.05).astype(int)

    # 3. Volume & Momentum Composite
    if "rel_volume" in df.columns and "price_chg_pct_1m" in df.columns:
        result["volume_momentum_score"] = df["rel_volume"] * df["price_chg_pct_1m"]

    # Breakout signal: EMA crossover + near 52W high
    if "ema_crossover_20_50" in result.columns and "near_52w_high_flag" in result.columns:
        result["breakout_signal"] = (
            (result["ema_crossover_20_50"] == 1) & (result["near_52w_high_flag"] == 1)
        ).astype(int)

    logger.info("Engineered technical analysis features (Phase 9.3 Schema 1.3)")
    return result


def engineer_valuation_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer valuation time-series features using extended EV/Sales, EV/EBITDA, and P/E data.

    Phase 9.3 Schema Version 1.3: Leverages new valuation multiples time-series columns
    (EV/Sales variants, EV/EBITDA extended, P/E extended history).

    Features created:
    - Valuation momentum indicators (1Y, 3Y trends)
    - Mean reversion metrics (current vs 3Y average)
    - Forward vs trailing valuation spreads
    - Quarterly valuation stability metrics

    Args:
        df: Input DataFrame with valuation time-series columns

    Returns:
        DataFrame with valuation time-series features added

    Example:
        >>> df_val = engineer_valuation_timeseries_features(stocks_df)
        >>> print(df_val[['ev_sales_trend_1y', 'p_e_forward_discount']].head())
    """
    result = df.copy()

    # 1. Valuation Momentum Indicators
    # EV/Sales trend (1Y)
    if "ev_sales_ltm" in df.columns and "ev_sales_1fyltm" in df.columns:
        result["ev_sales_trend_1y"] = _safe_div(
            df["ev_sales_ltm"] - df["ev_sales_1fyltm"], df["ev_sales_1fyltm"]
        )

    # EV/Sales trend (3Y) - using 3-year lookback
    if all(
        c in df.columns
        for c in [
            "ev_sales_ltm",
            "ev_sales_1fyltm",
            "ev_sales_2fyltm",
            "ev_sales_3fyltm",
        ]
    ):
        # Linear trend slope approximation
        y_vals = pd.concat(
            [
                df["ev_sales_3fyltm"],
                df["ev_sales_2fyltm"],
                df["ev_sales_1fyltm"],
                df["ev_sales_ltm"],
            ],
            axis=1,
        )
        result["ev_sales_trend_3y"] = y_vals.apply(
            lambda row: (
                (row.iloc[-1] - row.iloc[0]) / (row.iloc[0] + 1e-9)
                if row.notna().sum() >= 2
                else np.nan
            ),
            axis=1,
        )

    # EV/EBITDA momentum
    if "ev_ebitda_ltm" in df.columns and "ev_ebitda_1fyltm" in df.columns:
        result["ev_ebitda_momentum"] = _safe_div(
            df["ev_ebitda_ltm"] - df["ev_ebitda_1fyltm"], df["ev_ebitda_1fyltm"]
        )

    # P/E momentum (YoY)
    if "p_e_ltm" in df.columns and "p_e_1fyltm" in df.columns:
        result["p_e_momentum_yoy"] = _safe_div(df["p_e_ltm"] - df["p_e_1fyltm"], df["p_e_1fyltm"])

    # P/E momentum (QoQ)
    if "p_e_ltm" in df.columns and "p_e_1fqltm" in df.columns:
        result["p_e_momentum_qoq"] = _safe_div(df["p_e_ltm"] - df["p_e_1fqltm"], df["p_e_1fqltm"])

    # 2. Valuation Mean Reversion Features
    if "ev_sales_ltm" in df.columns and "ev_sales_3yavgltm" in df.columns:
        # Z-score: (current - mean) / std, approximated using deviation from 3Y avg
        deviation = df["ev_sales_ltm"] - df["ev_sales_3yavgltm"]
        result["ev_sales_vs_3y_avg"] = _safe_div(deviation, df["ev_sales_3yavgltm"])

    if "ev_ebitda_ltm" in df.columns and "ev_ebitda_3yavgltm" in df.columns:
        deviation = df["ev_ebitda_ltm"] - df["ev_ebitda_3yavgltm"]
        result["ev_ebitda_vs_3y_avg"] = _safe_div(deviation, df["ev_ebitda_3yavgltm"])

    if "p_e_ltm" in df.columns and "p_e_3yavgltm" in df.columns:
        deviation = df["p_e_ltm"] - df["p_e_3yavgltm"]
        result["p_e_vs_3y_avg"] = _safe_div(deviation, df["p_e_3yavgltm"])

    # Valuation extreme flag (>2 std dev from mean, approximated as >200% deviation)
    if "ev_sales_vs_3y_avg" in result.columns:
        result["valuation_extreme_flag"] = (result["ev_sales_vs_3y_avg"].abs() > 2.0).astype(int)

    # 3. Forward vs Trailing Valuation
    if "ev_sales_ntm" in df.columns and "ev_sales_ltm" in df.columns:
        result["ev_sales_forward_discount"] = _safe_div(
            df["ev_sales_ntm"] - df["ev_sales_ltm"], df["ev_sales_ltm"]
        )

    if "ev_ebitda_ntm" in df.columns and "ev_ebitda_ltm" in df.columns:
        result["ev_ebitda_forward_discount"] = _safe_div(
            df["ev_ebitda_ntm"] - df["ev_ebitda_ltm"], df["ev_ebitda_ltm"]
        )

    if "p_e_est_fy1" in df.columns and "p_e_ltm" in df.columns:
        result["p_e_forward_discount"] = _safe_div(df["p_e_est_fy1"] - df["p_e_ltm"], df["p_e_ltm"])

    # Growth implied by valuation (forward discount as proxy for growth expectations)
    if "ev_sales_forward_discount" in result.columns:
        # Negative discount implies growth expectations
        result["growth_implied_by_valuation"] = -result["ev_sales_forward_discount"]

    # 4. Quarterly Valuation Stability
    if all(
        c in df.columns
        for c in [
            "ev_sales_1fqltm",
            "ev_sales_2fqltm",
            "ev_sales_3fqltm",
            "ev_sales_4fqltm",
        ]
    ):
        quarterly_vals = pd.concat(
            [
                df["ev_sales_1fqltm"],
                df["ev_sales_2fqltm"],
                df["ev_sales_3fqltm"],
                df["ev_sales_4fqltm"],
            ],
            axis=1,
        )
        result["ev_sales_quarterly_volatility"] = quarterly_vals.std(axis=1)
        result["valuation_stability_score"] = _safe_div(
            1.0, result["ev_sales_quarterly_volatility"] + 0.01
        )

        # Valuation trend consistency (monotonicity check across quarters)
        def check_monotonicity(row):
            vals = row.dropna().values
            if len(vals) < 2:
                return 0
            increasing = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
            decreasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            return 1 if increasing else (-1 if decreasing else 0)

        result["valuation_trend_consistency"] = quarterly_vals.apply(check_monotonicity, axis=1)

    logger.info("Engineered valuation time-series features (Phase 9.3 Schema 1.3)")
    return result


def engineer_revenue_forecast_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer revenue forecasting and analyst consensus features.

    Phase 9.3 Schema Version 1.3: Leverages new revenue estimate columns
    (Revenues Est Avg/Med for NTM and FY1E).

    Features created:
    - Analyst consensus metrics (estimate spreads, disagreement)
    - Forward revenue expectations (implied growth rates)
    - Estimate quality indicators (avg vs median bias, confidence flags)

    Args:
        df: Input DataFrame with revenue estimate columns

    Returns:
        DataFrame with revenue forecast features added

    Example:
        >>> df_rev = engineer_revenue_forecast_features(stocks_df)
        >>> print(df_rev[['revenue_estimate_spread_ntm', 'revenue_growth_implied_fy1e']].head())
    """
    result = df.copy()

    # 1. Analyst Consensus Metrics
    # Revenue estimate spread NTM (disagreement indicator)
    if "revenues_est_avg_ntm" in df.columns and "revenues_est_med_ntm" in df.columns:
        result["revenue_estimate_spread_ntm"] = _safe_div(
            df["revenues_est_avg_ntm"] - df["revenues_est_med_ntm"],
            df["revenues_est_med_ntm"],
        )

    # Revenue estimate spread FY1E
    if "revenues_est_avg_fy1e" in df.columns and "revenues_est_med_fy1e" in df.columns:
        result["revenue_estimate_spread_fy1e"] = _safe_div(
            df["revenues_est_avg_fy1e"] - df["revenues_est_med_fy1e"],
            df["revenues_est_med_fy1e"],
        )

    # Revenue consensus uncertainty score (composite of both spreads)
    if (
        "revenue_estimate_spread_ntm" in result.columns
        and "revenue_estimate_spread_fy1e" in result.columns
    ):
        result["revenue_consensus_uncertainty_score"] = (
            result["revenue_estimate_spread_ntm"].abs()
            + result["revenue_estimate_spread_fy1e"].abs()
        ) / 2.0

    # 2. Forward Revenue Expectations
    # Revenue growth implied NTM
    if "revenues_est_avg_ntm" in df.columns and "total_revenues_1fy" in df.columns:
        result["revenue_growth_implied_ntm"] = _safe_div(
            df["revenues_est_avg_ntm"] - df["total_revenues_1fy"],
            df["total_revenues_1fy"],
        )

    # Revenue growth implied FY1E
    if "revenues_est_avg_fy1e" in df.columns and "total_revenues_1fy" in df.columns:
        result["revenue_growth_implied_fy1e"] = _safe_div(
            df["revenues_est_avg_fy1e"] - df["total_revenues_1fy"],
            df["total_revenues_1fy"],
        )

    # Revenue growth acceleration (FY1E growth vs historical CAGR)
    if (
        "revenue_growth_implied_fy1e" in result.columns
        and "total_revenues_cagr_5y_fy" in df.columns
    ):
        result["revenue_growth_acceleration"] = (
            result["revenue_growth_implied_fy1e"] - df["total_revenues_cagr_5y_fy"]
        )

    # 3. Estimate Quality Indicators
    # Avg vs median bias (systematic difference between average and median)
    if "revenue_estimate_spread_ntm" in result.columns:
        result["avg_vs_median_bias"] = result["revenue_estimate_spread_ntm"]

    # Estimate confidence flag (low spread = high confidence)
    # Threshold: spread < 5% indicates high confidence
    if "revenue_consensus_uncertainty_score" in result.columns:
        result["estimate_confidence_flag"] = (
            result["revenue_consensus_uncertainty_score"] < 0.05
        ).astype(int)

    # Growth surprise potential (gap between estimate and trend)
    if (
        "revenue_growth_implied_fy1e" in result.columns
        and "total_revenues_cagr_5y_fy" in df.columns
    ):
        result["growth_surprise_potential"] = (
            result["revenue_growth_implied_fy1e"] - df["total_revenues_cagr_5y_fy"]
        ).abs()

    logger.info("Engineered revenue forecast features (Phase 9.3 Schema 1.3)")
    return result


def engineer_dividend_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer dividend reliability and income stock features.

    Phase 9.3 Schema Version 1.3: Leverages new dividend record columns
    (Dividend Record dates, frequency, currency, amount, streak).

    Features created:
    - Dividend consistency metrics (streak years, consistency score, income stock flag)
    - Dividend coverage & safety (payout ratio, FCF coverage, safety score)
    - Dividend growth features (growth trend, yield vs sector, aristocrat flag)
    - Dividend event features (days since ex-date, frequency encoding, currency risk)

    Args:
        df: Input DataFrame with dividend record columns

    Returns:
        DataFrame with dividend reliability features added

    Example:
        >>> df_div = engineer_dividend_reliability_features(stocks_df)
        >>> print(df_div[['dividend_consistency_score', 'dividend_safety_score']].head())
    """
    result = df.copy()

    # 1. Dividend Consistency Metrics
    # Dividend streak years (already numeric in schema)
    if "dividend_streak" in df.columns:
        result["dividend_streak_years"] = df["dividend_streak"]

        # Dividend consistency score (weighted: streak + frequency)
        # Assuming quarterly frequency is ideal (4 payments/year)
        if "dividend_record_frequency" in df.columns:
            freq_map = {"quarterly": 4, "semi-annual": 2, "annual": 1, "monthly": 12}
            freq_encoded = df["dividend_record_frequency"].map(freq_map).fillna(1)
            # Score: (streak / 10) * 0.7 + (freq / 4) * 0.3, normalized to 0-100
            result["dividend_consistency_score"] = (df["dividend_streak"] / 10.0).clip(
                0, 1
            ) * 70 + (freq_encoded / 4.0).clip(0, 1) * 30
        else:
            # Without frequency, just use streak
            result["dividend_consistency_score"] = (df["dividend_streak"] / 10.0).clip(0, 1) * 100

        # Income stock flag (reliable dividend payers: streak > 5 years)
        result["income_stock_flag"] = (df["dividend_streak"] > 5).astype(int)

    # 2. Dividend Coverage & Safety
    # Dividend payout ratio (Dividend Amount / EPS)
    if "dividend_per_share" in df.columns and "eps_adj_ltm" in df.columns:
        result["dividend_payout_ratio"] = _safe_div(df["dividend_per_share"], df["eps_adj_ltm"])

    # FCF dividend coverage (FCF LTM / Total Dividends Paid)
    if "fcf_ltm" in df.columns and "common_dividends_paid_ltm" in df.columns:
        result["fcf_dividend_coverage"] = _safe_div(df["fcf_ltm"], df["common_dividends_paid_ltm"])

    # Dividend safety score (composite coverage metric)
    # Safe if payout ratio < 0.8 and FCF coverage > 1.2
    if "dividend_payout_ratio" in result.columns and "fcf_dividend_coverage" in result.columns:
        payout_safe = (result["dividend_payout_ratio"] < 0.8) | result[
            "dividend_payout_ratio"
        ].isna()
        fcf_safe = (result["fcf_dividend_coverage"] > 1.2) | result["fcf_dividend_coverage"].isna()
        result["dividend_safety_score"] = payout_safe.astype(int) * 50 + fcf_safe.astype(int) * 50

    # 3. Dividend Growth Features
    # Dividend growth trend (change in Dividend Per Share LTM vs historical)
    if "dividend_per_share_ltm" in df.columns:
        # Approximate growth using current vs lagged values (if available)
        # For now, create a placeholder for future enhancement
        result["dividend_growth_trend"] = 0.0  # Placeholder

    # Dividend yield vs sector (sector-relative ranking)
    if "div_yield_ltm" in result.columns and "sector" in result.columns:
        sector_median = result.groupby("sector")["div_yield_ltm"].transform("median")
        result["dividend_yield_vs_sector"] = result["div_yield_ltm"] - sector_median

    # Dividend aristocrat flag (long streak + positive growth)
    if "dividend_streak" in df.columns:
        # Aristocrats typically have 25+ years of consecutive increases
        result["dividend_aristocrat_flag"] = (df["dividend_streak"] >= 25).astype(int)

    # 4. Dividend Event Features
    # Days since ex-date (recency of last dividend)
    if "dividend_record_ex_date" in df.columns:
        try:
            ex_dates = pd.to_datetime(df["dividend_record_ex_date"], errors="coerce")
            today = pd.Timestamp.now()
            result["days_since_ex_date"] = (today - ex_dates).dt.days
        except Exception:
            result["days_since_ex_date"] = np.nan

    # Dividend frequency encoded (numerical encoding)
    if "dividend_record_frequency" in df.columns:
        freq_map = {"monthly": 12, "quarterly": 4, "semi-annual": 2, "annual": 1}
        result["dividend_frequency_encoded"] = (
            df["dividend_record_frequency"].map(freq_map).fillna(0)
        )

    # Currency risk flag (non-USD dividend currency)
    if "dividend_record_currency" in df.columns:
        result["currency_risk_flag"] = (
            df["dividend_record_currency"].fillna("USD") != "USD"
        ).astype(int)

    logger.info("Engineered dividend reliability features (Phase 9.3 Schema 1.3)")
    return result


def engineer_employment_dynamics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer employment dynamics and growth signal features.

    Phase 9.3 Schema Version 1.3: Leverages new employee count columns
    (Total Employees FY/FQ) and existing employee averages.

    Features created:
    - Employee growth metrics (YoY, QoQ, 5Y CAGR, acceleration)
    - Productivity & efficiency (revenue/profit per employee, trends)
    - Scale & workforce indicators (large employer flag, volatility, hiring intensity)

    Args:
        df: Input DataFrame with employment columns

    Returns:
        DataFrame with employment dynamics features added

    Example:
        >>> df_emp = engineer_employment_dynamics_features(stocks_df)
        >>> print(df_emp[['employee_growth_yoy', 'revenue_per_employee_fy']].head())
    """
    result = df.copy()

    # 1. Employee Growth Metrics
    # Employee growth YoY (using FY data)
    if "full_time_employees_fy" in df.columns and "full_time_employees_1fy" in df.columns:
        # Approximate prior year using full_time_employees_1fy as proxy
        result["employee_growth_yoy"] = _safe_div(
            df["full_time_employees_fy"] - df["full_time_employees_1fy"],
            df["full_time_employees_1fy"],
        )

    # Employee growth CAGR 5Y
    if "full_time_employees_fy" in df.columns and "avg_employees_5yavgfy" in df.columns:
        # CAGR = (End/Start)^(1/5) - 1
        # Approximate using current vs 5Y avg
        ratio = _safe_div(df["full_time_employees_fy"], df["avg_employees_5yavgfy"])
        result["employee_growth_cagr_5y"] = (ratio**0.2) - 1.0

    # Employee growth acceleration (change in growth rate)
    if "employee_growth_yoy" in result.columns and "employee_growth_cagr_5y" in result.columns:
        result["employee_growth_acceleration"] = (
            result["employee_growth_yoy"] - result["employee_growth_cagr_5y"]
        )

    # 2. Productivity & Efficiency
    # Revenue per employee (FY)
    if "total_revenues_fy" in df.columns and "full_time_employees_fy" in df.columns:
        result["revenue_per_employee_fy"] = _safe_div(
            df["total_revenues_fy"], df["full_time_employees_fy"]
        )

    # Revenue per employee (1FY)
    if "total_revenues_1fy" in df.columns and "full_time_employees_1fy" in df.columns:
        result["revenue_per_employee_1fy"] = _safe_div(
            df["total_revenues_1fy"], df["full_time_employees_1fy"]
        )

    # Revenue per employee trend (YoY change in productivity)
    if "revenue_per_employee_fy" in result.columns and "revenue_per_employee_1fy" in result.columns:
        result["revenue_per_employee_trend"] = _safe_div(
            result["revenue_per_employee_fy"] - result["revenue_per_employee_1fy"],
            result["revenue_per_employee_1fy"],
        )

    # Profit per employee (Net Income / Total Employees)
    if "normalized_net_income_fy" in df.columns and "full_time_employees_fy" in df.columns:
        result["profit_per_employee"] = _safe_div(
            df["normalized_net_income_fy"], df["full_time_employees_fy"]
        )
    elif "normalized_net_income_fq" in df.columns and "full_time_employees_fq" in df.columns:
        result["profit_per_employee"] = _safe_div(
            df["normalized_net_income_fq"], df["full_time_employees_fq"]
        )

    # 3. Scale & Workforce Indicators
    # Large employer flag (>10,000 employees)
    if "full_time_employees_fy" in df.columns:
        result["employee_base_scale_flag"] = (df["full_time_employees_fy"] > 10000).astype(int)

    # Workforce volatility (std dev of employee counts)
    if "full_time_employees_fq" in df.columns and "full_time_employees_fy" in df.columns:
        # Approximate volatility using difference between FQ and LTM avg
        result["workforce_volatility"] = (
            (df["full_time_employees_fq"] - df["full_time_employees_fy"]).abs()
            / df["full_time_employees_fy"]
        ).fillna(0)

    # Hiring intensity score (employee growth relative to sector)
    if "employee_growth_yoy" in result.columns and "sector" in df.columns:
        sector_median_growth = result.groupby("sector")["employee_growth_yoy"].transform("median")
        result["hiring_intensity_score"] = result["employee_growth_yoy"] - sector_median_growth

    logger.info("Engineered employment dynamics features (Phase 9.3 Schema 1.3)")
    return result


def engineer_estimated_vs_actual_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Estimated vs. Actual earnings analytics features.

    Phase 9.3 Enhanced Earnings Analytics: Compares forward estimates against
    actual reported metrics to identify earnings surprises, estimate momentum,
    and analyst forecast accuracy.

    Features created:
    - EPS Surprise %: (eps_actual - eps_estimated) / |eps_estimated| * 100
    - Revenue Surprise %: (revenue_actual - revenue_estimated) / |revenue_estimated| * 100
    - EBITDA Surprise %: (ebitda_actual - ebitda_estimated) / |ebitda_estimated| * 100
    - Earnings Beat Indicator: Boolean flag for positive EPS surprise
    - Surprise Momentum Score: Weighted average of multi-period revisions (1M, 3M, 6M)
    - Surprise Magnitude: Categorical (small/moderate/large) based on surprise %
    - Consensus Uncertainty: Spread between estimate and actual as volatility proxy
    - Estimate Revision Trend: Acceleration in estimate revisions over time

    Input Columns (from COLUMN_SCHEMA):
    - Actuals: eps_adj_ltm, net_eps_basic_ltm, total_revenues_ltm, ebitda_ltm
    - Estimates: eps_norm_est_avg_ntm, eps_norm_est_avg_fy1e, revenues_est_avg_ntm,
                 revenues_est_avg_fy1e, ebitda_est_avg_ntm, ebitda_est_avg_fy1e
    - Revisions: eps_est_avg_rev_pct_fy1e_1m, eps_est_avg_rev_pct_fy1e_3m,
                 eps_est_avg_rev_pct_fy1e_6m, eps_est_avg_rev_pct_fy1e_1y

    Args:
        df: Input DataFrame with EPS, revenue, and estimate columns

    Returns:
        DataFrame with estimated vs. actual analytics features added

    Example:
        >>> df_earnings = engineer_estimated_vs_actual_analytics(stocks_df)
        >>> print(df_earnings[['eps_surprise_pct', 'earnings_beat_indicator']].head())
        >>> # Identify stocks beating estimates consistently
        >>> beats = df_earnings[df_earnings['earnings_beat_indicator'] == True]
    """
    result = df.copy()

    # =========================================================================
    # 1. EPS Surprise Analytics
    # =========================================================================
    # Primary EPS Surprise: Actual vs. Next Twelve Months (NTM) estimate
    actual_eps_cols = ["eps_adj_ltm", "net_eps_basic_ltm", "eps"]
    estimate_eps_cols = ["eps_norm_est_avg_ntm", "eps_norm_est_avg_fy1e"]

    eps_actual = None
    for col in actual_eps_cols:
        if col in df.columns:
            eps_actual = pd.to_numeric(df[col], errors="coerce")
            break

    eps_estimate = None
    for col in estimate_eps_cols:
        if col in df.columns:
            eps_estimate = pd.to_numeric(df[col], errors="coerce")
            break

    if eps_actual is not None and eps_estimate is not None:
        # EPS Surprise Percentage
        result["eps_surprise_pct"] = (
            _safe_div((eps_actual - eps_estimate), eps_estimate.abs()) * 100
        )

        # Earnings Beat Indicator (boolean flag)
        result["earnings_beat_indicator"] = (result["eps_surprise_pct"] > 0).fillna(False)

        # Surprise Magnitude Categorization
        surprise_abs = result["eps_surprise_pct"].abs()
        eps_magnitude_cat = pd.cut(
            surprise_abs,
            bins=[0, 5, 15, float("inf")],
            labels=["small", "moderate", "large"],
            include_lowest=True,
        )
        # Add "unknown" to categories before filling NaN, then convert to category dtype
        eps_magnitude_cat = eps_magnitude_cat.cat.add_categories(["unknown"])
        result["eps_surprise_magnitude"] = eps_magnitude_cat.fillna("unknown").astype("category")

        logger.info(f"Computed EPS surprise for {result['eps_surprise_pct'].notna().sum()} stocks")

    # =========================================================================
    # 2. Revenue Surprise Analytics
    # =========================================================================
    actual_revenue_cols = ["total_revenues_ltm", "total_revenues_fy", "revenue"]
    estimate_revenue_cols = [
        "revenues_est_avg_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_ntm",
    ]

    revenue_actual = None
    for col in actual_revenue_cols:
        if col in df.columns:
            revenue_actual = pd.to_numeric(df[col], errors="coerce")
            break

    revenue_estimate = None
    for col in estimate_revenue_cols:
        if col in df.columns:
            revenue_estimate = pd.to_numeric(df[col], errors="coerce")
            break

    if revenue_actual is not None and revenue_estimate is not None:
        # Revenue Surprise Percentage
        result["revenue_surprise_pct"] = (
            _safe_div((revenue_actual - revenue_estimate), revenue_estimate.abs()) * 100
        )

        # Revenue Beat Indicator
        result["revenue_beat_indicator"] = (result["revenue_surprise_pct"] > 0).fillna(False)

        logger.info(
            f"Computed revenue surprise for {result['revenue_surprise_pct'].notna().sum()} stocks"
        )

    # =========================================================================
    # 3. EBITDA Surprise Analytics
    # =========================================================================
    actual_ebitda_cols = ["ebitda_ltm", "ebitda_fy", "ebitda"]
    estimate_ebitda_cols = ["ebitda_est_avg_ntm", "ebitda_est_avg_fy1e"]

    ebitda_actual = None
    for col in actual_ebitda_cols:
        if col in df.columns:
            ebitda_actual = pd.to_numeric(df[col], errors="coerce")
            break

    ebitda_estimate = None
    for col in estimate_ebitda_cols:
        if col in df.columns:
            ebitda_estimate = pd.to_numeric(df[col], errors="coerce")
            break

    if ebitda_actual is not None and ebitda_estimate is not None:
        # EBITDA Surprise Percentage
        result["ebitda_surprise_pct"] = (
            _safe_div((ebitda_actual - ebitda_estimate), ebitda_estimate.abs()) * 100
        )

        logger.info(
            f"Computed EBITDA surprise for {result['ebitda_surprise_pct'].notna().sum()} stocks"
        )

    # =========================================================================
    # 4. Surprise Momentum Score (Multi-Period Revision Trend)
    # =========================================================================
    revision_cols = {
        "eps_est_avg_rev_pct_fy1e_1m": 0.5,  # 50% weight to 1-month revision
        "eps_est_avg_rev_pct_fy1e_3m": 0.3,  # 30% weight to 3-month revision
        "eps_est_avg_rev_pct_fy1e_6m": 0.2,  # 20% weight to 6-month revision
    }

    revision_components = []
    for col, weight in revision_cols.items():
        if col in df.columns:
            rev_data = pd.to_numeric(df[col], errors="coerce")
            revision_components.append(rev_data * weight)

    if revision_components:
        # Weighted sum of revision trends
        result["surprise_momentum_score"] = pd.concat(revision_components, axis=1).sum(
            axis=1, skipna=True
        )

        # Identify stocks with consistent positive momentum (all revisions > 0)
        if all(
            col in df.columns
            for col in ["eps_est_avg_rev_pct_fy1e_1m", "eps_est_avg_rev_pct_fy1e_3m"]
        ):
            result["positive_revision_momentum"] = (
                (pd.to_numeric(df["eps_est_avg_rev_pct_fy1e_1m"], errors="coerce") > 0)
                & (pd.to_numeric(df["eps_est_avg_rev_pct_fy1e_3m"], errors="coerce") > 0)
            ).fillna(False)

        logger.info(
            f"Computed surprise momentum score for {result['surprise_momentum_score'].notna().sum()} stocks"
        )

    # =========================================================================
    # 5. Consensus Uncertainty (Estimate Spread as Volatility Proxy)
    # =========================================================================
    if "eps_surprise_pct" in result.columns:
        # Use absolute surprise as proxy for consensus uncertainty
        result["consensus_uncertainty_score"] = result["eps_surprise_pct"].abs()

    # =========================================================================
    # 6. Estimate Revision Acceleration
    # =========================================================================
    if all(
        col in df.columns for col in ["eps_est_avg_rev_pct_fy1e_1m", "eps_est_avg_rev_pct_fy1e_3m"]
    ):
        rev_1m = pd.to_numeric(df["eps_est_avg_rev_pct_fy1e_1m"], errors="coerce")
        rev_3m = pd.to_numeric(df["eps_est_avg_rev_pct_fy1e_3m"], errors="coerce")

        # Acceleration: Recent revision change faster than historical average
        result["estimate_revision_acceleration"] = rev_1m - rev_3m

        # Flag accelerating upgrades
        result["accelerating_upgrades_flag"] = (
            (rev_1m > rev_3m) & (rev_1m > 0) & (rev_3m > 0)
        ).fillna(False)

        logger.info(
            f"Computed estimate revision acceleration for {result['estimate_revision_acceleration'].notna().sum()} stocks"
        )

    logger.info("Engineered Estimated vs. Actual analytics features (Phase 9.3)")
    return result


def engineer_gaap_vs_adjusted_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer GAAP vs. Adjusted earnings quality analytics features.

    Phase 9.3 Enhanced Earnings Quality: Compares GAAP (reported) metrics against
    Adjusted (non-GAAP) metrics to assess earnings quality, identify aggressive
    accounting adjustments, and flag potential red flags.

    Features created:
    - Adjustment Spreads: Dollar differences between adjusted and GAAP metrics
    - Adjustment Ratios: Adjusted / GAAP for EPS, Net Income, EBITDA, EBIT
    - Earnings Quality Flags: Warning indicators for excessive adjustments (>20%)
    - Adjustment Consistency Score: Temporal stability of adjustment patterns
    - Exceptional Items Impact: Non-recurring item magnitude relative to core earnings
    - Quality Score: Composite 0-100 score based on adjustment magnitudes

    Input Columns (from COLUMN_SCHEMA):
    - GAAP EPS: net_eps_basic_ltm, net_eps_basic_fq, net_eps_basic_fy
    - Adjusted EPS: eps_adj_ltm, eps_adj_fy, eps_adj_1fy
    - GAAP Estimates: eps_gaap_est_avg_fy1e, eps_gaap_est_avg_ntm
    - GAAP Net Income: net_income_is_ltm, net_income_is_fy, net_income_is_fq
    - Adjusted Net Income: net_income_adj_ltm, net_income_adj_fy, net_income_adj_1fy
    - GAAP EBITDA: ebitda_ltm, ebitda_fy, ebitda_fq
    - Adjusted EBITDA: ebitda_adj_ltm, ebitda_adj_fy, ebitda_adj_1fy
    - GAAP EBIT: ebit_ltm, ebit_fy, ebit_fq
    - Adjusted EBIT: ebit_adj_ltm, ebit_adj_fy, ebit_adj_1fy

    Args:
        df: Input DataFrame with GAAP and adjusted earnings columns

    Returns:
        DataFrame with GAAP vs. Adjusted analytics features added

    Example:
        >>> df_quality = engineer_gaap_vs_adjusted_analytics(stocks_df)
        >>> print(df_quality[['eps_adjustment_ratio', 'earnings_quality_flag']].head())
        >>> # Identify companies with aggressive adjustments
        >>> aggressive = df_quality[df_quality['earnings_quality_flag'] == True]
    """
    result = df.copy()

    # =========================================================================
    # 1. EPS Adjustment Analytics (LTM - Last Twelve Months)
    # =========================================================================
    if "eps_adj_ltm" in df.columns and "net_eps_basic_ltm" in df.columns:
        eps_adj = pd.to_numeric(df["eps_adj_ltm"], errors="coerce")
        eps_gaap = pd.to_numeric(df["net_eps_basic_ltm"], errors="coerce")

        # EPS Adjustment Spread (dollar difference)
        result["eps_adjustment_spread_ltm"] = eps_adj - eps_gaap

        # EPS Adjustment Ratio (adjusted / GAAP)
        result["eps_adjustment_ratio_ltm"] = _safe_div(eps_adj, eps_gaap)

        # EPS Adjustment Percentage
        result["eps_adjustment_pct_ltm"] = _safe_div((eps_adj - eps_gaap), eps_gaap.abs()) * 100

        # Earnings Quality Flag: Warn if adjustment > 20%
        result["eps_quality_flag_ltm"] = (result["eps_adjustment_pct_ltm"].abs() > 20).fillna(False)

        logger.info(
            f"Computed EPS GAAP vs. Adjusted for {result['eps_adjustment_ratio_ltm'].notna().sum()} stocks (LTM)"
        )

    # EPS Adjustment Analytics (FY - Fiscal Year)
    if "eps_adj_fy" in df.columns and "net_eps_basic_fy" in df.columns:
        eps_adj_fy = pd.to_numeric(df["eps_adj_fy"], errors="coerce")
        eps_gaap_fy = pd.to_numeric(df["net_eps_basic_fy"], errors="coerce")

        result["eps_adjustment_spread_fy"] = eps_adj_fy - eps_gaap_fy
        result["eps_adjustment_ratio_fy"] = _safe_div(eps_adj_fy, eps_gaap_fy)
        result["eps_adjustment_pct_fy"] = (
            _safe_div((eps_adj_fy - eps_gaap_fy), eps_gaap_fy.abs()) * 100
        )

    # =========================================================================
    # 2. Net Income Adjustment Analytics
    # =========================================================================
    if "net_income_adj_ltm" in df.columns and "net_income_is_ltm" in df.columns:
        ni_adj = pd.to_numeric(df["net_income_adj_ltm"], errors="coerce")
        ni_gaap = pd.to_numeric(df["net_income_is_ltm"], errors="coerce")

        # Net Income Adjustment Spread (dollar difference)
        result["net_income_adjustment_spread_ltm"] = ni_adj - ni_gaap

        # Net Income Adjustment Ratio
        result["net_income_adjustment_ratio_ltm"] = _safe_div(ni_adj, ni_gaap)

        # Net Income Adjustment Percentage
        result["net_income_adjustment_pct_ltm"] = _safe_div((ni_adj - ni_gaap), ni_gaap.abs()) * 100

        logger.info(
            f"Computed Net Income GAAP vs. Adjusted for {result['net_income_adjustment_ratio_ltm'].notna().sum()} stocks (LTM)"
        )

    # Net Income Adjustment (FY)
    if "net_income_adj_fy" in df.columns and "net_income_is_fy" in df.columns:
        ni_adj_fy = pd.to_numeric(df["net_income_adj_fy"], errors="coerce")
        ni_gaap_fy = pd.to_numeric(df["net_income_is_fy"], errors="coerce")

        result["net_income_adjustment_spread_fy"] = ni_adj_fy - ni_gaap_fy
        result["net_income_adjustment_ratio_fy"] = _safe_div(ni_adj_fy, ni_gaap_fy)

    # =========================================================================
    # 3. EBITDA Adjustment Analytics
    # =========================================================================
    # Note: ebitda_adjustment_ratio_ltm/fy already computed in engineer_profitability_ratios()
    # Adding spread and percentage metrics
    if "ebitda_adj_ltm" in df.columns and "ebitda_ltm" in df.columns:
        ebitda_adj = pd.to_numeric(df["ebitda_adj_ltm"], errors="coerce")
        ebitda_gaap = pd.to_numeric(df["ebitda_ltm"], errors="coerce")

        result["ebitda_adjustment_spread_ltm"] = ebitda_adj - ebitda_gaap

        result["ebitda_adjustment_pct_ltm"] = (
            _safe_div((ebitda_adj - ebitda_gaap), ebitda_gaap.abs()) * 100
        )

        logger.info(
            f"Computed EBITDA GAAP vs. Adjusted for {result['ebitda_adjustment_spread_ltm'].notna().sum()} stocks (LTM)"
        )

    if "ebitda_adj_fy" in df.columns and "ebitda_fy" in df.columns:
        ebitda_adj_fy = pd.to_numeric(df["ebitda_adj_fy"], errors="coerce")
        ebitda_gaap_fy = pd.to_numeric(df["ebitda_fy"], errors="coerce")

        result["ebitda_adjustment_spread_fy"] = ebitda_adj_fy - ebitda_gaap_fy

    # =========================================================================
    # 4. EBIT Adjustment Analytics
    # =========================================================================
    if "ebit_adj_ltm" in df.columns and "ebit_ltm" in df.columns:
        ebit_adj = pd.to_numeric(df["ebit_adj_ltm"], errors="coerce")
        ebit_gaap = pd.to_numeric(df["ebit_ltm"], errors="coerce")

        result["ebit_adjustment_spread_ltm"] = ebit_adj - ebit_gaap

        result["ebit_adjustment_pct_ltm"] = _safe_div((ebit_adj - ebit_gaap), ebit_gaap.abs()) * 100

    if "ebit_adj_fy" in df.columns and "ebit_fy" in df.columns:
        ebit_adj_fy = pd.to_numeric(df["ebit_adj_fy"], errors="coerce")
        ebit_gaap_fy = pd.to_numeric(df["ebit_fy"], errors="coerce")

        result["ebit_adjustment_spread_fy"] = ebit_adj_fy - ebit_gaap_fy

    # =========================================================================
    # 5. Adjustment Consistency Score (Temporal Stability)
    # =========================================================================
    # Compare LTM vs. FY adjustment ratios to assess consistency
    if "eps_adjustment_ratio_ltm" in result.columns and "eps_adjustment_ratio_fy" in result.columns:
        ratio_diff = (result["eps_adjustment_ratio_ltm"] - result["eps_adjustment_ratio_fy"]).abs()

        # Lower difference = higher consistency (invert scale)
        # Clip to [0, 2] range and invert: consistency = 2 - diff (normalized to 0-100)
        result["adjustment_consistency_score"] = (2.0 - ratio_diff.clip(0, 2)) / 2.0 * 100

        logger.info(
            f"Computed adjustment consistency score for {result['adjustment_consistency_score'].notna().sum()} stocks"
        )

    # =========================================================================
    # 6. Composite Earnings Quality Flag
    # =========================================================================
    # Aggregate quality warning flag from multiple indicators
    quality_flags = []

    if "eps_quality_flag_ltm" in result.columns:
        quality_flags.append(result["eps_quality_flag_ltm"])

    if "ebitda_adjustment_pct_ltm" in result.columns:
        quality_flags.append(result["ebitda_adjustment_pct_ltm"].abs() > 20)

    if "net_income_adjustment_pct_ltm" in result.columns:
        quality_flags.append(result["net_income_adjustment_pct_ltm"].abs() > 20)

    if quality_flags:
        # Overall quality flag: ANY metric exceeds threshold
        result["earnings_quality_warning_flag"] = (
            pd.concat(quality_flags, axis=1).any(axis=1).fillna(False)
        )

        logger.info(
            f"Flagged {result['earnings_quality_warning_flag'].sum()} stocks with earnings quality warnings"
        )

    # =========================================================================
    # 7. Normalized Earnings Quality Score (0-100)
    # =========================================================================
    # Higher score = better quality (lower adjustments)
    # Formula: 100 - weighted average of adjustment percentages
    score_components = []

    if "eps_adjustment_pct_ltm" in result.columns:
        # Cap at 50% for scoring purposes
        eps_adj_impact = result["eps_adjustment_pct_ltm"].abs().clip(0, 50)
        score_components.append(eps_adj_impact * 0.4)  # 40% weight

    if "ebitda_adjustment_pct_ltm" in result.columns:
        ebitda_adj_impact = result["ebitda_adjustment_pct_ltm"].abs().clip(0, 50)
        score_components.append(ebitda_adj_impact * 0.3)  # 30% weight

    if "net_income_adjustment_pct_ltm" in result.columns:
        ni_adj_impact = result["net_income_adjustment_pct_ltm"].abs().clip(0, 50)
        score_components.append(ni_adj_impact * 0.3)  # 30% weight

    if score_components:
        total_adjustment_impact = pd.concat(score_components, axis=1).sum(axis=1, skipna=True)
        result["earnings_quality_score"] = (100 - total_adjustment_impact).clip(0, 100)

        logger.info(
            f"Computed earnings quality score (mean: {result['earnings_quality_score'].mean():.1f})"
        )

    # =========================================================================
    # 8. Exceptional Items Impact Ratio
    # =========================================================================
    # Leverage existing adjustment ratios as proxy for exceptional items magnitude
    exceptional_cols = [
        "impairment_of_goodwill_ltm",
        "asset_writedown_ltm",
        "restructuring_charges_ltm",
        "merger_and_restructuring_charges_ltm",
    ]

    exceptional_items_present = [col for col in exceptional_cols if col in df.columns]

    if exceptional_items_present and "net_income_is_ltm" in df.columns:
        # Sum of absolute exceptional items
        exceptional_sum = (
            df[exceptional_items_present]
            .apply(lambda x: pd.to_numeric(x, errors="coerce"), axis=0)
            .abs()
            .sum(axis=1)
        )

        ni_gaap_abs = pd.to_numeric(df["net_income_is_ltm"], errors="coerce").abs()

        # Exceptional items as % of net income
        result["exceptional_items_impact_ratio"] = _safe_div(exceptional_sum, ni_gaap_abs)

        logger.info(
            f"Computed exceptional items impact for {result['exceptional_items_impact_ratio'].notna().sum()} stocks"
        )

    logger.info("Engineered GAAP vs. Adjusted analytics features (Phase 9.3)")
    return result
