"""Employment and productivity feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

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
    existing_fte = [c for c in fte_cols if c in df.columns]
    if len(existing_fte) >= 2:
        fte_matrix = df[existing_fte].astype(float)
        # Calculate year-over-year percentage changes
        yoy_changes = fte_matrix.pct_change(axis=1).iloc[:, 1:]
        result["workforce_volatility_pct"] = yoy_changes.std(axis=1) * 100

    logger.info("Engineered employee productivity features")
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
