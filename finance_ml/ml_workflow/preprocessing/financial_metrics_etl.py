"""
Financial Metrics ETL Pipeline.

⚠️ DEPRECATION NOTICE ⚠️
This module is consolidated into finance_ml.ml_workflow.preprocessing.etl.
Use etl_with_financial_metrics() or set financial metrics flags in ETLConfig instead.

Migration Guide:
    # OLD (deprecated):
    from finance_ml.ml_workflow.preprocessing.financial_metrics_etl import (
        run_financial_metrics_etl
    )
    df = run_etl_pipeline(source='csv', data_dir='data/')
    df, metrics = run_financial_metrics_etl(df, output_dir=output_dir)

    # NEW (recommended):
    from finance_ml.ml_workflow.preprocessing.etl import etl_with_financial_metrics
    df, metrics = etl_with_financial_metrics(
        source='csv',
        data_dir='data/',
        output_dir=output_dir
    )

Dedicated ETL pipeline for computing financial metrics with sector-specific handling,
data quality alerts, and metrics dashboard generation.

This module extends the base ETL pipeline (etl.py) with specialized financial
metrics computation following the 4-category framework:
- Valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
- Profitability metrics (margins, ROE, ROA)
- Growth metrics (revenue, EBITDA, earnings growth)
- Leverage metrics (debt ratios)

TDD Implementation following code_guidelines.md Section 8 conventions.

Version: 1.1.0 (deprecated - consolidated into etl.py)
Created: 2025-11-30
Deprecated: 2025-12-04
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Emit deprecation warning when module is imported
warnings.warn(
    "DEPRECATION: financial_metrics_etl module is consolidated into "
    "finance_ml.ml_workflow.preprocessing.etl. "
    "Use etl_with_financial_metrics() or set financial metrics flags in ETLConfig instead. "
    "This module will be removed in v2.0.",
    DeprecationWarning,
    stacklevel=2,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants (Section 8.1 compliance)
# =============================================================================

# Severity thresholds for data quality alerts
LOW_MISSING_THRESHOLD = 0.05  # 5%
MEDIUM_MISSING_THRESHOLD = 0.25  # 25%
HIGH_MISSING_THRESHOLD = 0.50  # 50%
CRITICAL_MISSING_THRESHOLD = 0.75  # 75%

# Sector-specific metric mappings
FINANCIALS_SPECIFIC_METRICS = ["tbv_ltm", "tbv_fy", "p_tbv_ltm", "efficiency_ratio"]
TECH_SPECIFIC_METRICS = [
    "r_d_expenses_ltm",
    "r_d_intensity",
    "rule_of_40",
    "cash_burn_rate",
    "marketing_efficiency",
]
HEALTHCARE_SPECIFIC_METRICS = ["r_d_expenses_ltm", "r_d_intensity"]

# Valuation metric column mappings
VALUATION_METRICS = {
    "p_e_ratio": {"numerator": "market_cap", "denominator": "net_income_is_ltm"},
    "p_s_ratio": {"numerator": "market_cap", "denominator": "total_revenues_ltm"},
    "ev_ebitda_ratio": {"numerator": "enterprise_value", "denominator": "ebitda_ltm"},
    "ev_sales_ratio": {"numerator": "enterprise_value", "denominator": "total_revenues_ltm"},
}

# Profitability metric column mappings
PROFITABILITY_METRICS = {
    "gross_margin_pct": {"numerator": "gross_profit_ltm", "denominator": "total_revenues_ltm"},
    "operating_margin_pct": {
        "numerator": "operating_income_ltm",
        "denominator": "total_revenues_ltm",
    },
    "net_margin_pct": {"numerator": "net_income_is_ltm", "denominator": "total_revenues_ltm"},
    "roe": {"numerator": "net_income_is_ltm", "denominator": "total_equity_fy"},
    "roa": {"numerator": "net_income_is_ltm", "denominator": "total_assets_fy"},
}

# Growth metric column mappings (LTM vs FY for YoY growth)
GROWTH_METRICS = {
    "revenue_growth": {"current": "total_revenues_ltm", "prior": "total_revenues_fy"},
    "ebitda_growth": {"current": "ebitda_ltm", "prior": "ebitda_fy"},
    "earnings_growth": {"current": "net_income_is_ltm", "prior": "net_income_is_fy"},
}

# Leverage metric column mappings
LEVERAGE_METRICS = {
    "debt_to_equity": {"numerator": "total_debt_fy", "denominator": "total_equity_fy"},
    "debt_to_assets": {"numerator": "total_debt_fy", "denominator": "total_assets_fy"},
}


# =============================================================================
# Configuration Dataclass
# =============================================================================


@dataclass
class FinancialMetricsETLConfig:
    """
    Configuration for financial metrics ETL pipeline.

    Attributes:
        compute_valuation_metrics: Compute valuation ratios (P/E, P/S, EV/EBITDA)
        compute_profitability_metrics: Compute profitability metrics (margins, ROE, ROA)
        compute_growth_metrics: Compute growth metrics (revenue, EBITDA, earnings)
        compute_leverage_metrics: Compute leverage metrics (debt ratios)
        handle_sector_specific_metrics: Apply sector-specific metric handling
        critical_missing_threshold: Threshold for critical data quality alerts
        generate_quality_alerts: Generate data quality alert JSON
        generate_metrics_dashboard: Generate metrics dashboard JSON
        output_subdir: Subdirectory for output files (default: 'financial_metrics')
        detect_outliers: Enable outlier detection (default: False)
        outlier_method: Outlier detection method ('iqr', 'zscore') (default: 'iqr')
        outlier_threshold: Threshold for outlier detection (default: 2.5 for IQR, 3.0 for z-score)
        winsorize_ratios: Enable winsorization of computed ratios (default: False)
        winsorize_lower: Lower percentile for winsorization (default: 0.01)
        winsorize_upper: Upper percentile for winsorization (default: 0.99)
        winsorize_by_sector: Apply winsorization separately by sector (default: True)
        scale_features: Enable feature scaling (default: False)
        scaler_type: Scaler type ('robust', 'standard', 'minmax') (default: 'robust')
        scale_by_sector: Apply scaling separately by sector (default: True)
    """

    # Metric computation flags
    compute_valuation_metrics: bool = True
    compute_profitability_metrics: bool = True
    compute_growth_metrics: bool = True
    compute_leverage_metrics: bool = True
    compute_target_vs_price: bool = True

    # Sector-specific handling
    handle_sector_specific_metrics: bool = True

    # Data quality thresholds
    critical_missing_threshold: float = CRITICAL_MISSING_THRESHOLD
    high_missing_threshold: float = HIGH_MISSING_THRESHOLD
    medium_missing_threshold: float = MEDIUM_MISSING_THRESHOLD
    low_missing_threshold: float = LOW_MISSING_THRESHOLD

    # Output options
    generate_quality_alerts: bool = True
    generate_metrics_dashboard: bool = True
    output_subdir: str = "financial_metrics"

    # Outlier detection options
    detect_outliers: bool = False
    outlier_method: str = "iqr"  # 'iqr' or 'zscore'
    outlier_threshold: float = 2.5  # IQR multiplier or z-score threshold

    # Winsorization options
    winsorize_ratios: bool = False
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    winsorize_by_sector: bool = True

    # Scaling options
    scale_features: bool = False
    scaler_type: str = "robust"  # 'robust', 'standard', 'minmax'
    scale_by_sector: bool = True

    # Imputation options (for computed metrics with missing values)
    impute_computed_metrics: bool = False
    imputation_method: str = "sector_median"  # 'sector_median', 'global_median', 'zero'
    imputation_columns: Optional[List[str]] = None  # None = all computed metrics
    min_sector_samples: int = 5  # Minimum samples for sector-specific imputation


# =============================================================================
# Valuation Metrics Computation
# =============================================================================


def compute_valuation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute valuation metrics from financial data.

    Computes:
    - P/E ratio (Market Cap / Net Income)
    - P/S ratio (Market Cap / Revenue)
    - EV/EBITDA ratio (Enterprise Value / EBITDA)
    - EV/Sales ratio (Enterprise Value / Revenue)

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with added valuation metric columns
    """
    result = df.copy()

    for metric_name, components in VALUATION_METRICS.items():
        numerator_col = components["numerator"]
        denominator_col = components["denominator"]

        if numerator_col in result.columns and denominator_col in result.columns:
            numerator = pd.to_numeric(result[numerator_col], errors="coerce")
            denominator = pd.to_numeric(result[denominator_col], errors="coerce")

            # Avoid division by zero and negative denominators for some ratios
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = numerator / denominator

            # Replace inf with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)

            result[metric_name] = ratio
            logger.debug(f"Computed {metric_name}: {ratio.notna().sum()} valid values")
        else:
            logger.warning(
                f"Cannot compute {metric_name}: missing columns "
                f"({numerator_col} or {denominator_col})",
            )
            result[metric_name] = np.nan

    return result


# =============================================================================
# Profitability Metrics Computation
# =============================================================================


def compute_profitability_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute profitability metrics from financial data.

    Computes:
    - Gross margin (%) = Gross Profit / Revenue * 100
    - Operating margin (%) = Operating Income / Revenue * 100
    - Net margin (%) = Net Income / Revenue * 100
    - ROE (%) = Net Income / Equity * 100
    - ROA (%) = Net Income / Assets * 100

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with added profitability metric columns
    """
    result = df.copy()

    for metric_name, components in PROFITABILITY_METRICS.items():
        numerator_col = components["numerator"]
        denominator_col = components["denominator"]

        if numerator_col in result.columns and denominator_col in result.columns:
            numerator = pd.to_numeric(result[numerator_col], errors="coerce")
            denominator = pd.to_numeric(result[denominator_col], errors="coerce")

            # Compute ratio as percentage
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (numerator / denominator) * 100

            # Replace inf with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)

            result[metric_name] = ratio
            logger.debug(f"Computed {metric_name}: {ratio.notna().sum()} valid values")
        else:
            logger.warning(
                f"Cannot compute {metric_name}: missing columns "
                f"({numerator_col} or {denominator_col})",
            )
            result[metric_name] = np.nan

    return result


# =============================================================================
# Growth Metrics Computation
# =============================================================================


def compute_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute growth metrics from financial data.

    Computes YoY growth rates:
    - Revenue growth (%) = (LTM - FY) / FY * 100
    - EBITDA growth (%) = (LTM - FY) / FY * 100
    - Earnings growth (%) = (LTM - FY) / FY * 100

    Args:
        df: DataFrame with financial data (must have LTM and FY columns)

    Returns:
        DataFrame with added growth metric columns
    """
    result = df.copy()

    for metric_name, components in GROWTH_METRICS.items():
        current_col = components["current"]
        prior_col = components["prior"]

        if current_col in result.columns and prior_col in result.columns:
            current = pd.to_numeric(result[current_col], errors="coerce")
            prior = pd.to_numeric(result[prior_col], errors="coerce")

            # Compute YoY growth as percentage
            with np.errstate(divide="ignore", invalid="ignore"):
                growth = ((current - prior) / prior.abs()) * 100

            # Replace inf with NaN
            growth = growth.replace([np.inf, -np.inf], np.nan)

            result[metric_name] = growth
            logger.debug(f"Computed {metric_name}: {growth.notna().sum()} valid values")
        else:
            logger.warning(
                f"Cannot compute {metric_name}: missing columns " f"({current_col} or {prior_col})",
            )
            result[metric_name] = np.nan

    return result


# =============================================================================
# Leverage Metrics Computation
# =============================================================================


def compute_leverage_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute leverage metrics from financial data.

    Computes:
    - Debt to Equity = Total Debt / Total Equity
    - Debt to Assets = Total Debt / Total Assets

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with added leverage metric columns
    """
    result = df.copy()

    for metric_name, components in LEVERAGE_METRICS.items():
        numerator_col = components["numerator"]
        denominator_col = components["denominator"]

        if numerator_col in result.columns and denominator_col in result.columns:
            numerator = pd.to_numeric(result[numerator_col], errors="coerce")
            denominator = pd.to_numeric(result[denominator_col], errors="coerce")

            # Compute ratio
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = numerator / denominator

            # Replace inf with NaN
            ratio = ratio.replace([np.inf, -np.inf], np.nan)

            result[metric_name] = ratio
            logger.debug(f"Computed {metric_name}: {ratio.notna().sum()} valid values")
        else:
            logger.warning(
                f"Cannot compute {metric_name}: missing columns "
                f"({numerator_col} or {denominator_col})",
            )
            result[metric_name] = np.nan

    return result


# =============================================================================
# Target vs Price Metrics Computation
# =============================================================================


def compute_target_vs_price_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute target vs price metrics from price target and last price data.

    Computes:
    - target_vs_price (%) = (price_target - last_price) / last_price * 100
    - target_vs_price_median (%) = (price_target_median - last_price) / last_price * 100

    This metric represents the expected return if the stock reaches its analyst price target.
    Positive values indicate potential upside, negative values indicate potential downside.

    Args:
        df: DataFrame with price target and last price data

    Returns:
        DataFrame with added target_vs_price metric columns
    """
    result = df.copy()

    # Define price target columns to check (in priority order)
    price_target_variants = [
        "price_target",
        "Price Target",
        "price_target_mean",
        "Price_Target",
    ]

    price_target_median_variants = [
        "price_target_median",
        "Price Target Median",
        "price_target_med",
        "Price_Target_Median",
    ]

    last_price_variants = [
        "last_price",
        "Last Price",
        "last",
        "Last_Price",
    ]

    # Find available columns
    price_target_col = None
    for col in price_target_variants:
        if col in result.columns:
            price_target_col = col
            break

    price_target_median_col = None
    for col in price_target_median_variants:
        if col in result.columns:
            price_target_median_col = col
            break

    last_price_col = None
    for col in last_price_variants:
        if col in result.columns:
            last_price_col = col
            break

    # Compute target_vs_price if both price_target and last_price are available
    if price_target_col and last_price_col:
        price_target = pd.to_numeric(result[price_target_col], errors="coerce")
        last_price = pd.to_numeric(result[last_price_col], errors="coerce")

        # Compute percentage difference
        with np.errstate(divide="ignore", invalid="ignore"):
            target_vs_price = ((price_target - last_price) / last_price) * 100

        # Replace inf with NaN
        target_vs_price = target_vs_price.replace([np.inf, -np.inf], np.nan)

        result["target_vs_price"] = target_vs_price
        valid_count = target_vs_price.notna().sum()
        logger.info(
            f"Computed target_vs_price: {valid_count} valid values "
            f"(mean={target_vs_price.mean():.2f}%, median={target_vs_price.median():.2f}%)",
        )
    else:
        missing_cols = []
        if not price_target_col:
            missing_cols.append("price_target")
        if not last_price_col:
            missing_cols.append("last_price")
        logger.warning(
            f"Cannot compute target_vs_price: missing columns ({', '.join(missing_cols)})",
        )
        result["target_vs_price"] = np.nan

    # Compute target_vs_price_median if both price_target_median and last_price are available
    if price_target_median_col and last_price_col:
        price_target_median = pd.to_numeric(result[price_target_median_col], errors="coerce")
        last_price = pd.to_numeric(result[last_price_col], errors="coerce")

        # Compute percentage difference
        with np.errstate(divide="ignore", invalid="ignore"):
            target_vs_price_median = ((price_target_median - last_price) / last_price) * 100

        # Replace inf with NaN
        target_vs_price_median = target_vs_price_median.replace([np.inf, -np.inf], np.nan)

        result["target_vs_price_median"] = target_vs_price_median
        valid_count = target_vs_price_median.notna().sum()
        logger.info(
            f"Computed target_vs_price_median: {valid_count} valid values "
            f"(mean={target_vs_price_median.mean():.2f}%, median={target_vs_price_median.median():.2f}%)",
        )
    else:
        if not price_target_median_col:
            logger.debug(
                "price_target_median column not found, skipping target_vs_price_median calculation"
            )
        result["target_vs_price_median"] = np.nan

    return result


# =============================================================================
# Sector-Specific Metrics Handling
# =============================================================================


def handle_sector_specific_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle sector-specific metrics with appropriate missing value treatment.

    For sector-specific columns (e.g., TBV for Financials, R&D for Tech),
    missing values in non-applicable sectors are expected and should not
    trigger critical alerts.

    Args:
        df: DataFrame with sector column and sector-specific metrics

    Returns:
        DataFrame with sector-specific handling applied
    """
    result = df.copy()

    if "sector" not in result.columns:
        logger.warning("No 'sector' column found, skipping sector-specific handling")
        return result

    # Handle Financials-specific metrics
    for col in FINANCIALS_SPECIFIC_METRICS:
        if col in result.columns:
            # For non-Financials, NaN is expected
            non_financials_mask = result["sector"] != "Financials"
            # Log the expected missing pattern
            logger.debug(
                f"Column {col}: {non_financials_mask.sum()} non-Financials rows "
                f"(expected to be NaN)",
            )

    # Handle Tech-specific metrics
    for col in TECH_SPECIFIC_METRICS:
        if col in result.columns:
            # For non-Tech sectors (except Healthcare for R&D), NaN is expected
            non_tech_mask = ~result["sector"].isin(["Technology", "Healthcare"])
            logger.debug(
                f"Column {col}: {non_tech_mask.sum()} non-Tech/Healthcare rows "
                f"(expected to be NaN)",
            )

    return result


def compute_sector_specific_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute sector-specific financial ratios.

    - P/TBV for Financials sector (Price to Tangible Book Value)
    - R&D Intensity for Tech/Healthcare (R&D Expense / Revenue)
    - Rule of 40 for Tech/SaaS (Revenue Growth + EBITDA Margin)
    - Cash Burn Rate for growth companies (Cash / Monthly Burn)
    - Efficiency Ratio for all sectors (Operating Expenses / Revenue)
    - Marketing Efficiency for all sectors (Revenue / SG&A)
    - Tangible Book Value computation when tbv_ltm is missing

    Args:
        df: DataFrame with financial data

    Returns:
        DataFrame with sector-specific ratio columns added
    """
    result = df.copy()

    # =========================================================================
    # 1. Tangible Book Value (compute when missing)
    # =========================================================================
    # Use existing tbv_ltm if available, otherwise compute from equity - goodwill - intangibles
    if "total_equity_ltm" in result.columns:
        equity = pd.to_numeric(result["total_equity_ltm"], errors="coerce")
        goodwill = pd.to_numeric(result.get("goodwill_ltm", 0), errors="coerce").fillna(0)
        intangibles = pd.to_numeric(
            result.get("gross_intangible_assets_ltm", 0), errors="coerce"
        ).fillna(0)

        # Compute tangible book value
        computed_tbv = equity - goodwill - intangibles

        # Use existing tbv_ltm where available, otherwise use computed value
        if "tbv_ltm" in result.columns:
            existing_tbv = pd.to_numeric(result["tbv_ltm"], errors="coerce")
            result["tangible_book_value"] = existing_tbv.fillna(computed_tbv)
        else:
            result["tangible_book_value"] = computed_tbv

        logger.debug(
            f"Computed tangible_book_value: {result['tangible_book_value'].notna().sum()} valid values"
        )
    else:
        # Fallback to tbv_ltm if available
        if "tbv_ltm" in result.columns:
            result["tangible_book_value"] = pd.to_numeric(result["tbv_ltm"], errors="coerce")
            logger.debug(f"Using existing tbv_ltm as tangible_book_value")

    # =========================================================================
    # 2. P/TBV ratio (primarily for Financials)
    # =========================================================================
    if "market_cap" in result.columns:
        market_cap = pd.to_numeric(result["market_cap"], errors="coerce")

        # Use computed tangible_book_value or tbv_ltm
        if "tangible_book_value" in result.columns:
            tbv = pd.to_numeric(result["tangible_book_value"], errors="coerce")
        elif "tbv_ltm" in result.columns:
            tbv = pd.to_numeric(result["tbv_ltm"], errors="coerce")
        else:
            tbv = None

        if tbv is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                p_tbv = market_cap / tbv

            p_tbv = p_tbv.replace([np.inf, -np.inf], np.nan)
            result["p_tbv_ratio"] = p_tbv
            logger.debug(f"Computed p_tbv_ratio: {p_tbv.notna().sum()} valid values")

    # =========================================================================
    # 3. R&D Intensity (fixed column name: r_d_expenses_ltm - plural)
    # =========================================================================
    # Check for correct column name (plural "expenses")
    rd_col = None
    if "r_d_expenses_ltm" in result.columns:
        rd_col = "r_d_expenses_ltm"
    elif "r_d_expense_ltm" in result.columns:  # Fallback for legacy data
        rd_col = "r_d_expense_ltm"

    if rd_col and "total_revenues_ltm" in result.columns:
        rd_expense = pd.to_numeric(result[rd_col], errors="coerce")
        revenue = pd.to_numeric(result["total_revenues_ltm"], errors="coerce")

        # Apply zero-imputation for missing R&D expenses when revenue exists
        # Missing R&D data treated as 0 (company doesn't report R&D or has none)
        has_revenue = revenue.notna() & (revenue > 0)
        rd_expense = rd_expense.where(~(rd_expense.isna() & has_revenue), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            rd_intensity = (rd_expense / revenue) * 100

        rd_intensity = rd_intensity.replace([np.inf, -np.inf], np.nan)
        result["r_d_intensity"] = rd_intensity
        logger.debug(
            f"Computed r_d_intensity using {rd_col}: {rd_intensity.notna().sum()} valid values"
        )

    # =========================================================================
    # 4. Efficiency Ratio (Operating Expenses / Revenue * 100)
    # =========================================================================
    if "total_operating_expenses_ltm" in result.columns and "total_revenues_ltm" in result.columns:
        op_expenses = pd.to_numeric(result["total_operating_expenses_ltm"], errors="coerce")
        revenue = pd.to_numeric(result["total_revenues_ltm"], errors="coerce")

        # Apply zero-imputation for missing operating expenses when revenue exists
        # Missing operating expenses treated as 0 (not reported or minimal)
        has_revenue = revenue.notna() & (revenue > 0)
        op_expenses = op_expenses.where(~(op_expenses.isna() & has_revenue), 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            efficiency_ratio = (op_expenses / revenue) * 100

        efficiency_ratio = efficiency_ratio.replace([np.inf, -np.inf], np.nan)
        # Clip to reasonable range (0-200%)
        efficiency_ratio = efficiency_ratio.clip(lower=0, upper=200)
        result["efficiency_ratio"] = efficiency_ratio
        logger.debug(f"Computed efficiency_ratio: {efficiency_ratio.notna().sum()} valid values")

    # =========================================================================
    # 5. Marketing Efficiency (Revenue / SG&A - higher is better)
    # =========================================================================
    # Check for SG&A columns with different naming conventions
    sga_col = None
    for col_name in [
        "selling_general_and_admin_expenses_total_fy",
        "selling_general_and_admin_expenses_total_ltm",
        "selling_general_admin_expenses_total_fy",
    ]:
        if col_name in result.columns:
            sga_col = col_name
            break

    if sga_col and "total_revenues_ltm" in result.columns:
        sga = pd.to_numeric(result[sga_col], errors="coerce")
        revenue = pd.to_numeric(result["total_revenues_ltm"], errors="coerce")

        # Apply zero-imputation for missing SG&A when revenue exists
        # Missing SG&A treated as minimal value to avoid division issues
        # Use small positive value (0.01% of revenue) instead of pure zero
        has_revenue = revenue.notna() & (revenue > 0)
        sga_imputed = sga.where(~(sga.isna() & has_revenue), revenue * 0.0001)

        with np.errstate(divide="ignore", invalid="ignore"):
            marketing_efficiency = revenue / sga_imputed

        marketing_efficiency = marketing_efficiency.replace([np.inf, -np.inf], np.nan)
        result["marketing_efficiency"] = marketing_efficiency
        logger.debug(
            f"Computed marketing_efficiency using {sga_col}: {marketing_efficiency.notna().sum()} valid values"
        )

    # =========================================================================
    # 6. Cash Burn Rate (months of runway for cash-burning companies)
    # =========================================================================
    # Cash burn rate = Cash / (Monthly cash burn)
    # Only meaningful for companies with negative operating cash flow
    if "cash_and_equivalents_ltm" in result.columns and "cfo_ltm" in result.columns:
        cash = pd.to_numeric(result["cash_and_equivalents_ltm"], errors="coerce")
        cfo = pd.to_numeric(result["cfo_ltm"], errors="coerce")

        # Apply zero-imputation for missing cash when company is cash-burning (negative CFO)
        # This treats missing cash as 0 (depleted) for cash-burning companies
        # Profitable companies (positive CFO) will get NaN regardless
        cash_burning_mask = cfo < 0
        cash = cash.where(~(cash.isna() & cash_burning_mask), 0.0)

        # Monthly burn rate (only for negative CFO - cash burning companies)
        # CFO is annual, so divide by 12 for monthly
        with np.errstate(divide="ignore", invalid="ignore"):
            monthly_burn = -cfo / 12  # Negative CFO becomes positive burn
            cash_burn_rate = cash / monthly_burn  # Months of runway

        # Only keep positive burn rates (companies actually burning cash)
        # Companies with positive CFO get NaN (not burning cash)
        cash_burn_rate = cash_burn_rate.where(cfo < 0, np.nan)
        cash_burn_rate = cash_burn_rate.replace([np.inf, -np.inf], np.nan)
        # Clip to reasonable range (0-1000 months = ~83 years max)
        cash_burn_rate = cash_burn_rate.clip(lower=0, upper=1000)
        result["cash_burn_rate"] = cash_burn_rate
        logger.debug(f"Computed cash_burn_rate: {cash_burn_rate.notna().sum()} valid values")

    # =========================================================================
    # 7. Rule of 40 (for SaaS/Tech companies)
    # =========================================================================
    if "revenue_growth" in result.columns:
        revenue_growth = pd.to_numeric(result.get("revenue_growth", np.nan), errors="coerce")

        # Use operating margin or compute EBITDA margin
        if "operating_margin_pct" in result.columns:
            margin = pd.to_numeric(result["operating_margin_pct"], errors="coerce")
        elif "ebitda_ltm" in result.columns and "total_revenues_ltm" in result.columns:
            ebitda = pd.to_numeric(result["ebitda_ltm"], errors="coerce")
            revenue = pd.to_numeric(result["total_revenues_ltm"], errors="coerce")
            with np.errstate(divide="ignore", invalid="ignore"):
                margin = (ebitda / revenue) * 100
            margin = margin.replace([np.inf, -np.inf], np.nan)
        else:
            margin = np.nan

        if not isinstance(margin, float):
            result["rule_of_40"] = revenue_growth + margin
            logger.debug(f"Computed rule_of_40: {result['rule_of_40'].notna().sum()} valid values")

    return result


# =============================================================================
# Imputation for Computed Metrics
# =============================================================================

# Metrics that should NOT be imputed (they are only valid for specific conditions)
# cash_burn_rate: Only valid for companies with negative CFO (burning cash)
# Employee metrics: Only valid when employee count data is available
# These metrics having NaN is correct behavior, not missing data
CONDITIONAL_METRICS = [
    "cash_burn_rate",
    # Employee productivity metrics - NaN when employee data unavailable
    "revenue_per_employee",
    "revenue_per_employee_ltm",
    "revenue_per_employee_fy",
    "revenue_per_employee_trend",
    "revenue_per_employee_vs_5y_pct",
    "assets_per_employee",
    "ebitda_per_employee",
    "operating_income_per_employee",
    "profit_per_employee",
    "employee_growth_yoy",
    "employee_growth_yoy_pct",
    "employee_growth_qoq",
    "employee_growth_cagr_5y",
    "employee_growth_acceleration",
    "workforce_volatility",
    "hiring_intensity_score",
]

# Default list of computed metrics that can be imputed
IMPUTABLE_METRICS = [
    # Valuation metrics
    "p_e_ratio",
    "p_s_ratio",
    "ev_ebitda_ratio",
    "ev_sales_ratio",
    # Profitability metrics
    "gross_margin_pct",
    "operating_margin_pct",
    "net_margin_pct",
    "roe",
    "roa",
    # Growth metrics
    "revenue_growth",
    "ebitda_growth",
    "earnings_growth",
    # Leverage metrics
    "debt_to_equity",
    "debt_to_assets",
    # Sector-specific metrics (except conditional ones)
    "p_tbv_ratio",
    "r_d_intensity",
    "efficiency_ratio",
    "marketing_efficiency",
    "tangible_book_value",
    "rule_of_40",
]


def impute_computed_metrics(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "sector_median",
    sector_column: str = "sector",
    min_sector_samples: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Impute missing values in computed financial metrics.

    Uses sector-aware imputation to preserve sector-specific characteristics.
    Excludes conditional metrics (like cash_burn_rate) that are only valid
    for specific business conditions.

    Args:
        df: DataFrame with computed metrics
        columns: Columns to impute (default: IMPUTABLE_METRICS present in df)
        method: Imputation method
            - 'sector_median': Use sector median, fallback to global median
            - 'global_median': Use global median for all
            - 'zero': Fill with zero (use with caution)
        sector_column: Column name for sector grouping
        min_sector_samples: Minimum samples required for sector-specific imputation

    Returns:
        Tuple of (imputed DataFrame, imputation statistics dict)

    Note:
        cash_burn_rate is NOT imputed because NaN indicates the company is
        profitable (positive CFO) - this is correct behavior, not missing data.
    """
    result = df.copy()

    # Determine columns to impute
    if columns is None:
        columns = [c for c in IMPUTABLE_METRICS if c in result.columns]
    else:
        # Filter out conditional metrics even if explicitly specified
        columns = [c for c in columns if c not in CONDITIONAL_METRICS and c in result.columns]

    imputation_stats = {
        "method": method,
        "columns_imputed": [],
        "values_imputed": 0,
        "columns_skipped": [],
        "conditional_metrics_preserved": [c for c in CONDITIONAL_METRICS if c in df.columns],
    }

    if not columns:
        logger.info("No columns to impute")
        return result, imputation_stats

    logger.info(f"Imputing {len(columns)} computed metrics using {method} method")

    for col in columns:
        if col not in result.columns:
            continue

        missing_before = result[col].isna().sum()

        if missing_before == 0:
            continue

        if method == "sector_median" and sector_column in result.columns:
            # Sector-aware imputation
            for sector in result[sector_column].dropna().unique():
                sector_mask = result[sector_column] == sector
                sector_data = result.loc[sector_mask, col]

                # Check if enough non-null samples for sector median
                non_null_count = sector_data.notna().sum()

                if non_null_count >= min_sector_samples:
                    # Use sector median
                    sector_median = sector_data.median()
                    result.loc[sector_mask & result[col].isna(), col] = sector_median
                else:
                    # Fallback to global median for this sector
                    global_median = result[col].median()
                    if pd.notna(global_median):
                        result.loc[sector_mask & result[col].isna(), col] = global_median

            # Handle any remaining NaN (e.g., missing sector)
            remaining_nan = result[col].isna().sum()
            if remaining_nan > 0:
                global_median = result[col].median()
                if pd.notna(global_median):
                    result[col] = result[col].fillna(global_median)

        elif method == "global_median":
            # Global median imputation
            global_median = result[col].median()
            if pd.notna(global_median):
                result[col] = result[col].fillna(global_median)

        elif method == "zero":
            # Zero imputation (use with caution)
            result[col] = result[col].fillna(0)

        missing_after = result[col].isna().sum()
        values_filled = missing_before - missing_after

        if values_filled > 0:
            imputation_stats["columns_imputed"].append(col)
            imputation_stats["values_imputed"] += values_filled
            logger.debug(
                f"Imputed {col}: {values_filled} values filled ({missing_after} still missing)"
            )
        else:
            imputation_stats["columns_skipped"].append(col)

    logger.info(
        f"Imputation complete: {imputation_stats['values_imputed']} values filled "
        f"in {len(imputation_stats['columns_imputed'])} columns",
    )

    return result, imputation_stats


# =============================================================================
# Data Quality Alerts Generation
# =============================================================================


def generate_data_quality_alerts(
    df: pd.DataFrame,
    critical_threshold: float = CRITICAL_MISSING_THRESHOLD,
    high_threshold: float = HIGH_MISSING_THRESHOLD,
    medium_threshold: float = MEDIUM_MISSING_THRESHOLD,
    low_threshold: float = LOW_MISSING_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Generate data quality alerts based on missing value rates.

    Severity levels:
    - critical: > 75% missing
    - high: > 50% missing
    - medium: > 25% missing
    - low: > 5% missing

    Args:
        df: DataFrame to analyze
        critical_threshold: Threshold for critical severity
        high_threshold: Threshold for high severity
        medium_threshold: Threshold for medium severity
        low_threshold: Threshold for low severity

    Returns:
        List of alert dictionaries with severity, message, column, count, is_conditional
    """
    alerts = []
    n_rows = len(df)

    if n_rows == 0:
        return alerts

    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / n_rows

        if missing_rate > low_threshold:
            # Determine severity
            if missing_rate > critical_threshold:
                severity = "critical"
            elif missing_rate > high_threshold:
                severity = "high"
            elif missing_rate > medium_threshold:
                severity = "medium"
            else:
                severity = "low"

            # Check if this is a conditional metric (expected to have high missing)
            is_conditional = col in CONDITIONAL_METRICS

            # Adjust message for conditional metrics
            if is_conditional:
                message = (
                    f"Column '{col}' has {missing_count} missing values ({missing_rate * 100:.1f}%) "
                    f"[EXPECTED: conditional metric - requires specific source data]"
                )
                # Downgrade severity for conditional metrics
                if severity == "critical":
                    severity = "medium"
                elif severity == "high":
                    severity = "low"
            else:
                message = (
                    f"Column '{col}' has {missing_count} missing values ({missing_rate * 100:.1f}%)"
                )

            alert = {
                "severity": severity,
                "message": message,
                "column": col,
                "count": int(missing_count),
                "is_conditional": is_conditional,
            }
            alerts.append(alert)

    # Sort by severity (critical first) then by count
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    alerts.sort(key=lambda x: (severity_order[x["severity"]], -x["count"]))

    return alerts


# =============================================================================
# Metrics Dashboard Generation
# =============================================================================


def generate_metrics_dashboard(
    df: pd.DataFrame,
    sector_column: str = "sector",
) -> Dict[str, Any]:
    """
    Generate a metrics dashboard JSON structure.

    Dashboard includes:
    - Timestamp
    - Total stock count
    - By-sector breakdown with valuation, profitability, growth, leverage metrics
    - Per-group statistics (by sector)

    Args:
        df: DataFrame with computed financial metrics
        sector_column: Column name for sector grouping

    Returns:
        Dictionary suitable for JSON serialization
    """
    dashboard = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks": len(df),
        "by_sector": {},
    }

    def compute_stats(series: pd.Series) -> Dict[str, Any]:
        """Compute summary statistics for a numeric series."""
        numeric = pd.to_numeric(series, errors="coerce")
        return {
            "mean": float(numeric.mean()) if not numeric.isna().all() else None,
            "median": float(numeric.median()) if not numeric.isna().all() else None,
            "std": float(numeric.std()) if not numeric.isna().all() else None,
            "min": float(numeric.min()) if not numeric.isna().all() else None,
            "max": float(numeric.max()) if not numeric.isna().all() else None,
            "count": int(numeric.notna().sum()),
        }

    # Valuation metrics
    valuation_metrics = {}
    if "p_e_ratio" in df.columns:
        valuation_metrics["p_e"] = compute_stats(df["p_e_ratio"])
    if "p_b_ratio" in df.columns:
        valuation_metrics["p_b"] = compute_stats(df["p_b_ratio"])
    if "p_s_ratio" in df.columns:
        valuation_metrics["p_s"] = compute_stats(df["p_s_ratio"])
    if "ev_ebitda_ratio" in df.columns:
        valuation_metrics["ev_ebitda"] = compute_stats(df["ev_ebitda_ratio"])
    if "ev_sales_ratio" in df.columns:
        valuation_metrics["ev_sales"] = compute_stats(df["ev_sales_ratio"])

    dashboard["by_sector"]["valuation"] = valuation_metrics

    # Profitability metrics
    profitability_metrics = {}
    if "gross_margin_pct" in df.columns:
        profitability_metrics["gross_margin"] = compute_stats(df["gross_margin_pct"])
    if "operating_margin_pct" in df.columns:
        profitability_metrics["operating_margin"] = compute_stats(df["operating_margin_pct"])
    if "net_margin_pct" in df.columns:
        profitability_metrics["net_margin"] = compute_stats(df["net_margin_pct"])
    if "roe" in df.columns:
        profitability_metrics["roe"] = compute_stats(df["roe"])
    if "roa" in df.columns:
        profitability_metrics["roa"] = compute_stats(df["roa"])

    dashboard["by_sector"]["profitability"] = profitability_metrics

    # Growth metrics
    growth_metrics = {}
    if "revenue_growth" in df.columns:
        growth_metrics["revenue_growth"] = compute_stats(df["revenue_growth"])
    if "ebitda_growth" in df.columns:
        growth_metrics["ebitda_growth"] = compute_stats(df["ebitda_growth"])
    if "earnings_growth" in df.columns:
        growth_metrics["earnings_growth"] = compute_stats(df["earnings_growth"])

    dashboard["by_sector"]["growth"] = growth_metrics

    # Leverage metrics
    leverage_metrics = {}
    if "debt_to_equity" in df.columns:
        leverage_metrics["debt_to_equity"] = compute_stats(df["debt_to_equity"])
    if "debt_to_assets" in df.columns:
        leverage_metrics["debt_to_assets"] = compute_stats(df["debt_to_assets"])

    dashboard["by_sector"]["leverage"] = leverage_metrics

    # Per-group breakdown (by sector)
    if sector_column in df.columns:
        by_group = {}
        for sector in df[sector_column].dropna().unique():
            sector_df = df[df[sector_column] == sector]
            sector_stats = {
                "count": len(sector_df),
                "valuation": {},
                "profitability": {},
                "growth": {},
                "leverage": {},
            }

            # Add sector-specific stats
            if "p_e_ratio" in sector_df.columns:
                sector_stats["valuation"]["p_e"] = compute_stats(sector_df["p_e_ratio"])
            if "roe" in sector_df.columns:
                sector_stats["profitability"]["roe"] = compute_stats(sector_df["roe"])
            if "revenue_growth" in sector_df.columns:
                sector_stats["growth"]["revenue_growth"] = compute_stats(
                    sector_df["revenue_growth"],
                )
            if "debt_to_equity" in sector_df.columns:
                sector_stats["leverage"]["debt_to_equity"] = compute_stats(
                    sector_df["debt_to_equity"],
                )

            by_group[str(sector)] = sector_stats

        dashboard["by_group"] = by_group

    return dashboard


# =============================================================================
# Main Pipeline Function
# =============================================================================


def run_financial_metrics_etl(
    df: pd.DataFrame,
    config: Optional[FinancialMetricsETLConfig] = None,
    output_dir: Optional[Path] = None,
    return_metrics: bool = True,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Run the complete financial metrics ETL pipeline.

    Pipeline stages:
    1. Compute valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
    2. Compute profitability metrics (margins, ROE, ROA)
    3. Compute growth metrics (revenue, EBITDA, earnings growth)
    4. Compute leverage metrics (debt ratios)
    4.5. Compute target vs price metrics (target_vs_price, target_vs_price_median)
    5. Handle sector-specific metrics and compute sector-specific ratios
    5.5. Imputation of computed metrics (optional) - sector or global median
    6. Outlier detection (optional) - IQR or z-score based
    7. Winsorization (optional) - clip extreme values by percentile
    8. Feature scaling (optional) - robust, standard, or minmax scaling
    9. Generate data quality alerts (optional)
    10. Generate metrics dashboard (optional)

    Args:
        df: Input DataFrame with financial data
        config: Pipeline configuration (uses defaults if None)
        output_dir: Directory to save output JSON files (optional)
        return_metrics: If True, return (DataFrame, metrics) tuple

    Returns:
        DataFrame with computed metrics, or (DataFrame, metrics) tuple if return_metrics=True
    """
    if config is None:
        config = FinancialMetricsETLConfig()

    metrics = {
        "valuation_metrics_added": 0,
        "profitability_metrics_added": 0,
        "growth_metrics_added": 0,
        "leverage_metrics_added": 0,
        "target_vs_price_metrics_added": 0,
        "sector_specific_metrics_added": 0,
        "quality_alerts_count": 0,
        "dashboard_generated": False,
    }

    result = df.copy()
    initial_cols = set(result.columns)

    logger.info(f"Starting financial metrics ETL pipeline with {len(result)} rows")

    # Stage 1: Valuation metrics
    if config.compute_valuation_metrics:
        result = compute_valuation_metrics(result)
        new_cols = set(result.columns) - initial_cols
        metrics["valuation_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        logger.info(f"Stage 1: Added {metrics['valuation_metrics_added']} valuation metrics")

    # Stage 2: Profitability metrics
    if config.compute_profitability_metrics:
        result = compute_profitability_metrics(result)
        new_cols = set(result.columns) - initial_cols
        metrics["profitability_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        logger.info(
            f"Stage 2: Added {metrics['profitability_metrics_added']} profitability metrics",
        )

    # Stage 3: Growth metrics
    if config.compute_growth_metrics:
        result = compute_growth_metrics(result)
        new_cols = set(result.columns) - initial_cols
        metrics["growth_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        logger.info(f"Stage 3: Added {metrics['growth_metrics_added']} growth metrics")

    # Stage 4: Leverage metrics
    if config.compute_leverage_metrics:
        result = compute_leverage_metrics(result)
        new_cols = set(result.columns) - initial_cols
        metrics["leverage_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        logger.info(f"Stage 4: Added {metrics['leverage_metrics_added']} leverage metrics")

    # Stage 4.5: Target vs Price metrics
    if config.compute_target_vs_price:
        result = compute_target_vs_price_metrics(result)
        new_cols = set(result.columns) - initial_cols
        metrics["target_vs_price_metrics_added"] = len(new_cols)
        initial_cols = set(result.columns)
        logger.info(
            f"Stage 4.5: Added {metrics['target_vs_price_metrics_added']} target vs price metrics",
        )

    # Stage 5: Sector-specific handling
    if config.handle_sector_specific_metrics:
        result = handle_sector_specific_metrics(result)
        result = compute_sector_specific_ratios(result)
        new_cols = set(result.columns) - initial_cols
        metrics["sector_specific_metrics_added"] = len(new_cols)
        logger.info(
            f"Stage 5: Added {metrics['sector_specific_metrics_added']} sector-specific metrics",
        )

    # Stage 5.5: Imputation of computed metrics
    if config.impute_computed_metrics:
        result, imputation_stats = impute_computed_metrics(
            result,
            columns=config.imputation_columns,
            method=config.imputation_method,
            sector_column="sector",
            min_sector_samples=config.min_sector_samples,
        )
        metrics["imputation_applied"] = True
        metrics["imputation_method"] = imputation_stats["method"]
        metrics["imputation_values_filled"] = imputation_stats["values_imputed"]
        metrics["imputation_columns"] = imputation_stats["columns_imputed"]
        metrics["conditional_metrics_preserved"] = imputation_stats["conditional_metrics_preserved"]
        logger.info(
            f"Stage 5.5: Imputed {imputation_stats['values_imputed']} values "
            f"in {len(imputation_stats['columns_imputed'])} columns",
        )
    else:
        metrics["imputation_applied"] = False
        metrics["imputation_method"] = None
        metrics["imputation_values_filled"] = 0
        metrics["imputation_columns"] = []

    # Stage 6: Outlier Detection
    if config.detect_outliers:
        from finance_ml.ml_workflow.preprocessing.outliers import (
            detect_outliers_iqr,
            detect_outliers_zscore,
        )

        initial_cols_outlier = set(result.columns)

        if config.outlier_method == "iqr":
            result = detect_outliers_iqr(
                result,
                columns=None,  # All numeric columns
                by_sector=True,
                iqr_multiplier=config.outlier_threshold,
            )
        elif config.outlier_method == "zscore":
            result = detect_outliers_zscore(
                result,
                columns=None,  # All numeric columns
                threshold=config.outlier_threshold,
                by_sector=True,
            )
        else:
            logger.warning(f"Unknown outlier method: {config.outlier_method}, skipping")

        outlier_cols = [c for c in result.columns if "_outlier" in c]
        total_outliers = sum(result[c].sum() for c in outlier_cols if c in result.columns)
        metrics["outliers_detected"] = int(total_outliers)
        metrics["outlier_method"] = config.outlier_method
        metrics["outlier_columns_added"] = len(set(result.columns) - initial_cols_outlier)
        logger.info(
            f"Stage 6: Detected {metrics['outliers_detected']} outliers using {config.outlier_method} method",
        )

    # Stage 7: Winsorization
    if config.winsorize_ratios:
        from finance_ml.ml_workflow.preprocessing.outliers import winsorize_by_sector

        # Get computed ratio columns for winsorization
        ratio_columns = [
            col
            for col in result.columns
            if col.endswith("_ratio") or col.endswith("_pct") or col in ["roe", "roa"]
        ]

        if ratio_columns:
            result = winsorize_by_sector(
                result,
                columns=ratio_columns,
                lower_percentile=config.winsorize_lower,
                upper_percentile=config.winsorize_upper,
                by_sector=config.winsorize_by_sector,
                exclude_price_columns=True,
                exclude_ratio_columns=False,  # We want to winsorize ratios
            )
            metrics["winsorization_applied"] = True
            metrics["winsorize_bounds"] = {
                "lower": config.winsorize_lower,
                "upper": config.winsorize_upper,
            }
            metrics["columns_winsorized"] = len(ratio_columns)
            logger.info(
                f"Stage 7: Winsorized {len(ratio_columns)} ratio columns "
                f"({config.winsorize_lower:.1%}-{config.winsorize_upper:.1%})",
            )
        else:
            metrics["winsorization_applied"] = False
            metrics["winsorize_bounds"] = None
            metrics["columns_winsorized"] = 0
            logger.info("Stage 7: No ratio columns found to winsorize")
    else:
        metrics["winsorization_applied"] = False
        metrics["winsorize_bounds"] = None

    # Stage 8: Feature Scaling
    if config.scale_features:
        from finance_ml.ml_workflow.preprocessing.scaling import scale_features

        # Get numeric columns for scaling (excluding outlier flag columns)
        scalable_columns = [
            col
            for col in result.select_dtypes(include=[np.number]).columns
            if "_outlier" not in col
        ]

        if scalable_columns:
            result = scale_features(
                result,
                columns=scalable_columns,
                scaler_type=config.scaler_type,
                by_sector=config.scale_by_sector,
                exclude_price_columns=True,  # Preserve price columns for business metrics
            )
            metrics["scaling_applied"] = True
            metrics["scaler_type"] = config.scaler_type
            metrics["columns_scaled"] = len(scalable_columns)
            logger.info(
                f"Stage 8: Scaled {len(scalable_columns)} columns using {config.scaler_type} scaler",
            )
        else:
            metrics["scaling_applied"] = False
            metrics["scaler_type"] = None
            metrics["columns_scaled"] = 0
            logger.info("Stage 8: No columns found to scale")
    else:
        metrics["scaling_applied"] = False
        metrics["scaler_type"] = None
        metrics["columns_scaled"] = 0

    # Stage 9: Data quality alerts
    quality_alerts = []
    if config.generate_quality_alerts:
        quality_alerts = generate_data_quality_alerts(
            result,
            critical_threshold=config.critical_missing_threshold,
            high_threshold=config.high_missing_threshold,
            medium_threshold=config.medium_missing_threshold,
            low_threshold=config.low_missing_threshold,
        )
        metrics["quality_alerts_count"] = len(quality_alerts)
        logger.info(f"Stage 6: Generated {len(quality_alerts)} data quality alerts")

        # Save to file if output_dir provided
        if output_dir is not None:
            output_path = Path(output_dir) / config.output_subdir
            output_path.mkdir(parents=True, exist_ok=True)
            alerts_file = output_path / "data_quality_alerts.json"
            with open(alerts_file, "w") as f:
                json.dump(quality_alerts, f, indent=2)
            logger.info(f"Saved quality alerts to {alerts_file}")

    # Stage 7: Metrics dashboard
    dashboard = {}
    if config.generate_metrics_dashboard:
        dashboard = generate_metrics_dashboard(result)
        metrics["dashboard_generated"] = True
        logger.info("Stage 7: Generated metrics dashboard")

        # Save to file if output_dir provided
        if output_dir is not None:
            output_path = Path(output_dir) / config.output_subdir
            output_path.mkdir(parents=True, exist_ok=True)
            dashboard_file = output_path / "metrics_dashboard.json"
            with open(dashboard_file, "w") as f:
                json.dump(dashboard, f, indent=2)
            logger.info(f"Saved metrics dashboard to {dashboard_file}")

    # Add alerts and dashboard to metrics for return
    metrics["quality_alerts"] = quality_alerts
    metrics["dashboard"] = dashboard

    logger.info(f"Financial metrics ETL complete: {result.shape[1]} total columns")

    if return_metrics:
        return result, metrics
    return result


# =============================================================================
# Convenience Functions
# =============================================================================


def etl_financial_metrics(
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to run financial metrics ETL with default config.

    Args:
        df: Input DataFrame
        output_dir: Optional output directory for JSON files

    Returns:
        (DataFrame, metrics) tuple
    """
    return run_financial_metrics_etl(
        df,
        config=FinancialMetricsETLConfig(),
        output_dir=output_dir,
        return_metrics=True,
    )


def compute_all_financial_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all financial metrics without generating reports.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with all computed metrics
    """
    config = FinancialMetricsETLConfig(
        generate_quality_alerts=False,
        generate_metrics_dashboard=False,
    )
    return run_financial_metrics_etl(df, config=config, return_metrics=False)
