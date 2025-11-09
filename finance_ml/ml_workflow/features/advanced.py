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
from typing import Optional, List

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
    "engineer_accounting_quality_features",
    "engineer_employee_productivity_features",
    "build_comprehensive_features",
]


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf/NaN with appropriate values.

    Args:
        numer: Numerator Series
        denom: Denominator Series

    Returns:
        Result Series with inf/NaN handled
    """
    result = numer.astype(float) / denom.astype(float).replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


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
    if "p_e_ratio" in result.columns and "earnings_growth_pct" in df.columns:
        result["peg_ratio"] = _safe_div(result["p_e_ratio"], df["earnings_growth_pct"])

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
    if "cogs" in df.columns and "inventory" in df.columns:
        result["inventory_turnover"] = _safe_div(df["cogs"], df["inventory"])

    # Receivables Turnover (Revenue / Accounts Receivable)
    if "revenue" in df.columns and "accounts_receivable" in df.columns:
        result["receivables_turnover"] = _safe_div(df["revenue"], df["accounts_receivable"])

    # Revenue per Employee
    if "revenue" in df.columns and "employees" in df.columns:
        result["revenue_per_employee"] = _safe_div(df["revenue"], df["employees"])

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
            _safe_div((df["revenue"] - df["revenue_previous_year"]), df["revenue_previous_year"])
            * 100
        )

    # EPS Growth
    if "eps" in df.columns and "eps_previous_year" in df.columns:
        result["eps_growth_yoy"] = (
            _safe_div((df["eps"] - df["eps_previous_year"]), df["eps_previous_year"]) * 100
        )

    # EBITDA Growth
    if "ebitda" in df.columns and "ebitda_previous_year" in df.columns:
        result["ebitda_growth_yoy"] = (
            _safe_div((df["ebitda"] - df["ebitda_previous_year"]), df["ebitda_previous_year"]) * 100
        )

    logger.info("Engineered growth metrics")
    return result


def engineer_sector_specific_features(df: pd.DataFrame, sector_col: str = "sector") -> pd.DataFrame:
    """Engineer sector-specific features based on industry best practices.

    Args:
        df: Input DataFrame
        sector_col: Name of sector column

    Returns:
        DataFrame with sector-specific features added
    """
    result = df.copy()

    if sector_col not in df.columns:
        logger.warning(f"Sector column '{sector_col}' not found, skipping sector-specific features")
        return result

    # Financials sector features
    financials_mask = df[sector_col].str.contains("Financial", case=False, na=False)
    if financials_mask.any():
        # Add Tangible Book Value features if applicable
        if "total_equity" in df.columns and "intangible_assets" in df.columns:
            result.loc[financials_mask, "tangible_book_value"] = df.loc[
                financials_mask, "total_equity"
            ] - df.loc[financials_mask, "intangible_assets"].fillna(0)

            # Price to Tangible Book Value ratio
            if "last_price" in df.columns and "shares_outstanding" in df.columns:
                tbv_per_share = (
                    result.loc[financials_mask, "tangible_book_value"]
                    / df.loc[financials_mask, "shares_outstanding"]
                )
                result.loc[financials_mask, "p_tbv_ratio"] = _safe_div(
                    df.loc[financials_mask, "last_price"], tbv_per_share
                )

        # Net Interest Margin
        if all(
            col in df.columns for col in ["interest_income", "interest_expense", "earning_assets"]
        ):
            net_interest_income = (
                df.loc[financials_mask, "interest_income"]
                - df.loc[financials_mask, "interest_expense"]
            )
            result.loc[financials_mask, "net_interest_margin"] = (
                _safe_div(net_interest_income, df.loc[financials_mask, "earning_assets"]) * 100
            )

        # Efficiency Ratio
        if "operating_expenses" in df.columns and "revenue" in df.columns:
            result.loc[financials_mask, "efficiency_ratio"] = (
                _safe_div(
                    df.loc[financials_mask, "operating_expenses"],
                    df.loc[financials_mask, "revenue"],
                )
                * 100
            )

    # Energy/Materials sector features
    energy_mask = df[sector_col].str.contains("Energy|Materials", case=False, na=False)
    if energy_mask.any():
        # CAPEX Intensity
        if "capex" in df.columns and "revenue" in df.columns:
            result.loc[energy_mask, "capex_intensity"] = (
                _safe_div(df.loc[energy_mask, "capex"], df.loc[energy_mask, "revenue"]) * 100
            )

        # Asset Turnover
        if "revenue" in df.columns and "total_assets" in df.columns:
            result.loc[energy_mask, "asset_turnover"] = _safe_div(
                df.loc[energy_mask, "revenue"], df.loc[energy_mask, "total_assets"]
            )

    # Technology sector features
    tech_mask = df[sector_col].str.contains("Technology|Information", case=False, na=False)
    if tech_mask.any():
        # R&D Intensity
        if "r_d_expenses" in df.columns and "revenue" in df.columns:
            result.loc[tech_mask, "r_d_intensity"] = (
                _safe_div(df.loc[tech_mask, "r_d_expenses"], df.loc[tech_mask, "revenue"]) * 100
            )

        # SG&A Efficiency
        if "sga_expenses" in df.columns and "revenue" in df.columns:
            result.loc[tech_mask, "sga_efficiency"] = (
                _safe_div(df.loc[tech_mask, "sga_expenses"], df.loc[tech_mask, "revenue"]) * 100
            )

        # Rule of 40 (Growth + Margin)
        if "revenue_growth_yoy" in df.columns and "operating_margin_pct" in df.columns:
            result.loc[tech_mask, "rule_of_40"] = (
                df.loc[tech_mask, "revenue_growth_yoy"] + df.loc[tech_mask, "operating_margin_pct"]
            )

        # Cash Burn Rate
        if "operating_cash_flow" in df.columns and "capex" in df.columns:
            result.loc[tech_mask, "cash_burn_rate"] = (
                df.loc[tech_mask, "operating_cash_flow"] - df.loc[tech_mask, "capex"]
            )

    # Healthcare sector features
    health_mask = df[sector_col].str.contains("Health", case=False, na=False)
    if health_mask.any():
        # R&D intensity for healthcare
        if "r_d_expenses" in df.columns and "revenue" in df.columns:
            result.loc[health_mask, "r_d_intensity"] = (
                _safe_div(df.loc[health_mask, "r_d_expenses"], df.loc[health_mask, "revenue"]) * 100
            )

    # Consumer sector features (Consumer Discretionary, Consumer Staples)
    consumer_mask = df[sector_col].str.contains("Consumer", case=False, na=False)
    if consumer_mask.any():
        # Inventory Days
        if "inventory" in df.columns and "cost_of_goods_sold" in df.columns:
            result.loc[consumer_mask, "inventory_days"] = (
                _safe_div(
                    df.loc[consumer_mask, "inventory"], df.loc[consumer_mask, "cost_of_goods_sold"]
                )
                * 365
            )

        # Marketing Efficiency
        if "marketing_expenses" in df.columns and "revenue" in df.columns:
            result.loc[consumer_mask, "marketing_efficiency"] = (
                _safe_div(
                    df.loc[consumer_mask, "marketing_expenses"], df.loc[consumer_mask, "revenue"]
                )
                * 100
            )

    # Industrials sector features
    industrials_mask = df[sector_col].str.contains("Industrial", case=False, na=False)
    if industrials_mask.any():
        # CAPEX Intensity
        if "capex" in df.columns and "revenue" in df.columns:
            result.loc[industrials_mask, "capex_intensity"] = (
                _safe_div(df.loc[industrials_mask, "capex"], df.loc[industrials_mask, "revenue"])
                * 100
            )

        # CAPEX to Depreciation ratio
        if "capex" in df.columns and "depreciation_amortization" in df.columns:
            result.loc[industrials_mask, "capex_to_depreciation"] = _safe_div(
                df.loc[industrials_mask, "capex"],
                df.loc[industrials_mask, "depreciation_amortization"],
            )

        # Working Capital Efficiency
        if all(col in df.columns for col in ["current_assets", "current_liabilities", "revenue"]):
            working_capital = (
                df.loc[industrials_mask, "current_assets"]
                - df.loc[industrials_mask, "current_liabilities"]
            )
            result.loc[industrials_mask, "working_capital_efficiency"] = (
                _safe_div(working_capital, df.loc[industrials_mask, "revenue"]) * 100
            )

    # Utilities sector features
    utilities_mask = df[sector_col].str.contains("Utilities", case=False, na=False)
    if utilities_mask.any():
        # Dividend Payout Ratio
        if "dividends_paid" in df.columns and "net_income" in df.columns:
            result.loc[utilities_mask, "dividend_payout_ratio"] = (
                _safe_div(
                    df.loc[utilities_mask, "dividends_paid"], df.loc[utilities_mask, "net_income"]
                )
                * 100
            )

    logger.info(f"Engineered sector-specific features")
    return result


def engineer_temporal_features(
    df: pd.DataFrame, date_col: str = "next_earnings", reference_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """Engineer temporal features from date columns.

    Args:
        df: Input DataFrame
        date_col: Name of date column to extract features from
        reference_date: Optional reference date for calculating days since

    Returns:
        DataFrame with temporal features added
    """
    result = df.copy()

    if date_col not in df.columns:
        logger.warning(f"Date column '{date_col}' not found, skipping temporal features")
        return result

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
        try:
            result[date_col] = pd.to_datetime(result[date_col])
        except Exception as e:
            logger.warning(f"Could not convert {date_col} to datetime: {e}")
            return result

    # Extract fiscal quarter (1-4)
    result["fiscal_quarter"] = result[date_col].dt.quarter

    # Extract month (1-12)
    result["month"] = result[date_col].dt.month

    # Extract year
    result["year"] = result[date_col].dt.year

    # Days since reference date
    if reference_date is not None:
        result["days_since_reference"] = (result[date_col] - reference_date).dt.days

    logger.info(f"Engineered temporal features from {date_col}")
    return result


def engineer_market_microstructure_features(
    df: pd.DataFrame,
    price_col: str = "last_price",
    high_col: str = "high_price",
    low_col: str = "low_price",
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
        features = ["market_cap", "p_e_ratio", "roe", "debt_to_equity", "revenue_growth_yoy"]
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
    """Engineer analyst quality and coverage features.

    Features computed:
    - Analyst coverage (number of analysts)
    - Price target spread (high - low / median)
    - Price target consensus strength
    - Rating distribution features
    - Target vs current price deviation

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with analyst quality features added
    """
    result = df.copy()

    # Analyst coverage
    if "price_target_" in df.columns:
        result["analyst_coverage"] = df["price_target_"].fillna(0)

    # Price target spread (indicates consensus uncertainty)
    if all(
        col in df.columns
        for col in ["price_target_high", "price_target_low", "price_target_median"]
    ):
        target_range = df["price_target_high"] - df["price_target_low"]
        result["price_target_spread_pct"] = _safe_div(target_range, df["price_target_median"]) * 100

    # Consensus strength (low spread = high consensus)
    if "price_target_spread_pct" in result.columns:
        result["consensus_strength"] = 100 - result["price_target_spread_pct"].clip(upper=100)

    # Rating distribution features
    if all(
        col in df.columns
        for col in [
            "_strong_buy_ratings",
            "_buy_ratings",
            "_hold_ratings",
            "_sell_ratings",
            "_strong_sell_ratings",
        ]
    ):
        total_ratings = (
            df["_strong_buy_ratings"].fillna(0)
            + df["_buy_ratings"].fillna(0)
            + df["_hold_ratings"].fillna(0)
            + df["_sell_ratings"].fillna(0)
            + df["_strong_sell_ratings"].fillna(0)
        )

        # Bullish sentiment (Strong Buy + Buy) / Total
        bullish_ratings = df["_strong_buy_ratings"].fillna(0) + df["_buy_ratings"].fillna(0)
        result["analyst_bullish_pct"] = _safe_div(bullish_ratings, total_ratings) * 100

        # Bearish sentiment (Sell + Strong Sell) / Total
        bearish_ratings = df["_sell_ratings"].fillna(0) + df["_strong_sell_ratings"].fillna(0)
        result["analyst_bearish_pct"] = _safe_div(bearish_ratings, total_ratings) * 100

    # Target vs current price deviation
    if "price_target" in df.columns and "last_price" in df.columns:
        result["target_price_upside_pct"] = (
            _safe_div((df["price_target"] - df["last_price"]), df["last_price"]) * 100
        )

    logger.info("Engineered analyst quality features")
    return result


def engineer_accounting_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer accounting quality and red flag features.

    Features computed:
    - Exceptional items flags (goodwill impairment, asset writedowns, restructuring)
    - Accounting red flags (large one-time items)
    - Goodwill to assets ratio
    - Intangibles to assets ratio
    - Working capital quality

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with accounting quality features added
    """
    result = df.copy()

    # Goodwill impairment flag (red flag if present)
    if "impairment_of_goodwill_ltm" in df.columns:
        result["has_goodwill_impairment"] = (
            df["impairment_of_goodwill_ltm"].fillna(0) != 0
        ).astype(int)

    # Asset writedown flag
    if "asset_writedown_ltm" in df.columns:
        result["has_asset_writedown"] = (df["asset_writedown_ltm"].fillna(0) != 0).astype(int)

    # Restructuring charges flag
    if "restructuring_charges_ltm" in df.columns:
        result["has_restructuring"] = (df["restructuring_charges_ltm"].fillna(0) != 0).astype(int)

    # Goodwill to total assets ratio (high ratio can be risky)
    if "goodwill_ltm" in df.columns and "total_assets_ltm" in df.columns:
        result["goodwill_to_assets_pct"] = (
            _safe_div(df["goodwill_ltm"], df["total_assets_ltm"]) * 100
        )

    # Intangibles intensity
    if "intangible_assets" in df.columns and "total_assets_ltm" in df.columns:
        result["intangibles_to_assets_pct"] = (
            _safe_div(df["intangible_assets"], df["total_assets_ltm"]) * 100
        )

    # Exceptional items as % of net income (quality check)
    if all(
        col in df.columns
        for col in [
            "impairment_of_goodwill_ltm",
            "asset_writedown_ltm",
            "restructuring_charges_ltm",
            "net_income_ltm",
        ]
    ):
        exceptional_items = (
            df["impairment_of_goodwill_ltm"].fillna(0).abs()
            + df["asset_writedown_ltm"].fillna(0).abs()
            + df["restructuring_charges_ltm"].fillna(0).abs()
        )
        result["exceptional_items_to_ni_pct"] = (
            _safe_div(exceptional_items, df["net_income_ltm"].abs()) * 100
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


def engineer_employee_productivity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer employee productivity and efficiency features.

    Features computed:
    - Revenue per employee
    - Profit per employee
    - Assets per employee
    - EBITDA per employee
    - Employee growth trends

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with employee productivity features added
    """
    result = df.copy()

    # Check for employee data
    employee_col = None
    for col in ["avg_employees_ltm", "avg_employees_fy", "employees"]:
        if col in df.columns:
            employee_col = col
            break

    if employee_col is None:
        logger.warning("No employee data found, skipping employee productivity features")
        return result

    employees = df[employee_col]

    # Revenue per employee
    if "total_revenues_ltm" in df.columns:
        result["revenue_per_employee"] = _safe_div(df["total_revenues_ltm"], employees)

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

    # Employee growth (if historical data available)
    if "avg_employees_ltm" in df.columns and "avg_employees_fy" in df.columns:
        result["employee_growth_yoy_pct"] = (
            _safe_div((df["avg_employees_ltm"] - df["avg_employees_fy"]), df["avg_employees_fy"])
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
            _safe_div((result["revenue_per_employee"] - avg_5y_rev_per_emp), avg_5y_rev_per_emp)
            * 100
        )

    logger.info("Engineered employee productivity features")
    return result


def build_comprehensive_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_relative_values: bool = True,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build comprehensive feature set by applying all feature engineering functions.

    This orchestrator function applies all available feature engineering in sequence:
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

    Returns:
        DataFrame with comprehensive engineered features

    Example:
        >>> from finance_ml.ml_workflow.features.advanced import build_comprehensive_features
        >>> features_df = build_comprehensive_features(
        ...     raw_data,
        ...     include_interactions=True,
        ...     include_relative_values=True,
        ...     sector_col="sector"
        ... )
    """
    result = df.copy()

    # Apply all feature engineering functions in sequence
    result = engineer_valuation_ratios(result)
    result = engineer_profitability_ratios(result)
    result = engineer_leverage_ratios(result)
    result = engineer_liquidity_ratios(result)
    result = engineer_efficiency_ratios(result)
    result = engineer_growth_metrics(result)
    result = engineer_sector_specific_features(result, sector_col=sector_col)
    result = engineer_analyst_quality_features(result)
    result = engineer_accounting_quality_features(result)
    result = engineer_employee_productivity_features(result)

    # Temporal features (if date column exists)
    if "next_earnings" in result.columns:
        result = engineer_temporal_features(result, date_col="next_earnings")

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

    logger.info(
        f"Built comprehensive features: {len(result.columns)} total features "
        f"({len(result.columns) - len(df.columns)} new features added)"
    )
    return result
