"""
finance_ml.advanced_features - Advanced feature engineering for Phase 9.3

This module implements sophisticated feature engineering techniques including:
- Comprehensive financial ratios (valuation, profitability, leverage, liquidity, efficiency, growth)
- Sector-specific features for all major sectors
- Feature interactions and polynomial features
- Target encoding and relative value features
- Automated feature selection frameworks

Part of Phase 9.3 implementation.
"""

from __future__ import annotations

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression, RFE, RFECV
from sklearn.linear_model import Ridge

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
    "calculate_feature_importance_mutual_info",
    "calculate_feature_importance_rf",
    "calculate_feature_importance_shap",
    "calculate_feature_importance_rfe",
    "build_comprehensive_features",
]


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf/NaN with 0 or appropriate value.

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
    df: pd.DataFrame, date_col: str = "report_date", reference_date: Optional[pd.Timestamp] = None
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


def calculate_feature_importance_mutual_info(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None
) -> pd.DataFrame:
    """Calculate feature importance using mutual information.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    # Handle missing and invalid values
    X_clean = X.copy()
    
    # Replace infinite values with NaN first
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN with median
    X_clean = X_clean.fillna(X_clean.median())
    
    # For any remaining NaN, fill with 0
    X_clean = X_clean.fillna(0)
    
    # Clip extreme values
    for col in X_clean.columns:
        col_data = X_clean[col]
        if col_data.std() > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_clean[col] = np.clip(col_data, lower_bound, upper_bound)
    
    # Handle target variable
    y_clean = y.replace([np.inf, -np.inf], np.nan)
    y_clean = y_clean.fillna(y_clean.median())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(y_clean.mean())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(0)

    # Calculate mutual information
    mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)

    # Create result DataFrame
    importance_df = pd.DataFrame({"feature": X.columns, "importance": mi_scores}).sort_values(
        "importance", ascending=False
    )

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated mutual information for {len(X.columns)} features")
    return importance_df


def calculate_feature_importance_rf(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None, n_estimators: int = 100
) -> pd.DataFrame:
    """Calculate feature importance using Random Forest.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)
        n_estimators: Number of trees in Random Forest

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    # Handle missing and invalid values
    X_clean = X.copy()

    # Replace infinite values with NaN first
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    X_clean = X_clean.fillna(X_clean.median())

    # For any remaining NaN (e.g., all-NaN columns), fill with 0
    X_clean = X_clean.fillna(0)

    # Clip extremely large values to prevent overflow
    # Use 3 standard deviations as threshold for each column
    for col in X_clean.columns:
        col_data = X_clean[col]
        if col_data.std() > 0:
            mean_val = col_data.mean()
            std_val = col_data.std()
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_clean[col] = np.clip(col_data, lower_bound, upper_bound)

    # Handle target variable
    y_clean = y.replace([np.inf, -np.inf], np.nan)
    y_clean = y_clean.fillna(y_clean.median())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(y_clean.mean())
    if y_clean.isna().any():
        y_clean = y_clean.fillna(0)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_clean, y_clean)

    # Get feature importance
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated Random Forest importance for {len(X.columns)} features")
    return importance_df


def calculate_feature_importance_shap(
    X: pd.DataFrame, y: pd.Series, top_k: Optional[int] = None, n_estimators: int = 50
) -> pd.DataFrame:
    """Calculate feature importance using SHAP values.

    Args:
        X: Feature DataFrame
        y: Target variable
        top_k: Return only top k features (default: all)
        n_estimators: Number of trees for tree-based model

    Returns:
        DataFrame with features and importance scores, sorted by importance
    """
    # Clean data
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

    # Train a simple model for SHAP
    model = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42, max_depth=5, n_jobs=-1
    )
    model.fit(X_clean, y_clean)

    try:
        # Try to use SHAP if available
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean)

        # Calculate mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)
    except ImportError:
        # Fallback to Random Forest feature importance if SHAP not available
        logger.warning("SHAP not available, falling back to Random Forest feature importance")
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

    if top_k is not None:
        importance_df = importance_df.head(top_k)

    logger.info(f"Calculated SHAP importance for {len(X.columns)} features")
    return importance_df


def calculate_feature_importance_rfe(
    X: pd.DataFrame, y: pd.Series, n_features_to_select: int = 10, cv: Optional[int] = None
) -> List[str]:
    """Select features using Recursive Feature Elimination.

    Args:
        X: Feature DataFrame
        y: Target variable
        n_features_to_select: Number of features to select
        cv: Number of cross-validation folds (uses RFECV if provided)

    Returns:
        List of selected feature names
    """
    # Clean data
    X_clean = X.copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_clean = y.replace([np.inf, -np.inf], np.nan).fillna(y.median())

    # Use Ridge regression as the base estimator (fast and stable)
    estimator = Ridge(alpha=1.0, random_state=42)

    if cv is not None and cv > 1:
        # Use RFECV for cross-validated selection
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            min_features_to_select=max(1, n_features_to_select // 2),
        )
    else:
        # Use RFE for simple selection
        selector = RFE(
            estimator=estimator,
            n_features_to_select=min(n_features_to_select, len(X.columns)),
            step=1,
        )

    # Fit and get selected features
    selector.fit(X_clean, y_clean)
    selected_features = X.columns[selector.support_].tolist()

    logger.info(f"RFE selected {len(selected_features)} features from {len(X.columns)}")
    return selected_features


def build_comprehensive_features(
    df: pd.DataFrame,
    include_interactions: bool = True,
    include_relative_values: bool = True,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build comprehensive feature set with all advanced features.

    This is the main function that orchestrates all feature engineering steps.

    Args:
        df: Input DataFrame
        include_interactions: Whether to include interaction features
        include_relative_values: Whether to include relative value features
        sector_col: Name of sector column

    Returns:
        DataFrame with all engineered features
    """
    logger.info("Building comprehensive feature set...")

    result = df.copy()

    # Step 1: Core financial ratios
    result = engineer_valuation_ratios(result)
    result = engineer_profitability_ratios(result)
    result = engineer_leverage_ratios(result)
    result = engineer_liquidity_ratios(result)
    result = engineer_efficiency_ratios(result)
    result = engineer_growth_metrics(result)

    # Step 2: Sector-specific features
    result = engineer_sector_specific_features(result, sector_col=sector_col)

    # Step 3: Interaction features (optional)
    if include_interactions:
        result = create_feature_interactions(result)

    # Step 4: Relative value features (optional)
    if include_relative_values:
        result = create_relative_value_features(result, sector_col=sector_col)

    n_new_features = len(result.columns) - len(df.columns)
    logger.info(f"✓ Feature engineering complete: Added {n_new_features} new features")

    return result
