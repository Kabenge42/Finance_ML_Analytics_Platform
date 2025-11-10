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

    # Adjustment ratios (adj/LTM) as robustness/quality proxies
    if "ebitda_adj_ltm" in df.columns and "ebitda_ltm" in df.columns:
        result["ebitda_adjustment_ratio"] = _safe_div(
            df["ebitda_adj_ltm"].abs(), df["ebitda_ltm"].abs()
        )
    if "ebit_adj_ltm" in df.columns and "ebit_ltm" in df.columns:
        result["ebit_adjustment_ratio"] = _safe_div(df["ebit_adj_ltm"].abs(), df["ebit_ltm"].abs())

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
    """Engineer temporal and seasonality features.

    Adds:
    - fiscal_quarter, month, year from date_col
    - days_since_reference if reference_date provided
    - days_to_earnings: (next_earnings - last_updated).days when both present
    - earnings_report_recency: (reference_date - last_updated).days if both provided
    - reporting_lag: (last_updated - income_statement_report_date).days when both present
    - ltm_vs_5yavg_revenue: (total_revenues_ltm - 5Y avg)/5Y avg
    - fq_vs_5yavg_ebitda: (ebitda_fq - ebitda_5yavgfq)/ebitda_5yavgfq
    - quarterly_volatility_score: coefficient of variation across available quarterly EBITDA columns
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

    if result[date_col].isna().all():
        logger.warning(f"All values in '{date_col}' are NaT. Skipping temporal features.")
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

    # Additional earnings/reporting timing features
    if "next_earnings" in result.columns and "last_updated" in result.columns:
        ne = pd.to_datetime(result["next_earnings"], errors="coerce")
        lu = pd.to_datetime(result["last_updated"], errors="coerce")
        result["days_to_earnings"] = (ne - lu).dt.days
    if reference_date is not None and "last_updated" in result.columns:
        lu = pd.to_datetime(result["last_updated"], errors="coerce")
        result["earnings_report_recency"] = (reference_date - lu).dt.days
    if "income_statement_report_date" in result.columns and "last_updated" in result.columns:
        isrd = pd.to_datetime(result["income_statement_report_date"], errors="coerce")
        lu = pd.to_datetime(result["last_updated"], errors="coerce")
        result["reporting_lag"] = (lu - isrd).dt.days

    # Seasonality vs 5Y averages
    rev_5y_cols = [
        c
        for c in ("total_revenues_5yavg", "total_revenues_5yavgfq", "revenue_5yavg")
        if c in result.columns
    ]
    if "total_revenues_ltm" in result.columns and rev_5y_cols:
        base = result[rev_5y_cols[0]].astype(float)
        result["ltm_vs_5yavg_revenue"] = _safe_div(
            result["total_revenues_ltm"].astype(float) - base, base
        )

    if "ebitda_fq" in result.columns and "ebitda_5yavgfq" in result.columns:
        base = result["ebitda_5yavgfq"].astype(float)
        result["fq_vs_5yavg_ebitda"] = _safe_div(result["ebitda_fq"].astype(float) - base, base)

    # Quarterly volatility score (across available quarterly EBITDA columns)
    quarterly_cols = [c for c in result.columns if c.startswith("ebitda_fq_q")]
    if quarterly_cols:
        qmat = pd.concat([result[c].astype(float) for c in quarterly_cols], axis=1)
        mean = qmat.mean(axis=1)
        std = qmat.std(axis=1, ddof=0)
        result["quarterly_volatility_score"] = _safe_div(std, mean)

    logger.info(f"Engineered temporal features from {date_col}")
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

    # Moving averages and signals using daily history + last_price
    def compute_ma_row(row: pd.Series, window: int) -> float:
        """Compute moving average for a single row over specified window."""
        vals = []
        for d in range(window - 1, 0, -1):
            col = f"price_{d}d_ago"
            vals.append(row.get(col, np.nan))
        vals.append(row.get("last_price", np.nan))
        arr = np.asarray(vals, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        return float(np.mean(arr))

    ma20 = (
        df.apply(lambda r: compute_ma_row(r, 20), axis=1)
        if "last_price" in df.columns
        else pd.Series([np.nan] * len(df))
    )
    ma50 = (
        df.apply(lambda r: compute_ma_row(r, 50), axis=1)
        if "last_price" in df.columns
        else pd.Series([np.nan] * len(df))
    )

    if isinstance(ma20, pd.Series):
        result["ma_20d_simple"] = ma20
    if isinstance(ma50, pd.Series):
        result["ma_50d_simple"] = ma50

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
        "strong_buy_ratings",
        "buy_ratings",
        "hold_ratings",
        "sell_ratings",
        "strong_sell_ratings",
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
        for c in ["impairment_of_goodwill_ltm", "asset_writedown_ltm", "restructuring_charges_ltm"]
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
        for c in ["impairment_of_goodwill_1fy", "asset_writedown_1fy", "restructuring_charges_1fy"]
    ):
        exceptional_items_1fy = (
            df["impairment_of_goodwill_1fy"].fillna(0).abs()
            + df["asset_writedown_1fy"].fillna(0).abs()
            + df["restructuring_charges_1fy"].fillna(0).abs()
        )
        if "total_exceptional_items_ltm" in result.columns:
            result["exceptional_items_trend"] = _safe_div(
                result["total_exceptional_items_ltm"] - exceptional_items_1fy, exceptional_items_1fy
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
    - Uses normalized column names (e.g., cfo_ltm, net_income_ltm, fcf_ltm, total_revenues_ltm)
    - Safe divisions via _safe_div; returns NaNs when inputs missing.
    """
    result = df.copy()

    # Core ratios
    if "cfo_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["cfo_to_net_income"] = _safe_div(df["cfo_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["fcf_to_net_income"] = _safe_div(df["fcf_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "total_revenues_ltm" in df.columns:
        result["fcf_margin"] = _safe_div(df["fcf_ltm"], df["total_revenues_ltm"])

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
    if "capital_expenditure_ltm" in df.columns and "total_revenues_ltm" in df.columns:
        result["capex_intensity"] = _safe_div(
            df["capital_expenditure_ltm"], df["total_revenues_ltm"]
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
        for c in ("capital_expenditure_ltm", "capital_expenditure_fy", "capital_expenditure_1fy")
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
    if "total_revenues_ltm" in df.columns and "working_capital_ltm" in df.columns:
        result["working_capital_efficiency"] = _safe_div(
            df["total_revenues_ltm"], df["working_capital_ltm"]
        )
    if all(
        c in df.columns
        for c in ["working_capital_ltm", "working_capital_1fy", "total_revenues_ltm"]
    ):
        result["working_capital_trend"] = _safe_div(
            df["working_capital_ltm"] - df["working_capital_1fy"], df["total_revenues_ltm"]
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


def engineer_margin_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer profitability margin trends and quality metrics.

    Features computed (when inputs exist):
    - ebitda_margin_trend: (ebitda_ltm/total_revenues_ltm) - (ebitda_1fy/total_revenues_1fy)
    - gross_margin_trend: (gross_profit_ltm/total_revenues_ltm) - (gross_profit_fy/revenue_fy)
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
        for c in ("ebitda_ltm", "total_revenues_ltm", "ebitda_1fy", "total_revenues_1fy")
    ):
        cur = _safe_div(df["ebitda_ltm"].astype(float), df["total_revenues_ltm"].astype(float))
        prev = _safe_div(df["ebitda_1fy"].astype(float), df["total_revenues_1fy"].astype(float))
        result["ebitda_margin_trend"] = cur - prev

    # Gross margin trend (FY reference for previous)
    if all(
        c in df.columns
        for c in ("gross_profit_ltm", "total_revenues_ltm", "gross_profit_fy", "revenue_fy")
    ):
        cur = _safe_div(
            df["gross_profit_ltm"].astype(float), df["total_revenues_ltm"].astype(float)
        )
        prev = _safe_div(df["gross_profit_fy"].astype(float), df["revenue_fy"].astype(float))
        result["gross_margin_trend"] = cur - prev

    # Operating leverage = (%ΔEBIT)/(%ΔRevenue)
    if all(
        c in df.columns
        for c in ("ebit_ltm", "ebit_1fy", "total_revenues_ltm", "total_revenues_1fy")
    ):
        delta_ebit = _safe_div(
            df["ebit_ltm"].astype(float) - df["ebit_1fy"].astype(float),
            df["ebit_1fy"].astype(float),
        )
        delta_rev = _safe_div(
            df["total_revenues_ltm"].astype(float) - df["total_revenues_1fy"].astype(float),
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
    - short_interest_ratio: Pass-through of short_int_pct (already percent units)
    - beta_stability: Population variance (ddof=0) across available betas (beta_1y, beta_2y, beta_5y)
    - systematic_risk_trend: beta_1y - beta_5y (risk profile change)

    Args:
        df: Input DataFrame with normalized column names

    Returns:
        DataFrame with market sentiment features added
    """
    result = df.copy()

    # Short interest (percent already)
    if "short_int_pct" in df.columns:
        result["short_interest_ratio"] = df["short_int_pct"].astype(float)

    # Beta metrics
    beta_cols = [c for c in ("beta_1y", "beta_2y", "beta_5y") if c in df.columns]
    if beta_cols:
        beta_mat = df[beta_cols].astype(float)
        # Population variance across the provided beta horizons
        result["beta_stability"] = beta_mat.var(axis=1, ddof=0)

    if "beta_1y" in df.columns and "beta_5y" in df.columns:
        result["systematic_risk_trend"] = df["beta_1y"].astype(float) - df["beta_5y"].astype(float)

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
            df["cash_and_equivalents"].astype(float), df["current_liabilities"].astype(float)
        )
    if all(c in df.columns for c in ("working_capital_ltm", "total_assets_ltm")):
        result["working_capital_ratio"] = _safe_div(
            df["working_capital_ltm"].astype(float), df["total_assets_ltm"].astype(float)
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

    Minimal implementation focusing on quality score per plan:
    - composite_quality_score: mean of available {distress_risk_score, accounting_quality_score}.
    - momentum_score: optional, if price_momentum_1y and return_stability_score are present, scaled to 0-100.
    - value_score: optional placeholder using inverse of p_e_ratio percentile within sector if available (not used in tests).
    """
    result = df.copy()

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

    logger.info("Engineered composite scores")
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
