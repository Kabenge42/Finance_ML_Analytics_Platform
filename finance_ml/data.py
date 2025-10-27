"""
finance_ml.data - Data loading, normalization, and validation

This module provides functions for loading equity data from CSV files or databases,
normalizing column names, and validating data quality.

Functions extracted from ml_finance_model_v8_2.py as part of Phase 7 refactoring.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

# Optional import for database access
try:
    from sqlalchemy import create_engine  # type: ignore
except ImportError:  # pragma: no cover
    create_engine = None  # type: ignore


def setup_logging(level: str = "INFO") -> None:
    """Configure basic logging for the module.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get environment variable with optional default value.

    Args:
        name: Environment variable name
        default: Default value if variable not set

    Returns:
        Environment variable value (stripped) or default
    """
    v = os.environ.get(name, default)
    if v is not None and isinstance(v, str):
        v = v.strip()
    return v


def normalize_columns(df: pd.DataFrame, preserve_schema: bool = True) -> pd.DataFrame:
    """Normalize DataFrame column names to match database schema or lowercase with underscores.

    When preserve_schema=True, maps database column names to lowercase underscore format
    while maintaining consistency. When preserve_schema=False, converts to lowercase
    with underscores (legacy behavior).

    Args:
        df: Input DataFrame
        preserve_schema: If True, apply database schema column name mapping (default: True)

    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()

    if preserve_schema:
        # Create mapping from database schema column names to normalized names
        schema_mapping = {
            "Ticker": "ticker",
            "ISIN": "isin",
            "Name": "name",
            "Description": "description",
            "Exchange": "exchange",
            "Unit": "unit",
            "Sector": "sector",
            "Industry": "industry",
            "Last Updated": "last_updated",
            "Income Statement Report Date": "income_statement_report_date",
            "Next Earnings": "next_earnings",
            "Style Class": "style_class",
            "Next Earnings (Status)": "next_earnings_status",
            "Size Class": "size_class",
            "Flag": "flag",
            "Region": "region",
            "Country": "country",
            "Trading Country": "trading_country",
            "Market Cap": "market_cap",
            "Enterprise Value": "enterprise_value",
            "Last Price": "last_price",
            "Price Target (YTD Ago)": "price_target_ytd_ago",
            "Total Return (YTD)": "total_return_ytd",
            "Price Target": "price_target",
            "Price Target - Low": "price_target_low",
            "Price Target - Median": "price_target_median",
            "Price Target - High": "price_target_high",
            "Price Target - #": "price_target_count",
            "P/E (NTM)": "p_e_ntm",
            "P/E (LTM)": "p_e_ltm",
            "Altman Z-Score (FY)": "altman_z_score_fy",
            "Altman Z-Score (FQ)": "altman_z_score_fq",
            "Altman Z-Score (LTM)": "altman_z_score_ltm",
            "Beta (1Y)": "beta_1y",
            "Beta (2Y)": "beta_2y",
            "Beta (5Y)": "beta_5y",
            "Analyst Rating": "analyst_rating",
            "# Strong Sell Ratings": "strong_sell_ratings",
            "# Strong Buys Ratings": "strong_buys_ratings",
            "# Hold Ratings": "hold_ratings",
            "# Buys Ratings": "buys_ratings",
            "# Sell Ratings": "sell_ratings",
            "Total Revenues/CAGR (5Y FY)": "total_revenues_cagr_5y_fy",
            "Total Revenues (FQ)": "total_revenues_fq",
            "Total Revenues (-1FY)": "total_revenues_1fy",
            "Total Revenues (FY)": "total_revenues_fy",
            "Total Revenues (LTM)": "total_revenues_ltm",
            "Total Operating Expenses (LTM)": "total_operating_expenses_ltm",
            "P/TBV (LTM)": "p_tbv_ltm",
            "TBV (FY)": "tbv_fy",
            "TBV (LTM)": "tbv_ltm",
            "Market Cap (Country R)": "market_cap_country_r",
            "Tot. Return %/CAGR (3Y)": "tot_return_pct_cagr_3y",
            "Tot. Return %/CAGR (10Y)": "tot_return_pct_cagr_10y",
            "Total Return (5Y)": "total_return_5y",
            "Total Return (10Y)": "total_return_10y",
            "Net Income/Adj. (-1FY)": "net_income_adj_1fy",
            "CFF (LTM)": "cff_ltm",
            "CFI (LTM)": "cfi_ltm",
            "FCF (LTM)": "fcf_ltm",
            "CFO (LTM)": "cfo_ltm",
            "EBITDA (FQ)": "ebitda_fq",
            "EBITDA (LTM)": "ebitda_ltm",
            "EBITDA (FY)": "ebitda_fy",
            "EBITDA (-1FY)": "ebitda_1fy",
            "EBITDA/Adj. (LTM)": "ebitda_adj_ltm",
            "EBITDA/Adj. (FY)": "ebitda_adj_fy",
            "EBITDA/Adj. (-1FY)": "ebitda_adj_1fy",
            "EBIT (FQ)": "ebit_fq",
            "EBIT (LTM)": "ebit_ltm",
            "EBIT (FY)": "ebit_fy",
            "EBIT (-1FY)": "ebit_1fy",
            "EBIT/Adj. (-1FY)": "ebit_adj_1fy",
            "EBIT/Adj. (FY)": "ebit_adj_fy",
            "EBIT/Adj. (LTM)": "ebit_adj_ltm",
            "EBIT - Est Med (FY1E)": "ebit_est_med_fy1e",
            "EBIT - Est Med (NTM)": "ebit_est_med_ntm",
            "Return On Equity % (LTM)": "return_on_equity_pct_ltm",
            "Return On Equity % (FY)": "return_on_equity_pct_fy",
            "Net Income - (IS) (FY)": "net_income_is_fy",
            "Net Income - (IS) (LTM)": "net_income_is_ltm",
            "Normalized Net Income (FY)": "normalized_net_income_fy",
            "Normalized Net Income (LTM)": "normalized_net_income_ltm",
            "Net Income/Adj. (FY)": "net_income_adj_fy",
            "Net Income/Adj. (LTM)": "net_income_adj_ltm",
            "Net Income Margin % (FY)": "net_income_margin_pct_fy",
            "Net Income Margin % (LTM)": "net_income_margin_pct_ltm",
            "Volatility (1M)": "volatility_1m",
            "Volatility (3M)": "volatility_3m",
            "Volatility (6M)": "volatility_6m",
            "Volatility (1Y)": "volatility_1y",
            "Volume (Shrs)": "volume_shrs",
            "Short Int. (%)": "short_int_pct",
            "Dividend Per Share (LTM)": "dividend_per_share_ltm",
            "Div Yield (Ind)": "div_yield_ind",
            "Div Yield (LTM)": "div_yield_ltm",
            "Total Debt (FY)": "total_debt_fy",
            "Total Equity (FY)": "total_equity_fy",
            "Total Equity (LTM)": "total_equity_ltm",
            "Total Debt (LTM)": "total_debt_ltm",
            "Total Assets (LTM)": "total_assets_ltm",
            "Total Assets (FY)": "total_assets_fy",
            "Current Ratio (FY)": "current_ratio_fy",
            "Current Ratio (LTM)": "current_ratio_ltm",
            "Gross Profit Margin % (FY)": "gross_profit_margin_pct_fy",
            "Gross Profit Margin % (LTM)": "gross_profit_margin_pct_ltm",
            "Asset Turnover (FY)": "asset_turnover_fy",
            "Asset Turnover (LTM)": "asset_turnover_ltm",
            "Gross Profit (LTM)": "gross_profit_ltm",
            "Gross Profit (FY)": "gross_profit_fy",
            "EPS Norm - Est Avg (NTM)": "eps_norm_est_avg_ntm",
            "EPS/Adj. (-1FY)": "eps_adj_1fy",
            "EPS/Adj. (FY)": "eps_adj_fy",
            "EPS/Adj. (LTM)": "eps_adj_ltm",
            "EPS Norm - Est Avg (FY1E)": "eps_norm_est_avg_fy1e",
            "Gain (Loss) On Sale Of Assets (LTM)": "gain_loss_on_sale_of_assets_ltm",
            "Cost Of Revenues (LTM)": "cost_of_revenues_ltm",
            "Cash Acquisitions (LTM)": "cash_acquisitions_ltm",
            "Cash Acquisitions (FY)": "cash_acquisitions_fy",
            "Cash Acquisitions (-1FY)": "cash_acquisitions_1fy",
            "Inventory (LTM)": "inventory_ltm",
            "Goodwill (FQ)": "goodwill_fq",
            "Goodwill (LTM)": "goodwill_ltm",
            "Goodwill (FY)": "goodwill_fy",
            "Goodwill (-1FY)": "goodwill_1fy",
            "Impairment of Goodwill (FQ)": "impairment_of_goodwill_fq",
            "Impairment of Goodwill (LTM)": "impairment_of_goodwill_ltm",
            "Impairment of Goodwill (-1FY)": "impairment_of_goodwill_1fy",
            "Impairment of Goodwill (FY)": "impairment_of_goodwill_fy",
            "Operating Income (LTM)": "operating_income_ltm",
            "Asset Writedown (LTM)": "asset_writedown_ltm",
            "Asset Writedown (FY)": "asset_writedown_fy",
            "Asset Writedown (-1FY)": "asset_writedown_1fy",
            "Operating Income (FY)": "operating_income_fy",
            "Capital Expenditure (LTM)": "capital_expenditure_ltm",
            "Capital Expenditure (-1FY)": "capital_expenditure_1fy",
            "Capital Expenditure (FY)": "capital_expenditure_fy",
            "Retained Earnings (LTM)": "retained_earnings_ltm",
            "Total Current Assets (LTM)": "total_current_assets_ltm",
            "Total Current Liabilities (LTM)": "total_current_liabilities_ltm",
            "R&D Expenses (LTM)": "r_d_expenses_ltm",
            "Restructuring Charges (LTM)": "restructuring_charges_ltm",
            "Restructuring Charges (FQ)": "restructuring_charges_fq",
            "Restructuring Charges (-1FY)": "restructuring_charges_1fy",
            "Restructuring Charges (FY)": "restructuring_charges_fy",
            "Interest Expense/Total (LTM)": "interest_expense_total_ltm",
            "Merger & Restructuring Charges (LTM)": "merger_restructuring_charges_ltm",
            "Working Capital (LTM)": "working_capital_ltm",
            "Other Unusual Items/Total (LTM)": "other_unusual_items_total_ltm",
            "Interest Income On Investments (LTM)": "interest_income_on_investments_ltm",
            "Buyback Yield (LTM)": "buyback_yield_ltm",
            "Return on Assets (ROA) % (LTM)": "return_on_assets_roa_pct_ltm",
            "Return on Assets (ROA) % (FY)": "return_on_assets_roa_pct_fy",
            "Net Income - (IS) (-1FY)": "net_income_is_1fy",
            "Normalized Net Income (-1FY)": "normalized_net_income_1fy",
            "P/E (-1FYLTM)": "p_e_1fyltm",
            "CFF (FY)": "cff_fy",
            "CFF (-1FY)": "cff_1fy",
            "CFI (FY)": "cfi_fy",
            "CFI (-1FY)": "cfi_1fy",
            "CFO (FY)": "cfo_fy",
            "CFO (-1FY)": "cfo_1fy",
            "Div Yield (-1FYInd)": "div_yield_1fyind",
            "FCF (FY)": "fcf_fy",
            "Capital Expenditure (FQ)": "capital_expenditure_fq",
            "Capital Expenditure (5YAVGFQ)": "capital_expenditure_5yavgfq",
            "CFF (FQ)": "cff_fq",
            "CFI (FQ)": "cfi_fq",
            "CFO (FQ)": "cfo_fq",
            "FCF (FQ)": "fcf_fq",
            "Total Revenues (5YAVGFQ)": "total_revenues_5yavgfq",
            "EBITDA (5YAVGFQ)": "ebitda_5yavgfq",
            "EBIT (5YAVGFQ)": "ebit_5yavgfq",
            "P/E (5YAVGLTM)": "p_e_5yavgltm",
            "FCF (5YAVGFQ)": "fcf_5yavgfq",
            "Cash Acquisitions (FQ)": "cash_acquisitions_fq",
            "Cash Acquisitions (5YAVGFQ)": "cash_acquisitions_5yavgfq",
            "Asset Writedown (FQ)": "asset_writedown_fq",
            "Asset Writedown (5YAVGFQ)": "asset_writedown_5yavgfq",
            "Impairment of Goodwill (5YAVGFQ)": "impairment_of_goodwill_5yavgfq",
            "Operating Income (FQ)": "operating_income_fq",
            "Operating Income (5YAVGFQ)": "operating_income_5yavgfq",
            "P/B (LTM)": "p_b_ltm",
            "P/B (-1FY)": "p_b_1fy",
            "P/B (5YAVG)": "p_b_5yavg",
            "Cash And Equivalents (LTM)": "cash_and_equivalents_ltm",
            "Cash And Equivalents (FQ)": "cash_and_equivalents_fq",
            "Cash And Equivalents (FY)": "cash_and_equivalents_fy",
            "Cash And Equivalents (5YAVGFQ)": "cash_and_equivalents_5yavgfq",
            "Inventory (FQ)": "inventory_fq",
            "Inventory (FY)": "inventory_fy",
            "Goodwill (5YAVGFQ)": "goodwill_5yavgfq",
            "Inventory (5YAVGFQ)": "inventory_5yavgfq",
            "Avg Employees (LTM)": "avg_employees_ltm",
            "Avg Employees (FY)": "avg_employees_fy",
            "Avg Employees (5YAVGFY)": "avg_employees_5yavgfy",
            "Retained Earnings (FQ)": "retained_earnings_fq",
            "Retained Earnings (FY)": "retained_earnings_fy",
            "Retained Earnings (5YAVGFQ)": "retained_earnings_5yavgfq",
            "Working Capital (FQ)": "working_capital_fq",
            "Working Capital (FY)": "working_capital_fy",
            "Working Capital (5YAVGFY)": "working_capital_5yavgfy",
            "Div Yield (TTM)": "div_yield_ttm",
            "Div Yield (NTM)": "div_yield_ntm",
            "Div Yield (5YAVGLTM)": "div_yield_5yavgltm",
            "Gross Intangible Assets (LTM)": "gross_intangible_assets_ltm",
            "Gross Intangible Assets (FY)": "gross_intangible_assets_fy",
            "Gross Intangible Assets (5YAVGFQ)": "gross_intangible_assets_5yavgfq",
            "Restructuring Charges (5YAVGFQ)": "restructuring_charges_5yavgfq",
            "Merger & Restructuring Charges (FQ)": "merger_restructuring_charges_fq",
            "Merger & Restructuring Charges (FY)": "merger_restructuring_charges_fy",
            "Merger & Restructuring Charges (5YAVGFQ)": "merger_restructuring_charges_5yavgfq",
            "Normalized Net Income (FQ)": "normalized_net_income_fq",
            "Normalized Net Income (5YAVGFQ)": "normalized_net_income_5yavgfq",
            "Net Income/Adj. (FQ)": "net_income_adj_fq",
            "Net Income/Adj. (5YAVGFQ)": "net_income_adj_5yavgfq",
            "Net Income - (IS) (FQ)": "net_income_is_fq",
            "Net Income - (IS) (5YAVGFQ)": "net_income_is_5yavgfq",
            "Net Income - (IS) (5YAVGLTM)": "net_income_is_5yavgltm",
            "Normalized Net Income (5YAVGLTM)": "normalized_net_income_5yavgltm",
            "EBITDA (5YAVGLTM)": "ebitda_5yavgltm",
            "EBIT (5YAVGLTM)": "ebit_5yavgltm",
            "Total Revenues (5YAVGLTM)": "total_revenues_5yavgltm",
            "Revenues - Est YoY % (FY1E)": "revenues_est_yoy_pct_fy1e",
        }

        # Apply mapping to existing columns
        df.columns = [schema_mapping.get(col, col) for col in df.columns]

        # For any unmapped columns, apply the normalization
        df.columns = [
            (
                col
                if col in schema_mapping.values()
                else str(col).replace(r"[^0-9a-zA-Z]+", "_", regex=False).strip("_").lower()
            )
            for col in df.columns
        ]

        # Add column aliases for commonly used generic names
        # These create additional columns with simplified names pointing to the most relevant variant
        column_aliases = {
            "p_e": "p_e_ltm",  # Default P/E to Last Twelve Months
            "revenue": "total_revenues_ltm",  # Default revenue to LTM
            "ebitda": "ebitda_ltm",  # Default EBITDA to LTM
            "net_income": "net_income_is_ltm",  # Default net income to Income Statement LTM
            "p_b": "p_b_ltm",  # Default P/B to LTM
            "ev_ebitda": None,  # Not directly available, would need to be calculated
            "gross_margin": "gross_profit_margin_pct_ltm",  # Gross margin percentage LTM
        }

        for alias, source_col in column_aliases.items():
            if source_col and source_col in df.columns and alias not in df.columns:
                df[alias] = df[source_col]

    else:
        # Legacy behavior: convert to lowercase with underscores
        df.columns = (
            df.columns.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.strip("_").str.lower()
        )

    return df


def infer_region_from_filename(path: Path) -> Optional[str]:
    """Infer region from CSV filename.

    Detects region based on filename patterns:
    - screening_us.csv -> US
    - screening_eu.csv -> EU
    - screening_apac.csv -> APAC
    - screening_rotw.csv -> ROTW

    Args:
        path: Path to CSV file

    Returns:
        Region code (US, EU, APAC, ROTW) or None if not recognized
    """
    name = path.name.lower()
    if "us" in name:
        return "US"
    if "eu" in name:
        return "EU"
    if "apac" in name:
        return "APAC"
    if "rotw" in name:
        return "ROTW"
    return None


def load_from_csv(data_dir: Path, limit: Optional[int] = None) -> pd.DataFrame:
    """Load equity data from regional CSV files.

    Loads and combines data from four regional CSV files:
    - screening_us.csv
    - screening_eu.csv
    - screening_apac.csv
    - screening_rotw.csv

    Automatically adds 'Region' column based on filename and normalizes
    all column names to lowercase with underscores.

    Args:
        data_dir: Directory containing CSV files
        limit: Optional row limit for loaded data

    Returns:
        Combined DataFrame with normalized columns

    Raises:
        FileNotFoundError: If no CSV files found in directory
    """
    logging.info("Loading CSVs from %s", data_dir)
    csvs = [
        data_dir / "screening_us.csv",
        data_dir / "screening_eu.csv",
        data_dir / "screening_apac.csv",
        data_dir / "screening_rotw.csv",
    ]
    parts: List[pd.DataFrame] = []
    for p in csvs:
        if not p.exists():
            logging.warning("CSV not found: %s", p)
            continue
        df = pd.read_csv(p)
        # Only set Region if not provided
        if "Region" not in df.columns and "region" not in df.columns:
            region = infer_region_from_filename(p)
            if region:
                df["Region"] = region
        parts.append(df)
    if not parts:
        raise FileNotFoundError("No CSV files found in data/ directory.")
    all_df = pd.concat(parts, ignore_index=True)

    # Normalize column names
    all_df = normalize_columns(all_df)

    if limit is not None:
        all_df = all_df.head(limit)
    return all_df


def load_from_db(db_url: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load equity data from PostgreSQL database.

    Connects to PostgreSQL and loads data from the main equities table
    declared in the postgres database (schema: public by default), for all
    regions (US, EU, APAC, ROTW).

    This function expects the database to be initialized with the provided SQL
    scripts in the repository root:
      - create_equities_schema.sql
      - import_equities_data.sql

    Environment overrides:
      - DB_SCHEMA (default: 'public')
      - DB_TABLE (default: 'equities')

    Args:
        db_url: SQLAlchemy database URL (e.g., postgresql+psycopg2://user:pass@host:5432/postgres)
        limit: Optional row limit for loaded data

    Returns:
        DataFrame with normalized columns

    Raises:
        ImportError: If SQLAlchemy not available
    """
    if create_engine is None:
        raise ImportError(
            "SQLAlchemy not available. Install psycopg2-binary and SQLAlchemy or use CSV data source."
        )

    # Resolve schema and table from environment, default to public.equities
    schema = os.environ.get("DB_SCHEMA", "public")
    table = os.environ.get("DB_TABLE", "equities")
    # Fully qualified table reference
    table_ref = f"{schema}.{table}"

    logging.info("Loading from PostgreSQL: %s (table: %s)", db_url, table_ref)
    engine = create_engine(db_url)

    base_query = f"SELECT * FROM {table_ref} WHERE \"Region\" IN ('US','EU','APAC','ROTW')"
    query = base_query if limit is None else f"SELECT * FROM ( {base_query} ) q LIMIT {int(limit)}"

    df = pd.read_sql(query, engine)

    # Normalize column names
    df = normalize_columns(df)

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess DataFrame with basic cleaning and type conversions.

    Steps:
    1. Normalizes column names
    2. Validates schema (lenient mode)
    3. Drops rows with missing ticker, sector, or last_price
    4. Converts object columns to numeric where possible

    Args:
        df: Input DataFrame

    Returns:
        Preprocessed DataFrame
    """
    logging.info("Normalizing column names and basic filtering...")
    df = normalize_columns(df)

    # Validate schema (target optional in general preprocessing)
    validate_schema(df, require_target=False)

    # Basic required fields if present
    for col in ["ticker", "sector", "last_price"]:
        if col in df.columns:
            df = df[~df[col].isna()]

    # Coerce numerics where possible
    numeric_like = df.columns[df.dtypes == object]
    for c in numeric_like:
        # If object but looks numeric, coerce
        coerced = pd.to_numeric(df[c], errors="coerce")
        df[c] = coerced

    return df


def validate_schema(df: pd.DataFrame, require_target: bool = False) -> None:
    """Validate presence of critical columns.

    Args:
        df: DataFrame to validate
        require_target: If True, requires price_target or price_target_median column

    Raises:
        ValueError: If required columns are missing
    """
    cols = set(df.columns)

    # Always validate core columns
    missing = [c for c in ["ticker", "sector", "last_price"] if c not in cols]

    if require_target:
        # Additionally require a target column
        target_candidates = ["price_target", "price_target_median"]
        has_target = any(t in cols for t in target_candidates)

        if missing or not has_target:
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing required columns: {', '.join(missing)}")
            if not has_target:
                msg_parts.append(
                    "and at least one target column among: price_target, price_target_median"
                )
            raise ValueError(" ".join(msg_parts))
    else:
        # Only validate core columns
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")


def check_missing_values(df: pd.DataFrame) -> dict:
    """Check missing values in DataFrame and return detailed report.

    Args:
        df: DataFrame to check

    Returns:
        Dictionary mapping column names to dict with 'count' and 'percentage' keys.
        Only includes columns that have missing values.
    """
    report = {}
    total_rows = len(df)
    if total_rows == 0:
        return report

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        if missing_count > 0:
            percentage = (missing_count / total_rows) * 100.0
            report[col] = {"count": missing_count, "percentage": percentage}
    return report


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> List[int]:
    """Detect outliers in a numeric column using IQR method.

    Args:
        df: DataFrame containing the column
        column: Column name to check for outliers
        multiplier: IQR multiplier for outlier bounds (default: 1.5)

    Returns:
        List of integer indices where outliers are detected
    """
    series = pd.to_numeric(df[column], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (series < lower_bound) | (series > upper_bound)
    outlier_indices = outlier_mask[outlier_mask].index.tolist()
    return outlier_indices


def validate_numeric_ranges(df: pd.DataFrame) -> dict:
    """Validate numeric columns for invalid ranges.

    Checks for negative values in price and market cap columns.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with column names as keys and list of invalid indices as values
        Only includes columns that have validation issues
    """
    issues = {}

    # Check price columns (should be positive)
    price_cols = [c for c in df.columns if "price" in c.lower()]
    for col in price_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = series < 0
            invalid_indices = invalid_mask[invalid_mask].index.tolist()
            if invalid_indices:
                issues[col] = invalid_indices

    # Check market cap columns (should be positive)
    cap_cols = [c for c in df.columns if "market_cap" in c.lower() or "marketcap" in c.lower()]
    for col in cap_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            invalid_mask = series < 0
            invalid_indices = invalid_mask[invalid_mask].index.tolist()
            if invalid_indices:
                issues[col] = invalid_indices

    return issues


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Safely divide two Series, replacing inf with NaN.

    Args:
        numer: Numerator Series
        denom: Denominator Series

    Returns:
        Result Series with inf values replaced by NaN
    """
    result = numer.astype(float) / denom.astype(float)
    # Replace +/- inf with NaN
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def create_sample_financial_dataset(
    n_stocks: int = 100, random_seed: Optional[int] = None
) -> pd.DataFrame:
    """Create a sample financial dataset for demonstration purposes.

    Parameters
    ----------
    n_stocks : int
        Number of pseudo-stocks to generate.
    random_seed : Optional[int]
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic financial dataset with typical columns used by the package.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    sectors = [
        "Information Technology",
        "Healthcare",
        "Financials",
        "Consumer Discretionary",
        "Industrials",
        "Energy",
    ]
    regions = ["US", "EU", "APAC", "ROTW"]

    rows = []
    for i in range(n_stocks):
        last_price = np.random.uniform(10, 500)
        price_target = last_price * np.random.uniform(0.8, 1.3)
        market_cap = np.random.lognormal(mean=20, sigma=2)
        profit_margin = np.random.uniform(-0.2, 0.4)
        rows.append(
            {
                "ticker": f"STOCK{i:03d}",
                "name": f"Company {i}",
                "sector": np.random.choice(sectors),
                "region": np.random.choice(regions),
                "trading_country": np.random.choice(regions),
                "last_price": round(last_price, 2),
                "price_target": round(price_target, 2),
                "market_cap": round(market_cap, 0),
                "p_e_ntm": round(np.random.uniform(5, 50), 2),
                "profit_margin": round(profit_margin, 4),
                "ev": round(market_cap * np.random.uniform(0.9, 1.2), 0),
                "ebitda_ltm": round(market_cap * profit_margin * 0.01, 2),
                "revenue_ltm": round(market_cap * np.random.uniform(0.5, 2.0), 0),
                "volatility_90d": round(np.random.uniform(0.1, 0.6), 4),
                "data_source": "Sample",
            }
        )

    df = pd.DataFrame(rows)
    return df


def validate_financial_data_quality(df: pd.DataFrame, region: str) -> dict:
    """Validate data quality and return comprehensive metrics.

    Args:
        df: DataFrame to validate
        region: Region identifier for logging

    Returns:
        Dictionary with quality metrics including:
        - total_rows: Total number of rows
        - infinity_values: Count of infinity values
        - null_values: Count of null values
        - extreme_outliers: Count of extreme outliers
        - data_quality_score: Overall quality score (0-1)
    """
    results = {
        "region": region,
        "total_rows": len(df),
        "infinity_values": 0,
        "null_values": 0,
        "extreme_outliers": 0,
    }

    # Check for infinity values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        results["infinity_values"] += int(inf_count)

    # Check for null values
    results["null_values"] = int(df.isnull().sum().sum())

    # Check for extreme outliers (values > 1e10 or < -1e10)
    for col in numeric_cols:
        extreme_mask = (df[col].abs() > 1e10) & (~np.isinf(df[col]))
        results["extreme_outliers"] += int(extreme_mask.sum())

    # Calculate quality score
    total_cells = len(df) * len(df.columns)
    if total_cells > 0:
        issues = results["infinity_values"] + results["null_values"] + results["extreme_outliers"]
        results["data_quality_score"] = max(0.0, 1.0 - (issues / total_cells))
    else:
        results["data_quality_score"] = 0.0

    logging.info(
        "Data quality for %s: score=%.3f, inf=%d, null=%d, extreme=%d",
        region,
        results["data_quality_score"],
        results["infinity_values"],
        results["null_values"],
        results["extreme_outliers"],
    )

    return results


def sanitize_dataframe_with_logging(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize DataFrame with comprehensive logging.

    Args:
        df: Input DataFrame

    Returns:
        Sanitized DataFrame with infinity and NaN values handled
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Track sanitization actions
    inf_replaced = 0
    nan_filled = 0
    extreme_capped = 0

    for col in numeric_cols:
        # Replace infinity with NaN
        inf_mask = np.isinf(df[col])
        inf_count = int(inf_mask.sum())
        if inf_count > 0:
            df.loc[inf_mask, col] = np.nan
            inf_replaced += inf_count
            logging.debug("Replaced %d infinity values in column '%s'", inf_count, col)

        # Cap extreme values (> 1e10 or < -1e10)
        extreme_high_mask = df[col] > 1e10
        extreme_low_mask = df[col] < -1e10
        extreme_count = int((extreme_high_mask | extreme_low_mask).sum())
        if extreme_count > 0:
            df.loc[extreme_high_mask, col] = 1e10
            df.loc[extreme_low_mask, col] = -1e10
            extreme_capped += extreme_count
            logging.debug("Capped %d extreme values in column '%s'", extreme_count, col)

        # Fill NaN with median
        if df[col].isnull().any():
            nan_count = int(df[col].isnull().sum())
            median_val = df[col].median()
            if pd.notna(median_val):
                df[col].fillna(median_val, inplace=True)
                nan_filled += nan_count
                logging.debug(
                    "Filled %d NaN values in column '%s' with median %.3f",
                    nan_count,
                    col,
                    median_val,
                )
            else:
                # If median is NaN, fill with 0
                df[col].fillna(0.0, inplace=True)
                nan_filled += nan_count
                logging.debug("Filled %d NaN values in column '%s' with 0.0", nan_count, col)

    logging.info(
        "Sanitization complete: %d infinity replaced, %d NaN filled, %d extreme values capped",
        inf_replaced,
        nan_filled,
        extreme_capped,
    )

    return df


def perform_early_pipeline_validation(df: pd.DataFrame) -> dict:
    """Perform early pipeline validation with recommendations.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results including:
        - validation_score: Overall validation score (0-1)
        - total_checks: Total number of checks performed
        - passed_checks: Number of checks passed
        - warnings: List of warning messages
        - recommendations: List of recommended actions
    """
    results = {
        "validation_score": 1.0,
        "total_checks": 0,
        "passed_checks": 0,
        "warnings": [],
        "recommendations": [],
    }

    # Check 1: Minimum row count
    results["total_checks"] += 1
    if len(df) < 10:
        results["warnings"].append(f"Low row count: {len(df)} rows (minimum 10 recommended)")
        results["recommendations"].append("Consider loading more data for robust model training")
    else:
        results["passed_checks"] += 1

    # Check 2: Required columns present
    results["total_checks"] += 1
    required_cols = ["ticker", "sector", "last_price"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        results["warnings"].append(f"Missing required columns: {', '.join(missing_cols)}")
        results["recommendations"].append("Ensure data source includes all required columns")
    else:
        results["passed_checks"] += 1

    # Check 3: Negative prices
    results["total_checks"] += 1
    if "last_price" in df.columns:
        negative_prices = (df["last_price"] < 0).sum()
        if negative_prices > 0:
            results["warnings"].append(f"Found {negative_prices} negative prices")
            results["recommendations"].append("Review and clean price data before modeling")
        else:
            results["passed_checks"] += 1

    # Check 4: Zero prices
    results["total_checks"] += 1
    if "last_price" in df.columns:
        zero_prices = (df["last_price"] == 0).sum()
        if zero_prices > 0:
            results["warnings"].append(f"Found {zero_prices} zero prices")
            results["recommendations"].append("Consider removing or imputing zero prices")
        else:
            results["passed_checks"] += 1

    # Check 5: Extreme P/E ratios
    results["total_checks"] += 1
    pe_col = next((c for c in df.columns if "p_e" in c.lower()), None)
    if pe_col:
        extreme_pe = ((df[pe_col].abs() > 1000) & df[pe_col].notna()).sum()
        if extreme_pe > 0:
            results["warnings"].append(f"Found {extreme_pe} extreme P/E ratios (>1000)")
            results["recommendations"].append("Consider capping or filtering extreme P/E ratios")
        else:
            results["passed_checks"] += 1

    # Check 6: Future earnings dates
    results["total_checks"] += 1
    earnings_col = next(
        (c for c in df.columns if "next_earnings" in c.lower() and "days" in c.lower()), None
    )
    if earnings_col:
        past_earnings = (df[earnings_col] < 0).sum()
        if past_earnings > 0:
            results["warnings"].append(f"Found {past_earnings} past earnings dates")
            results["recommendations"].append("Update earnings dates to reflect future events")
        else:
            results["passed_checks"] += 1

    # Check 7: High missing value percentage
    results["total_checks"] += 1
    missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    if missing_pct > 30:
        results["warnings"].append(f"High missing value percentage: {missing_pct:.1f}%")
        results["recommendations"].append("Consider imputation strategies or feature selection")
    else:
        results["passed_checks"] += 1

    # Calculate validation score
    if results["total_checks"] > 0:
        results["validation_score"] = results["passed_checks"] / results["total_checks"]

    logging.info(
        "Pipeline validation: score=%.3f, checks=%d/%d passed",
        results["validation_score"],
        results["passed_checks"],
        results["total_checks"],
    )

    return results
