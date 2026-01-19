"""
Unified schema module - Single Source of Truth.

This module is the ONLY place where column definitions exist.
All other modules MUST import from here.

Phase 9.3 Feature Input Categorization (v1.15)
Total: 460+ features across 21 categories
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, TypedDict, Optional

# =============================================================================
# Type Aliases for Schema Definition
# =============================================================================

# DType: Maps to pandas/numpy dtype strings for ETL casting
DType = Literal[
    "float",  # float64 - default for numeric financial data (use Float64 for nullable)
    "Float64",  # pandas nullable float - use when pd.NA may be present
    "int",  # int64 - discrete counts, integer IDs
    "Int64",  # pandas nullable int - use when pd.NA may be present
    "string",  # object/string - text data
    "category",  # pandas Categorical - low-cardinality (sector, region)
    "datetime64[ns]",  # datetime columns
    "bool",  # boolean flags
    "boolean",  # pandas nullable boolean - use when pd.NA may be present
]

# Role: Semantic role determining preprocessing and pipeline treatment
Role = Literal[
    "id",  # Identifier columns (ticker, isin, name, description) - never used as features
    "target",  # Primary prediction target (price_target)
    "target_fallback",  # Alternative targets (price_target_median, last_price)
    "date",  # Temporal columns for time-series features
    "categorical",  # Grouping columns (sector, region, industry, exchange, unit, country, trading_country)
    "feature",  # General ML features from phase_93
    "market",  # Market/trading data (price, volume, market cap, dividends, shares outstanding)
    "financial_statement",  # P&L line items (revenues, expenses, recurring/non-recurring items)
    "balance_sheet",  # Balance sheet items (assets, liabilities, equity, working capital)
    "cash_flow",  # Cash flow statement items (CFO, CFI, CFF, FCF, capex)
    "ratio",  # Pre-normalized ratios (P/E, P/B, EV/EBITDA, ROE, ROA, EPS, turnover)
    "percentage",  # Bounded [0-100] metrics (margins, growth rates, returns, volatility, beta)
    "count",  # Discrete integers (analyst ratings, employees, shares, dividend streak)
    "auxiliary",  # Legacy aliases, optional - excluded from diagnostics
    "label",  # Classification targets (multi-label)
    "non_recurring",  # Non-recurring exceptional items (impairments, restructuring) - zero imputation
]


CATEGORICAL_DEFAULT_VALUE: str = "n/a"
NUMERIC_ZERO_DEFAULT: int = 0

# Role-level default values used for ingestion alignment (SQL defaults / COALESCE)
ROLE_DEFAULTS: Dict[Role, Any] = {
    "categorical": CATEGORICAL_DEFAULT_VALUE,
    "date": None,
    "financial_statement": NUMERIC_ZERO_DEFAULT,
    "balance_sheet": NUMERIC_ZERO_DEFAULT,
    "cash_flow": NUMERIC_ZERO_DEFAULT,
    "non_recurring": NUMERIC_ZERO_DEFAULT,
}


class ColumnMeta(TypedDict, total=False):
    dtype: DType
    role: Role
    sql_name: Optional[str]  # Original SQL column name
    description: Optional[str]  # Column description


# =============================================================================
# SQL Type to Python Dtype Mapping
# =============================================================================

SQL_TYPE_TO_DTYPE: Dict[str, DType] = {
    "text": "string",
    "varchar": "string",
    "character varying": "string",
    "char": "string",
    "integer": "Int64",
    "int": "Int64",
    "int4": "Int64",
    "bigint": "Int64",
    "int8": "Int64",
    "smallint": "Int64",
    "int2": "Int64",
    "numeric": "Float64",
    "decimal": "Float64",
    "real": "Float64",
    "float4": "Float64",
    "double precision": "Float64",
    "float8": "Float64",
    "date": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "timestamp without time zone": "datetime64[ns]",
    "timestamp with time zone": "datetime64[ns]",
    "boolean": "boolean",
    "bool": "boolean",
}


# =============================================================================
# Column Role Metadata - Maps SQL column names to semantic roles
# =============================================================================

# This dictionary provides the semantic metadata that cannot be inferred from SQL types
COLUMN_ROLE_METADATA: Dict[str, Dict[str, Any]] = {
    # ID columns
    "Ticker": {"role": "id"},
    "ISIN": {"role": "id"},
    "Name": {"role": "id"},
    "Description": {"role": "auxiliary"},
    # Categorical columns
    "Region": {"role": "categorical"},
    "Country": {"role": "categorical"},
    "Trading Country": {"role": "categorical"},
    "Exchange": {"role": "categorical"},
    "Unit": {"role": "categorical"},
    "Sector": {"role": "categorical"},
    "Industry": {"role": "categorical"},
    "Style Class": {"role": "categorical"},
    "Size Class": {"role": "categorical"},
    "FY End": {"role": "categorical"},
    "Next Earnings (When)": {"role": "categorical"},
    "Next Earnings (Status)": {"role": "categorical"},
    "Dividend Record (Currency)": {"role": "categorical"},
    "Dividend Record (Frequency)": {"role": "categorical"},
    "Current Fiscal Quarter": {"role": "categorical"},
    "Next Fiscal Quarter": {"role": "categorical"},
    "Next Earnings (Report)": {"role": "categorical"},
    "Earnings Report (Frequency)": {"role": "categorical"},
    # Date columns
    "Last Updated": {"role": "date"},
    "Income Statement Report Date": {"role": "date"},
    "Next Earnings": {"role": "date"},
    "Dividend Record (Announce Date)": {"role": "date"},
    "Dividend Record (Payable Date)": {"role": "date"},
    "Dividend Record (Record Date)": {"role": "date"},
    "Dividend Record (Ex Date)": {"role": "date"},
    "Reference Date": {"role": "date"},
    "FY End Date": {"role": "date"},
    "Next FY End Date": {"role": "date"},
    "Next Income Statement Report Date": {"role": "date"},
    # Target columns
    "Price Target": {"role": "target"},
    "Price Target - Median": {"role": "target_fallback"},
    # Market columns
    "Dividend Record (Amount)": {"role": "market"},
    "Market Cap": {"role": "market"},
    "Enterprise Value": {"role": "market"},
    "Last Price": {"role": "market"},
    "Price Target (YTD Ago)": {"role": "market"},
    "Price Target - Low": {"role": "market"},
    "Price Target - High": {"role": "market"},
    "Market Cap (Country R)": {"role": "market"},
    "Volume (Shrs)": {"role": "market"},
    "Dividend Per Share (LTM)": {"role": "market"},
    "Rel. Volume": {"role": "market"},
    "52W High/Adj": {"role": "market"},
    "52W Low/Adj": {"role": "market"},
    "EMA (20D)": {"role": "market"},
    "EMA (50D)": {"role": "market"},
    "EMA (100D)": {"role": "market"},
    "EMA (250D)": {"role": "market"},
    # Count columns
    "Dividend Streak": {"role": "count"},
    "Price Target - #": {"role": "count"},
    "Analyst Rating": {"role": "count"},
    "# Strong Sell Ratings": {"role": "count"},
    "# Strong Buys Ratings": {"role": "count"},
    "# Hold Ratings": {"role": "count"},
    "# Buys Ratings": {"role": "count"},
    "# Sell Ratings": {"role": "count"},
    "Shrs Out": {"role": "count"},
    "Shrs Out (-1FY)": {"role": "count"},
    "Full Time Employees (FQ)": {"role": "count"},
    "Full Time Employees (FY)": {"role": "count"},
    "Full Time Employees (-1FY)": {"role": "count"},
    "Full Time Employees (-2FY)": {"role": "count"},
    "Full Time Employees (-3FY)": {"role": "count"},
    "Avg Employees (5YAVGFY)": {"role": "count"},
    "EPS Norm - Est # (FY1E)": {"role": "count"},
    # Feature columns
    "Reporting Interval": {"role": "feature"},
    "Fiscal Month": {"role": "feature"},
    "Fiscal Quarter": {"role": "feature"},
    "Fiscal Year": {"role": "feature"},
    "Reporting Lag": {"role": "feature"},
    # Percentage columns
    "Total Return (YTD)": {"role": "percentage"},
    "Beta (1Y)": {"role": "percentage"},
    "Beta (2Y)": {"role": "percentage"},
    "Beta (5Y)": {"role": "percentage"},
    "Total Revenues/CAGR (5Y FY)": {"role": "percentage"},
    "Tot. Return %/CAGR (3Y)": {"role": "percentage"},
    "Tot. Return %/CAGR (10Y)": {"role": "percentage"},
    "Total Return (5Y)": {"role": "percentage"},
    "Total Return (10Y)": {"role": "percentage"},
    "Net Income Margin % (FY)": {"role": "percentage"},
    "Net Income Margin % (LTM)": {"role": "percentage"},
    "Volatility (1M)": {"role": "percentage"},
    "Volatility (3M)": {"role": "percentage"},
    "Volatility (6M)": {"role": "percentage"},
    "Volatility (1Y)": {"role": "percentage"},
    "Div Yield (Ind)": {"role": "percentage"},
    "Div Yield (LTM)": {"role": "percentage"},
    "Gross Profit Margin % (FY)": {"role": "percentage"},
    "Gross Profit Margin % (LTM)": {"role": "percentage"},
    "Buyback Yield (LTM)": {"role": "percentage"},
    "Div Yield (-1FYInd)": {"role": "percentage"},
    "Div Yield (TTM)": {"role": "percentage"},
    "Div Yield (NTM)": {"role": "percentage"},
    "Div Yield (5YAVGLTM)": {"role": "percentage"},
    "Revenues - Est YoY % (FY1E)": {"role": "percentage"},
    "Price Chg. % (1M)": {"role": "percentage"},
    "Price Chg. % (3M)": {"role": "percentage"},
    "1-Day %": {"role": "percentage"},
    "Div Yield (-2FYInd)": {"role": "percentage"},
    "Div Yield (-3FYInd)": {"role": "percentage"},
    "Div Yield (-4FYInd)": {"role": "percentage"},
    "Div Yield (-5FYInd)": {"role": "percentage"},
}


def _map_sql_type_to_dtype(sql_type: str) -> DType:
    """Map SQL type string to pandas dtype."""
    sql_type_lower = sql_type.lower().strip()

    # Handle parameterized types like varchar(255), numeric(10,2)
    base_type = re.split(r"[\(\[]", sql_type_lower)[0].strip()

    return SQL_TYPE_TO_DTYPE.get(base_type, "Float64")


def _infer_role_from_sql_name(sql_name: str) -> Role:
    """Infer column role from SQL column name patterns."""
    name_lower = sql_name.lower()

    # Check explicit metadata first
    if sql_name in COLUMN_ROLE_METADATA:
        return COLUMN_ROLE_METADATA[sql_name].get("role", "feature")

    # Pattern-based inference
    # Non-recurring items
    if any(
        kw in name_lower
        for kw in ["impairment", "writedown", "restructuring", "unusual", "gain (loss)"]
    ):
        return "non_recurring"

    # Ratio patterns
    if any(
        pat in name_lower
        for pat in [
            "p/e",
            "p/b",
            "p/tbv",
            "ev/",
            "eps",
            "return on",
            "current ratio",
            "asset turnover",
            "altman",
        ]
    ):
        return "ratio"

    # Cash flow patterns
    if any(
        kw in name_lower
        for kw in [
            "cfo",
            "cfi",
            "cff",
            "fcf",
            "cash acquisitions",
            "capital expenditure",
            "dividends paid",
        ]
    ):
        return "cash_flow"

    # Balance sheet patterns
    if any(
        kw in name_lower
        for kw in [
            "total assets",
            "total equity",
            "total debt",
            "inventory",
            "goodwill",
            "retained earnings",
            "working capital",
            "cash and equivalents",
            "tbv",
            "current assets",
            "current liabilities",
            "intangible",
        ]
    ):
        return "balance_sheet"

    # Financial statement patterns
    if any(
        kw in name_lower
        for kw in [
            "revenues",
            "ebitda",
            "ebit",
            "net income",
            "gross profit",
            "operating income",
            "operating expenses",
            "cost of",
            "r&d",
            "selling general",
            "marketing expenses",
            "interest expense",
            "interest income",
        ]
    ):
        return "financial_statement"

    # Market data patterns
    if any(
        kw in name_lower
        for kw in [
            "price",
            "ema",
            "volume",
            "market cap",
            "enterprise value",
            "dividend per share",
            "52w",
        ]
    ):
        return "market"

    # Count patterns
    if any(
        kw in name_lower
        for kw in ["# ", "num ", "employees", "shrs out", "analyst rating"]
    ):
        return "count"

    # Percentage patterns
    if any(
        kw in name_lower
        for kw in [
            "margin %",
            "return %",
            "yield",
            "volatility",
            "beta",
            "cagr",
            "chg. %",
            "day %",
        ]
    ):
        return "percentage"

    # Default to feature
    return "feature"


def get_equities_schema_from_sql(
    connection_string: Optional[str] = None,
    schema: str = "public",
    table_name: str = "equities",
) -> Dict[str, ColumnMeta]:
    """
    Retrieve the equities table schema from PostgreSQL and build COLUMN_SCHEMA.

    Uses SQLAlchemy to introspect the database schema and maps SQL types
    to pandas dtypes with semantic role information.

    Args:
        connection_string: SQLAlchemy connection string. If None, uses environment
                          variable DATABASE_URL or defaults to localhost.
        schema: Database schema name (default: 'public')
        table_name: Table name to introspect (default: 'equities')

    Returns:
        Dictionary mapping normalized column names to ColumnMeta TypedDict

    Example:
        >>> schema = get_equities_schema_from_sql("postgresql://user:pass@localhost/db")
        >>> schema["ticker"]
        {'dtype': 'string', 'role': 'id', 'sql_name': 'Ticker'}
    """
    import os
    from sqlalchemy import create_engine, inspect, text

    # Get connection string
    if connection_string is None:
        connection_string = os.environ.get(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
        )

    engine = create_engine(connection_string)
    inspector = inspect(engine)

    # Get column information
    columns = inspector.get_columns(table_name, schema=schema)

    # Also get column comments if available
    comments: Dict[str, str] = {}
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    f"""
                SELECT column_name, col_description(
                    (SELECT oid FROM pg_class WHERE relname = :table_name),
                    ordinal_position
                ) as comment
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table_name
            """
                ),
                {"table_name": table_name, "schema": schema},
            )
            for row in result:
                if row.comment:
                    comments[row.column_name] = row.comment
    except Exception:
        pass  # Comments are optional

    column_schema: Dict[str, ColumnMeta] = {}

    for col in columns:
        sql_name = col["name"]
        sql_type = str(col["type"]).lower()

        # Normalize column name for Python
        normalized_name = normalize_column_name(sql_name)

        # Map SQL type to pandas dtype
        dtype = _map_sql_type_to_dtype(sql_type)

        # Infer role from column name
        role = _infer_role_from_sql_name(sql_name)

        # Build column metadata
        meta: ColumnMeta = {
            "dtype": dtype,
            "role": role,
            "sql_name": sql_name,
        }

        # Add description if available
        if sql_name in comments:
            meta["description"] = comments[sql_name]

        column_schema[normalized_name] = meta

    return column_schema


def refresh_column_schema_from_sql(
    connection_string: Optional[str] = None,
) -> Dict[str, ColumnMeta]:
    """
    Refresh COLUMN_SCHEMA from the database.

    Call this function to update the schema after database changes.

    Args:
        connection_string: Optional SQLAlchemy connection string

    Returns:
        Updated COLUMN_SCHEMA dictionary
    """
    global COLUMN_SCHEMA
    COLUMN_SCHEMA = get_equities_schema_from_sql(connection_string)
    return COLUMN_SCHEMA


def generate_static_schema_code(connection_string: Optional[str] = None) -> str:
    """
    Generate Python code for the static schema fallback.

    Use this to update the _get_static_column_schema() function after
    database schema changes.

    Args:
        connection_string: Optional SQLAlchemy connection string

    Returns:
        Python code string that can be pasted into _get_static_column_schema()
    """
    schema = get_equities_schema_from_sql(connection_string)

    lines = ["return {"]
    for col_name, meta in sorted(schema.items()):
        meta_str = ", ".join(f'"{k}": "{v}"' for k, v in meta.items())
        lines.append(f'    "{col_name}": {{{meta_str}}},')
    lines.append("}")

    return "\n".join(lines)


# Master schema - auto-generates SQL
# NOTE: This is a truncated version of the 555 entries.
# In a real scenario, all 555 entries would be here.
COLUMN_SCHEMA: Dict[str, ColumnMeta] = {
    "ticker": {"dtype": "string", "role": "id", "sql_name": "Ticker"},
    "isin": {"dtype": "string", "role": "id", "sql_name": "ISIN"},
    "name": {"dtype": "string", "role": "id", "sql_name": "Name"},
    "description": {"dtype": "string", "role": "auxiliary", "sql_name": "Description"},
    "sector": {"dtype": "category", "role": "categorical", "sql_name": "Sector"},
    "industry": {"dtype": "category", "role": "categorical", "sql_name": "Industry"},
    "region": {"dtype": "category", "role": "categorical", "sql_name": "Region"},
    "country": {"dtype": "category", "role": "categorical", "sql_name": "Country"},
    "trading_country": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Trading Country",
    },
    "exchange": {"dtype": "category", "role": "categorical", "sql_name": "Exchange"},
    "unit": {"dtype": "category", "role": "categorical", "sql_name": "Unit"},
    "style_class": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Style Class",
    },
    "size_class": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Size Class",
    },
    "next_earnings_status": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (Status)",
    },
    "next_earnings_report": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (Report)",
        "description": "Next earnings report type (Full Year/Interim)",
    },
    "earnings_report_frequency": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Earnings Report (Frequency)",
        "description": "Earnings report frequency (e.g., Quarterly, Semi-Annual)",
    },
    "last_updated": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Last Updated",
    },
    "income_statement_report_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Income Statement Report Date",
    },
    "next_income_statement_report_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Next Income Statement Report Date",
        "description": "Next income statement report date",
    },
    "fy_end": {"dtype": "category", "role": "categorical", "sql_name": "FY End"},
    "fy_end_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "FY End Date",
        "description": "Fiscal year end date (parsed from FY End text)",
    },
    "next_fy_end_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Next FY End Date",
        "description": "Next fiscal year end date",
    },
    "fiscal_month": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Month",
        "description": "Months between Income Statement Report Date and FY End Date",
    },
    "fiscal_quarter": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Quarter",
        "description": "Fiscal quarter (1-4) from report date",
    },
    "fiscal_year": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Fiscal Year",
        "description": "Fiscal year from report date",
    },
    "reporting_interval": {
        "dtype": "Int64",
        "role": "feature",
        "sql_name": "Reporting Interval",
        "description": "Reporting interval in days",
    },
    "current_fiscal_quarter": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Current Fiscal Quarter",
        "description": "Current fiscal quarter (formatted as Q4 2025)",
    },
    "next_fiscal_quarter": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Fiscal Quarter",
        "description": "Next fiscal quarter (formatted as Q4 2025)",
    },
    "next_earnings": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Next Earnings",
    },
    "next_earnings_when": {
        "dtype": "category",
        "role": "categorical",
        "sql_name": "Next Earnings (When)",
    },
    "dividend_record_announce_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Announce Date)",
    },
    "dividend_record_ex_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Ex Date)",
    },
    "dividend_record_payable_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Payable Date)",
    },
    "dividend_record_record_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Dividend Record (Record Date)",
    },
    "reference_date": {
        "dtype": "datetime64[ns]",
        "role": "date",
        "sql_name": "Reference Date",
    },
    "last_price": {"dtype": "Float64", "role": "market", "sql_name": "Last Price"},
    "price_target": {"dtype": "Float64", "role": "target", "sql_name": "Price Target"},
    "price_target_ytd_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price Target (YTD Ago)",
    },
    "price_target_low": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price Target - Low",
    },
    "price_target_median": {
        "dtype": "Float64",
        "role": "target_fallback",
        "sql_name": "Price Target - Median",
    },
    "price_target_high": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price Target - High",
    },
    "price_target_num": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - #",
    },
    "price_target_count": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - #",
    },
    # Price Target Historical Analyst Count columns
    "price_target_num_1w_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (1W Ago)",
        "description": "Number of analysts with price targets 1 week ago",
    },
    "price_target_num_1m_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (1M Ago)",
        "description": "Number of analysts with price targets 1 month ago",
    },
    "price_target_num_mtd_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (MTD Ago)",
        "description": "Number of analysts with price targets MTD ago",
    },
    "price_target_num_qtd_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (QTD Ago)",
        "description": "Number of analysts with price targets QTD ago",
    },
    "price_target_num_3m_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (3M Ago)",
        "description": "Number of analysts with price targets 3 months ago",
    },
    "price_target_num_6m_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (6M Ago)",
        "description": "Number of analysts with price targets 6 months ago",
    },
    "price_target_num_ytd_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (YTD Ago)",
        "description": "Number of analysts with price targets YTD ago",
    },
    "price_target_num_1y_ago": {
        "dtype": "Int64",
        "role": "count",
        "sql_name": "Price Target - # (1Y Ago)",
        "description": "Number of analysts with price targets 1 year ago",
    },
    "price_5d_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (5D Ago)",
    },
    "price_1w_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (1W Ago)",
    },
    "price_1m_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (1M Ago)",
    },
    "price_3m_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (3M Ago)",
    },
    "price_6m_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (6M Ago)",
    },
    "price_1y_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (1Y Ago)",
    },
    "price_3y_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (3Y Ago)",
    },
    "price_5y_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (5Y Ago)",
    },
    "price_qtd_ago": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Price (QTD Ago)",
    },
    "market_cap": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Market Cap",
        "description": "Market capitalization",
    },
    "enterprise_value": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Enterprise Value",
        "description": "Enterprise value",
    },
    "market_cap_country_r": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Market Cap (Country R)",
    },
    "p_e_ntm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (NTM)"},
    "p_e_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (LTM)"},
    "p_fcf": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "P/FCF",
        "description": "Price to Free Cash Flow",
    },
    "forward_pe": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Forward P/E",
        "description": "Forward Price to Earnings",
    },
    "peg_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "PEG Ratio",
        "description": "Price/Earnings to Growth Ratio",
    },
    "fcf_per_share": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "FCF Per Share",
        "description": "Free Cash Flow Per Share",
    },
    "revenue_per_share": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Revenue Per Share",
        "description": "Revenue Per Share",
    },
    "ev_fcf": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/FCF",
        "description": "Enterprise Value to Free Cash Flow",
    },
    "p_e_1fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-1FYLTM)"},
    "p_b_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (LTM)"},
    "p_b_1fy": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (-1FY)"},
    "p_b_5yavg": {"dtype": "Float64", "role": "ratio", "sql_name": "P/B (5YAVG)"},
    "p_tbv_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/TBV (LTM)"},
    "ev_sales_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (LTM)"},
    "ev_sales_ntm": {"dtype": "Float64", "role": "ratio", "sql_name": "EV/Sales (NTM)"},
    "ev_sales_est_fy1": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (EST FY1)",
    },
    "ev_ebitda_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (LTM)",
    },
    "ev_ebitda_ntm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (NTM)",
    },
    "ev_ebitda_est_fy1": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (EST FY1)",
    },
    "p_e_est_fy1": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (EST FY1)"},
    "ev_sales_1fyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-1FYLTM)",
    },
    "ev_sales_2fyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-2FYLTM)",
    },
    "ev_sales_3fyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-3FYLTM)",
    },
    "ev_sales_3yavgltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (3YAVGLTM)",
    },
    "ev_sales_1fqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-1FQLTM)",
    },
    "ev_sales_2fqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-2FQLTM)",
    },
    "ev_sales_3fqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-3FQLTM)",
    },
    "ev_sales_4fqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/Sales (-4FQLTM)",
    },
    "ev_ebitda_1fyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (-1FYLTM)",
    },
    "ev_ebitda_1fqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (-1FQLTM)",
    },
    "ev_ebitda_3yavgltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EV/EBITDA (3YAVGLTM)",
    },
    "p_e_2fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-2FYLTM)"},
    "p_e_3fyltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-3FYLTM)"},
    "p_e_3yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (3YAVGLTM)"},
    "p_e_1fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-1FQLTM)"},
    "p_e_2fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-2FQLTM)"},
    "p_e_3fqltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (-3FQLTM)"},
    "p_e_5yavgltm": {"dtype": "Float64", "role": "ratio", "sql_name": "P/E (5YAVGLTM)"},
    "p_e_0fqqoqltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "P/E (-0FQQoQLTM)",
    },
    "p_e_0fyyoyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "P/E (-0FYYoYLTM)",
    },
    "p_e_1fyyoyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "P/E (-1FYYoYLTM)",
    },
    "p_e_0fqyoyltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "P/E (-0FQYoYLTM)",
    },
    "altman_z_score_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Altman Z-Score (FY)",
    },
    "altman_z_score_fq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Altman Z-Score (FQ)",
    },
    "altman_z_score_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Altman Z-Score (LTM)",
    },
    "beta_1y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (1Y)"},
    "beta_2y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (2Y)"},
    "beta_5y": {"dtype": "Float64", "role": "percentage", "sql_name": "Beta (5Y)"},
    "total_return_ytd": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Return (YTD)",
    },
    "total_return_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Return (5Y)",
    },
    "total_return_10y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Return (10Y)",
    },
    "tot_return_pct_cagr_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Tot. Return %/CAGR (3Y)",
    },
    "tot_return_pct_cagr_10y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Tot. Return %/CAGR (10Y)",
    },
    "price_chg_pct_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Price Chg. % (1M)",
    },
    "price_chg_pct_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Price Chg. % (3M)",
    },
    "1_day_pct": {"dtype": "Float64", "role": "percentage", "sql_name": "1-Day %"},
    "one_day_pct": {"dtype": "Float64", "role": "percentage", "sql_name": "1-Day %"},
    "analyst_rating": {"dtype": "float", "role": "count", "sql_name": "Analyst Rating"},
    "num_strong_sell_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Strong Sell Ratings",
    },
    "num_strong_buys_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Strong Buys Ratings",
    },
    "num_hold_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Hold Ratings",
    },
    "num_buys_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Buys Ratings",
    },
    "num_sell_ratings": {
        "dtype": "float",
        "role": "count",
        "sql_name": "# Sell Ratings",
    },
    "ema_20d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (20D)"},
    "ema_50d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (50D)"},
    "ema_100d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (100D)"},
    "ema_250d": {"dtype": "Float64", "role": "market", "sql_name": "EMA (250D)"},
    "ma_20d_simple": {"dtype": "Float64", "role": "market"},
    "ma_50d_simple": {"dtype": "Float64", "role": "market"},
    "52w_high_adj": {"dtype": "Float64", "role": "market", "sql_name": "52W High/Adj"},
    "52w_low_adj": {"dtype": "Float64", "role": "market", "sql_name": "52W Low/Adj"},
    "volatility_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Volatility (1M)",
    },
    "volatility_30d": {"dtype": "Float64", "role": "percentage"},
    "volatility_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Volatility (3M)",
    },
    "volatility_60d": {"dtype": "Float64", "role": "percentage"},
    "volatility_6m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Volatility (6M)",
    },
    "volatility_90d": {"dtype": "Float64", "role": "percentage"},
    "volatility_1y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Volatility (1Y)",
    },
    "volume_shrs": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Volume (Shrs)",
        "description": "Trading volume in shares",
    },
    "rel_volume": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Rel. Volume",
        "description": "Relative trading volume ratio",
    },
    "shrs_out": {"dtype": "float", "role": "count", "sql_name": "Shrs Out"},
    "shares_outstanding": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Shrs Out",
        "description": "Shares outstanding",
    },
    "shrs_out_1fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Shrs Out (-1FY)",
        "description": "Shares outstanding (previous FY)",
    },
    "total_revenues_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (FY)",
        "description": "Total revenues (Fiscal Year)",
    },
    "total_revenues_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (LTM)",
        "description": "Total revenues (Last Twelve Months)",
    },
    "total_revenues_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (FQ)",
    },
    "total_revenues_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (-1FY)",
    },
    "total_revenues_cagr_5y_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Total Revenues/CAGR (5Y FY)",
    },
    "total_revenues_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (5YAVGFQ)",
    },
    "total_revenues_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Revenues (5YAVGLTM)",
    },
    "revenues_est_avg_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Avg (NTM)",
    },
    "revenues_est_avg_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Avg (FY1E)",
    },
    "revenues_est_med_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Med (NTM)",
    },
    "revenues_est_med_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Revenues - Est Med (FY1E)",
    },
    "revenues_est_yoy_pct_fy1e": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Revenues - Est YoY % (FY1E)",
    },
    "total_operating_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Total Operating Expenses (LTM)",
    },
    "ebitda_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (FY)",
        "description": "EBITDA (Fiscal Year)",
    },
    "ebitda_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (LTM)",
        "description": "EBITDA (Last Twelve Months)",
    },
    "ebitda_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (FQ)",
    },
    "ebitda_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-1FY)",
    },
    "ebitda_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (LTM)",
    },
    "ebitda_adj_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (FY)",
    },
    "ebitda_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-1FY)",
    },
    "ebitda_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (5YAVGFQ)",
    },
    "ebitda_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (5YAVGLTM)",
    },
    "ebitda_est_avg_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA - Est Avg (FY1E)",
    },
    "ebitda_est_avg_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA - Est Avg (NTM)",
    },
    "ebit_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (FY)",
    },
    "ebit_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (LTM)",
    },
    "ebit_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (FQ)",
    },
    "ebit_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-1FY)",
    },
    "ebit_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (LTM)",
    },
    "ebit_adj_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (FY)",
    },
    "ebit_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-1FY)",
    },
    "ebit_est_med_fy1e": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT - Est Med (FY1E)",
    },
    "ebit_est_med_ntm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT - Est Med (NTM)",
    },
    "ebit_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (5YAVGFQ)",
    },
    "ebit_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (5YAVGLTM)",
    },
    "net_income_is_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (FY)",
        "description": "Net income from income statement (Fiscal Year)",
    },
    "net_income_is_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (LTM)",
        "description": "Net income from income statement (Last Twelve Months)",
    },
    "net_income_is_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (FQ)",
    },
    "net_income_is_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-1FY)",
    },
    "net_income_is_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (5YAVGFQ)",
    },
    "net_income_is_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (5YAVGLTM)",
    },
    "normalized_net_income_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (FY)",
    },
    "normalized_net_income_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (LTM)",
    },
    "normalized_net_income_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (FQ)",
    },
    "normalized_net_income_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-1FY)",
    },
    "normalized_net_income_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (5YAVGFQ)",
    },
    "normalized_net_income_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (5YAVGLTM)",
    },
    "net_income_adj_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (FY)",
    },
    "net_income_adj_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (LTM)",
    },
    "net_income_adj_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (FQ)",
    },
    "net_income_adj_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-1FY)",
    },
    "net_income_adj_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (5YAVGFQ)",
    },
    "operating_income_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (LTM)",
    },
    "operating_income_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (FY)",
    },
    "operating_income_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (FQ)",
    },
    "operating_income_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Operating Income (5YAVGFQ)",
    },
    "net_income_margin_pct_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Net Income Margin % (FY)",
    },
    "net_income_margin_pct_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Net Income Margin % (LTM)",
    },
    "gross_profit_margin_pct_fy": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Gross Profit Margin % (FY)",
    },
    "gross_profit_margin_pct_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Gross Profit Margin % (LTM)",
    },
    "gross_profit_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Gross Profit (LTM)",
    },
    "gross_profit_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Gross Profit (FY)",
    },
    "return_on_equity_pct_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return On Equity % (LTM)",
        "description": "Return on equity percentage (Last Twelve Months)",
    },
    "return_on_equity_pct_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return On Equity % (FY)",
    },
    "return_on_assets_roa_pct_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return on Assets (ROA) % (LTM)",
        "description": "Return on assets percentage (Last Twelve Months)",
    },
    "return_on_assets_roa_pct_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Return on Assets (ROA) % (FY)",
    },
    "cfo_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "CFO (LTM)",
        "description": "Cash from operations (Last Twelve Months)",
    },
    "cfo_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (FY)"},
    "cfo_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (FQ)"},
    "cfo_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-1FY)"},
    "fcf_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "FCF (LTM)",
        "description": "Free cash flow (Last Twelve Months)",
    },
    "fcf_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (FY)"},
    "fcf_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (FQ)"},
    "fcf_5yavgfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (5YAVGFQ)"},
    "cfi_ltm": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (LTM)"},
    "cfi_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (FY)"},
    "cfi_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (FQ)"},
    "cfi_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-1FY)"},
    "cff_ltm": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (LTM)"},
    "cff_fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (FY)"},
    "cff_fq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (FQ)"},
    "cff_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-1FY)"},
    "total_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Assets (LTM)",
        "description": "Total assets (Last Twelve Months)",
    },
    "total_assets_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Assets (FY)",
        "description": "Total assets (Fiscal Year)",
    },
    "total_equity_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Equity (FY)",
        "description": "Total equity (Fiscal Year)",
    },
    "total_equity_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Equity (LTM)",
    },
    "total_debt_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Debt (FY)",
        "description": "Total debt (Fiscal Year)",
    },
    "total_debt_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Debt (LTM)",
    },
    "total_current_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Current Assets (LTM)",
    },
    "total_current_liabilities_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Total Current Liabilities (LTM)",
    },
    "current_ratio_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Current Ratio (FY)",
    },
    "current_ratio_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Current Ratio (LTM)",
    },
    "working_capital_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (LTM)",
    },
    "working_capital_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (FQ)",
    },
    "working_capital_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (FY)",
    },
    "working_capital_5yavgfy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Working Capital (5YAVGFY)",
    },
    "tbv_fy": {"dtype": "float", "role": "balance_sheet", "sql_name": "TBV (FY)"},
    "tbv_ltm": {"dtype": "float", "role": "balance_sheet", "sql_name": "TBV (LTM)"},
    "cash_and_equivalents_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (LTM)",
    },
    "cash_and_equivalents_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (FQ)",
    },
    "cash_and_equivalents_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (FY)",
    },
    "cash_and_equivalents_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Cash And Equivalents (5YAVGFQ)",
    },
    "retained_earnings_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (LTM)",
    },
    "retained_earnings_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (FQ)",
    },
    "retained_earnings_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (FY)",
    },
    "retained_earnings_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Retained Earnings (5YAVGFQ)",
    },
    "inventory_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Inventory (LTM)",
    },
    "inventory_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Inventory (FQ)",
    },
    "inventory_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Inventory (FY)",
    },
    "inventory_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Inventory (5YAVGFQ)",
    },
    "goodwill_fq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (FQ)",
    },
    "goodwill_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (LTM)",
    },
    "goodwill_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (FY)",
    },
    "goodwill_1fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (-1FY)",
    },
    "goodwill_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Goodwill (5YAVGFQ)",
    },
    "intangible_assets": {"dtype": "Float64", "role": "feature"},
    "gross_intangible_assets_ltm": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (LTM)",
    },
    "gross_intangible_assets_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (FY)",
    },
    "gross_intangible_assets_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Gross Intangible Assets (5YAVGFQ)",
    },
    "capital_expenditure_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (LTM)",
        "description": "Capital expenditure (Last Twelve Months)",
    },
    "capital_expenditure_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (FY)",
    },
    "capital_expenditure_fq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (FQ)",
    },
    "capital_expenditure_1fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (-1FY)",
    },
    "capital_expenditure_5yavgfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Capital Expenditure (5YAVGFQ)",
    },
    "asset_turnover_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Asset Turnover (FY)",
    },
    "asset_turnover_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Asset Turnover (LTM)",
    },
    "cash_acquisitions_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (LTM)",
    },
    "cash_acquisitions_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (FY)",
    },
    "cash_acquisitions_fq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (FQ)",
    },
    "cash_acquisitions_1fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-1FY)",
    },
    "cash_acquisitions_5yavgfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (5YAVGFQ)",
    },
    "impairment_of_goodwill_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (FQ)",
    },
    "impairment_of_goodwill_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (LTM)",
    },
    "impairment_of_goodwill_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-1FY)",
    },
    "impairment_of_goodwill_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (FY)",
    },
    "impairment_of_goodwill_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (5YAVGFQ)",
    },
    "asset_writedown_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (LTM)",
    },
    "asset_writedown_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (FY)",
    },
    "asset_writedown_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (FQ)",
    },
    "asset_writedown_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-1FY)",
    },
    "asset_writedown_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (5YAVGFQ)",
    },
    "restructuring_charges_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (LTM)",
    },
    "restructuring_charges_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (FQ)",
    },
    "restructuring_charges_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-1FY)",
    },
    "restructuring_charges_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (FY)",
    },
    "restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (5YAVGFQ)",
    },
    "merger_and_restructuring_charges_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (LTM)",
    },
    "merger_and_restructuring_charges_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (FQ)",
    },
    "merger_and_restructuring_charges_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (FY)",
    },
    "merger_and_restructuring_charges_5yavgfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Merger & Restructuring Charges (5YAVGFQ)",
    },
    # --- NEW COLUMNS: Address missing schema warnings ---
    "r_d_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "R&D Expenses (LTM)",
        "description": "Research and development expenses (Last Twelve Months)",
    },
    "price_target_number": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - #",
        "description": "Number of analyst price targets (alias for price_target_num)",
    },
    "other_unusual_items_total_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Other Unusual Items/Total (LTM)",
    },
    "gain_loss_on_sale_of_assets_ltm": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (LTM)",
    },
    # --- NEW: Impairment of Goodwill Historical ---
    "impairment_of_goodwill_1fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-1FQFQ)",
    },
    "impairment_of_goodwill_2fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-2FQFQ)",
    },
    "impairment_of_goodwill_3fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-3FQFQ)",
    },
    "impairment_of_goodwill_4fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-4FQFQ)",
    },
    "impairment_of_goodwill_2fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-2FY)",
    },
    "impairment_of_goodwill_3fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-3FY)",
    },
    "impairment_of_goodwill_4fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Impairment of Goodwill (-4FY)",
    },
    # --- NEW: Asset Writedown Historical ---
    "asset_writedown_1fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-1FQFQ)",
    },
    "asset_writedown_2fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-2FQFQ)",
    },
    "asset_writedown_3fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-3FQFQ)",
    },
    "asset_writedown_4fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-4FQFQ)",
    },
    "asset_writedown_2fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-2FY)",
    },
    "asset_writedown_3fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-3FY)",
    },
    "asset_writedown_4fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-4FY)",
    },
    "asset_writedown_5fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Asset Writedown (-5FY)",
    },
    # --- NEW: Gain (Loss) On Sale Of Assets Historical ---
    "gain_loss_on_sale_of_assets_fq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (FQ)",
    },
    "gain_loss_on_sale_of_assets_fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (FY)",
    },
    "gain_loss_on_sale_of_assets_1fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-1FQFQ)",
    },
    "gain_loss_on_sale_of_assets_2fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-2FQFQ)",
    },
    "gain_loss_on_sale_of_assets_3fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-3FQFQ)",
    },
    "gain_loss_on_sale_of_assets_4fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-4FQFQ)",
    },
    "gain_loss_on_sale_of_assets_1fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-1FY)",
    },
    "gain_loss_on_sale_of_assets_2fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-2FY)",
    },
    "gain_loss_on_sale_of_assets_3fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-3FY)",
    },
    "gain_loss_on_sale_of_assets_4fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Gain (Loss) On Sale Of Assets (-4FY)",
    },
    # --- NEW: Restructuring Charges Historical ---
    "restructuring_charges_1fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-1FQFQ)",
    },
    "restructuring_charges_2fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-2FQFQ)",
    },
    "restructuring_charges_3fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-3FQFQ)",
    },
    "restructuring_charges_4fqfq": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-4FQFQ)",
    },
    "restructuring_charges_2fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-2FY)",
    },
    "restructuring_charges_3fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-3FY)",
    },
    "restructuring_charges_4fy": {
        "dtype": "float",
        "role": "non_recurring",
        "sql_name": "Restructuring Charges (-4FY)",
    },
    # --- NEW: Net Income - (IS) Historical ---
    "net_income_is_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-1FQFQ)",
    },
    "net_income_is_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-2FQFQ)",
    },
    "net_income_is_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-3FQFQ)",
    },
    "net_income_is_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-4FQFQ)",
    },
    "net_income_is_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-2FY)",
    },
    "net_income_is_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-3FY)",
    },
    "net_income_is_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income - (IS) (-4FY)",
    },
    # --- NEW: Normalized Net Income Historical ---
    "normalized_net_income_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-1FQFQ)",
    },
    "normalized_net_income_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-2FQFQ)",
    },
    "normalized_net_income_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-3FQFQ)",
    },
    "normalized_net_income_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-4FQFQ)",
    },
    "normalized_net_income_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-2FY)",
    },
    "normalized_net_income_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-3FY)",
    },
    "normalized_net_income_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Normalized Net Income (-4FY)",
    },
    # --- NEW: Net Income/Adj. Historical ---
    "net_income_adj_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-1FQFQ)",
    },
    "net_income_adj_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-2FQFQ)",
    },
    "net_income_adj_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-3FQFQ)",
    },
    "net_income_adj_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-4FQFQ)",
    },
    "net_income_adj_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-2FY)",
    },
    "net_income_adj_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-3FY)",
    },
    "net_income_adj_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Net Income/Adj. (-4FY)",
    },
    # --- NEW: EBIT Historical ---
    "ebit_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-1FQFQ)",
    },
    "ebit_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-2FQFQ)",
    },
    "ebit_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-3FQFQ)",
    },
    "ebit_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-4FQFQ)",
    },
    "ebit_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-2FY)",
    },
    "ebit_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-3FY)",
    },
    "ebit_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT (-4FY)",
    },
    # --- NEW: EBIT/Adj. Historical ---
    "ebit_adj_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (FQ)",
    },
    "ebit_adj_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-1FQFQ)",
    },
    "ebit_adj_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-2FQFQ)",
    },
    "ebit_adj_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-3FQFQ)",
    },
    "ebit_adj_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-4FQFQ)",
    },
    "ebit_adj_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-2FY)",
    },
    "ebit_adj_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-3FY)",
    },
    "ebit_adj_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBIT/Adj. (-4FY)",
    },
    # --- NEW: EBITDA Historical ---
    "ebitda_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-1FQFQ)",
    },
    "ebitda_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-2FQFQ)",
    },
    "ebitda_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-3FQFQ)",
    },
    "ebitda_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-4FQFQ)",
    },
    "ebitda_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-2FY)",
    },
    "ebitda_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-3FY)",
    },
    "ebitda_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA (-4FY)",
    },
    # --- NEW: EBITDA/Adj. Historical ---
    "ebitda_adj_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (FQ)",
    },
    "ebitda_adj_1fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-1FQFQ)",
    },
    "ebitda_adj_2fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-2FQFQ)",
    },
    "ebitda_adj_3fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-3FQFQ)",
    },
    "ebitda_adj_4fqfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-4FQFQ)",
    },
    "ebitda_adj_2fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-2FY)",
    },
    "ebitda_adj_3fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-3FY)",
    },
    "ebitda_adj_4fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "EBITDA/Adj. (-4FY)",
    },
    # --- NEW: Basic EPS - Cont Historical ---
    "basic_eps_cont_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (LTM)",
    },
    "basic_eps_cont_fq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (FQ)",
    },
    "basic_eps_cont_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (FY)",
    },
    "basic_eps_cont_1fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-1FQFQ)",
    },
    "basic_eps_cont_2fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-2FQFQ)",
    },
    "basic_eps_cont_3fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-3FQFQ)",
    },
    "basic_eps_cont_4fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-4FQFQ)",
    },
    "basic_eps_cont_1fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-1FY)",
    },
    "basic_eps_cont_2fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-2FY)",
    },
    "basic_eps_cont_3fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-3FY)",
    },
    "basic_eps_cont_4fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Basic EPS - Cont (-4FY)",
    },
    # --- NEW: EPS/Adj. Historical ---
    "eps_adj_fq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (FQ)",
    },
    "eps_adj_1fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-1FQFQ)",
    },
    "eps_adj_2fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-2FQFQ)",
    },
    "eps_adj_3fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-3FQFQ)",
    },
    "eps_adj_4fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-4FQFQ)",
    },
    "eps_adj_2fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-2FY)",
    },
    "eps_adj_3fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-3FY)",
    },
    "eps_adj_4fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS/Adj. (-4FY)",
    },
    "cost_of_revenues_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Cost Of Revenues (LTM)",
    },
    "randd_expenses_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "R&D Expenses (LTM)",
    },
    "r_d_expenses": {"dtype": "Float64", "role": "feature"},
    "selling_general_and_admin_expenses_total_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (FQ)",
    },
    "selling_general_and_admin_expenses_total_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (FY)",
    },
    "selling_general_and_admin_expenses_total_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (-1FY)",
    },
    "selling_general_and_admin_expenses_total_5yavgfq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Selling General & Admin Expenses/Total (5YAVGFQ)",
    },
    "sga_expenses": {
        "dtype": "Float64",
        "role": "financial_statement",
        "sql_name": "SG&A Expenses",
        "description": "Selling, General & Administrative expenses (LTM)",
    },
    "accounts_receivable_total_fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (FY)",
    },
    "accounts_receivable_total_1fy": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (-1FY)",
    },
    "accounts_receivable_total_5yavgfq": {
        "dtype": "float",
        "role": "balance_sheet",
        "sql_name": "Accounts Receivable/Total (5YAVGFQ)",
    },
    "marketing_expenses": {"dtype": "Float64", "role": "feature"},
    "marketing_expenses_fq": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (FQ)",
    },
    "marketing_expenses_fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (FY)",
    },
    "marketing_expenses_1fy": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (-1FY)",
    },
    "marketing_expenses_5yavgltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Marketing Expenses (5YAVGLTM)",
    },
    "eps_adj_1fy": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (-1FY)"},
    "eps_adj_fy": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (FY)"},
    "eps_adj_ltm": {"dtype": "Float64", "role": "ratio", "sql_name": "EPS/Adj. (LTM)"},
    "net_eps_basic_ltm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (LTM)",
    },
    "net_eps_basic_fq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (FQ)",
    },
    "net_eps_basic_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (FY)",
    },
    "net_eps_basic_1fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-1FQFQ)",
    },
    "net_eps_basic_2fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-2FQFQ)",
    },
    "net_eps_basic_3fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-3FQFQ)",
    },
    "net_eps_basic_4fqfq": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-4FQFQ)",
    },
    "net_eps_basic_1fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-1FY)",
    },
    "net_eps_basic_2fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-2FY)",
    },
    "net_eps_basic_3fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-3FY)",
    },
    "net_eps_basic_4fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-4FY)",
    },
    "net_eps_basic_5fy": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "Net EPS - Basic (-5FY)",
    },
    "eps_norm_est_avg_ntm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS Norm - Est Avg (NTM)",
    },
    "eps_norm_est_avg_fy1e": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS Norm - Est Avg (FY1E)",
    },
    "eps_norm_est_num_fy1e": {
        "dtype": "float",
        "role": "count",
        "sql_name": "EPS Norm - Est # (FY1E)",
    },
    "eps_est_avg_rev_pct_fy1e_1w": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1W)",
    },
    "eps_est_avg_rev_pct_fy1e_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1M)",
    },
    "eps_est_avg_rev_pct_fy1e_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 3M)",
    },
    "eps_est_avg_rev_pct_fy1e_6m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 6M)",
    },
    "eps_est_avg_rev_pct_fy1e_1y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Est Avg Rev % (FY1E - 1Y)",
    },
    "eps_gaap_est_avg_fy1e": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS GAAP - Est Avg (FY1E)",
    },
    "eps_gaap_est_avg_ntm": {
        "dtype": "Float64",
        "role": "ratio",
        "sql_name": "EPS GAAP - Est Avg (NTM)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 1M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_3m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 3M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_6m": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 6M)",
    },
    "eps_gaap_est_avg_rev_pct_fy1e_1y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS GAAP Est Avg Rev % (FY1E - 1Y)",
    },
    "eps_previous_year": {"dtype": "Float64", "role": "feature"},
    "dividend_per_share_ltm": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Dividend Per Share (LTM)",
    },
    "div_yield_ind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (Ind)",
    },
    "div_yield_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (LTM)",
    },
    "payout_ratio_ttm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Payout Ratio (TTM)",
        "description": "Payout Ratio (Trailing Twelve Months)",
    },
    "dps_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "DPS Growth",
        "description": "Dividend Per Share Growth",
    },
    "eps_growth_ttm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Growth (TTM)",
        "description": "EPS Growth (Trailing Twelve Months)",
    },
    "revenue_growth_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Revenue Growth (3Y)",
        "description": "Revenue Growth (3 Year)",
    },
    "revenue_growth_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Revenue Growth (5Y)",
        "description": "Revenue Growth (5 Year)",
    },
    "eps_growth_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Growth (3Y)",
        "description": "EPS Growth (3 Year)",
    },
    "eps_growth_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "EPS Growth (5Y)",
        "description": "EPS Growth (5 Year)",
    },
    "fcf_margin": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "FCF Margin",
        "description": "Free Cash Flow Margin",
    },
    "div_yield_ttm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (TTM)",
    },
    "div_yield_ntm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (NTM)",
    },
    "div_yield_1fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-1FYInd)",
    },
    "div_yield_2fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-2FYInd)",
    },
    "div_yield_3fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-3FYInd)",
    },
    "div_yield_4fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-4FYInd)",
    },
    "div_yield_5fyind": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (-5FYInd)",
    },
    "div_yield_5yavgltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Div Yield (5YAVGLTM)",
    },
    "common_dividends_paid_ltm": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Common Dividends Paid (LTM)",
    },
    "common_dividends_paid_fy": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Common Dividends Paid (FY)",
    },
    "dividend_record_frequency": {
        "dtype": "string",
        "role": "categorical",
        "sql_name": "Dividend Record (Frequency)",
    },
    "dividend_record_currency": {
        "dtype": "string",
        "role": "categorical",
        "sql_name": "Dividend Record (Currency)",
    },
    "dividend_record_amount": {
        "dtype": "Float64",
        "role": "market",
        "sql_name": "Dividend Record (Amount)",
        "description": "Dividend amount per share",
    },
    "dividend_streak": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Dividend Streak",
        "description": "Consecutive years of dividend payments",
    },
    "days_to_dividend": {"dtype": "Float64", "role": "feature"},
    "buyback_yield_ltm": {
        "dtype": "Float64",
        "role": "percentage",
        "sql_name": "Buyback Yield (LTM)",
    },
    "interest_expense_total_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Interest Expense/Total (LTM)",
    },
    "interest_income_on_investments_ltm": {
        "dtype": "float",
        "role": "financial_statement",
        "sql_name": "Interest Income On Investments (LTM)",
    },
    "employees": {"dtype": "float", "role": "count"},
    "avg_employees_ltm": {"dtype": "float", "role": "count"},
    "avg_employees_fy": {"dtype": "float", "role": "count"},
    "avg_employees_5yavgfy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Avg Employees (5YAVGFY)",
    },
    "total_employees_fy": {"dtype": "float", "role": "count"},
    "total_employees_fq": {"dtype": "float", "role": "count"},
    "full_time_employees_fq": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (FQ)",
        "description": "Full time employees (Fiscal Quarter)",
    },
    "full_time_employees_fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (FY)",
        "description": "Full time employees (Fiscal Year)",
    },
    "full_time_employees_1fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-1FY)",
    },
    "full_time_employees_2fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-2FY)",
    },
    "full_time_employees_3fy": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Full Time Employees (-3FY)",
    },
    "p_e": {"dtype": "Float64", "role": "ratio"},
    "p_b": {"dtype": "Float64", "role": "ratio"},
    "revenue": {"dtype": "float", "role": "financial_statement"},
    "ebitda": {"dtype": "float", "role": "financial_statement"},
    "ebit": {"dtype": "float", "role": "financial_statement"},
    "net_income": {"dtype": "float", "role": "financial_statement"},
    "net_income_ltm": {"dtype": "float", "role": "financial_statement"},
    "gross_margin": {"dtype": "Float64", "role": "percentage"},
    "eps": {"dtype": "Float64", "role": "ratio"},
    "total_equity": {"dtype": "float", "role": "balance_sheet"},
    "total_assets": {"dtype": "float", "role": "balance_sheet"},
    "total_debt": {"dtype": "float", "role": "balance_sheet"},
    "inventory": {"dtype": "float", "role": "balance_sheet"},
    "capex": {"dtype": "float", "role": "cash_flow"},
    "cash_and_equivalents": {"dtype": "float", "role": "balance_sheet"},
    "current_assets": {"dtype": "float", "role": "balance_sheet"},
    "current_liabilities": {"dtype": "Float64", "role": "market"},
    "working_capital": {"dtype": "float", "role": "balance_sheet"},
    "retained_earnings": {"dtype": "float", "role": "balance_sheet"},
    "cfo": {"dtype": "float", "role": "cash_flow"},
    "cfi": {"dtype": "float", "role": "cash_flow"},
    "cff": {"dtype": "float", "role": "cash_flow"},
    "fcf": {"dtype": "float", "role": "cash_flow"},
    "gross_profit": {"dtype": "float", "role": "financial_statement"},
    "operating_income": {"dtype": "float", "role": "financial_statement"},
    "interest_expense": {"dtype": "float", "role": "financial_statement"},
    "goodwill": {"dtype": "float", "role": "balance_sheet"},
    "dividend_per_share": {"dtype": "Float64", "role": "market"},
    "operating_expenses": {"dtype": "float", "role": "financial_statement"},
    "operating_cash_flow": {"dtype": "Float64", "role": "market"},
    "dividends_paid": {"dtype": "float", "role": "cash_flow"},
    "dividends_paid_ltm": {"dtype": "float", "role": "cash_flow"},
    "volatility_1y_pct": {"dtype": "Float64", "role": "percentage"},
    "tangible_book_value": {"dtype": "Float64", "role": "market"},
    "marketing_efficiency": {"dtype": "Float64", "role": "ratio"},
    "r_d_intensity": {"dtype": "Float64", "role": "percentage"},
    "rule_of_40": {"dtype": "Float64", "role": "percentage"},
    "operating_leverage": {"dtype": "Float64", "role": "ratio"},
    "one_day_chg": {"dtype": "Float64", "role": "percentage"},
    "market_cap_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "market_cap_x_roe": {"dtype": "Float64", "role": "feature"},
    "p_e_ratio_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "p_e_ratio_x_roe": {"dtype": "Float64", "role": "feature"},
    "roe_x_debt_to_equity": {"dtype": "Float64", "role": "feature"},
    "log_operating_income": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda": {"dtype": "float", "role": "financial_statement"},
    "log_net_income": {"dtype": "float", "role": "financial_statement"},
    "log_capex": {"dtype": "float", "role": "cash_flow"},
    "log_operating_cash_flow": {"dtype": "Float64", "role": "market"},
    "log_total_equity": {"dtype": "float", "role": "balance_sheet"},
    "log_market_cap": {"dtype": "Float64", "role": "market"},
    "log_total_assets": {"dtype": "float", "role": "balance_sheet"},
    "log_gross_profit": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents": {"dtype": "float", "role": "balance_sheet"},
    "log_total_debt": {"dtype": "float", "role": "balance_sheet"},
    "log_revenue": {"dtype": "float", "role": "financial_statement"},
    "log_enterprise_value": {"dtype": "Float64", "role": "market"},
    "log_gross_profit_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_operating_income_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_total_revenues_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_fq": {"dtype": "float", "role": "cash_flow"},
    "log_total_revenues_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_fy": {"dtype": "float", "role": "financial_statement"},
    "log_total_assets_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_ebitda_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_operating_income_fy": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_ltm": {"dtype": "float", "role": "cash_flow"},
    "log_revenues_est_avg_ntm": {"dtype": "float", "role": "financial_statement"},
    "log_total_revenues_fy": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_is_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_fcf_fq": {"dtype": "float", "role": "cash_flow"},
    "log_total_equity_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_total_revenues_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_adj_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_total_equity_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_total_debt_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_revenue_previous_year": {"dtype": "float", "role": "financial_statement"},
    "log_revenue_fy": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_5yavgfq": {"dtype": "float", "role": "cash_flow"},
    "log_net_income_is_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_cash_acquisitions_fy": {"dtype": "float", "role": "cash_flow"},
    "log_total_assets_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_net_income_adj_fy": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_5yavgltm": {"dtype": "float", "role": "financial_statement"},
    "log_revenues_est_avg_fy1e": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_1fy": {"dtype": "float", "role": "financial_statement"},
    "log_revenues_est_med_ntm": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents_fy": {"dtype": "float", "role": "balance_sheet"},
    "log_net_income_is_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_cash_and_equivalents_5yavgfq": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf_ltm": {"dtype": "float", "role": "cash_flow"},
    "log_total_debt_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf": {"dtype": "float", "role": "cash_flow"},
    "log_gross_profit_fy": {"dtype": "float", "role": "financial_statement"},
    "log_market_cap_country_r": {"dtype": "Float64", "role": "market"},
    "log_cash_and_equivalents_ltm": {"dtype": "float", "role": "balance_sheet"},
    "log_fcf_5yavgfq": {"dtype": "float", "role": "cash_flow"},
    "log_ebitda_5yavgfq": {"dtype": "float", "role": "financial_statement"},
    "log_fcf_fy": {"dtype": "float", "role": "cash_flow"},
    "log_revenues_est_med_fy1e": {"dtype": "float", "role": "financial_statement"},
    "log_total_assets_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "log_operating_income_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_net_income_is_fq": {"dtype": "float", "role": "financial_statement"},
    "log_ebitda_adj_ltm": {"dtype": "float", "role": "financial_statement"},
    "log_gross_profit_ltm": {"dtype": "float", "role": "financial_statement"},
    "p_e_ratio": {"dtype": "Float64", "role": "ratio"},
    "p_s_ratio": {"dtype": "Float64", "role": "ratio"},
    "ev_ebitda_ratio": {"dtype": "Float64", "role": "ratio"},
    "ev_sales_ratio": {"dtype": "Float64", "role": "ratio"},
    "gross_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "operating_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "net_margin_pct": {"dtype": "Float64", "role": "percentage"},
    "roe": {"dtype": "Float64", "role": "ratio"},
    "roa": {"dtype": "Float64", "role": "ratio"},
    "revenue_growth": {"dtype": "Float64", "role": "percentage"},
    "ebitda_growth": {"dtype": "Float64", "role": "percentage"},
    "earnings_growth": {"dtype": "Float64", "role": "percentage"},
    "debt_to_equity": {"dtype": "Float64", "role": "ratio"},
    "debt_to_assets": {"dtype": "Float64", "role": "ratio"},
    "target_vs_price": {"dtype": "Float64", "role": "ratio"},
    "target_vs_price_median": {"dtype": "Float64", "role": "ratio"},
    "peg_ratio": {"dtype": "Float64", "role": "ratio"},
    "dividend_yield": {"dtype": "Float64", "role": "percentage"},
    "roic": {"dtype": "Float64", "role": "ratio"},
    "revenue_previous_year": {"dtype": "float", "role": "financial_statement"},
    "ebitda_previous_year": {"dtype": "float", "role": "financial_statement"},
    "total_equity_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "total_assets_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "gross_profit_previous_year": {"dtype": "float", "role": "financial_statement"},
    "accounts_receivable_previous_year": {"dtype": "float", "role": "balance_sheet"},
    "roa_previous_year": {"dtype": "Float64", "role": "ratio"},
    "current_ratio_previous_year": {"dtype": "Float64", "role": "ratio"},
    "shares_outstanding_previous_year": {"dtype": "float", "role": "count"},
    "gross_margin_pct_previous_year": {"dtype": "Float64", "role": "percentage"},
    "asset_turnover_previous_year": {"dtype": "Float64", "role": "ratio"},
    "revenue_fy": {"dtype": "float", "role": "financial_statement"},
    "working_capital_1fy": {"dtype": "float", "role": "balance_sheet"},
    "cash_burn_rate": {"dtype": "Float64", "role": "feature"},
    "cash_burn_rate_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_ltm": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_ltm_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_fy": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_fy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_trend": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_trend_applicable": {"dtype": "bool", "role": "auxiliary"},
    "revenue_per_employee_vs_5y_pct": {"dtype": "Float64", "role": "feature"},
    "revenue_per_employee_vs_5y_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "assets_per_employee": {"dtype": "Float64", "role": "feature"},
    "assets_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "ebitda_per_employee": {"dtype": "Float64", "role": "feature"},
    "ebitda_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "operating_income_per_employee": {"dtype": "Float64", "role": "feature"},
    "operating_income_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "profit_per_employee": {"dtype": "Float64", "role": "feature"},
    "profit_per_employee_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy": {"dtype": "Float64", "role": "feature"},
    "employee_growth_yoy_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_yoy_pct": {"dtype": "Float64", "role": "feature"},
    "employee_growth_yoy_pct_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_qoq": {"dtype": "Float64", "role": "feature"},
    "employee_growth_qoq_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_cagr_5y": {"dtype": "Float64", "role": "feature"},
    "employee_growth_cagr_5y_applicable": {"dtype": "bool", "role": "auxiliary"},
    "employee_growth_acceleration": {"dtype": "Float64", "role": "feature"},
    "employee_growth_acceleration_applicable": {"dtype": "bool", "role": "auxiliary"},
    "workforce_volatility": {"dtype": "Float64", "role": "feature"},
    "workforce_volatility_applicable": {"dtype": "bool", "role": "auxiliary"},
    "hiring_intensity_score": {"dtype": "Float64", "role": "feature"},
    "hiring_intensity_score_applicable": {"dtype": "bool", "role": "auxiliary"},
    "altman_z_score": {"dtype": "Float64", "role": "feature"},
    "beneish_m_score": {"dtype": "Float64", "role": "feature"},
    "composite_quality_score": {"dtype": "Float64", "role": "feature"},
    "momentum_score": {"dtype": "Float64", "role": "feature"},
    "eps_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "eps_surprise_magnitude": {"dtype": "category", "role": "categorical"},
    "revenue_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "revenue_beat_indicator": {"dtype": "bool", "role": "feature"},
    "ebitda_surprise_pct": {"dtype": "Float64", "role": "percentage"},
    "earnings_beat_indicator": {"dtype": "bool", "role": "feature"},
    "surprise_momentum_score": {"dtype": "Float64", "role": "feature"},
    "positive_revision_momentum": {"dtype": "bool", "role": "feature"},
    "consensus_uncertainty_score": {"dtype": "Float64", "role": "feature"},
    "estimate_revision_acceleration": {"dtype": "Float64", "role": "percentage"},
    "accelerating_upgrades_flag": {"dtype": "bool", "role": "feature"},
    "eps_adjustment_spread_ltm": {"dtype": "Float64", "role": "feature"},
    "eps_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "eps_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "eps_quality_flag_ltm": {"dtype": "bool", "role": "feature"},
    "eps_adjustment_spread_fy": {"dtype": "Float64", "role": "feature"},
    "eps_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "eps_adjustment_pct_fy": {"dtype": "Float64", "role": "percentage"},
    "net_income_adjustment_spread_ltm": {
        "dtype": "float",
        "role": "financial_statement",
    },
    "net_income_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "net_income_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "net_income_adjustment_spread_fy": {
        "dtype": "float",
        "role": "financial_statement",
    },
    "net_income_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_spread_ltm": {"dtype": "float", "role": "financial_statement"},
    "ebitda_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "ebitda_adjustment_spread_fy": {"dtype": "float", "role": "financial_statement"},
    "ebit_adjustment_spread_ltm": {"dtype": "float", "role": "financial_statement"},
    "ebit_adjustment_pct_ltm": {"dtype": "Float64", "role": "percentage"},
    "ebit_adjustment_spread_fy": {"dtype": "float", "role": "financial_statement"},
    "adjustment_consistency_score": {"dtype": "Float64", "role": "feature"},
    "earnings_quality_warning_flag": {"dtype": "bool", "role": "feature"},
    "earnings_quality_score": {"dtype": "Float64", "role": "feature"},
    "exceptional_items_impact_ratio": {"dtype": "Float64", "role": "ratio"},
    "ebit_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "ebit_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_ratio_ltm": {"dtype": "Float64", "role": "ratio"},
    "ebitda_adjustment_ratio_fy": {"dtype": "Float64", "role": "ratio"},
    "ebitda_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "gross_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "net_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "operating_margin_trend": {"dtype": "Float64", "role": "percentage"},
    "days_to_earnings": {"dtype": "Float64", "role": "feature"},
    "earnings_report_recency": {"dtype": "Float64", "role": "feature"},
    "reporting_lag": {
        "dtype": "Float64",
        "role": "feature",
        "sql_name": "Reporting Lag",
    },
    # Temporal features that may contain pd.NA (use nullable Float64)
    "ltm_vs_5yavg_revenue": {
        "dtype": "Float64",
        "role": "feature",
        "description": "LTM revenue vs 5-year average ratio",
    },
    "fq_vs_5yavg_ebitda": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Fiscal quarter EBITDA vs 5-year average ratio",
    },
    "quarterly_volatility_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Coefficient of variation across quarterly EBITDA",
    },
    # =========================================================================
    # MISSING PHASE 9.3 FEATURES - Coverage Gap Fill
    # =========================================================================
    # Analyst Sentiment
    "analyst_coverage_quality": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Quality score based on analyst coverage breadth and consistency",
    },
    "price_target_revision": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Recent price target revision percentage",
    },
    # Technical Analysis
    "rsi_14d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "14-day Relative Strength Index",
    },
    "rsi_30d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "30-day Relative Strength Index",
    },
    "momentum_20d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "20-day price momentum indicator",
    },
    # Quality & Risk
    "distress_risk_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite financial distress probability score",
    },
    "altman_z_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Year-over-year change in Altman Z-Score",
    },
    "z_score_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Standard deviation of Altman Z-Score over time",
    },
    "exceptional_items_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in exceptional/non-recurring items over time",
    },
    # Employee Productivity
    "fte_cagr_3y_pct": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Full-time employee 3-year compound annual growth rate",
    },
    "fte_growth_1y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 1-year growth percentage",
    },
    "fte_growth_2y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 2-year growth percentage",
    },
    "fte_growth_3y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Full-time employee 3-year growth percentage",
    },
    "revenue_per_employee_1fy": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Revenue per employee from previous fiscal year",
    },
    "workforce_volatility_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Workforce size volatility as percentage",
    },
    # Balance Sheet Dynamics
    "asset_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total asset growth rate",
    },
    "balance_sheet_expansion": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite balance sheet expansion indicator",
    },
    "current_ratio_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in current ratio over time",
    },
    "debt_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total debt growth rate",
    },
    "equity_growth_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year total equity growth rate",
    },
    "earnings_retention_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Proportion of earnings retained vs distributed",
    },
    "retained_earnings_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year retained earnings growth",
    },
    "working_capital_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Working capital as ratio of total assets",
    },
    # Revenue Forecasting
    "revenue_forecast_accuracy": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Historical accuracy of revenue forecasts",
    },
    # Valuation Timeseries
    "valuation_extreme_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating extreme valuation vs historical norms",
    },
    "valuation_stability_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of valuation multiples over time",
    },
    "valuation_trend_consistency": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Consistency of valuation trend direction",
    },
    # Earnings Quality
    "earnings_quality_score_composite": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite earnings quality score combining multiple factors",
    },
    "eps_adjustment_ratio_fy": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Ratio of adjusted to GAAP EPS for fiscal year",
    },
    # Dividend Reliability
    "dividend_coverage_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Earnings coverage of dividend payments",
    },
    "dividend_growth_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "3-year dividend growth rate",
    },
    "dividend_growth_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "5-year dividend growth rate",
    },
    "dividend_yield_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of dividend yield over time",
    },
    "fcf_dividend_coverage": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Free cash flow coverage of dividends",
    },
    "payout_consistency_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Consistency of dividend payout ratio over time",
    },
    "sustainable_dividend_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating dividend sustainability",
    },
    # Composite Scores
    "value_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Composite value investing score",
    },
    "piotroski_f_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Piotroski F-Score (0-9 financial strength)",
    },
    # Efficiency Ratios
    "inventory_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Inventory turnover ratio",
    },
    "receivables_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Accounts receivable turnover ratio",
    },
    "asset_turnover": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Asset turnover ratio",
    },
    # Cash Flow
    "cfo_growth_yoy": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year cash from operations growth",
    },
    "cfo_to_net_income": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Cash from operations to net income ratio",
    },
    "fcf_margin": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Free cash flow margin (FCF/Revenue)",
    },
    "fcf_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of free cash flow over time",
    },
    "fcf_to_net_income": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Free cash flow to net income ratio",
    },
    # Leverage & Liquidity
    "cash_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Cash ratio (cash/current liabilities)",
    },
    "current_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Current ratio (current assets/current liabilities)",
    },
    "equity_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Equity ratio (equity/total assets)",
    },
    "interest_coverage": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Interest coverage ratio (EBIT/interest expense)",
    },
    "net_debt_to_ebitda": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Net debt to EBITDA ratio",
    },
    "quick_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Quick ratio (liquid assets/current liabilities)",
    },
    "working_capital_to_sales": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Working capital as percentage of sales",
    },
    # Capital Allocation
    "payout_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Dividend payout ratio",
    },
    "reinvestment_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Rate of earnings reinvestment",
    },
    "retention_rate": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Earnings retention rate (1 - payout ratio)",
    },
    "cash_conversion_cycle": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days in cash conversion cycle",
    },
    # Growth Metrics
    "book_value_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year book value growth",
    },
    "fcf_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year free cash flow growth",
    },
    "operating_income_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Year-over-year operating income growth",
    },
    # Market Sentiment
    "beta_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Stability of beta coefficient over time",
    },
    "systematic_risk_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Trend in systematic risk exposure",
    },
    "price_range_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Price range as percentage of mid-price",
    },
    # Valuation Ratios
    "book_value_per_share": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Book value per share",
    },
    "p_b_ratio": {
        "dtype": "Float64",
        "role": "ratio",
        "description": "Price to book ratio",
    },
    # Temporal Patterns
    "days_since_reference": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days since reference date",
    },
    "quarter_end_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating quarter-end proximity",
    },
    "month_end_flag": {
        "dtype": "bool",
        "role": "feature",
        "description": "Flag indicating month-end proximity",
    },
    "week_of_year": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Week number within year",
    },
    "day_of_week": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Day of week (0=Monday)",
    },
    "month": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Month number (1-12)",
    },
    "year": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Calendar year",
    },
    "fcf_1fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-1FY)"},
    "cfo_1fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-1FQFQ)"},
    "cfo_2fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-2FQFQ)"},
    "cfo_3fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-3FQFQ)"},
    "cfo_4fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-4FQFQ)"},
    "cfi_1fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-1FQFQ)"},
    "cfi_2fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-2FQFQ)"},
    "cfi_3fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-3FQFQ)"},
    "cfi_4fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-4FQFQ)"},
    "cfi_2fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-2FY)"},
    "cfi_3fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-3FY)"},
    "cfi_4fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFI (-4FY)"},
    "fcf_1fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-1FQFQ)"},
    "fcf_2fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-2FQFQ)"},
    "fcf_3fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-3FQFQ)"},
    "fcf_4fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-4FQFQ)"},
    "cff_2fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-2FY)"},
    "cff_3fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-3FY)"},
    "cff_4fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-4FY)"},
    "cff_1fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-1FQFQ)"},
    "cff_2fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-2FQFQ)"},
    "cff_3fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-3FQFQ)"},
    "cff_4fqfq": {"dtype": "float", "role": "cash_flow", "sql_name": "CFF (-4FQFQ)"},
    "cfo_2fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-2FY)"},
    "cfo_3fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-3FY)"},
    "cfo_4fy": {"dtype": "float", "role": "cash_flow", "sql_name": "CFO (-4FY)"},
    "cash_acquisitions_1fqfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-1FQFQ)",
    },
    "cash_acquisitions_2fqfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-2FQFQ)",
    },
    "cash_acquisitions_3fqfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-3FQFQ)",
    },
    "cash_acquisitions_4fqfq": {
        "dtype": "float",
        "role": "cash_flow",
        "sql_name": "Cash Acquisitions (-4FQFQ)",
    },
    "fcf_2fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-2FY)"},
    "fcf_3fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-3FY)"},
    "fcf_4fy": {"dtype": "float", "role": "cash_flow", "sql_name": "FCF (-4FY)"},
    "price_target_1w_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (1W Ago)",
    },
    "price_target_1m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (1M Ago)",
    },
    "price_target_3m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (3M Ago)",
    },
    "price_target_6m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (6M Ago)",
    },
    "price_target_mtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (MTD Ago)",
    },
    "price_target_qtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (QTD Ago)",
    },
    "price_target_1y_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target (1Y Ago)",
    },
    "price_target_count_3m_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (3M Ago)",
    },
    "price_target_count_6m_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (6M Ago)",
    },
    "price_target_count_ytd_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (YTD Ago)",
    },
    "price_target_count_1y_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (1Y Ago)",
    },
    "price_target_count_1w_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (1W Ago)",
    },
    "price_target_count_1m_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (1M Ago)",
    },
    "price_target_count_mtd_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (MTD Ago)",
    },
    "price_target_count_qtd_ago": {
        "dtype": "float",
        "role": "count",
        "sql_name": "Price Target - # (QTD Ago)",
    },
    "price_target_high_1w_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (1W Ago)",
    },
    "price_target_high_1m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (1M Ago)",
    },
    "price_target_high_6m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (6M Ago)",
    },
    "price_target_high_mtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (MTD Ago)",
    },
    "price_target_high_3m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (3M Ago)",
    },
    "price_target_high_qtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (QTD Ago)",
    },
    "price_target_high_1y_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (1Y Ago)",
    },
    "price_target_high_ytd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - High (YTD Ago)",
    },
    "price_target_low_1w_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (1W Ago)",
    },
    "price_target_low_1m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (1M Ago)",
    },
    "price_target_low_3m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (3M Ago)",
    },
    "price_target_low_6m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (6M Ago)",
    },
    "price_target_low_mtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (MTD Ago)",
    },
    "price_target_low_qtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (QTD Ago)",
    },
    "price_target_low_ytd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (YTD Ago)",
    },
    "price_target_low_1y_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Low (1Y Ago)",
    },
    "price_target_median_1w_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (1W Ago)",
    },
    "price_target_median_1m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (1M Ago)",
    },
    "price_target_median_3m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (3M Ago)",
    },
    "price_target_median_6m_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (6M Ago)",
    },
    "price_target_median_mtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (MTD Ago)",
    },
    "price_target_median_qtd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (QTD Ago)",
    },
    "price_target_median_ytd_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (YTD Ago)",
    },
    "price_target_median_1y_ago": {
        "dtype": "float",
        "role": "market",
        "sql_name": "Price Target - Median (1Y Ago)",
    },
    # =============================================================================
    # NEW FEATURES: Phase 9.3 v1.14 Temporal Enhancements
    # =============================================================================
    # --- Price Target Dynamics (Analyst Sentiment) ---
    "pt_momentum_1w": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum (1-week change %)",
    },
    "pt_momentum_1m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum (1-month change %)",
    },
    "pt_momentum_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum (3-month change %)",
    },
    "pt_momentum_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum (6-month change %)",
    },
    "pt_momentum_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum (1-year change %)",
    },
    "pt_acceleration_short": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum acceleration (1M vs 3M)",
    },
    "pt_acceleration_long": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target momentum acceleration (3M vs 1Y)",
    },
    "pt_consensus_convergence": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst consensus convergence (spread narrowing)",
    },
    "analyst_coverage_change_1m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (1-month)",
    },
    "analyst_coverage_change_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (3-month)",
    },
    "pt_vs_price_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target vs price momentum divergence",
    },
    "pt_qtd_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target quarter-to-date momentum",
    },
    "pt_ytd_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target year-to-date momentum",
    },
    "pt_skew_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst estimate skewness trend (mean vs median)",
    },
    "pt_high_low_spread_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "High-low price target spread evolution",
    },
    # NEW: Extended Price Target Dynamics
    "pt_mtd_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target month-to-date momentum",
    },
    "pt_median_momentum_1w": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median momentum (1-week change %)",
    },
    "pt_median_momentum_1m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median momentum (1-month change %)",
    },
    "pt_median_momentum_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median momentum (3-month change %)",
    },
    "pt_median_momentum_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median momentum (6-month change %)",
    },
    "pt_median_momentum_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median momentum (1-year change %)",
    },
    "pt_median_momentum_mtd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median MTD momentum",
    },
    "pt_median_momentum_qtd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median QTD momentum",
    },
    "pt_median_momentum_ytd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target median YTD momentum",
    },
    "pt_high_momentum_1w": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target high estimate 1-week momentum",
    },
    "pt_high_momentum_1m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target high estimate 1-month momentum",
    },
    "pt_high_momentum_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target high estimate 3-month momentum",
    },
    "pt_high_momentum_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target high estimate 6-month momentum",
    },
    "pt_high_momentum_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target high estimate 1-year momentum",
    },
    "pt_low_momentum_1w": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target low estimate 1-week momentum",
    },
    "pt_low_momentum_1m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target low estimate 1-month momentum",
    },
    "pt_low_momentum_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target low estimate 3-month momentum",
    },
    "pt_low_momentum_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target low estimate 6-month momentum",
    },
    "pt_low_momentum_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target low estimate 1-year momentum",
    },
    "pt_median_acceleration_short": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Median price target momentum acceleration (1M vs 3M)",
    },
    "pt_consensus_convergence_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst consensus convergence over 3 months",
    },
    "pt_consensus_convergence_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst consensus convergence over 6 months",
    },
    "pt_consensus_convergence_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst consensus convergence over 1 year",
    },
    "pt_spread_trend_3m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target spread trend over 3 months",
    },
    "pt_spread_trend_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target spread trend over 6 months",
    },
    "pt_spread_trend_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price target spread trend over 1 year",
    },
    "analyst_coverage_change_1w": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (1-week)",
    },
    "analyst_coverage_change_6m": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (6-month)",
    },
    "analyst_coverage_change_1y": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (1-year)",
    },
    "analyst_coverage_change_mtd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (MTD)",
    },
    "analyst_coverage_change_qtd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (QTD)",
    },
    "analyst_coverage_change_ytd": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Change in analyst coverage count (YTD)",
    },
    "analyst_coverage_acceleration": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst coverage momentum acceleration",
    },
    "analyst_interest_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Normalized analyst interest relative to market cap",
    },
    "analyst_rating_normalized": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst rating normalized to 0-100 scale",
    },
    "analyst_rating_conviction": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Analyst rating distance from neutral (conviction strength)",
    },
    "eps_revision_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Weighted EPS estimate revision momentum across time horizons",
    },
    "eps_revision_acceleration": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EPS revision acceleration (short-term vs long-term)",
    },
    "eps_gaap_revision_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Weighted GAAP EPS estimate revision momentum",
    },
    "eps_revision_gaap_divergence": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Divergence between adjusted and GAAP EPS revisions",
    },
    # --- Cash Flow Temporal (Cash Flow) ---
    "fcf_quarterly_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "FCF trend slope across last 5 quarters (normalized)",
    },
    "fcf_quarterly_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "FCF coefficient of variation across quarters",
    },
    "fcf_positive_ratio": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Ratio of positive FCF quarters (0-1)",
    },
    "cfo_quarterly_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "CFO trend slope across last 5 quarters",
    },
    "cfo_yoy_quarterly": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "CFO year-over-year quarterly growth",
    },
    "investment_intensity_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "CFI investment intensity trend (normalized)",
    },
    "cfo_5y_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "CFO 5-year trend slope",
    },
    "cfo_5y_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "CFO 5-year stability score (1 - CV)",
    },
    "cfo_margin_current": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Current CFO margin (CFO/Revenue)",
    },
    "cfo_margin_trend": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "CFO margin trend (current vs prior year)",
    },
    "acquisition_activity_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Cash acquisition activity trend",
    },
    "acquisition_quarters_active": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Number of quarters with acquisition activity",
    },
    # --- EPS Trajectory (Earnings Quality) ---
    "eps_quarterly_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EPS quarterly trend slope (normalized)",
    },
    "eps_quarterly_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EPS quarterly coefficient of variation",
    },
    "eps_yoy_quarterly_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "EPS year-over-year quarterly growth",
    },
    "eps_qoq_growth": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "EPS quarter-over-quarter growth",
    },
    "eps_positive_streak": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Count of positive EPS quarters (last 5)",
    },
    "eps_cagr_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "EPS 5-year compound annual growth rate",
    },
    "eps_cagr_3y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "EPS 3-year compound annual growth rate",
    },
    "eps_annual_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EPS annual trend slope (normalized)",
    },
    "eps_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current EPS vs 5-year average ratio",
    },
    "eps_growth_acceleration": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EPS growth acceleration (3Y CAGR - 5Y CAGR)",
    },
    "normalized_vs_gaap_spread": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Spread between normalized and GAAP net income",
    },
    "normalized_vs_gaap_ratio": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Ratio of normalized to GAAP net income",
    },
    "forward_eps_gaap_adjusted_spread": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Forward GAAP vs adjusted EPS estimate spread",
    },
    "earnings_stability_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Earnings stability (normalized 5Y avg vs current)",
    },
    # --- Balance Sheet Dynamics (Leverage) ---
    "working_capital_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current working capital vs 5-year average",
    },
    "cash_stability_ratio": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Cash position stability ratio (current vs 5Y avg)",
    },
    "inventory_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Inventory efficiency vs 5-year average",
    },
    "goodwill_stability": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Goodwill concentration stability vs 5-year average",
    },
    # --- Profitability Enhancements ---
    "ebitda_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EBITDA vs 5-year average baseline",
    },
    "ebitda_stability_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EBITDA stability score (higher is more stable)",
    },
    "ebit_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EBIT vs 5-year average baseline",
    },
    "operating_leverage_ratio": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Operating income growth vs revenue growth ratio",
    },
    "gross_margin_consistency": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Gross margin consistency (LTM vs FY)",
    },
    # --- Workforce Analytics (Employment) ---
    "fte_growth_1y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "1-year full-time employee growth percentage",
    },
    "fte_growth_2y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "2-year full-time employee growth percentage",
    },
    "fte_cagr_3y_pct": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "3-year FTE compound annual growth rate",
    },
    "fte_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current FTE vs 5-year average",
    },
    "workforce_stability_score": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Workforce stability indicator vs 5Y average",
    },
    # --- Accounting Quality & Risk (Quality) ---
    "impairment_of_goodwill_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current goodwill impairment vs 5-year average",
    },
    "asset_writedown_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current asset writedown vs 5-year average",
    },
    "restructuring_charges_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current restructuring charges vs 5-year average",
    },
    "merger_and_restructuring_charges_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current merger/restructuring charges vs 5Y average",
    },
    "other_unusual_to_ebitda": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Other unusual items relative to EBITDA",
    },
    "exceptional_items_frequency": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Frequency count of exceptional items (impairments, etc.)",
    },
    # --- Momentum & Technical (Momentum) ---
    "price_momentum_5d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Short-term 5-day price momentum",
    },
    "price_vs_ema_100d": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Price position relative to 100-day EMA",
    },
    "volatility_regime": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Volatility regime indicator (current vs long-term)",
    },
    "volatility_compression": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Volatility compression (1Y - 1M volatility)",
    },
    "volatility_term_structure": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Volatility term structure (3M - 6M volatility)",
    },
    "high_volume_flag": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Indicator flag for high relative volume (>1.5)",
    },
    "low_volume_flag": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Indicator flag for low relative volume (<0.5)",
    },
    "return_acceleration": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Return acceleration (3Y CAGR - 10Y CAGR)",
    },
    # --- Valuation Timeseries (Valuation) ---
    "ev_sales_quarterly_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "EV/Sales quarterly trajectory volatility",
    },
    "ev_sales_trend_consistency": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Consistency of EV/Sales trend direction",
    },
    "p_e_qoq_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Sequential P/E quarter-over-quarter momentum",
    },
    "p_e_yoy_momentum": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Sequential P/E year-over-year momentum",
    },
    "p_b_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current P/B vs 5-year average baseline",
    },
    "p_b_mean_reversion_signal": {
        "dtype": "Int64",
        "role": "feature",
        "description": "P/B mean reversion signal based on 5Y avg",
    },
    # --- Dividend Reliability (Dividends) ---
    "dividend_yield_volatility": {
        "dtype": "Float64",
        "role": "feature",
        "description": "3-year dividend yield volatility",
    },
    "dividend_yield_trend": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Annualized dividend yield trend",
    },
    "dividend_yield_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current dividend yield vs 5-year average",
    },
    "dividend_payout_growth": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Growth in actual common dividends paid",
    },
    "dividend_consistency_years": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Count of years with positive dividend yield (last 5)",
    },
    "dividend_yield_cagr_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "5-year dividend yield CAGR",
    },
    # --- Revenue Forecasting (Revenue) ---
    "revenue_estimate_skew": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Skewness in revenue estimates (avg vs median)",
    },
    "ebitda_margin_improvement_expected": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Expected improvement in EBITDA margin (forward vs current)",
    },
    "forward_ebit_margin": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Forward EBIT margin estimate",
    },
    "analyst_estimate_coverage": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Number of analysts providing forward estimates",
    },
    "high_coverage_flag": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Flag for high analyst coverage (>=10)",
    },
    "revenue_estimate_alignment": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Alignment between NTM and FY1E revenue estimates",
    },
    # --- Growth Metrics (Growth) ---
    "revenue_cagr_5y": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Directly available 5-year revenue CAGR",
    },
    "revenue_vs_5y_avg": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Current revenue vs 5-year average growth baseline",
    },
    "revenue_above_5y_avg_flag": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Flag for revenue significantly above 5Y average (>10%)",
    },
    "operating_income_growth_yoy": {
        "dtype": "Float64",
        "role": "percentage",
        "description": "Operating income year-over-year growth",
    },
    # --- Fiscal Calendar (Temporal Patterns) ---
    "fiscal_year_progress": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Progress through fiscal year (0-1 scale)",
    },
    "days_to_quarter_end": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Days until fiscal quarter end",
    },
    "fiscal_half": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Fiscal half indicator (1 or 2)",
    },
    "reporting_lag_zscore": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Reporting lag z-score vs median",
    },
    "late_reporter_flag": {
        "dtype": "boolean",
        "role": "feature",
        "description": "Flag for late reporting (>60 days)",
    },
    "days_since_fy_end": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Days since fiscal year end",
    },
    "days_to_next_fy_end": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Days until next fiscal year end",
    },
    "earnings_imminent": {
        "dtype": "boolean",
        "role": "feature",
        "description": "Earnings within 14 days flag",
    },
    "pre_earnings_window": {
        "dtype": "boolean",
        "role": "feature",
        "description": "Within 30-day pre-earnings window",
    },
    # --- Dividend Timing (Dividend Reliability) ---
    "days_to_dividend_ex_date": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days until dividend ex-date",
    },
    "days_to_dividend_record_date": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days until dividend record date",
    },
    "days_to_dividend_payable_date": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days until dividend payable date",
    },
    "approaching_ex_date": {
        "dtype": "boolean",
        "role": "feature",
        "description": "Within 7 days of ex-dividend date",
    },
    "recently_ex_dividend": {
        "dtype": "boolean",
        "role": "feature",
        "description": "Went ex-dividend within last 7 days",
    },
    "dividend_cycle_days": {
        "dtype": "Int64",
        "role": "feature",
        "description": "Days in dividend payment cycle",
    },
    "dividend_cycle_position": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Position within dividend cycle (0-1)",
    },
    "dividend_announcement_recency": {
        "dtype": "Float64",
        "role": "feature",
        "description": "Days since last dividend announcement",
    },
}

# Phase 9.3 Feature Input Categorization (v1.15)
# Total: 460+ features across 21 categories
PHASE93_FEATURE_CATEGORIES: Dict[str, List[str]] = {
    # =========================================================================
    # MOMENTUM & TECHNICAL (25 features)
    # =========================================================================
    "Momentum & Technical": [
        "52w_range_position",
        "breakout_signal",
        "ema_crossover_20_50",
        "ema_crossover_50_250",
        "ema_slope_20d",
        "ema_trend_consistency",
        "ma_20d_simple",
        "ma_50d_simple",
        "ma_crossover_signal",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "pct_above_52w_low",
        "pct_off_52w_high",
        "price_acceleration_3m",
        "price_distance_from_ma",
        "price_momentum_1m",
        "price_momentum_1y",
        "price_momentum_3m",
        "price_momentum_6m",
        "price_vs_ema_20d",
        "price_vs_ema_250d",
        "return_stability_score",
        "sharpe_proxy",
        "total_return_1y_pct",
        "volume_momentum_score",
    ],
    # =========================================================================
    # VALUATION RATIOS (25 features)
    # =========================================================================
    "Valuation Ratios": [
        "book_value_per_share",
        "dividend_yield",
        "ev_ebitda_forward_discount",
        "ev_ebitda_momentum",
        "ev_ebitda_ratio",
        "ev_ebitda_vs_3y_avg",
        "ev_sales_forward_discount",
        "ev_sales_quarterly_volatility",
        "ev_sales_ratio",
        "ev_sales_trend_1y",
        "ev_sales_trend_3y",
        "ev_sales_vs_3y_avg",
        "growth_implied_by_valuation",
        "p_b",
        "p_b_ratio",
        "p_e_forward_discount",
        "p_e_momentum_qoq",
        "p_e_momentum_yoy",
        "p_e_ratio",
        "p_e_vs_3y_avg",
        "p_s_ratio",
        "peg_ratio",
        "valuation_extreme_flag",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ],
    # =========================================================================
    # PROFITABILITY (16 features)
    # =========================================================================
    "Profitability": [
        "ebit_adjustment_ratio_fy",
        "ebit_adjustment_ratio_ltm",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_margin_trend",
        "gross_margin_pct",
        "gross_margin_trend",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
        "net_margin_pct",
        "net_margin_trend",
        "operating_leverage",
        "operating_margin_pct",
        "operating_margin_trend",
        "roa",
        "roe",
        "roic",
    ],
    # =========================================================================
    # QUALITY & RISK (18 features)
    # =========================================================================
    "Quality & Risk": [
        "accounting_quality_score",
        "altman_z_score",
        "altman_z_trend",
        "beneish_m_score",
        "distress_risk_score",
        "exceptional_items_to_ebitda",
        "exceptional_items_to_ni_pct",
        "exceptional_items_trend",
        "goodwill_change_rate",
        "goodwill_impairment_flag",
        "goodwill_to_assets",
        "goodwill_to_assets_pct",
        "has_asset_writedown",
        "has_goodwill_impairment",
        "has_restructuring",
        "intangible_intensity",
        "intangibles_to_assets_pct",
        "restructuring_intensity",
        "total_exceptional_items_ltm",
        "z_score_volatility",
    ],
    # =========================================================================
    # CASH FLOW (17 features)
    # =========================================================================
    "Cash Flow": [
        "cfo_growth_yoy",
        "cfo_to_net_income",
        "fcf_margin",
        "fcf_stability",
        "fcf_to_net_income",
        # NEW: Cash Flow Temporal (12)
        "fcf_quarterly_trend",
        "fcf_quarterly_volatility",
        "fcf_positive_ratio",
        "cfo_quarterly_trend",
        "cfo_yoy_quarterly",
        "investment_intensity_trend",
        "cfo_5y_trend",
        "cfo_5y_stability",
        "cfo_margin_current",
        "cfo_margin_trend",
        "acquisition_activity_trend",
        "acquisition_quarters_active",
    ],
    # =========================================================================
    # CAPITAL ALLOCATION (23 features)
    # =========================================================================
    "Capital Allocation": [
        "buyback_yield_ltm",
        "cash_conversion_cycle",
        "common_dividends_paid_fy",
        "common_dividends_paid_ltm",
        "dividend_per_share",
        "dividend_per_share_ltm",
        "dividend_record_amount",
        "dividend_streak",
        "div_yield_1fyind",
        "div_yield_2fyind",
        "div_yield_3fyind",
        "div_yield_4fyind",
        "div_yield_5fyind",
        "div_yield_5yavgltm",
        "div_yield_ind",
        "div_yield_ltm",
        "div_yield_ntm",
        "div_yield_ttm",
        "dividends_paid",
        "dividends_paid_ltm",
        "payout_ratio",
        "reinvestment_rate",
        "retention_rate",
    ],
    # =========================================================================
    # ANALYST SENTIMENT (65+ features)
    # =========================================================================
    "Analyst Sentiment": [
        "analyst_bullish_pct",
        "analyst_bearish_pct",
        "analyst_conviction",
        "analyst_coverage_quality",
        "consensus_strength",
        "price_target_range",
        "price_target_revision",
        "price_target_spread_pct",
        "target_price_upside_pct",
        "upside_potential",
        "analyst_rating_normalized",
        "analyst_rating_conviction",
        "eps_revision_momentum",
        "eps_revision_acceleration",
        "eps_gaap_revision_momentum",
        "eps_revision_gaap_divergence",
        # NEW: Price Target Dynamics (30+)
        "pt_momentum_1w",
        "pt_momentum_1m",
        "pt_momentum_3m",
        "pt_momentum_6m",
        "pt_momentum_1y",
        "pt_mtd_momentum",
        "pt_qtd_momentum",
        "pt_ytd_momentum",
        "pt_median_momentum_1w",
        "pt_median_momentum_1m",
        "pt_median_momentum_3m",
        "pt_median_momentum_6m",
        "pt_median_momentum_1y",
        "pt_median_momentum_mtd",
        "pt_median_momentum_qtd",
        "pt_median_momentum_ytd",
        "pt_high_momentum_1w",
        "pt_high_momentum_1m",
        "pt_high_momentum_3m",
        "pt_high_momentum_6m",
        "pt_high_momentum_1y",
        "pt_low_momentum_1w",
        "pt_low_momentum_1m",
        "pt_low_momentum_3m",
        "pt_low_momentum_6m",
        "pt_low_momentum_1y",
        "pt_acceleration_short",
        "pt_acceleration_long",
        "pt_median_acceleration_short",
        "pt_consensus_convergence",
        "pt_consensus_convergence_3m",
        "pt_consensus_convergence_6m",
        "pt_consensus_convergence_1y",
        "pt_spread_trend_3m",
        "pt_spread_trend_6m",
        "pt_spread_trend_1y",
        "analyst_coverage_change_1w",
        "analyst_coverage_change_1m",
        "analyst_coverage_change_3m",
        "analyst_coverage_change_6m",
        "analyst_coverage_change_1y",
        "analyst_coverage_change_mtd",
        "analyst_coverage_change_qtd",
        "analyst_coverage_change_ytd",
        "analyst_coverage_trend",
        "analyst_coverage_acceleration",
        "analyst_interest_score",
        "pt_vs_price_momentum",
        "pt_skew_trend",
        "pt_high_low_spread_trend",
    ],
    # =========================================================================
    # MARKET SENTIMENT (4 features)
    # =========================================================================
    "Market Sentiment": [
        "beta_stability",
        "one_day_chg",
        "systematic_risk_trend",
        "price_range_pct",
    ],
    # =========================================================================
    # LEVERAGE & LIQUIDITY (9 features)
    # =========================================================================
    "Leverage & Liquidity": [
        "cash_ratio",
        "current_ratio",
        "debt_to_assets",
        "debt_to_equity",
        "equity_ratio",
        "interest_coverage",
        "net_debt_to_ebitda",
        "quick_ratio",
        "working_capital_to_sales",
    ],
    # =========================================================================
    # TEMPORAL PATTERNS (26 features)
    # =========================================================================
    "Temporal Patterns": [
        "reference_date",
        "days_since_reference",
        "days_to_dividend",
        "days_to_earnings",
        "earnings_report_recency",
        "fiscal_quarter",
        "fiscal_year",
        "month",
        "reporting_lag",
        "year",
        "quarter_end_flag",
        "month_end_flag",
        "week_of_year",
        "day_of_week",
        "ltm_vs_5yavg_revenue",
        "fq_vs_5yavg_ebitda",
        "quarterly_volatility_score",
        # NEW: Fiscal Calendar (9)
        "fiscal_year_progress",
        "days_to_quarter_end",
        "fiscal_half",
        "reporting_lag_zscore",
        "late_reporter_flag",
        "days_since_fy_end",
        "days_to_next_fy_end",
        "earnings_imminent",
        "pre_earnings_window",
    ],
    # =========================================================================
    # COMPOSITE SCORES (5 features)
    # =========================================================================
    "Composite Scores": [
        "composite_quality_score",
        "earnings_quality_score",
        "momentum_score",
        "piotroski_f_score",
        "value_score",
    ],
    # =========================================================================
    # GROWTH METRICS (9 features)
    # =========================================================================
    "Growth Metrics": [
        "earnings_growth",
        "ebitda_growth",
        "ebitda_growth_yoy",
        "eps_growth_yoy",
        "revenue_growth",
        "revenue_growth_yoy",
        "book_value_growth",
        "fcf_growth",
        "operating_income_growth",
    ],
    # =========================================================================
    # EFFICIENCY RATIOS (4 features)
    # =========================================================================
    "Efficiency Ratios": [
        "asset_turnover",
        "inventory_turnover",
        "receivables_turnover",
        "revenue_per_employee",
    ],
    # =========================================================================
    # EMPLOYEE PRODUCTIVITY (21 features)
    # =========================================================================
    "Employee Productivity": [
        "assets_per_employee",
        "ebitda_per_employee",
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_yoy",
        "fte_cagr_3y_pct",
        "fte_growth_1y_pct",
        "fte_growth_2y_pct",
        "fte_growth_3y_pct",
        "operating_income_per_employee",
        "profit_per_employee",
        "revenue_per_employee_1fy",
        "revenue_per_employee_fy",
        "revenue_per_employee_trend",
        "workforce_volatility",
        "workforce_volatility_pct",
        "hiring_intensity_score",
        "revenue_per_employee_ltm",
        "revenue_per_employee_vs_5y_pct",
        "employee_growth_yoy_pct",
    ],
    # =========================================================================
    # BALANCE SHEET DYNAMICS (9 features)
    # =========================================================================
    "Balance Sheet Dynamics": [
        "asset_growth_rate",
        "balance_sheet_expansion",
        "cash_ratio",
        "current_ratio_trend",
        "debt_growth_rate",
        "earnings_retention_rate",
        "equity_growth_rate",
        "retained_earnings_growth",
        "working_capital_ratio",
    ],
    # =========================================================================
    # REVENUE FORECASTING (9 features)
    # =========================================================================
    "Revenue Forecasting": [
        "eps_est_avg_rev_pct_fy1e_1m",
        "eps_est_avg_rev_pct_fy1e_1w",
        "eps_est_avg_rev_pct_fy1e_1y",
        "eps_est_avg_rev_pct_fy1e_3m",
        "eps_est_avg_rev_pct_fy1e_6m",
        "revenue_forecast_accuracy",
        "revenues_est_avg_fy1e",
        "revenues_est_avg_ntm",
        "revenues_est_yoy_pct_fy1e",
    ],
    # =========================================================================
    # EARNINGS QUALITY (43 features)
    # =========================================================================
    "Earnings Quality": [
        "accelerating_upgrades_flag",
        "consensus_uncertainty_score",
        "earnings_beat_indicator",
        "eps_surprise_magnitude",
        "eps_surprise_pct",
        "estimate_revision_acceleration",
        "ebitda_surprise_pct",
        "positive_revision_momentum",
        "revenue_beat_indicator",
        "revenue_surprise_pct",
        "surprise_momentum_score",
        # GAAP vs. Adjusted Analytics (22 features)
        "adjustment_consistency_score",
        "earnings_quality_score_composite",
        "earnings_quality_warning_flag",
        "ebit_adjustment_ratio_fy",
        "ebit_adjustment_ratio_ltm",
        "ebit_adjustment_spread_fy",
        "ebit_adjustment_spread_ltm",
        "ebitda_adjustment_ratio_fy",
        "ebitda_adjustment_ratio_ltm",
        "ebitda_adjustment_spread_fy",
        "ebitda_adjustment_spread_ltm",
        "eps_adjustment_ratio_fy",
        "eps_adjustment_ratio_ltm",
        "eps_adjustment_spread_fy",
        "eps_adjustment_spread_ltm",
        "eps_quality_flag_ltm",
        "exceptional_items_impact_ratio",
        "net_income_adjustment_pct_ltm",
        "net_income_adjustment_ratio_fy",
        "net_income_adjustment_ratio_ltm",
        "net_income_adjustment_spread_fy",
        "net_income_adjustment_spread_ltm",
        # NEW: EPS Trajectory (10)
        "eps_quarterly_trend",
        "eps_quarterly_volatility",
        "eps_yoy_quarterly_growth",
        "eps_qoq_growth",
        "eps_positive_streak",
        "eps_cagr_5y",
        "eps_cagr_3y",
        "eps_annual_trend",
        "eps_vs_5y_avg",
        "eps_growth_acceleration",
    ],
    # =========================================================================
    # TECHNICAL ANALYSIS (15 features)
    # =========================================================================
    "Technical Analysis": [
        "52w_range_position",
        "breakout_signal",
        "ema_crossover_20_50",
        "ema_crossover_50_250",
        "ema_slope_20d",
        "ema_trend_consistency",
        "momentum_20d",
        "near_52w_high_flag",
        "near_52w_low_flag",
        "pct_above_52w_low",
        "pct_off_52w_high",
        "price_vs_ema_20d",
        "price_vs_ema_250d",
        "rsi_14d",
        "rsi_30d",
    ],
    # =========================================================================
    # VALUATION TIMESERIES (16 features)
    # =========================================================================
    "Valuation Timeseries": [
        "ev_ebitda_forward_discount",
        "ev_ebitda_momentum",
        "ev_ebitda_vs_3y_avg",
        "ev_sales_forward_discount",
        "ev_sales_quarterly_volatility",
        "ev_sales_trend_1y",
        "ev_sales_trend_3y",
        "ev_sales_vs_3y_avg",
        "growth_implied_by_valuation",
        "p_e_forward_discount",
        "p_e_momentum_qoq",
        "p_e_momentum_yoy",
        "p_e_vs_3y_avg",
        "valuation_extreme_flag",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ],
    # =========================================================================
    # DIVIDEND RELIABILITY (20 features)
    # =========================================================================
    "Dividend Reliability": [
        "days_to_dividend",
        "dividend_coverage_ratio",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_payout_ratio",
        "dividend_reliability_score",
        "dividend_streak",
        "dividend_yield_stability",
        "div_yield_5yavgltm",
        "fcf_dividend_coverage",
        "payout_consistency_score",
        "sustainable_dividend_flag",
        # NEW: Dividend Timing (8)
        "days_to_dividend_ex_date",
        "days_to_dividend_record_date",
        "days_to_dividend_payable_date",
        "approaching_ex_date",
        "recently_ex_dividend",
        "dividend_cycle_days",
        "dividend_cycle_position",
        "dividend_announcement_recency",
    ],
    # =========================================================================
    # EMPLOYMENT DYNAMICS (10 features)
    # =========================================================================
    "Employment Dynamics": [
        "employee_base_scale_flag",
        "employee_growth_acceleration",
        "employee_growth_cagr_5y",
        "employee_growth_yoy",
        "hiring_intensity_score",
        "profit_per_employee",
        "revenue_per_employee_fy",
        "revenue_per_employee_trend",
        "workforce_volatility",
        "fte_cagr_3y_pct",
    ],
}


def get_sql_column_name(normalized_name: str) -> str:
    """Get original SQL column name from normalized Python name."""
    meta = COLUMN_SCHEMA.get(normalized_name)
    if meta and "sql_name" in meta and meta["sql_name"]:
        return meta["sql_name"]
    return normalized_name


def normalize_column_name(column: str) -> str:
    """Standardize column names to lowercase with underscores."""
    if "R&D" in column or "r&d" in column.lower():
        column = column.replace("R&D", "RandD").replace("r&d", "randd")

    normalized = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")
        .replace("%", "pct")
        .replace("&", "and")
    )
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def generate_sql_schema() -> str:
    """Generate CREATE TABLE statement from COLUMN_SCHEMA."""
    lines = ["CREATE TABLE IF NOT EXISTS equities ("]
    for col_name, meta in COLUMN_SCHEMA.items():
        sql_name = meta.get("sql_name") or col_name
        dtype = meta.get("dtype", "float")
        sql_type = {
            "float": "NUMERIC",
            "Float64": "NUMERIC",
            "int": "INTEGER",
            "Int64": "INTEGER",
            "string": "TEXT",
            "category": "TEXT",
            "datetime64[ns]": "DATE",
            "bool": "BOOLEAN",
            "boolean": "BOOLEAN",
        }.get(dtype, "NUMERIC")
        lines.append(f'  "{sql_name}" {sql_type},')
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def get_expected_dtype(column: str) -> str:
    """Get the expected pandas-compatible dtype string for a column."""
    meta = COLUMN_SCHEMA.get(column, {})
    return meta.get("dtype", "float")


def list_numeric_feature_cols() -> List[str]:
    """List all numeric feature columns from COLUMN_SCHEMA."""
    numeric_dtypes = {"float", "Float64", "int", "Int64"}
    feature_roles = {
        "feature",
        "target",
        "target_fallback",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
        "count",
    }

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") in numeric_dtypes and meta.get("role") in feature_roles
    ]


def list_categorical_cols() -> List[str]:
    """List all categorical columns from COLUMN_SCHEMA."""
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") == "category" or meta.get("role") == "categorical"
    ]


def list_date_cols() -> List[str]:
    """List all date/datetime columns from COLUMN_SCHEMA."""
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("dtype") == "datetime64[ns]" or meta.get("role") == "date"
    ]


def get_role_default_value(role: Role) -> Any:
    """Default fill value for a given semantic role (ingestion alignment)."""
    return ROLE_DEFAULTS.get(role)


def get_column_default_value(column: str) -> Any:
    """Default fill value inferred from a column's role."""
    meta = COLUMN_SCHEMA.get(column, {})
    role = meta.get("role")
    return ROLE_DEFAULTS.get(role)


def list_etl_generated_column_patterns() -> List[str]:
    """List regex patterns for columns legitimately generated during ETL."""
    return [
        r"^log_[0-9a-z_]+$",  # Log-transformed columns
        r"^.*_applicable$",  # Conditional metric applicability flags
        r"^event_prob_.*$",  # Classification probabilities
        r"^sector_[0-9a-z]+_x_[0-9a-z_]+$",  # Sector interactions
        r"^.*_(ratio|pct|margin|growth|yoy)$",  # Common semantic/derived suffixes
        r"^.*_formatted$",  # Standardized date string representations
        r"^fy_end_vs_isrd_days$",  # Fiscal year-end to income statement report delta
        r"^fiscal_quarter_inferred$",  # Inferred fiscal quarter label
    ]


def list_required_schema_columns_for_etl(include_extended_financials: bool = False) -> List[str]:
    """List columns required for minimal ETL operations."""
    required = [
        "ticker", "isin", "sector", "region", "country", "trading_country",
        "last_price", "price_target", "market_cap"
    ]
    if include_extended_financials:
        required.extend(["enterprise_value", "ebitda_ltm", "total_revenues_ltm"])
    return required


def list_non_recurring_cols() -> List[str]:
    """List all non-recurring exceptional item columns from COLUMN_SCHEMA.

    These columns represent rare/exceptional events where missing values
    typically mean the event did not occur. Zero is the economically
    correct imputation for these items.

    Returns:
        List of column names with role='non_recurring'
    """
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == "non_recurring"]


def list_knn_imputable_cols() -> List[str]:
    """List all columns suitable for KNN imputation from COLUMN_SCHEMA.

    These are core financial metrics where KNN can leverage sector relationships
    and correlations to provide better estimates than simple statistics.

    Includes columns with roles: feature, market, financial_statement,
    balance_sheet, cash_flow, ratio, percentage.

    Excludes: non_recurring (zero imputation), count (median imputation),
    id, categorical, date, target, auxiliary.

    Returns:
        List of column names suitable for KNN imputation
    """
    knn_roles = {
        "feature",
        "market",
        "financial_statement",
        "balance_sheet",
        "cash_flow",
        "ratio",
        "percentage",
    }

    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") in knn_roles
        and meta.get("dtype") in ["float", "Float64", "int", "Int64", "bool", "boolean"]
    ]


def list_count_cols() -> List[str]:
    """List all count columns from COLUMN_SCHEMA.

    These are discrete integer columns (analyst ratings, employees, shares)
    that should use median imputation rather than KNN.

    Returns:
        List of column names with role='count'
    """
    return [
        col
        for col, meta in COLUMN_SCHEMA.items()
        if meta.get("role") == "count"
        and meta.get("dtype") in ["float", "Float64", "int", "Int64"]
    ]


def list_price_cols() -> List[str]:
    """List all price-related columns from COLUMN_SCHEMA.

    These columns should be used for price imputation (filling missing
    price targets with last_price as fallback).

    Returns:
        List of column names with role in ('market', 'target', 'target_fallback')
        and containing price-related semantics
    """
    price_roles = {"market", "target", "target_fallback"}
    price_keywords = {"price", "target", "ema_", "ma_", "52w_"}

    result = []
    for col, meta in COLUMN_SCHEMA.items():
        role = meta.get("role", "")
        if role in price_roles:
            # Check if column name suggests price-related data
            if any(kw in col.lower() for kw in price_keywords):
                result.append(col)

    return result


def get_pandas_nullable_dtype(dtype: str) -> str:
    """Convert schema dtype to pandas nullable-safe equivalent.

    Use this when the resulting Series may contain pd.NA values
    (e.g., after division with .replace(0, pd.NA)).

    Args:
        dtype: Schema dtype string (e.g., 'float', 'int', 'bool')

    Returns:
        Pandas nullable dtype string (e.g., 'Float64', 'Int64', 'boolean')
    """
    nullable_map = {
        "float": "Float64",
        "int": "Int64",
        "bool": "boolean",
    }
    return nullable_map.get(dtype, dtype)


def get_numpy_dtype(dtype: str) -> str:
    """Convert schema dtype to numpy-compatible equivalent.

    Use this when you need standard numpy dtypes (e.g., for scikit-learn).
    Note: Will fail if Series contains pd.NA - convert NA to np.nan first.

    Args:
        dtype: Schema dtype string

    Returns:
        NumPy-compatible dtype string
    """
    numpy_map = {
        "Float64": "float64",
        "Int64": "int64",
        "boolean": "bool",
        "float": "float64",
        "int": "int64",
    }
    return numpy_map.get(dtype, dtype)
