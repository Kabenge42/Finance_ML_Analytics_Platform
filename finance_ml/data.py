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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names to lowercase with underscores.
    
    Converts column names to lowercase, replaces non-alphanumeric characters
    with underscores, and strips leading/trailing underscores.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with normalized column names
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^0-9a-zA-Z]+", "_", regex=True)
        .str.strip("_")
        .str.lower()
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
        raise ImportError("SQLAlchemy not available. Install psycopg2-binary and SQLAlchemy or use CSV data source.")

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
                msg_parts.append("and at least one target column among: price_target, price_target_median")
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
            report[col] = {
                'count': missing_count,
                'percentage': percentage
            }
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
    series = pd.to_numeric(df[column], errors='coerce')
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
    price_cols = [c for c in df.columns if 'price' in c.lower()]
    for col in price_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
            invalid_mask = series < 0
            invalid_indices = invalid_mask[invalid_mask].index.tolist()
            if invalid_indices:
                issues[col] = invalid_indices
    
    # Check market cap columns (should be positive)
    cap_cols = [c for c in df.columns if 'market_cap' in c.lower() or 'marketcap' in c.lower()]
    for col in cap_cols:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors='coerce')
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
