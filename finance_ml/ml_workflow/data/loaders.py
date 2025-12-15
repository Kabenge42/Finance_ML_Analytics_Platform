"""
Data loading utilities for Finance ML.

Phase 8 (Restructuring Plan):
This module consolidates data loading functions that were previously
in preprocessing/data.py, providing a clean interface for loading
financial data from various sources.

Functions:
- load_from_csv: Load data from CSV files with region inference
- load_from_db: Load data from PostgreSQL database
- load_from_all_stocks: Load from all_stocks directory structure
- load_multiple_csvs: Load and combine multiple CSV files

Usage:
    from finance_ml.ml_workflow.data.loaders import load_from_csv, load_from_db

    # Load from CSV
    df = load_from_csv("data/screening_us.csv")

    # Load from database
    df = load_from_db(db_url="postgresql://localhost/postgres")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, List, Union

import pandas as pd

logger = logging.getLogger(__name__)


def infer_region_from_filename(filename: str) -> Optional[str]:
    """Infer region from filename pattern.

    Args:
        filename: Path or filename string

    Returns:
        Region code ('US', 'EU', 'APAC', 'ROTW') or None

    Example:
        >>> infer_region_from_filename("screening_us.csv")
        'US'
        >>> infer_region_from_filename("screening_apac.csv")
        'APAC'
    """
    fname_lower = str(filename).lower()

    if "_us" in fname_lower or "screening_us" in fname_lower:
        return "US"
    elif "_eu" in fname_lower or "screening_eu" in fname_lower:
        return "EU"
    elif "_apac" in fname_lower or "screening_apac" in fname_lower:
        return "APAC"
    elif "_rotw" in fname_lower or "screening_rotw" in fname_lower:
        return "ROTW"

    return None


def load_from_csv(
    filepath: Union[str, Path],
    region: Optional[str] = None,
    normalize_columns: bool = True,
    encoding: str = "utf-8",
    validate_schema: bool = False,
) -> pd.DataFrame:
    """Load financial data from a CSV file.

    Args:
        filepath: Path to CSV file
        region: Region code to assign (if None, inferred from filename)
        normalize_columns: Whether to normalize column names
        encoding: File encoding
        validate_schema: Whether to validate columns against COLUMN_SCHEMA

    Returns:
        DataFrame with loaded data

    Example:
        >>> df = load_from_csv("data/screening_us.csv")
        >>> df.columns
        Index(['ticker', 'sector', 'last_price', ...])
        >>> df = load_from_csv("data/screening_us.csv", validate_schema=True)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    logger.info(f"Loading CSV from {filepath}")

    # Load with string dtype initially to avoid mixed type issues
    df = pd.read_csv(filepath, encoding=encoding, low_memory=False)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Infer region if not provided
    if region is None:
        region = infer_region_from_filename(str(filepath))

    # Add or update region column
    if region is not None:
        if "Region" in df.columns:
            df["Region"] = df["Region"].fillna(region)
        elif "region" in df.columns:
            df["region"] = df["region"].fillna(region)
        else:
            df["region"] = region
        logger.info(f"Region set to: {region}")

    # Normalize column names if requested
    if normalize_columns:
        df = _normalize_column_names(df)

    # Optional schema validation
    if validate_schema:
        from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

        unknown_cols = [col for col in df.columns if col not in COLUMN_SCHEMA]
        if unknown_cols:
            logger.warning(
                f"Found {len(unknown_cols)} columns not in COLUMN_SCHEMA: "
                f"{unknown_cols[:5]}{'...' if len(unknown_cols) > 5 else ''}"
            )

    return df


def load_from_db(
    db_url: Optional[str] = None,
    table_name: str = "equities",
    query: Optional[str] = None,
    regions: Optional[List[str]] = None,
    validate_schema: bool = False,
) -> pd.DataFrame:
    """Load financial data from PostgreSQL database.

    Args:
        db_url: SQLAlchemy connection URL (or use DB_URL env var)
        table_name: Table name to query
        query: Custom SQL query (overrides table_name)
        regions: List of regions to filter
        validate_schema: Whether to validate columns against COLUMN_SCHEMA

    Returns:
        DataFrame with loaded data

    Example:
        >>> df = load_from_db(regions=["US", "EU"])
        >>> df = load_from_db(regions=["US"], validate_schema=True)
    """
    import os

    try:
        from sqlalchemy import create_engine
    except ImportError:
        raise ImportError("SQLAlchemy required. Install with: pip install sqlalchemy")

    # Get database URL
    if db_url is None:
        db_url = os.environ.get("DB_URL")

    if db_url is None:
        raise ValueError("Database URL required. Provide db_url or set DB_URL env var")

    logger.info(f"Connecting to database...")
    engine = create_engine(db_url)

    # Build query
    if query is None:
        if regions:
            region_list = ", ".join(f"'{r}'" for r in regions)
            query = f'SELECT * FROM {table_name} WHERE "Region" IN ({region_list})'
        else:
            query = f"SELECT * FROM {table_name}"

    logger.info(f"Executing query: {query[:100]}...")

    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} rows from database")

    # Normalize column names
    df = _normalize_column_names(df)

    # Optional schema validation
    if validate_schema:
        from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

        unknown_cols = [col for col in df.columns if col not in COLUMN_SCHEMA]
        if unknown_cols:
            logger.warning(
                f"Found {len(unknown_cols)} columns not in COLUMN_SCHEMA: "
                f"{unknown_cols[:5]}{'...' if len(unknown_cols) > 5 else ''}"
            )

    return df


def load_from_all_stocks(
    data_dir: Union[str, Path] = "all_stocks",
    pattern: str = "*.csv",
) -> pd.DataFrame:
    """Load and combine all CSV files from the all_stocks directory.

    Args:
        data_dir: Directory containing CSV files
        pattern: Glob pattern for files to load

    Returns:
        Combined DataFrame from all matching files
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    csv_files = list(data_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files matching {pattern} in {data_dir}")

    logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")

    dfs = []
    for csv_file in csv_files:
        try:
            df = load_from_csv(csv_file, normalize_columns=True)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {csv_file}: {e}")

    if not dfs:
        raise ValueError("No CSV files could be loaded")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined)} total rows from {len(dfs)} files")

    return combined


def load_multiple_csvs(
    filepaths: List[Union[str, Path]],
    normalize_columns: bool = True,
) -> pd.DataFrame:
    """Load and combine multiple CSV files.

    Args:
        filepaths: List of paths to CSV files
        normalize_columns: Whether to normalize column names

    Returns:
        Combined DataFrame
    """
    dfs = []

    for filepath in filepaths:
        try:
            df = load_from_csv(filepath, normalize_columns=normalize_columns)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {filepath}: {e}")

    if not dfs:
        raise ValueError("No CSV files could be loaded")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined {len(combined)} total rows from {len(dfs)} files")

    return combined


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase with underscores.

    Uses canonical normalization from schema.py to ensure consistency
    with COLUMN_SCHEMA keys. This handles special cases like:
    - R&D -> randd (not r_and_d)
    - # -> num (e.g., "# Strong Sell Ratings" -> "num_strong_sell_ratings")
    - % -> pct (e.g., "Price Chg. % (1M)" -> "price_chg_pct_1m")
    - & -> and (e.g., "Selling General & Admin" -> "selling_general_and_admin")

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with normalized column names
    """
    from finance_ml.ml_workflow.data.schema import normalize_column_name

    # Use canonical normalization function for consistency with COLUMN_SCHEMA
    df.columns = [normalize_column_name(col) for col in df.columns]
    return df
