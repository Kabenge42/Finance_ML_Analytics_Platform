"""
Notebook utility helpers for Finance ML Analytics Platform.

These functions encapsulate common display/reporting concerns and a clean
strategy-based data loading routine for use in notebooks.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from .config import FinanceMLConfig
from .data import (
    load_from_db,
    load_from_csv,
    create_sample_financial_dataset,
    check_missing_values,
    validate_schema,
    )

logger = logging.getLogger(__name__)


# -----------------------------
# Configuration display helpers
# -----------------------------


def display_config_summary(config: FinanceMLConfig) -> None:
    """Display configuration summary in a readable format.

    Parameters
    ----------
    config : FinanceMLConfig
        Loaded configuration instance.
    """
    print("Configuration loaded:")
    config_items = [
        ("Data directory", config.data_dir),
        ("Output directory", config.output_dir),
        ("Model directory", config.model_dir),
        ("Cache directory", config.cache_dir),
        ("Model version", config.model_version),
        ("Random seed", config.random_seed),
        ("N jobs", config.n_jobs),
        ("Log level", config.log_level),
        ("Memory limit", config.memory_limit),
        ("DB URL", "configured" if getattr(config, "db_url", None) else "not configured"),
    ]
    for label, value in config_items:
        print(f"  {label}: {value}")


# -----------------
# Data load helpers
# -----------------


def load_stock_data(config: FinanceMLConfig) -> pd.DataFrame:
    """Load stock data with a simple strategy: DB -> CSV -> Sample.

    Attempts to load from database if a DB URL is configured, then from CSV
    files (data/ directory), and finally falls back to a generated sample
    dataset for demonstration.
    """
    # Strategy 1: Database
    if getattr(config, "db_url", None):
        try:
            logger.info("Loading from database...")
            df = load_from_db(config.db_url, limit=None)
            logger.info("✓ Loaded %d stocks from database", len(df))
            return df
        except Exception as e:  # noqa: BLE001 (surface to user and continue)
            logger.warning("Database load failed: %s", e)

    # Strategy 2: CSV
    try:
        logger.info("Loading from CSV files...")
        df = load_from_csv(config.data_dir, limit=None)
        logger.info("✓ Loaded %d stocks from CSV", len(df))
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning("CSV load failed: %s", e)

    # Strategy 3: Sample data
    logger.info("Using sample dataset for demonstration...")
    df = create_sample_financial_dataset(n_stocks=100, random_seed=config.random_seed)
    logger.info("✓ Created sample dataset with %d stocks", len(df))
    return df


def display_data_summary(df: pd.DataFrame) -> None:
    """Display summary of the loaded dataset."""
    print("\n" + "=" * 70)
    print("Data Loading Summary")
    print("=" * 70)
    print(f"Loaded stocks: {len(df):,}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns (first 10): {list(df.columns)[:10]}")


# --------------------
# Validation reporting
# --------------------


def display_validation_results(is_valid: bool, errors: List[str]) -> None:
    """Display schema validation results in a consistent format."""
    if is_valid:
        logger.info("✓ Schema validation passed")
        print("✓ Schema validation passed")
    else:
        logger.warning("Schema validation issues: %s", errors)
        print(f"⚠ Schema validation issues found: {len(errors)}")
        for error in errors[:5]:
            print(f"  - {error}")


def display_missing_values_summary(missing_info: dict) -> None:
    """Display missing values summary."""
    print("\n" + "=" * 70)
    print("Missing Values Summary")
    print("=" * 70)
    print(f"Total missing: {missing_info.get('total_missing', 0):,}")
    print(f"Missing percentage: {missing_info.get('missing_percentage', 0):.2f}%")
    if missing_info.get("columns_with_missing"):
        print(f"Columns with missing data: {len(missing_info['columns_with_missing'])}")


def validate_and_display_data(df: pd.DataFrame) -> None:
    """Run schema and missing-value checks and display concise reports."""
    try:
        is_valid, errors = validate_schema(df)
        display_validation_results(is_valid, errors)
    except Exception as e:  # noqa: BLE001
        logger.error("Schema validation failed: %s", e)
        print(f"✗ Schema validation failed: {e}")

    try:
        missing_info = check_missing_values(df)
        display_missing_values_summary(missing_info)
    except Exception as e:  # noqa: BLE001
        logger.error("Missing value check failed: %s", e)
        print(f"✗ Missing value check failed: {e}")


# ---------------
# EDA convenience
# ---------------


def perform_and_display_eda(df: pd.DataFrame) -> Optional[dict]:
    """Perform EDA and print a compact human-readable summary.

    Returns the EDA results dict if available.
    """
    from .eval import simple_eda  # local import to avoid cycles

    try:
        eda_results = simple_eda(df)
        print("\n" + "=" * 70)
        print("Exploratory Data Analysis")
        print("=" * 70)
        print(f"Numeric columns: {eda_results.get('n_numeric', 0)}")
        print(f"Categorical columns: {eda_results.get('n_categorical', 0)}")

        print("\nDataset Overview:")
        print(df.info())

        if "sector" in df.columns:
            print("\nSector Distribution:")
            print(df["sector"].value_counts())
        if "region" in df.columns:
            print("\nRegion Distribution:")
            print(df["region"].value_counts())
        return eda_results
    except Exception as e:  # noqa: BLE001
        logger.error("EDA failed: %s", e)
        print(f"✗ EDA failed: {e}")
        return None
