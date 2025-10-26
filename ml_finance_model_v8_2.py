"""
Finance ML Analytics Platform — Python script version of ml_finance_model_main.ipynb

This script provides a notebook-to-script translation with a focus on:
- Loading and preprocessing data (from PostgreSQL if available, else from CSVs in data/)
- Light EDA summaries and column normalization
- Feature engineering placeholders aligned to guidelines
- Simple baseline models (classification optional, regression baseline)
- Metrics and optional exports

Design goals:
- Safe defaults: run without a database by falling back to CSVs
- Minimal hard dependencies beyond requirements.txt
- Clearly marked TODOs where domain-specific logic is required

Usage examples:
- Windows (PowerShell):
  - python ml_finance_model_v8_2.py --data-source auto
- macOS/Linux (bash):
  - python ml_finance_model_v8_2.py --data-source auto

Options:
- --data-source {auto,csv,db}
- --db-url postgresql+psycopg2://postgres:@localhost:5432/postgres (optional; env DB_URL also supported)
- --limit 5000  (limit number of rows for a quick run)
- --out-dir outputs  (directory to write artifacts)
- --dry-run  (run steps without training heavy models)

Environment variables used (optional):
- DATA_DIR, MODEL_DIR, CACHE_DIR, MODEL_VERSION, RANDOM_SEED, N_JOBS
- TF_CPP_MIN_LOG_LEVEL (see environment_variables.txt)

Note: Column names are normalized to pythonic snake_case for processing.

Phase 7 Refactoring (TDD):
This file has been refactored to import all functions from the finance_ml package
instead of duplicating code. All functionality is now in the modular finance_ml package.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Fix for _ARRAY_API AttributeError - ensure NumPy is imported first
import numpy as np

# Verify NumPy array API is available (workaround for version compatibility)
if not hasattr(np, "_ARRAY_API"):
    np._ARRAY_API = True  # Set a default value if missing

# Import all functions from the finance_ml package (Phase 7 TDD refactoring complete)
from finance_ml import (
    # Utilities and data loading
    setup_logging,
    get_env,
    load_from_csv,
    load_from_db,
    preprocess,
    # Feature engineering
    # Modeling
    train_and_evaluate_regression,
    train_and_evaluate_regression_by_sector,
    # Evaluation and analytics
    simple_eda,
    )

# Optional imports for DB access (kept optional per guidelines)
try:
    from sqlalchemy import create_engine  # type: ignore
except Exception:  # pragma: no cover - optional
    create_engine = None  # type: ignore


def main() -> int:
    """Main entry point for the Finance ML Analytics Platform script."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Finance ML Analytics Platform — main script")
    parser.add_argument("--data-source", choices=["auto", "csv", "db"], default="auto")
    parser.add_argument(
        "--db-url", default=None, help="SQLAlchemy URL; if not provided, use DB_URL env var"
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(get_env("DATA_DIR", default="data"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_seed = int(get_env("RANDOM_SEED", "42"))
    n_jobs_env = get_env("N_JOBS", "-1")
    try:
        n_jobs = int(n_jobs_env) if n_jobs_env is not None else -1
    except ValueError:
        n_jobs = -1

    # Decide data source
    db_url = args.db_url or get_env("DB_URL")
    source = args.data_source
    if source == "auto":
        if db_url and create_engine is not None:
            source = "db"
        else:
            if db_url and create_engine is None:
                logging.info(
                    "DB_URL is set but SQLAlchemy is not installed; falling back to CSV. "
                    "Hint: pip install SQLAlchemy psycopg2-binary to enable database access."
                )
            source = "csv"

    logging.info(
        "Configuration: source=%s, limit=%s, out_dir=%s, n_jobs=%d, seed=%d",
        source,
        args.limit,
        out_dir,
        n_jobs,
        random_seed,
    )

    # Load data
    if source == "db":
        if not db_url:
            logging.error(
                "--data-source db requested but DB URL is missing. Provide --db-url or DB_URL env variable."
            )
            return 2
        df_raw = load_from_db(db_url, limit=args.limit)
    else:
        df_raw = load_from_csv(data_dir, limit=args.limit)

    # Preprocess and EDA
    df = preprocess(df_raw)
    simple_eda(df, out_dir)

    # Train basic regression model if target exists
    _ = train_and_evaluate_regression(df, out_dir, n_jobs=n_jobs, dry_run=args.dry_run)

    # Also compute baseline per-sector metrics if target exists
    try:
        _metrics = train_and_evaluate_regression_by_sector(df, out_dir)
    except Exception as e:
        logging.warning("Per-sector metrics step skipped: %s", e)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def check_missing_values():
    return None


def validate_schema():
    return None


def detect_outliers_iqr():
    return None


def validate_numeric_ranges():
    return None


def normalize_columns():
    return None
