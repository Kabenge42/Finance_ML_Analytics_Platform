"""
Command-line interface for Finance ML Analytics Platform.

Provides console_scripts entry points for:
- finance-ml: Main analysis pipeline
- finance-ml-analyze: Quick data analysis
- finance-ml-validate: Data validation
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from finance_ml.config import load_config, FinanceMLConfig
from finance_ml.data import (
    load_from_csv,
    load_from_db,
    preprocess,
    setup_logging,
    validate_schema,
    check_missing_values,
)
from finance_ml.eval import simple_eda
from finance_ml.models import train_and_evaluate_regression, train_and_evaluate_regression_by_sector

logger = logging.getLogger(__name__)


def main() -> int:
    """Main CLI entry point for Finance ML Analytics Platform."""
    parser = argparse.ArgumentParser(
        description="Finance ML Analytics Platform — Equity screening and ML analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with auto data source detection
  finance-ml --data-source auto --limit 5000

  # Run with database
  finance-ml --data-source db --db-url postgresql://postgres:@localhost/postgres

  # Run with CSV files
  finance-ml --data-source csv --data-dir ./data

  # Dry run (skip training)
  finance-ml --data-source csv --dry-run

  # Use config file
  finance-ml --config config.json
        """
    )

    # Configuration options
    parser.add_argument("--config", type=str, help="Path to JSON/YAML config file")
    parser.add_argument("--data-source", choices=["auto", "csv", "db"], default="auto",
                        help="Data source: auto (try db then csv), csv, or db")
    parser.add_argument("--db-url", help="Database URL (e.g., postgresql://user:pass@host/db)")
    parser.add_argument("--data-dir", type=str, help="Directory containing CSV files")
    parser.add_argument("--output-dir", type=str, help="Output directory for artifacts")

    # Data options
    parser.add_argument("--limit", type=int, help="Limit number of rows to process")

    # Execution options
    parser.add_argument("--dry-run", action="store_true",
                        help="Run pipeline without training models")
    parser.add_argument("--skip-eda", action="store_true", help="Skip EDA step")
    parser.add_argument("--skip-sector-models", action="store_true",
                        help="Skip per-sector model training")

    # Performance options
    parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs (-1 for all cores)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Logging level")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(level=log_level)

    # Load configuration
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = load_config()
        logger.info("Using environment-based configuration")

    # Override config with CLI arguments
    if args.db_url:
        config.db_url = args.db_url
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.n_jobs is not None:
        config.n_jobs = args.n_jobs
    if args.seed is not None:
        config.random_seed = args.seed

    # Apply config to environment
    config.apply_to_env()

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Configuration: source={args.data_source}, limit={args.limit}, "
                f"out_dir={config.output_dir}, n_jobs={config.n_jobs}, seed={config.random_seed}")

    try:
        # Load data
        df_raw = _load_data(args.data_source, config, args.limit)
        logger.info(f"Loaded {len(df_raw)} rows")

        # Preprocess
        df = preprocess(df_raw)
        logger.info(f"After preprocessing: {len(df)} rows")

        # EDA
        if not args.skip_eda:
            logger.info("Running EDA...")
            simple_eda(df, config.output_dir)
            logger.info(f"EDA complete. Results saved to {config.output_dir}")

        # Train regression model
        if not args.dry_run:
            logger.info("Training baseline regression model...")
            metrics = train_and_evaluate_regression(
                df, config.output_dir, n_jobs=config.n_jobs, dry_run=False
            )
            logger.info(f"Baseline model metrics: {metrics}")

            # Per-sector models
            if not args.skip_sector_models:
                logger.info("Training per-sector models...")
                sector_metrics = train_and_evaluate_regression_by_sector(df, config.output_dir)
                logger.info(f"Sector models trained: {len(sector_metrics)} sectors")

        logger.info("Pipeline complete!")
        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


def analyze_main() -> int:
    """Quick analysis CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Finance ML Quick Analysis — Fast EDA and data profiling"
    )

    parser.add_argument("--data-source", choices=["csv", "db"], default="csv",
                        help="Data source: csv or db")
    parser.add_argument("--db-url", help="Database URL")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing CSV files")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit rows")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    try:
        # Load data
        if args.data_source == "db":
            if not args.db_url:
                logger.error("--db-url required for database source")
                return 1
            df = load_from_db(args.db_url, limit=args.limit)
        else:
            df = load_from_csv(Path(args.data_dir), limit=args.limit)

        logger.info(f"Loaded {len(df)} rows")

        # Preprocess
        df = preprocess(df)
        logger.info(f"After preprocessing: {len(df)} rows")

        # Run EDA
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        simple_eda(df, output_dir)

        logger.info(f"Analysis complete! Results saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


def validate_main() -> int:
    """Data validation CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Finance ML Data Validator — Validate data quality"
    )

    parser.add_argument("--data-source", choices=["csv", "db"], default="csv",
                        help="Data source: csv or db")
    parser.add_argument("--db-url", help="Database URL")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing CSV files")
    parser.add_argument("--limit", type=int, help="Limit rows")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    try:
        # Load data
        if args.data_source == "db":
            if not args.db_url:
                logger.error("--db-url required for database source")
                return 1
            df = load_from_db(args.db_url, limit=args.limit)
        else:
            df = load_from_csv(Path(args.data_dir), limit=args.limit)

        logger.info(f"Loaded {len(df)} rows")

        # Validate schema
        required_cols = ['ticker', 'sector', 'last_price']
        is_valid = validate_schema(df, required_cols)

        if is_valid:
            logger.info("✓ Schema validation passed")
        else:
            logger.error("✗ Schema validation failed")
            return 1

        # Check missing values
        missing_report = check_missing_values(df)
        if missing_report:
            logger.warning(f"Missing values detected in {len(missing_report)} columns")
            for col, info in missing_report.items():
                logger.warning(f"  {col}: {info['count']} missing ({info['percent']:.1f}%)")
        else:
            logger.info("✓ No missing values detected")

        logger.info("Validation complete!")
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        return 1


def _load_data(source: str, config: FinanceMLConfig, limit: Optional[int] = None):
    """Load data from specified source."""
    # Auto-detect source
    if source == "auto":
        try:
            from sqlalchemy import create_engine
            has_sqlalchemy = True
        except ImportError:
            has_sqlalchemy = False

        if config.db_url and has_sqlalchemy:
            source = "db"
        else:
            source = "csv"
        logger.info(f"Auto-detected data source: {source}")

    # Load from selected source
    if source == "db":
        if not config.db_url:
            raise ValueError("DB_URL not set. Provide --db-url or set DB_URL environment variable")
        return load_from_db(config.db_url, limit=limit)
    else:
        return load_from_csv(config.data_dir, limit=limit)


if __name__ == "__main__":
    sys.exit(main())
