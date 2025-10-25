#!/usr/bin/env python3
"""
CSV Data Validation Script for Equities Import
This script validates CSV files before importing them into PostgreSQL,
using the validation functions from ml_finance_model_v8_2.py.
Usage:
    python validate_csv_import.py [--region {us|eu|apac|rotw|all}] [--fix-issues]
Examples:
    # Validate all CSV files
    python validate_csv_import.py
    # Validate only US region
    python validate_csv_import.py --region us
    # Validate and create cleaned versions
    python validate_csv_import.py --fix-issues
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Import validation functions from the main script
from ml_finance_model_v8_2 import (
    validate_schema,
    check_missing_values,
    validate_numeric_ranges,
    )


@dataclass
class ValidationResult:
    """Encapsulates validation results for a CSV file."""

    region: str
    file: str
    exists: bool = False
    row_count: int = 0
    column_count: int = 0
    schema_valid: bool = False
    missing_values: Dict = field(default_factory=dict)
    numeric_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_error(self, error: str) -> None:
        """Add an error message to the results."""
        self.errors.append(error)

    def add_warning(self, warning: str) -> None:
        """Add a warning message to the results."""
        self.warnings.append(warning)

    def is_valid(self) -> bool:
        """Check if validation passed without errors."""
        return len(self.errors) == 0

    def print_summary(self) -> None:
        """Print validation summary."""
        print(f"\n{'-' * 70}")
        print(f"Validation Summary for {self.region.upper()}:")
        print(f"  Rows: {self.row_count:,}")
        print(f"  Columns: {self.column_count}")
        print(f"  Schema Valid: {'✓ Yes' if self.schema_valid else '❌ No'}")
        print(f"  Warnings: {len(self.warnings)}")
        print(f"  Errors: {len(self.errors)}")
        if self.is_valid():
            print(f"\n✓ CSV file is ready for import!")
        else:
            print(f"\n❌ Please fix errors before importing")


def get_csv_files(region: Optional[str] = None) -> Dict[str, Path]:
    """Get CSV file paths for specified region(s)."""
    data_dir = Path("data")
    all_files = {
        "us": data_dir / "screening_us.csv",
        "eu": data_dir / "screening_eu.csv",
        "apac": data_dir / "screening_apac.csv",
        "rotw": data_dir / "screening_rotw.csv",
    }
    if region and region != "all":
        region_lower = region.lower()
        if region_lower in all_files:
            return {region_lower: all_files[region_lower]}
        else:
            print(f"Error: Unknown region '{region}'. Valid options: us, eu, apac, rotw, all")
            sys.exit(1)
    return all_files


def validate_csv_file(csv_path: Path, region_name: str) -> ValidationResult:
    """
    Validate a single CSV file and return validation results.
    Args:
        csv_path: Path to the CSV file
        region_name: Name of the region (for display)
    Returns:
        ValidationResult object with validation results
    """
    print(f"\n{'=' * 70}")
    print(f"Validating {region_name.upper()} Region: {csv_path.name}")
    print(f"{'=' * 70}")

    results = ValidationResult(region=region_name, file=str(csv_path))

    # Check if file exists
    if not csv_path.exists():
        error_msg = f"File not found: {csv_path}"
        print(f"❌ ERROR: {error_msg}")
        results.add_error(error_msg)
        return results

    results.exists = True
    print(f"✓ File exists: {csv_path}")

    try:
        # Load CSV
        print(f"\nLoading CSV file...")
        df = pd.read_csv(csv_path)
        results.row_count = len(df)
        results.column_count = len(df.columns)
        print(f"✓ Loaded {results.row_count:,} rows × {results.column_count} columns")

        # Validate schema
        print(f"\nValidating schema...")
        try:
            validate_schema(df, require_target=False)
            results.schema_valid = True
            print(f"✓ Schema validation passed")
        except ValueError as e:
            error_msg = f"Schema validation failed: {str(e)}"
            print(f"❌ {error_msg}")
            results.add_error(error_msg)

        # Check missing values
        print(f"\nChecking for missing values...")
        missing_report = check_missing_values(df)
        results.missing_values = missing_report

        # Display missing value summary
        if missing_report:
            print(f"⚠ Missing values found in {len(missing_report)} columns:")
            for col, stats in list(missing_report.items())[:10]:  # Show top 10
                pct = stats["percent"]
                count = stats["count"]
                if pct > 50:
                    print(f"  ⚠ {col}: {count:,} missing ({pct:.1f}%)")
                    results.add_warning(f"{col}: {pct:.1f}% missing")
                else:
                    print(f"    {col}: {count:,} missing ({pct:.1f}%)")
            if len(missing_report) > 10:
                print(f"  ... and {len(missing_report) - 10} more columns with missing values")
        else:
            print(f"✓ No missing values detected")

        # Check critical columns
        print(f"\nChecking critical columns...")
        critical_cols = ["Ticker", "Sector", "Last Price"]
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                empty_count = (df[col] == "").sum() if df[col].dtype == object else 0
                total_missing = null_count + empty_count
                if total_missing > 0:
                    warning = f"{col}: {total_missing} missing/empty values"
                    print(f"  ⚠ {warning}")
                    results.add_warning(warning)
                else:
                    print(f"  ✓ {col}: No missing values")
            else:
                error_msg = f"Critical column '{col}' not found"
                print(f"  ❌ {error_msg}")
                results.add_error(error_msg)

        # Validate numeric ranges (sample check)
        print(f"\nValidating numeric ranges...")
        try:
            numeric_warnings = validate_numeric_ranges(df)
            if numeric_warnings:
                print(f"⚠ Found {len(numeric_warnings)} numeric range warnings:")
                for warning in numeric_warnings[:5]:  # Show first 5
                    print(f"  ⚠ {warning}")
                    results.add_warning(warning)
                if len(numeric_warnings) > 5:
                    print(f"  ... and {len(numeric_warnings) - 5} more warnings")
            else:
                print(f"✓ Numeric ranges look reasonable")
        except Exception as e:
            warning = f"Could not validate numeric ranges: {str(e)}"
            print(f"⚠ {warning}")
            results.add_warning(warning)

        # Check for duplicate tickers
        print(f"\nChecking for duplicates...")
        if "Ticker" in df.columns:
            dup_count = df["Ticker"].duplicated().sum()
            if dup_count > 0:
                warning = f"Found {dup_count} duplicate tickers"
                print(f"  ⚠ {warning}")
                results.add_warning(warning)
            else:
                print(f"  ✓ No duplicate tickers")

        # Check data types
        print(f"\nChecking data types...")
        numeric_cols_sample = ["Last Price", "Market Cap", "P/E (NTM)"]
        for col in numeric_cols_sample:
            if col in df.columns:
                # Try to convert to numeric and check for errors
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                non_numeric = df[col].notna() & numeric_series.isna()
                non_numeric_count = non_numeric.sum()
                if non_numeric_count > 0:
                    issue = f"{col}: {non_numeric_count} non-numeric values"
                    results.numeric_issues.append(issue)
                    print(f"  ⚠ {issue}")

        if not results.numeric_issues:
            print(f"  ✓ Sample numeric columns have appropriate types")

    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        print(f"\n❌ {error_msg}")
        results.add_error(error_msg)

    # Summary
    results.print_summary()

    return results
