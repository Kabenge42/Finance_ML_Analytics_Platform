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

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Import validation functions from the main script
from ml_finance_model_v8_2 import (
    validate_schema,
    check_missing_values,
    detect_outliers_iqr,
    validate_numeric_ranges,
    normalize_columns
)


def get_csv_files(region: Optional[str] = None) -> Dict[str, Path]:
    """Get CSV file paths for specified region(s)."""
    data_dir = Path("data")
    
    all_files = {
        "us": data_dir / "screening_us.csv",
        "eu": data_dir / "screening_eu.csv",
        "apac": data_dir / "screening_apac.csv",
        "rotw": data_dir / "screening_rotw.csv"
    }
    
    if region and region != "all":
        region_lower = region.lower()
        if region_lower in all_files:
            return {region_lower: all_files[region_lower]}
        else:
            print(f"Error: Unknown region '{region}'. Valid options: us, eu, apac, rotw, all")
            sys.exit(1)
    
    return all_files


def validate_csv_file(csv_path: Path, region_name: str) -> Dict:
    """
    Validate a single CSV file and return validation results.
    
    Args:
        csv_path: Path to the CSV file
        region_name: Name of the region (for display)
    
    Returns:
        Dictionary with validation results
    """
    print(f"\n{'='*70}")
    print(f"Validating {region_name.upper()} Region: {csv_path.name}")
    print(f"{'='*70}")
    
    results = {
        "region": region_name,
        "file": str(csv_path),
        "exists": False,
        "row_count": 0,
        "column_count": 0,
        "schema_valid": False,
        "missing_values": {},
        "numeric_issues": [],
        "warnings": [],
        "errors": []
    }
    
    # Check if file exists
    if not csv_path.exists():
        error_msg = f"File not found: {csv_path}"
        print(f"❌ ERROR: {error_msg}")
        results["errors"].append(error_msg)
        return results
    
    results["exists"] = True
    print(f"✓ File exists: {csv_path}")
    
    try:
        # Load CSV
        print(f"\nLoading CSV file...")
        df = pd.read_csv(csv_path)
        results["row_count"] = len(df)
        results["column_count"] = len(df.columns)
        print(f"✓ Loaded {results['row_count']:,} rows × {results['column_count']} columns")
        
        # Validate schema
        print(f"\nValidating schema...")
        try:
            validate_schema(df, require_target=False)
            results["schema_valid"] = True
            print(f"✓ Schema validation passed")
        except ValueError as e:
            error_msg = f"Schema validation failed: {str(e)}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            results["schema_valid"] = False
        
        # Check missing values
        print(f"\nChecking for missing values...")
        missing_report = check_missing_values(df)
        results["missing_values"] = missing_report
        
        # Display missing value summary
        if missing_report:
            print(f"⚠ Missing values found in {len(missing_report)} columns:")
            for col, stats in list(missing_report.items())[:10]:  # Show top 10
                pct = stats['percent']
                count = stats['count']
                if pct > 50:
                    print(f"  ⚠ {col}: {count:,} missing ({pct:.1f}%)")
                    results["warnings"].append(f"{col}: {pct:.1f}% missing")
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
                empty_count = (df[col] == '').sum() if df[col].dtype == object else 0
                total_missing = null_count + empty_count
                
                if total_missing > 0:
                    warning = f"{col}: {total_missing} missing/empty values"
                    print(f"  ⚠ {warning}")
                    results["warnings"].append(warning)
                else:
                    print(f"  ✓ {col}: No missing values")
            else:
                error_msg = f"Critical column '{col}' not found"
                print(f"  ❌ {error_msg}")
                results["errors"].append(error_msg)
        
        # Validate numeric ranges (sample check)
        print(f"\nValidating numeric ranges...")
        try:
            numeric_warnings = validate_numeric_ranges(df)
            if numeric_warnings:
                print(f"⚠ Found {len(numeric_warnings)} numeric range warnings:")
                for warning in numeric_warnings[:5]:  # Show first 5
                    print(f"  ⚠ {warning}")
                    results["warnings"].append(warning)
                if len(numeric_warnings) > 5:
                    print(f"  ... and {len(numeric_warnings) - 5} more warnings")
            else:
                print(f"✓ Numeric ranges look reasonable")
        except Exception as e:
            warning = f"Could not validate numeric ranges: {str(e)}"
            print(f"⚠ {warning}")
            results["warnings"].append(warning)
        
        # Check for duplicate tickers
        print(f"\nChecking for duplicates...")
        if "Ticker" in df.columns:
            dup_count = df["Ticker"].duplicated().sum()
            if dup_count > 0:
                warning = f"Found {dup_count} duplicate tickers"
                print(f"  ⚠ {warning}")
                results["warnings"].append(warning)
            else:
                print(f"  ✓ No duplicate tickers")
        
        # Check data types
        print(f"\nChecking data types...")
        numeric_cols_sample = ["Last Price", "Market Cap", "P/E (NTM)"]
        type_issues = []
        for col in numeric_cols_sample:
            if col in df.columns:
                # Try to convert to numeric and check for errors
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                non_numeric = df[col].notna() & numeric_series.isna()
                non_numeric_count = non_numeric.sum()
                
                if non_numeric_count > 0:
                    issue = f"{col}: {non_numeric_count} non-numeric values"
                    type_issues.append(issue)
                    print(f"  ⚠ {issue}")
        
        if type_issues:
            results["numeric_issues"] = type_issues
        else:
            print(f"  ✓ Sample numeric columns have appropriate types")
    
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        print(f"\n❌ {error_msg}")
        results["errors"].append(error_msg)
    
    # Summary
    print(f"\n{'-'*70}")
    print(f"Validation Summary for {region_name.upper()}:")
    print(f"  Rows: {results['row_count']:,}")
    print(f"  Columns: {results['column_count']}")
    print(f"  Schema Valid: {'✓ Yes' if results['schema_valid'] else '❌ No'}")
    print(f"  Warnings: {len(results['warnings'])}")
    print(f"  Errors: {len(results['errors'])}")
    
    if not results["errors"]:
        print(f"\n✓ CSV file is ready for import!")
    else:
        print(f"\n❌ Please fix errors before importing")
    
    return results


def create_cleaned_csv(csv_path: Path, output_path: Path) -> bool:
    """
    Create a cleaned version of the CSV with proper NULL handling.
    
    Args:
        csv_path: Path to original CSV
        output_path: Path for cleaned CSV
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"\nCreating cleaned version: {output_path.name}")
        df = pd.read_csv(csv_path)
        
        # Export with empty strings for NULL
        df.to_csv(output_path, index=False, na_rep='')
        print(f"✓ Cleaned CSV saved to: {output_path}")
        return True
    
    except Exception as e:
        print(f"❌ Error creating cleaned CSV: {str(e)}")
        return False


def main():
    """Main validation routine."""
    parser = argparse.ArgumentParser(
        description="Validate CSV files before importing to PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_csv_import.py                    # Validate all regions
  python validate_csv_import.py --region us        # Validate only US
  python validate_csv_import.py --fix-issues       # Create cleaned versions
        """
    )
    parser.add_argument(
        "--region",
        choices=["us", "eu", "apac", "rotw", "all"],
        default="all",
        help="Region to validate (default: all)"
    )
    parser.add_argument(
        "--fix-issues",
        action="store_true",
        help="Create cleaned CSV files with proper NULL handling"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("CSV Data Validation for Equities Import")
    print("="*70)
    
    # Get CSV files to validate
    csv_files = get_csv_files(args.region)
    
    # Validate each file
    all_results = []
    for region_name, csv_path in csv_files.items():
        results = validate_csv_file(csv_path, region_name)
        all_results.append(results)
        
        # Create cleaned version if requested
        if args.fix_issues and results["exists"]:
            cleaned_path = csv_path.parent / f"{csv_path.stem}_cleaned.csv"
            create_cleaned_csv(csv_path, cleaned_path)
    
    # Overall summary
    print(f"\n{'='*70}")
    print(f"Overall Validation Summary")
    print(f"{'='*70}")
    
    total_errors = sum(len(r["errors"]) for r in all_results)
    total_warnings = sum(len(r["warnings"]) for r in all_results)
    total_rows = sum(r["row_count"] for r in all_results)
    
    print(f"\nFiles validated: {len(all_results)}")
    print(f"Total rows across all files: {total_rows:,}")
    print(f"Total warnings: {total_warnings}")
    print(f"Total errors: {total_errors}")
    
    if total_errors == 0:
        print(f"\n✓ All CSV files are ready for import!")
        print(f"\nNext step:")
        print(f"  psql -h localhost -p 5432 -U postgres -d postgres -f import_equities_data.sql")
    else:
        print(f"\n❌ Some files have errors. Please review and fix before importing.")
        sys.exit(1)


if __name__ == "__main__":
    main()
