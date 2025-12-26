"""
Analyze imputation strategy coverage against COLUMN_SCHEMA.

This script identifies columns in COLUMN_SCHEMA that are not explicitly
covered by any imputation strategy function.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA
from finance_ml.ml_workflow.preprocessing.imputation import (
    get_zero_imputation_columns,
    get_median_imputation_columns,
    get_knn_imputation_columns,
    get_categorical_imputation_config,
)


def analyze_coverage():
    """Analyze imputation coverage for all schema columns."""
    
    # Get all columns from each imputation strategy
    zero_cols = set(get_zero_imputation_columns())
    median_cols = set(get_median_imputation_columns())
    knn_cols = set(get_knn_imputation_columns())
    categorical_config = get_categorical_imputation_config()
    categorical_cols = set(categorical_config.keys())
    
    # Get price columns from schema (handled by apply_price_imputation)
    price_roles = {"price", "target", "target_fallback"}
    price_cols = set()
    for col, meta in COLUMN_SCHEMA.items():
        role = meta.get("role", "")
        if role in price_roles:
            price_cols.add(col)
    
    # Date columns are auto-detected by pattern matching
    # Patterns must match apply_datetime_imputation_and_formatting()
    date_patterns = ["date", "updated", "earnings", "report", "fy_end", "dividend"]
    
    # Get all schema columns by role
    schema_by_role = {}
    for col, meta in COLUMN_SCHEMA.items():
        role = meta["role"]
        if role not in schema_by_role:
            schema_by_role[role] = []
        schema_by_role[role].append(col)
    
    print("=" * 80)
    print("IMPUTATION COVERAGE ANALYSIS")
    print("=" * 80)
    print()
    
    # Summary statistics
    print("IMPUTATION STRATEGY COVERAGE:")
    print(f"  Zero imputation:        {len(zero_cols):4d} columns")
    print(f"  Median imputation:      {len(median_cols):4d} columns")
    print(f"  KNN imputation:         {len(knn_cols):4d} columns")
    print(f"  Price imputation:       {len(price_cols):4d} columns")
    print(f"  Categorical imputation: {len(categorical_cols):4d} columns")
    print()
    
    # Analyze by role
    print("SCHEMA COLUMNS BY ROLE:")
    for role, cols in sorted(schema_by_role.items()):
        print(f"  {role:20s}: {len(cols):4d} columns")
    print()
    
    # Identify uncovered columns
    print("=" * 80)
    print("COVERAGE GAPS ANALYSIS")
    print("=" * 80)
    print()
    
    # Columns that should be explicitly covered
    # (exclude auxiliary, id, and some special roles)
    roles_requiring_coverage = {
        "feature", "target", "target_fallback", "price", "market_value",
        "ratio", "percentage", "count", "categorical", "date"
    }
    
    uncovered_by_role = {}
    
    for role in roles_requiring_coverage:
        if role not in schema_by_role:
            continue
            
        uncovered = []
        for col in schema_by_role[role]:
            # Check if column is covered by any strategy
            is_covered = (
                col in zero_cols or
                col in median_cols or
                col in knn_cols or
                col in price_cols or
                col in categorical_cols or
                any(pattern in col for pattern in date_patterns)
            )
            
            if not is_covered:
                uncovered.append(col)
        
        if uncovered:
            uncovered_by_role[role] = uncovered
    
    # Report uncovered columns
    if uncovered_by_role:
        print("UNCOVERED COLUMNS (by role):")
        print()
        for role, cols in sorted(uncovered_by_role.items()):
            print(f"{role.upper()} ({len(cols)} columns):")
            for col in sorted(cols):
                dtype = COLUMN_SCHEMA[col]["dtype"]
                print(f"  - {col:50s} (dtype: {dtype})")
            print()
    else:
        print("✓ All columns with required roles are covered by imputation strategies!")
        print()
    
    # Check for date columns specifically
    print("=" * 80)
    print("DATE COLUMNS ANALYSIS")
    print("=" * 80)
    print()
    
    date_cols = schema_by_role.get("date", [])
    print(f"Total date columns in schema: {len(date_cols)}")
    print()
    
    print("Date columns (auto-detected by pattern matching):")
    for col in sorted(date_cols):
        matches = [p for p in date_patterns if p in col]
        if matches:
            print(f"  ✓ {col:50s} (matches: {', '.join(matches)})")
        else:
            print(f"  ✗ {col:50s} (NO PATTERN MATCH - will use fallback)")
    print()
    
    # Check for dividend-related columns
    print("=" * 80)
    print("DIVIDEND-RELATED COLUMNS")
    print("=" * 80)
    print()
    
    dividend_cols = [col for col in COLUMN_SCHEMA.keys() if "dividend" in col]
    print(f"Total dividend-related columns: {len(dividend_cols)}")
    print()
    
    for col in sorted(dividend_cols):
        meta = COLUMN_SCHEMA[col]
        role = meta["role"]
        dtype = meta["dtype"]
        
        # Check coverage
        is_covered = (
            col in zero_cols or
            col in median_cols or
            col in knn_cols or
            col in price_cols or
            col in categorical_cols or
            any(pattern in col for pattern in date_patterns)
        )
        
        status = "✓" if is_covered else "✗"
        print(f"  {status} {col:50s} (role: {role:15s}, dtype: {dtype})")
    print()
    
    # Summary
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    if uncovered_by_role:
        print("1. Add uncovered columns to appropriate imputation strategies:")
        for role, cols in sorted(uncovered_by_role.items()):
            print(f"   - {role}: {len(cols)} columns need explicit coverage")
        print()
    
    # Check if dividend date columns need explicit handling
    dividend_date_cols = [col for col in date_cols if "dividend" in col]
    if dividend_date_cols:
        print("2. Dividend date columns detected:")
        for col in dividend_date_cols:
            print(f"   - {col}")
        print("   Consider adding explicit handling in apply_datetime_imputation_and_formatting()")
        print()
    
    print("3. Update date_patterns in apply_datetime_imputation_and_formatting() to include:")
    print('   date_patterns = ["date", "updated", "earnings", "report", "dividend"]')
    print()


if __name__ == "__main__":
    analyze_coverage()
