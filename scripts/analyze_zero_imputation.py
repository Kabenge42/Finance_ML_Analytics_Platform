"""
Analyze zero imputation columns in Out_7.csv to identify the root cause
of non-zero values in columns that should be zero-imputed.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.preprocessing.imputation import get_zero_imputation_columns

def analyze_zero_imputation():
    """Analyze zero imputation columns in Out_7.csv."""
    
    # Load the data
    df = pd.read_csv('all_stocks/Out_7.csv')
    
    # Get zero imputation columns
    zero_cols = get_zero_imputation_columns()
    
    print("=" * 80)
    print("ZERO IMPUTATION ANALYSIS - Out_7.csv")
    print("=" * 80)
    print()
    
    print(f"Total rows in Out_7.csv: {len(df)}")
    print(f"Total zero-imputation columns defined: {len(zero_cols)}")
    print()
    
    # Check which columns exist in the dataframe
    existing_zero_cols = [col for col in zero_cols if col in df.columns]
    missing_zero_cols = [col for col in zero_cols if col not in df.columns]
    
    print(f"Zero-imputation columns present in Out_7.csv: {len(existing_zero_cols)}")
    print(f"Zero-imputation columns missing from Out_7.csv: {len(missing_zero_cols)}")
    print()
    
    if missing_zero_cols:
        print("Missing columns:")
        for col in missing_zero_cols[:10]:
            print(f"  - {col}")
        if len(missing_zero_cols) > 10:
            print(f"  ... and {len(missing_zero_cols) - 10} more")
        print()
    
    # Analyze existing zero-imputation columns
    print("=" * 80)
    print("ANALYSIS OF EXISTING ZERO-IMPUTATION COLUMNS")
    print("=" * 80)
    print()
    
    problematic_cols = []
    
    for col in existing_zero_cols:
        series = df[col]
        
        # Statistics
        total_count = len(series)
        missing_count = series.isna().sum()
        zero_count = (series == 0).sum()
        non_zero_count = ((series != 0) & series.notna()).sum()
        
        # Check if there are non-zero values
        if non_zero_count > 0:
            problematic_cols.append(col)
            
            print(f"{col}:")
            print(f"  dtype: {series.dtype}")
            print(f"  Total values: {total_count}")
            print(f"  Missing (NaN): {missing_count} ({100*missing_count/total_count:.1f}%)")
            print(f"  Zero values: {zero_count} ({100*zero_count/total_count:.1f}%)")
            print(f"  Non-zero values: {non_zero_count} ({100*non_zero_count/total_count:.1f}%)")
            
            # Show sample non-zero values
            non_zero_vals = series[series != 0].dropna().head(10).tolist()
            print(f"  Sample non-zero values: {non_zero_vals}")
            
            # Show statistics of non-zero values
            non_zero_series = series[series != 0].dropna()
            if len(non_zero_series) > 0:
                print(f"  Non-zero stats: min={non_zero_series.min():.2f}, "
                      f"max={non_zero_series.max():.2f}, "
                      f"mean={non_zero_series.mean():.2f}, "
                      f"median={non_zero_series.median():.2f}")
            print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if problematic_cols:
        print(f"❌ ISSUE CONFIRMED: {len(problematic_cols)} zero-imputation columns have non-zero values!")
        print()
        print("Problematic columns:")
        for col in problematic_cols:
            non_zero_count = ((df[col] != 0) & df[col].notna()).sum()
            print(f"  - {col}: {non_zero_count} non-zero values")
        print()
        
        print("ROOT CAUSE ANALYSIS:")
        print("  The zero imputation step is either:")
        print("  1. Not being applied at all")
        print("  2. Being applied but then overwritten by subsequent imputation steps (KNN, median)")
        print("  3. Being applied to the wrong column names (normalization issue)")
        print()
        
        print("RECOMMENDED FIX:")
        print("  1. Ensure apply_zero_imputation() is called FIRST in the 6-step strategy")
        print("  2. Protect zero-imputed columns from being overwritten by subsequent steps")
        print("  3. Add a mask to track which values were zero-imputed")
        print("  4. Restore zero values after all other imputation steps complete")
    else:
        print("✓ All zero-imputation columns correctly have only zero or missing values!")
    
    print()

if __name__ == "__main__":
    analyze_zero_imputation()
