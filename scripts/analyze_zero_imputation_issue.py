"""
Analyze zero imputation issue in Out_7.csv.

This script examines zero-imputation columns to identify why they contain
non-zero values instead of zeros for missing data.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.preprocessing.imputation import get_zero_imputation_columns

def analyze_zero_imputation_issue():
    """Analyze zero imputation columns in Out_7.csv."""
    
    print("=" * 80)
    print("ZERO IMPUTATION ISSUE ANALYSIS - Out_7.csv")
    print("=" * 80)
    print()
    
    # Load data
    df = pd.read_csv('all_stocks/Out_7.csv', nrows=100)
    print(f"Loaded {len(df)} rows from Out_7.csv")
    print()
    
    # Get zero-imputation columns
    zero_cols = get_zero_imputation_columns()
    print(f"Zero-imputation columns defined: {len(zero_cols)}")
    print()
    
    # Check which columns exist in the data
    existing_zero_cols = [col for col in zero_cols if col in df.columns]
    missing_zero_cols = [col for col in zero_cols if col not in df.columns]
    
    print(f"Zero-imputation columns in data: {len(existing_zero_cols)}")
    print(f"Zero-imputation columns missing: {len(missing_zero_cols)}")
    print()
    
    if missing_zero_cols:
        print("Missing columns:")
        for col in missing_zero_cols[:5]:
            print(f"  - {col}")
        if len(missing_zero_cols) > 5:
            print(f"  ... and {len(missing_zero_cols) - 5} more")
        print()
    
    # Analyze existing zero-imputation columns
    print("=" * 80)
    print("ANALYSIS OF EXISTING ZERO-IMPUTATION COLUMNS")
    print("=" * 80)
    print()
    
    issue_found = False
    
    for col in existing_zero_cols:
        missing_count = df[col].isna().sum()
        zero_count = (df[col] == 0).sum()
        nonzero_count = ((df[col] != 0) & df[col].notna()).sum()
        
        # Check if there are non-zero values (potential issue)
        if nonzero_count > 0:
            issue_found = True
            print(f"{col}:")
            print(f"  dtype: {df[col].dtype}")
            print(f"  missing: {missing_count} ({missing_count/len(df)*100:.1f}%)")
            print(f"  zeros: {zero_count} ({zero_count/len(df)*100:.1f}%)")
            print(f"  non-zeros: {nonzero_count} ({nonzero_count/len(df)*100:.1f}%)")
            
            # Show sample non-zero values
            nonzero_vals = df[col][(df[col] != 0) & df[col].notna()].head(5).tolist()
            print(f"  sample non-zero values: {nonzero_vals}")
            print()
    
    if not issue_found:
        print("✓ All zero-imputation columns contain only zeros and missing values")
        print()
    else:
        print("=" * 80)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 80)
        print()
        print("ISSUE: Zero-imputation columns contain non-zero values")
        print()
        print("POSSIBLE CAUSES:")
        print("1. Source data contains actual reported values (not missing)")
        print("   → This is CORRECT behavior - preserve actual reported values")
        print()
        print("2. Zero imputation is being overwritten by subsequent steps")
        print("   → Check if KNN/median imputation includes these columns")
        print()
        print("3. Zero imputation is not being applied at all")
        print("   → Check if apply_zero_imputation() is called in ETL pipeline")
        print()
        
        # Check if values look like they came from imputation
        print("DIAGNOSTIC: Checking if values look imputed...")
        print()
        
        for col in existing_zero_cols[:3]:
            if ((df[col] != 0) & df[col].notna()).sum() > 0:
                nonzero_vals = df[col][(df[col] != 0) & df[col].notna()]
                print(f"{col}:")
                print(f"  mean: {nonzero_vals.mean():.2f}")
                print(f"  median: {nonzero_vals.median():.2f}")
                print(f"  std: {nonzero_vals.std():.2f}")
                print(f"  min: {nonzero_vals.min():.2f}")
                print(f"  max: {nonzero_vals.max():.2f}")
                print()
        
        print("If values are negative and have reasonable distributions,")
        print("they are likely ACTUAL REPORTED VALUES from source data.")
        print("This is CORRECT - zero imputation should only fill MISSING values.")
        print()

if __name__ == "__main__":
    analyze_zero_imputation_issue()
