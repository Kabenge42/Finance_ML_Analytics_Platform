"""
Test script to validate the zero imputation fix.

This script verifies that zero-imputation columns maintain zero values
after all imputation steps in the 6-step strategy.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.preprocessing.imputation import (
    get_zero_imputation_columns,
    apply_enhanced_imputation_strategy_6step,
)

def test_zero_imputation_fix():
    """Test that zero imputation is preserved after all imputation steps."""
    
    print("=" * 80)
    print("ZERO IMPUTATION FIX VALIDATION TEST")
    print("=" * 80)
    print()
    
    # Load a sample of the data
    print("Loading data from all_stocks/Out_7.csv...")
    df = pd.read_csv('all_stocks/Out_7.csv', nrows=100)
    print(f"Loaded {len(df)} rows")
    print()
    
    # Get zero imputation columns
    zero_cols = get_zero_imputation_columns()
    existing_zero_cols = [col for col in zero_cols if col in df.columns]
    
    print(f"Zero-imputation columns in dataset: {len(existing_zero_cols)}")
    print()
    
    # Check values BEFORE imputation
    print("BEFORE IMPUTATION:")
    print("-" * 80)
    for col in existing_zero_cols[:5]:  # Show first 5 as sample
        series = df[col]
        non_zero_count = ((series != 0) & series.notna()).sum()
        missing_count = series.isna().sum()
        print(f"{col}:")
        print(f"  Missing: {missing_count}, Non-zero: {non_zero_count}")
    print()
    
    # Apply the 6-step imputation strategy
    print("Applying 6-step imputation strategy...")
    df_imputed = apply_enhanced_imputation_strategy_6step(
        df,
        sector_column='sector',
        n_neighbors=5,
        price_column='last_price',
        handle_categoricals=True,
        handle_dates=True,
    )
    print("Imputation complete")
    print()
    
    # Check values AFTER imputation
    print("AFTER IMPUTATION:")
    print("-" * 80)
    
    all_pass = True
    failed_cols = []
    
    for col in existing_zero_cols:
        series = df_imputed[col]
        non_zero_count = ((series != 0) & series.notna()).sum()
        missing_count = series.isna().sum()
        zero_count = (series == 0).sum()
        
        # Check if column has only zero or missing values
        if non_zero_count > 0:
            all_pass = False
            failed_cols.append(col)
            print(f"❌ {col}:")
            print(f"  Missing: {missing_count}, Zero: {zero_count}, Non-zero: {non_zero_count}")
            # Show sample non-zero values
            non_zero_vals = series[series != 0].dropna().head(5).tolist()
            print(f"  Sample non-zero values: {non_zero_vals}")
        else:
            print(f"✓ {col}: All values are zero or missing (Zero: {zero_count}, Missing: {missing_count})")
    
    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    
    if all_pass:
        print("✓ TEST PASSED!")
        print(f"All {len(existing_zero_cols)} zero-imputation columns have only zero or missing values.")
        print()
        print("The fix successfully prevents KNN and median imputation from overwriting")
        print("zero values set in Step 1 of the 6-step imputation strategy.")
    else:
        print(f"❌ TEST FAILED!")
        print(f"{len(failed_cols)} out of {len(existing_zero_cols)} zero-imputation columns have non-zero values:")
        for col in failed_cols:
            print(f"  - {col}")
        print()
        print("The fix did not fully prevent overwriting of zero values.")
        print("Additional investigation required.")
    
    print()
    return all_pass

if __name__ == "__main__":
    success = test_zero_imputation_fix()
    sys.exit(0 if success else 1)
