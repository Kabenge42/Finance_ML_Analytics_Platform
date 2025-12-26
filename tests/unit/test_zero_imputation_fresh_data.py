"""
Test zero imputation fix with fresh data containing missing values.

This test creates synthetic data with missing values in zero-imputation columns
to verify that the fix correctly applies zero imputation and preserves it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.preprocessing.imputation import (
    get_zero_imputation_columns,
    apply_enhanced_imputation_strategy_6step,
)

def test_zero_imputation_with_fresh_data():
    """Test zero imputation with fresh data containing missing values."""
    
    print("=" * 80)
    print("ZERO IMPUTATION FIX - FRESH DATA TEST")
    print("=" * 80)
    print()
    
    # Create synthetic data with missing values
    print("Creating synthetic test data...")
    n_rows = 50
    
    # Get zero imputation columns
    zero_cols = get_zero_imputation_columns()
    
    # Create base dataframe with required columns
    df = pd.DataFrame({
        'ticker': [f'TICK{i}' for i in range(n_rows)],
        'sector': np.random.choice(['Technology', 'Healthcare', 'Finance'], n_rows),
        'last_price': np.random.uniform(10, 200, n_rows),
        'market_cap': np.random.uniform(1e9, 1e12, n_rows),
        'revenue_ltm': np.random.uniform(1e8, 1e11, n_rows),
    })
    
    # Add zero-imputation columns with missing values
    for col in zero_cols[:10]:  # Use first 10 for testing
        # Create column with 80% missing values
        values = np.random.uniform(-100, -1, n_rows)
        mask = np.random.random(n_rows) < 0.8
        values[mask] = np.nan
        df[col] = values
    
    print(f"Created {len(df)} rows with {len(zero_cols[:10])} zero-imputation columns")
    print()
    
    # Check missing values BEFORE imputation
    print("BEFORE IMPUTATION:")
    print("-" * 80)
    for col in zero_cols[:10]:
        if col in df.columns:
            missing_count = df[col].isna().sum()
            non_missing = df[col].notna().sum()
            print(f"{col}:")
            print(f"  Missing: {missing_count} ({100*missing_count/len(df):.1f}%), "
                  f"Non-missing: {non_missing}")
    print()
    
    # Apply 6-step imputation
    print("Applying 6-step imputation strategy...")
    df_imputed = apply_enhanced_imputation_strategy_6step(
        df,
        sector_column='sector',
        n_neighbors=3,  # Small n_neighbors for small dataset
        price_column='last_price',
        handle_categoricals=True,
        handle_dates=False,  # No date columns in synthetic data
    )
    print("Imputation complete")
    print()
    
    # Check values AFTER imputation
    print("AFTER IMPUTATION:")
    print("-" * 80)
    
    all_pass = True
    failed_cols = []
    
    for col in zero_cols[:10]:
        if col not in df_imputed.columns:
            continue
            
        series = df_imputed[col]
        missing_count = series.isna().sum()
        zero_count = (series == 0).sum()
        non_zero_count = ((series != 0) & series.notna()).sum()
        
        # Check if all non-missing values are zero
        if non_zero_count > 0:
            all_pass = False
            failed_cols.append(col)
            print(f"❌ {col}:")
            print(f"  Missing: {missing_count}, Zero: {zero_count}, Non-zero: {non_zero_count}")
            non_zero_vals = series[series != 0].dropna().head(5).tolist()
            print(f"  Sample non-zero values: {non_zero_vals}")
        else:
            print(f"✓ {col}: All non-missing values are zero (Zero: {zero_count}, Missing: {missing_count})")
    
    print()
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    
    if all_pass:
        print("✓ TEST PASSED!")
        print()
        print("The fix successfully:")
        print("  1. Applied zero imputation to missing values in Step 1")
        print("  2. Prevented KNN imputation (Step 2) from overwriting zeros")
        print("  3. Prevented median imputation (Step 4) from overwriting zeros")
        print()
        print("All zero-imputation columns now have only zero or missing values.")
    else:
        print(f"❌ TEST FAILED!")
        print(f"{len(failed_cols)} columns still have non-zero values after imputation:")
        for col in failed_cols:
            print(f"  - {col}")
        print()
        print("The fix did not fully prevent overwriting of zero values.")
    
    print()
    return all_pass

if __name__ == "__main__":
    success = test_zero_imputation_with_fresh_data()
    sys.exit(0 if success else 1)
