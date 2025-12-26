"""
Comprehensive test of zero imputation fix using actual source data.

This test verifies that:
1. Zero-imputation columns with missing values are set to 0 in Step 1
2. Zeros are preserved through Steps 2-6 (not overwritten by KNN/median)
3. All 22 zero-imputation columns are correctly handled
"""

import sys

import pandas as pd

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    get_zero_imputation_columns,
)

print("="*80)
print("COMPREHENSIVE ZERO IMPUTATION FIX TEST")
print("="*80)

# Load actual source data
print("\n1. Loading source data (equities.csv)...")
df_source = pd.read_csv('all_stocks/equities.csv', nrows=100)
print(f"   Loaded {len(df_source)} rows")

# Normalize column names to match imputation functions
print("\n2. Normalizing column names...")
df_source.columns = (
    df_source.columns
    .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
    .str.strip('_')
    .str.lower()
)

# Get zero-imputation columns
zero_cols = get_zero_imputation_columns()
print(f"\n3. Zero-imputation columns defined: {len(zero_cols)}")

# Find which zero-imputation columns are in the dataframe
zero_cols_in_df = [col for col in zero_cols if col in df_source.columns]
print(f"   Zero-imputation columns in dataframe: {len(zero_cols_in_df)}")

# Analyze BEFORE imputation
print("\n" + "="*80)
print("BEFORE IMPUTATION (Source Data)")
print("="*80)

before_stats = {}
for col in zero_cols_in_df:
    missing = df_source[col].isna().sum()
    zeros = (df_source[col] == 0).sum()
    nonzeros = ((df_source[col] != 0) & df_source[col].notna()).sum()
    
    before_stats[col] = {
        'missing': missing,
        'zeros': zeros,
        'nonzeros': nonzeros,
    }
    
    print(f"\n{col}:")
    print(f"  Missing: {missing}")
    print(f"  Zeros: {zeros}")
    print(f"  Non-zeros: {nonzeros}")
    
    if nonzeros > 0:
        sample = df_source[col].dropna()[df_source[col] != 0].head(3).tolist()
        print(f"  Sample non-zero values: {sample}")

# Apply 6-step imputation
print("\n" + "="*80)
print("APPLYING 6-STEP IMPUTATION WITH FIX")
print("="*80)

df_imputed = apply_enhanced_imputation_strategy_6step(
    df_source,
    sector_column='sector',
    n_neighbors=5,
    price_column='last_price',
    handle_categoricals=True,
    handle_dates=True,
)

# Analyze AFTER imputation
print("\n" + "="*80)
print("AFTER IMPUTATION (With Fix)")
print("="*80)

after_stats = {}
issues = []

for col in zero_cols_in_df:
    missing = df_imputed[col].isna().sum()
    zeros = (df_imputed[col] == 0).sum()
    nonzeros = ((df_imputed[col] != 0) & df_imputed[col].notna()).sum()
    
    after_stats[col] = {
        'missing': missing,
        'zeros': zeros,
        'nonzeros': nonzeros,
    }
    
    print(f"\n{col}:")
    print(f"  Missing: {missing}")
    print(f"  Zeros: {zeros}")
    print(f"  Non-zeros: {nonzeros}")
    
    # Check if zeros were properly imputed
    before_missing = before_stats[col]['missing']
    before_nonzeros = before_stats[col]['nonzeros']
    expected_zeros = before_missing + before_stats[col]['zeros']
    
    if before_missing > 0:
        if zeros < expected_zeros:
            issues.append({
                'column': col,
                'before_missing': before_missing,
                'expected_zeros': expected_zeros,
                'actual_zeros': zeros,
                'issue': f'Expected {expected_zeros} zeros but got {zeros}'
            })
            print(f"  ⚠️  ISSUE: Expected {expected_zeros} zeros but got {zeros}")
        else:
            print(f"  ✓ Correctly imputed {before_missing} missing values to zero")
    
    # Check if non-zeros were preserved
    if nonzeros != before_nonzeros:
        print(f"  ⚠️  WARNING: Non-zero count changed from {before_nonzeros} to {nonzeros}")

# Final validation
print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

if issues:
    print(f"\n✗ TEST FAILED - Found {len(issues)} issues:\n")
    for i, issue in enumerate(issues, 1):
        print(f"{i}. {issue['column']}")
        print(f"   Before: {issue['before_missing']} missing values")
        print(f"   Expected: {issue['expected_zeros']} zeros after imputation")
        print(f"   Actual: {issue['actual_zeros']} zeros")
        print(f"   Issue: {issue['issue']}")
        print()
else:
    print("\n✓ TEST PASSED - All zero-imputation columns correctly handled!")
    print("\nSummary:")
    total_missing_before = sum(s['missing'] for s in before_stats.values())
    total_zeros_after = sum(s['zeros'] for s in after_stats.values())
    print(f"  - Total missing values before: {total_missing_before}")
    print(f"  - Total zeros after imputation: {total_zeros_after}")
    print(f"  - All missing values correctly imputed to zero")
    print(f"  - Zeros preserved through all 6 steps")

# Check overall imputation completeness
total_missing_after = df_imputed.isna().sum().sum()
print(f"\n  - Total missing values remaining: {total_missing_after}")

if total_missing_after == 0:
    print("  ✓ Complete imputation achieved (zero missing values)")
else:
    print(f"  ⚠️  {total_missing_after} missing values remain in other columns")

print("\n" + "="*80)
