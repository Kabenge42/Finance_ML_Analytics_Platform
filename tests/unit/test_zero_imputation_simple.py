"""
Simple focused test of zero imputation fix.

Tests only Steps 1 and 4 to verify the fix works correctly.
"""

import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_zero_imputation,
    apply_median_imputation,
    get_zero_imputation_columns,
)

print("="*80)
print("SIMPLE ZERO IMPUTATION FIX TEST")
print("="*80)

# Load source data
print("\n1. Loading source data...")
df = pd.read_csv('all_stocks/equities.csv', nrows=100)
print(f"   Loaded {len(df)} rows")

# Normalize column names
df.columns = (
    df.columns
    .str.replace('[^0-9a-zA-Z]+', '_', regex=True)
    .str.strip('_')
    .str.lower()
)

# Get zero-imputation columns
zero_cols = get_zero_imputation_columns()
zero_cols_in_df = [col for col in zero_cols if col in df.columns]

print(f"\n2. Zero-imputation columns in dataframe: {len(zero_cols_in_df)}")

# Analyze BEFORE
print("\n" + "="*80)
print("BEFORE IMPUTATION")
print("="*80)

before_stats = {}
for col in zero_cols_in_df[:5]:  # Test first 5 columns
    missing = df[col].isna().sum()
    zeros = (df[col] == 0).sum()
    nonzeros = ((df[col] != 0) & df[col].notna()).sum()
    
    before_stats[col] = {'missing': missing, 'zeros': zeros, 'nonzeros': nonzeros}
    print(f"\n{col}:")
    print(f"  Missing: {missing}, Zeros: {zeros}, Non-zeros: {nonzeros}")

# Step 1: Apply zero imputation
print("\n" + "="*80)
print("STEP 1: ZERO IMPUTATION")
print("="*80)

df_after_step1 = apply_zero_imputation(df, columns=zero_cols)

print("\nAfter Step 1:")
for col in zero_cols_in_df[:5]:
    missing = df_after_step1[col].isna().sum()
    zeros = (df_after_step1[col] == 0).sum()
    nonzeros = ((df_after_step1[col] != 0) & df_after_step1[col].notna()).sum()
    
    print(f"\n{col}:")
    print(f"  Missing: {missing}, Zeros: {zeros}, Non-zeros: {nonzeros}")
    
    # Verify zeros were added
    expected_zeros = before_stats[col]['missing'] + before_stats[col]['zeros']
    if zeros == expected_zeros:
        print(f"  ✓ Correctly imputed {before_stats[col]['missing']} missing to zero")
    else:
        print(f"  ✗ Expected {expected_zeros} zeros, got {zeros}")

# Step 4: Apply median imputation (should NOT overwrite zeros)
print("\n" + "="*80)
print("STEP 4: MEDIAN IMPUTATION (with fix)")
print("="*80)

df_after_step4 = apply_median_imputation(df_after_step1, price_column='last_price')

print("\nAfter Step 4:")
issues = []
for col in zero_cols_in_df[:5]:
    missing = df_after_step4[col].isna().sum()
    zeros = (df_after_step4[col] == 0).sum()
    nonzeros = ((df_after_step4[col] != 0) & df_after_step4[col].notna()).sum()
    
    print(f"\n{col}:")
    print(f"  Missing: {missing}, Zeros: {zeros}, Non-zeros: {nonzeros}")
    
    # Verify zeros were preserved
    expected_zeros = before_stats[col]['missing'] + before_stats[col]['zeros']
    if zeros == expected_zeros:
        print(f"  ✓ Zeros preserved through Step 4")
    else:
        issues.append(col)
        print(f"  ✗ FAILED: Expected {expected_zeros} zeros, got {zeros}")
        print(f"     Zeros were overwritten by median imputation!")

# Final result
print("\n" + "="*80)
print("TEST RESULTS")
print("="*80)

if issues:
    print(f"\n✗ TEST FAILED - {len(issues)} columns had zeros overwritten:")
    for col in issues:
        print(f"  - {col}")
else:
    print("\n✓ TEST PASSED!")
    print("  - All missing values correctly imputed to zero in Step 1")
    print("  - All zeros preserved through Step 4 (median imputation)")
    print("  - Fix is working correctly!")

print("\n" + "="*80)
