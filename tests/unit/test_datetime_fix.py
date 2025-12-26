"""Test that the datetime imputation fix prevents false conversions."""

import pandas as pd

from finance_ml.ml_workflow.preprocessing.imputation import apply_datetime_imputation_and_formatting

# Create test dataframe with problematic columns
test_data = {
    # Numeric columns that should NOT be converted to datetime
    'retained_earnings_ltm': [107908.0, -14264.0, 297226.0, 254873.0, 229344.0],
    'dividend_per_share_ltm': [0.0, 1.2, 1.5, 3.0, 0.0],
    'dividend_record_amount': [0.0, 0.5, 0.75, 1.0, 0.0],
    'dividend_streak': [0.0, 5.0, 10.0, 15.0, 2.0],
    
    # Categorical columns that should NOT be converted to datetime
    'next_earnings_when': ['Before Market Open', 'After Market Close', 'Before Market Open', None, 'After Market Close'],
    'dividend_record_frequency': ['Quarterly', 'Annual', 'Semi-Annual', 'Quarterly', None],
    'next_earnings_status': ['Confirmed', 'Estimated', 'Confirmed', 'Estimated', 'Confirmed'],
    'dividend_record_currency': ['USD', 'EUR', 'USD', 'GBP', 'USD'],
    
    # True date columns that SHOULD be converted to datetime
    'last_updated': ['2025-12-22', '2025-12-21', '2025-12-20', None, '2025-12-19'],
    'next_earnings': ['2026-02-25', '2026-03-15', None, '2026-01-20', '2026-04-10'],
    'dividend_record_announce_date': ['2025-11-19', None, '2025-10-15', '2025-09-20', '2025-08-10'],
}

df = pd.DataFrame(test_data)

print("=" * 80)
print("TESTING DATETIME IMPUTATION FIX")
print("=" * 80)

print("\n1. BEFORE apply_datetime_imputation_and_formatting():")
print("-" * 80)
for col in df.columns:
    print(f"{col:40s} dtype: {str(df[col].dtype):15s} sample: {df[col].iloc[0]}")

# Apply the fixed function
df_result = apply_datetime_imputation_and_formatting(df, strategy='forward_fill')

print("\n2. AFTER apply_datetime_imputation_and_formatting():")
print("-" * 80)
for col in df_result.columns:
    print(f"{col:40s} dtype: {str(df_result[col].dtype):15s} sample: {df_result[col].iloc[0]}")

# Verify expectations
print("\n3. VALIDATION RESULTS:")
print("-" * 80)

errors = []

# Check numeric columns remain numeric
numeric_cols = ['retained_earnings_ltm', 'dividend_per_share_ltm', 'dividend_record_amount', 'dividend_streak']
for col in numeric_cols:
    if not pd.api.types.is_numeric_dtype(df_result[col]):
        errors.append(f"❌ {col} should be numeric but is {df_result[col].dtype}")
    else:
        print(f"✓ {col} correctly preserved as numeric ({df_result[col].dtype})")

# Check categorical columns remain categorical/object
categorical_cols = ['next_earnings_when', 'dividend_record_frequency', 'next_earnings_status', 'dividend_record_currency']
for col in categorical_cols:
    if pd.api.types.is_datetime64_any_dtype(df_result[col]):
        errors.append(f"❌ {col} should be categorical but is {df_result[col].dtype}")
    else:
        print(f"✓ {col} correctly preserved as categorical ({df_result[col].dtype})")

# Check true date columns are datetime
date_cols = ['last_updated', 'next_earnings', 'dividend_record_announce_date']
for col in date_cols:
    if not pd.api.types.is_datetime64_any_dtype(df_result[col]):
        errors.append(f"❌ {col} should be datetime but is {df_result[col].dtype}")
    else:
        print(f"✓ {col} correctly converted to datetime ({df_result[col].dtype})")

# Summary
print("\n" + "=" * 80)
if errors:
    print("❌ TEST FAILED - Issues found:")
    for error in errors:
        print(f"  {error}")
else:
    print("✓ TEST PASSED - All columns have correct dtypes!")
print("=" * 80)
