"""Analyze Out_7.csv to identify datetime conversion issues."""

import pandas as pd

# Load the CSV
df = pd.read_csv('all_stocks/Out_7.csv', nrows=20)

# Problematic columns mentioned in the issue
problematic_cols = [
    'retained_earnings_ltm',
    'dividend_per_share_ltm',
    'next_earnings_when',
    'dividend_record_frequency',
    'dividend_record_amount'
]

print("=" * 80)
print("ANALYSIS OF PROBLEMATIC COLUMNS IN Out_7.csv")
print("=" * 80)

for col in problematic_cols:
    print(f"\n{col}:")
    print("-" * 80)
    if col in df.columns:
        print(f"  Current dtype: {df[col].dtype}")
        print(f"  Sample values (first 10):")
        for i, val in enumerate(df[col].head(10)):
            print(f"    [{i}] {repr(val)}")
        print(f"  Unique values: {df[col].nunique()}")
        print(f"  Missing values: {df[col].isna().sum()}")
    else:
        print("  ❌ Column not found in CSV")

# Also check all columns with 'date' or 'earnings' or 'dividend' in name
print("\n" + "=" * 80)
print("ALL DATE/EARNINGS/DIVIDEND COLUMNS IN CSV")
print("=" * 80)

date_related = [col for col in df.columns if any(
    pattern in col.lower() for pattern in ['date', 'earnings', 'dividend']
)]

for col in date_related[:15]:  # First 15 to avoid too much output
    print(f"\n{col}: {df[col].dtype}")
    print(f"  Sample: {df[col].iloc[0] if len(df) > 0 else 'N/A'}")
