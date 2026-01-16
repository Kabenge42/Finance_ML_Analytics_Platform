"""
Compare source data (equities.csv) with transformed data (Out_7.csv)
to identify false imputation in zero-imputation columns.
"""

import sys

import pandas as pd

# Configure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Load data
print("Loading data...")
source = pd.read_csv('all_stocks/equities.csv', nrows=100)
out7 = pd.read_csv('all_stocks/Out_7.csv', nrows=100)

# Zero-imputation columns (normalized names)
zero_cols = [
    'impairment_of_goodwill_fq',
    'impairment_of_goodwill_ltm',
    'asset_writedown_ltm',
    'asset_writedown_fq',
    'restructuring_charges_ltm',
    'restructuring_charges_fq',
    'merger_and_restructuring_charges_ltm',
    'merger_and_restructuring_charges_fq',
    'gain_loss_on_sale_of_assets_ltm',
    'other_unusual_items_total_ltm',
]

print("\n" + "="*80)
print("COMPARISON: Source (equities.csv) vs Transformed (Out_7.csv)")
print("="*80)

# Map normalized names to source column names
def normalize_to_source(col):
    """Convert normalized column name to source CSV format."""
    # Remove underscores and capitalize
    parts = col.split('_')
    result = []
    for part in parts:
        if part == 'ltm':
            result.append('(LTM)')
        elif part == 'fq':
            result.append('(FQ)')
        elif part == 'fy':
            result.append('(FY)')
        elif part == 'of':
            result.append('of')
        elif part == 'on':
            result.append('on')
        elif part == 'and':
            result.append('&')
        else:
            result.append(part.capitalize())
    return ' '.join(result)

issues_found = []

for col in zero_cols:
    src_col = normalize_to_source(col)
    
    print(f"\n{col}:")
    print(f"  Source column name: {src_col}")
    
    # Check Out_7.csv
    if col in out7.columns:
        out7_missing = out7[col].isna().sum()
        out7_zeros = (out7[col] == 0).sum()
        out7_nonzeros = ((out7[col] != 0) & out7[col].notna()).sum()
        
        print(f"  Out_7.csv: missing={out7_missing}, zeros={out7_zeros}, non-zeros={out7_nonzeros}")
        
        if out7_nonzeros > 0:
            sample = out7[col].dropna()[out7[col] != 0].head(3).tolist()
            print(f"    Sample non-zero values: {sample}")
    else:
        print(f"  Out_7.csv: Column not found")
        continue
    
    # Check source CSV
    if src_col in source.columns:
        src_missing = source[src_col].isna().sum()
        src_zeros = (source[src_col] == 0).sum()
        src_nonzeros = ((source[src_col] != 0) & source[src_col].notna()).sum()
        
        print(f"  Source CSV: missing={src_missing}, zeros={src_zeros}, non-zeros={src_nonzeros}")
        
        if src_nonzeros > 0:
            sample_src = source[src_col].dropna()[source[src_col] != 0].head(3).tolist()
            print(f"    Source sample: {sample_src}")
        
        # Identify issue: if source is all missing but Out_7 has non-zeros
        if src_missing == 100 and out7_nonzeros > 0:
            issues_found.append({
                'column': col,
                'source_missing': src_missing,
                'out7_nonzeros': out7_nonzeros,
                'issue': 'FALSE IMPUTATION - Source is 100% missing but Out_7 has non-zero values'
            })
            print(f"  ⚠️  ISSUE: Source is 100% missing but Out_7 has {out7_nonzeros} non-zero values!")
        elif src_missing > 0 and out7_zeros < src_missing:
            issues_found.append({
                'column': col,
                'source_missing': src_missing,
                'out7_zeros': out7_zeros,
                'issue': f'INCOMPLETE ZERO IMPUTATION - Source has {src_missing} missing but Out_7 only has {out7_zeros} zeros'
            })
            print(f"  ⚠️  ISSUE: Source has {src_missing} missing but Out_7 only has {out7_zeros} zeros!")
    else:
        print(f"  Source CSV: Column not found (tried: {src_col})")
        # Try alternative formats
        alt_formats = [
            src_col.replace('&', 'and'),
            src_col.replace(' ', '_'),
            col.upper(),
        ]
        for alt in alt_formats:
            if alt in source.columns:
                print(f"  Found alternative: {alt}")
                break

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if issues_found:
    print(f"\n⚠️  FOUND {len(issues_found)} ISSUES:\n")
    for i, issue in enumerate(issues_found, 1):
        print(f"{i}. {issue['column']}")
        print(f"   {issue['issue']}")
        print()
    
    print("\nROOT CAUSE:")
    print("Zero-imputation columns are being falsely imputed with non-zero values")
    print("during the ETL transformation process, despite being empty in source data.")
    print("\nThis indicates that KNN or median imputation is overwriting the zero")
    print("values set in Step 1 of the 6-step imputation strategy.")
else:
    print("\n✓ No issues found - all zero-imputation columns are correctly handled")

print("\n" + "="*80)
