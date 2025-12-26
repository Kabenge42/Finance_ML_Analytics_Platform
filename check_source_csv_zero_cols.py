"""
Check source CSV files for zero-imputation columns.

This script examines the original CSV files to determine if non-zero values
in zero-imputation columns come from source data or are artifacts of ETL.
"""

import glob

import pandas as pd


def check_source_csv_files():
    """Check source CSV files for zero-imputation column values."""
    
    print("=" * 80)
    print("SOURCE CSV FILES - ZERO-IMPUTATION COLUMNS CHECK")
    print("=" * 80)
    print()
    
    # Get all screening CSV files
    csv_files = glob.glob('data/screening_*.csv')
    
    if not csv_files:
        print("No CSV files found in data/ directory")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    print()
    
    # Columns to check (original CSV column names with spaces)
    check_cols = [
        'Impairment of Goodwill (FQ)',
        'Impairment of Goodwill (LTM)',
        'Asset Writedown (FQ)',
        'Asset Writedown (LTM)',
        'Restructuring Charges (FQ)',
        'Restructuring Charges (LTM)',
        'Gain/Loss on Sale of Assets (LTM)',
        'Other Unusual Items/Total (LTM)',
    ]
    
    for csv_file in csv_files:
        print(f"File: {csv_file}")
        print("-" * 80)
        
        try:
            # Load first 50 rows
            df = pd.read_csv(csv_file, nrows=50)
            
            print(f"Loaded {len(df)} rows")
            print()
            
            found_cols = 0
            
            for col in check_cols:
                if col in df.columns:
                    found_cols += 1
                    missing_count = df[col].isna().sum()
                    zero_count = (df[col] == 0).sum()
                    nonzero_count = ((df[col] != 0) & df[col].notna()).sum()
                    
                    print(f"{col}:")
                    print(f"  missing: {missing_count} ({missing_count/len(df)*100:.1f}%)")
                    print(f"  zeros: {zero_count} ({zero_count/len(df)*100:.1f}%)")
                    print(f"  non-zeros: {nonzero_count} ({nonzero_count/len(df)*100:.1f}%)")
                    
                    if nonzero_count > 0:
                        # Show sample non-zero values
                        nonzero_vals = df[col][(df[col] != 0) & df[col].notna()].head(3).tolist()
                        print(f"  sample non-zero: {nonzero_vals}")
                    
                    print()
            
            if found_cols == 0:
                print("  None of the zero-imputation columns found in this file")
                print()
            
        except Exception as e:
            print(f"  Error reading file: {e}")
            print()
        
        print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("If source CSV files contain non-zero values in these columns,")
    print("then Out_7.csv is correctly preserving actual reported values.")
    print()
    print("Zero imputation should ONLY fill missing values with zero,")
    print("NOT replace actual reported exceptional items.")
    print()

if __name__ == "__main__":
    check_source_csv_files()
