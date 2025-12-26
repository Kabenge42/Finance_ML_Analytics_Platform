"""
Analyze column order mismatches between CSV files and SQL schema.
"""
import csv
from pathlib import Path

# Read CSV header
csv_path = Path("data/screening_us.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_columns = next(reader)

print(f"CSV has {len(csv_columns)} columns")
print("\n" + "="*80)
print("CSV COLUMN ORDER (first 30 columns)")
print("="*80)
for i, col in enumerate(csv_columns[:30], 1):
    print(f"{i:3d}. {col}")

# Read schema columns from create_equities_schema.sql
schema_path = Path("create_equities_schema.sql")
schema_columns = []
with open(schema_path, 'r', encoding='utf-8') as f:
    in_create_table = False
    for line in f:
        if 'CREATE TABLE equities' in line:
            in_create_table = True
            continue
        if in_create_table:
            if line.strip().startswith(')'):
                break
            # Extract column name from lines like: "Column Name" TYPE,
            if '"' in line:
                parts = line.strip().split('"')
                if len(parts) >= 2:
                    col_name = parts[1]
                    if col_name and not col_name.startswith('--'):
                        schema_columns.append(col_name)

print(f"\n\nSchema has {len(schema_columns)} columns")
print("\n" + "="*80)
print("SCHEMA COLUMN ORDER (first 30 columns)")
print("="*80)
for i, col in enumerate(schema_columns[:30], 1):
    print(f"{i:3d}. {col}")

# Compare positions
print("\n" + "="*80)
print("CRITICAL MISMATCHES (first 30 columns)")
print("="*80)
mismatches = []
for i in range(min(30, len(csv_columns), len(schema_columns))):
    csv_col = csv_columns[i]
    schema_col = schema_columns[i]
    if csv_col != schema_col:
        mismatches.append((i+1, csv_col, schema_col))
        print(f"Position {i+1:3d}: CSV='{csv_col}' vs SCHEMA='{schema_col}'")

print(f"\n\nTotal mismatches in first 30: {len(mismatches)}")

# Find where Region appears
print("\n" + "="*80)
print("REGION COLUMN POSITION")
print("="*80)
csv_region_pos = csv_columns.index("Region") + 1 if "Region" in csv_columns else None
schema_region_pos = schema_columns.index("Region") + 1 if "Region" in schema_columns else None
print(f"CSV:    Position {csv_region_pos}")
print(f"Schema: Position {schema_region_pos}")

# Find Dividend Record columns
print("\n" + "="*80)
print("DIVIDEND RECORD COLUMNS POSITIONS")
print("="*80)
dividend_cols = [col for col in csv_columns if "Dividend Record" in col]
print(f"Found {len(dividend_cols)} Dividend Record columns in CSV:")
for col in dividend_cols:
    csv_pos = csv_columns.index(col) + 1
    schema_pos = schema_columns.index(col) + 1 if col in schema_columns else "NOT FOUND"
    print(f"  {col:40s} - CSV pos: {csv_pos:3d}, Schema pos: {schema_pos}")

# Write full comparison to file
output_path = Path("column_comparison_report.txt")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("FULL COLUMN COMPARISON REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"CSV columns: {len(csv_columns)}\n")
    f.write(f"Schema columns: {len(schema_columns)}\n\n")
    
    f.write("POSITION-BY-POSITION COMPARISON\n")
    f.write("="*80 + "\n")
    max_len = max(len(csv_columns), len(schema_columns))
    for i in range(max_len):
        csv_col = csv_columns[i] if i < len(csv_columns) else "MISSING"
        schema_col = schema_columns[i] if i < len(schema_columns) else "MISSING"
        match = "✓" if csv_col == schema_col else "✗"
        f.write(f"{i+1:3d} {match} CSV: {csv_col:50s} | SCHEMA: {schema_col}\n")
    
    f.write("\n\nCSV COLUMNS NOT IN SCHEMA\n")
    f.write("="*80 + "\n")
    for col in csv_columns:
        if col not in schema_columns:
            f.write(f"  - {col}\n")
    
    f.write("\n\nSCHEMA COLUMNS NOT IN CSV\n")
    f.write("="*80 + "\n")
    for col in schema_columns:
        if col not in csv_columns:
            f.write(f"  - {col}\n")

print(f"\n\nFull report written to: {output_path}")
