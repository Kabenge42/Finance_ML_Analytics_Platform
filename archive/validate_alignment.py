"""
Validate that CSV, schema, and import script all have consistent column order.
"""
import csv
from pathlib import Path

print("="*80)
print("COLUMN ALIGNMENT VALIDATION")
print("="*80)

# 1. Read CSV columns
csv_path = Path("data/screening_us.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_columns = next(reader)

print(f"\n1. CSV columns: {len(csv_columns)}")

# 2. Read schema columns
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
            if '"' in line and not line.strip().startswith('--'):
                parts = line.strip().split('"')
                if len(parts) >= 2:
                    col_name = parts[1]
                    if col_name:
                        schema_columns.append(col_name)

print(f"2. Schema columns: {len(schema_columns)}")

# 3. Read import script staging table columns
import_path = Path("import_equities_data.sql")
import_columns = []
with open(import_path, 'r', encoding='utf-8') as f:
    in_staging_table = False
    for line in f:
        if 'CREATE TEMP TABLE screening_staging' in line:
            in_staging_table = True
            continue
        if in_staging_table:
            if line.strip().startswith(')'):
                break
            if '"' in line and not line.strip().startswith('--'):
                parts = line.strip().split('"')
                if len(parts) >= 2:
                    col_name = parts[1]
                    if col_name:
                        import_columns.append(col_name)

print(f"3. Import staging columns: {len(import_columns)}")

# 4. Validate all three match
print("\n" + "="*80)
print("VALIDATION RESULTS")
print("="*80)

all_match = True
mismatches = []

# Check counts
if len(csv_columns) != len(schema_columns):
    print(f"❌ Column count mismatch: CSV={len(csv_columns)}, Schema={len(schema_columns)}")
    all_match = False
else:
    print(f"✓ Column counts match: {len(csv_columns)}")

if len(csv_columns) != len(import_columns):
    print(f"❌ Column count mismatch: CSV={len(csv_columns)}, Import={len(import_columns)}")
    all_match = False
else:
    print(f"✓ Column counts match (CSV vs Import): {len(csv_columns)}")

# Check order
print("\n" + "="*80)
print("POSITION-BY-POSITION VALIDATION")
print("="*80)

max_len = max(len(csv_columns), len(schema_columns), len(import_columns))
for i in range(max_len):
    csv_col = csv_columns[i] if i < len(csv_columns) else "MISSING"
    schema_col = schema_columns[i] if i < len(schema_columns) else "MISSING"
    import_col = import_columns[i] if i < len(import_columns) else "MISSING"
    
    if csv_col == schema_col == import_col:
        if i < 10 or i >= max_len - 5:  # Show first 10 and last 5
            print(f"  {i+1:3d}. ✓ {csv_col}")
    else:
        print(f"  {i+1:3d}. ❌ CSV='{csv_col}' | Schema='{schema_col}' | Import='{import_col}'")
        mismatches.append((i+1, csv_col, schema_col, import_col))
        all_match = False

if len(csv_columns) > 15:
    print(f"  ... ({len(csv_columns) - 15} middle columns not shown)")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if all_match:
    print("✅ SUCCESS: All files have consistent column order!")
    print(f"   - CSV has {len(csv_columns)} columns")
    print(f"   - Schema has {len(schema_columns)} columns")
    print(f"   - Import has {len(import_columns)} columns")
    print(f"   - All columns match in order")
else:
    print(f"❌ FAILURE: Found {len(mismatches)} mismatches")
    print("\nFirst 5 mismatches:")
    for pos, csv_col, schema_col, import_col in mismatches[:5]:
        print(f"  Position {pos}:")
        print(f"    CSV:    {csv_col}")
        print(f"    Schema: {schema_col}")
        print(f"    Import: {import_col}")

# Key columns check
print("\n" + "="*80)
print("KEY COLUMNS VERIFICATION")
print("="*80)

key_columns = {
    "Region": 5,
    "Dividend Record (Currency)": 20,
    "Dividend Record (Amount)": 21,
    "Market Cap": 28,
    "Last Price": 30
}

for col_name, expected_pos in key_columns.items():
    csv_pos = csv_columns.index(col_name) + 1 if col_name in csv_columns else None
    schema_pos = schema_columns.index(col_name) + 1 if col_name in schema_columns else None
    import_pos = import_columns.index(col_name) + 1 if col_name in import_columns else None
    
    if csv_pos == schema_pos == import_pos == expected_pos:
        print(f"✓ {col_name:35s} at position {csv_pos} (expected {expected_pos})")
    else:
        print(f"❌ {col_name:35s} CSV={csv_pos}, Schema={schema_pos}, Import={import_pos} (expected {expected_pos})")

print("\n" + "="*80)
if all_match:
    print("✅ VALIDATION PASSED: Files are properly aligned!")
else:
    print("❌ VALIDATION FAILED: Files need correction!")
print("="*80)
