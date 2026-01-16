"""
Generate import_equities_data.sql with staging table columns in exact CSV order.
"""
import csv
from pathlib import Path

# Read CSV header to get correct order
csv_path = Path("data/screening_us.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_columns = next(reader)

print(f"CSV has {len(csv_columns)} columns")

# Read the original import script to preserve header comments
import_path = Path("import_equities_data.sql")
with open(import_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

# Extract header comments (everything before CREATE TEMP TABLE)
header_end = original_content.find('CREATE TEMP TABLE')
header_comments = original_content[:header_end].strip()

# Generate new import script
output_lines = []
output_lines.append(header_comments)
output_lines.append("")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- STAGING TABLE CREATION")
output_lines.append("-- ===================================================================")
output_lines.append("-- Create staging table with TEXT columns to avoid type conversion errors")
output_lines.append("-- Column order matches CSV file exactly")
output_lines.append("")
output_lines.append("DROP TABLE IF EXISTS screening_staging;")
output_lines.append("")
output_lines.append("CREATE TEMP TABLE screening_staging")
output_lines.append("(")

# Add all columns as TEXT in CSV order
for i, col_name in enumerate(csv_columns):
    comma = "," if i < len(csv_columns) - 1 else ""
    padding = " " * max(1, 50 - len(col_name))
    output_lines.append(f'    "{col_name}"{padding}TEXT{comma}')

output_lines.append(");")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- DATA IMPORT FROM CSV FILES")
output_lines.append("-- ===================================================================")
output_lines.append("-- Import data from regional CSV files into staging table")
output_lines.append("-- Uses \\copy for client-side import (works with relative paths)")
output_lines.append("-- NULL '' treats empty strings as NULL values")
output_lines.append("-- ENCODING 'UTF8' ensures proper character handling")
output_lines.append("")
output_lines.append("-- US Region")
output_lines.append("\\echo 'Importing US data...'")
output_lines.append("\\copy screening_staging FROM 'data/screening_us.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');")
output_lines.append("")
output_lines.append("-- EU Region")
output_lines.append("\\echo 'Importing EU data...'")
output_lines.append("\\copy screening_staging FROM 'data/screening_eu.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');")
output_lines.append("")
output_lines.append("-- APAC Region")
output_lines.append("\\echo 'Importing APAC data...'")
output_lines.append("\\copy screening_staging FROM 'data/screening_apac.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');")
output_lines.append("")
output_lines.append("-- ROTW Region")
output_lines.append("\\echo 'Importing ROTW data...'")
output_lines.append("\\copy screening_staging FROM 'data/screening_rotw.csv' WITH (FORMAT csv, HEADER true, NULL '', ENCODING 'UTF8');")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- DATA VALIDATION")
output_lines.append("-- ===================================================================")
output_lines.append("\\echo 'Validating imported data...'")
output_lines.append("SELECT 'Total rows in staging:' AS info, COUNT(*) AS count FROM screening_staging;")
output_lines.append("SELECT 'Rows by Region:' AS info, \"Region\", COUNT(*) AS count FROM screening_staging GROUP BY \"Region\" ORDER BY \"Region\";")
output_lines.append("SELECT 'Rows with missing Ticker:' AS info, COUNT(*) AS count FROM screening_staging WHERE \"Ticker\" IS NULL OR \"Ticker\" = '';")
output_lines.append("SELECT 'Rows with missing Sector:' AS info, COUNT(*) AS count FROM screening_staging WHERE \"Sector\" IS NULL OR \"Sector\" = '';")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- INSERT INTO MAIN TABLE")
output_lines.append("-- ===================================================================")
output_lines.append("-- Insert data from staging into main equities table")
output_lines.append("-- Cast TEXT columns to appropriate types (NUMERIC, DATE)")
output_lines.append("-- NULLIF handles empty strings")
output_lines.append("")
output_lines.append("\\echo 'Inserting data into equities table...'")
output_lines.append("")
output_lines.append("INSERT INTO equities (")

# Add column names for INSERT
for i, col_name in enumerate(csv_columns):
    comma = "," if i < len(csv_columns) - 1 else ""
    output_lines.append(f'    "{col_name}"{comma}')

output_lines.append(")")
output_lines.append("SELECT")

# Add SELECT with type casting
for i, col_name in enumerate(csv_columns):
    comma = "," if i < len(csv_columns) - 1 else ""
    
    # Determine type casting based on column name
    if any(keyword in col_name.lower() for keyword in ['date', 'updated', 'end', 'earnings', 'announce', 'payable', 'record', 'ex date']):
        # DATE columns
        cast_expr = f'NULLIF(TRIM("{col_name}"), \'\')::DATE'
    elif any(keyword in col_name.lower() for keyword in ['ticker', 'isin', 'name', 'description', 'exchange', 'unit', 'sector', 'industry', 'class', 'status', 'region', 'country', 'when', 'currency', 'frequency']):
        # TEXT columns
        cast_expr = f'NULLIF(TRIM("{col_name}"), \'\')'
    else:
        # NUMERIC columns
        cast_expr = f'NULLIF(TRIM("{col_name}"), \'\')::NUMERIC'
    
    padding = " " * max(1, 50 - len(col_name))
    output_lines.append(f'    {cast_expr}{padding}-- "{col_name}"{comma}')

output_lines.append("FROM screening_staging")
output_lines.append("ON CONFLICT DO NOTHING;")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- FINAL VALIDATION")
output_lines.append("-- ===================================================================")
output_lines.append("\\echo 'Final validation...'")
output_lines.append("SELECT 'Total rows in equities:' AS info, COUNT(*) AS count FROM equities;")
output_lines.append("SELECT 'Rows by Region:' AS info, \"Region\", COUNT(*) AS count FROM equities GROUP BY \"Region\" ORDER BY \"Region\";")
output_lines.append("SELECT 'Rows by Sector (top 10):' AS info, \"Sector\", COUNT(*) AS count FROM equities GROUP BY \"Sector\" ORDER BY COUNT(*) DESC LIMIT 10;")
output_lines.append("")
output_lines.append("-- ===================================================================")
output_lines.append("-- CLEANUP")
output_lines.append("-- ===================================================================")
output_lines.append("DROP TABLE IF EXISTS screening_staging;")
output_lines.append("")
output_lines.append("\\echo 'Import complete!'")
output_lines.append("")

# Write new import script
output_path = Path("import_equities_data_aligned.sql")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nNew aligned import script written to: {output_path}")
print(f"Total columns: {len(csv_columns)}")
print("\nFirst 10 columns in staging table:")
for i, col in enumerate(csv_columns[:10], 1):
    print(f"  {i:2d}. {col}")
