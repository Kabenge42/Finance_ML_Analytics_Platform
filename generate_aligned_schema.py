"""
Generate create_equities_schema.sql with columns in exact CSV order.
Preserves semantic comments and data types from original schema.
"""
import csv
import re
from pathlib import Path

# Read CSV header to get correct order
csv_path = Path("data/screening_us.csv")
with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    csv_columns = next(reader)

print(f"CSV has {len(csv_columns)} columns")

# Parse original schema to extract column definitions with comments
schema_path = Path("create_equities_schema.sql")
column_definitions = {}
current_column = None
current_def = []

with open(schema_path, 'r', encoding='utf-8') as f:
    in_create_table = False
    for line in f:
        if 'CREATE TABLE equities' in line:
            in_create_table = True
            continue
        if in_create_table:
            if line.strip().startswith(')'):
                # Save last column
                if current_column:
                    column_definitions[current_column] = ''.join(current_def)
                break
            
            # Check if this line starts a new column definition
            if '"' in line and not line.strip().startswith('--'):
                # Save previous column
                if current_column:
                    column_definitions[current_column] = ''.join(current_def)
                
                # Extract column name
                parts = line.strip().split('"')
                if len(parts) >= 2:
                    current_column = parts[1]
                    current_def = [line]
            else:
                # Continuation of current column (comment line)
                if current_column:
                    current_def.append(line)

print(f"Parsed {len(column_definitions)} column definitions from schema")

# Map data types from schema
column_types = {}
for col_name, col_def in column_definitions.items():
    # Extract type from definition
    # Format: "Column Name" TYPE, -- comment
    match = re.search(r'"\s+(\w+)', col_def)
    if match:
        col_type = match.group(1)
        column_types[col_name] = col_type
    else:
        # Default to NUMERIC for most columns
        if any(keyword in col_name.lower() for keyword in ['date', 'updated', 'end', 'earnings', 'announce', 'payable', 'record', 'ex date']):
            column_types[col_name] = 'DATE'
        elif any(keyword in col_name.lower() for keyword in ['ticker', 'isin', 'name', 'description', 'exchange', 'unit', 'sector', 'industry', 'class', 'status', 'region', 'country', 'when', 'currency', 'frequency']):
            column_types[col_name] = 'TEXT'
        else:
            column_types[col_name] = 'NUMERIC'

print(f"Mapped types for {len(column_types)} columns")

# Read the header and semantic comments from original schema
with open(schema_path, 'r', encoding='utf-8') as f:
    original_content = f.read()

# Extract the semantic category classification header
header_start = original_content.find('/*\n * SEMANTIC CATEGORY CLASSIFICATION SYSTEM')
header_end = original_content.find('*/', header_start) + 2
semantic_header = original_content[header_start:header_end]

# Extract the footer comments
footer_start = original_content.find('/*\n * =======================================\n * SEMANTIC CATEGORY SUMMARY')
footer_end = original_content.find('*/', footer_start) + 2
semantic_footer = original_content[footer_start:footer_end]

# Generate new schema with CSV column order
output_lines = []
output_lines.append("-- Drop existing table if it exists")
output_lines.append('DROP TABLE IF EXISTS "Equities";')
output_lines.append("DROP TABLE IF EXISTS equities;")
output_lines.append("-- Drop existing table if it exists")
output_lines.append('DROP TABLE IF EXISTS "Equities";')
output_lines.append("DROP TABLE IF EXISTS equities;")
output_lines.append("")
output_lines.append(semantic_header)
output_lines.append("")
output_lines.append("-- Create equities table with appropriate data types")
output_lines.append("CREATE TABLE equities")
output_lines.append("(")

# Add columns in CSV order
for i, col_name in enumerate(csv_columns):
    col_type = column_types.get(col_name, 'NUMERIC')
    
    # Get semantic comment if available
    semantic_comment = ""
    if col_name in column_definitions:
        # Extract inline comment from original definition
        col_def = column_definitions[col_name]
        comment_match = re.search(r'--\s*(.+)', col_def)
        if comment_match:
            semantic_comment = f" -- {comment_match.group(1).strip()}"
    
    # Format column definition
    comma = "," if i < len(csv_columns) - 1 else ""
    padding = " " * (50 - len(col_name))
    output_lines.append(f'    "{col_name}"{padding}{col_type}{comma}{semantic_comment}')

output_lines.append(") TABLESPACE pg_default;")
output_lines.append("")
output_lines.append("-- Set table ownership")
output_lines.append("ALTER TABLE equities")
output_lines.append("    OWNER TO postgres;")
output_lines.append("")
output_lines.append("-- Add comments")
output_lines.append("COMMENT ON TABLE equities IS 'Equities screening data with financial metrics and company information';")
output_lines.append("")
output_lines.append(semantic_footer)
output_lines.append("")

# Write new schema
output_path = Path("create_equities_schema_aligned.sql")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(output_lines))

print(f"\nNew aligned schema written to: {output_path}")
print(f"Total columns: {len(csv_columns)}")
print("\nFirst 10 columns in new schema:")
for i, col in enumerate(csv_columns[:10], 1):
    print(f"  {i:2d}. {col}")
