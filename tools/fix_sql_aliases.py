import re
import sys

# Load metadata mappings
mapping = {}
with open('equities_schema_metadata_setup.sql', 'r') as f:
    content = f.read()
    # Look for ('Original Name', 'alias', ...) pattern
    matches = re.findall(r"'\s*([^']+)\s*',\s*'\s*([^']+)\s*'", content)
    for col_name, alias in matches:
        mapping[col_name] = alias

def replace_aliases(file_path):
    print(f'Processing {file_path}...')
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    changes = 0
    # Sort mapping by key length descending to avoid partial replacements
    sorted_mapping = sorted(mapping.items(), key=lambda x: len(x[0]), reverse=True)

    for line in lines:
        old_line = line
        new_line = line
        
        for col_name, alias in sorted_mapping:
            # If alias starts with a number, it MUST be quoted in PostgreSQL
            safe_alias = f'"{alias}"' if alias[0].isdigit() else alias
            
            # Use regex for AS "col_name" to avoid matching parts of other strings
            # We want to match: AS "col_name" followed by a comma, newline or space
            pattern_as = r'AS\s+"' + re.escape(col_name) + r'"'
            new_line = re.sub(pattern_as, f'AS {safe_alias}', new_line)

            # For feature_registry.sql vw_identifier_columns: e."col_name"
            pattern_e = r'e\."' + re.escape(col_name) + r'"'
            new_line = re.sub(pattern_e, f'e.{safe_alias}', new_line)

            # For other quoted usages in feature_registry.sql (expressions): "col_name"
            # We must be careful not to replace aliases that were ALREADY replaced or are already correct.
            # Only replace if it's NOT preceded by AS.
            # This is tricky. Let's look for "col_name" that is NOT part of AS "col_name"
            # Actually, most expressions use quoted names like "Div Yield (LTM)"
            # Let's use a negative lookbehind if possible, but standard re.sub is easier with specific patterns.
            
            # Replace quoted source names in expressions
            # Example: text_to_numeric_safe(s."Price Target")
            # We want s."Price Target" -> s.price_target
            pattern_s = r's\."' + re.escape(col_name) + r'"'
            new_line = re.sub(pattern_s, f's.{safe_alias}', new_line)
            
            # General quoted usage in functions
            # Example: "Net EPS - Basic (FY)" > 0
            # We use word boundaries or just assume if it's quoted and matches exactly, it's our column.
            # But avoid AS safe_alias (unquoted) or already safe_alias.
            # If we already replaced it to safe_alias (unquoted), it won't have double quotes.
            # If we replaced it to "safe_alias" (quoted), it might match again if safe_alias == col_name.
            # But our col_names usually have spaces/special chars, while aliases are snake_case.
            if col_name != safe_alias.strip('"'):
                new_line = new_line.replace(f'"{col_name}"', safe_alias)

        if new_line != old_line:
            changes += 1
        new_lines.append(new_line)
    
    if changes > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f'Done. {changes} changes made.')
    else:
        print('No changes needed.')

replace_aliases('import_equities_data.sql')
replace_aliases('feature_registry.sql')
