import re

with open("create_equities_schema.sql", "r") as f:
    content = f.read()

roles_to_zero_fill = ["financial_statement", "balance_sheet", "cash_flow", "count"]


def replacer(match):
    line = match.group(0)
    role = match.group(2)
    if role in roles_to_zero_fill:
        # Check if already has DEFAULT 0
        if "DEFAULT 0" in line:
            return line
        return line.replace("NUMERIC,", "NUMERIC DEFAULT 0,")
    return line


# Pattern to match: "Column Name" NUMERIC, -- role: ...
pattern = r'^\s*"([^"]+)"\s+NUMERIC,\s+--\s+([^:]+):.*$'
new_content = re.sub(pattern, replacer, content, flags=re.MULTILINE)

with open("create_equities_schema.sql", "w") as f:
    f.write(new_content)
