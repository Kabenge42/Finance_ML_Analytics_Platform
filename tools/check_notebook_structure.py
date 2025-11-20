import json

with open("ml_finance_model_main.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}\n")
print("Cells 20-35 structure:\n")

for i in range(20, min(36, len(nb["cells"]))):
    cell = nb["cells"][i]
    cell_type = cell["cell_type"]
    source = cell.get("source", [])

    if isinstance(source, list):
        first_line = source[0] if source else "(empty)"
    else:
        first_line = source[:80] if source else "(empty)"

    # Truncate to 80 chars
    first_line = str(first_line)[:80].replace("\n", " ")

    print(f"Cell {i:2d}: {cell_type:8s} | {first_line}")
