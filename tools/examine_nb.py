import json
import sys

nb_path = sys.argv[1] if len(sys.argv) > 1 else "etl_data_explorer.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Notebook: {nb_path}")
print(f"Total cells: {len(nb['cells'])}\n")

for i, cell in enumerate(nb["cells"]):
    cell_type = cell["cell_type"]
    source = "".join(cell["source"])[:100] if cell["source"] else "empty"
    # Remove newlines and non-ASCII for compact display
    source = source.replace("\n", " ").strip()
    source = source.encode("ascii", "ignore").decode("ascii")
    print(f"{i:3d}. [{cell_type:8s}] {source}")
