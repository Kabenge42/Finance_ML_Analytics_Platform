"""Compare embedded notebook content with current cells."""

import json

with open("ml_finance_model_main2_0.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Get the embedded notebook from first raw cell
first_cell = nb["cells"][0]
embedded_json = "".join(first_cell.get("source", []))
embedded_nb = json.loads(embedded_json)

print("=== CURRENT NOTEBOOK ===")
print(f"Total cells: {len(nb['cells'])}")
cell_types = {}
for c in nb["cells"]:
    ct = c["cell_type"]
    cell_types[ct] = cell_types.get(ct, 0) + 1
print(f"Cell types: {cell_types}")

print("\n=== EMBEDDED NOTEBOOK (inside raw cell) ===")
print(f"Total cells: {len(embedded_nb['cells'])}")
emb_cell_types = {}
for c in embedded_nb["cells"]:
    ct = c["cell_type"]
    emb_cell_types[ct] = emb_cell_types.get(ct, 0) + 1
print(f"Cell types: {emb_cell_types}")

# Compare first few cells of each (skipping the raw cell in current)
print("\n=== COMPARISON ===")
print("Current cells (starting from index 1, after raw cell):")
for i, c in enumerate(nb["cells"][1:6], 1):
    src = "".join(c.get("source", []))[:60].replace("\n", "\\n")
    print(f"  [{i}] {c['cell_type']}: {src}...")

print("\nEmbedded cells (first 5):")
for i, c in enumerate(embedded_nb["cells"][:5]):
    src = "".join(c.get("source", []))[:60].replace("\n", "\\n")
    print(f"  [{i}] {c['cell_type']}: {src}...")

# Check if embedded has proper structure
print(f"\n=== EMBEDDED NOTEBOOK METADATA ===")
print(f"nbformat: {embedded_nb.get('nbformat')}")
print(f"nbformat_minor: {embedded_nb.get('nbformat_minor')}")
print(f"Has kernelspec: {'kernelspec' in embedded_nb.get('metadata', {})}")
