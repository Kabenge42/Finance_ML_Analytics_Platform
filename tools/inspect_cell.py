import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell = nb["cells"][61]
print(f"Cell 61 info:")
print(f"  Type: {cell['cell_type']}")
print(f"  Number of lines: {len(cell['source'])}")
print(f"  Total chars: {sum(len(line) for line in cell['source'])}")
print(f"  First line length: {len(cell['source'][0]) if cell['source'] else 0}")
print(f"\nFirst 3 lines:")
for i, line in enumerate(cell["source"][:3]):
    print(f"  Line {i}: {repr(line[:100])}...")
print(f"\nAll lines (showing first 100 chars of each):")
for i, line in enumerate(cell["source"][:10]):
    print(f"  {i}: {repr(line[:100])}")
