import json

with open("ml_finance_model_main_backup.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

cell_147 = data["cells"][147]
source = "".join(cell_147.get("source", []))

with open("cell_147_content.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("CELL 147 - PHASE 9.5 MAIN IMPLEMENTATION\n")
    f.write("=" * 80 + "\n\n")
    f.write(source)

print(f"Cell 147 extracted successfully")
print(f"Length: {len(source)} characters")
print(f"Lines: {source.count(chr(10)) + 1}")
print(f"Output saved to: cell_147_content.txt")
