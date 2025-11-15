import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "class FeatureEngineeringReporter" in source:
        with open("cell_78_content.txt", "w", encoding="utf-8") as out:
            out.write(f"Cell {i} content:\n")
            out.write("=" * 80 + "\n")
            out.write(source)
        print(f"Cell {i} content written to cell_78_content.txt")
        break
