import json

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    source = "".join(cell.get("source", []))
    if "class FeatureEngineeringReporter" in source:
        print(f"Found in cell {i}")
        print("=" * 80)
        print(source)
        print("=" * 80)
        break
