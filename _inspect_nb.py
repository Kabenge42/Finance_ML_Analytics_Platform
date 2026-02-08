import json

with open("feature_analytics.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        print(f"=== Cell {i} ===")
        print(src[:3000])
        print()
