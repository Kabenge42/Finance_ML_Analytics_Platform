import json

# Read the notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and modify the cell
for i, cell in enumerate(nb["cells"]):
    source = cell.get("source", [])
    source_str = "".join(source)

    if "class FeatureEngineeringReporter" in source_str:
        print(f"Found FeatureEngineeringReporter in cell {i}")

        # Check if import already exists
        if "from finance_ml.eval import calculate_feature_importance_rf" in source_str:
            print("Import already exists!")
            break

        # Find the position to insert the import (after existing imports)
        new_source = []
        import_added = False

        for j, line in enumerate(source):
            new_source.append(line)
            # Add import after "from typing import List, Optional"
            if "from typing import List, Optional" in line and not import_added:
                # Add newline if the next line isn't already blank
                if j + 1 < len(source) and source[j + 1].strip():
                    new_source.append("\n")
                new_source.append("from finance_ml.eval import calculate_feature_importance_rf\n")
                import_added = True

        cell["source"] = new_source
        print("Import statement added successfully!")
        break

# Write the modified notebook
with open("ml_finance_model_main.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully!")
