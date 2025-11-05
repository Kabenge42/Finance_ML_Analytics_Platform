import json

# Load the notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

# Find Phase 9.7 cell (search for the cell with Phase 9.7 header)
for idx, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if "PHASE 9.7 — VALUATION AND STOCK IDENTIFICATION" in source and len(source) > 10000:
            print(f"Found Phase 9.7 cell at index {idx}")
            print(f"Cell length: {len(source)} characters")
            print("\n" + "=" * 80)
            print("PHASE 9.7 CELL CONTENT:")
            print("=" * 80)
            print(source)

            # Also save to file for review
            with open("phase97_cell_content.txt", "w", encoding="utf-8") as out:
                out.write(source)
            print("\n" + "=" * 80)
            print(f"Content saved to phase97_cell_content.txt")
            break
else:
    print("Phase 9.7 cell not found")
