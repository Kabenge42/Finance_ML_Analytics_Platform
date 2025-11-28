import json
import sys
import io

# Force UTF-8 output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]

for i, cell_source in enumerate(cells):
    cell_text = "".join(cell_source) if isinstance(cell_source, list) else cell_source

    if "build_predictions_frame" in cell_text and "try:" in cell_text:
        print(f"\n{'='*80}")
        print(f"CELL {i}: Contains build_predictions_frame with try/except")
        print("=" * 80)
        # Write to file to avoid console encoding issues
        with open(f"tools/cell_{i}_content.txt", "w", encoding="utf-8") as out:
            out.write(cell_text)
        print(f"Full content written to tools/cell_{i}_content.txt")

    if "Failed to export enhanced predictions" in cell_text:
        print(f"\n{'='*80}")
        print(f"CELL {i}: Contains 'Failed to export enhanced predictions'")
        print("=" * 80)
        with open(f"tools/cell_{i}_error.txt", "w", encoding="utf-8") as out:
            out.write(cell_text)
        print(f"Full content written to tools/cell_{i}_error.txt")
