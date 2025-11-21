import json

# Load and validate notebook
with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f'✓ Notebook valid JSON: {len(nb["cells"])} cells')

# Check Cell 107 (Section 10.2)
cell_107 = nb["cells"][107]
source_107 = "".join(cell_107["source"])

print(f"\n✓ Cell 107 (Section 10.2):")
print(f'  - Lines: {len(cell_107["source"])}')
print(f"  - Total chars: {len(source_107)}")
print(f'  - Uses MIN_PORTFOLIO_CANDIDATES: {"MIN_PORTFOLIO_CANDIDATES" in source_107}')
print(f'  - Uses DEFAULT_EXPECTED_RETURN: {"DEFAULT_EXPECTED_RETURN" in source_107}')
print(f'  - Uses TRAIN_SIZE: {"TRAIN_SIZE" in source_107}')
print(f'  - Uses LAG_PERIODS: {"LAG_PERIODS" in source_107}')
print(f'  - Uses TECHNICAL_INDICATORS: {"TECHNICAL_INDICATORS" in source_107}')
print(f'  - Uses TARGET_COL_FALLBACK: {"TARGET_COL_FALLBACK" in source_107}')
print(f'  - Uses MIN_DATES_FOR_TIMESERIES: {"MIN_DATES_FOR_TIMESERIES" in source_107}')
print(f'  - Uses MIN_DATES_FOR_RELIABLE_ML: {"MIN_DATES_FOR_RELIABLE_ML" in source_107}')
print(
    f'  - Has logging statements: {"logger.info" in source_107 or "logger.warning" in source_107}'
)

# Find config import cell
import_cells = [
    i
    for i, c in enumerate(nb["cells"])
    if "from finance_ml.ml_workflow.analytics import" in "".join(c.get("source", []))
    and "MIN_PORTFOLIO_CANDIDATES" in "".join(c.get("source", []))
]

if import_cells:
    print(f"\n✓ Config import cell found at index: {import_cells[0]}")
    import_cell = nb["cells"][import_cells[0]]
    import_source = "".join(import_cell["source"])
    print(f'  - Imports MIN_PORTFOLIO_CANDIDATES: {"MIN_PORTFOLIO_CANDIDATES" in import_source}')
    print(f'  - Imports DEFAULT_EXPECTED_RETURN: {"DEFAULT_EXPECTED_RETURN" in import_source}')
else:
    print("\n⚠️  Config import cell not found!")

print("\n✓ Notebook validation complete")
