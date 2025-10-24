import json

nb_path = (
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_v8_2.ipynb"
)
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

total_cells = len(nb["cells"])
code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
markdown_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")

print(f"Total cells: {total_cells}")
print(f"Code cells: {code_cells}")
print(f"Markdown cells: {markdown_cells}")
print("\nFirst 10 cell types and previews:")
for i, cell in enumerate(nb["cells"][:10]):
    if cell["source"]:
        # Handle both list and string formats
        if isinstance(cell["source"], list):
            source_preview = "".join(cell["source"])[:100]
        else:
            source_preview = cell["source"][:100]
    else:
        source_preview = "(empty)"
    print(f"{i}: {cell['cell_type']} - {source_preview[:60]}...")

# Check for specific functions
functions_to_find = [
    "def check_missing_values",
    "def engineer_margin_features",
    "def create_event_labels",
    "def calculate_mispricing_score",
]

print("\nSearching for TDD functions:")
for func in functions_to_find:
    found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if func in source:
                found = True
                break
    print(f"  {func}: {'FOUND' if found else 'NOT FOUND'}")
