import json

nb_path = (
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_v8_2.ipynb"
)
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print(f"Code cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}")
print(f"Markdown cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')}")

# Check for specific TDD functions
functions_to_find = [
    "def check_missing_values",
    "def detect_outliers_iqr",
    "def validate_numeric_ranges",
    "def engineer_margin_features",
    "def engineer_volatility_features",
    "def engineer_revenue_cagr",
    "def create_event_labels",
    "def train_event_classifier",
    "def calculate_mispricing_score",
    "def rank_undervalued_stocks",
    "def rank_overvalued_stocks",
    "def rank_stocks_by_sector",
]


def flatten_source(source):
    """Recursively flatten source to string."""
    if isinstance(source, str):
        return source
    elif isinstance(source, list):
        result = []
        for item in source:
            result.append(flatten_source(item))
        return "".join(result)
    else:
        return str(source)


print("\nSearching for TDD functions:")
found_count = 0
for func in functions_to_find:
    found = False
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = flatten_source(cell["source"])
            if func in source:
                found = True
                found_count += 1
                break
    status = "✓ FOUND" if found else "✗ NOT FOUND"
    print(f"  {func}: {status}")

print(f"\nSummary: {found_count}/{len(functions_to_find)} TDD functions found in notebook")
