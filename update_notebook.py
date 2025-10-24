import json
import re

# Load the notebook
nb_path = (
    r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_v8_2.ipynb"
)
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Load the Python script to extract functions
py_path = r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_v8_2.py"
with open(py_path, "r", encoding="utf-8") as f:
    py_content = f.read()


# Extract specific functions with their complete code
def extract_function(content, func_name):
    """Extract a complete function definition including docstring."""
    pattern = rf"(def {func_name}\([^)]*\)[^:]*:.*?)(?=\n(?:def |class |\Z))"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).rstrip()
    return None


# Functions to add, organized by phase
functions_to_add = {
    "Phase 1: Data Quality Checks": [
        "check_missing_values",
        "detect_outliers_iqr",
        "validate_numeric_ranges",
    ],
    "Phase 2: Additional Feature Engineering": [
        "engineer_margin_features",
        "engineer_volatility_features",
        "engineer_revenue_cagr",
    ],
    "Phase 3: Event Classification": [
        "create_event_labels",
        "train_event_classifier",
    ],
    "Phase 5: Analytics and Stock Ranking": [
        "calculate_mispricing_score",
        "rank_undervalued_stocks",
        "rank_overvalued_stocks",
        "rank_stocks_by_sector",
    ],
}


# Helper function to create a new cell
def create_markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def create_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n"),
    }


# Find insertion point - after imports, before data loading
insert_index = 2  # After first markdown and imports

new_cells = []

# Add header
new_cells.append(
    create_markdown_cell(
        [
            "## TDD Implementation: Enhanced Functions\n",
            "\n",
            "The following functions were added using Test-Driven Development (TDD) methodology.\n",
            "All functions are covered by unit tests in the `tests/` directory.",
        ]
    )
)

# Add each phase with functions
for phase_title, func_names in functions_to_add.items():
    # Add phase header
    new_cells.append(create_markdown_cell([f"### {phase_title}\n"]))

    # Add each function
    for func_name in func_names:
        func_code = extract_function(py_content, func_name)
        if func_code:
            new_cells.append(create_code_cell(func_code))
            print(f"Added: {func_name}")
        else:
            print(f"WARNING: Could not find function: {func_name}")

# Insert the new cells into the notebook
nb["cells"] = nb["cells"][:insert_index] + new_cells + nb["cells"][insert_index:]

# Save the updated notebook
output_path = nb_path  # Overwrite the original
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nNotebook updated successfully!")
print(f"Added {len(new_cells)} cells")
print(f"Total cells now: {len(nb['cells'])}")
