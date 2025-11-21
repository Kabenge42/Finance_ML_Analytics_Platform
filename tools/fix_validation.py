"""Fix validation column names in notebook Cell 63 to match 5-class system."""

import json
import sys


def fix_notebook():
    notebook_path = (
        r"C:\Users\markm\PycharmProjects\Finance_ML_Analytics_Platform\ml_finance_model_main.ipynb"
    )

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find and update cell 63
    cell = nb["cells"][63]

    if cell.get("id") != "bb4c4e64f77d4a3":
        print(f"ERROR: Expected cell ID 'bb4c4e64f77d4a3', got '{cell.get('id')}'")
        return False

    # Get source as string
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

    # Replace the old 3-class validation with 5-class validation
    old_validation = """# Verify classification columns exist
required_classification_cols = [
    'event_label', 'y_proba_negative', 'y_proba_neutral', 'y_proba_positive'
]"""

    new_validation = """# Verify classification columns exist (5-class system - code_guidelines.md Section 2.2.1)
# Phase 9.4 creates: event_prob_strong_negative, event_prob_negative, event_prob_neutral,
#                     event_prob_positive, event_prob_strong_positive, event_class_predicted, event_confidence
required_classification_cols = [
    'event_prob_strong_negative',
    'event_prob_negative',
    'event_prob_neutral',
    'event_prob_positive',
    'event_prob_strong_positive',
    'event_class_predicted',
    'event_confidence'
]"""

    if old_validation not in source:
        print("ERROR: Could not find old validation code to replace")
        print("Source preview:", source[:500])
        return False

    # Replace
    source = source.replace(old_validation, new_validation)

    # Convert back to list format for notebook
    cell["source"] = source.split("\n")
    # Ensure each line ends with \n except the last
    cell["source"] = [
        line + "\n" if i < len(cell["source"]) - 1 else line
        for i, line in enumerate(cell["source"])
    ]

    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print("Success: Updated validation in Cell 63")
    print("\nOLD columns:")
    print("  - event_label")
    print("  - y_proba_negative")
    print("  - y_proba_neutral")
    print("  - y_proba_positive")
    print("\nNEW columns (5-class system):")
    print("  - event_prob_strong_negative")
    print("  - event_prob_negative")
    print("  - event_prob_neutral")
    print("  - event_prob_positive")
    print("  - event_prob_strong_positive")
    print("  - event_class_predicted")
    print("  - event_confidence")
    return True


if __name__ == "__main__":
    success = fix_notebook()
    sys.exit(0 if success else 1)
