"""Validate Phase 9.5 notebook structure and imports."""

import json
import sys


def validate_notebook():
    """Validate the notebook structure."""
    try:
        with open("ml_finance_model_main.ipynb", "r", encoding="utf-8") as f:
            nb = json.load(f)

        print("✓ Notebook loaded successfully")
        print(f"  Total cells: {len(nb['cells'])}")

        code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
        print(f"  Code cells: {len(code_cells)}")

        # Check for Phase 9.5 section
        phase95_cells = []
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell.get("source", []))
                if "PHASE 9.5" in source or "Phase 9.5" in source:
                    phase95_cells.append(i)

        print(f"  Phase 9.5 cells found: {len(phase95_cells)}")

        # Check for syntax errors in Phase 9.5 imports
        for i in phase95_cells:
            source = "".join(nb["cells"][i].get("source", []))
            if "import" in source and "finance_ml" in source:
                # Check for double 'import' keyword
                if "import  import" in source or "import import" in source:
                    print(f"✗ Syntax error found in cell {i}: double 'import' keyword")
                    return False
                else:
                    print(f"✓ Cell {i}: imports look correct")

        # Check for key Phase 9.5 functions
        phase95_source = "\n".join(
            ["".join(nb["cells"][i].get("source", [])) for i in phase95_cells]
        )

        required_functions = [
            "prepare_regression_data",
            "create_classification_interactions",
            "compare_regressors",
            "train_stacking_regressor",
            "train_quantile_regressor",
        ]

        for func in required_functions:
            if func in phase95_source:
                print(f"✓ Function used: {func}")
            else:
                print(f"⚠ Function not found: {func}")

        # Check for error handling improvements
        error_handling_checks = [
            ("isinstance(all_stocks_phase94, pd.DataFrame)", "DataFrame type check"),
            ("test_indices.intersection", "Index alignment validation"),
            ("AttributeError", "Specific exception handling"),
            ("invalid_intervals", "Quantile validation"),
        ]

        for pattern, description in error_handling_checks:
            if pattern in phase95_source:
                print(f"✓ {description} implemented")
            else:
                print(f"⚠ {description} not found")

        print("\n✓ Notebook structure validation complete")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


if __name__ == "__main__":
    success = validate_notebook()
    sys.exit(0 if success else 1)
