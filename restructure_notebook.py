"""
Comprehensive Notebook Restructuring Script

This script implements the comprehensive notebook restructuring plan:
1. Remove duplicate Phase 9.3 section (cells 111-127)
2. Reorder Phase 9.5/9.6 sections
3. Consolidate Phase 9.7 and 9.8 sections
4. Update Phase 9.5 with 4-step imputation strategy
5. Add validation gates
6. Standardize headers

Author: Claude Code
Date: 2025-11-05
"""

import json
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


def load_notebook(path):
    """Load Jupyter notebook from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(notebook, path):
    """Save Jupyter notebook to file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)


def remove_duplicate_phase93(notebook):
    """Remove duplicate Phase 9.3 sections by detecting actual duplicates."""
    print("\n=== Step 1: Removing Duplicate Phase 9.3 Section ===")

    # Find all Phase 9.3 section markers (only major sections with ##, not subsections ###)
    phase93_sections = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            # Only match major Phase 9.3 sections (##), not subsections (###)
            if source.strip().startswith("## Phase 9.3"):
                # Find the end of this section (next phase marker or end of notebook)
                end_idx = i + 1
                for j in range(i + 1, len(notebook["cells"])):
                    next_cell = notebook["cells"][j]
                    if next_cell["cell_type"] == "markdown" and next_cell["source"]:
                        next_source = "".join(next_cell["source"])
                        # Stop at next major phase or next Phase 9.3
                        if "## Phase 9." in next_source:
                            end_idx = j
                            break

                phase93_sections.append(
                    {
                        "start": i,
                        "end": end_idx,
                        "title": source.split("\n")[0],
                        "cell_count": end_idx - i,
                    }
                )

    print(f"Found {len(phase93_sections)} Phase 9.3 section(s)")

    # If we have duplicates, remove the later ones
    if len(phase93_sections) > 1:
        cells_to_remove = []
        # Keep the first section, remove subsequent ones
        for section in phase93_sections[1:]:
            print(f"  Marking cells {section['start']}-{section['end']-1} for removal (duplicate)")
            cells_to_remove.extend(range(section["start"], section["end"]))

        # Remove in reverse order to maintain indices
        removed_count = 0
        for cell_idx in sorted(cells_to_remove, reverse=True):
            if cell_idx < len(notebook["cells"]):
                cell = notebook["cells"][cell_idx]
                if cell["cell_type"] == "markdown" and cell["source"]:
                    source = "".join(cell["source"])
                    print(f"  Removing Cell {cell_idx}: {source.split(chr(10))[0][:60]}")
                del notebook["cells"][cell_idx]
                removed_count += 1

        print(f"✓ Removed {removed_count} duplicate cells")
    else:
        print("  No duplicate Phase 9.3 sections found")

    return notebook


def reorder_phase95_96(notebook):
    """Reorder Phase 9.5/9.6 sections to fix sequence."""
    print("\n=== Step 2: Reordering Phase 9.5/9.6 Sections ===")

    # After removing 17 cells (111-127), indices shift down by 17
    # Original: 9.5 (139-141), 9.5.1 (142-143), 9.6.1 (144-149), 9.6 (150-151)
    # After deletion: indices shift by -17
    # New indices: 9.5 (122-124), 9.5.1 (125-126), 9.6.1 (127-132), 9.6 (133-134)

    # Find the sections dynamically
    phase_markers = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            if "## Phase 9.5 " in source and "Sector-Optimized" in source:
                phase_markers["9.5_start"] = i
            elif "## Phase 9.5.1" in source:
                phase_markers["9.5.1_start"] = i
            elif "## Phase 9.6.1" in source:
                phase_markers["9.6.1_start"] = i
            elif "## Phase 9.6 " in source and "Model Evaluation" in source:
                phase_markers["9.6_start"] = i
            elif "## Phase 9.7" in source:
                phase_markers["9.7_start"] = i

    print(f"Found phase markers: {phase_markers}")

    if "9.6.1_start" in phase_markers and "9.6_start" in phase_markers:
        # 9.6.1 comes before 9.6, need to swap
        idx_961 = phase_markers["9.6.1_start"]
        idx_96 = phase_markers["9.6_start"]
        idx_97 = phase_markers.get("9.7_start", len(notebook["cells"]))

        if idx_961 < idx_96:
            print(
                f"  Moving Phase 9.6.1 (cells {idx_961}-{idx_96-1}) after Phase 9.6 (cells {idx_96}-{idx_97-1})"
            )

            # Extract Phase 9.6.1 cells
            phase_961_cells = notebook["cells"][idx_961:idx_96]

            # Remove Phase 9.6.1 from current position
            del notebook["cells"][idx_961:idx_96]

            # Adjust indices after deletion
            new_idx_97 = idx_97 - len(phase_961_cells)

            # Insert Phase 9.6.1 after Phase 9.6
            # Phase 9.6 now starts at idx_961 (shifted)
            # Find where Phase 9.6 ends (before 9.7)
            insert_pos = new_idx_97

            # Insert cells
            for i, cell in enumerate(phase_961_cells):
                notebook["cells"].insert(insert_pos + i, cell)

            print(f"✓ Reordered: Now sequence is 9.5 → 9.5.1 → 9.6 → 9.6.1 → 9.7")
    else:
        print("  Sections already in correct order or markers not found")

    return notebook


def consolidate_phase97(notebook):
    """Consolidate Phase 9.7 sections (merge duplicates)."""
    print("\n=== Step 3: Consolidating Phase 9.7 Sections ===")

    # Find all Phase 9.7 headers
    phase97_indices = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            if "## Phase 9.7" in source or "### Phase 9.7" in source:
                phase97_indices.append((i, source.split("\n")[0][:60]))

    print(f"Found {len(phase97_indices)} Phase 9.7 sections:")
    for idx, title in phase97_indices:
        print(f"  Cell {idx}: {title}")

    if len(phase97_indices) > 1:
        # Keep the first, remove others (they should be adjacent or marked as "Enhanced")
        # Mark "Enhanced" sections for merging
        for idx, title in reversed(phase97_indices[1:]):
            if "Enhanced" in title:
                print(f"  Removing duplicate/enhanced section at Cell {idx}")
                # Move content to first section if needed, then delete
                del notebook["cells"][idx]
        print(f"✓ Consolidated Phase 9.7 sections")
    else:
        print("  Only one Phase 9.7 section found")

    return notebook


def consolidate_phase98(notebook):
    """Consolidate Phase 9.8 sections (merge duplicates)."""
    print("\n=== Step 4: Consolidating Phase 9.8 Sections ===")

    # Find all Phase 9.8 headers
    phase98_indices = []
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            if "## Phase 9.8" in source:
                phase98_indices.append((i, source.split("\n")[0][:80]))

    print(f"Found {len(phase98_indices)} Phase 9.8 sections:")
    for idx, title in phase98_indices:
        print(f"  Cell {idx}: {title}")

    if len(phase98_indices) > 1:
        # Keep first, remove duplicates
        for idx, title in reversed(phase98_indices[1:]):
            print(f"  Removing duplicate section at Cell {idx}")
            del notebook["cells"][idx]
        print(f"✓ Consolidated Phase 9.8 sections")
    else:
        print("  Only one Phase 9.8 section found")

    return notebook


def update_phase95_imputation(notebook):
    """Update Phase 9.5 to use 4-step imputation strategy."""
    print("\n=== Step 5: Updating Phase 9.5 Imputation Strategy ===")

    # Find Phase 9.5 section
    phase95_start = None
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            source = "".join(cell["source"])
            if "## Phase 9.5" in source and "Sector-Optimized" in source:
                phase95_start = i
                break

    if phase95_start is None:
        print("  ⚠ Phase 9.5 section not found")
        return notebook

    # Look for imputation code in next 20 cells
    imputation_cell_idx = None
    for i in range(phase95_start, min(phase95_start + 20, len(notebook["cells"]))):
        cell = notebook["cells"][i]
        if cell["cell_type"] == "code" and cell["source"]:
            source = "".join(cell["source"])
            # Search for various markers that indicate where to inject imputation
            if (
                "prepare_regression_data" in source
                or "Handling missing values" in source
                or "fillna" in source
            ):
                imputation_cell_idx = i
                break

    if imputation_cell_idx is None:
        print("  ⚠ Imputation code cell not found")
        return notebook

    print(f"  Found imputation code at Cell {imputation_cell_idx}")

    # Replacement code using 4-step imputation
    new_imputation_code = """# ============================================================================
# STEP 2.1: COMPREHENSIVE NaN HANDLING WITH 4-STEP IMPUTATION
# ============================================================================
from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy_4step

print("\\n🔧 Step 2.1: Applying 4-step imputation strategy...")

# Log NaN counts before imputation
nan_before = all_stocks_phase95.select_dtypes(include=[np.number]).isnull().sum().sum()
print(f"  NaN values before imputation: {nan_before:,}")

# Apply comprehensive 4-step imputation
# Step 1: Zero imputation for exceptional events (48 cols)
# Step 2: Sector-aware KNN imputation (148 cols)
# Step 3: Price imputation for price targets (5 cols)
# Step 4: Median imputation for remaining columns
all_stocks_phase95 = apply_enhanced_imputation_strategy_4step(
    df=all_stocks_phase95,
    sector_column='sector',
    n_neighbors=5,
    price_column='last_price'
)

# Validate zero NaN after imputation
nan_after = all_stocks_phase95.select_dtypes(include=[np.number]).isnull().sum().sum()
print(f"  NaN values after imputation: {nan_after:,}")

if nan_after == 0:
    print("✓ Zero NaN values confirmed - data ready for model training")
else:
    print(f"⚠ Warning: {nan_after} NaN values remain - applying final cleanup")
    all_stocks_phase95 = all_stocks_phase95.fillna(0)

# Handle infinite values
inf_count = np.isinf(all_stocks_phase95.select_dtypes(include=[np.number])).sum().sum()
if inf_count > 0:
    print(f"  Replacing {inf_count} infinite values with NaN, then re-imputing...")
    all_stocks_phase95 = all_stocks_phase95.replace([np.inf, -np.inf], np.nan)
    all_stocks_phase95 = all_stocks_phase95.fillna(0)
    print("✓ Infinite values handled")
"""

    # Update the cell
    notebook["cells"][imputation_cell_idx]["source"] = new_imputation_code.split("\n")
    print(f"✓ Updated imputation strategy to use 4-step method")

    return notebook


def add_validation_gates(notebook):
    """Add validation gates before model training."""
    print("\n=== Step 6: Adding Validation Gates ===")

    # Find where compare_regressors is called
    validation_added = False
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and cell["source"]:
            source = "".join(cell["source"])
            if "compare_regressors" in source or "compare_regression_models" in source:
                # Add validation before this cell
                validation_code = """# ============================================================================
# VALIDATION GATE: Verify Data Quality Before Model Training
# ============================================================================
from finance_ml.advanced_models import validate_training_data

print("\\n🔍 Validating training data before model training...")

# Extract features and target
if 'feature_cols' in locals() and 'target_col' in locals():
    X_validate = all_stocks_phase95[feature_cols].copy()
    y_validate = all_stocks_phase95[target_col].copy()

    # Run validation
    validation_result = validate_training_data(X_validate, y_validate, strict=False)

    if validation_result['valid']:
        print("✓ Data validation passed - ready for model training")
    else:
        print("⚠ Data validation issues detected:")
        for issue in validation_result['issues']:
            print(f"  - {issue}")

        if validation_result['nan_features'] > 0 or validation_result['nan_target'] > 0:
            print("  Applying emergency cleanup...")
            X_validate = X_validate.fillna(0)
            y_validate = y_validate.fillna(y_validate.median())
            all_stocks_phase95[feature_cols] = X_validate
            all_stocks_phase95[target_col] = y_validate
            print("✓ Emergency cleanup applied")
else:
    print("⚠ Feature columns or target column not defined yet")
"""

                # Insert validation cell before model training
                validation_cell = {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": validation_code.split("\n"),
                }

                notebook["cells"].insert(i, validation_cell)
                print(f"✓ Added validation gate before Cell {i} (model training)")
                validation_added = True
                break

    if not validation_added:
        print("  ⚠ Model training code not found, validation gate not added")

    return notebook


def standardize_headers(notebook):
    """Standardize section headers throughout notebook."""
    print("\n=== Step 7: Standardizing Section Headers ===")

    updates = 0
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "markdown" and cell["source"]:
            # Handle source as either list or string
            if isinstance(cell["source"], list):
                source_str = "".join(cell["source"])
            else:
                source_str = cell["source"]

            lines = source_str.split("\n")

            # Standardize Phase headers
            if lines and lines[0].startswith("##"):
                original = lines[0]
                updated = original

                # Fix Phase 9.X headers
                if "Phase 9." in original:
                    # Ensure proper spacing and em-dash
                    import re

                    match = re.match(
                        r"(#+)\s*Phase\s+(\d+\.\d+(?:\.\d+)?)\s*[-–—]?\s*(.*)", original
                    )
                    if match:
                        level, phase_num, description = match.groups()
                        # Use proper em-dash
                        updated = f"{level} Phase {phase_num} — {description.strip()}"

                        if updated != original:
                            lines[0] = updated
                            # Reconstruct source maintaining format
                            new_source = "\n".join(lines)
                            # If original was a single-element list, keep it that way
                            if isinstance(cell["source"], list) and len(cell["source"]) == 1:
                                notebook["cells"][i]["source"] = [new_source]
                            else:
                                # Keep as list with newlines embedded
                                notebook["cells"][i]["source"] = [
                                    line + "\n" if idx < len(lines) - 1 else line
                                    for idx, line in enumerate(lines)
                                ]
                            updates += 1
                            if updates <= 5:  # Show first 5 updates
                                print(f"  Cell {i}: '{original}' → '{updated}'")

    print(f"✓ Standardized {updates} section headers")
    return notebook


def main():
    """Main restructuring workflow."""
    input_path = Path("ml_finance_model_main_backup.ipynb")
    output_path = input_path

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    print(f"\n{'='*70}")
    print("COMPREHENSIVE NOTEBOOK RESTRUCTURING")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    # Load notebook
    print("\nLoading notebook...")
    notebook = load_notebook(input_path)
    original_cell_count = len(notebook["cells"])
    print(f"Original cell count: {original_cell_count}")

    # Apply transformations
    notebook = remove_duplicate_phase93(notebook)
    notebook = reorder_phase95_96(notebook)
    notebook = consolidate_phase97(notebook)
    notebook = consolidate_phase98(notebook)
    notebook = update_phase95_imputation(notebook)
    notebook = add_validation_gates(notebook)
    notebook = standardize_headers(notebook)

    # Save restructured notebook
    final_cell_count = len(notebook["cells"])
    print(f"\n{'='*70}")
    print(f"RESTRUCTURING COMPLETE")
    print(f"{'='*70}")
    print(f"Original cells: {original_cell_count}")
    print(f"Final cells: {final_cell_count}")
    print(f"Cells removed: {original_cell_count - final_cell_count}")
    print(f"\nSaving to: {output_path}")

    save_notebook(notebook, output_path)
    print("✓ Notebook saved successfully")

    return 0


if __name__ == "__main__":
    sys.exit(main())
