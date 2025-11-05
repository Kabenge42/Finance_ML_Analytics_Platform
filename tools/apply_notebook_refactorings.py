#!/usr/bin/env python3
"""Apply comprehensive refactorings to notebook cells."""

import json
import shutil
from datetime import datetime


def create_backup(filename):
    """Create timestamped backup of notebook."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{filename}.backup_{timestamp}"
    shutil.copy(filename, backup_name)
    print(f"✓ Backup created: {backup_name}")
    return backup_name


def remove_duplicate_cells(notebook, cell_indices):
    """Remove duplicate cells from notebook."""
    print(f"\n📝 Removing duplicate cells: {cell_indices}")

    # Remove cells in reverse order to maintain indices
    for idx in sorted(cell_indices, reverse=True):
        if idx < len(notebook["cells"]):
            removed_cell = notebook["cells"].pop(idx)
            print(f"  ✓ Removed cell {idx} ({len(removed_cell['source'])} lines)")

    return notebook


def extract_heatmap_function_phase95(cell_content):
    """Extract heatmap visualization code into a function for Phase 9.5."""

    # Find the heatmap code (starts around line 662)
    lines = cell_content.split("\n")

    # Find where the heatmap code starts (after the main try-except block)
    heatmap_start = None
    for i, line in enumerate(lines):
        if "# Sector-Region market cap heatmap" in line or (
            "sector" in line
            and "region" in line
            and "market_cap" in line
            and line.strip().startswith("if ")
        ):
            heatmap_start = i
            break

    if heatmap_start is None:
        print("  ⚠ Heatmap code not found in expected location")
        return cell_content

    # Extract the heatmap code
    heatmap_code = "\n".join(lines[heatmap_start:])
    code_before = "\n".join(lines[:heatmap_start])

    # Create the new function
    new_function = '''

def create_sector_region_heatmap(all_stocks: pd.DataFrame) -> None:
    """Create and display sector-region market cap heatmap.
    
    Args:
        all_stocks: DataFrame with sector, region, and market_cap columns
    """
    if 'sector' not in all_stocks.columns or 'region' not in all_stocks.columns or \\
       'market_cap' not in all_stocks.columns:
        print("\\n⚠ Skipping sector-region heatmap: required columns not found")
        return
    
    heatmap_data = all_stocks.groupby(['sector', 'region'])['market_cap'].sum().unstack(fill_value=0)
    
    # Get top 10 sectors by total market cap
    top_sectors = heatmap_data.sum(axis=1).nlargest(10).index
    heatmap_data = heatmap_data.loc[top_sectors]
    
    if not heatmap_data.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data / 1e9, annot=True, fmt='.1f', cmap='YlOrRd',
                    linewidths=0.5, cbar_kws={'label': 'Market Cap ($B)'})
        plt.title('Market Cap Distribution: Top 10 Sectors by Region', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Region')
        plt.ylabel('Sector')
        plt.tight_layout()
        plt.show()


# Call the heatmap function
create_sector_region_heatmap(all_stocks)
'''

    # Combine: code before + new function + function call
    refactored_content = code_before.rstrip() + new_function

    print("  ✓ Extracted heatmap code into create_sector_region_heatmap() function")
    return refactored_content


def extract_importance_display_methods_phase93(cell_content):
    """Extract type-specific display logic from calculate_and_display_importance."""

    lines = cell_content.split("\n")

    # Find the calculate_and_display_importance method
    method_start = None
    for i, line in enumerate(lines):
        if "def calculate_and_display_importance" in line:
            method_start = i
            break

    if method_start is None:
        print("  ⚠ calculate_and_display_importance method not found")
        return cell_content

    # Find the end of the method
    method_end = None
    indent_level = len(lines[method_start]) - len(lines[method_start].lstrip())

    for i in range(method_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith("#"):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= indent_level and not line.strip().startswith(('"""', "'''")):
                method_end = i
                break

    if method_end is None:
        method_end = len(lines)

    # Extract new helper methods to add before calculate_and_display_importance
    helper_methods = '''
    def _display_dataframe_importance(self, importance_scores: pd.DataFrame, top_k: int) -> None:
        """Display importance scores from DataFrame format."""
        if 'feature' not in importance_scores.columns or 'importance' not in importance_scores.columns:
            raise ValueError("DataFrame must have 'feature' and 'importance' columns")
        
        top_features = importance_scores.head(top_k)
        for rank, (_, row) in enumerate(top_features.iterrows(), start=1):
            feature_name = row['feature']
            score_value = float(row['importance'])
            print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")
    
    def _display_series_importance(self, importance_scores: pd.Series, top_k: int) -> None:
        """Display importance scores from Series format."""
        top_features = importance_scores.head(top_k)
        for rank, (feature_name, score) in enumerate(top_features.items(), start=1):
            score_value = float(score)
            print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")
    
    def _display_dict_importance(self, importance_scores: dict, top_k: int) -> None:
        """Display importance scores from dict format."""
        importance_series = pd.Series(importance_scores)
        top_features = importance_series.sort_values(ascending=False).head(top_k)
        for rank, (feature_name, score) in enumerate(top_features.items(), start=1):
            score_value = float(score)
            print(f"  {rank:2d}. {feature_name:<40s}: {score_value:.4f}")

'''

    # Simplified calculate_and_display_importance method
    simplified_method = '''    def calculate_and_display_importance(self, dataframe, top_k=20, exclude_cols=None):
        """Calculate and display feature importance using Random Forest."""
        if exclude_cols is None:
            exclude_cols = ['ticker', 'sector', 'region', 'price_target', 'last_price']

        # Prepare features
        feature_cols = [c for c in dataframe.columns if c not in exclude_cols]
        X_features = dataframe[feature_cols]
        y_target = dataframe['price_target'].fillna(dataframe['last_price'])

        # Calculate importance scores
        importance_scores = calculate_feature_importance_rf(
            X_features,
            y_target,
            top_k=top_k,
            n_estimators=100
        )

        # Display results using visualization method
        self._display_importance_scores(importance_scores, top_k)

        # Display formatted list using type-specific methods
        if isinstance(importance_scores, pd.DataFrame):
            self._display_dataframe_importance(importance_scores, top_k)
        elif isinstance(importance_scores, pd.Series):
            self._display_series_importance(importance_scores, top_k)
        else:
            self._display_dict_importance(importance_scores, top_k)
'''

    # Insert helper methods before calculate_and_display_importance
    new_lines = (
        lines[:method_start] + helper_methods.split("\n") + [simplified_method] + lines[method_end:]
    )

    refactored_content = "\n".join(new_lines)

    print("  ✓ Extracted type-specific display methods")
    print("    - _display_dataframe_importance()")
    print("    - _display_series_importance()")
    print("    - _display_dict_importance()")

    return refactored_content


def main():
    """Apply all refactorings to the notebook."""
    notebook_file = "ml_finance_model_main.ipynb"

    print("=" * 80)
    print("APPLYING COMPREHENSIVE NOTEBOOK REFACTORINGS")
    print("=" * 80)

    # Create backup
    backup_file = create_backup(notebook_file)

    # Load notebook
    with open(notebook_file, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    print(f"\n📊 Original notebook: {len(notebook['cells'])} cells")

    # 1. Remove duplicate Phase 9.2 cells (keep 59, remove 62 and 65)
    print("\n1️⃣  PHASE 9.2 - Removing duplicate cells")
    notebook = remove_duplicate_cells(notebook, [65, 62])  # Remove in reverse order
    print(f"  ✓ After removal: {len(notebook['cells'])} cells")

    # 2. Refactor Phase 9.5 (cell index 144, but now 142 after removing 2 cells)
    print("\n2️⃣  PHASE 9.5 - Extracting heatmap function")
    phase95_index = 142  # Adjusted for removed cells
    if phase95_index < len(notebook["cells"]):
        cell_content = "".join(notebook["cells"][phase95_index]["source"])
        refactored_content = extract_heatmap_function_phase95(cell_content)
        notebook["cells"][phase95_index]["source"] = refactored_content.split("\n")

    # 3. Refactor Phase 9.3 (cell index 85, unchanged)
    print("\n3️⃣  PHASE 9.3 - Extracting type-specific display methods")
    phase93_index = 85
    if phase93_index < len(notebook["cells"]):
        cell_content = "".join(notebook["cells"][phase93_index]["source"])
        refactored_content = extract_importance_display_methods_phase93(cell_content)
        notebook["cells"][phase93_index]["source"] = refactored_content.split("\n")

    # Save refactored notebook
    with open(notebook_file, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✓ REFACTORING COMPLETE")
    print("=" * 80)
    print(f"  Final notebook: {len(notebook['cells'])} cells")
    print(f"  Backup: {backup_file}")
    print(f"  Refactored: {notebook_file}")
    print("\nSummary of changes:")
    print("  1. Removed 2 duplicate Phase 9.2 cells")
    print("  2. Extracted heatmap function in Phase 9.5")
    print("  3. Extracted type-specific display methods in Phase 9.3")


if __name__ == "__main__":
    main()
