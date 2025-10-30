#!/usr/bin/env python3
"""
Add Phase 9.2 integration examples to ml_finance_model_main.ipynb.

This script adds cells demonstrating feature importance and multivariate analysis
integration into simple_eda().
"""

import json
from pathlib import Path


def add_phase92_integration_cells():
    """Add Phase 9.2 integration cells to the notebook."""

    notebook_path = Path("ml_finance_model_main.ipynb")

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the last cell containing simple_eda (should be in Phase 9.2 section)
    last_eda_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell.get("source", []))
            if "simple_eda" in source and "Phase 9.2" in source:
                last_eda_idx = i

    if last_eda_idx == -1:
        print("Could not find Phase 9.2 simple_eda cell. Appending to end.")
        insert_idx = len(nb["cells"])
    else:
        insert_idx = last_eda_idx + 1
        print(f"Found Phase 9.2 simple_eda at cell {last_eda_idx}. Inserting after it.")

    # Create new cells
    new_cells = []

    # Markdown: Feature Importance Integration
    new_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 9.2.1 Feature Importance Analysis Integration\n",
                "\n",
                "**Phase 9.2 Enhancement**: `simple_eda()` now supports feature importance analysis when a target column is provided.\n",
                "\n",
                "The function integrates:\n",
                "- **Mutual Information**: Measures statistical dependency between features and target\n",
                "- **Random Forest Importance**: Feature importance from ensemble model\n",
                "- **SHAP Values**: Model-agnostic feature importance (optional, may be slow)\n",
                "\n",
                "This helps identify which features are most predictive of the target variable.",
            ],
        }
    )

    # Code: Feature Importance Example
    new_cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Demonstrate feature importance analysis with target column\n",
                "# Using a subset for faster computation\n",
                "sample_stocks = all_stocks.head(100).copy()\n",
                "\n",
                "# Ensure we have a target column (e.g., price_target or last_price)\n",
                "if 'price_target' in sample_stocks.columns:\n",
                "    target_col = 'price_target'\n",
                "elif 'last_price' in sample_stocks.columns:\n",
                "    target_col = 'last_price'\n",
                "else:\n",
                '    print("No suitable target column found. Skipping feature importance demo.")\n',
                "    target_col = None\n",
                "\n",
                "if target_col:\n",
                '    print(f"Running simple_eda with feature importance (target: {target_col})...")\n',
                "    eda_with_importance = simple_eda(\n",
                "        sample_stocks,\n",
                "        out_dir=output_dir / 'eda_with_importance',\n",
                "        target_column=target_col\n",
                "    )\n",
                "    \n",
                "    # Display feature importance results\n",
                "    if 'feature_importance' in eda_with_importance:\n",
                '        print("\\n=== Feature Importance Results ===")\n',
                "        \n",
                "        # Mutual Information\n",
                "        if 'mutual_information' in eda_with_importance['feature_importance']:\n",
                "            mi_scores = eda_with_importance['feature_importance']['mutual_information']\n",
                "            if mi_scores:\n",
                '                print("\\nTop 5 features by Mutual Information:")\n',
                "                sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "                for feat, score in sorted_mi:\n",
                '                    print(f"  {feat}: {score:.4f}")\n',
                "        \n",
                "        # Random Forest\n",
                "        if 'random_forest' in eda_with_importance['feature_importance']:\n",
                "            rf_scores = eda_with_importance['feature_importance']['random_forest']\n",
                "            if rf_scores:\n",
                '                print("\\nTop 5 features by Random Forest Importance:")\n',
                "                sorted_rf = sorted(rf_scores.items(), key=lambda x: x[1], reverse=True)[:5]\n",
                "                for feat, score in sorted_rf:\n",
                '                    print(f"  {feat}: {score:.4f}")\n',
                "    else:\n",
                '        print("No feature importance results (may need more data or numeric features)")\n',
            ],
        }
    )

    # Markdown: Multivariate Analysis Integration
    new_cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 9.2.2 Multivariate Analysis Integration\n",
                "\n",
                "**Phase 9.2 Enhancement**: `simple_eda()` now supports multivariate dimensionality reduction analysis.\n",
                "\n",
                "When `include_multivariate=True`, the function performs:\n",
                "- **PCA (Principal Component Analysis)**: Identifies main variance directions\n",
                "- **t-SNE**: Non-linear dimensionality reduction for visualization (optional)\n",
                "\n",
                "This helps understand high-dimensional data structure and detect patterns.",
            ],
        }
    )

    # Code: Multivariate Analysis Example
    new_cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Demonstrate multivariate analysis (PCA, t-SNE)\n",
                "# Using a subset for faster computation\n",
                "sample_stocks_mv = all_stocks.head(100).copy()\n",
                "\n",
                'print("Running simple_eda with multivariate analysis (PCA, t-SNE)...")\n',
                "eda_with_multivariate = simple_eda(\n",
                "    sample_stocks_mv,\n",
                "    out_dir=output_dir / 'eda_with_multivariate',\n",
                "    include_multivariate=True\n",
                ")\n",
                "\n",
                "# Display multivariate analysis results\n",
                "if 'multivariate_analysis' in eda_with_multivariate:\n",
                '    print("\\n=== Multivariate Analysis Results ===")\n',
                "    \n",
                "    # PCA Results\n",
                "    if 'pca' in eda_with_multivariate['multivariate_analysis']:\n",
                "        pca_result = eda_with_multivariate['multivariate_analysis']['pca']\n",
                "        if pca_result:\n",
                '            print("\\nPCA Results:")\n',
                "            print(f\"  Number of components: {pca_result.get('n_components', 'N/A')}\")\n",
                "            if 'explained_variance_ratio' in pca_result:\n",
                "                evr = pca_result['explained_variance_ratio']\n",
                '                print(f"  Explained variance ratio: {evr}")\n',
                "            if 'cumulative_variance' in pca_result:\n",
                "                cv = pca_result['cumulative_variance']\n",
                '                print(f"  Cumulative variance: {cv}")\n',
                '                print(f"  Variance explained by {len(cv)} components: {cv[-1]:.2%}")\n',
                "    \n",
                "    # t-SNE Results\n",
                "    if 'tsne' in eda_with_multivariate['multivariate_analysis']:\n",
                "        tsne_result = eda_with_multivariate['multivariate_analysis']['tsne']\n",
                "        if tsne_result:\n",
                '            print("\\nt-SNE Results:")\n',
                "            print(f\"  Number of components: {tsne_result.get('n_components', 'N/A')}\")\n",
                "            print(f\"  Components shape: {tsne_result.get('components_shape', 'N/A')}\")\n",
                "else:\n",
                '    print("No multivariate analysis results (may need more data or numeric features)")\n',
            ],
        }
    )

    # Insert cells
    for i, cell in enumerate(new_cells):
        nb["cells"].insert(insert_idx + i, cell)

    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Added {len(new_cells)} cells to notebook at position {insert_idx}")
    print(f"  - Markdown: Feature Importance Integration")
    print(f"  - Code: Feature Importance Example")
    print(f"  - Markdown: Multivariate Analysis Integration")
    print(f"  - Code: Multivariate Analysis Example")
    print(f"\nNotebook now has {len(nb['cells'])} cells (was {len(nb['cells']) - len(new_cells)})")


if __name__ == "__main__":
    add_phase92_integration_cells()
