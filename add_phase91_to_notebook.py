"""
Add Phase 9.1 Enhanced Imputation Strategy to notebook.
"""
import json
import sys

# Load notebook
with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
    nb = f.read()
    nb_data = json.loads(nb)

# Phase 9.1 cell content
phase91_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Phase 9.1: Enhanced Two-Step Imputation Strategy\n",
        "\n",
        "This section implements a sophisticated two-step imputation approach:\n",
        "1. **Zero imputation** for exceptional event columns (impairments, restructuring, acquisitions)\n",
        "2. **Sector-aware KNN imputation** for core financial metrics\n",
        "\n",
        "The strategy recognizes that missing values have different meanings in different contexts."
    ]
}

phase91_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Phase 9.1: Enhanced Two-Step Imputation Strategy\n",
        "print(\"\\n\" + \"=\"*80)\n",
        "print(\"Phase 9.1: Enhanced Two-Step Imputation Strategy\")\n",
        "print(\"=\"*80)\n",
        "\n",
        "from finance_ml.advanced_preprocessing import (\n",
        "    apply_enhanced_imputation_strategy,\n",
        "    get_zero_imputation_columns,\n",
        "    get_knn_imputation_columns\n",
        ")\n",
        "\n",
        "# Show column lists\n",
        "print(\"\\nStep 1: Zero Imputation Columns (Exceptional Events)\")\n",
        "zero_cols = get_zero_imputation_columns()\n",
        "print(f\"Total columns for zero imputation: {len(zero_cols)}\")\n",
        "if 'all_stocks' in dir():\n",
        "    print(f\"Available in dataset: {sum(1 for c in zero_cols if c in all_stocks.columns)}\")\n",
        "print(f\"Sample columns: {zero_cols[:5]}\")\n",
        "\n",
        "print(\"\\nStep 2: KNN Imputation Columns (Core Financial Metrics)\")\n",
        "knn_cols = get_knn_imputation_columns()\n",
        "print(f\"Total columns for KNN imputation: {len(knn_cols)}\")\n",
        "if 'all_stocks' in dir():\n",
        "    print(f\"Available in dataset: {sum(1 for c in knn_cols if c in all_stocks.columns)}\")\n",
        "print(f\"Sample columns: {knn_cols[:5]}\")\n",
        "\n",
        "# Apply enhanced imputation strategy\n",
        "if 'all_stocks' in dir():\n",
        "    print(\"\\nApplying two-step imputation strategy...\")\n",
        "    missing_before = all_stocks.select_dtypes(include=[np.number]).isna().sum().sum()\n",
        "    \n",
        "    all_stocks_imputed = apply_enhanced_imputation_strategy(\n",
        "        all_stocks,\n",
        "        sector_column='sector',\n",
        "        n_neighbors=5\n",
        "    )\n",
        "    \n",
        "    missing_after = all_stocks_imputed.select_dtypes(include=[np.number]).isna().sum().sum()\n",
        "    reduction = missing_before - missing_after\n",
        "    pct_reduction = (reduction / missing_before * 100) if missing_before > 0 else 0\n",
        "    \n",
        "    print(f\"\\nImputation Results:\")\n",
        "    print(f\"  Missing values before: {missing_before:,}\")\n",
        "    print(f\"  Missing values after:  {missing_after:,}\")\n",
        "    print(f\"  Reduction:            {reduction:,} ({pct_reduction:.1f}%)\")\n",
        "    \n",
        "    # Visualize imputation impact\n",
        "    import matplotlib.pyplot as plt\n",
        "    \n",
        "    fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "    \n",
        "    # Plot 1: Missing values by column type\n",
        "    zero_cols_present = [c for c in zero_cols if c in all_stocks.columns]\n",
        "    knn_cols_present = [c for c in knn_cols if c in all_stocks.columns]\n",
        "    \n",
        "    missing_by_type = pd.DataFrame({\n",
        "        'Before': [\n",
        "            all_stocks[zero_cols_present].isna().sum().sum() if zero_cols_present else 0,\n",
        "            all_stocks[knn_cols_present].isna().sum().sum() if knn_cols_present else 0,\n",
        "        ],\n",
        "        'After': [\n",
        "            all_stocks_imputed[zero_cols_present].isna().sum().sum() if zero_cols_present else 0,\n",
        "            all_stocks_imputed[knn_cols_present].isna().sum().sum() if knn_cols_present else 0,\n",
        "        ]\n",
        "    }, index=['Zero Imputation Cols', 'KNN Imputation Cols'])\n",
        "    \n",
        "    missing_by_type.plot(kind='bar', ax=axes[0], color=['#e74c3c', '#27ae60'])\n",
        "    axes[0].set_title('Missing Values Before/After Imputation by Strategy')\n",
        "    axes[0].set_ylabel('Missing Value Count')\n",
        "    axes[0].set_xlabel('Column Type')\n",
        "    axes[0].legend(title='Status')\n",
        "    axes[0].grid(axis='y', alpha=0.3)\n",
        "    axes[0].tick_params(axis='x', rotation=45)\n",
        "    \n",
        "    # Plot 2: Top columns with most imputation\n",
        "    top_imputed = (all_stocks.isna().sum() - all_stocks_imputed.isna().sum()).nlargest(10)\n",
        "    if len(top_imputed) > 0:\n",
        "        top_imputed.plot(kind='barh', ax=axes[1], color='#3498db')\n",
        "        axes[1].set_title('Top 10 Columns by Imputation Count')\n",
        "        axes[1].set_xlabel('Values Imputed')\n",
        "        axes[1].grid(axis='x', alpha=0.3)\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    if 'outputs_dir' in dir():\n",
        "        plt.savefig(outputs_dir / 'phase_9_1_imputation_impact.png', dpi=300, bbox_inches='tight')\n",
        "    plt.show()\n",
        "    \n",
        "    print(\"\\n✓ Phase 9.1 Enhanced imputation strategy applied successfully!\")\n",
        "    \n",
        "    # Update all_stocks to use imputed version\n",
        "    all_stocks = all_stocks_imputed\n",
        "else:\n",
        "    print(\"\\n⚠ all_stocks dataframe not found. Skipping Phase 9.1 imputation.\")"
    ]
}

# Find a good place to insert (after data loading/preprocessing, before modeling)
# Look for cells containing "Missing values" or similar
insert_index = None
for i, cell in enumerate(nb_data['cells']):
    source = ''.join(cell.get('source', []))
    if 'Missing values check' in source or 'missing values' in source.lower():
        insert_index = i + 1
        break

# If not found, add near the end but before modeling sections
if insert_index is None:
    for i, cell in enumerate(nb_data['cells']):
        source = ''.join(cell.get('source', []))
        if 'model' in source.lower() or 'training' in source.lower():
            insert_index = i
            break

# Default to adding after first 20 cells if still not found
if insert_index is None:
    insert_index = min(20, len(nb_data['cells']))

print(f"Inserting Phase 9.1 cells at position {insert_index}")

# Insert the cells
nb_data['cells'].insert(insert_index, phase91_cell)
nb_data['cells'].insert(insert_index + 1, phase91_code_cell)

# Save notebook
with open('ml_finance_model_main.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_data, f, indent=1, ensure_ascii=False)

print("✓ Phase 9.1 cells added to notebook successfully!")
print(f"  - Markdown cell at index {insert_index}")
print(f"  - Code cell at index {insert_index + 1}")
