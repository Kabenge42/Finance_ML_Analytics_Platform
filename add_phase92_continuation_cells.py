#!/usr/bin/env python3
"""Add Phase 9.2 continuation examples to ml_finance_model_main.ipynb."""

import json
from pathlib import Path


def add_phase92_continuation_cells():
    """Add cells demonstrating distance correlation, outlier visualizations, and UMAP."""

    notebook_path = Path("ml_finance_model_main.ipynb")

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find the Phase 9.2 section (after existing simple_eda integration cells)
    # Look for a cell containing "Phase 9.2" or add after the last Phase 9.2 cell
    insert_index = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            if "Phase 9.2" in source and "Integration" in source:
                # Find the end of Phase 9.2 section
                insert_index = i + 1
                # Keep looking for more Phase 9.2 cells
                for j in range(i + 1, len(nb["cells"])):
                    cell_source = (
                        "".join(nb["cells"][j]["source"])
                        if isinstance(nb["cells"][j]["source"], list)
                        else nb["cells"][j]["source"]
                    )
                    if "simple_eda" in cell_source or "Phase 9.2" in cell_source:
                        insert_index = j + 1
                    elif nb["cells"][j]["cell_type"] == "markdown" and (
                        "Phase 9.3" in cell_source
                        or "## 9.3" in cell_source
                        or "###" in cell_source
                    ):
                        break

    if insert_index is None:
        # If no Phase 9.2 section found, add after EDA section
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "markdown":
                source = (
                    "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
                )
                if "Exploratory Data Analysis" in source or "EDA" in source:
                    insert_index = i + 1
                    break

    if insert_index is None:
        print("⚠ Could not find appropriate insertion point")
        return False

    # New cells to add
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Phase 9.2 Continuation: Advanced Correlation & Outlier Analysis\n",
                "\n",
                "Demonstrating newly implemented features:\n",
                "1. **Distance Correlation** - Captures non-linear dependencies (requires `dcor`)\n",
                "2. **Outlier Visualizations** - Box plots, violin plots, and scatter plots with z-scores\n",
                "3. **UMAP Integration** - Additional dimensionality reduction (requires `umap-learn`)",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example: Distance Correlation Analysis\n",
                "# Distance correlation can detect both linear and non-linear relationships\n",
                "\n",
                "try:\n",
                "    from finance_ml.eval import calculate_distance_correlation\n",
                "    \n",
                "    # Select numeric columns for analysis\n",
                "    numeric_features = ['last_price', 'market_cap', 'pe_ratio', 'pb_ratio', 'ev_ebitda']\n",
                "    available_features = [col for col in numeric_features if col in all_stocks.columns]\n",
                "    \n",
                "    if len(available_features) >= 2:\n",
                "        # Calculate distance correlation matrix\n",
                "        dcor_matrix = calculate_distance_correlation(all_stocks, available_features)\n",
                "        \n",
                '        print("\\n📊 Distance Correlation Matrix (captures non-linear dependencies):")\n',
                "        print(dcor_matrix.round(3))\n",
                "        \n",
                "        # Compare with Pearson correlation\n",
                "        pearson_matrix = all_stocks[available_features].corr(method='pearson')\n",
                '        print("\\n📈 Pearson Correlation Matrix (linear dependencies only):")\n',
                "        print(pearson_matrix.round(3))\n",
                "        \n",
                '        print("\\n💡 Distance correlation detects dependencies Pearson might miss!")\n',
                "    else:\n",
                '        print("⚠ Need at least 2 numeric columns for correlation analysis")\n',
                "        \n",
                "except ImportError:\n",
                "    print(\"ℹ Distance correlation requires 'dcor' library\")\n",
                '    print("  Install with: pip install dcor")\n',
                "except Exception as e:\n",
                '    print(f"⚠ Distance correlation analysis failed: {e}")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example: Outlier Visualization Functions\n",
                "# Visualize outliers using multiple methods\n",
                "\n",
                "from finance_ml.eval import (\n",
                "    plot_outlier_boxplots,\n",
                "    plot_outlier_violins,\n",
                "    plot_outlier_scatter\n",
                ")\n",
                "\n",
                "# Select columns for outlier analysis\n",
                "outlier_features = ['last_price', 'market_cap', 'pe_ratio', 'pb_ratio']\n",
                "available_outlier = [col for col in outlier_features if col in all_stocks.columns]\n",
                "\n",
                "if len(available_outlier) >= 2:\n",
                '    print("\\n📦 Creating Outlier Visualizations...")\n',
                "    \n",
                "    # Box plots - show quartiles and outliers\n",
                "    try:\n",
                "        fig_box = plot_outlier_boxplots(\n",
                "            all_stocks, \n",
                "            columns=available_outlier[:4],\n",
                "            out_path=outputs_dir / 'outlier_boxplots.png'\n",
                "        )\n",
                "        if fig_box:\n",
                '            print("  ✓ Box plots saved to outputs/outlier_boxplots.png")\n',
                "    except Exception as e:\n",
                '        print(f"  ⚠ Box plots failed: {e}")\n',
                "    \n",
                "    # Violin plots - show distribution density\n",
                "    try:\n",
                "        fig_violin = plot_outlier_violins(\n",
                "            all_stocks, \n",
                "            columns=available_outlier[:4],\n",
                "            out_path=outputs_dir / 'outlier_violins.png'\n",
                "        )\n",
                "        if fig_violin:\n",
                '            print("  ✓ Violin plots saved to outputs/outlier_violins.png")\n',
                "    except Exception as e:\n",
                '        print(f"  ⚠ Violin plots failed: {e}")\n',
                "    \n",
                "    # Scatter plot with z-scores - color by outlier severity\n",
                "    try:\n",
                "        fig_scatter = plot_outlier_scatter(\n",
                "            all_stocks, \n",
                "            columns=available_outlier[:2],\n",
                "            out_path=outputs_dir / 'outlier_scatter.png',\n",
                "            z_threshold=3.0  # Highlight points with |z-score| > 3\n",
                "        )\n",
                "        if fig_scatter:\n",
                '            print("  ✓ Scatter plot saved to outputs/outlier_scatter.png")\n',
                '            print("    (Points colored by z-score, outliers highlighted in red)")\n',
                "    except Exception as e:\n",
                '        print(f"  ⚠ Scatter plot failed: {e}")\n',
                "    \n",
                '    print("\\n💡 These plots are also generated automatically when save_plots=True in simple_eda()")\n',
                "else:\n",
                '    print("⚠ Need at least 2 numeric columns for outlier analysis")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example: Complete EDA with all Phase 9.2 features\n",
                "# This demonstrates distance correlation + outlier viz + UMAP integration\n",
                "\n",
                'print("\\n🔬 Running Enhanced EDA with Phase 9.2 Continuation Features...\\n")\n',
                "\n",
                "# Create sample for faster analysis\n",
                "sample_size = min(500, len(all_stocks))\n",
                "sample_df = all_stocks.sample(n=sample_size, random_state=42)\n",
                "\n",
                "# Run enhanced EDA\n",
                "enhanced_summary = simple_eda(\n",
                "    sample_df,\n",
                "    out_dir=outputs_dir / 'enhanced_eda',\n",
                "    save_plots=True,  # Generates all visualizations including outlier plots\n",
                "    target_column=None,  # Set to enable feature importance\n",
                "    include_multivariate=True  # Enables PCA, t-SNE, and UMAP\n",
                ")\n",
                "\n",
                'print("\\n📊 Enhanced EDA Summary:")\n',
                "print(f\"  Total Features Analyzed: {enhanced_summary['column_count']}\")\n",
                "print(f\"  Numeric Features: {enhanced_summary['numeric_cols_count']}\")\n",
                "\n",
                "# Check correlation analysis results\n",
                "if 'correlation_analysis' in enhanced_summary:\n",
                "    corr_methods = [k for k in enhanced_summary['correlation_analysis'].keys() if enhanced_summary['correlation_analysis'][k]]\n",
                "    print(f\"\\n  Correlation Methods Available: {', '.join(corr_methods)}\")\n",
                "    if 'distance' in corr_methods:\n",
                '        print("    ✓ Distance correlation computed (captures non-linear relationships)")\n',
                "    else:\n",
                '        print("    ℹ Distance correlation skipped (dcor library not installed)")\n',
                "\n",
                "# Check multivariate analysis results\n",
                "if 'multivariate_analysis' in enhanced_summary:\n",
                "    multi_methods = [k for k in enhanced_summary['multivariate_analysis'].keys() if enhanced_summary['multivariate_analysis'][k]]\n",
                "    if multi_methods:\n",
                "        print(f\"\\n  Dimensionality Reduction Methods: {', '.join(multi_methods)}\")\n",
                "        if 'umap' in multi_methods:\n",
                '            print("    ✓ UMAP analysis completed (captures non-linear structure)")\n',
                "        else:\n",
                '            print("    ℹ UMAP skipped (umap-learn library not installed or insufficient data)")\n',
                "\n",
                "# Check visualizations\n",
                "viz_dir = outputs_dir / 'enhanced_eda'\n",
                "expected_plots = [\n",
                "    'eda_distributions.png',\n",
                "    'eda_correlation.png',\n",
                "    'eda_outlier_boxplots.png',\n",
                "    'eda_outlier_violins.png',\n",
                "    'eda_outlier_scatter.png'\n",
                "]\n",
                "\n",
                "generated_plots = [f for f in expected_plots if (viz_dir / f).exists()]\n",
                'print(f"\\n  Visualizations Generated: {len(generated_plots)}/{len(expected_plots)}")\n',
                "for plot in generated_plots:\n",
                '    print(f"    ✓ {plot}")\n',
                "\n",
                'print(f"\\n✅ Enhanced EDA complete! Results saved to {viz_dir}")\n',
                'print("\\n💡 Key Features:")\n',
                'print("   • Distance correlation detects non-linear dependencies")\n',
                'print("   • Outlier visualizations (box/violin/scatter) identify anomalies")\n',
                'print("   • UMAP provides non-linear dimensionality reduction")\n',
                'print("   • All features gracefully degrade when optional libraries unavailable")',
            ],
        },
    ]

    # Insert cells
    for i, cell in enumerate(new_cells):
        nb["cells"].insert(insert_index + i, cell)

    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Added {len(new_cells)} Phase 9.2 continuation cells to notebook")
    print(f"  Inserted at position {insert_index}")
    print(f"  Total cells now: {len(nb['cells'])}")
    print("\nCells demonstrate:")
    print("  1. Distance correlation analysis")
    print("  2. Outlier visualization functions")
    print("  3. Complete enhanced EDA with all features")

    return True


if __name__ == "__main__":
    add_phase92_continuation_cells()
