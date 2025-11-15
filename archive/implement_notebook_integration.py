"""
Implement Notebook Integration - Phase 9 Complete Workflow

This script reorganizes ml_finance_model_main.ipynb to implement the complete
Phase 9 workflow in the correct order with all missing phases added.

Correct Order:
1. Setup & Configuration
2. Phase 9.1 - Advanced Preprocessing
3. Phase 9.2 - Enhanced EDA (move from before 9.1)
4. Phase 9.3 - Advanced Feature Engineering (ADD NEW)
5. Phase 9.4 - Classification
6. Phase 9.5 - Regression
7. Phase 9.6 - Model Evaluation (ADD NEW)
8. Phase 9.7 - Valuation & Stock Identification (ADD NEW)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_ROOT / "ml_finance_model_main.ipynb"


def parse_notebook(path: Path) -> Dict:
    """Parse notebook JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_notebook(nb_data: Dict, path: Path):
    """Save notebook JSON with proper formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)


def get_cell_text(cell: Dict) -> str:
    """Extract text from a notebook cell."""
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def create_markdown_cell(content: str) -> Dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split("\n") if "\n" in content else [content],
    }


def create_code_cell(content: str) -> Dict:
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": content.split("\n") if "\n" in content else [content],
    }


def find_phase_boundaries(cells: List[Dict]) -> Dict[str, Tuple[int, int]]:
    """
    Find start and end indices for each phase section.
    Returns dict mapping phase name to (start_idx, end_idx).
    """
    phase_patterns = {
        "setup": (0, None),  # Will be updated
        "9.1": (None, None),
        "9.2": (None, None),
        "9.4": (None, None),
        "9.5": (None, None),
    }

    phase_starts = {}

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            text = get_cell_text(cell)

            # Look for phase markers
            if re.search(r"##\s*Phase\s*9\.1", text, re.IGNORECASE):
                phase_starts["9.1"] = idx
            elif re.search(r"##\s*Phase\s*9\.2", text, re.IGNORECASE):
                phase_starts["9.2"] = idx
            elif re.search(r"##\s*Phase\s*9\.4", text, re.IGNORECASE):
                phase_starts["9.4"] = idx
            elif re.search(r"##\s*Phase\s*9\.5", text, re.IGNORECASE):
                phase_starts["9.5"] = idx

    # Calculate boundaries
    boundaries = {}
    sorted_phases = sorted(phase_starts.items(), key=lambda x: x[1])

    # Setup section: 0 to first phase
    if sorted_phases:
        boundaries["setup"] = (0, sorted_phases[0][1])

    # Each phase: start to next phase start
    for i, (phase, start) in enumerate(sorted_phases):
        if i < len(sorted_phases) - 1:
            end = sorted_phases[i + 1][1]
        else:
            end = len(cells)
        boundaries[phase] = (start, end)

    return boundaries


def create_phase_9_3_cells() -> List[Dict]:
    """Create Phase 9.3 (Feature Engineering) cells."""
    markdown = create_markdown_cell(
        "## Phase 9.3 — Advanced Feature Engineering\n"
        "\n"
        "Comprehensive feature engineering using Phase 9.3 functions:\n"
        "1. Valuation ratios (P/E, P/B, P/S, EV/EBITDA, etc.)\n"
        "2. Profitability ratios (ROE, ROA, ROIC, margins)\n"
        "3. Leverage ratios (Debt/Equity, Net Debt/EBITDA)\n"
        "4. Liquidity and efficiency ratios\n"
        "5. Growth metrics (revenue CAGR, earnings growth)\n"
        "6. Sector-specific features\n"
        "7. Feature interactions and relative value features"
    )

    code = create_code_cell(
        'print("=" * 80)\n'
        'print("PHASE 9.3 — ADVANCED FEATURE ENGINEERING")\n'
        'print("=" * 80)\n'
        "\n"
        'print("\\n🔧 Building Comprehensive Features...")\n'
        "all_stocks_featured = build_comprehensive_features(\n"
        "    all_stocks_processed,\n"
        "    include_interactions=True,\n"
        "    include_relative_values=True,\n"
        "    sector_col='sector'\n"
        ")\n"
        "\n"
        'print(f"\\n✓ Feature engineering complete")\n'
        'print(f"  Original features: {all_stocks_processed.shape[1]}")\n'
        'print(f"  Engineered features: {all_stocks_featured.shape[1]}")\n'
        'print(f"  New features added: {all_stocks_featured.shape[1] - all_stocks_processed.shape[1]}")\n'
        "\n"
        "# Display sample of new features\n"
        "new_cols = [c for c in all_stocks_featured.columns if c not in all_stocks_processed.columns]\n"
        "if new_cols:\n"
        '    print(f"\\n📊 Sample of {min(10, len(new_cols))} new features:")\n'
        "    for col in new_cols[:10]:\n"
        '        print(f"  - {col}")\n'
        "\n"
        "# Calculate feature importance\n"
        "if 'price_target' in all_stocks_featured.columns:\n"
        '    print("\\n📈 Calculating Feature Importance...")\n'
        "    feature_cols = [c for c in all_stocks_featured.columns \n"
        "                   if c not in ['price_target', 'ticker', 'index'] \n"
        "                   and all_stocks_featured[c].dtype in ['float64', 'int64']]\n"
        "    \n"
        "    X_importance = all_stocks_featured[feature_cols].fillna(0)\n"
        "    y_importance = all_stocks_featured['price_target'].fillna(all_stocks_featured['last_price'])\n"
        "    \n"
        "    importance_scores = calculate_feature_importance_rf(X_importance, y_importance, top_k=20)\n"
        "    \n"
        '    print(f"\\n🔝 Top 20 Most Important Features:")\n'
        "    for i, (feature, score) in enumerate(importance_scores.items(), 1):\n"
        '        print(f"  {i:2d}. {feature:<40s}: {score:.4f}")'
    )

    return [markdown, code]


def create_phase_9_6_cells() -> List[Dict]:
    """Create Phase 9.6 (Model Evaluation) cells."""
    markdown = create_markdown_cell(
        "## Phase 9.6 — Model Evaluation and Error Analysis\n"
        "\n"
        "Comprehensive evaluation of regression regression:\n"
        "1. Comprehensive regression metrics (MAE, RMSE, MAPE, R², Median AE, Max Error)\n"
        "2. Metrics by segment (sector, region, market cap, volatility)\n"
        "3. Residual analysis (normality tests, Q-Q plots, histograms)\n"
        "4. Error bucketing analysis\n"
        "5. Cross-validation strategies"
    )

    code = create_code_cell(
        'print("=" * 80)\n'
        'print("PHASE 9.6 — MODEL EVALUATION AND ERROR ANALYSIS")\n'
        'print("=" * 80)\n'
        "\n"
        "if 'predicted_price_target' in all_stocks_featured.columns:\n"
        "    y_true = all_stocks_featured['price_target'].fillna(all_stocks_featured['last_price'])\n"
        "    y_pred = all_stocks_featured['predicted_price_target']\n"
        "    \n"
        "    # 1. Comprehensive regression metrics\n"
        '    print("\\n📊 Comprehensive Regression Metrics:")\n'
        "    metrics = comprehensive_regression_metrics(y_true, y_pred)\n"
        "    for metric, value in metrics.items():\n"
        '        print(f"  {metric}: {value:.4f}")\n'
        "    \n"
        "    # 2. Metrics by segment\n"
        "    if 'sector' in all_stocks_featured.columns:\n"
        '        print("\\n📈 Metrics by Sector:")\n'
        "        sector_metrics = compute_metrics_by_segment(\n"
        "            all_stocks_featured.assign(y_true=y_true, y_pred=y_pred),\n"
        "            'y_true', 'y_pred', 'sector'\n"
        "        )\n"
        "        for sector, metrics in list(sector_metrics.items())[:5]:\n"
        '            print(f"\\n  {sector}:")\n'
        "            print(f\"    MAE: {metrics['MAE']:.2f}\")\n"
        "            print(f\"    RMSE: {metrics['RMSE']:.2f}\")\n"
        "            print(f\"    R²: {metrics['R2']:.4f}\")\n"
        "    \n"
        "    # 3. Residual analysis\n"
        '    print("\\n🔍 Residual Analysis:")\n'
        "    residuals = residual_analysis_suite(y_true, y_pred, output_dir=config.output_dir)\n"
        "    print(f\"  Mean residual: {residuals['mean_residual']:.4f}\")\n"
        "    print(f\"  Std residual: {residuals['std_residual']:.4f}\")\n"
        "else:\n"
        '    print("\\n⚠ No predictions available. Run Phase 9.5 first.")'
    )

    return [markdown, code]


def create_phase_9_7_cells() -> List[Dict]:
    """Create Phase 9.7 (Valuation) cells."""
    markdown = create_markdown_cell(
        "## Phase 9.7 — Identification of Under/Overvalued Stocks with Visualization\n"
        "\n"
        "Comprehensive stock valuation and ranking:\n"
        "1. Valuation categories (Strong Buy/Buy/Hold/Sell/Strong Sell)\n"
        "2. Sector z-scores for relative valuation\n"
        "3. Multi-factor scoring (valuation, quality, growth)\n"
        "4. Interactive visualizations\n"
        "5. Stock rankings and exports"
    )

    code = create_code_cell(
        'print("=" * 80)\n'
        'print("PHASE 9.7 — VALUATION AND STOCK IDENTIFICATION")\n'
        'print("=" * 80)\n'
        "\n"
        "if 'predicted_price_target' in all_stocks_featured.columns:\n"
        "    # Calculate mispricing scores\n"
        '    print("\\n💰 Calculating Mispricing Scores...")\n'
        "    all_stocks_valued = calculate_mispricing_score(all_stocks_featured)\n"
        "    \n"
        "    # Assign valuation categories\n"
        '    print("\\n📊 Assigning Valuation Categories...")\n'
        "    categories = assign_valuation_category(all_stocks_valued['mispricing_score'])\n"
        "    all_stocks_valued['valuation_category'] = categories\n"
        "    \n"
        "    # Display distribution\n"
        '    print("\\n📈 Valuation Category Distribution:")\n'
        "    category_counts = all_stocks_valued['valuation_category'].value_counts()\n"
        "    for category, count in category_counts.items():\n"
        "        pct = (count / len(all_stocks_valued)) * 100\n"
        '        print(f"  {category}: {count:,} stocks ({pct:.1f}%)")\n'
        "    \n"
        "    # Rankings\n"
        '    print("\\n🏆 Top 10 Undervalued Stocks:")\n'
        "    top_undervalued = rank_undervalued_stocks(all_stocks_valued, top_n=10)\n"
        "    for i, row in top_undervalued.iterrows():\n"
        "        ticker = row.get('ticker', 'N/A')\n"
        "        sector = row.get('sector', 'N/A')\n"
        "        mispricing = row.get('mispricing_score', 0)\n"
        '        print(f"  {ticker:<10s} | {sector:<25s} | {mispricing:>6.1f}%")\n'
        "    \n"
        "    # Visualizations\n"
        '    print("\\n📊 Creating Interactive Visualizations...")\n'
        "    scatter_path = config.output_dir / 'valuation_scatter_plot.html'\n"
        "    create_valuation_scatter_plot(all_stocks_valued, out_path=scatter_path, color_by='sector')\n"
        '    print(f"  ✓ Scatter plot: {scatter_path}")\n'
        "    \n"
        "    # Export to Excel\n"
        "    excel_path = config.output_dir / 'stock_valuation_analysis.xlsx'\n"
        "    export_predictions_to_excel(all_stocks_valued, excel_path, include_summary=True)\n"
        '    print(f"  ✓ Excel report: {excel_path}")\n'
        "    \n"
        '    print("\\n✓ PHASE 9 COMPLETE — END-TO-END ML ANALYTICS PLATFORM")\n'
        '    print("📊 Business Objective Achieved: Stock Price Target Predictions with Valuation Analysis")\n'
        "else:\n"
        '    print("\\n⚠ No predictions available. Run Phase 9.5 first.")'
    )

    return [markdown, code]


def reorganize_notebook():
    """Main function to reorganize the notebook."""
    print("Loading notebook...")
    nb_data = parse_notebook(NOTEBOOK_PATH)
    cells = nb_data.get("cells", [])

    print(f"Original notebook: {len(cells)} cells")

    # Find phase boundaries
    boundaries = find_phase_boundaries(cells)
    print(f"\nFound phases: {list(boundaries.keys())}")
    for phase, (start, end) in boundaries.items():
        print(f"  {phase}: cells {start}-{end}")

    # Extract sections
    setup_cells = cells[boundaries["setup"][0] : boundaries["setup"][1]]
    phase_9_1_cells = cells[boundaries["9.1"][0] : boundaries["9.1"][1]]
    phase_9_2_cells = cells[boundaries["9.2"][0] : boundaries["9.2"][1]]
    phase_9_4_cells = cells[boundaries["9.4"][0] : boundaries["9.4"][1]]
    phase_9_5_cells = cells[boundaries["9.5"][0] : boundaries["9.5"][1]]

    # Create new phase cells
    phase_9_3_cells = create_phase_9_3_cells()
    phase_9_6_cells = create_phase_9_6_cells()
    phase_9_7_cells = create_phase_9_7_cells()

    # Assemble in correct order
    new_cells = []
    new_cells.extend(setup_cells)
    new_cells.extend(phase_9_1_cells)
    new_cells.extend(phase_9_2_cells)  # Moved after 9.1
    new_cells.extend(phase_9_3_cells)  # NEW
    new_cells.extend(phase_9_4_cells)
    new_cells.extend(phase_9_5_cells)
    new_cells.extend(phase_9_6_cells)  # NEW
    new_cells.extend(phase_9_7_cells)  # NEW

    print(f"\nReorganized notebook: {len(new_cells)} cells")
    print(f"  Setup: {len(setup_cells)} cells")
    print(f"  Phase 9.1: {len(phase_9_1_cells)} cells")
    print(f"  Phase 9.2: {len(phase_9_2_cells)} cells (moved after 9.1)")
    print(f"  Phase 9.3: {len(phase_9_3_cells)} cells (NEW)")
    print(f"  Phase 9.4: {len(phase_9_4_cells)} cells")
    print(f"  Phase 9.5: {len(phase_9_5_cells)} cells")
    print(f"  Phase 9.6: {len(phase_9_6_cells)} cells (NEW)")
    print(f"  Phase 9.7: {len(phase_9_7_cells)} cells (NEW)")

    # Update notebook
    nb_data["cells"] = new_cells

    # Save
    print(f"\nSaving reorganized notebook to {NOTEBOOK_PATH}...")
    save_notebook(nb_data, NOTEBOOK_PATH)
    print("✓ Notebook reorganized successfully!")

    return True


if __name__ == "__main__":
    try:
        success = reorganize_notebook()
        if success:
            print("\n" + "=" * 80)
            print("NOTEBOOK INTEGRATION COMPLETE")
            print("=" * 80)
            print("\nNext steps:")
            print("1. Run tests: python -m pytest tests/test_notebook_integration.py -v")
            print("2. Verify in Jupyter: jupyter notebook ml_finance_model_main.ipynb")
            print("3. Run full test suite: python -m unittest -v")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
