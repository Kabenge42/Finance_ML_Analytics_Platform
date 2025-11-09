#!/usr/bin/env python3
"""
Integrate missing Phase 9 sections (9.5-9.8) into ml_finance_model_main_backup.ipynb
"""
import json
from pathlib import Path
from datetime import datetime

def create_phase_95_cells():
    """Create Phase 9.5 cells (Sector-Optimized Regression Models)"""
    cells = []

    # Phase 9.5 header
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Phase 9.5 — Sector-Optimized Regression Models\n",
                "\n",
                "Train sector-specific regression regression with:\n",
                "- Multiple algorithms: Ridge, XGBoost, LightGBM, CatBoost\n",
                "- Classification features integrated\n",
                "- Ensemble methods (Stacking, Voting)\n",
                "- Non-negative prediction constraints",
            ],
        }
    )

    # Phase 9.5 code cell
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Phase 9.5: Sector-Optimized Regression Models\n",
            "from finance_ml.advanced_models import (\n",
            "    prepare_regression_data,\n",
            "    train_sector_specific_models,\n",
            "    compare_regressors,\n",
            "    validate_training_data\n",
            ")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"PHASE 9.5: SECTOR-OPTIMIZED REGRESSION MODELS\")\n",
            "print(\"=\"*80)\n",
            "\n",
            "# Prepare regression data\n",
            "exclude_cols = ['ticker', 'sector', 'region', 'last_price']\n",
            "X_train_reg, X_test_reg, y_train_reg, y_test_reg = prepare_regression_data(\n",
            "    all_stocks_phase94,\n",
            "    target_col='price_target',\n",
            "    exclude_cols=exclude_cols,\n",
            "    test_size=0.2,\n",
            "    random_state=42\n",
            ")\n",
            "\n",
            "print(f\"✓ Regression data prepared: {X_train_reg.shape}\")\n",
            "\n",
            "# Validate training data\n",
            "validation_result = validate_training_data(X_train_reg, y_train_reg, strict=False)\n",
            "if validation_result['valid']:\n",
            "    print(\"✓ Training data validated\")\n",
            "else:\n",
            "    print(f\"⚠ Validation warnings: {validation_result['issues']}\")\n"
        ]
    })

    return cells

def create_phase_951_cells():
    """Create Phase 9.5.1 cells (Model Optimization)"""
    cells = []
    cells.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Phase 9.5.1 — Model Optimization Enhancements\n",
                "\n",
                "Optimize regression with:\n",
                "- Hyperparameter tuning\n",
                "- Ensemble stacking\n",
                "- Quantile regression",
            ],
        }
    )
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Train stacking ensemble\n",
            "from finance_ml.advanced_models import train_stacking_regressor\n",
            "\n",
            "stacking_model = train_stacking_regressor(\n",
            "    X_train_reg, y_train_reg, cv=5, random_state=42, ensure_nonnegative=True\n",
            ")\n",
            "y_pred_stacking = stacking_model.predict(X_test_reg)\n",
            "print(\"✓ Stacking ensemble trained\")\n"
        ]
    })
    return cells

def create_phase_96_cells():
    """Create Phase 9.6 cells (Evaluation)"""
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 9.6 — Model Evaluation and Error Analysis"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from finance_ml.eval import comprehensive_regression_metrics, compute_metrics_by_segment\n",
            "\n",
            "metrics = comprehensive_regression_metrics(y_test_reg, y_pred_stacking)\n",
            "print(\"Overall Performance:\")\n",
            "for k, v in metrics.items():\n",
            "    print(f\"  {k}: {v:.4f}\")\n"
        ]
    })
    return cells

def create_phase_97_cells():
    """Create Phase 9.7 cells (Valuation)"""
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 9.7 — Identification of Under/Overvalued Stocks"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from finance_ml.eval import calculate_mispricing_score, rank_undervalued_stocks\n",
            "\n",
            "all_stocks_valued = all_stocks_phase94.copy()\n",
            "all_stocks_valued['predicted_price_target'] = stacking_model.predict(\n",
            "    all_stocks_valued[X_train_reg.columns]\n",
            ")\n",
            "all_stocks_valued['mispricing_score'] = calculate_mispricing_score(\n",
            "    all_stocks_valued, 'predicted_price_target', 'last_price'\n",
            ")\n",
            "top_undervalued = rank_undervalued_stocks(all_stocks_valued, top_n=20)\n",
            "print(\"Top 20 Undervalued Stocks:\")\n",
            "print(top_undervalued[['ticker', 'sector', 'mispricing_score']].head(20))\n"
        ]
    })
    return cells

def create_phase_961_cells():
    """Create Phase 9.6.1 cells (Enhanced Error Analysis)"""
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### Phase 9.6.1 — Enhanced Error Analysis\n",
                   "\n",
                   "SHAP analysis and residual diagnostics"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from finance_ml.eval import create_shap_summary_plot, residual_analysis_suite\n",
            "\n",
            "# SHAP analysis\n",
            "OUTPUT_DIR.mkdir(exist_ok=True)\n",
            "shap_dir = OUTPUT_DIR / 'shap'\n",
            "shap_dir.mkdir(exist_ok=True)\n",
            "create_shap_summary_plot(\n",
            "    stacking_model, X_test_reg, \n",
            "    output_path=shap_dir / 'shap_summary.png',\n",
            "    model_type='tree', n_samples=100\n",
            ")\n",
            "print(\"✓ SHAP analysis complete\")\n",
            "\n",
            "# Residual analysis\n",
            "residual_analysis_suite(y_test_reg, y_pred_stacking, output_dir=OUTPUT_DIR / 'residuals')\n",
            "print(\"✓ Residual analysis complete\")\n"
        ]
    })
    return cells

def create_phase_98_cells():
    """Create Phase 9.8 cells (Comprehensive Analytics)"""
    cells = []
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Phase 9.8 — Comprehensive Analytics\n",
                   "\n",
                   "Generate reports:\n",
                   "- Excel report with predictions\n",
                   "- PDF valuation report\n",
                   "- Interactive dashboards"]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from finance_ml.eval import (\n",
            "    generate_prediction_analyst_excel_report,\n",
            "    generate_enhanced_pdf_report,\n",
            "    export_predictions_to_excel\n",
            ")\n",
            "\n",
            "reports_dir = OUTPUT_DIR / 'reports'\n",
            "reports_dir.mkdir(exist_ok=True)\n",
            "\n",
            "# Excel report\n",
            "generate_prediction_analyst_excel_report(\n",
            "    all_stocks_valued,\n",
            "    excel_path=reports_dir / 'prediction_analyst_comparison.xlsx',\n",
            "    top_n_opportunities=50\n",
            ")\n",
            "print(\"✓ Excel report generated\")\n",
            "\n",
            "# PDF report\n",
            "generate_enhanced_pdf_report(\n",
            "    all_stocks_valued,\n",
            "    pdf_path=reports_dir / 'stock_valuation_report.pdf',\n",
            "    title='Stock Price Target Analysis - Comprehensive Report',\n",
            "    include_financial_dashboard=True,\n",
            "    include_quality_alerts=True,\n",
            "    include_hypothesis_tests=False,\n",
            "    include_charts=False\n",
            ")\n",
            "print(\"✓ PDF report generated\")\n",
            "\n",
            "print(\"\\n\" + \"=\"*80)\n",
            "print(\"PHASE 9 COMPLETE - ALL ANALYSES FINISHED\")\n",
            "print(\"=\"*80)\n"
        ]
    })
    return cells

def integrate_phases_into_notebook(notebook_path, output_path=None):
    """Integrate missing phases into notebook"""
    print(f"Loading notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    print(f"Original cell count: {len(nb['cells'])}")
    
    # Create backup
    backup_path = notebook_path.parent / f"{notebook_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb"
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"Backup created: {backup_path}")
    
    # Generate all phase cells
    print("\nGenerating phase cells...")
    all_new_cells = []
    all_new_cells.extend(create_phase_95_cells())
    print(f"  Phase 9.5: {len([c for c in create_phase_95_cells() if c['cell_type'] == 'code'])} code cells")
    all_new_cells.extend(create_phase_951_cells())
    print(f"  Phase 9.5.1: {len([c for c in create_phase_951_cells() if c['cell_type'] == 'code'])} code cells")
    all_new_cells.extend(create_phase_96_cells())
    print(f"  Phase 9.6: {len([c for c in create_phase_96_cells() if c['cell_type'] == 'code'])} code cells")
    all_new_cells.extend(create_phase_961_cells())
    print(f"  Phase 9.6.1: {len([c for c in create_phase_961_cells() if c['cell_type'] == 'code'])} code cells")
    all_new_cells.extend(create_phase_97_cells())
    print(f"  Phase 9.7: {len([c for c in create_phase_97_cells() if c['cell_type'] == 'code'])} code cells")
    all_new_cells.extend(create_phase_98_cells())
    print(f"  Phase 9.8: {len([c for c in create_phase_98_cells() if c['cell_type'] == 'code'])} code cells")
    
    # Append to notebook
    nb['cells'].extend(all_new_cells)
    
    # Save
    output = output_path or notebook_path
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✓ Integrated all phases. New cell count: {len(nb['cells'])}")
    print(f"✓ Saved to: {output}")
    
    return len(all_new_cells)

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main_backup.ipynb')
    integrate_phases_into_notebook(notebook_path)
