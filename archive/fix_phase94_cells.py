#!/usr/bin/env python3
"""
Fix malformed Phase 9.4-6 cells in ml_finance_model_main.ipynb.
Extracts cells from end, fixes formatting, and reorganizes into proper sections.
"""
import json
import re
from pathlib import Path
from datetime import datetime

def create_backup(notebook_path):
    """Create timestamped backup of notebook."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = notebook_path.parent / f"{notebook_path.stem}.backup.{timestamp}{notebook_path.suffix}"
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Backup created: {backup_path}")
    return backup_path

def fix_code_formatting(code_text):
    """Fix malformed code by adding proper line breaks and spacing."""
    
    # Fix common patterns
    patterns = [
        (r'print\(', '\nprint('),  # Add newline before print
        (r'\)print\(', ')\nprint('),  # Add newline between prints
        (r'import ', '\nimport '),  # Add newline before imports
        (r'\)import ', ')\nimport '),  # Add newline after statement before import
        (r'from ', '\nfrom '),  # Add newline before from
        (r'\)from ', ')\nfrom '),  # Add newline after statement before from
        (r'if ', '\nif '),  # Add newline before if
        (r'\)if ', ')\nif '),  # Add newline after statement before if
        (r'else:', '\nelse:'),  # Add newline before else
        (r'\)else:', ')\nelse:'),  # Add newline after statement before else
        (r'# ', '\n# '),  # Add newline before comments
        (r'\)# ', ')\n# '),  # Add newline after statement before comment
        (r'path =', '\npath ='),  # Variable assignments
        (r'dir =', '\ndir ='),
        (r'_df =', '\n_df ='),
        (r'_path =', '\n_path ='),
    ]
    
    fixed = code_text
    for pattern, replacement in patterns:
        fixed = re.sub(pattern, replacement, fixed)
    
    # Clean up excessive newlines
    while '\n\n\n' in fixed:
        fixed = fixed.replace('\n\n\n', '\n\n')
    
    # Remove leading newlines
    fixed = fixed.lstrip('\n')
    
    return fixed

def create_phase94_cells():
    """Create properly formatted Phase 9.4 cells following NOTEBOOK_INTEGRATION_GUIDE.md."""
    
    cells = []
    
    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 9.4: Uncertainty Quantification & Conformal Calibration\n",
            "\n",
            "**Objectives:**\n",
            "\n",
            "- Quantify prediction interval quality with coverage diagnostics\n",
            "- Validate conformal calibration effectiveness\n",
            "- Analyze uncertainty by sector and region\n",
            "- Generate reliability diagrams and interactive visualizations\n",
            "\n",
            "**Inputs:**\n",
            "\n",
            "- `outputs/regression/regression_predictions_detailed.csv` (standardized predictions schema)\n",
            "\n",
            "**Outputs:**\n",
            "\n",
            "- `outputs/uncertainty/quantile_predictions_diagnostics.csv`\n",
            "- `outputs/uncertainty/coverage_by_sector.json`\n",
            "- `outputs/uncertainty/uncertainty_summary.json`\n",
            "- 4 interactive HTML visualizations"
        ]
    })
    
    # Cell 2: Build Quantile Diagnostics
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.4] Build quantile diagnostics\n",
            "print(\"\\n\" + \"=\" * 80)\n",
            "print(\"PHASE 9.4: UNCERTAINTY QUANTIFICATION\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "from pathlib import Path\n",
            "import pandas as pd\n",
            "\n",
            "# Setup paths\n",
            "output_dir = Path(\"outputs\")\n",
            "uncertainty_dir = output_dir / \"uncertainty\"\n",
            "uncertainty_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Load predictions\n",
            "predictions_path = output_dir / \"regression\" / \"regression_predictions_detailed.csv\"\n",
            "if not predictions_path.exists():\n",
            "    print(f\"⚠️  Predictions file not found: {predictions_path}\")\n",
            "    print(\"   Please run Phase 9.5 (regression) first to generate predictions.\")\n",
            "else:\n",
            "    print(f\"📂 Loading predictions from: {predictions_path}\")\n",
            "    predictions_df = pd.read_csv(predictions_path)\n",
            "    print(f\"   Loaded {len(predictions_df):,} predictions\")\n",
            "\n",
            "    # Build quantile diagnostics\n",
            "    print(\"\\n🔍 Building quantile diagnostics...\")\n",
            "    diagnostics_df = build_quantile_diagnostics(\n",
            "        predictions_df=predictions_df,\n",
            "        output_dir=uncertainty_dir,\n",
            "        y_true_col=\"y_true\",\n",
            "        pred_cols={\"p10\": \"pred_p10\", \"p50\": \"pred_p50\", \"p90\": \"pred_p90\"},\n",
            "        sector_col=\"sector\",\n",
            "        region_col=\"region\",\n",
            "        target_coverage=0.8\n",
            "    )\n",
            "\n",
            "    print(f\"✓ Diagnostics computed for {len(diagnostics_df):,} predictions\")\n",
            "    print(f\"✓ Artifacts saved to: {uncertainty_dir}\")"
        ]
    })
    
    # Cell 3: Coverage and Width Visuals
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.4] Coverage and width visuals\n",
            "print(\"\\n📊 Generating interval coverage visualizations...\")\n",
            "\n",
            "plot_interval_coverage(\n",
            "    diagnostics_df=diagnostics_df,\n",
            "    output_dir=uncertainty_dir,\n",
            "    last_price_col=\"last_price\"\n",
            ")\n",
            "\n",
            "print(\"✓ Coverage visualizations created:\")\n",
            "print(f\"  - {uncertainty_dir / 'interval_width_by_bucket.html'}\")\n",
            "print(f\"  - {uncertainty_dir / 'coverage_heatmap_region_sector.html'}\")"
        ]
    })
    
    # Cell 4: Reliability Diagram
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.4] Reliability diagram for conformal calibration\n",
            "print(\"\\n📈 Creating reliability diagram...\")\n",
            "\n",
            "plot_reliability_diagram(\n",
            "    diagnostics_df=diagnostics_df,\n",
            "    output_dir=uncertainty_dir,\n",
            "    pre_calibration_df=None  # Set to pre-calibration df if available\n",
            ")\n",
            "\n",
            "print(f\"✓ Reliability diagram created: {uncertainty_dir / 'reliability_diagram_conformal.html'}\")"
        ]
    })
    
    # Cell 5: Summary and QA
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.4] Summary + QA\n",
            "print(\"\\n📋 Uncertainty Quantification Summary:\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "# Load summary\n",
            "import json\n",
            "\n",
            "summary_path = uncertainty_dir / \"uncertainty_summary.json\"\n",
            "if summary_path.exists():\n",
            "    with open(summary_path, 'r') as f:\n",
            "        summary = json.load(f)\n",
            "\n",
            "    print(f\"Overall Coverage: {summary.get('overall_coverage', 0):.1%}\")\n",
            "    print(f\"Target Coverage: {summary.get('target_coverage', 0.8):.1%}\")\n",
            "    print(f\"Within Tolerance: {'✓' if summary.get('within_tolerance', False) else '✗'}\")\n",
            "\n",
            "    under_covered = summary.get('under_covered_sectors', [])\n",
            "    over_covered = summary.get('over_covered_sectors', [])\n",
            "\n",
            "    if under_covered:\n",
            "        print(f\"\\n⚠️  Under-covered sectors: {', '.join(under_covered)}\")\n",
            "    if over_covered:\n",
            "        print(f\"⚠️  Over-covered sectors: {', '.join(over_covered)}\")\n",
            "\n",
            "    print(\"\\n✅ Uncertainty quantification complete!\")\n",
            "else:\n",
            "    print(\"⚠️  Summary file not found\")"
        ]
    })
    
    return cells

def create_phase95_cells():
    """Create properly formatted Phase 9.5 cells."""
    
    cells = []
    
    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 9.5: Outlier Safety Rails & Non-Negative Constraints\n",
            "\n",
            "**Objectives:**\n",
            "\n",
            "- Track winsorization effects on feature distributions\n",
            "- Validate non-negativity constraint adherence\n",
            "- Analyze safety rails sensitivity across thresholds\n",
            "- Generate interactive safety dashboards\n",
            "\n",
            "**Inputs:**\n",
            "\n",
            "- Raw and winsorized feature dataframes\n",
            "- Predictions dataframe\n",
            "\n",
            "**Outputs:**\n",
            "\n",
            "- `outputs/safety_rails/clipping_effect_summary.json`\n",
            "- `outputs/safety_rails/non_negative_violations.json`\n",
            "- `outputs/safety_rails/safety_rails_summary.json`\n",
            "- 3 interactive HTML visualizations"
        ]
    })
    
    # Cell 2: Winsorization Effects
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.5] Winsorization effects\n",
            "print(\"\\n\" + \"=\" * 80)\n",
            "print(\"PHASE 9.5: SAFETY RAILS & NON-NEGATIVE CONSTRAINTS\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "safety_rails_dir = output_dir / \"safety_rails\"\n",
            "safety_rails_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Note: This requires access to raw and winsorized dataframes from earlier phases\n",
            "# If not available, skip this cell\n",
            "if 'all_stocks_raw' in dir() and 'all_stocks_winsorized' in dir():\n",
            "    print(\"\\n🔍 Analyzing winsorization effects...\")\n",
            "\n",
            "    # Get numeric columns\n",
            "    numeric_cols = all_stocks_winsorized.select_dtypes(include=[np.number]).columns.tolist()\n",
            "\n",
            "    summary_dict = summarize_winsorization_effects(\n",
            "        df_raw=all_stocks_raw,\n",
            "        df_winsorized=all_stocks_winsorized,\n",
            "        numeric_cols=numeric_cols[:20],  # Limit to first 20 for demo\n",
            "        output_dir=safety_rails_dir,\n",
            "        sector_col=\"sector\"\n",
            "    )\n",
            "\n",
            "    print(f\"✓ Winsorization summary created for {len(numeric_cols[:20])} features\")\n",
            "    print(f\"✓ Artifacts saved to: {safety_rails_dir}\")\n",
            "else:\n",
            "    print(\"⚠️  Raw/winsorized dataframes not available. Skipping winsorization analysis.\")"
        ]
    })
    
    # Cell 3: Constraint Violations
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.5] Track constraint violations\n",
            "print(\"\\n🛡️  Checking non-negativity constraint violations...\")\n",
            "\n",
            "if predictions_path.exists():\n",
            "    violations_dict = track_constraint_violations(\n",
            "        predictions_df=predictions_df,\n",
            "        output_dir=safety_rails_dir,\n",
            "        pred_col=\"y_pred\",\n",
            "        sector_col=\"sector\"\n",
            "    )\n",
            "\n",
            "    total_violations = violations_dict.get(\"total_violations\", 0)\n",
            "    violation_rate = violations_dict.get(\"violation_rate\", 0)\n",
            "\n",
            "    print(f\"Total Violations: {total_violations}\")\n",
            "    print(f\"Violation Rate: {violation_rate:.2%}\")\n",
            "\n",
            "    if total_violations == 0:\n",
            "        print(\"✅ Non-negativity constraint satisfied!\")\n",
            "    else:\n",
            "        print(f\"⚠️  Found {total_violations} violations\")\n",
            "        violations_by_sector = violations_dict.get(\"violations_by_sector\", {})\n",
            "        for sector, count in violations_by_sector.items():\n",
            "            if count > 0:\n",
            "                print(f\"   - {sector}: {count} violations\")"
        ]
    })
    
    # Cell 4: Safety Rails Sensitivity
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.5] Interactive robustness sliders\n",
            "if 'all_stocks_raw' in dir():\n",
            "    print(\"\\n📊 Creating safety rails sensitivity dashboard...\")\n",
            "\n",
            "    safety_rails_sensitivity_app(\n",
            "        df_raw=all_stocks_raw,\n",
            "        output_dir=safety_rails_dir,\n",
            "        thresholds=[0.01, 0.05, 0.1]\n",
            "    )\n",
            "\n",
            "    print(f\"✓ Sensitivity dashboard created: {safety_rails_dir / 'safety_rails_sensitivity_dashboard.html'}\")"
        ]
    })
    
    # Cell 5: Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.5] Summary + QA\n",
            "print(\"\\n📋 Safety Rails Summary:\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "summary_path = safety_rails_dir / \"safety_rails_summary.json\"\n",
            "if summary_path.exists():\n",
            "    with open(summary_path, 'r') as f:\n",
            "        summary = json.load(f)\n",
            "\n",
            "    print(f\"Winsorization Features: {summary.get('winsorization', {}).get('n_features', 0)}\")\n",
            "    print(f\"Constraint Violations: {summary.get('violations', {}).get('total_violations', 0)}\")\n",
            "\n",
            "    print(\"\\n✅ Safety rails monitoring complete!\")"
        ]
    })
    
    return cells

def create_phase96_cells():
    """Create properly formatted Phase 9.6 cells."""
    
    cells = []
    
    # Cell 1: Markdown Header
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Section 9.6: Data Split and Leakage Policy Validation\n",
            "\n",
            "**Objectives:**\n",
            "\n",
            "- Validate CV fold construction and grouping rules\n",
            "- Check for ticker/sector overlaps across folds\n",
            "- Detect time-based leakage violations\n",
            "- Ensure stratification balance\n",
            "\n",
            "**Inputs:**\n",
            "\n",
            "- Fold assignments dictionary (from CV training)\n",
            "- Training dataframe with snapshot dates\n",
            "\n",
            "**Outputs:**\n",
            "\n",
            "- `outputs/splits/fold_overlap_heatmap.html`\n",
            "- `outputs/splits/grouped_cv_balance_metrics.json`\n",
            "- `outputs/splits/leakage_report.json`"
        ]
    })
    
    # Cell 2: Fold Overlap
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.6] Fold overlap analysis\n",
            "print(\"\\n\" + \"=\" * 80)\n",
            "print(\"PHASE 9.6: DATA SPLIT AND LEAKAGE POLICY VALIDATION\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "splits_dir = output_dir / \"splits\"\n",
            "splits_dir.mkdir(parents=True, exist_ok=True)\n",
            "\n",
            "# Note: This requires fold_assignments from CV training\n",
            "# Example: fold_assignments = {0: [idx_list], 1: [idx_list], ...}\n",
            "if 'fold_assignments' in dir():\n",
            "    print(\"\\n🔍 Computing fold overlap...\")\n",
            "\n",
            "    overlap_dict = compute_fold_overlap(\n",
            "        fold_assignments=fold_assignments,\n",
            "        output_dir=splits_dir,\n",
            "        group_col=\"ticker\"\n",
            "    )\n",
            "\n",
            "    print(f\"✓ Fold overlap analysis complete\")\n",
            "    print(f\"  Zero overlap validated: {overlap_dict.get('zero_overlap_validated', False)}\")\n",
            "else:\n",
            "    print(\"⚠️  fold_assignments not available. Skipping overlap analysis.\")"
        ]
    })
    
    # Cell 3: CV Balance
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.6] CV balance metrics\n",
            "if 'fold_assignments' in dir() and 'all_stocks_features' in dir():\n",
            "    print(\"\\n📊 Summarizing grouped CV balance...\")\n",
            "\n",
            "    balance_dict = summarize_grouped_cv_balance(\n",
            "        df=all_stocks_features,\n",
            "        fold_assignments=fold_assignments,\n",
            "        output_dir=splits_dir,\n",
            "        stratify_cols=[\"sector\", \"region\"]\n",
            "    )\n",
            "\n",
            "    print(f\"✓ Balance metrics computed for {len(fold_assignments)} folds\")"
        ]
    })
    
    # Cell 4: Time Leakage
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.6] Time leakage checks\n",
            "if 'fold_assignments' in dir() and 'all_stocks_features' in dir() and 'snapshot_date' in all_stocks_features.columns:\n",
            "    print(\"\\n🕐 Checking time-based leakage...\")\n",
            "\n",
            "    leakage_report = time_leakage_checks(\n",
            "        df=all_stocks_features,\n",
            "        fold_assignments=fold_assignments,\n",
            "        output_dir=splits_dir,\n",
            "        date_col=\"snapshot_date\"\n",
            "    )\n",
            "\n",
            "    violations = leakage_report.get(\"violations\", 0)\n",
            "    print(f\"  Leakage violations: {violations}\")\n",
            "\n",
            "    if violations == 0:\n",
            "        print(\"✅ No time-based leakage detected!\")"
        ]
    })
    
    # Cell 5: Summary
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# %% [PHASE 9.6] Summary + QA\n",
            "print(\"\\n📋 Data Split Validation Summary:\")\n",
            "print(\"=\" * 80)\n",
            "\n",
            "leakage_path = splits_dir / \"leakage_report.json\"\n",
            "if leakage_path.exists():\n",
            "    with open(leakage_path, 'r') as f:\n",
            "        report = json.load(f)\n",
            "\n",
            "    print(f\"Violations: {report.get('violations', 0)}\")\n",
            "    print(f\"Severity: {report.get('severity', 'NONE')}\")\n",
            "\n",
            "    print(\"\\n✅ Data split validation complete!\")"
        ]
    })
    
    return cells

def fix_notebook(notebook_path):
    """Main function to fix the notebook."""
    
    # Create backup
    backup_path = create_backup(notebook_path)
    
    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells = nb.get('cells', [])
    print(f"\nOriginal notebook has {len(cells)} cells")
    
    # Find and remove malformed cells at the end
    # Typically these are the last 15-20 cells based on the example
    malformed_start_idx = None
    for idx in range(len(cells) - 1, max(0, len(cells) - 30), -1):
        cell = cells[idx]
        source = ''.join(cell.get('source', []))
        # Look for Phase 9.4 marker
        if 'PHASE 9.4' in source or '[PHASE 9.4]' in source:
            malformed_start_idx = idx
            break
    
    if malformed_start_idx:
        print(f"Found malformed cells starting at index {malformed_start_idx}")
        # Remove malformed cells
        cells = cells[:malformed_start_idx]
        print(f"Removed {len(nb['cells']) - len(cells)} malformed cells")
    
    # Create properly formatted cells
    phase94_cells = create_phase94_cells()
    phase95_cells = create_phase95_cells()
    phase96_cells = create_phase96_cells()
    
    print(f"\nCreated {len(phase94_cells)} Phase 9.4 cells")
    print(f"Created {len(phase95_cells)} Phase 9.5 cells")
    print(f"Created {len(phase96_cells)} Phase 9.6 cells")
    
    # Find insertion point (after Phase 9.3 or Classification)
    insertion_idx = len(cells)  # Default to end if not found
    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        # Look for Phase 9.7 or Analytics section
        if 'PHASE 9.7' in source or 'Phase 9.7' in source or 'Section 9.7' in source:
            insertion_idx = idx
            print(f"Found Phase 9.7 at index {idx}, will insert before it")
            break
    
    # Insert new cells
    all_new_cells = phase94_cells + phase95_cells + phase96_cells
    cells = cells[:insertion_idx] + all_new_cells + cells[insertion_idx:]
    
    # Update notebook
    nb['cells'] = cells
    
    # Save fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\n✅ Notebook fixed successfully!")
    print(f"   Total cells: {len(cells)}")
    print(f"   Phase 9.4: 5 cells (markdown + 4 code)")
    print(f"   Phase 9.5: 5 cells (markdown + 4 code)")
    print(f"   Phase 9.6: 5 cells (markdown + 4 code)")
    print(f"   Backup saved to: {backup_path}")

if __name__ == '__main__':
    notebook_path = Path('ml_finance_model_main.ipynb')
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found: {notebook_path}")
        exit(1)
    
    fix_notebook(notebook_path)
