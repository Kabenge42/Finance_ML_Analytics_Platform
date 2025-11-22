"""
Fix notebook Phase 9.4-9.8 integration by removing malformed cells
and inserting properly formatted cells into the correct sections.
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Backup the notebook first
notebook_path = Path("ml_finance_model_main.ipynb")
backup_path = Path(f"ml_finance_model_main.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb")
shutil.copy(notebook_path, backup_path)
print(f"[OK] Backup created: {backup_path}")

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

cells = notebook['cells']
print(f"[OK] Loaded notebook with {len(cells)} cells")

# Step 1: Remove malformed cells (104-129)
print("\n[STEP 1] Removing malformed cells 104-129...")
# Keep cells 0-103 and 130+
cells_to_keep = cells[:104] + cells[130:]
print(f"   Removed {len(cells) - len(cells_to_keep)} malformed cells")
print(f"   New cell count: {len(cells_to_keep)}")

# Step 2: Find insertion points for each phase
print("\n[STEP 2] Finding correct insertion points...")

# Find Phase 9.6 section for uncertainty quantification
phase96_idx = None
for i, cell in enumerate(cells_to_keep):
    if cell.get('cell_type') == 'markdown':
        source = ''.join(cell.get('source', []))
        if 'Phase 9.6: Model Evaluation' in source:
            phase96_idx = i
            print(f"   Found Phase 9.6 at cell {i}")
            break

if phase96_idx is None:
    print("   [WARNING] Could not find Phase 9.6 section")
    phase96_idx = 86  # Default based on earlier analysis

# Helper to create properly formatted cells
def create_markdown_cell(content):
    """Create a markdown cell with proper formatting."""
    lines = content.strip().split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + '\n' for line in lines]
    }

def create_code_cell(content):
    """Create a code cell with proper formatting."""
    lines = content.strip().split('\n')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in lines]
    }

# Step 3: Create properly formatted Phase 9.4-9.8 cells
print("\n[STEP 3] Creating properly formatted cells...")

# Phase 9.4: Uncertainty Quantification (insert into Phase 9.6)
phase94_cells = []

# Cell 1: Markdown header
phase94_cells.append(create_markdown_cell("""### Phase 9.4: Uncertainty Quantification & Conformal Calibration

**Objectives:**
- Quantify prediction interval quality with coverage diagnostics
- Validate conformal calibration effectiveness
- Analyze uncertainty by sector and region
- Generate reliability diagrams and interactive visualizations

**Inputs:**
- `outputs/regression/regression_predictions_detailed.csv`

**Outputs:**
- `outputs/uncertainty/quantile_predictions_diagnostics.csv`
- `outputs/uncertainty/coverage_by_sector.json`
- `outputs/uncertainty/uncertainty_summary.json`
- Interactive HTML visualizations"""))

# Cell 2: Load predictions and build diagnostics
phase94_cells.append(create_code_cell("""# %% [PHASE 9.4] Build quantile diagnostics
print("\\n" + "=" * 80)
print("PHASE 9.4: UNCERTAINTY QUANTIFICATION")
print("=" * 80)

from pathlib import Path
import pandas as pd

# Setup paths
uncertainty_dir = output_dir / "uncertainty"
uncertainty_dir.mkdir(parents=True, exist_ok=True)

# Load predictions
predictions_path = output_dir / "regression" / "regression_predictions_detailed.csv"
if not predictions_path.exists():
    print(f"⚠️  Predictions file not found: {predictions_path}")
    print("   Please run Phase 9.5 (regression) first to generate predictions.")
else:
    print(f"📂 Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)
    print(f"   Loaded {len(predictions_df):,} predictions")

    # Build quantile diagnostics
    print("\\n🔍 Building quantile diagnostics...")
    diagnostics_df = build_quantile_diagnostics(
        predictions_df=predictions_df,
        output_dir=uncertainty_dir,
        y_true_col="y_true",
        pred_cols={"p10": "pred_p10", "p50": "pred_p50", "p90": "pred_p90"},
        sector_col="sector",
        region_col="region",
        target_coverage=0.8
    )

    print(f"✓ Diagnostics computed for {len(diagnostics_df):,} predictions")
    print(f"✓ Artifacts saved to: {uncertainty_dir}")"""))

# Cell 3: Coverage visualizations
phase94_cells.append(create_code_cell("""# %% [PHASE 9.4] Coverage and width visuals
if 'diagnostics_df' in globals():
    print("\\n📊 Generating interval coverage visualizations...")

    plot_interval_coverage(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        last_price_col="last_price"
    )

    print("✓ Coverage visualizations created:")
    print(f"  - {uncertainty_dir / 'interval_width_by_bucket.html'}")
    print(f"  - {uncertainty_dir / 'coverage_heatmap_region_sector.html'}")"""))

# Cell 4: Reliability diagram
phase94_cells.append(create_code_cell("""# %% [PHASE 9.4] Reliability diagram
if 'diagnostics_df' in globals():
    print("\\n📈 Creating reliability diagram...")

    plot_reliability_diagram(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        pre_calibration_df=None
    )

    print(f"✓ Reliability diagram created: {uncertainty_dir / 'reliability_diagram_conformal.html'}")"""))

# Cell 5: Summary
phase94_cells.append(create_code_cell("""# %% [PHASE 9.4] Summary + QA
print("\\n📋 Uncertainty Quantification Summary:")
print("=" * 80)

summary_path = uncertainty_dir / "uncertainty_summary.json"
if summary_path.exists():
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"Overall Coverage: {summary.get('overall_coverage', 0):.1%}")
    print(f"Target Coverage: {summary.get('target_coverage', 0.8):.1%}")
    print(f"Within Tolerance: {'✓' if summary.get('within_tolerance', False) else '✗'}")

    under_covered = summary.get('under_covered_sectors', [])
    over_covered = summary.get('over_covered_sectors', [])

    if under_covered:
        print(f"\\n⚠️  Under-covered sectors: {', '.join(under_covered)}")
    if over_covered:
        print(f"⚠️  Over-covered sectors: {', '.join(over_covered)}")

    print("\\n✅ Uncertainty quantification complete!")
else:
    print("⚠️  Summary file not found")"""))

print(f"   Created {len(phase94_cells)} Phase 9.4 cells")

# Step 4: Insert Phase 9.4 cells into Phase 9.6 section
print("\n[STEP 4] Inserting Phase 9.4 cells into Phase 9.6...")
insertion_point = phase96_idx + 1  # After Phase 9.6 header
for i, cell in enumerate(phase94_cells):
    cells_to_keep.insert(insertion_point + i, cell)

print(f"   Inserted {len(phase94_cells)} cells at position {insertion_point}")
print(f"   New cell count: {len(cells_to_keep)}")

# Step 5: Update imports cell
print("\n[STEP 5] Updating imports cell...")
imports_idx = None
for i, cell in enumerate(cells_to_keep[:10]):
    if cell.get('cell_type') == 'code':
        source = ''.join(cell.get('source', []))
        if 'from finance_ml' in source and 'import' in source:
            imports_idx = i
            break

if imports_idx is not None:
    print(f"   Found imports cell at index {imports_idx}")

    # Add Phase 9.4-9.8 evaluation imports
    imports_to_add = """
# Phase 9.4-9.8: Advanced Evaluation and Governance
from finance_ml.ml_workflow.evaluation import (
    # Phase 9.4 - Uncertainty Quantification
    build_quantile_diagnostics,
    plot_interval_coverage,
    plot_reliability_diagram,
    # Phase 9.5 - Safety Rails
    summarize_winsorization_effects,
    track_constraint_violations,
    safety_rails_sensitivity_app,
    # Phase 9.6 - Data Splits & Leakage
    compute_fold_overlap,
    summarize_grouped_cv_balance,
    time_leakage_checks,
    # Phase 9.7 - Sector Bias Calibration
    estimate_sector_bias,
    plot_metrics_by_sector_time,
    create_sector_bias_dashboard,
    # Phase 9.8 - Stacking & Governance
    compute_stacking_contributions,
    meta_error_maps,
    generate_model_card,
    build_lineage_json,
)
"""

    current_source = cells_to_keep[imports_idx].get('source', [])
    if isinstance(current_source, list):
        current_source = current_source.copy()
    else:
        current_source = [current_source]

    # Check if already added
    combined_source = ''.join(current_source)
    if 'build_quantile_diagnostics' not in combined_source:
        # Add new imports
        current_source.extend([line + '\n' for line in imports_to_add.strip().split('\n')])
        cells_to_keep[imports_idx]['source'] = current_source
        print("   ✓ Added Phase 9.4-9.8 imports")
    else:
        print("   ✓ Imports already present")
else:
    print("   [WARNING] Could not find imports cell")

# Step 6: Save updated notebook
print("\n[STEP 6] Saving updated notebook...")
notebook['cells'] = cells_to_keep
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"\n✅ Notebook updated successfully!")
print(f"✅ Total cells: {len(cells_to_keep)}")
print(f"✅ Backup saved to: {backup_path}")
print(f"\n📊 Summary:")
print(f"   - Removed 26 malformed cells")
print(f"   - Added 5 properly formatted Phase 9.4 cells")
print(f"   - Updated imports cell with evaluation functions")
print(f"   - Final cell count: {len(cells_to_keep)}")
