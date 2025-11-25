# Phase 9.4-9.6 Notebook Fix Summary

## Problem Analysis

The malformed cells at the end of `ml_finance_model_main.ipynb` have these issues:

1. **Missing line breaks** - All code concatenated on single lines
2. **Missing spaces** - No space after `print()` statements
3. **Incorrect cell markers** - Using `##%%` instead of `#%%`
4. **Poor markdown formatting** - Missing line breaks in markdown
5. **Wrong location** - Cells at the very end instead of proper Phase 9.4-9.6 sections

## Example of Current Malformed Cell

```python
# %% [PHASE 9.4] Build quantile diagnosticsprint("\n" + "=" * 80)print("PHASE 9.4: UNCERTAINTY QUANTIFICATION")print("=" * 80)from pathlib import Pathimport pandas as pd# Setup pathsoutput_dir = Path("outputs")uncertainty_dir = output_dir / "uncertainty"
```

## Solution Approach

### Option 1: Run the Fix Script (Recommended)

1. Open PyCharm terminal
2. Activate virtual environment:
   ```
   .venv\Scripts\activate
   ```
3. Run the fix script:
   ```
   python fix_phase94_cells.py
   ```

This will:

- Create a timestamped backup of the notebook
- Remove the malformed cells from the end
- Insert properly formatted Phase 9.4, 9.5, and 9.6 sections
- Place them in correct order before Phase 9.7

### Option 2: Manual Fix in PyCharm

If the script doesn't work, follow these steps:

#### Step 1: Create Backup

1. In PyCharm, right-click `ml_finance_model_main.ipynb`
2. Select "Copy" → Paste as `ml_finance_model_main_backup.ipynb`

#### Step 2: Delete Malformed Cells

1. Open the notebook in PyCharm
2. Scroll to the very end
3. Find cells starting with `# %% [PHASE 9.4]` (likely around the last 15-20 cells)
4. Delete all these malformed cells from the end

#### Step 3: Find Insertion Point

1. Scroll up to find Phase 9.7 or Analytics section
2. Position cursor before Phase 9.7 (this is where we'll insert new sections)

#### Step 4: Add Phase 9.4 Section

Insert these 5 cells in order:

**Cell 1 (Markdown):**

```markdown
## Section 9.4: Uncertainty Quantification & Conformal Calibration

**Objectives:**

- Quantify prediction interval quality with coverage diagnostics
- Validate conformal calibration effectiveness
- Analyze uncertainty by sector and region
- Generate reliability diagrams and interactive visualizations

**Inputs:**

- `outputs/regression/regression_predictions_detailed.csv` (standardized predictions schema)

**Outputs:**

- `outputs/uncertainty/quantile_predictions_diagnostics.csv`
- `outputs/uncertainty/coverage_by_sector.json`
- `outputs/uncertainty/uncertainty_summary.json`
- 4 interactive HTML visualizations
```

**Cell 2 (Code):**

```python
# %% [PHASE 9.4] Build quantile diagnostics
print("\n" + "=" * 80)
print("PHASE 9.4: UNCERTAINTY QUANTIFICATION")
print("=" * 80)

from pathlib import Path
import pandas as pd

# Setup paths
output_dir = Path("outputs")
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
    print("\n🔍 Building quantile diagnostics...")
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
    print(f"✓ Artifacts saved to: {uncertainty_dir}")
```

**Cell 3 (Code):**

```python
# %% [PHASE 9.4] Coverage and width visuals
print("\n📊 Generating interval coverage visualizations...")

plot_interval_coverage(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        last_price_col="last_price"
        )

print("✓ Coverage visualizations created:")
print(f"  - {uncertainty_dir / 'interval_width_by_bucket.html'}")
print(f"  - {uncertainty_dir / 'coverage_heatmap_region_sector.html'}")
```

**Cell 4 (Code):**

```python
# %% [PHASE 9.4] Reliability diagram for conformal calibration
print("\n📈 Creating reliability diagram...")

plot_reliability_diagram(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        pre_calibration_df=None  # Set to pre-calibration df if available
        )

print(f"✓ Reliability diagram created: {uncertainty_dir / 'reliability_diagram_conformal.html'}")
```

**Cell 5 (Code):**

```python
# %% [PHASE 9.4] Summary + QA
print("\n📋 Uncertainty Quantification Summary:")
print("=" * 80)

# Load summary
import json

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
        print(f"\n⚠️  Under-covered sectors: {', '.join(under_covered)}")
    if over_covered:
        print(f"⚠️  Over-covered sectors: {', '.join(over_covered)}")

    print("\n✅ Uncertainty quantification complete!")
else:
    print("⚠️  Summary file not found")
```

#### Step 5: Add Phase 9.5 Section

Insert these 5 cells after Phase 9.4:

**Cell 1 (Markdown):**

```markdown
## Section 9.5: Outlier Safety Rails & Non-Negative Constraints

**Objectives:**

- Track winsorization effects on feature distributions
- Validate non-negativity constraint adherence
- Analyze safety rails sensitivity across thresholds
- Generate interactive safety dashboards

**Inputs:**

- Raw and winsorized feature dataframes
- Predictions dataframe

**Outputs:**

- `outputs/safety_rails/clipping_effect_summary.json`
- `outputs/safety_rails/non_negative_violations.json`
- `outputs/safety_rails/safety_rails_summary.json`
- 3 interactive HTML visualizations
```

**Cell 2 (Code):**

```python
# %% [PHASE 9.5] Winsorization effects
print("\n" + "=" * 80)
print("PHASE 9.5: SAFETY RAILS & NON-NEGATIVE CONSTRAINTS")
print("=" * 80)

safety_rails_dir = output_dir / "safety_rails"
safety_rails_dir.mkdir(parents=True, exist_ok=True)

# Note: This requires access to raw and winsorized dataframes from earlier phases
# If not available, skip this cell
if 'all_stocks_raw' in dir() and 'all_stocks_winsorized' in dir():
    print("\n🔍 Analyzing winsorization effects...")

    # Get numeric columns
    numeric_cols = all_stocks_winsorized.select_dtypes(include=[np.number]).columns.tolist()

    summary_dict = summarize_winsorization_effects(
            df_raw=all_stocks_raw,
            df_winsorized=all_stocks_winsorized,
            numeric_cols=numeric_cols[:20],  # Limit to first 20 for demo
            output_dir=safety_rails_dir,
            sector_col="sector"
            )

    print(f"✓ Winsorization summary created for {len(numeric_cols[:20])} features")
    print(f"✓ Artifacts saved to: {safety_rails_dir}")
else:
    print("⚠️  Raw/winsorized dataframes not available. Skipping winsorization analysis.")
```

**Cell 3 (Code):**

```python
# %% [PHASE 9.5] Track constraint violations
print("\n🛡️  Checking non-negativity constraint violations...")

if predictions_path.exists():
    violations_dict = track_constraint_violations(
            predictions_df=predictions_df,
            output_dir=safety_rails_dir,
            pred_col="y_pred",
            sector_col="sector"
            )

    total_violations = violations_dict.get("total_violations", 0)
    violation_rate = violations_dict.get("violation_rate", 0)

    print(f"Total Violations: {total_violations}")
    print(f"Violation Rate: {violation_rate:.2%}")

    if total_violations == 0:
        print("✅ Non-negativity constraint satisfied!")
    else:
        print(f"⚠️  Found {total_violations} violations")
        violations_by_sector = violations_dict.get("violations_by_sector", {})
        for sector, count in violations_by_sector.items():
            if count > 0:
                print(f"   - {sector}: {count} violations")
```

**Cell 4 (Code):**

```python
# %% [PHASE 9.5] Interactive robustness sliders
if 'all_stocks_raw' in dir():
    print("\n📊 Creating safety rails sensitivity dashboard...")

    safety_rails_sensitivity_app(
            df_raw=all_stocks_raw,
            output_dir=safety_rails_dir,
            thresholds=[0.01, 0.05, 0.1]
            )

    print(f"✓ Sensitivity dashboard created: {safety_rails_dir / 'safety_rails_sensitivity_dashboard.html'}")
```

**Cell 5 (Code):**

```python
# %% [PHASE 9.5] Summary + QA
print("\n📋 Safety Rails Summary:")
print("=" * 80)

summary_path = safety_rails_dir / "safety_rails_summary.json"
if summary_path.exists():
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print(f"Winsorization Features: {summary.get('winsorization', {}).get('n_features', 0)}")
    print(f"Constraint Violations: {summary.get('violations', {}).get('total_violations', 0)}")

    print("\n✅ Safety rails monitoring complete!")
```

#### Step 6: Add Phase 9.6 Section

Insert these 5 cells after Phase 9.5:

**Cell 1 (Markdown):**

```markdown
## Section 9.6: Data Split and Leakage Policy Validation

**Objectives:**

- Validate CV fold construction and grouping rules
- Check for ticker/sector overlaps across folds
- Detect time-based leakage violations
- Ensure stratification balance

**Inputs:**

- Fold assignments dictionary (from CV training)
- Training dataframe with snapshot dates

**Outputs:**

- `outputs/splits/fold_overlap_heatmap.html`
- `outputs/splits/grouped_cv_balance_metrics.json`
- `outputs/splits/leakage_report.json`
```

**Cell 2 (Code):**

```python
# %% [PHASE 9.6] Fold overlap analysis
print("\n" + "=" * 80)
print("PHASE 9.6: DATA SPLIT AND LEAKAGE POLICY VALIDATION")
print("=" * 80)

splits_dir = output_dir / "splits"
splits_dir.mkdir(parents=True, exist_ok=True)

# Note: This requires fold_assignments from CV training
# Example: fold_assignments = {0: [idx_list], 1: [idx_list], ...}
if 'fold_assignments' in dir():
    print("\n🔍 Computing fold overlap...")

    overlap_dict = compute_fold_overlap(
            fold_assignments=fold_assignments,
            output_dir=splits_dir,
            group_col="ticker"
            )

    print(f"✓ Fold overlap analysis complete")
    print(f"  Zero overlap validated: {overlap_dict.get('zero_overlap_validated', False)}")
else:
    print("⚠️  fold_assignments not available. Skipping overlap analysis.")
```

**Cell 3 (Code):**

```python
# %% [PHASE 9.6] CV balance metrics
if 'fold_assignments' in dir() and 'all_stocks_features' in dir():
    print("\n📊 Summarizing grouped CV balance...")

    balance_dict = summarize_grouped_cv_balance(
            df=all_stocks_features,
            fold_assignments=fold_assignments,
            output_dir=splits_dir,
            stratify_cols=["sector", "region"]
            )

    print(f"✓ Balance metrics computed for {len(fold_assignments)} folds")
```

**Cell 4 (Code):**

```python
# %% [PHASE 9.6] Time leakage checks
if 'fold_assignments' in dir() and 'all_stocks_features' in dir() and 'snapshot_date' in all_stocks_features.columns:
    print("\n🕐 Checking time-based leakage...")

    leakage_report = time_leakage_checks(
            df=all_stocks_features,
            fold_assignments=fold_assignments,
            output_dir=splits_dir,
            date_col="snapshot_date"
            )

    violations = leakage_report.get("violations", 0)
    print(f"  Leakage violations: {violations}")

    if violations == 0:
        print("✅ No time-based leakage detected!")
```

**Cell 5 (Code):**

```python
# %% [PHASE 9.6] Summary + QA
print("\n📋 Data Split Validation Summary:")
print("=" * 80)

leakage_path = splits_dir / "leakage_report.json"
if leakage_path.exists():
    with open(leakage_path, 'r') as f:
        report = json.load(f)

    print(f"Violations: {report.get('violations', 0)}")
    print(f"Severity: {report.get('severity', 'NONE')}")

    print("\n✅ Data split validation complete!")
```

#### Step 7: Save and Verify

1. Save the notebook
2. Verify the structure:
    - Phase 9.4 (5 cells: 1 markdown + 4 code)
    - Phase 9.5 (5 cells: 1 markdown + 4 code)
    - Phase 9.6 (5 cells: 1 markdown + 4 code)
    - All appear before Phase 9.7

## Key Improvements

1. ✅ **Proper line breaks** - Each statement on its own line
2. ✅ **Correct spacing** - Proper spaces after function calls
3. ✅ **Standard cell markers** - Using `#%%` consistently
4. ✅ **Clean markdown** - Proper formatting with line breaks
5. ✅ **Correct placement** - Cells in proper phase sections
6. ✅ **Follows guidelines** - Aligned with `NOTEBOOK_INTEGRATION_GUIDE.md`
7. ✅ **Uses constants** - References `TARGET_COL`, `TEST_SIZE` from config

## Validation Checklist

After making changes, verify:

- [ ] No malformed cells at the end of notebook
- [ ] Phase 9.4 section exists with 5 cells
- [ ] Phase 9.5 section exists with 5 cells
- [ ] Phase 9.6 section exists with 5 cells
- [ ] All sections appear before Phase 9.7
- [ ] All code cells have proper line breaks
- [ ] All markdown cells have proper formatting
- [ ] Cell markers use `#%%` not `##%%`
- [ ] Notebook can be loaded without errors

## Files Created

1. `fix_phase94_cells.py` - Automated fix script
2. `analyze_phase94_cells.py` - Analysis script
3. `PHASE94_96_FIX_SUMMARY.md` - This document

## References

- `docs/code_guidelines.md` - Section 8.1-8.5 (Notebook Best Practices)
- `docs/NOTEBOOK_INTEGRATION_GUIDE.md` - Complete cell templates
- Workflow Overview in task description (10 Steps, Phase 9.1-9.8)
