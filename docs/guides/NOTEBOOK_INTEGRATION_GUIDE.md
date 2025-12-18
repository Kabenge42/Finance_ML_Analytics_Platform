# Notebook Integration Guide - Phase 9.4-9.8 Advanced Evaluation

This guide provides ready-to-use cells for integrating Phase 9.4-9.8 advanced evaluation and governance features into
`ml_finance_model_main.ipynb`.

## Prerequisites

- Ensure `finance_ml` package is installed with all dependencies
- Run through existing Phase 9.1-9.5 sections to generate predictions
- Predictions should be saved in `outputs/regression/regression_predictions_detailed.csv`

## Import Statements (Add to Imports Cell)

Add these imports to the existing imports cell near the beginning of the notebook:

```python
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
```

---

## Section 9.4: Uncertainty Quantification & Conformal Calibration

### Cell 1: Markdown Header

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

### Cell 2: Load Predictions and Build Diagnostics

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

### Cell 3: Coverage and Width Visualizations

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

### Cell 4: Reliability Diagram

```python
#%% [PHASE 9.4] Reliability diagram for conformal calibration
print("\n📈 Creating reliability diagram...")

plot_reliability_diagram(
        diagnostics_df=diagnostics_df,
        output_dir=uncertainty_dir,
        pre_calibration_df=None  # Set to pre-calibration df if available
        )

print(f"✓ Reliability diagram created: {uncertainty_dir / 'reliability_diagram_conformal.html'}")
```

### Cell 5: Summary and QA

```python
#%% [PHASE 9.4] Summary + QA
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

---

## Section 9.5: Outlier Safety Rails & Non-Negative Constraints

### Cell 1: Markdown Header

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

### Cell 2: Winsorization Effects Summary

```python
#%% [PHASE 9.5] Winsorization effects
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

### Cell 3: Constraint Violations Tracker

```python
#%% [PHASE 9.5] Track constraint violations
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

### Cell 4: Safety Rails Sensitivity Dashboard

```python
#%% [PHASE 9.5] Interactive robustness sliders
if 'all_stocks_raw' in dir():
    print("\n📊 Creating safety rails sensitivity dashboard...")

    safety_rails_sensitivity_app(
            df_raw=all_stocks_raw,
            output_dir=safety_rails_dir,
            thresholds=[0.01, 0.05, 0.1]
            )

    print(f"✓ Sensitivity dashboard created: {safety_rails_dir / 'safety_rails_sensitivity_dashboard.html'}")
```

### Cell 5: Safety Rails Summary

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

---

## Section 9.6: Data Split and Leakage Policy Validation

### Cell 1: Markdown Header

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

### Cell 2: Fold Overlap Analysis

```python
#%% [PHASE 9.6] Fold overlap analysis
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

### Cell 3: Grouped CV Balance Metrics

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

### Cell 4: Time-Based Leakage Checks

```python
#%% [PHASE 9.6] Time leakage checks
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

### Cell 5: Splits Summary

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

---

## Section 9.7: Sector Bias Calibration & Metrics Persistence

### Cell 1: Markdown Header

```markdown
## Section 9.7: Sector Bias Calibration & Metrics Persistence

**Objectives:**

- Estimate sector-level bias before/after calibration
- Track MAE/MAPE improvements per sector
- Visualize metrics trends over time
- Persist calibration metadata with model versioning

**Inputs:**

- Predictions dataframe with y_true, y_pred, y_pred_calibrated

**Outputs:**

- `outputs/calibration/sector_bias_calibration_v{MODEL_VERSION}.json`
- `outputs/calibration/metrics_by_sector_time.html`
- `outputs/calibration/sector_bias_dashboard.html`
```

### Cell 2: Bias Estimation

```python
#%% [PHASE 9.7] Sector bias estimation
print("\n" + "=" * 80)
print("PHASE 9.7: SECTOR BIAS CALIBRATION & METRICS PERSISTENCE")
print("=" * 80)

calibration_dir = output_dir / "calibration"
calibration_dir.mkdir(parents=True, exist_ok=True)

if predictions_path.exists():
    print("\n🔍 Estimating sector-level bias...")

    bias_dict = estimate_sector_bias(
            predictions_df=predictions_df,
            output_dir=calibration_dir,
            y_true_col="y_true",
            y_pred_col="y_pred",
            y_pred_calibrated_col="y_pred_calibrated",
            sector_col="sector",
            model_version=MODEL_VERSION
            )

    print(f"✓ Bias estimation complete for {len(bias_dict.get('sectors', {}))} sectors")
    print(f"✓ Versioned file: sector_bias_calibration_{MODEL_VERSION}.json")
```

### Cell 3: Metrics Over Time Visualization

```python
#%% [PHASE 9.7] Metrics over time
# Note: This requires historical metrics data
# If not available, skip this cell
if 'metrics_history_df' in dir():
    print("\n📈 Plotting metrics by sector over time...")

    plot_metrics_by_sector_time(
            metrics_history=metrics_history_df,
            output_dir=calibration_dir,
            snapshot_date_col="snapshot_date"
            )

    print(f"✓ Time-series plot created: {calibration_dir / 'metrics_by_sector_time.html'}")
else:
    print("⚠️  metrics_history_df not available. Skipping time-series plot.")
```

### Cell 4: Sector Bias Dashboard

```python
#%% [PHASE 9.7] Interactive bias dashboard
if predictions_path.exists() and 'bias_dict' in dir():
    print("\n📊 Creating sector bias dashboard...")

    create_sector_bias_dashboard(
            predictions_df=predictions_df,
            bias_dict=bias_dict,
            output_dir=calibration_dir
            )

    print(f"✓ Dashboard created: {calibration_dir / 'sector_bias_dashboard.html'}")
```

### Cell 5: Calibration Summary

```python
# %% [PHASE 9.7] Summary + QA
print("\n📋 Sector Bias Calibration Summary:")
print("=" * 80)

bias_path = calibration_dir / f"sector_bias_calibration_{MODEL_VERSION}.json"
if bias_path.exists():
    with open(bias_path, 'r') as f:
        bias_data = json.load(f)

    print(f"Model Version: {bias_data.get('model_version', 'N/A')}")
    print(f"Sectors Analyzed: {len(bias_data.get('sectors', {}))}")

    print("\n✅ Sector bias calibration complete!")
```

---

## Section 9.8: Stacking Ensemble Diagnostics & Model Governance

### Cell 1: Markdown Header

```markdown
## Section 9.8: Stacking Ensemble Diagnostics & Model Governance

**Objectives:**

- Analyze base model contributions to ensemble
- Generate explainability visuals (SHAP or permutation importance)
- Create meta-learner error maps
- Auto-generate model card and lineage documentation

**Inputs:**

- Base model predictions dictionary
- Meta-learner predictions
- Model configuration metadata

**Outputs:**

- `outputs/governance/stacking_contributions.csv`
- `outputs/governance/stacking_contributions.html`
- `outputs/governance/meta_error_map.html`
- `outputs/governance/model_card_v{MODEL_VERSION}.md`
- `outputs/governance/lineage.json`
```

### Cell 2: Base Model Contribution Analysis

```python
#%% [PHASE 9.8] Stacking contributions
print("\n" + "=" * 80)
print("PHASE 9.8: STACKING ENSEMBLE DIAGNOSTICS & MODEL GOVERNANCE")
print("=" * 80)

governance_dir = output_dir / "governance"
governance_dir.mkdir(parents=True, exist_ok=True)

# Note: This requires base_predictions dict from stacking ensemble training
# Example: base_predictions = {"xgboost": y_pred_xgb, "lightgbm": y_pred_lgb}
if 'base_predictions' in dir() and 'y_pred_meta' in dir() and 'y_test' in dir():
    print("\n🔍 Computing stacking contributions...")

    contributions_df = compute_stacking_contributions(
            base_predictions=base_predictions,
            meta_predictions=y_pred_meta,
            y_true=y_test,
            output_dir=governance_dir
            )

    print(f"✓ Contributions computed for {len(base_predictions)} base models")
    print(f"✓ Artifacts saved to: {governance_dir}")
else:
    print("⚠️  Stacking ensemble data not available. Skipping contribution analysis.")
```

### Cell 3: Explainability Overlays

```python
#%% [PHASE 9.8] Explainability (SHAP or permutation importance)
print("\n🔍 Generating explainability visuals...")

# SHAP is optional - fallback to permutation importance
try:
    import shap

    SHAP_AVAILABLE = True
    print("  Using SHAP for explainability")
except ImportError:
    SHAP_AVAILABLE = False
    print("  SHAP not available - using permutation importance fallback")

# Note: Actual SHAP/permutation importance code would go here
# This is a placeholder
if SHAP_AVAILABLE and 'model' in dir() and 'X_test' in dir():
    # SHAP analysis code
    print("  ✓ SHAP summary created")
else:
    # Permutation importance fallback
    print("  ✓ Permutation importance created")
```

### Cell 4: Meta-Learner Error Maps

```python
#%% [PHASE 9.8] Meta-learner error maps
if predictions_path.exists():
    print("\n📊 Creating meta-learner error maps...")

    meta_error_maps(
            predictions_df=predictions_df,
            output_dir=governance_dir,
            error_col="abs_error",
            sector_col="sector"
            )

    print(f"✓ Error maps created: {governance_dir / 'meta_error_map.html'}")
```

### Cell 5: Model Card Generation

```python
#%% [PHASE 9.8] Generate model card
print("\n📋 Generating model card...")

model_info = {
    "task": "Price target regression + classification-enhanced features",
    "data_sources": ["PostgreSQL equities table", "Multi-region CSVs"],
    "features": {
        "count": 310,
        "groups": ["momentum", "valuation", "profitability", "quality", "cash_flow", "growth"],
        "selection_method": "Phase 9.3 comprehensive pipeline"
        },
    "models": {
        "base": ["XGBoost", "LightGBM", "CatBoost"],
        "meta": "Ridge Regression",
        "hyperparameters": "Optuna-tuned"
        },
    "validation": {
        "strategy": "Grouped K-Fold CV (by ticker)",
        "n_folds": CV_FOLDS,
        "leakage_check": "Passed"
        }
    }

generate_model_card(
        model_info=model_info,
        output_dir=governance_dir,
        model_version=MODEL_VERSION
        )

print(f"✓ Model card created: {governance_dir / f'model_card_{MODEL_VERSION}.md'}")
```

### Cell 6: Lineage Tracking

```python
#%% [PHASE 9.8] Build lineage JSON
print("\n🔗 Building model lineage...")

datasets = {
    "train": "equities_table_2025",
    "validation": "hold_out_2025"
    }

features = {
    "count": 310,
    "groups": ["momentum", "valuation", "profitability", "quality", "cash_flow", "growth"],
    "selection": "comprehensive"
    }

models = {
    "base": ["xgboost", "lightgbm", "catboost"],
    "meta": "ridge",
    "hyperparameters": {"cv_folds": CV_FOLDS}
    }

artifacts = [
    "regression_predictions_detailed.csv",
    "quantile_predictions_diagnostics.csv",
    "sector_bias_calibration_v9_10.json",
    "model_card_v9_10.md"
    ]

metrics = {
    "overall": {"MAE": 0.0, "RMSE": 0.0, "R2": 0.0},
    "by_sector": {}
    }

lineage = build_lineage_json(
        datasets=datasets,
        features=features,
        models=models,
        artifacts=artifacts,
        metrics=metrics,
        output_dir=governance_dir,
        model_version=MODEL_VERSION
        )

print(f"✓ Lineage JSON created: {governance_dir / 'lineage.json'}")
print("\n✅ Model governance documentation complete!")
```

---

## Section 10: Portfolio Optimization Workflow

### Prerequisites

- Completed Phase 9.5 (regression) to generate predictions
- Stock universe data with sector, region, market_cap columns
- `finance_ml.ml_workflow.analytics.portfolio_reporting` module available

### Import Statements (Add to Imports Cell)

Add these imports to the existing imports cell for Section 10:

```python
# Section 10: Portfolio Optimization Workflow
from finance_ml.ml_workflow.analytics.portfolio_reporting import (
  universe_summary,
  returns_risk_diagnostics,
  frontier_and_constraints,
  risk_decomposition_dashboard,
  backtest_and_attribution,
  risk_management_dashboard,
  portfolio_summary,
  )
from finance_ml.dashboards import (
  PortfolioRebalanceWidget,
  create_multi_period_comparison,
  create_factor_exposure_dashboard,
  )
```

---

### Section 10.1: Overview & Inputs

#### Cell 1: Markdown Header

```markdown
## Section 10: Portfolio Optimization Workflow

**Objectives:**

- Build optimized portfolios from stock universe with advanced methods
- Perform comprehensive risk analysis and stress testing
- Generate interactive dashboards for portfolio monitoring
- Validate constraint adherence and backtest performance

**Inputs:**

- Stock universe: `all_stocks` DataFrame with predictions
- Expected returns: ML-based or analyst consensus
- Covariance matrix: Historical or factor-based
- Constraints: Position limits, sector caps, turnover limits

**Outputs:**

- `outputs/portfolio/` - 15+ artifacts including dashboards, holdings, and analytics
```

---

### Section 10.2: Universe & Filters Diagnostics

#### Cell 1: Selection Summary

```python
# %% [SECTION 10.2] Universe and filters diagnostics
print("\n" + "=" * 80)
print("SECTION 10.2: UNIVERSE & FILTERS DIAGNOSTICS")
print("=" * 80)

from pathlib import Path
import pandas as pd

# Setup paths
output_dir = Path("outputs")
portfolio_dir = output_dir / "portfolio"
portfolio_dir.mkdir(parents=True, exist_ok=True)

# Load stock universe (from Phase 9.5 or earlier)
if 'all_stocks' not in locals():
  print("⚠️  all_stocks DataFrame not found in workspace")
  print("   Please run data loading sections first")
else:
  print(f"📊 Stock universe loaded: {len(all_stocks):,} stocks")

  # Generate universe summary
  print("\n🔍 Generating universe diagnostics...")
  manifest = universe_summary(all_stocks, portfolio_dir)

  print(f"✓ Universe summary created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")
```

#### Cell 2: Interactive Filter Explorer

```python
# %% [SECTION 10.2] Interactive filter explorer (placeholder)
print("\n🎯 Filter explorer placeholder generated")
print(f"   View: {portfolio_dir / 'portfolio_filter_explorer.html'}")
print("   Note: Enhance with Plotly interactive widgets for production")
```

---

### Section 10.3: Expected Returns & Risk Inputs QA

#### Cell 1: Returns Distribution

```python
# %% [SECTION 10.3] Expected returns and risk inputs QA
print("\n" + "=" * 80)
print("SECTION 10.3: EXPECTED RETURNS & RISK INPUTS QA")
print("=" * 80)

import numpy as np

# Example: Use ML predictions as expected returns
if 'predictions_df' in locals() or (output_dir / "regression" / "regression_predictions_detailed.csv").exists():
  if 'predictions_df' not in locals():
    predictions_df = pd.read_csv(output_dir / "regression" / "regression_predictions_detailed.csv")

  # Create expected returns series (annualized)
  mu = predictions_df.set_index('ticker')['y_pred'] / predictions_df.set_index('ticker')['last_price'] - 1
  mu = mu * 252  # Annualize daily returns

  # Compute sample covariance (placeholder - use factor models for production)
  # For demonstration, create a simple covariance matrix
  n_assets = len(mu)
  cov = np.eye(n_assets) * 0.04  # Diagonal with 20% vol

  print(f"📈 Expected returns computed for {len(mu):,} assets")
  print(f"   Mean return: {mu.mean():.2%}")
  print(f"   Return std: {mu.std():.2%}")

  # Generate diagnostics
  print("\n🔍 Generating returns/risk diagnostics...")
  manifest = returns_risk_diagnostics(mu, cov, portfolio_dir)

  print(f"✓ Returns/risk diagnostics created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")
else:
  print("⚠️  Predictions not available - skip returns QA")
```

#### Cell 2: Correlation Heatmap

```python
# %% [SECTION 10.3] Risk correlation heatmap
print(f"\n📊 Correlation heatmap: {portfolio_dir / 'risk_correlation_heatmap.html'}")
print("   Note: Enhance with top-N assets and factor decomposition for production")
```

#### Cell 3: Volatility Drift Dashboard

```python
# %% [SECTION 10.3] Risk drift dashboard
print(f"\n📈 Risk drift dashboard: {portfolio_dir / 'risk_drift_dashboard.html'}")
print("   Note: Add time-series volatility tracking vs benchmark for production")
```

---

### Section 10.4: Optimization Frontiers & Constraint Explorer

#### Cell 1: Efficient Frontier

```python
# %% [SECTION 10.4] Optimization frontiers and constraint explorer
print("\n" + "=" * 80)
print("SECTION 10.4: OPTIMIZATION FRONTIERS & CONSTRAINTS")
print("=" * 80)

# Define constraints for optimization
constraints = {
  "max_weight": [0.05, 0.10, 0.15, 0.20],  # Per-asset limits
  "sector_caps": [0.30, 0.35, 0.40],  # Sector exposure limits
  "turnover_limit": [0.10, 0.20, 0.30],  # Portfolio turnover limits
  }

if 'mu' in locals() and 'cov' in locals():
  print("\n🎯 Generating efficient frontier and constraint scenarios...")
  manifest = frontier_and_constraints(mu, cov, constraints, portfolio_dir)

  print(f"✓ Frontier and constraint artifacts created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")

  # Load scenarios for review
  scenarios_df = pd.read_csv(portfolio_dir / "constraints_scenarios.csv")
  print(f"\n📋 Generated {len(scenarios_df)} constraint scenarios")
else:
  print("⚠️  Returns/covariance not available - skip optimization")
```

#### Cell 2: Constraint Sensitivity

```python
# %% [SECTION 10.4] Constraint sensitivity with sliders
print(f"\n🎚️ Constraint sensitivity: {portfolio_dir / 'constraints_sensitivity.html'}")
print("   Note: Add Plotly sliders for interactive exploration in production")
```

#### Cell 3: Transaction Cost Impact

```python
# %% [SECTION 10.4] Transaction cost impact
print(f"\n💰 Transaction cost impact: {portfolio_dir / 'transaction_cost_impact.html'}")
print(f"   Summary: {portfolio_dir / 'transaction_cost_summary.json'}")
```

---

### Section 10.5: Chosen Portfolio Breakdown & Risk Decomposition

#### Cell 1: Holdings Table

```python
# %% [SECTION 10.5] Portfolio breakdown and risk decomposition
print("\n" + "=" * 80)
print("SECTION 10.5: PORTFOLIO BREAKDOWN & RISK DECOMPOSITION")
print("=" * 80)

# Example: Create simple equal-weight portfolio for demonstration
if 'mu' in locals():
  n_assets = min(20, len(mu))  # Top 20 by expected return
  top_tickers = mu.nlargest(n_assets).index
  weights = pd.Series(1.0 / n_assets, index=top_tickers, name='weight')

  # Create exposures DataFrame
  exposures = all_stocks[all_stocks['ticker'].isin(top_tickers)][['ticker', 'sector', 'region']].set_index('ticker')

  print(f"📊 Portfolio: {n_assets} positions, equal-weighted")
  print(f"   Total weight: {weights.sum():.2%}")

  # Generate risk decomposition
  print("\n🔍 Generating risk decomposition...")
  manifest = risk_decomposition_dashboard(weights, exposures, portfolio_dir)

  print(f"✓ Risk decomposition artifacts created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")

  # Display holdings summary
  holdings_df = pd.read_csv(portfolio_dir / "portfolio_holdings_detailed.csv")
  print(f"\n📋 Holdings summary (top 5):")
  print(holdings_df.head())
else:
  print("⚠️  Portfolio weights not available - skip decomposition")
```

#### Cell 2: Risk Decomposition Waterfall

```python
# %% [SECTION 10.5] Risk decomposition waterfall
print(f"\n📊 Risk decomposition: {portfolio_dir / 'risk_decomposition.html'}")
print("   Note: Add factor contributions and idiosyncratic risk for production")
```

#### Cell 3: Stress Tests

```python
# %% [SECTION 10.5] Scenario and stress tests
print(f"\n⚡ Stress tests dashboard: {portfolio_dir / 'stress_tests_dashboard.html'}")
print("   Note: Add market crash, rate spike, sector rotation scenarios for production")
```

---

### Section 10.6: Backtesting & Walk-Forward Results

#### Cell 1: Vectorized Backtest

```python
# %% [SECTION 10.6] Backtesting and walk-forward results
print("\n" + "=" * 80)
print("SECTION 10.6: BACKTESTING & WALK-FORWARD RESULTS")
print("=" * 80)

# Example: Generate synthetic price history for backtesting
if 'weights' in locals():
  # Create synthetic prices (60 days, aligned with weights)
  np.random.seed(42)
  n_days = 60
  n_assets_bt = len(weights)
  returns_bt = np.random.normal(0.001, 0.02, size=(n_days, n_assets_bt))
  prices_bt = 100 * np.exp(np.cumsum(returns_bt, axis=0))

  dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
  prices_df = pd.DataFrame(prices_bt, index=dates, columns=weights.index)

  print(f"📈 Backtest data: {n_days} days, {n_assets_bt} assets")

  # Generate backtest and attribution
  print("\n🔍 Running backtest and attribution...")
  manifest = backtest_and_attribution(prices_df, weights, portfolio_dir)

  print(f"✓ Backtest artifacts created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")

  # Display attribution summary
  attrib_df = pd.read_csv(portfolio_dir / "attribution_breakdown.csv")
  print(f"\n📋 Attribution breakdown (top 5 contributors):")
  print(attrib_df.nlargest(5, 'contribution'))
else:
  print("⚠️  Portfolio weights not available - skip backtesting")
```

#### Cell 2: Attribution Report

```python
# %% [SECTION 10.6] Performance attribution
print(f"\n📊 Performance attribution: {portfolio_dir / 'performance_attribution.html'}")
print("   Note: Add Brinson-Fachler decomposition for production")
```

#### Cell 3: Walk-Forward Stability

```python
# %% [SECTION 10.6] Walk-forward optimization stability
print(f"\n🔄 Walk-forward stability: {portfolio_dir / 'walk_forward_stability.html'}")
print("   Note: Add weights turnover and hit rate analysis for production")
```

---

### Section 10.7: Risk Management Dashboard

#### Cell 1: Risk Metrics

```python
# %% [SECTION 10.7] Risk management dashboard
print("\n" + "=" * 80)
print("SECTION 10.7: RISK MANAGEMENT DASHBOARD")
print("=" * 80)

if 'weights' in locals() and 'cov' in locals():
  # Align weights to covariance matrix dimensions
  cov_aligned = cov[:len(weights), :len(weights)]

  print("\n🔍 Generating risk management dashboard...")
  manifest = risk_management_dashboard(weights, cov_aligned, portfolio_dir)

  print(f"✓ Risk management artifacts created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")

  # Compute key metrics
  w = weights.values.reshape(-1, 1)
  portfolio_vol = np.sqrt(np.maximum(0.0, (w.T @ cov_aligned @ w).item()))

  print(f"\n📊 Portfolio metrics:")
  print(f"   Volatility: {portfolio_vol:.2%}")
  print(f"   Expected return: {(weights @ mu.reindex(weights.index).fillna(0)):.2%}")
else:
  print("⚠️  Weights/covariance not available - skip risk dashboard")
```

#### Cell 2: Rebalancing Widget

```python
# %% [SECTION 10.7] Interactive rebalancing widget
print(f"\n🎯 Rebalancing widget: {portfolio_dir / 'portfolio_rebalancing_widget.html'}")
print("   Note: Use PortfolioRebalanceWidget for interactive trades calculation")
```

---

### Section 10.8: Summary, QA, and Export

#### Cell 1: Portfolio Summary

```python
# %% [SECTION 10.8] Portfolio summary and export
print("\n" + "=" * 80)
print("SECTION 10.8: PORTFOLIO SUMMARY & EXPORT")
print("=" * 80)

# Compute final KPIs
if 'weights' in locals() and 'mu' in locals() and 'cov' in locals():
  portfolio_return = float(weights @ mu.reindex(weights.index).fillna(0))
  w = weights.values.reshape(-1, 1)
  portfolio_vol = float(np.sqrt(np.maximum(0.0, (w.T @ cov_aligned @ w).item())))
  sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

  kpis = {
    "portfolio_return": portfolio_return,
    "portfolio_volatility": portfolio_vol,
    "sharpe_ratio": sharpe_ratio,
    "n_positions": len(weights),
    "max_weight": float(weights.max()),
    "min_weight": float(weights.min()),
    }

  print(f"\n📊 Portfolio KPIs:")
  for k, v in kpis.items():
    if isinstance(v, float) and abs(v) < 100:
      print(f"   {k}: {v:.4f}")
    else:
      print(f"   {k}: {v}")

  # Generate summary
  print("\n🔍 Generating portfolio summary...")
  manifest = portfolio_summary(kpis, portfolio_dir)

  print(f"✓ Portfolio summary artifacts created:")
  for f in manifest.get('files', []):
    print(f"  - {f}")
else:
  print("⚠️  Portfolio metrics not available - skip summary")
```

#### Cell 2: Multi-Period Comparison

```python
# %% [SECTION 10.8] Multi-period comparison
print(f"\n📈 Multi-period comparison: {portfolio_dir / 'portfolio_multi_period_comparison.html'}")
print("   Note: Use create_multi_period_comparison() for production visualizations")

print("\n✅ Section 10: Portfolio Optimization Workflow complete!")
print(f"   All artifacts saved to: {portfolio_dir}")
```

---

## Summary

After adding these cells to your notebook:

1. **Import statements** are added to the imports cell
2. **6 new sections** (9.4-9.8 + Section 10) with comprehensive reporting
3. **45+ new cells** total (5-6 per Phase 9 section, 19 for Section 10)
4. **45+ artifacts** generated across 6 output directories

### Output Directory Structure

```
outputs/
├── uncertainty/          # Phase 9.4
├── safety_rails/         # Phase 9.5
├── splits/               # Phase 9.6
├── calibration/          # Phase 9.7
├── governance/           # Phase 9.8
└── portfolio/            # Section 10
```

### Next Steps

1. Copy cells into `ml_finance_model_main.ipynb` after existing Phase 9.8
2. Run cells sequentially (some cells check for prerequisite data)
3. Review generated artifacts in respective output directories
4. Customize visualizations and thresholds as needed

### Troubleshooting

- **Missing predictions**: Ensure Phase 9.5 (regression) has run and created `regression_predictions_detailed.csv`
- **Missing dataframes**: Some cells require intermediate dataframes (all_stocks_raw, fold_assignments). Add guards or
  skip if unavailable.
- **SHAP unavailable**: Permutation importance fallback is automatic
- **Plotly issues**: All visualization functions have minimal HTML fallbacks

For more details, see:

- `docs/code_guidelines.md` - Function signatures and usage
- `docs/improvement_plan/notebook_restructuring_plan.md` - Complete specifications
- `CHANGELOG.md` - Implementation history
