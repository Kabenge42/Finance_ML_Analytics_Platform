# Notebook Gap Analysis: ml_finance_model_main.ipynb

## Analysis Date: 2025-11-14

## Executive Summary

The notebook contains all 8 phases (9.1-9.8) but has **structural misalignment** between:

- **Current**: Numbered business sections (1-10) with phase references embedded
- **Required**: Phase-aligned architecture (9.1-9.8 as primary organization per code_guidelines.md and
  improvement_plan.md)

---

## 1. Section Organization Misalignment

### Current Structure (Sections 1-10):

1. Configuration and Setup
2. Loading and Preprocessing Financial Data
3. Exploratory Data Analysis of Financial Metrics
4. Advanced Feature Engineering with Sector-Specific Optimizations
5. Multi-Class Classification of Financial Events
6. Sector-Optimized Regression Models with Classification Features
7. Model Evaluation and Error Analysis
8. Identification of Under/Overvalued Stocks with Visualization
9. Comprehensive Analytics: Predicted vs. Analyst Price Target Comparison
10. Portfolio Optimization with Risk Metrics

### Required Structure (Phase 9.1-9.8):

- **Phase 9.1**: Loading and preprocessing with 6-step imputation strategy
- **Phase 9.2**: Enhanced exploratory data analysis with statistical testing and benchmarking
- **Phase 9.3**: Advanced feature engineering with sector-specific optimizations
- **Phase 9.4**: Multi-class event classification using neural networks and ensembles
- **Phase 9.5**: Sector-optimized regression models with hyperparameter tuning and quantile models
- **Phase 9.6**: Model evaluation and comprehensive error analysis
- **Phase 9.7**: Identification of under/overvalued stocks with visualization and analyst comparison
- **Phase 9.8**: Comprehensive analytics and reporting

### Mapping Issues:

- Section 2 → Should be "Phase 9.1" (correct content, wrong header)
- Section 3 → Should be "Phase 9.2" (correct content, wrong header)
- Section 4 → Should be "Phase 9.3" (correct content, wrong header)
- Section 5 → Should be "Phase 9.4" (correct content, wrong header)
- Section 6 → Should be "Phase 9.5" (correct content, wrong header)
- Section 7 → Should be "Phase 9.6" (correct content, wrong header)
- Sections 8, 9, 10 → Should be consolidated into "Phase 9.7" and "Phase 9.8"

**ACTION REQUIRED**: Restructure section headers to match Phase 9.1-9.8 organization

---

## 2. v1.2 Standards Compliance Check

### 2.1 Uncertainty and Prediction Intervals

**Required** (code_guidelines.md lines 1541-1579):

- Quantile regression at q ∈ {0.1, 0.5, 0.9}
- Conformal calibration
- Monotonicity enforcement (p10 ≤ p50 ≤ p90)
- Non-negativity (clip lower bounds at 0)
- Coverage target: 80% ± 5%
- Output: `outputs/regression/quantile_predictions.csv`

**Current Status**: Need to verify implementation in Section 6.5

**ACTION REQUIRED**: Verify quantile regression implementation includes all v1.2 requirements

### 2.2 Outlier Safety Rails Policy

**Required** (code_guidelines.md lines 1582-1593):

- Target winsorization at [1st, 99th] percentiles
- Robust loss (Huber)
- Post-prediction clipping (mean ± 3·std)
- Non-negativity guard
- Diagnostics (counts >100%, >1000% error)

**Current Status**: Need to verify implementation

**ACTION REQUIRED**: Verify outlier safety rails implementation

### 2.3 Data Split and Leakage Policy

**Required** (code_guidelines.md lines 1596-1610):
Priority order:

1. TimeSeriesSplit if time column exists
2. GroupKFold by ticker
3. Stratify by sector
4. Random with seed

**Current Status**: Section 6.5.1 mentions Time-Series Cross-Validation

**ACTION REQUIRED**: Verify data split policy adherence

### 2.4 Standardized Predictions Schema

**Required** (code_guidelines.md lines 1613-1628):
Required columns:

- ticker, isin, sector, region, snapshot_date
- last_price, y_true, y_pred, y_pred_calibrated
- pred_p10, pred_p50, pred_p90, interval_width
- abs_error, pct_error
- model_version

Output: `outputs/regression/regression_predictions_detailed.csv`

**Current Status**: Need to verify output schema

**ACTION REQUIRED**: Verify predictions output includes all required columns

### 2.5 Sector Metrics and Calibration

**Required** (code_guidelines.md lines 1631-1643):

- Per-sector metrics: `outputs/models/regression_metrics_by_sector.csv`
- Sector-specific bias calibration

**Current Status**: Need to verify

**ACTION REQUIRED**: Verify sector metrics persistence and calibration

---

## 3. API Usage Alignment

### 3.1 Phase 9.1 API (Preprocessing)

**Required API** (improvement_plan.md lines 190-192):

```python
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data

X_prep, stats = prepare_phase91_data(df, sector_column="sector", price_column="last_price", n_neighbors=5,
                                     return_stats=True)
```

**Current Status**: Notebook uses individual imputation functions

**ACTION REQUIRED**: Check if `prepare_phase91_data()` orchestrator is used or if manual orchestration is needed

### 3.2 Phase 9.3 API (Features)

**Required API** (improvement_plan.md lines 207-211):

```python
from finance_ml.ml_workflow.features.api import build_features

X_feat = build_features(X_prep, preset="sector_optimized", include_interactions=True, include_relative=True)
```

**Current Status**: Section 4/Cell 30 mentions "Phase 9.3 API - Feature Engineering Presets (New in v0.7.0)"

**ACTION REQUIRED**: Verify API usage matches specification

### 3.3 Phase 9.4 API (Classification)

**Required API** (improvement_plan.md lines 213-220):

```python
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels
from finance_ml.ml_workflow.classification.models import fit_classifier

labels = create_enhanced_event_labels(X_feat, method="price_momentum", threshold_positive=10, threshold_negative=-10)
clf_res = fit_classifier(X_feat, labels, model="lightgbm", tuning={"n_trials": 50}, cv={"sector_stratified": True})
```

**Current Status**: Cell 38 mentions "Hyperparameter optimization with Phase 9.4 function"

**ACTION REQUIRED**: Verify API usage and standardized return dict

### 3.4 Phase 9.5 API (Regression)

**Required API** (improvement_plan.md lines 224-232):

```python
from finance_ml.ml_workflow.regression.dataset import integrate_classification_features
from finance_ml.ml_workflow.regression.models import fit_regressor

X_reg = integrate_classification_features(X_feat, clf_res.artifacts["probabilities"])
reg_res = fit_regressor(X_reg, y=target, model="xgboost", ensure_nonnegative=True)
```

**Current Status**: Need to verify

**ACTION REQUIRED**: Verify classification meta-features integration and API usage

---

## 4. Missing Components

### 4.1 Clear Phase Headers

**Issue**: Phase markers are embedded in comments, not as primary section headers

**ACTION REQUIRED**: Add clear markdown headers for each phase:

```markdown
## Phase 9.1: Loading and Preprocessing with 6-Step Imputation Strategy

## Phase 9.2: Enhanced Exploratory Data Analysis

## Phase 9.3: Advanced Feature Engineering

## Phase 9.4: Multi-Class Event Classification

## Phase 9.5: Sector-Optimized Regression Models

## Phase 9.6: Model Evaluation and Error Analysis

## Phase 9.7: Stock Ranking and Analyst Comparison

## Phase 9.8: Comprehensive Analytics and Reporting
```

### 4.2 Phase Validation Checkpoints

**Required**: Each phase should have a validation checkpoint

**Current Status**: Phase 9.1 has validation checkpoint (Cell 18)

**ACTION REQUIRED**: Add validation checkpoints for all phases

### 4.3 Phase Output Artifacts

**Required**: Each phase should persist its artifacts to phase-specific output directories

**Current Status**: Output directories created (Cell 6), but need to verify usage

**ACTION REQUIRED**: Verify each phase saves artifacts to correct directories

---

## 5. Code Structure Issues

### 5.1 Standardized Return Types

**Required** (code_guidelines.md lines 18-26):
All `train_*` functions must return dict with:

- model
- metrics
- y_pred
- y_proba (optional)
- artifacts (optional)

**ACTION REQUIRED**: Verify all model training calls follow this pattern

### 5.2 Column Naming

**Required** (code_guidelines.md lines 764-790):

- Canonical names: last_price, price_target, sector, region, ticker
- Must normalize early via `normalize_columns()`

**ACTION REQUIRED**: Verify column normalization is applied early

---

## 6. Documentation and Clarity

### 6.1 Business Objective Alignment

**Required**: Primary Goal: Predict Stock Price Targets (README.md line 46)

**Current Status**: Cell 1 includes "Business Objective" section

**ACTION REQUIRED**: Ensure alignment is clear throughout

### 6.2 Phase Workflow Description

**Required**: Each phase should have clear description of purpose, inputs, outputs

**ACTION REQUIRED**: Add comprehensive phase descriptions

---

## Summary of Required Changes

### Priority 1 (Critical - Structure):

1. ✅ Restructure section headers from numbered (1-10) to phase-based (9.1-9.8)
2. ✅ Add clear markdown phase headers
3. ✅ Add validation checkpoints for each phase
4. ✅ Consolidate Sections 8, 9, 10 into Phases 9.7 and 9.8

### Priority 2 (High - v1.2 Standards):

5. ✅ Verify uncertainty quantification implementation (quantile regression + conformal calibration)
6. ✅ Verify outlier safety rails implementation
7. ✅ Verify data split policy adherence
8. ✅ Verify standardized predictions schema output
9. ✅ Verify sector metrics persistence

### Priority 3 (Medium - API Alignment):

10. ✅ Verify Phase 9.1 API usage (prepare_phase91_data)
11. ✅ Verify Phase 9.3 API usage (build_features with presets)
12. ✅ Verify Phase 9.4 API usage (fit_classifier)
13. ✅ Verify Phase 9.5 API usage (fit_regressor with meta-features)
14. ✅ Verify standardized return types for all train_* calls

### Priority 4 (Low - Documentation):

15. ✅ Add comprehensive phase descriptions
16. ✅ Ensure business objective alignment is clear
17. ✅ Add phase workflow diagrams/descriptions

---

## Conclusion

The notebook has **good functional coverage** (all phases present) but needs **structural refactoring** to align with
the phase-based architecture specified in code_guidelines.md and improvement_plan.md.

**Main Actions**:

1. Restructure section headers to Phase 9.1-9.8
2. Verify v1.2 standards compliance
3. Ensure API usage follows specifications
4. Add validation checkpoints and phase descriptions
