# ML Workflow Guidelines

**Version**: 1.1  
**Last Updated**: 2025-12-14  
**Aligned with**: code_guidelines.md v1.11, ml-project-checklist.md

## Overview

This document provides comprehensive guidelines for the Finance ML Analytics Platform's 8-phase ML workflow (Phase
9.1-9.8) and 7-phase Portfolio Optimization workflow. It is based on analysis of the `ml_finance_model_main.ipynb`
workflow, outputs, and inspection results.

**Compliance Monitoring**: See `outputs/governance/ml_workflow_log.json` for runtime compliance tracking and issue
logging.

---

## Critical Issues Identified

### Data Processing Issues

1. **Zero Predictions**: Several stocks (PLTR, BAC, UBER, HD) have `y_pred = 0.0` indicating model prediction failures
2. **High Error Rates**: Prediction errors range from -100% to 597% (`pct_error`), suggesting calibration or model
   training issues
3. **Outlier Prevalence**: Data quality alerts show 5-10% outliers in critical columns like `market_cap`, `beta_*`,
   `analyst_rating`

### Implementation Gaps

1. **Non-Negativity Enforcement**: `ensure_nonnegative=False` used during training, allowing negative price predictions
2. **Missing Validation**: No post-prediction validation catching zero/negative predictions before export
3. **Calibration Issues**: `y_pred_calibrated` values differ significantly from raw predictions (e.g., BRKA: 1.2M →
   550K)

### Workflow Alignment Gaps

1. **Phase 9.5 → 9.6 Transition**: Model evaluation doesn't validate prediction distributions
2. **Quantile Monotonicity**: Some predictions may violate `pred_p10 < pred_p50 < pred_p90`
3. **Sector Bias**: Large cap stocks (BRKA) show extreme errors, suggesting sector-specific calibration needed

### New Issues Identified (2025-12-14)

1. **Cross-Validation GroupKFold Error** (Phase 9.4): Cross-validation fails with error "The number of folds must be of
   Integral type. GroupKFold(...) was passed." Root cause: `cross_validate_with_sector_stratification` passes GroupKFold
   object directly to `cross_val_score` instead of using `GroupKFold.split()` method.

2. **Perfect Model Scores Indicate Data Leakage** (Phase 9.5): Ridge model achieves R²=1.0 and MAE=0.0, which is
   impossible for real financial prediction. This strongly suggests target leakage in the feature engineering pipeline.

3. **Classification Overfitting** (Phase 9.4): LightGBM achieves F1=1.0000 during hyperparameter optimization, which is
   unrealistic for financial event classification. Requires validation on held-out test set.

### Overfitting Warning Signs

⚠️ **CRITICAL**: The following metrics indicate potential overfitting or data leakage:

| Metric      | Observed Value | Expected Range | Status      |
|-------------|----------------|----------------|-------------|
| Ridge R²    | 1.0000         | 0.60-0.85      | 🔴 CRITICAL |
| Ridge MAE   | 0.00           | >100           | 🔴 CRITICAL |
| LightGBM F1 | 1.0000         | 0.60-0.85      | 🟡 WARNING  |
| Lasso R²    | 1.0000         | 0.60-0.85      | 🔴 CRITICAL |

**Recommended Actions**:

1. Audit feature engineering pipeline for target leakage
2. Ensure temporal separation in train/test split
3. Validate on truly held-out test set
4. Review if `price_target` or related columns are included in features

---

## ML Workflow Guidelines (8-Phase Architecture)

### Phase 9.1: Unified ETL Pipeline

**Process Definition:**

- Single entry point: `etl_with_features(source, data_dir, feature_preset, return_metrics)`
- 11-stage pipeline: Extract → Normalize → Dtype Cast → Semantic Classification → Imputation → Semantic Transforms →
  Winsorization → Scaling → Feature Engineering → Post-Feature Imputation → **Schema Validation**
- 6-step imputation: Zero-fill → KNN → Price-based → Median → Categorical → Datetime
- Semantic column classification: 5 categories (price, market_value, ratio, percentage, count)
- **NEW (v1.11)**: Automated schema alignment validation (Stage 11)

**Acceptance Criteria:**

- ✅ Zero missing values after imputation
- ✅ 21 price columns preserved (never transformed)
- ✅ ETLMetrics returned with all tracking attributes
- ✅ Schema alignment score ≥ 95% (503 columns in COLUMN_SCHEMA)
- ✅ Unknown columns count ≤ 10
- ✅ No critical missing columns (ticker, sector, last_price, price_target)

**Success Metrics:**

| Metric                 | Target       | Validation                               |
|------------------------|--------------|------------------------------------------|
| Missing Values         | 0            | `df.isna().sum().sum() == 0`             |
| Price Columns          | 21 preserved | `assert_price_columns_preserved()`       |
| Row Count              | >6,000       | `len(df) > 6000`                         |
| Schema Alignment Score | ≥ 0.95       | `metrics.schema_alignment_score >= 0.95` |
| Unknown Columns        | ≤ 10         | `metrics.unknown_columns_count <= 10`    |
| Missing Expected       | ≤ 5          | `metrics.missing_expected_columns_count` |
| Dtype Mismatches       | 0            | `metrics.dtype_mismatches_count == 0`    |

**Validation Checkpoint:**

```python
assert not df.empty, "DataFrame must not be empty"
assert df.isna().sum().sum() == 0, "No missing values allowed"
assert 'ticker' in df.columns and 'sector' in df.columns

# NEW: Schema alignment validation (v1.11)
assert metrics.schema_alignment_score >= 0.95, \
    f"Schema alignment below 95%: {metrics.schema_alignment_score:.2%}"
assert metrics.unknown_columns_count <= 10, \
    f"Too many unknown columns: {metrics.unknown_columns_count}"
```

---

### Phase 9.2: Enhanced Exploratory Data Analysis

**Process Definition:**

- Statistical testing and sector/region benchmarking
- Correlation analysis with multicollinearity detection (threshold: 0.85)
- Distribution analysis and outlier detection per sector

**Acceptance Criteria:**

- ✅ EDA summary JSON exported to `outputs/eda/eda_summary.json`
- ✅ Data quality alerts documented with severity levels
- ✅ Benchmarking report per sector generated

**Success Metrics:**

| Metric            | Target                | Validation          |
|-------------------|-----------------------|---------------------|
| Correlation Pairs | <20 highly correlated | VIF analysis        |
| Outlier Rate      | <10% per column       | IQR × 2.5 threshold |
| Sector Coverage   | All 11 sectors        | Count validation    |

**Validation Checkpoint:**

```python
assert len(df['sector'].unique()) >= 10, "Minimum sector coverage"
assert (df.select_dtypes(include=[np.number]).describe().loc['count'] > 100).all()
```

---

### Phase 9.3: Advanced Feature Engineering

**Process Definition:**

- 196 features across 16 categories (Momentum, Valuation, Profitability, Quality/Risk, etc.)
- Sector-specific transformations with log-transforms for market values
- Feature presets: basic, momentum, quality, standard, comprehensive

**Acceptance Criteria:**

- ✅ ≥90% feature coverage (182/196 features)
- ✅ No NaN introduced by feature engineering
- ✅ Feature categories documented in `phase93_category_summary_statistics.json`

**Success Metrics:**

| Metric           | Target | Validation                     |
|------------------|--------|--------------------------------|
| Feature Coverage | ≥90%   | `get_phase93_coverage_stats()` |
| New Features     | 196    | Column count delta             |
| Execution Time   | <5s    | For 6,974 rows                 |

**Validation Checkpoint:**

```python
coverage_stats = get_phase93_coverage_stats(df)
assert sum(coverage_stats.values()) / 196 >= 0.90
```

---

### Phase 9.4: Multi-Class Event Classification

**Process Definition:**

- 13 label generation methods for event classification
- 5-class output: strong_negative, negative, neutral, positive, strong_positive
- Classification probabilities used as meta-features for regression

**Acceptance Criteria:**

- ✅ All 5 probability columns generated
- ✅ Probabilities sum to 1.0 per row
- ✅ Classification model accuracy >60%
- ⚠️ Cross-validation must complete without errors

**Success Metrics:**

| Metric        | Target        | Validation                                      |
|---------------|---------------|-------------------------------------------------|
| Accuracy      | >60%          | Cross-validation                                |
| F1 Score      | 0.55-0.90     | Macro average (⚠️ F1=1.0 indicates overfitting) |
| Class Balance | No class <10% | Distribution check                              |

**Known Issues (2025-12-14):**

⚠️ **GroupKFold Cross-Validation Error**: The `cross_validate_with_sector_stratification` function fails with:

```
TypeError: The number of folds must be of Integral type. GroupKFold(n_splits=5, ...) was passed.
```

**Fix Required**: Use `GroupKFold.split(X, y, groups)` method instead of passing GroupKFold object to `cross_val_score`:

```python
# INCORRECT (current):
cv_results = cross_val_score(model, X, y, cv=GroupKFold(n_splits=5))

# CORRECT (fix):
gkf = GroupKFold(n_splits=5)
cv_results = cross_val_score(model, X, y, cv=gkf.split(X, y, groups=sector_groups))
```

⚠️ **Overfitting Warning**: LightGBM achieved F1=1.0000 during hyperparameter optimization. Perfect scores are
unrealistic
for financial event classification and indicate:

- Possible data leakage from target-related features
- Insufficient regularization (max_depth=12 may be too deep)
- Need for held-out test set validation

**Validation Checkpoint:**

```python
prob_cols = ['event_prob_strong_negative', 'event_prob_negative',
             'event_prob_neutral', 'event_prob_positive', 'event_prob_strong_positive']
assert all(col in df.columns for col in prob_cols)
assert np.allclose(df[prob_cols].sum(axis=1), 1.0, atol=0.01)
# NEW: Overfitting check
assert f1_score < 0.95, "F1 >= 0.95 suggests overfitting - validate on held-out set"
```

---

### Phase 9.5: Sector-Optimized Regression

**Process Definition:**

- Stacking ensemble: RF + GradientBoosting + XGBoost + Ridge
- Quantile models (10th, 50th, 90th percentiles) for uncertainty
- Huber loss for robustness to outliers

**Acceptance Criteria:**

- ✅ Non-negative predictions enforced (prices ≥ 0)
- ✅ Quantile monotonicity: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- ✅ **CRITICAL**: No zero predictions for valid stocks
- ⚠️ R² must be realistic (0.60-0.90 range for financial data)

**Success Metrics:**

| Metric           | Target           | Validation                             |
|------------------|------------------|----------------------------------------|
| R²               | 0.60-0.90        | Test set (⚠️ R²=1.0 indicates leakage) |
| MAE              | >0, <15% of mean | Per sector (⚠️ MAE=0 is impossible)    |
| Zero Predictions | 0                | `(y_pred == 0).sum() == 0`             |

**Known Issues (2025-12-14):**

🔴 **CRITICAL - Data Leakage Detected**: Model comparison results show impossible metrics:

| Model                | MAE     | RMSE     | R²   | Status      |
|----------------------|---------|----------|------|-------------|
| Ridge                | 0.00    | 0.00     | 1.00 | 🔴 LEAKAGE  |
| Lasso                | 189.16  | 1469.69  | 1.00 | 🔴 LEAKAGE  |
| ExtraTrees           | 1190.29 | 31807.30 | 0.83 | ✅ Realistic |
| GradientBoosting     | 1260.77 | 37370.82 | 0.77 | ✅ Realistic |
| RandomForest         | 1738.64 | 42769.25 | 0.70 | ✅ Realistic |
| HistGradientBoosting | 2485.57 | 53915.17 | 0.52 | ✅ Realistic |

**Root Cause Analysis**:

- Ridge/Lasso achieving R²=1.0 with MAE=0.0 is mathematically impossible for real financial prediction
- Linear models are likely fitting on features that directly encode the target variable
- Possible culprits: `price_target`, `price_target_median`, or derived columns included in features

**Recommended Fix**:

```python
# Add target leakage check before model training
target_related_cols = ['price_target', 'price_target_median', 'price_target_high', 
                       'price_target_low', 'y_true', 'target']
leaky_features = [c for c in X_train.columns if any(t in c.lower() for t in target_related_cols)]
assert len(leaky_features) == 0, f"Target leakage detected: {leaky_features}"
```

**Validation Checkpoint:**

```python
assert (y_pred >= 0).all(), "Non-negativity constraint violated"
assert (y_pred > 0).sum() == len(y_pred), "Zero predictions detected"
assert (pred_p10 <= pred_p50).all() and (pred_p50 <= pred_p90).all()
# NEW: Leakage detection
assert r2_score < 0.95, "R² >= 0.95 suggests data leakage - audit features"
assert mae > 0, "MAE = 0 is impossible - check for target in features"
```

---

### Phase 9.6: Model Evaluation and Error Analysis

**Process Definition:**

- Sector-level metrics: MAE, RMSE, R², MAPE per sector
- Residual analysis and diagnostic plots
- Uncertainty calibration (80% prediction interval coverage)

**Acceptance Criteria:**

- ✅ Sector metrics exported to `regression_metrics_by_sector.csv`
- ✅ Calibration report with coverage statistics
- ✅ Error distribution analysis per market cap bucket

**Success Metrics:**

| Metric            | Target   | Validation            |
|-------------------|----------|-----------------------|
| Interval Coverage | 80% ± 5% | Conformal calibration |
| MAPE              | <25%     | Overall               |
| Sector R²         | >0.60    | All sectors           |

**Validation Checkpoint:**

```python
coverage = ((y_true >= pred_p10) & (y_true <= pred_p90)).mean()
assert 0.75 <= coverage <= 0.85, f"Coverage {coverage:.2%} outside bounds"
```

---

### Phase 9.7: Under/Overvalued Stock Identification

**Process Definition:**

- Mispricing score: `(Predicted_Target - Last_Price) / Last_Price`
- Top-N rankings per sector and region
- Analyst comparison analytics

**Acceptance Criteria:**

- ✅ Mispricing scores calculated for all stocks
- ✅ Rankings exported with confidence intervals
- ✅ Analyst disagreement analysis documented

**Success Metrics:**

| Metric            | Target     | Validation          |
|-------------------|------------|---------------------|
| Ranking Coverage  | 100%       | All valid tickers   |
| Confidence Score  | Calculated | For top 50          |
| Disagreement Rate | Documented | vs. analyst targets |

---

### Phase 9.8: Comprehensive Analytics and Reporting

**Process Definition:**

- Model governance: model cards, lineage tracking
- Interactive dashboards and visualizations
- Executive reporting with quality alerts

**Acceptance Criteria:**

- ✅ Standardized predictions schema (Section 11)
- ✅ All required output artifacts generated
- ✅ Model version tracked (`v9_9`)

**Required Output Columns:**

```
ticker, isin, sector, region, last_price, y_true, y_pred, 
y_pred_calibrated, pred_p10, pred_p50, pred_p90, interval_width,
abs_error, pct_error, model_version, snapshot_date
```

---

## Portfolio Optimization Workflow (7-Phase)

| Phase | Description           | Key Function                      | Validation                    |
|-------|-----------------------|-----------------------------------|-------------------------------|
| 1     | Stock Selection       | `select_portfolio_candidates()`   | ≥150 candidates               |
| 2     | ML Return Prediction  | `train_linear_return_predictor()` | Returns bounded [-0.50, 0.29] |
| 3     | Advanced Optimization | `optimize_black_litterman()`      | Sharpe < 3.0                  |
| 4     | Risk Management       | `calculate_expected_shortfall()`  | CVaR calculated               |
| 5     | Backtesting           | `run_vectorized_backtest()`       | Walk-forward validated        |
| 6     | Dashboards            | `PortfolioRebalanceWidget`        | Interactive plots             |
| 7     | ML Validation         | `validate_expected_returns()`     | Realistic bounds              |

---

## Recommended Fixes for Critical Issues

### 1. Zero Predictions Fix

```python
# Add validation after prediction
if (y_pred == 0).any():
    zero_indices = np.where(y_pred == 0)[0]
    logger.warning(f"Zero predictions for {len(zero_indices)} stocks")
    # Apply fallback: use last_price * sector_median_ratio
    y_pred[zero_indices] = df.loc[zero_indices, 'last_price'] * 1.05
```

### 2. Non-Negativity Enforcement

```python
# Change from ensure_nonnegative=False to True, OR add post-processing:
y_pred = np.maximum(y_pred, 0.01)  # Minimum 1 cent
```

### 3. High Error Validation

```python
# Add pct_error bounds check
extreme_errors = (np.abs(pct_error) > 100).sum()
if extreme_errors > len(pct_error) * 0.05:
    logger.warning(f">{extreme_errors} predictions with >100% error")
```

---

## Configuration Constants (Single Source of Truth)

```python
# Target Configuration
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'

# Data Split
TEST_SIZE = 0.2
CV_FOLDS = 5

# Quantiles
QUANTILES = [0.1, 0.5, 0.9]

# Sector Analysis
MIN_SECTOR_SAMPLES = 20

# Outlier Detection
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# Portfolio Constraints
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10
MAX_EXPECTED_RETURN = 0.29
MIN_EXPECTED_RETURN = -0.50

# Reproducibility
RANDOM_SEED = 42
MODEL_VERSION = 'v9_9'
```

---

## Test Coverage Requirements

| Phase     | Test Module                               | Min Tests | Coverage |
|-----------|-------------------------------------------|-----------|----------|
| 9.1       | `test_etl_unified_pipeline.py`            | 63        | ≥85%     |
| 9.3       | `test_data_types_detection.py`            | 9         | ≥80%     |
| 9.4       | `test_classification.py`                  | TBD       | ≥75%     |
| 9.5       | `test_phase95_nonnegative_predictions.py` | TBD       | ≥80%     |
| 9.6       | `test_finance_ml_eval.py`                 | TBD       | ≥75%     |
| Portfolio | `test_portfolio_*.py`                     | 23        | ≥80%     |

---

## Document References

- **Code Guidelines**: `docs/code_guidelines.md` v1.10
- **ML Project Checklist**: `reference material/ml-project-checklist.md`
- **Phase Implementation Plans**: `docs/improvement_plan/phase_9.*.md`
- **Test Suite**: `tests/test_*.py` (86 modules, 85+ tests)

---

## Appendix: DataFrame Stage Naming Convention

This project follows a **6-stage ML pipeline** (Section 8.2 of code_guidelines.md v1.11):

### Core Pipeline Stages (Required)

1. **`all_stocks_preprocessed`**: ETL pipeline output (~655 columns) - normalization, validation, imputation, scaling
2. **`all_stocks_features`**: Feature engineering output (~656 columns) - Phase 9.3 categories
3. **`all_stocks_classification`**: Classification meta-features (~663 columns) - event probabilities
4. **`all_stocks_enhanced`**: Final regression-ready dataset (~928 columns) - with interaction terms

### Optional ML-Specific Stages

5. **`all_stocks_selected`**: After feature selection (~392 columns) - importance/correlation filtering
6. **`all_stocks_balanced`**: SMOTE-balanced for classification training

### Auxiliary DataFrames (Not Pipeline Stages)

- **`all_stocks_multilabel`**: Multi-label target matrix (8 label columns only)

### Deprecated (Handled by ETL)

- ~~`all_stocks_typed`~~, ~~`all_stocks_winsorized`~~, ~~`all_stocks_imputed`~~, ~~`all_stocks_scaled`~~ → Use ETL
  pipeline

---

*This document should be updated as the workflow evolves. Version control is maintained alongside code_guidelines.md.*
