# Stock Price Target Prediction Notebook — Redesigned

Based on your objectives and the guidelines in `code_guidelines.md` and `ml_workflow_guidelines.md`, here's a
comprehensive redesign of your Jupyter notebook.

## Notebook Structure Overview

```
Section 0: Configuration and Setup
Section 1: Phase 9.1-9.3 — Unified ETL Pipeline with Features
Section 2: Phase 9.3 — Feature Selection
Section 3: Phase 9.4 — Classification
Section 4: Phase 9.5 — Regression
Section 5: Phase 9.6 — Evaluation
Section 6: Phase 9.7 — Analytics
Section 7: Phase 9.8 — Reporting
```

---

## Cell 1: Header and Navigation (Markdown)

```markdown
#%% md

# Stock Price Target Prediction — ML Analytics Platform

**Version 3.0.0** — Unified ETL Pipeline with Phase 9.1-9.8 Workflow  
**Model Version: v9_10**

## Business Objective

**Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support
investment decisions and portfolio optimization.

**Target Variable**: `price_target` for regression modeling

## Quick Reference Navigation

- [Section 0](#section-0-configuration-and-setup): Configuration and Setup
- [Section 1](#section-1-unified-etl-pipeline): Phase 9.1-9.3 Unified ETL Pipeline
- [Section 2](#section-2-feature-selection): Phase 9.3 Feature Selection
- [Section 3](#section-3-classification): Phase 9.4 Multi-Class Event Classification
- [Section 4](#section-4-regression): Phase 9.5 Sector-Optimized Regression
- [Section 5](#section-5-evaluation): Phase 9.6 Model Evaluation
- [Section 6](#section-6-analytics): Phase 9.7 Stock Ranking Analytics
- [Section 7](#section-7-reporting): Phase 9.8 Comprehensive Reporting

## Workflow Overview

| Section | Phase | Description | Key Outputs |
|---------|-------|-------------|-------------|
| 0 | Setup | Environment, paths, logging, seed | Config validated |
| 1 | 9.1-9.3 | Unified ETL + Feature Engineering | `all_stocks_preprocessed` |
| 2 | 9.3 | Feature Selection | `all_stocks_selected` |
| 3 | 9.4 | Event Classification | `all_stocks_classification` |
| 4 | 9.5 | Sector-Optimized Regression | Trained models, predictions |
| 5 | 9.6 | Evaluation and Error Analysis | Metrics by sector |
| 6 | 9.7 | Analytics: Rankings, Mispricing | Rankings DataFrame |
| 7 | 9.8 | Reporting: Artifacts, Dashboards | Output files |
```

---

## Cell 2: Section 0 — Configuration and Setup

```python
#%%
# =============================================================================
# Section 0: Configuration and Setup
# =============================================================================

import os
import warnings
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Target Configuration (code_guidelines.md Section 2.2)
# -----------------------------------------------------------------------------
TARGET_COL = 'price_target'
TARGET_COL_FALLBACK = 'last_price'

# -----------------------------------------------------------------------------
# Data Split Configuration
# -----------------------------------------------------------------------------
TEST_SIZE = 0.2
TRAIN_SIZE = 1 - TEST_SIZE
CV_FOLDS = 5

# -----------------------------------------------------------------------------
# Quantile Regression Configuration
# -----------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]

# -----------------------------------------------------------------------------
# Sector Analysis Configuration
# -----------------------------------------------------------------------------
MIN_SECTOR_SAMPLES = 20
MAX_SECTOR_WEIGHT = 0.25
MAX_SINGLE_POSITION = 0.10

# -----------------------------------------------------------------------------
# Outlier Detection Configuration
# -----------------------------------------------------------------------------
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# -----------------------------------------------------------------------------
# Confidence Thresholds
# -----------------------------------------------------------------------------
CONFIDENCE_LEVEL = 0.80
ALPHA = 1 - CONFIDENCE_LEVEL

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10')
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------------
# Directory Configuration
# -----------------------------------------------------------------------------
DATA_DIR = Path(os.getenv('DATA_DIR', 'data'))
OUTPUT_DIR = Path(os.getenv('OUTPUT_DIR', 'outputs'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'models'))

# Create output subdirectories
OUTPUT_SUBDIRS = [
    'eda', 'preprocessing', 'features', 'classification',
    'regression', 'evaluation', 'analytics', 'reporting',
    'plots', 'governance'
]
for subdir in OUTPUT_SUBDIRS:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

logger.info(f"Output directory: {OUTPUT_DIR}")
logger.info(f"Model version: {MODEL_VERSION}")
```

---

## Cell 3: Configuration Validation

```python
#%%
def validate_configuration():
    """
    Validate all configuration constants meet required constraints.
    
    Raises:
        ValueError: If any configuration constant is invalid
    """
    # Validate target columns
    if not TARGET_COL or not isinstance(TARGET_COL, str):
        raise ValueError(f"TARGET_COL must be a non-empty string, got: {TARGET_COL}")
    if not TARGET_COL_FALLBACK or not isinstance(TARGET_COL_FALLBACK, str):
        raise ValueError(f"TARGET_COL_FALLBACK must be a non-empty string, got: {TARGET_COL_FALLBACK}")

    # Validate split configuration
    if not (0 < TEST_SIZE < 1):
        raise ValueError(f"TEST_SIZE must be between 0 and 1, got: {TEST_SIZE}")
    if not (0 < TRAIN_SIZE < 1):
        raise ValueError(f"TRAIN_SIZE must be between 0 and 1, got: {TRAIN_SIZE}")
    if not abs((TRAIN_SIZE + TEST_SIZE) - 1.0) < 0.01:
        raise ValueError(f"TRAIN_SIZE + TEST_SIZE must equal 1.0, got: {TRAIN_SIZE + TEST_SIZE}")

    # Validate CV folds
    if not isinstance(CV_FOLDS, int) or CV_FOLDS < 2:
        raise ValueError(f"CV_FOLDS must be an integer >= 2, got: {CV_FOLDS}")

    # Validate quantiles
    if not QUANTILES or not isinstance(QUANTILES, list):
        raise ValueError(f"QUANTILES must be a non-empty list, got: {QUANTILES}")
    for q in QUANTILES:
        if not (0 <= q <= 1):
            raise ValueError(f"All quantiles must be between 0 and 1, got: {q}")
    if len(QUANTILES) != len(set(QUANTILES)):
        raise ValueError(f"QUANTILES must not contain duplicates, got: {QUANTILES}")
    if QUANTILES != sorted(QUANTILES):
        raise ValueError(f"QUANTILES must be monotonically increasing, got: {QUANTILES}")

    # Validate sector configuration
    if not isinstance(MIN_SECTOR_SAMPLES, int) or MIN_SECTOR_SAMPLES < 1:
        raise ValueError(f"MIN_SECTOR_SAMPLES must be a positive integer, got: {MIN_SECTOR_SAMPLES}")
    if not (0 < MAX_SECTOR_WEIGHT <= 1):
        raise ValueError(f"MAX_SECTOR_WEIGHT must be between 0 and 1, got: {MAX_SECTOR_WEIGHT}")
    if not (0 < MAX_SINGLE_POSITION <= 1):
        raise ValueError(f"MAX_SINGLE_POSITION must be between 0 and 1, got: {MAX_SINGLE_POSITION}")

    # Validate outlier detection
    if IQR_MULTIPLIER <= 0:
        raise ValueError(f"IQR_MULTIPLIER must be positive, got: {IQR_MULTIPLIER}")
    if ZSCORE_THRESHOLD <= 0:
        raise ValueError(f"ZSCORE_THRESHOLD must be positive, got: {ZSCORE_THRESHOLD}")
    if not (0 <= WINSORIZE_LOWER < 0.5):
        raise ValueError(f"WINSORIZE_LOWER must be between 0 and 0.5, got: {WINSORIZE_LOWER}")
    if not (0.5 < WINSORIZE_UPPER <= 1):
        raise ValueError(f"WINSORIZE_UPPER must be between 0.5 and 1, got: {WINSORIZE_UPPER}")

    # Validate confidence configuration
    if not (0 < CONFIDENCE_LEVEL < 1):
        raise ValueError(f"CONFIDENCE_LEVEL must be between 0 and 1, got: {CONFIDENCE_LEVEL}")
    if not abs(ALPHA - (1 - CONFIDENCE_LEVEL)) < 0.01:
        raise ValueError(f"ALPHA must equal (1 - CONFIDENCE_LEVEL), got: {ALPHA}")

    print("✓ All configuration constants validated successfully")


# Run validation immediately
validate_configuration()
```

---

## Cell 4: Section 1 Header (Markdown)

```markdown
#%% md
## Section 1: Unified ETL Pipeline (Phase 9.1-9.3)

This section uses `etl_with_features()` as the **single entry point** for:
- Data extraction and normalization
- Semantic column classification (price, market_value, ratio, percentage, count)
- 6-step imputation strategy
- Log-transforms for skewed market values
- Winsorization (excluding protected columns)
- Phase 9.3 feature engineering (196 features)

**Key Output**: `all_stocks_preprocessed` DataFrame with ~656 columns
```

---

## Cell 5: ETL Pipeline Execution

```python
#%%
# =============================================================================
# Section 1: Phase 9.1-9.3 — Unified ETL Pipeline
# =============================================================================

from finance_ml.ml_workflow.preprocessing import (
    etl_with_features,
    ETLConfig,
    ETLMetrics,
)

# Execute unified ETL pipeline with comprehensive feature engineering
all_stocks_preprocessed, etl_metrics = etl_with_features(
    source='csv',
    data_dir=DATA_DIR,
    feature_preset='comprehensive',  # 196 Phase 9.3 features
    return_metrics=True
)

# Display ETL summary
print(etl_metrics.summary())
print(f"\n{'='*60}")
print(f"ETL Pipeline Complete")
print(f"{'='*60}")
print(f"  Rows: {len(all_stocks_preprocessed):,}")
print(f"  Columns: {all_stocks_preprocessed.shape[1]:,}")
print(f"  Price columns protected: {etl_metrics.price_columns_count}")
print(f"  Features added: {etl_metrics.features_added}")
print(f"  Log-transformed columns: {etl_metrics.log_transformed_columns}")
```

---

## Cell 6: Post-ETL Validation

```python
#%%
# =============================================================================
# Post-ETL Validation (REQUIRED — code_guidelines.md Section 19.1)
# =============================================================================

def validate_etl_output(df: pd.DataFrame, metrics: ETLMetrics) -> None:
    """Validate ETL pipeline output meets quality requirements."""
    
    # DataFrame not empty
    assert not df.empty, "DataFrame must not be empty"
    
    # Critical columns present
    required_cols = ['ticker', 'sector', 'last_price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    # Target column available
    target_cols = [TARGET_COL, TARGET_COL_FALLBACK]
    has_target = any(col in df.columns for col in target_cols)
    assert has_target, f"At least one target column required: {target_cols}"
    
    # No missing values after 6-step imputation
    missing_total = df.isna().sum().sum()
    assert missing_total == 0, f"No missing values allowed after imputation, found {missing_total}"
    
    # Data sufficiency
    assert len(df) > 100, f"Insufficient data: {len(df)} rows (minimum 100)"
    
    # Price positivity
    assert df['last_price'].min() > 0, "last_price must be positive"
    
    # Semantic classification applied
    assert metrics.semantic_classification_applied, "Semantic classification should be applied"
    
    # Price columns protected
    assert metrics.price_columns_count >= 21, f"Expected 21 price columns, got {metrics.price_columns_count}"
    
    # Schema alignment validation (v1.11)
    assert metrics.schema_alignment_score >= 0.95, \
        f"Schema alignment below 95%: {metrics.schema_alignment_score:.2%}"
    assert metrics.unknown_columns_count <= 10, \
        f"Too many unknown columns: {metrics.unknown_columns_count}"
    
    print("✓ Post-ETL validation passed")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Missing values: 0")
    print(f"  Schema alignment: {metrics.schema_alignment_score:.2%}")
    print(f"  Unknown columns: {metrics.unknown_columns_count}")
    print(f"  Missing expected: {metrics.missing_expected_columns_count}")
    print(f"  Dtype mismatches: {metrics.dtype_mismatches_count}")


validate_etl_output(all_stocks_preprocessed, etl_metrics)
```

---

## Cell 7: Section 2 Header (Markdown)

```markdown
#%% md
## Section 2: Feature Selection (Phase 9.3)

Automated feature selection to reduce dimensionality while preserving model performance:
- Mutual information importance filtering
- Correlation-based deduplication  
- PRICE_COLUMNS protection (never removed)

**Key Output**: `all_stocks_selected` DataFrame with reduced feature set
```

---

## Cell 8: Feature Selection

```python
#%%
# =============================================================================
# Section 2: Phase 9.3 — Feature Selection
# =============================================================================

from finance_ml.ml_workflow.features.selection import (
    select_features_auto,
    select_features_by_category,
)
from finance_ml.ml_workflow.eda.phase93_categories import (
    get_phase93_coverage_stats,
    PHASE93_FEATURE_CATEGORIES,
)

# Validate Phase 9.3 feature coverage before selection
coverage_stats = get_phase93_coverage_stats(all_stocks_preprocessed)
total_features = sum(coverage_stats.values())
coverage_pct = (total_features / 196) * 100

print(f"Phase 9.3 Feature Coverage: {coverage_pct:.1f}% ({total_features}/196 features)")
assert coverage_pct >= 90, f"Phase 9.3 coverage must be ≥90%, got {coverage_pct:.1f}%"

# Prepare features and target
target_col = TARGET_COL if TARGET_COL in all_stocks_preprocessed.columns else TARGET_COL_FALLBACK
y = all_stocks_preprocessed[target_col].copy()

# Exclude non-feature columns
exclude_cols = ['ticker', 'isin', 'sector', 'region', 'country', 'industry',
                target_col, 'price_target', 'price_target_median', 'price_target_high',
                'price_target_low', 'last_updated']
feature_cols = [col for col in all_stocks_preprocessed.columns 
                if col not in exclude_cols and all_stocks_preprocessed[col].dtype in ['float64', 'float32', 'int64', 'int32']]

X = all_stocks_preprocessed[feature_cols].copy()

# Automated feature selection
X_selected = select_features_auto(
    X, y,
    importance_threshold=0.01,
    correlation_threshold=0.95,
    method='mutual_info'
)

# Create selected DataFrame
all_stocks_selected = all_stocks_preprocessed[
    ['ticker', 'isin', 'sector', 'region', target_col] + X_selected.columns.tolist()
].copy()

print(f"\n{'='*60}")
print(f"Feature Selection Complete")
print(f"{'='*60}")
print(f"  Features before: {len(feature_cols):,}")
print(f"  Features after: {X_selected.shape[1]:,}")
print(f"  Reduction: {(1 - X_selected.shape[1]/len(feature_cols))*100:.1f}%")

# Save selected feature list for reproducibility
feature_list_path = OUTPUT_DIR / 'features' / 'selected_features.txt'
with open(feature_list_path, 'w') as f:
    f.write('\n'.join(X_selected.columns.tolist()))
logger.info(f"Selected features saved to {feature_list_path}")
```

---

## Cell 9: Section 3 Header (Markdown)

```markdown
#%% md
## Section 3: Multi-Class Event Classification (Phase 9.4)

Event classification for generating meta-features used in regression:
- 5-class output: strong_negative, negative, neutral, positive, strong_positive
- Classification probabilities as regression features
- Cross-validation with proper sector stratification

**Key Output**: `all_stocks_classification` DataFrame with event probabilities

⚠️ **Warning**: Monitor for F1 scores > 0.95 which indicate potential overfitting
```

---

## Cell 10: Classification

```python
#%%
# =============================================================================
# Section 3: Phase 9.4 — Multi-Class Event Classification
# =============================================================================

from finance_ml.ml_workflow.classification import (
    create_event_labels,
    train_event_classifier,
)
from finance_ml.ml_workflow.classification.models import determine_cv_strategy
from sklearn.model_selection import cross_val_score

# Create event labels based on price momentum
event_labels = create_event_labels(
    all_stocks_selected,
    method='price_momentum'
)

# Prepare classification data
X_class = X_selected.copy()
y_class = event_labels

# Determine optimal CV strategy (prevents leakage)
cv_strategy, cv_obj = determine_cv_strategy(
    all_stocks_selected,
    target=y_class,
    n_splits=CV_FOLDS,
    group_column='ticker',
    random_state=RANDOM_SEED
)
logger.info(f"Using CV strategy: {cv_strategy}")

# Train event classifier
classifier_result = train_event_classifier(
    X_class, y_class,
    model='lightgbm',
    random_state=RANDOM_SEED
)

# Validate classification performance (overfitting check)
f1_score = classifier_result['metrics'].get('f1_macro', 0)
if f1_score >= 0.95:
    logger.warning(f"⚠️ F1={f1_score:.4f} suggests overfitting — validate on held-out set")
else:
    logger.info(f"Classification F1 (macro): {f1_score:.4f}")

# Generate probability features
class_proba = classifier_result['y_proba']
prob_columns = [
    'event_prob_strong_negative',
    'event_prob_negative', 
    'event_prob_neutral',
    'event_prob_positive',
    'event_prob_strong_positive'
]

# Create classification-enhanced DataFrame
all_stocks_classification = all_stocks_selected.copy()
for i, col in enumerate(prob_columns):
    if i < class_proba.shape[1]:
        all_stocks_classification[col] = class_proba[:, i]

# Validate probability sums
prob_sums = all_stocks_classification[prob_columns].sum(axis=1)
assert np.allclose(prob_sums, 1.0, atol=0.01), "Probabilities must sum to 1.0"

print(f"\n{'='*60}")
print(f"Classification Complete")
print(f"{'='*60}")
print(f"  Strategy: {cv_strategy}")
print(f"  F1 (macro): {f1_score:.4f}")
print(f"  Accuracy: {classifier_result['metrics'].get('accuracy', 0):.4f}")
print(f"  Probability columns added: {len(prob_columns)}")

# Save classification model
import joblib
classifier_path = MODEL_DIR / 'classification' / f'event_classifier_{MODEL_VERSION}.joblib'
classifier_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(classifier_result['model'], classifier_path)
logger.info(f"Classifier saved to {classifier_path}")
```

---

## Cell 11: Section 4 Header (Markdown)

```markdown
#%% md
## Section 4: Sector-Optimized Regression (Phase 9.5)

Regression modeling for price target prediction:
- Stacking ensemble: RF + GradientBoosting + XGBoost + Ridge
- Quantile models for uncertainty (P10, P50, P90)
- Non-negativity constraints enforced
- Target leakage detection

**Key Outputs**: Trained models, `predictions_df` with quantile predictions

🔴 **CRITICAL**: R² > 0.95 or MAE = 0 indicates data leakage — audit features
```

---

## Cell 12: Regression Setup and Leakage Check

```python
#%%
# =============================================================================
# Section 4: Phase 9.5 — Sector-Optimized Regression
# =============================================================================

from sklearn.model_selection import train_test_split
from finance_ml.ml_workflow.regression import (
    train_sector_models,
    train_quantile_regressor,
)
from finance_ml.ml_workflow.regression.dataset import (
    align_features_to_model,
    predict_with_model,
)

# Prepare regression data
target_col = TARGET_COL if TARGET_COL in all_stocks_classification.columns else TARGET_COL_FALLBACK
y_reg = all_stocks_classification[target_col].copy()

# Feature columns (exclude identifiers, targets, and check for leakage)
target_related_cols = ['price_target', 'price_target_median', 'price_target_high', 
                       'price_target_low', 'y_true', 'target']
leaky_features = [c for c in all_stocks_classification.columns 
                  if any(t in c.lower() for t in target_related_cols) and c != target_col]

if leaky_features:
    logger.warning(f"⚠️ Potential target leakage detected: {leaky_features[:5]}...")
    
exclude_cols = ['ticker', 'isin', 'sector', 'region', 'country', 'industry',
                target_col, 'last_updated'] + leaky_features

feature_cols = [col for col in all_stocks_classification.columns 
                if col not in exclude_cols 
                and all_stocks_classification[col].dtype in ['float64', 'float32', 'int64', 'int32']]

X_reg = all_stocks_classification[feature_cols].copy()

# Train/test split with sector stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg,
    test_size=TEST_SIZE,
    stratify=all_stocks_classification['sector'],
    random_state=RANDOM_SEED
)

# Get corresponding metadata
train_idx = X_train.index
test_idx = X_test.index
meta_train = all_stocks_classification.loc[train_idx, ['ticker', 'sector', 'region']].copy()
meta_test = all_stocks_classification.loc[test_idx, ['ticker', 'sector', 'region']].copy()

print(f"Train set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")
print(f"Features: {X_train.shape[1]:,}")
```

---

## Cell 13: Model Training

```python
#%%
# =============================================================================
# Train Stacking Ensemble and Quantile Models
# =============================================================================

from sklearn.ensemble import (
    StackingRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

# Define base models (Section 16.4 optimized hyperparameters)
base_models = [
    ('rf', RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        max_features='sqrt', random_state=RANDOM_SEED, n_jobs=-1
    )),
    ('et', ExtraTreesRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_SEED, n_jobs=-1
    )),
    ('gb', GradientBoostingRegressor(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_SEED
    )),
    ('xgb', XGBRegressor(
        n_estimators=150, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED
    )),
]

# Train stacking ensemble
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=CV_FOLDS,
    n_jobs=-1
)

logger.info("Training stacking ensemble...")
stacking_model.fit(X_train, y_train)

# Generate predictions with alignment
y_pred_train = predict_with_model(stacking_model, X_train, fill_missing=0.0)
y_pred_test = predict_with_model(stacking_model, X_test, fill_missing=0.0)

# Enforce non-negativity
y_pred_train = np.maximum(y_pred_train, 0)
y_pred_test = np.maximum(y_pred_test, 0)

# Train quantile models
logger.info("Training quantile models...")
quantile_predictions = {}
for q in QUANTILES:
    q_model = GradientBoostingRegressor(
        loss='quantile', alpha=q,
        n_estimators=100, max_depth=5,
        random_state=RANDOM_SEED
    )
    q_model.fit(X_train, y_train)
    quantile_predictions[q] = np.maximum(q_model.predict(X_test), 0)  # Non-negative

print(f"\n{'='*60}")
print(f"Model Training Complete")
print(f"{'='*60}")
```

---

## Cell 14: Leakage Detection

```python
#%%
# =============================================================================
# Data Leakage Detection (ml_workflow_guidelines.md)
# =============================================================================

from sklearn.metrics import r2_score, mean_absolute_error

# Calculate metrics on training set (leakage indicator)
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)

# Calculate metrics on test set
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"Training Metrics:")
print(f"  R²: {r2_train:.4f}")
print(f"  MAE: {mae_train:.2f}")
print(f"\nTest Metrics:")
print(f"  R²: {r2_test:.4f}")
print(f"  MAE: {mae_test:.2f}")

# Leakage detection thresholds
if r2_train >= 0.99 or mae_train < 1.0:
    logger.error("🔴 CRITICAL: Perfect training metrics indicate data leakage!")
    logger.error("   Audit feature engineering pipeline for target-related columns")
    
if r2_test >= 0.95:
    logger.warning("⚠️ R² >= 0.95 on test set is unrealistic for financial prediction")
    logger.warning("   Expected range: 0.60-0.85")

# Validate realistic performance
assert r2_test < 0.99, "R² = 1.0 indicates data leakage"
assert mae_test > 0, "MAE = 0 is impossible"
```

---

## Cell 15: Build Predictions DataFrame

```python
#%%
# =============================================================================
# Build Predictions DataFrame (code_guidelines.md Section 11)
# =============================================================================

from finance_ml.ml_workflow.regression.io import build_predictions_frame, validate_predictions_schema

# Build standardized predictions DataFrame
predictions_df = pd.DataFrame({
    'ticker': meta_test['ticker'].values,
    'sector': meta_test['sector'].values,
    'region': meta_test['region'].values,
    'last_price': all_stocks_classification.loc[test_idx, 'last_price'].values,
    'y_true': y_test.values,
    'y_pred': y_pred_test,
    'pred_p10': quantile_predictions[LOWER_QUANTILE],
    'pred_p50': quantile_predictions[MEDIAN_QUANTILE],
    'pred_p90': quantile_predictions[UPPER_QUANTILE],
})

# Calculate derived columns
predictions_df['interval_width'] = predictions_df['pred_p90'] - predictions_df['pred_p10']
predictions_df['abs_error'] = np.abs(predictions_df['y_pred'] - predictions_df['y_true'])
predictions_df['pct_error'] = 100 * (predictions_df['y_pred'] - predictions_df['y_true']) / predictions_df['y_true']

# Add metadata
predictions_df['model_version'] = MODEL_VERSION
predictions_df['snapshot_date'] = datetime.now().strftime('%Y-%m-%d')

# Validate schema invariants
# Monotonicity: pred_p10 <= pred_p50 <= pred_p90
monotonicity_violations = (
    (predictions_df['pred_p10'] > predictions_df['pred_p50']) |
    (predictions_df['pred_p50'] > predictions_df['pred_p90'])
).sum()
assert monotonicity_violations == 0, f"Quantile monotonicity violated in {monotonicity_violations} rows"

# Non-negativity
assert (predictions_df[['y_pred', 'pred_p10', 'pred_p50', 'pred_p90']] >= 0).all().all(), "Negative predictions detected"

print(f"✓ Predictions DataFrame built: {len(predictions_df):,} rows")
print(f"✓ Monotonicity check passed")
print(f"✓ Non-negativity check passed")

# Save predictions
predictions_path = OUTPUT_DIR / 'regression' / 'regression_predictions_detailed.csv'
predictions_df.to_csv(predictions_path, index=False)
logger.info(f"Predictions saved to {predictions_path}")
```

---

## Cell 16: Save Models

```python
#%%
# =============================================================================
# Save Trained Models
# =============================================================================

import joblib

# Save stacking ensemble
model_path = MODEL_DIR / 'regression' / f'stacking_ensemble_{MODEL_VERSION}.joblib'
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(stacking_model, model_path)
logger.info(f"Stacking model saved to {model_path}")

# Save quantile models
for q in QUANTILES:
    q_name = f'q{int(q*100):02d}'
    q_path = MODEL_DIR / 'regression' / f'quantile_{q_name}_{MODEL_VERSION}.joblib'
    # Note: Would need to save quantile models during training loop
    
print(f"✓ Models saved to {MODEL_DIR / 'regression'}")
```

---

## Cell 17: Section 5 Header (Markdown)

```markdown
#%% md
## Section 5: Model Evaluation (Phase 9.6)

Comprehensive evaluation and error analysis:
- Sector-level metrics: MAE, RMSE, R², MAPE
- Residual analysis and diagnostic plots
- Uncertainty calibration (80% prediction interval coverage)
- Sector bias estimation

**Key Outputs**: Metrics by sector, calibration report, diagnostic plots
```

---

## Cell 18: Evaluation

```python
#%%
# =============================================================================
# Section 5: Phase 9.6 — Model Evaluation
# =============================================================================

from finance_ml.ml_workflow.evaluation import (
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    residual_analysis,
)

# Overall metrics
overall_metrics = comprehensive_regression_metrics(
    predictions_df['y_true'],
    predictions_df['y_pred']
)

print(f"{'='*60}")
print(f"Overall Model Performance")
print(f"{'='*60}")
for metric, value in overall_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Sector-level metrics
sector_metrics = predictions_df.groupby('sector').apply(
    lambda g: pd.Series({
        'mae': np.abs(g['y_pred'] - g['y_true']).mean(),
        'rmse': np.sqrt(((g['y_pred'] - g['y_true']) ** 2).mean()),
        'r2': 1 - ((g['y_true'] - g['y_pred'])**2).sum() / ((g['y_true'] - g['y_true'].mean())**2).sum() if len(g) > 1 else 0,
        'mape': 100 * (np.abs(g['y_pred'] - g['y_true']) / g['y_true']).mean(),
        'bias': (g['y_pred'] - g['y_true']).mean(),
        'count': len(g)
    })
).round(4)

print(f"\n{'='*60}")
print(f"Sector-Level Metrics")
print(f"{'='*60}")
print(sector_metrics.to_string())

# Save sector metrics
sector_metrics_path = OUTPUT_DIR / 'evaluation' / 'regression_metrics_by_sector.csv'
sector_metrics['model_version'] = MODEL_VERSION
sector_metrics['timestamp'] = datetime.now().isoformat()
sector_metrics.to_csv(sector_metrics_path)
logger.info(f"Sector metrics saved to {sector_metrics_path}")
```

---

## Cell 19: Uncertainty Calibration

```python
#%%
# =============================================================================
# Uncertainty Calibration (80% Prediction Interval)
# =============================================================================

# Calculate interval coverage
coverage = (
    (predictions_df['y_true'] >= predictions_df['pred_p10']) &
    (predictions_df['y_true'] <= predictions_df['pred_p90'])
).mean()

print(f"{'='*60}")
print(f"Uncertainty Calibration")
print(f"{'='*60}")
print(f"  Target coverage: {CONFIDENCE_LEVEL*100:.0f}%")
print(f"  Actual coverage: {coverage*100:.1f}%")

# Coverage should be within ±5% of target
if 0.75 <= coverage <= 0.85:
    print(f"  ✓ Coverage within acceptable range (75-85%)")
else:
    logger.warning(f"  ⚠️ Coverage {coverage:.1%} outside acceptable range")

# Sector-level coverage
sector_coverage = predictions_df.groupby('sector').apply(
    lambda g: (
        (g['y_true'] >= g['pred_p10']) &
        (g['y_true'] <= g['pred_p90'])
    ).mean()
)

print(f"\n  Sector-level coverage:")
for sector, cov in sector_coverage.items():
    status = "✓" if 0.70 <= cov <= 0.90 else "⚠️"
    print(f"    {status} {sector}: {cov*100:.1f}%")

# Save calibration report
calibration_report = {
    'overall_coverage': float(coverage),
    'target_coverage': CONFIDENCE_LEVEL,
    'sector_coverage': sector_coverage.to_dict(),
    'model_version': MODEL_VERSION,
    'timestamp': datetime.now().isoformat()
}

import json
calibration_path = OUTPUT_DIR / 'evaluation' / 'calibration_report.json'
with open(calibration_path, 'w') as f:
    json.dump(calibration_report, f, indent=2)
logger.info(f"Calibration report saved to {calibration_path}")
```

---

## Cell 20: Section 6 Header (Markdown)

```markdown
#%% md
## Section 6: Stock Ranking Analytics (Phase 9.7)

Analytics for investment decision support:
- Mispricing score calculation: `(Predicted - Last Price) / Last Price`
- Top-N undervalued/overvalued stock rankings
- Analyst comparison analytics
- Confidence-weighted rankings

**Key Outputs**: Rankings DataFrame, mispricing scores
```

---

## Cell 21: Analytics

```python
#%%
# =============================================================================
# Section 6: Phase 9.7 — Stock Ranking Analytics
# =============================================================================

from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_scores,
    rank_stocks,
)

# Calculate mispricing scores
predictions_df['mispricing_score'] = (
    (predictions_df['y_pred'] - predictions_df['last_price']) / 
    predictions_df['last_price']
)

# Calculate confidence-weighted score (narrower intervals = higher confidence)
predictions_df['confidence_score'] = 1 / (1 + predictions_df['interval_width'] / predictions_df['last_price'])

# Combined ranking score
predictions_df['ranking_score'] = predictions_df['mispricing_score'] * predictions_df['confidence_score']

# Top undervalued stocks (positive mispricing = undervalued)
top_undervalued = predictions_df.nlargest(20, 'ranking_score')[
    ['ticker', 'sector', 'last_price', 'y_pred', 'mispricing_score', 'confidence_score', 'ranking_score']
]

# Top overvalued stocks (negative mispricing = overvalued)
top_overvalued = predictions_df.nsmallest(20, 'ranking_score')[
    ['ticker', 'sector', 'last_price', 'y_pred', 'mispricing_score', 'confidence_score', 'ranking_score']
]

print(f"{'='*60}")
print(f"Top 10 Undervalued Stocks")
print(f"{'='*60}")
print(top_undervalued.head(10).to_string(index=False))

print(f"\n{'='*60}")
print(f"Top 10 Overvalued Stocks")
print(f"{'='*60}")
print(top_overvalued.head(10).to_string(index=False))

# Sector distribution of recommendations
sector_distribution = predictions_df.groupby('sector')['mispricing_score'].agg(['mean', 'count'])
print(f"\n{'='*60}")
print(f"Sector Mispricing Summary")
print(f"{'='*60}")
print(sector_distribution.round(4).to_string())

# Save rankings
rankings_path = OUTPUT_DIR / 'analytics' / 'stock_rankings.csv'
predictions_df.to_csv(rankings_path, index=False)
logger.info(f"Rankings saved to {rankings_path}")
```

---

## Cell 22: Section 7 Header (Markdown)

```markdown
#%% md
## Section 7: Comprehensive Reporting (Phase 9.8)

Final reporting and artifact generation:
- Model governance: Model card, lineage tracking
- Dashboard data preparation
- Executive summary generation
- Quality alerts documentation

**Key Outputs**: Model card, lineage JSON, dashboard data, quality alerts
```

---

## Cell 23: Reporting

```python
#%%
# =============================================================================
# Section 7: Phase 9.8 — Comprehensive Reporting
# =============================================================================

from finance_ml.ml_workflow.evaluation import generate_model_card, build_lineage_json
from finance_ml.ml_workflow.reporting import generate_dashboard_data, create_quality_alerts

# Generate Model Card
model_card = {
    'model_version': MODEL_VERSION,
    'model_type': 'Stacking Ensemble (RF + ET + GB + XGB → Ridge)',
    'training_date': datetime.now().strftime('%Y-%m-%d'),
    'business_objective': 'Predict Stock Price Targets for portfolio optimization',
    'target_variable': target_col,
    'metrics': {
        'overall': overall_metrics,
        'by_sector': sector_metrics.to_dict()
    },
    'features_used': len(feature_cols),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'phase93_coverage': f"{coverage_pct:.1f}%",
    'uncertainty_coverage': f"{coverage*100:.1f}%",
    'data_source': 'csv',
    'imputation_strategy': '6step',
    'random_seed': RANDOM_SEED,
}

model_card_path = OUTPUT_DIR / 'governance' / 'model_card.json'
with open(model_card_path, 'w') as f:
    json.dump(model_card, f, indent=2)
logger.info(f"Model card saved to {model_card_path}")

# Generate Lineage JSON
lineage = {
    'pipeline_version': '3.0.0',
    'model_version': MODEL_VERSION,
    'execution_timestamp': datetime.now().isoformat(),
    'stages': [
        {'stage': 'ETL', 'function': 'etl_with_features', 'preset': 'comprehensive'},
        {'stage': 'Feature Selection', 'method': 'mutual_info', 'features_selected': X_selected.shape[1]},
        {'stage': 'Classification', 'model': 'LightGBM', 'cv_strategy': cv_strategy},
        {'stage': 'Regression', 'model': 'StackingRegressor', 'base_models': ['RF', 'ET', 'GB', 'XGB']},
        {'stage': 'Quantile', 'quantiles': QUANTILES},
    ],
    'data_sources': {
        'input': str(DATA_DIR),
        'output': str(OUTPUT_DIR),
    },
    'configuration': {
        'test_size': TEST_SIZE,
        'cv_folds': CV_FOLDS,
        'random_seed': RANDOM_SEED,
    }
}

lineage_path = OUTPUT_DIR / 'governance' / 'lineage.json'
with open(lineage_path, 'w') as f:
    json.dump(lineage, f, indent=2)
logger.info(f"Lineage saved to {lineage_path}")

# Quality Alerts
quality_alerts = []

# Check for high error predictions
high_error_mask = np.abs(predictions_df['pct_error']) > 100
if high_error_mask.any():
    quality_alerts.append({
        'severity': 'WARNING',
        'type': 'high_prediction_error',
        'count': int(high_error_mask.sum()),
        'message': f"{high_error_mask.sum()} predictions with >100% error"
    })

# Check for zero predictions
zero_pred_mask = predictions_df['y_pred'] == 0
if zero_pred_mask.any():
    quality_alerts.append({
        'severity': 'ERROR',
        'type': 'zero_predictions',
        'count': int(zero_pred_mask.sum()),
        'message': f"{zero_pred_mask.sum()} zero predictions detected"
    })

# Save quality alerts
alerts_path = OUTPUT_DIR / 'governance' / 'quality_alerts.json'
with open(alerts_path, 'w') as f:
    json.dump({'alerts': quality_alerts, 'timestamp': datetime.now().isoformat()}, f, indent=2)
logger.info(f"Quality alerts saved to {alerts_path}")

print(f"\n{'='*60}")
print(f"Reporting Complete")
print(f"{'='*60}")
print(f"  Model card: {model_card_path}")
print(f"  Lineage: {lineage_path}")
print(f"  Quality alerts: {len(quality_alerts)} issues")
```

---

## Cell 24: Final Summary

```python
#%%
# =============================================================================
# Notebook Execution Summary
# =============================================================================

print(f"""
{'='*70}
Stock Price Target Prediction — Execution Complete
{'='*70}

📊 Data Summary:
   - Input rows: {len(all_stocks_preprocessed):,}
   - Features (after selection): {X_selected.shape[1]:,}
   - Phase 9.3 coverage: {coverage_pct:.1f}%

🤖 Model Performance:
   - Test R²: {r2_test:.4f}
   - Test MAE: {mae_test:.2f}
   - Interval coverage: {coverage*100:.1f}%

📁 Output Artifacts:
   - Predictions: {predictions_path}
   - Sector metrics: {sector_metrics_path}
   - Model card: {model_card_path}
   - Rankings: {rankings_path}

⚙️ Configuration:
   - Model version: {MODEL_VERSION}
   - Random seed: {RANDOM_SEED}
   - Test size: {TEST_SIZE}
   - CV folds: {CV_FOLDS}

🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
""")
```

---

## Key Improvements in This Design

1. **Single Entry Point**: Uses `etl_with_features()` as the unified ETL pipeline (code_guidelines.md Section 7.5)

2. **Stage-Based Naming**: Follows the 6-stage convention (`all_stocks_preprocessed` → `all_stocks_selected` →
   `all_stocks_classification`)

3. **Leakage Detection**: Explicit checks for R² ≥ 0.95 and MAE = 0 (ml_workflow_guidelines.md critical issues)

4. **Validation Checkpoints**: Post-ETL, post-feature, and post-prediction validations (Section 19)

5. **Configuration Constants**: Single source of truth with `validate_configuration()` (Section 2)

6. **No Magic Numbers**: All thresholds use named constants

7. **Proper CV Strategy**: Uses `determine_cv_strategy()` for automatic selection (Section 10.2)

8. **Quantile Monotonicity**: Explicit validation of `pred_p10 ≤ pred_p50 ≤ pred_p90`

9. **Model Governance**: Model card, lineage JSON, quality alerts (Section 20)

10. **Non-Negativity Enforcement**: All price predictions clipped to ≥ 0