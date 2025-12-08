# Notebook Update Plan for Unified ETL Pipeline

## Summary

Update all three notebooks to align with code_guidelines.md v1.7 Section 8 (Notebook Best Practices) and Section 9 (
Column Schema and DataFrame Conventions), incorporating the new unified ETL pipeline with semantic transformations and
feature engineering.

## Key Changes Required

### 1. Import Updates

**Old approach:**

```python
from finance_ml.ml_workflow.preprocessing import (
    load_from_csv, preprocess, apply_enhanced_imputation_strategy_6step,
    winsorize_by_sector, scale_features
)
```

**New approach (unified):**

```python
from finance_ml.ml_workflow.preprocessing import (
    etl_with_features,  # Single entry point for complete pipeline
    ETLConfig,          # Configuration with semantic-aware options
    ETLMetrics,         # Comprehensive metrics tracking
)
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    PRICE_COLUMNS,      # Protected columns that must preserve original units
    classify_columns,   # Semantic column classification
)
```

### 2. Data Loading and Preprocessing (Phase 9.1)

**Old approach (scattered across multiple cells):**

```python
# Cell 1: Load data
all_stocks = load_from_csv(data_dir)

# Cell 2: Normalize columns
all_stocks.columns = normalize_columns(all_stocks.columns)

# Cell 3: Dtype casting
all_stocks = detect_and_cast_dtypes(all_stocks)

# Cell 4: Imputation
all_stocks = apply_enhanced_imputation_strategy_6step(all_stocks)

# Cell 5: Winsorization
all_stocks = winsorize_by_sector(all_stocks)

# Cell 6: Scaling
all_stocks = scale_features(all_stocks)

# Cell 7: Financial metrics
all_stocks = compute_financial_metrics(all_stocks)
```

**New approach (unified single call):**

```python
# Phase 9.1: Unified ETL Pipeline with Semantic Transformations and Feature Engineering
# Aligned with code_guidelines.md v1.7 Section 8.5 and Section 8.6

from finance_ml.ml_workflow.preprocessing import etl_with_features, ETLConfig
from pathlib import Path

# Configure ETL pipeline with semantic-aware transformations
etl_config = ETLConfig(
    # Data source
    data_dir=Path('data'),
    
    # Semantic-aware transformations (Section 8.5)
    use_semantic_column_classification=True,
    preserve_price_columns=True,              # Never transform price columns
    log_transform_market_values=True,         # Log-transform skewed market values
    exclude_ratios_from_winsorization=True,   # Ratios are pre-normalized
    exclude_percentages_from_winsorization=True,  # Percentages are bounded
    
    # Feature engineering (Section 9.3)
    apply_feature_engineering=True,
    feature_preset="comprehensive",           # Options: "basic", "momentum", "quality", "comprehensive"
    
    # Standard preprocessing
    normalize_columns=True,
    apply_dtype_casting=True,
    apply_imputation=True,
    sanitize_data=True,
    apply_scaling=True,
    
    # Financial metrics
    compute_valuation_metrics=True,
    compute_profitability_metrics=True,
    compute_growth_metrics=True,
    compute_leverage_metrics=True,
    compute_target_vs_price=True,
    
    # Quality validation
    validate_quality=True,
    validate_schema=True,
)

# Run unified ETL pipeline
all_stocks, etl_metrics = etl_with_features(
    source='csv',
    data_dir=Path('data'),
    feature_preset='comprehensive',
    config=etl_config,
    return_metrics=True,
)

# Display ETL metrics
print("\n" + "="*80)
print("ETL PIPELINE METRICS")
print("="*80)
print(etl_metrics.summary())

# Verify semantic transformations
print("\n" + "="*80)
print("SEMANTIC TRANSFORMATIONS APPLIED")
print("="*80)
print(f"Price columns protected: {etl_metrics.price_columns_count}")
print(f"Log-transformed columns: {etl_metrics.log_transformed_columns}")
print(f"Feature engineering: {etl_metrics.feature_engineering_applied}")
print(f"Features added: {etl_metrics.features_added}")
```

### 3. DataFrame Stage Naming (Section 8.3)

**Convention for notebook cells:**

```python
# Stage 1: Raw data (immediately after loading)
all_stocks_raw = etl_with_features(source='csv', ...)

# Stage 2: Normalized (column names normalized)
all_stocks_normalized = all_stocks_raw.copy()  # If needed

# Stage 3-8: Use unified pipeline (handles internally):
# - all_stocks_typed (dtype casting)
# - all_stocks_winsorized (outlier handling)
# - all_stocks_imputed (missing value imputation)
# - all_stocks_scaled (feature scaling)
# - all_stocks_features (feature engineering)
# - all_stocks_enhanced (financial metrics)

# Final output from etl_with_features is fully processed
all_stocks = all_stocks_raw  # Rename for clarity
```

### 4. Centralized Configuration Constants (Section 8.4)

**Add configuration cell at the top of each notebook:**

```python
# ============================================================================
# CENTRALIZED CONFIGURATION CONSTANTS (code_guidelines.md v1.7 Section 8.4)
# ============================================================================

# Target columns
TARGET_COL = "price_target"
TARGET_COL_FALLBACK = "price_target_median"

# Data splits
TEST_SIZE = 0.2
TRAIN_SIZE = 0.8
CV_FOLDS = 5

# Quantiles for uncertainty estimation
QUANTILES = [0.1, 0.5, 0.9]

# Sector constraints
MIN_SECTOR_SAMPLES = 30
MAX_SECTOR_WEIGHT = 0.30
MAX_SINGLE_POSITION = 0.10

# Outlier handling
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# Confidence thresholds
CONFIDENCE_ALPHA = 0.2  # 80% prediction intervals
CONFIDENCE_LEVEL = 0.8

# Reproducibility
RANDOM_SEED = 42
N_JOBS = -1  # Use all CPU cores
```

### 5. Price Column Protection Validation

**Add validation cell after ETL:**

```python
# ============================================================================
# VALIDATION: Price Column Preservation (Section 8.5.2)
# ============================================================================

from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

# Verify price columns were not transformed
price_cols_in_df = [c for c in all_stocks.columns if c in PRICE_COLUMNS]

print(f"\nPrice columns preserved: {len(price_cols_in_df)}")
for col in price_cols_in_df:
    print(f"  - {col}: min={all_stocks[col].min():.2f}, max={all_stocks[col].max():.2f}")

# Verify no scaling was applied to price columns
if 'last_price' in all_stocks.columns:
    original_price_range = all_stocks['last_price'].max() - all_stocks['last_price'].min()
    print(f"\nLast Price range: ${original_price_range:.2f} (should be in original dollar units)")
```

## Specific Notebook Updates

### ml_finance_model_main.ipynb (152 cells)

- **Cell 7-8**: Replace with unified ETL pipeline call
- **Cell 9-20**: Remove scattered preprocessing cells, consolidate into single ETL call
- Add centralized configuration constants cell after imports
- Add price column validation cell after ETL

### ml_finance_model_main2_0.ipynb (166 cells)

- **Cell 7-8**: Replace with unified ETL pipeline call
- **Cell 9-18**: Consolidate preprocessing cells
- Add semantic transformation validation
- Add feature engineering metrics display

### etl_data_explorer.ipynb (40 cells)

- **Cell 4**: Update to use `etl_with_features()` instead of separate ETL steps
- **Cell 5**: Add semantic classification display
- Add interactive visualization of price column preservation
- Add log-transformed columns analysis

## Benefits of This Approach

1. **Single Entry Point**: One function call replaces 7-10 scattered cells
2. **Semantic Awareness**: Price columns automatically protected
3. **Feature Engineering**: Integrated Phase 9.3 features (196 features)
4. **Traceability**: Comprehensive metrics for audit trail
5. **Consistency**: All notebooks use same unified pipeline
6. **Maintainability**: Configuration changes in one place
7. **Code Guidelines Compliance**: Aligns with v1.7 Sections 8 and 9

## Implementation Notes

- Keep existing EDA, modeling, and evaluation cells unchanged
- Only update data loading and preprocessing sections
- Preserve all comments and markdown explanations
- Test each notebook after updates to ensure execution
- Update cell numbering/titles to reflect consolidated structure
