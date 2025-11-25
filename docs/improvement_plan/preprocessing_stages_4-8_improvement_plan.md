# Preprocessing Pipeline Stages 4-8 Improvement Plan

**Version**: 1.0  
**Date**: 2025-11-25  
**Status**: Active  
**Scope**: Stages 4-8 Data Quality Improvements (Winsorization, Imputation, Scaling, Features)

---

## Executive Summary

### Business Objective Alignment

**Primary Goal** (from README.md): ML-based stock valuation and mispricing detection  
**Key Metric**: `(Predicted_Target - Last_Price) / Last_Price`

**Critical Issue Identified**: Current preprocessing pipeline corrupts the core business metric by:

1. **Winsorizing price columns** (last_price, price_target, market_cap) - artificially caps valid extreme prices
2. **Scaling price columns** - destroys dollar interpretability needed for valuation comparison
3. **Treating all numeric columns uniformly** - ignores semantic differences between prices, ratios, and counts

### Current State Analysis

| Stage                              | Current Behavior                                                              | Business Impact                                           | Priority |
|------------------------------------|-------------------------------------------------------------------------------|-----------------------------------------------------------|----------|
| **Stage 4: all_stocks_winsorized** | Winsorizes ALL numeric columns including last_price, price_target, market_cap | 🔴 **CRITICAL**: Distorts core valuation metric           | P0       |
| **Stage 5: all_stocks_imputed**    | 6-step imputation strategy (already enhanced)                                 | ✅ **GOOD**: Comprehensive, sector-aware                   | P2       |
| **Stage 6: all_stocks_scaled**     | Scales ALL numeric columns including prices                                   | 🔴 **HIGH**: Loss of price interpretability               | P0       |
| **Stage 7: all_stocks_features**   | Advanced feature engineering                                                  | 🟡 **MODERATE**: May create redundant/correlated features | P1       |
| **Stage 8: all_stocks_enhanced**   | Final enhanced dataset                                                        | 🟡 **MODERATE**: Validation needed                        | P1       |

### Root Cause Analysis

**Problem**: Both `winsorize_by_sector()` and `scale_features()` use the same flawed logic:

```python
# finance_ml/ml_workflow/preprocessing/outliers.py:209-210
if columns is None:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
```

```python
# finance_ml/ml_workflow/preprocessing/scaling.py:68-69
if columns is None:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()
```

**Impact**: No semantic understanding of column types:

- **Price columns** (last_price, price_target): Require original scale for business logic
- **Market cap/volume**: Highly skewed, better handled with log-transforms
- **Ratios** (P/E, P/B): Already normalized, may not need winsorization
- **Percentages**: Bounded [0, 100], inappropriate for percentile capping
- **Counts** (employees, analyst ratings): Discrete, inappropriate for continuous scaling

---

## Improvement Tasks (TDD Approach)

### Task Group 1: Selective Winsorization with Semantic Column Classification (P0)

**Objective**: Exclude price/valuation columns from winsorization; apply semantic-aware transformations

#### Task 1.1: Define Column Semantic Categories

**Test Module**: `tests/test_column_semantics.py` (NEW)

**Test Cases**:

```python
def test_identify_price_columns():
    """Price columns should be identified and excluded from winsorization."""
    # Expected: last_price, price_target, price_target_median, price_target_ytd_ago
    
def test_identify_market_value_columns():
    """Market cap/value columns requiring log-transforms."""
    # Expected: market_cap, ev, total_assets, revenue, total_debt
    
def test_identify_ratio_columns():
    """Financial ratios already normalized."""
    # Expected: p_e, p_b, p_s, ev_ebitda, roe, roa, etc.
    
def test_identify_percentage_columns():
    """Percentage columns bounded [0, 100]."""
    # Expected: margin columns, growth rates, volatility
    
def test_identify_count_columns():
    """Discrete count columns."""
    # Expected: num_analysts, num_employees, num_strong_buy_ratings
```

**Implementation**: Create `finance_ml/ml_workflow/preprocessing/column_semantics.py`:

```python
from typing import Dict, List, Set
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

PRICE_COLUMNS = {
    'last_price',
    'price_target',
    'price_target_median',
    'price_target_ytd_ago',
    'price_target_12m_ago',
}

MARKET_VALUE_COLUMNS = {
    'market_cap',
    'ev',
    'total_assets',
    'revenue',
    'total_debt',
    'ebitda',
    'operating_income',
    'net_income',
    'cash_and_equivalents',
}

RATIO_COLUMNS = {
    'p_e', 'p_b', 'p_s', 'p_fcf', 'p_tbv',
    'ev_ebitda', 'ev_sales', 'ev_fcf',
    'roe', 'roa', 'roic', 'roce',
    'debt_equity', 'current_ratio', 'quick_ratio',
}

PERCENTAGE_COLUMNS = {
    'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin',
    'revenue_growth_yoy', 'earnings_growth_yoy', 'ebitda_growth_yoy',
    'volatility_20d', 'volatility_60d', 'volatility_1y',
}

COUNT_COLUMNS = {
    'num_analysts', 'num_employees',
    'num_strong_buy_ratings', 'num_buy_ratings', 'num_hold_ratings',
    'num_sell_ratings', 'num_strong_sell_ratings',
}

def classify_columns(df_columns: List[str]) -> Dict[str, Set[str]]:
    """Classify DataFrame columns by semantic type."""
    # Implementation with fuzzy matching for normalized names
    pass

def get_winsorizable_columns(df_columns: List[str]) -> List[str]:
    """Return columns safe for winsorization (exclude prices, ratios, percentages)."""
    pass

def get_log_transform_columns(df_columns: List[str]) -> List[str]:
    """Return columns requiring log-transform (market values)."""
    pass

def get_scalable_columns(df_columns: List[str]) -> List[str]:
    """Return columns safe for scaling (exclude prices)."""
    pass
```

**Acceptance Criteria**:

- ✅ Price columns correctly identified and excluded from winsorization
- ✅ Market value columns identified for log-transforms
- ✅ Ratio/percentage columns identified as pre-normalized
- ✅ All classification functions have ≥90% test coverage

---

#### Task 1.2: Enhanced Winsorization with Column Exclusions

**Test Module**: `tests/test_selective_winsorization.py` (NEW)

**Test Cases**:

```python
def test_winsorize_excludes_price_columns():
    """Winsorization should skip price columns."""
    # Given: DataFrame with last_price, price_target, market_cap, p_e
    # When: winsorize_by_sector called
    # Then: last_price, price_target unchanged; p_e may be winsorized
    
def test_winsorize_excludes_ratios():
    """Financial ratios should not be winsorized (already bounded)."""
    
def test_winsorize_respects_column_whitelist():
    """Only whitelisted columns should be winsorized."""
    
def test_winsorize_by_sector_with_semantic_columns():
    """Sector-specific winsorization respects semantic classification."""
    
def test_winsorize_diagnostics_show_excluded_columns():
    """Diagnostics should report which columns were excluded and why."""
```

**Implementation**: Update `finance_ml/ml_workflow/preprocessing/outliers.py`:

```python
def winsorize_by_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    by_sector: bool = True,
    exclude_price_columns: bool = True,  # NEW
    exclude_ratio_columns: bool = True,  # NEW
) -> pd.DataFrame:
    """
    Winsorize extreme values with semantic column exclusions.
    
    Args:
        exclude_price_columns: If True, exclude price/valuation columns
        exclude_ratio_columns: If True, exclude pre-normalized ratios
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        get_winsorizable_columns, 
        PRICE_COLUMNS, 
        RATIO_COLUMNS
    )
    
    result = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # NEW: Apply semantic filtering
    excluded = set()
    if exclude_price_columns:
        excluded.update(PRICE_COLUMNS)
    if exclude_ratio_columns:
        excluded.update(RATIO_COLUMNS)
    
    winsorizable = [c for c in columns if c not in excluded and c in df.columns]
    excluded_present = [c for c in columns if c in excluded and c in df.columns]
    
    logger.info(f"Winsorizing {len(winsorizable)} columns, excluding {len(excluded_present)} semantic columns")
    
    # ... rest of implementation
```

**Acceptance Criteria**:

- ✅ Price columns (last_price, price_target, market_cap) never winsorized
- ✅ Backward compatible: existing code works with default exclude_price_columns=True
- ✅ Diagnostics log excluded columns and reasons
- ✅ Test coverage ≥85%

---

### Task Group 2: Log-Transform Alternatives for Skewed Financials (P0)

**Objective**: Replace winsorization with log-transforms for highly skewed market value columns

#### Task 2.1: Log-Transform Pipeline for Market Value Columns

**Test Module**: `tests/test_log_transforms.py` (NEW)

**Test Cases**:

```python
def test_log_transform_market_cap():
    """Market cap should be log-transformed to handle skewness."""
    # Given: market_cap with high skewness (>2.0)
    # When: apply_log_transforms called
    # Then: log_market_cap has reduced skewness (<1.0)
    
def test_log_transform_handles_zeros():
    """Log-transform should handle zero values (log1p)."""
    
def test_log_transform_handles_negatives():
    """Negative values should be handled (signed log)."""
    
def test_log_transform_preserves_nulls():
    """Null values should remain null after transform."""
    
def test_inverse_log_transform():
    """Log-transform should be reversible."""
    
def test_log_transform_reduces_outlier_impact():
    """Log-transformed columns should have fewer IQR outliers."""
```

**Implementation**: Create `finance_ml/ml_workflow/preprocessing/transforms.py`:

```python
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

def apply_log_transforms(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = 'log1p',  # 'log1p', 'signed_log', 'boxcox'
) -> pd.DataFrame:
    """
    Apply log-transforms to skewed financial columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform (default: auto-detect skewed)
        method: Transform method
            - 'log1p': log(1 + x) for non-negative
            - 'signed_log': sign(x) * log(1 + |x|) for any value
            - 'boxcox': Box-Cox power transform (requires positive)
    
    Returns:
        DataFrame with log-transformed columns (original columns replaced)
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        get_log_transform_columns
    )
    
    result = df.copy()
    
    if columns is None:
        columns = get_log_transform_columns(df.columns.tolist())
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'log1p':
            # Handles zeros, requires non-negative
            result[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
        elif method == 'signed_log':
            # Handles negative values
            result[f'log_{col}'] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
        else:
            raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Applied {method} transforms to {len(columns)} columns")
    return result

def inverse_log_transform(
    df: pd.DataFrame,
    columns: List[str],
    method: str = 'log1p',
) -> pd.DataFrame:
    """Reverse log-transforms for interpretability."""
    pass
```

**Acceptance Criteria**:

- ✅ Log-transforms reduce skewness for market_cap, revenue, total_assets
- ✅ Handles zero and negative values appropriately
- ✅ Preserves null values
- ✅ Creates new log_* columns (keeps originals for interpretability)
- ✅ Test coverage ≥85%

---

#### Task 2.2: Update Notebook Stage 4 to Use Log-Transforms

**Test Module**: `tests/test_integration_notebook_stage4.py` (NEW)

**Test Cases**:

```python
def test_stage4_excludes_price_columns_from_winsorization():
    """Stage 4 should not winsorize price columns."""
    
def test_stage4_applies_log_transforms_to_market_values():
    """Stage 4 should create log_market_cap, log_revenue, etc."""
    
def test_stage4_preserves_original_price_columns():
    """Original last_price and price_target must be preserved."""
    
def test_stage4_pipeline_deterministic():
    """Stage 4 transformations should be reproducible."""
```

**Implementation**: Update `ml_finance_model_main.ipynb` Section 5 (Stage 4):

```python
# BEFORE (lines ~922-1003):
financial_metrics = [c for c in numeric_cols if c not in ['ticker', 'isin']]
all_stocks_winsorized = winsorize_by_sector(
    all_stocks_typed,
    columns=financial_metrics[:50],
    lower_percentile=WINSORIZE_LOWER,
    upper_percentile=WINSORIZE_UPPER,
    by_sector=True
)

# AFTER:
from finance_ml.ml_workflow.preprocessing.column_semantics import get_winsorizable_columns, get_log_transform_columns
from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms

# Step 1: Apply log-transforms to skewed market value columns
all_stocks_log_transformed = apply_log_transforms(
    all_stocks_typed,
    method='signed_log'  # Handles negative values (debt, income)
)

# Step 2: Selective winsorization (excludes prices, ratios)
winsorizable_cols = get_winsorizable_columns(all_stocks_log_transformed.columns.tolist())
all_stocks_winsorized = winsorize_by_sector(
    all_stocks_log_transformed,
    columns=winsorizable_cols,
    lower_percentile=WINSORIZE_LOWER,
    upper_percentile=WINSORIZE_UPPER,
    by_sector=True,
    exclude_price_columns=True,
    exclude_ratio_columns=True
)

print(f"✓ Stage 4 Complete: Log-transformed {len([c for c in all_stocks_winsorized.columns if c.startswith('log_')])} columns")
print(f"✓ Winsorized {len(winsorizable_cols)} columns (excluded price/ratio columns)")
```

**Acceptance Criteria**:

- ✅ Notebook Section 5 updated with new logic
- ✅ Code cells include explanatory comments
- ✅ Output shows excluded column counts
- ✅ Integration test passes

---

### Task Group 3: Selective Scaling Improvements (P0)

**Objective**: Exclude price columns from scaling to preserve interpretability

#### Task 3.1: Enhanced Scaling with Column Exclusions

**Test Module**: `tests/test_selective_scaling.py` (NEW)

**Test Cases**:

```python
def test_scale_features_excludes_price_columns():
    """Scaling should skip price columns to preserve interpretability."""
    
def test_scale_features_handles_log_transformed_columns():
    """Log-transformed columns should be scaled."""
    
def test_scale_features_by_sector_with_exclusions():
    """Sector-specific scaling respects exclusions."""
    
def test_scale_features_backward_compatible():
    """Existing code should work with exclude_price_columns=True default."""
```

**Implementation**: Update `finance_ml/ml_workflow/preprocessing/scaling.py`:

```python
def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler_type: str = "robust",
    by_sector: bool = True,
    exclude_price_columns: bool = True,  # NEW: default True for safety
) -> pd.DataFrame:
    """
    Scale features with semantic column exclusions.
    
    Args:
        exclude_price_columns: If True, exclude price/valuation columns from scaling
    """
    from finance_ml.ml_workflow.preprocessing.column_semantics import (
        get_scalable_columns,
        PRICE_COLUMNS
    )
    
    result = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # NEW: Apply semantic filtering
    if exclude_price_columns:
        excluded = [c for c in columns if c in PRICE_COLUMNS and c in df.columns]
        scalable = [c for c in columns if c not in PRICE_COLUMNS and c in df.columns]
        logger.info(f"Scaling {len(scalable)} columns, excluding {len(excluded)} price columns")
        columns = scalable
    
    # ... rest of implementation
```

**Acceptance Criteria**:

- ✅ Price columns (last_price, price_target) excluded from scaling by default
- ✅ Backward compatible with existing code
- ✅ Log-transformed columns (log_market_cap) are scaled
- ✅ Test coverage ≥85%

---

#### Task 3.2: Update Notebook Stage 6 Scaling

**Test Module**: `tests/test_integration_notebook_stage6.py` (NEW)

**Implementation**: Update `ml_finance_model_main.ipynb` Section 6 (Stage 6):

```python
# AFTER Stage 5 (imputation)
from finance_ml.ml_workflow.preprocessing.scaling import scale_features

# Scale features (excludes price columns by default)
all_stocks_scaled = scale_features(
    all_stocks_imputed,
    scaler_type='robust',
    by_sector=True,
    exclude_price_columns=True  # Explicit for clarity
)

# Verify price columns preserved
price_cols = ['last_price', 'price_target', 'price_target_median']
for col in price_cols:
    if col in all_stocks_scaled.columns:
        assert all_stocks_scaled[col].equals(all_stocks_imputed[col]), f"{col} should not be scaled"

print(f"✓ Stage 6 Complete: Scaled features while preserving {len(price_cols)} price columns")
```

---

### Task Group 4: Feature Engineering Quality Enhancements (P1)

**Objective**: Improve feature quality and reduce redundancy in stages 7-8

#### Task 4.1: Feature Correlation Analysis and Pruning

**Test Module**: `tests/test_feature_quality.py` (NEW)

**Test Cases**:

```python
def test_detect_highly_correlated_features():
    """Identify feature pairs with correlation > 0.95."""
    
def test_prune_redundant_features():
    """Remove redundant features based on correlation threshold."""
    
def test_feature_importance_ranking():
    """Rank features by importance for target prediction."""
    
def test_phase93_features_no_leakage():
    """Phase 9.3 features should not leak target information."""
```

**Implementation**: Create `finance_ml/ml_workflow/features/quality.py`:

```python
def detect_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    method: str = 'pearson',
) -> List[Tuple[str, str, float]]:
    """Detect highly correlated feature pairs."""
    pass

def prune_correlated_features(
    df: pd.DataFrame,
    threshold: float = 0.95,
    keep_strategy: str = 'first',  # 'first', 'importance', 'variance'
) -> pd.DataFrame:
    """Remove redundant correlated features."""
    pass
```

**Acceptance Criteria**:

- ✅ Identifies feature pairs with correlation > threshold
- ✅ Provides multiple pruning strategies
- ✅ Test coverage ≥80%

---

#### Task 4.2: Feature Set Validation

**Test Module**: `tests/test_feature_validation.py` (NEW)

**Test Cases**:

```python
def test_no_infinite_values_in_features():
    """Features should not contain infinite values."""
    
def test_no_constant_features():
    """Constant features should be removed."""
    
def test_feature_dtypes_consistent():
    """Feature dtypes should be consistent across pipeline."""
    
def test_feature_nulls_below_threshold():
    """Features with >50% nulls should be flagged."""
```

---

### Task Group 5: Data Quality Validation and Reporting (P1)

**Objective**: Add comprehensive quality checks for stages 4-8

#### Task 5.1: Stage-Level Quality Metrics

**Test Module**: `tests/test_stage_quality_metrics.py` (NEW)

**Test Cases**:

```python
def test_stage4_quality_metrics():
    """Stage 4 should report winsorization effects and log-transform success."""
    
def test_stage5_quality_metrics():
    """Stage 5 should report imputation success rates."""
    
def test_stage6_quality_metrics():
    """Stage 6 should report scaling statistics and excluded columns."""
    
def test_stage7_quality_metrics():
    """Stage 7 should report feature count, correlation matrix."""
    
def test_stage8_quality_metrics():
    """Stage 8 should report final data quality score."""
```

**Implementation**: Create `finance_ml/ml_workflow/preprocessing/quality_metrics.py`:

```python
def calculate_stage_quality_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    stage_name: str,
) -> Dict[str, Any]:
    """
    Calculate quality metrics for a preprocessing stage.
    
    Returns:
        Dict with metrics:
        - rows_changed: Number of rows modified
        - columns_added: New columns created
        - null_reduction: Change in null percentage
        - outlier_reduction: Change in outlier count
        - skewness_change: Change in distribution skewness
    """
    pass
```

---

## Implementation Roadmap

### Phase 1: Critical Fixes (Week 1) - P0 Tasks

**Goal**: Fix business-critical issues with price column handling

1. **Day 1-2**: Task 1.1 - Column semantic classification
    - Create `column_semantics.py` module
    - Write 12 test cases
    - Achieve ≥90% coverage

2. **Day 3-4**: Task 1.2 & 3.1 - Enhanced winsorization and scaling
    - Update `outliers.py` and `scaling.py`
    - Add exclude_price_columns parameter
    - Write 16 test cases

3. **Day 5**: Task 2.1 - Log-transform pipeline
    - Create `transforms.py` module
    - Implement log1p and signed_log methods
    - Write 8 test cases

### Phase 2: Notebook Integration (Week 2) - P0 Tasks

**Goal**: Update notebook to use new semantic-aware preprocessing

4. **Day 6-7**: Task 2.2 & 3.2 - Update notebook stages 4 & 6
    - Modify notebook cells
    - Add integration tests
    - Verify output quality

5. **Day 8**: Integration testing
    - Run full notebook end-to-end
    - Verify price columns preserved
    - Check prediction quality

### Phase 3: Feature Quality (Week 3) - P1 Tasks

**Goal**: Improve feature engineering quality

6. **Day 9-10**: Task 4.1 - Feature correlation analysis
    - Create `features/quality.py` module
    - Implement correlation detection and pruning

7. **Day 11-12**: Task 4.2 - Feature validation
    - Add validation checks
    - Create quality reports

### Phase 4: Quality Metrics (Week 4) - P1 Tasks

**Goal**: Add comprehensive quality monitoring

8. **Day 13-14**: Task 5.1 - Stage quality metrics
    - Create `quality_metrics.py` module
    - Add stage-level reporting

9. **Day 15**: Documentation and final testing
    - Update code_guidelines.md
    - Run full test suite
    - Create improvement summary report

---

## Success Metrics

### Quantitative Goals

1. **Test Coverage**: ≥85% for all new modules
2. **Price Column Preservation**: 100% (zero modifications to last_price, price_target)
3. **Skewness Reduction**: ≥50% for log-transformed columns
4. **Feature Redundancy**: <5% highly correlated features (r > 0.95)
5. **Pipeline Execution Time**: <10% increase vs baseline

### Qualitative Goals

1. **Business Alignment**: Price columns remain in original units for valuation analysis
2. **Interpretability**: Clear separation between raw prices and engineered features
3. **Robustness**: Outlier handling preserves valid extreme values
4. **Maintainability**: Semantic column classification reusable across pipeline

---

## Risk Mitigation

### Risk 1: Breaking Changes to Existing Notebook

**Mitigation**:

- Add backward compatibility flags (exclude_price_columns defaults to True)
- Create integration tests before modifying notebook
- Keep deprecated functions with warnings

### Risk 2: Performance Degradation

**Mitigation**:

- Profile log-transform overhead (expected <1% impact)
- Cache semantic column classifications
- Use vectorized operations

### Risk 3: Semantic Classification Errors

**Mitigation**:

- Comprehensive unit tests for column identification
- Fuzzy matching for normalized column names
- Manual review of classification results
- Allow user overrides in function parameters

---

## Appendix A: Column Semantic Reference

### Price Columns (Never Transform)

```python
PRICE_COLUMNS = {
    'last_price',           # Current market price
    'price_target',         # Analyst consensus target
    'price_target_median',  # Median analyst target
    'price_target_ytd_ago', # Historical target (YTD)
    'price_target_12m_ago', # Historical target (12M)
}
```

### Market Value Columns (Log-Transform)

```python
MARKET_VALUE_COLUMNS = {
    'market_cap',           # High skewness (>3.0 typical)
    'ev',                   # Enterprise value
    'total_assets',         # Balance sheet item
    'revenue',              # Income statement item
    'total_debt',           # Liability
    'ebitda',               # Earnings proxy
    'operating_income',     # Operating metric
    'net_income',           # Bottom line (can be negative)
    'cash_and_equivalents', # Liquidity
}
```

### Ratio Columns (Pre-Normalized, Optional Winsorization)

```python
RATIO_COLUMNS = {
    # Valuation ratios
    'p_e', 'p_b', 'p_s', 'p_fcf', 'p_tbv',
    'ev_ebitda', 'ev_sales', 'ev_fcf',
    
    # Profitability ratios
    'roe', 'roa', 'roic', 'roce',
    
    # Leverage ratios
    'debt_equity', 'net_debt_ebitda',
    
    # Liquidity ratios
    'current_ratio', 'quick_ratio', 'cash_ratio',
}
```

### Percentage Columns (Bounded [0, 100])

```python
PERCENTAGE_COLUMNS = {
    # Margins
    'gross_margin', 'operating_margin', 'net_margin', 'ebitda_margin',
    
    # Growth rates
    'revenue_growth_yoy', 'earnings_growth_yoy', 'ebitda_growth_yoy',
    
    # Volatility
    'volatility_20d', 'volatility_60d', 'volatility_1y',
}
```

---

## Appendix B: Code Guidelines Updates

**Proposed additions to `docs/code_guidelines.md` v1.5**:

### Section 8.5: Preprocessing Stage Naming (NEW)

**8.5.1 Column Semantic Classification**

All preprocessing functions must respect semantic column types:

```python
# GOOD: Semantic-aware preprocessing
from finance_ml.ml_workflow.preprocessing.column_semantics import get_winsorizable_columns

winsorizable = get_winsorizable_columns(df.columns.tolist())
df_processed = winsorize_by_sector(df, columns=winsorizable, exclude_price_columns=True)

# BAD: Treating all numeric columns uniformly
df_processed = winsorize_by_sector(df)  # Corrupts price columns!
```

**8.5.2 Price Column Preservation Policy**

Price columns (last_price, price_target, market_cap) must NEVER be:

- Winsorized
- Scaled (StandardScaler, MinMaxScaler)
- Log-transformed (except as new columns: log_market_cap)
- Clipped or capped

**Rationale**: The core business metric `(Predicted_Target - Last_Price) / Last_Price` requires original price scale.

**8.5.3 Alternative Transformations for Skewed Data**

Use log-transforms instead of winsorization for highly skewed market value columns:

```python
# GOOD: Log-transform preserves information
from finance_ml.ml_workflow.preprocessing.transforms import apply_log_transforms

df = apply_log_transforms(df, method='signed_log')  # Creates log_market_cap, etc.

# BAD: Winsorization loses information about extreme but valid values
df = winsorize_by_sector(df, columns=['market_cap'])  # Caps mega-cap stocks artificially
```

---

## Appendix C: Test Execution Strategy

### Fast Tests (Run on Every Commit)

```bash
# Unit tests for semantic classification (~5 seconds)
python -m unittest tests.test_column_semantics -v

# Unit tests for transforms (~10 seconds)
python -m unittest tests.test_log_transforms tests.test_selective_winsorization tests.test_selective_scaling -v
```

### Medium Tests (Run Before PR)

```bash
# Integration tests (~1-2 minutes)
python -m unittest tests.test_integration_notebook_stage4 tests.test_integration_notebook_stage6 -v

# Feature quality tests (~30 seconds)
python -m unittest tests.test_feature_quality tests.test_feature_validation -v
```

### Full Suite (CI/CD)

```bash
# All preprocessing tests (~5 minutes)
python -m unittest discover -s tests -p "test_*preprocessing*.py" -v
python -m unittest discover -s tests -p "test_*stage*.py" -v
```

---

## References

1. **Business Requirements**: `README.md` - ML-based stock valuation objective
2. **Code Standards**: `docs/code_guidelines.md` v1.4 - TDD conventions, dataframe stages
3. **Existing Plan**: `docs/improvement_plan/data_preprocessing improvement_plan.md` - Schema alignment
4. **Phase 9.3 Features**: `docs/improvement_plan/Phase_9.3_feature_enhancement_plan.md` - Feature categories
5. **Current Implementation**:
    - `finance_ml/ml_workflow/preprocessing/outliers.py` - Winsorization
    - `finance_ml/ml_workflow/preprocessing/scaling.py` - Feature scaling
    - `ml_finance_model_main.ipynb` - Notebook pipeline

---

**Document Owner**: Development Team  
**Review Cycle**: Weekly during implementation  
**Status Updates**: Track progress in CHANGELOG.md
