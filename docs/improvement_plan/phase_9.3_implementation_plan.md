# Phase 9.3 TDD Implementation Plan

**Date:** 2025-12-09  
**Status:** ACTIVE  
**Version:** 1.0  
**Model Version Target:** v9_9  
**Alignment:** code_guidelines.md v1.10

---

## Executive Summary

This TDD implementation plan addresses Phase 9.3 Feature Engineering gaps identified in the current state analysis. It
focuses on two high-priority tasks that enhance the unified ETL pipeline and semantic preprocessing capabilities:

- **Task 1**: Automated Feature Selection Pipeline (High Priority)
- **Task 3**: Semantic Column Classification Enhancement (High Priority)

**Business Objective**: Predict Stock Price Targets for portfolio optimization by reducing feature noise, improving
model interpretability, and ensuring correct semantic transformations across all 318 schema columns.

**Current State Strengths**:

- 196 features across 16 semantic categories
- Unified ETL pipeline via `etl_with_features()` with feature presets
- Price column preservation policy (21 columns protected)
- Log-transforms for market value columns

**Gaps to Address**:

- No automated feature selection in unified ETL pipeline
- Feature importance thresholds hardcoded (0.01) not configurable
- Missing correlation-based redundancy detection
- 487 columns in "OTHER" semantic category needing assignment

---

## Implementation Overview

### Sprint Assignment

**Sprint 1 (High Priority)** - Foundation improvements  
**Estimated Duration**: 2 weeks  
**Dependencies**: Existing ETL pipeline, semantic classification module

### Test Modules

- `tests/test_feature_selection_auto.py` (4 tests - Task 1)
- `tests/test_semantic_classification.py` (extend existing, 3 new tests - Task 3)

**Total New Tests**: 7 tests

---

## Task 1: Automated Feature Selection Pipeline

### Priority: High

### Complexity: Medium

### Business Impact: Reduces noise, improves accuracy, enhances interpretability

### Objective

Integrate automated feature selection into the unified ETL pipeline to reduce dimensionality while preserving price
columns and high-importance features.

### Current Implementation

- Manual feature engineering in `finance_ml/ml_workflow/features/advanced.py`
- Static feature presets: basic, momentum, quality, comprehensive (196 features)
- No automatic pruning of low-importance or redundant features

### Target Implementation

**File**: `finance_ml/ml_workflow/features/selection.py` (new module)

### TDD Test Specifications

#### Test 1: `test_select_features_by_importance_threshold`

**Purpose**: Verify features below threshold are pruned

```python
def test_select_features_by_importance_threshold(self):
    """Features below importance threshold should be removed."""
    # Given: DataFrame with features of varying importance
    X = pd.DataFrame({
        'high_importance': np.random.randn(100),
        'medium_importance': np.random.randn(100) * 0.5,
        'low_importance': np.random.randn(100) * 0.01,
        'last_price': np.random.uniform(10, 100, 100)  # Price column
        })
    y = X['high_importance'] * 2 + np.random.randn(100)

    # When: Select features with threshold=0.05
    selected = select_features_auto(
            X, y,
            importance_threshold=0.05,
            method='mutual_info'
            )

    # Then: Low importance removed, price column preserved
    self.assertIn('high_importance', selected.columns)
    self.assertIn('last_price', selected.columns)
    self.assertNotIn('low_importance', selected.columns)
```

#### Test 2: `test_select_features_removes_correlated_redundancy`

**Purpose**: Verify highly correlated features (>0.95) are deduplicated

```python
def test_select_features_removes_correlated_redundancy(self):
    """Highly correlated features should be deduplicated."""
    # Given: Features with high correlation
    X = pd.DataFrame({
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100)
        })
    X['feature_b_duplicate'] = X['feature_b'] + np.random.randn(100) * 0.01
    y = X['feature_a'] + np.random.randn(100)

    # When: Select with correlation_threshold=0.95
    selected = select_features_auto(
            X, y,
            correlation_threshold=0.95,
            method='correlation'
            )

    # Then: Only one of correlated pair kept
    correlated_kept = sum([
        'feature_b' in selected.columns,
        'feature_b_duplicate' in selected.columns
        ])
    self.assertEqual(correlated_kept, 1)
```

#### Test 3: `test_select_features_preserves_price_columns`

**Purpose**: Verify PRICE_COLUMNS never removed by selection

```python
def test_select_features_preserves_price_columns(self):
    """Price columns must never be removed by selection."""
    # Given: Price columns with low calculated importance
    price_cols = ['last_price', 'price_target', 'price_target_median']
    X = pd.DataFrame({
        col: np.random.uniform(10, 100, 100)
        for col in price_cols + ['unrelated_feature']
        })
    y = np.random.randn(100)  # Target unrelated to price

    # When: Select features with strict threshold
    selected = select_features_auto(
            X, y,
            importance_threshold=0.9,  # Very strict
            method='mutual_info'
            )

    # Then: All price columns preserved
    for col in price_cols:
        self.assertIn(col, selected.columns)
```

#### Test 4: `test_select_features_by_category`

**Purpose**: Verify category-based selection (momentum, valuation, etc.)

```python
def test_select_features_by_category(self):
    """Category-based selection should respect semantic groups."""
    # Given: Features from different Phase 9.3 categories
    X = pd.DataFrame({
        'momentum_rsi': np.random.randn(100),
        'momentum_macd': np.random.randn(100),
        'valuation_pe': np.random.randn(100),
        'quality_altman_z': np.random.randn(100)
        })
    y = np.random.randn(100)

    # When: Select only momentum category
    selected = select_features_by_category(
            X,
            categories=['momentum']
            )

    # Then: Only momentum features included
    self.assertIn('momentum_rsi', selected.columns)
    self.assertIn('momentum_macd', selected.columns)
    self.assertNotIn('valuation_pe', selected.columns)
```

### Implementation Requirements

#### Function Signature

```python
def select_features_auto(
        X: pd.DataFrame,
        y: pd.Series,
        importance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        method: str = 'combined',  # 'mutual_info', 'rf_importance', 'correlation', 'combined'
        preserve_columns: Optional[List[str]] = None,
        return_scores: bool = False
        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, float]]]:
    """
    Automated feature selection combining multiple methods.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    importance_threshold : float, default=0.01
        Minimum importance score to retain feature
    correlation_threshold : float, default=0.95
        Maximum correlation to consider features redundant
    method : str, default='combined'
        Selection method: 'mutual_info', 'rf_importance', 'correlation', 'combined'
    preserve_columns : list, optional
        Columns to always preserve (defaults to PRICE_COLUMNS)
    return_scores : bool, default=False
        Whether to return importance scores
        
    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Selected features, optionally with importance scores
    """
```

#### Integration with ETLConfig

Add to `finance_ml/ml_workflow/preprocessing/config.py`:

```python
@dataclass
class ETLConfig:
    # ... existing fields ...

    # Feature Selection
    auto_feature_selection: bool = False
    feature_importance_threshold: float = 0.01
    feature_correlation_threshold: float = 0.95
    feature_selection_method: str = 'combined'
```

#### Integration with etl_with_features()

Update `finance_ml/ml_workflow/preprocessing/api.py`:

```python
def etl_with_features(
        # ... existing parameters ...
        auto_feature_selection: bool = False,
        feature_importance_threshold: float = 0.01,
        **kwargs
        ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ETLMetrics]]:
    """
    ... existing docstring ...
    
    auto_feature_selection : bool, default=False
        Apply automated feature selection after feature engineering
    feature_importance_threshold : float, default=0.01
        Minimum importance threshold for feature selection
    """
```

### Acceptance Criteria

- [x] All 4 tests pass in `test_feature_selection_auto.py`
- [x] `select_features_auto()` function implemented in `features/selection.py`
- [x] `select_features_by_category()` function implemented in `features/selection.py`
- [x] PRICE_COLUMNS always preserved regardless of importance scores
- [x] ETLConfig extended with feature selection parameters (5 new parameters added)
- [x] `etl_with_features()` integrates feature selection as optional Stage 10
- [x] Documentation updated in code_guidelines.md Section 9.3.1

---

## Task 3: Semantic Column Classification Enhancement

### Priority: High

### Complexity: Low

### Business Impact: Better preprocessing, fewer "OTHER" columns, correct transformations

### Objective

Reduce the "OTHER" semantic category from 487 columns to <50 by implementing pattern-based classification and schema
lookup enhancements.

### Current Implementation

- `finance_ml/ml_workflow/preprocessing/column_semantics.py` (324 lines)
- `classify_columns()` function with basic regex patterns
- 487 of 591 columns classified as "OTHER" (82%)

### Target Implementation

**File**: `finance_ml/ml_workflow/preprocessing/column_semantics.py` (extend existing)

### TDD Test Specifications

#### Test 1: `test_classify_unknown_columns_by_pattern`

**Purpose**: Verify regex patterns classify _ltm, _fy, _fq suffixes

```python
def test_classify_unknown_columns_by_pattern(self):
    """Suffix patterns should guide semantic classification."""
    # Given: Columns with standard suffixes
    test_columns = [
        'debt_to_equity_ltm',  # Ratio
        'total_revenues_fy',  # Count/Market Value
        'net_income_fq',  # Count/Market Value
        'roe_ltm',  # Percentage
        'operating_margin_fy',  # Percentage
        'ev_ebitda_ltm'  # Ratio
        ]

    # When: Classify with pattern inference
    classifications = classify_columns_with_patterns(test_columns)

    # Then: Correct semantic categories assigned
    self.assertEqual(classifications['debt_to_equity_ltm'], 'RATIO')
    self.assertEqual(classifications['roe_ltm'], 'PERCENTAGE')
    self.assertIn(classifications['total_revenues_fy'], ['COUNT', 'MARKET_VALUE'])
```

#### Test 2: `test_classify_unknown_columns_by_schema`

**Purpose**: Verify COLUMN_SCHEMA lookup for 487 'OTHER' columns

```python
def test_classify_unknown_columns_by_schema(self):
    """COLUMN_SCHEMA should provide fallback classification."""
    # Given: Columns in schema but not classified
    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

    unclassified = [
        col for col in COLUMN_SCHEMA.keys()
        if col not in ['ticker', 'sector', 'region']
        ][:50]  # Sample 50

    # When: Classify with schema lookup
    classifications = classify_columns_with_schema_fallback(unclassified)

    # Then: Schema dtype informs semantic category
    for col, category in classifications.items():
        if COLUMN_SCHEMA[col]['dtype'] == 'float64':
            self.assertIn(category, ['RATIO', 'PERCENTAGE', 'MARKET_VALUE', 'COUNT'])
        elif COLUMN_SCHEMA[col]['dtype'] == 'object':
            self.assertEqual(category, 'CATEGORICAL')
```

#### Test 3: `test_semantic_classification_coverage_above_90pct`

**Purpose**: ≥90% of columns should have semantic category

```python
def test_semantic_classification_coverage_above_90pct(self):
    """After enhancements, <10% should be in OTHER category."""
    # Given: All 591 preprocessed columns
    all_columns = pd.read_json(
            'outputs/catalog/preprocessed_stocks_metadata.json'
            )['columns']

    # When: Classify with enhanced pipeline
    result = classify_columns(all_columns)

    # Then: OTHER category <10%
    total = len(all_columns)
    other_count = sum(1 for cat in result.values() if cat == 'OTHER')
    coverage_pct = 100 * (1 - other_count / total)

    self.assertGreaterEqual(coverage_pct, 90.0)
    self.assertLessEqual(other_count, 59)  # 10% of 591
```

### Implementation Requirements

#### Enhanced Pattern Matching

Add to `column_semantics.py`:

```python
# Suffix-based classification rules
SUFFIX_PATTERNS = {
    'RATIO': [
        r'_to_',  # debt_to_equity, price_to_book
        r'ev_ebitda',  # ev/ebitda variants
        r'p_e_',  # p/e variants
        r'p_b_',  # p/b variants
        r'_turnover',  # asset_turnover, inventory_turnover
        r'_coverage'  # interest_coverage
        ],
    'PERCENTAGE': [
        r'_margin',  # operating_margin, net_margin
        r'_pct',  # growth_pct, change_pct
        r'^roe',  # return on equity
        r'^roa',  # return on assets
        r'_yoy',  # year-over-year growth
        r'_yield'  # dividend_yield
        ],
    'MARKET_VALUE': [
        r'^market_cap',
        r'^enterprise_value',
        r'^total_assets',
        r'^total_revenues',
        r'^ebitda',
        r'^net_income'
        ],
    'COUNT': [
        r'^num_',  # num_employees, num_analysts
        r'_count$',  # analyst_count, rating_count
        r'^shares_'  # shares_outstanding
        ]
    }


def classify_columns_with_patterns(columns: List[str]) -> Dict[str, str]:
    """
    Classify columns using suffix and prefix patterns.
    
    Parameters
    ----------
    columns : list of str
        Column names to classify
        
    Returns
    -------
    dict
        Mapping of column -> semantic category
    """
    classifications = {}

    for col in columns:
        col_lower = col.lower()
        classified = False

        for category, patterns in SUFFIX_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, col_lower):
                    classifications[col] = category
                    classified = True
                    break
            if classified:
                break

        if not classified:
            classifications[col] = 'OTHER'

    return classifications
```

#### Schema-Based Fallback

```python
def classify_columns_with_schema_fallback(columns: List[str]) -> Dict[str, str]:
    """
    Use COLUMN_SCHEMA as fallback for unclassified columns.
    
    Parameters
    ----------
    columns : list of str
        Column names to classify
        
    Returns
    -------
    dict
        Mapping of column -> semantic category
    """
    from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

    classifications = {}

    for col in columns:
        if col in COLUMN_SCHEMA:
            dtype = COLUMN_SCHEMA[col]['dtype']
            role = COLUMN_SCHEMA[col].get('role', 'feature')

            # Infer from dtype and role
            if 'price' in col.lower() or 'target' in col.lower():
                classifications[col] = 'PRICE'
            elif role == 'categorical':
                classifications[col] = 'CATEGORICAL'
            elif dtype in ['float64', 'float32']:
                # Default numeric to RATIO if unknown
                classifications[col] = 'RATIO'
            elif dtype == 'int64':
                classifications[col] = 'COUNT'
            else:
                classifications[col] = 'OTHER'
        else:
            classifications[col] = 'OTHER'

    return classifications
```

#### Integration into classify_columns()

Update existing function:

```python
def classify_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Enhanced semantic classification with pattern and schema fallback.
    
    ... existing docstring ...
    """
    # Step 1: Existing price/market_value/ratio detection
    result = _classify_existing_patterns(df)

    # Step 2: NEW - Pattern-based classification for unclassified
    unclassified = [col for col, cat in result.items() if cat == 'OTHER']
    pattern_classifications = classify_columns_with_patterns(unclassified)
    result.update(pattern_classifications)

    # Step 3: NEW - Schema-based fallback for remaining OTHER
    still_unclassified = [col for col, cat in result.items() if cat == 'OTHER']
    schema_classifications = classify_columns_with_schema_fallback(still_unclassified)
    result.update(schema_classifications)

    return result
```

### Acceptance Criteria

- [x] All 3 tests pass in `test_semantic_classification.py`
- [x] SUFFIX_PATTERNS dictionary implemented with 4 categories
- [x] `classify_columns_with_patterns()` function implemented
- [x] `classify_columns_with_schema_fallback()` function implemented
- [x] `classify_columns()` integrates both new methods
- [x] OTHER category reduced from 487 to 27 columns (93.8% coverage achieved, exceeds 90% target)
- [x] Documentation updated in code_guidelines.md Section 8.5.4

---

## Success Metrics

### Quantitative Targets

- **Feature Selection**: Reduce feature dimensionality by 20-30% while maintaining R² > 0.90 of full model
- **Semantic Coverage**: Achieve ≥90% classification coverage (OTHER <59 of 591 columns)
- **Test Coverage**: 100% pass rate on 7 new tests
- **Performance**: Feature selection execution time <5 seconds for 6974 rows × 591 columns

### Qualitative Targets

- Improved model interpretability through automated feature pruning
- Reduced preprocessing errors from incorrect semantic transformations
- Enhanced ETL pipeline configurability through ETLConfig parameters

---

## Dependencies and Risks

### Dependencies

- Existing ETL pipeline (`etl_with_features()`)
- Schema registry (`finance_ml/ml_workflow/data/schema.py`)
- Column semantics module (`column_semantics.py`)

### Risks and Mitigation

1. **Risk**: Feature selection may remove important domain-specific features
    - **Mitigation**: PRICE_COLUMNS preservation policy, configurable thresholds, validation against baseline

2. **Risk**: Pattern-based classification may misclassify edge cases
    - **Mitigation**: Schema fallback, manual review of reclassified columns, test coverage for common patterns

3. **Risk**: Integration with ETL pipeline may introduce breaking changes
    - **Mitigation**: Optional `auto_feature_selection` flag (default=False), backward compatibility maintained

---

## Next Steps

### Immediate Actions

1. Create `tests/test_feature_selection_auto.py` with 4 test cases
2. Extend `tests/test_semantic_classification.py` with 3 new test cases
3. Implement `features/selection.py` module with TDD approach
4. Enhance `column_semantics.py` with pattern and schema fallback

### Sprint 2 Handoff

After Phase 9.3 completion, proceed to:

- **Phase 9.4 Implementation Plan**: Classification robustness (Tasks 2, 4, 5)
- **Phase 9.5 Implementation Plan**: Regression improvements (Tasks 6, 7)

---

## Document Control

**Reviewed By**: TBD  
**Approved By**: TBD  
**Last Modified**: 2025-12-09  
**Related Documents**:

- `code_guidelines.md` v1.10
- `Phase_9.3_feature_enhancement_plan.md` v1.1
- Current State Analysis (2025-12-09)
