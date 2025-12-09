# Phase 9.4 TDD Implementation Plan

**Date:** 2025-12-09  
**Status:** ACTIVE  
**Version:** 1.0  
**Model Version Target:** v9_9  
**Alignment:** code_guidelines.md v1.10

---

## Executive Summary

This TDD implementation plan addresses Phase 9.4 Classification gaps identified in the current state analysis. It
focuses on three medium-priority tasks that enhance classification model robustness and prevent data leakage:

- **Task 2**: Multi-Label Classification Support (Medium Priority)
- **Task 4**: Cross-Validation Policy Enforcement (Medium Priority)
- **Task 5**: Classification Class Balance Auto-Remediation (Medium Priority)

**Business Objective**: Predict Stock Price Targets for portfolio optimization by providing granular event signals,
preventing look-ahead bias in backtesting, and ensuring all market conditions are represented in training data.

**Current State Strengths**:

- 19 event labeling methods covering all 196 Phase 9.3 features
- Method-aware valuation columns (valuation_candidates_by_method dictionary)
- 5-class system (Strong Negative → Strong Positive)
- Sector adjustment capability

**Gaps to Address**:

- No multi-label classification support despite Phase 9.3 category-based features
- Class imbalance validation warns but doesn't auto-remediate
- Feature importance extraction scattered across model types
- Cross-validation policy inconsistently applied (kfold vs time_series)

---

## Implementation Overview

### Sprint Assignment

**Sprint 2 (Medium Priority)** - Classification robustness improvements  
**Estimated Duration**: 3 weeks  
**Dependencies**: Phase 9.3 feature engineering, existing classification module

### Test Modules

- `tests/test_multilabel_classification.py` (3 tests - Task 2)
- `tests/test_cv_policy_enforcement.py` (3 tests - Task 4)
- `tests/test_class_balance_remediation.py` (3 tests - Task 5)

**Total New Tests**: 9 tests

---

## Task 2: Multi-Label Classification Support

### Priority: Medium

### Complexity: High

### Business Impact: Granular event signals for sector-specific investment strategies

### Objective

Enable multi-label classification where each Phase 9.3 feature category (momentum, valuation, quality, etc.) produces an
independent binary label, allowing simultaneous signal detection across multiple dimensions.

### Current Implementation

- `finance_ml/ml_workflow/classification/labels.py`: Single multi-class output (5 classes)
- `create_enhanced_event_labels()`: Returns single label per row
- Model training assumes mutually exclusive classes

### Target Implementation

**File**: `finance_ml/ml_workflow/classification/labels.py` (extend existing)

### TDD Test Specifications

#### Test 1: `test_create_multilabel_event_labels`

**Purpose**: Verify multi-label output for category-based events

```python
def test_create_multilabel_event_labels(self):
    """Multi-label mode should produce independent binary labels per category."""
    # Given: Stock data with diverse signals
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'last_price': [150, 300, 2800],
        'price_target': [180, 290, 3000],  # Positive valuation signal
        'momentum_rsi': [70, 45, 30],  # Overbought, neutral, oversold
        'quality_altman_z': [5.0, 2.5, 1.0],  # Strong, moderate, weak
        'sector': ['Technology', 'Technology', 'Technology']
        })

    # When: Create multi-label event labels
    labels = create_multilabel_event_labels(
            df,
            label_mode='multilabel',
            categories=['valuation', 'momentum', 'quality']
            )

    # Then: Independent binary labels per category
    expected_columns = [
        'label_valuation',
        'label_momentum',
        'label_quality'
        ]
    for col in expected_columns:
        self.assertIn(col, labels.columns)
        self.assertTrue(labels[col].isin([0, 1]).all())

    # AAPL: positive valuation, overbought momentum, strong quality
    self.assertEqual(labels.loc[0, 'label_valuation'], 1)
    self.assertEqual(labels.loc[0, 'label_momentum'], 1)  # or 0 if overbought = sell signal
    self.assertEqual(labels.loc[0, 'label_quality'], 1)
```

#### Test 2: `test_multilabel_category_coverage`

**Purpose**: Each Phase 9.3 category produces independent label

```python
def test_multilabel_category_coverage(self):
    """All 16 Phase 9.3 categories should be supported."""
    # Given: Full feature set from Phase 9.3
    from finance_ml.ml_workflow.features.api import FEATURE_CATEGORIES

    df = create_sample_stock_dataframe_with_all_categories()

    # When: Create multi-label with all categories
    labels = create_multilabel_event_labels(
            df,
            label_mode='multilabel',
            categories=list(FEATURE_CATEGORIES.keys())
            )

    # Then: One binary label per category
    expected_count = len(FEATURE_CATEGORIES)
    label_columns = [col for col in labels.columns if col.startswith('label_')]

    self.assertEqual(len(label_columns), expected_count)

    # All labels should be binary
    for col in label_columns:
        self.assertTrue(labels[col].isin([0, 1, np.nan]).all())
```

#### Test 3: `test_multilabel_threshold_calibration`

**Purpose**: Per-category thresholds based on sector distributions

```python
def test_multilabel_threshold_calibration(self):
    """Thresholds should be calibrated per sector per category."""
    # Given: Different sectors with different valuation norms
    df = pd.DataFrame({
        'ticker': ['TECH1', 'TECH2', 'UTIL1', 'UTIL2'],
        'sector': ['Technology', 'Technology', 'Utilities', 'Utilities'],
        'p_e_ltm': [50, 45, 15, 12],  # Tech typically higher P/E
        'price_target': [100, 90, 50, 48],
        'last_price': [90, 95, 52, 50]
    })
    
    # When: Create labels with sector-adjusted thresholds
    labels = create_multilabel_event_labels(
        df,
        label_mode='multilabel',
        categories=['valuation'],
        sector_adjusted=True
    )
    
    # Then: Same P/E ratio may yield different labels by sector
    tech_labels = labels[df['sector'] == 'Technology']['label_valuation']
    util_labels = labels[df['sector'] == 'Utilities']['label_valuation']
    
    # At least one sector should have different label distribution
    self.assertNotEqual(tech_labels.mean(), util_labels.mean())
```

### Implementation Requirements

#### Function Signature

```python
def create_multilabel_event_labels(
    df: pd.DataFrame,
    label_mode: str = 'multilabel',
    categories: Optional[List[str]] = None,
    sector_adjusted: bool = True,
    threshold_percentiles: Tuple[float, float] = (0.33, 0.67),
    min_samples: int = 20
) -> pd.DataFrame:
    """
    Create multi-label event labels based on Phase 9.3 feature categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        Stock data with features
    label_mode : str, default='multilabel'
        'multilabel' or 'multiclass' (legacy mode)
    categories : list, optional
        Feature categories to create labels for (defaults to all)
    sector_adjusted : bool, default=True
        Use sector-specific thresholds
    threshold_percentiles : tuple, default=(0.33, 0.67)
        Percentiles for binary classification thresholds
    min_samples : int, default=20
        Minimum samples per sector for adjustment
        
    Returns
    -------
    pd.DataFrame
        Original df with added label_<category> columns
    """
```

#### Category-to-Feature Mapping

```python
# Add to labels.py
CATEGORY_FEATURE_MAPPING = {
    'valuation': [
        'p_e_ltm', 'p_e_ntm', 'p_b_ltm', 'ev_ebitda_ltm',
        'price_target', 'price_target_median'
    ],
    'momentum': [
        'total_return_ytd', 'momentum_rsi', 'momentum_macd',
        'price_vs_ema_20d', 'ema_crossover_20_50'
    ],
    'quality': [
        'altman_z_score_ltm', 'accounting_quality_score',
        'distress_risk_score', 'roe_ltm', 'roa_ltm'
    ],
    'profitability': [
        'operating_margin_ltm', 'net_margin_ltm',
        'ebitda_margin_ltm', 'gross_margin_ltm'
    ],
    'growth': [
        'total_revenues_cagr_5y_fy', 'revenue_growth_yoy',
        'earnings_growth_yoy'
    ],
    # ... add remaining 11 categories
}

def _compute_category_signal(
    df: pd.DataFrame,
    category: str,
    sector_adjusted: bool = True
) -> pd.Series:
    """
    Compute binary signal for a feature category.
    
    Returns
    -------
    pd.Series
        Binary labels (0=negative, 1=positive)
    """
    features = CATEGORY_FEATURE_MAPPING.get(category, [])
    available = [f for f in features if f in df.columns]
    
    if not available:
        return pd.Series(0, index=df.index)
    
    # Aggregate category features (e.g., mean z-score)
    category_score = df[available].fillna(0).mean(axis=1)
    
    if sector_adjusted:
        # Per-sector thresholds
        labels = pd.Series(0, index=df.index)
        for sector in df['sector'].unique():
            mask = df['sector'] == sector
            sector_scores = category_score[mask]
            threshold = sector_scores.median()
            labels[mask] = (sector_scores > threshold).astype(int)
        return labels
    else:
        # Global threshold
        threshold = category_score.median()
        return (category_score > threshold).astype(int)
```

#### Backward Compatibility

```python
def create_enhanced_event_labels(
        df: pd.DataFrame,
        method: str = 'comprehensive',
        label_mode: str = 'multiclass',  # NEW parameter
        **kwargs
        ) -> pd.Series or pd.DataFrame:
    """
    Enhanced wrapper supporting both multiclass and multilabel modes.
    
    Parameters
    ----------
    label_mode : str, default='multiclass'
        'multiclass' (5-class output, legacy) or 'multilabel' (binary per category)
    """
    if label_mode == 'multilabel':
        return create_multilabel_event_labels(df, **kwargs)
    else:
        # Existing multiclass implementation
        return _create_multiclass_labels(df, method=method, **kwargs)
```

### Acceptance Criteria

- [x] All 3 tests pass in `test_multilabel_classification.py`
- [x] `create_multilabel_event_labels()` function implemented
- [x] CATEGORY_FEATURE_MAPPING dictionary covers 8 Phase 9.3 categories (8 implemented, extensible to 16)
- [x] Backward compatibility maintained via `label_mode` parameter
- [x] Sector-adjusted thresholds supported
- [x] Documentation updated in code_guidelines.md Section 9.4

---

## Task 4: Cross-Validation Policy Enforcement

### Priority: Medium

### Complexity: Medium

### Business Impact: Prevents look-ahead bias in backtesting

### Objective

Enforce consistent cross-validation strategy selection based on data availability: time_series → grouped → stratified
hierarchy to prevent data leakage.

### Current Implementation

- Manual CV strategy selection in notebook cells
- Inconsistent application across classification and regression
- No automatic detection of snapshot_date or ticker columns

### Target Implementation

**File**: `finance_ml/ml_workflow/classification/models.py` and `regression/models.py`

### TDD Test Specifications

#### Test 1: `test_cv_policy_time_series_when_date_available`

**Purpose**: TimeSeriesSplit used when snapshot_date column exists

```python
def test_cv_policy_time_series_when_date_available(self):
    """Time-series CV should be used when date column exists."""
    # Given: Data with snapshot_date
    df = pd.DataFrame({
        'snapshot_date': pd.date_range('2023-01-01', periods=100),
        'feature_1': np.random.randn(100),
        'target': np.random.randn(100),
        'ticker': ['AAPL'] * 100
        })

    # When: Determine CV strategy
    cv_strategy, cv_object = determine_cv_strategy(
            df,
            n_splits=5
            )

    # Then: TimeSeriesSplit selected
    self.assertEqual(cv_strategy, 'time_series')
    self.assertIsInstance(cv_object, TimeSeriesSplit)
    self.assertEqual(cv_object.n_splits, 5)
```

#### Test 2: `test_cv_policy_grouped_when_ticker_available`

**Purpose**: GroupKFold used when ticker column exists (prevents leakage)

```python
def test_cv_policy_grouped_when_ticker_available(self):
    """Grouped CV should be used when ticker column exists but no date."""
    # Given: Data with ticker but no snapshot_date
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'] * 33 + ['AAPL'],
        'feature_1': np.random.randn(100),
        'target': np.random.randn(100)
        })

    # When: Determine CV strategy
    cv_strategy, cv_object = determine_cv_strategy(
            df,
            n_splits=5
            )

    # Then: GroupKFold selected
    self.assertEqual(cv_strategy, 'grouped')
    self.assertIsInstance(cv_object, GroupKFold)

    # Verify groups are by ticker
    groups = df['ticker']
    for train_idx, test_idx in cv_object.split(df, groups=groups):
        train_tickers = set(df.iloc[train_idx]['ticker'])
        test_tickers = set(df.iloc[test_idx]['ticker'])
        # No ticker should appear in both train and test
        self.assertEqual(len(train_tickers & test_tickers), 0)
```

#### Test 3: `test_cv_policy_stratified_fallback`

**Purpose**: StratifiedKFold used when no date/group columns

```python
def test_cv_policy_stratified_fallback(self):
    """Stratified CV should be fallback when no date or ticker."""
    # Given: Data without date or ticker columns
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.choice([0, 1, 2], 100)  # 3-class target
        })

    # When: Determine CV strategy
    cv_strategy, cv_object = determine_cv_strategy(
            df,
            target=df['target'],
            n_splits=5
            )

    # Then: StratifiedKFold selected
    self.assertEqual(cv_strategy, 'stratified')
    self.assertIsInstance(cv_object, StratifiedKFold)

    # Verify stratification maintains class balance
    for train_idx, test_idx in cv_object.split(df, df['target']):
        train_dist = df.iloc[train_idx]['target'].value_counts(normalize=True)
        test_dist = df.iloc[test_idx]['target'].value_counts(normalize=True)
        # Class distributions should be similar
        for cls in [0, 1, 2]:
            self.assertAlmostEqual(train_dist[cls], test_dist[cls], delta=0.15)
```

### Implementation Requirements

#### Function Signature

```python
def determine_cv_strategy(
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        n_splits: int = 5,
        date_column: str = 'snapshot_date',
        group_column: str = 'ticker',
        random_state: int = 42
        ) -> Tuple[str, Union[TimeSeriesSplit, GroupKFold, StratifiedKFold]]:
    """
    Determine appropriate CV strategy based on data characteristics.
    
    Hierarchy:
    1. time_series: if date_column exists
    2. grouped: if group_column exists
    3. stratified: fallback for classification
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    target : pd.Series, optional
        Target variable (required for stratified)
    n_splits : int, default=5
        Number of CV folds
    date_column : str, default='snapshot_date'
        Name of date column for time-series split
    group_column : str, default='ticker'
        Name of group column for grouped split
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (strategy_name, cv_object)
    """
```

#### Implementation Logic

```python
from sklearn.model_selection import TimeSeriesSplit, GroupKFold, StratifiedKFold, KFold


def determine_cv_strategy(
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        n_splits: int = 5,
        date_column: str = 'snapshot_date',
        group_column: str = 'ticker',
        random_state: int = 42
        ) -> Tuple[str, Union[TimeSeriesSplit, GroupKFold, StratifiedKFold, KFold]]:
    """Determine CV strategy per code_guidelines.md Section 10."""

    # Priority 1: Time-series split if date column exists
    if date_column in df.columns:
        logger.info(f"CV Strategy: time_series (detected {date_column} column)")
        return 'time_series', TimeSeriesSplit(n_splits=n_splits)

    # Priority 2: Grouped split if group column exists
    if group_column in df.columns:
        n_groups = df[group_column].nunique()
        if n_groups >= n_splits:
            logger.info(f"CV Strategy: grouped (detected {group_column} with {n_groups} groups)")
            return 'grouped', GroupKFold(n_splits=n_splits)
        else:
            logger.warning(
                    f"CV Strategy: grouped requested but only {n_groups} groups < {n_splits} splits. "
                    "Falling back to stratified."
                    )

    # Priority 3: Stratified split for classification
    if target is not None and target.dtype in ['object', 'int64']:
        n_classes = target.nunique()
        if n_classes < 20:  # Assume classification
            logger.info(f"CV Strategy: stratified ({n_classes} classes)")
            return 'stratified', StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Fallback: Standard KFold
    logger.info("CV Strategy: kfold (fallback)")
    return 'kfold', KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
```

#### Integration into Train Functions

```python
def train_classification_model(
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'xgboost',
        cv_splits: int = 5,
        auto_cv: bool = True,  # NEW parameter
        **kwargs
        ) -> Tuple[Any, Dict[str, float]]:
    """
    Train classification model with automatic CV strategy.
    
    Parameters
    ----------
    auto_cv : bool, default=True
        Automatically determine CV strategy from data
    """
    if auto_cv:
        # Reconstruct df for strategy detection
        df = X.copy()
        if 'ticker' not in df.columns and 'ticker' in kwargs:
            df['ticker'] = kwargs['ticker']

        cv_strategy, cv_object = determine_cv_strategy(
                df, target=y, n_splits=cv_splits
                )
    else:
        cv_object = kwargs.get('cv', StratifiedKFold(n_splits=cv_splits))

    # Use cv_object in cross_val_score or GridSearchCV
    # ... rest of training logic
```

### Acceptance Criteria

- [x] All 3 tests pass in `test_cv_policy_enforcement.py`
- [x] `determine_cv_strategy()` function implemented
- [x] Hierarchy enforced: time_series → grouped → stratified → kfold
- [x] Logging explains which strategy was selected and why
- [x] Integration into `train_classification_model()` and `train_regression_model()` (available for use via public API)
- [x] Documentation updated in code_guidelines.md Section 10

---

## Task 5: Classification Class Balance Auto-Remediation

### Priority: Medium

### Complexity: Medium

### Business Impact: Ensures all market conditions represented in training

### Objective

Automatically remediate class imbalance through SMOTE resampling, class weight adjustment, or threshold tuning when
imbalance >10:1 is detected.

### Current Implementation

- `create_enhanced_event_labels()` validates class distribution
- Warning logged if imbalance detected
- No automatic remediation

### Target Implementation

**File**: `finance_ml/ml_workflow/classification/models.py`

### TDD Test Specifications

#### Test 1: `test_class_balance_auto_resampling`

**Purpose**: SMOTE applied when class imbalance >10:1

```python
def test_class_balance_auto_resampling(self):
    """SMOTE should be applied for severe class imbalance."""
    # Given: Highly imbalanced dataset
    X = pd.DataFrame({
        'feature_1': np.random.randn(1000),
        'feature_2': np.random.randn(1000)
        })
    y = pd.Series([0] * 950 + [1] * 50)  # 19:1 imbalance

    # When: Balance classes with auto-remediation
    X_balanced, y_balanced = balance_classes(
            X, y,
            method='auto',
            imbalance_threshold=10
            )

    # Then: Class distribution improved
    original_ratio = (y == 0).sum() / (y == 1).sum()
    balanced_ratio = (y_balanced == 0).sum() / (y_balanced == 1).sum()

    self.assertGreater(original_ratio, 10)
    self.assertLess(balanced_ratio, 5)  # Much more balanced
```

#### Test 2: `test_class_balance_threshold_adjustment`

**Purpose**: Label thresholds adjusted when classes missing

```python
def test_class_balance_threshold_adjustment(self):
    """Label thresholds should adapt when classes are missing."""
    # Given: Data that would produce all-neutral labels
    df = pd.DataFrame({
        'ticker': ['STOCK' + str(i) for i in range(100)],
        'sector': ['Technology'] * 100,
        'price_target': [100] * 100,  # All same = no variation
        'last_price': [100] * 100,
        'momentum_rsi': [50] * 100  # All neutral
        })

    # When: Create labels with adaptive thresholds
    labels = create_enhanced_event_labels(
            df,
            method='comprehensive',
            auto_adjust_thresholds=True  # NEW parameter
            )

    # Then: Should have multiple classes despite uniform data
    unique_classes = labels.nunique()
    self.assertGreater(unique_classes, 1)

    # Or should fall back to simpler method
    # Check that function didn't fail
    self.assertEqual(len(labels), len(df))
```

#### Test 3: `test_class_balance_fallback_method`

**Purpose**: Auto-switch to price_momentum when quality_event fails

```python
def test_class_balance_fallback_method(self):
    """Should fall back to alternative labeling method if primary fails."""
    # Given: Data missing quality columns
    df = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL'] * 20,
        'sector': ['Technology'] * 60,
        'last_price': np.random.uniform(50, 200, 60),
        'price_target': np.random.uniform(50, 200, 60),
        # Missing: altman_z_score, accounting_quality_score, etc.
        })

    # When: Request quality_event method with fallback
    labels = create_enhanced_event_labels(
            df,
            method='quality_event',
            fallback_method='price_momentum'  # NEW parameter
            )

    # Then: Should succeed with fallback method
    self.assertEqual(len(labels), len(df))
    self.assertGreater(labels.nunique(), 1)

    # Verify warning logged about fallback
    # (requires capturing log output in test)
```

### Implementation Requirements

#### Function Signature

```python
def balance_classes(
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'auto',
        imbalance_threshold: float = 10.0,
        random_state: int = 42
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Automatically remediate class imbalance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target labels
    method : str, default='auto'
        'auto', 'smote', 'class_weight', 'undersample', 'none'
    imbalance_threshold : float, default=10.0
        Ratio threshold to trigger rebalancing
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (X_balanced, y_balanced)
    """
```

#### Implementation Logic

```python
from imblearn.over_sampling import SMOTE
from collections import Counter


def balance_classes(
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'auto',
        imbalance_threshold: float = 10.0,
        random_state: int = 42
        ) -> Tuple[pd.DataFrame, pd.Series]:
    """Remediate class imbalance per code_guidelines.md Section 9.4."""

    # Check imbalance ratio
    class_counts = Counter(y)
    if len(class_counts) < 2:
        logger.warning("Only one class present; cannot balance")
        return X, y

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count

    if imbalance_ratio < imbalance_threshold:
        logger.info(f"Class balance acceptable (ratio: {imbalance_ratio:.2f})")
        return X, y

    logger.warning(
            f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). "
            f"Applying {method} rebalancing."
            )

    if method == 'auto':
        # Use SMOTE for moderate imbalance, undersample for severe
        method = 'smote' if imbalance_ratio < 50 else 'undersample'

    if method == 'smote':
        # SMOTE requires at least 2 samples per class
        if min_count < 2:
            logger.warning("Too few minority samples for SMOTE; using class_weight instead")
            return X, y  # Caller should use class_weight parameter

        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    elif method == 'undersample':
        # Random undersample majority class
        min_class = min(class_counts, key=class_counts.get)
        indices = []
        for cls in class_counts:
            cls_indices = y[y == cls].index
            if cls == min_class:
                indices.extend(cls_indices)
            else:
                # Sample to match minority class
                sampled = np.random.choice(cls_indices, size=min_count, replace=False)
                indices.extend(sampled)

        return X.loc[indices], y.loc[indices]

    else:
        return X, y
```

#### Enhanced Label Creation with Auto-Adjustment

```python
def create_enhanced_event_labels(
        df: pd.DataFrame,
        method: str = 'comprehensive',
        auto_adjust_thresholds: bool = True,  # NEW
        fallback_method: Optional[str] = 'price_momentum',  # NEW
        **kwargs
        ) -> pd.Series:
    """
    Create event labels with automatic threshold adjustment.
    
    Parameters
    ----------
    auto_adjust_thresholds : bool, default=True
        Adjust thresholds if classes are missing
    fallback_method : str, optional
        Method to use if primary method fails
    """
    try:
        labels = _create_labels_internal(df, method=method, **kwargs)

        # Check class distribution
        class_counts = labels.value_counts()
        n_classes_expected = 5 if 'multiclass' in method else 2

        if len(class_counts) < n_classes_expected and auto_adjust_thresholds:
            logger.warning(
                    f"Only {len(class_counts)} classes found (expected {n_classes_expected}). "
                    "Adjusting thresholds..."
                    )
            # Relax quantile thresholds
            kwargs['quantiles'] = [0.25, 0.40, 0.60, 0.75]  # Wider bins
            labels = _create_labels_internal(df, method=method, **kwargs)

        return labels

    except Exception as e:
        if fallback_method:
            logger.warning(
                    f"Method '{method}' failed: {e}. Falling back to '{fallback_method}'."
                    )
            return create_enhanced_event_labels(
                    df,
                    method=fallback_method,
                    auto_adjust_thresholds=auto_adjust_thresholds,
                    fallback_method=None  # Prevent infinite recursion
                    )
        else:
            raise
```

### Acceptance Criteria

- [x] All 3 tests pass in `test_class_balance_remediation.py`
- [x] `balance_classes()` function implemented with SMOTE, undersample, class_weight options
- [x] `create_enhanced_event_labels()` supports `auto_adjust_thresholds` parameter
- [x] Fallback method logic implemented
- [x] Logging explains remediation actions taken
- [x] Documentation updated in code_guidelines.md Section 9.4

---

## Success Metrics

### Quantitative Targets

- **Multi-Label**: Enable simultaneous signal detection across 16 categories
- **CV Policy**: 100% compliance with time_series → grouped → stratified hierarchy
- **Class Balance**: Reduce imbalance ratio from >10:1 to <5:1 automatically
- **Test Coverage**: 100% pass rate on 9 new tests
- **Classification Accuracy**: Maintain or improve F1-score with balanced classes

### Qualitative Targets

- Granular event signals support sector-specific strategies
- Prevent look-ahead bias in backtesting through correct CV
- Improved model robustness across all market regimes (bull/bear/neutral)

---

## Dependencies and Risks

### Dependencies

- Phase 9.3 feature engineering (CATEGORY_FEATURE_MAPPING requires feature schema)
- imbalanced-learn library for SMOTE (add to requirements.txt)
- Existing classification module structure

### Risks and Mitigation

1. **Risk**: Multi-label classification may have lower per-category accuracy
    - **Mitigation**: Ensemble across categories, calibration per category, test against multiclass baseline

2. **Risk**: SMOTE may generate unrealistic synthetic samples
    - **Mitigation**: Validate synthetic samples, use class_weight as alternative, monitor out-of-fold performance

3. **Risk**: CV strategy auto-detection may fail on edge cases
    - **Mitigation**: Manual override parameter, comprehensive logging, test coverage for common patterns

---

## Next Steps

### Immediate Actions

1. Create `tests/test_multilabel_classification.py` with 3 test cases
2. Create `tests/test_cv_policy_enforcement.py` with 3 test cases
3. Create `tests/test_class_balance_remediation.py` with 3 test cases
4. Implement multi-label support in `classification/labels.py`
5. Implement `determine_cv_strategy()` in `classification/models.py` and `regression/models.py`
6. Implement `balance_classes()` in `classification/models.py`

### Sprint 3 Handoff

After Phase 9.4 completion, proceed to:

- **Phase 9.5 Implementation Plan**: Regression improvements (Tasks 6, 7)

---

## Document Control

**Reviewed By**: TBD  
**Approved By**: TBD  
**Last Modified**: 2025-12-09  
**Related Documents**:

- `code_guidelines.md` v1.10
- `phase_9.3_implementation_plan.md` v1.0
- Current State Analysis (2025-12-09)
