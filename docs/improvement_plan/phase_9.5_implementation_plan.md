# Phase 9.5 TDD Implementation Plan

**Date:** 2025-12-09  
**Status:** ACTIVE  
**Version:** 1.0  
**Model Version Target:** v9_9  
**Alignment:** code_guidelines.md v1.10

---

## Executive Summary

This TDD implementation plan addresses Phase 9.5 Regression gaps identified in the current state analysis. It focuses on
two tasks: one low-priority advanced capability and one high-priority operational improvement:

- **Task 6**: Stacking Ensemble Hyperparameter Tuning (Low Priority)
- **Task 7**: Unified Test Data Alignment (High Priority)

**Business Objective**: Predict Stock Price Targets for portfolio optimization by maximizing prediction accuracy through
automated hyperparameter tuning and eliminating runtime errors from feature misalignment.

**Current State Strengths**:

- Stacking ensemble with Huber loss (outlier-robust)
- Quantile regression (p10, p50, p90) for uncertainty quantification
- Classification meta-features integration
- Non-negativity constraints applied post-calibration

**Gaps to Address**:

- No automated hyperparameter tuning for stacking base models
- Feature alignment between train/test requires manual intervention
- Interaction feature regeneration duplicated in multiple cells
- Prediction clipping strategy documented but bounds not configurable

---

## Implementation Overview

### Sprint Assignment

**Sprint 3 (Mixed Priority)** - Regression improvements  
**Estimated Duration**: 2 weeks  
**Dependencies**: Phase 9.4 classification, existing regression module

### Test Modules

- `tests/test_stacking_hyperparameter.py` (3 tests - Task 6)
- `tests/test_feature_alignment.py` (3 tests - Task 7)

**Total New Tests**: 6 tests

---

## Task 7: Unified Test Data Alignment

### Priority: High

### Complexity: Low

### Business Impact: Eliminates runtime errors in batch prediction pipelines

### Objective

Automatically align test features to trained model's expected feature set, filling missing columns with zeros and
removing extra columns, to prevent prediction-time errors.

### Current Implementation

- Manual feature alignment in notebook cells (Section 6.4)
- Duplicate code across multiple prediction cells
- Runtime errors when test data has different feature set than training

### Target Implementation

**File**: `finance_ml/ml_workflow/regression/dataset.py`

### TDD Test Specifications

#### Test 1: `test_align_test_features_to_model`

**Purpose**: X_test aligned to model.feature_names_in_ automatically

```python
def test_align_test_features_to_model(self):
    """Test features should be aligned to model's expected features."""
    # Given: Trained model expecting specific features
    X_train = pd.DataFrame({
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100),
        'feature_c': np.random.randn(100)
        })
    y_train = np.random.randn(100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # And: Test data with different features
    X_test = pd.DataFrame({
        'feature_a': np.random.randn(20),
        'feature_b': np.random.randn(20),
        'feature_d': np.random.randn(20)  # Extra feature
        # Missing: feature_c
        })

    # When: Align test features to model
    X_test_aligned = align_features_to_model(X_test, model)

    # Then: Features match model's expectations
    self.assertListEqual(
            list(X_test_aligned.columns),
            list(model.feature_names_in_)
            )
    self.assertEqual(X_test_aligned.shape[1], X_train.shape[1])
```

#### Test 2: `test_align_fills_missing_with_zero`

**Purpose**: Missing columns filled with 0, not dropped

```python
def test_align_fills_missing_with_zero(self):
    """Missing features should be filled with zero."""
    # Given: Model trained on features [a, b, c]
    X_train = pd.DataFrame({
        'feature_a': [1, 2, 3],
        'feature_b': [4, 5, 6],
        'feature_c': [7, 8, 9]
        })
    y_train = [10, 11, 12]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # And: Test data missing feature_c
    X_test = pd.DataFrame({
        'feature_a': [1.5],
        'feature_b': [4.5]
        })

    # When: Align features
    X_test_aligned = align_features_to_model(X_test, model)

    # Then: Missing feature_c filled with 0
    self.assertIn('feature_c', X_test_aligned.columns)
    self.assertEqual(X_test_aligned['feature_c'].iloc[0], 0.0)

    # And: Prediction doesn't raise error
    prediction = model.predict(X_test_aligned)
    self.assertEqual(len(prediction), 1)
```

#### Test 3: `test_align_removes_extra_columns`

**Purpose**: Extra columns in X_test not passed to model

```python
def test_align_removes_extra_columns(self):
    """Extra features not in training should be removed."""
    # Given: Model trained on features [a, b]
    X_train = pd.DataFrame({
        'feature_a': np.random.randn(100),
        'feature_b': np.random.randn(100)
        })
    y_train = np.random.randn(100)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # And: Test data with extra features
    X_test = pd.DataFrame({
        'feature_a': np.random.randn(20),
        'feature_b': np.random.randn(20),
        'feature_z': np.random.randn(20),  # Extra
        'feature_y': np.random.randn(20)  # Extra
        })

    # When: Align features
    X_test_aligned = align_features_to_model(X_test, model)

    # Then: Extra features removed
    self.assertNotIn('feature_z', X_test_aligned.columns)
    self.assertNotIn('feature_y', X_test_aligned.columns)
    self.assertEqual(len(X_test_aligned.columns), 2)
```

### Implementation Requirements

#### Function Signature

```python
def align_features_to_model(
    X_test: pd.DataFrame,
    model: Any,
    fill_value: float = 0.0,
    warn_missing: bool = True,
    warn_extra: bool = True
) -> pd.DataFrame:
    """
    Align test features to match trained model's expected features.
    
    Parameters
    ----------
    X_test : pd.DataFrame
        Test features to align
    model : sklearn estimator or compatible
        Trained model with feature_names_in_ attribute
    fill_value : float, default=0.0
        Value to fill missing features
    warn_missing : bool, default=True
        Log warning for missing features
    warn_extra : bool, default=True
        Log warning for extra features
        
    Returns
    -------
    pd.DataFrame
        Aligned test features matching model's feature order
    """
```

#### Implementation Logic

```python
import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

def align_features_to_model(
    X_test: pd.DataFrame,
    model: Any,
    fill_value: float = 0.0,
    warn_missing: bool = True,
    warn_extra: bool = True
) -> pd.DataFrame:
    """
    Align test features to trained model per code_guidelines.md Section 7.5.
    
    This function ensures X_test has exactly the features the model expects,
    in the correct order, preventing prediction-time errors.
    """
    # Get expected features from model
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_)
    elif hasattr(model, 'feature_name_'):  # LightGBM
        expected_features = list(model.feature_name_)
    elif hasattr(model, 'get_booster'):  # XGBoost
        expected_features = model.get_booster().feature_names
    else:
        logger.warning(
            "Model does not expose feature names; returning X_test unchanged"
        )
        return X_test
    
    # Identify missing and extra features
    test_features = set(X_test.columns)
    expected_features_set = set(expected_features)
    
    missing_features = expected_features_set - test_features
    extra_features = test_features - expected_features_set
    
    # Log warnings
    if missing_features and warn_missing:
        logger.warning(
            f"X_test missing {len(missing_features)} features expected by model. "
            f"Filling with {fill_value}. Missing: {sorted(missing_features)[:5]}..."
        )
    
    if extra_features and warn_extra:
        logger.warning(
            f"X_test has {len(extra_features)} extra features not in model. "
            f"Removing. Extra: {sorted(extra_features)[:5]}..."
        )
    
    # Create aligned dataframe
    X_aligned = X_test.copy()
    
    # Add missing features with fill_value
    for feature in missing_features:
        X_aligned[feature] = fill_value
    
    # Select only expected features in correct order
    X_aligned = X_aligned[expected_features]
    
    return X_aligned
```

#### Integration into Prediction Wrapper

```python
def predict_with_model(
        model: Any,
        X_test: pd.DataFrame,
        auto_align: bool = True,
        **kwargs
        ) -> np.ndarray:
    """
    Predict with automatic feature alignment.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    auto_align : bool, default=True
        Automatically align features to model
    **kwargs
        Additional arguments passed to model.predict()
        
    Returns
    -------
    np.ndarray
        Predictions
    """
    if auto_align:
        X_test = align_features_to_model(X_test, model)

    return model.predict(X_test, **kwargs)
```

#### Notebook Integration Example

```python
# Before (50+ lines of manual alignment):
# if set(X_test.columns) != set(model.feature_names_in_):
#     missing = set(model.feature_names_in_) - set(X_test.columns)
#     for col in missing:
#         X_test[col] = 0
#     X_test = X_test[model.feature_names_in_]

# After (1 line):
from finance_ml.ml_workflow.regression.dataset import predict_with_model

predictions = predict_with_model(stacking_model, X_test)
```

### Acceptance Criteria

- [ ] All 3 tests pass in `test_feature_alignment.py`
- [ ] `align_features_to_model()` function implemented in `regression/dataset.py`
- [ ] `predict_with_model()` wrapper integrates alignment
- [ ] Support for sklearn, XGBoost, LightGBM, CatBoost models
- [ ] Notebook code reduced by ~50 lines in Section 6.4
- [ ] Documentation updated in code_guidelines.md Section 7.5

---

## Task 6: Stacking Ensemble Hyperparameter Tuning

### Priority: Low

### Complexity: High

### Business Impact: Marginal accuracy gain (~0.5-1% R² improvement)

### Objective

Automate hyperparameter tuning for stacking ensemble base models using Optuna, selecting top-performing configurations
within a time budget.

### Current Implementation

- Manual hyperparameter selection for base models
- Default parameters used for XGBoost, LightGBM, CatBoost
- No systematic hyperparameter search

### Target Implementation

**File**: `finance_ml/ml_workflow/regression/models.py`

### TDD Test Specifications

#### Test 1: `test_stacking_hyperparameter_search`

**Purpose**: Optuna tunes base model params within time budget

```python
def test_stacking_hyperparameter_search(self):
    """Hyperparameter search should complete within time budget."""
    # Given: Training data for regression
    X = pd.DataFrame({
        'feature_1': np.random.randn(200),
        'feature_2': np.random.randn(200),
        'feature_3': np.random.randn(200)
        })
    y = X['feature_1'] * 2 + X['feature_2'] - 0.5 * X['feature_3'] + np.random.randn(200) * 0.1

    # When: Tune hyperparameters with time budget
    start_time = time.time()
    best_params, best_score = tune_stacking_hyperparameters(
            X, y,
            model_type='xgboost',
            n_trials=10,
            timeout=30  # 30 seconds max
            )
    elapsed = time.time() - start_time

    # Then: Completes within timeout
    self.assertLessEqual(elapsed, 35)  # 5 second buffer
    self.assertIsInstance(best_params, dict)
    self.assertIn('learning_rate', best_params)
    self.assertIsInstance(best_score, float)
```

#### Test 2: `test_stacking_base_model_selection`

**Purpose**: Top-3 base models selected from comparison results

```python
def test_stacking_base_model_selection(self):
    """Should select best base models for stacking."""
    # Given: Comparison results from multiple models
    comparison_results = {
        'xgboost': {'mae': 10.5, 'rmse': 15.2, 'r2': 0.85},
        'lightgbm': {'mae': 10.2, 'rmse': 14.8, 'r2': 0.87},
        'catboost': {'mae': 10.8, 'rmse': 15.5, 'r2': 0.83},
        'ridge': {'mae': 12.0, 'rmse': 17.0, 'r2': 0.75},
        'lasso': {'mae': 12.5, 'rmse': 17.5, 'r2': 0.73}
        }

    # When: Select top base models
    selected = select_stacking_base_models(
            comparison_results,
            metric='r2',
            top_k=3
            )

    # Then: Top 3 by R² selected
    self.assertEqual(len(selected), 3)
    self.assertIn('lightgbm', selected)
    self.assertIn('xgboost', selected)
    self.assertIn('catboost', selected)
    self.assertNotIn('ridge', selected)
```

#### Test 3: `test_stacking_meta_learner_selection`

**Purpose**: Meta-learner chosen based on CV performance

```python
def test_stacking_meta_learner_selection(self):
    """Meta-learner should be selected via cross-validation."""
    # Given: Base model predictions
    X_base = pd.DataFrame({
        'pred_xgb': np.random.uniform(50, 150, 100),
        'pred_lgb': np.random.uniform(50, 150, 100),
        'pred_cat': np.random.uniform(50, 150, 100)
        })
    y = np.random.uniform(50, 150, 100)

    # When: Select best meta-learner
    best_meta, cv_scores = select_meta_learner(
            X_base, y,
            candidates=['ridge', 'lasso', 'huber'],
            cv=5
            )

    # Then: Best meta-learner selected
    self.assertIn(best_meta, ['ridge', 'lasso', 'huber'])
    self.assertIsInstance(cv_scores, dict)
    self.assertEqual(len(cv_scores), 3)

    # Best meta has highest score
    best_score = cv_scores[best_meta]
    for score in cv_scores.values():
        self.assertGreaterEqual(best_score, score)
```

### Implementation Requirements

#### Function Signature

```python
def tune_stacking_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'xgboost',
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    cv: int = 3,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Dict[str, Any], float]:
    """
    Tune hyperparameters for stacking base models using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Training features
    y : pd.Series
        Training target
    model_type : str, default='xgboost'
        'xgboost', 'lightgbm', 'catboost'
    n_trials : int, default=50
        Number of Optuna trials
    timeout : int, optional, default=300
        Time budget in seconds (None for unlimited)
    cv : int, default=3
        Cross-validation folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=False
        Print trial progress
        
    Returns
    -------
    tuple
        (best_params, best_score)
    """
```

#### Implementation Logic

```python
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

def tune_stacking_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'xgboost',
    n_trials: int = 50,
    timeout: Optional[int] = 300,
    cv: int = 3,
    random_state: int = 42,
    verbose: bool = False
) -> Tuple[Dict[str, Any], float]:
    """Hyperparameter tuning per code_guidelines.md Section 16."""
    
    def objective(trial):
        """Optuna objective function."""
        if model_type == 'xgboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': 100,
                'random_state': random_state,
                'verbosity': 0
            }
            model = xgb.XGBRegressor(**params)
        
        elif model_type == 'lightgbm':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'n_estimators': 100,
                'random_state': random_state,
                'verbose': -1
            }
            model = lgb.LGBMRegressor(**params)
        
        elif model_type == 'catboost':
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'iterations': 100,
                'random_state': random_state,
                'verbose': False
            }
            model = CatBoostRegressor(**params)
        
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        
        # Cross-validate (negative MAE)
        scorer = make_scorer(mean_absolute_error, greater_is_better=False)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        return -scores.mean()  # Optuna minimizes
    
    # Run optimization
    sampler = TPESampler(seed=random_state)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=verbose
    )
    
    logger.info(
        f"Optuna optimization complete: {len(study.trials)} trials, "
        f"best MAE: {study.best_value:.4f}"
    )
    
    return study.best_params, study.best_value
```

#### Base Model Selection

```python
def select_stacking_base_models(
        comparison_results: Dict[str, Dict[str, float]],
        metric: str = 'r2',
        top_k: int = 3
        ) -> List[str]:
    """
    Select top base models for stacking ensemble.
    
    Parameters
    ----------
    comparison_results : dict
        Results from compare_regressors()
    metric : str, default='r2'
        Metric to rank by ('mae', 'rmse', 'r2')
    top_k : int, default=3
        Number of base models to select
        
    Returns
    -------
    list
        Names of selected models
    """
    # Sort by metric (descending for r2, ascending for mae/rmse)
    ascending = metric in ['mae', 'rmse', 'mape']

    sorted_models = sorted(
            comparison_results.items(),
            key=lambda x: x[1][metric],
            reverse=not ascending
            )

    selected = [model_name for model_name, _ in sorted_models[:top_k]]

    logger.info(f"Selected {top_k} base models by {metric}: {selected}")
    return selected
```

#### Meta-Learner Selection

```python
def select_meta_learner(
        X_base: pd.DataFrame,
        y: pd.Series,
        candidates: List[str] = ['ridge', 'lasso', 'huber'],
        cv: int = 5,
        random_state: int = 42
        ) -> Tuple[str, Dict[str, float]]:
    """
    Select best meta-learner via cross-validation.
    
    Parameters
    ----------
    X_base : pd.DataFrame
        Base model predictions (meta-features)
    y : pd.Series
        Target
    candidates : list, default=['ridge', 'lasso', 'huber']
        Meta-learner candidates
    cv : int, default=5
        Cross-validation folds
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    tuple
        (best_meta_learner_name, cv_scores_dict)
    """
    from sklearn.linear_model import Ridge, Lasso, HuberRegressor
    from sklearn.model_selection import cross_val_score

    meta_learners = {
        'ridge': Ridge(random_state=random_state),
        'lasso': Lasso(random_state=random_state),
        'huber': HuberRegressor()
        }

    cv_scores = {}
    for name in candidates:
        if name not in meta_learners:
            logger.warning(f"Unknown meta-learner: {name}. Skipping.")
            continue

        model = meta_learners[name]
        scores = cross_val_score(
                model, X_base, y,
                cv=cv,
                scoring='r2',
                n_jobs=-1
                )
        cv_scores[name] = scores.mean()
        logger.info(f"Meta-learner {name}: R² = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_meta = max(cv_scores, key=cv_scores.get)
    logger.info(f"Selected meta-learner: {best_meta} (R² = {cv_scores[best_meta]:.4f})")

    return best_meta, cv_scores
```

### Acceptance Criteria

- [ ] All 3 tests pass in `test_stacking_hyperparameter.py`
- [ ] `tune_stacking_hyperparameters()` function implemented with Optuna
- [ ] `select_stacking_base_models()` selects top-k models by metric
- [ ] `select_meta_learner()` cross-validates meta-learner candidates
- [ ] Timeout protection prevents indefinite runs
- [ ] Documentation updated in code_guidelines.md Section 16

---

## Success Metrics

### Quantitative Targets

- **Feature Alignment**: Zero prediction-time errors from feature mismatch
- **Code Reduction**: Notebook code reduced by ~50 lines in Section 6.4
- **Hyperparameter Tuning**: 0.5-1% R² improvement over default parameters
- **Test Coverage**: 100% pass rate on 6 new tests
- **Performance**: Tuning completes within 5 minutes per base model

### Qualitative Targets

- Simplified prediction workflow through automatic alignment
- Systematic hyperparameter optimization for reproducibility
- Reduced manual intervention in notebook cells

---

## Dependencies and Risks

### Dependencies

- Task 7: sklearn models with `feature_names_in_` attribute
- Task 6: Optuna library (add to requirements.txt)
- Existing regression module structure

### Risks and Mitigation

1. **Risk**: Optuna hyperparameter search may overfit to validation set
    - **Mitigation**: Use nested CV, monitor out-of-sample performance, limit search space

2. **Risk**: Feature alignment may mask data quality issues
    - **Mitigation**: Log warnings for missing/extra features, optional strict mode that raises errors

3. **Risk**: Hyperparameter tuning may be too slow for large datasets
    - **Mitigation**: Timeout protection, reduced n_trials, parallel execution

---

## Implementation Priority

### Rationale

**Task 7 (High Priority)** should be implemented first:

- Immediate operational value (eliminates prediction errors)
- Low complexity (simple function)
- High impact on notebook usability

**Task 6 (Low Priority)** can be deferred:

- Marginal accuracy improvement (~0.5-1%)
- High complexity (Optuna integration)
- Manual hyperparameter tuning works adequately

### Recommended Sequence

1. Implement Task 7 in Sprint 3 Week 1
2. Validate Task 7 with existing models
3. Implement Task 6 in Sprint 3 Week 2 if time permits
4. Task 6 can be moved to Sprint 4 if needed

---

## Next Steps

### Immediate Actions (Task 7 - High Priority)

1. Create `tests/test_feature_alignment.py` with 3 test cases
2. Implement `align_features_to_model()` in `regression/dataset.py`
3. Implement `predict_with_model()` wrapper
4. Update notebook Section 6.4 to use new functions
5. Validate against existing trained models

### Deferred Actions (Task 6 - Low Priority)

1. Create `tests/test_stacking_hyperparameter.py` with 3 test cases (if time permits)
2. Add Optuna to requirements.txt
3. Implement `tune_stacking_hyperparameters()` in `regression/models.py`
4. Implement `select_stacking_base_models()` and `select_meta_learner()`
5. Optional: Add notebook cell for automated tuning

### Post-Sprint 3

After Phase 9.5 completion:

- Review all three implementation plans (9.3, 9.4, 9.5)
- Update code_guidelines.md with new functions and policies
- Run full test suite (85 existing + 22 new = 107 tests)
- Update IMPROVEMENT_PLAN.md with completed phases

---

## Document Control

**Reviewed By**: TBD  
**Approved By**: TBD  
**Last Modified**: 2025-12-09  
**Related Documents**:

- `code_guidelines.md` v1.10
- `phase_9.3_implementation_plan.md` v1.0
- `phase_9.4_implementation_plan.md` v1.0
- Current State Analysis (2025-12-09)
