# Finance ML Workflow TDD Implementation Plan

**Document Version:** 1.0  
**Date:** 2025-12-10  
**Status:** READY FOR IMPLEMENTATION  
**Alignment:** `code_guidelines.md` v1.10, `PHASE_93_94_NOTEBOOK_INTEGRATION_GUIDE.md`

---

## Executive Summary

This document provides minimal, actionable TDD implementation tasks to fix critical issues in Phases 9.4-8 (
Classification, Regression, Evaluation, Price Target Predictions) that prevent accurate price_target predictions.

### Critical Issues Identified

**Issue 1: Market Cap Feature Leakage (ROOT CAUSE)**

- **Impact**: Predictions on market_cap scale (~880K) instead of price scale (~736K for BRKA)
- **Location**: `finance_ml/ml_workflow/regression/dataset.py` line 429
- **Root Cause**: `market_cap` included in base regression features + sector interactions
- **Severity**: CRITICAL - Makes all predictions unusable for investment decisions

**Issue 2: Classification Model Collapse**

- **Impact**: F1-macro=0.143, classes 1-3 never predicted (0% precision/recall)
- **Root Cause**: Missing `balance_classes()` and `determine_cv_strategy()` integration
- **Severity**: HIGH - Classification meta-features provide no value to regression

**Issue 3: Sector Calibration Degradation**

- **Impact**: MAE worsens by -93% (Financials) to -431% (Communication Services)
- **Root Cause**: Calibration applied to biased baseline from market_cap leakage
- **Severity**: HIGH - Makes calibrated predictions worse than raw predictions

### Expected Outcomes After Implementation

| Metric                      | Current                    | Target       | Improvement           |
|-----------------------------|----------------------------|--------------|-----------------------|
| Regression MAE (Financials) | 913                        | <300         | 67% reduction         |
| Classification F1-macro     | 0.143                      | >0.30        | 110% increase         |
| Calibration success rate    | 45% sectors                | >80% sectors | 78% increase          |
| Prediction scale accuracy   | Off by orders of magnitude | Within ±20%  | Fundamental fix       |
| Class recall (classes 1-3)  | 0%                         | >15%         | Classes become active |

---

## Implementation Tasks

### Priority 1: Fix Market Cap Feature Leakage (CRITICAL - 1 day)

#### Task 1.1: Remove market_cap from Regression Features

**File to Modify:** `finance_ml/ml_workflow/regression/dataset.py`

**Code Change (Line 425-431):**

```python
# BEFORE:
base_cols = [
    "p_e_ratio",
    "ev_ebitda_ratio", 
    "gross_margin",
    "market_cap",  # ❌ REMOVE - causes feature leakage
    "beta_5y",
]

# AFTER:
base_cols = [
    "p_e_ratio",
    "ev_ebitda_ratio",
    "gross_margin",
    "beta_5y",
    "debt_to_equity",  # Add fundamental risk metric instead
]
```

**Test Specification:** `tests/test_regression_no_leakage.py`

```python
def test_market_cap_not_in_features():
    """Ensure market_cap excluded from regression features (data leakage prevention)."""
    df = create_sample_stocks_dataframe()
    X_train, X_test, y_train, y_test, meta = prepare_regression_data(
        df, target_col='price_target'
    )
    
    # Market cap should NOT appear in features
    assert 'market_cap' not in X_train.columns, "market_cap leaks target information"
    assert 'log_market_cap' not in X_train.columns, "log_market_cap leaks target"
    
    # No sector interactions with market_cap
    market_cap_interactions = [
        col for col in X_train.columns if 'market_cap' in col.lower()
    ]
    assert len(market_cap_interactions) == 0, (
        f"Market cap interactions found: {market_cap_interactions}"
    )
    
    # But enterprise_value IS allowed (forward-looking metric)
    assert 'log_enterprise_value' in X_train.columns or 'enterprise_value' in X_train.columns

def test_enterprise_value_allowed_as_feature():
    """Verify enterprise_value can be used (not circular like market_cap)."""
    df = create_sample_stocks_dataframe()
    X_train, _, _, _, _ = prepare_regression_data(df, target_col='price_target')
    
    # Enterprise value allowed because it's forward-looking
    ev_cols = [col for col in X_train.columns if 'enterprise_value' in col.lower()]
    assert len(ev_cols) > 0, "Enterprise value should be available as feature"
```

**Notebook Integration:** Add validation cell after Phase 9.5 Section 6.2 (~line 4920)

```python
#%% Feature Leakage Prevention Check
print("\n🔍 Feature Leakage Prevention Check")
print("=" * 80)

# Verify no market cap leakage
leakage_cols = [
    col for col in X_train.columns 
    if 'market_cap' in col.lower()
]
if leakage_cols:
    raise ValueError(f"⚠️ FEATURE LEAKAGE DETECTED: {leakage_cols}")
else:
    print("✓ No market_cap feature leakage detected")

# Log feature statistics
print(f"\n📊 Feature Statistics:")
print(f"  Total features: {X_train.shape[1]}")
print(f"  Numeric features: {len(meta.get('numeric_features', []))}")
print(f"  Classification meta-features: {len(meta.get('classification_features', []))}")
```

**Expected Impact:**

- Predictions shift from market_cap scale to true price scale
- MAE improves by 30-50% once model learns valuation patterns
- R² may initially drop but predictions become more generalizable

---

### Priority 2: Fix Classification Collapse (HIGH - 1 day)

#### Task 2.1: Integrate balance_classes() in Notebook

**File to Modify:** `ml_finance_model_main.ipynb`  
**Location:** Insert BEFORE Phase 9.4 classification training (~line 4400)

**New Cell: Balance Classes**

```python
#%% Phase 9.4: Class Balance Analysis and Adjustment
print("\n🔧 Phase 9.4: Class Balance Analysis and Adjustment")
print("=" * 80)

from finance_ml.ml_workflow.classification.training import balance_classes

# Analyze class distribution
y_train_dist = y_train_clf.value_counts().sort_index()
print(f"\n📊 Original Class Distribution:")
for cls, count in y_train_dist.items():
    pct = count / len(y_train_clf) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

# Apply balancing (SMOTE for minority, undersample majority)
X_train_balanced, y_train_balanced = balance_classes(
    X_train_clf, 
    y_train_clf,
    method='auto',  # Auto-selects SMOTE or undersampling
    min_samples=MIN_SECTOR_SAMPLES
)

y_balanced_dist = y_train_balanced.value_counts().sort_index()
print(f"\n✓ Balanced Class Distribution:")
for cls, count in y_balanced_dist.items():
    pct = count / len(y_train_balanced) * 100
    print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")

print(f"\n  Resampling: {len(X_train_clf):,} → {len(X_train_balanced):,} samples")
```

**Test Specification:** `tests/test_classification_balance.py`

```python
def test_balance_classes_improves_minority():
    """Verify balance_classes increases minority class representation."""
    from finance_ml.ml_workflow.classification.training import balance_classes
    
    X, y = create_imbalanced_classification_data()  # 5 classes, 50-500 samples
    X_bal, y_bal = balance_classes(X, y, method='auto')
    
    # All classes should have reasonable representation (≥20% of majority)
    class_counts = pd.Series(y_bal).value_counts()
    max_count = class_counts.max()
    min_count = class_counts.min()
    
    assert min_count >= 0.2 * max_count, (
        f"Class imbalance persists: {class_counts.to_dict()}"
    )
    assert len(set(y_bal)) == 5, "Should preserve all 5 classes"

def test_all_classes_predicted_after_balance():
    """Verify trained model predicts all 5 classes after balancing."""
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = create_imbalanced_classification_data()
    X_bal, y_bal = balance_classes(X, y, method='auto')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.2, random_state=42
    )
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    predicted_classes = set(y_pred)
    assert len(predicted_classes) >= 4, (
        f"Model should predict at least 4 of 5 classes, got {len(predicted_classes)}"
    )
```

#### Task 2.2: Integrate determine_cv_strategy() in Notebook

**New Cell: CV Strategy Selection**

```python
#%% Phase 9.4: Cross-Validation Strategy Selection
print("\n🔧 Phase 9.4: Cross-Validation Strategy Selection")
print("=" * 80)

from finance_ml.ml_workflow.classification.training import determine_cv_strategy

cv_strategy, cv_params = determine_cv_strategy(
    all_stocks_features,
    y_train_clf,
    date_col='snapshot_date',
    group_col='ticker',
    stratify_col='sector'
)

print(f"\n✓ Selected CV Strategy: {cv_strategy}")
print(f"  Parameters: {cv_params}")

# Use in cross-validation
if cv_strategy == 'TimeSeriesSplit':
    from sklearn.model_selection import TimeSeriesSplit
    cv = TimeSeriesSplit(**cv_params)
elif cv_strategy == 'GroupKFold':
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(**cv_params)
else:
    from sklearn.model_selection import StratifiedKFold
    cv = StratifiedKFold(**cv_params)
```

**Expected Impact:**

- Precision/recall for classes 1-3 improves from 0% to 15-30%
- F1-macro improves from 0.143 to 0.30-0.45
- Overall accuracy improves from 24% to 40-50%

---

### Priority 3: Fix Sector Calibration Logic (HIGH - 0.5 days)

#### Task 3.1: Add Calibration Pre-Check

**File to Modify:** `finance_ml/ml_workflow/evaluation/calibration.py`  
**Location:** Start of `apply_sector_calibration()` function

**Code Change:**

```python
def apply_sector_calibration(predictions_df, calibration_dict, model_version):
    """Apply sector-specific bias correction with validation."""
    
    # NEW: Pre-check - only apply if improves ≥50% of sectors
    if calibration_dict and 'sectors' in calibration_dict:
        improved_sectors = sum(
            1 for s, metrics in calibration_dict['sectors'].items()
            if metrics.get('mae_improvement_pct', 0) > 0
        )
        total_sectors = len(calibration_dict['sectors'])
        
        if improved_sectors < total_sectors * 0.5:
            logging.warning(
                f"⚠️ Calibration improves only {improved_sectors}/{total_sectors} "
                f"sectors. Skipping (likely underlying model issue)."
            )
            predictions_df['y_pred_calibrated'] = predictions_df['y_pred'].copy()
            return predictions_df
    
    # Existing calibration logic...
```

**Test Specification:** `tests/test_calibration_validation.py`

```python
def test_calibration_skipped_if_degrading_majority():
    """Calibration should be skipped if it worsens >50% of sectors."""
    preds = create_sample_predictions_by_sector()
    
    # Bad calibration (worsens 3 of 5 sectors)
    bad_calibration = {
        'sectors': {
            'Financials': {'bias_raw': 100, 'mae_improvement_pct': -50},
            'Technology': {'bias_raw': 50, 'mae_improvement_pct': 20},
            'Energy': {'bias_raw': 80, 'mae_improvement_pct': -30},
            'Healthcare': {'bias_raw': 60, 'mae_improvement_pct': -20},
            'Materials': {'bias_raw': 40, 'mae_improvement_pct': 10},
        }
    }
    
    result = apply_sector_calibration(preds, bad_calibration, 'v9_9')
    
    # Should NOT apply calibration
    assert (result['y_pred_calibrated'] == result['y_pred']).all()

def test_calibration_applied_if_improving_majority():
    """Calibration should be applied if it improves ≥50% of sectors."""
    preds = create_sample_predictions_by_sector()
    
    # Good calibration (improves 4 of 5 sectors)
    good_calibration = {
        'sectors': {
            'Financials': {'bias_raw': 100, 'mae_improvement_pct': 30},
            'Technology': {'bias_raw': 50, 'mae_improvement_pct': 20},
            'Energy': {'bias_raw': 80, 'mae_improvement_pct': 40},
            'Healthcare': {'bias_raw': 60, 'mae_improvement_pct': 25},
            'Materials': {'bias_raw': 40, 'mae_improvement_pct': -10},
        }
    }
    
    result = apply_sector_calibration(preds, good_calibration, 'v9_9')
    
    # Should apply calibration
    assert not (result['y_pred_calibrated'] == result['y_pred']).all()
```

**Notebook Integration:** Add after calibration application (~line 5650)

```python
#%% Calibration Quality Check
print("\n🔍 Calibration Quality Check")
print("=" * 80)

with open(OUTPUT_DIR / 'calibration' / f'sector_bias_calibration_{MODEL_VERSION}.json') as f:
    calib_data = json.load(f)

improved = [
    s for s, m in calib_data['sectors'].items()
    if m['mae_improvement_pct'] > 0
]
degraded = [
    s for s, m in calib_data['sectors'].items()
    if m['mae_improvement_pct'] < 0
]

print(f"✓ Improved: {len(improved)} sectors")
print(f"⚠️ Degraded: {len(degraded)} sectors")

if len(degraded) > len(improved):
    print("\n⚠️ WARNING: Calibration degraded majority of sectors!")
    print("   Root cause: Underlying model has systematic bias (check feature leakage)")
```

**Expected Impact:**

- After fixing market_cap leakage, calibration improves 80%+ of sectors
- MAE improvement positive for all major sectors

---

## Implementation Timeline

### Phase 1: Critical Fixes (2 days)

1. **Day 1**: Task 1.1 (Remove market_cap feature leakage + tests)
2. **Day 2**: Task 2.1-2.2 (Integrate balance_classes + determine_cv_strategy + tests)
3. **Day 2**: Task 3.1 (Add calibration pre-check + tests)

### Phase 2: Validation & Refinement (1 day)

4. **Day 3**: Re-run full notebook pipeline
5. **Day 3**: Validate metrics meet targets (MAE <300, F1-macro >0.30, calibration >80%)
6. **Day 3**: Update documentation and create regression test suite

---

## Testing Strategy

### Unit Tests (New Files)

- `tests/test_regression_no_leakage.py` (3 tests)
- `tests/test_classification_balance.py` (2 tests)
- `tests/test_calibration_validation.py` (2 tests)

### Integration Tests (Notebook Execution)

```bash
# Run notebook cells 1-50 (ETL + EDA + Features + Classification + Regression)
jupyter nbconvert --execute --to notebook ml_finance_model_main.ipynb \
    --ExecutePreprocessor.timeout=600

# Validate outputs
python -c "
import pandas as pd
df = pd.read_csv('outputs/regression/regression_predictions_detailed.csv')
print(f'MAE: {df.abs_error.mean():.2f}')
assert df.abs_error.mean() < 400, 'MAE too high'
assert df.y_pred.min() >= 0, 'Negative predictions found'
"
```

### Validation Checkpoints

After each priority:

1. ✅ All new unit tests pass
2. ✅ No regressions in existing test suite
3. ✅ Notebook cells execute without errors
4. ✅ Output files validated against schema
5. ✅ Metrics meet target thresholds

---

## Success Metrics

### Critical Success Criteria

- ✅ MAE (Financials) < 300 (currently 913)
- ✅ Classification F1-macro > 0.30 (currently 0.143)
- ✅ Calibration improves >80% sectors (currently 45%)
- ✅ All 5 classes have recall >10% (currently 3 classes at 0%)
- ✅ Zero negative predictions (non-negativity enforced)

### Performance Targets

- Training time: No increase >20%
- Memory usage: No increase >30%
- Prediction latency: <100ms per stock

---

## Alignment with Code Guidelines

| Issue                  | code_guidelines.md Section             | Status After Fix     |
|------------------------|----------------------------------------|----------------------|
| Market cap leakage     | Section 5.4 (Feature Categories)       | ✅ Compliant          |
| Classification balance | Section 2.2.1 (5-class system)         | ✅ All classes active |
| Calibration validation | Section 7 (Sector Metrics)             | ✅ Improves 80%+      |
| Feature selection      | Section 9.3 (Automated Selection)      | ✅ Integrated         |
| CV strategy            | Section 6 (Data Split Policy)          | ✅ Fully applied      |
| Prediction intervals   | Section 4 (Uncertainty Quantification) | ✅ Tighter bounds     |

---

## References

- **Code Guidelines**: `docs/code_guidelines.md` v1.10
- **Phase 9.3/9.4 Integration**: `PHASE_93_94_NOTEBOOK_INTEGRATION_GUIDE.md`
- **Previous Analysis**: See session history for detailed issue breakdown
- **Test Suite**: `tests/` directory (1815 existing tests, +7 new tests)

---

**Document Status**: READY FOR IMPLEMENTATION  
**Next Action**: Begin Priority 1 - Task 1.1 (Remove market_cap feature leakage)
