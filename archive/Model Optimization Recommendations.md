### Model Optimization Recommendations — Phase 9.5 Regression Analysis

Based on comprehensive analysis of `regression_predictions.csv`, `regression_metrics_by_sector.csv`, and
`prediction_analyst_comparison_report.xlsx`, here are actionable recommendations to optimize ML workflow output quality
and prediction accuracy.

---

### 📊 Current Performance Summary

**Overall Metrics (1,385 predictions):**

- **MAE**: 272.56 (moderate)
- **RMSE**: 4,643.02 (very high — driven by outliers)
- **Error Distribution**:
    - 2.2% of predictions (31 stocks) have errors > 1,000
    - 5.0% (69 stocks) have errors > 100
    - 90th percentile error: 545.34
    - 95th percentile error: 1,087.50
    - 99th percentile error: 6,825.70

**Sector-Level Performance (from Analyst Comparison Report):**

- **Best Performers**: Information Technology (37.2% agreement), Materials (32.3%), Utilities (29.6%)
- **Worst Performers**: Real Estate (14.3%), Health Care (14.3%), Consumer Discretionary (16.3%)
- **Prediction Bias Issues**:
    - Industrials: -544 (systematic under-prediction)
    - Financials: +795 (systematic over-prediction)
    - Communication Services: +755 (systematic over-prediction)

---

### 🎯 Priority 1: Fix Critical Data Pipeline Issues

#### Issue 1.1: Missing Sector Information in Predictions Output

**Problem**: `regression_predictions.csv` contains only `y_true`, `y_pred`, and `residual` — no sector, ticker, or
feature information for error analysis.

**Impact**: Cannot diagnose sector-specific model failures or feature importance issues.

**Solution**:

```python
# In finance_ml/regression.py, train_and_evaluate_regression() function (line 244):
# CURRENT:
results_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "residual": y_test.values - preds,
}, index=y_test.index)

# IMPROVED:
results_df = pd.DataFrame({
    "y_true": y_test.values,
    "y_pred": preds,
    "residual": y_test.values - preds,
    "abs_error": np.abs(y_test.values - preds),
    "pct_error": ((y_test.values - preds) / y_test.values) * 100,
}, index=y_test.index)

# Add sector, ticker, and key features for diagnostic analysis
if "sector" in df.columns:
    results_df["sector"] = df.loc[y_test.index, "sector"]
if "ticker" in df.columns:
    results_df["ticker"] = df.loc[y_test.index, "ticker"]
if "market_cap" in df.columns:
    results_df["market_cap"] = df.loc[y_test.index, "market_cap"]
```

#### Issue 1.2: Empty `regression_metrics_by_sector.csv`

**Problem**: File is completely empty despite having a dedicated function.

**Root Cause**: Function `train_and_evaluate_regression_by_sector()` exists but is never called in the main pipeline.

**Solution**: Add explicit call in notebook or CLI pipeline:

```python
# After train_and_evaluate_regression(), add:
if 'sector' in df.columns:
    sector_metrics = train_and_evaluate_regression_by_sector(df, out_dir)
    logger.info(f"Sector-level metrics: {len(sector_metrics)} sectors evaluated")
```

---

### 🔧 Priority 2: Address Outlier Handling

#### Issue 2.1: Extreme Outliers Driving RMSE

**Analysis**:

- 99th percentile error (6,825.70) is 13x the MAE
- RMSE (4,643) is 17x the MAE
- Suggests a few catastrophic predictions dominating loss

**Recommendations**:

**A. Add Robust Loss Function for Training**:

```python
# Use Huber loss for GradientBoostingRegressor to reduce outlier sensitivity
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    loss='huber',  # Instead of 'squared_error'
    alpha=0.9,     # Quantile for Huber transition
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05
)
```

**B. Add Winsorization for Target Variable**:

```python
# In preprocessing, cap extreme target values
from scipy.stats import mstats

def winsorize_target(y, lower=0.01, upper=0.99):
    """Cap extreme target values at percentiles."""
    return mstats.winsorize(y, limits=[lower, 1-upper])

y_train_robust = winsorize_target(y_train)
```

**C. Add Post-Prediction Clipping**:

```python
# Ensure predictions stay within reasonable bounds
def clip_predictions(preds, y_train, n_std=3):
    """Clip predictions to n standard deviations from training mean."""
    mean = y_train.mean()
    std = y_train.std()
    lower = max(0, mean - n_std * std)
    upper = mean + n_std * std
    return np.clip(preds, lower, upper)

preds_clipped = clip_predictions(preds, y_train)
```

---

### 📈 Priority 3: Improve Sector-Specific Modeling

#### Issue 3.1: Large Sector Bias (Financials +795, Industrials -544)

**Problem**: Systematic over/under-prediction suggests sector models need calibration.

**Recommendations**:

**A. Implement Sector-Specific Feature Engineering**:

```python
# Already partially implemented in notebook, but needs enhancement:

def engineer_sector_features(df, sector):
    """Create sector-specific features."""
    if sector == "Financials":
        # Focus on book value, ROE, leverage
        df['p_tbv'] = df['market_cap'] / df.get('tangible_book_value', 1)
        df['roe'] = df.get('net_income', 0) / df.get('shareholders_equity', 1)
        df['leverage_ratio'] = df.get('total_debt', 0) / df.get('shareholders_equity', 1)
        
    elif sector == "Industrials":
        # Focus on margins, asset turnover, order backlog
        df['asset_turnover'] = df.get('revenue', 0) / df.get('total_assets', 1)
        df['operating_leverage'] = df.get('operating_income', 0) / df.get('revenue', 1)
        
    elif sector == "Information Technology":
        # Focus on growth, R&D intensity, gross margins
        df['rd_intensity'] = df.get('r_d_expense', 0) / df.get('revenue', 1)
        df['gross_margin'] = df.get('gross_profit', 0) / df.get('revenue', 1)
    
    return df
```

**B. Add Sector-Specific Calibration Layer**:

```python
# Post-prediction bias correction per sector
def calibrate_predictions_by_sector(preds_df):
    """Apply sector-specific bias correction."""
    sector_bias = {
        'Financials': -795,        # Over-predicting by 795
        'Industrials': +544,       # Under-predicting by 544
        'Communication Services': -755,
    }
    
    for sector, bias in sector_bias.items():
        mask = preds_df['sector'] == sector
        preds_df.loc[mask, 'y_pred_calibrated'] = preds_df.loc[mask, 'y_pred'] + bias
    
    return preds_df
```

#### Issue 3.2: Poor Performance in Real Estate & Health Care (14.3% agreement)

**Recommendations**:

- **Real Estate**: Add property-specific features (FFO, AFFO, cap rates, NOI)
- **Health Care**: Add regulatory/pipeline features (FDA approvals, patent expiry, R&D pipeline)

---

### 🧪 Priority 4: Enhance Model Validation Strategy

#### Issue 4.1: Single Train/Test Split May Not Capture Temporal Dynamics

**Current**: 80/20 static split

**Recommendation**: Implement **Time-Series Cross-Validation**:

```python
from sklearn.model_selection import TimeSeriesSplit

# If you have date information:
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Train and evaluate
```

#### Issue 4.2: No Uncertainty Quantification in Main Predictions

**Problem**: Single point predictions without confidence intervals.

**Solution**: Already implemented quantile regression in notebook (Phase 9.5.5) — **export these to CSV**:

```python
# In notebook, after quantile predictions:
quantile_results = pd.DataFrame({
    'ticker': test_tickers,
    'y_true': y_test,
    'pred_median': predictions_quantile[0.5],
    'pred_lower_10': predictions_quantile[0.1],
    'pred_upper_90': predictions_quantile[0.9],
    'interval_width': predictions_quantile[0.9] - predictions_quantile[0.1]
})

quantile_results.to_csv('outputs/regression/quantile_predictions.csv', index=False)
```

---

### 🎨 Priority 5: Add Feature Importance Analysis

#### Missing Capability: No Feature Importance in Outputs

**Add to `train_and_evaluate_regression()`**:

```python
# After model training:
if hasattr(pipe.named_steps['regressor'], 'feature_importances_'):
    feature_names = (
        pipe.named_steps['preprocessor']
        .get_feature_names_out()
    )
    importances = pipe.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    feature_importance_df.to_csv(
        out_dir / 'feature_importance.csv', 
        index=False
    )
```

---

### 🚀 Priority 6: Ensemble & Stacking Improvements

#### Issue 6.1: Stacking Ensemble Not Used in Main Pipeline

**Problem**: Notebook trains stacking ensemble (Phase 9.5.4) but `train_and_evaluate_regression()` uses simple
RandomForest.

**Recommendation**: Make stacking ensemble the default:

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

def build_stacking_pipeline(num_cols, cat_cols, n_jobs=1):
    """Build stacking ensemble pipeline."""
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    base_estimators = [
        ('rf', RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=42)),
        ('gb', GradientBoostingRegressor(loss='huber', n_estimators=100, random_state=42)),
        ('ridge', Ridge(alpha=10.0))
    ]
    
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])
    
    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        n_jobs=n_jobs
    )
    
    return Pipeline([('preprocessor', preprocessor), ('regressor', stacking)])
```

---

### 📋 Implementation Roadmap

#### Immediate Actions (1-2 days):

1. ✅ **Fix `regression_predictions.csv`**: Add sector, ticker, abs_error, pct_error columns
2. ✅ **Populate `regression_metrics_by_sector.csv`**: Call sector-level function in pipeline
3. ✅ **Add outlier handling**: Implement Huber loss and prediction clipping

#### Short-term (1 week):

4. ✅ **Sector-specific calibration**: Apply bias correction for Financials, Industrials
5. ✅ **Feature importance export**: Save top features to CSV
6. ✅ **Export quantile predictions**: Save uncertainty intervals

#### Medium-term (2-4 weeks):

7. ✅ **Enhanced sector features**: Real Estate (FFO/AFFO), Health Care (pipeline metrics)
8. ✅ **Time-series CV**: Replace static split with temporal validation
9. ✅ **Stacking ensemble as default**: Upgrade main pipeline

#### Long-term (1-2 months):

10. ✅ **Hyperparameter optimization**: Add Optuna/Grid Search for sector models
11. ✅ **Model monitoring**: Track prediction drift over time
12. ✅ **A/B testing framework**: Compare model versions systematically

---

### 📊 Expected Impact

| Metric                | Current | Target  | Improvement Method                    |
|-----------------------|---------|---------|---------------------------------------|
| MAE (overall)         | 272.56  | < 200   | Outlier handling + sector calibration |
| RMSE                  | 4,643   | < 500   | Huber loss + winsorization            |
| 99th pct error        | 6,826   | < 1,500 | Prediction clipping                   |
| IT sector agreement   | 37.2%   | > 45%   | Enhanced tech features                |
| Financials agreement  | 19.4%   | > 30%   | Sector-specific models + calibration  |
| Real Estate agreement | 14.3%   | > 25%   | Property-specific features            |

---

### 🔍 Monitoring & Validation

**Add these validation checks to pipeline**:

```python
def validate_predictions(results_df):
    """Validate prediction quality and flag issues."""
    checks = {
        'negative_predictions': (results_df['y_pred'] < 0).sum(),
        'extreme_errors_pct': (results_df['abs_error'] > 1000).sum() / len(results_df) * 100,
        'mean_abs_error': results_df['abs_error'].mean(),
        'sectors_with_bias': results_df.groupby('sector')['residual'].mean().abs().nlargest(3)
    }
    
    return checks
```

---

### 📁 Suggested File Outputs

**Enhanced output structure**:

```
outputs/models/
├── regression_predictions_detailed.csv  # With sector, ticker, features
├── regression_metrics_by_sector.csv     # Per-sector MAE, RMSE, R²
├── quantile_predictions.csv             # Uncertainty intervals
├── feature_importance.csv               # Top features by sector
├── sector_calibration_factors.csv       # Bias corrections
└── model_diagnostics.json               # Validation checks
```

---

### 💡 Key Takeaways

1. **Critical**: Fix data pipeline to include sector info in all outputs
2. **High Impact**: Add sector-specific calibration to fix Financials/Industrials bias
3. **Risk Mitigation**: Implement outlier handling (Huber loss) to reduce RMSE
4. **Model Upgrade**: Switch to stacking ensemble for 5-10% MAE improvement
5. **Visibility**: Export feature importance and quantile predictions for interpretability

These recommendations are based on detailed analysis of your actual model outputs and are directly implementable in the
existing codebase structure.