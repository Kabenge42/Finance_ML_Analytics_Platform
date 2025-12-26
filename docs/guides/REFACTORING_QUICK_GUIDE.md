# Quick Refactoring Guide for ml_finance_model_main.ipynb

## Priority Summary

### 🔴 CRITICAL (Do First)

1. **52 Missing Imports** - Add these to import cells:
    - Feature engineering: `engineer_technical_analysis_features`, etc. → from `features.advanced`
    - Regression: `regression_train_stacking`, etc. → from `regression.*`
    - Analytics: `analytics_calculate_mispricing`, etc. → from `analytics.*`
    - Reporting: `ExcelReportConfig`, etc. → from `reporting.*`

2. **4 Missing Variables**:
    - `fold_assignments` → Initialize with CV splitter
    - `metrics_history_df` → Initialize as empty DataFrame
    - `top_candidates` → Filter from results DataFrame

### 🟠 HIGH (Do Second)

3. **11 Type Issues**:
    - Line 668: Type `DATA_SOURCE` as `Literal["csv", "db", "all_stocks"]`
    - Line 1888: Convert 0.7 to `timedelta(days=0.7)`
    - Lines 2958, 6699, 6705: Fix `marker_color` from string to dict
    - Line 8234: Convert Series to array with `.values`

4. **2 Incorrect Arguments**:
    - Lines 1453-1454: Remove `output_dir` and `sector_col` arguments

5. **4 Deprecated Imports**:
    - Line 4887: Replace `finance_ml.ml_workflow.models` → `regression.pipeline`
    - Lines 6258, 6782, 7083: Replace `analytics.eval` → specific modules

### 🟡 MEDIUM (Do Third)

6. **35 Unbound Variables**:
    - Initialize before conditional blocks
    - Add default values before use
    - Ensure proper error handling

### 🟢 LOW (Optional)

7. **47 Unused Imports**: Remove to clean up
8. **12 Type Hint Issues**: Fix for strict type checking
9. **17 Name Shadowing**: Rename variables (style issue)
10. **7 Missing Docstrings**: Add for documentation

## Quick Fix Template

### Adding Missing Import

```python
# Find import cell (usually near top)
# Add to existing import block or create new one:
from finance_ml.features.advanced import (
    engineer_technical_analysis_features,
    engineer_valuation_timeseries_features
    )
```

### Fixing Unbound Variable

```python
# BEFORE use, add initialization:
variable_name = default_value  # e.g., pd.DataFrame(), {}, []

# Then use conditionally:
if condition:
    variable_name = actual_value
```

### Fixing Type Issue

```python
# String to Literal
DATA_SOURCE: Literal["csv", "db", "all_stocks"] = "csv"

# Float to timedelta
from datetime import timedelta

param = timedelta(days=0.7)

# String to dict for Plotly
marker = dict(color='green')  # NOT marker_color='green'

# Series to array
returns = mean_returns.values  # NOT returns=mean_returns
```

## Testing Strategy

After each phase:

1. Save notebook
2. Restart kernel
3. Run all cells (or run until first error)
4. Fix any runtime errors
5. Repeat

## Files Created

1. **NOTEBOOK_REFACTORING_DETAILED_FIXES.md** - Complete line-by-line fixes
2. **REFACTORING_QUICK_GUIDE.md** - This file (quick reference)

## Estimated Effort

- **Critical Fixes** (Priority 1-2): 2-3 hours
- **High Fixes** (Priority 3-4): 1-2 hours
- **Medium Fixes** (Priority 5): 1 hour
- **Low Fixes** (Priority 6-10): 30 mins (optional)

**Total**: 4-6 hours for production-ready code

## Most Common Issues

1. **Import not found**: Check actual module structure with:
   ```bash
   python -c "from finance_ml.ml_workflow.MODULE import FUNCTION"
   ```

2. **Function name mismatch**: Search codebase:
   ```bash
   grep -rn "def function_name" finance_ml/
   ```

3. **False positives**: Some issues may be due to notebook cell execution order - verify by running notebook

## Next Steps

1. Review **NOTEBOOK_REFACTORING_DETAILED_FIXES.md** for specific line numbers
2. Start with Section 1 (Critical Missing Imports)
3. Test after each section
4. Commit changes after each working phase

## Contact

If functions don't exist in expected modules:

- Check if module needs to be created
- Check if function was renamed
- Verify against project structure
