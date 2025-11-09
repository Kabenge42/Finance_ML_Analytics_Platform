# Finance ML Analytics Platform — Code Guidelines

These guidelines codify conventions for function signatures, return types, dataset preparation, column naming/schema,
and general Python best practices. They align with the project’s business objectives and the Phase 9.1–9.8 modular
design described in README.md and the improvement plans.

Goals

- Maximize code quality, maintainability, and testability
- Ensure consistent APIs across modules and phases
- Guarantee schema and naming consistency end-to-end
- Make downstream analytics, notebooks, and CLI predictable and robust

1) Standardized Function Signatures and Return Types

1.1 Training functions (train_*)

- Contract: All model-training functions return a dict with these keys:
    - model: The fitted estimator or pipeline
    - metrics: Dict[str, float] — evaluation metrics (e.g., accuracy, f1_macro for classification; mae, rmse, r2 for
      regression)
    - y_pred: 1D array-like or pandas Series/DataFrame of predictions aligned to input indices
    - y_proba: Optional 2D array-like or DataFrame of class probabilities (classification only). Omit or set to None for
      regressors.
    - artifacts: Optional Dict[str, Any] — auxiliary items (e.g., feature_importance, confusion_matrix, oof_predictions,
      cv_results)

- Examples:

```python
# Classification
res = train_event_classifier(X, y, model="lightgbm")
assert set(res).issuperset({"model", "metrics", "y_pred"})
acc = res["metrics"].get("accuracy")
f1m = res["metrics"].get("f1_macro")
y_proba = res.get("y_proba")  # May be None if estimator has no predict_proba

# Regression
res = train_and_evaluate_regression(df)
mae = res["metrics"].get("mae")
r2 = res["metrics"].get("r2")
y_pred = res["y_pred"]  # Series/DataFrame aligned to df index
```

- Backward compatibility: Where legacy code expects top-level metric keys (e.g., res["mae"]) provide shims during
  transition, but write new code to use res["metrics"]["mae"].

1.2 Dataset preparation

- Contract: Dataset prep functions return a 5-tuple or a dataclass:
    - (X_train, X_test, y_train, y_test, meta)
    - Where meta is a dict or small dataclass including feature_names, categorical_features, target_name, indices, and
      any scalers/encoders if applicable.
- Dataclass option:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class DatasetSplit:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    meta: Dict[str, Any]
```

- Rationale: A consistent shape across phases simplifies tests and integration between classification, regression, and
  analytics.

2) Column Naming and Schema (all_stocks dataframe)

2.1 Normalized column names based on `create_equities_schema.sql schema`

- Always normalize DataFrame column names early via `finance_ml.data.normalize_columns`.
- Canonical names (must exist if relevant in your workflow):
    - last_price
    - price_target (preferred target)
    - price_target_median (optional, fallback target)
    - sector
    - region
    - ticker (identifier)
- Normalization rules:
    - Lowercase snake_case
    - Replace non-alphanumeric with underscores
    - Trim leading/trailing underscores
    - Preserve data types

2.2 Downstream assumptions

- All modules must assume normalized names. Do not mix raw CSV header style (e.g., "Last Price" or "Price Target").
- When joining/merging, preserve index alignment and canonical names.
- Tests assume normalized columns for loaders and downstream utilities.

2.3 Validation

- Use `validate_schema(df, require_target: bool)` to assert required fields.
- For notebook/script workflows, validate after normalization and before heavy processing.

3) DataFrame Shape and Feature References

- Keep `all_stocks` as the single, unified DataFrame across regions.
- Clearly separate identifiers/targets from features:
    - Identifiers: ticker, sector, region
    - Targets: price_target (and optional price_target_median)
    - Features: everything else numeric/categorical after filtering
- Use helper utilities to extract feature columns; avoid hard-coded lists spread across files.

4) Typing, Logging, and Errors

- Typing: add type hints for public APIs and internal utilities.
- Logging: prefer `logging` over prints. For notebooks, prints are ok for user feedback, but underlying package
  functions should log.
- Errors: raise specific exceptions; avoid broad `except Exception` unless re-raising with context. Provide actionable
  messages.

5) Reproducibility and Configuration

- Respect environment variables and config objects (e.g., RANDOM_SEED, N_JOBS, DATA_DIR, DB_URL).
- Avoid hard-coded paths; use `pathlib.Path` and config.
- Document default behavior in docstrings.

6) Testing Conventions

- Use unittest; keep tests deterministic and fast where possible.
- Mock external services (DB) for unit tests.
- Provide small sample data for functional tests.
- Ensure coverage for normalization, dataset prep returns, and train_* result schema.

7) Notebook and CLI Alignment

- Notebooks and CLI should:
    - Normalize columns immediately after loading using `normalize_columns()`
    - Validate schema with `validate_schema()` using canonical names
    - Expect training functions to return the standardized dict and read metrics via `res["metrics"]`
- Maintain light wrapper logic in notebooks; delegate work to `finance_ml` package APIs.

Appendix A — Quick Reference

- Train result schema: {model, metrics, y_pred, y_proba?, artifacts?}
- Dataset prep return: (X_train, X_test, y_train, y_test, meta) or DatasetSplit dataclass
- Canonical columns: last_price, price_target, price_target_median, sector, region, ticker
- Normalize early: `df = normalize_columns(df)`
- Validate before modeling: `validate_schema(df, require_target=True)`

Appendix B — Migration Notes

- Legacy code reading top-level metrics (e.g., `res['rmse']`) should be updated to `res['metrics']['rmse']`. Provide
  shims in training functions during the migration window.
- Replace references to raw column names (e.g., `"Last Price"`) with canonical names after `normalize_columns()`.

Appendix C — Preprocessing Function Parameter Differences (Phase 9.1 Migration)

**IMPORTANT**: During Phase 9.1 refactoring, preprocessing functions were moved from `data.py` to
`finance_ml.ml_workflow.preprocessing/` with updated signatures. The package-level imports may route to OLD versions
with DIFFERENT parameter names.

**Issue**: The old `data.py` functions expect singular `column` parameter, while new preprocessing functions accept
plural `columns`. Calling with wrong parameter names causes
`TypeError: unexpected keyword argument 'columns'. Did you mean 'column'?`

**Affected Functions**:

1. **detect_outliers_iqr**
    - OLD (data.py): `def detect_outliers_iqr(df, column: str, multiplier: float = 1.5)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_iqr(df, columns: Optional[List[str]] = None, by_sector: bool = True, iqr_multiplier: float = 1.5)`
    - **Solution**: Loop through each column individually when using old version:
      ```python
      outliers_iqr = {}
      for col in financial_metrics[:20]:
          outliers_iqr[col] = detect_outliers_iqr(all_stocks, column=col, multiplier=1.5)
      ```

2. **detect_outliers_zscore**
    - OLD (data.py): `def detect_outliers_zscore(df, column: str, threshold: float = 3.0)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_zscore(df, columns: Optional[List[str]] = None, threshold: float = 3.0, by_sector: bool = True)`
    - **Solution**: Loop through each column individually:
      ```python
      outliers_zscore = {}
      for col in financial_metrics[:20]:
          outliers_zscore[col] = detect_outliers_zscore(all_stocks, column=col, threshold=3.0)
      ```

3. **detect_outliers_isolation_forest**
    - OLD (data.py): `def detect_outliers_isolation_forest(df, column: str, contamination: float = 0.1)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_isolation_forest(df, columns: Optional[List[str]] = None, contamination: float = 0.1)`
    - **Solution**: Loop through each column individually:
      ```python
      outliers_iforest = {}
      for col in financial_metrics[:20]:
          outliers_iforest[col] = detect_outliers_isolation_forest(all_stocks, column=col, contamination=0.1, random_state=42)
      ```

4. **winsorize_by_sector**
    - OLD (data.py):
      `def winsorize_by_sector(df, columns: List[str], sector_column: str = "sector", lower: float = 0.01, upper: float = 0.99)`
    - NEW (preprocessing/outliers.py):
      `def winsorize_by_sector(df, columns: Optional[List[str]] = None, lower_percentile: float = 0.01, upper_percentile: float = 0.99, by_sector: bool = True)`
    - **Solution**: Use correct parameter names for old version:
      ```python
      all_stocks = winsorize_by_sector(
          all_stocks,
          columns=financial_metrics[:20],
          lower=0.01,  # NOT lower_percentile
          upper=0.99,  # NOT upper_percentile
          sector_column='sector'  # NOT by_sector=True
      )
      ```

**Best Practices**:

1. **Explicit imports**: Import directly from the subpackage you intend to use:
   ```python
   # Use new Phase 9.1 version explicitly
   from finance_ml.ml_workflow.preprocessing.outliers import detect_outliers_iqr
   ```

2. **Check function signature**: Before calling, verify parameter names match the version you're using:
   ```python
   import inspect
   print(inspect.signature(detect_outliers_iqr))
   ```

3. **Per-column processing**: When using old data.py versions, always loop through columns individually rather than
   passing a list.

4. **Migration path**: Update imports to use new Phase 9.1 preprocessing modules once they're stable and fully tested.

**Testing**: Add unit tests that verify function signatures match expected parameter names to catch these issues early.
