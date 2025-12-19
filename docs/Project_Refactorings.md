Based on my analysis of the codebase and alignment with `code_guidelines.md`, here are actionable refactoring tasks
organized by priority:

## High Priority Refactoring Tasks

### 1. **Consolidate Inline Helper Functions into `finance_ml` Package**

The attached `fix_phase95_refactor.py` contains a `get_r2_score()` helper function that should be moved to the package:

**Action:** Move to `finance_ml/ml_workflow/regression/utils.py`

```python
# finance_ml/ml_workflow/regression/utils.py
def get_r2_score(item):
    """
    Safely extract R² score from various data structures.
    
    Handles training function return formats per code_guidelines.md Section 7.1:
    - Dict with 'metrics' sub-dict containing 'r2' or 'r2_score'
    - Dict with direct 'r2_score' key (legacy format)
    - Tuple from dict.items() -> (key, value)
    - Direct numeric values
    
    Args:
        item: Result from training function or (key, value) tuple
    
    Returns:
        float: Extracted R² score, defaults to 0.0 if not found
    """
    # ... implementation from fix_phase95_refactor.py
```

---

### 2. **Standardize Training Function Return Types (Section 7.1 Compliance)**

The `get_r2_score()` function handles multiple return formats, indicating inconsistent return types across training
functions.

**Action:** Audit and standardize all `train_*` functions to return:

```python
{
    "model": fitted_estimator,
    "metrics": {"r2": float, "mae": float, "rmse": float, ...},
    "y_pred": array_like,
    "y_proba": Optional[array_like],
    "artifacts": Optional[Dict]
}
```

**Files to update:**

- `finance_ml/ml_workflow/regression/models.py` - `train_random_forest_regressor`, `train_extra_trees_regressor`
- Ensure `metrics` sub-dict is always present (not top-level keys)

---

### 3. **Extract Magic Numbers from `etl_data_explorer.ipynb`**

The notebook defines constants locally that should reference `code_guidelines.md` Section 2.1:

```python
# Replace local definitions with imports from config
from finance_ml.ml_workflow.config import (
    TARGET_COL,
    TARGET_COL_FALLBACK,
    TEST_SIZE,
    CV_FOLDS,
    QUANTILES,
    MIN_SECTOR_SAMPLES,
    WINSORIZE_LOWER,
    WINSORIZE_UPPER,
    RANDOM_SEED,
    MODEL_VERSION,
)

# Remove duplicate definitions:
# TARGET_COL = 'price_target'  # DELETE - use import
# TEST_SIZE = 0.2  # DELETE - use import
```

---

### 4. **Refactor `equities_dashboard_app.py` - Extract Long Functions**

The `create_app()` function spans ~88K characters (lines 35300-123216). Per Section 6.2, functions should follow single
responsibility principle.

**Action:** Extract into separate modules:

```
finance_ml/dashboards/
├── equities_dashboard_app.py      # Main app entry, routing
├── components/
│   ├── filters.py                 # _safe_options, apply_filters
│   ├── kpi_cards.py              # _kpi_cards, _monitoring_kpi_cards
│   ├── charts.py                 # _target_vs_price_scatter, _market_cap_distribution
│   ├── earnings.py               # create_earnings_events_chart
│   └── artifacts.py              # _list_artifacts, _render_artifact
└── callbacks/                     # Dash callback handlers
```

---

### 5. **Consolidate Duplicate Column Classification Logic**

Both `etl_data_explorer.ipynb` and `equities_dashboard_app.py` define similar column handling. Use the canonical source:

**Action:** Always import from `finance_ml.ml_workflow.preprocessing.column_semantics`:

```python
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    PRICE_COLUMNS,
    classify_columns,
    get_winsorizable_columns,
)
```

---

## Medium Priority Refactoring Tasks

### 6. **Migrate `refactor_notebook.py` Logic to Package**

The `refactor_notebook.py` script contains useful notebook analysis functions that should be part of the quality module:

**Action:** Move to `finance_ml/ml_workflow/quality/notebook_review.py`:

```python
# finance_ml/ml_workflow/quality/notebook_review.py
def is_function_definition_cell(cell: dict, finance_ml_functions: list) -> bool:
    """Check if a cell contains function definitions that should be in package."""
    ...

def is_legacy_import_cell(cell: dict) -> bool:
    """Check if cell contains deprecated imports."""
    ...

def should_keep_cell(cell: dict, index: int) -> bool:
    """Determine if a cell should be kept in refactored notebook."""
    ...
```

---

### 7. **Add Type Hints to Dashboard Functions**

Per Section 6.2, type hints are required for function signatures:

```python
# ... existing code ...
def load_data_csv_first(
        data_dir: Path,
        limit: Optional[int] = None,
        normalize: bool = True
) -> pd.DataFrame:
    """Load data from CSV files with automatic fallback."""
    # ... existing code ...


def validate_required_columns(
        df: pd.DataFrame,
        required: List[str]
) -> Tuple[bool, List[str]]:
    """Validate that required columns exist in DataFrame."""
    # ... existing code ...
```

---

### 8. **Standardize Temporal Calculations (Section 9.3.0)**

Per v1.13 guidelines, all temporal calculations must use `reference_date`:

**Action:** Audit and update:

- `engineer_temporal_features()` - Verify `reference_date` parameter usage
- `create_earnings_calendar_dashboard()` - Use consistent `reference_date`
- Dashboard components calculating `days_to_earnings`

---

### 9. **Create Unified ETL Configuration Factory**

The `etl_data_explorer.ipynb` has extensive ETLConfig setup. Create a factory function:

```python
# finance_ml/ml_workflow/preprocessing/etl_presets.py
def get_etl_config_comprehensive() -> ETLConfig:
    """Get comprehensive ETL config for full feature engineering."""
    return ETLConfig(
        extraction=DataExtractionConfig(normalize_column_names=True),
        validation=SchemaValidationConfig(
            validate_schema=True,
            schema_alignment_threshold=0.80,
        ),
        # ... standard comprehensive config ...
    )

def get_etl_config_quick() -> ETLConfig:
    """Get minimal ETL config for fast iteration."""
    ...
```

---

## Low Priority Refactoring Tasks

### 10. **Add Deprecation Warnings for Legacy Patterns**

Per Section 4.4, deprecated modules should emit warnings:

```python
# finance_ml/ml_workflow/advanced_preprocessing.py
import warnings

def preprocess_data(*args, **kwargs):
    warnings.warn(
        "advanced_preprocessing.preprocess_data is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.etl.etl_with_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # ... fallback implementation ...
```

---

### 11. **Enhance `analysis_summary.html` Template**

The HTML report shows unrealistic metrics (e.g., "10629341.84% mispricing"). Add validation:

```python
# finance_ml/ml_workflow/reporting/html_templates.py
def validate_mispricing_display(value: float) -> str:
    """Format mispricing with sanity bounds."""
    if abs(value) > 1000:  # 1000% threshold
        return f">{1000 if value > 0 else -1000}%*"
    return f"{value:.2f}%"
```

---

### 12. **Document and Test Earnings Quality Features (v1.12)**

The new 33 earnings quality features need comprehensive tests:

**Action:** Create `tests/test_earnings_quality_features.py`:

```python
class TestEarningsQualityFeatures(unittest.TestCase):
    def test_engineer_estimated_vs_actual_analytics(self):
        """Test EPS/revenue surprise calculations."""
        ...
    
    def test_engineer_gaap_vs_adjusted_analytics(self):
        """Test GAAP vs Adjusted earnings comparison."""
        ...
    
    def test_earnings_quality_score_bounds(self):
        """Verify earnings_quality_score is in [0, 100]."""
        ...
```

---

## Summary Checklist

| Priority  | Task                                    | Guideline Reference | Estimated Effort |
|-----------|-----------------------------------------|---------------------|------------------|
| 🔴 High   | Consolidate `get_r2_score()` to package | §7.1                | 1 hour           |
| 🔴 High   | Standardize training function returns   | §7.1                | 4 hours          |
| 🔴 High   | Extract magic numbers from notebook     | §8.1, §8.3          | 2 hours          |
| 🔴 High   | Refactor `create_app()` monolith        | §6.2                | 8 hours          |
| 🔴 High   | Consolidate column classification       | §8.5                | 1 hour           |
| 🟡 Medium | Migrate notebook refactor logic         | §6.2                | 2 hours          |
| 🟡 Medium | Add type hints to dashboards            | §6.2                | 3 hours          |
| 🟡 Medium | Standardize temporal calculations       | §9.3.0              | 2 hours          |
| 🟡 Medium | Create ETL config factory               | §7.5                | 2 hours          |
| 🟢 Low    | Add deprecation warnings                | §4.4                | 1 hour           |
| 🟢 Low    | Enhance HTML report validation          | §17                 | 2 hours          |
| 🟢 Low    | Test earnings quality features          | §9.3                | 4 hours          |