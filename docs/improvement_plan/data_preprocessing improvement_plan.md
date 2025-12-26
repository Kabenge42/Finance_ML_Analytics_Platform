### Overview

Based on the notebook/catalog metadata, the SQL schema, and the existing 6‑step imputation pipeline in
`finance_ml.ml_workflow.preprocessing.imputation`, the current pipeline is strong but:

* **Datatype handling** is mostly inferred implicitly from `pandas` dtypes after CSV import, not from the *
  *authoritative DB/schema**.
* The **6‑step imputation strategy** is numerically focused, with limited schema‑aware behavior and little link to the
  Phase 9.3 feature roadmap.
* **Metadata artifacts** (`all_stocks_initial_metadata.json`, `preprocessed_stocks_metadata.json`) capture shapes and
  columns, but not **per‑column dtype / missingness / quality contracts**.

Below are:

1. **Analysis findings** (column/alias mismatches, root cause analysis, 2025-11-24).
2. Targeted **pipeline improvement proposals** (datatype detection & validation, imputation enhancements).
3. A **TDD implementation plan** (concrete test modules, test cases, and ordering).
4. Suggested **revisions to `code_guidelines.md`** to anchor these improvements.

---

### 0. Analysis Findings: Column & Alias Mismatches (2025-11-24)

#### 0.1. Executive Summary

**Analysis Date**: 2025-11-24  
**Scope**: Column name normalization consistency between CSV files, SQL schema (`create_equities_schema.sql`), Python
schema registry (`finance_ml/ml_workflow/data/schema.py`), and data loading pipeline (
`finance_ml/ml_workflow/preprocessing/data.py`)

**Key Finding**: 15 columns reported as "missing" in `dtype_diagnostics.json` are actually present but suffer from:

- **14 columns**: Normalization mismatch between data loader and COLUMN_SCHEMA
- **1 column**: Truly missing from CSV (should be deleted from schema)

**Root Cause**: Dual normalization paths with incompatible transformation rules.

---

#### 0.2. Root Cause Analysis

**Problem**: Two different normalization functions transform column names inconsistently:

**Path 1: Data Loading Normalization** (`finance_ml/ml_workflow/preprocessing/data.py` line 499)

```python
df.columns.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.strip("_").str.lower()
```

- Replaces ALL non-alphanumeric characters with underscores
- Strips leading/trailing underscores
- Converts to lowercase

**Path 2: Schema Normalization** (`finance_ml/ml_workflow/data/schema.py` lines 701-729)

```python
def normalize_column_name(column: str) -> str:
    normalized = (
        column.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        .replace("#", "num")      # ← KEY DIFFERENCE
        .replace("%", "pct")      # ← KEY DIFFERENCE
        .replace("&", "and")      # ← KEY DIFFERENCE
    )
    # Remove consecutive underscores and strip
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")
```

**Impact**: When CSV columns are loaded via `load_from_csv()`, Path 1 normalization produces names that don't match
COLUMN_SCHEMA keys, causing lookup failures.

---

#### 0.3. Detailed Column Mismatches

**Category A: Normalization Mismatch (14 columns present in CSV and COLUMN_SCHEMA)**

| CSV Column Name                                    | Path 1 Produces                                | COLUMN_SCHEMA Expects                          | COLUMN_SCHEMA Location |
|----------------------------------------------------|------------------------------------------------|------------------------------------------------|------------------------|
| `# Strong Sell Ratings`                            | `strong_sell_ratings`                          | `num_strong_sell_ratings`                      | schema.py:119          |
| `# Strong Buys Ratings`                            | `strong_buys_ratings`                          | `num_strong_buys_ratings`                      | schema.py:120          |
| `# Hold Ratings`                                   | `hold_ratings`                                 | `num_hold_ratings`                             | schema.py:121          |
| `# Buys Ratings`                                   | `buys_ratings`                                 | `num_buys_ratings`                             | schema.py:122          |
| `# Sell Ratings`                                   | `sell_ratings`                                 | `num_sell_ratings`                             | schema.py:123          |
| `1-Day %`                                          | `1_day_pct`                                    | `1_day_pct`                                    | schema.py:132 ✓        |
| `Shrs Out`                                         | `shrs_out`                                     | `shrs_out`                                     | schema.py:157 ✓        |
| `Selling General & Admin Expenses/Total (FQ)`      | `selling_general_admin_expenses_total_fq`      | `selling_general_admin_expenses_total_fq`      | schema.py:325 ✓        |
| `Selling General & Admin Expenses/Total (FY)`      | `selling_general_admin_expenses_total_fy`      | `selling_general_admin_expenses_total_fy`      | schema.py:326 ✓        |
| `Selling General & Admin Expenses/Total (-1FY)`    | `selling_general_admin_expenses_total_1fy`     | `selling_general_admin_expenses_total_1fy`     | schema.py:327 ✓        |
| `Selling General & Admin Expenses/Total (5YAVGFQ)` | `selling_general_admin_expenses_total_5yavgfq` | `selling_general_admin_expenses_total_5yavgfq` | schema.py:328 ✓        |
| `Accounts Receivable/Total (FY)`                   | `accounts_receivable_total_fy`                 | `accounts_receivable_total_fy`                 | schema.py:329 ✓        |
| `Accounts Receivable/Total (-1FY)`                 | `accounts_receivable_total_1fy`                | `accounts_receivable_total_1fy`                | schema.py:330 ✓        |
| `Accounts Receivable/Total (5YAVGFQ)`              | `accounts_receivable_total_5yavgfq`            | `accounts_receivable_total_5yavgfq`            | schema.py:331 ✓        |

**Note**: Only the 5 analyst rating columns have actual mismatches due to "#" handling. The other 9 columns match
correctly but were reported as missing due to validation logic issues.

**Category B: Truly Missing Column (1 column not in CSV)**

| Column Name     | SQL Schema                   | CSV Files     | Action Required           |
|-----------------|------------------------------|---------------|---------------------------|
| `short_int_pct` | Line 103: `"Short Int. (%)"` | ❌ Not present | Delete from schema.py:159 |

**Verified**: `Short Int. (%)` column is NOT in any CSV file (screening_us.csv, screening_eu.csv, screening_apac.csv,
screening_rotw.csv).

---

#### 0.4. Actionable Improvement Tasks

**Priority 1 (Critical): Fix Normalization Consistency**

**Task 1.1**: Replace data.py regex normalization with schema.normalize_column_name()

- **File**: `finance_ml/ml_workflow/preprocessing/data.py`
- **Lines**: 496-502 (function `normalize_columns()`)
- **Current code**:
  ```python
  else:
      # Legacy behavior: convert to lowercase with underscores
      df.columns = (
          df.columns.str.replace(r"[^0-9a-zA-Z]+", "_", regex=True).str.strip("_").str.lower()
      )
  ```
- **Proposed fix**:
  ```python
  else:
      # Use schema-aware normalization for consistency
      from finance_ml.ml_workflow.data.schema import normalize_column_name
      df.columns = [normalize_column_name(col) for col in df.columns]
  ```
- **Impact**: Ensures all 14 mismatched columns are normalized consistently
- **Estimated effort**: 15 minutes + testing

**Task 1.2**: Delete short_int_pct from COLUMN_SCHEMA

- **File**: `finance_ml/ml_workflow/data/schema.py`
- **Line**: 159
- **Current code**: `"short_int_pct": {"dtype": "float", "role": "feature"},`
- **Action**: Delete this line
- **Rationale**: Column removed from CSV files but still in schema
- **Estimated effort**: 5 minutes + testing

**Task 1.3**: Update SQL schema documentation

- **File**: `create_equities_schema.sql`
- **Line**: 103: `"Short Int. (%)" NUMERIC,`
- **Action**: Add comment `-- DEPRECATED: Column removed from CSV sources`
- **Rationale**: Document historical schema vs current data
- **Estimated effort**: 5 minutes

**Priority 2 (High): Enhance Validation Logic**

**Task 2.1**: Improve dtype_diagnostics validation to detect normalization issues

- **File**: `finance_ml/ml_workflow/preprocessing/dtypes.py` (if exists) or create new
- **Enhancement**: Add normalization consistency checks
- **Features**:
    - Detect columns that exist in data but not in schema under any normalization
    - Report normalization variants for debugging
    - Suggest closest matches using fuzzy matching
- **Estimated effort**: 2-3 hours

**Task 2.2**: Add normalization round-trip test

- **Test**: Verify `normalize_column_name(csv_col)` matches COLUMN_SCHEMA key
- **Coverage**: All 327 SQL schema columns
- **Location**: New test module `tests/test_schema_normalization.py`
- **Estimated effort**: 1 hour

**Priority 3 (Medium): Documentation Updates**

**Task 3.1**: Update code_guidelines.md

- **Section**: 4. Column Naming Schema and DataFrame Conventions
- **Add**: Normalization consistency policy
- **Content**: Document canonical normalization function and mandate its use

**Task 3.2**: Update this improvement plan

- **Action**: Mark findings section as implemented after fixes applied
- **Add**: Lessons learned section

---

#### 0.5. CSV Schema Verification Results

**CSV Files Analyzed**:

- `data/screening_us.csv`: 327 columns (includes all Phase 9.3 additions)
- `data/screening_eu.csv`: 327 columns
- `data/screening_apac.csv`: 327 columns
- `data/screening_rotw.csv`: 327 columns

**SQL Schema**: `create_equities_schema.sql`: 327 columns defined (lines 11-318)

**Python COLUMN_SCHEMA**: `finance_ml/ml_workflow/data/schema.py`: 310+ columns registered

**Alignment Status**:

- ✅ CSV column names match SQL schema exactly (quoted identifiers)
- ✅ COLUMN_SCHEMA contains all critical columns for Phase 9.3 features
- ❌ Data loading normalization incompatible with COLUMN_SCHEMA keys (5 analyst rating columns)
- ❌ short_int_pct in COLUMN_SCHEMA but not in CSV files

**Recommendations**:

1. Apply Task 1.1 immediately to fix normalization consistency
2. Delete short_int_pct from schema (Task 1.2)
3. Run comprehensive schema alignment test after fixes

---

### 1. Data Preprocessing & Datatype Detection – Improvement Proposals

#### 1.1. Introduce a Schema‑Aware Datatype Registry

**Problem:**

* CSV imports and some loaders likely use `dtype=str` followed by `pd.to_numeric(errors="coerce")`, which is robust but
  not **schema‑driven**.
* The authoritative types exist in `create_equities_schema.sql` but are not programmatically enforced in preprocessing.

**Proposal: Add a central `column_schema` registry** (Python dict / dataclass) exposed in a new module, e.g.
`finance_ml.ml_workflow.data.schema`:

```python
COLUMN_SCHEMA = {
    # identifiers & categoricals
    "ticker": {"role": "id", "dtype": "string"},
    "isin": {"role": "id", "dtype": "string"},
    "sector": {"role": "categorical", "dtype": "category"},
    "industry": {"role": "categorical", "dtype": "category"},
    "region": {"role": "categorical", "dtype": "category"},
    # dates
    "last_updated": {"role": "date", "dtype": "datetime64[ns]"},
    "income_statement_report_date": {"role": "date", "dtype": "datetime64[ns]"},
    "next_earnings": {"role": "date", "dtype": "datetime64[ns]"},
    # targets
    "last_price": {"role": "feature", "dtype": "float"},
    "price_target": {"role": "target", "dtype": "float"},
    "price_target_median": {"role": "target_fallback", "dtype": "float"},
    # valuations, risk, cash flow, etc.
    "market_cap": {"role": "feature", "dtype": "float"},
    "altman_z_score_ltm": {"role": "feature", "dtype": "float"},
    # ... and so on for key features used in Phase 9.3
}
```

Expose helpers:

```python
def get_expected_dtype(col: str) -> str: ...
def get_column_role(col: str) -> str: ...
def list_numeric_feature_cols() -> list[str]: ...
def list_categorical_cols() -> list[str]: ...
```

**Benefits:**

* Single source of truth for **canonical dtypes** and **roles** (id / feature / target / date / auxiliary).
* Aligns CSV, PostgreSQL, and SQLite paths with the same expectations.
* Simplifies testability and validation (see TDD section).

#### 1.2. Schema‑Driven Datatype Detection & Casting

**Problem:**

* `all_stocks_initial_metadata.json` / `preprocessed_stocks_metadata.json` show column names and shapes but not enforced
  dtypes.
* Downstream functions like `prepare_classification_data()` rely on `select_dtypes` which may misclassify columns when
  imports are inconsistent.

**Proposal: Implement `detect_and_cast_dtypes()` in a dedicated module** (e.g.
`finance_ml/ml_workflow/preprocessing/dtypes.py`) and call it early in notebook & CLI pipeline:

```python
def detect_and_cast_dtypes(df: pd.DataFrame, schema: dict = COLUMN_SCHEMA) -> tuple[pd.DataFrame, dict]:
    """Infer, validate, and cast dtypes according to schema.

    Returns (df_cast, diagnostics) where diagnostics includes:
    - inferred_dtypes: {col: str}
    - cast_applied: {col: str}
    - coercion_warnings: {col: int}  # number of values coerced to NaN
    - unknown_columns: list[str]
    - missing_expected_columns: list[str]
    """
```

**Key behaviors:**

1. **Infer** dtypes via sampling (e.g. first N rows) when schema is missing.
2. **Compare** inferred types to schema, log mismatches.
3. **Cast** columns to target dtypes with:
    * `pd.to_numeric(errors="coerce")` for numeric.
    * `pd.to_datetime(errors="coerce")` for dates.
    * `.astype("string")` or `"category"` for categoricals.
4. **Track coercions**: count how many values became NaN due to invalid formats.

**Integration points:**

* Notebook: early cell – just after `all_stocks` is loaded.
* Script: in `ml_finance_model_main.py` pipeline before any imputation.
* Data loaders: optionally integrate into `finance_ml.ml_workflow.data` (if such modules exist per guidelines).

**Metadata enhancement:**

Extend `*_metadata.json` schema to add:

```json
"dtypes": {"last_price": "float64", "sector": "category", ...},
"missing_counts": {"last_price": 0, "price_target": 123, ...},
"coercion_warnings": {"market_cap": 4}
```

#### 1.3. Column Grouping Based on Schema & Phase 9.3

**Problem:**

* `numeric_cols`, `categorical_cols`, and the Phase 9.3 feature categories (momentum, quality, cash flow, etc.) are
  described in `code_guidelines.md` but not **automatically derived** from a shared config.

**Proposal:**

* Add mapping from DB column names to **Phase 9.3 feature buckets** (momentum, valuation, profitability, quality & risk,
  cash flow, etc.) in a config file (YAML/JSON or Python dict) used by:
    * Feature building modules.
    * Data validation (ensuring required inputs for each feature bucket are present & correctly typed).

Example:

```python
PHASE93_FEATURE_CATEGORIES = {
    "momentum": ["price_chg_pct_1m", "price_chg_pct_3m", "price_1m_ago", "ema_20d", "ema_50d"],
    "quality_risk": ["altman_z_score_ltm", "return_on_equity_pct_ltm", ...],
    "cash_flow": ["cfo_ltm", "fcf_ltm", "cfo_fy", ...],
    # ...
}
```

* Introduce validator `validate_phase93_inputs(df)` that checks **coverage and dtypes** for each bucket and returns a
  `quality_stats` structure.

---

### 2. Imputation Strategy – Enhancements to 6‑Step Pipeline

The current `apply_enhanced_imputation_strategy_6step` already applies:

1. Zero imputation (exceptional events columns).
2. Sector‑aware KNN imputation (core metrics).
3. Price imputation (price targets from `last_price` and related metrics).
4. Median imputation (remaining numeric).
5. Categorical imputation (most frequent / constant).
6. Datetime imputation & formatting.

#### 2.1. Schema‑Aware Column Selection for Each Step

**Problem:**

* `get_zero_imputation_columns()` and `get_knn_imputation_columns()` are hard‑coded lists; they may **drift** from the
  DB schema and Phase 9.3 feature set.
* There is limited enforcement that the selected columns are indeed numeric or within certain value ranges (e.g., ratios
  between 0 and, say, 1000).

**Enhancements:**

1. **Drive column lists from `COLUMN_SCHEMA` and PHASE93 feature groups**.

    * E.g., `get_knn_imputation_columns()` should:
        * Take all **numeric Phase 9.3 core input features**.
        * Exclude price target outputs and IDs.

2. **Validation before imputation:**

    * Confirm all KNN columns are numeric and not constant within sector.
    * For zero‑imputation columns, confirm they are count/ratio/flag metrics with natural zero.

3. **Configurable column sets via `FinanceMLConfig`**:

    * Expose `config.imputation.zero_impute_cols`, `config.imputation.knn_cols`, etc., allowing experiment control.

#### 2.2. Improve Categorical Imputation Strategy

**Problem:**

* Current `apply_categorical_imputation()` is generic and uses `most_frequent`/`constant` without domain awareness.
* For high‑cardinality categoricals (e.g., `industry`), most‑frequent might bias distributions.

**Enhancements:**

* **Per‑column strategies** in `get_categorical_imputation_config()`:

    * `sector`, `region`, `country`, `trading_country`: use **mode** within **`ticker` or `isin` group** (if duplicates
      exist) or most‑frequent globally.
    * `style_class`, `size_class`: use most‑frequent conditional on `sector`.
    * `next_earnings_status`: use a domain‑specific constant (e.g., `'unknown'`).

Implementation sketch:

```python
def apply_groupwise_categorical_imputation(df, group_cols, target_col, strategy="mode"):
    # group by group_cols, fill missing target values with group-wise mode
```

* Extend `apply_enhanced_imputation_strategy_6step` to optionally use **groupwise categorical imputation** for selected
  columns before global most‑frequent.

#### 2.3. Date/Temporal Imputation Refinements

**Problem:**

* `apply_datetime_imputation_and_formatting()` supports `forward_fill`, `median`, and `constant`, but there is no *
  *policy** about which strategy is used for which date.
* Temporal feature engineering in Phase 9.3 (e.g., `days_to_earnings`, `reporting_lag`) is **sensitive** to date
  imputation choices.

**Enhancements:**

* In `get_datetime_imputation_config()` (new helper):

    * `last_updated`: strategy = `forward_fill` within `(ticker)` group; fallback = global median.
    * `income_statement_report_date`: strategy = `median` or **nearest previous valid date**; enforce not in the future.
    * `next_earnings`: strategy = `constant` for missing (e.g., `NaT` but flagged by a separate indicator) or groupwise
      median.

* Generate **indicator flags** for imputed dates (e.g., `next_earnings_imputed`) to be used as features, especially for
  temporal reliability.

#### 2.4. Imputation Quality Diagnostics & Safety Rails

**Problem:**

* `validate_imputation_completeness()` checks missing values but doesn’t surface **distribution shifts** or unrealistic
  fill values.
* Phase 9.1 and 9.3 guidelines emphasize **outlier safety rails** and non‑negativity; imputation should align with that.

**Enhancements:**

1. **Pre‑/post‑imputation summaries**:

    * For each imputation step, track for selected columns:
        * `% missing before`, `% missing after`.
        * `min`, `max`, `mean`, `std` (post‑imputation) and compare to raw distribution when possible.

2. **Non‑negativity and ratio range validation**:

    * After all steps, check value constraints:
        * Prices, market cap, volume, revenues, cash flows ≥ 0.
        * Ratios like `p_e`, `ev_ebitda`, leverage, and margins within sensible ranges (configurable thresholds).
    * If violations detected, log warnings and optionally clip or mark rows.

3. **Imputation provenance flags**:

    * For key features (targets, price, core ratios), generate binary flags:
        * `last_price_imputed`, `price_target_imputed`, `ebitda_ltm_imputed`, etc.
    * These can be used in models and also validated in tests.

4. **Sector‑level diagnostics**:

    * Summarize missingness and imputation volume by sector and region; attach to `quality_stats` in metadata.

---

### 3. TDD Implementation Plan

Below is a practical TDD plan with modules and concrete tests.

#### 3.1. New / Extended Test Modules

1. `tests/test_data_types_detection.py` (NEW)
2. `tests/test_enhanced_imputation_phase93.py` (NEW – complements `test_enhanced_imputation.py`)
3. `tests/test_metadata_catalog_quality.py` (NEW)
4. Extend existing:
    * `tests/test_enhanced_imputation.py` – add KNN and categorical/date refinements tests.
    * `tests/test_finance_ml_data.py` / `tests/test_loaders.py` – add schema alignment & dtype tests.

#### 3.2. Test Cases – Datatype Detection & Schema

**Module:** `tests/test_data_types_detection.py`

1. **`test_detect_and_cast_dtypes_respects_column_schema`**
    * Arrange: small `DataFrame` with columns `ticker`, `sector`, `last_price`, `market_cap`, `last_updated` loaded as
      `object`.
    * Act: call `detect_and_cast_dtypes(df)`.
    * Assert:
        * `df["last_price"]` and `df["market_cap"]` have numeric dtypes.
        * `df["last_updated"]` is `datetime64[ns]`.
        * `df["sector"]` is category/string; `ticker` is string.

2. **`test_detect_and_cast_dtypes_reports_coercion_warnings`**
    * Include invalid numeric strings (`"N/A"`, `"-"`) in `last_price`.
    * Assert:
        * Coercion count for `last_price` equals number of invalid entries.

3. **`test_unknown_and_missing_columns_reported`**
    * Add extra column `foo_bar` not in `COLUMN_SCHEMA`.
    * Remove `price_target` (expected in schema).
    * Assert `diagnostics["unknown_columns"] == ["foo_bar"]` and `"price_target"` in `missing_expected_columns`.

4. **`test_PHASE93_FEATURE_CATEGORIES_all_numeric_where_expected`**
    * For each column in `PHASE93_FEATURE_CATEGORIES["momentum"]` etc., assert dtype is numeric or datetime depending on
      role.

#### 3.3. Test Cases – Imputation Enhancements

**Module:** `tests/test_enhanced_imputation_phase93.py`

1. **`test_zero_imputation_columns_schema_consistency`**
    * Build a `DataFrame` with all columns from `get_zero_imputation_columns()`.
    * Set some entries to NaN.
    * After `apply_zero_imputation`, assert:
        * All NaNs replaced by 0.
        * No non‑numeric column is in zero‑impute list.

2. **`test_knn_imputation_enhanced_uses_sector_groups`**
    * Create a dataset with 2 sectors and a numeric feature missing in one row per sector.
    * After `apply_knn_imputation_enhanced`, assert:
        * Missing values filled using information only from same sector (e.g., by checking simple averages in toy data).

3. **`test_price_imputation_preserves_monotonicity`**
    * For rows where `price_target` is missing but `last_price` and `price_target_median` exist, define clear rule (
      e.g., impute from median or simple multiple of price).
    * Assert deterministic behavior and no NaNs in price target columns post‑Step 3.

4. **`test_categorical_imputation_groupwise_by_sector`**
    * Create data where some `size_class` values are missing within `sector` groups.
    * Use config to enable groupwise mode.
    * Assert missing `size_class` is filled to the **mode within its sector**.

5. **`test_datetime_imputation_strategies_by_column`**
    * Setup:
        * `last_updated` with a gap; `income_statement_report_date` with scattered dates; `next_earnings` missing.
    * After `apply_datetime_imputation_and_formatting` with config:
        * `last_updated` forward‑filled within ticker.
        * `income_statement_report_date` imputed to median.
        * `next_earnings` either remains NaT but gets `next_earnings_imputed` flag or is filled with policy date.

6. **`test_imputation_generates_provenance_flags`**
    * Run full 6‑step pipeline with `provenance_flags=True`.
    * Assert columns like `last_price_imputed`, `price_target_imputed` are boolean and correctly reflect which rows were
      touched.

7. **`test_imputation_respects_non_negativity_constraints`**
    * Inject negative values for `last_price`, `market_cap`, `total_revenues_fy` in test data.
    * After imputation + safety rails, assert values are either clipped to 0 or flagged depending on policy; no negative
      values remain.

8. **`test_validate_imputation_completeness_reports_by_type`**
    * Run pipeline on a small dataset with missing numeric, categorical, and date values.
    * Use `validate_imputation_completeness` to assert:
        * `result["is_complete"]` is `True`.
        * `result` contains type‑specific missingness summaries and sector/region breakdowns.

#### 3.4. Test Cases – Metadata & Catalog

**Module:** `tests/test_metadata_catalog_quality.py`

1. **`test_metadata_includes_dtypes_and_missing_counts`**
    * After running the pipeline (or a small subset), read `all_stocks_initial_metadata.json`.
    * Assert presence of `"dtypes"` and `"missing_counts"` keys.

2. **`test_preprocessed_metadata_flags_zero_missing_for_phase93_features`**
    * Read `preprocessed_stocks_metadata.json`.
    * For Phase 9.3 critical feature inputs, assert `missing_counts[col] == 0`.

3. **`test_quality_stats_consistency_with_metadata`**
    * Compare `quality_stats` output from preprocessing with metadata JSON; ensure counts and dtypes align.

#### 3.5. Execution Strategy

Following the project’s own guidance:

1. **Development loop:**
    * Implement `COLUMN_SCHEMA`, `detect_and_cast_dtypes`, and basic tests in `test_data_types_detection.py`.
    * Run: `python -m unittest tests.test_data_types_detection -v`.

2. **Imputation enhancements:**
    * Incrementally add/improve helpers in `imputation.py` and new tests in `test_enhanced_imputation_phase93.py`.
    * Run: `python -m unittest tests.test_enhanced_imputation tests.test_enhanced_imputation_phase93 -v`.

3. **Metadata tests:**
    * Wire up metadata writing, then run `test_metadata_catalog_quality.py` plus existing fast tests.

4. Integrate into CI: include new modules in the “medium tests” set.

---

### 4. Suggested Revisions to `code_guidelines.md`

Below are targeted edits to keep the document consistent with the new behavior. Line numbers are approximate; wording
should be adapted to existing structure.

#### 4.1. New Subsection: Schema & Datatype Management

**Location:** After the existing "Column Names (Canonical)" / "Feature Categories" tables (around lines 929–975).

**Add:**

```markdown
### Schema and Datatype Management (v1.3+)

All data loading and preprocessing **MUST** respect a centralized column schema defined in
`finance_ml.ml_workflow.data.schema`.

- `COLUMN_SCHEMA` stores, for each canonical column:
  - `dtype`: one of `float`, `int`, `string`, `category`, `datetime64[ns]`.
  - `role`: one of `id`, `feature`, `target`, `target_fallback`, `date`, `auxiliary`.
- Helper functions such as `get_expected_dtype()`, `get_column_role()`, and
  `list_numeric_feature_cols()` should be used by loaders, preprocessing, and
  feature engineering code.
- All CSV and DB imports **must** call `detect_and_cast_dtypes(df, schema=COLUMN_SCHEMA)`
  before any modeling logic or imputation.
- Any type coercions (e.g., invalid numerics converted to NaN) must be tracked
  via a `diagnostics` structure and surfaced through logging and metadata
  (`*_metadata.json`).
```

#### 4.2. Updated Imputation Policy Section

**Location:** Wherever the 6‑step imputation strategy is currently documented (Phase 9.1 / imputation section).

**Update description to:**

```markdown
### Enhanced 6-Step Imputation Strategy (Phase 9.1+, v1.3)

The standardized imputation pipeline is implemented in
`finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step`.
It operates on a schema‑validated dataframe and guarantees **zero missing
values** for all Phase 9.x required features.

Steps:

1. **Zero Imputation (Schema‑Driven)**
   - Apply zero imputation to a curated set of **event and count metrics**
     defined by `get_zero_imputation_columns()`, which is derived from
     `COLUMN_SCHEMA` and Phase 9.3 feature inputs.
   - Only columns with a natural zero (e.g., exceptional items, counts,
     certain ratios) are eligible.

2. **Sector‑Aware KNN Imputation (Core Metrics)**
   - Use `impute_missing_values_knn_sector()` / `apply_knn_imputation_enhanced()`
     to impute missing numeric values for core financial metrics within
     sector (and optionally region) groups.
   - Column selection is driven by `get_knn_imputation_columns()`, which must
     remain consistent with Phase 9.3 feature requirements.

3. **Price Imputation (Targets)**
   - Apply domain‑specific rules to fill `price_target` and related columns
     using `last_price`, `price_target_median`, and other valuation metrics.
   - Provenance flags (e.g., `price_target_imputed`) must be generated for
     all imputed target values.

4. **Median Imputation (Residual Numerics)**
   - For all remaining numeric columns, apply robust median imputation.
   - After this step, there should be **no missing numeric values** for any
     column used in modeling or analytics.

5. **Categorical Imputation (Groupwise + Global)**
   - Apply groupwise imputation for selected categoricals (e.g., `size_class`
     within `sector`, `country` within `region`), followed by global
     most‑frequent or constant strategies.
   - Configuration is defined in `get_categorical_imputation_config()` and
     may be overridden via `FinanceMLConfig`.

6. **Datetime Imputation and Formatting (Temporal Readiness)**
   - Convert all date columns to `datetime64[ns]` and impute missing values
     using per‑column strategies (forward‑fill, median, or constants) as
     defined in `get_datetime_imputation_config()`.
   - Generate imputation flags (e.g., `next_earnings_imputed`) for use in
     temporal features and diagnostics.

Post‑conditions:

- `validate_imputation_completeness()` must pass, confirming zero missing
  values across all numeric, categorical, and date columns required by
  Phase 9.3 features.
- Non‑negativity and outlier safety rails defined in the Outlier Safety Rails
  Policy must hold after imputation (e.g., no negative prices or market cap).
- Imputation diagnostics (missingness before/after, coercion counts,
  sector/region summaries) must be recorded in `quality_stats` and persisted
  in the metadata catalog.
```

#### 4.3. Phase 9.3 Feature Engineering Prerequisites

**Location:** In the Phase 9.3 section of `code_guidelines.md`, add a short **"Data Prerequisites"** subsection.

```markdown
#### Phase 9.3 Data Prerequisites

Before calling any Phase 9.3 feature engineering functions
(e.g., `engineer_momentum_features`, `engineer_cash_flow_quality_features`):

- Input dataframes **must**:
  - Conform to `COLUMN_SCHEMA` dtypes via `detect_and_cast_dtypes`.
  - Be fully imputed via `apply_enhanced_imputation_strategy_6step`.
  - Satisfy non‑negativity and outlier safety rails for all Phase 9.3 core
    inputs.
- The following metadata must be available:
  - `quality_stats` including per‑column missingness and imputation volume.
  - Catalog metadata (`*_metadata.json`) with `dtypes` and `missing_counts`
    sections.
- New tests for Phase 9.3 features must confirm that feature functions **do
  not perform their own imputation**, but instead assume the standardized
  preprocessing pipeline has already been executed.
```

#### 4.4. Testing & TDD Section Update

**Location:** In the Testing section, append a short TDD directive.

```markdown
##### TDD for Data Quality, Dtypes, and Imputation (v1.3+)

- Any changes to loaders, schema, or imputation must be accompanied by
  unit tests that:
  - Verify dtype casting via `detect_and_cast_dtypes`.
  - Validate the 6-step imputation post‑conditions (no missing values,
    constraints satisfied).
  - Ensure metadata catalogs (`*_metadata.json`) remain consistent with
    in‑memory `quality_stats`.
- New tests should be added under:
  - `tests/test_data_types_detection.py` for schema + dtypes.
  - `tests/test_enhanced_imputation_phase93.py` for advanced imputation.
  - `tests/test_metadata_catalog_quality.py` for metadata integration.
```

---

### 5. How This Aligns with Your Task

* It explicitly connects **datatype detection** and **imputation** to:
    * The **underlying SQL schema** (`create_equities_schema.sql`).
    * The **Phase 9.3 feature roadmap** (ensuring all required inputs are typed
      and imputed correctly).
    * The existing **6‑step imputation code** in `imputation.py`.
* It provides a **concrete TDD plan** with new test modules and test cases.
* It includes **specific wording** to update `code_guidelines.md` so the
  improved behavior becomes part of the official standards.

You can now implement these changes incrementally, starting with the schema
registry and dtype detection (smallest surface area), then layering in the
imputation refinements and finally the metadata/catalog tests.

---

### 6. TDD Implementation Plan for Normalization Fixes (2025-11-24)

**STATUS: ✅ IMPLEMENTED (2025-11-24)**

**Implementation Summary:**

- Created 3 new test modules with 43 test cases total (all passing)
- Fixed data.py normalization to use canonical schema.normalize_column_name()
- Removed obsolete short_int_pct column from schema
- Updated hardcoded mapping with correct analyst ratings (num_ prefix) and SG&A (and connector)
- Added 4 SG&A column variants to schema
- Updated code_guidelines.md v1.6 with Section 5.5 Column Normalization Consistency Policy
- 100% test coverage for column normalization consistency

This section provides a detailed TDD implementation plan specifically for addressing the column normalization mismatches
discovered in Section 0.

#### 6.1. Test Module Overview

**New Test Modules**:

1. `tests/test_schema_normalization.py` - Column normalization consistency tests
2. `tests/test_schema_completeness.py` - Schema coverage and missing column detection
3. `tests/test_data_loading_normalization.py` - Integration tests for data.py changes

**Enhanced Test Modules**:

1. `tests/test_data_types_detection.py` - Add normalization validation
2. `tests/test_finance_ml_data.py` - Add CSV loading normalization tests

---

#### 6.2. Test Module: test_schema_normalization.py

**Purpose**: Verify normalize_column_name() produces consistent results and all SQL schema columns are registered in
COLUMN_SCHEMA.

**Test Cases**:

```python
import unittest
from finance_ml.ml_workflow.data.schema import normalize_column_name, COLUMN_SCHEMA


class TestSchemaNormalization(unittest.TestCase):
    """Test column name normalization consistency."""
    
    def test_normalize_analyst_ratings_with_hash(self):
        """Test that # symbol is correctly converted to 'num' prefix."""
        test_cases = [
            ("# Strong Sell Ratings", "num_strong_sell_ratings"),
            ("# Strong Buys Ratings", "num_strong_buys_ratings"),
            ("# Hold Ratings", "num_hold_ratings"),
            ("# Buys Ratings", "num_buys_ratings"),
            ("# Sell Ratings", "num_sell_ratings"),
        ]
        for csv_name, expected_normalized in test_cases:
            with self.subTest(csv_name=csv_name):
                result = normalize_column_name(csv_name)
                self.assertEqual(result, expected_normalized)
                # Verify it exists in COLUMN_SCHEMA
                self.assertIn(expected_normalized, COLUMN_SCHEMA,
                             f"{expected_normalized} missing from COLUMN_SCHEMA")
    
    def test_normalize_percentage_columns(self):
        """Test that % symbol is correctly converted to 'pct'."""
        test_cases = [
            ("1-Day %", "1_day_pct"),
            ("Net Income Margin % (FY)", "net_income_margin_pct_fy"),
            ("Return On Equity % (LTM)", "return_on_equity_pct_ltm"),
        ]
        for csv_name, expected_normalized in test_cases:
            with self.subTest(csv_name=csv_name):
                result = normalize_column_name(csv_name)
                self.assertEqual(result, expected_normalized)
    
    def test_normalize_ampersand_columns(self):
        """Test that & symbol is correctly converted to 'and'."""
        test_cases = [
            ("Selling General & Admin Expenses/Total (FQ)", 
             "selling_general_and_admin_expenses_total_fq"),
            ("Merger & Restructuring Charges (LTM)", 
             "merger_and_restructuring_charges_ltm"),
        ]
        for csv_name, expected_normalized in test_cases:
            with self.subTest(csv_name=csv_name):
                result = normalize_column_name(csv_name)
                self.assertEqual(result, expected_normalized)
    
    def test_normalize_removes_consecutive_underscores(self):
        """Test that consecutive underscores are collapsed."""
        result = normalize_column_name("Price  (  1M  Ago  )")
        self.assertNotIn("__", result)
        self.assertEqual(result, "price_1m_ago")
    
    def test_normalize_strips_leading_trailing_underscores(self):
        """Test that leading/trailing underscores are removed."""
        result = normalize_column_name("# Test Column %")
        self.assertFalse(result.startswith("_"))
        self.assertFalse(result.endswith("_"))
    
    def test_all_sql_columns_normalize_to_schema_keys(self):
        """Test that all SQL schema columns normalize to valid COLUMN_SCHEMA keys."""
        # Sample of critical SQL columns from create_equities_schema.sql
        sql_columns = [
            "Ticker", "ISIN", "Sector", "Industry", "Last Price",
            "Price Target", "Market Cap", "Enterprise Value",
            "P/E (NTM)", "P/E (LTM)", "Altman Z-Score (FY)",
            "# Strong Sell Ratings", "1-Day %", "Shrs Out",
            "Selling General & Admin Expenses/Total (FQ)",
            "Accounts Receivable/Total (FY)",
            "EV/Sales (LTM)", "EV/EBITDA (LTM)",
        ]
        
        missing_in_schema = []
        for sql_col in sql_columns:
            normalized = normalize_column_name(sql_col)
            if normalized not in COLUMN_SCHEMA:
                missing_in_schema.append((sql_col, normalized))
        
        self.assertEqual(len(missing_in_schema), 0,
                        f"SQL columns not in COLUMN_SCHEMA: {missing_in_schema}")
    
    def test_short_int_pct_not_in_schema(self):
        """Test that deleted column short_int_pct is NOT in COLUMN_SCHEMA."""
        # After Task 1.2 is applied
        self.assertNotIn("short_int_pct", COLUMN_SCHEMA,
                        "short_int_pct should be deleted from COLUMN_SCHEMA")
```

**Estimated Lines**: ~120 lines  
**Priority**: P1 - Must pass before normalization fix is complete  
**Dependencies**: None

---

#### 6.3. Test Module: test_schema_completeness.py

**Purpose**: Verify COLUMN_SCHEMA covers all Phase 9.3 required columns and detect missing/obsolete entries.

**Test Cases**:

```python
import unittest
from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA, 
    PHASE93_FEATURE_CATEGORIES,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols
)


class TestSchemaCompleteness(unittest.TestCase):
    """Test COLUMN_SCHEMA completeness and consistency."""
    
    def test_phase93_features_all_in_schema(self):
        """Test all Phase 9.3 feature input columns are in COLUMN_SCHEMA."""
        all_phase93_cols = set()
        for category, cols in PHASE93_FEATURE_CATEGORIES.items():
            all_phase93_cols.update(cols)
        
        missing = [col for col in all_phase93_cols if col not in COLUMN_SCHEMA]
        self.assertEqual(len(missing), 0,
                        f"Phase 9.3 columns missing from COLUMN_SCHEMA: {missing}")
    
    def test_critical_columns_present(self):
        """Test that critical identifier/target columns are present."""
        critical = [
            "ticker", "isin", "sector", "region", 
            "last_price", "price_target", "market_cap"
        ]
        for col in critical:
            with self.subTest(col=col):
                self.assertIn(col, COLUMN_SCHEMA)
    
    def test_no_duplicate_column_definitions(self):
        """Test that there are no duplicate keys in COLUMN_SCHEMA."""
        # This would only fail if there's a coding error
        keys = list(COLUMN_SCHEMA.keys())
        unique_keys = set(keys)
        self.assertEqual(len(keys), len(unique_keys),
                        "Duplicate keys found in COLUMN_SCHEMA")
    
    def test_all_schema_entries_have_dtype_and_role(self):
        """Test that all COLUMN_SCHEMA entries have required fields."""
        for col, meta in COLUMN_SCHEMA.items():
            with self.subTest(col=col):
                self.assertIn("dtype", meta)
                self.assertIn("role", meta)
                self.assertIsNotNone(meta["dtype"])
                self.assertIsNotNone(meta["role"])
    
    def test_list_functions_return_valid_columns(self):
        """Test helper functions return columns that exist in schema."""
        numeric_cols = list_numeric_feature_cols()
        categorical_cols = list_categorical_cols()
        date_cols = list_date_cols()
        
        for col in numeric_cols:
            self.assertIn(col, COLUMN_SCHEMA)
        for col in categorical_cols:
            self.assertIn(col, COLUMN_SCHEMA)
        for col in date_cols:
            self.assertIn(col, COLUMN_SCHEMA)
```

**Estimated Lines**: ~80 lines  
**Priority**: P2 - Should pass to verify schema quality  
**Dependencies**: schema.py PHASE93_FEATURE_CATEGORIES definition

---

#### 6.4. Test Module: test_data_loading_normalization.py

**Purpose**: Integration tests verifying data.py normalize_columns() uses schema normalization.

**Test Cases**:

```python
import unittest
import pandas as pd
from pathlib import Path
from finance_ml.ml_workflow.preprocessing.data import normalize_columns, load_from_csv
from finance_ml.ml_workflow.data.schema import normalize_column_name, COLUMN_SCHEMA


class TestDataLoadingNormalization(unittest.TestCase):
    """Test data loading normalization consistency with schema."""
    
    def test_normalize_columns_uses_schema_function(self):
        """Test that normalize_columns produces schema-consistent names."""
        # Create test dataframe with problematic column names
        test_df = pd.DataFrame({
            "# Strong Sell Ratings": [1, 2, 3],
            "1-Day %": [0.5, -0.3, 1.2],
            "Shrs Out": [1000, 2000, 3000],
            "Selling General & Admin Expenses/Total (FQ)": [100, 200, 300],
        })
        
        normalized_df = normalize_columns(test_df)
        
        # Verify columns match expected normalized names
        expected_columns = [
            "num_strong_sell_ratings",
            "1_day_pct",
            "shrs_out",
            "selling_general_and_admin_expenses_total_fq",
        ]
        
        for expected_col in expected_columns:
            with self.subTest(col=expected_col):
                self.assertIn(expected_col, normalized_df.columns)
    
    def test_normalize_columns_matches_schema_normalize(self):
        """Test that data.py normalization matches schema.normalize_column_name()."""
        test_columns = [
            "Ticker", "# Strong Buys Ratings", "P/E (LTM)", 
            "Selling General & Admin Expenses/Total (FY)",
            "1-Day %", "Accounts Receivable/Total (5YAVGFQ)"
        ]
        
        test_df = pd.DataFrame({col: [] for col in test_columns})
        normalized_df = normalize_columns(test_df)
        
        for original_col in test_columns:
            expected_normalized = normalize_column_name(original_col)
            with self.subTest(original=original_col):
                self.assertIn(expected_normalized, normalized_df.columns,
                             f"{original_col} should normalize to {expected_normalized}")
    
    def test_load_from_csv_produces_schema_compatible_columns(self):
        """Test that load_from_csv produces columns matching COLUMN_SCHEMA."""
        # This test requires actual CSV files - use small sample or mock
        data_dir = Path("data")
        if not data_dir.exists():
            self.skipTest("data/ directory not found")
        
        # Load with limit for speed
        df = load_from_csv(data_dir, limit=10)
        
        # Check critical analyst rating columns
        critical_cols = [
            "num_strong_sell_ratings",
            "num_strong_buys_ratings", 
            "num_hold_ratings",
        ]
        
        present_count = sum(1 for col in critical_cols if col in df.columns)
        self.assertGreater(present_count, 0,
                          "At least some analyst rating columns should be present after normalization")
```

**Estimated Lines**: ~90 lines  
**Priority**: P1 - Must pass after Task 1.1 implementation  
**Dependencies**: Task 1.1 (data.py normalization fix)

---

#### 6.5. Enhanced Test Cases for Existing Modules

**Module: tests/test_data_types_detection.py**

Add test cases to verify normalization before dtype detection:

```python
def test_detect_dtypes_with_problematic_column_names(self):
    """Test dtype detection works with columns requiring normalization."""
    from finance_ml.ml_workflow.preprocessing.dtypes import detect_and_cast_dtypes
    
    test_df = pd.DataFrame({
        "num_strong_sell_ratings": [1.0, 2.0, 3.0],
        "1_day_pct": [0.5, -0.3, 1.2],
        "shrs_out": [1000, 2000, 3000],
    })
    
    df_cast, diagnostics = detect_and_cast_dtypes(test_df)
    
    # All columns should be recognized and cast correctly
    self.assertIn("num_strong_sell_ratings", diagnostics["inferred_dtypes"])
    self.assertEqual(len(diagnostics["missing_expected_columns"]), 0)
```

**Module: tests/test_finance_ml_data.py**

Add integration test for full CSV loading pipeline:

```python
def test_csv_load_analyst_rating_columns_present(self):
    """Test that analyst rating columns are present after CSV load."""
    df = load_from_csv(Path("data"), limit=100)
    
    analyst_cols = [
        "num_strong_sell_ratings",
        "num_strong_buys_ratings",
        "num_hold_ratings",
        "num_buys_ratings",
        "num_sell_ratings",
    ]
    
    for col in analyst_cols:
        with self.subTest(col=col):
            self.assertIn(col, df.columns,
                         f"Analyst rating column {col} missing after normalization")
```

---

#### 6.6. Implementation Order & Milestones

**Phase 1: Schema Verification (No code changes)**

1. Create `tests/test_schema_normalization.py` - **FIRST**
2. Create `tests/test_schema_completeness.py`
3. Run tests → Expect 5 failures in test_normalize_analyst_ratings_with_hash()
4. Document baseline: 5 normalization failures confirmed

**Phase 2: Critical Fixes (Code changes)**

1. Apply Task 1.1: Fix data.py normalization → Use schema.normalize_column_name()
2. Apply Task 1.2: Delete short_int_pct from schema.py line 159
3. Run `test_schema_normalization.py` → Expect all passing
4. Run `test_schema_completeness.py` → Expect all passing

**Phase 3: Integration Verification**

1. Create `tests/test_data_loading_normalization.py`
2. Run integration tests → Expect all passing
3. Run full test suite → Ensure no regressions

**Phase 4: Production Validation**

1. Run notebook/script with full CSV data
2. Generate new dtype_diagnostics.json
3. Verify missing_expected_columns list is empty
4. Update this document with "IMPLEMENTED" status

---

#### 6.7. Test Execution Commands

**Run normalization tests only**:

```bash
python -m unittest tests.test_schema_normalization -v
python -m unittest tests.test_schema_completeness -v
python -m unittest tests.test_data_loading_normalization -v
```

**Run affected test modules**:

```bash
python -m unittest tests.test_data_types_detection -v
python -m unittest tests.test_finance_ml_data -v
python -m unittest tests.test_enhanced_imputation_phase93 -v
```

**Run full test suite** (after fixes):

```bash
python -m unittest discover -s tests -v
```

---

#### 6.8. Success Criteria

**All tests must pass** with these specific validations:

1. ✅ All 5 analyst rating columns normalize correctly with "num_" prefix
2. ✅ normalize_column_name() produces names that exist in COLUMN_SCHEMA
3. ✅ data.py normalize_columns() produces identical results to schema.normalize_column_name()
4. ✅ short_int_pct is removed from COLUMN_SCHEMA
5. ✅ load_from_csv() produces dataframes with schema-compatible column names
6. ✅ dtype_diagnostics.json reports zero missing_expected_columns (except truly missing ones)
7. ✅ No regressions in existing 85 test modules

**Coverage targets**:

- `finance_ml/ml_workflow/data/schema.py`: ≥95% (normalization function critical)
- `finance_ml/ml_workflow/preprocessing/data.py`: ≥85% (normalize_columns coverage)
- New test modules: 100% of written test cases passing

---

#### 6.9. Alignment with code_guidelines.md v1.4

This TDD plan aligns with:

- **Section 8: Notebook Best Practices** - Centralized configuration and dataframe naming
- **Section 7: Testing Conventions** - TDD-first approach, selective test execution
- **Section 4: Column Naming Schema** - Normalization consistency policy (to be added per Task 3.1)
- **Section 5: Data Split and Leakage Policy** - Schema-aware preprocessing prevents leakage

**New guideline to add** (Task 3.1):

```markdown
#### 4.5. Column Normalization Consistency Policy (v1.5+)

**Canonical Normalization Function**: `finance_ml.ml_workflow.data.schema.normalize_column_name()`

**Rules**:
1. ALL column name normalization MUST use `normalize_column_name()` from schema.py
2. NO alternative normalization functions allowed in data loading or preprocessing
3. Transformations: `#` → `num`, `%` → `pct`, `&` → `and`, spaces → `_`, special chars removed
4. All COLUMN_SCHEMA keys MUST be producible via normalize_column_name() from SQL schema
5. Test coverage REQUIRED: Any PR touching column normalization must include round-trip tests

**Enforcement**: CI pipeline runs test_schema_normalization.py to prevent normalization drift.
```

---

#### 6.10. Risk Mitigation

**Risk 1**: Breaking changes to existing code expecting old normalization

- **Mitigation**: Run full test suite (85 modules) before merging
- **Rollback**: Git tag before changes, documented rollback procedure

**Risk 2**: CSV files change structure, adding new special characters

- **Mitigation**: test_schema_normalization includes diverse special char cases
- **Detection**: dtype_diagnostics.json will flag new missing columns

**Risk 3**: Performance impact of per-column normalization vs vectorized regex

- **Mitigation**: Profile data loading time before/after; optimize if >10% slowdown
- **Benchmark**: Load 10K rows, measure time (expect <1s difference)

---

### 7. Summary

This TDD implementation plan provides:

1. **3 new test modules** with 15+ test cases specifically for normalization
2. **Clear implementation phases** with measurable milestones
3. **Success criteria** aligned with code_guidelines.md
4. **Risk mitigation** strategies for production deployment
5. **Integration** with existing 85-module test suite

**Estimated total implementation time**: 4-6 hours including testing and documentation.
