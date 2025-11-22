# Fix Summary: Notebook Import and Configuration Issues

**Date:** 2025-11-22  
**Issue:** Malformed import statement and undefined output_dir variable  
**Status:** ✅ RESOLVED

---

## Issues Identified

### 1. Malformed Import Statement (Line 187)

**Problem:**
The Phase 9.4-9.8 import statement was concatenated without proper line breaks, making it invalid Python syntax:

```python
# Phase 9.4-9.8: Advanced Evaluation and Governancefrom finance_ml.ml_workflow.evaluation import (    # Phase 9.4 - Uncertainty Quantification    build_quantile_diagnostics,    plot_interval_coverage,    plot_reliability_diagram, ...
```

This caused a syntax error preventing the notebook from executing.

**Root Cause:**
Import block was improperly formatted during previous integration, missing newlines between comment and import
statement.

### 2. Undefined output_dir Variable (Line 4768)

**Problem:**
Phase 9.4 cells used `output_dir` variable without defining it:

```python
uncertainty_dir = output_dir / "uncertainty"
```

This caused `NameError: name 'output_dir' is not defined` at runtime.

**Root Cause:**
Cells were copied from NOTEBOOK_INTEGRATION_GUIDE.md without the recommended configuration setup using
`config.output_dir`.

### 3. Incorrect Section Header Format (Line 4741)

**Problem:**
Markdown header used "### Phase 9.4:" instead of "## Section 9.4:", causing test failures because the test suite
expects "Section 9.4:" format.

---

## Fixes Applied

### Fix 1: Reformatted Import Statement (Lines 187-211)

**Before:**

```python
# Phase 9.4-9.8: Advanced Evaluation and Governancefrom finance_ml.ml_workflow.evaluation import (    # Phase 9.4 - Uncertainty Quantification    build_quantile_diagnostics, ...
```

**After:**

```python
# Phase 9.4-9.8: Advanced Evaluation and Governance
from finance_ml.ml_workflow.evaluation import (
    # Phase 9.4 - Uncertainty Quantification
    build_quantile_diagnostics,
    plot_interval_coverage,
    plot_reliability_diagram,
    # Phase 9.5 - Safety Rails
    summarize_winsorization_effects,
    track_constraint_violations,
    safety_rails_sensitivity_app,
    # Phase 9.6 - Data Splits & Leakage
    compute_fold_overlap,
    summarize_grouped_cv_balance,
    time_leakage_checks,
    # Phase 9.7 - Sector Bias Calibration
    estimate_sector_bias,
    plot_metrics_by_sector_time,
    create_sector_bias_dashboard,
    # Phase 9.8 - Stacking & Governance
    compute_stacking_contributions,
    meta_error_maps,
    generate_model_card,
    build_lineage_json,
    )
```

**Result:** ✅ Valid Python import statement with proper formatting

### Fix 2: Added output_dir Configuration (Lines 4764-4770)

**Before:**

```python
from pathlib import Path
import pandas as pd

# Setup paths
uncertainty_dir = output_dir / "uncertainty"  # ❌ output_dir not defined
```

**After:**

```python
from pathlib import Path
import pandas as pd
import json

# Setup paths - use config.output_dir from FinanceMLConfig
output_dir = config.output_dir  # ✅ Properly defined from config
uncertainty_dir = output_dir / "uncertainty"
```

**Approach:** Best-practice solution using `config.output_dir` from FinanceMLConfig system (defined at line 175) rather
than hard-coding `Path("outputs")`.

**Result:** ✅ output_dir properly defined and accessible throughout Phase 9.4 cells

### Fix 3: Corrected Section Header (Line 4741)

**Before:**

```markdown
### Phase 9.4: Uncertainty Quantification & Conformal Calibration
```

**After:**

```markdown
## Section 9.4: Uncertainty Quantification & Conformal Calibration
```

**Result:** ✅ Header now matches test expectations and uses proper heading level

---

## Validation Results

### Test Suite Status

Ran: `python -m unittest tests.test_notebook_phase94_98_structure -v`

**Phase 9.4 Tests (4/4 PASS):**

- ✅ `test_phase94_imports_present` - All required imports present
- ✅ `test_phase94_section_exists` - Section marker found
- ✅ `test_phase94_has_minimum_cells` - Minimum 4 cells present
- ✅ `test_notebook_exists` - Valid JSON structure

**Phase 9.5-9.8 Tests (expected failures - not yet implemented):**

- ❌ Phase 9.5-9.8 sections do not exist in notebook (separate implementation task)

### Notebook Validation

Ran: `python validate_notebook.py`

```
✅ Notebook is valid JSON
✅ Total cells: 120
✅ Notebook format: 4.5
✅ Phase 9.4 cells: 4
✅ Phase 9.5 cells: 0
✅ Phase 9.6 cells: 0
✅ Phase 9.7 cells: 0
✅ Phase 9.8 cells: 0
✅ Total Phase 9.4-9.8 cells: 4
✅ Notebook validation successful!
```

---

## Files Modified

1. **ml_finance_model_main.ipynb**
    - Line 187-211: Reformatted import statement
    - Line 4741: Updated section header format
    - Line 4764-4770: Added output_dir configuration and json import

---

## Recommendations for Future Work

### Phase 9.5-9.8 Full Implementation

The following sections are referenced in test suite but not yet implemented in the notebook:

1. **Phase 9.5: Outlier Safety Rails & Non-Negative Constraints**
    - 5 cells needed
    - Outputs to: `outputs/safety_rails/`
    - See: NOTEBOOK_INTEGRATION_GUIDE.md lines 179-301

2. **Phase 9.6: Data Split and Leakage Policy Validation**
    - 5 cells needed
    - Outputs to: `outputs/splits/`
    - See: NOTEBOOK_INTEGRATION_GUIDE.md lines 305-413

3. **Phase 9.7: Sector Bias Calibration & Metrics Persistence**
    - 5 cells needed
    - Outputs to: `outputs/calibration/`
    - See: NOTEBOOK_INTEGRATION_GUIDE.md lines 417-522

4. **Phase 9.8: Stacking Ensemble Diagnostics & Model Governance**
    - 6 cells needed
    - Outputs to: `outputs/governance/`
    - See: NOTEBOOK_INTEGRATION_GUIDE.md lines 526-709

**Implementation Steps:**

1. Follow NOTEBOOK_INTEGRATION_GUIDE.md cell-by-cell structure
2. Ensure each section has proper markdown header: "## Section 9.X:"
3. Add `output_dir = config.output_dir` at the start of first code cell in each section
4. Add required imports (`json`, `pathlib.Path`, etc.) as needed
5. Use cell markers: `[PHASE 9.5]`, `[PHASE 9.6]`, `[PHASE 9.7]`, `[PHASE 9.8]`
6. Validate with: `python -m unittest tests.test_notebook_phase94_98_structure -v`

---

## Summary

**Issues Resolved:**

- ✅ Malformed import statement fixed with proper Python formatting
- ✅ output_dir properly configured using best-practice config.output_dir approach
- ✅ Section header corrected to match test expectations
- ✅ Phase 9.4 implementation complete and passing all tests
- ✅ Notebook JSON structure validated

**Remaining Work:**

- ⚠️ Phase 9.5-9.8 sections need to be added (separate implementation task per notebook_restructuring_plan.md)

**Documentation References:**

- `docs/NOTEBOOK_INTEGRATION_GUIDE.md` - Cell-by-cell implementation guide
- `docs/improvement_plan/notebook_restructuring_plan.md` - Phase 9.4-9.8 specifications
- `docs/code_guidelines.md` v1.2+ - Configuration and best practices
- `tests/test_notebook_phase94_98_structure.py` - Validation test suite
