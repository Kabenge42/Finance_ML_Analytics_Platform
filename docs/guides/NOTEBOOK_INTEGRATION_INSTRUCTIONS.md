# Notebook Integration Instructions - Section 9.1.8

## Overview

This document provides instructions for manually integrating the **Section 9.1.8: Enhanced 6-step Imputation Strategy**
into `ml_finance_model_main.ipynb`.

## Status

✅ **Code Implementation**: Complete (7 functions in `finance_ml/advanced_preprocessing.py`)  
✅ **Testing**: Complete (21/21 tests passing in `tests/test_enhanced_imputation.py`)  
✅ **Documentation**: Complete (README.md, IMPROVEMENT_PLAN.md, guidelines.md)  
✅ **Test Organization**: Complete (guidelines.md section 3D)  
📝 **Notebook Integration**: Manual task (instructions below)

---

## Integration Point

**Location**: After Section 9.1.6.5 (line 886), before Phase 9.2 (line 887)

**Current structure**:

```
Line 866: #%% md
          ### 9.1.6.5 Summary of Phase 9.1 Enhancements
Line 867: #%%
          [Code cell with enhancement summary]
Line 886: print("=" * 80)
Line 887: #%% md
          ## Phase 9.2 — Advanced Exploratory Data Analysis
```

**After integration**:

```
Line 866-886: [Existing Section 9.1.6.5]
[NEW] Line 887: #%% md
              ### 9.1.8 Enhanced 6-step Imputation Strategy (Phase 9.1 Complete)
[NEW] Line 888: #%%
              [Code cell with 6-step imputation demonstration]
Line 889+: #%% md
           ## Phase 9.2 — Advanced Exploratory Data Analysis
```

---

## Integration Steps

### Option 1: Manual Editing (Recommended for Safety)

1. **Open the notebook** in Jupyter Notebook or JupyterLab:
   ```powershell
   jupyter notebook ml_finance_model_main.ipynb
   ```

2. **Navigate** to Section 9.1.6.5 (after the "Next Steps: Phase 9.2..." print statement)

3. **Insert a new Markdown cell** with this content:
   ```markdown
   ### 9.1.8 Enhanced 6-step Imputation Strategy (Phase 9.1 Complete)

   Complete imputation pipeline ensuring zero missing values:
   1. **Step 1: Zero Imputation** (48 columns) - Exceptional events (impairments, restructuring)
   2. **Step 2: KNN Imputation** (148 columns) - Sector-aware financial metrics
   3. **Step 3: Price Imputation** (5 columns) - Price targets from last_price
   4. **Step 4: Median Imputation** (remaining) - Fallback for all other numerical columns
   ```

4. **Insert a new Code cell** below the markdown cell and copy the content from `notebook_section_9_1_8.txt` (lines
   11-129)

5. **Save the notebook** and verify the cell ordering

### Option 2: Using Jupyter nbformat (Programmatic)

If you prefer programmatic insertion, use this Python script:

```python
import json
import nbformat
from nbformat.v4 import new_markdown_cell, new_code_cell

# Load notebook
with open('ml_finance_model_main.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Find Phase 9.2 cell index
phase92_idx = next(
    i for i, cell in enumerate(nb.cells) 
    if cell.cell_type == 'markdown' and 'Phase 9.2' in cell.source
)

# Create new cells
markdown_content = """### 9.1.8 Enhanced 6-step Imputation Strategy (Phase 9.1 Complete)

Complete imputation pipeline ensuring zero missing values:
1. **Step 1: Zero Imputation** (48 columns) - Exceptional events (impairments, restructuring)
2. **Step 2: KNN Imputation** (148 columns) - Sector-aware financial metrics
3. **Step 3: Price Imputation** (5 columns) - Price targets from last_price
4. **Step 4: Median Imputation** (remaining) - Fallback for all other numerical columns"""

# Read code content from prepared file
with open('notebook_section_9_1_8.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    code_content = content.split('---CODE_CELL---\n')[1].strip()

markdown_cell = new_markdown_cell(markdown_content)
code_cell = new_code_cell(code_content)

# Insert cells before Phase 9.2
nb.cells.insert(phase92_idx, code_cell)
nb.cells.insert(phase92_idx, markdown_cell)

# Save notebook
with open('ml_finance_model_main.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"✓ Section 9.1.8 inserted at index {phase92_idx}")
print(f"✓ Total cells: {len(nb.cells)}")
```

---

## Verification Steps

After integration:

1. **Run the notebook** through Section 9.1.8 to verify:
    - All imports resolve correctly
    - The 6-step imputation executes without errors
    - Visualizations render correctly
    - Output shows zero missing values after imputation

2. **Check outputs**:
    - Missing values before/after statistics
    - 4-panel visualization saved to `outputs/phase_9_1_4step_imputation.png`
    - Success message with zero missing values confirmation

3. **Expected output**:
   ```
   9.1.8 ENHANCED 6-step IMPUTATION STRATEGY
   ================================================================================
   
   6-step Imputation Strategy:
     Step 1: Zero Imputation      → 48 columns (exceptional events)
     Step 2: KNN Imputation       → 148 columns (financial metrics)
     Step 3: Price Imputation     → 5 columns (price targets)
     Step 4: Median Imputation    → Remaining numerical columns
   
   Step 1 (Zero): 48 defined, X available
   Step 2 (KNN): 148 defined, Y available
   Step 3 (Price): 5 defined, Z available
   
   Applying 6-step imputation strategy...
   
   Imputation Results:
     Missing values before: N,NNN
     Missing values after:  0
     Reduction:            N,NNN (100.0%)
   
   ✓ Enhanced 6-step imputation strategy completed successfully!
     ✓ Zero missing values in output: True
     ✓ Visualization saved to: outputs\phase_9_1_4step_imputation.png
   ```

---

## Related Files

- **Implementation**: `finance_ml/advanced_preprocessing.py` (lines 594-1165)
- **Tests**: `tests/test_enhanced_imputation.py` (21 tests, all passing)
- **Documentation**:
    - `docs/improvement_plan/IMPROVEMENT_PLAN.md` (Phase 9.1 section updated)
    - `README.md` (advanced_preprocessing section updated)
    - `.junie/guidelines.md` (test execution strategies added)
- **Prepared content**: `notebook_section_9_1_8.txt`

---

## Function Reference

The section uses these functions from `finance_ml.advanced_preprocessing`:

1. `get_zero_imputation_columns()` - Returns 48 column names for zero imputation
2. `get_knn_imputation_columns()` - Returns 148 column names for KNN imputation
3. `apply_enhanced_imputation_strategy_4step()` - Master function executing all 4 steps

All functions are fully tested and integrated into the `finance_ml` package.

---

## Troubleshooting

**Issue**: Imports fail  
**Solution**: Ensure `finance_ml` package is installed: `pip install -e .`

**Issue**: Sector column not found  
**Solution**: The function will fall back to global KNN imputation (expected behavior)

**Issue**: Visualization doesn't render  
**Solution**: Ensure `outputs_dir` variable is defined in earlier cells

**Issue**: Still have missing values after imputation  
**Solution**: Check warning messages for columns that couldn't be imputed (e.g., all-NaN columns)

---

## Contact

For questions or issues with integration, refer to:

- Implementation guide: `docs/improvement_plan/Implement__9.1_Loading_and_Preprocessing_Enhanced.md`
- Test suite: Run `python -m unittest tests.test_enhanced_imputation -v`
