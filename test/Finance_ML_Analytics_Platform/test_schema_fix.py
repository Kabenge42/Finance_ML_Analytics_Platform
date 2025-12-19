"""Test script to verify Phase 9.3 Schema 1.3 columns are in COLUMN_SCHEMA."""

from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA, PHASE93_FEATURE_INPUTS
from finance_ml.ml_workflow.preprocessing import detect_and_cast_dtypes
import pandas as pd

print("=" * 70)
print("Testing Phase 9.3 Schema 1.3 Fix")
print("=" * 70)

# The 21 columns that were causing warnings
phase93_cols = [
    "ev_sales_1fyltm",
    "ev_sales_2fyltm",
    "ev_sales_3fyltm",
    "ev_sales_3yavgltm",
    "ev_sales_1fqltm",
    "ev_sales_2fqltm",
    "ev_sales_3fqltm",
    "ev_sales_4fqltm",
    "ev_ebitda_1fyltm",
    "ev_ebitda_1fqltm",
    "ev_ebitda_3yavgltm",
    "p_e_2fyltm",
    "p_e_3fyltm",
    "p_e_3yavgltm",
    "p_e_1fqltm",
    "p_e_2fqltm",
    "p_e_3fqltm",
    "p_e_0fqqoqltm",
    "p_e_0fyyoyltm",
    "p_e_1fyyoyltm",
    "p_e_0fqyoyltm",
]

# Test 1: Check COLUMN_SCHEMA
print("\n1. Checking COLUMN_SCHEMA...")
missing_in_schema = [c for c in phase93_cols if c not in COLUMN_SCHEMA]
if missing_in_schema:
    print(f"   [FAIL] {len(missing_in_schema)} columns missing from COLUMN_SCHEMA")
    for col in missing_in_schema:
        print(f"      - {col}")
else:
    print(f"   [PASS] All 21 Phase 9.3 columns found in COLUMN_SCHEMA")

# Test 2: Check PHASE93_FEATURE_INPUTS
print("\n2. Checking PHASE93_FEATURE_INPUTS...")
valuation_cols = PHASE93_FEATURE_INPUTS.get("valuation", [])
missing_in_phase93 = [c for c in phase93_cols if c not in valuation_cols]
if missing_in_phase93:
    print(f"   [FAIL] {len(missing_in_phase93)} columns missing from PHASE93_FEATURE_INPUTS")
    for col in missing_in_phase93:
        print(f"      - {col}")
else:
    print(f"   [PASS] All 21 Phase 9.3 columns found in PHASE93_FEATURE_INPUTS")

# Test 3: Test detect_and_cast_dtypes with Phase 9.3 columns
print("\n3. Testing detect_and_cast_dtypes with Phase 9.3 columns...")
test_data = {col: [1.0, 2.0, 3.0] for col in phase93_cols}
test_data["ticker"] = ["AAPL", "MSFT", "GOOGL"]
test_df = pd.DataFrame(test_data)

df_cast, diagnostics = detect_and_cast_dtypes(test_df)

unknown_cols = diagnostics.get("unknown_columns", [])
phase93_unknown = [c for c in phase93_cols if c in unknown_cols]
if phase93_unknown:
    print(f"   [FAIL] {len(phase93_unknown)} Phase 9.3 columns reported as unknown")
    for col in phase93_unknown:
        print(f"      - {col}")
else:
    print(f"   [PASS] No Phase 9.3 columns reported as unknown")

# Summary
print("\n" + "=" * 70)
print("Summary:")
print(f"  Total COLUMN_SCHEMA entries: {len(COLUMN_SCHEMA)}")
print(f"  Valuation columns in PHASE93_FEATURE_INPUTS: {len(valuation_cols)}")
print(f"  Phase 9.3 columns in COLUMN_SCHEMA: {21 - len(missing_in_schema)}/21")
print(f"  Phase 9.3 columns in PHASE93_FEATURE_INPUTS: {21 - len(missing_in_phase93)}/21")
print(f"  Phase 9.3 columns passing dtype detection: {21 - len(phase93_unknown)}/21")
print("=" * 70)

if not missing_in_schema and not missing_in_phase93 and not phase93_unknown:
    print("\n[SUCCESS] ALL TESTS PASSED - Schema fix is complete!")
    exit(0)
else:
    print("\n[FAILED] SOME TESTS FAILED - Schema fix incomplete")
    exit(1)
