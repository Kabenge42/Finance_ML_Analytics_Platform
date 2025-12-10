"""Quick debug script for schema fallback classification."""

from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
    classify_columns_with_schema_fallback,
)
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA

# Test with all schema columns
cols = list(COLUMN_SCHEMA.keys())
print(f"Total columns in COLUMN_SCHEMA: {len(cols)}")

# Test full pipeline
result = classify_columns(cols)
print("\nFull classification result:")
for cat, items in result.items():
    print(f"  {cat}: {len(items)}")

# Test schema fallback on OTHER columns directly
other_cols = list(result["other"])[:20]  # Sample first 20
print(f"\nSample of {len(other_cols)} OTHER columns:")
for col in other_cols[:10]:
    print(f"  - {col}")

# Test schema fallback directly
schema_result = classify_columns_with_schema_fallback(other_cols)
print(f"\nSchema fallback on OTHER columns:")
for col, cat in list(schema_result.items())[:10]:
    schema_info = COLUMN_SCHEMA.get(col.lower(), {})
    print(f"  {col} -> {cat} (dtype: {schema_info.get('dtype', 'N/A')})")
