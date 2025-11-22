"""
Analyze preprocessed_stocks_metadata.json and generate correct Phase 9.3 benchmarking metrics.
"""

import json
from pathlib import Path

# Load metadata
metadata_path = Path("outputs/catalog/preprocessed_stocks_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("=" * 80)
print("PREPROCESSED STOCKS METADATA ANALYSIS")
print("=" * 80)
print(f"Dataset: {metadata['name']}")
print(f"Version: {metadata['version']}")
print(f"Shape: {metadata['shape']} (rows x columns)")
print(f"Total columns: {len(metadata['columns'])}")
print(f"Quality score: {metadata['quality_score']:.4f}")
print()

# Load Phase 9.3 categories registry
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    categorize_dataframe_columns,
)
import pandas as pd

# Create a dummy DataFrame with just column names to use categorization
df_cols = pd.DataFrame(columns=metadata["columns"])

print("=" * 80)
print("PHASE 9.3 CATEGORY COVERAGE ANALYSIS")
print("=" * 80)

# Categorize columns
categorized = {}
for category, expected_features in PHASE93_FEATURE_CATEGORIES.items():
    available = [feat for feat in expected_features if feat in metadata["columns"]]
    categorized[category] = available

    coverage_pct = (len(available) / len(expected_features) * 100) if expected_features else 0
    print(f"\n{category}:")
    print(f"  Expected features: {len(expected_features)}")
    print(f"  Available in data: {len(available)}")
    print(f"  Coverage: {coverage_pct:.1f}%")

    if available:
        print(f"  Available features: {', '.join(available[:5])}")
        if len(available) > 5:
            print(f"    ... and {len(available) - 5} more")
    else:
        print(f"  ⚠️  No features from this category found in preprocessed data")

print()
print("=" * 80)
print("ACTUAL METRICS AVAILABLE BY CATEGORY")
print("=" * 80)

# Find which actual columns belong to which categories
actual_categorized = {cat: [] for cat in PHASE93_FEATURE_CATEGORIES.keys()}
uncategorized = []

for col in metadata["columns"]:
    categorized_flag = False
    for category, expected_features in PHASE93_FEATURE_CATEGORIES.items():
        if col in expected_features:
            actual_categorized[category].append(col)
            categorized_flag = True
            break
    if not categorized_flag:
        uncategorized.append(col)

total_categorized = sum(len(v) for v in actual_categorized.values())
print(f"\nTotal columns: {len(metadata['columns'])}")
print(f"Categorized as Phase 9.3 features: {total_categorized}")
print(f"Uncategorized (raw/auxiliary columns): {len(uncategorized)}")
print()

for category, cols in actual_categorized.items():
    if cols:
        print(f"\n{category}: {len(cols)} metrics")
        print(f"  {', '.join(cols)}")

print()
print("=" * 80)
print("CORRECTED BENCHMARKING METRICS")
print("=" * 80)
print("\nMetrics Availability by Category:")
for category, cols in actual_categorized.items():
    total_expected = len(PHASE93_FEATURE_CATEGORIES[category])
    coverage_pct = (len(cols) / total_expected * 100) if total_expected else 0
    print(f"  {category}: {len(cols)}/{total_expected} metrics ({coverage_pct:.0f}% coverage)")

print()
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("\nThe benchmarking section should:")
print("1. Use actual column names from preprocessed_stocks_metadata.json")
print("2. Report accurate coverage percentages based on available features")
print("3. Skip visualization for categories with 0 available metrics")
print("4. Focus analysis on categories with >0 metrics (Valuation, Profitability, etc.)")
