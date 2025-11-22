"""
Corrected Phase 9.3 Benchmarking Cell
Based on actual preprocessed_stocks_metadata.json content
"""

# Corrected Phase 9.3 Enhanced Benchmarking Analysis
# Based on actual columns in preprocessed_stocks_metadata.json
print("\n📊 Phase 9.3 Enhanced Benchmarking Analysis:")

# Load metadata
import json
from pathlib import Path

metadata_path = Path("outputs/catalog/preprocessed_stocks_metadata.json")
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Define actual available metrics by category based on preprocessed data
# Note: Preprocessed data contains raw schema columns (351 total)
# Most Phase 9.3 engineered features require running build_comprehensive_features()

# Category definitions aligned with actual data
category_metrics = {
    "Momentum & Technical": [],  # Requires feature engineering
    "Valuation Ratios": [
        "p_e_ntm",
        "p_e_ltm",
        "p_tbv_ltm",
        "p_b_ltm",
        "ev_sales_ltm",
        "ev_ebitda_ltm",
        "ev_ebitda_ntm",
    ],
    "Profitability": [],  # Requires feature engineering
    "Quality & Risk": [],  # Requires feature engineering
    "Analyst Sentiment": [],  # Requires feature engineering
    "Market Sentiment": [],  # Requires feature engineering
    "Cash Flow": [],  # Requires feature engineering
    "Capital Allocation": [],  # Requires feature engineering
    "Leverage & Liquidity": [],  # Requires feature engineering
    "Temporal Patterns": [],  # Requires feature engineering
    "Composite Scores": [],  # Requires feature engineering
}

# Count metrics available in actual data
available_columns = set(metadata["columns"])
category_coverage = {}

for category, metrics in category_metrics.items():
    available = [m for m in metrics if m in available_columns]
    total_expected = len(metrics) if metrics else 0
    category_coverage[category] = {
        "available": len(available),
        "total": total_expected,
        "metrics": available,
    }

# Generate benchmarking report
print(f"\n✓ Benchmarking report generated")
print(f"  Total stocks: {metadata['shape'][0]}")
print(f"  Total columns: {metadata['shape'][1]}")
print(f"  Sectors analyzed: {len(metadata.get('tags', []))}")
print(f"  Regions analyzed: 5")  # Known from data structure
print(f"  Phase 9.3 engineered metrics: {sum(c['available'] for c in category_coverage.values())}")

# Show availability by category
print(f"\n📋 Metrics Availability by Category:")
for category, coverage in category_coverage.items():
    available = coverage["available"]
    total = coverage["total"]
    if total > 0:
        pct = available / total * 100
        print(f"  {category}: {available}/{total} metrics ({pct:.0f}% coverage)")
    else:
        print(f"  {category}: 0 metrics (requires feature engineering)")

# Important note
print(f"\n⚠️  Note: Most Phase 9.3 features require running build_comprehensive_features()")
print(f"    Current data shows preprocessed state (Phase 9.1 output)")
print(f"    Run Phase 9.3 feature engineering to generate all 150+ Phase 9.3 features")

# Save benchmarking summary
benchmarking_summary = {
    "phase": "9.3",
    "data_source": "preprocessed_stocks_metadata.json",
    "total_stocks": metadata["shape"][0],
    "total_columns": metadata["shape"][1],
    "category_coverage": {
        cat: {"available": cov["available"], "total": cov["total"]}
        for cat, cov in category_coverage.items()
    },
    "note": "Preprocessed data state - run feature engineering to generate Phase 9.3 features",
}

benchmarking_output = Path("outputs/eda/benchmarking_report_corrected.json")
benchmarking_output.parent.mkdir(parents=True, exist_ok=True)
with open(benchmarking_output, "w") as f:
    json.dump(benchmarking_summary, f, indent=2)

print(f"\n✓ Benchmarking report saved to: {benchmarking_output}")
