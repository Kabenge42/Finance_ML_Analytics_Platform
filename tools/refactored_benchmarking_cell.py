# Refactored Phase 9.3 Enhanced Benchmarking Analysis
# Analyzes actual engineered features in all_stocks_features DataFrame
print("\n📊 Phase 9.3 Enhanced Benchmarking Analysis:")
print("=" * 80)

# Import Phase 9.3 category detection modules
from finance_ml.ml_workflow.eda.phase93_categories import (
    categorize_dataframe_columns,
    get_phase93_coverage_stats,
    get_category_description,
    PHASE93_FEATURE_CATEGORIES,
)

# Validate that feature engineering has completed
if "all_stocks_features" not in dir() or all_stocks_features is None:
    print("⚠️  ERROR: all_stocks_features not found!")
    print("   Please run Phase 9.3 feature engineering cells first.")
else:
    print(f"\n✓ Analyzing engineered features DataFrame")
    print(f"  Total stocks: {all_stocks_features.shape[0]}")
    print(f"  Total columns: {all_stocks_features.shape[1]}")

    # Categorize features by Phase 9.3 families
    categorized = categorize_dataframe_columns(all_stocks_features)
    coverage_stats = get_phase93_coverage_stats(all_stocks_features)

    # Calculate total Phase 9.3 features present
    total_phase93_features = sum(coverage_stats.values())

    # Get expected feature counts per category
    expected_counts = {cat: len(features) for cat, features in PHASE93_FEATURE_CATEGORIES.items()}
    total_expected = sum(expected_counts.values())

    print(
        f"  Phase 9.3 engineered features present: {total_phase93_features}/{total_expected} ({total_phase93_features/total_expected*100:.1f}%)"
    )

    # Sector/region distribution
    if "sector" in all_stocks_features.columns:
        sectors = all_stocks_features["sector"].nunique()
        print(f"  Sectors analyzed: {sectors}")
    if "region" in all_stocks_features.columns:
        regions = all_stocks_features["region"].nunique()
        print(f"  Regions analyzed: {regions}")

    # Show availability by category
    print(f"\n📋 Phase 9.3 Feature Coverage by Category:")
    print("=" * 80)

    for category in sorted(PHASE93_FEATURE_CATEGORIES.keys()):
        present = coverage_stats.get(category, 0)
        expected = expected_counts[category]

        if present > 0:
            pct = (present / expected * 100) if expected > 0 else 0
            print(f"  ✓ {category}: {present}/{expected} features ({pct:.1f}% coverage)")

            # Show sample features for this category
            if category in categorized:
                sample_features = categorized[category][:3]
                for feat in sample_features:
                    non_null = all_stocks_features[feat].notna().sum()
                    print(f"      • {feat}: {non_null}/{len(all_stocks_features)} non-null")
        else:
            print(f"  ✗ {category}: 0/{expected} features (not yet engineered)")

    # Generate summary report
    print(f"\n📊 Benchmarking Summary:")
    print("=" * 80)

    categories_with_features = len([c for c in coverage_stats.values() if c > 0])
    categories_total = len(PHASE93_FEATURE_CATEGORIES)

    print(f"  Categories with features: {categories_with_features}/{categories_total}")
    print(f"  Total Phase 9.3 features: {total_phase93_features}")
    print(f"  Overall coverage: {total_phase93_features/total_expected*100:.1f}%")

    # Export benchmarking report
    benchmarking_summary = {
        "phase": "9.3",
        "data_source": "all_stocks_features DataFrame (post-feature-engineering)",
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_stocks": int(all_stocks_features.shape[0]),
        "total_columns": int(all_stocks_features.shape[1]),
        "phase93_features_present": int(total_phase93_features),
        "phase93_features_expected": int(total_expected),
        "coverage_percentage": float(total_phase93_features / total_expected * 100),
        "category_coverage": {
            cat: {
                "present": int(coverage_stats.get(cat, 0)),
                "expected": int(expected_counts[cat]),
                "coverage_pct": float(
                    (coverage_stats.get(cat, 0) / expected_counts[cat] * 100)
                    if expected_counts[cat] > 0
                    else 0
                ),
            }
            for cat in PHASE93_FEATURE_CATEGORIES.keys()
        },
        "categories_with_features": int(categories_with_features),
        "note": "Analysis performed on engineered features DataFrame after Phase 9.3 completion",
    }

    # Save report
    from pathlib import Path

    benchmarking_output = Path("outputs/eda/phase93_benchmarking_post_engineering.json")
    benchmarking_output.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(benchmarking_output, "w") as f:
        json.dump(benchmarking_summary, f, indent=2)

    print(f"\n✓ Benchmarking report saved to: {benchmarking_output}")
    print("\n" + "=" * 80)
