"""
Audit Phase 9.3 feature generators to extract actual feature names.

This script parses advanced.py to identify the actual features produced by each
generator function and compares them with the registry in phase93_categories.py.
"""

import json
import re
from pathlib import Path
from typing import Set

# Category to generator function mapping
CATEGORY_GENERATORS = {
    "Momentum & Technical": [
        "engineer_momentum_features",
        "engineer_technical_analysis_features",
    ],
    "Valuation Ratios": [
        "engineer_valuation_ratios",
        "engineer_valuation_timeseries_features",
    ],
    "Profitability": [
        "engineer_profitability_ratios",
        "engineer_margin_trends",
    ],
    "Quality & Risk": [
        "engineer_accounting_quality_features",
        "engineer_financial_distress_features",
    ],
    "Cash Flow": [
        "engineer_cash_flow_quality_features",
    ],
    "Capital Allocation": [
        "engineer_capital_allocation_features",
        "engineer_dividend_reliability_features",
    ],
    "Analyst Sentiment": [
        "engineer_analyst_quality_features",
    ],
    "Market Sentiment": [
        "engineer_market_sentiment_features",
        "engineer_market_microstructure_features",
    ],
    "Leverage & Liquidity": [
        "engineer_leverage_ratios",
        "engineer_liquidity_ratios",
    ],
    "Temporal Patterns": [
        "engineer_temporal_features",
    ],
    "Composite Scores": [
        "engineer_composite_scores",
    ],
}


def extract_features_from_function(source_code: str, func_name: str) -> Set[str]:
    """Extract feature names assigned in a function."""
    features = set()

    # Pattern: result["feature_name"] = ...
    pattern1 = r'result\["([a-z_0-9]+)"\]\s*='
    # Pattern: result['feature_name'] = ...
    pattern2 = r"result\['([a-z_0-9]+)'\]\s*="

    for match in re.finditer(pattern1, source_code):
        features.add(match.group(1))
    for match in re.finditer(pattern2, source_code):
        features.add(match.group(1))

    return features


def extract_function_source(file_path: Path, func_name: str) -> str:
    """Extract source code of a specific function."""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find function start
    func_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(f"def {func_name}("):
            func_start = i
            break

    if func_start is None:
        return ""

    # Find function end (next def at same indentation or end of file)
    func_indent = len(lines[func_start]) - len(lines[func_start].lstrip())
    func_end = len(lines)

    for i in range(func_start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line.strip().startswith("#"):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= func_indent and line.strip().startswith("def "):
                func_end = i
                break

    return "".join(lines[func_start:func_end])


def audit_generators():
    """Audit all generator functions and compare with registry."""
    advanced_path = Path("finance_ml.features.advanced.py")

    if not advanced_path.exists():
        print(f"Error: {advanced_path} not found")
        return

    print("=" * 80)
    print("Phase 9.3 Feature Generator Audit")
    print("=" * 80)

    category_features = {}

    for category, generators in CATEGORY_GENERATORS.items():
        print(f"\n{category}:")
        print("-" * 40)

        all_features = set()

        for gen_func in generators:
            source = extract_function_source(advanced_path, gen_func)
            if not source:
                print(f"  ⚠ Function {gen_func} not found")
                continue

            features = extract_features_from_function(source, gen_func)
            print(f"  {gen_func}: {len(features)} features")
            for feat in sorted(features):
                print(f"    - {feat}")

            all_features.update(features)

        category_features[category] = sorted(all_features)
        print(f"  Total unique: {len(all_features)} features")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total_features = sum(len(feats) for feats in category_features.values())
    print(f"Total features across all categories: {total_features}")
    print("\nFeatures per category:")
    for category, features in category_features.items():
        print(f"  {category}: {len(features)} features")

    # Save to JSON
    output_path = Path("phase93_actual_features.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(category_features, f, indent=2)
    print(f"\n✓ Saved actual features to: {output_path}")

    # Load registry for comparison
    try:
        import sys

        sys.path.insert(0, str(Path.cwd()))
        from finance_ml.ml_workflow.eda.phase93_categories import PHASE93_FEATURE_CATEGORIES

        print("\n" + "=" * 80)
        print("REGISTRY COMPARISON")
        print("=" * 80)

        for category in CATEGORY_GENERATORS.keys():
            actual = set(category_features.get(category, []))
            registered = set(PHASE93_FEATURE_CATEGORIES.get(category, []))

            missing = registered - actual
            extra = actual - registered

            print(f"\n{category}:")
            print(f"  Registered: {len(registered)} | Actual: {len(actual)}")

            if missing:
                print(f"  ⚠ In registry but NOT generated ({len(missing)}):")
                for feat in sorted(missing):
                    print(f"    - {feat}")

            if extra:
                print(f"  ⚠ Generated but NOT in registry ({len(extra)}):")
                for feat in sorted(extra):
                    print(f"    - {feat}")

            if not missing and not extra:
                print("  ✓ Perfect match!")

    except Exception as e:
        print(f"\n⚠ Could not compare with registry: {e}")


if __name__ == "__main__":
    audit_generators()
