"""
TDD tests for semantic column classification enhancements.

Tests for Phase 9.3 Task 3: Semantic Column Classification Enhancement
Aligned with phase_9.3_implementation_plan.md

Test Coverage:
- Test 1: classify_unknown_columns_by_pattern
- Test 2: classify_unknown_columns_by_schema
- Test 3: semantic_classification_coverage_above_90pct

Business Objective: Reduce "OTHER" semantic category from 487 columns to <59
by implementing pattern-based classification and schema lookup enhancements.

Model Version: v9_9
Alignment: code_guidelines.md v1.10
"""

import unittest
import pandas as pd

from finance_ml.ml_workflow.preprocessing.column_semantics import (
    classify_columns,
    classify_columns_with_patterns,
    classify_columns_with_schema_fallback,
)
from finance_ml.ml_workflow.data.schema import COLUMN_SCHEMA


class TestSemanticClassification(unittest.TestCase):
    """Test suite for semantic column classification enhancements."""

    def test_classify_unknown_columns_by_pattern(self):
        """Suffix patterns should guide semantic classification."""
        # Given: Columns with standard suffixes
        test_columns = [
            "debt_to_equity_ltm",  # Ratio
            "total_revenues_fy",  # Count/Market Value
            "net_income_fq",  # Count/Market Value
            "roe_ltm",  # Percentage
            "operating_margin_fy",  # Percentage
            "ev_ebitda_ltm",  # Ratio
        ]

        # When: Classify with pattern inference
        classifications = classify_columns_with_patterns(test_columns)

        # Then: Correct semantic categories assigned
        self.assertEqual(classifications["debt_to_equity_ltm"], "RATIO")
        self.assertEqual(classifications["roe_ltm"], "PERCENTAGE")
        self.assertIn(classifications["total_revenues_fy"], ["COUNT", "MARKET_VALUE"])
        self.assertEqual(classifications["ev_ebitda_ltm"], "RATIO")
        self.assertEqual(classifications["operating_margin_fy"], "PERCENTAGE")

    def test_classify_unknown_columns_by_schema(self):
        """COLUMN_SCHEMA should provide fallback classification."""
        # Given: Columns in schema but not classified
        unclassified = [
            col for col in COLUMN_SCHEMA.keys() if col not in ["ticker", "sector", "region"]
        ][
            :50
        ]  # Sample 50

        # When: Classify with schema lookup
        classifications = classify_columns_with_schema_fallback(unclassified)

        # Then: Schema dtype informs semantic category
        for col, category in classifications.items():
            if col in COLUMN_SCHEMA:
                dtype = COLUMN_SCHEMA[col]["dtype"]
                if dtype in ["float64", "float32"]:
                    self.assertIn(
                        category, ["RATIO", "PERCENTAGE", "MARKET_VALUE", "COUNT", "PRICE"]
                    )
                elif dtype == "object":
                    self.assertEqual(category, "CATEGORICAL")

    def test_semantic_classification_coverage_above_90pct(self):
        """After enhancements, <10% should be in OTHER category."""
        # Given: Sample of preprocessed columns from metadata
        # Use a realistic set of columns from the schema
        all_columns = list(COLUMN_SCHEMA.keys())

        # When: Classify with enhanced pipeline
        result = classify_columns(all_columns)

        # Then: OTHER category <10%
        total = len(all_columns)
        other_count = len(result.get("other", set()))
        coverage_pct = 100 * (1 - other_count / total) if total > 0 else 100

        self.assertGreaterEqual(
            coverage_pct,
            90.0,
            f"Coverage {coverage_pct:.1f}% below 90% target ({other_count}/{total} in OTHER)",
        )
        # 10% of ~318 columns = ~32 columns max in OTHER
        max_other = int(total * 0.1)
        self.assertLessEqual(
            other_count, max_other, f"Too many columns in OTHER: {other_count} > {max_other}"
        )


if __name__ == "__main__":
    unittest.main()
