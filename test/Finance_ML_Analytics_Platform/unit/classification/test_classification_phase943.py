"""
Test Suite for Phase 9.4.3 - Enhanced Feature Importance Analysis

Tests for:
- analyze_feature_importance_by_groups: Group features by Phase 9.3 categories
- analyze_feature_importance_by_sector: Sector-specific feature importance
- analyze_shap_by_feature_groups: SHAP values grouped by feature categories

Author: Finance ML Team
Date: 2025-11-09
Version: 9.4.3
"""

import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from finance_ml.ml_workflow.classification.evaluation import (
    analyze_feature_importance_by_groups,
    analyze_feature_importance_by_sector,
    analyze_shap_by_feature_groups,
)


class TestAnalyzeFeatureImportanceByGroups(unittest.TestCase):
    """Test analyze_feature_importance_by_groups function."""

    def test_categorize_analyst_quality_features(self):
        """Test that analyst quality features are correctly categorized."""
        importance_dict = {
            "analyst_coverage": 0.15,
            "analyst_consensus_strength": 0.10,
            "price_target_spread": 0.08,
            "rating_buy_ratio": 0.05,
            "p_e_ratio": 0.20,
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        self.assertIn("group_totals", result)
        self.assertIn("group_percentages", result)
        self.assertIn("feature_groups", result)
        self.assertIn("top_features_per_group", result)

        # Check analyst quality group
        analyst_total = result["group_totals"]["analyst_quality"]
        self.assertAlmostEqual(analyst_total, 0.38, places=2)  # 0.15 + 0.10 + 0.08 + 0.05

    def test_categorize_accounting_quality_features(self):
        """Test that accounting quality features are correctly categorized."""
        importance_dict = {
            "exceptional_items_intensity": 0.12,
            "goodwill_intensity": 0.09,
            "intangibles_ratio": 0.07,
            "accounting_quality_score": 0.06,
            "revenue": 0.25,
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        # Check accounting quality group
        accounting_total = result["group_totals"]["accounting_quality"]
        self.assertAlmostEqual(accounting_total, 0.34, places=2)  # 0.12 + 0.09 + 0.07 + 0.06

    def test_categorize_employee_productivity_features(self):
        """Test that employee productivity features are correctly categorized."""
        importance_dict = {
            "revenue_per_employee": 0.18,
            "profit_per_employee": 0.14,
            "assets_per_employee": 0.11,
            "employee_growth_rate": 0.08,
            "market_cap": 0.30,
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        # Check employee productivity group
        employee_total = result["group_totals"]["employee_productivity"]
        self.assertAlmostEqual(employee_total, 0.51, places=2)  # 0.18 + 0.14 + 0.11 + 0.08

    def test_basic_features_categorization(self):
        """Test that non-Phase 9.3 features go to basic group."""
        importance_dict = {
            "p_e_ratio": 0.20,
            "market_cap": 0.18,
            "debt_to_equity": 0.15,
            "analyst_coverage": 0.10,  # This goes to analyst_quality
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        # Check that non-analyst features are in basic group
        self.assertEqual(result["feature_groups"]["p_e_ratio"], "basic")
        self.assertEqual(result["feature_groups"]["market_cap"], "basic")
        self.assertEqual(result["feature_groups"]["debt_to_equity"], "basic")
        self.assertEqual(result["feature_groups"]["analyst_coverage"], "analyst_quality")

    def test_group_percentages_sum_to_100(self):
        """Test that group percentages sum to 100%."""
        importance_dict = {
            "analyst_coverage": 0.25,
            "exceptional_items_intensity": 0.20,
            "revenue_per_employee": 0.30,
            "p_e_ratio": 0.25,
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        total_percentage = sum(result["group_percentages"].values())
        self.assertAlmostEqual(total_percentage, 100.0, places=1)

    def test_top_features_per_group(self):
        """Test that top features are correctly extracted per group."""
        importance_dict = {
            "analyst_coverage": 0.15,
            "analyst_consensus_strength": 0.10,
            "price_target_spread": 0.05,
            "p_e_ratio": 0.30,
        }

        result = analyze_feature_importance_by_groups(importance_dict, top_n_per_group=2)

        analyst_top = result["top_features_per_group"]["analyst_quality"]
        self.assertEqual(len(analyst_top), 2)  # Top 2 analyst features (limited by top_n_per_group)
        self.assertEqual(analyst_top[0]["feature"], "analyst_coverage")
        self.assertAlmostEqual(analyst_top[0]["importance"], 0.15, places=2)

    def test_empty_importance_dict(self):
        """Test handling of empty importance dictionary."""
        importance_dict = {}

        result = analyze_feature_importance_by_groups(importance_dict)

        self.assertEqual(result["group_totals"]["analyst_quality"], 0.0)
        self.assertEqual(result["group_totals"]["accounting_quality"], 0.0)
        self.assertEqual(result["group_totals"]["employee_productivity"], 0.0)
        self.assertEqual(result["group_totals"]["basic"], 0.0)


class TestAnalyzeFeatureImportanceBySector(unittest.TestCase):
    """Test analyze_feature_importance_by_sector function."""

    def setUp(self):
        """Set up test data with sectors."""
        np.random.seed(42)
        n_samples = 100

        # Create features with sector column
        self.X = pd.DataFrame(
            {
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "feature_1": np.random.randn(n_samples),
                "feature_2": np.random.randn(n_samples),
                "feature_3": np.random.randn(n_samples),
            }
        )

        # Create labels
        self.y = np.random.randint(0, 3, n_samples)

        # Create model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)

    def test_sector_specific_importance_structure(self):
        """Test that sector-specific importance returns correct structure."""
        result = analyze_feature_importance_by_sector(
            self.model, self.X, self.y, sector_col="sector", top_n=3
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("Sector", result.columns)
        self.assertIn("Feature", result.columns)
        self.assertIn("Importance", result.columns)
        self.assertIn("Rank", result.columns)

    def test_sector_specific_importance_content(self):
        """Test that importance is computed for each sector."""
        result = analyze_feature_importance_by_sector(
            self.model, self.X, self.y, sector_col="sector", top_n=3
        )

        if not result.empty:
            sectors_in_result = result["Sector"].unique()
            self.assertTrue(len(sectors_in_result) > 0)

            # Check ranks are assigned correctly
            for sector in sectors_in_result:
                sector_data = result[result["Sector"] == sector]
                ranks = sector_data["Rank"].values
                self.assertTrue(all(rank in [1, 2, 3] for rank in ranks))

    def test_missing_sector_column(self):
        """Test handling when sector column is missing."""
        X_no_sector = self.X.drop(columns=["sector"])

        result = analyze_feature_importance_by_sector(
            self.model, X_no_sector, self.y, sector_col="sector"
        )

        self.assertTrue(result.empty)
        self.assertListEqual(list(result.columns), ["Sector", "Feature", "Importance", "Rank"])

    def test_top_n_parameter(self):
        """Test that top_n parameter limits features per sector."""
        top_n = 2
        result = analyze_feature_importance_by_sector(
            self.model, self.X, self.y, sector_col="sector", top_n=top_n
        )

        if not result.empty:
            for sector in result["Sector"].unique():
                sector_features = result[result["Sector"] == sector]
                self.assertLessEqual(len(sector_features), top_n)

    def test_small_sector_samples(self):
        """Test that sectors with <10 samples are skipped."""
        # Create data with one tiny sector
        X_small = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 5,
                "feature_1": np.random.randn(55),
                "feature_2": np.random.randn(55),
            }
        )
        y_small = np.random.randint(0, 3, 55)

        result = analyze_feature_importance_by_sector(
            self.model, X_small, y_small, sector_col="sector", top_n=2
        )

        # Finance sector should be skipped
        if not result.empty:
            sectors_in_result = result["Sector"].unique()
            self.assertNotIn("Finance", sectors_in_result)


class TestAnalyzeShapByFeatureGroups(unittest.TestCase):
    """Test analyze_shap_by_feature_groups function."""

    def test_shap_grouping_with_mock_values(self):
        """Test SHAP grouping with mocked SHAP values."""
        # Mock SHAP values
        n_samples = 50
        n_features = 6

        shap_array = np.random.randn(n_samples, n_features) * 0.1

        feature_names = [
            "analyst_coverage",
            "analyst_consensus_strength",
            "exceptional_items_intensity",
            "revenue_per_employee",
            "p_e_ratio",
            "market_cap",
        ]

        # Mock shap_values object
        mock_shap = Mock()
        mock_shap.values = shap_array

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True):
            result = analyze_shap_by_feature_groups(mock_shap, feature_names)

            self.assertIn("group_mean_abs_shap", result)
            self.assertIn("group_percentages", result)
            self.assertIn("top_features_per_group", result)
            self.assertIn("feature_groups", result)

            # Check that all groups are present
            self.assertIn("analyst_quality", result["group_mean_abs_shap"])
            self.assertIn("accounting_quality", result["group_mean_abs_shap"])
            self.assertIn("employee_productivity", result["group_mean_abs_shap"])
            self.assertIn("basic", result["group_mean_abs_shap"])

    def test_shap_without_values_attribute(self):
        """Test SHAP grouping when values is a plain array."""
        shap_array = np.random.randn(30, 4) * 0.1
        feature_names = ["analyst_coverage", "p_e_ratio", "revenue_per_employee", "market_cap"]

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True):
            result = analyze_shap_by_feature_groups(shap_array, feature_names)

            self.assertIn("group_percentages", result)
            total_percentage = sum(result["group_percentages"].values())
            self.assertAlmostEqual(total_percentage, 100.0, places=1)

    def test_shap_multiclass_values(self):
        """Test SHAP grouping with multi-class SHAP values (3D array)."""
        # Multi-class SHAP: (n_samples, n_features, n_classes)
        shap_array_3d = np.random.randn(40, 5, 3) * 0.1

        feature_names = [
            "analyst_coverage",
            "exceptional_items_intensity",
            "revenue_per_employee",
            "p_e_ratio",
            "debt_ratio",
        ]

        mock_shap = Mock()
        mock_shap.values = shap_array_3d

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True):
            result = analyze_shap_by_feature_groups(mock_shap, feature_names)

            # Should handle 3D array by averaging across classes
            self.assertIn("group_mean_abs_shap", result)
            self.assertTrue(
                all(isinstance(v, float) for v in result["group_mean_abs_shap"].values())
            )

    def test_shap_top_features_per_group(self):
        """Test that top features are correctly extracted per group."""
        shap_array = np.array(
            [
                [0.1, 0.15, 0.05, 0.20],  # Sample 1
                [0.12, 0.14, 0.06, 0.18],  # Sample 2
            ]
        )

        feature_names = [
            "analyst_coverage",
            "analyst_consensus_strength",
            "p_e_ratio",
            "market_cap",
        ]

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True):
            result = analyze_shap_by_feature_groups(shap_array, feature_names, top_n_per_group=2)

            analyst_top = result["top_features_per_group"]["analyst_quality"]
            self.assertEqual(len(analyst_top), 2)  # Top 2 from analyst group

    def test_shap_not_available(self):
        """Test handling when SHAP is not available."""
        shap_array = np.random.randn(20, 3)
        feature_names = ["feature_1", "feature_2", "feature_3"]

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", False):
            result = analyze_shap_by_feature_groups(shap_array, feature_names)

            self.assertEqual(result, {})

    def test_shap_error_handling(self):
        """Test error handling in SHAP analysis."""
        # Invalid SHAP object that will cause an exception
        invalid_shap = "not_a_valid_shap_object"
        feature_names = ["feature_1", "feature_2"]

        with patch("finance_ml.ml_workflow.classification.evaluation.HAVE_SHAP", True):
            result = analyze_shap_by_feature_groups(invalid_shap, feature_names)

            # Should return empty dict on error
            self.assertEqual(result, {})


class TestPhase943Integration(unittest.TestCase):
    """Integration tests for Phase 9.4.3 functions."""

    def test_all_functions_importable(self):
        """Test that all Phase 9.4.3 functions can be imported."""
        from finance_ml.ml_workflow.classification import (
            analyze_feature_importance_by_groups,
            analyze_feature_importance_by_sector,
            analyze_shap_by_feature_groups,
        )

        self.assertTrue(callable(analyze_feature_importance_by_groups))
        self.assertTrue(callable(analyze_feature_importance_by_sector))
        self.assertTrue(callable(analyze_shap_by_feature_groups))

    def test_phase93_feature_detection(self):
        """Test that Phase 9.3 features are correctly detected across all functions."""
        # Test with all Phase 9.3 feature types
        importance_dict = {
            # Analyst quality
            "analyst_coverage": 0.10,
            "price_target_spread": 0.08,
            # Accounting quality
            "goodwill_intensity": 0.12,
            "accounting_quality_score": 0.09,
            # Employee productivity
            "revenue_per_employee": 0.15,
            "profit_per_employee": 0.11,
            # Basic
            "p_e_ratio": 0.20,
            "market_cap": 0.15,
        }

        result = analyze_feature_importance_by_groups(importance_dict)

        # All three Phase 9.3 groups should have features
        self.assertGreater(result["group_totals"]["analyst_quality"], 0)
        self.assertGreater(result["group_totals"]["accounting_quality"], 0)
        self.assertGreater(result["group_totals"]["employee_productivity"], 0)
        self.assertGreater(result["group_totals"]["basic"], 0)


if __name__ == "__main__":
    unittest.main()
