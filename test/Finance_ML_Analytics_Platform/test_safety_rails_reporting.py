"""
Test safety rails reporting and visualization (Phase 9.5).

Tests for notebook-friendly safety rails functions that generate:
- clipping_effect_summary.json
- non_negative_violations.json
- safety_rails_summary.json
- HTML visualizations (pre/post winsorization, violation heatmap, sensitivity dashboard)
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import json


class TestSummarizeWinsorizationEffects(unittest.TestCase):
    """Test summarize_winsorization_effects function."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 200

        # Create pre-winsorization data with outliers
        self.features_raw = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n_samples),
                "feature_1": np.concatenate(
                    [
                        np.random.normal(100, 20, n_samples - 10),
                        np.array(
                            [500, 600, 700, 800, 900, -100, -200, -300, -400, -500]
                        ),  # Outliers
                    ]
                ),
                "feature_2": np.random.normal(50, 10, n_samples),
            }
        )

        # Create post-winsorization data (outliers clipped)
        self.features_winsorized = self.features_raw.copy()
        self.features_winsorized["feature_1"] = np.clip(
            self.features_winsorized["feature_1"],
            self.features_winsorized["feature_1"].quantile(0.05),
            self.features_winsorized["feature_1"].quantile(0.95),
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_summarize_winsorization_creates_json(self):
        """Test that clipping_effect_summary.json is created."""
        from finance_ml.ml_workflow.evaluation.safety_rails import summarize_winsorization_effects

        result = summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
        )

        json_path = self.output_dir / "clipping_effect_summary.json"
        self.assertTrue(json_path.exists())

        with open(json_path, "r") as f:
            summary = json.load(f)

        # Should have statistics per feature
        self.assertIn("feature_1", summary)
        self.assertIn("feature_2", summary)

    def test_summarize_winsorization_computes_stats(self):
        """Test that winsorization statistics are computed correctly."""
        from finance_ml.ml_workflow.evaluation.safety_rails import summarize_winsorization_effects

        summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
        )

        json_path = self.output_dir / "clipping_effect_summary.json"
        with open(json_path, "r") as f:
            summary = json.load(f)

        # feature_1 should show winsorization effects
        f1_stats = summary["feature_1"]
        self.assertIn("raw_mean", f1_stats)
        self.assertIn("winsorized_mean", f1_stats)
        self.assertIn("raw_std", f1_stats)
        self.assertIn("winsorized_std", f1_stats)
        self.assertIn("pct_values_changed", f1_stats)

        # feature_1 should have values changed (outliers clipped)
        self.assertGreater(f1_stats["pct_values_changed"], 0)

    def test_summarize_winsorization_creates_html(self):
        """Test that pre_post_winsorization_distributions.html is created."""
        from finance_ml.ml_workflow.evaluation.safety_rails import summarize_winsorization_effects

        result = summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
        )

        html_path = self.output_dir / "pre_post_winsorization_distributions.html"
        self.assertTrue(html_path.exists())


class TestTrackConstraintViolations(unittest.TestCase):
    """Test track_constraint_violations function."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 100

        # Create predictions with some intentional violations
        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n_samples),
                "y_pred_raw": np.concatenate(
                    [
                        np.random.uniform(50, 200, n_samples - 5),
                        np.array([-10, -5, -2, -1, -0.5]),  # Negative predictions (violations)
                    ]
                ),
                "y_pred_clipped": np.concatenate(
                    [
                        np.random.uniform(50, 200, n_samples - 5),
                        np.array([0, 0, 0, 0, 0]),  # After clipping
                    ]
                ),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_track_violations_creates_json(self):
        """Test that non_negative_violations.json is created."""
        from finance_ml.ml_workflow.evaluation.safety_rails import track_constraint_violations

        track_constraint_violations(predictions_df=self.predictions_df, output_dir=self.output_dir)

        json_path = self.output_dir / "non_negative_violations.json"
        self.assertTrue(json_path.exists())

        with open(json_path, "r") as f:
            violations = json.load(f)

        self.assertIn("total_violations", violations)
        self.assertIn("violations_by_sector", violations)

    def test_track_violations_detects_negatives(self):
        """Test that negative prediction violations are detected."""
        from finance_ml.ml_workflow.evaluation.safety_rails import track_constraint_violations

        track_constraint_violations(predictions_df=self.predictions_df, output_dir=self.output_dir)

        json_path = self.output_dir / "non_negative_violations.json"
        with open(json_path, "r") as f:
            violations = json.load(f)

        # Should detect the 5 negative predictions in raw data
        self.assertEqual(violations["total_violations"], 5)

    def test_track_violations_post_clipping_zero(self):
        """Test that violations are zero after clipping."""
        from finance_ml.ml_workflow.evaluation.safety_rails import track_constraint_violations

        # Use only clipped predictions
        df_clipped = self.predictions_df[["ticker", "sector", "y_pred_clipped"]].copy()
        df_clipped.rename(columns={"y_pred_clipped": "y_pred_raw"}, inplace=True)

        track_constraint_violations(predictions_df=df_clipped, output_dir=self.output_dir)

        json_path = self.output_dir / "non_negative_violations.json"
        with open(json_path, "r") as f:
            violations = json.load(f)

        # Should have zero violations after clipping
        self.assertEqual(violations["total_violations"], 0)

    def test_track_violations_creates_heatmap(self):
        """Test that violation_heatmap_by_feature_sector.html is created."""
        from finance_ml.ml_workflow.evaluation.safety_rails import track_constraint_violations

        track_constraint_violations(predictions_df=self.predictions_df, output_dir=self.output_dir)

        html_path = self.output_dir / "violation_heatmap_by_feature_sector.html"
        self.assertTrue(html_path.exists())


class TestSafetyRailsSummary(unittest.TestCase):
    """Test safety_rails_summary.json generation."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 100

        self.features_raw = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare"], n_samples),
                "feature_1": np.random.normal(100, 20, n_samples),
            }
        )

        self.features_winsorized = self.features_raw.copy()
        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare"], n_samples),
                "y_pred_raw": np.random.uniform(50, 200, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_combined_functions_create_summary(self):
        """Test that calling both functions creates a complete summary."""
        from finance_ml.ml_workflow.evaluation.safety_rails import (
            summarize_winsorization_effects,
            track_constraint_violations,
        )

        summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
        )

        track_constraint_violations(predictions_df=self.predictions_df, output_dir=self.output_dir)

        # Check all expected files exist
        self.assertTrue((self.output_dir / "clipping_effect_summary.json").exists())
        self.assertTrue((self.output_dir / "non_negative_violations.json").exists())


class TestSafetyRailsSensitivityApp(unittest.TestCase):
    """Test safety_rails_sensitivity_app function."""

    def setUp(self):
        """Create test data and temporary output directory."""
        self.output_dir = Path(tempfile.mkdtemp())

        np.random.seed(42)
        n_samples = 150

        # Create sample data for sensitivity analysis
        self.data_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(["Technology", "Healthcare", "Financials"], n_samples),
                "feature_value": np.concatenate(
                    [
                        np.random.normal(100, 20, n_samples - 10),
                        np.random.uniform(200, 500, 10),  # Outliers
                    ]
                ),
                "y_true": np.random.uniform(50, 200, n_samples),
            }
        )

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.output_dir)

    def test_sensitivity_app_creates_html(self):
        """Test that safety_rails_sensitivity_dashboard.html is created."""
        from finance_ml.ml_workflow.evaluation.safety_rails import safety_rails_sensitivity_app

        html_path = safety_rails_sensitivity_app(data_df=self.data_df, output_dir=self.output_dir)

        self.assertTrue(Path(html_path).exists())
        self.assertTrue(str(html_path).endswith(".html"))
        self.assertIn("sensitivity", str(html_path).lower())

    def test_sensitivity_app_handles_threshold_params(self):
        """Test that sensitivity app accepts threshold parameters."""
        from finance_ml.ml_workflow.evaluation.safety_rails import safety_rails_sensitivity_app

        html_path = safety_rails_sensitivity_app(
            data_df=self.data_df,
            output_dir=self.output_dir,
            default_lower_pct=0.05,
            default_upper_pct=0.95,
        )

        self.assertTrue(Path(html_path).exists())


if __name__ == "__main__":
    unittest.main()
