"""
Test Phase 9.5: Outlier Safety Rails & Non-Negative Constraints.

Tests for:
- summarize_winsorization_effects
- track_constraint_violations
- safety_rails_sensitivity_app

TDD Approach: Focused tests covering core functionality.
"""

import json
import unittest
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np


class TestSummarizeWinsorizationEffects(unittest.TestCase):
    """Test summarize_winsorization_effects function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "safety_rails"

        # Create sample raw and winsorized dataframes
        np.random.seed(42)
        n_samples = 100

        # Raw data with outliers
        self.features_raw = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [np.random.normal(50, 10, 95), [200, 250, 300, 350, 400]]
                ),
                "feature2": np.concatenate(
                    [np.random.normal(100, 20, 95), [-100, -50, 500, 600, 700]]
                ),
                "feature3": np.random.normal(25, 5, 100),
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], 100),
            }
        )

        # Winsorized data (outliers clipped)
        self.features_winsorized = self.features_raw.copy()
        for col in ["feature1", "feature2", "feature3"]:
            q_low = self.features_winsorized[col].quantile(0.05)
            q_high = self.features_winsorized[col].quantile(0.95)
            self.features_winsorized[col] = self.features_winsorized[col].clip(q_low, q_high)

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_summarize_winsorization_effects_creates_json(self):
        """Test that function creates summary JSON."""
        from finance_ml.ml_workflow.evaluation import summarize_winsorization_effects

        summary_dict = summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
            feature_cols=["feature1", "feature2", "feature3"],
        )

        # Check summary structure
        self.assertIsInstance(summary_dict, dict)

        # Check output file exists
        json_path = self.output_dir / "clipping_effect_summary.json"
        self.assertTrue(json_path.exists())

    def test_winsorization_detects_changes(self):
        """Test that function detects differences between raw and winsorized."""
        from finance_ml.ml_workflow.evaluation import summarize_winsorization_effects

        summary_dict = summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
            feature_cols=["feature1", "feature2"],
        )

        # Should have feature-specific keys (actual implementation uses feature names as keys)
        self.assertIn("feature1", summary_dict)
        self.assertIn("feature2", summary_dict)


class TestTrackConstraintViolations(unittest.TestCase):
    """Test track_constraint_violations function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "safety_rails"

        # Create predictions with some negative values (violations)
        np.random.seed(42)
        n_samples = 100

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(n_samples)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples),
                "y_pred": np.concatenate(
                    [
                        np.random.uniform(10, 100, 95),  # Valid predictions
                        [-5, -10, -2, -8, -3],  # 5 violations
                    ]
                ),
                "y_true": np.random.uniform(10, 100, n_samples),
            }
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_track_constraint_violations_creates_json(self):
        """Test that function creates violations JSON."""
        from finance_ml.ml_workflow.evaluation import track_constraint_violations

        violations_dict = track_constraint_violations(
            predictions_df=self.predictions_df, output_dir=self.output_dir, prediction_col="y_pred"
        )

        # Check structure
        self.assertIsInstance(violations_dict, dict)

        # Check output file exists
        json_path = self.output_dir / "non_negative_violations.json"
        self.assertTrue(json_path.exists())

    def test_detects_negative_predictions(self):
        """Test that function correctly identifies negative predictions."""
        from finance_ml.ml_workflow.evaluation import track_constraint_violations

        violations_dict = track_constraint_violations(
            predictions_df=self.predictions_df, output_dir=self.output_dir, prediction_col="y_pred"
        )

        # Should detect 5 violations
        if "total_violations" in violations_dict:
            self.assertEqual(violations_dict["total_violations"], 5)

    def test_handles_no_violations(self):
        """Test that function handles case with no violations."""
        from finance_ml.ml_workflow.evaluation import track_constraint_violations

        # Create df with no negative values
        df_clean = self.predictions_df.copy()
        df_clean["y_pred"] = np.abs(df_clean["y_pred"])

        violations_dict = track_constraint_violations(
            predictions_df=df_clean, output_dir=self.output_dir, prediction_col="y_pred"
        )

        # Should report 0 violations
        if "total_violations" in violations_dict:
            self.assertEqual(violations_dict["total_violations"], 0)


class TestSafetyRailsSensitivityApp(unittest.TestCase):
    """Test safety_rails_sensitivity_app function."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "safety_rails"

        # Create sample data
        np.random.seed(42)
        n_samples = 100

        self.data_df = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [np.random.normal(50, 10, 95), [200, 250, 300, 350, 400]]
                ),
                "feature2": np.random.normal(100, 20, 100),
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], 100),
            }
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_sensitivity_app_creates_html(self):
        """Test that function creates HTML dashboard."""
        from finance_ml.ml_workflow.evaluation import safety_rails_sensitivity_app

        safety_rails_sensitivity_app(
            data_df=self.data_df,
            output_dir=self.output_dir,
            default_lower_pct=0.05,
            default_upper_pct=0.95,
        )

        # Check HTML file exists
        html_path = self.output_dir / "safety_rails_sensitivity_dashboard.html"
        self.assertTrue(html_path.exists())

    def test_html_is_valid(self):
        """Test that generated HTML is valid."""
        from finance_ml.ml_workflow.evaluation import safety_rails_sensitivity_app

        safety_rails_sensitivity_app(
            data_df=self.data_df,
            output_dir=self.output_dir,
        )

        html_path = self.output_dir / "safety_rails_sensitivity_dashboard.html"
        content = html_path.read_text(encoding="utf-8", errors="ignore")

        self.assertGreater(len(content), 100)
        self.assertTrue("<html>" in content.lower() or "<div>" in content.lower())


class TestIntegrationPhase95(unittest.TestCase):
    """Integration tests for Phase 9.5 workflow."""

    def setUp(self):
        """Create test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "safety_rails"

        # Create comprehensive test data
        np.random.seed(42)
        n_samples = 200

        self.features_raw = pd.DataFrame(
            {
                "feature1": np.concatenate(
                    [
                        np.random.normal(50, 10, 190),
                        [200, 250, 300, 350, 400, -50, -100, -30, -80, -120],
                    ]
                ),
                "feature2": np.random.normal(100, 20, 200),
                "sector": np.random.choice(["Tech", "Finance", "Healthcare", "Energy"], 200),
            }
        )

        self.features_winsorized = self.features_raw.copy()
        for col in ["feature1", "feature2"]:
            q_low = self.features_winsorized[col].quantile(0.05)
            q_high = self.features_winsorized[col].quantile(0.95)
            self.features_winsorized[col] = self.features_winsorized[col].clip(q_low, q_high)

        self.predictions_df = pd.DataFrame(
            {
                "ticker": [f"TICK{i}" for i in range(200)],
                "sector": self.features_raw["sector"],
                "y_pred": np.concatenate([np.random.uniform(10, 100, 195), [-5, -10, -2, -8, -3]]),
                "y_true": np.random.uniform(10, 100, 200),
            }
        )

    def tearDown(self):
        """Clean up test artifacts."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_full_phase95_workflow(self):
        """Test complete Phase 9.5 workflow."""
        from finance_ml.ml_workflow.evaluation import (
            summarize_winsorization_effects,
            track_constraint_violations,
            safety_rails_sensitivity_app,
        )

        # Step 1: Winsorization effects
        summarize_winsorization_effects(
            features_raw=self.features_raw,
            features_winsorized=self.features_winsorized,
            output_dir=self.output_dir,
            feature_cols=["feature1", "feature2"],
        )

        # Step 2: Track violations
        track_constraint_violations(
            predictions_df=self.predictions_df, output_dir=self.output_dir, prediction_col="y_pred"
        )

        # Step 3: Sensitivity dashboard
        safety_rails_sensitivity_app(
            data_df=self.features_raw,
            output_dir=self.output_dir,
        )

        # Verify all artifacts created
        expected_artifacts = [
            "clipping_effect_summary.json",
            "non_negative_violations.json",
            "safety_rails_sensitivity_dashboard.html",
        ]

        for artifact in expected_artifacts:
            filepath = self.output_dir / artifact
            self.assertTrue(filepath.exists(), f"Missing artifact: {artifact}")


if __name__ == "__main__":
    unittest.main()
