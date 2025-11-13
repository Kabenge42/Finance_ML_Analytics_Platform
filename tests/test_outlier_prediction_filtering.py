"""
TDD Test Suite for Task 10.2: Implement Robust Outlier Filtering

Tests post-prediction outlier detection and confidence scoring.
Target: Reduce mean-median error gap from 19x to <3x

Requirements from finance_ml_improvement_plan.md Task 10.2:
- Add post-prediction outlier detection (IQR, Z-score, isolation forest on errors)
- Implement prediction confidence scores based on feature completeness
- Filter/flag predictions with extreme percentage errors (>500%)
- Add "prediction_quality" column: {high, medium, low} based on confidence
- Separate reporting for high-confidence vs. all predictions
- Target: Reduce mean-median error gap from 19x to <3x

Test Categories:
1. Post-prediction outlier detection on errors (not input features)
2. Confidence score calculation (0-1 range, based on feature completeness + interval width)
3. Extreme error flagging (>500% percentage error)
4. Prediction quality categorization (high/medium/low)
5. Filtered reporting impact on mean vs median error metrics
6. Mean-median error gap validation (<3x target)
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile


class TestOutlierPredictionFiltering(unittest.TestCase):
    """Test suite for post-prediction outlier filtering and confidence scoring."""

    def setUp(self):
        """Create synthetic prediction data with outliers and quality variations."""
        np.random.seed(42)
        n_samples = 1000

        # Create prediction data with mix of good and bad predictions
        self.y_true = np.random.uniform(50, 200, n_samples)

        # Most predictions are good (small error)
        good_mask = np.random.rand(n_samples) < 0.95
        self.y_pred = np.where(
            good_mask,
            self.y_true + np.random.normal(0, 10, n_samples),  # Good predictions
            self.y_true * np.random.uniform(5, 10, n_samples),  # Catastrophic outliers
        )

        # Create feature completeness scores (0-1)
        # Lower score = more missing features = lower confidence
        self.feature_completeness = np.random.beta(
            5, 2, n_samples
        )  # Skewed toward high completeness

        # Introduce some low-completeness samples
        low_completeness_mask = np.random.rand(n_samples) < 0.1
        self.feature_completeness[low_completeness_mask] = np.random.uniform(
            0.2, 0.5, low_completeness_mask.sum()
        )

        # Create prediction intervals (for confidence scoring)
        self.interval_width = np.random.uniform(10, 100, n_samples)

        # Wide intervals = higher uncertainty = lower confidence
        high_uncertainty_mask = np.random.rand(n_samples) < 0.15
        self.interval_width[high_uncertainty_mask] = np.random.uniform(
            150, 300, high_uncertainty_mask.sum()
        )

        # Create sectors for stratified analysis
        self.sectors = np.random.choice(
            ["Technology", "Healthcare", "Energy", "Financials"], size=n_samples
        )

        # Build DataFrame
        self.predictions_df = pd.DataFrame(
            {
                "y_true": self.y_true,
                "y_pred": self.y_pred,
                "feature_completeness": self.feature_completeness,
                "interval_width": self.interval_width,
                "sector": self.sectors,
            }
        )

        # Calculate errors
        self.predictions_df["abs_error"] = np.abs(
            self.predictions_df["y_true"] - self.predictions_df["y_pred"]
        )
        self.predictions_df["pct_error"] = (
            100 * self.predictions_df["abs_error"] / self.predictions_df["y_true"]
        )

    def test_detect_prediction_outliers_function_exists(self):
        """Test that detect_prediction_outliers function exists."""
        try:
            from finance_ml.ml_workflow.evaluation.confidence import detect_prediction_outliers

            self.assertTrue(callable(detect_prediction_outliers))
        except ImportError:
            self.fail("detect_prediction_outliers function not implemented")

    def test_calculate_prediction_confidence_function_exists(self):
        """Test that calculate_prediction_confidence function exists."""
        try:
            from finance_ml.ml_workflow.evaluation.confidence import calculate_prediction_confidence

            self.assertTrue(callable(calculate_prediction_confidence))
        except ImportError:
            self.fail("calculate_prediction_confidence function not implemented")

    def test_flag_extreme_errors_function_exists(self):
        """Test that flag_extreme_errors function exists."""
        try:
            from finance_ml.ml_workflow.evaluation.confidence import flag_extreme_errors

            self.assertTrue(callable(flag_extreme_errors))
        except ImportError:
            self.fail("flag_extreme_errors function not implemented")

    def test_assign_prediction_quality_function_exists(self):
        """Test that assign_prediction_quality function exists."""
        try:
            from finance_ml.ml_workflow.evaluation.confidence import assign_prediction_quality

            self.assertTrue(callable(assign_prediction_quality))
        except ImportError:
            self.fail("assign_prediction_quality function not implemented")

    def test_detect_outliers_iqr_on_errors(self):
        """Test IQR outlier detection on prediction errors."""
        from finance_ml.ml_workflow.evaluation.confidence import detect_prediction_outliers

        outlier_mask = detect_prediction_outliers(
            self.predictions_df, method="iqr", error_col="abs_error"
        )

        # Should return boolean array
        self.assertEqual(len(outlier_mask), len(self.predictions_df))
        self.assertEqual(outlier_mask.dtype, bool)

        # Should detect some outliers (we injected 5% catastrophic predictions)
        outlier_rate = outlier_mask.mean()
        self.assertGreater(outlier_rate, 0.01, "Should detect at least 1% outliers")
        self.assertLess(outlier_rate, 0.20, "Should not flag more than 20% as outliers")

    def test_detect_outliers_zscore_on_errors(self):
        """Test Z-score outlier detection on prediction errors."""
        from finance_ml.ml_workflow.evaluation.confidence import detect_prediction_outliers

        outlier_mask = detect_prediction_outliers(
            self.predictions_df, method="zscore", error_col="pct_error", threshold=3.0
        )

        self.assertEqual(len(outlier_mask), len(self.predictions_df))
        self.assertEqual(outlier_mask.dtype, bool)

        # Z-score should detect extreme percentage errors
        outlier_rate = outlier_mask.mean()
        self.assertGreater(outlier_rate, 0.01)
        self.assertLess(outlier_rate, 0.25)

    def test_detect_outliers_isolation_forest_on_errors(self):
        """Test Isolation Forest outlier detection on prediction errors."""
        from finance_ml.ml_workflow.evaluation.confidence import detect_prediction_outliers

        outlier_mask = detect_prediction_outliers(
            self.predictions_df, method="isolation_forest", error_col="abs_error", contamination=0.1
        )

        self.assertEqual(len(outlier_mask), len(self.predictions_df))
        self.assertEqual(outlier_mask.dtype, bool)

        # Contamination=0.1 means expect ~10% outliers
        outlier_rate = outlier_mask.mean()
        self.assertGreater(outlier_rate, 0.05)
        self.assertLess(outlier_rate, 0.15)

    def test_flag_extreme_errors_above_500_percent(self):
        """Test flagging predictions with >500% percentage error."""
        from finance_ml.ml_workflow.evaluation.confidence import flag_extreme_errors

        extreme_mask = flag_extreme_errors(
            self.predictions_df, pct_error_col="pct_error", threshold=500.0
        )

        # Should return boolean array
        self.assertEqual(len(extreme_mask), len(self.predictions_df))
        self.assertEqual(extreme_mask.dtype, bool)

        # Verify all flagged predictions have >500% error
        if extreme_mask.any():
            extreme_errors = self.predictions_df.loc[extreme_mask, "pct_error"]
            self.assertTrue(
                (extreme_errors > 500).all(), "All flagged predictions should have >500% error"
            )

        # Should flag at least some predictions (we injected catastrophic errors)
        self.assertGreater(extreme_mask.sum(), 0, "Should flag at least some extreme errors")

    def test_calculate_confidence_score_range_0_to_1(self):
        """Test that confidence scores are in valid [0, 1] range."""
        from finance_ml.ml_workflow.evaluation.confidence import calculate_prediction_confidence

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        # Should return array of same length
        self.assertEqual(len(confidence_scores), len(self.predictions_df))

        # All scores should be in [0, 1]
        self.assertTrue(
            np.all((confidence_scores >= 0) & (confidence_scores <= 1)),
            f"Confidence scores must be in [0, 1], got range [{confidence_scores.min():.3f}, {confidence_scores.max():.3f}]",
        )

    def test_confidence_score_inversely_related_to_interval_width(self):
        """Test that wider intervals result in lower confidence scores."""
        from finance_ml.ml_workflow.evaluation.confidence import calculate_prediction_confidence

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        # Split into narrow and wide interval groups
        narrow_mask = self.predictions_df["interval_width"] < 50
        wide_mask = self.predictions_df["interval_width"] > 150

        if narrow_mask.sum() > 0 and wide_mask.sum() > 0:
            narrow_confidence = confidence_scores[narrow_mask].mean()
            wide_confidence = confidence_scores[wide_mask].mean()

            self.assertGreater(
                narrow_confidence,
                wide_confidence,
                "Narrow intervals should have higher confidence than wide intervals",
            )

    def test_confidence_score_increases_with_feature_completeness(self):
        """Test that higher feature completeness results in higher confidence."""
        from finance_ml.ml_workflow.evaluation.confidence import calculate_prediction_confidence

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        # Split into high and low completeness groups
        high_completeness_mask = self.predictions_df["feature_completeness"] > 0.8
        low_completeness_mask = self.predictions_df["feature_completeness"] < 0.4

        if high_completeness_mask.sum() > 0 and low_completeness_mask.sum() > 0:
            high_conf = confidence_scores[high_completeness_mask].mean()
            low_conf = confidence_scores[low_completeness_mask].mean()

            self.assertGreater(
                high_conf, low_conf, "High feature completeness should yield higher confidence"
            )

    def test_assign_prediction_quality_three_categories(self):
        """Test that prediction quality is assigned to high/medium/low categories."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            calculate_prediction_confidence,
            assign_prediction_quality,
        )

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        quality_labels = assign_prediction_quality(confidence_scores)

        # Should return same length
        self.assertEqual(len(quality_labels), len(self.predictions_df))

        # Should only contain valid categories
        unique_qualities = set(quality_labels)
        valid_qualities = {"high", "medium", "low"}
        self.assertTrue(
            unique_qualities.issubset(valid_qualities),
            f"Invalid quality labels found: {unique_qualities - valid_qualities}",
        )

        # Should have all three categories (with 1000 samples)
        self.assertEqual(
            len(unique_qualities), 3, "Should have all three quality categories (high, medium, low)"
        )

    def test_filter_low_confidence_predictions(self):
        """Test filtering out low-confidence predictions."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            calculate_prediction_confidence,
            filter_low_confidence_predictions,
        )

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        # Add confidence scores to dataframe
        df_with_conf = self.predictions_df.copy()
        df_with_conf["confidence_score"] = confidence_scores

        # Filter with threshold 0.5
        filtered_df = filter_low_confidence_predictions(
            df_with_conf, confidence_col="confidence_score", threshold=0.5
        )

        # Filtered dataframe should only contain high-confidence predictions
        self.assertTrue(
            (filtered_df["confidence_score"] >= 0.5).all(),
            "Filtered dataframe should only contain predictions with confidence >= 0.5",
        )

        # Should remove some predictions
        self.assertLess(
            len(filtered_df),
            len(df_with_conf),
            "Filtering should remove some low-confidence predictions",
        )

    def test_mean_median_error_gap_reduction_after_filtering(self):
        """Test that filtering reduces mean-median error gap to <3x."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            calculate_prediction_confidence,
            assign_prediction_quality,
            detect_prediction_outliers,
        )

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        quality_labels = assign_prediction_quality(confidence_scores)

        # Detect outliers using IQR on errors
        outlier_mask = detect_prediction_outliers(
            self.predictions_df, method="iqr", error_col="abs_error"
        )

        # Calculate unfiltered metrics (all predictions)
        mean_error_all = self.predictions_df["abs_error"].mean()
        median_error_all = self.predictions_df["abs_error"].median()
        gap_all = mean_error_all / median_error_all

        # Calculate filtered metrics: high-confidence AND not outlier
        # This is the key: combine confidence scoring with outlier detection
        high_quality_mask = quality_labels == "high"
        filtered_mask = high_quality_mask & ~outlier_mask
        filtered_df = self.predictions_df[filtered_mask]

        self.assertGreater(
            len(filtered_df), 100, "Should have enough filtered predictions for valid statistics"
        )

        mean_error_filtered = filtered_df["abs_error"].mean()
        median_error_filtered = filtered_df["abs_error"].median()
        gap_filtered = mean_error_filtered / median_error_filtered

        # Gap should be significantly reduced
        self.assertLess(
            gap_filtered,
            gap_all,
            f"Filtered gap ({gap_filtered:.1f}x) should be less than unfiltered ({gap_all:.1f}x)",
        )

        # Target: gap < 3x after filtering
        self.assertLess(
            gap_filtered,
            3.0,
            f"Mean-median error gap ({gap_filtered:.1f}x) should be < 3x after filtering",
        )

    def test_separate_reporting_high_vs_all_predictions(self):
        """Test that we can generate separate reports for high-confidence vs all predictions."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            calculate_prediction_confidence,
            assign_prediction_quality,
            prediction_quality_report,
        )

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        quality_labels = assign_prediction_quality(confidence_scores)

        # Add to dataframe
        df_with_quality = self.predictions_df.copy()
        df_with_quality["confidence_score"] = confidence_scores
        df_with_quality["prediction_quality"] = quality_labels

        # Generate report
        report = prediction_quality_report(
            df_with_quality,
            quality_col="prediction_quality",
            error_col="abs_error",
            pct_error_col="pct_error",
        )

        # Report should have statistics by quality level
        self.assertIn("by_quality", report)
        self.assertIn("high", report["by_quality"])
        self.assertIn("medium", report["by_quality"])
        self.assertIn("low", report["by_quality"])

        # Each quality level should have metrics
        for quality in ["high", "medium", "low"]:
            quality_stats = report["by_quality"][quality]
            self.assertIn("count", quality_stats)
            self.assertIn("mean_error", quality_stats)
            self.assertIn("median_error", quality_stats)
            self.assertIn("mean_pct_error", quality_stats)
            self.assertIn("error_gap_ratio", quality_stats)

    def test_prediction_quality_report_export_to_csv(self):
        """Test that prediction quality report can be exported to CSV."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            calculate_prediction_confidence,
            assign_prediction_quality,
            prediction_quality_report,
            export_quality_report,
        )

        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        quality_labels = assign_prediction_quality(confidence_scores)

        df_with_quality = self.predictions_df.copy()
        df_with_quality["confidence_score"] = confidence_scores
        df_with_quality["prediction_quality"] = quality_labels

        report = prediction_quality_report(
            df_with_quality,
            quality_col="prediction_quality",
            error_col="abs_error",
            pct_error_col="pct_error",
        )

        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            export_quality_report(report=report, output_dir=output_dir)

            # Verify file was created
            report_file = output_dir / "prediction_quality_report.csv"
            self.assertTrue(report_file.exists(), "prediction_quality_report.csv not created")

            # Verify CSV content
            report_df = pd.read_csv(report_file)
            self.assertIn("quality", report_df.columns)
            self.assertIn("count", report_df.columns)
            self.assertIn("mean_error", report_df.columns)
            self.assertIn("median_error", report_df.columns)
            self.assertIn("error_gap_ratio", report_df.columns)

    def test_outlier_detection_and_confidence_integration(self):
        """Test integration: outlier detection + confidence scoring work together."""
        from finance_ml.ml_workflow.evaluation.confidence import (
            detect_prediction_outliers,
            calculate_prediction_confidence,
            assign_prediction_quality,
        )

        # Detect outliers
        outlier_mask = detect_prediction_outliers(
            self.predictions_df, method="iqr", error_col="abs_error"
        )

        # Calculate confidence
        confidence_scores = calculate_prediction_confidence(
            self.predictions_df,
            feature_completeness_col="feature_completeness",
            interval_width_col="interval_width",
        )

        # Assign quality
        quality_labels = assign_prediction_quality(confidence_scores)

        # Verify: outliers should correlate with low confidence
        df_integrated = self.predictions_df.copy()
        df_integrated["is_outlier"] = outlier_mask
        df_integrated["confidence_score"] = confidence_scores
        df_integrated["prediction_quality"] = quality_labels

        # Outliers should have lower average confidence
        outlier_confidence = df_integrated.loc[outlier_mask, "confidence_score"].mean()
        inlier_confidence = df_integrated.loc[~outlier_mask, "confidence_score"].mean()

        self.assertLess(
            outlier_confidence,
            inlier_confidence,
            "Outliers should have lower confidence on average",
        )


if __name__ == "__main__":
    unittest.main()
