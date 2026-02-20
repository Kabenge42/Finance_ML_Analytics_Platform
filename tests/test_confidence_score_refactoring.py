"""
Tests for confidence score refactoring:
- compute_beta_confidence_score utility function
- Dynamic n_total proxy path
- AUC discrimination penalty
- Grid posterior smoothness (50 points)
"""

import unittest

import numpy as np
import pandas as pd


class TestComputeBetaConfidenceScore(unittest.TestCase):
    """Test the shared compute_beta_confidence_score utility."""

    def setUp(self):
        from finance_ml.analytics.probability_analytics import compute_beta_confidence_score
        self.compute = compute_beta_confidence_score

    def test_scalar_returns_float(self):
        result = self.compute(4.0, 4.0, prior_alpha=2.0, prior_beta=2.0)
        self.assertIsInstance(float(result), float)

    def test_output_bounded_0_1(self):
        """Confidence score must always be in [0, 1]."""
        test_cases = [
            (2.0, 2.0),   # no data beyond prior
            (10.0, 2.0),  # strong beat signal
            (2.0, 10.0),  # strong miss signal
            (22.0, 22.0), # large symmetric sample
            (100.0, 5.0), # extreme skew
        ]
        for alpha, beta in test_cases:
            score = self.compute(alpha, beta)
            self.assertGreaterEqual(float(score), 0.0, f"Failed for alpha={alpha}, beta={beta}")
            self.assertLessEqual(float(score), 1.0, f"Failed for alpha={alpha}, beta={beta}")

    def test_no_data_beyond_prior_gives_low_confidence(self):
        """When posterior equals prior, confidence should be very low."""
        score = self.compute(2.0, 2.0, prior_alpha=2.0, prior_beta=2.0)
        self.assertLess(float(score), 0.4, "No data should yield low confidence")

    def test_more_data_increases_confidence(self):
        """More observations should increase confidence."""
        score_small = self.compute(4.0, 4.0, prior_alpha=2.0, prior_beta=2.0)
        score_large = self.compute(12.0, 12.0, prior_alpha=2.0, prior_beta=2.0)
        self.assertGreater(float(score_large), float(score_small))

    def test_decisive_posterior_increases_confidence(self):
        """A posterior far from 0.5 should score higher than one at 0.5, same sample size."""
        # Same total (alpha+beta=12), but different decisiveness
        score_balanced = self.compute(6.0, 6.0, prior_alpha=2.0, prior_beta=2.0)
        score_decisive = self.compute(10.0, 2.0, prior_alpha=2.0, prior_beta=2.0)
        self.assertGreater(float(score_decisive), float(score_balanced))

    def test_vectorized_input(self):
        """Should work with numpy arrays."""
        alphas = np.array([4.0, 6.0, 10.0])
        betas = np.array([4.0, 6.0, 2.0])
        result = self.compute(alphas, betas)
        self.assertEqual(len(result), 3)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))

    def test_constant_confidence_bug_is_fixed(self):
        """
        The old formula gave concentration/20 = (alpha+beta)/20 which was constant
        when n_total was fixed. The new formula should vary with posterior shape.
        """
        # Simulate proxy path: prior (2,2) + different beat counts with n_total=5
        scores = []
        for n_beats in range(6):  # 0 through 5
            post_alpha = 2.0 + n_beats
            post_beta = 2.0 + (5 - n_beats)
            score = self.compute(post_alpha, post_beta)
            scores.append(float(score))

        # Old formula: all would be 9/20 = 0.45
        # New formula: should show variation due to decisiveness component
        self.assertGreater(max(scores) - min(scores), 0.01,
                           "Confidence scores should vary across different beat counts")

    def test_custom_normalization_factor(self):
        """Custom normalization factor should affect volume component."""
        # Use asymmetric posterior to ensure non-zero decisiveness
        score_default = self.compute(15.0, 5.0, normalization_factor=20.0)
        score_smaller = self.compute(15.0, 5.0, normalization_factor=10.0)
        self.assertGreater(float(score_smaller), float(score_default))


class TestDynamicNTotalProxy(unittest.TestCase):
    """Test that proxy paths use dynamic n_total instead of hardcoded 5."""

    def test_analyze_dataframe_proxy_varying_confidence(self):
        """Proxy path should produce varying confidence scores."""
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel

        model = EarningsBeatProbabilityModel()

        # No eps_positive_years or eps_positive_streak → forces graduated proxy path
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "name": ["Co A", "Co B", "Co C", "Co D", "Co E"],
            "sector": ["Tech"] * 5,
            "eps_trajectory_score": [10.0, 30.0, 50.0, 70.0, 90.0],
        })

        # Use non-existent columns to force proxy for all rows
        result = model.analyze_dataframe(df, beats_col="_none_", total_col="_none_")
        self.assertGreater(len(result), 0)
        self.assertIn("confidence_score", result.columns)

        scores = result["confidence_score"].values
        # With varying trajectory scores, confidence should not be constant
        self.assertGreater(scores.max() - scores.min(), 0.01,
                           "Confidence scores should vary with different trajectory scores")

    def test_analyze_dataframe_enhanced_proxy_varying_confidence(self):
        """Enhanced proxy path should produce varying confidence scores."""
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel

        model = EarningsBeatProbabilityModel()

        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "name": ["Co A", "Co B", "Co C", "Co D", "Co E"],
            "sector": ["Tech"] * 5,
            "eps_trajectory_score": [10.0, 30.0, 50.0, 70.0, 90.0],
        })

        result = model.analyze_dataframe_enhanced(df)
        self.assertGreater(len(result), 0)
        self.assertIn("confidence_score", result.columns)

        scores = result["confidence_score"].values
        self.assertGreater(scores.max() - scores.min(), 0.01,
                           "Confidence scores should vary with different trajectory scores")

    def test_proxy_dynamic_n_total_with_eps_positive_years(self):
        """When eps_positive_years is available, use it for n_total in proxy path."""
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel

        model = EarningsBeatProbabilityModel()

        # Use non-default column names so eps_positive_years is free for proxy n_total
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "name": ["Co A", "Co B", "Co C"],
            "sector": ["Tech"] * 3,
            "eps_trajectory_score": [50.0, 50.0, 50.0],
            "eps_positive_years": [3, 8, 15],
        })

        result = model.analyze_dataframe(df, beats_col="_none_", total_col="_none_")
        # Different eps_positive_years → different total_reports
        totals = result["total_reports"].values
        self.assertGreater(totals.max() - totals.min(), 0,
                           "total_reports should vary with eps_positive_years")

    def test_proxy_graduated_n_total_without_data_columns(self):
        """Without data columns, n_total should graduate by trajectory score."""
        from finance_ml.analytics.probability_analytics import EarningsBeatProbabilityModel

        model = EarningsBeatProbabilityModel()

        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "name": ["Co A", "Co B", "Co C", "Co D"],
            "sector": ["Tech"] * 4,
            "eps_trajectory_score": [10.0, 30.0, 65.0, 85.0],
        })

        result = model.analyze_dataframe_enhanced(df)
        totals = result["total_reports"].values
        # Graduated: 10→3, 30→4, 65→6, 85→8
        self.assertEqual(list(totals), [3, 4, 6, 8])


class TestAUCDiscriminationPenalty(unittest.TestCase):
    """Test that AUC < 0.5 is penalized in overall confidence."""

    def test_auc_below_half_reduces_overall(self):
        """AUC below 0.5 should reduce overall confidence score."""
        from finance_ml.analytics.probability_analytics import ModelConfidenceEstimator

        estimator = ModelConfidenceEstimator()

        np.random.seed(42)
        n = 200
        actual = np.random.binomial(1, 0.5, n)
        # Anti-discriminating: predict opposite of actual
        anti_probs = np.where(actual == 1, 0.3, 0.7).astype(float)
        # Random baseline
        random_probs = np.full(n, 0.5)

        result_anti = estimator.compute_confidence_metrics(anti_probs, actual, "Anti")
        result_random = estimator.compute_confidence_metrics(random_probs, actual, "Random")

        self.assertLess(result_anti.overall_confidence, result_random.overall_confidence,
                        "Anti-discriminating model should score lower than random")

    def test_auc_above_half_no_penalty(self):
        """AUC above 0.5 should not receive penalty."""
        from finance_ml.analytics.probability_analytics import ModelConfidenceEstimator

        estimator = ModelConfidenceEstimator()

        np.random.seed(42)
        n = 200
        actual = np.random.binomial(1, 0.5, n)
        # Good discriminator
        good_probs = np.where(actual == 1, 0.7, 0.3).astype(float)

        result = estimator.compute_confidence_metrics(good_probs, actual, "Good")
        # With good AUC, overall should be reasonably high
        self.assertGreater(result.overall_confidence, 60)


class TestGridPosteriorSmoothness(unittest.TestCase):
    """Test that grid posterior uses finer grid (50 points)."""

    def test_grid_has_50_points(self):
        import market_analytics
        grid = market_analytics._BEAT_MODEL_P_GRID
        self.assertEqual(len(grid), 50)

    def test_grid_range(self):
        import market_analytics
        grid = market_analytics._BEAT_MODEL_P_GRID
        self.assertAlmostEqual(grid[0], 0.01)
        self.assertAlmostEqual(grid[-1], 0.99)


class TestNumericCastCols(unittest.TestCase):
    """Test that _NUMERIC_CAST_COLS includes previously missing columns."""

    def test_missing_columns_now_included(self):
        from finance_ml.analytics.probability_analytics import _NUMERIC_CAST_COLS

        expected_additions = [
            "gaap_norm_spread",
            "revision_trend_short",
            "revision_trend_medium",
            "eps_norm_est_ntm",
            "eps_gaap_est_ntm",
            "eps_gaap_est_fy1e",
        ]
        for col in expected_additions:
            self.assertIn(col, _NUMERIC_CAST_COLS, f"{col} should be in _NUMERIC_CAST_COLS")


if __name__ == "__main__":
    unittest.main()
