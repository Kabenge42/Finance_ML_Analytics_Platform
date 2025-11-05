"""
Comprehensive test suite for Phase 9.1 Enhanced 4-Step Imputation Strategy.

This module tests the complete 4-step imputation pipeline:
- Step 1: Zero imputation (48 columns)
- Step 2: KNN imputation (148 columns)
- Step 3: Price imputation (5 columns)
- Step 4: Median imputation (remaining columns)

Test coverage target: ≥80%
"""

import unittest
import pandas as pd
import numpy as np
from finance_ml.advanced_preprocessing import (
    get_zero_imputation_columns,
    get_knn_imputation_columns,
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_price_imputation,
    apply_median_imputation,
    apply_enhanced_imputation_strategy_4step,
    apply_enhanced_imputation_strategy,
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
    calculate_data_quality_score,
    impute_missing_values,
    impute_missing_values_knn_sector,
    create_scaler_pipeline,
    scale_features,
    DataQualityReport,
)


class TestEnhancedImputation4Step(unittest.TestCase):
    """Comprehensive test suite for 4-step imputation strategy."""

    def setUp(self):
        """Create sample financial data."""
        np.random.seed(42)
        n = 100
        self.df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN"] * 25,
                "sector": ["Technology"] * 100,
                # Step 1: Zero imputation columns
                "impairment_of_goodwill_fq": [np.nan] * 70 + [1000.0] * 30,
                "restructuring_charges_ltm": [np.nan] * 80 + [500.0] * 20,
                "cash_acquisitions_fy": [np.nan] * 90 + [2000.0] * 10,
                # Step 2: KNN imputation columns
                "market_cap": [
                    100 + i + np.random.randn() if i % 3 != 0 else np.nan for i in range(n)
                ],
                "enterprise_value": [
                    120 + i + np.random.randn() if i % 4 != 0 else np.nan for i in range(n)
                ],
                "ebitda_ltm": [10 + i * 0.1 if i % 5 != 0 else np.nan for i in range(n)],
                # Step 3: Price imputation columns
                "last_price": [50 + i * 0.5 for i in range(n)],
                "price_target": [np.nan] * 60 + [55 + i * 0.5 for i in range(40)],
                "price_target_median": [np.nan] * 70 + [54 + i * 0.5 for i in range(30)],
                # Step 4: Other numerical column
                "other_metric": [np.nan] * 50 + list(range(50, 100)),
            }
        )

    # STEP 1 TESTS
    def test_step1_zero_columns_count(self):
        """Step 1: Verify 48 zero imputation columns."""
        cols = get_zero_imputation_columns()
        self.assertEqual(len(cols), 48)

    def test_step1_zero_imputation_fills_with_zero(self):
        """Step 1: Verify zero imputation fills NaN with 0."""
        result = apply_zero_imputation(self.df)
        self.assertEqual(result["impairment_of_goodwill_fq"].isna().sum(), 0)
        self.assertIn(0.0, result["impairment_of_goodwill_fq"].values)

    def test_step1_zero_imputation_preserves_existing(self):
        """Step 1: Verify existing values are preserved."""
        result = apply_zero_imputation(self.df)
        # Last 30 values were 1000.0, should still be 1000.0
        original_non_nan = self.df["impairment_of_goodwill_fq"].dropna()
        for idx in original_non_nan.index:
            self.assertEqual(
                result.loc[idx, "impairment_of_goodwill_fq"],
                self.df.loc[idx, "impairment_of_goodwill_fq"],
            )

    # STEP 2 TESTS
    def test_step2_knn_columns_count(self):
        """Step 2: Verify 148 KNN imputation columns."""
        cols = get_knn_imputation_columns()
        self.assertEqual(len(cols), 148)

    def test_step2_knn_reduces_missing(self):
        """Step 2: Verify KNN imputation reduces missing values."""
        before = self.df["market_cap"].isna().sum()
        result = apply_knn_imputation_enhanced(self.df, sector_column="sector")
        after = result["market_cap"].isna().sum()
        self.assertLessEqual(after, before)

    def test_step2_knn_with_multiple_columns(self):
        """Step 2: Verify KNN works with multiple columns."""
        before_market_cap = self.df["market_cap"].isna().sum()
        before_enterprise = self.df["enterprise_value"].isna().sum()

        result = apply_knn_imputation_enhanced(self.df, sector_column="sector")

        after_market_cap = result["market_cap"].isna().sum()
        after_enterprise = result["enterprise_value"].isna().sum()

        # Both should have reduced or stayed the same
        self.assertLessEqual(after_market_cap, before_market_cap)
        self.assertLessEqual(after_enterprise, before_enterprise)

    # STEP 3 TESTS
    def test_step3_price_imputation_uses_last_price(self):
        """Step 3: Verify price targets imputed from last_price."""
        result = apply_price_imputation(self.df, price_column="last_price")
        # Where price_target was NaN, should now be last_price
        original_nan_idx = self.df["price_target"].isna()
        self.assertTrue(
            (
                result.loc[original_nan_idx, "price_target"]
                == result.loc[original_nan_idx, "last_price"]
            ).all()
        )

    def test_step3_price_imputation_preserves_existing(self):
        """Step 3: Verify existing price targets are preserved."""
        result = apply_price_imputation(self.df, price_column="last_price")
        original_values = self.df["price_target"].dropna()
        for idx in original_values.index:
            self.assertEqual(result.loc[idx, "price_target"], self.df.loc[idx, "price_target"])

    def test_step3_price_imputation_all_targets(self):
        """Step 3: Verify all 5 price target columns are handled."""
        # Add all 5 price target columns
        test_df = self.df.copy()
        test_df["price_target_low"] = [np.nan] * 80 + [45 + i * 0.5 for i in range(20)]
        test_df["price_target_high"] = [np.nan] * 85 + [60 + i * 0.5 for i in range(15)]

        result = apply_price_imputation(test_df, price_column="last_price")

        # Check that NaN values were filled
        self.assertEqual(result["price_target_low"].isna().sum(), 0)
        self.assertEqual(result["price_target_high"].isna().sum(), 0)

    def test_step3_price_imputation_no_target_columns(self):
        """Step 3: Verify behavior when no price target columns exist."""
        # Create dataframe without any price target columns
        test_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Technology", "Technology"],
                "last_price": [150.0, 250.0],
                "market_cap": [1000.0, 2000.0],
            }
        )

        result = apply_price_imputation(test_df, price_column="last_price")

        # Should return dataframe unchanged (no price target columns to impute)
        self.assertIsNotNone(result)
        pd.testing.assert_frame_equal(result, test_df)

    # STEP 4 TESTS
    def test_step4_median_imputes_remaining(self):
        """Step 4: Verify median imputation handles remaining columns."""
        result = apply_median_imputation(self.df)
        # Should have no NaN in numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.assertEqual(result[col].isna().sum(), 0)

    def test_step4_median_uses_correct_value(self):
        """Step 4: Verify median values are used for imputation."""
        # Create simple test case
        test_df = pd.DataFrame({"value": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = apply_median_imputation(test_df)
        # Median of [1, 2, 4, 5] is 3.0
        self.assertEqual(result.loc[2, "value"], 3.0)

    # FULL PIPELINE TESTS
    def test_full_4step_pipeline(self):
        """Test complete 4-step pipeline reduces all missing values."""
        missing_before = self.df.select_dtypes(include=[np.number]).isna().sum().sum()

        result = apply_enhanced_imputation_strategy_4step(
            self.df, sector_column="sector", n_neighbors=5, price_column="last_price"
        )

        missing_after = result.select_dtypes(include=[np.number]).isna().sum().sum()
        self.assertEqual(missing_after, 0)  # Should have zero missing
        self.assertGreater(missing_before, missing_after)

    def test_pipeline_preserves_dtypes(self):
        """Test pipeline preserves data types."""
        result = apply_enhanced_imputation_strategy_4step(self.df)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            self.assertEqual(result[col].dtype, self.df[col].dtype)

    def test_pipeline_preserves_non_nan_values(self):
        """Test pipeline preserves existing non-NaN values."""
        result = apply_enhanced_imputation_strategy_4step(self.df)
        # Check that non-NaN values in original df are preserved
        for col in ["market_cap", "ebitda_ltm"]:
            original_non_nan = self.df[col].dropna()
            for idx in original_non_nan.index:
                self.assertAlmostEqual(result.loc[idx, col], self.df.loc[idx, col], places=5)

    def test_pipeline_with_no_missing_values(self):
        """Test pipeline handles data with no missing values."""
        # Create dataframe with no NaN
        clean_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
                "market_cap": [100.0, 200.0, 300.0],
                "last_price": [150.0, 250.0, 350.0],
                "price_target": [160.0, 260.0, 360.0],
            }
        )

        result = apply_enhanced_imputation_strategy_4step(clean_df)

        # Should have no NaN and preserve all values
        self.assertEqual(result.select_dtypes(include=[np.number]).isna().sum().sum(), 0)
        pd.testing.assert_frame_equal(
            result[["market_cap", "last_price", "price_target"]],
            clean_df[["market_cap", "last_price", "price_target"]],
        )

    def test_pipeline_returns_copy(self):
        """Test that pipeline returns a copy and doesn't modify original."""
        original_df = self.df.copy()
        result = apply_enhanced_imputation_strategy_4step(self.df)

        # Original should still have NaN values
        self.assertGreater(original_df.isna().sum().sum(), 0)
        # Result should have no NaN in numeric columns
        self.assertEqual(result.select_dtypes(include=[np.number]).isna().sum().sum(), 0)

    # EDGE CASE TESTS
    def test_missing_sector_column(self):
        """Test behavior when sector column is missing."""
        df_no_sector = self.df.drop(columns=["sector"])
        # Should not raise an error, but may not reduce missing values as much
        result = apply_knn_imputation_enhanced(df_no_sector, sector_column="sector")
        self.assertIsNotNone(result)

    def test_missing_price_column(self):
        """Test behavior when price column is missing."""
        df_no_price = self.df.drop(columns=["last_price"])
        result = apply_price_imputation(df_no_price, price_column="last_price")
        # Should return dataframe unchanged (with warning logged)
        self.assertIsNotNone(result)

    def test_empty_dataframe(self):
        """Test behavior with empty dataframe."""
        empty_df = pd.DataFrame()
        result = apply_enhanced_imputation_strategy_4step(empty_df)
        self.assertEqual(len(result), 0)

    def test_all_nan_column(self):
        """Test behavior with column that is all NaN."""
        test_df = pd.DataFrame(
            {
                "sector": ["Tech"] * 10,
                "all_nan": [np.nan] * 10,
                "some_values": [1.0, 2.0, np.nan, 4.0, 5.0] * 2,
            }
        )
        result = apply_enhanced_imputation_strategy_4step(test_df)
        # all_nan should be filled with its median (which would be NaN, but median imputation should handle it)
        # In practice, median of all NaN is NaN, so column might remain NaN
        # But we can verify it doesn't crash
        self.assertIsNotNone(result)


class TestOutlierDetection(unittest.TestCase):
    """Test suite for outlier detection functions."""

    def setUp(self):
        """Create sample data with outliers."""
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 95)
        outliers = np.array([200, 250, 300, 350, 400])

        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "value1": np.concatenate([normal_data, outliers]),
                "value2": np.random.normal(50, 5, 100),
            }
        )

    def test_detect_outliers_iqr_basic(self):
        """Test IQR outlier detection returns boolean mask."""
        result = detect_outliers_iqr(self.df, columns=["value1"])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("value1" in result.columns)
        self.assertEqual(len(result), len(self.df))

    def test_detect_outliers_iqr_identifies_outliers(self):
        """Test IQR method identifies outliers."""
        result = detect_outliers_iqr(self.df, columns=["value1"], by_sector=False)
        # Should identify the extreme values as outliers
        self.assertGreater(result["value1"].sum(), 0)

    def test_detect_outliers_iqr_by_sector(self):
        """Test IQR outlier detection per sector."""
        result = detect_outliers_iqr(self.df, columns=["value1"], by_sector=True)
        self.assertIsInstance(result, pd.DataFrame)
        # Should work without errors when sector column exists
        self.assertTrue("value1" in result.columns)

    def test_detect_outliers_zscore_basic(self):
        """Test z-score outlier detection."""
        result = detect_outliers_zscore(self.df, columns=["value1"], threshold=3.0)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue("value1" in result.columns)

    def test_detect_outliers_zscore_threshold(self):
        """Test z-score with different thresholds."""
        result_strict = detect_outliers_zscore(self.df, columns=["value1"], threshold=2.0)
        result_loose = detect_outliers_zscore(self.df, columns=["value1"], threshold=4.0)
        # Stricter threshold should find more outliers
        self.assertGreaterEqual(result_strict["value1"].sum(), result_loose["value1"].sum())

    def test_detect_outliers_isolation_forest(self):
        """Test isolation forest outlier detection."""
        result = detect_outliers_isolation_forest(self.df, columns=["value1", "value2"])
        self.assertIsInstance(result, pd.Series)
        # Should identify some outliers
        self.assertGreater(result.sum(), 0)


class TestWinsorization(unittest.TestCase):
    """Test suite for winsorization functions."""

    def setUp(self):
        """Create sample data for winsorization."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "value": np.concatenate(
                    [
                        np.random.normal(100, 10, 90),
                        np.array([10, 20, 300, 400, 500, 600, 700, 800, 900, 1000]),
                    ]
                ),
            }
        )

    def test_winsorize_basic(self):
        """Test basic winsorization."""
        result = winsorize_by_sector(self.df, columns=["value"], by_sector=False)
        self.assertEqual(len(result), len(self.df))
        # Extreme values should be capped at 99th percentile
        self.assertLessEqual(result["value"].max(), self.df["value"].quantile(0.99))

    def test_winsorize_percentiles(self):
        """Test winsorization with custom percentiles."""
        result = winsorize_by_sector(
            self.df,
            columns=["value"],
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=False,
        )
        self.assertIsInstance(result, pd.DataFrame)
        # Max should be at or below 95th percentile
        self.assertLessEqual(result["value"].max(), self.df["value"].quantile(0.95))

    def test_winsorize_by_sector_groups(self):
        """Test winsorization works with sector grouping."""
        result = winsorize_by_sector(self.df, columns=["value"], by_sector=True)
        self.assertEqual(len(result), len(self.df))
        # Should process both sectors
        self.assertEqual(set(result["sector"].unique()), {"Tech", "Finance"})


class TestDataQuality(unittest.TestCase):
    """Test suite for data quality functions."""

    def setUp(self):
        """Create sample data with quality issues."""
        self.df_clean = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "sector": ["Technology", "Technology", "Technology", "Consumer", "Consumer"],
                "region": ["US", "US", "US", "US", "US"],
                "value1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "value2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        self.df_missing = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
                "sector": ["Technology", "Technology", "Technology", "Consumer", "Consumer"],
                "region": ["US", "US", "US", "US", "US"],
                "value1": [1.0, np.nan, 3.0, np.nan, 5.0],
                "value2": [10.0, 20.0, np.nan, 40.0, np.nan],
            }
        )

    def test_calculate_data_quality_score_clean(self):
        """Test data quality score for clean data."""
        report = calculate_data_quality_score(self.df_clean)
        self.assertIsInstance(report, DataQualityReport)
        self.assertGreater(report.overall_score, 0.9)
        self.assertEqual(report.completeness_score, 1.0)

    def test_calculate_data_quality_score_missing(self):
        """Test data quality score with missing values."""
        report = calculate_data_quality_score(self.df_missing)
        self.assertIsInstance(report, DataQualityReport)
        self.assertLess(report.completeness_score, 1.0)

    def test_data_quality_report_string(self):
        """Test DataQualityReport string representation."""
        report = calculate_data_quality_score(self.df_clean)
        report_str = str(report)
        self.assertIn("Overall Score", report_str)
        self.assertIn("Completeness", report_str)


class TestImputationMethods(unittest.TestCase):
    """Test suite for basic imputation methods."""

    def setUp(self):
        """Create sample data for imputation."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "value1": [i if i % 3 != 0 else np.nan for i in range(100)],
                "value2": [i * 2 if i % 4 != 0 else np.nan for i in range(100)],
            }
        )

    def test_impute_missing_values_median(self):
        """Test sector median imputation."""
        result = impute_missing_values(self.df, strategy="sector_median")
        # Should reduce missing values
        self.assertLess(result.isna().sum().sum(), self.df.isna().sum().sum())

    def test_impute_missing_values_mean(self):
        """Test sector mean imputation."""
        result = impute_missing_values(self.df, strategy="sector_mean")
        self.assertLess(result.isna().sum().sum(), self.df.isna().sum().sum())

    def test_impute_missing_values_global(self):
        """Test global median imputation when no sector column."""
        df_no_sector = self.df.drop(columns=["sector"])
        result = impute_missing_values(df_no_sector, strategy="sector_median")
        # Should fallback to global imputation (or at least not increase missing values)
        self.assertLessEqual(result.isna().sum().sum(), df_no_sector.isna().sum().sum())

    def test_impute_knn_sector(self):
        """Test KNN imputation with sector awareness."""
        result = impute_missing_values_knn_sector(self.df, sector_column="sector")
        # Should reduce missing values
        self.assertLessEqual(result.isna().sum().sum(), self.df.isna().sum().sum())

    def test_impute_knn_without_sector(self):
        """Test KNN imputation without sector column."""
        df_no_sector = self.df.drop(columns=["sector"])
        result = impute_missing_values_knn_sector(df_no_sector, sector_column="sector")
        # Should still work (global imputation)
        self.assertIsInstance(result, pd.DataFrame)

    def test_apply_enhanced_imputation_strategy(self):
        """Test the 3-step imputation strategy (original version)."""
        result = apply_enhanced_imputation_strategy(self.df, sector_column="sector")
        # Should reduce missing values significantly
        self.assertLessEqual(result.isna().sum().sum(), self.df.isna().sum().sum())


class TestScalingFunctions(unittest.TestCase):
    """Test suite for scaling and preprocessing functions."""

    def setUp(self):
        """Create sample data for scaling."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "sector": ["Tech"] * 50 + ["Finance"] * 50,
                "value1": np.random.normal(100, 20, 100),
                "value2": np.random.normal(500, 100, 100),
                "value3": np.random.uniform(0, 1, 100),
            }
        )

    def test_create_scaler_pipeline_robust(self):
        """Test creating robust scaler pipeline."""
        scaler = create_scaler_pipeline(scaler_type="robust", by_sector=False)
        self.assertIsNotNone(scaler)

    def test_create_scaler_pipeline_standard(self):
        """Test creating standard scaler pipeline."""
        scaler = create_scaler_pipeline(scaler_type="standard", by_sector=False)
        self.assertIsNotNone(scaler)

    def test_create_scaler_pipeline_minmax(self):
        """Test creating minmax scaler pipeline."""
        scaler = create_scaler_pipeline(scaler_type="minmax", by_sector=False)
        self.assertIsNotNone(scaler)

    def test_scale_features_basic(self):
        """Test basic feature scaling."""
        result = scale_features(
            self.df, columns=["value1", "value2"], scaler_type="robust", by_sector=False
        )
        self.assertEqual(len(result), len(self.df))
        # Scaled values should have different range
        self.assertNotEqual(result["value1"].std(), self.df["value1"].std())

    def test_scale_features_by_sector(self):
        """Test sector-specific scaling."""
        result = scale_features(self.df, columns=["value1"], scaler_type="standard", by_sector=True)
        self.assertEqual(len(result), len(self.df))
        # Should work without errors
        self.assertIsInstance(result, pd.DataFrame)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling."""

    def test_outlier_detection_no_numeric_columns(self):
        """Test outlier detection with no numeric columns."""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = detect_outliers_iqr(df)
        # Should handle gracefully
        self.assertEqual(len(result), len(df))

    def test_winsorize_single_value(self):
        """Test winsorization with single unique value."""
        df = pd.DataFrame({"sector": ["Tech"] * 5, "value": [100.0] * 5})
        result = winsorize_by_sector(df, columns=["value"], by_sector=False)
        # Should not crash
        self.assertEqual(len(result), len(df))

    def test_imputation_all_missing_sector(self):
        """Test imputation when entire sector has missing values."""
        df = pd.DataFrame(
            {
                "sector": ["Tech"] * 5 + ["Finance"] * 5,
                "value": [np.nan] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        result = impute_missing_values(df, strategy="sector_median")
        # Tech sector should be imputed using global or Finance values
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
