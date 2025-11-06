"""
Tests for Phase 9.1 - Advanced Data Loading and Preprocessing

This module tests enhanced preprocessing capabilities including:
- Outlier detection (IQR, Z-score, Isolation Forest)
- Winsorization
- Data quality scoring
- Temporal validation
- Advanced imputation strategies
"""

import unittest

import numpy as np
import pandas as pd

from finance_ml.data import (
    detect_outliers_iqr_advanced,
    detect_outliers_by_sector,
    detect_outliers_zscore,
    winsorize_column,
    winsorize_by_sector,
    calculate_completeness_score,
    calculate_consistency_score,
    impute_by_sector,
    safe_divide,
    create_temporal_split,
    create_expanding_windows,
    )


class TestOutlierDetection(unittest.TestCase):
    """Test suite for outlier detection methods"""

    def setUp(self):
        """Create sample financial data for testing"""
        np.random.seed(42)
        n = 100

        self.df = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n),
                "market_cap": np.random.lognormal(10, 2, n),
                "p_e": np.random.uniform(5, 50, n),
                "revenue": np.random.lognormal(8, 1.5, n),
                "net_income": np.random.normal(100, 50, n),
            }
        )

        # Add some outliers
        self.df.loc[0, "market_cap"] = 1e12  # Extreme outlier
        self.df.loc[1, "p_e"] = 500  # Extreme P/E
        self.df.loc[2, "net_income"] = -1000  # Large loss

    def test_detect_outliers_iqr_basic(self):
        """Test IQR outlier detection on single column"""
        outliers = detect_outliers_iqr_advanced(self.df, columns=["market_cap"])

        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertEqual(len(outliers.columns), 1)
        self.assertIn("market_cap", outliers.columns)

        # Should detect the extreme market_cap outlier
        self.assertTrue(outliers.loc[0, "market_cap"])

    def test_detect_outliers_iqr_multiple_columns(self):
        """Test IQR detection across multiple columns"""
        outliers = detect_outliers_iqr_advanced(
            self.df, columns=["market_cap", "p_e", "net_income"]
        )

        self.assertEqual(outliers.shape, (len(self.df), 3))
        self.assertTrue(outliers.loc[0, "market_cap"])
        self.assertTrue(outliers.loc[1, "p_e"])

    def test_detect_outliers_iqr_multiplier(self):
        """Test IQR with custom multiplier"""
        # More lenient outlier detection
        outliers_lenient = detect_outliers_iqr_advanced(self.df, columns=["p_e"], multiplier=3.0)

        # Stricter outlier detection
        outliers_strict = detect_outliers_iqr_advanced(self.df, columns=["p_e"], multiplier=1.0)

        # Strict should find more outliers
        self.assertGreaterEqual(outliers_strict["p_e"].sum(), outliers_lenient["p_e"].sum())

    def test_detect_outliers_by_sector(self):
        """Test sector-specific outlier detection"""
        outliers = detect_outliers_by_sector(self.df, columns=["market_cap", "p_e"])

        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertEqual(outliers.shape, (len(self.df), 2))

        # Should detect outliers within each sector
        self.assertIn("market_cap", outliers.columns)
        self.assertIn("p_e", outliers.columns)

    def test_detect_outliers_empty_data(self):
        """Test outlier detection with empty DataFrame"""
        empty_df = pd.DataFrame(columns=["market_cap", "p_e"])
        outliers = detect_outliers_iqr_advanced(empty_df, columns=["market_cap"])

        self.assertEqual(len(outliers), 0)

    def test_detect_outliers_missing_column(self):
        """Test behavior when column doesn't exist"""
        with self.assertRaises(KeyError):
            detect_outliers_iqr_advanced(self.df, columns=["nonexistent_column"])


class TestZScoreOutlierDetection(unittest.TestCase):
    """Test suite for Z-score based outlier detection"""

    def setUp(self):
        """Create sample data"""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "value": np.concatenate(
                    [
                        np.random.normal(100, 10, 95),  # Normal data
                        [200, 250, 300, 0, -50],  # Outliers
                    ]
                )
            }
        )

    def test_zscore_outlier_detection_basic(self):
        """Test basic Z-score outlier detection"""
        outliers = detect_outliers_zscore(self.df, columns=["value"])

        self.assertIsInstance(outliers, pd.DataFrame)
        self.assertTrue(any(outliers["value"]))  # Should find some outliers

    def test_zscore_outlier_detection_threshold(self):
        """Test Z-score with custom threshold"""
        outliers_strict = detect_outliers_zscore(self.df, columns=["value"], threshold=2.0)
        outliers_lenient = detect_outliers_zscore(self.df, columns=["value"], threshold=4.0)

        # Strict threshold should find more outliers
        self.assertGreaterEqual(outliers_strict["value"].sum(), outliers_lenient["value"].sum())


class TestWinsorization(unittest.TestCase):
    """Test suite for winsorization (capping extreme values)"""

    def setUp(self):
        """Create sample data with extreme values"""
        np.random.seed(42)
        self.df = pd.DataFrame({"price": np.random.uniform(10, 100, 100)})
        # Add extreme values
        self.df.loc[0, "price"] = 1000
        self.df.loc[99, "price"] = 1

    def test_winsorize_basic(self):
        """Test basic winsorization"""
        winsorized = winsorize_column(self.df["price"])

        # Check that extreme values are capped
        self.assertLess(winsorized.max(), self.df["price"].max())
        self.assertGreater(winsorized.min(), self.df["price"].min())

    def test_winsorize_by_sector(self):
        """Test sector-specific winsorization"""
        # Add sector column
        self.df["sector"] = np.random.choice(["Tech", "Finance"], len(self.df))

        # Apply sector-specific winsorization
        df_winsorized = winsorize_by_sector(self.df, columns=["price"])

        # Check that the output has the same shape
        self.assertEqual(df_winsorized.shape, self.df.shape)

        # Check that extreme values were capped
        self.assertLessEqual(df_winsorized["price"].max(), self.df["price"].max())


class TestDataQualityScoring(unittest.TestCase):
    """Test suite for data quality assessment"""

    def setUp(self):
        """Create data with varying quality"""
        self.df_good = pd.DataFrame(
            {
                "col1": range(100),
                "col2": range(100, 200),
            }
        )

        self.df_poor = pd.DataFrame(
            {
                "col1": [np.nan] * 50 + list(range(50)),
                "col2": [None, 1, 2] * 33 + [3],
            }
        )

    def test_data_quality_score_completeness(self):
        """Test completeness score calculation"""
        score_good = calculate_completeness_score(self.df_good)
        score_poor = calculate_completeness_score(self.df_poor)

        self.assertAlmostEqual(score_good, 100.0)
        self.assertLess(score_poor, 80.0)

    def test_data_quality_score_consistency(self):
        """Test data consistency checks"""
        df = pd.DataFrame(
            {
                "market_cap": [1000, 2000, -500, 3000],  # Negative is inconsistent
                "revenue": [100, 200, 300, 400],
            }
        )

        consistency_result = calculate_consistency_score(df)

        # Should report issues
        self.assertIsInstance(consistency_result, dict)
        self.assertIn("score", consistency_result)
        self.assertIn("issues", consistency_result)

        # Should detect the negative market cap
        self.assertGreater(len(consistency_result["issues"]), 0)


class TestTemporalValidation(unittest.TestCase):
    """Test suite for temporal validation and time-aware splits"""

    def setUp(self):
        """Create time-series financial data"""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        self.df = pd.DataFrame(
            {
                "date": dates,
                "ticker": "AAPL",
                "price": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1e6, 5e6, 100),
            }
        )

    def test_temporal_sort_validation(self):
        """Test that data is sorted by date"""
        # Shuffle the data
        df_shuffled = self.df.sample(frac=1, random_state=42)

        # Sort by date
        df_sorted = df_shuffled.sort_values("date")

        # Check that dates are in ascending order
        self.assertTrue((df_sorted["date"].diff()[1:] >= pd.Timedelta(0)).all())

    def test_no_future_data_leakage(self):
        """Test that training data doesn't include future information"""
        split_date = pd.Timestamp("2020-02-01")
        train, test = create_temporal_split(self.df, "date", split_date)

        # Ensure no overlap
        self.assertTrue((train["date"] <= split_date).all())
        self.assertTrue((test["date"] > split_date).all())
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)

    def test_expanding_window_cv(self):
        """Test expanding window cross-validation"""
        splits = create_expanding_windows(self.df, "date", n_splits=3)

        self.assertEqual(len(splits), 3)

        # Check that each split is valid
        for train_mask, test_mask in splits:
            self.assertGreater(train_mask.sum(), 0)
            self.assertGreater(test_mask.sum(), 0)
            # No overlap
            self.assertFalse((train_mask & test_mask).any())


class TestAdvancedImputation(unittest.TestCase):
    """Test suite for advanced missing value imputation"""

    def setUp(self):
        """Create data with missing values"""
        np.random.seed(42)
        n = 100

        self.df = pd.DataFrame(
            {
                "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n),
                "revenue": np.random.lognormal(8, 1, n),
                "profit_margin": np.random.uniform(0.05, 0.25, n),
                "debt_ratio": np.random.uniform(0.1, 0.8, n),
            }
        )

        # Introduce missing values
        missing_indices = np.random.choice(n, size=20, replace=False)
        self.df.loc[missing_indices, "profit_margin"] = np.nan

    def test_sector_specific_median_imputation(self):
        """Test imputation using sector-specific median"""
        df_imputed = impute_by_sector(self.df, "profit_margin", "sector", method="median")

        # Check that all missing values are filled
        self.assertEqual(df_imputed["profit_margin"].isnull().sum(), 0)

    def test_sector_specific_mean_imputation(self):
        """Test imputation using sector-specific mean"""
        df_imputed = impute_by_sector(self.df, "profit_margin", "sector", method="mean")

        # Check that all missing values are filled
        self.assertEqual(df_imputed["profit_margin"].isnull().sum(), 0)

    def test_knn_imputation_placeholder(self):
        """Test KNN-based imputation (placeholder)"""
        # KNN imputation will be implemented with scikit-learn
        # For now, just test that we have the data structure

        self.assertTrue("profit_margin" in self.df.columns)
        self.assertGreater(self.df["profit_margin"].isnull().sum(), 0)


class TestFinancialRatioHandling(unittest.TestCase):
    """Test suite for handling financial ratio edge cases"""

    def setUp(self):
        """Create data with edge cases for ratio calculations"""
        self.df = pd.DataFrame(
            {
                "market_cap": [1000, 2000, 3000, 4000],
                "book_value": [500, 0, -100, 1000],  # Zero and negative cases
                "ebitda": [100, 200, 0, 300],  # Zero case
                "revenue": [1000, 2000, 3000, 0],  # Zero case
            }
        )

    def test_safe_ratio_calculation(self):
        """Test safe division for ratio calculation"""
        p_b = safe_divide(self.df["market_cap"], self.df["book_value"])

        # Check that division by zero is handled
        self.assertTrue(np.isnan(p_b.iloc[1]))  # book_value = 0

        # Check that normal divisions work
        self.assertIsInstance(p_b, pd.Series)
        self.assertAlmostEqual(p_b.iloc[0], 1000 / 500, places=2)

    def test_handle_negative_denominators(self):
        """Test handling of negative denominators in ratios"""

        # P/B ratio with negative book value should be handled
        p_b = self.df["market_cap"] / self.df["book_value"].replace(0, np.nan)

        # Negative book value case
        self.assertLess(p_b.iloc[2], 0)

    def test_infinite_ratio_handling(self):
        """Test handling of infinite ratios"""

        ratios = self.df["market_cap"] / self.df["book_value"]

        # Replace infinities
        ratios_clean = ratios.replace([np.inf, -np.inf], np.nan)

        self.assertTrue(np.isfinite(ratios_clean.dropna()).all())


class TestEnhancedImputationStrategy(unittest.TestCase):
    """Test suite for Phase 9.1 enhanced imputation strategy."""
    
    def setUp(self):
        """Create sample financial dataset with missing values."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'] * 20,
            'sector': ['Technology', 'Technology', 'Technology', 
                      'Consumer Discretionary', 'Consumer Discretionary'] * 20,
            # Zero imputation columns
            'impairment_of_goodwill_fq': [np.nan] * 50 + [1000.0] * 50,
            'restructuring_charges_ltm': [np.nan] * 60 + [500.0] * 40,
            'cash_acquisitions_fy': [np.nan] * 70 + [2000.0] * 30,
            'asset_writedown_ltm': [np.nan] * 80 + [300.0] * 20,
            'r_d_expenses_ltm': [np.nan] * 30 + [1500.0] * 70,
            # KNN imputation columns
            'market_cap': [100 + np.random.randn() * 10 if i % 3 != 0 else np.nan 
                          for i in range(100)],
            'enterprise_value': [120 + np.random.randn() * 15 if i % 4 != 0 else np.nan 
                                for i in range(100)],
            'last_price': [50 + np.random.randn() * 5 if i % 5 != 0 else np.nan 
                          for i in range(100)],
            'p_e_ltm': [15 + np.random.randn() * 3 if i % 6 != 0 else np.nan 
                       for i in range(100)],
            'ebitda_ltm': [10 + np.random.randn() * 2 if i % 7 != 0 else np.nan 
                          for i in range(100)],
        })
    
    def test_get_zero_imputation_columns(self):
        """Test retrieval of zero imputation column list."""
        from finance_ml.advanced_preprocessing import get_zero_imputation_columns
        
        zero_cols = get_zero_imputation_columns()
        
        # Verify it returns a list
        self.assertIsInstance(zero_cols, list)
        # Verify key columns are present
        self.assertIn('impairment_of_goodwill_fq', zero_cols)
        self.assertIn('restructuring_charges_ltm', zero_cols)
        self.assertIn('cash_acquisitions_fy', zero_cols)
        # Verify minimum expected count (adjust based on your schema)
        self.assertGreaterEqual(len(zero_cols), 30)
    
    def test_get_knn_imputation_columns(self):
        """Test retrieval of KNN imputation column list."""
        from finance_ml.advanced_preprocessing import get_knn_imputation_columns
        
        knn_cols = get_knn_imputation_columns()
        
        # Verify it returns a list
        self.assertIsInstance(knn_cols, list)
        # Verify key columns are present
        self.assertIn('market_cap', knn_cols)
        self.assertIn('enterprise_value', knn_cols)
        self.assertIn('last_price', knn_cols)
        self.assertIn('p_e_ltm', knn_cols)
        # Verify minimum expected count
        self.assertGreaterEqual(len(knn_cols), 100)
    
    def test_apply_zero_imputation_basic(self):
        """Test basic zero imputation for exceptional event columns."""
        from finance_ml.advanced_preprocessing import apply_zero_imputation
        
        result = apply_zero_imputation(
            self.df, 
            columns=['impairment_of_goodwill_fq', 'restructuring_charges_ltm']
        )
        
        # Verify NaN values are replaced with 0
        self.assertEqual(result['impairment_of_goodwill_fq'].isna().sum(), 0)
        self.assertEqual(result['restructuring_charges_ltm'].isna().sum(), 0)
        # Verify non-NaN values are preserved
        self.assertGreater(result['impairment_of_goodwill_fq'].sum(), 0)
        # Verify original dataframe is not modified
        self.assertGreater(self.df['impairment_of_goodwill_fq'].isna().sum(), 0)
    
    def test_apply_zero_imputation_auto_detect(self):
        """Test automatic detection and imputation of zero-fill columns."""
        from finance_ml.advanced_preprocessing import apply_zero_imputation
        
        # Call without specifying columns (auto-detect)
        result = apply_zero_imputation(self.df)
        
        # Verify known zero-imputation columns have no NaN
        zero_cols_present = ['impairment_of_goodwill_fq', 'restructuring_charges_ltm', 
                            'cash_acquisitions_fy', 'asset_writedown_ltm']
        for col in zero_cols_present:
            if col in result.columns:
                self.assertEqual(result[col].isna().sum(), 0, 
                               f"Column {col} should have no NaN after zero imputation")
    
    def test_apply_knn_imputation_with_sector(self):
        """Test KNN imputation with sector awareness."""
        from finance_ml.advanced_preprocessing import apply_knn_imputation_enhanced
        
        result = apply_knn_imputation_enhanced(
            self.df,
            columns=['market_cap', 'enterprise_value', 'last_price'],
            sector_column='sector',
            n_neighbors=3
        )
        
        # Verify NaN values are reduced (should be 0 if enough neighbors)
        for col in ['market_cap', 'enterprise_value', 'last_price']:
            self.assertLessEqual(result[col].isna().sum(), 
                               self.df[col].isna().sum())
    
    def test_apply_enhanced_imputation_strategy_full_pipeline(self):
        """Test complete two-step imputation pipeline."""
        from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy
        
        result = apply_enhanced_imputation_strategy(
            self.df,
            sector_column='sector',
            n_neighbors=5
        )
        
        # Verify zero-imputation columns have no NaN
        zero_cols_present = [col for col in ['impairment_of_goodwill_fq', 
                                              'restructuring_charges_ltm'] 
                            if col in result.columns]
        for col in zero_cols_present:
            self.assertEqual(result[col].isna().sum(), 0)
        
        # Verify KNN-imputation columns have reduced NaN
        knn_cols_present = [col for col in ['market_cap', 'enterprise_value'] 
                           if col in result.columns]
        for col in knn_cols_present:
            self.assertLessEqual(result[col].isna().sum(), self.df[col].isna().sum())
    
    def test_imputation_preserves_dtypes(self):
        """Test that imputation preserves numeric dtypes."""
        from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy
        
        original_dtypes = self.df.dtypes
        result = apply_enhanced_imputation_strategy(self.df)
        
        for col in result.select_dtypes(include=[np.number]).columns:
            if col in original_dtypes:
                self.assertEqual(result[col].dtype, original_dtypes[col])
    
    def test_imputation_edge_cases(self):
        """Test imputation with edge cases."""
        from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy
        
        # Test with all NaN column
        df_edge = self.df.copy()
        df_edge['all_nan_col'] = np.nan
        result = apply_enhanced_imputation_strategy(df_edge)
        
        # Should handle gracefully
        self.assertIn('all_nan_col', result.columns)
    
    def test_imputation_logging(self):
        """Test that imputation logs appropriate information."""
        from finance_ml.advanced_preprocessing import apply_enhanced_imputation_strategy
        import logging
        
        with self.assertLogs(level='INFO') as log:
            apply_enhanced_imputation_strategy(self.df)
            
        # Verify logging occurred
        self.assertTrue(any('imputation' in msg.lower() for msg in log.output))


if __name__ == "__main__":
    unittest.main()
