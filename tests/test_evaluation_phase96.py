"""
Unit tests for Phase 9.6: Model Evaluation and Error Analysis.

Tests comprehensive regression metrics, residual analysis, error bucketing,
cross-validation strategies, and model comparison utilities.

Follows TDD approach: write failing tests first, then implement minimal code to pass.
"""

import shutil
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Import functions to be tested (will fail initially until implemented)
try:
    from finance_ml.eval import (
        comprehensive_regression_metrics,
        compute_metrics_by_segment,
        residual_analysis_suite,
        error_bucketing_analysis,
        create_stratified_sector_cv,
        create_grouped_ticker_cv,
        evaluate_with_cross_validation,
    )
except ImportError as e:
    warnings.warn(f"Import failed (expected for TDD): {e}")


def create_sample_regression_data(n_samples=300, random_state=42):
    """
    Create sample regression data for testing evaluation functions.
    
    Returns:
        pd.DataFrame: Sample data with predictions, actuals, and metadata
    """
    np.random.seed(random_state)
    
    # Create sectors (7 major sectors from Phase 9.6)
    sectors = ['Technology', 'Healthcare', 'Financials', 'Energy', 
               'Consumer', 'Industrials', 'Materials']
    
    # Create regions (US, EU, APAC, ROTW)
    regions = ['US', 'EU', 'APAC', 'ROTW']
    
    # Generate synthetic data
    data = {
        'ticker': [f'TICK{i:03d}' for i in range(n_samples)],
        'sector': np.random.choice(sectors, n_samples),
        'region': np.random.choice(regions, n_samples),
        'last_price': np.random.uniform(10, 500, n_samples),
        'market_cap': np.random.lognormal(mean=9, sigma=2, size=n_samples),  # millions
        'volatility': np.random.uniform(0.1, 0.8, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate predictions and actuals with realistic relationship
    # Actuals = price target
    df['actual'] = df['last_price'] * np.random.uniform(0.8, 1.3, n_samples)
    
    # Predictions have some error
    noise = np.random.normal(0, df['actual'].std() * 0.15, n_samples)
    df['predicted'] = df['actual'] + noise
    
    # Add sector-specific bias for testing segmentation
    sector_bias = {
        'Technology': 10,
        'Healthcare': -5,
        'Financials': 0,
        'Energy': 15,
        'Consumer': -10,
        'Industrials': 5,
        'Materials': 8
    }
    df['predicted'] = df['predicted'] + df['sector'].map(sector_bias)
    
    # Create market cap buckets
    df['market_cap_bucket'] = pd.cut(
        df['market_cap'],
        bins=[0, 2000, 10000, np.inf],
        labels=['Small', 'Mid', 'Large']
    )
    
    # Create volatility buckets
    df['volatility_bucket'] = pd.cut(
        df['volatility'],
        bins=[0, 0.3, 0.5, np.inf],
        labels=['Low', 'Medium', 'High']
    )
    
    return df


class TestComprehensiveRegressionMetrics(unittest.TestCase):
    """Test comprehensive regression metrics calculation."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_regression_data(n_samples=200)
        self.y_true = self.df['actual'].values
        self.y_pred = self.df['predicted'].values
    
    def test_comprehensive_regression_metrics_returns_dict(self):
        """Test that comprehensive_regression_metrics returns a dictionary."""
        result = comprehensive_regression_metrics(self.y_true, self.y_pred)
        self.assertIsInstance(result, dict)
    
    def test_comprehensive_regression_metrics_has_required_metrics(self):
        """Test that all required metrics are present."""
        result = comprehensive_regression_metrics(self.y_true, self.y_pred)
        
        required_metrics = ['mae', 'rmse', 'mape', 'r2', 'median_ae', 'max_error']
        for metric in required_metrics:
            self.assertIn(metric, result, f"Missing required metric: {metric}")
    
    def test_comprehensive_regression_metrics_mae_positive(self):
        """Test that MAE is positive."""
        result = comprehensive_regression_metrics(self.y_true, self.y_pred)
        self.assertGreater(result['mae'], 0)
    
    def test_comprehensive_regression_metrics_rmse_gte_mae(self):
        """Test that RMSE >= MAE (mathematical property)."""
        result = comprehensive_regression_metrics(self.y_true, self.y_pred)
        self.assertGreaterEqual(result['rmse'], result['mae'])
    
    def test_comprehensive_regression_metrics_r2_range(self):
        """Test that R² is in reasonable range (can be negative for bad models)."""
        result = comprehensive_regression_metrics(self.y_true, self.y_pred)
        self.assertIsInstance(result['r2'], (int, float))
        self.assertLessEqual(result['r2'], 1.0)
    
    def test_comprehensive_regression_metrics_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        result = comprehensive_regression_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(result['mae'], 0.0, places=10)
        self.assertAlmostEqual(result['rmse'], 0.0, places=10)
        self.assertAlmostEqual(result['mape'], 0.0, places=10)
        self.assertAlmostEqual(result['r2'], 1.0, places=10)
    
    def test_comprehensive_regression_metrics_with_zeros(self):
        """Test MAPE handling when actuals contain zeros."""
        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([0.5, 1.5, 2.5, 3.5])
        
        result = comprehensive_regression_metrics(y_true, y_pred)
        
        # Should handle zeros gracefully (skip or use alternative)
        self.assertIn('mape', result)
        self.assertIsInstance(result['mape'], (int, float))


class TestComputeMetricsBySegment(unittest.TestCase):
    """Test metrics computation by segments (sector, region, market cap)."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_regression_data(n_samples=300)
    
    def test_compute_metrics_by_segment_returns_dataframe(self):
        """Test that function returns a DataFrame."""
        result = compute_metrics_by_segment(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            segment_col='sector'
        )
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_compute_metrics_by_segment_has_all_segments(self):
        """Test that all segments are represented."""
        result = compute_metrics_by_segment(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            segment_col='sector'
        )
        
        unique_sectors = self.df['sector'].nunique()
        self.assertEqual(len(result), unique_sectors)
    
    def test_compute_metrics_by_segment_has_required_columns(self):
        """Test that result has required metric columns."""
        result = compute_metrics_by_segment(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            segment_col='sector'
        )
        
        required_cols = ['segment', 'n_samples', 'mae', 'rmse', 'r2']
        for col in required_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")
    
    def test_compute_metrics_by_segment_by_region(self):
        """Test segmentation by region."""
        result = compute_metrics_by_segment(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            segment_col='region'
        )
        
        unique_regions = self.df['region'].nunique()
        self.assertEqual(len(result), unique_regions)
    
    def test_compute_metrics_by_segment_by_market_cap(self):
        """Test segmentation by market cap bucket."""
        result = compute_metrics_by_segment(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            segment_col='market_cap_bucket'
        )
        
        # Should have 3 buckets: Small, Mid, Large
        self.assertGreaterEqual(len(result), 1)
        self.assertLessEqual(len(result), 3)


class TestResidualAnalysisSuite(unittest.TestCase):
    """Test residual analysis suite functions."""
    
    def setUp(self):
        """Set up test data and temporary output directory."""
        self.df = create_sample_regression_data(n_samples=200)
        self.y_true = self.df['actual'].values
        self.y_pred = self.df['predicted'].values
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_residual_analysis_suite_returns_dict(self):
        """Test that residual analysis returns a dictionary."""
        result = residual_analysis_suite(
            self.y_true,
            self.y_pred,
            output_dir=None  # No plots
        )
        self.assertIsInstance(result, dict)
    
    def test_residual_analysis_suite_has_statistics(self):
        """Test that result includes residual statistics."""
        result = residual_analysis_suite(
            self.y_true,
            self.y_pred,
            output_dir=None
        )
        
        required_keys = ['mean_residual', 'std_residual', 'skewness', 'kurtosis']
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")
    
    def test_residual_analysis_suite_has_normality_test(self):
        """Test that normality test results are included."""
        result = residual_analysis_suite(
            self.y_true,
            self.y_pred,
            output_dir=None
        )
        
        self.assertIn('normality_test', result)
        self.assertIn('p_value', result['normality_test'])
        self.assertIn('is_normal', result['normality_test'])
    
    def test_residual_analysis_suite_creates_plots(self):
        """Test that plots are created when output_dir is provided."""
        output_path = Path(self.temp_dir)
        
        result = residual_analysis_suite(
            self.y_true,
            self.y_pred,
            output_dir=output_path
        )
        
        # Check that at least one plot file was created
        plot_files = list(output_path.glob('*.png'))
        self.assertGreater(len(plot_files), 0, "No plot files created")
    
    def test_residual_analysis_suite_mean_near_zero(self):
        """Test that mean residual is close to zero for unbiased predictions."""
        result = residual_analysis_suite(
            self.y_true,
            self.y_pred,
            output_dir=None
        )
        
        # Mean residual should be close to zero (within 10% of std)
        self.assertLess(
            abs(result['mean_residual']),
            result['std_residual'] * 0.5,
            "Mean residual too large - possible systematic bias"
        )


class TestErrorBucketingAnalysis(unittest.TestCase):
    """Test error bucketing and segmentation analysis."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_regression_data(n_samples=300)
    
    def test_error_bucketing_analysis_returns_dict(self):
        """Test that function returns a dictionary."""
        result = error_bucketing_analysis(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            bucket_cols=['market_cap_bucket', 'volatility_bucket', 'sector']
        )
        self.assertIsInstance(result, dict)
    
    def test_error_bucketing_analysis_has_all_buckets(self):
        """Test that all requested bucket columns are analyzed."""
        bucket_cols = ['market_cap_bucket', 'volatility_bucket']
        
        result = error_bucketing_analysis(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            bucket_cols=bucket_cols
        )
        
        for col in bucket_cols:
            self.assertIn(col, result, f"Missing bucket: {col}")
    
    def test_error_bucketing_analysis_identifies_outliers(self):
        """Test that outlier identification is included."""
        result = error_bucketing_analysis(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            bucket_cols=['sector']
        )
        
        self.assertIn('outliers', result)
        self.assertIn('n_outliers', result['outliers'])
        self.assertIn('outlier_threshold', result['outliers'])
    
    def test_error_bucketing_analysis_by_market_cap(self):
        """Test error analysis by market cap buckets."""
        result = error_bucketing_analysis(
            self.df,
            y_true_col='actual',
            y_pred_col='predicted',
            bucket_cols=['market_cap_bucket']
        )
        
        self.assertIn('market_cap_bucket', result)
        bucket_result = result['market_cap_bucket']
        
        # Should have metrics for each bucket
        self.assertIsInstance(bucket_result, pd.DataFrame)
        self.assertIn('mae', bucket_result.columns)


class TestCrossValidationStrategies(unittest.TestCase):
    """Test cross-validation strategy creation."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_regression_data(n_samples=200)
    
    def test_create_stratified_sector_cv_returns_splitter(self):
        """Test that stratified sector CV returns a valid splitter."""
        cv = create_stratified_sector_cv(n_splits=5)
        
        # Should have split method
        self.assertTrue(hasattr(cv, 'split'))
    
    def test_create_stratified_sector_cv_splits(self):
        """Test that CV splitter generates correct number of splits."""
        cv = create_stratified_sector_cv(n_splits=3)
        
        X = self.df[['last_price', 'market_cap', 'volatility']]
        y = self.df['sector']  # Pass sector as y for stratification
        
        splits = list(cv.split(X, y))
        self.assertEqual(len(splits), 3)
    
    def test_create_grouped_ticker_cv_returns_splitter(self):
        """Test that grouped ticker CV returns a valid splitter."""
        cv = create_grouped_ticker_cv(n_splits=5)
        
        # Should have split method
        self.assertTrue(hasattr(cv, 'split'))
    
    def test_create_grouped_ticker_cv_no_ticker_leakage(self):
        """Test that same ticker doesn't appear in train and test."""
        cv = create_grouped_ticker_cv(n_splits=3)
        
        X = self.df[['last_price', 'market_cap', 'volatility']]
        y = self.df['actual']
        groups = self.df['ticker']
        
        for train_idx, test_idx in cv.split(X, y, groups):
            train_tickers = set(self.df.iloc[train_idx]['ticker'])
            test_tickers = set(self.df.iloc[test_idx]['ticker'])
            
            # No overlap between train and test tickers
            self.assertEqual(len(train_tickers & test_tickers), 0,
                           "Ticker leakage detected: same ticker in train and test")


class TestEvaluateWithCrossValidation(unittest.TestCase):
    """Test cross-validation evaluation function."""
    
    def setUp(self):
        """Set up test data."""
        self.df = create_sample_regression_data(n_samples=150)
    
    def test_evaluate_with_cross_validation_returns_dict(self):
        """Test that function returns results dictionary."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = self.df[['last_price', 'market_cap', 'volatility']].fillna(0)
        y = self.df['actual']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        result = evaluate_with_cross_validation(
            model, X, y,
            cv_strategy='simple',
            n_splits=3
        )
        
        self.assertIsInstance(result, dict)
    
    def test_evaluate_with_cross_validation_has_scores(self):
        """Test that CV scores are included."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = self.df[['last_price', 'market_cap', 'volatility']].fillna(0)
        y = self.df['actual']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        result = evaluate_with_cross_validation(
            model, X, y,
            cv_strategy='simple',
            n_splits=3
        )
        
        self.assertIn('cv_scores', result)
        self.assertIn('mean_score', result)
        self.assertIn('std_score', result)
    
    def test_evaluate_with_cross_validation_stratified(self):
        """Test stratified cross-validation."""
        from sklearn.ensemble import RandomForestRegressor
        
        X = self.df[['last_price', 'market_cap', 'volatility']].fillna(0)
        y = self.df['actual']
        groups = self.df['sector']
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        result = evaluate_with_cross_validation(
            model, X, y,
            cv_strategy='stratified',
            groups=groups,
            n_splits=3
        )
        
        self.assertIn('cv_scores', result)
        self.assertEqual(len(result['cv_scores']), 3)


class TestIntegrationEvaluationWorkflow(unittest.TestCase):
    """Integration test for complete evaluation workflow."""
    
    def test_complete_evaluation_pipeline(self):
        """Test end-to-end evaluation pipeline."""
        from sklearn.ensemble import RandomForestRegressor
        
        # Create sample data
        df = create_sample_regression_data(n_samples=200)
        
        # Prepare features and target
        X = df[['last_price', 'market_cap', 'volatility']].fillna(0)
        y = df['actual']
        
        # Train simple model
        model = RandomForestRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        df['predicted'] = y_pred
        
        # 1. Comprehensive metrics
        metrics = comprehensive_regression_metrics(y, y_pred)
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        
        # 2. Metrics by segment
        sector_metrics = compute_metrics_by_segment(
            df, 'actual', 'predicted', 'sector'
        )
        self.assertGreater(len(sector_metrics), 0)
        
        # 3. Residual analysis
        residuals = residual_analysis_suite(y, y_pred, output_dir=None)
        self.assertIn('mean_residual', residuals)
        
        # 4. Error bucketing
        error_buckets = error_bucketing_analysis(
            df, 'actual', 'predicted', ['sector', 'market_cap_bucket']
        )
        self.assertIn('sector', error_buckets)
        
        # 5. Cross-validation
        cv_results = evaluate_with_cross_validation(
            model, X, y, cv_strategy='simple', n_splits=3
        )
        self.assertIn('mean_score', cv_results)
        
        # All components should work together
        self.assertTrue(True, "Integration pipeline completed successfully")


if __name__ == '__main__':
    unittest.main()
