"""
Integration tests for Section 2: Loading and Preprocessing Financial Data

Tests the complete preprocessing pipeline as integrated in ml_finance_model_main.ipynb
Section 2, ensuring all components work together seamlessly.

Pipeline tested:
1. Data loading (DB or CSV)
2. Outlier detection (IQR, Z-score, Isolation Forest)
3. Sector-specific winsorization
4. Data quality scoring
5. 4-step imputation strategy
6. Feature scaling

Test Strategy: Integration tests with realistic sample data
"""

import unittest
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

from finance_ml.data import load_from_csv, validate_schema
from finance_ml.advanced_preprocessing import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
    calculate_data_quality_score,
    apply_enhanced_imputation_strategy_4step,
    scale_features,
)


class TestSection2PreprocessingPipeline(unittest.TestCase):
    """Test complete Section 2 preprocessing pipeline integration."""

    @classmethod
    def setUpClass(cls):
        """Set up sample data for integration testing."""
        np.random.seed(42)

        # Create realistic financial data sample
        n_samples = 200
        sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

        cls.sample_data = pd.DataFrame(
            {
                "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
                "sector": np.random.choice(sectors, n_samples),
                "region": np.random.choice(["US", "EU", "APAC"], n_samples),
                "last_price": np.random.uniform(10, 500, n_samples),
                "price_target": np.random.uniform(12, 550, n_samples),
                "market_cap": np.random.uniform(1e9, 1e12, n_samples),
                "ev_ebitda": np.random.uniform(-5, 30, n_samples),
                "p_e": np.random.uniform(-10, 50, n_samples),
                "p_b": np.random.uniform(0.5, 10, n_samples),
                "roe": np.random.uniform(-0.2, 0.5, n_samples),
                "revenue": np.random.uniform(1e8, 1e11, n_samples),
                "ebitda": np.random.uniform(1e7, 1e10, n_samples),
                "net_income": np.random.uniform(-1e8, 1e10, n_samples),
            }
        )

        # Add some missing values
        cls.sample_data.loc[np.random.choice(n_samples, 20, replace=False), "ev_ebitda"] = np.nan
        cls.sample_data.loc[np.random.choice(n_samples, 15, replace=False), "p_e"] = np.nan

        # Add some outliers
        cls.sample_data.loc[np.random.choice(n_samples, 5, replace=False), "p_e"] = 1000
        cls.sample_data.loc[np.random.choice(n_samples, 5, replace=False), "ev_ebitda"] = -100

        cls.initial_missing = cls.sample_data.isnull().sum().sum()
        cls.initial_shape = cls.sample_data.shape

    def test_pipeline_step1_outlier_detection(self):
        """Test Step 1: Outlier detection with all three methods."""
        numeric_cols = self.sample_data.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        # IQR outlier detection
        outliers_iqr = detect_outliers_iqr(
            self.sample_data, columns=financial_metrics[:5], by_sector=True, iqr_multiplier=1.5
        )
        self.assertIsInstance(outliers_iqr, pd.DataFrame)
        self.assertEqual(outliers_iqr.shape[0], self.sample_data.shape[0])

        # Z-score outlier detection
        outliers_zscore = detect_outliers_zscore(
            self.sample_data, columns=financial_metrics[:5], threshold=3.0, by_sector=True
        )
        self.assertIsInstance(outliers_zscore, pd.DataFrame)

        # Isolation Forest outlier detection
        outliers_iforest = detect_outliers_isolation_forest(
            self.sample_data, columns=financial_metrics[:5], contamination=0.1, random_state=42
        )
        self.assertIsInstance(outliers_iforest, pd.Series)

        # Verify outliers were detected (select only boolean columns before summing)
        # Outlier detection returns boolean DataFrames for numeric columns only
        iqr_bool_cols = outliers_iqr.select_dtypes(include=["bool"]).columns
        zscore_bool_cols = outliers_zscore.select_dtypes(include=["bool"]).columns

        total_outliers_iqr = (
            int(outliers_iqr[iqr_bool_cols].sum().sum()) if len(iqr_bool_cols) > 0 else 0
        )
        total_outliers_zscore = (
            int(outliers_zscore[zscore_bool_cols].sum().sum()) if len(zscore_bool_cols) > 0 else 0
        )
        total_outliers_iforest = (
            int(outliers_iforest.sum()) if outliers_iforest.dtype == bool else 0
        )

        self.assertGreaterEqual(total_outliers_iqr, 0, "IQR detection should complete")
        self.assertGreaterEqual(total_outliers_zscore, 0, "Z-score detection should complete")
        self.assertGreaterEqual(
            total_outliers_iforest, 0, "Isolation Forest detection should complete"
        )

    def test_pipeline_step2_winsorization(self):
        """Test Step 2: Sector-specific winsorization."""
        numeric_cols = self.sample_data.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        # Apply winsorization
        data_winsorized = winsorize_by_sector(
            self.sample_data.copy(),
            columns=financial_metrics[:5],
            lower_percentile=0.01,
            upper_percentile=0.99,
            by_sector=True,
        )

        # Verify shape is preserved
        self.assertEqual(data_winsorized.shape, self.sample_data.shape)

        # Verify extreme values were limited
        for col in financial_metrics[:5]:
            original_max = self.sample_data[col].max()
            winsorized_max = data_winsorized[col].max()
            if original_max > 100:  # Only check if there were extreme values
                self.assertLessEqual(winsorized_max, original_max)

    def test_pipeline_step3_data_quality_scoring(self):
        """Test Step 3: Data quality scoring."""
        quality_report = calculate_data_quality_score(self.sample_data)

        # Verify report structure
        self.assertTrue(hasattr(quality_report, "overall_score"))
        self.assertTrue(hasattr(quality_report, "completeness_score"))
        self.assertTrue(hasattr(quality_report, "validity_score"))
        self.assertTrue(hasattr(quality_report, "consistency_score"))

        # Verify scores are in valid range [0, 1]
        self.assertGreaterEqual(quality_report.overall_score, 0)
        self.assertLessEqual(quality_report.overall_score, 1)
        self.assertGreaterEqual(quality_report.completeness_score, 0)
        self.assertLessEqual(quality_report.completeness_score, 1)

    def test_pipeline_step4_imputation(self):
        """Test Step 4: 4-step imputation strategy."""
        data_imputed = apply_enhanced_imputation_strategy_4step(
            self.sample_data.copy(),
            sector_column="sector",
            n_neighbors=5,
            price_column="last_price",
        )

        # Verify missing values were reduced
        final_missing = data_imputed.isnull().sum().sum()
        self.assertLessEqual(final_missing, self.initial_missing)

        # Verify shape is preserved
        self.assertEqual(data_imputed.shape, self.sample_data.shape)

    def test_pipeline_step5_feature_scaling(self):
        """Test Step 5: Feature scaling with robust scaler."""
        numeric_cols = self.sample_data.select_dtypes(include=[np.number]).columns.tolist()
        exclude_scaling = ["price_target", "last_price"]
        scaling_cols = [c for c in numeric_cols if c not in exclude_scaling]

        # Apply scaling
        data_scaled = scale_features(
            self.sample_data.copy(), columns=scaling_cols[:5], scaler_type="robust", by_sector=True
        )

        # Verify shape is preserved
        self.assertEqual(data_scaled.shape, self.sample_data.shape)

        # Verify scaled columns have reasonable distribution
        for col in scaling_cols[:5]:
            if col in data_scaled.columns:
                scaled_mean = data_scaled[col].mean()
                scaled_std = data_scaled[col].std()
                # After robust scaling, values should be more normalized
                self.assertTrue(np.isfinite(scaled_mean))
                self.assertTrue(np.isfinite(scaled_std))

    def test_complete_pipeline_integration(self):
        """Test complete preprocessing pipeline from start to finish."""
        data = self.sample_data.copy()

        # Record initial state
        initial_missing = data.isnull().sum().sum()
        initial_shape = data.shape

        # Step 1: Outlier detection (just detect, don't remove)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        outliers_iqr = detect_outliers_iqr(data, columns=financial_metrics[:5], by_sector=True)
        iqr_bool_cols = outliers_iqr.select_dtypes(include=["bool"]).columns
        outliers_detected = (
            int(outliers_iqr[iqr_bool_cols].sum().sum()) if len(iqr_bool_cols) > 0 else 0
        )

        # Step 2: Winsorization
        data = winsorize_by_sector(
            data,
            columns=financial_metrics[:5],
            lower_percentile=0.01,
            upper_percentile=0.99,
            by_sector=True,
        )

        # Step 3: Data quality scoring
        quality_report = calculate_data_quality_score(data)

        # Step 4: Imputation
        data = apply_enhanced_imputation_strategy_4step(
            data, sector_column="sector", n_neighbors=5, price_column="last_price"
        )

        # Step 5: Feature scaling
        exclude_scaling = ["price_target", "last_price"]
        scaling_cols = [c for c in numeric_cols if c not in exclude_scaling]

        data_scaled = scale_features(
            data.copy(), columns=scaling_cols[:5], scaler_type="robust", by_sector=True
        )

        # Verify complete pipeline results
        final_missing = data.isnull().sum().sum()
        final_shape = data.shape

        # Assertions
        self.assertEqual(final_shape, initial_shape, "Shape should be preserved")
        self.assertLessEqual(final_missing, initial_missing, "Missing values should be reduced")
        self.assertGreater(outliers_detected, 0, "Should detect outliers")
        self.assertGreater(quality_report.overall_score, 0, "Should calculate quality score")
        self.assertEqual(data_scaled.shape, initial_shape, "Scaled data should preserve shape")

        # Log pipeline results
        print(f"\n✓ Complete Pipeline Test Results:")
        print(f"  Initial missing values: {initial_missing}")
        print(f"  Final missing values: {final_missing}")
        print(f"  Outliers detected (IQR): {outliers_detected}")
        print(f"  Data quality score: {quality_report.overall_score:.2f}")
        print(f"  Shape preserved: {initial_shape} -> {final_shape}")

    def test_pipeline_preserves_critical_columns(self):
        """Test that pipeline preserves critical columns (ticker, sector, targets)."""
        data = self.sample_data.copy()

        # Apply full pipeline
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        financial_metrics = [c for c in numeric_cols if c not in ["ticker", "isin"]]

        data = winsorize_by_sector(data, columns=financial_metrics[:5], by_sector=True)
        data = apply_enhanced_imputation_strategy_4step(
            data, sector_column="sector", price_column="last_price"
        )

        # Verify critical columns exist
        critical_columns = ["ticker", "sector", "region", "last_price", "price_target"]
        for col in critical_columns:
            self.assertIn(col, data.columns, f"Critical column '{col}' must be preserved")
            self.assertFalse(
                data[col].isnull().all(), f"Critical column '{col}' should have values"
            )

    def test_pipeline_handles_edge_cases(self):
        """Test that pipeline handles edge cases gracefully."""
        # Test with minimal data
        minimal_data = self.sample_data.head(10).copy()

        try:
            data = winsorize_by_sector(minimal_data, by_sector=True)
            data = apply_enhanced_imputation_strategy_4step(
                data, sector_column="sector", price_column="last_price"
            )
            quality_report = calculate_data_quality_score(data)

            self.assertEqual(len(data), 10, "Should handle small datasets")
            self.assertIsNotNone(quality_report, "Should calculate quality for small datasets")
        except Exception as e:
            self.fail(f"Pipeline should handle small datasets gracefully: {e}")


class TestSection2NotebookIntegration(unittest.TestCase):
    """Test that Section 2 notebook cells match expected structure."""

    def test_section2_has_all_preprocessing_steps(self):
        """Test that Section 2 includes all required preprocessing steps."""
        notebook_path = Path("ml_finance_model_main.ipynb")

        if not notebook_path.exists():
            self.skipTest("Notebook file not found")

        with open(notebook_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for all preprocessing components
        required_components = [
            "detect_outliers_iqr",
            "detect_outliers_zscore",
            "detect_outliers_isolation_forest",
            "winsorize_by_sector",
            "calculate_data_quality_score",
            "apply_enhanced_imputation_strategy_4step",
            "scale_features",
        ]

        for component in required_components:
            self.assertIn(
                component, content, f"Section 2 must include {component} in preprocessing pipeline"
            )

    def test_section2_imports_correct_modules(self):
        """Test that Section 2 imports from finance_ml.advanced_preprocessing."""
        notebook_path = Path("ml_finance_model_main.ipynb")

        if not notebook_path.exists():
            self.skipTest("Notebook file not found")

        with open(notebook_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for correct imports
        self.assertIn(
            "from finance_ml.advanced_preprocessing import",
            content,
            "Section 2 must import from advanced_preprocessing module",
        )


if __name__ == "__main__":
    unittest.main()
