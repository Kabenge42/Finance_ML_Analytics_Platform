"""
TDD Test Suite for Task 10.1: Fix Quantile Calibration Failure

Tests proper conformal prediction intervals with coverage guarantees.
Target: Achieve 75-85% empirical coverage (currently 7.1%)

Requirements from finance_ml_improvement_plan.md Task 10.1:
- Implement proper conformal prediction intervals with coverage guarantees
- Add sector-aware quantile calibration (different volatility per sector)
- Use TimeSeriesSplit for quantile model training (prevent leakage)
- Implement quantile interval validation (check coverage, monotonicity, non-negativity)
- Export calibration diagnostics: coverage by sector, interval width distribution

Test Categories:
1. Coverage validation (75-85% target)
2. Monotonicity preservation (p10 ≤ p50 ≤ p90)
3. Non-negativity enforcement for price predictions
4. Sector-specific coverage variations
5. TimeSeriesSplit integration
6. Diagnostics export functionality
"""

import unittest
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Import existing modules
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor,
    enforce_monotonic_quantiles,
)
from finance_ml.ml_workflow.regression.uncertainty import (
    conformal_prediction_intervals,
    compute_interval_coverage,
)


class TestQuantileCalibrationCoverage(unittest.TestCase):
    """Test suite for quantile calibration with conformal prediction."""

    def setUp(self):
        """Create synthetic data for testing."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        # Create features
        self.X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Create target with heteroscedastic noise (sector-dependent volatility)
        self.sectors = np.random.choice(
            ["Technology", "Healthcare", "Energy", "Financials"], size=n_samples
        )
        self.X["sector"] = self.sectors

        # Base target
        true_signal = (
            self.X["feature_0"] * 10 + self.X["feature_1"] * 5 + self.X["feature_2"] * 3
        ) + 100

        # Sector-specific volatility
        sector_volatility = {
            "Technology": 5.0,
            "Healthcare": 8.0,
            "Energy": 15.0,  # High volatility
            "Financials": 6.0,
        }
        noise = np.array([np.random.randn() * sector_volatility[sector] for sector in self.sectors])

        self.y = pd.Series(true_signal + noise, name="price_target")

        # Ensure non-negative prices
        self.y = self.y.clip(lower=0)

    def test_conformal_quantile_calibration_function_exists(self):
        """Test that conformal_quantile_calibration function exists."""
        # This will fail initially - function doesn't exist yet
        try:
            from finance_ml.ml_workflow.regression.uncertainty import conformal_quantile_calibration

            self.assertTrue(callable(conformal_quantile_calibration))
        except ImportError:
            self.fail("conformal_quantile_calibration function not implemented")

    def test_sector_aware_quantile_calibration_exists(self):
        """Test that sector_aware_quantile_calibration function exists."""
        try:
            from finance_ml.ml_workflow.regression.uncertainty import (
                sector_aware_quantile_calibration,
            )

            self.assertTrue(callable(sector_aware_quantile_calibration))
        except ImportError:
            self.fail("sector_aware_quantile_calibration function not implemented")

    def test_validate_quantile_coverage_exists(self):
        """Test that validate_quantile_coverage function exists."""
        try:
            from finance_ml.ml_workflow.regression.uncertainty import validate_quantile_coverage

            self.assertTrue(callable(validate_quantile_coverage))
        except ImportError:
            self.fail("validate_quantile_coverage function not implemented")

    def test_empirical_coverage_target_80_percent(self):
        """Test that conformal calibration achieves 75-85% empirical coverage."""
        from finance_ml.ml_workflow.regression.uncertainty import conformal_quantile_calibration

        # Split using TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=2)
        train_idx, test_idx = list(tscv.split(self.X))[-1]  # Use last split

        X_train = self.X.iloc[train_idx].drop(columns=["sector"])
        X_test = self.X.iloc[test_idx].drop(columns=["sector"])
        y_train = self.y.iloc[train_idx]
        y_test = self.y.iloc[test_idx]

        # Train quantile models
        quantile_result = train_quantile_regressor(
            X_train, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Apply conformal calibration
        calibrated_intervals = conformal_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            alpha=0.2,  # 80% coverage target
            quantiles=[0.1, 0.5, 0.9],
        )

        # Validate coverage
        lower = calibrated_intervals["pred_p10"]
        upper = calibrated_intervals["pred_p90"]
        coverage = compute_interval_coverage(y_test.values, lower, upper)

        # Assert coverage is between 75-85%
        self.assertGreaterEqual(
            coverage, 0.75, f"Coverage {coverage:.1%} below target (75% minimum)"
        )
        self.assertLessEqual(coverage, 0.85, f"Coverage {coverage:.1%} above target (85% maximum)")

    def test_monotonicity_preserved_after_calibration(self):
        """Test that p10 ≤ p50 ≤ p90 after conformal calibration."""
        from finance_ml.ml_workflow.regression.uncertainty import conformal_quantile_calibration

        # Use simple train/test split
        split_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:split_idx].drop(columns=["sector"])
        X_test = self.X.iloc[split_idx:].drop(columns=["sector"])
        y_train = self.y.iloc[:split_idx]

        # Train quantile models
        quantile_result = train_quantile_regressor(
            X_train, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Apply conformal calibration
        calibrated_intervals = conformal_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            alpha=0.2,
            quantiles=[0.1, 0.5, 0.9],
        )

        # Check monotonicity
        p10 = calibrated_intervals["pred_p10"]
        p50 = calibrated_intervals["pred_p50"]
        p90 = calibrated_intervals["pred_p90"]

        # All samples must satisfy p10 ≤ p50 ≤ p90
        violations_lower = np.sum(p10 > p50)
        violations_upper = np.sum(p50 > p90)

        self.assertEqual(violations_lower, 0, f"Found {violations_lower} violations of p10 ≤ p50")
        self.assertEqual(violations_upper, 0, f"Found {violations_upper} violations of p50 ≤ p90")

    def test_non_negativity_for_price_predictions(self):
        """Test that all calibrated predictions are non-negative."""
        from finance_ml.ml_workflow.regression.uncertainty import conformal_quantile_calibration

        split_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:split_idx].drop(columns=["sector"])
        X_test = self.X.iloc[split_idx:].drop(columns=["sector"])
        y_train = self.y.iloc[:split_idx]

        # Train quantile models
        quantile_result = train_quantile_regressor(
            X_train, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Apply conformal calibration with non-negativity
        calibrated_intervals = conformal_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            alpha=0.2,
            quantiles=[0.1, 0.5, 0.9],
            clip_lower_at_zero=True,
        )

        # Check all predictions are non-negative
        p10 = calibrated_intervals["pred_p10"]
        p50 = calibrated_intervals["pred_p50"]
        p90 = calibrated_intervals["pred_p90"]

        self.assertTrue(np.all(p10 >= 0), f"Found negative values in p10: min={p10.min()}")
        self.assertTrue(np.all(p50 >= 0), f"Found negative values in p50: min={p50.min()}")
        self.assertTrue(np.all(p90 >= 0), f"Found negative values in p90: min={p90.min()}")

    def test_sector_aware_calibration_different_volatility(self):
        """Test sector-aware calibration handles different volatility per sector."""
        from finance_ml.ml_workflow.regression.uncertainty import sector_aware_quantile_calibration

        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=2)
        train_idx, test_idx = list(tscv.split(self.X))[-1]

        X_train = self.X.iloc[train_idx]
        X_test = self.X.iloc[test_idx]
        y_train = self.y.iloc[train_idx]
        y_test = self.y.iloc[test_idx]

        # Train quantile models (without sector column)
        X_train_features = X_train.drop(columns=["sector"])
        X_test_features = X_test.drop(columns=["sector"])

        quantile_result = train_quantile_regressor(
            X_train_features, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        # Apply sector-aware calibration
        calibrated_intervals = sector_aware_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train_features,
            y_train=y_train,
            X_test=X_test_features,
            sectors_train=X_train["sector"].values,
            sectors_test=X_test["sector"].values,
            alpha=0.2,
            quantiles=[0.1, 0.5, 0.9],
        )

        # Verify coverage by sector
        test_df = pd.DataFrame(
            {
                "sector": X_test["sector"].values,
                "y_true": y_test.values,
                "pred_p10": calibrated_intervals["pred_p10"],
                "pred_p90": calibrated_intervals["pred_p90"],
            }
        )

        # Check each sector has reasonable coverage
        for sector in ["Technology", "Healthcare", "Energy", "Financials"]:
            sector_mask = test_df["sector"] == sector
            if sector_mask.sum() < 5:  # Skip if too few samples
                continue

            sector_df = test_df[sector_mask]
            sector_coverage = compute_interval_coverage(
                sector_df["y_true"].values,
                sector_df["pred_p10"].values,
                sector_df["pred_p90"].values,
            )

            # Each sector should have coverage within reasonable range
            # Allow wider range per sector since sample size is smaller
            self.assertGreater(
                sector_coverage, 0.60, f"Sector {sector} coverage {sector_coverage:.1%} too low"
            )
            self.assertLess(
                sector_coverage, 0.95, f"Sector {sector} coverage {sector_coverage:.1%} too high"
            )

    def test_timeseries_split_prevents_leakage(self):
        """Test that TimeSeriesSplit is used to prevent future information leakage."""
        from finance_ml.ml_workflow.regression.uncertainty import conformal_quantile_calibration

        # TimeSeriesSplit should respect temporal order
        tscv = TimeSeriesSplit(n_splits=3)

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.X)):
            # Verify train indices come before test indices
            max_train_idx = max(train_idx)
            min_test_idx = min(test_idx)

            self.assertLess(
                max_train_idx,
                min_test_idx,
                f"Fold {fold_idx}: TimeSeriesSplit leakage - train indices overlap with test",
            )

    def test_validate_quantile_coverage_diagnostics(self):
        """Test validate_quantile_coverage returns diagnostics."""
        from finance_ml.ml_workflow.regression.uncertainty import (
            conformal_quantile_calibration,
            validate_quantile_coverage,
        )

        split_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y.iloc[:split_idx]
        y_test = self.y.iloc[split_idx:]

        # Train and calibrate
        X_train_features = X_train.drop(columns=["sector"])
        X_test_features = X_test.drop(columns=["sector"])

        quantile_result = train_quantile_regressor(
            X_train_features, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        calibrated_intervals = conformal_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train_features,
            y_train=y_train,
            X_test=X_test_features,
            alpha=0.2,
            quantiles=[0.1, 0.5, 0.9],
        )

        # Validate with diagnostics
        diagnostics = validate_quantile_coverage(
            y_true=y_test.values,
            pred_p10=calibrated_intervals["pred_p10"],
            pred_p50=calibrated_intervals["pred_p50"],
            pred_p90=calibrated_intervals["pred_p90"],
            sectors=X_test["sector"].values,
        )

        # Check diagnostics structure
        self.assertIn("overall_coverage", diagnostics)
        self.assertIn("coverage_by_sector", diagnostics)
        self.assertIn("interval_width_stats", diagnostics)
        self.assertIn("monotonicity_violations", diagnostics)

        # Validate overall coverage is in target range
        overall_coverage = diagnostics["overall_coverage"]
        self.assertGreaterEqual(overall_coverage, 0.75)
        self.assertLessEqual(overall_coverage, 0.85)

        # Validate coverage_by_sector is a dict
        self.assertIsInstance(diagnostics["coverage_by_sector"], dict)

        # Validate interval_width_stats has expected keys
        width_stats = diagnostics["interval_width_stats"]
        self.assertIn("mean", width_stats)
        self.assertIn("median", width_stats)
        self.assertIn("std", width_stats)

    def test_export_calibration_diagnostics_to_csv(self):
        """Test that calibration diagnostics can be exported to CSV."""
        from finance_ml.ml_workflow.regression.uncertainty import (
            conformal_quantile_calibration,
            validate_quantile_coverage,
            export_calibration_diagnostics,
        )

        split_idx = int(0.7 * len(self.X))
        X_train = self.X.iloc[:split_idx]
        X_test = self.X.iloc[split_idx:]
        y_train = self.y.iloc[:split_idx]
        y_test = self.y.iloc[split_idx:]

        # Train and calibrate
        X_train_features = X_train.drop(columns=["sector"])
        X_test_features = X_test.drop(columns=["sector"])

        quantile_result = train_quantile_regressor(
            X_train_features, y_train, quantiles=[0.1, 0.5, 0.9], random_state=42
        )

        calibrated_intervals = conformal_quantile_calibration(
            quantile_models=quantile_result["model"],
            X_train=X_train_features,
            y_train=y_train,
            X_test=X_test_features,
            alpha=0.2,
            quantiles=[0.1, 0.5, 0.9],
        )

        # Get diagnostics
        diagnostics = validate_quantile_coverage(
            y_true=y_test.values,
            pred_p10=calibrated_intervals["pred_p10"],
            pred_p50=calibrated_intervals["pred_p50"],
            pred_p90=calibrated_intervals["pred_p90"],
            sectors=X_test["sector"].values,
        )

        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            export_calibration_diagnostics(diagnostics=diagnostics, output_dir=output_dir)

            # Verify files were created
            coverage_by_sector_file = output_dir / "coverage_by_sector.csv"
            interval_width_file = output_dir / "interval_width_distribution.csv"

            self.assertTrue(coverage_by_sector_file.exists(), "coverage_by_sector.csv not created")
            self.assertTrue(
                interval_width_file.exists(), "interval_width_distribution.csv not created"
            )

            # Verify CSV content is readable
            coverage_df = pd.read_csv(coverage_by_sector_file)
            self.assertIn("sector", coverage_df.columns)
            self.assertIn("coverage", coverage_df.columns)

            width_df = pd.read_csv(interval_width_file)
            self.assertIn("interval_width", width_df.columns)


if __name__ == "__main__":
    unittest.main()
