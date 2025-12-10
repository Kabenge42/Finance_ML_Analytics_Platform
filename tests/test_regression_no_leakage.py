"""
Test suite for regression feature leakage prevention.

Tests ensure that market_cap is excluded from regression features to prevent
data leakage (market_cap scale predictions instead of price scale).

TDD Implementation for Priority 1 - Task 1.1:
- Remove market_cap feature leakage from regression pipeline
- Verify no sector interactions with market_cap
- Allow enterprise_value (forward-looking metric)

Aligned with:
- code_guidelines.md v1.10 Section 5.4 (Feature Categories)
- Finance ML Workflow TDD Implementation Plan v1.0
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data,
    add_sector_interactions_for_prediction,
)


def create_sample_stocks_dataframe(n_samples=100, random_state=42):
    """
    Create sample stock data with realistic features for testing.

    Includes both allowed features (enterprise_value, p_e_ratio, etc.)
    and the problematic feature (market_cap) that should be excluded.
    """
    np.random.seed(random_state)

    sectors = ["Technology", "Financials", "Healthcare", "Energy", "Materials"]

    df = pd.DataFrame(
        {
            "ticker": [f"TICK{i:03d}" for i in range(n_samples)],
            "sector": np.random.choice(sectors, n_samples),
            "last_price": np.random.uniform(10, 500, n_samples),
            "price_target": np.random.uniform(10, 500, n_samples),
            "market_cap": np.random.uniform(1e8, 1e12, n_samples),  # Should be EXCLUDED
            "enterprise_value": np.random.uniform(1e8, 1e12, n_samples),  # Should be ALLOWED
            "p_e_ratio": np.random.uniform(5, 50, n_samples),
            "ev_ebitda_ratio": np.random.uniform(5, 30, n_samples),
            "gross_margin": np.random.uniform(0.1, 0.9, n_samples),
            "beta_5y": np.random.uniform(0.5, 2.0, n_samples),
            "debt_to_equity": np.random.uniform(0.0, 2.5, n_samples),
        }
    )

    return df


class TestMarketCapLeakagePrevention(unittest.TestCase):
    """Test suite for market_cap feature leakage prevention."""

    def test_market_cap_not_in_features(self):
        """
        Ensure market_cap excluded from regression features (data leakage prevention).

        CRITICAL: market_cap directly correlates with price_target and causes
        predictions to be on market_cap scale (~880K) instead of price scale (~736K).

        Expected behavior:
        - market_cap NOT in feature columns
        - log_market_cap NOT in feature columns
        - No sector interactions with market_cap
        """
        df = create_sample_stocks_dataframe()
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target"
        )

        # Market cap should NOT appear in features
        self.assertNotIn(
            "market_cap",
            X_train.columns.tolist(),
            "market_cap leaks target information - causes predictions on wrong scale",
        )
        self.assertNotIn(
            "log_market_cap", X_train.columns.tolist(), "log_market_cap leaks target information"
        )

        # No sector interactions with market_cap
        market_cap_interactions = [col for col in X_train.columns if "market_cap" in col.lower()]
        self.assertEqual(
            len(market_cap_interactions),
            0,
            f"Market cap interactions found (leakage): {market_cap_interactions}",
        )

        # Verify same for test set
        self.assertNotIn("market_cap", X_test.columns.tolist())
        market_cap_interactions_test = [
            col for col in X_test.columns if "market_cap" in col.lower()
        ]
        self.assertEqual(len(market_cap_interactions_test), 0)

    def test_enterprise_value_allowed_as_feature(self):
        """
        Verify enterprise_value can be used (not circular like market_cap).

        Enterprise value is forward-looking and includes debt, preferred stock,
        and minority interests - not directly circular with stock price.
        """
        df = create_sample_stocks_dataframe()
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target"
        )

        # Enterprise value allowed because it's forward-looking
        ev_cols = [col for col in X_train.columns if "enterprise_value" in col.lower()]
        self.assertGreater(
            len(ev_cols),
            0,
            "Enterprise value should be available as feature (forward-looking metric)",
        )

        # Could be raw or log-transformed
        has_ev_raw = "enterprise_value" in X_train.columns
        has_ev_log = "log_enterprise_value" in X_train.columns
        self.assertTrue(
            has_ev_raw or has_ev_log,
            "Either enterprise_value or log_enterprise_value should be present",
        )

    def test_debt_to_equity_included_as_replacement(self):
        """
        Verify debt_to_equity is included as replacement for market_cap.

        Debt-to-equity ratio is a fundamental risk metric that doesn't leak
        target information and provides value for valuation models.
        """
        df = create_sample_stocks_dataframe()
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target"
        )

        # Debt-to-equity should be present (either raw or in interactions)
        dte_cols = [col for col in X_train.columns if "debt_to_equity" in col.lower()]
        self.assertGreater(
            len(dte_cols), 0, "debt_to_equity should be included as fundamental risk metric"
        )

    def test_sector_interactions_no_market_cap(self):
        """
        Verify sector interactions don't include market_cap.

        Sector interactions multiply sector dummies by base features.
        If market_cap is in base features, it creates leakage through
        interactions like sector_Technology__x__market_cap.
        """
        df = create_sample_stocks_dataframe()
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target"
        )

        # Check for sector interaction columns
        sector_interaction_cols = [
            col for col in X_train.columns if "sector_" in col and "__x__" in col
        ]

        if len(sector_interaction_cols) > 0:
            # If sector interactions exist, none should involve market_cap
            market_cap_sector_interactions = [
                col for col in sector_interaction_cols if "market_cap" in col.lower()
            ]
            self.assertEqual(
                len(market_cap_sector_interactions),
                0,
                f"Sector interactions with market_cap found: {market_cap_sector_interactions}",
            )

    def test_add_sector_interactions_for_prediction_no_market_cap(self):
        """
        Verify prediction-time sector interactions also exclude market_cap.

        The add_sector_interactions_for_prediction() function must use the
        same base_cols (without market_cap) to ensure training/prediction parity.
        """
        df = create_sample_stocks_dataframe()

        # Prepare features (without target)
        feature_cols = [
            "p_e_ratio",
            "ev_ebitda_ratio",
            "gross_margin",
            "beta_5y",
            "debt_to_equity",
            "enterprise_value",
        ]
        X = df[feature_cols].copy()

        # Add sector interactions using default base_cols
        X_with_interactions = add_sector_interactions_for_prediction(X, df_with_sector=df)

        # No market_cap in any column name
        market_cap_cols = [
            col for col in X_with_interactions.columns if "market_cap" in col.lower()
        ]
        self.assertEqual(
            len(market_cap_cols), 0, f"market_cap found in prediction features: {market_cap_cols}"
        )

    def test_prediction_scale_reasonable(self):
        """
        Verify predictions are on price scale, not market_cap scale.

        After fixing market_cap leakage, predictions should be in range
        of actual price targets (e.g., 10-500) not market_cap scale (1e8-1e12).

        This is an integration test that trains a simple model and checks
        prediction magnitude.
        """
        df = create_sample_stocks_dataframe(n_samples=200)
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target", test_size=0.3
        )

        # Train simple model
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Check prediction scale
        pred_mean = np.mean(y_pred)
        pred_std = np.std(y_pred)
        target_mean = np.mean(y_test)
        target_std = np.std(y_test)

        # Predictions should be within reasonable range of targets
        # Not orders of magnitude off (which happens with market_cap leakage)
        scale_ratio = pred_mean / target_mean if target_mean > 0 else 0

        self.assertGreater(
            scale_ratio,
            0.1,
            f"Predictions too small (mean: {pred_mean:.2f} vs target: {target_mean:.2f})",
        )
        self.assertLess(
            scale_ratio,
            10.0,
            f"Predictions too large (mean: {pred_mean:.2f} vs target: {target_mean:.2f}) - possible market_cap leakage",
        )

        # Predictions should have reasonable variability
        # Not all near zero or all huge
        self.assertGreater(
            pred_std, 0.01 * target_std, "Prediction std too small - model not learning variation"
        )
        self.assertLess(
            pred_std, 100 * target_std, "Prediction std too large - possible scale mismatch"
        )


class TestFeatureLeakageDocumentation(unittest.TestCase):
    """Test that feature metadata correctly documents exclusions."""

    def test_metadata_documents_excluded_features(self):
        """
        Verify that feature metadata tracks excluded features.

        The prepare_regression_data() function returns metadata dictionary
        that should document which features were intentionally excluded
        for leakage prevention.
        """
        df = create_sample_stocks_dataframe()
        X_train, X_test, y_train, y_test, meta = prepare_regression_data(
            df, target_col="price_target"
        )

        # Metadata should exist
        self.assertIsInstance(meta, dict)

        # Should have feature information
        self.assertIn("all_features", meta)

        # market_cap should not be in feature list
        all_features = meta["all_features"]
        market_cap_in_features = any("market_cap" in str(f).lower() for f in all_features)
        self.assertFalse(market_cap_in_features, "market_cap should not be in documented features")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
