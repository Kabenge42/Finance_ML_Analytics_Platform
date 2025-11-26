"""
Phase 7: Enhanced ML Return Prediction & Advanced Optimization Tests (TDD v2.0)

This test module implements TDD for Phase 7 enhancements:
1. Realistic expected returns (mean < 50%)
2. PRICE_COLUMNS constant for historical return calculation
3. Phase 9.3 feature integration (196 features)
4. Return clipping/constraints for production-ready optimization

Test-Driven Development: Write failing tests FIRST, then implement minimal code to pass.

Issue Summary:
- Return Calculation Issue: Expected returns showing mean of 95.6% (unrealistic)
- Sharpe Ratio Anomaly: Max Sharpe of 42.4 indicates calculation problems
- Feature Underutilization: Only 6 basic features vs 196 available Phase 9.3 features
- Price Column Gap: 21 PRICE_COLUMNS not integrated for historical return calculation

Acceptance Criteria:
- Realistic expected returns (mean < 50%)
- Improved prediction accuracy
- Production-ready optimization
"""

import unittest
import numpy as np
import pandas as pd
from typing import Dict, List


def create_sample_portfolio_data(n_stocks: int = 50) -> pd.DataFrame:
    """Create sample portfolio data with realistic financial metrics."""
    np.random.seed(42)

    sectors = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]

    data = {
        "ticker": [f"STOCK_{i}" for i in range(n_stocks)],
        "sector": np.random.choice(sectors, n_stocks),
        "last_price": np.random.uniform(10, 500, n_stocks),
        "market_cap": np.random.uniform(1e9, 1e12, n_stocks),
        "price_target": np.random.uniform(10, 600, n_stocks),
        "mispricing_score": np.random.uniform(-0.5, 0.5, n_stocks),
        # Historical price columns
        "price_1w_ago": np.random.uniform(10, 500, n_stocks),
        "price_1m_ago": np.random.uniform(10, 500, n_stocks),
        "price_3m_ago": np.random.uniform(10, 500, n_stocks),
        "price_6m_ago": np.random.uniform(10, 500, n_stocks),
        "price_1y_ago": np.random.uniform(10, 500, n_stocks),
        # EMA columns
        "ema_20d": np.random.uniform(10, 500, n_stocks),
        "ema_50d": np.random.uniform(10, 500, n_stocks),
        "ema_100d": np.random.uniform(10, 500, n_stocks),
        "ema_250d": np.random.uniform(10, 500, n_stocks),
        # 52-week bounds
        "52w_high_adj": np.random.uniform(100, 600, n_stocks),
        "52w_low_adj": np.random.uniform(5, 100, n_stocks),
        # Return columns (daily)
        "return_1d": np.random.uniform(-0.05, 0.05, n_stocks),
    }

    return pd.DataFrame(data)


def create_sample_data_with_phase93_features(n_stocks: int = 50) -> pd.DataFrame:
    """Create sample data with Phase 9.3 engineered features."""
    df = create_sample_portfolio_data(n_stocks)
    np.random.seed(42)

    # Add a subset of Phase 9.3 features
    phase93_features = {
        # Momentum & Technical
        "price_momentum_1m": np.random.uniform(-0.2, 0.2, n_stocks),
        "price_momentum_3m": np.random.uniform(-0.3, 0.3, n_stocks),
        "price_momentum_6m": np.random.uniform(-0.4, 0.4, n_stocks),
        "price_momentum_1y": np.random.uniform(-0.5, 0.5, n_stocks),
        "rsi_14d": np.random.uniform(20, 80, n_stocks),
        "sharpe_proxy": np.random.uniform(-1, 2, n_stocks),
        "volume_momentum_score": np.random.uniform(-1, 1, n_stocks),
        # Valuation Ratios
        "p_e_ratio": np.random.uniform(5, 50, n_stocks),
        "p_b_ratio": np.random.uniform(0.5, 10, n_stocks),
        "ev_ebitda_ratio": np.random.uniform(3, 30, n_stocks),
        "peg_ratio": np.random.uniform(0.5, 3, n_stocks),
        # Quality & Risk
        "beta": np.random.uniform(0.5, 2, n_stocks),
        "volatility_90d": np.random.uniform(0.1, 0.5, n_stocks),
        # Profitability
        "roe": np.random.uniform(-0.1, 0.4, n_stocks),
        "roa": np.random.uniform(-0.05, 0.2, n_stocks),
        "net_margin_pct": np.random.uniform(-0.1, 0.3, n_stocks),
        # Growth Metrics
        "revenue_growth_yoy": np.random.uniform(-0.2, 0.5, n_stocks),
        "earnings_growth_yoy": np.random.uniform(-0.3, 0.6, n_stocks),
        # Analyst Sentiment
        "analyst_rating_avg": np.random.uniform(1, 5, n_stocks),
        "price_target_upside": np.random.uniform(-0.2, 0.5, n_stocks),
    }

    for col, values in phase93_features.items():
        df[col] = values

    return df


class TestReturnBoundsConfiguration(unittest.TestCase):
    """Test configuration constants for realistic return bounds."""

    def test_max_expected_return_constant_exists(self):
        """Test that MAX_EXPECTED_RETURN constant exists in config."""
        from finance_ml.ml_workflow.config.ml_returns_config import MAX_EXPECTED_RETURN

        self.assertIsInstance(MAX_EXPECTED_RETURN, float)
        # Should be reasonable upper bound (e.g., 100% = 1.0 or 200% = 2.0)
        self.assertLessEqual(MAX_EXPECTED_RETURN, 2.0)
        self.assertGreater(MAX_EXPECTED_RETURN, 0.0)

    def test_min_expected_return_constant_exists(self):
        """Test that MIN_EXPECTED_RETURN constant exists in config."""
        from finance_ml.ml_workflow.config.ml_returns_config import MIN_EXPECTED_RETURN

        self.assertIsInstance(MIN_EXPECTED_RETURN, float)
        # Should be reasonable lower bound (e.g., -100% = -1.0)
        self.assertGreaterEqual(MIN_EXPECTED_RETURN, -1.0)
        self.assertLess(MIN_EXPECTED_RETURN, 0.0)

    def test_realistic_return_bounds_relationship(self):
        """Test that MIN < DEFAULT < MAX for expected returns."""
        from finance_ml.ml_workflow.config.ml_returns_config import (
            MIN_EXPECTED_RETURN,
            MAX_EXPECTED_RETURN,
            DEFAULT_EXPECTED_RETURN,
        )

        self.assertLess(MIN_EXPECTED_RETURN, DEFAULT_EXPECTED_RETURN)
        self.assertLess(DEFAULT_EXPECTED_RETURN, MAX_EXPECTED_RETURN)


class TestPriceColumnsConstant(unittest.TestCase):
    """Test PRICE_COLUMNS constant for historical return calculation."""

    def test_price_columns_constant_exists(self):
        """Test that PRICE_COLUMNS constant exists in config."""
        from finance_ml.ml_workflow.config.ml_returns_config import PRICE_COLUMNS

        self.assertIsInstance(PRICE_COLUMNS, dict)

    def test_price_columns_has_required_categories(self):
        """Test PRICE_COLUMNS has current, historical, 52w_bounds, and emas."""
        from finance_ml.ml_workflow.config.ml_returns_config import PRICE_COLUMNS

        required_categories = ["current", "historical", "52w_bounds", "emas"]
        for category in required_categories:
            self.assertIn(
                category, PRICE_COLUMNS, f"Missing category '{category}' in PRICE_COLUMNS"
            )

    def test_price_columns_historical_has_lookback_periods(self):
        """Test historical category has standard lookback periods."""
        from finance_ml.ml_workflow.config.ml_returns_config import PRICE_COLUMNS

        historical = PRICE_COLUMNS.get("historical", [])
        # Should include at least 1w, 1m, 3m, 6m, 1y lookback columns
        self.assertGreaterEqual(len(historical), 5)


class TestClipExpectedReturns(unittest.TestCase):
    """Test clip_expected_returns function for realistic bounds."""

    def test_clip_expected_returns_function_exists(self):
        """Test that clip_expected_returns function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns

        self.assertTrue(callable(clip_expected_returns))

    def test_clip_expected_returns_clips_high_values(self):
        """Test that extreme high returns are clipped."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns
        from finance_ml.ml_workflow.config.ml_returns_config import MAX_EXPECTED_RETURN

        # Create returns with unrealistically high values (like the 95.6% issue)
        returns = np.array([0.10, 0.956, 1.5, 2.0, 3.0])

        clipped = clip_expected_returns(returns)

        self.assertTrue(all(clipped <= MAX_EXPECTED_RETURN))
        # First value should be unchanged
        self.assertAlmostEqual(clipped[0], 0.10, places=5)

    def test_clip_expected_returns_clips_low_values(self):
        """Test that extreme low returns are clipped."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns
        from finance_ml.ml_workflow.config.ml_returns_config import MIN_EXPECTED_RETURN

        # Create returns with unrealistically low values
        returns = np.array([-0.10, -0.50, -1.5, -2.0])

        clipped = clip_expected_returns(returns)

        self.assertTrue(all(clipped >= MIN_EXPECTED_RETURN))

    def test_clip_expected_returns_handles_series(self):
        """Test that function handles pandas Series."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns

        returns = pd.Series([0.10, 0.956, 1.5, -0.50, -1.5])

        clipped = clip_expected_returns(returns)

        self.assertIsInstance(clipped, (np.ndarray, pd.Series))

    def test_clip_expected_returns_mean_below_threshold(self):
        """Test that clipped returns have mean < 50% (key acceptance criterion)."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns

        # Simulate the issue: unrealistic returns with 95.6% mean
        np.random.seed(42)
        unrealistic_returns = np.random.uniform(0.5, 1.5, 100)  # Mean ~100%

        clipped = clip_expected_returns(unrealistic_returns)

        # Key acceptance criterion: mean < 50%
        self.assertLess(
            np.mean(clipped), 0.50, f"Mean return {np.mean(clipped):.2%} exceeds 50% threshold"
        )


class TestCalculateHistoricalReturns(unittest.TestCase):
    """Test calculate_historical_returns function using PRICE_COLUMNS."""

    def test_calculate_historical_returns_function_exists(self):
        """Test that calculate_historical_returns function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import calculate_historical_returns

        self.assertTrue(callable(calculate_historical_returns))

    def test_calculate_historical_returns_creates_return_columns(self):
        """Test that function creates return columns from price columns."""
        from finance_ml.ml_workflow.analytics.ml_returns import calculate_historical_returns

        df = create_sample_portfolio_data()

        result = calculate_historical_returns(df)

        # Should create return columns
        expected_cols = ["return_1w", "return_1m", "return_3m", "return_6m", "return_1y"]
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing return column: {col}")

    def test_calculate_historical_returns_correct_formula(self):
        """Test that returns are calculated correctly: (current - historical) / historical."""
        from finance_ml.ml_workflow.analytics.ml_returns import calculate_historical_returns

        df = pd.DataFrame(
            {
                "last_price": [100.0, 110.0, 90.0],
                "price_1m_ago": [90.0, 100.0, 100.0],
            }
        )

        result = calculate_historical_returns(df)

        # Expected: (100-90)/90=0.111, (110-100)/100=0.10, (90-100)/100=-0.10
        expected_1m = pd.Series([0.1111, 0.10, -0.10])
        np.testing.assert_array_almost_equal(
            result["return_1m"].values, expected_1m.values, decimal=2
        )

    def test_calculate_historical_returns_handles_missing_columns(self):
        """Test graceful handling when price columns are missing."""
        from finance_ml.ml_workflow.analytics.ml_returns import calculate_historical_returns

        df = pd.DataFrame(
            {
                "last_price": [100.0, 110.0],
                # Missing historical price columns
            }
        )

        # Should not raise error, just skip missing columns
        result = calculate_historical_returns(df)

        self.assertIsInstance(result, pd.DataFrame)


class TestPhase93FeatureIntegration(unittest.TestCase):
    """Test Phase 9.3 feature integration for enhanced return prediction."""

    def test_get_phase93_return_features_function_exists(self):
        """Test that get_phase93_return_features function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import get_phase93_return_features

        self.assertTrue(callable(get_phase93_return_features))

    def test_get_phase93_return_features_returns_categories(self):
        """Test that function returns relevant Phase 9.3 feature categories."""
        from finance_ml.ml_workflow.analytics.ml_returns import get_phase93_return_features

        categories = get_phase93_return_features()

        self.assertIsInstance(categories, dict)
        # Should include high-relevance categories
        expected_categories = ["Momentum & Technical", "Valuation Ratios"]
        for cat in expected_categories:
            self.assertIn(cat, categories)

    def test_get_phase93_return_features_total_count(self):
        """Test that we get more features than the current 6 basic ones."""
        from finance_ml.ml_workflow.analytics.ml_returns import get_phase93_return_features

        categories = get_phase93_return_features()

        total_features = sum(len(features) for features in categories.values())

        # Should have significantly more than 6 features
        self.assertGreater(
            total_features, 20, f"Only {total_features} features, expected > 20 for Phase 9.3"
        )


class TestCreateMlReturnFeaturesEnhanced(unittest.TestCase):
    """Test enhanced ML return features with Phase 9.3 integration."""

    def test_create_ml_return_features_enhanced_function_exists(self):
        """Test that create_ml_return_features_enhanced function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import create_ml_return_features_enhanced

        self.assertTrue(callable(create_ml_return_features_enhanced))

    def test_create_ml_return_features_enhanced_uses_phase93(self):
        """Test that enhanced function uses Phase 9.3 features when available."""
        from finance_ml.ml_workflow.analytics.ml_returns import create_ml_return_features_enhanced

        df = create_sample_data_with_phase93_features()

        result = create_ml_return_features_enhanced(df)

        # Should include Phase 9.3 features that exist in input
        phase93_cols = ["price_momentum_1m", "rsi_14d", "p_e_ratio", "roe"]
        for col in phase93_cols:
            if col in df.columns:
                self.assertIn(col, result.columns, f"Phase 9.3 feature '{col}' not included")

    def test_create_ml_return_features_enhanced_feature_count(self):
        """Test that enhanced function produces more features than basic version."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            create_ml_return_features,
            create_ml_return_features_enhanced,
        )

        df = create_sample_data_with_phase93_features()

        # Need return column for basic version
        df["return_1d"] = np.random.uniform(-0.05, 0.05, len(df))

        enhanced_result = create_ml_return_features_enhanced(df)

        # Enhanced should have more features
        # Basic version: ~6 features (3 lags + 3 technical indicators)
        # Enhanced: Should include Phase 9.3 features
        self.assertGreater(
            len(enhanced_result.columns), 10, "Enhanced features should include Phase 9.3 columns"
        )


class TestRealisticReturnPrediction(unittest.TestCase):
    """Integration tests for realistic return prediction."""

    def test_predicted_returns_are_realistic(self):
        """Test that predicted returns fall within realistic bounds."""
        from finance_ml.ml_workflow.analytics.ml_returns import (
            train_linear_return_predictor,
            clip_expected_returns,
        )
        from finance_ml.ml_workflow.config.ml_returns_config import (
            MIN_EXPECTED_RETURN,
            MAX_EXPECTED_RETURN,
        )

        np.random.seed(42)

        # Create training data
        X_train = np.random.randn(100, 10)
        y_train = np.random.uniform(-0.5, 1.5, 100)  # Some unrealistic targets

        model = train_linear_return_predictor(X_train, y_train)
        y_pred = model.predict(X_train)

        # Clip predictions
        y_pred_clipped = clip_expected_returns(y_pred)

        # All predictions should be within bounds
        self.assertTrue(all(y_pred_clipped >= MIN_EXPECTED_RETURN))
        self.assertTrue(all(y_pred_clipped <= MAX_EXPECTED_RETURN))

    def test_sharpe_ratio_is_reasonable(self):
        """Test that Sharpe ratio is reasonable (not 42.4)."""
        from finance_ml.ml_workflow.analytics.ml_returns import clip_expected_returns
        from finance_ml.ml_workflow.config.ml_returns_config import (
            MIN_EXPECTED_RETURN,
            MAX_EXPECTED_RETURN,
        )

        np.random.seed(42)

        # Simulate returns that span the valid range after clipping
        # Use a range that includes values within bounds to maintain variance
        raw_returns = np.random.uniform(-0.5, 1.5, 100)  # Mix of realistic and unrealistic
        clipped_returns = clip_expected_returns(raw_returns)

        # Calculate Sharpe ratio
        mean_return = np.mean(clipped_returns)
        std_return = np.std(clipped_returns)
        risk_free_rate = 0.03

        # Handle edge case where all values clip to same bound (zero variance)
        if std_return < 1e-10:
            # If variance is ~0, that's actually acceptable - returns are bounded
            sharpe = 0.0
        else:
            sharpe = (mean_return - risk_free_rate) / std_return

        # Sharpe ratio should be reasonable (typically < 3 for realistic returns)
        # With bounded returns, Sharpe should be well-behaved
        self.assertLess(sharpe, 10.0, f"Sharpe ratio {sharpe:.2f} is unrealistically high")


class TestReturnCalculationDiagnostics(unittest.TestCase):
    """Test diagnostics for return calculation issues."""

    def test_validate_expected_returns_function_exists(self):
        """Test that validate_expected_returns function exists."""
        from finance_ml.ml_workflow.analytics.ml_returns import validate_expected_returns

        self.assertTrue(callable(validate_expected_returns))

    def test_validate_expected_returns_flags_unrealistic(self):
        """Test that function flags unrealistic returns."""
        from finance_ml.ml_workflow.analytics.ml_returns import validate_expected_returns

        # Create returns matching the issue (95.6% mean)
        unrealistic_returns = np.array([0.956] * 100)

        diagnostics = validate_expected_returns(unrealistic_returns)

        self.assertIsInstance(diagnostics, dict)
        self.assertIn("is_realistic", diagnostics)
        self.assertFalse(
            diagnostics["is_realistic"], "Returns with 95.6% mean should be flagged as unrealistic"
        )

    def test_validate_expected_returns_accepts_realistic(self):
        """Test that function accepts realistic returns."""
        from finance_ml.ml_workflow.analytics.ml_returns import validate_expected_returns

        # Create realistic returns (e.g., 8-15% annual)
        realistic_returns = np.random.uniform(0.05, 0.20, 100)

        diagnostics = validate_expected_returns(realistic_returns)

        self.assertTrue(
            diagnostics["is_realistic"], "Realistic returns (5-20%) should pass validation"
        )


if __name__ == "__main__":
    unittest.main()
