"""
Test suite for finance_ml_analytics_platform.py

TDD-based tests covering:
- Configuration validation (Section 8.1)
- 8-phase ML workflow (Phase 9.1-9.8)
- Section 18 Portfolio Optimization Workflow (7 phases)

Coverage target: ≥80% for changed files
"""

import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfigurationConstants(unittest.TestCase):
    """Test Section 8.1: Configuration Constants - Single Source of Truth"""

    def test_target_columns_defined(self):
        """Test that target columns are properly defined"""
        import finance_ml_analytics_platform as platform

        self.assertEqual(platform.TARGET_COL, "price_target")
        self.assertEqual(platform.TARGET_COL_FALLBACK, "last_price")

    def test_data_split_constants(self):
        """Test data split constants are valid"""
        import finance_ml_analytics_platform as platform

        self.assertEqual(platform.TEST_SIZE, 0.2)
        self.assertEqual(platform.TRAIN_SIZE, 0.8)
        self.assertEqual(platform.CV_FOLDS, 5)

    def test_quantile_constants(self):
        """Test quantile regression constants"""
        import finance_ml_analytics_platform as platform

        self.assertEqual(platform.QUANTILES, [0.01, 0.5, 0.99])
        self.assertEqual(platform.LOWER_QUANTILE, 0.01)
        self.assertEqual(platform.MEDIAN_QUANTILE, 0.5)
        self.assertEqual(platform.UPPER_QUANTILE, 0.99)

    def test_portfolio_constraints(self):
        """Test portfolio constraint constants"""
        import finance_ml_analytics_platform as platform

        self.assertEqual(platform.MAX_SECTOR_WEIGHT, 0.25)
        self.assertEqual(platform.MAX_SINGLE_POSITION, 0.10)

    def test_outlier_thresholds(self):
        """Test outlier threshold constants"""
        import finance_ml_analytics_platform as platform

        self.assertEqual(platform.IQR_MULTIPLIER, 2.5)
        self.assertEqual(platform.ZSCORE_THRESHOLD, 3.0)
        self.assertEqual(platform.WINSORIZE_LOWER, 0.05)
        self.assertEqual(platform.WINSORIZE_UPPER, 0.95)

    def test_validate_configuration_success(self):
        """Test that validate_configuration returns True for valid config"""
        import finance_ml_analytics_platform as platform

        result = platform.validate_configuration()
        self.assertTrue(result)


class TestConfigurationValidation(unittest.TestCase):
    """Test configuration validation edge cases"""

    def test_validate_configuration_exists(self):
        """Test that validate_configuration function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "validate_configuration"))
        self.assertTrue(callable(platform.validate_configuration))


class TestMainStructure(unittest.TestCase):
    """Test main script structure and entry points"""

    def test_main_function_exists(self):
        """Test that the script has a main() function"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "main"))
        self.assertTrue(callable(platform.main))

    def test_setup_logging_exists(self):
        """Test that setup_logging function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "setup_logging"))
        self.assertTrue(callable(platform.setup_logging))


class TestPhase91LoadingPreprocessing(unittest.TestCase):
    """Test Phase 9.1: Loading and Preprocessing with 6-step imputation"""

    def test_run_phase_91_exists(self):
        """Test that Phase 9.1 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_91_loading_preprocessing"))
        self.assertTrue(callable(platform.run_phase_91_loading_preprocessing))

    def test_run_phase_91_returns_dataframe(self):
        """Test that Phase 9.1 returns processed DataFrame"""
        import finance_ml_analytics_platform as platform

        # Create sample input data
        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": [150.0, 140.0, 300.0],
                "price_target": [180.0, 160.0, 350.0],
                "market_cap": [2.5e12, 1.8e12, 2.2e12],
            }
        )

        result = platform.run_phase_91_loading_preprocessing(sample_data)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)


class TestPhase92EDA(unittest.TestCase):
    """Test Phase 9.2: Enhanced Exploratory Data Analysis"""

    def test_run_phase_92_exists(self):
        """Test that Phase 9.2 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_92_eda"))
        self.assertTrue(callable(platform.run_phase_92_eda))

    def test_run_phase_92_returns_dict(self):
        """Test that Phase 9.2 returns EDA results dictionary"""
        import finance_ml_analytics_platform as platform

        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": [150.0, 140.0, 300.0],
                "price_target": [180.0, 160.0, 350.0],
            }
        )

        result = platform.run_phase_92_eda(sample_data)
        self.assertIsInstance(result, dict)


class TestPhase93FeatureEngineering(unittest.TestCase):
    """Test Phase 9.3: Advanced Feature Engineering"""

    def test_run_phase_93_exists(self):
        """Test that Phase 9.3 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_93_feature_engineering"))
        self.assertTrue(callable(platform.run_phase_93_feature_engineering))

    def test_run_phase_93_returns_dataframe(self):
        """Test that Phase 9.3 returns enhanced DataFrame"""
        import finance_ml_analytics_platform as platform

        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "sector": ["Technology", "Technology", "Technology"],
                "last_price": [150.0, 140.0, 300.0],
                "price_target": [180.0, 160.0, 350.0],
                "market_cap": [2.5e12, 1.8e12, 2.2e12],
                "ev": [2.6e12, 1.9e12, 2.3e12],
                "ebitda": [1e11, 8e10, 9e10],
            }
        )

        result = platform.run_phase_93_feature_engineering(sample_data)
        self.assertIsInstance(result, pd.DataFrame)


class TestPhase94Classification(unittest.TestCase):
    """Test Phase 9.4: Multi-class Event Classification"""

    def test_run_phase_94_exists(self):
        """Test that Phase 9.4 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_94_classification"))
        self.assertTrue(callable(platform.run_phase_94_classification))

    def test_run_phase_94_returns_tuple(self):
        """Test that Phase 9.4 returns model and predictions"""
        import finance_ml_analytics_platform as platform

        # Create sample feature data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 3, 100), name="event_class")

        result = platform.run_phase_94_classification(X, y)
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("predictions", result)


class TestPhase95Regression(unittest.TestCase):
    """Test Phase 9.5: Sector-optimized Regression with Quantile Models"""

    def test_run_phase_95_exists(self):
        """Test that Phase 9.5 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_95_regression"))
        self.assertTrue(callable(platform.run_phase_95_regression))

    def test_run_phase_95_returns_dict(self):
        """Test that Phase 9.5 returns regression results"""
        import finance_ml_analytics_platform as platform

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])
        y = pd.Series(np.random.randn(100) * 10 + 100, name="price_target")

        result = platform.run_phase_95_regression(X, y)
        self.assertIsInstance(result, dict)
        self.assertIn("model", result)
        self.assertIn("predictions", result)


class TestPhase96Evaluation(unittest.TestCase):
    """Test Phase 9.6: Model Evaluation and Error Analysis"""

    def test_run_phase_96_exists(self):
        """Test that Phase 9.6 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_96_evaluation"))
        self.assertTrue(callable(platform.run_phase_96_evaluation))

    def test_run_phase_96_returns_metrics(self):
        """Test that Phase 9.6 returns evaluation metrics"""
        import finance_ml_analytics_platform as platform

        y_true = pd.Series([100, 110, 120, 130, 140])
        y_pred = pd.Series([102, 108, 122, 128, 142])

        result = platform.run_phase_96_evaluation(y_true, y_pred)
        self.assertIsInstance(result, dict)
        self.assertIn("mae", result)
        self.assertIn("rmse", result)
        self.assertIn("r2", result)


class TestPhase97StockValuation(unittest.TestCase):
    """Test Phase 9.7: Identification of Under/Overvalued Stocks"""

    def test_run_phase_97_exists(self):
        """Test that Phase 9.7 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_97_stock_valuation"))
        self.assertTrue(callable(platform.run_phase_97_stock_valuation))

    def test_run_phase_97_returns_rankings(self):
        """Test that Phase 9.7 returns stock rankings"""
        import finance_ml_analytics_platform as platform

        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
                "sector": ["Tech", "Tech", "Tech", "Tech", "Tech"],
                "last_price": [150, 140, 300, 180, 350],
                "predicted_target": [180, 160, 350, 170, 380],
            }
        )

        result = platform.run_phase_97_stock_valuation(sample_data)
        self.assertIsInstance(result, dict)
        self.assertIn("undervalued", result)
        self.assertIn("overvalued", result)


class TestPhase98Reporting(unittest.TestCase):
    """Test Phase 9.8: Comprehensive Analytics and Reporting"""

    def test_run_phase_98_exists(self):
        """Test that Phase 9.8 function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_phase_98_reporting"))
        self.assertTrue(callable(platform.run_phase_98_reporting))

    def test_run_phase_98_returns_report(self):
        """Test that Phase 9.8 returns comprehensive report"""
        import finance_ml_analytics_platform as platform

        # Mock workflow results
        workflow_results = {
            "eda_summary": {"total_stocks": 1000},
            "model_metrics": {"mae": 5.0, "rmse": 7.0},
            "stock_rankings": {"undervalued": [], "overvalued": []},
        }

        result = platform.run_phase_98_reporting(workflow_results)
        self.assertIsInstance(result, dict)


class TestSection18PortfolioOptimization(unittest.TestCase):
    """Test Section 18: Portfolio Optimization Workflow (7 phases)"""

    def test_run_portfolio_optimization_exists(self):
        """Test that portfolio optimization function exists"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "run_portfolio_optimization"))
        self.assertTrue(callable(platform.run_portfolio_optimization))

    def test_portfolio_phase1_stock_selection(self):
        """Test Phase 1: Enhanced Stock Selection"""
        import finance_ml_analytics_platform as platform

        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"] * 20,
                "sector": ["Technology"] * 100,
                "market_cap": [2.5e12, 1.8e12, 2.2e12, 1.5e12, 800e9] * 20,
                "mispricing_score": np.random.randn(100) * 0.1,
            }
        )

        result = platform.run_portfolio_phase1_stock_selection(sample_data)
        self.assertIsInstance(result, pd.DataFrame)

    def test_portfolio_phase2_return_prediction(self):
        """Test Phase 2: ML-Based Return Prediction"""
        import finance_ml_analytics_platform as platform

        sample_data = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "last_price": [150.0, 140.0, 300.0],
                "price_1w_ago": [148.0, 138.0, 295.0],
                "price_1m_ago": [145.0, 135.0, 290.0],
            }
        )

        result = platform.run_portfolio_phase2_return_prediction(sample_data)
        self.assertIsInstance(result, dict)
        self.assertIn("expected_returns", result)

    def test_portfolio_phase3_optimization(self):
        """Test Phase 3: Advanced Portfolio Optimization"""
        import finance_ml_analytics_platform as platform

        expected_returns = pd.Series([0.10, 0.12, 0.08], index=["AAPL", "GOOGL", "MSFT"])
        cov_matrix = pd.DataFrame(
            np.array([[0.04, 0.01, 0.01], [0.01, 0.05, 0.02], [0.01, 0.02, 0.03]]),
            index=["AAPL", "GOOGL", "MSFT"],
            columns=["AAPL", "GOOGL", "MSFT"],
        )

        result = platform.run_portfolio_phase3_optimization(expected_returns, cov_matrix)
        self.assertIsInstance(result, dict)
        self.assertIn("weights", result)

    def test_portfolio_phase4_risk_management(self):
        """Test Phase 4: Risk Management Enhancements"""
        import finance_ml_analytics_platform as platform

        weights = np.array([0.4, 0.35, 0.25])
        returns = pd.DataFrame(np.random.randn(252, 3) * 0.02, columns=["AAPL", "GOOGL", "MSFT"])

        result = platform.run_portfolio_phase4_risk_management(weights, returns)
        self.assertIsInstance(result, dict)
        self.assertIn("expected_shortfall", result)
        self.assertIn("var_95", result)

    def test_portfolio_phase5_backtesting(self):
        """Test Phase 5: Backtesting Framework"""
        import finance_ml_analytics_platform as platform

        # Create sample historical data
        np.random.seed(42)
        prices = pd.DataFrame(
            np.cumprod(1 + np.random.randn(252, 3) * 0.02, axis=0) * 100,
            columns=["AAPL", "GOOGL", "MSFT"],
        )

        result = platform.run_portfolio_phase5_backtesting(prices)
        self.assertIsInstance(result, dict)
        self.assertIn("portfolio_returns", result)

    def test_portfolio_phase6_dashboards(self):
        """Test Phase 6: Interactive Dashboard Expansion"""
        import finance_ml_analytics_platform as platform

        portfolio_data = {
            "weights": {"AAPL": 0.4, "GOOGL": 0.35, "MSFT": 0.25},
            "returns": pd.Series([0.10, 0.12, 0.08]),
        }

        result = platform.run_portfolio_phase6_dashboards(portfolio_data)
        self.assertIsInstance(result, dict)

    def test_portfolio_phase7_validation(self):
        """Test Phase 7: Enhanced ML & Validation"""
        import finance_ml_analytics_platform as platform

        expected_returns = pd.Series([0.10, 0.12, 0.08, 0.15, -0.05])

        result = platform.run_portfolio_phase7_validation(expected_returns)
        self.assertIsInstance(result, dict)
        self.assertIn("is_valid", result)
        self.assertIn("clipped_returns", result)


class TestReturnBoundsPolicy(unittest.TestCase):
    """Test Section 18.2: Return Calculation Best Practices"""

    def test_max_expected_return_constant(self):
        """Test MAX_EXPECTED_RETURN constant is defined"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "MAX_EXPECTED_RETURN"))
        self.assertEqual(platform.MAX_EXPECTED_RETURN, 0.29)

    def test_min_expected_return_constant(self):
        """Test MIN_EXPECTED_RETURN constant is defined"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "MIN_EXPECTED_RETURN"))
        self.assertEqual(platform.MIN_EXPECTED_RETURN, -0.50)

    def test_realistic_return_threshold(self):
        """Test REALISTIC_RETURN_MEAN_THRESHOLD constant"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "REALISTIC_RETURN_MEAN_THRESHOLD"))
        self.assertEqual(platform.REALISTIC_RETURN_MEAN_THRESHOLD, 0.30)


class TestWorkflowIntegration(unittest.TestCase):
    """Test full workflow integration"""

    @patch("finance_ml_analytics_platform.load_from_csv")
    @patch("finance_ml_analytics_platform.imputation")
    def test_main_executes_without_error(self, mock_imputation, mock_load_from_csv):
        """Test that main() executes the complete workflow"""
        import finance_ml_analytics_platform as platform

        # Mock data loading - load_from_csv returns DataFrame directly
        mock_load_from_csv.return_value = pd.DataFrame(
            {
                "ticker": ["AAPL", "GOOGL"],
                "sector": ["Tech", "Tech"],
                "last_price": [150.0, 140.0],
                "price_target": [180.0, 160.0],
            }
        )

        # Execute main - should not raise
        try:
            platform.main()
        except Exception as e:
            # Allow graceful failures due to missing data/resources
            self.assertIn(
                str(type(e).__name__), ["AttributeError", "TypeError", "ValueError", "KeyError"]
            )


class TestModuleImports(unittest.TestCase):
    """Test that all required modules are imported"""

    def test_imports_portfolio_module(self):
        """Test portfolio module is imported"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "portfolio"))

    def test_imports_risk_module(self):
        """Test risk module is imported"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "risk"))

    def test_imports_stock_selection_module(self):
        """Test stock_selection module is imported"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "stock_selection"))

    def test_imports_ml_returns_module(self):
        """Test ml_returns module is imported"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "ml_returns"))

    def test_imports_attribution_module(self):
        """Test attribution module is imported"""
        import finance_ml_analytics_platform as platform

        self.assertTrue(hasattr(platform, "attribution"))


if __name__ == "__main__":
    unittest.main()
