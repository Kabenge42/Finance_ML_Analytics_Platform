"""
Test suite for Phase 9.3-9.8 notebook imports validation.

This test validates that all Phase 9.3-9.8 functions imported in ml_finance_model_main.ipynb
are available at the package level and have the expected signatures.

Test Coverage:
- Phase 9.3 (features): features_build_comprehensive, features_importance_rf, etc.
- Phase 9.4 (classification): classification_create_enhanced_event_labels, etc.
- Phase 9.5 (regression): regression_train_xgboost, regression_compare_regressors, etc.
- Phase 9.6 (evaluation): evaluation_comprehensive_metrics, etc.
- Phase 9.7 (analytics): analytics_calculate_mispricing, analytics_rank_undervalued, etc.
- Phase 9.8 (reporting): reporting_financial_metrics, reporting_quality_alerts
"""

import unittest
import inspect
from typing import Callable


class TestNotebookPhase93to98Imports(unittest.TestCase):
    """Test Phase 9.3-9.8 imports match notebook expectations."""

    def test_phase93_features_imports_available(self):
        """Verify Phase 9.3 feature engineering functions are available."""
        from finance_ml import (
            features_build_comprehensive,
            features_importance_rf,
            engineer_valuation_ratios,
            engineer_analyst_quality_features,
        )

        # Verify all are callable
        self.assertTrue(callable(features_build_comprehensive))
        self.assertTrue(callable(features_importance_rf))
        self.assertTrue(callable(engineer_valuation_ratios))
        self.assertTrue(callable(engineer_analyst_quality_features))

    def test_phase94_classification_imports_available(self):
        """Verify Phase 9.4 classification functions are available."""
        from finance_ml import (
            classification_create_enhanced_event_labels,
            classification_optimize_hyperparameters,
        )

        # Verify all are callable
        self.assertTrue(callable(classification_create_enhanced_event_labels))
        self.assertTrue(callable(classification_optimize_hyperparameters))

    def test_phase95_regression_imports_available(self):
        """Verify Phase 9.5 regression functions are available."""
        from finance_ml import (
            regression_prepare_data,
            regression_train_xgboost,
            regression_train_lightgbm,
            regression_train_catboost,
            regression_compare_regressors,
            regression_train_sector_models,
            regression_save_model,
            regression_load_model,
            regression_create_classification_interactions,
            regression_train_stacking,
            regression_train_quantile,
        )

        # Verify all are callable
        functions = [
            regression_prepare_data,
            regression_train_xgboost,
            regression_train_lightgbm,
            regression_train_catboost,
            regression_compare_regressors,
            regression_train_sector_models,
            regression_save_model,
            regression_load_model,
            regression_create_classification_interactions,
            regression_train_stacking,
            regression_train_quantile,
        ]
        for func in functions:
            self.assertTrue(callable(func), f"{func.__name__} is not callable")

    def test_phase96_evaluation_imports_available(self):
        """Verify Phase 9.6 evaluation functions are available."""
        from finance_ml import (
            evaluation_comprehensive_metrics,
            evaluation_metrics_by_segment,
        )

        # Verify all are callable
        self.assertTrue(callable(evaluation_comprehensive_metrics))
        self.assertTrue(callable(evaluation_metrics_by_segment))

    def test_phase97_analytics_imports_available(self):
        """Verify Phase 9.7 analytics functions are available."""
        from finance_ml import (
            analytics_calculate_mispricing,
            analytics_rank_undervalued,
            analytics_rank_overvalued,
            analytics_rank_by_sector,
        )

        # Verify all are callable
        self.assertTrue(callable(analytics_calculate_mispricing))
        self.assertTrue(callable(analytics_rank_undervalued))
        self.assertTrue(callable(analytics_rank_overvalued))
        self.assertTrue(callable(analytics_rank_by_sector))

    def test_phase98_reporting_imports_available(self):
        """Verify Phase 9.8 reporting functions are available."""
        from finance_ml import (
            reporting_financial_metrics,
            reporting_quality_alerts,
        )

        # Verify all are callable
        self.assertTrue(callable(reporting_financial_metrics))
        self.assertTrue(callable(reporting_quality_alerts))

    def test_phase92_eda_imports_available(self):
        """Verify Phase 9.2 EDA functions are available at package level."""
        from finance_ml import (
            generate_eda_report,
            generate_benchmarking_report,
            compare_sector_distributions,
            compare_regional_valuations,
            simple_eda,
        )

        # Verify all are callable
        self.assertTrue(callable(generate_eda_report))
        self.assertTrue(callable(generate_benchmarking_report))
        self.assertTrue(callable(compare_sector_distributions))
        self.assertTrue(callable(compare_regional_valuations))
        self.assertTrue(callable(simple_eda))

    def test_data_catalog_import_available(self):
        """Verify DataCatalog is available at package level."""
        from finance_ml import DataCatalog

        # Verify it's a class
        self.assertTrue(inspect.isclass(DataCatalog))

    def test_module_imports_available(self):
        """Verify ml_workflow module imports work."""
        from finance_ml.ml_workflow import analyst_comparison, portfolio_optimization, risk_metrics

        # Verify all are modules
        self.assertTrue(inspect.ismodule(analyst_comparison))
        self.assertTrue(inspect.ismodule(portfolio_optimization))
        self.assertTrue(inspect.ismodule(risk_metrics))

    def test_phase93_function_signatures_valid(self):
        """Verify Phase 9.3 functions have valid signatures."""
        from finance_ml import features_build_comprehensive, features_importance_rf

        # Check features_build_comprehensive accepts df parameter
        sig = inspect.signature(features_build_comprehensive)
        param_names = list(sig.parameters.keys())
        self.assertIn(
            "df", param_names, "features_build_comprehensive should accept 'df' parameter"
        )

        # Check features_importance_rf accepts required parameters
        sig_rf = inspect.signature(features_importance_rf)
        param_names_rf = list(sig_rf.parameters.keys())
        self.assertIn("X", param_names_rf, "features_importance_rf should accept 'X' parameter")
        self.assertIn("y", param_names_rf, "features_importance_rf should accept 'y' parameter")

    def test_phase95_regression_signatures_valid(self):
        """Verify Phase 9.5 regression training functions have valid signatures."""
        from finance_ml import regression_train_xgboost, regression_train_lightgbm

        # Check regression_train_xgboost signature (uses X, y parameters)
        sig = inspect.signature(regression_train_xgboost)
        param_names = list(sig.parameters.keys())
        self.assertIn("X", param_names, "regression_train_xgboost should accept 'X' parameter")
        self.assertIn("y", param_names, "regression_train_xgboost should accept 'y' parameter")

        # Check regression_train_lightgbm signature (uses X, y parameters)
        sig_lgbm = inspect.signature(regression_train_lightgbm)
        param_names_lgbm = list(sig_lgbm.parameters.keys())
        self.assertIn(
            "X", param_names_lgbm, "regression_train_lightgbm should accept 'X' parameter"
        )
        self.assertIn(
            "y", param_names_lgbm, "regression_train_lightgbm should accept 'y' parameter"
        )

    def test_phase97_analytics_signatures_valid(self):
        """Verify Phase 9.7 analytics functions have valid signatures."""
        from finance_ml import analytics_calculate_mispricing, analytics_rank_undervalued

        # Check analytics_calculate_mispricing signature
        sig = inspect.signature(analytics_calculate_mispricing)
        param_names = list(sig.parameters.keys())
        # Should accept some form of actual and predicted values
        self.assertTrue(
            len(param_names) >= 2,
            "analytics_calculate_mispricing should accept at least 2 parameters",
        )

        # Check analytics_rank_undervalued signature
        sig_rank = inspect.signature(analytics_rank_undervalued)
        param_names_rank = list(sig_rank.parameters.keys())
        # Should accept dataframe or scores
        self.assertTrue(
            len(param_names_rank) >= 1,
            "analytics_rank_undervalued should accept at least 1 parameter",
        )

    def test_no_deprecated_imports_in_notebook_pattern(self):
        """Verify the notebook doesn't use old deprecated import patterns."""
        # This test documents the correct import pattern
        # If the notebook used old patterns, these imports would fail
        try:
            from finance_ml import (
                features_build_comprehensive,  # Not build_comprehensive_features
                classification_create_enhanced_event_labels,  # Not create_enhanced_event_labels
                regression_train_xgboost,  # Not train_xgboost_regressor
                evaluation_comprehensive_metrics,  # Not comprehensive_regression_metrics
                analytics_calculate_mispricing,  # Not calculate_mispricing_score
                reporting_financial_metrics,  # Not calculate_financial_metrics_dashboard
            )

            # If we get here, new names are available
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"New Phase 9.3-9.8 names not available: {e}")

    def test_all_notebook_imports_in_single_block(self):
        """Verify all notebook imports work in a single import block."""
        # This mimics the exact import structure from the notebook
        try:
            # Phase 9.2
            from finance_ml import (
                generate_eda_report,
                generate_benchmarking_report,
                compare_sector_distributions,
                compare_regional_valuations,
                simple_eda,
            )

            # Phase 9.3
            from finance_ml import (
                features_build_comprehensive,
                features_importance_rf,
                engineer_valuation_ratios,
                engineer_analyst_quality_features,
            )

            # Phase 9.4
            from finance_ml import (
                classification_create_enhanced_event_labels,
                classification_optimize_hyperparameters,
            )

            # Phase 9.5
            from finance_ml import (
                regression_prepare_data,
                regression_train_xgboost,
                regression_train_lightgbm,
                regression_train_catboost,
                regression_compare_regressors,
                regression_train_sector_models,
                regression_save_model,
                regression_load_model,
                regression_create_classification_interactions,
                regression_train_stacking,
                regression_train_quantile,
            )

            # Phase 9.6
            from finance_ml import (
                evaluation_comprehensive_metrics,
                evaluation_metrics_by_segment,
            )

            # Phase 9.7
            from finance_ml import (
                analytics_calculate_mispricing,
                analytics_rank_undervalued,
                analytics_rank_overvalued,
                analytics_rank_by_sector,
            )

            # Phase 9.8
            from finance_ml import (
                reporting_financial_metrics,
                reporting_quality_alerts,
            )

            # Module imports
            from finance_ml.ml_workflow import (
                analyst_comparison,
                portfolio_optimization,
                risk_metrics,
            )

            # Data catalog
            from finance_ml import DataCatalog

            self.assertTrue(True, "All notebook imports successful")

        except ImportError as e:
            self.fail(f"Notebook import block failed: {e}")


if __name__ == "__main__":
    unittest.main()
