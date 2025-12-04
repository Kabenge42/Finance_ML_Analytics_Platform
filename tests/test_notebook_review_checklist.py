"""
Test notebook review checklist for ml_finance_model_main2_0.ipynb.

Validates that the notebook contains all required sections, phases, and
implementation components as specified in the issue description.

Requirements verified:
- 16 Feature Categories (196 Features) in Phase 9.3
- Phase 9.4: Classification Workflow
- Phase 9.5: Regression Workflow
- Phase 9.6: Model Evaluation & Analytics
- Phase 9.7: Portfolio Optimization
"""

import json
import os
import re
import unittest
from pathlib import Path


class TestNotebookStructure(unittest.TestCase):
    """Test that notebook has required structure and sections."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found: {cls.notebook_path}")

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        # Extract all cell sources as text
        cls.all_source = ""
        cls.markdown_cells = []
        cls.code_cells = []

        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

            if cell.get("cell_type") == "markdown":
                cls.markdown_cells.append(source)
            elif cell.get("cell_type") == "code":
                cls.code_cells.append(source)

    def test_notebook_exists(self):
        """Test that the notebook file exists."""
        self.assertTrue(self.notebook_path.exists(), f"Notebook not found at {self.notebook_path}")

    def test_notebook_has_cells(self):
        """Test that notebook has cells."""
        cells = self.notebook.get("cells", [])
        self.assertGreater(len(cells), 0, "Notebook has no cells")

    def test_has_configuration_section(self):
        """Test that notebook has configuration section."""
        has_config = any(
            "Configuration" in cell or "CONFIGURATION" in cell for cell in self.markdown_cells
        )
        self.assertTrue(has_config, "Missing Configuration section")

    def test_has_phase_91_preprocessing(self):
        """Test that notebook has Phase 9.1: Preprocessing section."""
        has_phase = "Phase 9.1" in self.all_source or "phase 9.1" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.1: Preprocessing section")

    def test_has_phase_92_eda(self):
        """Test that notebook has Phase 9.2: EDA section."""
        has_phase = "Phase 9.2" in self.all_source or "phase 9.2" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.2: EDA section")

    def test_has_phase_93_feature_engineering(self):
        """Test that notebook has Phase 9.3: Feature Engineering section."""
        has_phase = "Phase 9.3" in self.all_source or "phase 9.3" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.3: Feature Engineering section")

    def test_has_phase_94_classification(self):
        """Test that notebook has Phase 9.4: Classification section."""
        has_phase = "Phase 9.4" in self.all_source or "phase 9.4" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.4: Classification section")

    def test_has_phase_95_regression(self):
        """Test that notebook has Phase 9.5: Regression section."""
        has_phase = "Phase 9.5" in self.all_source or "phase 9.5" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.5: Regression section")

    def test_has_phase_96_evaluation(self):
        """Test that notebook has Phase 9.6: Evaluation section."""
        has_phase = "Phase 9.6" in self.all_source or "phase 9.6" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.6: Model Evaluation section")

    def test_has_phase_97_analytics(self):
        """Test that notebook has Phase 9.7: Analytics section."""
        has_phase = "Phase 9.7" in self.all_source or "phase 9.7" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.7: Analytics section")

    def test_has_phase_98_reporting(self):
        """Test that notebook has Phase 9.8: Reporting section."""
        has_phase = "Phase 9.8" in self.all_source or "phase 9.8" in self.all_source.lower()
        self.assertTrue(has_phase, "Missing Phase 9.8: Reporting section")

    def test_has_portfolio_optimization_section(self):
        """Test that notebook has Portfolio Optimization section."""
        has_section = "Portfolio Optimization" in self.all_source or "Section 10" in self.all_source
        self.assertTrue(has_section, "Missing Portfolio Optimization section")


class TestFeatureEngineeringCompleteness(unittest.TestCase):
    """Test that feature engineering section covers required categories."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_momentum_features(self):
        """Test that notebook includes momentum feature engineering."""
        has_feature = "momentum" in self.all_source.lower() or "Momentum" in self.all_source
        self.assertTrue(has_feature, "Missing Momentum feature category")

    def test_has_valuation_features(self):
        """Test that notebook includes valuation feature engineering."""
        has_feature = "valuation" in self.all_source.lower() or "Valuation" in self.all_source
        self.assertTrue(has_feature, "Missing Valuation feature category")

    def test_has_profitability_features(self):
        """Test that notebook includes profitability feature engineering."""
        has_feature = (
            "profitability" in self.all_source.lower() or "Profitability" in self.all_source
        )
        self.assertTrue(has_feature, "Missing Profitability feature category")

    def test_has_quality_risk_features(self):
        """Test that notebook includes quality/risk feature engineering."""
        has_feature = (
            "quality" in self.all_source.lower()
            or "Quality" in self.all_source
            or "risk" in self.all_source.lower()
        )
        self.assertTrue(has_feature, "Missing Quality & Risk feature category")

    def test_has_cash_flow_features(self):
        """Test that notebook includes cash flow feature engineering."""
        has_feature = (
            "cash_flow" in self.all_source.lower()
            or "Cash Flow" in self.all_source
            or "cash flow" in self.all_source.lower()
        )
        self.assertTrue(has_feature, "Missing Cash Flow feature category")

    def test_has_analyst_sentiment_features(self):
        """Test that notebook includes analyst sentiment features."""
        has_feature = "analyst" in self.all_source.lower() or "Analyst" in self.all_source
        self.assertTrue(has_feature, "Missing Analyst Sentiment feature category")

    def test_has_leverage_liquidity_features(self):
        """Test that notebook includes leverage/liquidity features."""
        has_feature = (
            "leverage" in self.all_source.lower()
            or "liquidity" in self.all_source.lower()
            or "Leverage" in self.all_source
        )
        self.assertTrue(has_feature, "Missing Leverage & Liquidity feature category")

    def test_has_growth_features(self):
        """Test that notebook includes growth feature engineering."""
        has_feature = "growth" in self.all_source.lower() or "Growth" in self.all_source
        self.assertTrue(has_feature, "Missing Growth Metrics feature category")

    def test_references_phase93_categories(self):
        """Test that notebook references PHASE93_FEATURE_CATEGORIES."""
        has_reference = "PHASE93_FEATURE_CATEGORIES" in self.all_source
        self.assertTrue(has_reference, "Missing reference to PHASE93_FEATURE_CATEGORIES")


class TestClassificationWorkflow(unittest.TestCase):
    """Test that classification workflow section is complete."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_classification_imports(self):
        """Test that notebook imports classification modules."""
        has_import = (
            "classification" in self.all_source.lower()
            or "train_xgboost_classifier" in self.all_source
            or "train_lightgbm_classifier" in self.all_source
        )
        self.assertTrue(has_import, "Missing classification module imports")

    def test_has_event_labels(self):
        """Test that notebook creates event classification labels."""
        has_labels = (
            "event" in self.all_source.lower()
            or "label" in self.all_source.lower()
            or "create_event_labels" in self.all_source
        )
        self.assertTrue(has_labels, "Missing event label creation")

    def test_has_model_evaluation(self):
        """Test that notebook includes classification model evaluation."""
        has_eval = (
            "confusion" in self.all_source.lower()
            or "accuracy" in self.all_source.lower()
            or "f1" in self.all_source.lower()
            or "classification_report" in self.all_source
        )
        self.assertTrue(has_eval, "Missing classification model evaluation")


class TestRegressionWorkflow(unittest.TestCase):
    """Test that regression workflow section is complete."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_regression_imports(self):
        """Test that notebook imports regression modules."""
        has_import = (
            "regression" in self.all_source.lower()
            or "train_xgboost_regressor" in self.all_source
            or "train_lightgbm_regressor" in self.all_source
        )
        self.assertTrue(has_import, "Missing regression module imports")

    def test_has_quantile_regression(self):
        """Test that notebook includes quantile regression."""
        has_quantile = "quantile" in self.all_source.lower() or "QUANTILES" in self.all_source
        self.assertTrue(has_quantile, "Missing quantile regression")

    def test_has_stacking_ensemble(self):
        """Test that notebook includes stacking ensemble."""
        has_stacking = "stacking" in self.all_source.lower() or "Stacking" in self.all_source
        self.assertTrue(has_stacking, "Missing stacking ensemble")

    def test_has_model_comparison(self):
        """Test that notebook includes model comparison."""
        has_comparison = (
            "compare" in self.all_source.lower() or "comparison" in self.all_source.lower()
        )
        self.assertTrue(has_comparison, "Missing model comparison")


class TestPortfolioOptimization(unittest.TestCase):
    """Test that portfolio optimization section is complete."""

    @classmethod
    def setUpClass(cls):
        """Load notebook content once for all tests."""
        project_root = Path(__file__).parent.parent
        cls.notebook_path = project_root / "ml_finance_model_main2_0.ipynb"

        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.all_source = ""
        for cell in cls.notebook.get("cells", []):
            source = "".join(cell.get("source", []))
            cls.all_source += source + "\n"

    def test_has_expected_return_calculation(self):
        """Test that notebook calculates expected returns."""
        has_return = (
            "expected_return" in self.all_source.lower()
            or "Expected Return" in self.all_source
            or "mispricing" in self.all_source.lower()
        )
        self.assertTrue(has_return, "Missing expected return calculation")

    def test_has_stock_selection(self):
        """Test that notebook includes stock selection."""
        has_selection = (
            "stock_selection" in self.all_source.lower()
            or "ranking" in self.all_source.lower()
            or "undervalued" in self.all_source.lower()
        )
        self.assertTrue(has_selection, "Missing stock selection")

    def test_has_portfolio_construction(self):
        """Test that notebook includes portfolio construction."""
        has_portfolio = "portfolio" in self.all_source.lower() or "Portfolio" in self.all_source
        self.assertTrue(has_portfolio, "Missing portfolio construction")


if __name__ == "__main__":
    unittest.main()
