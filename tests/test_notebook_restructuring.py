"""
Test suite for notebook restructuring - TDD approach for ml_finance_model_main.ipynb

This test suite validates the restructured notebook following the comprehensive
restructuring plan, ensuring:
1. Proper section structure (10 sections as per plan)
2. Package function usage instead of inline helpers
3. Correct module imports and configuration
4. Business objective alignment
5. Production-ready code quality

Following strict TDD: write failing tests first, implement, refactor.
"""

import unittest
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


class TestNotebookStructure(unittest.TestCase):
    """Test notebook structure and organization."""

    @classmethod
    def setUpClass(cls):
        """Load notebook file for analysis."""
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Parse notebook (simple JSON or cell-by-cell format)
                if content.startswith("{"):
                    cls.notebook = json.loads(content)
                    cls.cells = cls.notebook.get("cells", [])
                else:
                    # Parse #%% format
                    cls.cells = cls._parse_percent_format(content)
        else:
            cls.cells = []

    @staticmethod
    def _parse_percent_format(content: str) -> List[Dict]:
        """Parse #%% format notebook cells."""
        cells = []
        cell_pattern = r"#%%(?:\s+md)?\n(.*?)(?=#%%|\Z)"
        matches = re.finditer(cell_pattern, content, re.DOTALL)
        for match in matches:
            cell_content = match.group(1).strip()
            cell_type = "markdown" if "#%% md" in match.group(0) else "code"
            cells.append({"cell_type": cell_type, "source": cell_content})
        return cells

    def test_notebook_exists(self):
        """Test that notebook file exists."""
        self.assertTrue(self.notebook_path.exists(), "ml_finance_model_main.ipynb must exist")

    def test_notebook_has_required_sections(self):
        """Test notebook has all 10 required sections per restructuring plan."""
        required_sections = [
            "Business Objective",
            "Configuration and Setup",
            "Loading and Preprocessing",
            "Exploratory Data Analysis",
            "Feature Engineering",
            "Classification",
            "Regression",
            "Evaluation",
            "Valuation",
            "Analytics",
        ]

        content = "\n".join([str(cell.get("source", "")) for cell in self.cells])

        for section in required_sections:
            with self.subTest(section=section):
                self.assertIn(section, content, f"Section '{section}' must be present in notebook")

    def test_notebook_uses_notebook_config(self):
        """Test notebook uses NotebookConfig from finance_ml."""
        content = "\n".join([str(cell.get("source", "")) for cell in self.cells])

        self.assertIn(
            "from finance_ml import NotebookConfig", content, "Must import NotebookConfig"
        )
        self.assertIn("config = NotebookConfig(", content, "Must instantiate NotebookConfig")

    def test_notebook_imports_package_functions(self):
        """Test notebook imports from finance_ml package instead of inline definitions."""
        required_imports = [
            "finance_ml",
            "data",
            "features",
            "advanced_features",
            "classification",
            "advanced_models",
            "eval",
        ]

        content = "\n".join([str(cell.get("source", "")) for cell in self.cells])

        for module in required_imports:
            with self.subTest(module=module):
                # Check for import (either direct or from finance_ml)
                pattern = f"(from finance_ml import.*{module}|import finance_ml\\.{module})"
                self.assertTrue(
                    re.search(pattern, content) or f"finance_ml.{module}" in content,
                    f"Must import {module} from finance_ml package",
                )


class TestSection0Header(unittest.TestCase):
    """Test Section 0: Header and Business Objective."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_has_business_objective(self):
        """Test header includes clear business objective."""
        self.assertIn("Business Objective", self.content)
        self.assertIn("predict stock price target", self.content.lower())

    def test_has_workflow_overview(self):
        """Test header includes workflow overview."""
        self.assertIn("Workflow Overview", self.content)

    def test_has_version_info(self):
        """Test header includes version information."""
        # Should have version like 2.0.0 per plan
        version_pattern = r"Version\s+\d+\.\d+\.\d+"
        self.assertTrue(re.search(version_pattern, self.content), "Must include version number")


class TestSection1Configuration(unittest.TestCase):
    """Test Section 1: Configuration and Setup."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_imports_notebook_config(self):
        """Test imports NotebookConfig."""
        self.assertIn("from finance_ml import NotebookConfig", self.content)

    def test_initializes_config(self):
        """Test initializes config with proper flags."""
        self.assertIn("config = NotebookConfig(", self.content)

        # Check for key configuration flags
        config_flags = [
            "have_finance_prediction",
            "have_database_connection",
            "have_advanced_analytics",
        ]
        for flag in config_flags:
            with self.subTest(flag=flag):
                self.assertIn(flag, self.content, f"Config must include {flag} flag")

    def test_sets_random_seed(self):
        """Test sets random seed for reproducibility."""
        self.assertIn("RANDOM_SEED", self.content)
        self.assertIn("np.random.seed", self.content)

    def test_creates_output_directories(self):
        """Test creates output directory structure."""
        self.assertIn("OUTPUT_DIR", self.content)
        self.assertIn("mkdir", self.content.lower())


class TestSection2DataLoading(unittest.TestCase):
    """Test Section 2: Loading and Preprocessing."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_data_loading_functions(self):
        """Test uses package data loading functions."""
        # Should use load_from_db or load_from_csv from finance_ml.data
        self.assertTrue(
            "load_from_db" in self.content or "load_from_csv" in self.content,
            "Must use data loading functions from finance_ml.data",
        )

    def test_uses_4step_imputation(self):
        """Test uses 4-step imputation strategy from package."""
        self.assertIn(
            "apply_enhanced_imputation_strategy_4step",
            self.content,
            "Must use 4-step imputation from advanced_preprocessing",
        )

    def test_validates_schema(self):
        """Test includes schema validation."""
        self.assertIn("validate_schema", self.content, "Must validate data schema")


class TestSection3EDA(unittest.TestCase):
    """Test Section 3: Exploratory Data Analysis."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_eda_report_function(self):
        """Test uses generate_eda_report from package."""
        self.assertIn(
            "generate_eda_report", self.content, "Must use generate_eda_report from advanced_eda"
        )

    def test_uses_benchmarking(self):
        """Test includes benchmarking analysis."""
        self.assertIn("benchmarking", self.content.lower(), "Must include benchmarking analysis")


class TestSection4Features(unittest.TestCase):
    """Test Section 4: Feature Engineering."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_comprehensive_features(self):
        """Test uses build_comprehensive_features from package."""
        self.assertIn(
            "build_comprehensive_features", self.content, "Must use build_comprehensive_features"
        )

    def test_includes_feature_importance(self):
        """Test includes feature importance analysis."""
        self.assertIn(
            "feature_importance", self.content.lower(), "Must include feature importance analysis"
        )


class TestSection5Classification(unittest.TestCase):
    """Test Section 5: Multi-Class Classification."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_creates_event_labels(self):
        """Test creates event labels using package function."""
        self.assertIn(
            "create_enhanced_event_labels", self.content, "Must use create_enhanced_event_labels"
        )

    def test_compares_classifiers(self):
        """Test compares multiple classifiers."""
        self.assertIn("compare_classifiers", self.content, "Must use compare_classifiers")


class TestSection6Regression(unittest.TestCase):
    """Test Section 6: Regression Models."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_sector_specific_models(self):
        """Test uses sector-specific model training."""
        self.assertIn(
            "train_sector_specific_models", self.content, "Must use train_sector_specific_models"
        )

    def test_uses_ensemble_models(self):
        """Test includes ensemble/stacking regression."""
        self.assertTrue(
            "stacking" in self.content.lower() or "ensemble" in self.content.lower(),
            "Must include ensemble/stacking regression",
        )


class TestSection7Evaluation(unittest.TestCase):
    """Test Section 7: Model Evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_comprehensive_metrics(self):
        """Test uses comprehensive regression metrics."""
        self.assertIn(
            "comprehensive_regression_metrics",
            self.content,
            "Must use comprehensive_regression_metrics",
        )

    def test_includes_segment_analysis(self):
        """Test includes segment analysis by sector/region."""
        self.assertIn("compute_metrics_by_segment", self.content, "Must include segment analysis")


class TestSection8Valuation(unittest.TestCase):
    """Test Section 8: Stock Valuation."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_calculates_mispricing_score(self):
        """Test calculates mispricing scores."""
        self.assertIn(
            "calculate_mispricing_score", self.content, "Must calculate mispricing scores"
        )

    def test_ranks_stocks(self):
        """Test ranks undervalued/overvalued stocks."""
        self.assertTrue(
            "rank_undervalued_stocks" in self.content or "rank_overvalued_stocks" in self.content,
            "Must rank stocks by valuation",
        )


class TestSection9Analytics(unittest.TestCase):
    """Test Section 9: Comprehensive Analytics."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_analyst_comparison(self):
        """Test uses analyst comparison analytics."""
        self.assertIn("analyst_comparison", self.content.lower(), "Must include analyst comparison")

    def test_generates_reports(self):
        """Test generates Excel/PDF reports."""
        self.assertTrue(
            "excel" in self.content.lower() or "pdf" in self.content.lower(),
            "Must generate analytical reports",
        )


class TestSection10Portfolio(unittest.TestCase):
    """Test Section 10: Portfolio Optimization."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_uses_portfolio_optimization(self):
        """Test uses portfolio optimization functions."""
        self.assertIn(
            "portfolio_optimization", self.content.lower(), "Must use portfolio_optimization module"
        )

    def test_calculates_risk_metrics(self):
        """Test calculates portfolio risk metrics."""
        self.assertIn("risk_metrics", self.content.lower(), "Must calculate risk metrics")


class TestCodeQuality(unittest.TestCase):
    """Test overall code quality and production readiness."""

    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if cls.notebook_path.exists():
            with open(cls.notebook_path, "r", encoding="utf-8") as f:
                cls.content = f.read()
        else:
            cls.content = ""

    def test_no_excessive_inline_helpers(self):
        """Test notebook doesn't have excessive inline helper functions."""
        # Count function definitions - should be minimal (only utility functions)
        def_count = len(re.findall(r"\ndef\s+\w+", self.content))
        self.assertLess(
            def_count,
            10,
            f"Too many inline functions ({def_count}). Use package functions instead.",
        )

    def test_reasonable_length(self):
        """Test notebook is reasonably sized (not 5755 lines)."""
        line_count = len(self.content.split("\n"))
        self.assertLess(
            line_count, 2000, f"Notebook too long ({line_count} lines). Target is ~1170 lines."
        )

    def test_uses_pathlib(self):
        """Test uses pathlib.Path for path handling."""
        self.assertIn("from pathlib import Path", self.content, "Must use pathlib.Path for paths")

    def test_has_error_handling(self):
        """Test includes proper error handling."""
        self.assertIn("try:", self.content, "Must include error handling with try/except")


if __name__ == "__main__":
    unittest.main()
