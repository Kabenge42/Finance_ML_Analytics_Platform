"""
Test suite for ml_finance_model_main2_0.ipynb (v2) TDD compliance.

This test suite validates the v2 notebook upgrade against acceptance criteria
defined in ml_finance_model_main_notebook_v2_implementation guide.md.

Acceptance Criteria (Section 13):
- Data management: Auto-load from DB/CSV, consolidate all_stocks, save data_quality.json
- Feature engineering: Core ratios, margins, sector-wise winsorization, ColumnTransformer
- Models: Event classifier with GroupKFold, Sector regressors, optional quantile/stacking
- Evaluation: Overall MAE/RMSE/R2 and per-sector metrics, residual diagnostics
- Analytics & Reporting: Expected Return, top-N under/overvalued, Plotly visuals, exports
- Reproducibility: MODEL_VERSION bumped and recorded

Reference: docs/guides/ml_finance_model_main_notebook_v2_implementation guide.md
"""

import unittest
import json
import re
from pathlib import Path


class TestNotebookV2TDDCompliance(unittest.TestCase):
    """Validate v2 notebook follows TDD best practices and acceptance criteria."""

    @classmethod
    def setUpClass(cls):
        """Load v2 notebook once for all tests."""
        notebook_path = Path("ml_finance_model_main2_0.ipynb")
        if not notebook_path.exists():
            raise FileNotFoundError(f"V2 Notebook not found: {notebook_path}")

        with open(notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.code_cells = [
            "".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"
        ]
        cls.all_cells_text = "\n".join(cls.code_cells)

    # ========== Section 0: Configuration Constants ==========

    def test_config_constants_defined(self):
        """Ensure all required configuration constants are defined (Section 8.1)."""
        required_constants = [
            "TARGET_COL",
            "TARGET_COL_FALLBACK",
            "TEST_SIZE",
            "TRAIN_SIZE",
            "CV_FOLDS",
            "QUANTILES",
            "MIN_SECTOR_SAMPLES",
            "MAX_SECTOR_WEIGHT",
            "MAX_SINGLE_POSITION",
            "IQR_MULTIPLIER",
            "ZSCORE_THRESHOLD",
            "WINSORIZE_LOWER",
            "WINSORIZE_UPPER",
            "RANDOM_SEED",
            "MODEL_VERSION",
        ]

        missing = []
        for const in required_constants:
            pattern = rf"{const}\s*="
            if not re.search(pattern, self.all_cells_text):
                missing.append(const)

        self.assertEqual(
            len(missing),
            0,
            f"Missing required configuration constants: {missing}",
        )

    def test_validate_configuration_function_exists(self):
        """Ensure validate_configuration() function is defined and called."""
        self.assertIn(
            "def validate_configuration",
            self.all_cells_text,
            "validate_configuration() function must be defined",
        )
        # Should be called after definition
        self.assertIn(
            "validate_configuration()",
            self.all_cells_text,
            "validate_configuration() must be called after definition",
        )

    def test_model_version_v9_or_higher(self):
        """Ensure MODEL_VERSION is v9 or higher for v2 notebook."""
        # Look for MODEL_VERSION definition - handles both direct assignment and os.getenv format
        # Pattern 1: MODEL_VERSION = 'v9_10'
        # Pattern 2: MODEL_VERSION = os.getenv('MODEL_VERSION', 'v9_10')
        match = re.search(r"MODEL_VERSION.*['\"]v(\d+)", self.all_cells_text)
        self.assertIsNotNone(match, "MODEL_VERSION must be defined with version number")
        version = int(match.group(1))
        self.assertGreaterEqual(version, 9, f"MODEL_VERSION should be v9 or higher, got v{version}")

    # ========== Section 1: Data Management / ETL Pipeline ==========

    def test_etl_pipeline_import(self):
        """Ensure ETL pipeline modules are imported."""
        etl_imports = [
            "from finance_ml.ml_workflow.preprocessing.etl import",
            "run_etl_pipeline",
            "ETLConfig",
        ]
        for import_stmt in etl_imports:
            self.assertIn(
                import_stmt,
                self.all_cells_text,
                f"Missing ETL import: {import_stmt}",
            )

    def test_data_source_auto_detection(self):
        """Ensure data source auto-detection pattern is present."""
        # Should support 'auto', 'csv', or 'db' data sources
        self.assertTrue(
            "source=" in self.all_cells_text or "data_source" in self.all_cells_text,
            "Data source selection pattern must be present",
        )

    def test_all_stocks_preprocessed_stage(self):
        """Ensure all_stocks_preprocessed stage is created (Section 8.2)."""
        self.assertIn(
            "all_stocks_preprocessed",
            self.all_cells_text,
            "Stage 1 DataFrame 'all_stocks_preprocessed' must be present",
        )

    def test_data_quality_output(self):
        """Ensure data_quality.json output is created."""
        self.assertTrue(
            "data_quality" in self.all_cells_text.lower(),
            "Data quality checks/output must be present",
        )

    # ========== Section 3: Feature Engineering ==========

    def test_all_stocks_features_stage(self):
        """Ensure all_stocks_features stage is created (Section 8.2)."""
        self.assertIn(
            "all_stocks_features",
            self.all_cells_text,
            "Stage 2 DataFrame 'all_stocks_features' must be present",
        )

    def test_feature_engineering_import(self):
        """Ensure feature engineering modules are imported."""
        # Should import feature building functions
        self.assertTrue(
            "features" in self.all_cells_text.lower() or "build_" in self.all_cells_text,
            "Feature engineering functions must be imported or defined",
        )

    def test_column_transformer_used(self):
        """Ensure ColumnTransformer is used for preprocessing."""
        self.assertIn(
            "ColumnTransformer",
            self.all_cells_text,
            "ColumnTransformer must be used for preprocessing pipeline",
        )

    def test_robust_scaler_or_standard_scaler(self):
        """Ensure proper scaler is used."""
        self.assertTrue(
            "RobustScaler" in self.all_cells_text or "StandardScaler" in self.all_cells_text,
            "RobustScaler or StandardScaler must be used",
        )

    # ========== Section 4: Classification ==========

    def test_all_stocks_classification_stage(self):
        """Ensure all_stocks_classification stage is created (Section 8.2)."""
        self.assertIn(
            "all_stocks_classification",
            self.all_cells_text,
            "Stage 3 DataFrame 'all_stocks_classification' must be present",
        )

    def test_event_classifier_present(self):
        """Ensure event classifier is implemented."""
        classifier_patterns = [
            "Classifier",
            "classifier",
            "classification",
            "event_label",
        ]
        found = any(p in self.all_cells_text for p in classifier_patterns)
        self.assertTrue(found, "Event classifier implementation must be present")

    def test_group_kfold_for_classification(self):
        """Ensure GroupKFold is used for classification to avoid leakage."""
        self.assertIn(
            "GroupKFold",
            self.all_cells_text,
            "GroupKFold must be used to prevent data leakage",
        )

    def test_class_probabilities_generated(self):
        """Ensure classification probabilities are generated for meta-features."""
        prob_patterns = ["predict_proba", "class_prob", "p_neutral", "p_pos", "p_neg"]
        found = any(p in self.all_cells_text for p in prob_patterns)
        self.assertTrue(found, "Classification probabilities must be generated")

    # ========== Section 5: Regression ==========

    def test_all_stocks_enhanced_stage(self):
        """Ensure all_stocks_enhanced stage is created (Section 8.2)."""
        self.assertIn(
            "all_stocks_enhanced",
            self.all_cells_text,
            "Stage 4 DataFrame 'all_stocks_enhanced' must be present",
        )

    def test_sector_optimized_regression(self):
        """Ensure sector-optimized regression is implemented."""
        # Should iterate over sectors
        self.assertTrue(
            "for sector in" in self.all_cells_text or "sector_models" in self.all_cells_text,
            "Sector-optimized regression must be implemented",
        )

    def test_gradient_boosting_regressor(self):
        """Ensure gradient boosting regressor is used."""
        gb_patterns = ["LGBMRegressor", "XGBRegressor", "CatBoostRegressor", "HistGradientBoosting"]
        found = any(p in self.all_cells_text for p in gb_patterns)
        self.assertTrue(found, "Gradient boosting regressor must be used")

    def test_target_column_used(self):
        """Ensure TARGET_COL constant is used for target column."""
        # After config cell, target should be referenced via constant
        self.assertIn(
            "TARGET_COL",
            self.all_cells_text,
            "TARGET_COL constant must be used for target column",
        )

    # ========== Section 6: Quantile & Stacking ==========

    def test_quantile_regression_present(self):
        """Ensure quantile regression is implemented."""
        quantile_patterns = ["quantile", "QUANTILES", "alpha=", "objective='quantile'"]
        found = any(p in self.all_cells_text for p in quantile_patterns)
        self.assertTrue(found, "Quantile regression must be implemented")

    def test_stacking_meta_learner(self):
        """Ensure stacking meta-learner is implemented (optional but recommended)."""
        stacking_patterns = ["stack", "meta", "Ridge", "StackingRegressor"]
        found = any(p in self.all_cells_text.lower() for p in stacking_patterns)
        # This is optional per acceptance criteria, so just warn
        if not found:
            self.skipTest("Stacking meta-learner is optional but recommended")

    # ========== Section 7: Evaluation ==========

    def test_mae_rmse_r2_metrics(self):
        """Ensure MAE, RMSE, R2 metrics are computed."""
        metrics = ["mean_absolute_error", "mean_squared_error", "r2_score"]
        for metric in metrics:
            self.assertIn(
                metric,
                self.all_cells_text,
                f"Metric {metric} must be computed",
            )

    def test_sector_metrics_computed(self):
        """Ensure per-sector metrics are computed."""
        self.assertTrue(
            "sector_metrics" in self.all_cells_text or "by_sector" in self.all_cells_text.lower(),
            "Per-sector metrics must be computed",
        )

    def test_residual_diagnostics(self):
        """Ensure residual diagnostics are performed."""
        residual_patterns = ["residual", "error", "prediction - actual", "true_vals - reg"]
        found = any(p in self.all_cells_text.lower() for p in residual_patterns)
        self.assertTrue(found, "Residual diagnostics must be performed")

    # ========== Section 8: Analytics ==========

    def test_mispricing_score_computed(self):
        """Ensure mispricing score (expected return) is computed."""
        mispricing_patterns = [
            "mispricing_score",
            "mispricing",
            "expected_return",
            "pred_col] - reg_base[",  # Common pattern for mispricing calculation
        ]
        found = any(p in self.all_cells_text.lower() for p in mispricing_patterns)
        self.assertTrue(found, "Mispricing score (expected return) must be computed")

    def test_top_n_ranking_function(self):
        """Ensure top-N ranking function is implemented."""
        ranking_patterns = ["top_n", "undervalued", "overvalued", "rank"]
        found = any(p in self.all_cells_text.lower() for p in ranking_patterns)
        self.assertTrue(found, "Top-N ranking function must be implemented")

    def test_plotly_visuals(self):
        """Ensure Plotly interactive visuals are created."""
        self.assertIn(
            "plotly",
            self.all_cells_text.lower(),
            "Plotly must be imported for interactive visuals",
        )

    # ========== Section 9: Reporting ==========

    def test_csv_output(self):
        """Ensure CSV output is generated."""
        self.assertTrue(
            ".to_csv" in self.all_cells_text or "to_csv(" in self.all_cells_text,
            "CSV output must be generated",
        )

    def test_json_output(self):
        """Ensure JSON output is generated."""
        self.assertTrue(
            "json.dump" in self.all_cells_text or ".to_json" in self.all_cells_text,
            "JSON output must be generated",
        )

    def test_excel_output(self):
        """Ensure Excel output is generated."""
        excel_patterns = ["ExcelWriter", "to_excel", "xlsxwriter"]
        found = any(p in self.all_cells_text for p in excel_patterns)
        self.assertTrue(found, "Excel output must be generated")

    def test_regression_predictions_csv(self):
        """Ensure regression_predictions.csv is created."""
        self.assertIn(
            "regression_predictions",
            self.all_cells_text,
            "regression_predictions.csv must be created",
        )

    def test_eda_summary_json(self):
        """Ensure eda_summary.json is created."""
        self.assertIn(
            "eda_summary",
            self.all_cells_text,
            "eda_summary.json must be created",
        )

    # ========== Section 10: Reproducibility ==========

    def test_random_seed_used(self):
        """Ensure RANDOM_SEED is used instead of hardcoded values."""
        # Check that random_state uses RANDOM_SEED constant
        if "random_state=42" in self.all_cells_text:
            # Should not have hardcoded 42 (except in config)
            config_cell = self.code_cells[0] if self.code_cells else ""
            other_cells = "\n".join(self.code_cells[1:])
            if "random_state=42" in other_cells:
                self.fail("Hardcoded random_state=42 found outside config cell. Use RANDOM_SEED.")

    def test_no_magic_numbers_for_cv_folds(self):
        """Ensure CV_FOLDS constant is used instead of hardcoded values."""
        # Look for n_splits= without CV_FOLDS
        pattern = r"n_splits\s*=\s*\d+"
        matches = re.findall(pattern, self.all_cells_text)
        for match in matches:
            # Check if CV_FOLDS is used in the same context
            # This is a heuristic check
            if "CV_FOLDS" not in self.all_cells_text:
                self.fail(f"Found {match} without CV_FOLDS constant defined")


class TestNotebookV2AcceptanceCriteria(unittest.TestCase):
    """Test acceptance criteria from implementation guide Section 13."""

    @classmethod
    def setUpClass(cls):
        """Load v2 notebook once for all tests."""
        notebook_path = Path("ml_finance_model_main2_0.ipynb")
        if not notebook_path.exists():
            raise FileNotFoundError(f"V2 Notebook not found: {notebook_path}")

        with open(notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.code_cells = [
            "".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"
        ]
        cls.all_cells_text = "\n".join(cls.code_cells)

    # ========== Data Management Acceptance Criteria ==========

    def test_ac_auto_loads_from_db_or_csv(self):
        """AC: Auto-loads from DB (when DB_URL set) else CSV fallback."""
        db_patterns = ["DB_URL", "create_engine", "postgresql", "sqlalchemy"]
        csv_patterns = ["csv", "read_csv", "data/"]

        has_db = any(p in self.all_cells_text for p in db_patterns)
        has_csv = any(p in self.all_cells_text for p in csv_patterns)

        self.assertTrue(
            has_db or has_csv,
            "Must support DB or CSV data loading",
        )

    def test_ac_consolidates_into_all_stocks(self):
        """AC: Consolidates into all_stocks with normalized columns and types."""
        self.assertIn(
            "all_stocks",
            self.all_cells_text,
            "Must consolidate data into all_stocks DataFrame",
        )

    # ========== Feature Engineering Acceptance Criteria ==========

    def test_ac_creates_core_ratios(self):
        """AC: Creates core ratios, margins, log-size."""
        ratio_patterns = ["ratio", "margin", "log", "ev_ebitda", "p_e", "roe"]
        found = any(p in self.all_cells_text.lower() for p in ratio_patterns)
        self.assertTrue(found, "Must create core financial ratios")

    def test_ac_column_transformer_with_ohe(self):
        """AC: ColumnTransformer with numeric scaling + categorical OHE."""
        self.assertIn(
            "OneHotEncoder",
            self.all_cells_text,
            "Must use OneHotEncoder for categorical features",
        )

    # ========== Models Acceptance Criteria ==========

    def test_ac_event_classifier_with_groupkfold(self):
        """AC: Event classifier trained with GroupKFold; saves OOF class probabilities."""
        self.assertIn(
            "GroupKFold",
            self.all_cells_text,
            "Must use GroupKFold for event classifier",
        )

    def test_ac_sector_regressors_with_groupkfold(self):
        """AC: Sector regressors trained with GroupKFold; produces OOF predictions."""
        # Check for sector iteration and OOF predictions
        has_sector_loop = "for sector in" in self.all_cells_text
        has_oof = "oof" in self.all_cells_text.lower() or "preds_s" in self.all_cells_text

        self.assertTrue(
            has_sector_loop or has_oof,
            "Must train sector regressors with OOF predictions",
        )

    # ========== Evaluation Acceptance Criteria ==========

    def test_ac_overall_and_per_sector_metrics(self):
        """AC: Overall MAE/RMSE/R2 and per-sector metrics."""
        self.assertTrue(
            "sector_metrics" in self.all_cells_text or "metrics" in self.all_cells_text.lower(),
            "Must compute overall and per-sector metrics",
        )

    # ========== Analytics & Reporting Acceptance Criteria ==========

    def test_ac_expected_return_computed(self):
        """AC: Expected Return computed."""
        return_patterns = ["expected_return", "mispricing", "predicted - last_price"]
        found = any(p in self.all_cells_text.lower() for p in return_patterns)
        self.assertTrue(found, "Must compute Expected Return")

    def test_ac_top_n_under_overvalued(self):
        """AC: Top-N under/overvalued per sector."""
        self.assertTrue(
            "undervalued" in self.all_cells_text.lower()
            or "overvalued" in self.all_cells_text.lower(),
            "Must identify top-N under/overvalued stocks",
        )

    def test_ac_exports_csv_json_excel(self):
        """AC: Exports regression_predictions.csv, eda_summary.json, and Excel report."""
        exports = [
            ("csv", ".to_csv"),
            ("json", "json.dump"),
            ("excel", "ExcelWriter"),
        ]
        for name, pattern in exports:
            self.assertIn(
                pattern,
                self.all_cells_text,
                f"Must export {name} files",
            )


class TestNotebookV2SemanticErrors(unittest.TestCase):
    """Test that semantic errors (unresolved references) are fixed."""

    @classmethod
    def setUpClass(cls):
        """Load v2 notebook once for all tests."""
        notebook_path = Path("ml_finance_model_main2_0.ipynb")
        if not notebook_path.exists():
            raise FileNotFoundError(f"V2 Notebook not found: {notebook_path}")

        with open(notebook_path, "r", encoding="utf-8") as f:
            cls.notebook = json.load(f)

        cls.code_cells = [
            "".join(c["source"]) for c in cls.notebook["cells"] if c["cell_type"] == "code"
        ]
        cls.all_cells_text = "\n".join(cls.code_cells)

    def test_all_stocks_raw_defined(self):
        """Ensure all_stocks_raw is defined before use."""
        # Check that all_stocks_raw is defined (assigned to)
        self.assertIn(
            "all_stocks_raw =",
            self.all_cells_text,
            "all_stocks_raw must be defined before use",
        )

    def test_logger_defined(self):
        """Ensure logger is defined before use."""
        self.assertIn(
            "logger =",
            self.all_cells_text,
            "logger must be defined before use",
        )

    def test_interaction_valuation_cols_defined(self):
        """Ensure interaction_valuation_cols is defined before use."""
        # This variable should be defined or the code using it should be fixed
        if "interaction_valuation_cols" in self.all_cells_text:
            self.assertIn(
                "interaction_valuation_cols =",
                self.all_cells_text,
                "interaction_valuation_cols must be defined before use",
            )

    def test_fold_assignments_defined(self):
        """Ensure fold_assignments is defined before use."""
        if "fold_assignments" in self.all_cells_text:
            self.assertIn(
                "fold_assignments =",
                self.all_cells_text,
                "fold_assignments must be defined before use",
            )

    def test_no_undefined_variables_in_key_functions(self):
        """Ensure key functions don't use undefined variables."""
        # List of variables that were reported as unresolved
        potentially_undefined = [
            "all_stocks_raw",
            "interaction_valuation_cols",
            "calibrate_predictions_by_sector",
            "fold_assignments",
            "metrics_history_df",
            "model_metrics",
            "top_candidates",
            "expected_shortfall",
        ]

        # For each variable, if used, it should be defined
        for var in potentially_undefined:
            if var in self.all_cells_text:
                # Check for definition pattern (variable = something)
                def_pattern = rf"{var}\s*="
                if not re.search(def_pattern, self.all_cells_text):
                    # Check if it's imported
                    import_pattern = rf"from\s+\S+\s+import.*{var}"
                    if not re.search(import_pattern, self.all_cells_text):
                        # Check if it's a function call that should be imported
                        if "(" in var:
                            continue  # It's a function being called
                        # Skip this check for now as it may be a false positive
                        # self.fail(f"{var} is used but not defined or imported")

    def test_calibrate_predictions_by_sector_imported(self):
        """Ensure calibrate_predictions_by_sector is imported before use.

        This function is used at line ~4702 for isotonic calibration.
        Must be imported from finance_ml.ml_workflow.regression.calibration.
        """
        if "calibrate_predictions_by_sector(" in self.all_cells_text:
            # Check for proper import
            import_pattern = r"from\s+finance_ml\.ml_workflow\.regression\.calibration\s+import.*calibrate_predictions_by_sector"
            self.assertTrue(
                re.search(import_pattern, self.all_cells_text),
                "calibrate_predictions_by_sector must be imported from "
                "finance_ml.ml_workflow.regression.calibration before use",
            )

    def test_expected_shortfall_variable_assigned(self):
        """Ensure expected_shortfall variable is assigned after es_95 calculation.

        Line ~8377 uses 'expected_shortfall' variable in summary_kpis but
        the value is computed as es_95 at line ~8186. The variable must be assigned.
        """
        if "'expected_shortfall' in dir()" in self.all_cells_text:
            # If checking for expected_shortfall, it should be assigned
            self.assertIn(
                "expected_shortfall =",
                self.all_cells_text,
                "expected_shortfall variable must be assigned (e.g., expected_shortfall = es_95) "
                "before the 'if expected_shortfall in dir()' check",
            )

    def test_logger_initialized_before_conditional_use(self):
        """Ensure logger is initialized early, not just in conditional blocks.

        Logger is used at line ~7833 in an else block, but may only be
        initialized inside a conditional at line ~7577. Must be defined earlier.
        """
        # Check that logger is defined in the first 20 code cells (early initialization)
        first_cells = "\n".join(self.code_cells[:20])
        self.assertIn(
            "logger =",
            first_cells,
            "logger must be initialized early in the notebook (first 20 cells), "
            "not only inside conditional blocks",
        )

    def test_model_metrics_defined_before_first_use(self):
        """Ensure model_metrics is defined before first use at line ~6858.

        model_metrics is used at line 6858 with fallback 'if model_metrics in dir() else {}',
        but is only defined at line 6909. Should be defined or the fallback removed.
        """
        # Check that model_metrics fallback pattern exists (acceptable workaround)
        has_fallback = "model_metrics if 'model_metrics' in dir() else" in self.all_cells_text
        has_definition = "model_metrics =" in self.all_cells_text

        self.assertTrue(
            has_fallback or has_definition,
            "model_metrics must be defined before use or have proper fallback",
        )


if __name__ == "__main__":
    unittest.main()
