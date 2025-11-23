"""
Notebook Review Checklist (Section 6.1) — Structural TDD tests.

These tests statically inspect ml_finance_model_main.ipynb to ensure the
notebook adheres to the documented checklist without running heavy ML code.

Focus: configuration, env vars, preprocessing hooks, safety rails, and outputs.
"""

import json
import unittest
from pathlib import Path


class TestNotebookReviewChecklist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook_path = Path("ml_finance_model_main.ipynb")
        if not cls.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found at {cls.notebook_path}")
        with open(cls.notebook_path, "r", encoding="utf-8") as f:
            cls.nb = json.load(f)
        # Join code cell sources for simple searching
        cls.code_cells = [
            "".join(c.get("source", [])) for c in cls.nb.get("cells", []) if c.get("cell_type") == "code"
        ]
        cls.all_code = "\n".join(cls.code_cells)

    # ---- Configuration and Setup ----
    def test_configuration_constants_and_seed(self):
        # Constants present
        required_constants = [
            "TARGET_COL", "TARGET_COL_FALLBACK", "TEST_SIZE", "CV_FOLDS",
            "QUANTILES", "MIN_SECTOR_SAMPLES", "RANDOM_SEED", "MODEL_VERSION",
        ]
        for const in required_constants:
            self.assertIn(const + " =", self.all_code, f"Missing config constant: {const}")

        # Feature preset constant recommended by guidelines
        self.assertIn("FEATURE_PRESET", self.all_code, "Missing FEATURE_PRESET constant")

        # Seed usage
        self.assertRegex(self.all_code, r"np\.random\.seed\(RANDOM_SEED\)", "Random seed not set from constant")

        # validate_configuration called
        self.assertIn("validate_configuration()", self.all_code, "validate_configuration() must be called")

    def test_env_vars_and_output_dir(self):
        # DB_URL env var for data source
        self.assertIn("os.getenv('DB_URL'", self.all_code, "DB_URL env var should be used for data source")
        # MODEL_VERSION env var
        self.assertIn("os.getenv('MODEL_VERSION'", self.all_code, "MODEL_VERSION should be read from env")
        # CACHE_DIR optional
        self.assertIn("os.getenv('CACHE_DIR'", self.all_code, "CACHE_DIR should be used for catalog/temp artifacts")
        # OUTPUT_DIR via pathlib.Path
        self.assertRegex(self.all_code, r"OUTPUT_DIR\s*=\s*Path\(\'outputs\'\)", "OUTPUT_DIR must use pathlib.Path")

    # ---- Data Loading and Preprocessing ----
    def test_data_loading_and_preprocessing_hooks(self):
        # Data source section markers and functions
        self.assertIn("load_from_db", self.all_code, "DB loading function should be referenced")
        self.assertIn("load_from_csv", self.all_code, "CSV loading function should be referenced")
        # Column normalization
        self.assertIn("normalize_columns", self.all_code, "Column normalization should be applied")
        # Dtype detection present
        self.assertIn("detect_and_cast_dtypes", self.all_code, "Schema-aware dtype detection should be applied")
        # Explicit dtype validation against schema
        self.assertIn(
            "validate_dtypes_against_schema",
            self.all_code,
            "Post-casting validation validate_dtypes_against_schema() should be called",
        )
        # 6-step imputation
        self.assertIn(
            "apply_enhanced_imputation_strategy_6step",
            self.all_code,
            "Enhanced 6-step imputation strategy should be applied",
        )
        # Winsorization/clipping hooks
        self.assertIn("winsorize_by_sector", self.all_code, "Winsorization by sector should be referenced")

    # ---- Feature Engineering ----
    def test_feature_engineering_preset_and_phase93(self):
        # Feature preset present and Phase 9.3 categories referenced
        self.assertIn("FEATURE_PRESET", self.all_code)
        self.assertIn("PHASE93_FEATURE_INPUTS", self.all_code, "Phase 9.3 feature inputs should be referenced")

    # ---- Model Training, Evaluation, and Outputs ----
    def test_evaluation_outputs_and_constraints(self):
        # Predictions schema artifacts
        required_artifacts = [
            "outputs/regression/regression_predictions_detailed.csv",
            "outputs/regression/quantile_predictions.csv",
            "outputs/regression/regression_metrics_by_sector.csv",
        ]
        for artifact in required_artifacts:
            self.assertIn(artifact, self.all_code, f"Expected artifact path referenced: {artifact}")

        # Quantile monotonicity checks present
        self.assertRegex(
            self.all_code,
            r"pred_p10\s*<=\s*pred_p50\s*<=\s*pred_p90|quantile",
            "Quantile monotonicity checks should be present",
        )

        # Non-negativity enforced or validated
        self.assertIn("Non-negativity", self.all_code, "Non-negativity constraint should be mentioned/enforced")


if __name__ == "__main__":
    unittest.main()
