import json
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NB_PATH = PROJECT_ROOT / "ml_finance_model_v8_3.ipynb"
ENH_MD_PATH = PROJECT_ROOT / "NOTEBOOK_ENHANCEMENTS.md"


def _read_notebook_text(path: Path) -> str:
    """Return all textual content from the notebook for simple assertions.
    Attempts JSON parsing first (true .ipynb). If that fails, falls back to raw text.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Try to parse as JSON .ipynb
    try:
        nb = json.loads(text)
        parts = []
        for cell in nb.get("cells", []):
            src = cell.get("source", [])
            if isinstance(src, list):
                parts.append("".join(src))
            elif isinstance(src, str):
                parts.append(src)
        return "\n".join(parts)
    except Exception:
        # Not JSON or parsing failed; return raw text (e.g., percent-format fallback)
        return text


class TestNotebookEnhancements(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert NB_PATH.exists(), f"Notebook not found at {NB_PATH}"
        cls.nb_text = _read_notebook_text(NB_PATH)
        cls.enh_md = ENH_MD_PATH.read_text(encoding="utf-8", errors="ignore")

    def test_enhancement_plan_mentions_v8_3(self):
        # Sanity check that the enhancement doc targets v8_3
        self.assertIn("v8_3", self.enh_md)

    def test_model_version_marker_present_in_notebook(self):
        # Acceptance: The notebook must clearly mark the new model version v8_3
        self.assertIn(
            "v8_3", self.nb_text, msg="Notebook should include a visible 'v8_3' version marker"
        )

    def test_uses_finance_ml_package_functions(self):
        # Core functions expected from the enhancement plan and README
        expected_symbols = [
            "load_stock_data",
            "validate_and_display_data",
            "perform_and_display_eda",
            "preprocess",
            "build_features_and_target",
            "create_event_labels",
            "train_event_classifier",
            "train_and_evaluate_regression",
            "calculate_mispricing_score",
            "rank_undervalued_stocks",
            "export_predictions_to_excel",
        ]
        # Basic import presence
        self.assertIn("from finance_ml", self.nb_text)
        # Check individual symbol refs appear somewhere in the notebook text
        missing = [sym for sym in expected_symbols if sym not in self.nb_text]
        self.assertFalse(
            missing, msg=f"Expected finance_ml symbols not found in notebook: {missing}"
        )

    def test_advanced_preprocessing_pipeline_present(self):
        # ColumnTransformer with separate Numeric and Categorical transformers
        for token in [
            "ColumnTransformer",
            "StandardScaler",
            "OneHotEncoder",
            "numeric_features",
            "categorical_features",
        ]:
            self.assertIn(token, self.nb_text, msg=f"Expected token not found in notebook: {token}")

    def test_enhanced_modeling_sections_present(self):
        # Presence of per-sector, quantile, stacking, and excel export sections
        expected_phrases = [
            "PER-SECTOR REGRESSION METRICS",
            "QUANTILE REGRESSION BY SECTOR",
            "STACKING ENSEMBLE BY SECTOR",
            "export_predictions_to_excel",
        ]
        for phrase in expected_phrases:
            self.assertIn(phrase, self.nb_text, msg=f"Expected section/usage missing: {phrase}")

    def test_week1_validation_suite_present(self):
        # Functions referenced in the Week 1 enhancement validation suite
        expected_funcs = [
            "validate_financial_data_quality",
            "sanitize_dataframe_with_logging",
            "monitor_ensemble_training",
            "perform_early_pipeline_validation",
        ]
        for func in expected_funcs:
            self.assertIn(func, self.nb_text, msg=f"Expected validation helper not found: {func}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
