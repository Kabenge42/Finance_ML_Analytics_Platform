import io
import unittest
from contextlib import redirect_stdout


class TestNotebookConfig(unittest.TestCase):
    def test_defaults_and_is_feature_enabled(self):
        # Import inside test to follow TDD: it should fail before implementation exists
        from finance_ml import NotebookConfig

        cfg = NotebookConfig()
        # Default flags
        self.assertTrue(cfg.have_finance_prediction)
        self.assertFalse(cfg.have_database_connection)
        self.assertTrue(cfg.have_advanced_analytics)
        self.assertFalse(cfg.have_dim_reduction)
        self.assertFalse(cfg.debug_mode)
        self.assertTrue(cfg.enable_sector_analysis)
        self.assertTrue(cfg.enable_region_analysis)
        self.assertTrue(cfg.enable_interactive_plots)
        self.assertTrue(cfg.enable_excel_export)

        # is_feature_enabled should reflect attribute values
        self.assertTrue(cfg.is_feature_enabled("have_finance_prediction"))
        self.assertFalse(cfg.is_feature_enabled("have_database_connection"))
        # Unknown feature names should return False, not raise
        self.assertFalse(cfg.is_feature_enabled("non_existent_feature"))

    def test_overrides(self):
        from finance_ml import NotebookConfig

        cfg = NotebookConfig(
            have_finance_prediction=False,
            have_database_connection=True,
            enable_interactive_plots=False,
            debug_mode=True,
        )
        self.assertFalse(cfg.have_finance_prediction)
        self.assertTrue(cfg.have_database_connection)
        self.assertFalse(cfg.enable_interactive_plots)
        self.assertTrue(cfg.debug_mode)

        self.assertFalse(cfg.is_feature_enabled("have_finance_prediction"))
        self.assertTrue(cfg.is_feature_enabled("have_database_connection"))

    def test_display_summary_outputs_expected_lines(self):
        from finance_ml import NotebookConfig

        cfg = NotebookConfig()
        buf = io.StringIO()
        with redirect_stdout(buf):
            cfg.display_summary()
        out = buf.getvalue()

        # Check for header and a few key lines; avoid brittle exact spacing
        self.assertIn("FEATURE FLAGS CONFIGURATION", out)
        self.assertIn("Financial Prediction:", out)
        self.assertIn("Database Connection:", out)
        self.assertIn("Advanced Analytics:", out)
        self.assertIn("Dimensionality Reduction:", out)
        self.assertIn("Sector Analysis:", out)
        self.assertIn("Region Analysis:", out)
        self.assertIn("Interactive Plots:", out)
        self.assertIn("Excel Export:", out)


if __name__ == "__main__":
    unittest.main()
