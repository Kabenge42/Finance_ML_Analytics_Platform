"""
Tests for expected_returns_v3.py refactorings (Fixes #1–#7).

Covers:
  - Fix #1: _log_and_print helper (logging + print)
  - Fix #2: safe_divide usage in implied return calculations
  - Fix #3: df.copy() on fallback assignment
  - Fix #4: _has_required_columns guard
  - Fix #5: PipelineConfig dataclass + from_env
  - Fix #6: Result variable initialization (empty DataFrames)
  - Fix #7: configure_logging replaces logging.basicConfig
"""

from __future__ import annotations

import logging
import os
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd


class TestLogAndPrint(unittest.TestCase):
    """Fix #1/#7: _log_and_print helper logs and prints."""

    def test_log_and_print_calls_both(self):
        from expected_returns_v3 import _log_and_print

        with patch("expected_returns_v3.logger") as mock_logger, \
             patch("builtins.print") as mock_print:
            _log_and_print("hello world")
            mock_logger.log.assert_called_once_with(logging.INFO, "hello world")
            mock_print.assert_called_once_with("hello world")

    def test_log_and_print_warning_level(self):
        from expected_returns_v3 import _log_and_print

        with patch("expected_returns_v3.logger") as mock_logger, \
             patch("builtins.print") as mock_print:
            _log_and_print("warn msg", logging.WARNING)
            mock_logger.log.assert_called_once_with(logging.WARNING, "warn msg")
            mock_print.assert_called_once_with("warn msg")


class TestHasRequiredColumns(unittest.TestCase):
    """Fix #4: _has_required_columns guard."""

    def test_all_columns_present(self):
        from expected_returns_v3 import _has_required_columns

        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        self.assertTrue(_has_required_columns(df, ["a", "b"], "test"))

    def test_missing_columns(self):
        from expected_returns_v3 import _has_required_columns

        df = pd.DataFrame({"a": [1]})
        self.assertFalse(_has_required_columns(df, ["a", "x", "y"], "test"))

    def test_empty_columns_list(self):
        from expected_returns_v3 import _has_required_columns

        df = pd.DataFrame({"a": [1]})
        self.assertTrue(_has_required_columns(df, [], "test"))


class TestPipelineConfig(unittest.TestCase):
    """Fix #5: PipelineConfig dataclass."""

    def test_defaults(self):
        from expected_returns_v3 import PipelineConfig

        cfg = PipelineConfig()
        self.assertEqual(cfg.mc_simulations, 50_000)
        self.assertEqual(cfg.mc_max_stocks, 10_000)
        self.assertEqual(cfg.mcmc_chains, 4)
        self.assertEqual(cfg.mcmc_samples, 10_000)
        self.assertAlmostEqual(cfg.beat_threshold, 0.6)
        self.assertEqual(cfg.output_dir, "outputs/analytics")
        self.assertEqual(cfg.log_file, "logs/expected_returns_pipeline.log")
        self.assertEqual(cfg.log_level, logging.INFO)

    def test_custom_values(self):
        from expected_returns_v3 import PipelineConfig

        cfg = PipelineConfig(mc_simulations=100, mc_max_stocks=50)
        self.assertEqual(cfg.mc_simulations, 100)
        self.assertEqual(cfg.mc_max_stocks, 50)

    def test_from_env(self):
        from expected_returns_v3 import PipelineConfig

        env = {
            "ER_MC_SIMULATIONS": "500",
            "ER_MC_MAX_STOCKS": "200",
            "ER_MCMC_CHAINS": "2",
            "ER_MCMC_SAMPLES": "1000",
            "ER_OUTPUT_DIR": "/tmp/test",
            "ER_LOG_FILE": "/tmp/test.log",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = PipelineConfig.from_env()
            self.assertEqual(cfg.mc_simulations, 500)
            self.assertEqual(cfg.mc_max_stocks, 200)
            self.assertEqual(cfg.mcmc_chains, 2)
            self.assertEqual(cfg.mcmc_samples, 1000)
            self.assertEqual(cfg.output_dir, "/tmp/test")
            self.assertEqual(cfg.log_file, "/tmp/test.log")

    def test_from_env_defaults(self):
        from expected_returns_v3 import PipelineConfig

        # With no env vars set, should use defaults
        with patch.dict(os.environ, {}, clear=True):
            cfg = PipelineConfig.from_env()
            self.assertEqual(cfg.mc_simulations, 50_000)


class TestMainAcceptsConfig(unittest.TestCase):
    """Fix #5: main() accepts PipelineConfig parameter."""

    def test_main_signature_accepts_config(self):
        import inspect
        from expected_returns_v3 import main

        sig = inspect.signature(main)
        self.assertIn("config", sig.parameters)
        # Default should be None
        self.assertIs(sig.parameters["config"].default, None)


class TestSafeDivideUsage(unittest.TestCase):
    """Fix #2: safe_divide is imported and used."""

    def test_safe_divide_imported(self):
        import expected_returns_v3 as mod
        self.assertTrue(hasattr(mod, "safe_divide"))

    def test_safe_divide_zero_denominator(self):
        from expected_returns_v3 import safe_divide
        result = safe_divide(100.0, 0.0, default=1.0)
        self.assertEqual(result, 1.0)

    def test_safe_divide_normal(self):
        from expected_returns_v3 import safe_divide
        result = safe_divide(200.0, 100.0, default=1.0)
        self.assertAlmostEqual(result, 2.0)


class TestConfigureLoggingImport(unittest.TestCase):
    """Fix #7: configure_logging is imported from finance_ml.logging_config."""

    def test_configure_logging_imported(self):
        import expected_returns_v3 as mod
        self.assertTrue(hasattr(mod, "configure_logging"))


class TestDfCopyOnFallback(unittest.TestCase):
    """Fix #3: df_all = df.copy() prevents shared reference."""

    def test_no_print_of_bare_df_assignment(self):
        """Verify the source code uses df.copy() not bare df assignment."""
        import inspect
        from expected_returns_v3 import main

        source = inspect.getsource(main)
        # Should NOT have 'df_all = df' without .copy()
        # We check that 'df_all = df.copy()' exists
        self.assertIn("df_all = df.copy()", source)
        # And the bare assignment should not exist (except as part of .copy())
        lines = source.split("\n")
        bare_assignments = [
            l.strip() for l in lines
            if l.strip().startswith("df_all = df") and ".copy()" not in l
        ]
        self.assertEqual(len(bare_assignments), 0,
                         f"Found bare df_all = df assignments: {bare_assignments}")


class TestResultVariableInitialization(unittest.TestCase):
    """Fix #6: All result variables initialized before pipeline steps."""

    def test_result_vars_initialized(self):
        """Verify main() source initializes result DataFrames."""
        import inspect
        from expected_returns_v3 import main

        source = inspect.getsource(main)
        for var in ["mc", "pt", "kal", "beat", "credit", "div_safety",
                     "tri", "quad", "strong", "summary", "df_features"]:
            self.assertIn(f"{var} = pd.DataFrame()", source,
                          f"Missing initialization: {var} = pd.DataFrame()")
        self.assertIn("screens: dict", source)
        self.assertIn("category_analytics: dict", source)


class TestTryExceptWrapping(unittest.TestCase):
    """Fix #1: Pipeline steps wrapped in try/except."""

    def test_steps_have_try_except(self):
        """Verify key pipeline steps are wrapped in try/except."""
        import inspect
        from expected_returns_v3 import main

        source = inspect.getsource(main)
        # Steps 2-5 should have error handling
        for step_marker in [
            "Step 2 (Monte Carlo) failed",
            "Step 3 (Price Target) failed",
            "Step 4 (Kalman) failed",
            "Step 5 (Earnings Beat) failed",
        ]:
            self.assertIn(step_marker, source,
                          f"Missing try/except for: {step_marker}")


class TestLoadAnalyticsTable(unittest.TestCase):
    """Tests for refactored load_analytics_table function."""

    def test_signature_has_correct_params(self):
        """load_analytics_table should accept db_url, schema, earnings_date_filter, limit."""
        import inspect
        from expected_returns_v3 import load_analytics_table

        sig = inspect.signature(load_analytics_table)
        params = list(sig.parameters.keys())
        self.assertIn("db_url", params)
        self.assertIn("schema", params)
        self.assertIn("earnings_date_filter", params)
        self.assertIn("limit", params)
        # Old params should NOT be present
        self.assertNotIn("table_name", params)
        self.assertNotIn("views", params)

    def test_delegates_to_load_feature_data_from_db(self):
        """load_analytics_table should call load_feature_data_from_db."""
        from expected_returns_v3 import load_analytics_table

        mock_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with patch("expected_returns_v3.load_feature_data_from_db", return_value=mock_df) as mock_fn:
            result = load_analytics_table(db_url="test://url", schema="myschema")
            mock_fn.assert_called_once_with(
                db_url="test://url",
                schema="myschema",
                earnings_date_filter="2026-01-01",
                limit=None,
            )
            self.assertEqual(len(result), 2)

    def test_returns_empty_on_import_error(self):
        """load_analytics_table returns empty DataFrame on ImportError."""
        from expected_returns_v3 import load_analytics_table

        with patch("expected_returns_v3.load_feature_data_from_db", side_effect=ImportError("no sqlalchemy")):
            result = load_analytics_table()
            self.assertTrue(result.empty)

    def test_returns_empty_on_value_error(self):
        """load_analytics_table returns empty DataFrame on ValueError."""
        from expected_returns_v3 import load_analytics_table

        with patch("expected_returns_v3.load_feature_data_from_db", side_effect=ValueError("no DB_URL")):
            result = load_analytics_table()
            self.assertTrue(result.empty)

    def test_returns_empty_when_db_returns_none(self):
        """load_analytics_table returns empty DataFrame when db returns None."""
        from expected_returns_v3 import load_analytics_table

        with patch("expected_returns_v3.load_feature_data_from_db", return_value=None):
            result = load_analytics_table()
            self.assertTrue(result.empty)


class TestDataLoadingSectionIncludesAllStockFeatures(unittest.TestCase):
    """Verify Step 1 loads mv_all_stock_features via load_analytics_table."""

    def test_main_calls_load_analytics_table(self):
        """main() source should call load_analytics_table() in data loading."""
        import inspect
        from expected_returns_v3 import main

        source = inspect.getsource(main)
        self.assertIn("df_features = load_analytics_table()", source)

    def test_section_header_updated(self):
        """Step 1 header should mention All Stock Features MV."""
        import inspect
        from expected_returns_v3 import main

        source = inspect.getsource(main)
        self.assertIn("All Stock Features MV", source)


if __name__ == "__main__":
    unittest.main()
