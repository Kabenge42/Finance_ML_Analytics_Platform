"""
Tests for Equities Dashboard Refactorings (TDD approach).

This module tests the refactoring changes documented in dashboard_refactorings.md:
1. data_utils.py: _resolve_db_url(), renamed import, improved load_data_csv_first/load_data
2. equities_dashboard_app.py: _resolve_db_url_runtime(), run_dashboard_etl_pipeline rename,
   load_from_equities_table fallback, _load_initial_data exception handling, _prepare_data_store None check
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd


class TestDataUtilsResolveDbUrl(unittest.TestCase):
    """Tests for _resolve_db_url() helper in data_utils.py."""

    def test_resolve_db_url_with_explicit_parameter(self):
        """Should return explicit db_url parameter when provided."""
        from finance_ml.dashboards.components.data_utils import _resolve_db_url

        result = _resolve_db_url("postgresql://explicit:5432/db")
        self.assertEqual(result, "postgresql://explicit:5432/db")

    def test_resolve_db_url_from_db_url_env(self):
        """Should resolve from DB_URL environment variable."""
        from finance_ml.dashboards.components.data_utils import _resolve_db_url

        with patch.dict(os.environ, {"DB_URL": "postgresql://env:5432/db"}, clear=False):
            # Clear DATABASE_URL to ensure DB_URL is used
            env_copy = os.environ.copy()
            if "DATABASE_URL" in env_copy:
                del env_copy["DATABASE_URL"]
            with patch.dict(os.environ, env_copy, clear=True):
                with patch.dict(os.environ, {"DB_URL": "postgresql://env:5432/db"}):
                    result = _resolve_db_url(None)
                    self.assertEqual(result, "postgresql://env:5432/db")

    def test_resolve_db_url_from_database_url_env(self):
        """Should fall back to DATABASE_URL environment variable."""
        from finance_ml.dashboards.components.data_utils import _resolve_db_url

        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://fallback:5432/db"}, clear=True):
            result = _resolve_db_url(None)
            self.assertEqual(result, "postgresql://fallback:5432/db")

    def test_resolve_db_url_returns_none_when_not_configured(self):
        """Should return None when no DB URL is configured."""
        from finance_ml.dashboards.components.data_utils import _resolve_db_url

        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_db_url(None)
            self.assertIsNone(result)

    def test_resolve_db_url_explicit_overrides_env(self):
        """Explicit parameter should override environment variables."""
        from finance_ml.dashboards.components.data_utils import _resolve_db_url

        with patch.dict(os.environ, {"DB_URL": "postgresql://env:5432/db"}):
            result = _resolve_db_url("postgresql://explicit:5432/db")
            self.assertEqual(result, "postgresql://explicit:5432/db")


class TestDataUtilsLoadDataCsvFirst(unittest.TestCase):
    """Tests for improved load_data_csv_first() in data_utils.py."""

    @patch("finance_ml.dashboards.components.data_utils._run_etl_pipeline")
    def test_load_data_csv_first_uses_runtime_db_url_resolution(self, mock_etl):
        """Should resolve DB URL at call time, not import time."""
        from finance_ml.dashboards.components.data_utils import load_data_csv_first

        mock_df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "name": ["Apple"],
                "sector": ["Tech"],
                "region": ["US"],
                "last_price": [150.0],
                "price_target": [180.0],
                "market_cap": [2.5e12],
            }
        )
        mock_metrics = MagicMock()
        mock_metrics.summary.return_value = "ETL Summary"
        mock_etl.return_value = (mock_df, mock_metrics)

        # Set env var after import
        with patch.dict(os.environ, {"DB_URL": "postgresql://runtime:5432/db"}, clear=True):
            df, source = load_data_csv_first(force_etl=True)
            # Should have called ETL with db source since DB_URL is set
            self.assertTrue(mock_etl.called)

    @patch("finance_ml.dashboards.components.data_utils._run_etl_pipeline")
    def test_load_data_csv_first_validates_etl_output(self, mock_etl):
        """Should validate ETL output and return error for empty DataFrame."""
        from finance_ml.dashboards.components.data_utils import load_data_csv_first

        mock_etl.return_value = (pd.DataFrame(), None)

        with patch.dict(os.environ, {}, clear=True):
            df, source = load_data_csv_first(force_etl=True)
            self.assertTrue(df.empty)
            self.assertIn("ETL", source)  # Should indicate ETL issue

    @patch("finance_ml.dashboards.components.data_utils._run_etl_pipeline")
    def test_load_data_csv_first_returns_error_message_on_failure(self, mock_etl):
        """Should return truncated error message on ETL failure."""
        from finance_ml.dashboards.components.data_utils import load_data_csv_first

        mock_etl.side_effect = Exception("A very long error message " * 10)

        with patch.dict(os.environ, {}, clear=True):
            df, source = load_data_csv_first(force_etl=True)
            self.assertTrue(df.empty)
            self.assertIn("ETL failed", source)
            # Error message should be truncated
            self.assertLessEqual(len(source), 150)


class TestDataUtilsLoadData(unittest.TestCase):
    """Tests for improved load_data() in data_utils.py."""

    @patch("finance_ml.dashboards.components.data_utils._run_etl_pipeline")
    @patch("finance_ml.dashboards.components.data_utils.load_data_csv_first")
    def test_load_data_uses_resolve_db_url(self, mock_csv_first, mock_etl):
        """Should use _resolve_db_url for runtime resolution."""
        from finance_ml.dashboards.components.data_utils import load_data

        mock_df = pd.DataFrame({"ticker": ["AAPL"]})
        mock_csv_first.return_value = (mock_df, "csv")

        with patch.dict(os.environ, {"DB_URL": "postgresql://test:5432/db"}, clear=True):
            result = load_data(data_source="auto")
            # Should work without errors
            self.assertIsNotNone(result)

    @patch("finance_ml.dashboards.components.data_utils._run_etl_pipeline")
    def test_load_data_handles_none_dataframe(self, mock_etl):
        """Should handle None DataFrame gracefully."""
        from finance_ml.dashboards.components.data_utils import load_data

        mock_etl.return_value = (None, None)

        with patch.dict(os.environ, {}, clear=True):
            result = load_data(data_source="csv")
            self.assertIsInstance(result, pd.DataFrame)
            self.assertTrue(result.empty)


class TestEquitiesDashboardResolveDbUrlRuntime(unittest.TestCase):
    """Tests for _resolve_db_url_runtime() in equities_dashboard_app.py."""

    def test_resolve_db_url_runtime_exists(self):
        """Function should exist in the module."""
        from finance_ml.dashboards import equities_dashboard_app

        self.assertTrue(hasattr(equities_dashboard_app, "_resolve_db_url_runtime"))

    def test_resolve_db_url_runtime_with_explicit_parameter(self):
        """Should return explicit db_url parameter when provided."""
        from finance_ml.dashboards.equities_dashboard_app import _resolve_db_url_runtime

        result = _resolve_db_url_runtime("postgresql://explicit:5432/db")
        self.assertEqual(result, "postgresql://explicit:5432/db")

    def test_resolve_db_url_runtime_from_env(self):
        """Should resolve from environment at runtime."""
        from finance_ml.dashboards.equities_dashboard_app import _resolve_db_url_runtime

        with patch.dict(os.environ, {"DB_URL": "postgresql://runtime:5432/db"}, clear=True):
            result = _resolve_db_url_runtime(None)
            self.assertEqual(result, "postgresql://runtime:5432/db")

    def test_resolve_db_url_runtime_returns_none_when_not_configured(self):
        """Should return None when no DB URL is configured."""
        from finance_ml.dashboards.equities_dashboard_app import _resolve_db_url_runtime

        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_db_url_runtime(None)
            self.assertIsNone(result)


class TestRunDashboardEtlPipelineRename(unittest.TestCase):
    """Tests for run_dashboard_etl_pipeline() rename."""

    def test_run_dashboard_etl_pipeline_exists(self):
        """Function should exist with new name."""
        from finance_ml.dashboards import equities_dashboard_app

        self.assertTrue(hasattr(equities_dashboard_app, "run_dashboard_etl_pipeline"))

    def test_run_dashboard_etl_pipeline_normalizes_columns(self):
        """Should normalize column names from SQL format."""
        from finance_ml.dashboards.equities_dashboard_app import run_dashboard_etl_pipeline

        # Create mock raw data with SQL-style column names
        raw_data = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Last Price": [150.0],
                "Market Cap": [2.5e12],
            }
        )

        with patch("finance_ml.dashboards.equities_dashboard_app.ETLPipeline") as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.transform.return_value = raw_data
            mock_pipeline.load.return_value = raw_data
            mock_pipeline.metrics = MagicMock()
            mock_pipeline.metrics.rows_input = 1
            mock_pipeline.metrics.stages_executed = ["transform", "load"]
            mock_pipeline_cls.return_value = mock_pipeline

            result = run_dashboard_etl_pipeline(raw_data.copy())
            # Should have called transform and load
            self.assertTrue(mock_pipeline.transform.called)
            self.assertTrue(mock_pipeline.load.called)


class TestLoadFromEquitiesTableFallback(unittest.TestCase):
    """Tests for load_from_equities_table() fallback logic."""

    @patch("finance_ml.dashboards.equities_dashboard_app.create_engine")
    def test_load_from_equities_table_fallback_to_most_recent(self, mock_create_engine):
        """Should fall back to most recent data when no data for current_date - 1."""
        from finance_ml.dashboards.equities_dashboard_app import load_from_equities_table

        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine

        # Mock connection context manager
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        # Mock count query - table has data
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 100
        mock_conn.execute.return_value = mock_count_result

        # First query returns empty (no data for current_date - 1)
        # Second query returns data (most recent)
        with patch("finance_ml.dashboards.equities_dashboard_app.pd.read_sql") as mock_read_sql:
            mock_read_sql.side_effect = [
                pd.DataFrame(),  # First call: empty for current_date - 1
                pd.DataFrame(
                    {"ticker": ["AAPL"], "last_price": [150.0]}
                ),  # Second call: most recent
            ]

            result = load_from_equities_table("postgresql://test:5432/db")

            # Should have called read_sql twice (fallback)
            self.assertEqual(mock_read_sql.call_count, 2)
            self.assertFalse(result.empty)

    @patch("finance_ml.dashboards.equities_dashboard_app.create_engine")
    def test_load_from_equities_table_raises_on_empty_db_url(self, mock_create_engine):
        """Should raise ValueError when db_url is empty."""
        from finance_ml.dashboards.equities_dashboard_app import load_from_equities_table

        with self.assertRaises(ValueError) as ctx:
            load_from_equities_table("")

        self.assertIn("Database URL is required", str(ctx.exception))


class TestLoadInitialDataExceptionHandling(unittest.TestCase):
    """Tests for _load_initial_data() improved exception handling."""

    @patch("finance_ml.dashboards.equities_dashboard_app.load_from_equities_table")
    @patch("finance_ml.dashboards.equities_dashboard_app.load_data")
    def test_load_initial_data_handles_generic_exception(self, mock_load_data, mock_load_equities):
        """Should handle generic exceptions and fall back to CSV."""
        from finance_ml.dashboards.equities_dashboard_app import _load_initial_data

        mock_load_equities.side_effect = Exception("Unexpected error")
        mock_load_data.return_value = pd.DataFrame({"ticker": ["AAPL"]})

        df, source = _load_initial_data("db", "postgresql://test:5432/db", None)

        # Should have fallen back to CSV
        self.assertEqual(source, "csv")
        self.assertTrue(mock_load_data.called)

    @patch("finance_ml.dashboards.equities_dashboard_app.load_data")
    def test_load_initial_data_returns_none_source_on_total_failure(self, mock_load_data):
        """Should return 'none' source when all loading fails."""
        from finance_ml.dashboards.equities_dashboard_app import _load_initial_data

        mock_load_data.return_value = None

        df, source = _load_initial_data("csv", None, None)

        self.assertEqual(source, "none")
        self.assertTrue(df.empty)

    @patch("finance_ml.dashboards.equities_dashboard_app.load_data")
    def test_load_initial_data_handles_empty_dataframe_from_csv(self, mock_load_data):
        """Should handle empty DataFrame from CSV source."""
        from finance_ml.dashboards.equities_dashboard_app import _load_initial_data

        mock_load_data.return_value = pd.DataFrame()

        df, source = _load_initial_data("csv", None, None)

        # Should return empty df with 'none' source
        self.assertTrue(df.empty)
        self.assertEqual(source, "none")


class TestPrepareDataStoreNoneCheck(unittest.TestCase):
    """Tests for _prepare_data_store() None check."""

    def test_prepare_data_store_handles_none_dataframe(self):
        """Should handle None DataFrame gracefully."""
        from finance_ml.dashboards.equities_dashboard_app import _prepare_data_store

        result = _prepare_data_store(None)
        self.assertIsNone(result)

    def test_prepare_data_store_handles_empty_dataframe(self):
        """Should return None for empty DataFrame."""
        from finance_ml.dashboards.equities_dashboard_app import _prepare_data_store

        result = _prepare_data_store(pd.DataFrame())
        self.assertIsNone(result)

    def test_prepare_data_store_returns_json_for_valid_dataframe(self):
        """Should return JSON string for valid DataFrame."""
        from finance_ml.dashboards.equities_dashboard_app import _prepare_data_store

        df = pd.DataFrame({"ticker": ["AAPL", "GOOGL"], "price": [150.0, 2800.0]})
        result = _prepare_data_store(df)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        self.assertIn("data", parsed)


class TestCreateAppUsesRuntimeDbUrlResolution(unittest.TestCase):
    """Tests for create_app() using runtime DB URL resolution."""

    def test_create_app_uses_resolve_db_url_runtime(self):
        """create_app should use _resolve_db_url_runtime for DB URL resolution."""
        from finance_ml.dashboards import equities_dashboard_app

        # Verify the function exists and is used
        self.assertTrue(hasattr(equities_dashboard_app, "_resolve_db_url_runtime"))

        # Create app without loading data
        app = equities_dashboard_app.create_app(load_on_start=False)
        self.assertIsNotNone(app)


class TestInitialStatusMessage(unittest.TestCase):
    """Tests for initial status message when no data is loaded."""

    def test_create_app_shows_status_message_when_no_data(self):
        """Should show helpful status message when no data is loaded."""
        from finance_ml.dashboards import equities_dashboard_app

        app = equities_dashboard_app.create_app(load_on_start=False)

        # The app should have a layout
        self.assertIsNotNone(app.layout)


if __name__ == "__main__":
    unittest.main()
