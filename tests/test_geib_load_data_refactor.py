"""
Tests for refactored load_geib_data() in geib_dash_app.py.

Validates the migration from raw SQL / manual create_engine to data_utils
abstractions: get_analytics_engine, get_equities_schema, backfill_feature_columns,
load_all_feature_views, load_feature_categories_from_db, get_view_category_mapping,
validate_feature_alignment, reorder_with_identifiers, and safe_get_column.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Module path prefix used for patching inside geib_dash_app
# ---------------------------------------------------------------------------
_MOD = "finance_ml.dashboards.geib_dash_app"

# Preserve system env vars needed by matplotlib / pathlib during clear=True patches
_SYS_ENV = {k: v for k, v in os.environ.items()
            if k in ("HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
                     "SYSTEMROOT", "TEMP", "TMP", "PATH", "APPDATA",
                     "LOCALAPPDATA", "MPLCONFIGDIR")}


class TestLoadGeibDataReturnStructure(unittest.TestCase):
    """The returned dict must contain all legacy AND new keys."""

    EXPECTED_KEYS = {
        "summary",
        "tri_model",
        "earnings",
        "credit",
        "model_confidence",
        # New keys introduced by the refactor
        "equities",
        "feature_views",
        "feature_categories",
        "view_category_mapping",
        "schema_metadata",
        "validation",
    }

    def setUp(self):
        self.env_patcher = patch.dict(os.environ, _SYS_ENV, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_returns_all_keys_when_env_not_set(self):
        """Even without env vars the dict must contain every expected key."""
        from finance_ml.dashboards.geib_dash_app import load_geib_data

        data = load_geib_data()
        self.assertEqual(set(data.keys()), self.EXPECTED_KEYS)

    def test_empty_defaults_when_env_not_set(self):
        """All DataFrame values should be empty when env vars are missing."""
        from finance_ml.dashboards.geib_dash_app import load_geib_data

        data = load_geib_data()
        for key in ("summary", "tri_model", "earnings", "credit",
                     "model_confidence", "equities"):
            self.assertTrue(data[key].empty, f"{key} should be empty")
        self.assertEqual(data["feature_views"], {})
        self.assertEqual(data["feature_categories"], {})
        self.assertEqual(data["view_category_mapping"], {})
        self.assertEqual(data["schema_metadata"], {})
        self.assertEqual(data["validation"], {})


class TestLoadGeibDataNoGeibEnv(unittest.TestCase):
    """When GEIB_DASHBOARD != 'true', return early with empty data."""

    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ, {**_SYS_ENV, "DB_URL": "postgresql://u:p@localhost/db"}, clear=True
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_returns_empty_without_geib_flag(self):
        from finance_ml.dashboards.geib_dash_app import load_geib_data

        data = load_geib_data()
        self.assertTrue(data["summary"].empty)


class TestLoadGeibDataNoDbUrl(unittest.TestCase):
    """When DB_URL is missing, return early with empty data."""

    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ, {**_SYS_ENV, "GEIB_DASHBOARD": "true"}, clear=True
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    def test_returns_empty_without_db_url(self):
        from finance_ml.dashboards.geib_dash_app import load_geib_data

        data = load_geib_data()
        self.assertTrue(data["summary"].empty)


class TestLoadGeibDataUsesDataUtils(unittest.TestCase):
    """Verify that load_geib_data delegates to data_utils instead of raw SQL."""

    _MOCK_SUMMARY = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "expected_upside_pct": [15.5, 10.2],
        "expected_return_prob_weighted": [12.0, 8.5],
        "volume_shrs": [1_000_000, 2_000_000],
        "last_price": [150.0, 300.0],
    })

    _MOCK_SCHEMA_META = {
        "last_price": {"column_name": "Last Price", "role": "market_data",
                       "column_type": "numeric", "description": "", "column_count": 1},
        "volume_shrs": {"column_name": "Volume", "role": "market_data",
                        "column_type": "bigint", "description": "", "column_count": 1},
        "ticker": {"column_name": "Ticker", "role": "id",
                   "column_type": "text", "description": "", "column_count": 1},
    }

    _MOCK_EQUITIES = pd.DataFrame({
        "ticker": ["AAPL"],
        "last_price": [150.0],
        "sector": ["Technology"],
    })

    _MOCK_FEATURE_CATEGORIES = {
        "Valuation Ratios": ["p_e_ratio", "p_b_ratio"],
        "Momentum": ["rsi_14d", "macd_signal"],
    }

    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {
            **_SYS_ENV,
            "GEIB_DASHBOARD": "true",
            "DB_URL": "postgresql://user:pass@localhost:5432/testdb",
            "DB_ANALYTICS_SCHEMA": "analytics",
        }, clear=True)
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    # ------------------------------------------------------------------
    # 1. Uses get_analytics_engine() instead of manual create_engine
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema")
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_calls_get_analytics_engine(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_schema.return_value = self._MOCK_SCHEMA_META
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        load_geib_data()

        mock_engine.assert_called_once()

    # ------------------------------------------------------------------
    # 2. Uses get_equities_schema() for numeric column detection
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema")
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_calls_get_equities_schema(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_schema.return_value = self._MOCK_SCHEMA_META
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        mock_schema.assert_called_once()
        self.assertEqual(data["schema_metadata"], self._MOCK_SCHEMA_META)

    # ------------------------------------------------------------------
    # 3. Numeric columns derived from schema (not hardcoded)
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema")
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_numeric_cols_from_schema(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        """Numeric coercion should target columns whose schema column_type is numeric-like."""
        mock_engine.return_value = MagicMock()
        # Summary has last_price as string to verify coercion happens
        summary = pd.DataFrame({
            "ticker": ["AAPL"],
            "last_price": ["150.5"],
            "volume_shrs": ["1000000"],
            "expected_upside_pct": [10.0],
            "expected_return_prob_weighted": [8.0],
        })
        mock_read_sql.return_value = summary.copy()
        mock_schema.return_value = self._MOCK_SCHEMA_META
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        # last_price & volume_shrs should be coerced to numeric (from schema)
        self.assertTrue(pd.api.types.is_numeric_dtype(data["summary"]["last_price"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(data["summary"]["volume_shrs"]))

    # ------------------------------------------------------------------
    # 4. Calls load_equities_data_from_db + backfill + reorder
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_loads_equities_with_backfill(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        mock_eq_load.assert_called_once()
        mock_backfill.assert_called_once()
        mock_reorder.assert_called_once()
        self.assertFalse(data["equities"].empty)

    # ------------------------------------------------------------------
    # 5. Calls load_all_feature_views(return_dict=True)
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views")
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_loads_feature_views_as_dict(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()
        mock_views.return_value = {"vw_features_momentum": pd.DataFrame({"a": [1]})}

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        mock_views.assert_called_once()
        # Verify return_dict=True was passed
        call_kwargs = mock_views.call_args
        self.assertTrue(call_kwargs[1].get("return_dict", False) or
                        (len(call_kwargs[0]) > 0 and call_kwargs[0][-1] is True),
                        "load_all_feature_views should be called with return_dict=True")

    # ------------------------------------------------------------------
    # 6. Calls load_feature_categories_from_db + get_view_category_mapping
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping")
    @patch(f"{_MOD}.load_feature_categories_from_db")
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_loads_feature_categories_and_mapping(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()
        mock_cats.return_value = self._MOCK_FEATURE_CATEGORIES
        mock_vcm.return_value = {"momentum": "Momentum"}

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        mock_cats.assert_called_once()
        mock_vcm.assert_called_once()
        self.assertEqual(data["feature_categories"], self._MOCK_FEATURE_CATEGORIES)
        self.assertEqual(data["view_category_mapping"], {"momentum": "Momentum"})

    # ------------------------------------------------------------------
    # 7. Calls validate_feature_alignment when data is available
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment")
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db")
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_validates_feature_alignment(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()
        mock_cats.return_value = self._MOCK_FEATURE_CATEGORIES
        mock_validate.return_value = {
            "Valuation Ratios": {"coverage_pct": 50, "missing_count": 1},
            "Momentum": {"coverage_pct": 100, "missing_count": 0},
        }

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        mock_validate.assert_called_once()
        self.assertIn("Valuation Ratios", data["validation"])

    # ------------------------------------------------------------------
    # 8. tri_model uses .copy() instead of duplicate query
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db")
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_tri_model_is_copy_of_summary(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()
        mock_eq_load.return_value = self._MOCK_EQUITIES.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        # tri_model should be equal to summary but a separate object
        pd.testing.assert_frame_equal(data["tri_model"], data["summary"])
        self.assertIsNot(data["tri_model"], data["summary"])

    # ------------------------------------------------------------------
    # 9. Graceful fallback when equities load fails
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment", return_value={})
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db", side_effect=Exception("DB down"))
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_equities_load_failure_graceful(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        data = load_geib_data()

        # Should still succeed and have empty equities
        self.assertTrue(data["equities"].empty)
        self.assertFalse(data["summary"].empty)

    # ------------------------------------------------------------------
    # 10. Skips validation when equities empty or categories empty
    # ------------------------------------------------------------------
    @patch(f"{_MOD}.validate_feature_alignment")
    @patch(f"{_MOD}.get_view_category_mapping", return_value={})
    @patch(f"{_MOD}.load_feature_categories_from_db", return_value={})
    @patch(f"{_MOD}.load_all_feature_views", return_value={})
    @patch(f"{_MOD}.reorder_with_identifiers", side_effect=lambda df: df)
    @patch(f"{_MOD}.backfill_feature_columns", side_effect=lambda df: df)
    @patch(f"{_MOD}.load_equities_data_from_db", side_effect=Exception("fail"))
    @patch(f"{_MOD}.get_equities_schema", return_value={})
    @patch("pandas.read_sql")
    @patch(f"{_MOD}.get_analytics_engine")
    def test_skips_validation_when_no_data(
        self, mock_engine, mock_read_sql, mock_schema, mock_eq_load,
        mock_backfill, mock_reorder, mock_views, mock_cats, mock_vcm,
        mock_validate,
    ):
        mock_engine.return_value = MagicMock()
        mock_read_sql.return_value = self._MOCK_SUMMARY.copy()

        from finance_ml.dashboards.geib_dash_app import load_geib_data
        load_geib_data()

        mock_validate.assert_not_called()


class TestLoadGeibDataImportsStructure(unittest.TestCase):
    """Source-level checks that the correct data_utils imports are present."""

    @classmethod
    def setUpClass(cls):
        cls.source = (
            PROJECT_ROOT / "finance_ml" / "dashboards" / "geib_dash_app.py"
        ).read_text(encoding="utf-8")

    def test_imports_get_analytics_engine(self):
        self.assertIn("get_analytics_engine", self.source)

    def test_imports_get_equities_schema(self):
        self.assertIn("get_equities_schema", self.source)

    def test_imports_load_equities_data_from_db(self):
        self.assertIn("load_equities_data_from_db", self.source)

    def test_imports_load_all_feature_views(self):
        self.assertIn("load_all_feature_views", self.source)

    def test_imports_load_feature_categories_from_db(self):
        self.assertIn("load_feature_categories_from_db", self.source)

    def test_imports_get_view_category_mapping(self):
        self.assertIn("get_view_category_mapping", self.source)

    def test_imports_validate_feature_alignment(self):
        self.assertIn("validate_feature_alignment", self.source)

    def test_imports_validate_viz_column_coverage(self):
        self.assertIn("validate_viz_column_coverage", self.source)

    def test_imports_safe_get_column(self):
        self.assertIn("safe_get_column", self.source)

    def test_imports_backfill_feature_columns(self):
        self.assertIn("backfill_feature_columns", self.source)

    def test_imports_reorder_with_identifiers(self):
        self.assertIn("reorder_with_identifiers", self.source)

    def test_imports_load_identifier_columns(self):
        self.assertIn("load_identifier_columns", self.source)

    def test_no_direct_create_engine_import(self):
        """Should NOT import create_engine from sqlalchemy directly."""
        self.assertNotIn("from sqlalchemy import create_engine", self.source)

    def test_no_hardcoded_numeric_cols_list(self):
        """The old 30-item hardcoded numeric_cols list should be removed."""
        # The old code had "prob_positive_upside" as the first item in the list
        # followed by "last_price" etc. Check the old pattern is gone.
        # We check that the old block `numeric_cols = [` with hardcoded entries
        # no longer appears inside load_geib_data
        import re
        # Old pattern: a literal list assignment with 'prob_positive_upside'
        old_pattern = r'numeric_cols\s*=\s*\[\s*"prob_positive_upside"'
        self.assertIsNone(
            re.search(old_pattern, self.source),
            "Hardcoded numeric_cols list should be replaced by schema-driven detection",
        )


class TestVizColumnCoverageAtStartup(unittest.TestCase):
    """Verify VIZ_REQUIRED_COLUMNS dict and validate_viz_column_coverage usage."""

    @classmethod
    def setUpClass(cls):
        cls.source = (
            PROJECT_ROOT / "finance_ml" / "dashboards" / "geib_dash_app.py"
        ).read_text(encoding="utf-8")

    def test_viz_required_columns_defined(self):
        """A VIZ_REQUIRED_COLUMNS mapping should be defined."""
        self.assertIn("VIZ_REQUIRED_COLUMNS", self.source)

    def test_validate_viz_column_coverage_called(self):
        """validate_viz_column_coverage should be called at startup."""
        self.assertIn("validate_viz_column_coverage", self.source)


class TestSafeGetColumnUsage(unittest.TestCase):
    """Verify that safe_get_column is used somewhere in the dashboard callbacks."""

    @classmethod
    def setUpClass(cls):
        cls.source = (
            PROJECT_ROOT / "finance_ml" / "dashboards" / "geib_dash_app.py"
        ).read_text(encoding="utf-8")

    def test_safe_get_column_used_in_code(self):
        """safe_get_column should appear in the source beyond just the import."""
        import re
        # Count occurrences excluding import lines
        non_import = [
            line for line in self.source.splitlines()
            if "safe_get_column" in line and "import" not in line
        ]
        self.assertGreaterEqual(
            len(non_import), 1,
            "safe_get_column should be used at least once in dashboard code",
        )


if __name__ == "__main__":
    unittest.main()
