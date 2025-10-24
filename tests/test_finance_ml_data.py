"""
Test suite for finance_ml.data module

This module tests data loading, normalization, and validation functions
following TDD methodology for Phase 7 refactoring.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd


class TestNormalizeColumns(unittest.TestCase):
    """Test column normalization function"""

    def test_normalize_columns_spaces_to_underscores(self):
        """Should convert spaces to underscores and lowercase"""
        df = pd.DataFrame({"Last Price": [100, 200], "Market Cap": [1e9, 2e9]})
        from finance_ml.data import normalize_columns

        result = normalize_columns(df)
        self.assertIn("last_price", result.columns)
        self.assertIn("market_cap", result.columns)

    def test_normalize_columns_special_chars(self):
        """Should remove special characters and strip underscores"""
        df = pd.DataFrame({"P/E (NTM)": [15, 20], "EBITDA (-1FY)": [100, 200]})
        from finance_ml.data import normalize_columns

        result = normalize_columns(df)
        self.assertIn("p_e_ntm", result.columns)
        self.assertIn("ebitda_1fy", result.columns)

    def test_normalize_columns_preserves_data(self):
        """Should preserve data values after normalization"""
        df = pd.DataFrame({"Last Price": [100, 200, 300]})
        from finance_ml.data import normalize_columns

        result = normalize_columns(df)
        pd.testing.assert_series_equal(
            result["last_price"], pd.Series([100, 200, 300], name="last_price")
        )


class TestInferRegionFromFilename(unittest.TestCase):
    """Test region inference from file paths"""

    def test_infer_us_region(self):
        """Should detect US region from filename"""
        from finance_ml.data import infer_region_from_filename

        path = Path("data/screening_us.csv")
        self.assertEqual(infer_region_from_filename(path), "US")

    def test_infer_eu_region(self):
        """Should detect EU region from filename"""
        from finance_ml.data import infer_region_from_filename

        path = Path("data/screening_eu.csv")
        self.assertEqual(infer_region_from_filename(path), "EU")

    def test_infer_apac_region(self):
        """Should detect APAC region from filename"""
        from finance_ml.data import infer_region_from_filename

        path = Path("data/screening_apac.csv")
        self.assertEqual(infer_region_from_filename(path), "APAC")

    def test_infer_rotw_region(self):
        """Should detect ROTW region from filename"""
        from finance_ml.data import infer_region_from_filename

        path = Path("data/screening_rotw.csv")
        self.assertEqual(infer_region_from_filename(path), "ROTW")

    def test_infer_unknown_region(self):
        """Should return None for unrecognized filenames"""
        from finance_ml.data import infer_region_from_filename

        path = Path("data/other_file.csv")
        self.assertIsNone(infer_region_from_filename(path))


class TestLoadFromCSV(unittest.TestCase):
    """Test CSV loading function"""

    def setUp(self):
        """Create temporary CSV files for testing"""
        self.temp_dir = tempfile.mkdtemp()

        # Create a minimal test CSV
        csv_content = """Ticker,Name,Sector,Last Price,Market Cap
AAPL,Apple Inc.,Technology,150.0,2500000000000
MSFT,Microsoft,Technology,300.0,2200000000000"""

        csv_path = Path(self.temp_dir) / "screening_us.csv"
        csv_path.write_text(csv_content)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_from_csv_returns_dataframe(self):
        """Should return a pandas DataFrame"""
        from finance_ml.data import load_from_csv

        result = load_from_csv(Path(self.temp_dir))
        self.assertIsInstance(result, pd.DataFrame)

    def test_load_from_csv_has_region_column(self):
        """Should add Region column based on filename"""
        from finance_ml.data import load_from_csv

        result = load_from_csv(Path(self.temp_dir))
        self.assertIn("region", result.columns)
        self.assertTrue((result["region"] == "US").all())

    def test_load_from_csv_normalized_columns(self):
        """Should have normalized column names"""
        from finance_ml.data import load_from_csv

        result = load_from_csv(Path(self.temp_dir))
        self.assertIn("ticker", result.columns)
        self.assertIn("last_price", result.columns)
        self.assertIn("market_cap", result.columns)

    def test_load_from_csv_with_limit(self):
        """Should limit rows when limit parameter is provided"""
        from finance_ml.data import load_from_csv

        result = load_from_csv(Path(self.temp_dir), limit=1)
        self.assertEqual(len(result), 1)


class TestLoadFromDB(unittest.TestCase):
    """Test database loading function"""

    def test_load_from_db_requires_sqlalchemy(self):
        """Should raise ImportError if SQLAlchemy not available"""
        # This test checks that the function handles missing dependencies
        from finance_ml.data import load_from_db

        # If SQLAlchemy is not installed, should raise ImportError
        # If it is installed, we skip this test
        try:
            import sqlalchemy

            self.skipTest("SQLAlchemy is installed")
        except ImportError:
            with self.assertRaises(ImportError):
                load_from_db("postgresql://fake", limit=10)

    def test_load_from_db_validates_url(self):
        """Should validate database URL format"""
        from finance_ml.data import load_from_db

        try:
            import sqlalchemy

            # Test with invalid URL should raise an error
            with self.assertRaises(Exception):
                load_from_db("invalid_url", limit=10)
        except ImportError:
            self.skipTest("SQLAlchemy not installed")


class TestPreprocess(unittest.TestCase):
    """Test preprocessing function"""

    def test_preprocess_drops_missing_ticker(self):
        """Should drop rows with missing Ticker"""
        from finance_ml.data import preprocess

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", None, "MSFT"],
                "sector": ["Tech", "Tech", "Tech"],
                "last_price": [150, 200, 300],
            }
        )
        result = preprocess(df)
        self.assertEqual(len(result), 2)
        self.assertNotIn(None, result["ticker"].values)

    def test_preprocess_drops_missing_sector(self):
        """Should drop rows with missing Sector"""
        from finance_ml.data import preprocess

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOG"],
                "sector": ["Tech", None, "Tech"],
                "last_price": [150, 200, 300],
            }
        )
        result = preprocess(df)
        self.assertEqual(len(result), 2)

    def test_preprocess_converts_numeric_columns(self):
        """Should convert numeric columns with coercion"""
        from finance_ml.data import preprocess

        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Tech", "Tech"],
                "last_price": ["150", "200"],
                "market_cap": ["1e9", "2e9"],
            }
        )
        result = preprocess(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(result["last_price"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(result["market_cap"]))


class TestValidateSchema(unittest.TestCase):
    """Test schema validation function"""

    def test_validate_schema_requires_core_columns(self):
        """Should raise ValueError if core columns are missing"""
        from finance_ml.data import validate_schema

        df = pd.DataFrame({"ticker": ["AAPL"]})
        with self.assertRaises(ValueError):
            validate_schema(df, require_target=False)

    def test_validate_schema_with_target(self):
        """Should check for target columns when require_target=True"""
        from finance_ml.data import validate_schema

        df = pd.DataFrame({"ticker": ["AAPL"], "sector": ["Tech"], "last_price": [150]})
        with self.assertRaises(ValueError):
            validate_schema(df, require_target=True)

    def test_validate_schema_passes_with_valid_data(self):
        """Should pass validation with all required columns"""
        from finance_ml.data import validate_schema

        df = pd.DataFrame(
            {
                "ticker": ["AAPL"],
                "sector": ["Tech"],
                "last_price": [150],
                "market_cap": [1e9],
                "enterprise_value": [1.1e9],
                "ebitda_ltm": [1e8],
            }
        )
        # Should not raise
        validate_schema(df, require_target=False)


class TestCheckMissingValues(unittest.TestCase):
    """Test missing values check function"""

    def test_check_missing_values_returns_dict(self):
        """Should return dictionary with missing value info"""
        from finance_ml.data import check_missing_values

        df = pd.DataFrame({"ticker": ["AAPL", "MSFT", None], "last_price": [150, 200, 300]})
        result = check_missing_values(df)
        self.assertIsInstance(result, dict)
        self.assertIn("ticker", result)

    def test_check_missing_values_calculates_percentages(self):
        """Should calculate percentage of missing values"""
        from finance_ml.data import check_missing_values

        df = pd.DataFrame({"col1": [1, None, 3, None], "col2": [1, 2, 3, 4]})
        result = check_missing_values(df)
        # Returns dict with 'count' and 'percentage' keys for columns with missing values
        self.assertIn("col1", result)
        self.assertEqual(result["col1"]["count"], 2)
        self.assertAlmostEqual(result["col1"]["percentage"], 50.0)
        # col2 has no missing values, so it shouldn't be in the result
        self.assertNotIn("col2", result)


class TestDetectOutliersIQR(unittest.TestCase):
    """Test IQR-based outlier detection"""

    def test_detect_outliers_iqr_returns_series(self):
        """Should return list of indices"""
        from finance_ml.data import detect_outliers_iqr

        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 100]})
        result = detect_outliers_iqr(df, "values")
        self.assertIsInstance(result, list)

    def test_detect_outliers_iqr_identifies_outliers(self):
        """Should identify extreme outliers"""
        from finance_ml.data import detect_outliers_iqr

        df = pd.DataFrame({"values": [1, 2, 3, 4, 5, 1000]})
        result = detect_outliers_iqr(df, "values")
        # Should return list containing index 5 (last value)
        self.assertIn(5, result)

    def test_detect_outliers_iqr_with_multiplier(self):
        """Should respect multiplier parameter"""
        from finance_ml.data import detect_outliers_iqr

        df = pd.DataFrame({"values": list(range(100)) + [200]})
        result = detect_outliers_iqr(df, "values", multiplier=1.5)
        # Should identify at least one outlier
        self.assertGreater(len(result), 0)


class TestValidateNumericRanges(unittest.TestCase):
    """Test numeric range validation"""

    def test_validate_numeric_ranges_returns_dict(self):
        """Should return dictionary with validation results"""
        from finance_ml.data import validate_numeric_ranges

        df = pd.DataFrame({"last_price": [150, 200], "market_cap": [1e9, 2e9]})
        result = validate_numeric_ranges(df)
        self.assertIsInstance(result, dict)

    def test_validate_numeric_ranges_flags_negatives(self):
        """Should flag negative values in price columns"""
        from finance_ml.data import validate_numeric_ranges

        df = pd.DataFrame({"last_price": [-150, 200], "market_cap": [1e9, 2e9]})
        result = validate_numeric_ranges(df)
        # Returns dict with column names as keys and lists of invalid indices as values
        self.assertIn("last_price", result)
        self.assertIsInstance(result["last_price"], list)
        self.assertIn(0, result["last_price"])  # Index 0 has negative value

    def test_validate_numeric_ranges_checks_market_cap(self):
        """Should validate market cap is positive"""
        from finance_ml.data import validate_numeric_ranges

        df = pd.DataFrame({"last_price": [150, 200], "market_cap": [-1e9, 2e9]})
        result = validate_numeric_ranges(df)
        self.assertIn("market_cap", result)
        self.assertIsInstance(result["market_cap"], list)
        self.assertIn(0, result["market_cap"])  # Index 0 has negative value


class TestGetEnv(unittest.TestCase):
    """Test get_env utility function"""

    def test_get_env_returns_value(self):
        """Should return environment variable value"""
        from finance_ml.data import get_env

        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = get_env("TEST_VAR")
            self.assertEqual(result, "test_value")

    def test_get_env_strips_whitespace(self):
        """Should strip whitespace from environment variable values"""
        from finance_ml.data import get_env

        with patch.dict(os.environ, {"TEST_VAR": "  value_with_spaces  "}):
            result = get_env("TEST_VAR")
            self.assertEqual(result, "value_with_spaces")

    def test_get_env_returns_default(self):
        """Should return default when variable not set"""
        from finance_ml.data import get_env

        result = get_env("NONEXISTENT_VAR", default="default_value")
        self.assertEqual(result, "default_value")

    def test_get_env_returns_none_when_not_set(self):
        """Should return None when variable not set and no default"""
        from finance_ml.data import get_env

        result = get_env("NONEXISTENT_VAR")
        self.assertIsNone(result)


class TestLoadFromCSVErrors(unittest.TestCase):
    """Test load_from_csv error handling"""

    def test_load_from_csv_raises_when_no_files(self):
        """Should raise FileNotFoundError when no CSV files found"""
        from finance_ml.data import load_from_csv

        temp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_from_csv(Path(temp_dir))
            self.assertIn("No CSV files found", str(ctx.exception))
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)


class TestLoadFromDBFull(unittest.TestCase):
    """Test load_from_db functionality"""

    def test_load_from_db_executes_query(self):
        """Should execute SQL query and load data"""
        from finance_ml.data import load_from_db

        # Mock the create_engine and query execution
        mock_engine = MagicMock()
        mock_df = pd.DataFrame(
            {"Ticker": ["AAPL", "MSFT"], "Region": ["US", "US"], "Last Price": [150, 300]}
        )

        with patch("finance_ml.data.create_engine", return_value=mock_engine):
            with patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
                result = load_from_db("postgresql://test")

                # Verify query was executed
                mock_read_sql.assert_called_once()

                # Verify columns are normalized
                self.assertIn("ticker", result.columns)
                self.assertIn("region", result.columns)
                self.assertIn("last_price", result.columns)

    def test_load_from_db_with_limit(self):
        """Should apply limit to query"""
        from finance_ml.data import load_from_db

        mock_engine = MagicMock()
        mock_df = pd.DataFrame({"Ticker": ["AAPL"], "Region": ["US"]})

        with patch("finance_ml.data.create_engine", return_value=mock_engine):
            with patch("pandas.read_sql", return_value=mock_df) as mock_read_sql:
                result = load_from_db("postgresql://test", limit=100)

                # Verify limit was applied to query
                call_args = mock_read_sql.call_args
                query = call_args[0][0]
                self.assertIn("LIMIT", query)
                self.assertIn("100", query)


class TestCheckMissingValuesEdgeCases(unittest.TestCase):
    """Test check_missing_values edge cases"""

    def test_check_missing_values_empty_dataframe(self):
        """Should handle empty DataFrame gracefully"""
        from finance_ml.data import check_missing_values

        df = pd.DataFrame()
        result = check_missing_values(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)

    def test_check_missing_values_zero_rows(self):
        """Should return empty dict for DataFrame with columns but no rows"""
        from finance_ml.data import check_missing_values

        df = pd.DataFrame(columns=["col1", "col2"])
        result = check_missing_values(df)
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)


class TestSafeDiv(unittest.TestCase):
    """Test _safe_div helper function"""

    def test_safe_div_normal_division(self):
        """Should perform normal division for valid values"""
        from finance_ml.data import _safe_div

        numer = pd.Series([10, 20, 30])
        denom = pd.Series([2, 4, 5])
        result = _safe_div(numer, denom)
        expected = pd.Series([5.0, 5.0, 6.0])
        pd.testing.assert_series_equal(result, expected)

    def test_safe_div_handles_zero_denominator(self):
        """Should replace inf with NaN when dividing by zero"""
        from finance_ml.data import _safe_div

        numer = pd.Series([10, 20])
        denom = pd.Series([0, 2])
        result = _safe_div(numer, denom)
        self.assertTrue(pd.isna(result.iloc[0]))  # 10/0 -> inf -> NaN
        self.assertEqual(result.iloc[1], 10.0)  # 20/2 = 10.0

    def test_safe_div_handles_negative_inf(self):
        """Should replace -inf with NaN"""
        from finance_ml.data import _safe_div

        numer = pd.Series([-10, 20])
        denom = pd.Series([0, 2])
        result = _safe_div(numer, denom)
        self.assertTrue(pd.isna(result.iloc[0]))  # -10/0 -> -inf -> NaN
        self.assertEqual(result.iloc[1], 10.0)


if __name__ == "__main__":
    unittest.main()
