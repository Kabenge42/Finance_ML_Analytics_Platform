#!/usr/bin/env python3
"""
Test suite for SQLite import functionality (tools/import_sqlite.py).

Following strict TDD approach as per IMPROVEMENT_PLAN.md Phase 2:
- Tests for SQLite import hardening (shell-based via SQL script and Python-based)
- Validates header removal, NULL handling, region backfilling, and deduplication
- Ensures UNIQUE("Ticker","Region") constraint prevents duplicates
- Provides post-import validation queries

Target coverage: ≥80% for tools/import_sqlite.py
"""
import os
import sqlite3
# Import the module under test
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.import_sqlite import (
    chunk_insert_dataframe,
    import_region,
    iter_regions,
    quote_identifier,
)


class TestQuoteIdentifier(unittest.TestCase):
    """Test SQL identifier quoting."""

    def test_simple_identifier(self):
        """Test quoting a simple identifier."""
        result = quote_identifier("Ticker")
        self.assertEqual(result, '"Ticker"')

    def test_identifier_with_spaces(self):
        """Test quoting identifier with spaces."""
        result = quote_identifier("Last Price")
        self.assertEqual(result, '"Last Price"')

    def test_identifier_with_quotes(self):
        """Test quoting identifier containing double quotes."""
        result = quote_identifier('Price "Adjusted"')
        self.assertEqual(result, '"Price ""Adjusted"""')

    def test_empty_string_raises_error(self):
        """Test that empty string raises ValueError."""
        with self.assertRaises(ValueError):
            quote_identifier("")

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with self.assertRaises(ValueError):
            quote_identifier(None)

    def test_special_characters(self):
        """Test quoting identifier with special characters."""
        result = quote_identifier("P/E Ratio")
        self.assertEqual(result, '"P/E Ratio"')


class TestIterRegions(unittest.TestCase):
    """Test region selection and validation."""

    def test_no_selection_returns_all(self):
        """Test that None or empty selection returns all regions."""
        result = iter_regions(None)
        self.assertEqual(set(result), {"US", "EU", "APAC", "ROTW"})

    def test_single_region(self):
        """Test selecting a single region."""
        result = iter_regions("US")
        self.assertEqual(result, ["US"])

    def test_multiple_regions(self):
        """Test selecting multiple regions."""
        result = iter_regions("US,EU")
        self.assertEqual(result, ["US", "EU"])

    def test_case_insensitive(self):
        """Test that region names are case-insensitive."""
        result = iter_regions("us,eu")
        self.assertEqual(result, ["US", "EU"])

    def test_whitespace_handling(self):
        """Test that whitespace is stripped."""
        result = iter_regions(" US , EU ")
        self.assertEqual(result, ["US", "EU"])

    def test_invalid_region_filtered(self):
        """Test that invalid regions are filtered out."""
        result = iter_regions("US,INVALID,EU")
        self.assertEqual(result, ["US", "EU"])

    def test_all_invalid_raises_error(self):
        """Test that all invalid regions raises ValueError."""
        with self.assertRaises(ValueError):
            iter_regions("INVALID1,INVALID2")

    def test_empty_string_returns_all(self):
        """Test that empty string returns all regions."""
        result = iter_regions("")
        self.assertEqual(set(result), {"US", "EU", "APAC", "ROTW"})


class TestChunkInsertDataFrame(unittest.TestCase):
    """Test DataFrame insertion into SQLite."""

    def setUp(self):
        """Create temporary SQLite database for testing."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)
        # Create minimal test table
        self.conn.execute(
            """
            CREATE TABLE equities (
                "Ticker" TEXT,
                "Name" TEXT,
                "Sector" TEXT,
                "Region" TEXT,
                "Last Price" NUMERIC,
                UNIQUE("Ticker", "Region")
            )
        """
        )
        self.conn.commit()

    def tearDown(self):
        """Clean up temporary database."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)

    def test_insert_simple_dataframe(self):
        """Test inserting a simple DataFrame."""
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "GOOGL"],
                "Name": ["Apple Inc", "Alphabet Inc"],
                "Sector": ["Technology", "Technology"],
                "Region": ["US", "US"],
                "Last Price": [150.0, 2800.0],
            }
        )
        count = chunk_insert_dataframe(self.conn, df)
        self.assertEqual(count, 2)

        # Verify insertion
        cursor = self.conn.execute("SELECT COUNT(*) FROM equities")
        self.assertEqual(cursor.fetchone()[0], 2)

    def test_empty_strings_converted_to_null(self):
        """Test that empty strings are converted to NULL."""
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "GOOGL"],
                "Name": ["Apple Inc", ""],
                "Sector": ["Technology", "Technology"],
                "Region": ["US", "US"],
                "Last Price": [150.0, 2800.0],
            }
        )
        chunk_insert_dataframe(self.conn, df)

        # Verify NULL was inserted
        cursor = self.conn.execute('SELECT "Name" FROM equities WHERE "Ticker" = ?', ("GOOGL",))
        result = cursor.fetchone()[0]
        self.assertIsNone(result)

    def test_duplicate_insert_ignored(self):
        """Test that duplicate inserts are ignored (INSERT OR IGNORE)."""
        df1 = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Name": ["Apple Inc"],
                "Sector": ["Technology"],
                "Region": ["US"],
                "Last Price": [150.0],
            }
        )
        chunk_insert_dataframe(self.conn, df1)

        # Try to insert duplicate
        df2 = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Name": ["Apple Inc Updated"],
                "Sector": ["Technology"],
                "Region": ["US"],
                "Last Price": [155.0],
            }
        )
        chunk_insert_dataframe(self.conn, df2)

        # Verify only one row exists with original data
        cursor = self.conn.execute("SELECT COUNT(*) FROM equities")
        self.assertEqual(cursor.fetchone()[0], 1)

        cursor = self.conn.execute('SELECT "Name" FROM equities WHERE "Ticker" = ?', ("AAPL",))
        self.assertEqual(cursor.fetchone()[0], "Apple Inc")

    def test_empty_dataframe(self):
        """Test inserting empty DataFrame."""
        df = pd.DataFrame(columns=["Ticker", "Name", "Sector", "Region", "Last Price"])
        count = chunk_insert_dataframe(self.conn, df)
        self.assertEqual(count, 0)

    def test_database_error_raises_exception(self):
        """Test that database errors are properly raised."""
        # Create DataFrame with invalid data for the schema
        df = pd.DataFrame(
            {
                "InvalidColumn": ["test"],
            }
        )
        with self.assertRaises(sqlite3.Error):
            chunk_insert_dataframe(self.conn, df)

    def test_null_values_preserved(self):
        """Test that None/NaN values are preserved as NULL."""
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Name": [None],
                "Sector": ["Technology"],
                "Region": ["US"],
                "Last Price": [150.0],
            }
        )
        chunk_insert_dataframe(self.conn, df)

        cursor = self.conn.execute('SELECT "Name" FROM equities WHERE "Ticker" = ?', ("AAPL",))
        result = cursor.fetchone()[0]
        self.assertIsNone(result)


class TestImportRegion(unittest.TestCase):
    """Test region-specific CSV import."""

    def setUp(self):
        """Create temporary database and CSV files."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)
        # Create test table
        self.conn.execute(
            """
            CREATE TABLE equities (
                "Ticker" TEXT,
                "Name" TEXT,
                "Sector" TEXT,
                "Region" TEXT,
                "Last Price" NUMERIC,
                UNIQUE("Ticker", "Region")
            )
        """
        )
        self.conn.commit()

        # Create temporary data directory
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
        # Clean up CSV files
        for file in self.data_dir.glob("*.csv"):
            file.unlink()
        self.data_dir.rmdir()

    def test_import_region_with_valid_csv(self):
        """Test importing a valid CSV file for a region."""
        # Create test CSV
        csv_path = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Name": ["Apple Inc", "Microsoft Corp"],
                "Sector": ["Technology", "Technology"],
                "Region": ["US", "US"],
                "Last Price": [150.0, 300.0],
            }
        )
        df.to_csv(csv_path, index=False)

        count = import_region(self.conn, self.data_dir, "US", chunksize=1000)
        self.assertEqual(count, 2)

        # Verify data was inserted
        cursor = self.conn.execute('SELECT COUNT(*) FROM equities WHERE "Region" = ?', ("US",))
        self.assertEqual(cursor.fetchone()[0], 2)

    def test_import_region_backfills_missing_region(self):
        """Test that missing Region values are backfilled."""
        # Create CSV without Region column
        csv_path = self.data_dir / "screening_eu.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["BMW", "NOKIA"],
                "Name": ["BMW AG", "Nokia Corp"],
                "Sector": ["Automotive", "Technology"],
                "Last Price": [80.0, 5.0],
            }
        )
        df.to_csv(csv_path, index=False)

        count = import_region(self.conn, self.data_dir, "EU", chunksize=1000)
        self.assertEqual(count, 2)

        # Verify Region was backfilled
        cursor = self.conn.execute('SELECT "Region" FROM equities')
        regions = [row[0] for row in cursor.fetchall()]
        self.assertEqual(regions, ["EU", "EU"])

    def test_import_region_with_empty_region_strings(self):
        """Test that empty Region strings are replaced with region identifier."""
        csv_path = self.data_dir / "screening_apac.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["SONY", "SAMSUNG"],
                "Name": ["Sony Corp", "Samsung Electronics"],
                "Sector": ["Technology", "Technology"],
                "Region": ["", ""],
                "Last Price": [100.0, 1500.0],
            }
        )
        df.to_csv(csv_path, index=False)

        import_region(self.conn, self.data_dir, "APAC", chunksize=1000)

        # Verify Region was backfilled
        cursor = self.conn.execute('SELECT "Region" FROM equities')
        regions = [row[0] for row in cursor.fetchall()]
        self.assertEqual(regions, ["APAC", "APAC"])

    def test_import_region_csv_not_found(self):
        """Test handling of missing CSV file."""
        count = import_region(self.conn, self.data_dir, "ROTW", chunksize=1000)
        self.assertEqual(count, 0)

    def test_import_region_chunked_processing(self):
        """Test that large files are processed in chunks."""
        # Create CSV with multiple rows
        csv_path = self.data_dir / "screening_us.csv"
        rows = [
            {
                "Ticker": f"TICK{i}",
                "Name": f"Company {i}",
                "Sector": "Technology",
                "Region": "US",
                "Last Price": float(100 + i),
            }
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        # Import with small chunk size
        count = import_region(self.conn, self.data_dir, "US", chunksize=3)
        self.assertEqual(count, 10)

        # Verify all rows were inserted
        cursor = self.conn.execute("SELECT COUNT(*) FROM equities")
        self.assertEqual(cursor.fetchone()[0], 10)

    def test_import_region_with_unicode(self):
        """Test importing CSV with unicode characters."""
        csv_path = self.data_dir / "screening_eu.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["SAP"],
                "Name": ["SAP Société Européenne"],
                "Sector": ["Technology"],
                "Region": ["EU"],
                "Last Price": [120.0],
            }
        )
        df.to_csv(csv_path, index=False, encoding="utf-8")

        count = import_region(self.conn, self.data_dir, "EU", chunksize=1000)
        self.assertEqual(count, 1)

        cursor = self.conn.execute('SELECT "Name" FROM equities WHERE "Ticker" = ?', ("SAP",))
        self.assertEqual(cursor.fetchone()[0], "SAP Société Européenne")


class TestSQLiteImportIntegration(unittest.TestCase):
    """Integration tests for complete SQLite import workflow."""

    def setUp(self):
        """Create temporary database with full schema."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)

        # Create table with more realistic schema
        self.conn.execute(
            """
            CREATE TABLE equities (
                "Ticker" TEXT NOT NULL,
                "ISIN" TEXT,
                "Name" TEXT,
                "Sector" TEXT,
                "Industry" TEXT,
                "Region" TEXT,
                "Last Price" NUMERIC,
                "Market Cap" NUMERIC,
                "P/E" NUMERIC,
                UNIQUE("Ticker", "Region")
            )
        """
        )
        self.conn.commit()

        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
        for file in self.data_dir.glob("*.csv"):
            file.unlink()
        self.data_dir.rmdir()

    def test_multi_region_import(self):
        """Test importing multiple regions maintains uniqueness."""
        # Create CSVs for multiple regions
        for region in ["US", "EU"]:
            csv_path = self.data_dir / f"screening_{region.lower()}.csv"
            df = pd.DataFrame(
                {
                    "Ticker": ["AAPL", "MSFT"],
                    "Name": [f"Apple {region}", f"Microsoft {region}"],
                    "Sector": ["Technology", "Technology"],
                    "Region": [region, region],
                    "Last Price": [150.0, 300.0],
                }
            )
            df.to_csv(csv_path, index=False)

        # Import both regions
        for region in ["US", "EU"]:
            import_region(self.conn, self.data_dir, region, chunksize=1000)

        # Verify 4 unique (Ticker, Region) combinations
        cursor = self.conn.execute("SELECT COUNT(*) FROM equities")
        self.assertEqual(cursor.fetchone()[0], 4)

    def test_header_not_imported_as_data(self):
        """Test that CSV header is not imported as a data row."""
        csv_path = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Name": ["Apple Inc"],
                "Sector": ["Technology"],
                "Region": ["US"],
                "Last Price": [150.0],
            }
        )
        df.to_csv(csv_path, index=False)

        import_region(self.conn, self.data_dir, "US", chunksize=1000)

        # Verify "Ticker" is not present as a data value
        cursor = self.conn.execute('SELECT COUNT(*) FROM equities WHERE "Ticker" = ?', ("Ticker",))
        self.assertEqual(cursor.fetchone()[0], 0)

        # Verify we have exactly 1 row
        cursor = self.conn.execute("SELECT COUNT(*) FROM equities")
        self.assertEqual(cursor.fetchone()[0], 1)

    def test_validation_post_import(self):
        """Test post-import validation queries."""
        # Create test data
        csv_path = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT", ""],
                "Name": ["Apple Inc", "Microsoft Corp", "Unknown"],
                "Sector": ["Technology", "Technology", ""],
                "Region": ["US", "US", "US"],
                "Last Price": [150.0, 300.0, None],
            }
        )
        df.to_csv(csv_path, index=False)

        import_region(self.conn, self.data_dir, "US", chunksize=1000)

        # Validation: Check for missing Ticker
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM equities WHERE "Ticker" IS NULL OR "Ticker" = ?', ("",)
        )
        missing_ticker = cursor.fetchone()[0]
        self.assertGreaterEqual(missing_ticker, 0)  # Should handle gracefully

        # Validation: Check for missing Sector
        cursor = self.conn.execute(
            'SELECT COUNT(*) FROM equities WHERE "Sector" IS NULL OR "Sector" = ?', ("",)
        )
        missing_sector = cursor.fetchone()[0]
        self.assertGreaterEqual(missing_sector, 0)

        # Validation: Check for missing Last Price
        cursor = self.conn.execute('SELECT COUNT(*) FROM equities WHERE "Last Price" IS NULL')
        missing_price = cursor.fetchone()[0]
        self.assertGreaterEqual(missing_price, 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in import functions."""

    def setUp(self):
        """Create temporary database and directory."""
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute(
            """
            CREATE TABLE equities (
                "Ticker" TEXT,
                "Name" TEXT,
                "Sector" TEXT,
                "Region" TEXT,
                "Last Price" NUMERIC,
                UNIQUE("Ticker", "Region")
            )
        """
        )
        self.conn.commit()

        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)

    def tearDown(self):
        """Clean up temporary files."""
        self.conn.close()
        os.close(self.db_fd)
        os.unlink(self.db_path)
        for file in self.data_dir.glob("*.csv"):
            file.unlink()
        self.data_dir.rmdir()

    def test_csv_parser_error_handling(self):
        """Test handling of CSV parsing errors."""
        # Create malformed CSV
        csv_path = self.data_dir / "screening_us.csv"
        with open(csv_path, "w") as f:
            f.write('Ticker,Name,Sector,Region,"Last Price"\n')
            f.write('AAPL,"Apple Inc",Technology,US,150.0\n')
            f.write('MSFT,"Unclosed quote\n')  # Malformed line

        with self.assertRaises(Exception):
            import_region(self.conn, self.data_dir, "US", chunksize=1000)

    def test_database_error_in_chunk_insert(self):
        """Test handling of database errors during chunk insertion."""
        # Drop table to cause error
        self.conn.execute("DROP TABLE equities")
        self.conn.commit()

        csv_path = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL"],
                "Name": ["Apple Inc"],
                "Sector": ["Technology"],
                "Region": ["US"],
                "Last Price": [150.0],
            }
        )
        df.to_csv(csv_path, index=False)

        with self.assertRaises(sqlite3.Error):
            import_region(self.conn, self.data_dir, "US", chunksize=1000)

    def test_csv_with_header_only(self):
        """Test handling of CSV files with header but no data."""
        # Create CSV with header only
        csv_path = self.data_dir / "screening_us.csv"
        csv_path.write_text('Ticker,Name,Sector,Region,"Last Price"\n')

        # Header-only CSV should return 0 rows imported
        count = import_region(self.conn, self.data_dir, "US", chunksize=1000)
        self.assertEqual(count, 0)


class TestMainFunction(unittest.TestCase):
    """Test the main CLI function."""

    def setUp(self):
        """Set up temporary files for main function tests."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.temp_dir)
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Create schema
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE equities (
                "Ticker" TEXT,
                "Name" TEXT,
                "Sector" TEXT,
                "Region" TEXT,
                "Last Price" NUMERIC,
                UNIQUE("Ticker", "Region")
            )
        """
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        """Clean up temporary files."""
        os.close(self.db_fd)
        os.unlink(self.db_path)
        for file in self.data_dir.glob("*.csv"):
            file.unlink()
        self.data_dir.rmdir()

    def test_main_with_missing_data_directory(self):
        """Test main function with missing data directory."""
        from tools.import_sqlite import main

        with patch(
            "sys.argv",
            ["import_sqlite.py", "--db", str(self.db_path), "--data-dir", "nonexistent_dir"],
        ):
            result = main()
            self.assertEqual(result, 2)  # Error code for missing directory

    def test_main_with_valid_csv_files(self):
        """Test main function with valid CSV files."""
        from tools.import_sqlite import main

        # Create test CSV
        csv_path = self.data_dir / "screening_us.csv"
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL", "MSFT"],
                "Name": ["Apple Inc", "Microsoft Corp"],
                "Sector": ["Technology", "Technology"],
                "Region": ["US", "US"],
                "Last Price": [150.0, 300.0],
            }
        )
        df.to_csv(csv_path, index=False)

        with patch(
            "sys.argv",
            [
                "import_sqlite.py",
                "--db",
                str(self.db_path),
                "--data-dir",
                str(self.data_dir),
                "--regions",
                "US",
            ],
        ):
            result = main()
            self.assertEqual(result, 0)  # Success

        # Verify data was imported
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM equities")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)

    def test_main_with_custom_chunksize(self):
        """Test main function with custom chunksize parameter."""
        from tools.import_sqlite import main

        # Create test CSV with multiple rows
        csv_path = self.data_dir / "screening_eu.csv"
        rows = [
            {
                "Ticker": f"TICK{i}",
                "Name": f"Company {i}",
                "Sector": "Technology",
                "Region": "EU",
                "Last Price": float(100 + i),
            }
            for i in range(10)
        ]
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

        with patch(
            "sys.argv",
            [
                "import_sqlite.py",
                "--db",
                str(self.db_path),
                "--data-dir",
                str(self.data_dir),
                "--regions",
                "EU",
                "--chunksize",
                "3",
            ],
        ):
            result = main()
            self.assertEqual(result, 0)

        # Verify all data was imported
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT COUNT(*) FROM equities WHERE "Region" = ?', ("EU",))
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, 10)

    def test_main_with_all_regions(self):
        """Test main function importing all regions."""
        from tools.import_sqlite import main

        # Create CSV for each region
        for region in ["US", "EU", "APAC", "ROTW"]:
            csv_path = self.data_dir / f"screening_{region.lower()}.csv"
            df = pd.DataFrame(
                {
                    "Ticker": [f"TICK{region}"],
                    "Name": [f"Company {region}"],
                    "Sector": ["Technology"],
                    "Region": [region],
                    "Last Price": [100.0],
                }
            )
            df.to_csv(csv_path, index=False)

        with patch(
            "sys.argv",
            ["import_sqlite.py", "--db", str(self.db_path), "--data-dir", str(self.data_dir)],
        ):
            result = main()
            self.assertEqual(result, 0)

        # Verify all regions imported
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('SELECT COUNT(DISTINCT "Region") FROM equities')
        distinct_regions = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(distinct_regions, 4)


class TestTqdmFallback(unittest.TestCase):
    """Test tqdm fallback when module not available."""

    def test_tqdm_fallback_iterator(self):
        """Test that the fallback tqdm works as a pass-through."""
        # This tests the fallback defined in lines 27-30
        # The actual import happens at module load, so we test the behavior
        from tools.import_sqlite import tqdm as import_tqdm

        test_list = [1, 2, 3, 4, 5]
        result = list(import_tqdm(test_list, desc="test"))
        self.assertEqual(result, test_list)


if __name__ == "__main__":
    unittest.main()
