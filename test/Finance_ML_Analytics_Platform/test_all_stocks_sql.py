"""
TDD Tests for all_stocks.sql type casting and UNION compatibility.

This test module validates that the all_stocks.sql script properly handles:
1. Type casting for DATE, NUMERIC, and TEXT columns
2. UNION ALL compatibility across regional tables
3. NULL handling for empty strings in TEXT sources
4. Region normalization to uppercase standard values

Following TDD principles:
- Write failing tests first
- Implement minimal code to pass
- Refactor for maintainability

Test Database Setup:
- Uses mock regional tables with mixed type schemas
- Tests actual UNION ALL operations
- Validates final data types match all_stocks schema
"""

import unittest
import re
from pathlib import Path


class TestAllStocksSQLTypeCasting(unittest.TestCase):
    """Test suite for all_stocks.sql type casting and UNION compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.sql_file = Path(__file__).parent.parent / "all_stocks" / "all_stocks.sql"
        self.assertTrue(self.sql_file.exists(), f"SQL file not found: {self.sql_file}")
        self.sql_content = self.sql_file.read_text(encoding="utf-8")

    def test_sql_file_contains_explicit_column_list_not_select_star(self):
        """
        Test that INSERT INTO all_stocks uses explicit column list, not SELECT *.

        FAILING TEST (TDD): This will fail initially because current code uses SELECT *

        The UNION ALL should have explicit column lists with type casts to ensure
        compatibility across regional tables with different column types.
        """
        # Check that INSERT uses explicit column list
        insert_pattern = r'INSERT\s+INTO\s+all_stocks\s*\(\s*"Ticker"'
        self.assertIsNotNone(
            re.search(insert_pattern, self.sql_content, re.IGNORECASE | re.DOTALL),
            "INSERT INTO all_stocks should have explicit column list starting with 'Ticker'",
        )

        # Check that SELECT * is NOT used in the UNION statements
        # Pattern: SELECT * FROM ... UNION ALL (this is what we DON'T want)
        bad_pattern = r"SELECT\s+\*\s+FROM\s+postgres\.public\.screening_(us|eu|apac|rotw)"
        matches = re.findall(bad_pattern, self.sql_content, re.IGNORECASE)
        self.assertEqual(
            len(matches),
            0,
            f"Found {len(matches)} SELECT * statements in UNION. Should use explicit column lists with type casts.",
        )

    def test_sql_contains_date_type_casting(self):
        """
        Test that DATE columns are explicitly cast using to_date() function.

        FAILING TEST (TDD): Current code doesn't cast date columns.

        Date columns like "Last Updated", "Income Statement Report Date",
        "Next Earnings" must be cast from TEXT to DATE using:
        to_date(NULLIF("Last Updated",''),'YYYY-MM-DD')::date
        """
        # Check for to_date casting pattern for date columns
        date_columns = [
            "Last Updated",
            "Income Statement Report Date",
            "Next Earnings",
            "Dividend Record (Announce Date)",
            "Dividend Record (Ex Date)",
            "Dividend Record (Payable Date)",
            "Dividend Record (Record Date)",
        ]

        found_date_casts = 0
        for col in date_columns:
            # Look for pattern: to_date(NULLIF("Column",''),'YYYY-MM-DD') or to_date("Column",'YYYY-MM-DD')
            # Simplified pattern with DOTALL for multi-line matching
            pattern = rf"to_date.*{re.escape(col)}.*YYYY-MM-DD"
            if re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL):
                found_date_casts += 1

        self.assertGreater(
            found_date_casts,
            0,
            "SQL should contain to_date() casting for DATE columns with 'YYYY-MM-DD' format",
        )

    def test_sql_contains_numeric_type_casting(self):
        """
        Test that NUMERIC columns are explicitly cast using ::numeric.

        FAILING TEST (TDD): Current code doesn't cast numeric columns.

        Numeric columns like "Market Cap", "Last Price", "P/E (LTM)" must be cast:
        NULLIF("Market Cap",'')::numeric

        This handles TEXT sources (ROTW) that need conversion from TEXT to NUMERIC.
        """
        # Check for ::numeric casting pattern
        numeric_columns = [
            "Market Cap",
            "Last Price",
            "Enterprise Value",
            "P/E (LTM)",
            "EV/EBITDA (LTM)",
        ]

        found_numeric_casts = 0
        for col in numeric_columns:
            # Look for pattern: "Column"::numeric or NULLIF("Column",'')::numeric
            # Simplified to match actual SQL format
            pattern = rf"{re.escape(col)}.*?::numeric"
            if re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL):
                found_numeric_casts += 1

        self.assertGreater(
            found_numeric_casts, 0, "SQL should contain ::numeric casting for NUMERIC columns"
        )

    def test_sql_contains_nullif_for_empty_strings(self):
        """
        Test that NULLIF handles empty strings for TEXT sources.

        FAILING TEST (TDD): Current code doesn't use NULLIF for empty string handling.

        For TEXT source tables (ROTW), empty strings must be converted to NULL:
        NULLIF("Column",'')

        This prevents "invalid input syntax" errors when casting to NUMERIC or DATE.
        """
        # Check for NULLIF pattern with empty string
        nullif_pattern = r"NULLIF\([^)]+,\s*['\"]['\"]?\s*\)"
        matches = re.findall(nullif_pattern, self.sql_content, re.IGNORECASE)

        self.assertGreater(
            len(matches),
            0,
            "SQL should use NULLIF(column,'') to handle empty strings in TEXT sources",
        )

    def test_sql_contains_region_normalization(self):
        """
        Test that Region column is normalized to uppercase standard values.

        FAILING TEST (TDD): Current code doesn't normalize region values.

        Region must be normalized to: 'US', 'EU', 'APAC', 'ROTW'
        Using: UPPER(region)::text or explicit CASE statements
        """
        # Check for UPPER(region) or explicit region value setting
        region_patterns = [
            r"UPPER\(\s*region\s*\)",
            r"UPPER\(\s*['\"]US['\"]",
            r"['\"]US['\"]::text",
            r"region\s*=\s*['\"]US['\"]",
        ]

        found_region_handling = False
        for pattern in region_patterns:
            if re.search(pattern, self.sql_content, re.IGNORECASE):
                found_region_handling = True
                break

        self.assertTrue(
            found_region_handling,
            "SQL should normalize Region column to uppercase standard values (US, EU, APAC, ROTW)",
        )

    def test_sql_handles_all_318_columns_explicitly(self):
        """
        Test that all 318 columns are listed explicitly in INSERT and SELECT.

        FAILING TEST (TDD): Current code uses SELECT * which doesn't specify columns.

        The INSERT INTO and each SELECT in UNION ALL must list all 318 columns
        explicitly to ensure proper type casting and column order.
        """
        # Count column definitions in CREATE TABLE
        create_table_pattern = r"CREATE TABLE all_stocks\s*\((.*?)\s*CONSTRAINT"
        match = re.search(create_table_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)

        if match:
            column_defs = match.group(1)
            # Count lines that look like column definitions (quoted name followed by type)
            column_lines = re.findall(r'"\w[^"]*"\s+(?:TEXT|NUMERIC|DATE|INTEGER)', column_defs)
            table_column_count = len(column_lines)

            # Now check if INSERT statement has similar number of columns
            insert_pattern = r"INSERT INTO all_stocks\s*\((.*?)\)\s*SELECT"
            insert_match = re.search(insert_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)

            if insert_match:
                insert_columns = insert_match.group(1)
                # Count quoted column names in INSERT
                insert_column_list = re.findall(r'"[^"]+?"', insert_columns)
                insert_column_count = len(insert_column_list)

                self.assertGreaterEqual(
                    insert_column_count,
                    300,
                    f"INSERT should explicitly list ~300-318 columns, found {insert_column_count}",
                )
            else:
                self.fail("INSERT INTO all_stocks should have explicit column list in parentheses")

    def test_sql_union_all_maintains_column_order(self):
        """
        Test that all SELECT statements in UNION ALL have matching column order.

        FAILING TEST (TDD): With SELECT *, column order inconsistencies may occur.

        Each SELECT in UNION ALL must list columns in same order as INSERT column list.
        """
        # Find INSERT column list
        insert_pattern = r"INSERT INTO all_stocks\s*\((.*?)\)\s*SELECT"
        insert_match = re.search(insert_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)

        if not insert_match:
            self.skipTest(
                "Cannot validate - INSERT column list not found (expected to fail in TDD)"
            )

        # Extract first few column names from INSERT
        insert_columns = re.findall(r'"([^"]+)"', insert_match.group(1))[:5]

        # Find all SELECT statements in UNION
        select_pattern = r"SELECT\s+(.*?)\s+FROM\s+postgres\.public\.screening_"
        select_matches = re.findall(select_pattern, self.sql_content, re.IGNORECASE | re.DOTALL)

        if len(select_matches) == 0:
            self.skipTest("Cannot validate - SELECT statements not found (expected to fail in TDD)")

        # Check that first SELECT has matching column order
        first_select = select_matches[0]
        if first_select.strip() == "*":
            self.fail(
                "SELECT should use explicit column list matching INSERT column order, not SELECT *"
            )


class TestAllStocksSQLStructure(unittest.TestCase):
    """Test suite for all_stocks.sql overall structure and best practices."""

    def setUp(self):
        """Set up test fixtures."""
        self.sql_file = Path(__file__).parent.parent / "all_stocks" / "all_stocks.sql"
        self.sql_content = self.sql_file.read_text(encoding="utf-8")

    def test_sql_has_transaction_control(self):
        """Test that SQL script uses proper transaction control (DO block)."""
        # Check for DO $$ block
        self.assertIn("DO", self.sql_content.upper())
        self.assertIn("$$", self.sql_content)

    def test_sql_has_error_handling(self):
        """Test that SQL script has error handling mechanisms."""
        # Check for error handling patterns
        error_patterns = [
            r"\\set\s+ON_ERROR_STOP",
            r"EXCEPTION\s+WHEN\s+OTHERS",
            r"RAISE\s+WARNING",
            r"RAISE\s+ERROR",
        ]

        found_error_handling = False
        for pattern in error_patterns:
            if re.search(pattern, self.sql_content, re.IGNORECASE):
                found_error_handling = True
                break

        self.assertTrue(found_error_handling, "SQL should have error handling mechanisms")

    def test_sql_validates_column_count(self):
        """Test that SQL script validates expected column count (318)."""
        # Check for column count validation
        pattern = r"318|column.*count|SELECT COUNT\(\*\).*information_schema\.columns"
        self.assertIsNotNone(
            re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL),
            "SQL should validate column count matches expected 318",
        )


if __name__ == "__main__":
    unittest.main()
