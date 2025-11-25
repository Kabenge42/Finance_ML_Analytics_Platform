"""
Test module for column name normalization consistency.

TDD Implementation: Data Normalization Fixes (2025-11-24)
Phase 1.1 - Schema Normalization Tests

Purpose:
- Verify normalize_column_name() produces correct semantic replacements
- Validate all SQL schema columns normalize to COLUMN_SCHEMA keys
- Detect normalization mismatches between CSV loading and schema lookup

Test Coverage:
- Analyst rating columns with # prefix (5 columns)
- Percentage columns with % suffix
- Ampersand columns with &
- Special characters handling
- Round-trip normalization validation
- Obsolete column detection (short_int_pct)
"""

import unittest
from finance_ml.ml_workflow.data.schema import (
    normalize_column_name,
    COLUMN_SCHEMA,
    get_expected_dtype,
)


class TestNormalizeColumnName(unittest.TestCase):
    """Test the normalize_column_name() function for correctness."""

    def test_normalize_analyst_ratings_with_hash_strong_sell(self):
        """Test # Strong Sell Ratings normalizes to num_strong_sell_ratings."""
        input_col = "# Strong Sell Ratings"
        expected = "num_strong_sell_ratings"
        result = normalize_column_name(input_col)
        self.assertEqual(
            result,
            expected,
            f"Expected '{input_col}' → '{expected}', got '{result}'"
        )

    def test_normalize_analyst_ratings_with_hash_strong_buys(self):
        """Test # Strong Buys Ratings normalizes to num_strong_buys_ratings."""
        input_col = "# Strong Buys Ratings"
        expected = "num_strong_buys_ratings"
        result = normalize_column_name(input_col)
        self.assertEqual(
            result,
            expected,
            f"Expected '{input_col}' → '{expected}', got '{result}'"
        )

    def test_normalize_analyst_ratings_with_hash_hold(self):
        """Test # Hold Ratings normalizes to num_hold_ratings."""
        input_col = "# Hold Ratings"
        expected = "num_hold_ratings"
        result = normalize_column_name(input_col)
        self.assertEqual(
            result,
            expected,
            f"Expected '{input_col}' → '{expected}', got '{result}'"
        )

    def test_normalize_analyst_ratings_with_hash_buys(self):
        """Test # Buys Ratings normalizes to num_buys_ratings."""
        input_col = "# Buys Ratings"
        expected = "num_buys_ratings"
        result = normalize_column_name(input_col)
        self.assertEqual(
            result,
            expected,
            f"Expected '{input_col}' → '{expected}', got '{result}'"
        )

    def test_normalize_analyst_ratings_with_hash_sell(self):
        """Test # Sell Ratings normalizes to num_sell_ratings."""
        input_col = "# Sell Ratings"
        expected = "num_sell_ratings"
        result = normalize_column_name(input_col)
        self.assertEqual(
            result,
            expected,
            f"Expected '{input_col}' → '{expected}', got '{result}'"
        )

    def test_normalize_percentage_columns(self):
        """Test % suffix normalizes to _pct."""
        test_cases = [
            ("1-Day %", "1_day_pct"),
            ("Short Int. (%)", "short_int_pct"),
            ("Volatility %", "volatility_pct"),
        ]
        for input_col, expected in test_cases:
            with self.subTest(input_col=input_col):
                result = normalize_column_name(input_col)
                self.assertEqual(
                    result,
                    expected,
                    f"Expected '{input_col}' → '{expected}', got '{result}'"
                )

    def test_normalize_ampersand_columns(self):
        """Test & normalizes to _and_."""
        test_cases = [
            ("Selling General & Admin Expenses/Total (FQ)", "selling_general_and_admin_expenses_total_fq"),
            ("Cash And Equivalents (LTM)", "cash_and_equivalents_ltm"),
        ]
        for input_col, expected in test_cases:
            with self.subTest(input_col=input_col):
                result = normalize_column_name(input_col)
                self.assertEqual(
                    result,
                    expected,
                    f"Expected '{input_col}' → '{expected}', got '{result}'"
                )

    def test_normalize_spaces_to_underscores(self):
        """Test spaces normalize to single underscores."""
        input_col = "Market Cap"
        expected = "market_cap"
        result = normalize_column_name(input_col)
        self.assertEqual(result, expected)

    def test_normalize_multiple_special_chars(self):
        """Test multiple special characters are handled correctly."""
        input_col = "P/E (NTM)"
        expected = "p_e_ntm"
        result = normalize_column_name(input_col)
        self.assertEqual(result, expected)

    def test_normalize_strips_leading_trailing_underscores(self):
        """Test leading/trailing underscores are stripped."""
        test_cases = [
            ("_Leading", "leading"),
            ("Trailing_", "trailing"),
            ("_Both_", "both"),
        ]
        for input_col, expected in test_cases:
            with self.subTest(input_col=input_col):
                result = normalize_column_name(input_col)
                self.assertEqual(result, expected)

    def test_normalize_consecutive_underscores_collapsed(self):
        """Test consecutive underscores are collapsed to single underscore."""
        input_col = "Multiple___Underscores"
        expected = "multiple_underscores"
        result = normalize_column_name(input_col)
        self.assertEqual(result, expected)

    def test_normalize_case_insensitive(self):
        """Test normalization converts to lowercase."""
        test_cases = [
            ("UPPERCASE", "uppercase"),
            ("MixedCase", "mixedcase"),
            ("CamelCase", "camelcase"),
        ]
        for input_col, expected in test_cases:
            with self.subTest(input_col=input_col):
                result = normalize_column_name(input_col)
                self.assertEqual(result, expected)


class TestSchemaConsistency(unittest.TestCase):
    """Test COLUMN_SCHEMA consistency with SQL schema columns."""

    def test_analyst_rating_columns_in_schema(self):
        """Test all 5 analyst rating columns are in COLUMN_SCHEMA with correct keys."""
        sql_columns = [
            "# Strong Sell Ratings",
            "# Strong Buys Ratings",
            "# Hold Ratings",
            "# Buys Ratings",
            "# Sell Ratings",
        ]
        for sql_col in sql_columns:
            with self.subTest(sql_col=sql_col):
                normalized = normalize_column_name(sql_col)
                self.assertIn(
                    normalized,
                    COLUMN_SCHEMA,
                    f"Column '{sql_col}' normalized to '{normalized}' not found in COLUMN_SCHEMA"
                )

    def test_short_int_pct_not_in_schema(self):
        """Test short_int_pct is removed from COLUMN_SCHEMA (obsolete column)."""
        # This test will PASS after Task 1.2 is applied
        self.assertNotIn(
            "short_int_pct",
            COLUMN_SCHEMA,
            "Obsolete column 'short_int_pct' should be removed from COLUMN_SCHEMA"
        )

    def test_critical_columns_normalize_correctly(self):
        """Test critical Phase 9.3 columns normalize to schema keys."""
        critical_columns = [
            ("Ticker", "ticker"),
            ("Sector", "sector"),
            ("Region", "region"),
            ("Last Price", "last_price"),
            ("Price Target", "price_target"),
            ("Market Cap", "market_cap"),
            ("EBITDA (LTM)", "ebitda_ltm"),
            ("1-Day %", "1_day_pct"),
        ]
        for sql_col, expected_key in critical_columns:
            with self.subTest(sql_col=sql_col):
                normalized = normalize_column_name(sql_col)
                self.assertEqual(normalized, expected_key)
                self.assertIn(
                    normalized,
                    COLUMN_SCHEMA,
                    f"Critical column '{sql_col}' → '{normalized}' not in COLUMN_SCHEMA"
                )

    def test_all_schema_keys_are_normalized(self):
        """Test all COLUMN_SCHEMA keys are in normalized form (lowercase, underscores)."""
        for key in COLUMN_SCHEMA.keys():
            # Check lowercase
            self.assertEqual(key, key.lower(), f"Key '{key}' is not lowercase")
            # Check no special chars except underscores
            self.assertRegex(
                key,
                r'^[a-z0-9_]+$',
                f"Key '{key}' contains invalid characters (not alphanumeric or underscore)"
            )
            # Check no leading/trailing underscores
            self.assertEqual(key, key.strip('_'), f"Key '{key}' has leading/trailing underscores")


class TestRoundTripNormalization(unittest.TestCase):
    """Test round-trip normalization: SQL column → normalized → COLUMN_SCHEMA lookup."""

    # Sample of SQL schema columns that should normalize correctly
    SQL_COLUMNS = [
        "Ticker", "ISIN", "Name", "Sector", "Industry", "Region",
        "Market Cap", "Enterprise Value", "Last Price",
        "# Strong Sell Ratings", "# Strong Buys Ratings", "# Hold Ratings",
        "# Buys Ratings", "# Sell Ratings",
        "Total Revenues/CAGR (5Y FY)", "P/TBV (LTM)", "EBITDA (LTM)",
        "1-Day %", "Shrs Out",
        "Selling General & Admin Expenses/Total (FQ)",
        "Accounts Receivable/Total (FY)",
        "EV/Sales (LTM)", "EV/EBITDA (LTM)", "P/E (LTM)",
        "52W High/Adj", "52W Low/Adj",
        "EMA (20D)", "EMA (50D)", "EMA (100D)", "EMA (250D)",
    ]

    def test_sql_columns_round_trip_to_schema(self):
        """Test SQL columns normalize to keys present in COLUMN_SCHEMA."""
        missing = []
        for sql_col in self.SQL_COLUMNS:
            normalized = normalize_column_name(sql_col)
            if normalized not in COLUMN_SCHEMA:
                missing.append((sql_col, normalized))

        self.assertEqual(
            len(missing),
            0,
            f"The following SQL columns failed round-trip normalization:\n" +
            "\n".join([f"  '{sql}' → '{norm}' (not in COLUMN_SCHEMA)" for sql, norm in missing])
        )

    def test_normalized_keys_have_valid_dtypes(self):
        """Test normalized keys in COLUMN_SCHEMA have valid dtype entries."""
        for sql_col in self.SQL_COLUMNS:
            normalized = normalize_column_name(sql_col)
            if normalized in COLUMN_SCHEMA:
                with self.subTest(sql_col=sql_col):
                    dtype_info = get_expected_dtype(normalized)
                    self.assertIsNotNone(
                        dtype_info,
                        f"Column '{sql_col}' → '{normalized}' has no dtype info"
                    )


if __name__ == '__main__':
    unittest.main()
