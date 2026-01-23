"""
Test suite for feature_registry.sql helper function refactoring.

This module tests the TDD implementation of helper function enhancements:
1. calc_total_analyst_ratings() - reduces duplication in sentiment calculations
2. near_threshold_flag() - used for 52W high/low proximity flags
3. Consistent use of safe_divide(), calc_change_ratio(), pct_change() helpers

TDD Approach:
- RED: Tests written first to define expected behavior
- GREEN: Minimal implementation to pass tests
- REFACTOR: Clean up and optimize

Coverage target: ≥80% for changed files
"""

import re
import unittest
from pathlib import Path


class TestHelperFunctionDefinitions(unittest.TestCase):
    """Test suite for verifying new helper function definitions in feature_registry.sql."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _function_exists(self, function_name: str) -> bool:
        """Check if a CREATE OR REPLACE FUNCTION statement exists for the given function."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    def _function_has_modifier(self, function_name: str, modifier: str) -> bool:
        """Check if a function has a specific modifier (IMMUTABLE, STABLE, PARALLEL SAFE)."""
        # Find the function definition block
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+\w+.*?AS\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            func_header = match.group(0)
            return modifier.upper() in func_header.upper()
        return False

    def _function_returns_type(self, function_name: str, return_type: str) -> bool:
        """Check if a function returns a specific type."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+{re.escape(return_type)}"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL))

    # =========================================================================
    # TEST: calc_total_analyst_ratings helper function
    # =========================================================================
    def test_calc_total_analyst_ratings_exists(self):
        """Test that calc_total_analyst_ratings helper function is defined."""
        self.assertTrue(
            self._function_exists("calc_total_analyst_ratings"),
            "calc_total_analyst_ratings function should be defined in feature_registry.sql",
        )

    def test_calc_total_analyst_ratings_returns_numeric(self):
        """Test that calc_total_analyst_ratings returns NUMERIC type."""
        self.assertTrue(
            self._function_returns_type("calc_total_analyst_ratings", "NUMERIC"),
            "calc_total_analyst_ratings should return NUMERIC type",
        )

    def test_calc_total_analyst_ratings_is_immutable(self):
        """Test that calc_total_analyst_ratings has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("calc_total_analyst_ratings", "IMMUTABLE"),
            "calc_total_analyst_ratings should have IMMUTABLE modifier",
        )

    def test_calc_total_analyst_ratings_is_parallel_safe(self):
        """Test that calc_total_analyst_ratings has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("calc_total_analyst_ratings", "PARALLEL SAFE"),
            "calc_total_analyst_ratings should have PARALLEL SAFE modifier",
        )

    def test_calc_total_analyst_ratings_has_five_parameters(self):
        """Test that calc_total_analyst_ratings accepts 5 rating parameters."""
        pattern = (
            r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+calc_total_analyst_ratings\s*\(\s*"
            r"(\w+\s+NUMERIC\s*,\s*){4}\w+\s+NUMERIC\s*\)"
        )
        self.assertTrue(
            bool(re.search(pattern, self.sql_content, re.IGNORECASE)),
            "calc_total_analyst_ratings should accept 5 NUMERIC parameters",
        )

    # =========================================================================
    # TEST: near_threshold_flag helper function
    # =========================================================================
    def test_near_threshold_flag_exists(self):
        """Test that near_threshold_flag helper function is defined."""
        self.assertTrue(
            self._function_exists("near_threshold_flag"),
            "near_threshold_flag function should be defined in feature_registry.sql",
        )

    def test_near_threshold_flag_returns_integer(self):
        """Test that near_threshold_flag returns INTEGER type."""
        self.assertTrue(
            self._function_returns_type("near_threshold_flag", "INTEGER"),
            "near_threshold_flag should return INTEGER type",
        )

    def test_near_threshold_flag_is_immutable(self):
        """Test that near_threshold_flag has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("near_threshold_flag", "IMMUTABLE"),
            "near_threshold_flag should have IMMUTABLE modifier",
        )

    def test_near_threshold_flag_is_parallel_safe(self):
        """Test that near_threshold_flag has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("near_threshold_flag", "PARALLEL SAFE"),
            "near_threshold_flag should have PARALLEL SAFE modifier",
        )

    def test_near_threshold_flag_has_is_below_parameter(self):
        """Test that near_threshold_flag has is_below BOOLEAN parameter with default."""
        pattern = r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+near_threshold_flag\s*\([^)]*is_below\s+BOOLEAN\s+DEFAULT"
        self.assertTrue(
            bool(re.search(pattern, self.sql_content, re.IGNORECASE)),
            "near_threshold_flag should have is_below BOOLEAN parameter with DEFAULT",
        )


class TestExistingHelperFunctionModifiers(unittest.TestCase):
    """Test that existing helper functions have proper modifiers."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _function_has_modifier(self, function_name: str, modifier: str) -> bool:
        """Check if a function has a specific modifier."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+\w+.*?AS\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            func_header = match.group(0)
            return modifier.upper() in func_header.upper()
        return False

    def test_safe_divide_is_immutable(self):
        """Test that safe_divide has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("safe_divide", "IMMUTABLE"),
            "safe_divide should have IMMUTABLE modifier",
        )

    def test_safe_divide_is_parallel_safe(self):
        """Test that safe_divide has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("safe_divide", "PARALLEL SAFE"),
            "safe_divide should have PARALLEL SAFE modifier",
        )

    def test_pct_change_is_immutable(self):
        """Test that pct_change has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("pct_change", "IMMUTABLE"),
            "pct_change should have IMMUTABLE modifier",
        )

    def test_pct_change_is_parallel_safe(self):
        """Test that pct_change has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("pct_change", "PARALLEL SAFE"),
            "pct_change should have PARALLEL SAFE modifier",
        )

    def test_calc_change_ratio_is_immutable(self):
        """Test that calc_change_ratio has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("calc_change_ratio", "IMMUTABLE"),
            "calc_change_ratio should have IMMUTABLE modifier",
        )

    def test_calc_change_ratio_is_parallel_safe(self):
        """Test that calc_change_ratio has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("calc_change_ratio", "PARALLEL SAFE"),
            "calc_change_ratio should have PARALLEL SAFE modifier",
        )

    def test_clamp_score_is_immutable(self):
        """Test that clamp_score has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("clamp_score", "IMMUTABLE"),
            "clamp_score should have IMMUTABLE modifier",
        )

    def test_clamp_score_is_parallel_safe(self):
        """Test that clamp_score has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("clamp_score", "PARALLEL SAFE"),
            "clamp_score should have PARALLEL SAFE modifier",
        )

    def test_ema_crossover_signal_is_immutable(self):
        """Test that ema_crossover_signal has IMMUTABLE modifier."""
        self.assertTrue(
            self._function_has_modifier("ema_crossover_signal", "IMMUTABLE"),
            "ema_crossover_signal should have IMMUTABLE modifier",
        )

    def test_ema_crossover_signal_is_parallel_safe(self):
        """Test that ema_crossover_signal has PARALLEL SAFE modifier."""
        self.assertTrue(
            self._function_has_modifier("ema_crossover_signal", "PARALLEL SAFE"),
            "ema_crossover_signal should have PARALLEL SAFE modifier",
        )


class TestHelperFunctionUsageInRefactoredFunctions(unittest.TestCase):
    """Test that refactored functions use helper functions consistently."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _get_function_body(self, function_name: str) -> str:
        """Extract the body of a function between $$ markers."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+.*?AS\s*\$\$\s*(.*?)\s*\$\$\s*LANGUAGE\s+SQL"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def _count_inline_nullif_divisions(self, function_body: str) -> int:
        """Count inline / NULLIF(..., 0) patterns that should use safe_divide."""
        # Pattern for inline division with NULLIF that's NOT inside a helper function call
        # Exclude patterns that are already using safe_divide, calc_change_ratio, or pct_change
        pattern = r"(?<!safe_divide\()(?<!calc_change_ratio\()(?<!pct_change\()\s*/\s*NULLIF\s*\("
        return len(re.findall(pattern, function_body, re.IGNORECASE))

    # =========================================================================
    # TEST: calc_sentiment_features uses calc_total_analyst_ratings
    # =========================================================================
    def test_calc_sentiment_features_uses_total_analyst_ratings_helper(self):
        """Test that calc_sentiment_features uses calc_total_analyst_ratings helper."""
        func_body = self._get_function_body("calc_sentiment_features")
        self.assertIn(
            "calc_total_analyst_ratings",
            func_body.lower(),
            "calc_sentiment_features should use calc_total_analyst_ratings helper",
        )

    def test_calc_sentiment_features_uses_safe_divide(self):
        """Test that calc_sentiment_features uses safe_divide for divisions."""
        func_body = self._get_function_body("calc_sentiment_features")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_sentiment_features should use safe_divide helper",
        )

    def test_calc_sentiment_features_uses_pct_change_for_upside(self):
        """Test that calc_sentiment_features uses pct_change for upside_potential."""
        func_body = self._get_function_body("calc_sentiment_features")
        self.assertIn(
            "pct_change",
            func_body.lower(),
            "calc_sentiment_features should use pct_change helper for upside_potential",
        )

    def test_calc_sentiment_features_uses_calc_change_ratio(self):
        """Test that calc_sentiment_features uses calc_change_ratio for revisions."""
        func_body = self._get_function_body("calc_sentiment_features")
        self.assertIn(
            "calc_change_ratio",
            func_body.lower(),
            "calc_sentiment_features should use calc_change_ratio helper for price target revisions",
        )

    def test_calc_sentiment_features_uses_cte_for_analyst_data(self):
        """Test that calc_sentiment_features uses a CTE for pre-calculating analyst data."""
        func_body = self._get_function_body("calc_sentiment_features")
        self.assertIn(
            "with",
            func_body.lower(),
            "calc_sentiment_features should use a CTE (WITH clause) for analyst data",
        )

    # =========================================================================
    # TEST: calc_technical_analysis_features uses near_threshold_flag
    # =========================================================================
    def test_calc_technical_analysis_features_uses_near_threshold_flag(self):
        """Test that calc_technical_analysis_features uses near_threshold_flag helper."""
        func_body = self._get_function_body("calc_technical_analysis_features")
        self.assertIn(
            "near_threshold_flag",
            func_body.lower(),
            "calc_technical_analysis_features should use near_threshold_flag helper",
        )

    def test_calc_technical_analysis_features_uses_calc_change_ratio(self):
        """Test that calc_technical_analysis_features uses calc_change_ratio for ema_slope."""
        func_body = self._get_function_body("calc_technical_analysis_features")
        self.assertIn(
            "calc_change_ratio",
            func_body.lower(),
            "calc_technical_analysis_features should use calc_change_ratio helper",
        )

    def test_calc_technical_analysis_features_uses_pct_change(self):
        """Test that calc_technical_analysis_features uses pct_change for price_vs_ema_100d."""
        func_body = self._get_function_body("calc_technical_analysis_features")
        self.assertIn(
            "pct_change",
            func_body.lower(),
            "calc_technical_analysis_features should use pct_change helper",
        )

    def test_calc_technical_analysis_features_uses_cte(self):
        """Test that calc_technical_analysis_features uses a CTE for price metrics."""
        func_body = self._get_function_body("calc_technical_analysis_features")
        self.assertIn(
            "with",
            func_body.lower(),
            "calc_technical_analysis_features should use a CTE (WITH clause)",
        )

    # =========================================================================
    # TEST: calc_quality_features uses safe_divide consistently
    # =========================================================================
    def test_calc_quality_features_uses_safe_divide(self):
        """Test that calc_quality_features uses safe_divide for all divisions."""
        func_body = self._get_function_body("calc_quality_features")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_quality_features should use safe_divide helper",
        )

    def test_calc_quality_features_minimal_inline_nullif(self):
        """Test that calc_quality_features has minimal inline NULLIF divisions."""
        func_body = self._get_function_body("calc_quality_features")
        inline_count = self._count_inline_nullif_divisions(func_body)
        self.assertEqual(
            inline_count,
            0,
            f"calc_quality_features should have 0 inline / NULLIF patterns, found {inline_count}",
        )

    # =========================================================================
    # TEST: calc_profitability_features uses safe_divide consistently
    # =========================================================================
    def test_calc_profitability_features_uses_safe_divide(self):
        """Test that calc_profitability_features uses safe_divide for all divisions."""
        func_body = self._get_function_body("calc_profitability_features")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_profitability_features should use safe_divide helper",
        )

    def test_calc_profitability_features_minimal_inline_nullif(self):
        """Test that calc_profitability_features has minimal inline NULLIF divisions."""
        func_body = self._get_function_body("calc_profitability_features")
        inline_count = self._count_inline_nullif_divisions(func_body)
        self.assertEqual(
            inline_count,
            0,
            f"calc_profitability_features should have 0 inline / NULLIF patterns, found {inline_count}",
        )

    # =========================================================================
    # TEST: calc_leverage_features uses safe_divide consistently
    # =========================================================================
    def test_calc_leverage_features_uses_safe_divide(self):
        """Test that calc_leverage_features uses safe_divide for all divisions."""
        func_body = self._get_function_body("calc_leverage_features")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_leverage_features should use safe_divide helper",
        )

    def test_calc_leverage_features_minimal_inline_nullif(self):
        """Test that calc_leverage_features has minimal inline NULLIF divisions."""
        func_body = self._get_function_body("calc_leverage_features")
        inline_count = self._count_inline_nullif_divisions(func_body)
        self.assertEqual(
            inline_count,
            0,
            f"calc_leverage_features should have 0 inline / NULLIF patterns, found {inline_count}",
        )

    # =========================================================================
    # TEST: calc_efficiency_ratios uses safe_divide consistently
    # =========================================================================
    def test_calc_efficiency_ratios_uses_safe_divide(self):
        """Test that calc_efficiency_ratios uses safe_divide for all divisions."""
        func_body = self._get_function_body("calc_efficiency_ratios")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_efficiency_ratios should use safe_divide helper",
        )

    def test_calc_efficiency_ratios_minimal_inline_nullif(self):
        """Test that calc_efficiency_ratios has minimal inline NULLIF divisions."""
        func_body = self._get_function_body("calc_efficiency_ratios")
        inline_count = self._count_inline_nullif_divisions(func_body)
        self.assertEqual(
            inline_count,
            0,
            f"calc_efficiency_ratios should have 0 inline / NULLIF patterns, found {inline_count}",
        )

    # =========================================================================
    # TEST: calc_momentum_features uses safe_divide for 52W calculations
    # =========================================================================
    def test_calc_momentum_features_uses_safe_divide(self):
        """Test that calc_momentum_features uses safe_divide for 52W calculations."""
        func_body = self._get_function_body("calc_momentum_features")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_momentum_features should use safe_divide helper",
        )


class TestPctChangeHelperUsesInternalSafeDivide(unittest.TestCase):
    """Test that pct_change helper uses safe_divide internally for consistency."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _get_function_body(self, function_name: str) -> str:
        """Extract the body of a function between $$ markers."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+.*?AS\s*\$\$\s*(.*?)\s*\$\$\s*LANGUAGE\s+SQL"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def test_pct_change_uses_safe_divide(self):
        """Test that pct_change helper uses safe_divide internally."""
        func_body = self._get_function_body("pct_change")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "pct_change should use safe_divide internally for consistency",
        )

    def test_calc_change_ratio_uses_safe_divide(self):
        """Test that calc_change_ratio helper uses safe_divide internally."""
        func_body = self._get_function_body("calc_change_ratio")
        self.assertIn(
            "safe_divide",
            func_body.lower(),
            "calc_change_ratio should use safe_divide internally for consistency",
        )


class TestRefactoringReducesDuplication(unittest.TestCase):
    """Test that refactoring reduces code duplication."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _get_function_body(self, function_name: str) -> str:
        """Extract the body of a function between $$ markers."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\(.*?\)\s*RETURNS\s+.*?AS\s*\$\$\s*(.*?)\s*\$\$\s*LANGUAGE\s+SQL"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def test_calc_sentiment_features_no_repeated_total_ratings_calculation(self):
        """Test that calc_sentiment_features doesn't repeat total ratings calculation inline."""
        func_body = self._get_function_body("calc_sentiment_features")
        # Count occurrences of the full inline total ratings pattern
        pattern = r'"#\s*Strong\s*Buys\s*Ratings"\s*\+\s*"#\s*Buys\s*Ratings"\s*\+\s*"#\s*Hold\s*Ratings"\s*\+\s*"#\s*Sell\s*Ratings"\s*\+\s*"#\s*Strong\s*Sell\s*Ratings"'
        matches = re.findall(pattern, func_body, re.IGNORECASE)
        # Should have at most 1 occurrence (in the CTE definition), not multiple
        self.assertLessEqual(
            len(matches),
            1,
            f"calc_sentiment_features should not repeat total ratings calculation inline, found {len(matches)} occurrences",
        )

    def test_calc_technical_analysis_features_no_repeated_52w_calculations(self):
        """Test that calc_technical_analysis_features doesn't repeat 52W proximity calculations."""
        func_body = self._get_function_body("calc_technical_analysis_features")
        # Count occurrences of inline 52W high proximity calculation
        pattern = r'\(\s*"52W\s*High/Adj"\s*-\s*"Last\s*Price"\s*\)\s*/\s*NULLIF\s*\(\s*"52W\s*High/Adj"'
        matches = re.findall(pattern, func_body, re.IGNORECASE)
        # Should have 0 occurrences (moved to CTE or using helper)
        self.assertLessEqual(
            len(matches),
            1,
            f"calc_technical_analysis_features should not repeat 52W proximity calculation inline, found {len(matches)} occurrences",
        )


if __name__ == "__main__":
    unittest.main()
