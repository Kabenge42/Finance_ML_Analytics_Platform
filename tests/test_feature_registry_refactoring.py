"""
Test suite for feature_registry.sql helper functions refactoring.

This module tests the new helper functions and refactored main functions
as specified in the FEATURE_REGISTRY_REFACTORINGS issue.

TDD Approach:
- RED: Tests written first to define expected behavior
- GREEN: Minimal implementation to pass tests
- REFACTOR: Clean up and optimize

Coverage target: ≥80% for changed files

Key Improvements Being Tested:
1. pct_change helper function
2. calc_cagr helper function
3. calc_adjustment_pct helper function
4. positive_flag, improvement_flag, nonzero_flag helper functions
5. calc_liquidity_stress helper function
6. Refactored calc_quality_features using nonzero_flag
7. Refactored calc_financial_distress_features using calc_liquidity_stress
8. Refactored calc_eps_trajectory_features using new helpers
9. Refactored calc_gaap_adjusted_analytics using calc_adjustment_pct
10. Refactored calc_accounting_quality_features using helper functions
"""

import re
import unittest
from pathlib import Path


class TestHelperFunctionsExist(unittest.TestCase):
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

    def _function_is_immutable(self, function_name: str) -> bool:
        """Check if a function is marked as IMMUTABLE."""
        # Find the function definition and check for IMMUTABLE
        func_start_pattern = (
            rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        )
        match = re.search(func_start_pattern, self.sql_content, re.IGNORECASE)
        if match:
            # Look for IMMUTABLE between function declaration and $$ LANGUAGE
            func_section = self.sql_content[match.start() : match.start() + 500]
            return bool(re.search(r"\bIMMUTABLE\b", func_section, re.IGNORECASE))
        return False

    def _function_is_parallel_safe(self, function_name: str) -> bool:
        """Check if a function is marked as PARALLEL SAFE."""
        func_start_pattern = (
            rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        )
        match = re.search(func_start_pattern, self.sql_content, re.IGNORECASE)
        if match:
            func_section = self.sql_content[match.start() : match.start() + 500]
            return bool(re.search(r"PARALLEL\s+SAFE", func_section, re.IGNORECASE))
        return False

    def _function_returns_type(self, function_name: str, return_type: str) -> bool:
        """Check if a function returns a specific type."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\([^)]*\)\s*RETURNS\s+{re.escape(return_type)}"
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    # =========================================================================
    # TEST: pct_change helper function
    # =========================================================================
    def test_pct_change_function_exists(self):
        """Test that pct_change function is defined."""
        self.assertTrue(
            self._function_exists("pct_change"),
            "pct_change function should be defined in feature_registry.sql",
        )

    def test_pct_change_returns_numeric(self):
        """Test that pct_change returns NUMERIC type."""
        self.assertTrue(
            self._function_returns_type("pct_change", "NUMERIC"),
            "pct_change should return NUMERIC type",
        )

    def test_pct_change_is_immutable(self):
        """Test that pct_change is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("pct_change"),
            "pct_change should be IMMUTABLE",
        )

    def test_pct_change_is_parallel_safe(self):
        """Test that pct_change is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("pct_change"),
            "pct_change should be PARALLEL SAFE",
        )

    def test_pct_change_uses_calc_change_ratio(self):
        """Test that pct_change uses calc_change_ratio internally."""
        # Find pct_change function body
        pattern = r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+pct_change\s*\([^)]*\).*?\$\$\s*(.*?)\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(match, "pct_change function body should be found")
        func_body = match.group(1)
        self.assertIn(
            "calc_change_ratio",
            func_body.lower(),
            "pct_change should use calc_change_ratio internally",
        )

    # =========================================================================
    # TEST: calc_cagr helper function
    # =========================================================================
    def test_calc_cagr_function_exists(self):
        """Test that calc_cagr function is defined."""
        self.assertTrue(
            self._function_exists("calc_cagr"),
            "calc_cagr function should be defined in feature_registry.sql",
        )

    def test_calc_cagr_returns_numeric(self):
        """Test that calc_cagr returns NUMERIC type."""
        self.assertTrue(
            self._function_returns_type("calc_cagr", "NUMERIC"),
            "calc_cagr should return NUMERIC type",
        )

    def test_calc_cagr_is_immutable(self):
        """Test that calc_cagr is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("calc_cagr"),
            "calc_cagr should be IMMUTABLE",
        )

    def test_calc_cagr_is_parallel_safe(self):
        """Test that calc_cagr is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("calc_cagr"),
            "calc_cagr should be PARALLEL SAFE",
        )

    def test_calc_cagr_uses_power_function(self):
        """Test that calc_cagr uses POWER function for compound calculation."""
        pattern = r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+calc_cagr\s*\([^)]*\).*?\$\$\s*(.*?)\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(match, "calc_cagr function body should be found")
        func_body = match.group(1)
        self.assertIn(
            "power",
            func_body.lower(),
            "calc_cagr should use POWER function for CAGR calculation",
        )

    # =========================================================================
    # TEST: calc_adjustment_pct helper function
    # =========================================================================
    def test_calc_adjustment_pct_function_exists(self):
        """Test that calc_adjustment_pct function is defined."""
        self.assertTrue(
            self._function_exists("calc_adjustment_pct"),
            "calc_adjustment_pct function should be defined in feature_registry.sql",
        )

    def test_calc_adjustment_pct_returns_numeric(self):
        """Test that calc_adjustment_pct returns NUMERIC type."""
        self.assertTrue(
            self._function_returns_type("calc_adjustment_pct", "NUMERIC"),
            "calc_adjustment_pct should return NUMERIC type",
        )

    def test_calc_adjustment_pct_is_immutable(self):
        """Test that calc_adjustment_pct is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("calc_adjustment_pct"),
            "calc_adjustment_pct should be IMMUTABLE",
        )

    def test_calc_adjustment_pct_is_parallel_safe(self):
        """Test that calc_adjustment_pct is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("calc_adjustment_pct"),
            "calc_adjustment_pct should be PARALLEL SAFE",
        )

    # =========================================================================
    # TEST: positive_flag helper function
    # =========================================================================
    def test_positive_flag_function_exists(self):
        """Test that positive_flag function is defined."""
        self.assertTrue(
            self._function_exists("positive_flag"),
            "positive_flag function should be defined in feature_registry.sql",
        )

    def test_positive_flag_returns_integer(self):
        """Test that positive_flag returns INTEGER type."""
        self.assertTrue(
            self._function_returns_type("positive_flag", "INTEGER"),
            "positive_flag should return INTEGER type",
        )

    def test_positive_flag_is_immutable(self):
        """Test that positive_flag is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("positive_flag"),
            "positive_flag should be IMMUTABLE",
        )

    def test_positive_flag_is_parallel_safe(self):
        """Test that positive_flag is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("positive_flag"),
            "positive_flag should be PARALLEL SAFE",
        )

    # =========================================================================
    # TEST: improvement_flag helper function
    # =========================================================================
    def test_improvement_flag_function_exists(self):
        """Test that improvement_flag function is defined."""
        self.assertTrue(
            self._function_exists("improvement_flag"),
            "improvement_flag function should be defined in feature_registry.sql",
        )

    def test_improvement_flag_returns_integer(self):
        """Test that improvement_flag returns INTEGER type."""
        self.assertTrue(
            self._function_returns_type("improvement_flag", "INTEGER"),
            "improvement_flag should return INTEGER type",
        )

    def test_improvement_flag_is_immutable(self):
        """Test that improvement_flag is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("improvement_flag"),
            "improvement_flag should be IMMUTABLE",
        )

    def test_improvement_flag_is_parallel_safe(self):
        """Test that improvement_flag is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("improvement_flag"),
            "improvement_flag should be PARALLEL SAFE",
        )

    # =========================================================================
    # TEST: nonzero_flag helper function
    # =========================================================================
    def test_nonzero_flag_function_exists(self):
        """Test that nonzero_flag function is defined."""
        self.assertTrue(
            self._function_exists("nonzero_flag"),
            "nonzero_flag function should be defined in feature_registry.sql",
        )

    def test_nonzero_flag_returns_integer(self):
        """Test that nonzero_flag returns INTEGER type."""
        self.assertTrue(
            self._function_returns_type("nonzero_flag", "INTEGER"),
            "nonzero_flag should return INTEGER type",
        )

    def test_nonzero_flag_is_immutable(self):
        """Test that nonzero_flag is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("nonzero_flag"),
            "nonzero_flag should be IMMUTABLE",
        )

    def test_nonzero_flag_is_parallel_safe(self):
        """Test that nonzero_flag is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("nonzero_flag"),
            "nonzero_flag should be PARALLEL SAFE",
        )

    # =========================================================================
    # TEST: calc_liquidity_stress helper function
    # =========================================================================
    def test_calc_liquidity_stress_function_exists(self):
        """Test that calc_liquidity_stress function is defined."""
        self.assertTrue(
            self._function_exists("calc_liquidity_stress"),
            "calc_liquidity_stress function should be defined in feature_registry.sql",
        )

    def test_calc_liquidity_stress_returns_numeric(self):
        """Test that calc_liquidity_stress returns NUMERIC type."""
        self.assertTrue(
            self._function_returns_type("calc_liquidity_stress", "NUMERIC"),
            "calc_liquidity_stress should return NUMERIC type",
        )

    def test_calc_liquidity_stress_is_immutable(self):
        """Test that calc_liquidity_stress is marked as IMMUTABLE."""
        self.assertTrue(
            self._function_is_immutable("calc_liquidity_stress"),
            "calc_liquidity_stress should be IMMUTABLE",
        )

    def test_calc_liquidity_stress_is_parallel_safe(self):
        """Test that calc_liquidity_stress is marked as PARALLEL SAFE."""
        self.assertTrue(
            self._function_is_parallel_safe("calc_liquidity_stress"),
            "calc_liquidity_stress should be PARALLEL SAFE",
        )


class TestRefactoredFunctionsUseHelpers(unittest.TestCase):
    """Test suite for verifying refactored functions use the new helper functions."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _get_function_body(self, function_name: str) -> str:
        """Extract the body of a SQL function."""
        # Pattern to match function definition with optional parameters
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\([^)]*\).*?\$\$\s*(.*?)\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return ""

    # =========================================================================
    # TEST: calc_quality_features uses nonzero_flag
    # =========================================================================
    def test_calc_quality_features_uses_nonzero_flag(self):
        """Test that calc_quality_features uses nonzero_flag helper."""
        func_body = self._get_function_body("calc_quality_features")
        self.assertIn(
            "nonzero_flag",
            func_body.lower(),
            "calc_quality_features should use nonzero_flag helper",
        )

    def test_calc_quality_features_no_inline_nonzero_check(self):
        """Test that calc_quality_features doesn't use inline CASE WHEN <> 0 pattern."""
        func_body = self._get_function_body("calc_quality_features")
        # Should not have inline pattern like: CASE WHEN "Something" <> 0 THEN 1 ELSE 0 END
        inline_pattern = (
            r"CASE\s+WHEN\s+\"[^\"]+\"\s*<>\s*0\s+THEN\s+1\s+ELSE\s+0\s+END"
        )
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        self.assertEqual(
            len(matches),
            0,
            f"calc_quality_features should not have inline nonzero CASE patterns, found: {matches}",
        )

    # =========================================================================
    # TEST: calc_financial_distress_features uses calc_liquidity_stress
    # =========================================================================
    def test_calc_financial_distress_features_uses_calc_liquidity_stress(self):
        """Test that calc_financial_distress_features uses calc_liquidity_stress helper."""
        func_body = self._get_function_body("calc_financial_distress_features")
        self.assertIn(
            "calc_liquidity_stress",
            func_body.lower(),
            "calc_financial_distress_features should use calc_liquidity_stress helper",
        )

    def test_calc_financial_distress_features_uses_cte(self):
        """Test that calc_financial_distress_features uses CTE for consolidation."""
        func_body = self._get_function_body("calc_financial_distress_features")
        self.assertIn(
            "with",
            func_body.lower(),
            "calc_financial_distress_features should use CTE (WITH clause)",
        )

    def test_calc_financial_distress_features_no_duplicated_liquidity_logic(self):
        """Test that liquidity stress logic is not duplicated inline."""
        func_body = self._get_function_body("calc_financial_distress_features")
        # Count occurrences of the inline liquidity pattern
        inline_pattern = r"WHEN\s+\"Current\s+Ratio\s+\(LTM\)\"\s*<\s*1\.0\s+THEN\s+30"
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        # Should appear at most once (if at all) since it's in the helper now
        self.assertLessEqual(
            len(matches),
            1,
            "Liquidity stress logic should not be duplicated - use calc_liquidity_stress helper",
        )

    # =========================================================================
    # TEST: calc_eps_trajectory_features uses pct_change, calc_cagr, positive_flag, improvement_flag
    # =========================================================================
    def test_calc_eps_trajectory_features_uses_pct_change(self):
        """Test that calc_eps_trajectory_features uses pct_change helper."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        self.assertIn(
            "pct_change",
            func_body.lower(),
            "calc_eps_trajectory_features should use pct_change helper",
        )

    def test_calc_eps_trajectory_features_uses_calc_cagr(self):
        """Test that calc_eps_trajectory_features uses calc_cagr helper."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        self.assertIn(
            "calc_cagr",
            func_body.lower(),
            "calc_eps_trajectory_features should use calc_cagr helper",
        )

    def test_calc_eps_trajectory_features_uses_positive_flag(self):
        """Test that calc_eps_trajectory_features uses positive_flag helper."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        self.assertIn(
            "positive_flag",
            func_body.lower(),
            "calc_eps_trajectory_features should use positive_flag helper",
        )

    def test_calc_eps_trajectory_features_uses_improvement_flag(self):
        """Test that calc_eps_trajectory_features uses improvement_flag helper."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        self.assertIn(
            "improvement_flag",
            func_body.lower(),
            "calc_eps_trajectory_features should use improvement_flag helper",
        )

    def test_calc_eps_trajectory_features_no_inline_cagr_pattern(self):
        """Test that calc_eps_trajectory_features doesn't use inline CAGR calculation."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        # CAGR pattern: POWER(..., 1.0 / N) - 1
        inline_pattern = r"POWER\s*\([^)]+,\s*1\.0\s*/\s*\d+\.?\d*\s*\)\s*-\s*1"
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        self.assertEqual(
            len(matches),
            0,
            f"calc_eps_trajectory_features should not have inline CAGR patterns, found: {matches}",
        )

    def test_calc_eps_trajectory_features_no_inline_positive_check(self):
        """Test that calc_eps_trajectory_features doesn't use inline positive CASE pattern."""
        func_body = self._get_function_body("calc_eps_trajectory_features")
        # Should not have inline pattern like: CASE WHEN "Something" > 0 THEN 1 ELSE 0 END
        inline_pattern = r"CASE\s+WHEN\s+\"[^\"]+\"\s*>\s*0\s+THEN\s+1\s+ELSE\s+0\s+END"
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        self.assertEqual(
            len(matches),
            0,
            f"calc_eps_trajectory_features should not have inline positive CASE patterns, found: {matches}",
        )

    # =========================================================================
    # TEST: calc_gaap_adjusted_analytics uses calc_adjustment_pct
    # =========================================================================
    def test_calc_gaap_adjusted_analytics_uses_calc_adjustment_pct(self):
        """Test that calc_gaap_adjusted_analytics uses calc_adjustment_pct helper."""
        func_body = self._get_function_body("calc_gaap_adjusted_analytics")
        self.assertIn(
            "calc_adjustment_pct",
            func_body.lower(),
            "calc_gaap_adjusted_analytics should use calc_adjustment_pct helper",
        )

    def test_calc_gaap_adjusted_analytics_reduced_inline_adjustment_patterns(self):
        """Test that inline adjustment percentage patterns are significantly reduced."""
        func_body = self._get_function_body("calc_gaap_adjusted_analytics")
        # Pattern: (X - Y) / NULLIF(ABS(Y), 0) * 100
        inline_pattern = r"\([^)]+\s*-\s*[^)]+\)\s*/\s*NULLIF\s*\(\s*ABS\s*\([^)]+\)\s*,\s*0\s*\)\s*\*\s*100"
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        # After refactoring, we should have very few (or zero) of these patterns
        # Previously there were 20+ such patterns
        self.assertLessEqual(
            len(matches),
            5,
            f"calc_gaap_adjusted_analytics should have reduced inline adjustment patterns, found: {len(matches)}",
        )

    # =========================================================================
    # TEST: calc_accounting_quality_features uses helper functions
    # =========================================================================
    def test_calc_accounting_quality_features_uses_nonzero_flag(self):
        """Test that calc_accounting_quality_features uses nonzero_flag helper."""
        func_body = self._get_function_body("calc_accounting_quality_features")
        self.assertIn(
            "nonzero_flag",
            func_body.lower(),
            "calc_accounting_quality_features should use nonzero_flag helper",
        )

    def test_calc_accounting_quality_features_uses_positive_flag(self):
        """Test that calc_accounting_quality_features uses positive_flag helper."""
        func_body = self._get_function_body("calc_accounting_quality_features")
        self.assertIn(
            "positive_flag",
            func_body.lower(),
            "calc_accounting_quality_features should use positive_flag helper",
        )

    def test_calc_accounting_quality_features_uses_calc_change_ratio(self):
        """Test that calc_accounting_quality_features uses calc_change_ratio helper."""
        func_body = self._get_function_body("calc_accounting_quality_features")
        self.assertIn(
            "calc_change_ratio",
            func_body.lower(),
            "calc_accounting_quality_features should use calc_change_ratio helper",
        )

    def test_calc_accounting_quality_features_no_inline_nonzero_check(self):
        """Test that calc_accounting_quality_features doesn't use inline nonzero CASE pattern."""
        func_body = self._get_function_body("calc_accounting_quality_features")
        # Should not have inline pattern like: CASE WHEN "Something" <> 0 THEN N ELSE 0 END
        inline_pattern = (
            r"CASE\s+WHEN\s+\"[^\"]+\"\s*<>\s*0\s+THEN\s+\d+\s+ELSE\s+0\s+END"
        )
        matches = re.findall(inline_pattern, func_body, re.IGNORECASE)
        self.assertEqual(
            len(matches),
            0,
            f"calc_accounting_quality_features should not have inline nonzero CASE patterns, found: {matches}",
        )


class TestHelperFunctionsBehavior(unittest.TestCase):
    """Test suite for verifying helper function behavior through their implementations."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _get_function_body(self, function_name: str) -> str:
        """Extract the body of a SQL function."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\([^)]*\).*?\$\$\s*(.*?)\s*\$\$"
        match = re.search(pattern, self.sql_content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return ""

    def test_pct_change_multiplies_by_100(self):
        """Test that pct_change multiplies by 100 to get percentage."""
        func_body = self._get_function_body("pct_change")
        self.assertIn(
            "100", func_body, "pct_change should multiply by 100 to return percentage"
        )

    def test_calc_cagr_handles_zero_and_negative(self):
        """Test that calc_cagr handles zero/negative values with CASE/WHEN."""
        func_body = self._get_function_body("calc_cagr")
        # Should have some form of protection for invalid inputs
        self.assertTrue(
            "case" in func_body.lower() or "nullif" in func_body.lower(),
            "calc_cagr should handle zero/negative values",
        )

    def test_calc_adjustment_pct_uses_abs_in_denominator(self):
        """Test that calc_adjustment_pct uses ABS in denominator."""
        func_body = self._get_function_body("calc_adjustment_pct")
        self.assertIn(
            "abs",
            func_body.lower(),
            "calc_adjustment_pct should use ABS in denominator",
        )

    def test_positive_flag_returns_1_or_0(self):
        """Test that positive_flag returns 1 or 0."""
        func_body = self._get_function_body("positive_flag")
        self.assertIn(
            "1", func_body, "positive_flag should return 1 for positive values"
        )
        self.assertIn(
            "0", func_body, "positive_flag should return 0 for non-positive values"
        )

    def test_nonzero_flag_checks_not_equal_zero(self):
        """Test that nonzero_flag checks for <> 0."""
        func_body = self._get_function_body("nonzero_flag")
        self.assertTrue(
            "<> 0" in func_body or "!= 0" in func_body or "<>0" in func_body,
            "nonzero_flag should check for <> 0",
        )

    def test_calc_liquidity_stress_uses_current_ratio_thresholds(self):
        """Test that calc_liquidity_stress uses 1.0 and 1.5 thresholds."""
        func_body = self._get_function_body("calc_liquidity_stress")
        self.assertIn(
            "1.0", func_body, "calc_liquidity_stress should use 1.0 threshold"
        )
        self.assertIn(
            "1.5", func_body, "calc_liquidity_stress should use 1.5 threshold"
        )

    def test_calc_liquidity_stress_returns_30_15_0(self):
        """Test that calc_liquidity_stress returns 30.0, 15.0, or 0.0."""
        func_body = self._get_function_body("calc_liquidity_stress")
        self.assertIn(
            "30",
            func_body,
            "calc_liquidity_stress should return 30.0 for severe stress",
        )
        self.assertIn(
            "15",
            func_body,
            "calc_liquidity_stress should return 15.0 for moderate stress",
        )


class TestSQLSyntaxValidity(unittest.TestCase):
    """Test suite for basic SQL syntax validity checks."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def test_all_functions_have_language_declaration(self):
        """Test that all functions end with a LANGUAGE declaration (SQL or plpgsql)."""
        # Count CREATE FUNCTION statements
        create_count = len(
            re.findall(
                r"CREATE\s+OR\s+REPLACE\s+FUNCTION", self.sql_content, re.IGNORECASE
            )
        )
        # Count LANGUAGE statements (SQL or plpgsql)
        language_sql_count = len(
            re.findall(r"\$\$\s+LANGUAGE\s+SQL", self.sql_content, re.IGNORECASE)
        )
        language_plpgsql_count = len(
            re.findall(r"\$\$\s+LANGUAGE\s+plpgsql", self.sql_content, re.IGNORECASE)
        )
        total_language_count = language_sql_count + language_plpgsql_count
        self.assertEqual(
            create_count,
            total_language_count,
            f"All {create_count} functions should have LANGUAGE declaration, found {total_language_count}",
        )

    def test_no_undefined_pct_change_calls(self):
        """Test that pct_change is defined before it's used."""
        # Find first usage of pct_change
        first_use = self.sql_content.lower().find("pct_change(")
        # Find definition of pct_change
        definition = re.search(
            r"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+pct_change\s*\(",
            self.sql_content,
            re.IGNORECASE,
        )
        self.assertIsNotNone(definition, "pct_change function should be defined")
        self.assertLess(
            definition.start(),
            first_use,
            "pct_change should be defined before first use",
        )


class TestMvFeatureRegistryHelperFunctions(unittest.TestCase):
    """Test suite for verifying helper functions in mv_feature_registry.sql.

    This ensures mv_feature_registry.sql is self-contained and can be executed
    independently without requiring feature_registry.sql to be run first.

    Root cause fix for: ERROR: function safe_divide(numeric, numeric) does not exist
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - load SQL file content once for all tests."""
        cls.sql_file = Path(__file__).parent.parent / "mv_feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    def _function_exists(self, function_name: str) -> bool:
        """Check if a CREATE OR REPLACE FUNCTION statement exists for the given function."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        return bool(re.search(pattern, self.sql_content, re.IGNORECASE))

    def _get_function_position(self, function_name: str) -> int:
        """Get the position (character index) of a function definition."""
        pattern = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(function_name)}\s*\("
        match = re.search(pattern, self.sql_content, re.IGNORECASE)
        return match.start() if match else -1

    def test_safe_divide_exists_in_mv_feature_registry(self):
        """Test that safe_divide is defined in mv_feature_registry.sql."""
        self.assertTrue(
            self._function_exists("safe_divide"),
            "safe_divide function should be defined in mv_feature_registry.sql",
        )

    def test_calc_change_ratio_exists_in_mv_feature_registry(self):
        """Test that calc_change_ratio is defined in mv_feature_registry.sql."""
        self.assertTrue(
            self._function_exists("calc_change_ratio"),
            "calc_change_ratio function should be defined in mv_feature_registry.sql",
        )

    def test_safe_divide_defined_before_calc_change_ratio(self):
        """Test that safe_divide is defined BEFORE calc_change_ratio.

        This is the root cause fix for the error:
        ERROR: function safe_divide(numeric, numeric) does not exist
        Where: SQL function "calc_change_ratio" during inlining
        """
        safe_divide_pos = self._get_function_position("safe_divide")
        calc_change_ratio_pos = self._get_function_position("calc_change_ratio")

        self.assertGreater(safe_divide_pos, -1, "safe_divide should be defined")
        self.assertGreater(
            calc_change_ratio_pos, -1, "calc_change_ratio should be defined"
        )
        self.assertLess(
            safe_divide_pos,
            calc_change_ratio_pos,
            "safe_divide must be defined BEFORE calc_change_ratio",
        )

    def test_all_helper_functions_exist_in_mv_feature_registry(self):
        """Test that all required helper functions are defined in mv_feature_registry.sql."""
        required_helpers = [
            "safe_divide",
            "calc_change_ratio",
            "clamp_score",
            "ema_crossover_signal",
            "calc_total_analyst_ratings",
            "near_threshold_flag",
            "pct_change",
            "calc_cagr",
            "calc_adjustment_pct",
            "positive_flag",
            "improvement_flag",
            "nonzero_flag",
            "calc_liquidity_stress",
        ]
        for func_name in required_helpers:
            self.assertTrue(
                self._function_exists(func_name),
                f"{func_name} function should be defined in mv_feature_registry.sql",
            )

    def test_pct_change_defined_after_calc_change_ratio(self):
        """Test that pct_change is defined after calc_change_ratio (dependency order)."""
        calc_change_ratio_pos = self._get_function_position("calc_change_ratio")
        pct_change_pos = self._get_function_position("pct_change")

        self.assertGreater(
            pct_change_pos,
            calc_change_ratio_pos,
            "pct_change must be defined AFTER calc_change_ratio (it depends on it)",
        )

    def test_helper_functions_defined_before_materialized_view(self):
        """Test that all helper functions are defined before the materialized view."""
        mv_pattern = r"CREATE\s+MATERIALIZED\s+VIEW"
        mv_match = re.search(mv_pattern, self.sql_content, re.IGNORECASE)
        self.assertIsNotNone(mv_match, "Materialized view should exist")
        mv_position = mv_match.start()

        # All helper functions should be defined before the materialized view
        helper_functions = [
            "safe_divide",
            "calc_change_ratio",
            "pct_change",
            "clamp_score",
        ]
        for func_name in helper_functions:
            func_pos = self._get_function_position(func_name)
            self.assertLess(
                func_pos,
                mv_position,
                f"{func_name} must be defined BEFORE the materialized view",
            )


if __name__ == "__main__":
    unittest.main()
