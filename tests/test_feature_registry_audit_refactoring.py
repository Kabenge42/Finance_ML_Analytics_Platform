"""
Test suite for feature_registry.sql audit & refactoring recommendations.

TDD Approach:
- RED: Tests written first to define expected behavior
- GREEN: Minimal implementation to pass tests
- REFACTOR: Clean up and optimize

Coverage target: ≥80% for changed files

Key Improvements Being Tested:
1. [HIGH] Fix eps_stability NULL placeholder in calc_eps_trajectory_features
2. [HIGH] Fix ROIC formula in calc_profitability_features
3. [HIGH] New calc_volatility_surface_features function
4. [MEDIUM] New calc_forward_consensus_features function
5. [MEDIUM] New calc_price_target_achievement_features function
6. [MEDIUM] New calc_dividend_history_features function
7. [MEDIUM] Enhance calc_long_term_momentum_features with Total Return columns
8. [MEDIUM] DRY violation fix in calc_extended_valuation_timeseries
9. [LOW] Decompose calc_composite_scores into atomic functions
10. [LOW] New calc_size_liquidity_features function
11. [LOW] New calc_investment_income_temporal function
12. [LOW] Remove duplicate eps_trajectory_score computation
"""

import re
import unittest
from pathlib import Path


class _SQLTestBase(unittest.TestCase):
    """Base class that loads feature_registry.sql once."""

    sql_content: str = ""

    @classmethod
    def setUpClass(cls):
        cls.sql_file = Path(__file__).parent.parent / "feature_registry.sql"
        if not cls.sql_file.exists():
            raise FileNotFoundError(f"SQL file not found: {cls.sql_file}")
        cls.sql_content = cls.sql_file.read_text(encoding="utf-8")

    # -- helpers ---------------------------------------------------------------

    def _function_exists(self, name: str) -> bool:
        pat = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(name)}\s*\("
        return bool(re.search(pat, self.sql_content, re.IGNORECASE))

    def _function_body(self, name: str) -> str:
        """Return the text between the first $$ ... $$ for *name*."""
        pat = (
            rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(name)}\s*\(" r".*?\$\$\s*(.*?)\s*\$\$"
        )
        m = re.search(pat, self.sql_content, re.IGNORECASE | re.DOTALL)
        return m.group(1) if m else ""

    def _function_header(self, name: str, chars: int = 1500) -> str:
        """Return the first *chars* characters starting from CREATE ... FUNCTION name."""
        pat = rf"CREATE\s+OR\s+REPLACE\s+FUNCTION\s+{re.escape(name)}\s*\("
        m = re.search(pat, self.sql_content, re.IGNORECASE)
        if m:
            return self.sql_content[m.start() : m.start() + chars]
        return ""

    def _function_returns_table_columns(self, name: str) -> list[str]:
        """Extract column names from RETURNS TABLE(...)."""
        header = self._function_header(name, 3000)
        table_match = re.search(
            r"RETURNS\s+TABLE\s*\((.*?)\)\s*(STABLE|IMMUTABLE|VOLATILE)",
            header,
            re.IGNORECASE | re.DOTALL,
        )
        if not table_match:
            return []
        cols_text = table_match.group(1)
        return [
            line.strip().split()[0]
            for line in cols_text.splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]

    def _is_stable_parallel_safe(self, name: str) -> bool:
        header = self._function_header(name, 1500)
        return bool(
            re.search(r"\bSTABLE\b", header, re.IGNORECASE)
            and re.search(r"PARALLEL\s+SAFE", header, re.IGNORECASE)
        )


# =============================================================================
# 1. HIGH: Fix eps_stability NULL placeholder
# =============================================================================
class TestEpsStabilityFix(_SQLTestBase):
    """eps_stability in calc_eps_trajectory_features must compute a real value."""

    def test_eps_stability_not_null_placeholder(self):
        """eps_stability should NOT be a NULL::NUMERIC placeholder."""
        body = self._function_body("calc_eps_trajectory_features")
        self.assertNotIn(
            "NULL::NUMERIC",
            body,
            "eps_stability should be computed, not NULL::NUMERIC",
        )

    def test_eps_stability_uses_coefficient_of_variation(self):
        """eps_stability should use a coefficient-of-variation approach (SQRT, POWER)."""
        body = self._function_body("calc_eps_trajectory_features").upper()
        self.assertIn("SQRT", body, "eps_stability should use SQRT for std-dev")
        self.assertIn("POWER", body, "eps_stability should use POWER for variance")

    def test_eps_stability_uses_5_year_eps_values(self):
        """eps_stability should reference 5 years of EPS data."""
        body = self._function_body("calc_eps_trajectory_features")
        for suffix in ["(FY)", "(-1FY)", "(-2FY)", "(-3FY)", "(-4FY)"]:
            col = f'"Net EPS - Basic {suffix}"'
            self.assertIn(
                col,
                body,
                f"eps_stability should reference {col}",
            )

    def test_eps_stability_bounded_0_to_1(self):
        """eps_stability should be bounded between 0 and 1."""
        body = self._function_body("calc_eps_trajectory_features").upper()
        self.assertIn("LEAST", body, "eps_stability should use LEAST to cap at 1")


# =============================================================================
# 2. HIGH: Fix ROIC formula in calc_profitability_features
# =============================================================================
class TestRoicFormulaFix(_SQLTestBase):
    """ROIC should use NOPAT (EBIT-based), not Net Income."""

    def test_roic_uses_ebit(self):
        """ROIC calculation should reference EBIT, not Net Income."""
        body = self._function_body("calc_profitability_features")
        # Find the ROIC line specifically
        roic_match = re.search(r"(.+AS\s+roic)", body, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(roic_match, "roic column should exist")
        roic_section = roic_match.group(1)
        self.assertIn(
            '"EBIT (LTM)"',
            roic_section,
            "ROIC should use EBIT (LTM) for NOPAT proxy",
        )

    def test_roic_does_not_use_net_income(self):
        """ROIC should NOT use Net Income directly."""
        body = self._function_body("calc_profitability_features")
        roic_match = re.search(r"(.+AS\s+roic)", body, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(roic_match)
        roic_section = roic_match.group(1)
        # Split by AS to get only the ROIC expression (last column before roic alias)
        # We need the expression that maps to roic, not the whole body
        lines = roic_section.strip().splitlines()
        # Find the line(s) that compute roic
        roic_lines = []
        for i, line in enumerate(lines):
            if "AS roic" in line.lower() or (i > 0 and "roic" in lines[i].lower()):
                # Collect this line and preceding lines until previous AS
                for j in range(i, -1, -1):
                    roic_lines.insert(0, lines[j])
                    if j < i and " AS " in lines[j].upper():
                        break
                break
        roic_expr = "\n".join(roic_lines)
        self.assertNotIn(
            '"Net Income - (IS) (LTM)"',
            roic_expr,
            "ROIC should not use Net Income",
        )

    def test_roic_subtracts_cash(self):
        """ROIC denominator should be Invested Capital = Equity + Debt - Cash."""
        body = self._function_body("calc_profitability_features")
        roic_match = re.search(r"(.+AS\s+roic)", body, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(roic_match)
        roic_section = roic_match.group(1)
        self.assertIn(
            '"Cash And Equivalents (LTM)"',
            roic_section,
            "ROIC denominator should subtract Cash And Equivalents",
        )

    def test_roic_applies_tax_rate(self):
        """ROIC should apply a tax rate assumption (e.g., 0.25)."""
        body = self._function_body("calc_profitability_features")
        roic_match = re.search(r"(.+AS\s+roic)", body, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(roic_match)
        roic_section = roic_match.group(1)
        # Should contain a tax multiplier like (1 - 0.25) or 0.75
        has_tax = "0.25" in roic_section or "0.75" in roic_section
        self.assertTrue(has_tax, "ROIC should apply a tax rate (0.25 or 0.75)")


# =============================================================================
# 3. HIGH: New calc_volatility_surface_features
# =============================================================================
class TestVolatilitySurfaceFeatures(_SQLTestBase):
    """New function: calc_volatility_surface_features."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_volatility_surface_features"),
            "calc_volatility_surface_features should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_volatility_surface_features"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_volatility_surface_features")
        expected = [
            "isin",
            "vol_1m",
            "vol_3m",
            "vol_6m",
            "vol_1y",
            "vol_term_spread_short",
            "vol_term_spread_long",
            "vol_ratio_3m_1y",
            "vol_hump",
            "beta_1y",
            "beta_2y",
            "beta_5y",
            "beta_term_structure",
            "beta_convexity",
            "realized_vs_implied_proxy",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_uses_safe_divide(self):
        body = self._function_body("calc_volatility_surface_features")
        self.assertIn("safe_divide", body, "Should use safe_divide helper")

    def test_uses_calc_change_ratio(self):
        body = self._function_body("calc_volatility_surface_features")
        self.assertIn(
            "calc_change_ratio", body, "Should use calc_change_ratio for beta_term_structure"
        )

    def test_references_beta_2y(self):
        body = self._function_body("calc_volatility_surface_features")
        self.assertIn('"Beta (2Y)"', body, "Should reference Beta (2Y)")

    def test_references_all_volatility_tenors(self):
        body = self._function_body("calc_volatility_surface_features")
        for tenor in ["1M", "3M", "6M", "1Y"]:
            self.assertIn(
                f'"Volatility ({tenor})"',
                body,
                f"Should reference Volatility ({tenor})",
            )


# =============================================================================
# 4. MEDIUM: New calc_forward_consensus_features
# =============================================================================
class TestForwardConsensusFeatures(_SQLTestBase):
    """New function: calc_forward_consensus_features."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_forward_consensus_features"),
            "calc_forward_consensus_features should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_forward_consensus_features"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_forward_consensus_features")
        expected = [
            "isin",
            "pe_ntm",
            "pe_est_fy1",
            "pe_forward_discount",
            "eps_gaap_vs_norm_ntm",
            "eps_gaap_vs_norm_fy1e",
            "forward_adjustment_trend",
            "ebitda_est_ntm",
            "ebitda_est_fy1e",
            "ev_ebitda_est_fy1",
            "ebitda_forward_growth",
            "earnings_revision_divergence",
            "forward_pe_vs_sector_proxy",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_uses_previously_unused_columns(self):
        """Should reference columns that were previously unused."""
        body = self._function_body("calc_forward_consensus_features")
        self.assertIn('"P/E (NTM)"', body)
        self.assertIn('"EPS GAAP - Est Avg (NTM)"', body)
        self.assertIn('"EV/EBITDA (EST FY1)"', body)


# =============================================================================
# 5. MEDIUM: New calc_price_target_achievement_features
# =============================================================================
class TestPriceTargetAchievementFeatures(_SQLTestBase):
    """New function: calc_price_target_achievement_features."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_price_target_achievement_features"),
            "calc_price_target_achievement_features should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_price_target_achievement_features"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_price_target_achievement_features")
        expected = [
            "isin",
            "pt_achievement_1y",
            "pt_accuracy_1y",
            "pt_optimism_bias",
            "pt_range_hit_rate",
            "pt_median_vs_mean_spread",
            "pt_high_low_convergence_1y",
            "analyst_count_stability",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_references_price_target_1y_ago(self):
        body = self._function_body("calc_price_target_achievement_features")
        self.assertIn('"Price Target (1Y Ago)"', body)


# =============================================================================
# 6. MEDIUM: New calc_dividend_history_features
# =============================================================================
class TestDividendHistoryFeatures(_SQLTestBase):
    """New function: calc_dividend_history_features."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_dividend_history_features"),
            "calc_dividend_history_features should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_dividend_history_features"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_dividend_history_features")
        expected = [
            "isin",
            "div_yield_2fy",
            "div_yield_3fy",
            "div_yield_4fy",
            "div_yield_5fy",
            "div_yield_trend_3y",
            "div_yield_volatility",
            "div_yield_declining_flag",
            "div_yield_mean_5y",
            "div_yield_vs_5y_mean",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_references_historical_div_yields(self):
        body = self._function_body("calc_dividend_history_features")
        for fy in ["-2FYInd", "-3FYInd", "-4FYInd", "-5FYInd"]:
            self.assertIn(
                f'"Div Yield ({fy})"',
                body,
                f"Should reference Div Yield ({fy})",
            )


# =============================================================================
# 7. MEDIUM: Enhance calc_long_term_momentum_features with Total Return
# =============================================================================
class TestLongTermMomentumTotalReturn(_SQLTestBase):
    """calc_long_term_momentum_features should include Total Return columns."""

    def test_has_total_return_columns(self):
        cols = self._function_returns_table_columns("calc_long_term_momentum_features")
        expected_new = [
            "total_return_ytd",
            "total_return_5y",
            "total_return_10y",
            "return_cagr_3y",
            "return_cagr_10y",
            "return_vs_price_momentum",
            "return_consistency_score",
        ]
        for col in expected_new:
            self.assertIn(col, cols, f"Missing new column: {col}")

    def test_references_total_return_source_columns(self):
        body = self._function_body("calc_long_term_momentum_features")
        self.assertIn('"Total Return (YTD)"', body)
        self.assertIn('"Total Return (5Y)"', body)
        self.assertIn('"Total Return (10Y)"', body)
        self.assertIn('"Tot. Return %/CAGR (3Y)"', body)
        self.assertIn('"Tot. Return %/CAGR (10Y)"', body)

    def test_return_consistency_score_uses_safe_divide(self):
        body = self._function_body("calc_long_term_momentum_features")
        self.assertIn("safe_divide", body, "return_consistency_score should use safe_divide")


# =============================================================================
# 8. MEDIUM: DRY violation fix in calc_extended_valuation_timeseries
# =============================================================================
class TestExtendedValuationTimeseriesDRY(_SQLTestBase):
    """calc_extended_valuation_timeseries should use helper functions, not inline NULLIF."""

    def test_uses_calc_change_ratio(self):
        body = self._function_body("calc_extended_valuation_timeseries")
        self.assertIn(
            "calc_change_ratio",
            body,
            "Should use calc_change_ratio instead of inline division",
        )

    def test_no_inline_nullif_division_pattern(self):
        """Should not have inline (X - Y) / NULLIF(Y, 0) patterns."""
        body = self._function_body("calc_extended_valuation_timeseries")
        # Count remaining inline NULLIF patterns for change-ratio-like calculations
        # Pattern: (something - something) / NULLIF(something, 0)
        inline_pattern = re.findall(
            r"\([^)]+\)\s*/\s*NULLIF\s*\([^)]+,\s*0\)",
            body,
        )
        # Allow at most 1 for p_e_percentile_proxy which has a special 0.5 multiplier
        self.assertLessEqual(
            len(inline_pattern),
            1,
            f"Found {len(inline_pattern)} inline NULLIF division patterns; "
            f"should use calc_change_ratio instead",
        )


# =============================================================================
# 9. LOW: Decompose calc_composite_scores — new atomic functions
# =============================================================================
class TestCompositeScoreDecomposition(_SQLTestBase):
    """calc_composite_scores should be decomposed into atomic functions."""

    def test_calc_piotroski_f_score_exists(self):
        self.assertTrue(
            self._function_exists("calc_piotroski_f_score"),
            "calc_piotroski_f_score should be a standalone function",
        )

    def test_calc_piotroski_f_score_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_piotroski_f_score"))

    def test_calc_piotroski_f_score_returns_9_components(self):
        """Piotroski F-Score has 9 binary components summed to an integer."""
        cols = self._function_returns_table_columns("calc_piotroski_f_score")
        self.assertIn("isin", cols)
        self.assertIn("piotroski_f_score", cols)

    def test_calc_shareholder_dilution_features_exists(self):
        self.assertTrue(
            self._function_exists("calc_shareholder_dilution_features"),
            "calc_shareholder_dilution_features should be a standalone function",
        )

    def test_calc_quality_momentum_composite_exists(self):
        self.assertTrue(
            self._function_exists("calc_quality_momentum_composite"),
            "calc_quality_momentum_composite should be a standalone function",
        )

    def test_composite_scores_no_duplicate_eps_trajectory(self):
        """calc_composite_scores should NOT compute eps_trajectory_score (it's in calc_eps_trajectory_features)."""
        cols = self._function_returns_table_columns("calc_composite_scores")
        self.assertNotIn(
            "eps_trajectory_score",
            cols,
            "eps_trajectory_score should be removed from calc_composite_scores "
            "(already in calc_eps_trajectory_features)",
        )


# =============================================================================
# 10. LOW: New calc_size_liquidity_features
# =============================================================================
class TestSizeLiquidityFeatures(_SQLTestBase):
    """New function: calc_size_liquidity_features."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_size_liquidity_features"),
            "calc_size_liquidity_features should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_size_liquidity_features"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_size_liquidity_features")
        expected = [
            "isin",
            "market_cap",
            "market_cap_country_r",
            "log_market_cap",
            "volume_shrs",
            "relative_volume",
            "shares_outstanding",
            "daily_turnover_ratio",
            "size_class",
            "style_class",
            "liquidity_score",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_uses_log_for_market_cap(self):
        body = self._function_body("calc_size_liquidity_features").upper()
        self.assertIn("LN(", body, "Should use LN() for log_market_cap")


# =============================================================================
# 11. LOW: New calc_investment_income_temporal
# =============================================================================
class TestInvestmentIncomeTemporalFeatures(_SQLTestBase):
    """New function: calc_investment_income_temporal."""

    def test_function_exists(self):
        self.assertTrue(
            self._function_exists("calc_investment_income_temporal"),
            "calc_investment_income_temporal should be defined",
        )

    def test_is_stable_parallel_safe(self):
        self.assertTrue(self._is_stable_parallel_safe("calc_investment_income_temporal"))

    def test_returns_expected_columns(self):
        cols = self._function_returns_table_columns("calc_investment_income_temporal")
        expected = [
            "isin",
            "inv_income_ltm",
            "inv_income_fq",
            "inv_income_fy",
            "inv_income_qoq_growth",
            "inv_income_yoy_growth",
            "inv_income_to_revenue",
            "inv_income_trend_3y",
            "inv_income_positive_quarters",
            "financial_company_proxy",
        ]
        for col in expected:
            self.assertIn(col, cols, f"Missing column: {col}")

    def test_uses_helper_functions(self):
        body = self._function_body("calc_investment_income_temporal")
        self.assertIn("calc_change_ratio", body)
        self.assertIn("safe_divide", body)


# =============================================================================
# 13. View/MV fix: cs.eps_trajectory_score → etf.eps_trajectory_score
# =============================================================================
class TestViewEpsTrajectoryScoreReference(_SQLTestBase):
    """Views must not reference cs.eps_trajectory_score (removed from calc_composite_scores)."""

    def test_vw_features_composite_scores_no_cs_eps_trajectory(self):
        """vw_features_composite_scores should use etf.eps_trajectory_score, not cs."""
        # Extract the view definition
        pat = r"CREATE\s+OR\s+REPLACE\s+VIEW\s+vw_features_composite_scores\s+AS\s+(.*?);\s*\n"
        m = re.search(pat, self.sql_content, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(m, "vw_features_composite_scores should exist")
        view_body = m.group(1)
        self.assertNotIn(
            "cs.eps_trajectory_score",
            view_body,
            "vw_features_composite_scores must not reference cs.eps_trajectory_score",
        )
        self.assertIn(
            "etf.eps_trajectory_score",
            view_body,
            "vw_features_composite_scores should reference etf.eps_trajectory_score",
        )

    def test_vw_features_composite_scores_joins_eps_trajectory(self):
        """vw_features_composite_scores must JOIN calc_eps_trajectory_features."""
        pat = r"CREATE\s+OR\s+REPLACE\s+VIEW\s+vw_features_composite_scores\s+AS\s+(.*?);\s*\n"
        m = re.search(pat, self.sql_content, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(m)
        view_body = m.group(1)
        self.assertIn(
            "calc_eps_trajectory_features()",
            view_body,
            "vw_features_composite_scores must join calc_eps_trajectory_features",
        )

    def test_mv_all_stock_features_no_cs_eps_trajectory(self):
        """mv_all_stock_features should use etf.eps_trajectory_score, not cs."""
        pat = r"CREATE\s+MATERIALIZED\s+VIEW\s+mv_all_stock_features\s+AS\s+(.*?);\s*\n"
        m = re.search(pat, self.sql_content, re.IGNORECASE | re.DOTALL)
        self.assertIsNotNone(m, "mv_all_stock_features should exist")
        mv_body = m.group(1)
        self.assertNotIn(
            "cs.eps_trajectory_score",
            mv_body,
            "mv_all_stock_features must not reference cs.eps_trajectory_score",
        )


if __name__ == "__main__":
    unittest.main()
