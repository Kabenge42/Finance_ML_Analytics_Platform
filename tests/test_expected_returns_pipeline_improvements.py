"""
Tests for Expected Returns Analytics Pipeline Improvements (v3.1).

Covers:
- hierarchical_mcmc_multi_level (multi-level nested shrinkage)
- Outlier winsorization in build_expected_returns_summary
- Integration of multi-level MCMC in the pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_level_df() -> pd.DataFrame:
    """DataFrame with multiple categorical columns for multi-level MCMC."""
    np.random.seed(42)
    n = 600

    regions = ["North America", "Europe", "Asia"]
    countries = {
        "North America": ["USA", "Canada"],
        "Europe": ["UK", "Germany"],
        "Asia": ["Japan", "China"],
    }
    sectors = ["Technology", "Healthcare", "Financials"]
    industries = {
        "Technology": ["Software", "Hardware"],
        "Healthcare": ["Pharma", "Biotech"],
        "Financials": ["Banks", "Insurance"],
    }

    rows = []
    for i in range(n):
        region = regions[i % 3]
        country = countries[region][i % 2]
        sector = sectors[i % 3]
        industry = industries[sector][i % 2]
        rows.append({
            "ticker": f"T{i:04d}",
            "region": region,
            "country": country,
            "sector": sector,
            "industry": industry,
            "expected_upside_pct": np.random.normal(15, 30),
        })

    return pd.DataFrame(rows)


@pytest.fixture
def summary_with_outliers() -> pd.DataFrame:
    """Summary DataFrame with extreme outliers for winsorization testing."""
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "ticker": [f"T{i:04d}" for i in range(n)],
        "expected_upside_pct": np.random.normal(15, 20, n),
        "expected_return_prob_weighted": np.random.normal(20, 30, n),
        "filtered_upside": np.random.normal(18, 25, n),
        "prob_positive_upside": np.random.uniform(30, 90, n),
        "price_target_mc": np.random.uniform(10, 100, n),
        "var_5_pct": np.random.normal(-10, 5, n),
        "risk_reward_ratio": np.random.uniform(0.5, 3, n),
        "kalman_estimate": np.random.uniform(20, 100, n),
        "achievement_probability": np.random.uniform(0.3, 0.9, n),
        "confidence_level": np.random.choice(["High", "Medium", "Low"], n),
        "analyst_conviction": np.random.uniform(0.3, 1.0, n),
        "eps_revision_momentum": np.random.normal(0, 1, n),
        "analyst_rating_normalized": np.random.uniform(0, 1, n),
        "posterior_beat_prob": np.random.uniform(0.3, 0.8, n),
        "confidence_score": np.random.uniform(0.3, 0.9, n),
        "beat_classification": np.random.choice(["likely_beat", "likely_miss"], n),
        "industry": np.random.choice(["Software", "Banks", "Pharma"], n),
    })
    # Inject extreme outliers
    df.loc[0, "expected_return_prob_weighted"] = 4090.0
    df.loc[1, "expected_return_prob_weighted"] = -500.0
    df.loc[0, "filtered_upside"] = 749.0
    df.loc[1, "filtered_upside"] = -300.0
    return df


# ===========================================================================
# Tests for _HIERARCHICAL_CATEGORY_COLS
# ===========================================================================

class TestHierarchicalCategoryCols:
    """Verify the category columns constant is defined and correct."""

    def test_constant_exists(self):
        from finance_ml.analytics.statistical_analysis import _HIERARCHICAL_CATEGORY_COLS
        assert isinstance(_HIERARCHICAL_CATEGORY_COLS, list)

    def test_contains_expected_columns(self):
        from finance_ml.analytics.statistical_analysis import _HIERARCHICAL_CATEGORY_COLS
        expected = {"region", "country", "sector", "industry", "exchange"}
        assert expected.issubset(set(_HIERARCHICAL_CATEGORY_COLS))

    def test_has_nine_columns(self):
        from finance_ml.analytics.statistical_analysis import _HIERARCHICAL_CATEGORY_COLS
        assert len(_HIERARCHICAL_CATEGORY_COLS) == 9


# ===========================================================================
# Tests for hierarchical_mcmc_multi_level
# ===========================================================================

class TestHierarchicalMcmcMultiLevel:
    """Tests for the multi-level hierarchical MCMC function."""

    def test_returns_dict(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct")
        assert isinstance(result, dict)

    def test_has_global_key(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct")
        assert "global" in result
        assert "mean" in result["global"]
        assert "std" in result["global"]
        assert "n_obs" in result["global"]

    def test_has_levels_key(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct")
        assert "levels" in result
        assert isinstance(result["levels"], dict)

    def test_levels_contain_available_columns(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct",
                                               group_cols=["region", "sector", "industry"])
        for col in ["region", "sector", "industry"]:
            assert col in result["levels"], f"Missing level: {col}"

    def test_group_results_have_required_keys(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["region"])
        for group_name, group_data in result["levels"]["region"].items():
            for key in ["raw_mean", "posterior_mean", "shrinkage", "ci_95",
                        "prob_positive", "samples", "n_obs", "shrinkage_target"]:
                assert key in group_data, f"Missing key '{key}' in group '{group_name}'"

    def test_shrinkage_between_zero_and_one(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["region", "sector"])
        for level_name, level_data in result["levels"].items():
            for group_name, group_info in level_data.items():
                assert 0 < group_info["shrinkage"] < 1, (
                    f"Shrinkage out of range for {level_name}/{group_name}"
                )

    def test_small_groups_shrink_more(self, multi_level_df):
        """Groups with fewer observations should have lower shrinkage (more pooling)."""
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        # Create a df with one very small group
        df = multi_level_df.copy()
        # Make "Asia" have very few observations
        asia_mask = df["region"] == "Asia"
        df = df[~asia_mask].copy()
        small_rows = pd.DataFrame({
            "ticker": [f"SMALL{i}" for i in range(5)],
            "region": ["Asia"] * 5,
            "country": ["Japan"] * 5,
            "sector": ["Technology"] * 5,
            "industry": ["Software"] * 5,
            "expected_upside_pct": np.random.normal(15, 10, 5),
        })
        df = pd.concat([df, small_rows], ignore_index=True)

        result = hierarchical_mcmc_multi_level(df, "expected_upside_pct", group_cols=["region"], min_group_size=3)
        regions = result["levels"].get("region", {})
        if "Asia" in regions and "North America" in regions:
            assert regions["Asia"]["shrinkage"] < regions["North America"]["shrinkage"]

    def test_cross_level_summary_is_dataframe(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct")
        assert "cross_level_summary" in result
        assert isinstance(result["cross_level_summary"], pd.DataFrame)
        assert not result["cross_level_summary"].empty

    def test_cross_level_summary_columns(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct")
        xls = result["cross_level_summary"]
        expected_cols = {"level", "group", "n_obs", "raw_mean", "posterior_mean",
                         "shrinkage", "ci_95_low", "ci_95_high", "prob_positive"}
        assert expected_cols.issubset(set(xls.columns))

    def test_nested_shrinkage_uses_parent_mean(self, multi_level_df):
        """Industry should shrink toward sector mean, not global mean."""
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["sector", "industry"])
        # Check that industry groups have shrinkage_target != global mean
        global_mean = result["global"]["mean"]
        sector_level = result["levels"].get("sector", {})
        industry_level = result["levels"].get("industry", {})

        # At least some industry groups should shrink toward their parent sector
        has_parent_shrinkage = False
        for group_name, group_data in industry_level.items():
            if group_data["shrinkage_target"] != global_mean:
                has_parent_shrinkage = True
                break
        if sector_level and industry_level:
            assert has_parent_shrinkage, "Industry groups should shrink toward sector, not global"

    def test_missing_feature_returns_empty(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "nonexistent_column")
        assert result == {}

    def test_insufficient_data_returns_empty(self):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        small_df = pd.DataFrame({
            "ticker": ["A", "B"],
            "region": ["US", "EU"],
            "val": [1.0, 2.0],
        })
        result = hierarchical_mcmc_multi_level(small_df, "val")
        assert result == {}

    def test_no_categorical_columns_returns_global_only(self):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        np.random.seed(42)
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(100)],
            "val": np.random.normal(10, 5, 100),
        })
        result = hierarchical_mcmc_multi_level(df, "val")
        assert "global" in result
        assert "levels" not in result or not result.get("levels")

    def test_custom_shrinkage_strength(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result_low = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["region"],
                                                   shrinkage_strength=1.0)
        result_high = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["region"],
                                                    shrinkage_strength=100.0)
        # Higher shrinkage_strength → lower shrinkage values
        for region in result_low["levels"]["region"]:
            if region in result_high["levels"]["region"]:
                assert (result_low["levels"]["region"][region]["shrinkage"]
                        >= result_high["levels"]["region"][region]["shrinkage"])

    def test_samples_shape(self, multi_level_df):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        n_samples = 5000
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct", group_cols=["region"],
                                               n_samples=n_samples)
        for group_data in result["levels"]["region"].values():
            assert group_data["samples"].shape == (n_samples,)


# ===========================================================================
# Tests for outlier winsorization in build_expected_returns_summary
# ===========================================================================

class TestOutlierWinsorization:
    """Verify that build_expected_returns_summary clips extreme outliers."""

    def _build_summary_inputs(self, summary_with_outliers):
        """Build minimal mc/kal/pt/earn DataFrames for build_expected_returns_summary."""
        df = summary_with_outliers
        mc = df[["ticker", "expected_upside_pct", "price_target_mc",
                  "prob_positive_upside", "var_5_pct", "risk_reward_ratio",
                  "industry"]].copy()
        kal = df[["ticker", "filtered_upside", "kalman_estimate"]].copy()
        pt = df[["ticker", "expected_return_prob_weighted",
                  "achievement_probability", "confidence_level",
                  "analyst_conviction", "eps_revision_momentum",
                  "analyst_rating_normalized"]].copy()
        pt["price_target_prob_weighted"] = df["price_target_mc"] * 1.1
        earn = df[["ticker", "posterior_beat_prob", "confidence_score",
                    "beat_classification"]].copy()
        return mc, kal, pt, earn

    def test_expected_return_prob_weighted_clipped(self, summary_with_outliers):
        from expected_returns_v3 import build_expected_returns_summary
        mc, kal, pt, earn = self._build_summary_inputs(summary_with_outliers)
        result = build_expected_returns_summary(mc, kal, pt, earn)
        if not result.empty and "expected_return_prob_weighted" in result.columns:
            # After winsorization, extreme values should be absent
            assert result["expected_return_prob_weighted"].max() < 4090.0
            assert result["expected_return_prob_weighted"].min() > -500.0

    def test_filtered_upside_clipped(self, summary_with_outliers):
        from expected_returns_v3 import build_expected_returns_summary
        mc, kal, pt, earn = self._build_summary_inputs(summary_with_outliers)
        result = build_expected_returns_summary(mc, kal, pt, earn)
        if not result.empty and "filtered_upside" in result.columns:
            # After winsorization, extreme values should be absent
            assert result["filtered_upside"].max() < 749.0
            assert result["filtered_upside"].min() > -300.0

    def test_outlier_values_are_reduced(self, summary_with_outliers):
        """The extreme 4090% outlier should be clipped down."""
        from expected_returns_v3 import build_expected_returns_summary
        mc, kal, pt, earn = self._build_summary_inputs(summary_with_outliers)
        result = build_expected_returns_summary(mc, kal, pt, earn)
        if not result.empty and "expected_return_prob_weighted" in result.columns:
            assert result["expected_return_prob_weighted"].max() < 4090.0


# ===========================================================================
# Integration test: multi-level MCMC importable from statistical_analysis
# ===========================================================================

class TestMultiLevelMcmcIntegration:
    """Integration tests for the multi-level MCMC pipeline."""

    def test_importable(self):
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        assert callable(hierarchical_mcmc_multi_level)

    def test_constant_importable(self):
        from finance_ml.analytics.statistical_analysis import _HIERARCHICAL_CATEGORY_COLS
        assert isinstance(_HIERARCHICAL_CATEGORY_COLS, list)

    def test_end_to_end_with_realistic_data(self, multi_level_df):
        """Full pipeline: multi-level MCMC produces usable cross-level summary."""
        from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_multi_level
        result = hierarchical_mcmc_multi_level(multi_level_df, "expected_upside_pct",
                                               group_cols=["region", "country", "sector", "industry"],
                                               min_group_size=20, shrinkage_strength=10.0)
        assert "global" in result
        assert "levels" in result
        assert "cross_level_summary" in result

        xls = result["cross_level_summary"]
        assert len(xls) > 0
        assert xls["level"].nunique() >= 2  # At least 2 levels with enough data
