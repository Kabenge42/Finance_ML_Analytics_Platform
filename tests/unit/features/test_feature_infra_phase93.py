"""
Phase 9.3 (Week 1) — Test Infrastructure

This suite validates the new fixtures and helpers and performs a light integration
sanity-check against build_comprehensive_features to ensure basic compatibility.

Coverage goals:
- Helpers: >= 90% (small and critical)
- Integration: exercise core path without heavy computation
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from tests.fixtures.feature_engineering_samples import (
    make_minimal_sample,
    make_edge_case_sample,
    make_large_sample,
)
from tests.utils.feature_test_helpers import (
    assert_no_inf,
    assert_nan_ratio_below,
    assert_within_range,
    time_block,
)

# Prefer the consolidated module path if available; fall back to legacy for BC
try:
    from finance_ml.features.advanced import build_comprehensive_features
except Exception:  # pragma: no cover - legacy import path fallback
    from finance_ml.advanced_features import build_comprehensive_features  # type: ignore


class TestFeatureTestHelpers(unittest.TestCase):
    def test_assert_no_inf_passes_without_infinities(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 3.0]})
        assert_no_inf(df)

    def test_assert_no_inf_raises_on_inf(self):
        df = pd.DataFrame({"a": [np.inf, 2.0]})
        with self.assertRaises(AssertionError):
            assert_no_inf(df)

    def test_assert_nan_ratio_below(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 2.0, 3.0]})
        # 25% NaN, threshold 0.3 should pass; 0.2 should fail
        assert_nan_ratio_below(df, columns=["a"], max_ratio=0.3)
        with self.assertRaises(AssertionError):
            assert_nan_ratio_below(df, columns=["a"], max_ratio=0.2)

    def test_assert_nan_ratio_below_value_error(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        with self.assertRaises(ValueError):
            assert_nan_ratio_below(df, columns=["a"], max_ratio=2.0)

    def test_assert_within_range(self):
        df = pd.DataFrame({"x": [0.0, 0.5, 1.0, np.nan]})
        assert_within_range(df, column="x", min_value=0.0, max_value=1.0)
        df_bad = pd.DataFrame({"x": [-0.1, 0.5, 1.1]})
        with self.assertRaises(AssertionError):
            assert_within_range(df_bad, column="x", min_value=0.0, max_value=1.0)

    def test_time_block(self):
        with time_block(0.2):
            # quick operation well within budget
            sum(range(1000))


class TestIntegrationBuildComprehensiveFeatures(unittest.TestCase):
    def test_build_comprehensive_features_on_minimal_sample(self):
        df = make_minimal_sample()
        with time_block(0.5):
            result = build_comprehensive_features(
                df, include_interactions=False, include_relative_values=False
            )
        # basic sanity checks
        self.assertEqual(len(result), len(df))
        self.assertIn("p_e_ratio", result.columns)
        self.assertIn("roe", result.columns)
        # numerical hygiene
        assert_no_inf(result)
        # core ratios should not be entirely NaN on clean data
        assert_nan_ratio_below(result, columns=["p_e_ratio", "roe"], max_ratio=0.0)

    def test_build_comprehensive_features_on_edge_cases(self):
        df = make_edge_case_sample()
        result = build_comprehensive_features(
            df, include_interactions=False, include_relative_values=False
        )
        # Ensure pipeline does not introduce infinities
        assert_no_inf(result)
        # Allow NaNs due to safe divisions, but require not all NaN for certain robust ratios if inputs permit
        cols = [
            c for c in ["p_e_ratio", "p_b_ratio", "ev_ebitda_ratio", "roe"] if c in result.columns
        ]
        if cols:
            # at least one finite value among these columns overall
            finite_any = np.isfinite(result[cols].to_numpy(dtype=float, copy=False)).any()
            self.assertTrue(finite_any)

    def test_build_comprehensive_features_large_sample_perf(self):
        df = make_large_sample(1000)
        with time_block(2.0):  # should be very fast on 1K rows
            result = build_comprehensive_features(
                df, include_interactions=False, include_relative_values=False
            )
        self.assertEqual(len(result), len(df))
        assert_no_inf(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
