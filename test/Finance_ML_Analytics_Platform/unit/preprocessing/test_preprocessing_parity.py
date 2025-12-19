"""Parity tests for legacy advanced_preprocessing vs new preprocessing package.

Phase 4 (Restructuring Plan): Consolidate preprocessing duplicates.

These tests ensure that the legacy
``finance_ml.ml_workflow.advanced_preprocessing`` module acts as a thin
compatibility layer over the canonical implementations in
``finance_ml.ml_workflow.preprocessing`` for key functions:

- Outlier detection and winsorization
- Scaling
- Imputation helpers (column lists and 4/6-step strategies)

The goal is API continuity, not exhaustive behavioural duplication. We
keep checks lightweight and deterministic.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestPreprocessingParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        cls.df = pd.DataFrame(
            {
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy"] * 4,
                "feature_a": rng.normal(size=20),
                "feature_b": rng.normal(size=20) * 10 + 100,
                "last_price": rng.uniform(10, 200, size=20),
            }
        )

    def test_outlier_detection_iqr_parity(self):
        """advanced_preprocessing.detect_outliers_iqr should delegate to preprocessing.outliers."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import outliers as new_out

        cols = ["feature_a", "feature_b"]

        legacy_res = legacy.detect_outliers_iqr(self.df.copy(), columns=cols, by_sector=True)
        new_res = new_out.detect_outliers_iqr(self.df.copy(), columns=cols, by_sector=True)

        # Same columns and shape; values are expected to match because legacy is a thin wrapper.
        self.assertListEqual(list(legacy_res.columns), list(new_res.columns))
        self.assertEqual(legacy_res.shape, new_res.shape)

    def test_winsorize_by_sector_parity(self):
        """advanced_preprocessing.winsorize_by_sector should match preprocessing.outliers."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import outliers as new_out

        cols = ["feature_a", "feature_b"]

        legacy_res = legacy.winsorize_by_sector(
            self.df.copy(),
            columns=cols,
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=True,
        )
        new_res = new_out.winsorize_by_sector(
            self.df.copy(),
            columns=cols,
            lower_percentile=0.05,
            upper_percentile=0.95,
            by_sector=True,
        )

        self.assertListEqual(list(legacy_res.columns), list(new_res.columns))
        self.assertEqual(legacy_res.shape, new_res.shape)

    def test_scale_features_parity(self):
        """advanced_preprocessing.scale_features should delegate to preprocessing.scaling."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import scaling as new_scaling

        cols = ["feature_a", "feature_b"]

        legacy_res = legacy.scale_features(
            self.df.copy(), columns=cols, scaler_type="robust", by_sector=True
        )
        new_res = new_scaling.scale_features(
            self.df.copy(), columns=cols, scaler_type="robust", by_sector=True
        )

        # Column sets and shapes should be identical.
        self.assertSetEqual(set(legacy_res.columns), set(new_res.columns))
        self.assertEqual(legacy_res.shape, new_res.shape)

    def test_zero_imputation_columns_parity(self):
        """Column lists for zero imputation should be sourced from preprocessing.imputation."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import imputation as new_imp

        legacy_cols = legacy.get_zero_imputation_columns()
        new_cols = new_imp.get_zero_imputation_columns()

        self.assertIsInstance(legacy_cols, list)
        self.assertListEqual(legacy_cols, new_cols)

    def test_knn_imputation_columns_parity(self):
        """Column lists for KNN imputation should match preprocessing.imputation."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import imputation as new_imp

        legacy_cols = legacy.get_knn_imputation_columns()
        new_cols = new_imp.get_knn_imputation_columns()

        self.assertIsInstance(legacy_cols, list)
        self.assertListEqual(legacy_cols, new_cols)

    def test_enhanced_imputation_4step_parity(self):
        """apply_enhanced_imputation_strategy_4step should call preprocessing.imputation version."""

        from finance_ml.ml_workflow import advanced_preprocessing as legacy
        from finance_ml.ml_workflow.preprocessing import imputation as new_imp

        df = self.df.copy()
        # Introduce a few NaNs
        df.loc[0, "feature_a"] = np.nan
        df.loc[1, "feature_b"] = np.nan

        legacy_res = legacy.apply_enhanced_imputation_strategy_4step(
            df.copy(), sector_column="sector", n_neighbors=3, price_column="last_price"
        )
        new_res = new_imp.apply_enhanced_imputation_strategy_4step(
            df.copy(), sector_column="sector", n_neighbors=3, price_column="last_price"
        )

        self.assertEqual(legacy_res.shape, new_res.shape)
        self.assertSetEqual(set(legacy_res.columns), set(new_res.columns))


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
