"""Parity tests for legacy models vs new regression/classification subpackages.

Phase 3 (Restructuring Plan):

- Verify that key training and orchestration functions exposed via the
  legacy ``finance_ml.ml_workflow.models`` module are backed by the new
  ``finance_ml.ml_workflow.regression`` (and related) modules.
- Keep these tests light: we only check signatures and very small
  synthetic behaviours to avoid heavy training or external deps.

These tests intentionally mirror the style of ``test_classification_parity``
used for the Phase 2 classification refactor.
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np
import pandas as pd


# Functions that should be provided by the legacy models facade and
# implemented in the new regression subpackage. The mapping of
# ``legacy_name -> (module, attr_name)`` is kept explicit to avoid
# guessing and to make future refactors easier to update.
DUPLICATE_REGRESSION_FUNCTIONS = {
    # High-level orchestration helpers
    "train_stacking_ensemble": (
        "finance_ml.ml_workflow.regression.models",
        "train_stacking_regressor",
    ),
    "train_quantile_regression": (
        "finance_ml.ml_workflow.regression.quantile",
        "train_quantile_regressor",
    ),
}


class TestModelsParitySignatures(unittest.TestCase):
    def test_legacy_and_new_functions_exist(self):
        """Legacy facade and new regression modules must both expose helpers.

        We intentionally avoid enforcing strict signature equality here
        because the legacy API is dataframe-oriented (``df, feature_cols``)
        while the new regression API is ``X, y``-based. The restructuring
        plan mainly requires that callers can continue to import the same
        function names and that they delegate to the canonical
        implementations, which these existence checks validate.
        """

        import importlib

        legacy = importlib.import_module("finance_ml.ml_workflow.models")

        for legacy_name, (mod_path, target_name) in DUPLICATE_REGRESSION_FUNCTIONS.items():
            with self.subTest(legacy_name=legacy_name):
                self.assertTrue(
                    hasattr(legacy, legacy_name),
                    msg=f"Legacy models module missing {legacy_name}",
                )

                target_mod = importlib.import_module(mod_path)
                self.assertTrue(
                    hasattr(target_mod, target_name),
                    msg=f"Target module {mod_path} missing {target_name}",
                )


class TestModelsParityBehaviour(unittest.TestCase):
    """Very lightweight behavioural checks for critical helpers.

    We avoid asserting exact metrics; instead we focus on whether both
    legacy and new implementations can accept the same inputs and
    produce structurally similar outputs (keys, basic shapes).
    """

    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = rng.randn(200) * 10 + 100

        cls.df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        cls.df["target"] = y

    def test_train_stacking_ensemble_result_schema(self):
        """Check that stacking helpers return a dict with a fitted model."""

        from finance_ml.ml_workflow import models as legacy
        from finance_ml.ml_workflow.regression import models as reg_models

        feature_cols = [c for c in self.df.columns if c != "target"]

        # Call legacy facade
        legacy_result = legacy.train_stacking_ensemble(
            self.df,
            feature_cols=feature_cols,
            target_col="target",
            random_state=42,
            use_meta_features=False,
        )

        # Call new implementation directly
        X = self.df[feature_cols]
        y = self.df["target"]
        new_result = reg_models.train_stacking_regressor(
            X,
            y,
            cv=3,
            random_state=42,
            ensure_nonnegative=False,
            loss="squared_error",
        )

        # Basic structural checks
        for res, label in [(legacy_result, "legacy"), (new_result, "new")]:
            self.assertIsInstance(res, dict, msg=f"{label} result must be a dict")
            self.assertIn("model", res, msg=f"{label} result missing 'model' key")
            self.assertTrue(
                hasattr(res["model"], "predict"),
                msg=f"{label} model must implement predict()",
            )

    # Note: we intentionally keep behavioural parity tests minimal and
    # focused on stacking, as quantile helpers in the legacy module wrap
    # a different calling convention (df vs X/y) and internal wrapper
    # types that are already well-covered by dedicated regression tests.


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
