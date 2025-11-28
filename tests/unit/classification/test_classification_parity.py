"""
Parity tests for legacy classification.py vs consolidated classification/models.py.

TDD Phase 2.1: Ensure function signatures match and core behavior is equivalent
for the duplicated functions before migrating classification.py to a shim.

Scope: Focus on functions confirmed as duplicates in the restructuring plan.
We keep output checks lightweight and deterministic to avoid heavy deps.
"""

import inspect
import unittest
import numpy as np
import pandas as pd


DUPLICATE_FUNCTIONS = [
    "_prepare_categorical_features",
    "prepare_classification_data",
    "train_xgboost_classifier",
    "train_lightgbm_classifier",
    "train_catboost_classifier",
    "train_svm_classifier",
    "train_neural_network_classifier",
    "train_voting_classifier",
    "train_stacking_classifier",
    "apply_smote",
    "apply_adasyn",
    "apply_undersampling",
    "apply_combined_sampling",
    "export_classification_features",
    "clean_extreme_values",
    "validate_data_quality",
    "compare_classifiers",
]


class TestClassificationParitySignatures(unittest.TestCase):
    def test_signatures_match(self):
        import finance_ml.ml_workflow.classification as legacy
        from finance_ml.ml_workflow.classification import models as newmod

        for fn_name in DUPLICATE_FUNCTIONS:
            self.assertTrue(hasattr(legacy, fn_name), f"legacy missing {fn_name}")
            self.assertTrue(hasattr(newmod, fn_name), f"models missing {fn_name}")

            sig_legacy = inspect.signature(getattr(legacy, fn_name))
            sig_new = inspect.signature(getattr(newmod, fn_name))

            # Compare parameter names and kinds (allow default differences)
            params_legacy = [(p.name, p.kind) for p in sig_legacy.parameters.values()]
            params_new = [(p.name, p.kind) for p in sig_new.parameters.values()]
            self.assertEqual(
                params_legacy,
                params_new,
                f"Signature params differ for {fn_name}: {sig_legacy} vs {sig_new}",
            )


class TestClassificationParityOutputs(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0, 1e12],  # extreme value
                "num2": [10.0, 20.0, 30.0, 40.0],
                "cat": ["A", "B", "A", "B"],
            }
        )
        self.labels = np.array([0, 1, 0, 1])

    def test_clean_extreme_values_equivalence(self):
        import finance_ml.ml_workflow.classification as legacy
        from finance_ml.ml_workflow.classification import models as newmod

        out_legacy = legacy.clean_extreme_values(self.df.copy(), clip_threshold=1e6)
        out_new = newmod.clean_extreme_values(self.df.copy(), clip_threshold=1e6)

        # Ensure both clip the extreme value similarly and preserve columns
        self.assertListEqual(list(out_legacy.columns), list(out_new.columns))
        self.assertAlmostEqual(out_legacy["num1"].max(), out_new["num1"].max(), places=6)

    def test_export_classification_features_equivalence(self):
        import finance_ml.ml_workflow.classification as legacy
        from finance_ml.ml_workflow.classification import models as newmod

        # Create a simple 2-class probability array for 4 samples
        y_proba = np.array(
            [
                [0.7, 0.3],
                [0.4, 0.6],
                [0.8, 0.2],
                [0.55, 0.45],
            ]
        )

        out_legacy = legacy.export_classification_features(
            self.df.copy(), y_proba, class_names=["c0", "c1"]
        )
        out_new = newmod.export_classification_features(
            self.df.copy(), y_proba, class_names=["c0", "c1"]
        )

        # Compare resulting columns
        self.assertListEqual(list(out_legacy.columns), list(out_new.columns))

        # Compare any probability columns present (naming may vary between implementations)
        proba_cols = [c for c in out_legacy.columns if c.lower().startswith("proba")]
        for col in proba_cols:
            self.assertTrue(
                np.allclose(out_legacy[col].values, out_new[col].values),
                msg=f"Mismatch in probability column {col}",
            )

    def test_prepare_classification_data_basic_shapes(self):
        # Keep this test light: just verify both return the same split shapes and column sets.
        import finance_ml.ml_workflow.classification as legacy
        from finance_ml.ml_workflow.classification import models as newmod

        ret_legacy = legacy.prepare_classification_data(
            self.df.copy(), self.labels, test_size=0.5, random_state=42
        )
        ret_new = newmod.prepare_classification_data(
            self.df.copy(), self.labels, test_size=0.5, random_state=42
        )

        # Support both 4-tuple and 6-tuple return styles by slicing first 4 items
        X_train_l, X_test_l, y_train_l, y_test_l = ret_legacy[:4]
        X_train_n, X_test_n, y_train_n, y_test_n = ret_new[:4]

        self.assertEqual(X_train_l.shape, X_train_n.shape)
        self.assertEqual(X_test_l.shape, X_test_n.shape)
        self.assertEqual(y_train_l.shape, y_train_n.shape)
        self.assertEqual(y_test_l.shape, y_test_n.shape)
        self.assertSetEqual(set(X_train_l.columns), set(X_train_n.columns))


if __name__ == "__main__":
    unittest.main()
