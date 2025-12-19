"""
Python Script/Module Review Checklist (Section 6.2) — Structural TDD tests.

These tests statically analyze Python source text using
finance_ml.ml_workflow.quality.script_review to verify compliance with the
guidelines in docs/code_guidelines.md §6.2.

The analyzer is AST-based and fast; it does not import or execute the target
module. Tests use synthetic modules for precise control and include a smoke test
on a real project file.
"""

from __future__ import annotations

import unittest
from textwrap import dedent
from pathlib import Path

from finance_ml.ml_workflow.quality.script_review import (
    review_python_source,
    review_python_file,
)


class TestScriptModuleReviewChecklist(unittest.TestCase):
    def _issues_codes(self, result):
        return {issue.get("code", "") for issue in result.get("issues", [])}

    def test_good_module_passes_core_checks(self):
        good_module = dedent(
            r'''
            """Example module adhering to guidelines.

            Follows type hints, docstrings with Args/Returns/Raises, logging, and
            import grouping order (stdlib → third‑party → local).
            """

            import os
            import sys
            import logging

            import numpy as np
            import pandas as pd

            from finance_ml import __version__  # local package import last

            try:  # optional dependency
                import plotly  # type: ignore
            except ImportError:  # graceful degradation
                plotly = None

            logger = logging.getLogger(__name__)

            # Constants (allowed at module level)
            MODEL_VERSION = "v9_10"

            class DatasetSplit(tuple):
                pass

            def _helper_format_cols(cols: list[str]) -> list[str]:
                """Helper that enforces normalized, lower_snake_case columns.

                Args:
                    cols: List of candidate column names

                Returns:
                    Normalized list of columns

                Raises:
                    ValueError: if a column is not lower_snake_case
                """
                for c in cols:
                    if not c.replace("_", "").islower():
                        raise ValueError("column names must be lower_snake_case")
                return cols

            def make_dataset(n: int) -> DatasetSplit:
                """Create a small dummy dataset.

                Args:
                    n: number of rows to generate

                Returns:
                    DatasetSplit with train_X, train_y, valid_X, valid_y, meta
                """
                assert isinstance(n, int) and n > 0, "n must be positive int"
                columns = _helper_format_cols(["last_price", "price_target"])  # normalized
                return ([], [], [], [], {"columns": columns})

            def train_event_classifier(X: np.ndarray, y: np.ndarray) -> dict:
                """Train an example classifier.

                Args:
                    X: feature matrix
                    y: labels

                Returns:
                    dict with keys: model, metrics, y_pred, y_proba, artifacts

                Raises:
                    ValueError: on invalid inputs
                """
                if X is None or y is None:
                    raise ValueError("X and y are required")
                logger.info("training model…")
                return {
                    "model": object(),
                    "metrics": {"accuracy": 1.0},
                    "y_pred": [],
                    "y_proba": [],
                    "artifacts": {"feature_importance": []},
                }
            '''
        )
        result = review_python_source(good_module, filename="good_module.py")
        self.assertIsInstance(result, dict)
        self.assertIn("issues", result)
        self.assertEqual(
            len(result["issues"]), 0, f"Unexpected issues: {result['issues']}"
        )
        self.assertIn("summary", result)
        self.assertGreaterEqual(result["summary"].get("functions_checked", 0), 2)

    def test_bad_module_reports_common_issues(self):
        bad_module = dedent(
            r"""
            #!/usr/bin/env python3
            # Bad module violating multiple rules
            from finance_ml import something
            import pandas as pd
            import os
            import logging
            import numpy as np

            cache = {}  # global mutable state

            def train_model(X, y):
                print("training")  # print instead of logging
                return {"model": object(), "metrics": {}}  # missing required keys

            def make_dataset(n):
                return (1, 2, 3)  # not a 5‑tuple
            """
        )
        result = review_python_source(bad_module, filename="bad_module.py")
        codes = self._issues_codes(result)
        # Expect several core issues
        self.assertIn("imports_ordering", codes)
        self.assertIn("global_mutable_state", codes)
        self.assertIn("missing_type_hints", codes)
        self.assertIn("print_statements", codes)
        self.assertIn("training_return_schema", codes)
        self.assertIn("dataset_prep_return", codes)

    def test_smoke_review_real_file(self):
        # Smoke test on a real, moderate module; do not enforce zero issues.
        path = Path("finance_ml/ml_workflow/evaluation/stacking.py")
        self.assertTrue(path.exists(), "Expected real module path to exist")
        result = review_python_file(path)
        self.assertIn("issues", result)
        self.assertIn("summary", result)
        self.assertIsInstance(result["issues"], list)


if __name__ == "__main__":
    unittest.main()
