"""
Smoke tests for hypothesis/testing facade module.

These tests verify that the new focused import paths under
finance_ml.ml_workflow.evaluation.hypothesis provide access to the
expected functions while analytics.eval is being decomposed.
"""

import unittest


class TestHypothesisFacade(unittest.TestCase):
    def test_imports_and_callables(self):
        from finance_ml.ml_workflow.evaluation import hypothesis as hyp

        # Verify attributes exist and are callables
        for name in [
            "test_normality",
            "calculate_skewness_kurtosis",
            "detect_outliers_statistical",
            "compare_two_groups",
            "compare_sector_means",
            "perform_comprehensive_hypothesis_tests",
        ]:
            self.assertTrue(hasattr(hyp, name), f"Missing {name} in hypothesis facade")
            self.assertTrue(callable(getattr(hyp, name)), f"{name} is not callable")


if __name__ == "__main__":
    unittest.main()
