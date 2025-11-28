"""
Smoke tests for reporting facade module.

These tests verify that the new focused import paths under
finance_ml.ml_workflow.evaluation.reports expose the expected functions while
analytics.eval is being decomposed.
"""

import unittest


class TestReportsFacade(unittest.TestCase):
    def test_imports_and_callables(self):
        from finance_ml.ml_workflow.evaluation import reports

        for name in [
            "generate_sector_comparison_report",
            "generate_data_quality_dashboard",
            "export_profiling_report",
        ]:
            self.assertTrue(hasattr(reports, name), f"Missing {name} in reports facade")
            self.assertTrue(callable(getattr(reports, name)), f"{name} is not callable")


if __name__ == "__main__":
    unittest.main()
