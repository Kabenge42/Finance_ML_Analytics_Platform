"""
Simple smoke test to demonstrate how to run the test suite and collect coverage.
This test is intentionally trivial and has no external dependencies.
"""
import unittest


class TestCoverageSmoke(unittest.TestCase):
    def test_arithmetic(self) -> None:
        self.assertEqual(1 + 1, 2)

    def test_import_package(self) -> None:
        try:
            import finance_ml  # noqa: F401
        except Exception as e:  # pragma: no cover - test should not fail here
            self.fail(f"Importing finance_ml raised an exception: {e}")


if __name__ == "__main__":
    unittest.main()
