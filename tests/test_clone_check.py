import unittest
from sklearn.base import clone
from sklearn.linear_model import Ridge
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper


class TestClone(unittest.TestCase):
    def test_clone_wrapper(self):
        base = Ridge()
        wrapper = NonNegativeRegressionWrapper(base)
        try:
            cloned = clone(wrapper)
            print("Clone successful")
        except Exception as e:
            print(f"Clone failed: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
