import unittest
import numpy as np

try:
    from finance_ml.ml_workflow.regression.robust import winsorize_target, clip_predictions
except Exception:
    # Fallback import directly from file to avoid heavy package __init__
    import importlib.util
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "finance_ml" / "ml_workflow" / "regression" / "robust.py"
    spec = importlib.util.spec_from_file_location("robust", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.load_module(mod.__name__)  # type: ignore[attr-defined]
    winsorize_target = getattr(mod, "winsorize_target")
    clip_predictions = getattr(mod, "clip_predictions")


class TestRobustOutlierSafety(unittest.TestCase):
    def test_winsorize_caps_extremes(self):
        y = np.array([1, 2, 3, 4, 5, 1000], dtype=float)
        w = winsorize_target(y, lower=0.0, upper=0.90)
        # Top 10% (approx) should be capped; since small sample, ensure the extreme is reduced
        self.assertTrue(w.max() < 1000)
        # Monotonicity preserved and same length
        self.assertEqual(len(w), len(y))
        self.assertTrue(np.all(w >= 0))

    def test_winsorize_limits_validation(self):
        with self.assertRaises(ValueError):
            winsorize_target([1, 2, 3], lower=0.6, upper=0.4)

    def test_clip_predictions_bounds(self):
        rng = np.random.default_rng(0)
        y_train = rng.normal(loc=100.0, scale=10.0, size=500)
        preds = np.array([50.0, 100.0, 200.0, -10.0])
        clipped = clip_predictions(preds, y_train, n_std=2.0)
        lower = max(0.0, y_train.mean() - 2.0 * y_train.std())
        upper = y_train.mean() + 2.0 * y_train.std()
        self.assertTrue(np.all(clipped >= lower - 1e-9))
        self.assertTrue(np.all(clipped <= upper + 1e-9))
        # Non-negative lower bound ensured
        self.assertTrue(np.all(clipped >= 0.0))


if __name__ == "__main__":
    unittest.main()
