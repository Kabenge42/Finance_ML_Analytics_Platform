import unittest

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None

try:
    import finance_ml as mod
except Exception:
    mod = None


@unittest.skipIf(pd is None or mod is None or np is None, "pandas/numpy not installed")
class TestEventClassification(unittest.TestCase):
    """Phase 3: Event classification tests per IMPROVEMENT_PLAN.md"""

    def test_create_event_labels_basic(self):
        """Test event label creation from price target and analyst rating changes.

        Enhanced 5-class system (Phase 9.4):
        0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive
        """
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0],
                "price_target": [12.0, 19.0, 18.0, 25.0, 26.0],
                "analyst_rating": [5.0, 3.0, 4.5, 3.5, 2.0],
            }
        )
        labels = mod.create_event_labels(df)
        # Should return array of 0-4: 5-class system (Phase 9.4 enhanced)
        # 0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(label in [0, 1, 2, 3, 4] for label in labels))
        # Verify labels are valid integers in expected range
        self.assertTrue(all(isinstance(label, (int, np.integer)) for label in labels))

    def test_create_event_labels_with_volatility(self):
        """Test event label creation incorporating volatility spikes"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "last_price": [10.0, 20.0, 15.0],
                "price_target": [11.0, 21.0, 16.0],
                "price_volatility_30d": [0.5, 5.0, 1.0],  # B has high volatility spike
            }
        )
        labels = mod.create_event_labels(df, use_volatility=True)
        self.assertEqual(len(labels), 3)
        # Row 1 (B) has volatility spike (>3.0) → should influence classification
        # Implementation detail: high volatility might be considered negative catalyst

    def test_train_event_classifier_basic(self):
        """Test training a basic event classifier"""
        # Create synthetic dataset with clear patterns
        np.random.seed(42)
        n = 100
        df = pd.DataFrame(
            {
                "ticker": [f"T{i}" for i in range(n)],
                "sector": np.random.choice(["Tech", "Energy", "Finance"], n),
                "last_price": np.random.uniform(10, 100, n),
                "price_target": np.random.uniform(10, 100, n),
                "analyst_rating": np.random.uniform(1, 5, n),
                "feature_1": np.random.randn(n),
                "feature_2": np.random.randn(n),
            }
        )
        labels = mod.create_event_labels(df)

        # Train classifier - returns a dict with model and metrics
        result = mod.train_event_classifier(df, labels)

        # Should return dict with model and metrics (Phase 9.4 standardized format)
        self.assertIsNotNone(result)
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        # Phase 9.4 uses 'f1_score' key (not 'f1_macro')
        self.assertIn("f1_score", result)
        self.assertIsNotNone(result["model"])
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
