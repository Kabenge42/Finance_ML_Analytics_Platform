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
        """Test event label creation from price target and analyst rating changes"""
        df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "last_price": [10.0, 20.0, 15.0, 25.0, 30.0],
                "price_target": [12.0, 19.0, 18.0, 25.0, 26.0],
                "analyst_rating": [5.0, 3.0, 4.5, 3.5, 2.0],
            }
        )
        labels = mod.create_event_labels(df)
        # Should return array of 0/1/2: 0=Neutral, 1=Positive, 2=Negative
        # Labels use ±10% threshold: Positive >= +10%, Negative <= -10%, else Neutral
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(label in [0, 1, 2] for label in labels))
        # Row 0: price_target > last_price by 20% (12-10)/10 = +20% → Positive (1)
        self.assertEqual(labels[0], 1)
        # Row 1: price_target < last_price by 5% (19-20)/20 = -5% → Neutral (0) - below 10% threshold
        self.assertEqual(labels[1], 0)
        # Row 2: price_target > last_price by 20% (18-15)/15 = +20% → Positive (1)
        self.assertEqual(labels[2], 1)
        # Row 3: price_target == last_price (0%) → Neutral (0)
        self.assertEqual(labels[3], 0)
        # Row 4: price_target < last_price by 13.3% (26-30)/30 = -13.3% → Negative (2)
        self.assertEqual(labels[4], 2)

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

        # Should return dict with model and metrics
        self.assertIsNotNone(result)
        self.assertIn("model", result)
        self.assertIn("accuracy", result)
        self.assertIn("f1_macro", result)
        self.assertIsNotNone(result["model"])
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
