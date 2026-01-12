import unittest

import numpy as np
import pandas as pd

from finance_ml.ml_workflow.classification.labels import (
    LABEL_FEATURE_REGISTRY,
    get_features_for_label_method,
    validate_label_method,
    get_label_method_info,
    analyze_label_quality,
    create_enhanced_event_labels,
)


class TestLabelsRefactoring(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "ticker": ["A", "B", "C", "D", "E"],
                "sector": ["Tech", "Tech", "Finance", "Finance", "Energy"],
                "roe": [0.2, 0.15, 0.1, 0.05, 0.01],
                "roa": [0.1, 0.08, 0.05, 0.02, 0.005],
                "debt_to_equity": [0.1, 0.5, 1.0, 2.0, 5.0],
                "p_e_ratio": [10, 15, 20, 25, 30],
            }
        )

    def test_registry_exists(self):
        self.assertIsInstance(LABEL_FEATURE_REGISTRY, dict)
        self.assertIn("price_momentum", LABEL_FEATURE_REGISTRY)

    def test_get_features_for_label_method(self):
        features = get_features_for_label_method("profitability_event")
        self.assertIn("roe", features)
        self.assertIn("roa", features)

    def test_validate_label_method(self):
        validate_label_method("price_momentum")
        with self.assertRaises(ValueError):
            validate_label_method("invalid_method")

    def test_get_label_method_info(self):
        info = get_label_method_info("valuation")
        self.assertEqual(info["method"], "valuation")
        self.assertIn("description", info)
        self.assertIn("categories", info)

    def test_analyze_label_quality(self):
        labels = np.array([4, 3, 2, 1, 0])
        quality = analyze_label_quality(self.df, labels, "profitability_event")
        self.assertEqual(quality["total_samples"], 5)
        self.assertIn("class_distribution", quality)
        self.assertTrue(quality["balanced"])

    def test_enhanced_labels_use_registry_profitability(self):
        # This will test if the refactored create_enhanced_event_labels works as expected
        labels = create_enhanced_event_labels(self.df, method="profitability_event")
        self.assertEqual(len(labels), 5)
        self.assertTrue(all(l in [0, 1, 2, 3, 4] for l in labels))
        # High ROE should be 4
        self.assertEqual(labels[0], 4)
        # Low ROE should be 0
        self.assertEqual(labels[4], 0)

    def test_enhanced_labels_use_registry_leverage(self):
        labels = create_enhanced_event_labels(self.df, method="leverage_event")
        # Low debt should be positive (leverage higher_is_better = False)
        self.assertEqual(labels[0], 4)
        self.assertEqual(labels[4], 0)
