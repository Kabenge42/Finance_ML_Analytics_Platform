import unittest

from finance_ml.ml_workflow.preprocessing.etl import ETLConfig
from finance_ml.ml_workflow.preprocessing.etl_presets import (
    get_etl_config_comprehensive,
    get_etl_config_quick,
)


class TestETLPresets(unittest.TestCase):
    def test_comprehensive_factory_sets_semantic_defaults(self):
        cfg = get_etl_config_comprehensive()

        self.assertIsInstance(cfg, ETLConfig)
        self.assertTrue(cfg.normalize_columns)
        self.assertTrue(cfg.validate_schema)
        self.assertGreaterEqual(cfg.validation.schema_alignment_threshold, 0.80)
        self.assertTrue(cfg.use_semantic_column_classification)
        self.assertTrue(cfg.apply_imputation)
        self.assertTrue(cfg.apply_feature_engineering)
        self.assertEqual(cfg.feature_preset, "comprehensive")

    def test_quick_factory_prioritizes_speed(self):
        cfg = get_etl_config_quick()

        self.assertIsInstance(cfg, ETLConfig)
        self.assertTrue(cfg.normalize_columns)
        # Quick preset should skip heavy validation/feature steps
        self.assertFalse(cfg.validate_schema)
        self.assertFalse(cfg.apply_feature_engineering)
        self.assertFalse(cfg.apply_feature_selection)


if __name__ == "__main__":
    unittest.main()
