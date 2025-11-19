import unittest
from unittest.mock import patch

import pandas as pd


class TestPhase93ApiMigration(unittest.TestCase):
    """
    Phase 9.3 API Migration tests (TDD): ensure engineer_features uses the
    modern build_features() API with the comprehensive preset and the
    include_relative parameter, replacing the legacy features_build_comprehensive().

    This test is designed to fail before the migration is applied because
    ml_finance_model_main will not expose build_features in its namespace
    nor call it. After migration, it should pass.
    """

    def test_engineer_features_calls_build_features_with_expected_args(self):
        # Lazy import to ensure we patch symbols in the loaded module
        import ml_finance_model_main as main

        # Minimal input dataframe; actual feature generation is stubbed/mocked
        df = pd.DataFrame(
            {
                "sector": ["Tech", "Health"],
                "price_target": [100.0, 110.0],
                "last_price": [95.0, 120.0],
            }
        )

        cfg = main.PipelineConfig(have_advanced_analytics=False)

        called = {}

        def fake_build_features(
            df_in, *, preset, include_interactions, include_relative, sector_col
        ):
            # Capture arguments for assertion and return dataframe with a marker column
            called["args"] = (preset, include_interactions, include_relative, sector_col)
            return df_in.assign(dummy_feature=1)

        def fake_importance(df_in, target_col, n_features):
            # Lightweight importance stub to avoid heavy computation
            return {"dummy_feature": 1.0}

        # Patch build_features inside the script namespace and feature importance to a stub
        with (
            patch("ml_finance_model_main.build_features", side_effect=fake_build_features),
            patch("ml_finance_model_main.features_importance_rf", side_effect=fake_importance),
        ):
            df_out, meta = main.engineer_features(df, cfg)

        # Assert our stub was used and received the expected arguments
        self.assertIn("args", called, "build_features was not called by engineer_features")
        self.assertEqual(
            called["args"],
            ("comprehensive", True, True, "sector"),
            "build_features must be called with preset='comprehensive', include_interactions=True, include_relative=True, sector_col='sector'",
        )

        # Output dataframe should include the dummy feature returned by fake_build_features
        self.assertIn("dummy_feature", df_out.columns)

        # Metadata should include expected keys
        self.assertIsInstance(meta, dict)
        self.assertIn("feature_names", meta)
        self.assertIn("n_features", meta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
