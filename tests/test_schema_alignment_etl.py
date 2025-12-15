"""Tests for ETL schema alignment metrics.

These tests validate the scope-aware schema alignment behavior introduced for
Phase 9.1 Stage 11 (schema validation) when ETL runs after feature engineering.
"""

import unittest

import pandas as pd


try:
    from finance_ml.ml_workflow.preprocessing.etl import ETLPipeline
    from finance_ml.ml_workflow.data.schema import list_required_schema_columns_for_etl
except ImportError:
    ETLPipeline = None
    list_required_schema_columns_for_etl = None


class TestSchemaAlignmentETL(unittest.TestCase):
    def setUp(self):
        if ETLPipeline is None or list_required_schema_columns_for_etl is None:
            self.skipTest("ETL/schema modules not available")

    def _minimal_required_df(self) -> pd.DataFrame:
        required = list_required_schema_columns_for_etl(include_extended_financials=False)

        # Build a minimal DataFrame with required columns
        df = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB", "CCC"],
                "isin": ["US0000000001", "US0000000002", "US0000000003"],
                "sector": pd.Series(["Tech", "Tech", "Tech"], dtype="category"),
                "region": pd.Series(["US", "US", "US"], dtype="category"),
                "country": pd.Series(
                    ["United States", "United States", "United States"],
                    dtype="category",
                ),
                "trading_country": pd.Series(
                    ["United States", "United States", "United States"],
                    dtype="category",
                ),
                "last_price": [10.0, 20.0, 30.0],
                "price_target": [11.0, 22.0, 33.0],
                "price_target_median": [10.5, 21.0, 31.5],
                "price_target_ytd_ago": [9.0, 18.0, 27.0],
                "market_cap": [1_000_000.0, 2_000_000.0, 3_000_000.0],
                "enterprise_value": [1_200_000.0, 2_300_000.0, 3_400_000.0],
            }
        )

        # Ensure required columns are present (defensive)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise AssertionError(f"Test fixture missing required columns: {missing}")

        return df

    def test_schema_alignment_allows_phase93_features_and_meets_thresholds(self):
        pipeline = ETLPipeline()
        df = self._minimal_required_df()

        # Add a handful of Phase 9.3 engineered features (should not be treated as unknown)
        df = df.assign(
            rsi_14d=[50.0, 55.0, 60.0],
            price_momentum_1m=[0.02, -0.01, 0.03],
            ev_ebitda_ratio=[12.0, 10.5, 14.2],
        )

        result = pipeline._validate_schema_alignment(df)

        self.assertGreaterEqual(
            result["alignment_score"],
            0.95,
            f"Expected alignment_score >= 0.95, got {result['alignment_score']:.4f}",
        )
        self.assertEqual(
            result["missing_expected_columns"],
            [],
            f"Missing required columns unexpectedly: {result['missing_expected_columns']}",
        )
        self.assertEqual(
            result["unknown_columns"],
            [],
            f"Phase 9.3 features should be allowlisted; unknown: {result['unknown_columns']}",
        )

    def test_schema_alignment_penalizes_truly_unknown_columns(self):
        pipeline = ETLPipeline()
        df = self._minimal_required_df().assign(totally_unknown_feature=[1.0, 2.0, 3.0])

        result = pipeline._validate_schema_alignment(df)

        self.assertIn("totally_unknown_feature", result["unknown_columns"])
        self.assertLess(
            result["alignment_score"],
            0.95,
            "A truly unknown column should reduce recognition rate enough to drop below 0.95",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
