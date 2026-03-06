"""Tests for the extended InferenceData schema (Part 9/10 of inference_schema.py)."""
import unittest

import numpy as np
import pandas as pd


class TestIdentifierCoordinates(unittest.TestCase):
    def test_from_dataframe_basic(self):
        from finance_ml.analytics.inference_schema import IdentifierCoordinates

        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "sector": ["Tech", "Tech"],
            "region": ["NA", "NA"],
            "isin": ["US1", "US2"],
        })
        ic = IdentifierCoordinates.from_dataframe(df)
        self.assertEqual(len(ic.tickers), 2)
        self.assertEqual(len(ic.regions), 2)

    def test_to_equity_coordinates(self):
        from finance_ml.analytics.inference_schema import IdentifierCoordinates

        df = pd.DataFrame({"ticker": ["A"], "isin": ["X"], "name": ["Agilent"]})
        ic = IdentifierCoordinates.from_dataframe(df)
        ec = ic.to_equity_coordinates()
        self.assertEqual(len(ec.tickers), 1)
        self.assertEqual(ec.names[0], "Agilent")

    def test_to_xarray_coords(self):
        from finance_ml.analytics.inference_schema import IdentifierCoordinates

        df = pd.DataFrame({"ticker": ["A", "B"], "sector": ["S1", "S2"]})
        ic = IdentifierCoordinates.from_dataframe(df)
        coords = ic.to_xarray_coords()
        self.assertIn("equity", coords)
        self.assertIn("sector", coords)

    def test_missing_ticker_raises(self):
        from finance_ml.analytics.inference_schema import IdentifierCoordinates

        df = pd.DataFrame({"sector": ["Tech"]})
        with self.assertRaises(ValueError):
            IdentifierCoordinates.from_dataframe(df)


class TestEquitiesSchemaMetadata(unittest.TestCase):
    def test_role_filters(self):
        from finance_ml.analytics.inference_schema import EquitiesSchemaMetadata

        df = pd.DataFrame({
            "column_name": ["Ticker", "Sector", "Price"],
            "column_alias": ["ticker", "sector", "last_price"],
            "role": ["id", "categorical", "price"],
            "column_type": ["text", "text", "numeric"],
        })
        esm = EquitiesSchemaMetadata.from_dataframe(df)
        self.assertEqual(esm.id_columns(), ["ticker"])
        self.assertEqual(esm.categorical_columns(), ["sector"])
        self.assertEqual(esm.numeric_columns(), ["last_price"])
        self.assertEqual(esm.date_columns(), [])


class TestFeatureRegistryMetadata(unittest.TestCase):
    def test_functions_for_category(self):
        from finance_ml.analytics.inference_schema import FeatureRegistryMetadata

        df = pd.DataFrame({
            "function_name": ["fn_a", "fn_b", "fn_c"],
            "category": ["Growth", "Growth", "Value"],
            "feature_count": [3, 5, 2],
        })
        frm = FeatureRegistryMetadata.from_dataframe(df)
        self.assertEqual(frm.functions_for_category("Growth"), ["fn_a", "fn_b"])
        self.assertEqual(frm.functions_for_category("Value"), ["fn_c"])


class TestFeatureViewSpec(unittest.TestCase):
    def test_to_xarray_dataset(self):
        from finance_ml.analytics.inference_schema import FeatureViewSpec

        spec = FeatureViewSpec(
            view_name="vw_features_momentum",
            category="Momentum",
            feature_columns=["rsi_14", "macd_signal"],
        )
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "rsi_14": [55.0, 60.0],
            "macd_signal": [0.1, -0.2],
        })
        ds = spec.to_xarray_dataset(df)
        self.assertIn("rsi_14", ds.data_vars)
        self.assertEqual(ds.sizes["equity"], 2)


class TestFeatureViewRegistry(unittest.TestCase):
    def test_registry_has_17_views(self):
        from finance_ml.analytics.inference_schema import FEATURE_VIEW_REGISTRY

        self.assertEqual(len(FEATURE_VIEW_REGISTRY), 17)
        self.assertIn("vw_features_valuation_ratios", FEATURE_VIEW_REGISTRY)


class TestBuildFeatureViewInferenceData(unittest.TestCase):
    def test_builds_posterior(self):
        from finance_ml.analytics.inference_schema import build_feature_view_inference_data

        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "col1": [1.0, 2.0, 3.0],
            "col2": [4.0, 5.0, 6.0],
        })
        result = build_feature_view_inference_data(
            "vw_features_momentum", df, n_posterior_samples=10, n_chains=2
        )
        # Should return InferenceData or Dataset
        self.assertIsNotNone(result)


class TestEquitiesMaterializedViewSpec(unittest.TestCase):
    def test_from_dataframe(self):
        from finance_ml.analytics.inference_schema import EquitiesMaterializedViewSpec

        df = pd.DataFrame({
            "ticker": ["A"],
            "last_price": [100.0],
            "price_target_median": [120.0],
            "ema_50": [95.0],
            "total_revenues": [1e9],
            "last_price_1m_ago": [98.0],
        })
        spec = EquitiesMaterializedViewSpec.from_dataframe(df)
        self.assertIn("last_price", spec.price_columns)
        self.assertIn("ema_50", spec.price_columns)
        self.assertIn("price_target_median", spec.price_target_columns)
        self.assertIn("total_revenues", spec.financial_columns)
        self.assertIn("last_price_1m_ago", spec.historical_price_columns)


if __name__ == "__main__":
    unittest.main()
