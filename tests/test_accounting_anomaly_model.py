import unittest
import warnings

import pandas as pd

from finance_ml.analytics.probability_analytics import AccountingAnomalyProbabilityModel


class TestAccountingAnomalyProbabilityModel(unittest.TestCase):
    def test_analyze_dataframe_returns_expected_columns(self):
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "sector": ["Tech", "Tech", "Health"],
            "exceptional_items_frequency": [0.1, 0.8, 0.3],
            "gaap_adj_eps_gap_pct": [5.0, 50.0, 10.0],
        })
        model = AccountingAnomalyProbabilityModel()
        result = model.analyze_dataframe(df)
        for col in ["anomaly_severity_score", "anomaly_risk_rank",
                     "sector_anomaly_percentile", "multi_flag_alert"]:
            self.assertIn(col, result.columns)

    def test_custom_thresholds(self):
        df = pd.DataFrame({
            "ticker": ["X"], "sector": ["Fin"],
            "exceptional_items_frequency": [0.5],
        })
        model = AccountingAnomalyProbabilityModel(
            severity_anomaly_weight=0.5,
            severity_feature_weight=0.5,
            multi_flag_threshold=1,
        )
        result = model.analyze_dataframe(df)
        self.assertEqual(len(result), 1)

    def test_default_field_values(self):
        model = AccountingAnomalyProbabilityModel()
        self.assertAlmostEqual(model.severity_anomaly_weight, 0.7)
        self.assertAlmostEqual(model.severity_feature_weight, 0.3)
        self.assertEqual(model.multi_flag_threshold, 3)
        self.assertIsNone(model.anomaly_z_threshold)
        self.assertIsNone(model.tier_bins)
        self.assertIsNone(model.tier_labels)

    def test_deprecated_standalone_function_warns(self):
        from finance_ml.analytics.statistical_analysis import analyze_accounting_anomalies

        df = pd.DataFrame({
            "ticker": ["A"],
            "sector": ["Tech"],
            "exceptional_items_frequency": [0.1],
        })
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            analyze_accounting_anomalies(df)
            self.assertTrue(any(issubclass(x.category, DeprecationWarning) for x in w))

    def test_conditional_probabilities_returns_expected_columns(self):
        """Test that calculate_conditional_probabilities returns the right schema."""
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(20)],
            "sector": ["Tech"] * 10 + ["Health"] * 10,
            "exceptional_items_frequency": [0.1 * i for i in range(20)],
            "gaap_adj_eps_gap_pct": [5.0 + i * 3 for i in range(20)],
        })
        model = AccountingAnomalyProbabilityModel()
        result = model.analyze_dataframe(df)
        cond_probs = model.calculate_conditional_probabilities(result)
        for col in [
            "feature", "p_anomaly_high", "p_anomaly_low",
            "lift_high", "lift_low", "separation", "base_anomaly_rate",
        ]:
            self.assertIn(col, cond_probs.columns)

    def test_anomaly_conditional_probability_column_exists(self):
        """Test that analyze_dataframe adds anomaly_conditional_probability."""
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(20)],
            "sector": ["Tech"] * 10 + ["Health"] * 10,
            "exceptional_items_frequency": [0.1 * i for i in range(20)],
            "gaap_adj_eps_gap_pct": [5.0 + i * 3 for i in range(20)],
        })
        model = AccountingAnomalyProbabilityModel()
        result = model.analyze_dataframe(df)
        self.assertIn("anomaly_conditional_probability", result.columns)
        # Probabilities should be in [0, 1] range
        probs = result["anomaly_conditional_probability"].dropna()
        if len(probs) > 0:
            self.assertTrue((probs >= 0).all())
            self.assertTrue((probs <= 1).all())

    def test_conditional_probabilities_empty_when_no_score(self):
        """Test graceful handling when anomaly_severity_score is absent."""
        df = pd.DataFrame({"ticker": ["A"], "sector": ["Tech"]})
        model = AccountingAnomalyProbabilityModel()
        cond_probs = model.calculate_conditional_probabilities(df)
        self.assertEqual(len(cond_probs), 0)

    def test_separation_sorted_descending(self):
        """Test that conditional probabilities are sorted by separation."""
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(30)],
            "sector": ["Tech"] * 15 + ["Health"] * 15,
            "exceptional_items_frequency": [0.05 * i for i in range(30)],
            "gaap_adj_eps_gap_pct": [2.0 + i * 4 for i in range(30)],
        })
        model = AccountingAnomalyProbabilityModel()
        result = model.analyze_dataframe(df)
        cond_probs = model.calculate_conditional_probabilities(result)
        if len(cond_probs) > 1:
            separations = cond_probs["separation"].tolist()
            self.assertEqual(separations, sorted(separations, reverse=True))

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["ticker", "sector"])
        model = AccountingAnomalyProbabilityModel()
        result = model.analyze_dataframe(df)
        self.assertEqual(len(result), 0)


class TestAnomalySeverityDashboard(unittest.TestCase):
    """Tests for create_anomaly_severity_dashboard in quality_risk.py."""

    def _make_anomaly_df(self, n: int = 20) -> pd.DataFrame:
        """Build a small DataFrame mimicking AccountingAnomalyProbabilityModel output."""
        import numpy as np
        rng = np.random.RandomState(42)
        tiers = ["Clean", "Watch", "Flag", "Alert"]
        return pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "industry": rng.choice(["Tech", "Health", "Finance"], n),
            "accounting_anomaly_score": rng.uniform(0, 100, n),
            "accounting_anomaly_tier": rng.choice(tiers, n),
            "anomaly_severity_score": rng.uniform(0, 100, n),
            "anomaly_conditional_probability": rng.uniform(0, 1, n),
            "anomaly_risk_rank": rng.uniform(0, 100, n),
            "sector_anomaly_percentile": rng.uniform(0, 100, n),
            "multi_flag_alert": rng.choice([True, False], n),
        })

    def test_returns_figure(self):
        from finance_ml.analytics.visualizations.quality_risk import (
            create_anomaly_severity_dashboard,
        )
        import plotly.graph_objects as go

        df = self._make_anomaly_df()
        fig = create_anomaly_severity_dashboard(df)
        self.assertIsInstance(fig, go.Figure)

    def test_no_data_fallback(self):
        from finance_ml.analytics.visualizations.quality_risk import (
            create_anomaly_severity_dashboard,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({"ticker": ["A"]})
        fig = create_anomaly_severity_dashboard(df)
        self.assertIsInstance(fig, go.Figure)

    def test_without_optional_columns(self):
        """Dashboard should not crash when optional columns are absent."""
        from finance_ml.analytics.visualizations.quality_risk import (
            create_anomaly_severity_dashboard,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "anomaly_severity_score": [40.0, 80.0],
        })
        fig = create_anomaly_severity_dashboard(df)
        self.assertIsInstance(fig, go.Figure)


class TestAnomalyConditionalProbabilityChart(unittest.TestCase):
    """Tests for create_anomaly_conditional_probability_chart in probability_viz.py."""

    def _make_anomaly_df(self, n: int = 20) -> pd.DataFrame:
        import numpy as np
        rng = np.random.RandomState(42)
        tiers = ["Clean", "Watch", "Flag", "Alert"]
        return pd.DataFrame({
            "ticker": [f"T{i}" for i in range(n)],
            "industry": rng.choice(["Tech", "Health", "Finance"], n),
            "accounting_anomaly_score": rng.uniform(0, 100, n),
            "accounting_anomaly_tier": rng.choice(tiers, n),
            "anomaly_severity_score": rng.uniform(0, 100, n),
            "anomaly_conditional_probability": rng.uniform(0, 1, n),
        })

    def test_returns_figure(self):
        from finance_ml.analytics.visualizations.probability_viz import (
            create_anomaly_conditional_probability_chart,
        )
        import plotly.graph_objects as go

        df = self._make_anomaly_df()
        fig = create_anomaly_conditional_probability_chart(df)
        self.assertIsInstance(fig, go.Figure)

    def test_no_data_fallback(self):
        from finance_ml.analytics.visualizations.probability_viz import (
            create_anomaly_conditional_probability_chart,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({"ticker": ["A"]})
        fig = create_anomaly_conditional_probability_chart(df)
        self.assertIsInstance(fig, go.Figure)

    def test_with_explicit_cond_probs(self):
        from finance_ml.analytics.visualizations.probability_viz import (
            create_anomaly_conditional_probability_chart,
        )
        import plotly.graph_objects as go

        df = self._make_anomaly_df()
        cond_probs = pd.DataFrame({
            "feature": ["feat_a", "feat_b"],
            "p_anomaly_high": [0.8, 0.6],
            "p_anomaly_low": [0.2, 0.3],
            "lift_high": [1.6, 1.2],
            "lift_low": [0.4, 0.6],
            "separation": [0.6, 0.3],
            "base_anomaly_rate": [0.5, 0.5],
        })
        fig = create_anomaly_conditional_probability_chart(df, cond_probs=cond_probs)
        self.assertIsInstance(fig, go.Figure)

    def test_without_tier_column(self):
        from finance_ml.analytics.visualizations.probability_viz import (
            create_anomaly_conditional_probability_chart,
        )
        import plotly.graph_objects as go

        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "anomaly_conditional_probability": [0.3, 0.7],
            "anomaly_severity_score": [30.0, 70.0],
        })
        fig = create_anomaly_conditional_probability_chart(df)
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
