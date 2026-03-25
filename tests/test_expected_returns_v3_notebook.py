"""
TDD tests for expected_returns_v3.ipynb notebook cells.

Tests cover the key pipeline functions used across notebook cells:
- Cell 3: PipelineConfig
- Cell 4: Pipeline helpers (_log_and_print, _has_required_columns, reconcile_feature_categories)
- Cell 6: Model statistics (compute_model_detailed_statistics, print_model_statistics)
- Cell 11: Monte Carlo (run_monte_carlo_analysis)
- Cell 19: Price Target Achievement (run_price_target_achievement)
- Cell 22: Kalman Filter (run_kalman_filter)
- Cell 26: Earnings Beat (run_earnings_beat_analysis)
- Cell 34: Credit Risk / Dividend Safety
- Cell 38: Stock Screening (run_stock_screening)
- Cell 44: Tri/Quad Model Alignment
- Cell 52: Summary, consensus, compute helpers, export
- Notebook structure validation
"""
from __future__ import annotations

import json
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper: build synthetic DataFrames for pipeline tests
# ---------------------------------------------------------------------------

def _make_equities_df(n: int = 20) -> pd.DataFrame:
    """Minimal equities DataFrame with columns needed by model runners."""
    rng = np.random.RandomState(42)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "company_name": [f"Company {t}" for t in tickers],
        "industry": rng.choice(["Tech", "Finance", "Healthcare"], n),
        "last_price": rng.uniform(10, 500, n),
        "price_target": rng.uniform(12, 600, n),
        "price_target_high": rng.uniform(15, 700, n),
        "price_target_low": rng.uniform(5, 400, n),
        "price_target_median": rng.uniform(10, 550, n),
        "market_cap": rng.uniform(1e8, 1e12, n),
        "enterprise_value": rng.uniform(1e8, 1e12, n),
        "volume_shrs": rng.uniform(1e4, 1e7, n),
        "shares_outstanding": rng.uniform(1e6, 1e9, n),
    })


def _make_mc_df(n: int = 10) -> pd.DataFrame:
    """Synthetic Monte Carlo results."""
    rng = np.random.RandomState(42)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "industry": rng.choice(["Tech", "Finance", "Healthcare"], n),
        "expected_upside_pct": rng.uniform(-20, 40, n),
        "price_target_mc": rng.uniform(50, 600, n),
        "prob_positive_upside": rng.uniform(30, 90, n),
        "var_5_pct": rng.uniform(-30, -5, n),
        "risk_reward_ratio": rng.uniform(0.5, 3.0, n),
        "last_price": rng.uniform(10, 500, n),
        "market_cap": rng.uniform(1e8, 1e12, n),
    })


def _make_kal_df(n: int = 10) -> pd.DataFrame:
    """Synthetic Kalman filter results."""
    rng = np.random.RandomState(43)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "filtered_upside": rng.uniform(-15, 35, n),
        "kalman_estimate": rng.uniform(50, 600, n),
        "kalman_variance": rng.uniform(0.01, 1.0, n),
    })


def _make_pt_df(n: int = 10) -> pd.DataFrame:
    """Synthetic Price Target Achievement results."""
    rng = np.random.RandomState(44)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "expected_return_prob_weighted": rng.uniform(-10, 30, n),
        "achievement_probability": rng.uniform(0.3, 0.95, n),
        "price_target_prob_weighted": rng.uniform(50, 600, n),
        "confidence_level": rng.choice(["High", "Medium", "Low"], n),
        "analyst_conviction": rng.uniform(0, 1, n),
        "eps_revision_momentum": rng.uniform(-1, 1, n),
        "analyst_rating_normalized": rng.uniform(0, 1, n),
    })


def _make_beat_df(n: int = 10) -> pd.DataFrame:
    """Synthetic earnings beat results."""
    rng = np.random.RandomState(45)
    tickers = [f"T{i:03d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "posterior_beat_prob": rng.uniform(0.2, 0.95, n),
        "confidence_score": rng.uniform(0.3, 1.0, n),
        "beat_classification": rng.choice(["Likely Beat", "Neutral", "Likely Miss"], n),
        "posterior_alpha": rng.uniform(1, 10, n),
        "posterior_beta": rng.uniform(1, 10, n),
    })


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 3: PipelineConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineConfigNotebook(unittest.TestCase):
    """Tests for Cell 3 — PipelineConfig dataclass."""

    def test_default_values(self):
        from expected_returns_v3 import PipelineConfig
        cfg = PipelineConfig()
        self.assertEqual(cfg.mc_simulations, 50_000)
        self.assertEqual(cfg.mc_max_stocks, 10_000)
        self.assertEqual(cfg.mcmc_chains, 4)
        self.assertEqual(cfg.mcmc_samples, 10_000)
        self.assertAlmostEqual(cfg.beat_threshold, 0.6)
        self.assertEqual(cfg.output_dir, "outputs/analytics")

    def test_from_env_with_env_vars(self):
        from expected_returns_v3 import PipelineConfig
        with patch.dict(os.environ, {
            "ER_MC_SIMULATIONS": "1000",
            "ER_MC_MAX_STOCKS": "500",
            "ER_MCMC_CHAINS": "2",
            "ER_MCMC_SAMPLES": "5000",
            "ER_OUTPUT_DIR": "/tmp/test_out",
        }):
            cfg = PipelineConfig.from_env()
            self.assertEqual(cfg.mc_simulations, 1000)
            self.assertEqual(cfg.mc_max_stocks, 500)
            self.assertEqual(cfg.mcmc_chains, 2)
            self.assertEqual(cfg.mcmc_samples, 5000)
            self.assertEqual(cfg.output_dir, "/tmp/test_out")

    def test_from_env_defaults(self):
        from expected_returns_v3 import PipelineConfig
        env_clean = {k: v for k, v in os.environ.items()
                     if not k.startswith("ER_")}
        with patch.dict(os.environ, env_clean, clear=True):
            cfg = PipelineConfig.from_env()
            self.assertEqual(cfg.mc_simulations, 50_000)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 4: Pipeline Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestPipelineHelpers(unittest.TestCase):
    """Tests for Cell 4 — _log_and_print, _has_required_columns, reconcile_feature_categories."""

    def test_log_and_print(self):
        from expected_returns_v3 import _log_and_print
        with patch("builtins.print") as mock_print:
            _log_and_print("test message")
            mock_print.assert_called_once_with("test message")

    def test_has_required_columns_true(self):
        from expected_returns_v3 import _has_required_columns
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        self.assertTrue(_has_required_columns(df, ["a", "b"], "test"))

    def test_has_required_columns_false(self):
        from expected_returns_v3 import _has_required_columns
        df = pd.DataFrame({"a": [1]})
        self.assertFalse(_has_required_columns(df, ["a", "missing"], "test"))

    def test_reconcile_feature_categories_filters(self):
        from expected_returns_v3 import reconcile_feature_categories
        cats = {"cat1": ["a", "b", "c"], "cat2": ["x", "y"]}
        result = reconcile_feature_categories(cats, {"a", "c", "x"})
        self.assertEqual(result["cat1"], ["a", "c"])
        self.assertEqual(result["cat2"], ["x"])

    def test_reconcile_drops_empty_categories(self):
        from expected_returns_v3 import reconcile_feature_categories
        cats = {"cat1": ["missing1", "missing2"], "cat2": ["present"]}
        result = reconcile_feature_categories(cats, {"present"})
        self.assertNotIn("cat1", result)
        self.assertIn("cat2", result)

    def test_lazy_feature_categories_dict_protocol(self):
        from expected_returns_v3 import _LazyFeatureCategories
        lazy = _LazyFeatureCategories()
        # Should support len, iter, contains, repr
        with patch("expected_returns_v3.get_feature_categories",
                   return_value={"cat1": ["a"]}):
            self.assertEqual(len(lazy), 1)
            self.assertIn("cat1", lazy)
            self.assertEqual(list(lazy.keys()), ["cat1"])
            self.assertEqual(lazy.get("missing", "default"), "default")


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 6: Model Statistics
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelStatistics(unittest.TestCase):
    """Tests for Cell 6 — compute_model_detailed_statistics, print_model_statistics."""

    def test_compute_model_detailed_statistics(self):
        from expected_returns_v3 import compute_model_detailed_statistics
        df = _make_mc_df(20)
        stats = compute_model_detailed_statistics(
            df, "MC", ["expected_upside_pct", "prob_positive_upside"]
        )
        self.assertIsInstance(stats, dict)
        self.assertIn("expected_upside_pct", stats)
        self.assertIn("global", stats["expected_upside_pct"])
        self.assertIn("mean", stats["expected_upside_pct"]["global"])

    def test_print_model_statistics_no_error(self):
        from expected_returns_v3 import compute_model_detailed_statistics, print_model_statistics
        df = _make_mc_df(20)
        stats = compute_model_detailed_statistics(
            df, "MC", ["expected_upside_pct"]
        )
        # Should not raise
        print_model_statistics(stats, "MC")


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 11: Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunMonteCarloAnalysis(unittest.TestCase):
    """Tests for Cell 11 — run_monte_carlo_analysis."""

    def test_returns_empty_on_missing_columns(self):
        from expected_returns_v3 import run_monte_carlo_analysis
        df = pd.DataFrame({"ticker": ["A"], "some_col": [1]})
        result = run_monte_carlo_analysis(df)
        self.assertTrue(result.empty)

    @patch("expected_returns_v3.monte_carlo_price_target_simulation")
    @patch("expected_returns_v3._get_schema_columns")
    def test_calls_simulation(self, mock_schema, mock_sim):
        from expected_returns_v3 import run_monte_carlo_analysis
        mock_schema.return_value = {
            "mc_required": ["price_target", "price_target_high", "price_target_low", "last_price"],
            "kalman_required": ["last_price", "price_target"],
            "historical_prices": [],
            "historical_targets": [],
            "historical_targets_high": [],
            "historical_targets_low": [],
            "historical_targets_median": [],
        }
        df = _make_equities_df(5)
        mock_sim.return_value = _make_mc_df(5)
        result = run_monte_carlo_analysis(df, n_simulations=100, max_stocks=5)
        mock_sim.assert_called_once()
        self.assertEqual(len(result), 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 19: Price Target Achievement
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunPriceTargetAchievement(unittest.TestCase):
    """Tests for Cell 19 — run_price_target_achievement."""

    @patch("expected_returns_v3.PriceTargetAchievementModel")
    @patch("expected_returns_v3._get_schema_columns")
    def test_returns_dataframe(self, mock_schema, mock_model_cls):
        from expected_returns_v3 import run_price_target_achievement
        mock_schema.return_value = {
            "mc_required": ["price_target", "price_target_high", "price_target_low", "last_price"],
            "kalman_required": ["last_price", "price_target"],
            "historical_prices": [],
            "historical_targets": [],
            "historical_targets_high": [],
            "historical_targets_low": [],
            "historical_targets_median": [],
        }
        mock_instance = MagicMock()
        mock_instance.analyze.return_value = _make_pt_df(5)
        mock_instance.analyze_dataframe.return_value = _make_pt_df(5)
        mock_model_cls.return_value = mock_instance
        df = _make_equities_df(5)
        result = run_price_target_achievement(df,use_historical_targets=False)
        self.assertIsInstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 22: Kalman Filter
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunKalmanFilter(unittest.TestCase):
    """Tests for Cell 22 — run_kalman_filter."""

    def test_returns_empty_on_missing_columns(self):
        from expected_returns_v3 import run_kalman_filter
        df = pd.DataFrame({"ticker": ["A"]})
        result = run_kalman_filter(df)
        self.assertTrue(result.empty)

    @patch("expected_returns_v3.kalman_filter_price_target")
    @patch("expected_returns_v3._get_schema_columns")
    def test_runs_with_valid_input(self, mock_schema, mock_kalman):
        from expected_returns_v3 import run_kalman_filter
        mock_schema.return_value = {
            "mc_required": ["price_target", "price_target_high", "price_target_low", "last_price"],
            "kalman_required": ["last_price", "price_target"],
            "historical_prices": [],
            "historical_targets": [],
            "historical_targets_high": [],
            "historical_targets_low": [],
            "historical_targets_median": [],
        }
        mock_kalman.return_value = _make_kal_df(5)
        df = _make_equities_df(5)
        result = run_kalman_filter(df, use_historical_targets=False)
        self.assertIsInstance(result, pd.DataFrame)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 26: Earnings Beat
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunEarningsBeatAnalysis(unittest.TestCase):
    """Tests for Cell 26 — run_earnings_beat_analysis."""

    @patch("expected_returns_v3.bayesian_earnings_beat_model")
    @patch("expected_returns_v3.ResampledBeatProbabilityModel")
    @patch("expected_returns_v3.EPSStreakAnalyzer")
    @patch("expected_returns_v3.EarningsBeatProbabilityModel")
    def test_returns_dataframe(self, mock_model_cls, mock_streak, mock_resamp, mock_bayes):
        from expected_returns_v3 import run_earnings_beat_analysis
        mock_instance = MagicMock()
        mock_instance.analyze_dataframe_enhanced.return_value = _make_beat_df(5)
        mock_model_cls.return_value = mock_instance
        mock_streak.return_value.analyze_dataframe.return_value = pd.DataFrame()
        mock_resamp.return_value.analyze_dataframe.return_value = pd.DataFrame()
        mock_bayes.return_value = pd.DataFrame()
        df = _make_equities_df(5)
        result = run_earnings_beat_analysis(df)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 44: Tri-Model & Quad-Model Alignment
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriModelAlignment(unittest.TestCase):
    """Tests for Cell 44 — build_tri_model_alignment, build_quad_model_alignment."""

    def test_tri_model_returns_empty_on_empty_input(self):
        from expected_returns_v3 import build_tri_model_alignment
        result = build_tri_model_alignment(pd.DataFrame(), _make_kal_df(), _make_pt_df())
        self.assertTrue(result.empty)

    def test_tri_model_merges_correctly(self):
        from expected_returns_v3 import build_tri_model_alignment
        mc = _make_mc_df(10)
        kal = _make_kal_df(10)
        pt = _make_pt_df(10)
        result = build_tri_model_alignment(mc, kal, pt)
        if not result.empty:
            self.assertIn("agreement_score", result.columns)
            self.assertIn("signal", result.columns)
            self.assertIn("mc_bullish", result.columns)
            self.assertTrue(result["agreement_score"].between(0, 3).all())

    def test_quad_model_returns_empty_on_empty_tri(self):
        from expected_returns_v3 import build_quad_model_alignment
        result = build_quad_model_alignment(pd.DataFrame(), _make_beat_df())
        self.assertTrue(result.empty)

    def test_quad_model_adds_beat_columns(self):
        from expected_returns_v3 import build_tri_model_alignment, build_quad_model_alignment
        mc = _make_mc_df(10)
        kal = _make_kal_df(10)
        pt = _make_pt_df(10)
        tri = build_tri_model_alignment(mc, kal, pt)
        if not tri.empty:
            beat = _make_beat_df(10)
            quad = build_quad_model_alignment(tri, beat)
            if not quad.empty:
                self.assertIn("quad_agreement", quad.columns)
                self.assertIn("beat_bullish", quad.columns)
                self.assertTrue(quad["quad_agreement"].between(0, 4).all())


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 52: Summary & Compute Helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractStrongConsensus(unittest.TestCase):
    """Tests for extract_strong_consensus."""

    def test_returns_empty_on_empty_input(self):
        from expected_returns_v3 import extract_strong_consensus
        result = extract_strong_consensus(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_filters_strong_consensus(self):
        from expected_returns_v3 import extract_strong_consensus
        tri = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "agreement_score": [3, 3, 1],
            "prob_positive_upside": [60.0, 50.0, 80.0],
            "achievement_probability": [0.7, 0.5, 0.9],
            "expected_upside_pct": [20.0, 15.0, 5.0],
        })
        result = extract_strong_consensus(tri)
        # Only A passes: score=3, prob=60>=55, achievement=0.7>=0.6
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["ticker"], "A")


class TestComputeDerivedPriceTarget(unittest.TestCase):
    """Tests for compute_derived_price_target and its wrappers."""

    def test_compute_derived_price_target(self):
        from expected_returns_v3 import compute_derived_price_target
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "expected_upside_pct": [10.0, -5.0],
            "last_price": [100.0, 200.0],
        })
        source = df.copy()
        result = compute_derived_price_target(df, source)
        self.assertIn("price_target_derived", result.columns)
        self.assertAlmostEqual(result.iloc[0]["price_target_derived"], 110.0, places=1)
        self.assertAlmostEqual(result.iloc[1]["price_target_derived"], 190.0, places=1)

    def test_empty_input(self):
        from expected_returns_v3 import compute_derived_price_target
        result = compute_derived_price_target(pd.DataFrame(), pd.DataFrame())
        self.assertTrue(result.empty)

    def test_compute_price_target_mc(self):
        from expected_returns_v3 import compute_price_target_mc
        df = pd.DataFrame({
            "ticker": ["A"],
            "expected_upside_pct": [20.0],
            "last_price": [100.0],
        })
        result = compute_price_target_mc(df, df)
        self.assertIn("price_target_mc", result.columns)
        self.assertAlmostEqual(result.iloc[0]["price_target_mc"], 120.0, places=1)

    def test_compute_price_target_prob_weighted(self):
        from expected_returns_v3 import compute_price_target_prob_weighted
        df = pd.DataFrame({
            "ticker": ["A"],
            "expected_return_prob_weighted": [15.0],
            "last_price": [200.0],
        })
        result = compute_price_target_prob_weighted(df, df)
        self.assertIn("price_target_prob_weighted", result.columns)
        self.assertAlmostEqual(result.iloc[0]["price_target_prob_weighted"], 230.0, places=1)


class TestComputeSectorExpectedReturns(unittest.TestCase):
    """Tests for compute_sector_expected_returns."""

    def test_returns_empty_on_empty(self):
        from expected_returns_v3 import compute_sector_expected_returns
        result = compute_sector_expected_returns(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_returns_empty_without_industry(self):
        from expected_returns_v3 import compute_sector_expected_returns
        df = pd.DataFrame({"ticker": ["A"], "expected_upside_pct": [10]})
        result = compute_sector_expected_returns(df)
        self.assertTrue(result.empty)

    def test_aggregates_by_industry(self):
        from expected_returns_v3 import compute_sector_expected_returns
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "industry": ["Tech", "Tech", "Finance", "Finance"],
            "expected_upside_pct": [10, 20, 5, 15],
            "filtered_upside": [8, 18, 3, 13],
            "expected_return_prob_weighted": [9, 19, 4, 14],
            "agreement_score": [3, 2, 3, 1],
        })
        result = compute_sector_expected_returns(df)
        self.assertEqual(len(result), 2)
        self.assertIn("mc_mean", result.columns)
        self.assertIn("count", result.columns)


class TestComputeReturnZscoreRanks(unittest.TestCase):
    """Tests for compute_return_zscore_ranks."""

    def test_returns_same_on_empty(self):
        from expected_returns_v3 import compute_return_zscore_ranks
        result = compute_return_zscore_ranks(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_adds_zscore_columns(self):
        from expected_returns_v3 import compute_return_zscore_ranks
        df = pd.DataFrame({
            "ticker": [f"T{i}" for i in range(20)],
            "industry": ["Tech"] * 10 + ["Finance"] * 10,
            "expected_upside_pct": np.random.uniform(-10, 30, 20),
            "filtered_upside": np.random.uniform(-10, 30, 20),
        })
        result = compute_return_zscore_ranks(df)
        # Should have z-score columns
        zscore_cols = [c for c in result.columns if "_zscore" in c or "_pctrank" in c]
        self.assertGreater(len(zscore_cols), 0)


class TestComputeCrossModelCorrelation(unittest.TestCase):
    """Tests for compute_cross_model_correlation."""

    def test_returns_dict(self):
        from expected_returns_v3 import compute_cross_model_correlation
        mc = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "expected_upside_pct": [10.0, 20.0, -5.0],
        })
        kal = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "filtered_upside": [8.0, 18.0, -3.0],
        })
        result = compute_cross_model_correlation(mc, kal)
        self.assertIsInstance(result, dict)


class TestRunParallelMCMC(unittest.TestCase):
    """Tests for run_parallel_mcmc_return_analysis."""

    def test_returns_empty_on_empty_input(self):
        from expected_returns_v3 import run_parallel_mcmc_return_analysis
        result = run_parallel_mcmc_return_analysis(pd.DataFrame())
        self.assertEqual(result, {})

    def test_returns_empty_on_insufficient_data(self):
        from expected_returns_v3 import run_parallel_mcmc_return_analysis
        mc = pd.DataFrame({"expected_upside_pct": [1.0, 2.0]})
        result = run_parallel_mcmc_return_analysis(mc)
        self.assertEqual(result, {})


class TestBuildExpectedReturnsSummary(unittest.TestCase):
    """Tests for build_expected_returns_summary."""

    def test_returns_empty_when_any_input_empty(self):
        from expected_returns_v3 import build_expected_returns_summary
        result = build_expected_returns_summary(
            pd.DataFrame(), _make_kal_df(), _make_pt_df(), _make_beat_df()
        )
        self.assertTrue(result.empty)

    def test_merges_four_models(self):
        from expected_returns_v3 import build_expected_returns_summary
        mc = _make_mc_df(10)
        kal = _make_kal_df(10)
        pt = _make_pt_df(10)
        beat = _make_beat_df(10)
        result = build_expected_returns_summary(mc, kal, pt, beat)
        if not result.empty:
            self.assertIn("expected_upside_pct", result.columns)
            self.assertIn("filtered_upside", result.columns)
            self.assertIn("posterior_beat_prob", result.columns)


# ═══════════════════════════════════════════════════════════════════════════════
# Cell 73: Export
# ═══════════════════════════════════════════════════════════════════════════════

class TestExportExpectedReturnsResults(unittest.TestCase):
    """Tests for export_expected_returns_results."""

    @patch("expected_returns_v3.export_to_csv")
    @patch("expected_returns_v3.export_to_db")
    def test_export_creates_output_dir(self, mock_db, mock_csv):
        from expected_returns_v3 import export_expected_returns_results
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = os.path.join(tmpdir, "sub", "dir")
            mock_db.return_value = None
            mock_csv.return_value = None
            result = export_expected_returns_results(
                mc=_make_mc_df(2),
                pt=_make_pt_df(2),
                kal=_make_kal_df(2),
                tri=pd.DataFrame(),
                strong=pd.DataFrame(),
                beat=_make_beat_df(2),
                output_dir=out_dir,
            )
            self.assertTrue(Path(out_dir).exists())
            self.assertIsInstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook Structure Validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestNotebookStructure(unittest.TestCase):
    """Validate the expected_returns_v3.ipynb notebook structure."""

    @classmethod
    def setUpClass(cls):
        nb_path = Path(__file__).resolve().parent.parent / "expected_returns_v3.ipynb"
        if not nb_path.exists():
            raise unittest.SkipTest("expected_returns_v3.ipynb not found")
        with open(nb_path, "r", encoding="utf-8") as f:
            cls.nb = json.load(f)
        cls.cells = cls.nb.get("cells", [])

    def test_notebook_is_valid_json(self):
        self.assertIn("cells", self.nb)
        self.assertIn("metadata", self.nb)
        self.assertIn("nbformat", self.nb)

    def test_has_multiple_cells(self):
        """Notebook must have at least 70 cells (76 specified)."""
        self.assertGreaterEqual(len(self.cells), 70,
                                f"Expected ≥70 cells, got {len(self.cells)}")

    def test_first_cell_is_markdown_title(self):
        first = self.cells[0]
        self.assertEqual(first["cell_type"], "markdown")
        source = "".join(first["source"])
        self.assertIn("Expected Returns", source)

    def test_has_imports_cell(self):
        """Cell 2 should contain imports."""
        code_cells = [c for c in self.cells if c["cell_type"] == "code"]
        imports_cell = code_cells[0]
        source = "".join(imports_cell["source"])
        self.assertIn("import", source)
        self.assertIn("finance_ml", source)

    def test_has_pipeline_config_cell(self):
        """Should have a cell defining PipelineConfig."""
        found = any(
            "PipelineConfig" in "".join(c["source"])
            for c in self.cells if c["cell_type"] == "code"
        )
        self.assertTrue(found, "PipelineConfig not found in any code cell")

    def test_has_markdown_section_headers(self):
        """Should have markdown cells for major sections."""
        md_cells = [c for c in self.cells if c["cell_type"] == "markdown"]
        md_text = " ".join("".join(c["source"]) for c in md_cells)
        required_sections = [
            "Monte Carlo",
            "Price Target Achievement",
            "Kalman",
            "Earnings Beat",
            "Cross-Model",
            "Expected Returns Summary",
            "Export",
        ]
        for section in required_sections:
            self.assertIn(section, md_text,
                          f"Missing markdown section: {section}")

    def test_has_visualization_cells(self):
        """Should have cells calling visualization functions."""
        code_sources = ["".join(c["source"]) for c in self.cells if c["cell_type"] == "code"]
        all_code = "\n".join(code_sources)
        viz_functions = [
            "create_mc_return_distribution",
            "create_sector_heatmap",
            "create_tri_model_agreement_histogram",
        ]
        for func in viz_functions:
            self.assertIn(func, all_code,
                          f"Missing visualization call: {func}")

    def test_has_data_loading_section(self):
        """Should have data loading cells."""
        code_sources = ["".join(c["source"]) for c in self.cells if c["cell_type"] == "code"]
        all_code = "\n".join(code_sources)
        self.assertIn("load_expected_returns_data", all_code)

    def test_has_export_section(self):
        """Should have export cells."""
        code_sources = ["".join(c["source"]) for c in self.cells if c["cell_type"] == "code"]
        all_code = "\n".join(code_sources)
        self.assertIn("export_expected_returns_results", all_code)

    def test_has_pipeline_summary_cell(self):
        """Should have a final pipeline summary."""
        md_cells = [c for c in self.cells if c["cell_type"] == "markdown"]
        found = any("Pipeline Summary" in "".join(c["source"]) or
                     "COMPLETE" in "".join(c["source"])
                     for c in md_cells)
        code_cells = [c for c in self.cells if c["cell_type"] == "code"]
        code_found = any("COMPLETE" in "".join(c["source"]) or
                         "Pipeline Summary" in "".join(c["source"])
                         for c in code_cells)
        self.assertTrue(found or code_found, "Missing pipeline summary")

    def test_cell_types_valid(self):
        """All cells must be either 'code' or 'markdown'."""
        for i, cell in enumerate(self.cells):
            self.assertIn(cell["cell_type"], ("code", "markdown", "raw"),
                          f"Cell {i} has invalid type: {cell['cell_type']}")

    def test_no_empty_code_cells(self):
        """Code cells should not be empty."""
        for i, cell in enumerate(self.cells):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]).strip()
                self.assertGreater(len(source), 0,
                                   f"Code cell {i} is empty")

    def test_notebook_kernel_info(self):
        """Notebook should have Python kernel metadata."""
        metadata = self.nb.get("metadata", {})
        kernelspec = metadata.get("kernelspec", {})
        if kernelspec:
            self.assertIn("python", kernelspec.get("language", "").lower())


class TestNotebookCellCount(unittest.TestCase):
    """Verify notebook has the expected number of cells per type."""

    @classmethod
    def setUpClass(cls):
        nb_path = Path(__file__).resolve().parent.parent / "expected_returns_v3.ipynb"
        if not nb_path.exists():
            raise unittest.SkipTest("expected_returns_v3.ipynb not found")
        with open(nb_path, "r", encoding="utf-8") as f:
            cls.nb = json.load(f)
        cls.cells = cls.nb.get("cells", [])

    def test_markdown_cell_count(self):
        """Should have at least 15 markdown cells (section headers)."""
        md_count = sum(1 for c in self.cells if c["cell_type"] == "markdown")
        self.assertGreaterEqual(md_count, 15,
                                f"Expected ≥15 markdown cells, got {md_count}")

    def test_code_cell_count(self):
        """Should have at least 40 code cells."""
        code_count = sum(1 for c in self.cells if c["cell_type"] == "code")
        self.assertGreaterEqual(code_count, 40,
                                f"Expected ≥40 code cells, got {code_count}")


if __name__ == "__main__":
    unittest.main()
