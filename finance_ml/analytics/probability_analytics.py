"""
Probability Analytics Module for Market Analytics

This module provides comprehensive probability and model confidence analysis
for earnings beat predictions, EPS streak analysis, and posterior probability
estimation using Bayesian inference methods.

Features:
- Bayesian Earnings Beat Probability Model with posterior updates
- EPS Streak Analysis with predictive analytics
- Model Confidence Estimation with calibration metrics
- Interactive dashboards for probability visualization

References:
- Bayesian methods for financial forecasting
- Posterior probability estimation techniques
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from finance_ml.analytics.data_utils import export_to_analytics_db

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================


class BeatProbabilityEstimate(TypedDict):
    """Type definition for beat probability estimation results."""

    posterior_alpha: float
    posterior_beta: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_exceeds_threshold: float
    confidence_score: float


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================


@dataclass
class BeatProbabilityResult:
    """Result container for earnings beat probability analysis."""

    ticker: str
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    prior_mean: float
    posterior_mean: float
    posterior_std: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    beat_probability: float
    confidence_score: float
    historical_beat_rate: float
    n_observations: int


@dataclass
class EPSStreakResult:
    """Result container for EPS streak analysis."""

    ticker: str
    current_streak: int
    streak_type: str  # 'beat', 'miss', 'meet'
    max_streak_beat: int
    max_streak_miss: int
    streak_continuation_prob: float
    mean_reversion_prob: float
    expected_next_outcome: str
    confidence_level: float
    historical_pattern: list[str] = field(default_factory=list)


@dataclass
class ModelConfidenceResult:
    """Result container for model confidence estimation."""

    model_name: str
    brier_score: float
    log_loss: float
    calibration_error: float
    discrimination_auc: float
    reliability_diagram_data: dict
    confidence_intervals: dict
    overall_confidence: float


@dataclass(frozen=True)
class PriorParameters:
    """Immutable container for Beta distribution prior parameters."""

    alpha: float
    beta: float

    @property
    def expected_beat_rate(self) -> float:
        """Calculate the expected beat rate from prior parameters."""
        return self.alpha / (self.alpha + self.beta)

    def as_tuple(self) -> tuple[float, float]:
        """Return parameters as (alpha, beta) tuple."""
        return (self.alpha, self.beta)


# =============================================================================
# BAYESIAN EARNINGS BEAT PROBABILITY MODEL
# =============================================================================


class EarningsBeatProbabilityModel:
    """
    Bayesian model for estimating earnings beat probabilities.

    Uses Beta-Binomial conjugate prior framework to compute posterior
    probabilities of earnings beats given historical data. The model
    supports incremental updates as new earnings data becomes available.

    The posterior probability is computed using Bayes' theorem:
    P(beat | data) ∝ P(data | beat) × P(beat)

    With Beta prior: Beta(α, β)
    And Binomial likelihood for beats/misses
    Posterior: Beta(α + beats, β + misses)
    """

    # Quantile values for credible intervals
    CI_90_LOWER_QUANTILE = 0.05
    CI_90_UPPER_QUANTILE = 0.95
    CI_95_LOWER_QUANTILE = 0.025
    CI_95_UPPER_QUANTILE = 0.975

    # Confidence score normalization factor (based on effective sample size)
    CONFIDENCE_NORMALIZATION_FACTOR = 20

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        sector_priors: Optional[dict[str, tuple[float, float]]] = None,
    ):
        """
        Initialize the earnings beat probability model.

        Args:
            prior_alpha: Alpha parameter for Beta prior (default: 2.0 for mild optimism)
            prior_beta: Beta parameter for Beta prior (default: 2.0 for symmetry)
            sector_priors: Optional dict mapping sectors to (alpha, beta) tuples
                          for sector-specific priors based on historical patterns
        """
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.sector_priors = sector_priors or self._default_sector_priors()

    def _default_sector_priors(self) -> dict[str, tuple[float, float]]:
        """
        Default sector-specific priors based on typical beat rates.

        Technology tends to have higher beat rates (~70%), while
        cyclical sectors like Energy have more variable outcomes.
        """
        return {
            "Information Technology": (3.5, 1.5),  # ~70% prior beat rate
            "Health Care": (3.0, 1.5),  # ~67% prior beat rate
            "Consumer Discretionary": (2.5, 1.5),  # ~63% prior beat rate
            "Industrials": (2.5, 1.5),  # ~63% prior beat rate
            "Financials": (2.5, 2.0),  # ~56% prior beat rate
            "Consumer Staples": (2.5, 2.0),  # ~56% prior beat rate
            "Materials": (2.0, 2.0),  # ~50% prior beat rate
            "Energy": (2.0, 2.5),  # ~44% prior beat rate
            "Utilities": (2.0, 2.0),  # ~50% prior beat rate
            "Communication Services": (3.0, 1.5),  # ~67% prior beat rate
            "Real Estate": (2.0, 2.0),  # ~50% prior beat rate
        }

    def _get_prior_parameters(
        self,
        sector: Optional[str],
        use_sector_prior: bool,
    ) -> tuple[float, float]:
        """
        Get the appropriate prior parameters based on sector.

        Args:
            sector: Optional sector name
            use_sector_prior: Whether to use sector-specific priors

        Returns:
            Tuple of (alpha, beta) prior parameters
        """
        if use_sector_prior and sector and sector in self.sector_priors:
            return self.sector_priors[sector]
        return self.prior_alpha, self.prior_beta

    def _compute_posterior_statistics(
        self,
        alpha: float,
        beta: float,
    ) -> tuple[float, float]:
        """
        Compute mean and standard deviation of Beta posterior.

        Args:
            alpha: Posterior alpha parameter
            beta: Posterior beta parameter

        Returns:
            Tuple of (posterior_mean, posterior_std)
        """
        total = alpha + beta
        posterior_mean = alpha / total
        posterior_std = np.sqrt((alpha * beta) / (total**2 * (total + 1)))
        return posterior_mean, posterior_std

    def _compute_credible_intervals(
        self,
        distribution: stats.rv_continuous,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """
        Compute 90% and 95% credible intervals from posterior distribution.

        Args:
            distribution: Scipy Beta distribution object

        Returns:
            Tuple of (ci_90, ci_95) where each is a (lower, upper) tuple
        """
        ci_90 = (
            distribution.ppf(self.CI_90_LOWER_QUANTILE),
            distribution.ppf(self.CI_90_UPPER_QUANTILE),
        )
        ci_95 = (
            distribution.ppf(self.CI_95_LOWER_QUANTILE),
            distribution.ppf(self.CI_95_UPPER_QUANTILE),
        )
        return ci_90, ci_95

    def _compute_confidence_score(self, alpha: float, beta: float) -> float:
        """
        Compute confidence score based on posterior concentration.

        Higher α + β means more concentrated posterior, indicating
        more data and thus higher confidence in the estimate.

        Args:
            alpha: Posterior alpha parameter
            beta: Posterior beta parameter

        Returns:
            Confidence score between 0 and 1
        """
        effective_sample_size = alpha + beta - 2  # Subtract prior pseudo-counts
        return min(1.0, effective_sample_size / self.CONFIDENCE_NORMALIZATION_FACTOR)

    def compute_posterior(
        self,
        n_beats: int,
        n_total: int,
        sector: Optional[str] = None,
        use_sector_prior: bool = True,
    ) -> tuple[float, float]:
        """
        Compute posterior Beta parameters given observed beats.

        Args:
            n_beats: Number of earnings beats observed
            n_total: Total number of earnings observations
            sector: Optional sector for sector-specific prior
            use_sector_prior: Whether to use sector-specific priors

        Returns:
            Tuple of (posterior_alpha, posterior_beta)
        """
        n_misses = n_total - n_beats
        alpha, beta = self._get_prior_parameters(sector, use_sector_prior)

        # Conjugate update: posterior = Beta(α + beats, β + misses)
        posterior_alpha = alpha + n_beats
        posterior_beta = beta + n_misses

        return posterior_alpha, posterior_beta

    def compute_beat_probability(
        self,
        n_beats: int,
        n_total: int,
        sector: Optional[str] = None,
        threshold: float = 0.5,
    ) -> BeatProbabilityEstimate:
        """
        Compute the probability of future earnings beat.

        Args:
            n_beats: Number of historical beats
            n_total: Total observations
            sector: Optional sector name
            threshold: Probability threshold for "likely beat" classification

        Returns:
            Dictionary with probability estimates and confidence metrics
        """
        post_alpha, post_beta = self.compute_posterior(n_beats, n_total, sector)

        posterior_mean, posterior_std = self._compute_posterior_statistics(post_alpha, post_beta)

        dist = stats.beta(post_alpha, post_beta)
        ci_90, ci_95 = self._compute_credible_intervals(dist)

        prob_exceeds_threshold = 1 - dist.cdf(threshold)
        confidence_score = self._compute_confidence_score(post_alpha, post_beta)

        return {
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_90": ci_90,
            "credible_interval_95": ci_95,
            "prob_exceeds_threshold": prob_exceeds_threshold,
            "confidence_score": confidence_score,
        }

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        beats_col: str = "eps_beat_count",
        total_col: str = "eps_total_reports",
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Analyze earnings beat probabilities for entire DataFrame.

        Args:
            df: DataFrame with earnings data
            beats_col: Column name for beat counts
            total_col: Column name for total report counts
            sector_col: Column name for sector
            ticker_col: Column name for ticker

        Returns:
            DataFrame with probability analysis results
        """
        results = []

        for _, row in df.iterrows():
            # Handle missing data
            n_beats = row.get(beats_col, 0)
            n_total = row.get(total_col, 0)
            sector = row.get(sector_col, None)
            ticker = row.get(ticker_col, "UNKNOWN")

            if pd.isna(n_beats) or pd.isna(n_total) or n_total == 0:
                # Use proxy from EPS trajectory if available
                if "eps_trajectory_score" in df.columns and not pd.isna(
                    row.get("eps_trajectory_score")
                ):
                    # Estimate beats from trajectory score (0-100 scale)
                    trajectory = row["eps_trajectory_score"]
                    n_total = 5  # Assume 5 quarters of data
                    n_beats = int(trajectory / 100 * n_total)
                else:
                    continue

            n_beats = int(n_beats)
            n_total = int(n_total)

            if n_total == 0:
                continue

            prob_result = self.compute_beat_probability(n_beats, n_total, sector)

            # Determine classification based on posterior mean vs default threshold
            beat_classification = (
                "likely_beat" if prob_result["posterior_mean"] > 0.5 else "uncertain"
            )

            results.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "historical_beats": n_beats,
                    "total_reports": n_total,
                    "historical_beat_rate": n_beats / n_total,
                    "posterior_beat_prob": prob_result["posterior_mean"],
                    "posterior_std": prob_result["posterior_std"],
                    "ci_90_lower": prob_result["credible_interval_90"][0],
                    "ci_90_upper": prob_result["credible_interval_90"][1],
                    "ci_95_lower": prob_result["credible_interval_95"][0],
                    "ci_95_upper": prob_result["credible_interval_95"][1],
                    "confidence_score": prob_result["confidence_score"],
                    "beat_classification": beat_classification,
                }
            )

        return pd.DataFrame(results)


# =============================================================================
# EPS STREAK ANALYZER
# =============================================================================


class EPSStreakAnalyzer:
    """
    Analyzer for EPS beat/miss streaks with predictive capabilities.

    Uses Markov chain analysis and historical patterns to predict
    streak continuation vs. mean reversion probabilities.
    """

    def __init__(self, mean_reversion_weight: float = 0.3):
        """
        Initialize streak analyzer.

        Args:
            mean_reversion_weight: Weight for mean reversion in predictions (0-1)
                                  Higher values increase mean reversion tendency
        """
        self.mean_reversion_weight = mean_reversion_weight

    def compute_streak_from_trajectory(
        self,
        eps_trajectory_score: float,
        eps_positive_streak: Optional[int] = None,
        eps_improvement_count: Optional[int] = None,
    ) -> EPSStreakResult:
        """
        Compute streak analysis from trajectory score and related metrics.

        Args:
            eps_trajectory_score: EPS trajectory score (0-100)
            eps_positive_streak: Number of positive EPS quarters
            eps_improvement_count: Number of YoY improvements

        Returns:
            EPSStreakResult with analysis
        """
        # Estimate current streak from available metrics
        if eps_positive_streak is not None and not pd.isna(eps_positive_streak):
            current_streak = int(eps_positive_streak)
            streak_type = "beat" if current_streak > 0 else "miss"
        elif eps_trajectory_score is not None and not pd.isna(eps_trajectory_score):
            # Infer from trajectory score
            if eps_trajectory_score >= 80:
                current_streak = 4
                streak_type = "beat"
            elif eps_trajectory_score >= 60:
                current_streak = 3
                streak_type = "beat"
            elif eps_trajectory_score >= 40:
                current_streak = 1
                streak_type = "meet"
            elif eps_trajectory_score >= 20:
                current_streak = 2
                streak_type = "miss"
            else:
                current_streak = 3
                streak_type = "miss"
        else:
            current_streak = 0
            streak_type = "meet"

        # Compute continuation probability using geometric decay model
        # P(continue) = base_rate * decay^streak_length
        base_continuation = 0.65  # Base continuation probability
        decay_factor = 0.85  # Decay per streak length

        continuation_prob = base_continuation * (decay_factor ** abs(current_streak))

        # Apply mean reversion adjustment
        mean_reversion_prob = 1 - continuation_prob
        mean_reversion_prob = mean_reversion_prob * (
            1 - self.mean_reversion_weight
        ) + self.mean_reversion_weight * (1 - continuation_prob)

        # Confidence based on streak length (longer streaks = less confidence in continuation)
        confidence = max(0.3, 1 - abs(current_streak) * 0.1)

        # Expected next outcome
        if continuation_prob > 0.5:
            expected_next = streak_type
        else:
            expected_next = "beat" if streak_type == "miss" else "miss"

        return EPSStreakResult(
            ticker="",  # Set by caller
            current_streak=current_streak,
            streak_type=streak_type,
            max_streak_beat=max(current_streak if streak_type == "beat" else 0, 0),
            max_streak_miss=max(current_streak if streak_type == "miss" else 0, 0),
            streak_continuation_prob=continuation_prob,
            mean_reversion_prob=mean_reversion_prob,
            expected_next_outcome=expected_next,
            confidence_level=confidence,
        )

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        trajectory_col: str = "eps_trajectory_score",
        streak_col: str = "eps_positive_streak",
        improvement_col: str = "eps_improvement_count",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Analyze EPS streaks for entire DataFrame.

        Args:
            df: DataFrame with EPS data
            trajectory_col: Column for trajectory score
            streak_col: Column for positive streak count
            improvement_col: Column for improvement count
            ticker_col: Column for ticker

        Returns:
            DataFrame with streak analysis
        """
        results = []

        for _, row in df.iterrows():
            trajectory = row.get(trajectory_col, None)
            streak = row.get(streak_col, None)
            improvement = row.get(improvement_col, None)
            ticker = row.get(ticker_col, "UNKNOWN")

            if trajectory is None or pd.isna(trajectory):
                continue

            result = self.compute_streak_from_trajectory(
                eps_trajectory_score=trajectory,
                eps_positive_streak=streak,
                eps_improvement_count=improvement,
            )

            results.append(
                {
                    "ticker": ticker,
                    "current_streak": result.current_streak,
                    "streak_type": result.streak_type,
                    "continuation_probability": result.streak_continuation_prob,
                    "mean_reversion_probability": result.mean_reversion_prob,
                    "expected_next_outcome": result.expected_next_outcome,
                    "prediction_confidence": result.confidence_level,
                }
            )

        return pd.DataFrame(results)


# =============================================================================
# MODEL CONFIDENCE ESTIMATOR
# =============================================================================


class ModelConfidenceEstimator:
    """
    Estimator for model confidence and calibration metrics.

    Provides comprehensive confidence assessment including:
    - Brier score for probability calibration
    - Reliability diagrams
    - Confidence interval coverage
    """

    def __init__(self, n_bins: int = 10):
        """
        Initialize confidence estimator.

        Args:
            n_bins: Number of bins for calibration analysis
        """
        self.n_bins = n_bins

    def compute_brier_score(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> float:
        """
        Compute Brier score for probability predictions.

        Brier score = (1/N) * Σ(predicted - actual)²
        Lower is better, 0 = perfect, 0.25 = random for binary.
        """
        return np.mean((predicted_probs - actual_outcomes) ** 2)

    def compute_calibration_error(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> tuple[float, dict]:
        """
        Compute Expected Calibration Error (ECE) and reliability diagram data.

        ECE measures how well predicted probabilities match observed frequencies.
        """
        bins = np.linspace(0, 1, self.n_bins + 1)
        bin_indices = np.digitize(predicted_probs, bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        reliability_data = {
            "bin_centers": [],
            "observed_freq": [],
            "predicted_mean": [],
            "count": [],
        }

        total_samples = len(predicted_probs)
        ece = 0.0

        for i in range(self.n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_pred = predicted_probs[mask].mean()
                bin_actual = actual_outcomes[mask].mean()
                bin_count = mask.sum()

                reliability_data["bin_centers"].append((bins[i] + bins[i + 1]) / 2)
                reliability_data["observed_freq"].append(bin_actual)
                reliability_data["predicted_mean"].append(bin_pred)
                reliability_data["count"].append(bin_count)

                ece += (bin_count / total_samples) * abs(bin_actual - bin_pred)

        return ece, reliability_data

    def compute_confidence_metrics(
        self,
        predicted_probs: np.ndarray,
        actual_outcomes: np.ndarray,
        model_name: str = "Earnings Beat Model",
    ) -> ModelConfidenceResult:
        """
        Compute comprehensive confidence metrics for predictions.

        Args:
            predicted_probs: Array of predicted probabilities
            actual_outcomes: Array of actual binary outcomes (0 or 1)
            model_name: Name for the model

        Returns:
            ModelConfidenceResult with all metrics
        """
        # Brier score
        brier = self.compute_brier_score(predicted_probs, actual_outcomes)

        # Log loss (cross-entropy)
        eps = 1e-15
        clipped_probs = np.clip(predicted_probs, eps, 1 - eps)
        log_loss = -np.mean(
            actual_outcomes * np.log(clipped_probs)
            + (1 - actual_outcomes) * np.log(1 - clipped_probs)
        )

        # Calibration error and reliability data
        ece, reliability_data = self.compute_calibration_error(predicted_probs, actual_outcomes)

        # AUC-ROC for discrimination
        try:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(actual_outcomes, predicted_probs)
        except Exception:
            # Fallback: simple rank-based AUC approximation
            n_pos = actual_outcomes.sum()
            n_neg = len(actual_outcomes) - n_pos
            if n_pos > 0 and n_neg > 0:
                ranks = stats.rankdata(predicted_probs)
                auc = (ranks[actual_outcomes == 1].sum() - n_pos * (n_pos + 1) / 2) / (
                    n_pos * n_neg
                )
            else:
                auc = 0.5

        # Confidence intervals for predictions
        ci_coverage = self._compute_ci_coverage(predicted_probs, actual_outcomes)

        # Overall confidence score (0-100)
        # Weighted combination of metrics
        overall = (
            (1 - brier) * 30  # Lower brier is better
            + (1 - ece) * 30  # Lower ECE is better
            + auc * 40  # Higher AUC is better
        )
        overall = min(100, max(0, overall))

        return ModelConfidenceResult(
            model_name=model_name,
            brier_score=brier,
            log_loss=log_loss,
            calibration_error=ece,
            discrimination_auc=auc,
            reliability_diagram_data=reliability_data,
            confidence_intervals=ci_coverage,
            overall_confidence=overall,
        )

    def _compute_ci_coverage(
        self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray
    ) -> dict:
        """Compute confidence interval coverage rates."""
        # For Beta distribution CIs
        coverage_90 = 0.0
        coverage_95 = 0.0
        n = len(predicted_probs)

        for i, (prob, actual) in enumerate(zip(predicted_probs, actual_outcomes)):
            # Approximate CI from probability
            std = np.sqrt(prob * (1 - prob) / 10)  # Assume 10 observations
            ci_90 = (max(0, prob - 1.645 * std), min(1, prob + 1.645 * std))
            ci_95 = (max(0, prob - 1.96 * std), min(1, prob + 1.96 * std))

            if ci_90[0] <= actual <= ci_90[1]:
                coverage_90 += 1
            if ci_95[0] <= actual <= ci_95[1]:
                coverage_95 += 1

        return {
            "coverage_90": coverage_90 / n if n > 0 else 0,
            "coverage_95": coverage_95 / n if n > 0 else 0,
        }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def create_earnings_probability_dashboard(
    probability_df: pd.DataFrame,
    title: str = "Earnings Beat Probability Analysis",
) -> go.Figure:
    """
    Create comprehensive dashboard for earnings beat probabilities.

    Args:
        probability_df: DataFrame from EarningsBeatProbabilityModel.analyze_dataframe
        title: Dashboard title

    Returns:
        Plotly Figure with probability analysis dashboard
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Posterior Beat Probability Distribution",
            "Confidence Score by Sector",
            "Historical vs Posterior Beat Rate",
            "Probability Classification",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Color scheme matching Global Equity Research Dashboard theme
    colors = {
        "primary": "#0A7EA4",
        "secondary": "#00A878",
        "accent": "#6C63FF",
        "warning": "#FFD93D",
        "danger": "#E63946",
    }

    # 1. Posterior probability histogram
    fig.add_trace(
        go.Histogram(
            x=probability_df["posterior_beat_prob"],
            nbinsx=20,
            name="Posterior P(Beat)",
            marker_color=colors["primary"],
            opacity=0.8,
        ),
        row=1,
        col=1,
    )

    # Add vertical line at 0.5 threshold
    fig.add_vline(x=0.5, line_dash="dash", line_color=colors["danger"], row=1, col=1)

    # 2. Confidence by sector
    if "sector" in probability_df.columns:
        sector_conf = (
            probability_df.groupby("sector")["confidence_score"].mean().sort_values(ascending=True)
        )
        fig.add_trace(
            go.Bar(
                y=sector_conf.index,
                x=sector_conf.values,
                orientation="h",
                name="Avg Confidence",
                marker_color=colors["secondary"],
            ),
            row=1,
            col=2,
        )

    # 3. Historical vs Posterior scatter
    fig.add_trace(
        go.Scatter(
            x=probability_df["historical_beat_rate"],
            y=probability_df["posterior_beat_prob"],
            mode="markers",
            name="Stocks",
            marker=dict(
                size=8,
                color=probability_df["confidence_score"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Confidence", x=0.45),
            ),
            text=probability_df["ticker"],
            hovertemplate="<b>%{text}</b><br>Historical: %{x:.1%}<br>Posterior: %{y:.1%}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Add diagonal reference line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 4. Classification pie chart
    if "beat_classification" in probability_df.columns:
        classification_counts = probability_df["beat_classification"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=classification_counts.index,
                values=classification_counts.values,
                marker_colors=[colors["secondary"], colors["warning"]],
                hole=0.4,
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color="#1A2332")),
        height=700,
        showlegend=False,
        template="plotly_dark",
        font=dict(family="Inter, sans-serif"),
    )

    # Update axes labels
    fig.update_xaxes(title_text="P(Beat)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Confidence Score", row=1, col=2)
    fig.update_xaxes(title_text="Historical Beat Rate", row=2, col=1)
    fig.update_yaxes(title_text="Posterior Beat Probability", row=2, col=1)

    return fig


def create_confidence_calibration_chart(
    confidence_result: ModelConfidenceResult,
) -> go.Figure:
    """
    Create reliability diagram and confidence metrics chart.

    Args:
        confidence_result: ModelConfidenceResult from confidence estimator

    Returns:
        Plotly Figure with calibration analysis
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Reliability Diagram", "Model Confidence Metrics"),
        specs=[[{"type": "scatter"}, {"type": "indicator"}]],
    )

    reliability = confidence_result.reliability_diagram_data

    # Reliability diagram
    if reliability["bin_centers"]:
        fig.add_trace(
            go.Scatter(
                x=reliability["bin_centers"],
                y=reliability["observed_freq"],
                mode="markers+lines",
                name="Observed",
                marker=dict(size=10, color="#0A7EA4"),
            ),
            row=1,
            col=1,
        )

        # Perfect calibration line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Perfect Calibration",
            ),
            row=1,
            col=1,
        )

    # Confidence gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence_result.overall_confidence,
            title={"text": "Model Confidence"},
            delta={"reference": 70},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#0A7EA4"},
                "steps": [
                    {"range": [0, 40], "color": "#E63946"},
                    {"range": [40, 70], "color": "#FFD93D"},
                    {"range": [70, 100], "color": "#00A878"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title=f"Model Calibration: {confidence_result.model_name}",
        height=400,
        template="plotly_dark",
    )

    fig.update_xaxes(title_text="Predicted Probability", row=1, col=1)
    fig.update_yaxes(title_text="Observed Frequency", row=1, col=1)

    return fig


def create_eps_streak_analysis_chart(
    streak_df: pd.DataFrame,
    title: str = "EPS Streak Analysis & Predictions",
) -> go.Figure:
    """
    Create visualization for EPS streak analysis.

    Args:
        streak_df: DataFrame from EPSStreakAnalyzer.analyze_dataframe
        title: Chart title

    Returns:
        Plotly Figure with streak analysis
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Current Streak Distribution",
            "Continuation vs Reversion Probability",
            "Prediction Confidence by Streak Length",
            "Expected Outcomes",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ],
    )

    colors = {"beat": "#00A878", "miss": "#E63946", "meet": "#FFD93D"}

    # 1. Streak distribution
    fig.add_trace(
        go.Histogram(
            x=streak_df["current_streak"],
            nbinsx=15,
            name="Streak Length",
            marker_color="#0A7EA4",
        ),
        row=1,
        col=1,
    )

    # 2. Continuation vs Reversion
    fig.add_trace(
        go.Scatter(
            x=streak_df["continuation_probability"],
            y=streak_df["mean_reversion_probability"],
            mode="markers",
            marker=dict(
                size=8,
                color=[colors.get(t, "#0A7EA4") for t in streak_df["streak_type"]],
            ),
            text=streak_df["ticker"],
            hovertemplate="<b>%{text}</b><br>Continue: %{x:.1%}<br>Revert: %{y:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. Confidence by streak length
    streak_conf = streak_df.groupby("current_streak")["prediction_confidence"].mean()
    fig.add_trace(
        go.Scatter(
            x=streak_conf.index,
            y=streak_conf.values,
            mode="markers+lines",
            marker=dict(size=10, color="#6C63FF"),
            name="Avg Confidence",
        ),
        row=2,
        col=1,
    )

    # 4. Expected outcomes pie
    outcome_counts = streak_df["expected_next_outcome"].value_counts()
    fig.add_trace(
        go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            marker_colors=[colors.get(o, "#0A7EA4") for o in outcome_counts.index],
            hole=0.4,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=24)),
        height=700,
        showlegend=False,
        template="plotly_dark",
    )

    return fig


def export_probability_analytics_results(
    probability_df: pd.DataFrame,
    streak_df: pd.DataFrame,
    output_dir: Path,
    confidence_result: Optional[ModelConfidenceResult] = None,
) -> dict:
    """
    Export all probability analytics results to database and files.

    Args:
        probability_df: DataFrame with probability analysis
        streak_df: DataFrame with streak analysis
        output_dir: Output directory path
        confidence_result: Optional confidence metrics

    Returns:
        Dictionary with export information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = {}

    # 1. Export probability analysis to database
    try:
        export_to_analytics_db(probability_df, "earnings_probability_analysis")
        exports["probability_analysis_db"] = "analytics.earnings_probability_analysis"
    except Exception as e:
        logger.error(f"Failed to export probability analysis to database: {e}")

    # Export probability analysis to CSV (optional fallback/backup)
    prob_path = output_dir / "earnings_beat_probability_analysis.csv"
    probability_df.to_csv(prob_path, index=False)
    exports["probability_analysis_csv"] = str(prob_path)

    # 2. Export streak analysis to database
    try:
        export_to_analytics_db(streak_df, "eps_streak_analysis")
        exports["streak_analysis_db"] = "analytics.eps_streak_analysis"
    except Exception as e:
        logger.error(f"Failed to export streak analysis to database: {e}")

    # Export streak analysis to CSV
    streak_path = output_dir / "eps_streak_analysis.csv"
    streak_df.to_csv(streak_path, index=False)
    exports["streak_analysis_csv"] = str(streak_path)

    # 3. Export confidence metrics to database
    if confidence_result:
        conf_df = pd.DataFrame(
            [
                {
                    "model_name": confidence_result.model_name,
                    "brier_score": confidence_result.brier_score,
                    "log_loss": confidence_result.log_loss,
                    "calibration_error": confidence_result.calibration_error,
                    "discrimination_auc": confidence_result.discrimination_auc,
                    "overall_confidence": confidence_result.overall_confidence,
                }
            ]
        )
        try:
            export_to_analytics_db(conf_df, "model_confidence_metrics")
            exports["confidence_metrics_db"] = "analytics.model_confidence_metrics"
        except Exception as e:
            logger.error(f"Failed to export confidence metrics to database: {e}")

        # Export confidence metrics to CSV
        conf_path = output_dir / "model_confidence_metrics.csv"
        conf_df.to_csv(conf_path, index=False)
        exports["confidence_metrics_csv"] = str(conf_path)

    # 4. Create and export summary statistics to database
    summary_data = {
        "metric": [
            "Total Stocks Analyzed",
            "Mean Posterior Beat Probability",
            "Median Posterior Beat Probability",
            "Stocks Classified as Likely Beat",
            "Mean Confidence Score",
            "Mean Streak Length",
            "Stocks with Beat Streak",
            "Stocks with Miss Streak",
        ],
        "value": [
            float(len(probability_df)),
            float(probability_df["posterior_beat_prob"].mean()),
            float(probability_df["posterior_beat_prob"].median()),
            float((probability_df["beat_classification"] == "likely_beat").sum()),
            float(probability_df["confidence_score"].mean()),
            float(streak_df["current_streak"].abs().mean()),
            float((streak_df["streak_type"] == "beat").sum()),
            float((streak_df["streak_type"] == "miss").sum()),
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    try:
        export_to_analytics_db(summary_df, "probability_analytics_summary")
        exports["summary_db"] = "analytics.probability_analytics_summary"
    except Exception as e:
        logger.error(f"Failed to export summary statistics to database: {e}")

    # Export summary statistics to CSV
    summary_path = output_dir / "probability_analytics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    exports["summary_csv"] = str(summary_path)

    logger.info(f"Exported probability analytics results to database and {output_dir}")
    return exports
