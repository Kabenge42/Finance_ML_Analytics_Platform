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

from finance_ml.analytics.data_utils import (
    export_to_analytics_db,
    load_identifier_columns,
    reorder_with_identifiers,
    ExportConfig,
    export_to_db,
    export_to_csv,
    export_to_json,
)

logger = logging.getLogger(__name__)

# Identifier columns for model output propagation
_IDENTIFIER_COLS_CACHE: list[str] | None = None


def _get_identifier_cols() -> list[str]:
    """Cached access to identifier columns for model output."""
    global _IDENTIFIER_COLS_CACHE
    if _IDENTIFIER_COLS_CACHE is None:
        _IDENTIFIER_COLS_CACHE = load_identifier_columns()
    return _IDENTIFIER_COLS_CACHE


def _extract_identifiers(row: pd.Series) -> dict:
    """Extract all available identifier columns from a DataFrame row."""
    id_cols = _get_identifier_cols()
    return {
        col: row.get(col, None)
        for col in id_cols
        if col in row.index and pd.notna(row.get(col))
    }


# Columns that must be cast to numeric before export (Issue 7)
_NUMERIC_CAST_COLS = [
    "gaap_revision_momentum", "gaap_norm_spread", "revision_trend_short",
    "revision_trend_medium", "eps_norm_est_fy1e", "eps_norm_est_ntm",
    "eps_gaap_est_ntm", "eps_gaap_est_fy1e",
]
_INTEGER_CAST_COLS = ["analyst_count", "quarterly_beat_streak"]

# Lazy ArviZ import (consistent with inference_schema.py)
try:
    import arviz as az
    import xarray as xr

    ARVIZ_AVAILABLE = True
except (ImportError, OSError, PermissionError, Exception):
    az = None  # type: ignore[assignment]
    xr = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False


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
    # NEW: Interpretability enhancements
    prior_influence_pct: float  # How much prior vs data drove the result
    effective_sample_size: float  # Statistical power indicator
    classification_confidence: str  # 'High', 'Medium', 'Low'


# =============================================================================
# DATA CLASSES FOR STRUCTURED RESULTS
# =============================================================================


@dataclass
class CreditRiskResult:
    """Result container for credit risk probability analysis."""

    ticker: str
    name: str
    sector: str
    distress_probability: float
    liquidity_stress_score: float
    cash_runway_months: float
    altman_z_score: float
    risk_level: str  # 'Low', 'Medium', 'High', 'Distressed'
    confidence_interval: tuple[float, float]


@dataclass
class DividendSafetyResult:
    """Result container for dividend safety analysis."""

    ticker: str
    name: str
    dividend_cut_probability: float
    fcf_dividend_coverage: float
    payout_ratio: float
    dividend_streak: int
    safety_score: float
    risk_category: str  # 'Safe', 'Borderline', 'At Risk'


@dataclass
class PriceTargetResult:
    """Result container for price target achievement analysis."""

    ticker: str
    name: str
    achievement_probability: float
    upside_potential: float
    price_target_spread_pct: float
    analyst_rating_normalized: float
    expected_return_prob_weighted: float


@dataclass
class BeatProbabilityResult:
    """Result container for earnings beat probability analysis."""

    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    exchange: str
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
    # Forward estimate fields
    eps_norm_est_ntm: Optional[float] = None
    eps_norm_est_fy1e: Optional[float] = None
    eps_gaap_est_ntm: Optional[float] = None
    eps_gaap_est_fy1e: Optional[float] = None
    gaap_revision_momentum: Optional[float] = None
    gaap_norm_spread: Optional[float] = None
    # Next earnings context
    next_earnings_status: Optional[str] = None
    # Analyst coverage
    analyst_count: Optional[int] = None
    # Dynamic total derived from non-null reported EPS
    dynamic_total_reports: Optional[int] = None


@dataclass
class EPSStreakResult:
    """Result container for EPS streak analysis."""

    ticker: str
    name: str
    sector: str
    industry: str
    country: str
    exchange: str
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

    @property
    def mode(self) -> float | None:
        """Calculate the mode (most likely value) of the distribution."""
        if self.alpha > 1 and self.beta > 1:
            return (self.alpha - 1) / (self.alpha + self.beta - 2)
        return None  # Mode undefined for alpha <= 1 or beta <= 1

    @property
    def variance(self) -> float:
        """Calculate the variance of the distribution."""
        total = self.alpha + self.beta
        return (self.alpha * self.beta) / (total**2 * (total + 1))

    @property
    def concentration(self) -> float:
        """Return concentration parameter (higher = more confident prior)."""
        return self.alpha + self.beta

    def as_tuple(self) -> tuple[float, float]:
        """Return parameters as (alpha, beta) tuple."""
        return (self.alpha, self.beta)

    def strength_description(self) -> str:
        """Human-readable description of prior strength."""
        concentration = self.concentration
        if concentration < 4:
            return "Weak (data-driven)"
        elif concentration < 10:
            return "Moderate"
        else:
            return "Strong (informative)"


# =============================================================================
# REPORTED EPS HISTORY & FORWARD ESTIMATE SIGNALS
# =============================================================================


@dataclass
class ReportedEPSHistory:
    """Actual reported EPS data for quarterly and annual periods.

    Fields follow the naming convention from the equities schema:
    - eps_basic_fq: most recent fiscal quarter
    - eps_basic_1fqfq: one quarter ago, etc.
    - eps_basic_fy: most recent fiscal year
    - eps_basic_1fy: one year ago, etc.
    """

    # Quarterly Net EPS - Basic (newest first)
    eps_basic_fq: Optional[float] = None
    eps_basic_1fqfq: Optional[float] = None
    eps_basic_2fqfq: Optional[float] = None
    eps_basic_3fqfq: Optional[float] = None
    eps_basic_4fqfq: Optional[float] = None

    # Annual Net EPS - Basic (newest first)
    eps_basic_fy: Optional[float] = None
    eps_basic_1fy: Optional[float] = None
    eps_basic_2fy: Optional[float] = None
    eps_basic_3fy: Optional[float] = None
    eps_basic_4fy: Optional[float] = None
    eps_basic_5fy: Optional[float] = None

    # Adjusted EPS
    eps_adj_fy: Optional[float] = None
    eps_adj_1fy: Optional[float] = None
    eps_adj_ltm: Optional[float] = None
    eps_adj_fq: Optional[float] = None
    eps_adj_1fqfq: Optional[float] = None
    eps_adj_2fqfq: Optional[float] = None
    eps_adj_3fqfq: Optional[float] = None
    eps_adj_4fqfq: Optional[float] = None

    # Continuing EPS
    eps_cont_fq: Optional[float] = None
    eps_cont_1fqfq: Optional[float] = None
    eps_cont_2fqfq: Optional[float] = None
    eps_cont_3fqfq: Optional[float] = None
    eps_cont_4fqfq: Optional[float] = None

    @property
    def quarterly_series(self) -> list[float]:
        """Return non-None quarterly EPS values, newest first."""
        fields = [
            self.eps_basic_fq,
            self.eps_basic_1fqfq,
            self.eps_basic_2fqfq,
            self.eps_basic_3fqfq,
            self.eps_basic_4fqfq,
        ]
        return [v for v in fields if v is not None]

    @property
    def annual_series(self) -> list[float]:
        """Return non-None annual EPS values, newest first."""
        fields = [
            self.eps_basic_fy,
            self.eps_basic_1fy,
            self.eps_basic_2fy,
            self.eps_basic_3fy,
            self.eps_basic_4fy,
            self.eps_basic_5fy,
        ]
        return [v for v in fields if v is not None]

    def count_yoy_improvements(self) -> tuple[int, int]:
        """Count year-over-year improvements in annual EPS.

        Compares each consecutive pair (newer vs older).
        Returns (n_beats, n_total) where n_total is the number of
        consecutive pairs available.
        """
        series = self.annual_series
        if len(series) < 2:
            return (0, 0)
        n_beats = 0
        n_total = len(series) - 1
        for i in range(n_total):
            if series[i] > series[i + 1]:
                n_beats += 1
        return (n_beats, n_total)

    def quarterly_beat_streak(self) -> int:
        """Count consecutive positive EPS quarters from most recent."""
        streak = 0
        for v in self.quarterly_series:
            if v > 0:
                streak += 1
            else:
                break
        return streak

    def count_quarterly_beats_vs_estimate(self, estimate: Optional[float]) -> tuple[int, int]:
        """Count how many quarterly actuals exceeded a forward estimate.

        Args:
            estimate: Forward EPS estimate to compare against.

        Returns:
            (n_beats, n_total) tuple.
        """
        if estimate is None:
            return (0, 0)
        series = self.quarterly_series
        if not series:
            return (0, 0)
        n_beats = sum(1 for v in series if v > estimate)
        return (n_beats, len(series))

    @property
    def total_reports_count(self) -> int:
        """Total number of non-null reported EPS observations across all series.

        Counts unique non-null entries across quarterly basic, annual basic,
        adjusted, and continuing EPS fields to dynamically derive the total
        number of available data points for historical beat rate calculations.
        """
        all_fields = [
            # Quarterly basic
            self.eps_basic_fq,
            self.eps_basic_1fqfq,
            self.eps_basic_2fqfq,
            self.eps_basic_3fqfq,
            self.eps_basic_4fqfq,
            # Annual basic
            self.eps_basic_fy,
            self.eps_basic_1fy,
            self.eps_basic_2fy,
            self.eps_basic_3fy,
            self.eps_basic_4fy,
            self.eps_basic_5fy,
            # Adjusted
            self.eps_adj_fy,
            self.eps_adj_1fy,
            self.eps_adj_ltm,
            self.eps_adj_fq,
            self.eps_adj_1fqfq,
            self.eps_adj_2fqfq,
            self.eps_adj_3fqfq,
            self.eps_adj_4fqfq,
            # Continuing
            self.eps_cont_fq,
            self.eps_cont_1fqfq,
            self.eps_cont_2fqfq,
            self.eps_cont_3fqfq,
            self.eps_cont_4fqfq,
        ]
        return sum(1 for v in all_fields if v is not None)

    @property
    def annual_reports_count(self) -> int:
        """Count of non-null annual EPS observations (basic series only)."""
        return len(self.annual_series)

    @property
    def quarterly_reports_count(self) -> int:
        """Count of non-null quarterly EPS observations (basic series only)."""
        return len(self.quarterly_series)


@dataclass
class ForwardEstimateSignals:
    """Forward-looking analyst estimate and revision signals.

    Captures consensus EPS estimates (Normalized and GAAP) plus
    revision percentages across multiple time horizons.
    """

    # Consensus estimates
    eps_norm_ntm: Optional[float] = None
    eps_norm_fy1e: Optional[float] = None
    eps_gaap_ntm: Optional[float] = None
    eps_gaap_fy1e: Optional[float] = None

    # Normalized revision percentages
    revision_1w: Optional[float] = None
    revision_1m: Optional[float] = None
    revision_3m: Optional[float] = None
    revision_6m: Optional[float] = None
    revision_1y: Optional[float] = None

    # GAAP revision percentages
    gaap_revision_1m: Optional[float] = None
    gaap_revision_3m: Optional[float] = None
    gaap_revision_6m: Optional[float] = None
    gaap_revision_1y: Optional[float] = None

    # Coverage
    analyst_count: Optional[int] = None

    # Recency weights for revision momentum (1W most important)
    _REVISION_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "revision_1w": 0.35,
            "revision_1m": 0.30,
            "revision_3m": 0.20,
            "revision_6m": 0.10,
            "revision_1y": 0.05,
        },
        init=False,
        repr=False,
    )

    @property
    def gaap_revision_momentum(self) -> float:
        """Compute a 0-100 momentum score from revision data.

        Uses recency-weighted scoring: each available revision contributes
        its weight toward 100 (positive) or 0 (negative).
        Returns 50.0 when no revision data is available.
        """
        available = []
        for field_name, weight in self._REVISION_WEIGHTS.items():
            val = getattr(self, field_name)
            if val is not None:
                available.append((val, weight))
        if not available:
            return 50.0
        # Renormalize weights to sum to 1
        total_weight = sum(w for _, w in available)
        score = 0.0
        for val, weight in available:
            normalized_w = weight / total_weight
            # Map: positive revision → 100, negative → 0
            score += normalized_w * (100.0 if val > 0 else (50.0 if val == 0 else 0.0))
        return score

    @property
    def revision_trend_short(self) -> Optional[float]:
        """Short-term revision acceleration: 1W - 1M."""
        if self.revision_1w is not None and self.revision_1m is not None:
            return self.revision_1w - self.revision_1m
        return None

    @property
    def revision_trend_medium(self) -> Optional[float]:
        """Medium-term revision acceleration: 1M - 3M."""
        if self.revision_1m is not None and self.revision_3m is not None:
            return self.revision_1m - self.revision_3m
        return None

    @property
    def gaap_norm_spread(self) -> Optional[float]:
        """GAAP-vs-Norm divergence as percentage of Norm estimate.

        Returns (GAAP - Norm) / Norm * 100. Negative means GAAP < Norm
        (potential accounting quality concern).
        """
        if self.eps_gaap_fy1e is not None and self.eps_norm_fy1e is not None:
            if self.eps_norm_fy1e != 0:
                return (self.eps_gaap_fy1e - self.eps_norm_fy1e) / self.eps_norm_fy1e * 100.0
        return None

    @property
    def has_sufficient_data(self) -> bool:
        """Check if enough forward data is available for enhanced analysis.

        Requires at least a FY1E estimate and one revision data point.
        """
        has_estimate = self.eps_norm_fy1e is not None
        revision_fields = [
            self.revision_1w,
            self.revision_1m,
            self.revision_3m,
            self.revision_6m,
            self.revision_1y,
        ]
        has_revision = any(v is not None for v in revision_fields)
        return has_estimate and has_revision


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
        sector_priors: Optional[dict[str, PriorParameters]] = None,
    ):
        """
        Initialize the earnings beat probability model.

        Args:
            prior_alpha: Alpha parameter for Beta prior (default: 2.0 for mild optimism)
            prior_beta: Beta parameter for Beta prior (default: 2.0 for symmetry)
            sector_priors: Optional dict mapping sectors to PriorParameters
                          for sector-specific priors based on historical patterns
        """
        self.default_prior = PriorParameters(prior_alpha, prior_beta)
        self.sector_priors = sector_priors or self._create_default_sector_priors()

    def _create_default_sector_priors(self) -> dict[str, PriorParameters]:
        """
        Create default sector-specific priors based on typical beat rates.

        Technology tends to have higher beat rates (~70%), while
        cyclical sectors like Energy have more variable outcomes.
        """
        return {
            "Information Technology": PriorParameters(3.5, 1.5),  # ~70% prior beat rate
            "Health Care": PriorParameters(3.0, 1.5),  # ~67% prior beat rate
            "Consumer Discretionary": PriorParameters(2.5, 1.5),  # ~63% prior beat rate
            "Industrials": PriorParameters(2.5, 1.5),  # ~63% prior beat rate
            "Financials": PriorParameters(2.5, 2.0),  # ~56% prior beat rate
            "Consumer Staples": PriorParameters(2.5, 2.0),  # ~56% prior beat rate
            "Materials": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
            "Energy": PriorParameters(2.0, 2.5),  # ~44% prior beat rate
            "Utilities": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
            "Communication Services": PriorParameters(3.0, 1.5),  # ~67% prior beat rate
            "Real Estate": PriorParameters(2.0, 2.0),  # ~50% prior beat rate
        }

    def _get_prior_parameters(
        self,
        sector: Optional[str],
        use_sector_prior: bool,
    ) -> PriorParameters:
        """
        Get the appropriate prior parameters based on sector.

        Args:
            sector: Optional sector name
            use_sector_prior: Whether to use sector-specific priors

        Returns:
            PriorParameters for the given sector or default
        """
        if not use_sector_prior or not sector:
            return self.default_prior

        return self.sector_priors.get(sector, self.default_prior)

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

    def _compute_single_credible_interval(
        self,
        distribution: stats.rv_continuous,
        lower_quantile: float,
        upper_quantile: float,
    ) -> tuple[float, float]:
        """
        Compute a single credible interval from posterior distribution.

        Args:
            distribution: Scipy Beta distribution object
            lower_quantile: Lower quantile (e.g., 0.05 for 90% CI)
            upper_quantile: Upper quantile (e.g., 0.95 for 90% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        return (distribution.ppf(lower_quantile), distribution.ppf(upper_quantile))

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
        ci_90 = self._compute_single_credible_interval(
            distribution,
            self.CI_90_LOWER_QUANTILE,
            self.CI_90_UPPER_QUANTILE,
        )
        ci_95 = self._compute_single_credible_interval(
            distribution,
            self.CI_95_LOWER_QUANTILE,
            self.CI_95_UPPER_QUANTILE,
        )
        return ci_90, ci_95

    def _compute_confidence_score(self, alpha: float, beta: float) -> float:
        """
        Compute confidence score based on effective sample size.

        The confidence score reflects how much data supports the posterior
        estimate, normalized to a 0-1 scale.

        Args:
            alpha: Posterior alpha parameter
            beta: Posterior beta parameter

        Returns:
            Confidence score between 0 and 1
        """
        effective_sample_size = alpha + beta - 2  # Subtract prior contribution
        confidence = min(1.0, effective_sample_size / self.CONFIDENCE_NORMALIZATION_FACTOR)
        return max(0.0, confidence)

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
        prior = self._get_prior_parameters(sector, use_sector_prior)
        alpha, beta = prior.alpha, prior.beta

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
        name_col: str = "name",
    ) -> pd.DataFrame:
        """
        Analyze earnings beat probabilities for entire DataFrame.

        Args:
            df: DataFrame with earnings data
            beats_col: Column name for beat counts
            total_col: Column name for total report counts
            sector_col: Column name for sector
            ticker_col: Column name for ticker
            name_col: Column name for company name

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
            name = row.get(name_col, "UNKNOWN")

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
            prior = self._get_prior_parameters(sector, use_sector_prior=True)

            # Determine classification based on posterior mean vs default threshold
            beat_classification = (
                "likely_beat" if prob_result["posterior_mean"] > 0.5 else "uncertain"
            )

            record = _extract_identifiers(row)
            record.update({
                "historical_beats": n_beats,
                "total_reports": n_total,
                "historical_beat_rate": n_beats / n_total,
                "prior_alpha": prior.alpha,
                "prior_beta": prior.beta,
                "posterior_alpha": prob_result["posterior_alpha"],
                "posterior_beta": prob_result["posterior_beta"],
                "posterior_beat_prob": prob_result["posterior_mean"],
                "posterior_std": prob_result["posterior_std"],
                "ci_90_lower": prob_result["credible_interval_90"][0],
                "ci_90_upper": prob_result["credible_interval_90"][1],
                "ci_95_lower": prob_result["credible_interval_95"][0],
                "ci_95_upper": prob_result["credible_interval_95"][1],
                "confidence_score": prob_result["confidence_score"],
                "beat_classification": beat_classification,
            })
            results.append(record)

        return pd.DataFrame(results)

    # -----------------------------------------------------------------
    # Enhanced: Three-layer evidence fusion
    # -----------------------------------------------------------------

    # GAAP divergence threshold (%) below which no penalty is applied
    GAAP_DIVERGENCE_THRESHOLD = 20.0
    # Maximum pseudo-observations added by revision momentum
    MAX_REVISION_PSEUDO_OBS = 3.0

    def compute_forward_adjusted_beat_probability(
        self,
        reported_history: ReportedEPSHistory,
        forward_signals: ForwardEstimateSignals,
        sector: Optional[str] = None,
    ) -> BeatProbabilityEstimate:
        """Compute beat probability fusing historical, revision, and GAAP quality layers.

        Layer 1 – Historical beats from actual reported EPS (YoY improvements).
        Layer 2 – Revision momentum converted to pseudo-observations.
        Layer 3 – GAAP-vs-Norm divergence penalty (shrinks toward prior).

        Args:
            reported_history: Actual reported EPS data.
            forward_signals: Forward-looking analyst signals.
            sector: Optional sector for sector-specific prior.

        Returns:
            BeatProbabilityEstimate dict with full posterior statistics.
        """
        prior = self._get_prior_parameters(sector, use_sector_prior=True)

        # --- Layer 1: Historical beat counting ---
        n_beats, n_total = reported_history.count_yoy_improvements()
        if n_total == 0:
            # Fallback to quarterly streak as pseudo-observations
            streak = reported_history.quarterly_beat_streak()
            # Dynamically derive total from non-null quarterly data
            n_total = reported_history.quarterly_reports_count
            n_beats = min(streak, n_total)

        # If still zero, use the full non-null count as a last resort
        if n_total == 0:
            n_total = reported_history.total_reports_count

        # --- Layer 2: Revision momentum pseudo-observations ---
        momentum = forward_signals.gaap_revision_momentum  # 0-100
        # Convert to pseudo beat fraction and scale
        pseudo_beat_frac = momentum / 100.0
        pseudo_n = self.MAX_REVISION_PSEUDO_OBS if forward_signals.has_sufficient_data else 0.0
        pseudo_beats = pseudo_beat_frac * pseudo_n
        pseudo_misses = pseudo_n - pseudo_beats

        # --- Posterior before GAAP penalty ---
        post_alpha = prior.alpha + n_beats + pseudo_beats
        post_beta = prior.beta + (n_total - n_beats) + pseudo_misses

        # --- Layer 3: GAAP quality guard ---
        spread = forward_signals.gaap_norm_spread
        if spread is not None and abs(spread) > self.GAAP_DIVERGENCE_THRESHOLD:
            # Proportional shrinkage toward prior mean
            excess = abs(spread) - self.GAAP_DIVERGENCE_THRESHOLD
            # penalty_factor in (0, 1]; larger excess → stronger shrinkage
            penalty_factor = min(1.0, excess / 100.0)
            # Also penalise divergent GAAP revisions vs norm revisions
            if (
                forward_signals.gaap_revision_1m is not None
                and forward_signals.revision_1m is not None
            ):
                rev_sign_mismatch = (
                    forward_signals.revision_1m > 0 and forward_signals.gaap_revision_1m < 0
                )
                if rev_sign_mismatch:
                    penalty_factor = min(1.0, penalty_factor + 0.15)

            # Shrink posterior toward prior by blending
            prior_total = prior.alpha + prior.beta
            post_total = post_alpha + post_beta
            data_alpha = post_alpha - prior.alpha
            data_beta = post_beta - prior.beta
            post_alpha = prior.alpha + data_alpha * (1 - penalty_factor)
            post_beta = prior.beta + data_beta * (1 - penalty_factor)

        # --- Compute statistics ---
        posterior_mean, posterior_std = self._compute_posterior_statistics(post_alpha, post_beta)
        dist = stats.beta(post_alpha, post_beta)
        ci_90, ci_95 = self._compute_credible_intervals(dist)
        prob_exceeds = 1 - dist.cdf(0.5)
        confidence_score = self._compute_confidence_score(post_alpha, post_beta)

        # Prior influence
        prior_total = prior.alpha + prior.beta
        post_total = post_alpha + post_beta
        prior_influence = prior_total / post_total * 100.0

        effective_sample = post_total - prior_total

        # Classification confidence
        if confidence_score >= 0.6:
            classification_confidence = "High"
        elif confidence_score >= 0.3:
            classification_confidence = "Medium"
        else:
            classification_confidence = "Low"

        return {
            "posterior_alpha": post_alpha,
            "posterior_beta": post_beta,
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "credible_interval_90": ci_90,
            "credible_interval_95": ci_95,
            "prob_exceeds_threshold": prob_exceeds,
            "confidence_score": confidence_score,
            "prior_influence_pct": prior_influence,
            "effective_sample_size": effective_sample,
            "classification_confidence": classification_confidence,
        }

    # -----------------------------------------------------------------
    # Enhanced DataFrame analysis
    # -----------------------------------------------------------------

    # Column mappings from equities table to dataclass fields
    _FORWARD_COL_MAP: dict[str, str] = {
        "eps_norm_est_avg_ntm": "eps_norm_ntm",
        "eps_norm_est_avg_fy1e": "eps_norm_fy1e",
        "eps_gaap_est_avg_ntm": "eps_gaap_ntm",
        "eps_gaap_est_avg_fy1e": "eps_gaap_fy1e",
        "eps_est_avg_rev_pct_fy1e_1w": "revision_1w",
        "eps_est_avg_rev_pct_fy1e_1m": "revision_1m",
        "eps_est_avg_rev_pct_fy1e_3m": "revision_3m",
        "eps_est_avg_rev_pct_fy1e_6m": "revision_6m",
        "eps_est_avg_rev_pct_fy1e_1y": "revision_1y",
        "eps_gaap_est_avg_rev_pct_fy1e_1m": "gaap_revision_1m",
        "eps_gaap_est_avg_rev_pct_fy1e_3m": "gaap_revision_3m",
        "eps_gaap_est_avg_rev_pct_fy1e_6m": "gaap_revision_6m",
        "eps_gaap_est_avg_rev_pct_fy1e_1y": "gaap_revision_1y",
        "eps_norm_est_num_fy1e": "analyst_count",
    }

    _HISTORY_COL_MAP: dict[str, str] = {
        "net_eps_basic_fq": "eps_basic_fq",
        "net_eps_basic_1fqfq": "eps_basic_1fqfq",
        "net_eps_basic_2fqfq": "eps_basic_2fqfq",
        "net_eps_basic_3fqfq": "eps_basic_3fqfq",
        "net_eps_basic_4fqfq": "eps_basic_4fqfq",
        "net_eps_basic_fy": "eps_basic_fy",
        "net_eps_basic_1fy": "eps_basic_1fy",
        "net_eps_basic_2fy": "eps_basic_2fy",
        "net_eps_basic_3fy": "eps_basic_3fy",
        "net_eps_basic_4fy": "eps_basic_4fy",
        "net_eps_basic_5fy": "eps_basic_5fy",
        "eps_adj_ltm": "eps_adj_ltm",
        "eps_adj_fy": "eps_adj_fy",
        "eps_adj_1fy": "eps_adj_1fy",
        "eps_adj_fq": "eps_adj_fq",
        "eps_adj_1fqfq": "eps_adj_1fqfq",
        "eps_adj_2fqfq": "eps_adj_2fqfq",
        "eps_adj_3fqfq": "eps_adj_3fqfq",
        "eps_adj_4fqfq": "eps_adj_4fqfq",
        "eps_cont_fq": "eps_cont_fq",
        "eps_cont_1fqfq": "eps_cont_1fqfq",
        "eps_cont_2fqfq": "eps_cont_2fqfq",
        "eps_cont_3fqfq": "eps_cont_3fqfq",
        "eps_cont_4fqfq": "eps_cont_4fqfq",
    }

    def _row_to_forward_signals(self, row: pd.Series) -> Optional[ForwardEstimateSignals]:
        """Extract ForwardEstimateSignals from a DataFrame row."""
        kwargs: dict = {}
        any_present = False
        for df_col, field_name in self._FORWARD_COL_MAP.items():
            val = row.get(df_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                kwargs[field_name] = int(val) if field_name == "analyst_count" else float(val)
                any_present = True
        if not any_present:
            return None
        return ForwardEstimateSignals(**kwargs)

    def _row_to_history(self, row: pd.Series) -> ReportedEPSHistory:
        """Extract ReportedEPSHistory from a DataFrame row."""
        kwargs: dict = {}
        for df_col, field_name in self._HISTORY_COL_MAP.items():
            val = row.get(df_col)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                kwargs[field_name] = float(val)
        return ReportedEPSHistory(**kwargs)

    def analyze_dataframe_enhanced(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
        name_col: str = "name",
    ) -> pd.DataFrame:
        """Analyze earnings beat probabilities using enhanced three-layer fusion.

        Falls back to trajectory-proxy method when forward data is unavailable.

        Args:
            df: DataFrame with equities data.
            sector_col: Column name for sector.
            ticker_col: Column name for ticker.
            name_col: Column name for company name.

        Returns:
            DataFrame with enriched probability analysis results.
        """
        results = []

        for _, row in df.iterrows():
            ticker = row.get(ticker_col, "UNKNOWN")
            name = row.get(name_col, "UNKNOWN")
            sector = row.get(sector_col, None)

            forward_signals = self._row_to_forward_signals(row)
            history = self._row_to_history(row)

            if forward_signals is not None:
                # Enhanced path
                prob_result = self.compute_forward_adjusted_beat_probability(
                    reported_history=history,
                    forward_signals=forward_signals,
                    sector=sector,
                )
                n_beats, n_total = history.count_yoy_improvements()
                # Dynamically derive total_reports from non-null reported data
                dynamic_total = history.total_reports_count
                # Use the larger of YoY pair count and dynamic count for beat rate
                effective_total = max(n_total, dynamic_total) if dynamic_total > 0 else n_total
                historical_beat_rate = n_beats / effective_total if effective_total > 0 else 0.0

                beat_classification = (
                    "likely_beat" if prob_result["posterior_mean"] > 0.5 else "uncertain"
                )
                record = _extract_identifiers(row)
                record.update({
                    "historical_beats": n_beats,
                    "total_reports": effective_total,
                    "dynamic_total_reports": dynamic_total,
                    "historical_beat_rate": historical_beat_rate,
                    "posterior_beat_prob": prob_result["posterior_mean"],
                    "posterior_std": prob_result["posterior_std"],
                    "ci_90_lower": prob_result["credible_interval_90"][0],
                    "ci_90_upper": prob_result["credible_interval_90"][1],
                    "ci_95_lower": prob_result["credible_interval_95"][0],
                    "ci_95_upper": prob_result["credible_interval_95"][1],
                    "confidence_score": prob_result["confidence_score"],
                    "prior_influence_pct": prob_result["prior_influence_pct"],
                    "effective_sample_size": prob_result["effective_sample_size"],
                    "classification_confidence": prob_result["classification_confidence"],
                    "beat_classification": beat_classification,
                    "gaap_revision_momentum": forward_signals.gaap_revision_momentum,
                    "gaap_norm_spread": forward_signals.gaap_norm_spread,
                    "revision_trend_short": forward_signals.revision_trend_short,
                    "revision_trend_medium": forward_signals.revision_trend_medium,
                    "eps_norm_est_fy1e": forward_signals.eps_norm_fy1e,
                    "eps_norm_est_ntm": forward_signals.eps_norm_ntm,
                    "eps_gaap_est_ntm": forward_signals.eps_gaap_ntm,
                    "eps_gaap_est_fy1e": forward_signals.eps_gaap_fy1e,
                    "analyst_count": forward_signals.analyst_count,
                    "next_earnings_status": row.get("next_earnings_status", None),
                    "quarterly_beat_streak": history.quarterly_beat_streak(),
                    "data_source": "forward_enhanced",
                })
                results.append(record)
            elif "eps_trajectory_score" in df.columns and not pd.isna(
                row.get("eps_trajectory_score")
            ):
                prior = self._get_prior_parameters(sector, True)
                # Trajectory proxy fallback
                trajectory = row["eps_trajectory_score"]
                # Dynamically derive total from non-null reported data
                dynamic_total = history.total_reports_count
                n_total = dynamic_total if dynamic_total > 0 else 5
                n_beats = int(trajectory / 100 * n_total)
                prob_result = self.compute_beat_probability(n_beats, n_total, sector)
                beat_classification = (
                    "likely_beat" if prob_result["posterior_mean"] > 0.5 else "uncertain"
                )
                confidence_score = prob_result["confidence_score"]
                if confidence_score >= 0.6:
                    classification_confidence = "High"
                elif confidence_score >= 0.3:
                    classification_confidence = "Medium"
                else:
                    classification_confidence = "Low"

                prior = self._get_prior_parameters(sector, True)
                prior_total = prior.alpha + prior.beta
                post_total = prob_result["posterior_alpha"] + prob_result["posterior_beta"]

                record = _extract_identifiers(row)
                record.update({
                    "historical_beats": n_beats,
                    "total_reports": n_total,
                    "dynamic_total_reports": dynamic_total,
                    "historical_beat_rate": n_beats / n_total if n_total > 0 else 0.0,
                    "prior_alpha": prior.alpha,
                    "prior_beta": prior.beta,
                    "posterior_alpha": prob_result["posterior_alpha"],
                    "posterior_beta": prob_result["posterior_beta"],
                    "posterior_beat_prob": prob_result["posterior_mean"],
                    "posterior_std": prob_result["posterior_std"],
                    "ci_90_lower": prob_result["credible_interval_90"][0],
                    "ci_90_upper": prob_result["credible_interval_90"][1],
                    "ci_95_lower": prob_result["credible_interval_95"][0],
                    "ci_95_upper": prob_result["credible_interval_95"][1],
                    "confidence_score": confidence_score,
                    "prior_influence_pct": prior_total / post_total * 100.0,
                    "effective_sample_size": post_total - prior_total,
                    "classification_confidence": classification_confidence,
                    "beat_classification": beat_classification,
                    "gaap_revision_momentum": None,
                    "gaap_norm_spread": None,
                    "revision_trend_short": None,
                    "revision_trend_medium": None,
                    "eps_norm_est_fy1e": None,
                    "eps_norm_est_ntm": None,
                    "eps_gaap_est_ntm": None,
                    "eps_gaap_est_fy1e": None,
                    "analyst_count": None,
                    "next_earnings_status": row.get("next_earnings_status", None),
                    "quarterly_beat_streak": None,
                    "data_source": "trajectory_proxy",
                })
                results.append(record)
            # else: skip row with no data

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
        ticker: str = "",
        name: str = "",
        sector: str = "",
        industry: str = "",
        country: str = "",
        exchange: str = "",
        reported_history: Optional[ReportedEPSHistory] = None,
        forward_signals: Optional[ForwardEstimateSignals] = None,
    ) -> EPSStreakResult:
        """
        Compute streak analysis from trajectory score and related metrics.

        Args:
            eps_trajectory_score: EPS trajectory score (0-100)
            eps_positive_streak: Number of positive EPS quarters
            eps_improvement_count: Number of YoY improvements
            ticker: Ticker symbol
            name: Company name
            sector: Sector
            industry: Industry
            country: Country
            exchange: Exchange
            reported_history: Optional actual reported EPS data for dynamic counts
            forward_signals: Optional forward estimate signals for probability refinement

        Returns:
            EPSStreakResult with analysis
        """
        # --- Dynamically derive streak from reported history if available ---
        if reported_history is not None and reported_history.quarterly_reports_count > 0:
            dynamic_streak = reported_history.quarterly_beat_streak()
            dynamic_total = reported_history.quarterly_reports_count
        else:
            dynamic_streak = None
            dynamic_total = 0

        # Estimate current streak from available metrics
        if eps_positive_streak is not None and not pd.isna(eps_positive_streak):
            current_streak = int(eps_positive_streak)
            streak_type = "beat" if current_streak > 0 else "miss"
        elif dynamic_streak is not None and dynamic_streak > 0:
            current_streak = dynamic_streak
            streak_type = "beat"
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

        # --- Forward estimate adjustment ---
        # If revision momentum is positive and streak is a beat, boost continuation
        if forward_signals is not None and forward_signals.has_sufficient_data:
            momentum = forward_signals.gaap_revision_momentum  # 0-100
            # Positive momentum reinforces beat streaks, undermines miss streaks
            momentum_adjustment = (momentum - 50.0) / 200.0  # Range: -0.25 to +0.25
            if streak_type == "beat":
                continuation_prob += momentum_adjustment
            elif streak_type == "miss":
                continuation_prob -= momentum_adjustment

            # GAAP-Norm spread: large divergence reduces confidence in continuation
            spread = forward_signals.gaap_norm_spread
            if spread is not None and abs(spread) > 20.0:
                continuation_prob -= min(0.10, abs(spread) / 500.0)

            continuation_prob = max(0.05, min(0.95, continuation_prob))

        # Apply mean reversion adjustment
        mean_reversion_prob = 1 - continuation_prob
        mean_reversion_prob = mean_reversion_prob * (
            1 - self.mean_reversion_weight
        ) + self.mean_reversion_weight * (1 - continuation_prob)

        # Confidence based on streak length and data availability
        confidence = max(0.3, 1 - abs(current_streak) * 0.1)
        # Boost confidence if we have more dynamic data points
        if dynamic_total > 3:
            confidence = min(1.0, confidence + 0.1)
        if forward_signals is not None and forward_signals.has_sufficient_data:
            confidence = min(1.0, confidence + 0.05)

        # Expected next outcome
        if continuation_prob > 0.5:
            expected_next = streak_type
        else:
            expected_next = "beat" if streak_type == "miss" else "miss"

        return EPSStreakResult(
            ticker=ticker,
            name=name,
            sector=sector,
            industry=industry,
            country=country,
            exchange=exchange,
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
        ticker_col: str = "ticker",
        trajectory_col: str = "eps_trajectory_score",
        streak_col: str = "eps_positive_streak",
        improvement_col: str = "eps_improvement_count",
        name_col: str = "name",
        sector_col: str = "sector",
        industry_col: str = "industry",
        country_col: str = "country",
        exchange_col: str = "exchange",
    ) -> pd.DataFrame:
        """
        Analyze EPS streaks for entire DataFrame.

        Now dynamically derives total_reports from non-null reported EPS data
        and incorporates forward estimate signals when available.

        Args:
            df: DataFrame with EPS data
            trajectory_col: Column for trajectory score
            streak_col: Column for positive streak count
            improvement_col: Column for improvement count
            ticker_col: Column for ticker
            name_col: Column for company name
            sector_col: Column for sector
            industry_col: Column for industry
            country_col: Column for country
            exchange_col: Column for exchange

        Returns:
            DataFrame with streak analysis
        """
        # Check if forward/history columns are available
        has_history_cols = any(
            col in df.columns for col in EarningsBeatProbabilityModel._HISTORY_COL_MAP
        )
        has_forward_cols = any(
            col in df.columns for col in EarningsBeatProbabilityModel._FORWARD_COL_MAP
        )

        # Create a temporary model instance for column mapping helpers
        _model = EarningsBeatProbabilityModel()

        results = []

        for _, row in df.iterrows():
            trajectory = row.get(trajectory_col, None)
            streak = row.get(streak_col, None)
            improvement = row.get(improvement_col, None)
            ticker = row.get(ticker_col, "UNKNOWN")
            name = row.get(name_col, "")
            sector = row.get(sector_col, "")
            industry = row.get(industry_col, "")
            country = row.get(country_col, "")
            exchange = row.get(exchange_col, "")

            if trajectory is None or pd.isna(trajectory):
                continue

            # Build reported history and forward signals when columns exist
            reported_history = None
            forward_signals = None
            dynamic_total = 0

            if has_history_cols:
                reported_history = _model._row_to_history(row)
                dynamic_total = reported_history.total_reports_count

            if has_forward_cols:
                forward_signals = _model._row_to_forward_signals(row)

            result = self.compute_streak_from_trajectory(
                eps_trajectory_score=trajectory,
                eps_positive_streak=streak,
                eps_improvement_count=improvement,
                ticker=ticker,
                name=name,
                sector=sector,
                industry=industry,
                country=country,
                exchange=exchange,
                reported_history=reported_history,
                forward_signals=forward_signals,
            )

            # Historical beat rate from dynamically derived total
            n_beats_yoy, n_total_yoy = (
                reported_history.count_yoy_improvements()
                if reported_history is not None
                else (0, 0)
            )
            effective_total = max(n_total_yoy, dynamic_total) if dynamic_total > 0 else n_total_yoy
            historical_beat_rate = n_beats_yoy / effective_total if effective_total > 0 else 0.0

            record = _extract_identifiers(row)
            record.update({
                    "current_streak": result.current_streak,
                    "streak_type": result.streak_type,
                    "continuation_probability": result.streak_continuation_prob,
                    "mean_reversion_probability": result.mean_reversion_prob,
                    "expected_next_outcome": result.expected_next_outcome,
                    "prediction_confidence": result.confidence_level,
                    "dynamic_total_reports": dynamic_total,
                    "historical_beat_rate": historical_beat_rate,
                    "gaap_revision_momentum": (
                        forward_signals.gaap_revision_momentum
                        if forward_signals is not None
                        else None
                    ),
                    "next_earnings_status": row.get("next_earnings_status", None),
                }
            )
            results.append(record)

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


class CreditRiskProbabilityModel:
    """
    Bayesian framework to estimate likelihood of financial distress.

    Enhanced features: altman_z_score, altman_z_trend, liquidity_stress_score,
    cash_runway_months, accumulated_deficit_flag, combined_distress_score,
    wc_deteriorating_flag, debt_deleveraging, interest_coverage, quick_ratio,
    beta_stability_score
    """

    def __init__(
        self,
        distress_threshold: float = 1.81,
        prior_alpha: float = 2.0,
        prior_beta: float = 3.0,  # Slightly pessimistic prior
    ):
        self.distress_threshold = distress_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze dataframe for credit risk with enhanced features."""
        results = []

        for _, row in df.iterrows():
            # Core distress indicators
            z_score = row.get("altman_z_score", 3.0)
            z_trend = row.get("altman_z_trend", 0)  # NEW: Z-score trajectory
            liquidity_stress = row.get("liquidity_stress_score", 50)
            cash_runway = row.get("cash_runway_months", 24)
            accumulated_deficit = row.get("accumulated_deficit_flag", 0)

            # NEW: Additional risk factors from views
            combined_distress = row.get("combined_distress_score", 50)
            wc_deteriorating = row.get("wc_deteriorating_flag", 0)
            debt_deleveraging = row.get("debt_deleveraging", 0)  # Negative = more debt
            interest_coverage = row.get("interest_coverage", 5.0)
            quick_ratio = row.get("quick_ratio", 1.5)
            beta_stability = row.get("beta_stability_score", 50)

            # Bayesian-style probability estimation with enhanced inputs
            # Prior based on Z-Score zones
            if z_score < 1.81:
                base_prob = 0.75
            elif z_score < 2.67:  # Grey zone
                base_prob = 0.35
            elif z_score < 3.0:
                base_prob = 0.15
            else:
                base_prob = 0.05

            # Evidence-based adjustments
            adjustments = 0.0

            # Z-score trend (deteriorating = higher risk)
            if z_trend < -0.5:
                adjustments += 0.15
            elif z_trend > 0.5:
                adjustments -= 0.05

            # Liquidity stress
            if liquidity_stress > 70:
                adjustments += 0.12
            elif liquidity_stress < 30:
                adjustments -= 0.05

            # Cash runway
            if cash_runway < 6:
                adjustments += 0.18
            elif cash_runway < 12:
                adjustments += 0.08
            elif cash_runway > 24:
                adjustments -= 0.03

            # Working capital deterioration
            if wc_deteriorating == 1:
                adjustments += 0.10

            # Debt trends
            if debt_deleveraging is not None and debt_deleveraging < -0.1:
                adjustments += 0.08  # Increasing debt burden

            # Interest coverage
            if interest_coverage is not None and interest_coverage < 1.5:
                adjustments += 0.15
            elif interest_coverage is not None and interest_coverage < 3.0:
                adjustments += 0.05

            # Quick ratio
            if quick_ratio is not None and quick_ratio < 0.8:
                adjustments += 0.10

            if accumulated_deficit == 1:
                adjustments += 0.08

            prob = min(0.99, max(0.01, base_prob + adjustments))

            # Compute confidence interval width based on data completeness
            data_points = sum(
                [
                    1
                    for v in [
                        z_score,
                        liquidity_stress,
                        cash_runway,
                        interest_coverage,
                        quick_ratio,
                    ]
                    if v is not None and not pd.isna(v)
                ]
            )
            ci_width = 0.15 - (data_points * 0.02)  # Narrower CI with more data

            risk_level = "Low"
            if prob > 0.7:
                risk_level = "Distressed"
            elif prob > 0.5:
                risk_level = "High"
            elif prob > 0.3:
                risk_level = "Medium"

            record = _extract_identifiers(row)
            record.update({
                    "distress_probability": prob,
                    "liquidity_stress_score": liquidity_stress,
                    "cash_runway_months": cash_runway,
                    "altman_z_score": z_score,
                    "altman_z_trend": z_trend,
                    "interest_coverage": interest_coverage,
                    "quick_ratio": quick_ratio,
                    "risk_level": risk_level,
                    "ci_lower": max(0, prob - ci_width),
                    "ci_upper": min(1, prob + ci_width),
                    "data_quality_score": data_points / 5.0,
            })
            results.append(record)

        return pd.DataFrame(results)


class DividendCutProbabilityModel:
    """
    Identify high-yield stocks where distribution is likely to be reduced.

    Enhanced features: fcf_dividend_coverage, dividend_payout_ratio, dividend_streak,
    dividend_growth_expectation, sustainable_dividend_flag, dividend_consistency,
    dividend_yield_vs_5y_avg, recent_dividend_change
    """

    def __init__(self, high_payout_threshold: float = 0.9, min_coverage: float = 1.2):
        self.high_payout_threshold = high_payout_threshold
        self.min_coverage = min_coverage

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core dividend metrics
            fcf_coverage = row.get("fcf_dividend_coverage", 2.0)
            payout_ratio = row.get("dividend_payout_ratio", 50)
            streak = row.get("dividend_streak", 10)
            growth_exp = row.get("dividend_growth_expectation", 0)

            # NEW: Enhanced metrics from vw_features_dividends
            sustainable_flag = row.get("sustainable_dividend_flag", 1)
            consistency = row.get("dividend_consistency", 0.8)
            yield_vs_5y = row.get("dividend_yield_vs_5y_avg", 1.0)
            recent_change = row.get("recent_dividend_change", 0)
            high_yield_flag = row.get("high_yield_flag", 0)

            # Base probability with more granular assessment
            prob = 0.05  # Low base rate for established dividend payers

            # FCF coverage is the strongest predictor
            if fcf_coverage is not None and not pd.isna(fcf_coverage):
                if fcf_coverage < 0.5:
                    prob += 0.45
                elif fcf_coverage < 1.0:
                    prob += 0.30
                elif fcf_coverage < 1.2:
                    prob += 0.15
                elif fcf_coverage > 2.0:
                    prob -= 0.03

            # Payout ratio stress
            if payout_ratio is not None and not pd.isna(payout_ratio):
                if payout_ratio > 100:
                    prob += 0.25  # Paying from reserves
                elif payout_ratio > 90:
                    prob += 0.15
                elif payout_ratio > 75:
                    prob += 0.05

            # Streak provides historical reliability signal
            if streak is not None:
                if streak < 2:
                    prob += 0.10
                elif streak >= 10:
                    prob -= 0.05  # Dividend aristocrat effect
                elif streak >= 5:
                    prob -= 0.02

            # NEW: Sustainability flag from comprehensive calc
            if sustainable_flag == 0:
                prob += 0.12

            # NEW: Consistency score
            if consistency is not None and consistency < 0.5:
                prob += 0.10

            # NEW: Yield vs historical average (abnormally high = warning)
            if yield_vs_5y is not None and yield_vs_5y > 1.5:
                prob += 0.08  # Price has dropped, yield spiked

            # NEW: Recent dividend changes
            if recent_change is not None and recent_change < -10:
                prob += 0.15  # Already cutting

            # Negative growth expectation
            if growth_exp is not None and growth_exp < -5:
                prob += 0.12

            prob = min(0.95, max(0.03, prob))

            risk_cat = "Safe"
            if prob > 0.6:
                risk_cat = "At Risk"
            elif prob > 0.35:
                risk_cat = "Borderline"
            elif prob > 0.15:
                risk_cat = "Monitor"

            record = _extract_identifiers(row)
            record.update({
                    "dividend_cut_probability": prob,
                    "fcf_dividend_coverage": fcf_coverage,
                    "payout_ratio": payout_ratio,
                    "dividend_streak": streak,
                    "dividend_consistency": consistency,
                    "yield_vs_5y_avg": yield_vs_5y,
                    "sustainable_flag": sustainable_flag,
                    "safety_score": 100 * (1 - prob),
                    "risk_category": risk_cat,
            })
            results.append(record)
        return pd.DataFrame(results)


class PriceTargetAchievementModel:
    """
    Estimates probability of reaching consensus price target.

    Enhanced features: upside_potential, price_target_spread_pct, pt_momentum_1m,
    analyst_rating_normalized, pt_consensus_convergence, analyst_conviction,
    pt_acceleration_short, eps_revision_momentum, analyst_coverage_trend
    """

    def __init__(self, time_horizon_months: int = 12):
        self.time_horizon_months = time_horizon_months

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            # Core metrics
            upside = row.get("upside_potential", 10)
            spread = row.get("price_target_spread_pct", 20)
            pt_momentum = row.get("pt_momentum_1m", 0)
            rating = row.get("analyst_rating_normalized", 50)

            # NEW: Enhanced analyst sentiment features
            conviction = row.get("analyst_conviction", 50)
            consensus_convergence = row.get("pt_consensus_convergence", 0)
            pt_accel = row.get("pt_acceleration_short", 0)
            eps_revision = row.get("eps_revision_momentum", 0)
            coverage_trend = row.get("analyst_coverage_trend", 0)
            bullish_pct = row.get("analyst_bullish_pct", 50)

            # Base probability - inversely related to upside magnitude
            if upside is None or pd.isna(upside):
                base_prob = 0.5
            elif upside <= 0:
                base_prob = 0.85  # Already at/above target
            elif upside < 10:
                base_prob = 0.70
            elif upside < 20:
                base_prob = 0.55
            elif upside < 30:
                base_prob = 0.40
            elif upside < 50:
                base_prob = 0.25
            else:
                base_prob = 0.15

            adjustments = 0.0

            # PT momentum signals conviction strengthening
            if pt_momentum is not None and pt_momentum > 0.05:
                adjustments += 0.08
            elif pt_momentum is not None and pt_momentum < -0.05:
                adjustments -= 0.10

            # Strong analyst consensus (low spread)
            if spread is not None and spread < 15:
                adjustments += 0.08
            elif spread is not None and spread > 40:
                adjustments -= 0.08

            # NEW: Analyst conviction score
            if conviction is not None and conviction > 70:
                adjustments += 0.07
            elif conviction is not None and conviction < 30:
                adjustments -= 0.05

            # NEW: Consensus converging (analysts agreeing)
            if consensus_convergence is not None and consensus_convergence > 0:
                adjustments += 0.05

            # NEW: PT acceleration (momentum building)
            if pt_accel is not None and pt_accel > 0.02:
                adjustments += 0.06

            # NEW: EPS revisions supporting the price target
            if eps_revision is not None and eps_revision > 5:
                adjustments += 0.08
            elif eps_revision is not None and eps_revision < -5:
                adjustments -= 0.10

            # NEW: Growing analyst coverage = more attention
            if coverage_trend is not None and coverage_trend > 0:
                adjustments += 0.03

            # Rating strength
            if rating is not None and rating > 75:
                adjustments += 0.05

            prob = min(0.90, max(0.05, base_prob + adjustments))

            record = _extract_identifiers(row)
            record.update({
                    "achievement_probability": prob,
                    "upside_potential": upside,
                    "price_target_spread_pct": spread,
                    "analyst_conviction": conviction,
                    "eps_revision_momentum": eps_revision,
                    "analyst_rating_normalized": rating,
                    "expected_return_prob_weighted": (upside or 0) * prob,
                    "confidence_level": (
                        "High"
                        if spread and spread < 20
                        else "Medium" if spread and spread < 35 else "Low"
                    ),
            })
            results.append(record)
        return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def create_earnings_probability_dashboard(
    probability_df: pd.DataFrame,
    title: str = "Earnings Beat Probability Analysis",
) -> go.Figure:
    """
    Create comprehensive dashboard for earnings beat probabilities.

    Supports both legacy output (from analyze_dataframe) and enhanced output
    (from analyze_dataframe_enhanced) with revision momentum and GAAP divergence
    columns. When enhanced columns are present, additional panels are shown.

    Args:
        probability_df: DataFrame from EarningsBeatProbabilityModel.analyze_dataframe
            or analyze_dataframe_enhanced
        title: Dashboard title

    Returns:
        Plotly Figure with probability analysis dashboard
    """
    # Detect enhanced columns
    has_momentum = "gaap_revision_momentum" in probability_df.columns
    has_spread = "gaap_norm_spread" in probability_df.columns
    has_streak = "quarterly_beat_streak" in probability_df.columns
    is_enhanced = has_momentum or has_spread

    if is_enhanced:
        n_rows = 3
        subplot_titles = (
            "Posterior Beat Probability Distribution",
            "Confidence Score by Sector",
            "Historical vs Posterior Beat Rate",
            "Probability Classification",
            "Revision Momentum vs P(Beat)" if has_momentum else "Beat Streak Distribution",
            "GAAP-Norm Spread vs P(Beat)" if has_spread else "Beat Streak Distribution",
        )
        specs = [
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
            [{"type": "scatter"}, {"type": "scatter"}],
        ]
    else:
        n_rows = 2
        subplot_titles = (
            "Posterior Beat Probability Distribution",
            "Confidence Score by Sector",
            "Historical vs Posterior Beat Rate",
            "Probability Classification",
        )
        specs = [
            [{"type": "histogram"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "pie"}],
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=subplot_titles,
        specs=specs,
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

    # 5. Enhanced panel: Revision momentum vs posterior (row 3, col 1)
    if is_enhanced and has_momentum:
        plot_df = probability_df[["gaap_revision_momentum", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_revision_momentum"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="Momentum vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=colors["secondary"],
                        opacity=0.6,
                    ),
                    text=(
                        probability_df.loc[plot_df.index, "ticker"]
                        if "ticker" in probability_df.columns
                        else None
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Momentum: %{x:.0f}<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=1,
            )
    elif is_enhanced and has_streak:
        streak_data = probability_df["quarterly_beat_streak"].dropna()
        if len(streak_data) > 0:
            streak_counts = streak_data.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=streak_counts.index.astype(str),
                    y=streak_counts.values,
                    name="Beat Streak",
                    marker_color=colors["accent"],
                ),
                row=3,
                col=1,
            )

    # 6. Enhanced panel: GAAP spread vs posterior (row 3, col 2)
    if is_enhanced and has_spread:
        plot_df = probability_df[["gaap_norm_spread", "posterior_beat_prob"]].dropna()
        if len(plot_df) > 0:
            abs_spread = plot_df["gaap_norm_spread"].abs()
            fig.add_trace(
                go.Scatter(
                    x=plot_df["gaap_norm_spread"],
                    y=plot_df["posterior_beat_prob"],
                    mode="markers",
                    name="GAAP Spread vs P(Beat)",
                    marker=dict(
                        size=8,
                        color=abs_spread,
                        colorscale="YlOrRd",
                        showscale=False,
                        opacity=0.7,
                    ),
                    text=(
                        probability_df.loc[plot_df.index, "ticker"]
                        if "ticker" in probability_df.columns
                        else None
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Spread: %{x:.1f}%<br>"
                        "P(Beat): %{y:.1%}<extra></extra>"
                    ),
                ),
                row=3,
                col=2,
            )
    elif is_enhanced and has_streak:
        streak_data = probability_df["quarterly_beat_streak"].dropna()
        if len(streak_data) > 0:
            streak_counts = streak_data.value_counts().sort_index()
            fig.add_trace(
                go.Bar(
                    x=streak_counts.index.astype(str),
                    y=streak_counts.values,
                    name="Beat Streak",
                    marker_color=colors["accent"],
                ),
                row=3,
                col=2,
            )

    # Update layout
    height = 1000 if is_enhanced else 700
    fig.update_layout(
        title=dict(text=title, font=dict(size=24, color="#1A2332")),
        height=height,
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

    if is_enhanced:
        if has_momentum:
            fig.update_xaxes(title_text="Revision Momentum (0-100)", row=3, col=1)
            fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=1)
        if has_spread:
            fig.update_xaxes(title_text="GAAP-Norm Spread %", row=3, col=2)
            fig.update_yaxes(title_text="Posterior P(Beat)", row=3, col=2)

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


class CategoryProbabilityAnalyzer:
    """
    Probability analyzer for a specific feature category/view.

    Provides Bayesian estimation, confidence intervals, and
    probability distributions for all features in a category.
    """

    def __init__(self, category_name: str, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        self.category_name = category_name
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def analyze_view(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
    ) -> pd.DataFrame:
        """
        Analyze all features in a view and return probability metrics.

        Returns DataFrame with probability estimates per stock per feature.
        """
        results = []
        identifier_cols = load_identifier_columns()
        id_data = df[[c for c in identifier_cols if c in df.columns]].copy()

        for feat in feature_cols:
            if feat not in df.columns:
                continue

            data = pd.to_numeric(df[feat], errors="coerce")
            if data.dropna().empty:
                continue

            # Calculate percentile rank as probability proxy
            percentile = data.rank(pct=True)

            # Bayesian credible intervals
            mean_val = data.mean()
            std_val = data.std()

            feat_results = id_data.copy()
            feat_results["feature"] = feat
            feat_results["value"] = data
            feat_results["percentile"] = percentile
            feat_results["z_score"] = (data - mean_val) / std_val if std_val > 0 else 0
            feat_results["prob_above_median"] = (percentile > 0.5).astype(float)

            results.append(feat_results)

        if not results:
            return pd.DataFrame()

        return pd.concat(results, ignore_index=True)


# =============================================================================
# RESAMPLED BEAT PROBABILITY MODEL (ArviZ-enhanced)
# =============================================================================


@dataclass
class ResampledBeatEstimate:
    """Result container for resampled earnings beat probability with technical conditioning."""

    ticker: str
    name: str
    sector: str
    base_posterior_mean: float
    resampled_posterior_mean: float
    technical_adjustment: float
    momentum_signal: float
    volatility_regime_score: float
    credible_interval_90: tuple[float, float]
    credible_interval_95: tuple[float, float]
    prob_beat_given_momentum: float
    earnings_season_flag: Optional[int] = None
    pre_earnings_window: Optional[int] = None


class ResampledBeatProbabilityModel:
    """
    Extends EarningsBeatProbabilityModel with technical resampling priors.

    Conditions the Beta posterior on technical signals from
    ``vw_features_technical_analysis`` and ``vw_features_momentum``,
    then uses multi-timeframe resampled returns as informative priors.

    Parameters
    ----------
    base_model : EarningsBeatProbabilityModel
        Pre-configured base model for standard Bayesian beat probabilities.
    momentum_weight : float
        Weight of momentum signal in prior adjustment (0–1).
    volatility_weight : float
        Weight of volatility regime in prior adjustment (0–1).
    n_posterior_samples : int
        Number of posterior draws for ArviZ output.
    n_chains : int
        Number of MCMC chains.
    """

    _MOMENTUM_COLS = [
        "price_momentum_1m",
        "price_momentum_3m",
        "price_momentum_6m",
        "range_52w_position",
        "ema_crossover_20_50",
    ]
    _TECHNICAL_COLS = [
        "ema_slope_20d",
        "ema_trend_consistency",
        "breakout_signal",
        "volatility_compression",
    ]
    _TEMPORAL_COLS = [
        "earnings_season_flag",
        "pre_earnings_window",
        "days_to_earnings",
        "reporting_freshness_score",
    ]
    _EARNINGS_COLS = [
        "eps_trajectory_score",
        "eps_positive_streak",
        "revision_quality_divergence",
    ]

    def __init__(
        self,
        base_model: Optional["EarningsBeatProbabilityModel"] = None,
        momentum_weight: float = 0.3,
        volatility_weight: float = 0.2,
        n_posterior_samples: int = 4000,
        n_chains: int = 4,
        random_seed: int = 42,
    ):
        self.base_model = base_model or EarningsBeatProbabilityModel()
        self.momentum_weight = np.clip(momentum_weight, 0, 1)
        self.volatility_weight = np.clip(volatility_weight, 0, 1)
        self.n_posterior_samples = n_posterior_samples
        self.n_chains = n_chains
        self.rng = np.random.default_rng(random_seed)

    def _compute_momentum_signal(self, row: pd.Series) -> float:
        """Composite momentum signal from available features (normalised to [-1, 1])."""
        signals = []
        for col in self._MOMENTUM_COLS:
            if col in row.index and pd.notna(row[col]):
                signals.append(float(row[col]))
        if not signals:
            return 0.0
        raw = np.mean(signals)
        return float(np.clip(raw / 100.0, -1.0, 1.0))

    def _compute_volatility_regime(self, row: pd.Series) -> float:
        """Volatility regime score (0=high vol, 1=low/compressed vol)."""
        score = 0.5
        if "volatility_compression" in row.index and pd.notna(row["volatility_compression"]):
            score = float(np.clip(row["volatility_compression"], 0, 1))
        elif "volatility_term_structure" in row.index and pd.notna(
            row["volatility_term_structure"]
        ):
            score = float(np.clip(1.0 - abs(row["volatility_term_structure"]) / 100, 0, 1))
        return score

    def _adjust_prior(
        self,
        base_alpha: float,
        base_beta: float,
        momentum_signal: float,
        vol_regime: float,
    ) -> tuple[float, float]:
        """
        Adjust Beta prior parameters based on technical signals.

        Positive momentum + low volatility → shift prior toward higher beat rate.
        """
        adjustment = (
            self.momentum_weight * momentum_signal + self.volatility_weight * (vol_regime - 0.5) * 2
        )
        concentration = base_alpha + base_beta
        shift = adjustment * 0.2 * concentration

        adjusted_alpha = max(0.5, base_alpha + shift)
        adjusted_beta = max(0.5, base_beta - shift)
        return adjusted_alpha, adjusted_beta

    def analyze(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        Run resampled beat probability analysis on equities DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Merged feature data (ideally from multiple vw_features_* views).
        sector_col : str
            Sector grouping column.
        ticker_col : str
            Ticker identifier column.

        Returns
        -------
        pd.DataFrame
            Enhanced beat probability results with technical conditioning.
        """
        base_results = self.base_model.analyze_dataframe_enhanced(
            df, sector_col=sector_col, ticker_col=ticker_col
        )
        if base_results.empty:
            return pd.DataFrame()

        results = []
        for _, row in base_results.iterrows():
            ticker = row.get(ticker_col, row.get("ticker", ""))

            orig_mask = (
                df[ticker_col] == ticker
                if ticker_col in df.columns
                else pd.Series(False, index=df.index)
            )
            orig_row = df.loc[orig_mask].iloc[0] if orig_mask.any() else pd.Series(dtype=float)

            momentum = self._compute_momentum_signal(orig_row)
            vol_regime = self._compute_volatility_regime(orig_row)

            base_alpha = row.get("posterior_alpha", 2.0)
            base_beta = row.get("posterior_beta", 2.0)
            base_mean = base_alpha / (base_alpha + base_beta)

            adj_alpha, adj_beta = self._adjust_prior(base_alpha, base_beta, momentum, vol_regime)
            adj_mean = adj_alpha / (adj_alpha + adj_beta)

            ci_90 = (
                float(stats.beta.ppf(0.05, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.95, adj_alpha, adj_beta)),
            )
            ci_95 = (
                float(stats.beta.ppf(0.025, adj_alpha, adj_beta)),
                float(stats.beta.ppf(0.975, adj_alpha, adj_beta)),
            )

            results.append(
                ResampledBeatEstimate(
                    ticker=str(ticker),
                    name=str(row.get("name", "")),
                    sector=str(row.get(sector_col, row.get("sector", ""))),
                    base_posterior_mean=float(base_mean),
                    resampled_posterior_mean=float(adj_mean),
                    technical_adjustment=float(adj_mean - base_mean),
                    momentum_signal=momentum,
                    volatility_regime_score=vol_regime,
                    credible_interval_90=ci_90,
                    credible_interval_95=ci_95,
                    prob_beat_given_momentum=float(1.0 - stats.beta.cdf(0.5, adj_alpha, adj_beta)),
                    earnings_season_flag=(
                        int(orig_row["earnings_season_flag"])
                        if "earnings_season_flag" in orig_row.index
                        and pd.notna(orig_row.get("earnings_season_flag"))
                        else None
                    ),
                    pre_earnings_window=(
                        int(orig_row["pre_earnings_window"])
                        if "pre_earnings_window" in orig_row.index
                        and pd.notna(orig_row.get("pre_earnings_window"))
                        else None
                    ),
                )
            )

        return pd.DataFrame([vars(r) for r in results])

    def build_inference_data(
        self,
        df: pd.DataFrame,
        sector_col: str = "sector",
        ticker_col: str = "ticker",
    ) -> "az.InferenceData | xr.Dataset | None":
        """
        Build ArviZ InferenceData from resampled beat probability posteriors.

        Returns
        -------
        arviz.InferenceData, xr.Dataset, or None
        """
        result_df = self.analyze(df, sector_col=sector_col, ticker_col=ticker_col)
        if result_df.empty:
            return None

        tickers = result_df["ticker"].values
        n_equities = len(tickers)

        base_results = self.base_model.analyze_dataframe_enhanced(
            df, sector_col=sector_col, ticker_col=ticker_col
        )

        adj_alphas = np.full(n_equities, 2.0)
        adj_betas = np.full(n_equities, 2.0)

        for i, ticker in enumerate(tickers):
            base_row = base_results.loc[base_results["ticker"] == ticker]
            if base_row.empty:
                continue
            base_a = float(base_row["posterior_alpha"].iloc[0])
            base_b = float(base_row["posterior_beta"].iloc[0])
            mom = float(result_df.iloc[i]["momentum_signal"])
            vol = float(result_df.iloc[i]["volatility_regime_score"])
            adj_alphas[i], adj_betas[i] = self._adjust_prior(base_a, base_b, mom, vol)

        posterior_samples = np.stack(
            [
                self.rng.beta(adj_alphas, adj_betas, size=(self.n_posterior_samples, n_equities))
                for _ in range(self.n_chains)
            ]
        )

        pp_samples = (self.rng.random(posterior_samples.shape) < posterior_samples).astype(int)

        coords = {
            "chain": np.arange(self.n_chains),
            "draw": np.arange(self.n_posterior_samples),
            "equity": tickers,
        }

        if ARVIZ_AVAILABLE and az is not None:
            return az.from_dict(
                posterior={"beat_probability": posterior_samples},
                posterior_predictive={"beat_outcome": pp_samples},
                observed_data={
                    "base_posterior_mean": result_df["base_posterior_mean"].values,
                    "momentum_signal": result_df["momentum_signal"].values,
                },
                constant_data={
                    "momentum_weight": np.array([self.momentum_weight]),
                    "volatility_weight": np.array([self.volatility_weight]),
                },
                coords=coords,
                dims={
                    "beat_probability": ["chain", "draw", "equity"],
                    "beat_outcome": ["chain", "draw", "equity"],
                },
            )
        elif xr is not None:
            return xr.Dataset(
                {"beat_probability": (["chain", "draw", "equity"], posterior_samples)},
                coords=coords,
            )
        return None


def create_view_probability_dashboard(
    view_df: pd.DataFrame,
    view_name: str,
    category_name: str,
) -> "go.Figure":
    """
    Create interactive probability dashboard for a feature view.

    Parameters
    ----------
    view_df : pd.DataFrame
        DataFrame from a specific vw_features view
    view_name : str
        Name of the view (e.g., "vw_features_momentum")
    category_name : str
        Display name for the category

    Returns
    -------
    go.Figure
        Plotly figure with probability distributions
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from finance_ml.analytics.data_utils import get_identifier_cols_set

    identifier_cols = get_identifier_cols_set()

    feature_cols = [c for c in view_df.columns if c not in identifier_cols][:6]  # Top 6

    n_features = len(feature_cols)
    if n_features == 0:
        fig = go.Figure()
        fig.add_annotation(text="No features available", x=0.5, y=0.5)
        return fig

    rows = (n_features + 1) // 2
    fig = make_subplots(rows=rows, cols=2, subplot_titles=[f"{feat}" for feat in feature_cols])

    for idx, feat in enumerate(feature_cols):
        row = (idx // 2) + 1
        col = (idx % 2) + 1

        data = pd.to_numeric(view_df[feat], errors="coerce").dropna()
        if len(data) > 10:
            fig.add_trace(
                go.Histogram(x=data, name=feat, showlegend=False, nbinsx=30),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=f"{category_name} - Probability Distributions",
        height=300 * rows,
        showlegend=False,
    )

    return fig


def export_probability_analytics_results(
    probability_df: pd.DataFrame,
    streak_df: pd.DataFrame,
    output_dir: Path,
    confidence_result: Optional[ModelConfidenceResult] = None,
    credit_risk_df: Optional[pd.DataFrame] = None,
    dividend_safety_df: Optional[pd.DataFrame] = None,
    price_target_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Export all probability analytics results to database and files.

    Uses standardized identifier columns from vw_identifier_columns to
    build each output table with consistent identifier ordering.

    Args:
        probability_df: DataFrame with probability analysis
        streak_df: DataFrame with streak analysis
        output_dir: Output directory path
        confidence_result: Optional confidence metrics
        credit_risk_df: Optional credit risk analysis results
        dividend_safety_df: Optional dividend safety analysis results
        price_target_df: Optional price target achievement results

    Returns:
        Dictionary with export information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports = {}

    def _safe_export(df: pd.DataFrame, table_name: str, reorder: bool = True) -> None:
        """Export a DataFrame via ExportConfig pipeline with error handling."""
        try:
            ordered = reorder_with_identifiers(df) if reorder else df
            cfg = ExportConfig(
                table_name=table_name,
                output_dir=str(output_dir),
            )
            export_to_db(ordered, cfg)
            export_to_csv(ordered, cfg)
            export_to_json(ordered, cfg)
            exports[f"{table_name}_db"] = f"analytics.{table_name}"
            exports[f"{table_name}_csv"] = str(output_dir / f"{table_name}.csv")
        except Exception as e:
            logger.error("Failed to export %s: %s", table_name, e)

    # Issue 7: Cast mixed-type columns to proper numeric dtypes before export
    for col in _NUMERIC_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(probability_df[col], errors="coerce")
    for col in _INTEGER_CAST_COLS:
        if col in probability_df.columns:
            probability_df[col] = pd.to_numeric(probability_df[col], errors="coerce").astype("Int64")

    # 1. Export probability analysis (Issue 3: table_name is canonical for both DB and CSV)
    _safe_export(probability_df, "earnings_probability_analysis")

    # 2. Export streak analysis
    _safe_export(streak_df, "eps_streak_analysis")

    # 3. Export confidence metrics
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
        _safe_export(conf_df, "model_confidence_metrics", reorder=False)

    # 4. Create and export summary statistics (Issue 6: validate columns first)
    required_prob_cols = {"posterior_beat_prob", "beat_classification", "confidence_score"}
    required_streak_cols = {"current_streak", "streak_type"}
    missing_prob = required_prob_cols - set(probability_df.columns)
    missing_streak = required_streak_cols - set(streak_df.columns)

    if missing_prob or missing_streak:
        logger.warning(
            "Summary skipped — missing columns: prob=%s, streak=%s",
            missing_prob or "none",
            missing_streak or "none",
        )
    else:
        try:
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
            _safe_export(summary_df, "probability_analytics_summary", reorder=False)
        except Exception as e:
            logger.error("Failed to compute/export summary statistics: %s", e)

    # 5. Export credit risk results
    if credit_risk_df is not None and len(credit_risk_df) > 0:
        _safe_export(credit_risk_df, "credit_risk_analysis")

    # 6. Export dividend safety results
    if dividend_safety_df is not None and len(dividend_safety_df) > 0:
        _safe_export(dividend_safety_df, "dividend_safety_analysis")

    # 7. Export price target achievement results
    if price_target_df is not None and len(price_target_df) > 0:
        _safe_export(price_target_df, "price_target_achievement")

    logger.info("Exported probability analytics results to database and %s", output_dir)
    return exports
