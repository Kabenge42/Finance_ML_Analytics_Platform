"""
InferenceData schema specification for finance_ml analytics.

Provides a standardised bridge between the Bayesian probability models
in probability_analytics.py / statistical_analysis.py and the ArviZ
InferenceData schema (xarray-backed, NetCDF-compatible).

The schema is anchored to the postgres.public metadata tables:
  - equities_schema_metadata   → observed_data coords & column roles
  - calculated_features_registry → feature dimensions & categories

References:
  - ArviZ InferenceData schema: https://python.arviz.org/en/stable/schema/schema.html
  - xarray Dataset: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy ArviZ import (optional dependency)
# ---------------------------------------------------------------------------
try:
    import arviz as az

    ARVIZ_AVAILABLE = True
except ImportError:
    az = None  # type: ignore[assignment]
    ARVIZ_AVAILABLE = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_VALID_SCHEMA_RE = None  # lazy-compiled


def _safe_values(series_or_df: "pd.Series | pd.DataFrame") -> np.ndarray:
    """Return a **writable** NumPy copy from a pandas object.

    Pandas ≥ 2.0 Copy-on-Write may return read-only views from ``.values``.
    This helper guarantees the caller can safely mutate or pass the array
    to libraries (ArviZ, xarray) that may attempt internal writes.
    """
    return np.array(series_or_df.values, copy=True)


def _validate_schema_name(schema: str) -> str:
    """Validate that *schema* is a safe SQL identifier (letters, digits, underscores)."""
    import re

    global _VALID_SCHEMA_RE
    if _VALID_SCHEMA_RE is None:
        _VALID_SCHEMA_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
    if not _VALID_SCHEMA_RE.match(schema):
        raise ValueError(f"Invalid schema name: {schema!r}")
    return schema


# =============================================================================
# 1. Equity Metadata Coordinate Builders
# =============================================================================

# Column roles from equities_schema_metadata that serve as coordinates
_ROLE_TO_COORD_DIM = {
    "id": "equity",  # ticker / isin → primary equity dimension
    "categorical": "equity",  # sector, industry, etc. as coords on equity dim
    "date": "equity",  # reference dates as coords on equity dim
}

# Feature category dimension from calculated_features_registry
_FEATURE_DIM = "feature"
_CATEGORY_DIM = "category"


@dataclass(frozen=True)
class EquityCoordinates:
    """
    Coordinate specification derived from equities_schema_metadata.

    Maps database column roles → xarray coordinate arrays for the
    ``equity`` dimension used across all InferenceData groups.

    Attributes
    ----------
    tickers : np.ndarray
        Ticker symbols (primary equity identifier).
    isins : np.ndarray
        ISIN codes.
    names : np.ndarray
        Company names.
    sectors : np.ndarray
        Sector classification per equity.
    industries : np.ndarray
        Industry classification per equity.
    countries : np.ndarray
        Country of incorporation.
    exchanges : np.ndarray
        Primary exchange.
    """

    tickers: np.ndarray
    isins: np.ndarray = field(default_factory=lambda: np.array([]))
    names: np.ndarray = field(default_factory=lambda: np.array([]))
    sectors: np.ndarray = field(default_factory=lambda: np.array([]))
    industries: np.ndarray = field(default_factory=lambda: np.array([]))
    countries: np.ndarray = field(default_factory=lambda: np.array([]))
    exchanges: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_xarray_coords(self) -> dict[str, Any]:
        """Build xarray-compatible coordinate dict for the equity dimension."""
        coords: dict[str, Any] = {"equity": self.tickers}
        if len(self.isins) == len(self.tickers):
            coords["isin"] = ("equity", self.isins)
        if len(self.names) == len(self.tickers):
            coords["name"] = ("equity", self.names)
        if len(self.sectors) == len(self.tickers):
            coords["sector"] = ("equity", self.sectors)
        if len(self.industries) == len(self.tickers):
            coords["industry"] = ("equity", self.industries)
        if len(self.countries) == len(self.tickers):
            coords["country"] = ("equity", self.countries)
        if len(self.exchanges) == len(self.tickers):
            coords["exchange"] = ("equity", self.exchanges)
        return coords

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "EquityCoordinates":
        """
        Construct coordinates from a feature DataFrame.

        Column names follow the aliases defined in equities_schema_metadata
        (e.g. ``ticker``, ``isin``, ``sector``, ``industry``).
        """

        def _col(name: str) -> np.ndarray:
            return _safe_values(df[name]) if name in df.columns else np.array([])

        tickers = _col("ticker")
        if len(tickers) == 0:
            raise ValueError("DataFrame must contain a 'ticker' column")

        return cls(
            tickers=tickers,
            isins=_col("isin"),
            names=_col("name"),
            sectors=_col("sector"),
            industries=_col("industry"),
            countries=_col("country"),
            exchanges=_col("exchange"),
        )


@dataclass(frozen=True)
class FeatureCoordinates:
    """
    Coordinate specification derived from calculated_features_registry.

    Maps feature_key / feature_alias → xarray coordinate arrays for
    the ``feature`` dimension.

    Attributes
    ----------
    feature_keys : np.ndarray
        Feature keys from the registry.
    feature_aliases : np.ndarray
        Human-readable aliases.
    categories : np.ndarray
        Category per feature (e.g. 'Profitability', 'Momentum').
    source_functions : np.ndarray
        Source SQL function per feature.
    primary_source_cols : np.ndarray
        Primary source column from equities table.
    """

    feature_keys: np.ndarray
    feature_aliases: np.ndarray = field(default_factory=lambda: np.array([]))
    categories: np.ndarray = field(default_factory=lambda: np.array([]))
    source_functions: np.ndarray = field(default_factory=lambda: np.array([]))
    primary_source_cols: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_xarray_coords(self) -> dict[str, Any]:
        """Build xarray coordinate dict for the feature dimension."""
        coords: dict[str, Any] = {"feature": self.feature_keys}
        if len(self.feature_aliases) == len(self.feature_keys):
            coords["feature_alias"] = ("feature", self.feature_aliases)
        if len(self.categories) == len(self.feature_keys):
            coords["category"] = ("feature", self.categories)
        if len(self.source_functions) == len(self.feature_keys):
            coords["source_function"] = ("feature", self.source_functions)
        if len(self.primary_source_cols) == len(self.feature_keys):
            coords["primary_source_col"] = ("feature", self.primary_source_cols)
        return coords

    @classmethod
    def from_dataframe(cls, registry_df: pd.DataFrame) -> "FeatureCoordinates":
        """Construct from a calculated_features_registry query result."""

        def _col(name: str) -> np.ndarray:
            return _safe_values(registry_df[name]) if name in registry_df.columns else np.array([])

        return cls(
            feature_keys=_col("feature_key"),
            feature_aliases=_col("feature_alias"),
            categories=_col("category"),
            source_functions=_col("source_function"),
            primary_source_cols=_col("primary_source_col"),
        )


# =============================================================================
# 2. InferenceData Factory — Bayesian Beat Probability
# =============================================================================


def build_beat_probability_inference_data(
    beat_results_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    n_posterior_samples: int = 4000,
    n_chains: int = 4,
    random_seed: int = 42,
) -> "az.InferenceData | xr.Dataset":
    """
    Build an ArviZ InferenceData object from EarningsBeatProbabilityModel results.

    Follows the ArviZ InferenceData schema specification with groups:
      - **posterior**: Beta(α, β) samples per equity (chain × draw × equity)
      - **posterior_predictive**: Bernoulli beat/miss draws per equity
      - **observed_data**: Historical beat indicators per equity
      - **constant_data**: Prior parameters, feature metadata
      - **sample_stats**: Log-likelihood per draw

    Parameters
    ----------
    beat_results_df : pd.DataFrame
        Output from ``EarningsBeatProbabilityModel.analyze_dataframe_enhanced()``
        containing posterior_alpha, posterior_beta, prior_alpha, prior_beta, etc.
    observed_df : pd.DataFrame
        Source feature DataFrame with observed financial data.
    n_posterior_samples : int, default 4000
        Number of posterior draws per chain.
    n_chains : int, default 4
        Number of MCMC chains to simulate.
    random_seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    arviz.InferenceData or xr.Dataset
        InferenceData if ArviZ is available, otherwise a plain xr.Dataset
        with the posterior group.

    References
    ----------
    .. [1] ArviZ InferenceData schema:
       https://python.arviz.org/en/stable/schema/schema.html
    """
    rng = np.random.default_rng(random_seed)
    equity_coords = EquityCoordinates.from_dataframe(beat_results_df)
    n_equities = len(equity_coords.tickers)

    # --- Posterior samples from Beta(posterior_alpha, posterior_beta) ---
    post_alpha = _safe_values(beat_results_df["posterior_alpha"])
    post_beta = _safe_values(beat_results_df["posterior_beta"])

    # Shape: (n_chains, n_posterior_samples, n_equities)
    posterior_samples = np.stack(
        [
            rng.beta(post_alpha, post_beta, size=(n_posterior_samples, n_equities))
            for _ in range(n_chains)
        ]
    )

    # --- Posterior predictive: Bernoulli draws ---
    posterior_predictive = (
        rng.random((n_chains, n_posterior_samples, n_equities)) < posterior_samples
    ).astype(int)

    # --- Log-likelihood ---
    # Use historical beat rate as the observed outcome proxy
    if "historical_beat_rate" in beat_results_df.columns:
        observed_beat = (_safe_values(beat_results_df["historical_beat_rate"]) > 0.5).astype(float)
    else:
        observed_beat = np.ones(n_equities) * 0.5

    log_lik = np.where(
        observed_beat[np.newaxis, np.newaxis, :] == 1,
        np.log(posterior_samples + 1e-12),
        np.log(1 - posterior_samples + 1e-12),
    )

    # --- Build coordinate dicts ---
    equity_xr_coords = equity_coords.to_xarray_coords()
    dims = {
        "beat_probability": ["chain", "draw", "equity"],
    }
    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_posterior_samples),
        **equity_xr_coords,
    }

    # --- Constant data (priors + metadata) ---
    prior_alpha = (
        _safe_values(beat_results_df["prior_alpha"])
        if "prior_alpha" in beat_results_df.columns
        else np.ones(n_equities)
    )
    prior_beta = (
        _safe_values(beat_results_df["prior_beta"])
        if "prior_beta" in beat_results_df.columns
        else np.ones(n_equities)
    )

    constant_data = {
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
    }
    if "confidence_score" in beat_results_df.columns:
        constant_data["confidence_score"] = _safe_values(beat_results_df["confidence_score"])

    if ARVIZ_AVAILABLE:
        idata = az.from_dict(
            posterior={"beat_probability": posterior_samples},
            posterior_predictive={"beat_outcome": posterior_predictive},
            observed_data={"observed_beat": observed_beat},
            log_likelihood={"beat_probability": log_lik},
            constant_data=constant_data,
            coords=coords,
            dims=dims,
        )
        logger.info(
            "Built InferenceData: %d chains × %d draws × %d equities",
            n_chains,
            n_posterior_samples,
            n_equities,
        )
        return idata
    else:
        # Fallback: return xr.Dataset with posterior group only
        ds = xr.Dataset(
            {"beat_probability": (["chain", "draw", "equity"], posterior_samples)},
            coords=coords,
        )
        logger.warning("ArviZ not available; returning xr.Dataset (posterior only)")
        return ds


# =============================================================================
# 3. InferenceData Factory — Credit Risk / Ruin Probability
# =============================================================================


def build_credit_risk_inference_data(
    ruin_results_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    n_posterior_samples: int = 4000,
    n_chains: int = 4,
    random_seed: int = 42,
) -> "az.InferenceData | xr.Dataset":
    """
    Build InferenceData for credit risk / ruin probability models.

    Groups:
      - **posterior**: Ruin probability samples per equity
      - **observed_data**: distress_risk_score, altman_z_score
      - **constant_data**: Capital, cash_burn, volatility inputs
      - **sample_stats**: Divergence flags

    Parameters
    ----------
    ruin_results_df : pd.DataFrame
        Output from ``fast_ruin_probability()`` or ``calculate_ruin_probability()``.
    observed_df : pd.DataFrame
        Source feature DataFrame.
    n_posterior_samples : int
        Draws per chain.
    n_chains : int
        Number of chains.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    arviz.InferenceData or xr.Dataset
    """
    rng = np.random.default_rng(random_seed)
    equity_coords = EquityCoordinates.from_dataframe(ruin_results_df)
    n_equities = len(equity_coords.tickers)

    # Ruin probability → Beta posterior (moment-match from point estimate)
    ruin_p = np.clip(_safe_values(ruin_results_df["ruin_probability"]), 0.001, 0.999)
    # Moment-matched Beta: concentration κ = 50 (moderate certainty)
    kappa = 50.0
    alpha_ruin = ruin_p * kappa
    beta_ruin = (1 - ruin_p) * kappa

    posterior_samples = np.stack(
        [
            rng.beta(alpha_ruin, beta_ruin, size=(n_posterior_samples, n_equities))
            for _ in range(n_chains)
        ]
    )

    coords = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_posterior_samples),
        **equity_coords.to_xarray_coords(),
    }
    dims = {"ruin_probability": ["chain", "draw", "equity"]}

    # Observed financial health indicators — set index ONCE before the loop
    observed = {}
    obs_cols = ("distress_risk_score", "altman_z_score", "cash_runway_months")
    available_obs = [c for c in obs_cols if c in observed_df.columns]
    if available_obs:
        if "ticker" in observed_df.columns:
            obs_indexed = observed_df.set_index("ticker")
        elif observed_df.index.name is not None:
            obs_indexed = observed_df  # already has a named index
        else:
            obs_indexed = None  # can't align — skip observed data

        if obs_indexed is not None:
            for col in available_obs:
                observed[col] = _safe_values(obs_indexed[col].reindex(equity_coords.tickers))

    constant_data = {}
    for col in ("capital", "cash_burn", "volatility"):
        if col in ruin_results_df.columns:
            constant_data[col] = _safe_values(ruin_results_df[col])

    if "risk_tier" in ruin_results_df.columns:
        constant_data["risk_tier"] = _safe_values(ruin_results_df["risk_tier"].astype(str))

    if ARVIZ_AVAILABLE:
        return az.from_dict(
            posterior={"ruin_probability": posterior_samples},
            observed_data=observed if observed else None,
            constant_data=constant_data if constant_data else None,
            coords=coords,
            dims=dims,
        )
    else:
        return xr.Dataset(
            {"ruin_probability": (["chain", "draw", "equity"], posterior_samples)},
            coords=coords,
        )


# =============================================================================
# 4. InferenceData Factory — Monte Carlo Price Target
# =============================================================================


def build_monte_carlo_inference_data(
    mc_results_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    n_simulations: int = 10000,
    random_seed: int = 42,
) -> "az.InferenceData | xr.Dataset":
    """
    Build InferenceData for Monte Carlo price target simulations.

    Uses a single chain with ``n_simulations`` draws (forward simulation,
    not MCMC). Groups:
      - **posterior_predictive**: Simulated price targets (1 × draws × equity)
      - **observed_data**: last_price, pt_median
      - **constant_data**: pt_low, pt_high, pt_median inputs

    Parameters
    ----------
    mc_results_df : pd.DataFrame
        Output from ``fast_monte_carlo_simulation()``.
    observed_df : pd.DataFrame
        Source feature DataFrame.
    n_simulations : int
        Number of forward simulations per equity.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    arviz.InferenceData or xr.Dataset
    """
    from scipy import stats as sp_stats

    rng = np.random.default_rng(random_seed)
    equity_coords = EquityCoordinates.from_dataframe(mc_results_df)
    n_equities = len(equity_coords.tickers)

    # Reconstruct triangular simulations — always writable copies
    last_prices = _safe_values(mc_results_df["last_price"])
    pt_low = last_prices * 0.8
    pt_high = last_prices * 1.5
    pt_median = (
        _safe_values(mc_results_df["pt_median"])
        if "pt_median" in mc_results_df.columns
        else (pt_low + pt_high) / 2
    )

    # Retrieve exact columns from observed_df if available
    if "price_target_low" in observed_df.columns:
        for i, t in enumerate(equity_coords.tickers):
            row = observed_df[observed_df["ticker"] == t]
            if not row.empty:
                val_low = row["price_target_low"].iloc[0]
                val_high = row["price_target_high"].iloc[0]
                val_median = row["price_target_median"].iloc[0]
                if pd.notna(val_low) and val_low > 0:
                    pt_low[i] = val_low
                if pd.notna(val_high) and val_high > 0:
                    pt_high[i] = val_high
                if pd.notna(val_median) and val_median > 0:
                    pt_median[i] = val_median

    # Triangular distribution simulation
    scale = np.maximum(pt_high - pt_low, 0.01)
    c = np.clip((pt_median - pt_low) / scale, 0.01, 0.99)

    simulated_prices = np.zeros((1, n_simulations, n_equities))
    for i in range(n_equities):
        simulated_prices[0, :, i] = sp_stats.triang.rvs(
            c[i], loc=pt_low[i], scale=scale[i], size=n_simulations, random_state=rng
        )

    coords = {
        "chain": np.array([0]),
        "draw": np.arange(n_simulations),
        **equity_coords.to_xarray_coords(),
    }
    dims = {"simulated_price": ["chain", "draw", "equity"]}

    observed = {
        "last_price": last_prices,
    }
    constant = {
        "pt_low": pt_low,
        "pt_median": pt_median,
        "pt_high": pt_high,
    }

    if ARVIZ_AVAILABLE:
        return az.from_dict(
            posterior_predictive={"simulated_price": simulated_prices},
            observed_data=observed,
            constant_data=constant,
            coords=coords,
            dims=dims,
        )
    else:
        return xr.Dataset(
            {"simulated_price": (["chain", "draw", "equity"], simulated_prices)},
            coords=coords,
        )


# =============================================================================
# 5. InferenceData Factory — Bayesian Category Analysis
# =============================================================================


def build_category_analysis_inference_data(
    analysis_results: dict[str, dict],
    observed_df: pd.DataFrame,
    category_name: str,
    features: Sequence[str],
    n_posterior_samples: int = 4000,
    n_chains: int = 4,
    random_seed: int = 42,
    feature_coords: Optional[FeatureCoordinates] = None,
) -> "az.InferenceData | xr.Dataset":
    """
    Build InferenceData for ``bayesian_category_analysis()`` results.

    Groups:
      - **posterior**: Normal posterior samples for feature means
        (chain × draw × feature)
      - **observed_data**: Raw feature values (equity × feature)
      - **constant_data**: Prior mean, prior std, category metadata

    Parameters
    ----------
    analysis_results : dict
        Output from ``bayesian_category_analysis()``.
    observed_df : pd.DataFrame
        Source feature DataFrame.
    category_name : str
        Feature category name.
    features : Sequence[str]
        Feature names analysed.
    n_posterior_samples : int
        Draws per chain.
    n_chains : int
        Number of chains.
    random_seed : int
        Seed for reproducibility.
    feature_coords : FeatureCoordinates, optional
        Pre-built feature coordinates from calculated_features_registry.

    Returns
    -------
    arviz.InferenceData or xr.Dataset
    """
    rng = np.random.default_rng(random_seed)

    analysed_features = [f for f in features if f in analysis_results]
    n_features = len(analysed_features)
    if n_features == 0:
        raise ValueError(f"No analysed features found for category '{category_name}'")

    # Posterior samples: Normal(posterior_mean, posterior_std)
    posterior_means = np.array([analysis_results[f]["posterior_mean"] for f in analysed_features])
    posterior_stds = np.array([analysis_results[f]["posterior_std"] for f in analysed_features])

    posterior_samples = np.stack(
        [
            rng.normal(posterior_means, posterior_stds, size=(n_posterior_samples, n_features))
            for _ in range(n_chains)
        ]
    )  # (n_chains, n_posterior_samples, n_features)

    coords: dict[str, Any] = {
        "chain": np.arange(n_chains),
        "draw": np.arange(n_posterior_samples),
        "feature": np.array(analysed_features),
    }
    if feature_coords is not None:
        # Add registry metadata as feature-level coordinates
        mask = np.isin(feature_coords.feature_keys, analysed_features)
        if mask.sum() > 0 and len(feature_coords.categories) > 0:
            coords["category"] = ("feature", feature_coords.categories[mask])

    dims = {"feature_mean": ["chain", "draw", "feature"]}

    # Observed data: raw feature values per equity
    equity_coords = EquityCoordinates.from_dataframe(observed_df)
    # Writable copy; NaN values are preserved but won't crash ArviZ
    observed_matrix = _safe_values(observed_df[analysed_features])

    observed_ds = {
        "observed_values": observed_matrix,
    }
    constant = {
        "category_name": np.array([category_name]),
        "n_observations": np.array([analysis_results[f]["n_obs"] for f in analysed_features]),
        "sample_mean": np.array([analysis_results[f]["sample_mean"] for f in analysed_features]),
    }

    if ARVIZ_AVAILABLE:
        return az.from_dict(
            posterior={"feature_mean": posterior_samples},
            observed_data=observed_ds,
            constant_data=constant,
            coords=coords,
            dims=dims,
        )
    else:
        return xr.Dataset(
            {"feature_mean": (["chain", "draw", "feature"], posterior_samples)},
            coords=coords,
        )


# =============================================================================
# 6. Database Metadata Loaders
# =============================================================================


def load_equity_coordinates_from_db(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> EquityCoordinates:
    """
    Load EquityCoordinates from equities_schema_metadata + equities tables.

    Queries ``equities_schema_metadata`` for column roles, then fetches
    the actual identifier/categorical columns from the ``equities`` table.

    Parameters
    ----------
    db_url : str, optional
        Database URL. Falls back to DB_URL env var.
    schema : str
        Schema name.

    Returns
    -------
    EquityCoordinates
    """
    import os
    from sqlalchemy import create_engine, text

    schema = _validate_schema_name(schema)

    url = db_url or os.environ.get("DB_URL")
    if not url:
        raise ValueError("DB_URL not available")

    engine = create_engine(url)

    # Get identifier columns from metadata table
    meta_query = text(f"""
        SELECT column_alias
        FROM {schema}.equities_schema_metadata
        WHERE role IN ('id', 'categorical')
        AND column_alias IN ('ticker', 'isin', 'name', 'sector', 'industry', 'country', 'exchange')
        ORDER BY column_alias
    """)

    with engine.connect() as conn:
        available_cols = [row[0] for row in conn.execute(meta_query)]

    if not available_cols:
        raise ValueError("No identifier columns found in equities_schema_metadata")

    df = pd.read_sql(
        f"SELECT {', '.join(available_cols)} FROM {schema}.mv_all_stock_features", engine
    )
    return EquityCoordinates.from_dataframe(df)


def load_feature_coordinates_from_db(
    db_url: Optional[str] = None,
    schema: str = "public",
    category_filter: Optional[str] = None,
) -> FeatureCoordinates:
    """
    Load FeatureCoordinates from calculated_features_registry.

    Parameters
    ----------
    db_url : str, optional
        Database URL.
    schema : str
        Schema name.
    category_filter : str, optional
        Filter to a specific category.

    Returns
    -------
    FeatureCoordinates
    """
    import os
    from sqlalchemy import create_engine, text

    schema = _validate_schema_name(schema)

    url = db_url or os.environ.get("DB_URL")
    if not url:
        raise ValueError("DB_URL not available")

    engine = create_engine(url)
    query = f"""
        SELECT feature_key, feature_alias, category, source_function, primary_source_col
        FROM {schema}.calculated_features_registry
    """
    if category_filter:
        query += " WHERE category = :cat"

    query += " ORDER BY category, feature_key"

    df = pd.read_sql(
        text(query), engine, params={"cat": category_filter} if category_filter else None
    )
    return FeatureCoordinates.from_dataframe(df)


# =============================================================================
# 7. Convenience: Summary & Diagnostics
# =============================================================================


def summarize_inference_data(idata: Any) -> dict[str, Any]:
    """
    Produce a summary dict for an InferenceData (or xr.Dataset fallback).

    Returns
    -------
    dict
        Keys: groups, n_chains, n_draws, n_equities, variables, r_hat (if available).
    """
    summary: dict[str, Any] = {}

    if ARVIZ_AVAILABLE and isinstance(idata, az.InferenceData):
        summary["groups"] = list(idata.groups())

        # Prefer posterior; fall back to posterior_predictive (e.g. Monte Carlo)
        if hasattr(idata, "posterior"):
            group = idata.posterior
        elif hasattr(idata, "posterior_predictive"):
            group = idata.posterior_predictive
        else:
            group = None

        if group is not None:
            summary["n_chains"] = group.sizes.get("chain", 0)
            summary["n_draws"] = group.sizes.get("draw", 0)
            summary["n_equities"] = group.sizes.get("equity", group.sizes.get("feature", 0))
            summary["variables"] = list(group.data_vars)

            # R-hat convergence diagnostic (only meaningful for posterior)
            if hasattr(idata, "posterior"):
                try:
                    rhat = az.rhat(idata)
                    summary["r_hat"] = {
                        var: float(rhat[var].max().values) for var in rhat.data_vars
                    }
                except Exception:
                    summary["r_hat"] = None
    elif isinstance(idata, xr.Dataset):
        summary["groups"] = ["posterior (xr.Dataset fallback)"]
        summary["n_chains"] = idata.sizes.get("chain", 0)
        summary["n_draws"] = idata.sizes.get("draw", 0)
        summary["variables"] = list(idata.data_vars)
    else:
        summary["error"] = f"Unknown type: {type(idata)}"

    return summary
