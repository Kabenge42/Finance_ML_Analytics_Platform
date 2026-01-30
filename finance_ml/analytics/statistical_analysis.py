"""
Statistical analysis functions for feature analytics.

This module provides advanced statistical analysis including:
- Bayesian parameter estimation
- MCMC sampling (Metropolis-Hastings)
- Monte Carlo simulations
- Distribution fitting
- Conditional probability analysis
- Investor's ruin probability models
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def bayesian_category_analysis(
    df: pd.DataFrame,
    category_name: str,
    features: list,
    prior_mean: float = 0,
    prior_std: float = 10,
) -> dict:
    """
    Bayesian analysis of feature distributions within a category.

    Uses Normal-Normal conjugate prior for continuous features.
    Prior: μ ~ N(prior_mean, prior_std²)
    Likelihood: X | μ ~ N(μ, σ²)
    Posterior: μ | X ~ N(posterior_mean, posterior_var)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature data
    category_name : str
        Name of the feature category
    features : list
        List of feature names to analyze
    prior_mean : float, default 0
        Prior mean for the parameter
    prior_std : float, default 10
        Prior standard deviation

    Returns
    -------
    dict
        Dictionary mapping feature names to posterior statistics

    Examples
    --------
    >>> results = bayesian_category_analysis(df, 'Profitability', ['roe', 'roa'])
    >>> print(results['roe']['posterior_mean'])
    """
    results = {}

    for feature in features:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) < 50:
            continue

        n = len(data)
        sample_mean = data.mean()
        sample_var = data.var()

        # Posterior parameters (Normal-Normal conjugate)
        prior_var = prior_std**2
        posterior_var = 1 / (1 / prior_var + n / sample_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + n * sample_mean / sample_var)
        posterior_std = np.sqrt(posterior_var)

        # 95% Credible Interval
        ci_low = posterior_mean - 1.96 * posterior_std
        ci_high = posterior_mean + 1.96 * posterior_std

        # Probability that true mean > 0
        prob_positive = 1 - stats.norm.cdf(0, posterior_mean, posterior_std)

        results[feature] = {
            "n_obs": n,
            "sample_mean": sample_mean,
            "sample_std": np.sqrt(sample_var),
            "posterior_mean": posterior_mean,
            "posterior_std": posterior_std,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "prob_positive": prob_positive,
        }

    return results


def metropolis_hastings_sampler(
    data: np.ndarray,
    n_samples: int = 10000,
    burn_in: int = 2000,
    proposal_std: float = 0.5,
    prior_mean: float = 0,
    prior_std: float = 10,
) -> Tuple[np.ndarray, float]:
    """
    Metropolis-Hastings MCMC sampler for estimating posterior of mean parameter.

    Assumes: X ~ N(μ, σ²) with σ known from data
             Prior: μ ~ N(prior_mean, prior_std²)

    Parameters
    ----------
    data : np.ndarray
        Observed data
    n_samples : int, default 10000
        Number of MCMC samples to generate
    burn_in : int, default 2000
        Number of initial samples to discard
    proposal_std : float, default 0.5
        Standard deviation of proposal distribution
    prior_mean : float, default 0
        Prior mean
    prior_std : float, default 10
        Prior standard deviation

    Returns
    -------
    tuple
        (samples, acceptance_rate) - MCMC samples and acceptance rate

    Examples
    --------
    >>> samples, acc_rate = metropolis_hastings_sampler(data, n_samples=5000)
    >>> print(f"Acceptance rate: {acc_rate:.2%}")
    """
    data_mean = np.mean(data)
    data_std = np.std(data)
    n = len(data)

    # Initialize
    current = data_mean
    samples = np.zeros(n_samples)
    accepted = 0

    def log_posterior(mu):
        # Log-likelihood
        ll = -n / 2 * np.log(2 * np.pi * data_std**2) - np.sum((data - mu) ** 2) / (2 * data_std**2)
        # Log-prior
        lp = -0.5 * ((mu - prior_mean) / prior_std) ** 2
        return ll + lp

    current_log_post = log_posterior(current)

    for i in range(n_samples + burn_in):
        # Propose new value
        proposal = current + np.random.normal(0, proposal_std)
        proposal_log_post = log_posterior(proposal)

        # Acceptance ratio (log scale)
        log_alpha = proposal_log_post - current_log_post

        # Accept or reject
        if np.log(np.random.random()) < log_alpha:
            current = proposal
            current_log_post = proposal_log_post
            if i >= burn_in:
                accepted += 1

        if i >= burn_in:
            samples[i - burn_in] = current

    acceptance_rate = accepted / n_samples

    return samples, acceptance_rate


def mcmc_student_t(
    data: np.ndarray, n_samples: int = 10000, burn_in: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MCMC for Student's t location parameter with heavier tails.

    Better for financial data with outliers.

    Parameters
    ----------
    data : np.ndarray
        Observed data
    n_samples : int, default 10000
        Number of MCMC samples
    burn_in : int, default 2000
        Burn-in period

    Returns
    -------
    tuple
        (samples_mu, samples_df) - Location and degrees of freedom samples

    Examples
    --------
    >>> mu_samples, df_samples = mcmc_student_t(data)
    >>> print(f"Posterior mean: {mu_samples.mean():.2f}")
    """
    from scipy.stats import t as student_t

    # Initial estimates
    current_mu = np.median(data)
    current_df = 5  # degrees of freedom
    data_scale = stats.median_abs_deviation(data)

    samples_mu = np.zeros(n_samples)
    samples_df = np.zeros(n_samples)

    def log_likelihood(mu, df):
        return np.sum(student_t.logpdf(data, df, loc=mu, scale=data_scale))

    current_ll = log_likelihood(current_mu, current_df)

    for i in range(n_samples + burn_in):
        # Propose new mu and df
        prop_mu = current_mu + np.random.normal(0, 0.1)
        prop_df = max(2, current_df + np.random.normal(0, 0.5))

        prop_ll = log_likelihood(prop_mu, prop_df)

        if np.log(np.random.random()) < (prop_ll - current_ll):
            current_mu = prop_mu
            current_df = prop_df
            current_ll = prop_ll

        if i >= burn_in:
            samples_mu[i - burn_in] = current_mu
            samples_df[i - burn_in] = current_df

    return samples_mu, samples_df


def hierarchical_mcmc_by_sector(
    df: pd.DataFrame, feature: str, sector_col: str = "industry", n_samples: int = 8000
) -> dict:
    """
    Hierarchical MCMC: estimate sector-level means with pooling toward global mean.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature : str
        Feature name to analyze
    sector_col : str, default 'industry'
        Column name for sector grouping
    n_samples : int, default 8000
        Number of MCMC samples

    Returns
    -------
    dict
        Dictionary mapping sectors to posterior statistics

    Examples
    --------
    >>> results = hierarchical_mcmc_by_sector(df, 'roe')
    >>> print(results['Technology']['posterior_mean'])
    """
    results = {}
    sectors = df[sector_col].dropna().unique()

    # Global parameters
    global_data = df[feature].dropna()
    global_mean = global_data.mean()
    global_std = global_data.std()

    for sector in sectors:
        sector_data = df[df[sector_col] == sector][feature].dropna().values
        if len(sector_data) < 30:
            continue

        # Shrinkage toward global mean based on sample size
        n = len(sector_data)
        shrinkage = n / (n + 10)  # Simple shrinkage factor

        sector_mean = sector_data.mean()
        sector_std = sector_data.std()

        # Posterior with shrinkage
        posterior_mean = shrinkage * sector_mean + (1 - shrinkage) * global_mean
        posterior_std = sector_std / np.sqrt(n)

        # MCMC samples from posterior
        samples = np.random.normal(posterior_mean, posterior_std, n_samples)

        results[sector] = {
            "raw_mean": sector_mean,
            "posterior_mean": posterior_mean,
            "shrinkage": shrinkage,
            "samples": samples,
            "n_obs": n,
        }

    return results


def fit_distributions_by_category(
    df: pd.DataFrame, category: str, features: list, n_simulations: int = 10000
) -> dict:
    """
    Fit multiple distributions and select best fit using AIC.

    Simulate future scenarios using best-fit distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    category : str
        Category name
    features : list
        List of features to fit
    n_simulations : int, default 10000
        Number of simulations

    Returns
    -------
    dict
        Dictionary with fitted distributions and simulations

    Examples
    --------
    >>> fits = fit_distributions_by_category(df, 'Profitability', ['roe', 'roa'])
    >>> print(fits['roe']['best_distribution'])
    """
    from scipy.stats import norm, t, skewnorm

    results = {}

    for feature in features:
        if feature not in df.columns:
            continue

        data = df[feature].dropna()
        if len(data) < 100:
            continue

        # Remove extreme outliers for fitting
        q01, q99 = data.quantile([0.01, 0.99])
        data_clean = data[(data >= q01) & (data <= q99)]

        # Fit distributions
        fits = {}

        # Normal
        try:
            params_norm = norm.fit(data_clean)
            ll_norm = norm.logpdf(data_clean, *params_norm).sum()
            fits["normal"] = {"params": params_norm, "aic": 2 * 2 - 2 * ll_norm}
        except:
            pass

        # Student's t
        try:
            params_t = t.fit(data_clean)
            ll_t = t.logpdf(data_clean, *params_t).sum()
            fits["student_t"] = {"params": params_t, "aic": 2 * 3 - 2 * ll_t}
        except:
            pass

        # Skew Normal
        try:
            params_skew = skewnorm.fit(data_clean)
            ll_skew = skewnorm.logpdf(data_clean, *params_skew).sum()
            fits["skew_normal"] = {"params": params_skew, "aic": 2 * 3 - 2 * ll_skew}
        except:
            pass

        if not fits:
            continue

        # Select best fit by AIC
        best_dist = min(fits.keys(), key=lambda k: fits[k]["aic"])
        best_params = fits[best_dist]["params"]

        # Simulate from best distribution
        if best_dist == "normal":
            simulations = norm.rvs(*best_params, size=n_simulations)
        elif best_dist == "student_t":
            simulations = t.rvs(*best_params, size=n_simulations)
        else:
            simulations = skewnorm.rvs(*best_params, size=n_simulations)

        # Calculate VaR and CVaR
        var_5 = np.percentile(simulations, 5)
        cvar_5 = simulations[simulations <= var_5].mean()

        results[feature] = {
            "best_distribution": best_dist,
            "params": best_params,
            "aic": fits[best_dist]["aic"],
            "simulated_mean": simulations.mean(),
            "simulated_std": simulations.std(),
            "var_5_pct": var_5,
            "cvar_5_pct": cvar_5,
            "simulations": simulations,
        }

    return results


def calculate_ruin_probability(
    df: pd.DataFrame,
    initial_capital_col: str = "market_cap",
    cash_burn_col: str = "cash_burn_rate",
    volatility_col: str = "volatility_regime",
) -> pd.DataFrame:
    """
    Calculate investor's ruin probability using modified Gambler's Ruin framework.

    P(ruin) ≈ exp(-2 * μ * W / σ²) for μ > 0 (drift)
    where W = initial wealth, μ = expected return, σ = volatility

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with financial metrics
    initial_capital_col : str, default 'market_cap'
        Column for initial capital
    cash_burn_col : str, default 'cash_burn_rate'
        Column for cash burn rate
    volatility_col : str, default 'volatility_regime'
        Column for volatility measure

    Returns
    -------
    pd.DataFrame
        DataFrame with ruin probabilities and risk tiers

    Examples
    --------
    >>> ruin_df = calculate_ruin_probability(df)
    >>> high_risk = ruin_df[ruin_df['ruin_probability'] > 0.6]
    """
    result = df[["ticker", "name", "industry", "market_cap", "distress_risk_score"]].copy()

    # Proxy expected return from FCF yield and earnings trajectory
    if "fcf_yield" in df.columns and "eps_trajectory_score" in df.columns:
        fcf_norm = df["fcf_yield"].clip(-20, 50) / 100
        eps_norm = df["eps_trajectory_score"] / 100
        result["expected_drift"] = (fcf_norm * 0.6 + eps_norm * 0.4).fillna(0)
    else:
        result["expected_drift"] = 0.05  # Default 5% drift

    # Volatility proxy
    if volatility_col in df.columns:
        result["volatility"] = df[volatility_col].abs().clip(5, 80) / 100
    elif "beta_momentum" in df.columns:
        result["volatility"] = (df["beta_momentum"].abs() * 0.2).clip(0.1, 0.8)
    else:
        result["volatility"] = 0.25

    # Cash runway as wealth buffer
    if "cash_runway_months" in df.columns:
        result["wealth_buffer"] = df["cash_runway_months"].clip(0, 120) / 12
    else:
        result["wealth_buffer"] = 3  # Default 3 years

    # Calculate ruin probability
    mu = result["expected_drift"]
    sigma = result["volatility"]
    W = result["wealth_buffer"]

    sigma_sq = sigma**2 + 1e-6

    result["ruin_probability"] = np.where(
        mu > 0,
        np.exp(-2 * mu * W / sigma_sq).clip(0, 1),
        np.minimum(1.0, 0.5 + 0.5 * np.abs(mu) * W),
    )

    result["survival_probability"] = 1 - result["ruin_probability"]

    # Risk tier classification
    result["risk_tier"] = pd.cut(
        result["ruin_probability"],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"],
    )

    return result


def calculate_conditional_probabilities(
    df: pd.DataFrame, feature_categories: dict, distress_threshold: float = 30
) -> pd.DataFrame:
    """
    Calculate conditional probability of financial distress given feature conditions.

    P(Distress | High Feature) vs P(Distress | Low Feature)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    feature_categories : dict
        Dictionary of feature categories
    distress_threshold : float, default 30
        Threshold for distress classification

    Returns
    -------
    pd.DataFrame
        DataFrame with conditional probabilities

    Examples
    --------
    >>> cond_probs = calculate_conditional_probabilities(df, FEATURE_CATEGORIES)
    >>> top_features = cond_probs.nlargest(10, 'separation')
    """
    results = []

    df["is_distressed"] = df["distress_risk_score"] < distress_threshold
    base_distress_rate = df["is_distressed"].mean()

    for category, features in feature_categories.items():
        for feature in features[:5]:  # Top 5 features per category
            if feature not in df.columns:
                continue

            data = df[[feature, "is_distressed"]].dropna()
            if len(data) < 100:
                continue

            median_val = data[feature].median()

            # P(Distress | Feature > Median)
            high_mask = data[feature] > median_val
            p_distress_high = data.loc[high_mask, "is_distressed"].mean()

            # P(Distress | Feature <= Median)
            p_distress_low = data.loc[~high_mask, "is_distressed"].mean()

            # Lift ratio
            lift_high = p_distress_high / base_distress_rate if base_distress_rate > 0 else 1
            lift_low = p_distress_low / base_distress_rate if base_distress_rate > 0 else 1

            results.append(
                {
                    "category": category,
                    "feature": feature,
                    "p_distress_high": p_distress_high,
                    "p_distress_low": p_distress_low,
                    "lift_high": lift_high,
                    "lift_low": lift_low,
                    "separation": abs(p_distress_high - p_distress_low),
                }
            )

    return pd.DataFrame(results).sort_values("separation", ascending=False)


# =============================================================================
# Enhanced Statistical Methods
# =============================================================================


def kalman_filter_price_target(
    df: pd.DataFrame,
    observation_col: str = "last_price",
    target_col: str = "price_target",
    process_variance: float = 1e-5,
    measurement_variance: float = 0.1,
) -> pd.DataFrame:
    """
    Kalman filter for smoothing price targets and estimating true value.

    State-space model:
    - State: True underlying value
    - Observation: Noisy analyst price targets

    Parameters
    ----------
    df : pd.DataFrame
        Must contain observation_col and target_col
    observation_col : str, default 'last_price'
        Column with current price observations
    target_col : str, default 'price_target'
        Column with analyst price targets
    process_variance : float, default 1e-5
        Q - process noise covariance (how much true value changes)
    measurement_variance : float, default 0.1
        R - measurement noise covariance (analyst estimate error)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - ticker: Stock identifier
        - kalman_estimate: Filtered price estimate
        - kalman_variance: Estimation uncertainty
        - kalman_gain: Filter gain at each step
        - signal_strength: Confidence in the estimate (1/variance)

    Examples
    --------
    >>> kalman_df = kalman_filter_price_target(df)
    >>> high_confidence = kalman_df[kalman_df['signal_strength'] > 10]
    """
    if observation_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(
            columns=[
                "ticker",
                "kalman_estimate",
                "kalman_variance",
                "kalman_gain",
                "signal_strength",
            ]
        )

    results = []

    for idx, row in df.iterrows():
        # Get observation and target
        obs = row.get(observation_col)
        target = row.get(target_col)

        # Skip if missing data
        if pd.isna(obs) or pd.isna(target) or obs <= 0 or target <= 0:
            continue

        # Initialize state with observation
        x_est = float(obs)
        p_est = 1.0  # Initial covariance

        # Measurement (analyst target)
        z = float(target)

        # Predict step (no control input, random walk model)
        x_pred = x_est
        p_pred = p_est + process_variance

        # Update step
        kalman_gain = p_pred / (p_pred + measurement_variance)
        x_est = x_pred + kalman_gain * (z - x_pred)
        p_est = (1 - kalman_gain) * p_pred

        # Signal strength (inverse of variance)
        signal_strength = 1.0 / (p_est + 1e-10)

        results.append(
            {
                "ticker": row.get("ticker", str(idx)),
                "kalman_estimate": x_est,
                "kalman_variance": p_est,
                "kalman_gain": kalman_gain,
                "signal_strength": signal_strength,
                "original_price": obs,
                "original_target": target,
                "filtered_upside": (x_est - obs) / obs * 100 if obs > 0 else 0,
            }
        )

    return pd.DataFrame(results)


def kalman_momentum_filter(
    df: pd.DataFrame,
    momentum_cols: list = None,
    process_variance: float = 0.01,
    measurement_variance: float = 0.1,
) -> pd.DataFrame:
    """
    Apply Kalman filter to smooth noisy momentum indicators.

    Useful for reducing whipsaw signals in trend following.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with momentum columns
    momentum_cols : list, optional
        List of momentum column names. Default: ['price_momentum_1m',
        'price_momentum_3m', 'price_momentum_6m']
    process_variance : float, default 0.01
        Process noise variance
    measurement_variance : float, default 0.1
        Measurement noise variance

    Returns
    -------
    pd.DataFrame
        DataFrame with filtered momentum columns (suffixed with '_filtered')

    Examples
    --------
    >>> filtered_df = kalman_momentum_filter(df)
    >>> print(filtered_df['price_momentum_1m_filtered'].head())
    """
    if momentum_cols is None:
        momentum_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]

    available_cols = [col for col in momentum_cols if col in df.columns]

    if not available_cols:
        return df.copy()

    result = df.copy()

    for col in available_cols:
        data = df[col].values
        n = len(data)

        # Initialize
        x_est = np.zeros(n)
        p_est = np.ones(n)

        # First value initialization
        valid_idx = np.where(~np.isnan(data))[0]
        if len(valid_idx) == 0:
            result[f"{col}_filtered"] = np.nan
            continue

        first_valid = valid_idx[0]
        x_est[first_valid] = data[first_valid]
        p_est[first_valid] = 1.0

        # Forward pass
        for i in range(first_valid + 1, n):
            if np.isnan(data[i]):
                x_est[i] = x_est[i - 1]
                p_est[i] = p_est[i - 1] + process_variance
            else:
                # Predict
                x_pred = x_est[i - 1]
                p_pred = p_est[i - 1] + process_variance

                # Update
                k = p_pred / (p_pred + measurement_variance)
                x_est[i] = x_pred + k * (data[i] - x_pred)
                p_est[i] = (1 - k) * p_pred

        result[f"{col}_filtered"] = x_est
        result[f"{col}_variance"] = p_est

    return result


def fit_gaussian_copula(df: pd.DataFrame, features: list, n_simulations: int = 10000) -> dict:
    """
    Fit Gaussian copula to capture dependency structure between features.

    Useful for:
    - Understanding tail dependencies between risk factors
    - Generating correlated Monte Carlo samples
    - Stress testing with realistic correlation structures

    Parameters
    ----------
    df : pd.DataFrame
        Data with features to model
    features : list
        Column names to include in copula
    n_simulations : int, default 10000
        Number of samples to generate

    Returns
    -------
    dict
        Dictionary with:
        - correlation_matrix: Estimated correlation structure
        - features: List of features used
        - simulated_samples: Correlated uniform samples
        - tail_dependence: Lower/upper tail dependence coefficients
        - marginal_params: Parameters of marginal distributions

    Examples
    --------
    >>> copula = fit_gaussian_copula(df, ['roe', 'debt_to_equity', 'p_e_ratio'])
    >>> print(copula['correlation_matrix'])
    """
    # Filter to available features
    available_features = [f for f in features if f in df.columns]

    if len(available_features) < 2:
        return {
            "correlation_matrix": np.array([[1.0]]),
            "features": available_features,
            "simulated_samples": np.array([]),
            "tail_dependence": {"lower": np.array([]), "upper": np.array([])},
            "marginal_params": {},
        }

    # Extract data and handle missing values
    data = df[available_features].dropna()

    if len(data) < 50:
        return {
            "correlation_matrix": np.eye(len(available_features)),
            "features": available_features,
            "simulated_samples": np.array([]),
            "tail_dependence": {"lower": np.array([]), "upper": np.array([])},
            "marginal_params": {},
        }

    n_features = len(available_features)

    # Transform to uniform marginals using empirical CDF
    uniform_data = np.zeros((len(data), n_features))
    marginal_params = {}

    for i, feat in enumerate(available_features):
        col_data = data[feat].values
        ranks = stats.rankdata(col_data)
        uniform_data[:, i] = ranks / (len(col_data) + 1)

        # Store marginal statistics
        marginal_params[feat] = {
            "mean": float(np.mean(col_data)),
            "std": float(np.std(col_data)),
            "median": float(np.median(col_data)),
            "skew": float(stats.skew(col_data)),
            "kurtosis": float(stats.kurtosis(col_data)),
        }

    # Transform to normal and estimate correlation
    normal_data = stats.norm.ppf(uniform_data)
    normal_data = np.nan_to_num(normal_data, nan=0, posinf=0, neginf=0)

    # Estimate correlation matrix
    correlation_matrix = np.corrcoef(normal_data.T)

    # Ensure positive definiteness
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    correlation_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Generate correlated samples
    try:
        cholesky = np.linalg.cholesky(correlation_matrix)
        z = np.random.standard_normal((n_simulations, n_features))
        correlated_normal = z @ cholesky.T
        simulated_uniform = stats.norm.cdf(correlated_normal)
    except np.linalg.LinAlgError:
        # Fallback to independent samples if Cholesky fails
        simulated_uniform = np.random.uniform(0, 1, (n_simulations, n_features))

    # Calculate tail dependence
    tail_dep = _calculate_tail_dependence(uniform_data)

    return {
        "correlation_matrix": correlation_matrix,
        "features": available_features,
        "simulated_samples": simulated_uniform,
        "tail_dependence": tail_dep,
        "marginal_params": marginal_params,
        "n_observations": len(data),
    }


def _calculate_tail_dependence(uniform_data: np.ndarray, threshold: float = 0.05) -> dict:
    """
    Calculate lower and upper tail dependence coefficients.

    Parameters
    ----------
    uniform_data : np.ndarray
        Data transformed to uniform marginals
    threshold : float, default 0.05
        Threshold for tail definition

    Returns
    -------
    dict
        Dictionary with 'lower' and 'upper' tail dependence matrices
    """
    n_vars = uniform_data.shape[1]
    lower_dep = np.zeros((n_vars, n_vars))
    upper_dep = np.zeros((n_vars, n_vars))

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Lower tail: P(V < q | U < q)
            mask_lower = uniform_data[:, i] < threshold
            if mask_lower.sum() > 0:
                lower_dep[i, j] = (uniform_data[mask_lower, j] < threshold).mean()
                lower_dep[j, i] = lower_dep[i, j]

            # Upper tail: P(V > 1-q | U > 1-q)
            mask_upper = uniform_data[:, i] > (1 - threshold)
            if mask_upper.sum() > 0:
                upper_dep[i, j] = (uniform_data[mask_upper, j] > (1 - threshold)).mean()
                upper_dep[j, i] = upper_dep[i, j]

    # Set diagonal to 1
    np.fill_diagonal(lower_dep, 1.0)
    np.fill_diagonal(upper_dep, 1.0)

    return {"lower": lower_dep, "upper": upper_dep}


def parallel_mcmc_chains(
    data: np.ndarray, n_chains: int = 4, n_samples: int = 10000, n_jobs: int = -1
) -> dict:
    """
    Run multiple MCMC chains in parallel for better convergence diagnostics.

    Uses joblib (already in requirements.txt) for parallel execution.

    Parameters
    ----------
    data : np.ndarray
        Input data for sampling
    n_chains : int, default 4
        Number of parallel chains
    n_samples : int, default 10000
        Samples per chain
    n_jobs : int, default -1
        Number of parallel jobs (-1 = all cores)

    Returns
    -------
    dict
        Dictionary with:
        - chains: List of sample arrays
        - r_hat: Gelman-Rubin convergence diagnostic
        - combined_samples: Merged samples from all chains
        - converged: Boolean indicating if R-hat < 1.1
        - chain_means: Mean of each chain
        - chain_stds: Std of each chain

    Examples
    --------
    >>> result = parallel_mcmc_chains(data, n_chains=4)
    >>> if result['converged']:
    ...     print(f"Posterior mean: {result['combined_samples'].mean():.2f}")
    """
    try:
        from joblib import Parallel, delayed

        use_parallel = True
    except ImportError:
        use_parallel = False

    def run_single_chain(seed: int) -> np.ndarray:
        """Run a single MCMC chain with given seed."""
        np.random.seed(seed)
        samples, _ = metropolis_hastings_sampler(data, n_samples=n_samples, burn_in=n_samples // 5)
        return samples

    # Run chains
    if use_parallel and n_jobs != 1:
        chains = Parallel(n_jobs=n_jobs)(
            delayed(run_single_chain)(seed) for seed in range(n_chains)
        )
    else:
        # Sequential fallback
        chains = [run_single_chain(seed) for seed in range(n_chains)]

    # Calculate Gelman-Rubin diagnostic
    r_hat = _calculate_gelman_rubin(chains)

    # Combine samples
    combined_samples = np.concatenate(chains)

    # Chain statistics
    chain_means = [np.mean(c) for c in chains]
    chain_stds = [np.std(c) for c in chains]

    return {
        "chains": chains,
        "r_hat": r_hat,
        "combined_samples": combined_samples,
        "converged": r_hat < 1.1,
        "chain_means": chain_means,
        "chain_stds": chain_stds,
        "posterior_mean": np.mean(combined_samples),
        "posterior_std": np.std(combined_samples),
        "ci_95": (np.percentile(combined_samples, 2.5), np.percentile(combined_samples, 97.5)),
    }


def _calculate_gelman_rubin(chains: list) -> float:
    """
    Calculate R-hat (Gelman-Rubin) convergence diagnostic.

    R-hat < 1.1 indicates convergence.

    Parameters
    ----------
    chains : list
        List of MCMC sample arrays

    Returns
    -------
    float
        R-hat statistic
    """
    m = len(chains)  # number of chains
    n = len(chains[0])  # samples per chain

    if m < 2 or n < 10:
        return float("inf")

    chain_means = np.array([np.mean(c) for c in chains])
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n / (m - 1) * np.sum((chain_means - overall_mean) ** 2)

    # Within-chain variance
    W = np.mean([np.var(c, ddof=1) for c in chains])

    if W < 1e-10:
        return 1.0

    # Pooled variance estimate
    var_hat = (1 - 1 / n) * W + B / n

    return float(np.sqrt(var_hat / W))
