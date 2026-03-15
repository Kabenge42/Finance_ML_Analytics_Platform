"""
Statistical functions for the probabilistic_ml_model package.

Submodules
----------
- ``statistical_analysis`` — Bayesian, MCMC, Kalman, Copula, distribution fitting
- ``ensemble`` — Tri-/quad-model alignment, expected returns summary builders
- ``screening`` — Stock screening strategies
- ``probability_analytics`` — Probability models (earnings, credit, dividend, anomaly)
"""

from finance_ml.probabilistic_ml_model.statistical_functions.statistical_analysis import (
    bayesian_category_analysis,
    metropolis_hastings_sampler,
    mcmc_student_t,
    hierarchical_mcmc_by_sector,
    hierarchical_mcmc_multi_level,
    parallel_mcmc_chains,
    kalman_momentum_filter,
    kalman_filter_price_target,
    fit_gaussian_copula,
    monte_carlo_price_target_simulation,
    calculate_ruin_probability,
    calculate_conditional_probabilities,
    BayesianTechnicalResampler,
)

from finance_ml.probabilistic_ml_model.statistical_functions.ensemble import (
    build_tri_model_alignment,
    build_quad_model_alignment,
    build_expected_returns_summary,
    extract_strong_consensus,
)