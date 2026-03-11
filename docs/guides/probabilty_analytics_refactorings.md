Below are concrete refactoring tasks for each of the five probability models in `probability_analytics.py`, leveraging the advanced MCMC functions already available in `statistical_analysis.py`, and corresponding pipeline changes in `expected_returns_v3.py`.

---

### Core Problem

All five models currently use **heuristic if/else rule-based adjustments** to compute probabilities (e.g., `if z_score < 1.81: base_prob = 0.75; adjustments += 0.15`). This approach:
- Produces point estimates with no proper uncertainty quantification
- Has no principled posterior distribution
- Generates ad-hoc confidence intervals (`prob ± ci_width`)

The `statistical_analysis.py` module already contains production-ready Metropolis MCMC samplers that implement the full Metropolis algorithm (proposal distribution → acceptance ratio → decision rules), as demonstrated in `MCMC_dependent_sampling.ipynb`.

---

### Available MCMC Functions (from `statistical_analysis.py`)

| Function | Signature | Use Case |
|---|---|---|
| `metropolis_hastings_sampler()` | `(data, n_samples, burn_in, proposal_std, prior_mean, prior_std)` → `(samples, acceptance_rate)` | General posterior sampling for mean parameter with Normal likelihood + Normal prior |
| `mcmc_student_t()` | `(data, n_samples, burn_in)` → `(samples_mu, samples_df)` | Heavy-tailed financial data with outliers — samples location + degrees of freedom |
| `hierarchical_mcmc_by_sector()` | `(df, feature, sector_col, n_samples)` → `dict[sector → posterior_stats]` | Sector-level shrinkage estimation pooling toward global mean |
| `parallel_mcmc_chains()` | `(data, n_chains, n_samples)` → `dict` with chains + Gelman-Rubin diagnostics | Multi-chain MCMC with convergence diagnostics |
| `bayesian_category_analysis()` | `(df, category_name, features, prior_mean, prior_std)` → `dict[feature → posterior_stats]` | Conjugate Normal-Normal posterior per feature in a category |

---

### Task 1: `AccountingAnomalyProbabilityModel` (lines 225–489)

**Current approach**: Calls `detect_accounting_anomalies()` then computes `anomaly_severity_score` via weighted linear combination. `calculate_conditional_probabilities()` uses simple median-split separation scores.

**Refactoring tasks**:

1. **Add `mcmc_student_t()` for anomaly score posterior estimation** — Anomaly scores (Mahalanobis distance, robust z-scores) are heavy-tailed. Replace the deterministic severity score with a Student-t posterior:
   ```python
   from finance_ml.analytics.statistical_analysis import mcmc_student_t

   # In analyze_dataframe(), after Phase 1:
   anomaly_scores = result["accounting_anomaly_score"].dropna().values
   mu_samples, df_samples = mcmc_student_t(anomaly_scores, n_samples=5000, burn_in=1000)
   result["anomaly_posterior_mean"] = mu_samples.mean()
   result["anomaly_posterior_std"] = mu_samples.std()
   result["anomaly_ci_lower"] = np.percentile(mu_samples, 2.5)
   result["anomaly_ci_upper"] = np.percentile(mu_samples, 97.5)
   ```

2. **Add `hierarchical_mcmc_by_sector()` for sector-relative anomaly scoring** — Replace the simple `groupby().rank()` on line 317–321 with hierarchical MCMC that shrinks sector anomaly estimates toward the global mean:
   ```python
   from finance_ml.analytics.statistical_analysis import hierarchical_mcmc_by_sector

   sector_posteriors = hierarchical_mcmc_by_sector(
       result, feature="accounting_anomaly_score", sector_col=sector_col, n_samples=5000
   )
   # Map each stock's sector posterior mean/CI back to the result DataFrame
   ```

3. **Add `metropolis_hastings_sampler()` for per-feature conditional probability** — In `calculate_conditional_probabilities()` (line 389), replace the median-split heuristic with MCMC-sampled posterior P(Anomaly | Feature):
   ```python
   from finance_ml.analytics.statistical_analysis import metropolis_hastings_sampler

   # For each anomaly feature, sample the posterior mean of the anomalous subgroup
   high_group = data[data[feat] > median_val].values
   samples, acc_rate = metropolis_hastings_sampler(
       high_group, n_samples=5000, burn_in=1000, prior_mean=0, prior_std=10
   )
   ```

4. **Add constructor parameters**: `n_mcmc_samples: int = 5000`, `burn_in: int = 1000`, `use_mcmc: bool = True` to allow toggling.

---

### Task 2: `CreditRiskProbabilityModel` (lines 2278–2419)

**Current approach**: Row-by-row iteration with hardcoded if/else thresholds producing a single `distress_probability` point estimate. CI is `prob ± (0.15 - data_points * 0.02)`.

**Refactoring tasks**:

1. **Replace heuristic probability with `metropolis_hastings_sampler()` on distress indicators** — Collect the vector of distress-relevant features per stock, define a log-posterior over distress probability, and sample:
   ```python
   from finance_ml.analytics.statistical_analysis import metropolis_hastings_sampler

   # Vectorize: collect all z_scores as observed data
   z_scores = df["altman_z_score"].dropna().values
   samples, acc_rate = metropolis_hastings_sampler(
       z_scores, n_samples=10000, burn_in=2000,
       prior_mean=self.distress_threshold, prior_std=1.0
   )
   # Per-stock: P(distress) = P(Z < threshold | posterior)
   distress_prob_per_stock = np.mean(samples[:, None] > df["altman_z_score"].values[None, :], axis=0)
   ```

2. **Use `mcmc_student_t()` for robust distress estimation** — Financial distress indicators have fat tails. Replace the Normal assumption:
   ```python
   mu_samples, df_samples = mcmc_student_t(z_scores, n_samples=8000)
   # Derive P(Z < 1.81) from the posterior predictive
   ```

3. **Add `hierarchical_mcmc_by_sector()` for sector-calibrated risk** — Different sectors have different baseline distress rates. Use hierarchical shrinkage:
   ```python
   sector_results = hierarchical_mcmc_by_sector(
       df, feature="altman_z_score", sector_col="industry", n_samples=8000
   )
   # Use sector posterior mean as the sector-specific prior for each stock
   ```

4. **Replace ad-hoc CI** (line 2389 `ci_width = 0.15 - (data_points * 0.02)`) with proper MCMC credible intervals from the posterior samples.

5. **Vectorize the row loop** — The current `for _, row in df.iterrows()` pattern (line 2302) should be replaced with vectorized operations feeding into the MCMC sampler.

---

### Task 3: `DividendCutProbabilityModel` (lines 2422–2529)

**Current approach**: Row-by-row if/else adjustments on `fcf_coverage`, `payout_ratio`, `streak`, etc. producing a single `dividend_cut_probability`.

**Refactoring tasks**:

1. **Use `metropolis_hastings_sampler()` on FCF coverage distribution** — FCF coverage is the strongest predictor (per code comments). Sample its posterior to derive P(cut):
   ```python
   fcf_data = df["fcf_dividend_coverage"].dropna().values
   samples, acc_rate = metropolis_hastings_sampler(
       fcf_data, n_samples=8000, burn_in=2000,
       prior_mean=self.min_coverage, prior_std=1.0
   )
   # P(cut) ≈ P(FCF_coverage < threshold | posterior)
   cut_prob = np.mean(samples < self.min_coverage)
   ```

2. **Use `mcmc_student_t()` for payout ratio modeling** — Payout ratios can have extreme outliers (>100%). The Student-t sampler handles this:
   ```python
   payout_data = df["dividend_payout_ratio"].dropna().values
   mu_samples, df_samples = mcmc_student_t(payout_data)
   # Posterior predictive probability of payout > high_payout_threshold
   ```

3. **Add `hierarchical_mcmc_by_sector()`** — Dividend safety varies dramatically by sector (utilities vs. tech). Use sector-level hierarchical priors:
   ```python
   sector_posteriors = hierarchical_mcmc_by_sector(
       df, feature="fcf_dividend_coverage", sector_col="industry"
   )
   ```

4. **Combine multiple MCMC posteriors** — Create a composite posterior by multiplying the individual posterior probabilities from FCF coverage, payout ratio, and streak signals, rather than additive heuristic adjustments.

5. **Add `n_mcmc_samples`, `burn_in`, `use_mcmc` constructor parameters** for backward compatibility.

---

### Task 4: `PriceTargetAchievementModel` (lines 2532–2640)

**Current approach**: Heuristic base probability from upside buckets + additive adjustments from momentum, spread, conviction, etc.

**Refactoring tasks**:

1. **Use `monte_carlo_price_target_simulation()` + `metropolis_hastings_sampler()` jointly** — The pipeline already calls `monte_carlo_price_target_simulation()` separately in `run_monte_carlo_analysis()` (line 1241). Wire its output into the model:
   ```python
   from finance_ml.analytics.statistical_analysis import (
       metropolis_hastings_sampler, monte_carlo_price_target_simulation
   )

   # Sample posterior of expected return
   returns_data = df["upside_potential"].dropna().values
   samples, acc_rate = metropolis_hastings_sampler(
       returns_data, n_samples=10000, prior_mean=0, prior_std=20
   )
   # P(achievement) = P(return > 0 | posterior)
   achievement_prob = np.mean(samples > 0)
   ```

2. **Use `mcmc_student_t()` for heavy-tailed return modeling** — Price target returns are fat-tailed:
   ```python
   mu_samples, df_samples = mcmc_student_t(returns_data, n_samples=10000)
   # Per-stock achievement probability from posterior predictive
   ```

3. **Use `parallel_mcmc_chains()` for convergence diagnostics** — Price target achievement is a high-stakes estimate. Run multi-chain MCMC with Gelman-Rubin:
   ```python
   from finance_ml.analytics.statistical_analysis import parallel_mcmc_chains

   mcmc_result = parallel_mcmc_chains(returns_data, n_chains=4, n_samples=10000)
   # Check mcmc_result["gelman_rubin"] < 1.1 for convergence
   ```

4. **Replace `expected_return_prob_weighted`** (line 2631: `upside * prob`) with the posterior mean of the return distribution weighted by the MCMC-derived achievement probability.

---

### Task 5: `CategoryProbabilityAnalyzer` (lines 3102–3156)

**Current approach**: Simple percentile rank as "probability proxy" + z-score. No Bayesian estimation despite the class name.

**Refactoring tasks**:

1. **Replace with `bayesian_category_analysis()`** — This function already exists and does exactly what this class should do:
   ```python
   from finance_ml.analytics.statistical_analysis import bayesian_category_analysis

   def analyze_view(self, df, feature_cols):
       results = bayesian_category_analysis(
           df, category_name=self.category_name, features=feature_cols,
           prior_mean=0, prior_std=10
       )
       # results[feature] contains: posterior_mean, posterior_std, ci_95, mcmc_samples
   ```

2. **Add `metropolis_hastings_sampler()` per feature** — For each feature column, run MCMC to get a proper posterior:
   ```python
   for feat in feature_cols:
       data = df[feat].dropna().values
       samples, acc_rate = metropolis_hastings_sampler(
           data, n_samples=self.n_mcmc_samples, burn_in=self.burn_in,
           prior_mean=self.prior_alpha, prior_std=self.prior_beta  # repurpose params
       )
       feat_results["posterior_mean"] = samples.mean()
       feat_results["posterior_std"] = samples.std()
       feat_results["ci_lower_95"] = np.percentile(samples, 2.5)
       feat_results["ci_upper_95"] = np.percentile(samples, 97.5)
   ```

3. **Add `run_category_probability_analytics()` integration** — This function in `statistical_analysis.py` (line 2499) already combines `bayesian_category_analysis()` + `fit_distributions_by_category()`. Delegate to it:
   ```python
   from finance_ml.analytics.statistical_analysis import run_category_probability_analytics

   analytics = run_category_probability_analytics(
       df, category_name=self.category_name, features=feature_cols
   )
   ```

4. **Add constructor parameters**: `n_mcmc_samples: int = 5000`, `burn_in: int = 1000`, `use_student_t: bool = False` for heavy-tailed categories.

---

### Pipeline Changes in `expected_returns_v3.py`

#### Task 6: Wire MCMC into pipeline orchestration functions

1. **`run_credit_risk_analysis()` (line 1691)** — After `credit_model.analyze_dataframe()`, add MCMC enrichment:
   ```python
   # Add hierarchical sector-level MCMC
   try:
       sector_mcmc = hierarchical_mcmc_by_sector(credit_df, "altman_z_score")
       # Merge sector posterior stats into credit DataFrame
   except Exception as e:
       logger.warning("Hierarchical MCMC for credit risk failed: %s", e)
   ```

2. **`run_accounting_anomaly_analysis()` (line 1826)** — Add MCMC posterior estimation after the model runs:
   ```python
   # Add Student-t MCMC for anomaly score posterior
   try:
       anomaly_scores = anomaly_results["accounting_anomaly_score"].dropna().values
       if len(anomaly_scores) > 50:
           mu_samples, df_samples = mcmc_student_t(anomaly_scores)
           anomaly_results["anomaly_posterior_location"] = mu_samples.mean()
   except Exception as e:
       logger.warning("MCMC anomaly posterior failed: %s", e)
   ```

3. **`run_category_probability_analysis()` (line 1953)** — Replace or augment `CategoryProbabilityAnalyzer.analyze_view()` with `run_category_probability_analytics()`:
   ```python
   from finance_ml.analytics.statistical_analysis import run_category_probability_analytics

   for cat_name, features in categories.items():
       analytics = run_category_probability_analytics(df, cat_name, features)
       # Merge MCMC-derived posteriors into results
   ```

4. **`run_parallel_mcmc_return_analysis()` (line 3116)** — This function already exists and calls `parallel_mcmc_chains()`. Ensure its output is integrated into `build_expected_returns_summary()`.

---

### Summary of MCMC Function → Model Mapping

| Model | Primary MCMC Function | Secondary MCMC Function | Purpose |
|---|---|---|---|
| `AccountingAnomalyProbabilityModel` | `mcmc_student_t()` | `hierarchical_mcmc_by_sector()` | Heavy-tailed anomaly posteriors + sector shrinkage |
| `CreditRiskProbabilityModel` | `metropolis_hastings_sampler()` | `mcmc_student_t()`, `hierarchical_mcmc_by_sector()` | Distress probability posterior + sector calibration |
| `DividendCutProbabilityModel` | `metropolis_hastings_sampler()` | `hierarchical_mcmc_by_sector()` | FCF coverage posterior + sector-level dividend safety |
| `PriceTargetAchievementModel` | `mcmc_student_t()` | `parallel_mcmc_chains()` | Fat-tailed return posterior + convergence diagnostics |
| `CategoryProbabilityAnalyzer` | `bayesian_category_analysis()` | `metropolis_hastings_sampler()` | Replace percentile proxy with proper Bayesian posterior |

### Backward Compatibility

All models should add a `use_mcmc: bool = False` constructor parameter defaulting to `False`, so existing callers are unaffected. When `True`, the MCMC path is activated. This allows incremental rollout and A/B comparison of heuristic vs. MCMC estimates.
