### Refactoring Tasks for `expected_returns_v3.py` Pipeline

Based on analysis of the codebase (4,982 lines), today's pipeline logs (4 runs on 2026-03-13), and the reference documentation (`finance_ml_analytics_guide.md`, `probabilty_analytics_refactorings.md`, `PML_README.md`).

---

### Area 1: Monolithic `main()` Function (Lines 3592–4982)

The `main()` function is ~1,400 lines — a single function orchestrating 10+ steps with all logic inline.

**Task 1.1: Extract each pipeline step into its own function**
- Create dedicated functions like `_step_load_data(cfg)`, `_step_monte_carlo(df, cfg)`, `_step_screening(df_all, cfg)`, etc.
- Each returns a typed result dataclass/dict, making the `main()` function a ~50-line orchestrator.

**Task 1.2: Introduce a `PipelineResult` dataclass**
- Replace the 16+ loose variables initialized at lines 3639–3655 (`mc`, `pt`, `kal`, `beat`, `credit`, `div_safety`, `tri`, `quad`, `strong`, `summary`, `screens`, `category_analytics`, `corr_info`, `df_features`, `resampled_posterior`, `mcmc_result`, `anomaly_results`) with a single structured container.

---

### Area 2: Step 7b Performance Bottleneck (707.5s / ~77% of Total Runtime)

From today's log, Step 7b (`run_category_probability_analysis`) took **707.5 seconds** — processing 17 categories × 689 features with sequential MCMC sampling. Total pipeline time was ~920s.

**Task 2.1: Parallelize category-level MCMC with `joblib`**
- The log shows `Joblib: False`. Enable joblib and parallelize the 17 independent category analyses using `Parallel(n_jobs=cfg.n_jobs)(delayed(analyze_category)(cat) for cat in categories)`.

**Task 2.2: Add feature-level sampling budget control**
- 689 features × 5,000 MCMC samples each is excessive. Add a `max_features_per_category` parameter and/or reduce `n_mcmc_samples` for low-importance features.

**Task 2.3: Enable Numba JIT for MCMC inner loops**
- The log shows `Numba: False`. The `optimized_ops.py` module exists but isn't active. Enable Numba acceleration for the Metropolis-Hastings sampler hot loops.

**Task 2.4: Add result caching between runs**
- The pipeline ran 4 times today (02:11, 18:31, 18:56, 19:50) with identical data. Implement `joblib.Memory` or file-based caching for expensive MCMC results keyed by data hash + parameters.

---

### Area 3: Screening Threshold Failures (Recurring Warnings)

Every run today produced:
- `WARNING - Quality screen returned 0 stocks`
- `WARNING - Dividend screen returned only 15 stocks (0.3% of universe)`

**Task 3.1: Add adaptive screening thresholds**
- In `run_stock_screening()` (line 2314), implement percentile-based fallback thresholds when absolute thresholds yield <1% of the universe. Log the relaxed thresholds used.

**Task 3.2: Add screening threshold configuration to `PipelineConfig`**
- Move hardcoded screening parameters into `PipelineConfig` so they can be tuned via environment variables without code changes.

---

### Area 4: ArviZ / InferenceData Integration (Step 8 Always Skipped)

Every run logs: `⏭️ Step 8: ArviZ not available — skipping InferenceData`

**Task 4.1: Add ArviZ to dependencies or implement graceful degradation**
- Per `PML_README.md`, ArviZ/PyMC/xarray are in the tech stack. Either install the dependency or implement a lightweight alternative using the existing `inference_schema.py` (1,588 lines) that doesn't require ArviZ.

**Task 4.2: Gate ArviZ-dependent visualizations**
- Step 9 generates visualizations that could consume InferenceData. Ensure viz functions degrade gracefully when InferenceData is `None`, rather than silently producing incomplete outputs.

---

### Area 5: MCMC Integration into Probability Models (from `probabilty_analytics_refactorings.md`)

All five probability models use heuristic if/else rules instead of the production-ready MCMC samplers already in `statistical_analysis.py`.

**Task 5.1: `AccountingAnomalyProbabilityModel` — Add `mcmc_student_t()` for anomaly score posteriors**
- Replace deterministic severity scores with Student-t posterior estimation (heavy-tailed anomaly scores).
- Add `hierarchical_mcmc_by_sector()` to replace simple `groupby().rank()` with sector-shrinkage estimation.

**Task 5.2: `CreditRiskProbabilityModel` — Replace row-by-row heuristics with vectorized MCMC**
- Replace `for _, row in df.iterrows()` with vectorized `metropolis_hastings_sampler()` on distress indicators.
- Replace ad-hoc CI formula (`ci_width = 0.15 - (data_points * 0.02)`) with proper MCMC credible intervals.

**Task 5.3: `DividendCutProbabilityModel` — MCMC on FCF coverage**
- Use `metropolis_hastings_sampler()` on FCF coverage distribution to derive `P(cut)`.
- Add `hierarchical_mcmc_by_sector()` for sector-level dividend safety priors (utilities vs. tech).

**Task 5.4: `PriceTargetAchievementModel` — Fat-tailed return posteriors**
- Use `mcmc_student_t()` for heavy-tailed return modeling.
- Add `parallel_mcmc_chains()` with Gelman-Rubin convergence diagnostics for this high-stakes estimate.

**Task 5.5: `CategoryProbabilityAnalyzer` — Replace percentile proxy with Bayesian posterior**
- Delegate to existing `bayesian_category_analysis()` and `run_category_probability_analytics()` from `statistical_analysis.py` instead of simple percentile ranks.

**Task 5.6: Add `use_mcmc: bool = False` toggle to all models**
- All five models should accept `use_mcmc` constructor parameter (default `False`) for backward compatibility and incremental rollout.

---

### Area 6: Pipeline Orchestration in `expected_returns_v3.py` (from `probabilty_analytics_refactorings.md` Task 6)

**Task 6.1: Wire hierarchical MCMC into `run_credit_risk_analysis()` (line 1859)**
- After `credit_model.analyze_dataframe()`, add `hierarchical_mcmc_by_sector(credit_df, "altman_z_score")` for sector-calibrated risk.

**Task 6.2: Wire Student-t MCMC into `run_accounting_anomaly_analysis()` (line 2065)**
- After model execution, add `mcmc_student_t(anomaly_scores)` for posterior anomaly location estimation.

**Task 6.3: Wire `run_category_probability_analytics()` into `run_category_probability_analysis()` (line 2210)**
- Replace or augment `CategoryProbabilityAnalyzer.analyze_view()` with the existing `run_category_probability_analytics()` function.

**Task 6.4: Integrate `run_parallel_mcmc_return_analysis()` output into `build_expected_returns_summary()`**
- Step 7a (line 3437) already runs parallel MCMC but its output isn't merged into the summary. Wire the Gelman-Rubin diagnostics and posterior means into the final summary DataFrame.

---

### Area 7: Export Step Performance (Step 10: 100.6s)

**Task 7.1: Parallelize database exports**
- Step 10 exports ~20 tables sequentially (100.6s). Use concurrent DB connections or batch inserts.

**Task 7.2: Add selective export based on changed data**
- Skip re-exporting tables whose source DataFrames haven't changed since last run.

---

### Summary: Priority-Ordered Refactoring Roadmap

| Priority | Task | Impact | Effort |
|:---------|:-----|:-------|:-------|
| **P0** | 2.1 — Parallelize Step 7b with joblib | -600s runtime | Medium |
| **P0** | 1.1 — Extract `main()` into step functions | Maintainability | Medium |
| **P1** | 5.1–5.5 — MCMC integration into 5 probability models | Model quality | High |
| **P1** | 3.1 — Adaptive screening thresholds | Fix 0-stock screens | Low |
| **P1** | 6.1–6.4 — Wire MCMC into pipeline orchestration | End-to-end integration | Medium |
| **P2** | 2.4 — Result caching between runs | Avoid redundant compute | Medium |
| **P2** | 4.1 — ArviZ integration or alternative | Enable Step 8 | Medium |
| **P2** | 2.2–2.3 — Sampling budget + Numba | Further runtime reduction | Low |
| **P3** | 7.1–7.2 — Export optimization | -50s runtime | Low |
| **P3** | 1.2 — `PipelineResult` dataclass | Code clarity | Low |