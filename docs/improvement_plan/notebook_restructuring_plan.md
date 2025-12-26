### Objective

Provide a concrete, cell-by-cell restructuring plan to extend the notebook’s reporting and visualization from Phase 9.2
into Phases 9.4–9.8, and to elevate Section 10 (Portfolio Optimization Workflow) with diagnostics, dashboards, and
governance artifacts. The plan aligns with `docs/code_guidelines.md` v1.2+, reuses existing artifacts from Phases
9.2–9.3, and observes the Standardized Predictions Schema.

### Global Principles and Assumptions

- Inputs are produced by the existing pipeline into `outputs/` according to project conventions:
    - Regression predictions: `outputs/regression/regression_predictions_detailed.csv` (schema:
      `ticker, isin, sector, region, last_price, y_true, y_pred, y_pred_calibrated, pred_p10, pred_p50, pred_p90, interval_width, abs_error, pct_error, model_version, snapshot_date`)
    - Classification probabilities (if available): `outputs/classification/classification_predictions.csv`
- Environment variables: `OUTPUT_DIR`, `MODEL_VERSION`, `ENABLE_INTERACTIVE_PLOTS`, `REPORT_FORMAT` (HTML first),
  `N_JOBS`.
- Python 3.14 gating: SHAP 0.50.0 (with enhanced explainability features and improved performance) and `catboost` may be
  unavailable on Python 3.14. All SHAP-related visuals are optional and guarded by
  presence checks. Provide fallbacks (permutation importance, partial dependence).
- File naming convention: new artifacts live under dedicated subfolders to keep outputs organized and testable.

---

### Phase 9.4 — Uncertainty Quantification & Conformal Calibration

Goal: Make interval quality observable and auditable across sectors/regions with both static and interactive artifacts.

#### Notebook Integration (new Section 9.4, 5 cells)

1) [Markdown] 9.4 Overview — Uncertainty & Conformal Calibration
    - What: Objectives, inputs, outputs, success criteria.
    - Inputs: `regression_predictions_detailed.csv`.
2) [Code] Load predictions + derive diagnostics
    - Compute `coverage_flag_p90` (y_true within [p10, p90]), `interval_width`, `calibration_error` per group.
    - Save `outputs/uncertainty/quantile_predictions_diagnostics.csv`.
3) [Code] Coverage and width visuals
    - `coverage_by_sector.json` (grouped coverage stats)
    - `interval_width_by_bucket.html` (width vs. last_price buckets)
    - `coverage_heatmap_region_sector.html` (pivot heatmap)
4) [Code] Reliability diagrams for conformal calibration
    - `reliability_diagram_conformal.html` (pre vs post calibration)
    - Side-by-side residual plots per sector: `residuals_calibrated_vs_raw_by_sector.html`
5) [Code] Summary + QA
    - Validations: 80% target coverage within tolerance bands; non-empty artifacts; interval monotonicity (p10<=p50<
      =p90)
    - Export `uncertainty_summary.json`.

#### Outputs

- CSV: `outputs/uncertainty/quantile_predictions_diagnostics.csv`
- JSON: `outputs/uncertainty/coverage_by_sector.json`, `outputs/uncertainty/uncertainty_summary.json`
- HTML: `interval_width_by_bucket.html`, `coverage_heatmap_region_sector.html`, `reliability_diagram_conformal.html`,
  `residuals_calibrated_vs_raw_by_sector.html`

#### QA Checks

- Coverage within [0.75, 0.90] for 80% band across major sectors (configurable).
- Report sectors under-covered or over-covered with counts and deltas.
- Validate interval ordering and absence of NaN in critical columns.

---

### Phase 9.5 — Outlier Safety Rails & Non-Negative Constraints

Goal: Visualize and quantify the effect of winsorization, clipping, and non-negativity to enforce policy from
`code_guidelines.md`.

#### Notebook Integration (new Section 9.5, 5 cells)

1) [Markdown] 9.5 Overview — Safety Rails
    - Describe winsorization thresholds, clipping, non-negative predictions policy.
2) [Code] Pre vs post distributions
    - Input: Pre-/post-processed features and predictions (from cache or recompute lightweight view).
    - Visuals: `pre_post_winsorization_distributions.html` (facet grid by feature category),
      `clipping_effect_summary.json`.
3) [Code] Constraint violation tracker
    - Compute any violations (e.g., negative predictions where not allowed).
    - Artifacts: `non_negative_violations.json`, `violation_heatmap_by_feature_sector.html`.
4) [Code] Interactive robustness sliders (Plotly)
    - Threshold-sensitivity dashboard: `safety_rails_sensitivity_dashboard.html`.
5) [Code] Summary + QA
    - Ensure violation counts are zero (post-policy). Export `safety_rails_summary.json`.

#### Outputs

- HTML: `pre_post_winsorization_distributions.html`, `violation_heatmap_by_feature_sector.html`,
  `safety_rails_sensitivity_dashboard.html`
- JSON: `clipping_effect_summary.json`, `non_negative_violations.json`, `safety_rails_summary.json`

#### QA Checks

- Winsorization reduces extreme kurtosis without distorting medians beyond tolerance.
- Zero non-negative violations after final step.

---

### Phase 9.6 — Data Split and Leakage Policy Validation

Goal: Prove no leakage across folds; maintain class/sector balance and grouping rules.

#### Notebook Integration (new Section 9.6, 5 cells)

1) [Markdown] 9.6 Overview — Data Split & Leakage
2) [Code] Fold construction snapshot reader
    - Input: fold assignments used in training (persisted artifact) or reconstruct with same seed/policy.
    - Output: `fold_assignments.csv` (optional) under `outputs/splits/`.
3) [Code] Overlap and balance dashboards
    - `fold_overlap_heatmap.html` (Ticker/Sector overlaps across folds)
    - `grouped_cv_balance_metrics.json` (per-fold group/sector counts, target distribution)
4) [Code] Time-aware leakage checks
    - If `snapshot_date` available, ensure train dates < validation dates within folds; export `leakage_report.json`.
5) [Code] Summary + QA
    - Report leakage risks with severity and remediation hints.

#### Outputs

- HTML: `fold_overlap_heatmap.html`
- JSON: `grouped_cv_balance_metrics.json`, `leakage_report.json`
- CSV: `fold_assignments.csv` (optional)

#### QA Checks

- Zero direct overlaps for `ticker` across train/validation when grouped.
- Stratification deltas by `sector` and `region` within allowed tolerance.

---

### Phase 9.7 — Sector Bias Calibration & Metrics Persistence

Goal: Measure and persist bias corrections; relate to metric improvements over time.

#### Notebook Integration (new Section 9.7, 5 cells)

1) [Markdown] 9.7 Overview — Sector Bias & Persistence
2) [Code] Bias estimation
    - Compute sector/region bias pre- and post-calibration (using `y_pred_calibrated`).
    - Save `sector_bias_calibration_v{MODEL_VERSION}.json`.
3) [Code] Metrics over time visualization
    - `metrics_by_sector_time.html` (MAE/MAPE deltas before/after) using `snapshot_date`.
4) [Code] Drill-down comparison dashboard
    - `sector_bias_dashboard.html` (interactive: select sector, see bias, coverage, error trends).
5) [Code] Summary + QA
    - Validate persistence contract (file versioned by `MODEL_VERSION`), non-empty content, expected keys.

#### Outputs

- JSON: `outputs/calibration/sector_bias_calibration_v{MODEL_VERSION}.json`
- HTML: `metrics_by_sector_time.html`, `sector_bias_dashboard.html`

#### QA Checks

- Post-calibration errors reduced or unchanged per sector; flag regressions > tolerance.

---

### Phase 9.8 — Stacking Ensemble Diagnostics & Model Governance

Goal: Provide transparency for stacked models and document model lineage.

#### Notebook Integration (new Section 9.8, 6 cells)

1) [Markdown] 9.8 Overview — Stacking & Governance
2) [Code] Base-model contribution analysis
    - Bar charts of base model weights/contributions: `stacking_contributions.html` and CSV
      `stacking_contributions.csv`.
3) [Code] Explainability overlays
    - If `shap` available: `shap_summary.html` (sector-level tabs). Else: permutation importance:
      `permutation_importance.html`.
4) [Code] Meta-learner error maps
    - `meta_error_map.html` (error vs. key features and sectors)
5) [Code] Governance artifacts
    - `model_card_v{MODEL_VERSION}.md` (template-filled) and `lineage.json` mapping datasets → features → models →
      metrics.
6) [Code] Baseline vs stacked comparison
    - Interactive comparison with confidence bands per sector: `stacked_vs_baseline_comparison.html`.

#### Outputs

- HTML: `stacking_contributions.html`, `shap_summary.html` or `permutation_importance.html`, `meta_error_map.html`,
  `stacked_vs_baseline_comparison.html`
- CSV: `stacking_contributions.csv`
- Governance: `model_card_v{MODEL_VERSION}.md`, `lineage.json`

#### QA Checks

- Governance files exist and contain mandatory sections/keys (see template below).
- If SHAP unavailable, permutation importance artifacts exist instead.

#### Model Card Template (auto-filled)

```
# Model Card — {MODEL_VERSION}
- Task: Price target regression + classification-enhanced features
- Data: source(s), time range, snapshot_date policy
- Features: key groups, selection method, safety rails
- Models: base learners, meta-learner, hyperparameters
- Validation: split policy, metrics (MAE, RMSE, MAPE, R2), uncertainty coverage
- Fairness/Bias: sector/region calibration overview
- Risk/Limitations: non-negativity, data drift, missingness
- Versioning: code, data, dependency hashes
```

#### Lineage JSON Skeleton

```
{
  "model_version": "v9_10",
  "datasets": {"train": "...", "validation": "..."},
  "features": {"count": 310, "groups": ["momentum","valuation", ...]},
  "models": {"base": ["xgboost","lightgbm"], "meta": "linear"},
  "artifacts": ["regression_predictions_detailed.csv", "quantile_predictions_diagnostics.csv"],
  "metrics": {"overall": {"MAE": 0.0, "MAPE": 0.0}, "by_sector": {...}}
}
```

---

### Section 10 — Portfolio Optimization Workflow (Enhancement Plan)

Goal: Expand diagnostics, sensitivity analysis, risk decomposition, and dashboards. The six phases reported as complete
are the functional backbone; this plan focuses on reporting and visualization within the notebook.

#### Proposed Subsections (10.1 – 10.8) and Cells

10.1 Overview & Inputs (1 cell)

- [Markdown] Objectives, inputs (selected universe, expected returns from `analytics/ml_returns.py`,
  covariance/correlation matrix, constraints, benchmark).

10.2 Universe & Filters Diagnostics (2 cells)

- [Code] Selection summary table by `sector/region/market_cap_bucket`; export `portfolio_universe_summary.json` and
  `portfolio_universe_summary.html`.
- [Code] Interactive filter explorer (Plotly): `portfolio_filter_explorer.html`.

10.3 Expected Returns & Risk Inputs QA (3 cells)

- [Code] Visualize return estimates distribution and stability; export `expected_returns_diagnostics.json`,
  `expected_returns_distribution.html`.
- [Code] Correlation heatmap (top N): `risk_correlation_heatmap.html`.
- [Code] Volatility and correlation drift vs benchmark: `risk_drift_dashboard.html`.

10.4 Optimization Frontiers & Constraint Explorer (3 cells)

- [Code] Efficient frontier (mean-variance) with hover details; `efficient_frontier.html`.
- [Code] Constraint sensitivity (max weight, sector caps, turnover limits) with sliders; `constraints_sensitivity.html`
  and `constraints_scenarios.csv`.
- [Code] Transaction cost impact: `transaction_cost_impact.html` and `transaction_cost_summary.json`.

10.5 Chosen Portfolio Breakdown & Risk Decomposition (3 cells)

- [Code] Holdings table with sector/region exposures; `portfolio_holdings_detailed.csv`, `portfolio_exposures.html`.
- [Code] Risk decomposition (factor and idiosyncratic if available) + contribution-to-risk waterfall:
  `risk_decomposition.html`.
- [Code] Scenario & stress tests (from `analytics/risk.py`): `stress_tests_dashboard.html`.

10.6 Backtesting & Walk-Forward Results (3 cells)

- [Code] Vectorized backtest charts (cumulative perf, rolling Sharpe, drawdowns): `backtest_performance.html`.
- [Code] Attribution report: `performance_attribution.html` and `attribution_breakdown.csv`.
- [Code] Walk-forward optimization stability (weights turnover, hit rate): `walk_forward_stability.html`.

10.7 Risk Management Dashboard (2 cells)

- [Code] Expected Shortfall (CVaR), tracking error, risk budgets: `risk_management_dashboard.html`.
- [Code] Interactive rebalancing widget snapshot from `dashboards/portfolio_widgets.py`:
  `portfolio_rebalancing_widget.html` (export static snapshot if needed).

10.8 Summary, QA, and Export (2 cells)

- [Code] Portfolio analytics summary JSON covering KPIs, limits, and exceptions: `portfolio_summary.json`.
- [Code] Multi-period comparison (already exists): ensure link-out and cross-reference;
  `portfolio_multi_period_comparison.html`.

#### Portfolio Outputs Directory Map

- `outputs/portfolio/`:
    - `portfolio_universe_summary.json`, `portfolio_universe_summary.html`
    - `expected_returns_diagnostics.json`, `expected_returns_distribution.html`, `risk_correlation_heatmap.html`,
      `risk_drift_dashboard.html`
    - `efficient_frontier.html`, `constraints_sensitivity.html`, `constraints_scenarios.csv`
    - `transaction_cost_impact.html`, `transaction_cost_summary.json`
    - `portfolio_holdings_detailed.csv`, `portfolio_exposures.html`, `risk_decomposition.html`
    - `stress_tests_dashboard.html`, `backtest_performance.html`, `performance_attribution.html`,
      `attribution_breakdown.csv`
    - `walk_forward_stability.html`, `risk_management_dashboard.html`, `portfolio_rebalancing_widget.html`
    - `portfolio_summary.json`, `portfolio_multi_period_comparison.html`

#### Portfolio QA Checks

- Constraint adherence (sum weights=1, per-asset/sector caps, turnover ≤ limit).
- Risk budgets respected; TE and ES within configured bounds.
- Backtest integrity: no forward-looking data; rebalancing cadence respected.

---

### Imports and Cell Markers for Testability

Add to the notebook’s early import cell (mirroring Phase 9.2 approach):

```
from finance_ml.ml_workflow.eval.uncertainty import (
    build_quantile_diagnostics,
    plot_interval_coverage,
    plot_reliability_diagram,
)
from finance_ml.ml_workflow.eval.safety_rails import (
    summarize_winsorization_effects,
    track_constraint_violations,
    safety_rails_sensitivity_app,
)
from finance_ml.ml_workflow.eval.splits import (
    compute_fold_overlap,
    summarize_grouped_cv_balance,
    time_leakage_checks,
)
from finance_ml.ml_workflow.eval.calibration import (
    estimate_sector_bias,
    plot_metrics_by_sector_time,
)
from finance_ml.ml_workflow.eval.stacking import (
    compute_stacking_contributions,
    meta_error_maps,
    generate_model_card,
    build_lineage_json,
)
from finance_ml.ml_workflow.analytics.portfolio_reporting import (
    universe_summary,
    returns_risk_diagnostics,
    frontier_and_constraints,
    risk_decomposition_dashboard,
    backtest_and_attribution,
    risk_management_dashboard,
)
```

Each notebook cell should start with a short marker comment to support structure tests, e.g.:

```
# [PHASE 9.4] Build quantile diagnostics
# [PHASE 9.5] Safety rails: violations and sensitivity
# [PHASE 9.6] Split policy leakage checks
# [PHASE 9.7] Sector bias calibration persistence
# [PHASE 9.8] Stacking diagnostics & governance
# [SECTION 10] Portfolio: efficient frontier and risk
```

---

### Validation and TDD Alignment

Map to existing and recommended tests (selective execution to avoid timeouts):

- Uncertainty: `tests/test_uncertainty_calibration.py` — verify `uncertainty_summary.json`, coverage stats,
  monotonicity.
- Safety Rails: `tests/test_outlier_safety_rails.py`, `tests/test_phase95_nonnegative_predictions.py` — verify
  `non_negative_violations.json` empty and winsorization summaries present.
- Splits/Leakage: `tests/test_data_splits_policy.py` — assert `leakage_report.json` passes.
- Sector Metrics: `tests/test_regression_sector_metrics.py`, `tests/test_sector_bias_calibration.py` — assert
  calibration artifacts.
- Stacking: `tests/test_stacking_default.py` — check presence and consistency of stacking diagnostics.
- Portfolio: `tests/test_portfolio_optimization*.py`, `tests/test_portfolio_ml_prediction.py`,
  `tests/test_portfolio_risk_management.py`, `tests/test_portfolio_backtesting.py`,
  `tests/test_portfolio_dashboards.py` — validate creation and schema of portfolio artifacts.

Add one lightweight structural test (similar to Phase 9.2):

- `tests/test_notebook_phase94_98_structure.py` — asserts each new section exists and minimal cell markers appear in the
  notebook JSON.

---

### Success Criteria and Checklist

- Phase 9.4–9.8 sections exist with concise cells (≤6 per section) and clearly labeled markers.
- All declared artifacts are produced under the specified subfolders with non-empty content and required keys.
- QA JSONs summarize key diagnostics and pass associated tests.
- Governance artifacts (`model_card_v{MODEL_VERSION}.md`, `lineage.json`) exist and are versioned.
- Portfolio Section 10 emits the specified dashboards and CSV/JSON summaries, obeying constraints and risk budgets.

---

### Quick Implementation Notes

- Prefer reading existing artifacts (CSV/JSON) to keep cells quick; compute-heavy steps remain in library code.
- Guard SHAP imports:
  ```
  try:
      import shap
      SHAP_AVAILABLE = True
  except Exception:
      SHAP_AVAILABLE = False
  ```
- Use `Path(OUTPUT_DIR)/"subdir"` for file paths; avoid hardcoded strings.
- Reuse the standardized predictions DataFrame; verify columns before plotting.

---

### Deliverable Summary (New/Updated Artifacts)

- Uncertainty (9.4): `quantile_predictions_diagnostics.csv`, `coverage_by_sector.json`, `uncertainty_summary.json`, plus
  4 HTML visuals.
- Safety Rails (9.5): `clipping_effect_summary.json`, `non_negative_violations.json`, `safety_rails_summary.json`, plus
  3 HTML visuals.
- Splits (9.6): `grouped_cv_balance_metrics.json`, `leakage_report.json`, `fold_assignments.csv` (optional), plus
  heatmap.
- Calibration (9.7): `sector_bias_calibration_v{MODEL_VERSION}.json`, `metrics_by_sector_time.html`, dashboard.
- Stacking & Governance (9.8): `stacking_contributions.csv`, SHAP or permutation importance HTML, `meta_error_map.html`,
  `model_card_v{MODEL_VERSION}.md`, `lineage.json`.
- Portfolio (Section 10): full suite of dashboards and summaries in `outputs/portfolio/` as listed above.

This plan is ready to be transcribed into the notebook by adding section headers, import markers, and the specified
cells, using the corresponding library functions or thin wrappers that write the artifacts listed here.
