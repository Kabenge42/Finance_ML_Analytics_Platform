Based on the `expected_returns_v3.py` (3707 lines), the `expected_returns_analytics.ipynb` (data input pattern), and the `finance_ml/analytics/visualizations/` modules, here is the recommended notebook organization with cell-by-cell breakdown.

---

### Cell 1 — Markdown: Title & Description

```markdown
# Expected Returns Analytics Module (v3.1)

Automated pipeline for expected returns analysis using the v3.1+ analytics platform:
- **Monte Carlo Simulation** — Probabilistic upside/downside distributions
- **Price Target Achievement** — Probability-weighted expected returns by sector
- **Kalman Filtered Targets** — Noise-reduced price target signals
- **Earnings Beat Analysis** — Three-layer Bayesian earnings beat probability
- **Cross-Model Comparison** — MC vs Kalman vs Achievement model alignment
- **Quad-Model Agreement** — MC + Kalman + Achievement + Earnings Beat
- **Statistical Analysis** — Bayesian category analysis, copula dependency, MCMC
- **Probability Analytics** — Category-level probability distributions, credit risk
- **Stock Screening** — Quality, value, growth, dividend, GARP, health filters

Data sources (v3.2 — Equities MV + Feature Views):
- `public.mv_equities` (equities data via `load_equities_data_from_db`)
- `public.vw_features_*` (17 feature views via `load_all_feature_views`)
- `equities_schema_metadata` (dynamic column discovery via `get_equities_schema`)
```

---

### Cell 2 — Code: Imports & Setup (lines 31–188 of .py)

```python
#%% Imports & Setup
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats as sp_stats

# --- Data utilities ---
from finance_ml.analytics.data_utils import (
    ExportConfig, aggregate_probability_results, backfill_feature_columns,
    compute_metric_statistics, export_to_csv, export_to_db, export_to_json,
    get_equities_schema, get_identifier_cols_set, load_all_feature_views,
    load_feature_data_from_db, load_equities_data_from_db,
    load_feature_categories_from_db, load_identifier_columns,
    reorder_with_identifiers, validate_feature_alignment,
)

# --- Optimised operations ---
from finance_ml.analytics.optimized_ops import (
    fast_ruin_probability, get_optimization_status,
    vectorized_percentile_rank, vectorized_zscore,
)

# --- Probability models ---
from finance_ml.analytics.probability_analytics import (
    CategoryProbabilityAnalyzer, CreditRiskProbabilityModel,
    DividendCutProbabilityModel, EarningsBeatProbabilityModel,
    EPSStreakAnalyzer, PriceTargetAchievementModel,
    ResampledBeatProbabilityModel, create_earnings_probability_dashboard,
    export_probability_analytics_results,
)

# --- Screening ---
from finance_ml.analytics.screening import (
    create_enhanced_screener, create_sector_relative_ranking,
    rank_stocks_by_composite_score, screen_dividend_quality,
    screen_earnings_quality, screen_financial_health,
    screen_garp_opportunities, screen_growth_momentum,
    screen_high_yield_safe_dividends, screen_integrity_filtered_growth,
    screen_valuation_reversion_candidates, screen_value_opportunities,
)

# --- Statistical analysis ---
from finance_ml.analytics.statistical_analysis import (
    bayesian_category_analysis, bayesian_earnings_beat_model,
    calculate_conditional_probabilities, calculate_ruin_probability,
    detect_accounting_anomalies, fit_distributions_by_category,
    fit_gaussian_copula, hierarchical_mcmc_by_sector,
    kalman_filter_price_target, kalman_momentum_filter,
    mcmc_student_t, monte_carlo_price_target_simulation,
    parallel_mcmc_chains, resampled_posterior_returns,
    run_category_probability_analytics,
    analyze_employee_productivity_frontier, analyze_reporting_lag_sentiment,
)

# --- InferenceData schema (ArviZ) ---
try:
    from finance_ml.analytics.inference_schema import (
        ARVIZ_AVAILABLE, EquityCoordinates,
        build_beat_probability_inference_data, build_credit_risk_inference_data,
        build_monte_carlo_inference_data, summarize_inference_data,
    )
except ImportError:
    ARVIZ_AVAILABLE = False
    EquityCoordinates = None

# --- Visualizations: Probabilistic (ArviZ-backed) ---
from finance_ml.analytics.visualizations._shared import PLOTLY_TEMPLATE
from finance_ml.analytics.visualizations.probability_viz import (
    create_bayesian_category_ridge, create_beat_probability_posterior,
    create_posterior_return_forest, create_ruin_probability_diagnostic,
    create_tri_model_posterior_comparison,
)

# --- Visualizations: Quality & Risk ---
from finance_ml.analytics.visualizations.quality_risk import (
    create_altman_zscore_distribution, create_beneish_mscore_analysis,
    create_distress_early_warning_dashboard, create_piotroski_fscore_breakdown,
    create_quality_risk_quadrant, create_risk_tier_sunburst,
)

# --- Visualizations: Earnings Quality ---
from finance_ml.analytics.visualizations.earnings_quality import (
    create_enhanced_beat_probability_dashboard as create_enhanced_beat_prob_dash,
    create_gaap_divergence_plot, create_revision_momentum_chart,
)

# --- Visualizations: Expected Returns ---
from finance_ml.analytics.visualizations.expected_returns_viz import (
    create_beat_vs_achievement_scatter, create_kalman_vs_raw_scatter,
    create_mc_return_distribution, create_model_dispersion_dashboard,
    create_return_distribution_fit_chart, create_screening_summary_chart,
    create_sector_heatmap, create_sector_return_analytics_heatmap,
    create_sector_risk_reward_scatter, create_strong_consensus_bar,
    create_tri_model_agreement_histogram, create_var_analysis,
)

from finance_ml.logging_config import configure_logging
from finance_ml.ml_workflow.core.utils import safe_divide

px.defaults.template = PLOTLY_TEMPLATE
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
```

---

### Cell 3 — Code: Pipeline Configuration (lines 190–244)

```python
#%% Pipeline Configuration
@dataclass
class PipelineConfig:
    """Centralized configuration for the expected returns analytics pipeline."""
    mc_simulations: int = 50_000
    mc_max_stocks: int = 10_000
    mcmc_chains: int = 4
    mcmc_samples: int = 10_000
    beat_threshold: float = 0.6
    output_dir: str = "outputs/analytics"
    log_file: str | None = "logs/expected_returns_pipeline.log"
    log_level: int = logging.INFO

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        return cls(
            mc_simulations=int(os.environ.get("ER_MC_SIMULATIONS", 50_000)),
            mc_max_stocks=int(os.environ.get("ER_MC_MAX_STOCKS", 10_000)),
            mcmc_chains=int(os.environ.get("ER_MCMC_CHAINS", 4)),
            mcmc_samples=int(os.environ.get("ER_MCMC_SAMPLES", 10_000)),
            output_dir=os.environ.get("ER_OUTPUT_DIR", "outputs/analytics"),
            log_file=os.environ.get("ER_LOG_FILE", "logs/expected_returns_pipeline.log"),
        )

cfg = PipelineConfig.from_env()
output_dir = Path(cfg.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

configure_logging(level=cfg.log_level, log_file=cfg.log_file, console=True)
```

---

### Cell 4 — Code: Pipeline Helpers (lines 247–375)

```python
#%% Pipeline Helpers
# Include: _log_and_print, _has_required_columns, get_feature_categories,
# _LazyFeatureCategories, reconcile_feature_categories
# (lines 252–375 from expected_returns_v3.py)
```

---

### Cell 5 — Code: Schema Column Discovery (lines 378–503)

```python
#%% Schema Column Discovery & Backfill
# Include: _get_schema_columns, _apply_backfill_and_kalman
# (lines 395–503 from expected_returns_v3.py)
```

---

### Cell 6 — Code: Model Statistics Helpers (lines 506–631)

```python
#%% Model Statistics
# Include: compute_model_detailed_statistics, print_model_statistics
# (lines 511–631 from expected_returns_v3.py)
```

---

### Cell 7 — Markdown: Data Loading Section Header

```markdown
## 1. Data Loading — Equities & Feature Views (v3.2)

Load data from materialized views:
- `mv_equities` → primary equities data
- `vw_features_*` → 17 feature views  
- `mv_all_stock_features` → combined feature MV

Mirrors the SQL queries from `expected_returns_analytics.ipynb`:
- `SELECT * FROM monte_carlo_simulation`
- `SELECT * FROM price_target_achievement`
- `SELECT * FROM kalman_filtered_price_targets`
- `SELECT * FROM vw_features_analyst_sentiment`
- `SELECT * FROM vw_features_cashflow`
```

---

### Cell 8 — Code: Data Loading Functions (lines 634–796)

```python
#%% Data Loading Functions
# Include: load_expected_returns_data, load_all_stock_features, load_analytics_table
# (lines 639–796 from expected_returns_v3.py)
```

---

### Cell 9 — Code: Execute Data Loading (from main() lines 2762–2795)

```python
#%% Execute: Load Data
df = load_expected_returns_data()
print(f"✓ Loaded mv_equities: {len(df):,} stocks × {len(df.columns)} features")

df_all = load_all_stock_features()
if not df_all.empty:
    print(f"✓ Loaded feature views: {len(df_all):,} stocks × {len(df_all.columns)} features")
else:
    df_all = df.copy()

df_features = load_analytics_table()
if not df_features.empty:
    print(f"✓ Loaded mv_all_stock_features: {len(df_features):,} stocks × {len(df_features.columns)} features")

df.head()
```

---

### Cell 10 — Markdown: Monte Carlo Section

```markdown
## 2. Monte Carlo Simulation

Probabilistic price target simulation with historical target drift enrichment.
```

---

### Cell 11 — Code: MC Model Runner Functions (lines 800–996)

```python
#%% Monte Carlo Model Functions
# Include: ALL_HISTORICAL_PRICE_TARGET_COLS, _resolve_available_historical_cols,
# _log_historical_coverage, run_monte_carlo_analysis
# (lines 800–996 from expected_returns_v3.py)
```

---

### Cell 12 — Code: Execute Monte Carlo (from main() lines 2805–2887)

```python
#%% Execute: Monte Carlo Simulation
mc = run_monte_carlo_analysis(df, n_simulations=cfg.mc_simulations, max_stocks=cfg.mc_max_stocks, use_historical_targets=True)
mc = compute_price_target_mc(mc, df)
print(f"✓ {len(mc):,} stocks simulated")
print(f"  Mean upside: {mc['expected_upside_pct'].mean():.1f}%")
mc.head()
```

---

### Cell 13 — Code: MC Visualization 🎨

```python
#%% Visualization: MC Return Distribution
fig = create_mc_return_distribution(mc)
fig.show()
```

### Cell 14 — Code: MC Risk-Reward 🎨

```python
#%% Visualization: Sector Risk-Reward Scatter
fig = create_sector_risk_reward_scatter(mc)
fig.show()
```

### Cell 15 — Code: MC VaR Analysis 🎨

```python
#%% Visualization: VaR Analysis
fig = create_var_analysis(mc)
fig.show()
```

### Cell 16 — Code: MC Posterior Forest 🎨

```python
#%% Visualization: Posterior Return Forest
fig = create_posterior_return_forest(mc, top_n=25)
fig.show()
```

### Cell 17 — Code: Return Distribution Fit 🎨

```python
#%% Visualization: Return Distribution Fit
fig = create_return_distribution_fit_chart(mc)
fig.show()
```

---

### Cell 18 — Markdown: Price Target Achievement

```markdown
## 3. Price Target Achievement

Probability-weighted expected returns by sector with analyst conviction scoring.
```

---

### Cell 19 — Code: PT Functions (lines 999–1073)

```python
#%% Price Target Achievement Functions
# Include: run_price_target_achievement
```

### Cell 20 — Code: Execute PT (from main() lines 2896–2943)

```python
#%% Execute: Price Target Achievement
pt = run_price_target_achievement(df, use_historical_targets=True, feature_df=df_all)
pt = compute_price_target_prob_weighted(pt, df)
print(f"✓ {len(pt):,} stocks analyzed")
pt.head()
```

---

### Cell 21 — Markdown: Kalman Filter

```markdown
## 4. Kalman Filtered Targets

Noise-reduced price target signals using Kalman filtering.
```

### Cell 22 — Code: Kalman Functions (lines 1076–1282)

```python
#%% Kalman Filter Functions
# Include: run_kalman_filter, _enrich_with_historical_target_drift
```

### Cell 23 — Code: Execute Kalman

```python
#%% Execute: Kalman Filter
kal = run_kalman_filter(df, use_historical_targets=True)
print(f"✓ {len(kal):,} stocks filtered, mean filtered upside: {kal['filtered_upside'].mean():.1f}%")
kal.head()
```

### Cell 24 — Code: Kalman Visualization 🎨

```python
#%% Visualization: Kalman vs Raw Scatter
fig = create_kalman_vs_raw_scatter(kal)
fig.show()
```

---

### Cell 25 — Markdown: Earnings Beat

```markdown
## 5. Earnings Beat Analysis

Three-layer Bayesian earnings beat probability model.
```

### Cell 26 — Code: Earnings Beat Functions (lines 1285–1359)

```python
#%% Earnings Beat Functions
# Include: run_earnings_beat_analysis
```

### Cell 27 — Code: Execute Earnings Beat

```python
#%% Execute: Earnings Beat
beat = run_earnings_beat_analysis(df_all if not df_all.empty else df)
print(f"✓ {len(beat):,} stocks, mean P(beat): {beat['posterior_beat_prob'].mean():.3f}")
beat.head()
```

### Cell 28 — Code: Earnings Beat Visualizations 🎨

```python
#%% Visualization: Beat Probability Posterior
fig = create_beat_probability_posterior(beat, top_n=12)
fig.show()
```

### Cell 29 — Code: Earnings Probability Dashboard 🎨

```python
#%% Visualization: Earnings Probability Dashboard
fig = create_earnings_probability_dashboard(beat)
fig.show()
```

### Cell 30 — Code: Enhanced Beat Probability 🎨

```python
#%% Visualization: Enhanced Beat Probability Dashboard
fig = create_enhanced_beat_prob_dash(beat)
fig.show()
```

### Cell 31 — Code: Revision Momentum 🎨

```python
#%% Visualization: EPS Revision Momentum
if "eps_revision_momentum" in beat.columns:
    fig = create_revision_momentum_chart(beat, top_n=30)
    fig.show()
```

### Cell 32 — Code: GAAP Divergence 🎨

```python
#%% Visualization: GAAP Divergence
if "gaap_adj_eps_gap_pct" in beat.columns:
    fig = create_gaap_divergence_plot(beat)
    fig.show()
```

---

### Cell 33 — Markdown: Credit Risk & Dividend Safety

```markdown
## 5b. Credit Risk & Dividend Safety
```

### Cell 34 — Code: Credit/Dividend Functions (lines 1362–1494)

```python
#%% Credit Risk & Dividend Safety Functions
# Include: run_credit_risk_analysis, run_dividend_safety_analysis
```

### Cell 35 — Code: Execute Credit/Dividend

```python
#%% Execute: Credit Risk & Dividend Safety
credit = run_credit_risk_analysis(df, feature_df=df_all)
div_safety = run_dividend_safety_analysis(df_all)
```

### Cell 36 — Code: Ruin Probability Diagnostic 🎨

```python
#%% Visualization: Ruin Probability Diagnostic
if not credit.empty and "ruin_probability" in credit.columns:
    fig = create_ruin_probability_diagnostic(credit, top_n=20)
    fig.show()
```

---

### Cell 37 — Markdown: Stock Screening

```markdown
## 5c. Stock Screening

Quality, value, growth, dividend, GARP, health, and integrity-filtered screens.
```

### Cell 38 — Code: Screening Functions (lines 1592–1781)

```python
#%% Stock Screening Functions
# Include: run_stock_screening, filter_quality_stocks
```

### Cell 39 — Code: Execute Screening

```python
#%% Execute: Stock Screening
screens = run_stock_screening(df_all)
for name, screen_df in screens.items():
    if not screen_df.empty:
        print(f"  ✓ {name}: {len(screen_df):,} stocks")
```

### Cell 40 — Code: Screening Summary 🎨

```python
#%% Visualization: Screening Summary
if screens:
    fig = create_screening_summary_chart(screens)
    fig.show()
```

---

### Cell 41 — Markdown: Resampled Posterior

```markdown
## 5d. Resampled Bayesian Posterior Returns (v3.1)
```

### Cell 42 — Code: Execute Resampled Posterior

```python
#%% Execute: Resampled Posterior
resampled_posterior = run_resampled_posterior_analysis(df)
if not resampled_posterior.empty:
    print(f"✓ {len(resampled_posterior):,} stocks, mean posterior: {resampled_posterior['posterior_mean'].mean()*100:.2f}%")
```

---

### Cell 43 — Markdown: Cross-Model Alignment

```markdown
## 6. Cross-Model Alignment

Tri-model (MC + Kalman + Achievement) and quad-model (+ Earnings Beat) alignment.
```

### Cell 44 — Code: Alignment Functions (lines 1784–1911)

```python
#%% Alignment Functions
# Include: _SIGNAL_LABELS, _SIGNAL_LABELS_4, build_tri_model_alignment,
# build_quad_model_alignment
```

### Cell 45 — Code: Execute Alignment

```python
#%% Execute: Cross-Model Alignment
tri = build_tri_model_alignment(mc, kal, pt)
strong = extract_strong_consensus(tri)
quad = build_quad_model_alignment(tri, beat, beat_threshold=cfg.beat_threshold)
corr_info = compute_cross_model_correlation(mc, kal)
print(f"Tri-model: {len(tri):,}, Strong consensus: {len(strong)}, Quad full: {(quad['quad_agreement']==4).sum()}")
```

### Cell 46 — Code: Tri-Model Agreement 🎨

```python
#%% Visualization: Tri-Model Agreement
fig = create_tri_model_agreement_histogram(tri)
fig.show()
```

### Cell 47 — Code: Sector Heatmap 🎨

```python
#%% Visualization: Sector Heatmap
fig = create_sector_heatmap(tri)
fig.show()
```

### Cell 48 — Code: Strong Consensus 🎨

```python
#%% Visualization: Strong Consensus Picks
fig = create_strong_consensus_bar(strong)
fig.show()
```

### Cell 49 — Code: Tri-Model Posterior Comparison 🎨

```python
#%% Visualization: Tri-Model Posterior Comparison
fig = create_tri_model_posterior_comparison(strong, top_n=12)
fig.show()
```

### Cell 50 — Code: Beat vs Achievement 🎨

```python
#%% Visualization: Beat vs Achievement Scatter
fig = create_beat_vs_achievement_scatter(beat, pt)
fig.show()
```

---

### Cell 51 — Markdown: Expected Returns Summary

```markdown
## 7. Expected Returns Summary (4-Model Merge)

Merges MC, Kalman, Price Target, and Earnings Beat into a unified summary with quality filtering, z-score ranking, sector analytics, and hierarchical MCMC.
```

### Cell 52 — Code: Summary Functions (lines 1914–2601)

```python
#%% Summary & Analytics Functions
# Include: build_expected_returns_summary, extract_strong_consensus,
# compute_derived_price_target, compute_price_target_prob_weighted,
# compute_price_target_mc, compute_sector_expected_returns,
# compute_sector_return_analytics, compute_return_zscore_ranks,
# compute_cross_model_correlation, compute_cross_model_diagnostics,
# compute_return_distribution_analytics, run_parallel_mcmc_return_analysis,
# run_resampled_posterior_analysis
```

### Cell 53 — Code: Execute Summary

```python
#%% Execute: Build Summary
summary = build_expected_returns_summary(mc, kal, pt, beat, source_df=df_all)
summary = filter_quality_stocks(summary, df_all)
summary = compute_return_zscore_ranks(summary)
sector_analytics = compute_sector_return_analytics(summary)
print(f"✓ {len(summary):,} stocks in summary")
summary.head()
```

### Cell 54 — Code: Model Dispersion Dashboard 🎨

```python
#%% Visualization: Model Dispersion Dashboard
fig = create_model_dispersion_dashboard(summary)
fig.show()
```

### Cell 55 — Code: Summary Posterior 🎨

```python
#%% Visualization: Expected Returns Summary Posterior
fig = create_tri_model_posterior_comparison(summary, top_n=12)
fig.show()
```

### Cell 56 — Code: Sector Return Analytics 🎨

```python
#%% Visualization: Sector Return Analytics Heatmap
if not sector_analytics.empty:
    fig = create_sector_return_analytics_heatmap(sector_analytics)
    fig.show()
```

---

### Cell 57 — Markdown: Parallel MCMC

```markdown
## 7a. Parallel MCMC Return Analysis (v3.1)

Gelman-Rubin convergence diagnostics across parallel chains.
```

### Cell 58 — Code: Execute MCMC

```python
#%% Execute: Parallel MCMC
mcmc_result = run_parallel_mcmc_return_analysis(mc, n_chains=cfg.mcmc_chains, n_samples=cfg.mcmc_samples)
if mcmc_result:
    print(f"R̂={mcmc_result.get('r_hat', float('nan')):.4f}, converged={mcmc_result.get('converged', False)}")
```

---

### Cell 59 — Markdown: Category Bayesian Analytics

```markdown
## 7b. Per-Category Bayesian Probability Analytics
```

### Cell 60 — Code: Execute Category Analytics

```python
#%% Execute: Category Probability Analytics
category_analytics = run_category_probability_analysis(df_all)
for cat_name, cat_result in category_analytics.items():
    print(f"  {cat_name}: {cat_result.get('features_analyzed', 0)} features")
```

### Cell 61 — Code: Bayesian Sentiment Ridge 🎨

```python
#%% Visualization: Bayesian Analyst Sentiment Ridge
sentiment_features = [f for f in ["analyst_bullish_pct", "upside_potential", "eps_revision_momentum",
    "analyst_conviction", "pt_consensus_convergence"] if f in df_all.columns]
if sentiment_features:
    results = bayesian_category_analysis(df_all, "Analyst Sentiment", sentiment_features)
    fig = create_bayesian_category_ridge(results, category_name="Analyst Sentiment")
    fig.show()
```

### Cell 62 — Code: Bayesian Profitability Ridge 🎨

```python
#%% Visualization: Bayesian Profitability Ridge
prof_features = [f for f in ["roe", "roa", "roic", "gross_margin_pct", "operating_margin_pct"] if f in df_all.columns]
if prof_features:
    results = bayesian_category_analysis(df_all, "Profitability", prof_features)
    fig = create_bayesian_category_ridge(results, category_name="Profitability")
    fig.show()
```

---

### Cell 63 — Markdown: Quality & Risk Dashboards

```markdown
## 8. Quality & Risk Visualizations

Deep-dive quality and risk dashboards from `finance_ml.analytics.visualizations.quality_risk`.
```

### Cell 64 — Code: Quality Risk Quadrant 🎨

```python
#%% Visualization: Quality-Risk Quadrant
fig = create_quality_risk_quadrant(df)
fig.show()
```

### Cell 65 — Code: Distress Early Warning 🎨

```python
#%% Visualization: Distress Early Warning Dashboard
fig = create_distress_early_warning_dashboard(df)
fig.show()
```

### Cell 66 — Code: Piotroski F-Score 🎨

```python
#%% Visualization: Piotroski F-Score Breakdown
if "piotroski_f_score" in df.columns:
    fig = create_piotroski_fscore_breakdown(df)
    fig.show()
```

### Cell 67 — Code: Altman Z-Score 🎨

```python
#%% Visualization: Altman Z-Score Distribution
if "altman_z_score" in df.columns:
    fig = create_altman_zscore_distribution(df)
    fig.show()
```

### Cell 68 — Code: Beneish M-Score 🎨

```python
#%% Visualization: Beneish M-Score Analysis
fig = create_beneish_mscore_analysis(df)
fig.show()
```

### Cell 69 — Code: Risk Tier Sunburst 🎨

```python
#%% Visualization: Risk Tier Sunburst
if "distress_risk_score" in df.columns:
    fig = create_risk_tier_sunburst(df)
    fig.show()
```

---

### Cell 70 — Markdown: InferenceData (ArviZ)

```markdown
## 9. InferenceData (ArviZ)

Build ArviZ InferenceData objects for MC, Beat Probability, and Credit Risk models.
```

### Cell 71 — Code: Build InferenceData (from main() lines 3538–3593)

```python
#%% Build InferenceData
if ARVIZ_AVAILABLE:
    if not mc.empty:
        idata_mc = build_monte_carlo_inference_data(mc, df_all, n_simulations=25_000)
        print(f"✓ MC InferenceData: {summarize_inference_data(idata_mc)}")
    if not beat.empty and "posterior_alpha" in beat.columns:
        idata_beat = build_beat_probability_inference_data(beat, df_all, n_posterior_samples=4000, n_chains=4)
        print(f"✓ Beat InferenceData: {summarize_inference_data(idata_beat)}")
    if not credit.empty:
        idata_credit = build_credit_risk_inference_data(credit, df_all)
        print(f"✓ Credit Risk InferenceData: {summarize_inference_data(idata_credit)}")
else:
    print("⏭️ ArviZ not available — skipping InferenceData")
```

---

### Cell 72 — Markdown: Export

```markdown
## 10. Export Results

Export all analytics to database, CSV, and JSON in `outputs/analytics/`.
```

### Cell 73 — Code: Export Functions (lines 2615–2693)

```python
#%% Export Functions
# Include: export_expected_returns_results
```

### Cell 74 — Code: Execute Export (from main() lines 3601–3653)

```python
#%% Execute: Export
exports = export_expected_returns_results(
    mc=mc, pt=pt, kal=kal, tri=tri, strong=strong, beat=beat,
    summary=summary, credit=credit, div_safety=div_safety,
    screens=screens, output_dir=str(output_dir),
)
for name, dest in exports.items():
    print(f"  ✓ {name} → {dest}")
```

---

### Cell 75 — Markdown: Pipeline Summary

```markdown
## ✅ Pipeline Summary
```

### Cell 76 — Code: Final Summary (from main() lines 3658–3707)

```python
#%% Pipeline Summary
print("=" * 80)
print("✅ EXPECTED RETURNS ANALYTICS v3.1 COMPLETE")
print("=" * 80)
print(f"  mv_equities:              {len(df):,} stocks × {len(df.columns)} features")
print(f"  Monte Carlo:              {len(mc):,}")
print(f"  Price Target Achievement: {len(pt):,}")
print(f"  Kalman Filtered:          {len(kal):,}")
print(f"  Earnings Beat:            {len(beat):,}")
print(f"  Credit Risk:              {len(credit):,}")
print(f"  Tri-model aligned:        {len(tri):,}")
print(f"  Strong consensus:         {len(strong):,}")
print(f"  Summary:                  {len(summary):,}")
```

---

### Summary of Visualization Outputs (27 charts mapped to cells)

These correspond to the HTML files in `outputs/analytics/er_*.html`:

| Output File | Visualization Function | Cell |
|:---|:---|:---|
| `er_mc_distribution.html` | `create_mc_return_distribution(mc)` | 13 |
| `er_sector_risk_reward.html` | `create_sector_risk_reward_scatter(mc)` | 14 |
| `er_var_analysis.html` | `create_var_analysis(mc)` | 15 |
| `er_posterior_return_forest.html` | `create_posterior_return_forest(mc)` | 16 |
| `er_return_distribution_fit.html` | `create_return_distribution_fit_chart(mc)` | 17 |
| `er_kalman_vs_raw.html` | `create_kalman_vs_raw_scatter(kal)` | 24 |
| `er_beat_probability_posterior.html` | `create_beat_probability_posterior(beat)` | 28 |
| `er_earnings_probability_dashboard.html` | `create_earnings_probability_dashboard(beat)` | 29 |
| `er_enhanced_beat_probability.html` | `create_enhanced_beat_prob_dash(beat)` | 30 |
| `er_revision_momentum.html` | `create_revision_momentum_chart(beat)` | 31 |
| `er_gaap_divergence.html` | `create_gaap_divergence_plot(beat)` | 32 |
| `er_ruin_probability_diagnostic.html` | `create_ruin_probability_diagnostic(credit)` | 36 |
| `er_screening_summary.html` | `create_screening_summary_chart(screens)` | 40 |
| `er_tri_model_agreement.html` | `create_tri_model_agreement_histogram(tri)` | 46 |
| `er_sector_heatmap.html` | `create_sector_heatmap(tri)` | 47 |
| `er_strong_consensus.html` | `create_strong_consensus_bar(strong)` | 48 |
| `er_tri_model_posterior.html` | `create_tri_model_posterior_comparison(strong)` | 49 |
| `er_beat_vs_achievement.html` | `create_beat_vs_achievement_scatter(beat, pt)` | 50 |
| `er_model_dispersion_dashboard.html` | `create_model_dispersion_dashboard(summary)` | 54 |
| `er_expected_returns_summary_posterior.html` | `create_tri_model_posterior_comparison(summary)` | 55 |
| `er_sector_return_analytics.html` | `create_sector_return_analytics_heatmap(...)` | 56 |
| `er_bayesian_sentiment_ridge.html` | `create_bayesian_category_ridge(...)` | 61 |
| `er_bayesian_profitability_ridge.html` | `create_bayesian_category_ridge(...)` | 62 |
| `er_quality_risk_quadrant.html` | `create_quality_risk_quadrant(df)` | 64 |
| `er_distress_early_warning.html` | `create_distress_early_warning_dashboard(df)` | 65 |
| `er_piotroski_fscore.html` | `create_piotroski_fscore_breakdown(df)` | 66 |
| `er_altman_zscore.html` | `create_altman_zscore_distribution(df)` | 67 |
| `er_beneish_mscore.html` | `create_beneish_mscore_analysis(df)` | 68 |
| `er_risk_tier_sunburst.html` | `create_risk_tier_sunburst(df)` | 69 |

### Key Design Principles

1. **Mirrors `expected_returns_analytics.ipynb` pattern**: Data input cells followed by analysis cells, but expanded with the full v3.1 pipeline
2. **Each pipeline step gets its own markdown header + code cell**: Matching the 10-step `main()` structure
3. **Visualizations are inline** (`.show()` instead of `.write_html()`): Each visualization gets its own cell immediately after the analysis that produces its data
4. **Function definitions are separated from execution**: Define functions in one cell, execute in the next — enables re-running analysis without re-defining
5. **All 27 `er_*.html` outputs** from `outputs/analytics/` are represented as inline Plotly charts using the imported visualization functions from `finance_ml.analytics.visualizations`