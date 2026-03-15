Based on the analysis of `expected_returns_v3.py` (v3.1/v3.2/v3.3 changes), `expected_returns_summary.sql` schema (112 columns), `finance_ml_analytics_guide.md`, and the 2026-03-10 pipeline log outputs, the following gaps and improvements are identified.

---

### 1. Surface Missing Pipeline Artifacts (New Tabs / Embedded Vizualizations)

The pipeline (Step 9) generates **30+ HTML artifacts**, but many new ones from v3.1–v3.3 are **not served or embedded** in the dashboard. The following artifacts should be integrated:

#### MCMC Posterior Charts (v3.3 — Not Surfaced)
- `er_mcmc_anomaly_posterior.html` — MCMC-enhanced anomaly posterior
- `er_mcmc_credit_risk_posterior.html` — MCMC credit risk posterior
- `er_mcmc_dividend_cut_posterior.html` — MCMC dividend cut posterior
- `er_mcmc_price_target_posterior.html` — MCMC price target achievement posterior
- `er_mcmc_category_sentiment_posterior.html` — MCMC category sentiment posterior

**Suggestion**: Add an **"🔗 MCMC Posteriors"** tab (or sub-tabs within existing Credit Risk / Anomaly tabs) using `render_artifact_or_placeholder()`:

```python
render_artifact_or_placeholder("er_mcmc_credit_risk_posterior", "MCMC Credit Risk Posterior"),
render_artifact_or_placeholder("er_mcmc_anomaly_posterior", "MCMC Anomaly Posterior"),
render_artifact_or_placeholder("er_mcmc_dividend_cut_posterior", "MCMC Dividend Cut Posterior"),
render_artifact_or_placeholder("er_mcmc_price_target_posterior", "MCMC Price Target Posterior"),
```

#### Quality & Risk Deep-Dive (v3.2 — Not Surfaced)
- `er_piotroski_fscore.html` — Piotroski F-Score breakdown
- `er_altman_zscore.html` — Altman Z-Score distribution with distress zones
- `er_beneish_mscore.html` — Beneish M-Score manipulation probability
- `er_risk_tier_sunburst.html` — Sector → Industry → Risk Tier sunburst
- `er_posterior_return_forest.html` — Posterior return forest plot (top N)

**Suggestion**: Embed these in the existing **"🛡️ Credit Risk & Dividend Safety"** tab or create a new **"📊 Quality & Risk Deep-Dive"** tab.

#### Earnings Quality & Growth (v3.1 — Partially Surfaced)
- `er_enhanced_beat_probability.html` — Enhanced beat probability dashboard
- `er_revision_momentum.html` — Already at line 3097 ✓
- `er_gaap_divergence.html` — Already at line 3096 ✓
- `er_beat_rate_heatmap.html` — **Not surfaced**
- `er_earnings_consistency_matrix.html` — **Not surfaced**
- `er_earnings_quality_decomposition.html` — **Not surfaced**
- `er_growth_consistency_matrix.html` — **Not surfaced**
- `er_growth_vs_profitability.html` — **Not surfaced**
- `er_growth_acceleration.html` — **Not surfaced**
- `er_sustainable_growth.html` — **Not surfaced**

**Suggestion**: Add artifact panels within the **"📅 Earnings Calendar & Events"** tab or a new **"📈 Growth & Earnings Quality"** tab.

#### Other Missing Artifacts
- `er_kalman_vs_raw.html` — Kalman vs raw scatter
- `er_tri_model_agreement.html` — Tri-model agreement histogram
- `er_bayesian_sentiment_ridge.html` — Bayesian sentiment ridge
- `er_bayesian_profitability_ridge.html` — Bayesian profitability ridge

---

### 2. Missing Stock Screening Strategies

The pipeline log shows **13 screens** are generated, but the dashboard only runs **9** (lines 5339–5348). Missing screens:

| Screen | Pipeline Count | Dashboard |
|--------|---------------|-----------|
| `sector_relative` | 6,391 stocks | ❌ Missing |
| `low_vol_quality` | 2,261 stocks | ❌ Missing |
| `fcf_compounders` | 960 stocks | ❌ Missing |
| `total_return_leaders` | 6,391 stocks | ❌ Missing |

**Suggestion**: Import the missing screeners from `screening.py` and add them to the screening summary execution block (~line 5337):

```python
from finance_ml.analytics.screening import (
    screen_sector_relative_ranking,
    screen_low_volatility_quality,
    screen_fcf_compounders,
    screen_total_return_leaders,
)

# Add to the screens dict around line 5349:
("sector_relative", screen_sector_relative_ranking),
("low_vol_quality", screen_low_volatility_quality),
("fcf_compounders", screen_fcf_compounders),
("total_return_leaders", screen_total_return_leaders),
```

Also consider adding an **interactive screening tab** with a dropdown to select a screen strategy and view its filtered results in a `DataTable`.

---

### 3. New Summary Columns Not Surfaced

The `expected_returns_summary` SQL schema contains **many new columns** (from v3.1–v3.3) that are loaded but never displayed, filtered, or visualized:

#### Resampled Posterior / Technical Signal Columns
- `resampled_posterior_mean` — Bayesian resampled posterior return
- `technical_adjustment` — Technical signal adjustment factor
- `momentum_signal` — Composite momentum signal
- `volatility_regime_score` — Volatility regime indicator

#### Credible Intervals
- `credible_interval_90` — 90% credible interval (text, e.g., `[low, high]`)
- `credible_interval_95` — 95% credible interval

#### Enhanced Beat Probability
- `prob_beat_given_momentum` — Conditional P(beat|momentum)
- `eps_revision_momentum` — EPS revision momentum score
- `analyst_rating_normalized` — Normalized analyst rating

#### Z-Score & Percentile Ranks
- `expected_upside_pct_zscore`, `filtered_upside_zscore`, `expected_return_prob_weighted_zscore`
- `expected_upside_pct_pctile`, `filtered_upside_pctile`, `expected_return_prob_weighted_pctile`

**Suggestion**:
- Add these columns to `get_formatted_columns()` (lines 444–593) with proper `Format` specs
- Display `resampled_posterior_mean` and `momentum_signal` in the main overview table
- Add `credible_interval_90/95` as tooltip or expandable detail columns
- Surface z-score/percentile columns in the **"📊 Z-Score & Percentile Ranking"** tab as additional scatter axes

---

### 4. Dynamic Filter / Callback Enhancements

#### 4a. Add Missing Filter Dimensions

The `FILTER_CONFIG` (line 600–696) has 20 filters. Consider adding:

```python
{"label": "FY End", "id": "fy-end-dropdown", "column": "fy_end", "width": "18%"},
{"label": "Next Fiscal Quarter", "id": "next-fq-dropdown", "column": "next_fiscal_quarter", "width": "18%"},
```

These are present in the summary table and useful for filtering by reporting cycle.

#### 4b. Cascading / Dependent Filters

Currently, all filters operate independently (line 789–809, `apply_global_filters`). Implement **cascading dropdowns** so that selecting a `region` dynamically updates the `country`, `exchange`, and `trading_country` options:

```python
@app.callback(
    Output("country-dropdown", "options"),
    Input("region-dropdown", "value"),
)
def update_country_options(selected_regions):
    filtered = df if not selected_regions else df[df["region"].isin(selected_regions)]
    return build_filter_options(filtered, "country")
```

Similarly, `sector` → `industry` should be cascaded.

#### 4c. Numeric Range Filters

Add `dcc.RangeSlider` filters for key numeric dimensions:
- `market_cap` (size filtering)
- `expected_upside_pct` (upside range)
- `confidence_score` (confidence threshold)
- `composite_score` (composite quality threshold)
- `altman_z_score` (distress filtering)

These should be added to `build_filter_panel()` alongside the existing dropdowns.

#### 4d. Cross-Tab Filter Persistence

Ensure that filter selections persist across tab switches. Currently, each callback independently collects filter args via `collect_filter_values(*args)` (line 812–819). Consider using a single `dcc.Store` component to hold the current filter state:

```python
dcc.Store(id="global-filter-store", data={}),
```

And have a single callback update it, with all tab callbacks reading from the store.

---

### 5. Layout & Design Structure Refactoring

#### 5a. Modularize the 8,655-Line File

The file is monolithic. Extract into separate modules:

```
finance_ml/dashboards/
├── geib_dash_app.py          # App initialization, layout assembly, server entry
├── geib_layout/
│   ├── __init__.py
│   ├── filters.py            # FILTER_CONFIG, build_filter_panel, apply_global_filters
│   ├── tabs/
│   │   ├── overview.py       # Expected Returns Overview tab layout
│   │   ├── consensus.py      # Model Consensus tab
│   │   ├── screening.py      # Stock Screening tab (NEW)
│   │   ├── earnings.py       # Earnings Calendar tab
│   │   ├── credit_risk.py    # Credit Risk tab
│   │   ├── anomaly.py        # Anomaly Analytics tab
│   │   ├── monte_carlo.py    # Monte Carlo Simulator tab
│   │   ├── kelly.py          # Kelly Criterion tab
│   │   ├── beta_capm.py      # Beta & CAPM tab
│   │   └── efficient_frontier.py
│   ├── formatters.py         # get_formatted_columns, TABLE_STYLE_*, COLORS
│   └── artifacts.py          # render_artifact_or_placeholder, load_plotly_figure_from_html
├── geib_callbacks/
│   ├── __init__.py
│   ├── overview_callbacks.py
│   ├── screening_callbacks.py
│   ├── earnings_callbacks.py
│   └── ...
```

#### 5b. Remove Duplicate CSS

The slider tooltip CSS is duplicated between the `<head>` and `<body>` `<style>` blocks (lines 1569–1627 and 1636–1680). Consolidate into a single `<style>` block in `<head>` or move to an external CSS file in `assets/`.

#### 5c. KPI Cards Row

The KPI summary cards (line ~1715–1762) are generated inline. Extract a reusable `build_kpi_card()` function and add new KPIs:
- **Mean Resampled Posterior Return** — from `resampled_posterior_mean`
- **MCMC Convergence (R̂)** — from latest pipeline run
- **Screening Hit Rate** — percentage of universe passing any screen

---

### 6. Artifact Formatting Improvements

#### 6a. Standardize Artifact Loading

The `render_artifact_or_placeholder()` function (line 1128–1186) loads artifacts from `outputs/analytics/`. For artifacts generated dynamically in callbacks (e.g., screening summary at line 5333–5357), the pattern switches to calling `create_*` functions directly. Standardize by:

1. **Prefer dynamic generation** for filter-responsive artifacts (recalculated per filter state)
2. **Use `render_artifact_or_placeholder()`** only for static/expensive artifacts (MCMC diagnostics, ArviZ panels)

#### 6b. Conditional Table Formatting

Extend `TABLE_STYLE_DATA_CONDITIONAL` (line 171) to cover new columns:

```python
# Resampled posterior: green for positive, red for negative
{"if": {"column_id": "resampled_posterior_mean", "filter_query": "{resampled_posterior_mean} > 0"},
 "color": COLORS["success"], "fontWeight": "bold"},
{"if": {"column_id": "resampled_posterior_mean", "filter_query": "{resampled_posterior_mean} < 0"},
 "color": COLORS["danger"], "fontWeight": "bold"},

# Momentum signal
{"if": {"column_id": "momentum_signal", "filter_query": "{momentum_signal} > 0.5"},
 "color": COLORS["success"]},

# Altman Z-Score distress zones
{"if": {"column_id": "altman_z_score", "filter_query": "{altman_z_score} < 1.8"},
 "color": COLORS["danger"], "fontWeight": "bold"},
{"if": {"column_id": "altman_z_score", "filter_query": "{altman_z_score} > 3.0"},
 "color": COLORS["success"]},
```

#### 6c. Add Column Tooltips

Use `dash_table.DataTable`'s `tooltip_header` property to show column descriptions from the schema metadata, especially for technical columns like `var_5_pct`, `resampled_posterior_mean`, `volatility_regime_score`.

---

### 7. New Tab Suggestions

#### 7a. "📋 Stock Screening Explorer" Tab

A dedicated interactive screening tab:
- **Dropdown**: Select screening strategy (earnings_quality, value, growth, garp, dividend, healthy, valuation_reversion, integrity_growth, high_yield_safe, sector_relative, low_vol_quality, fcf_compounders, total_return_leaders)
- **DataTable**: Show filtered results with `get_formatted_columns()`
- **Summary bar chart**: `create_screening_summary_chart(screens)`
- **Download button**: Export selected screen to CSV

#### 7b. "🧪 Resampled Posterior & MCMC" Tab

Surface the v3.1/v3.3 Bayesian posterior analytics:
- Resampled posterior distribution scatter (from `resampled_posterior_mean` column)
- MCMC convergence diagnostics (static artifact)
- Hierarchical shrinkage diagnostic
- Category posterior diagnostics

#### 7c. "📊 Valuation Deep-Dive" Tab

The pipeline generates 5 valuation artifacts that are not surfaced:
- `er_valuation_multiples.html`
- `er_valuation_distribution.html`
- `er_relative_valuation_matrix.html`
- `er_valuation_vs_growth.html`
- `er_historical_valuation_percentile.html`

Embed via `render_artifact_or_placeholder()` or generate dynamically with the imported viz functions (already imported at lines 97–103).

---

### 8. Data Loading Enhancements (`load_geib_data`)

#### 8a. Load Resampled Posterior Data

The pipeline exports resampled posterior data but `load_geib_data()` doesn't load it. Add:

```python
# --- 4d. Resampled Posterior Data ---
try:
    data["resampled_posterior"] = pd.read_sql(
        f"SELECT * FROM {analytics_schema}.resampled_posterior_returns", engine
    )
except Exception:
    # Fallback: resampled columns in expected_returns_summary
    _resamp_cols = [c for c in ["ticker", "resampled_posterior_mean", "technical_adjustment",
                                 "momentum_signal", "volatility_regime_score"]
                    if c in data["summary"].columns]
    if _resamp_cols:
        data["resampled_posterior"] = data["summary"][_resamp_cols].copy()
```

#### 8b. Load Screening Results from DB

Instead of re-running all screens in the callback (lines 5337–5357), load pre-computed screens from the database (they're exported in Step 10):

```python
for screen_name in ["earnings_quality_stocks", "value_stocks", "growth_momentum_stocks", ...]:
    try:
        data[f"screen_{screen_name}"] = pd.read_sql(
            f"SELECT * FROM {analytics_schema}.{screen_name}", engine
        )
    except Exception:
        pass
```

This eliminates expensive re-computation on every callback trigger.

---

### 9. Summary of Priority Actions

| Priority | Action | Impact |
|----------|--------|--------|
| **High** | Surface MCMC posterior artifacts (5 new charts) | Major data coverage gap |
| **High** | Add 4 missing screening strategies | Incomplete screening coverage |
| **High** | Add new summary columns to tables & formatters | 20+ columns invisible to users |
| **High** | Cascading region→country→exchange filters | Better UX for 6,391-stock universe |
| **Medium** | New "Stock Screening Explorer" interactive tab | Key workflow missing dedicated UI |
| **Medium** | New "Valuation Deep-Dive" tab (5 artifacts) | Already generated, just not shown |
| **Medium** | Embed quality/risk artifacts (Piotroski, Altman, Beneish, sunburst) | Already generated |
| **Medium** | Numeric range sliders (market_cap, confidence_score) | Enables quantitative filtering |
| **Medium** | Load screens from DB instead of re-computing | Performance improvement |
| **Low** | Modularize 8,655-line file into sub-modules | Maintainability |
| **Low** | Remove duplicate CSS blocks | Code hygiene |
| **Low** | Add `dcc.Store` for cross-tab filter persistence | UX polish |
| **Low** | Add column tooltips from schema metadata | Discoverability |

