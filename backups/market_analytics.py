# %% [markdown]
# # Feature Engineering Analytics Dashboard
# 
# %% [markdown]
# ---
# ## 1. Setup & Configuration
# 
# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
# Cell 2: Setup and Configuration
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

# Import feature analytics module functions
from finance_ml.analytics.feature_analytics import (
    PLOTLY_TEMPLATE,
    analyze_distress_distribution,
    bayesian_earnings_beat_model,
    create_composite_quality_score,
    create_interactive_momentum_dashboard,
    create_interactive_valuation_heatmap,
    create_leverage_liquidity_quadrant,
    create_summary_dashboard,
    monte_carlo_price_target_simulation,
)

warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

# Dark theme for Plotly
px.defaults.template = PLOTLY_TEMPLATE

from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

# Optional import for database access
if TYPE_CHECKING:
    from sqlalchemy import Engine

try:
    from sqlalchemy import create_engine
except ImportError:  # pragma: no cover
    create_engine = None  # type: ignore

# Dark theme for Plotly
PLOTLY_TEMPLATE = "plotly_dark"
px.defaults.template = PLOTLY_TEMPLATE


def load_feature_data_from_db(
    db_url: Optional[str] = None,
    earnings_date_filter: str = "2026-01-01",
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load feature data from PostgreSQL database materialized view.

    Loads data from public.mv_all_stock_features with optional filtering
    by next_earnings date. This function replicates the SQL query from
    feature_analytics.ipynb notebook.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL (e.g., postgresql+psycopg2://user:pass@host:5432/postgres)
        If None, reads from DB_URL environment variable
    earnings_date_filter : str, default "2026-01-01"
        Filter stocks with next_earnings >= this date (ISO format: YYYY-MM-DD)
    limit : int, optional
        Maximum number of rows to return

    Returns
    -------
    pd.DataFrame
        DataFrame with feature data from mv_all_stock_features

    Raises
    ------
    ImportError
        If SQLAlchemy or psycopg2 not available
    ValueError
        If db_url is not provided and DB_URL environment variable is not set

    Examples
    --------
    >>> # Load from environment variable
    >>> df = load_feature_data_from_db()
    >>>
    >>> # Load with explicit URL
    >>> db_url = "postgresql+psycopg2://postgres:@localhost:5432/postgres"
    >>> df = load_feature_data_from_db(db_url=db_url)
    >>>
    >>> # Load with custom date filter
    >>> df = load_feature_data_from_db()
    """
    if create_engine is None:
        raise ImportError(
            "SQLAlchemy not available. Install psycopg2-binary and SQLAlchemy to use database loading."
        )

    # Resolve database URL
    if db_url is None:
        db_url = os.environ.get("DB_URL")
        if db_url is None:
            raise ValueError(
                "db_url parameter not provided and DB_URL environment variable not set. "
                "Please provide a database URL or set the DB_URL environment variable."
            )

    # Resolve schema from environment, default to public
    schema = os.environ.get("DB_SCHEMA", "public")
    view_name = "mv_all_stock_features"
    view_ref = f"{schema}.{view_name}"

    logging.info(
        "Loading feature data from %s (view: %s, earnings_date_filter: %s)",
        db_url.split("@")[-1] if "@" in db_url else db_url,  # Hide credentials in log
        view_ref,
        earnings_date_filter
    )

    # Create SQLAlchemy engine
    engine = create_engine(db_url)

    # Build SQL query matching the notebook
    base_query = f"""
        SELECT *
        FROM {view_ref}
        WHERE next_earnings >= DATE '{earnings_date_filter}'
        ORDER BY next_earnings
    """

    # Apply limit if specified
    if limit is not None:
        query = f"{base_query} LIMIT {int(limit)}"
    else:
        query = base_query

    # Execute query and load into DataFrame
    all_stocks_features = pd.read_sql(query, engine)

    logging.info("Loaded %d rows from %s", len(all_stocks_features), view_ref)

    return all_stocks_features
# Normalize SQL result and backfill expected columns for charts
if not isinstance(all_stocks_features, pd.DataFrame):
    try:
        all_stocks_features = pd.DataFrame(all_stocks_features)
    except Exception:
        pass

if isinstance(all_stocks_features, pd.DataFrame):
    # Backfill analyst_neutral_pct if missing but components exist
    if "analyst_neutral_pct" not in all_stocks_features.columns:
        bullish = all_stocks_features.get("analyst_bullish_pct")
        bearish = all_stocks_features.get("analyst_bearish_pct")
        if bullish is not None and bearish is not None:
            neutral = 100 - bullish - bearish
            all_stocks_features["analyst_neutral_pct"] = neutral.clip(lower=0, upper=100)

    # Map inventory_turnover to expected column name
    if "inventory_turnover_mv" not in all_stocks_features.columns:
        if "inventory_turnover" in all_stocks_features.columns:
            all_stocks_features["inventory_turnover_mv"] = all_stocks_features["inventory_turnover"]

    # Calculate inventory_days from turnover
    if "inventory_days" not in all_stocks_features.columns:
        turnover_col = all_stocks_features.get("inventory_turnover_mv")
        if turnover_col is not None:
            turnover = turnover_col.replace(0, pd.NA)
            all_stocks_features["inventory_days"] = 365 / turnover

    # Map R&D intensity columns
    if "rnd_intensity_ltm" not in all_stocks_features.columns:
        for src_col in ["rnd_intensity", "rnd_to_revenue"]:
            if src_col in all_stocks_features.columns:
                all_stocks_features["rnd_intensity_ltm"] = all_stocks_features[src_col]
                break

    # Map tangible book value columns
    if "tangible_book_value_ltm" not in all_stocks_features.columns:
        if "tangible_book_value" in all_stocks_features.columns:
            all_stocks_features["tangible_book_value_ltm"] = all_stocks_features["tangible_book_value"]

    # Map goodwill concentration
    if "goodwill_concentration" not in all_stocks_features.columns:
        for src_col in ["goodwill_to_equity", "goodwill_to_assets_pct"]:
            if src_col in all_stocks_features.columns:
                all_stocks_features["goodwill_concentration"] = all_stocks_features[src_col]
                break

    # Ensure industry column exists (prefer industry over sector)
    if "industry" not in all_stocks_features.columns and "sector" in all_stocks_features.columns:
        all_stocks_features["industry"] = all_stocks_features["sector"]

    print(f"✓ Backfill complete. Columns: {len(all_stocks_features.columns)}")

# %% [markdown]
# ---
# ## 3. Interactive Analytics Dashboard (Plotly)
# 
# ### 3.1 Interactive Momentum Analysis Dashboard
# 
# %%
# Generate interactive momentum dashboard using the feature_analytics module
fig = create_interactive_momentum_dashboard(all_stocks_features)
fig.show()

# %% [markdown]
# ### 3.2 Interactive Valuation Heatmap
# 
# %%
# Generate interactive valuation heatmap by industry
fig = create_interactive_valuation_heatmap(all_stocks_features)
fig.show()

# %% [markdown]
# ### 3.3 Leverage vs Liquidity Quadrant Analysis
# 
# %%
# Generate leverage vs liquidity quadrant analysis with distress coloring
fig = create_leverage_liquidity_quadrant(all_stocks_features)
fig.show()

# %% [markdown]
# ---
# ## 4. Probabilistic & Statistical Analytics
# 
# ### 4.1 Monte Carlo Price Target Simulation
# 
# %%
# Run Monte Carlo simulation for price targets
mc_results = monte_carlo_price_target_simulation(all_stocks_features, n_simulations=10000)

print("🎯 Monte Carlo Price Target Simulation Results:")
print(f"   Simulated {len(mc_results)} stocks with 10,000 iterations each\n")

# Display top opportunities by risk-reward ratio
top_opportunities = mc_results.nlargest(100, 'risk_reward_ratio')
display(top_opportunities[['ticker', 'name', 'industry', 'expected_upside_pct', 
                           'prob_positive_upside', 'var_5_pct', 'risk_reward_ratio']].round(2))

# %% [markdown]
# ### 4.2 Bayesian Earnings Beat Probability Model
# 
# %%
# Run Bayesian earnings beat probability model
bayesian_results = bayesian_earnings_beat_model(all_stocks_features)

# Refactored Bayesian Earnings Beat Probability Model Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if not bayesian_results.empty:
    print("📊 Bayesian Earnings Beat Probability Model - Enhanced Visualization\n")

    # Create a 2x2 subplot layout for comprehensive analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Posterior Beat Probability vs EPS Streak',
            'Beat Probability Distribution',
            'Model Confidence Distribution',
            'High Confidence Candidates by Industry'
        ],
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "histogram"}, {"type": "bar"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 1. Scatter plot: Posterior probability vs EPS streak (colored by confidence)
    fig.add_trace(
        go.Scatter(
            x=bayesian_results['eps_positive_streak'],
            y=bayesian_results['posterior_beat_prob'],
            mode='markers',
            marker=dict(
                size=8,
                color=bayesian_results['model_confidence'],
                colorscale='Viridis',
                opacity=0.6,
                colorbar=dict(title='Confidence', x=0.45, len=0.4, y=0.8)
            ),
            text=bayesian_results['ticker'],
            hovertemplate='<b>%{text}</b><br>EPS Streak: %{x}<br>P(Beat): %{y:.3f}<extra></extra>',
            name='Stocks'
        ),
        row=1, col=1
    )

    # Add trend line for reference
    streak_means = bayesian_results.groupby('eps_positive_streak')['posterior_beat_prob'].mean()
    fig.add_trace(
        go.Scatter(
            x=streak_means.index,
            y=streak_means.values,
            mode='lines+markers',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10, symbol='diamond'),
            name='Mean by Streak'
        ),
        row=1, col=1
    )

    # 2. Histogram: Distribution of posterior beat probabilities
    fig.add_trace(
        go.Histogram(
            x=bayesian_results['posterior_beat_prob'],
            nbinsx=30,
            marker_color='#00bc8c',
            opacity=0.7,
            name='Beat Prob'
        ),
        row=1, col=2
    )

    # Add vertical lines for key thresholds
    median_prob = bayesian_results['posterior_beat_prob'].median()
    fig.add_vline(x=0.5, line_dash="dash", line_color="white",
                  annotation_text="50%", row=1, col=2)
    fig.add_vline(x=median_prob, line_dash="dot", line_color="#f39c12",
                  annotation_text=f"Median: {median_prob:.2f}", row=1, col=2)

    # 3. Histogram: Model confidence distribution
    fig.add_trace(
        go.Histogram(
            x=bayesian_results['model_confidence'],
            nbinsx=25,
            marker_color='#3498db',
            opacity=0.7,
            name='Confidence'
        ),
        row=2, col=1
    )

    # 4. Bar chart: High confidence candidates by industry
    high_conf = bayesian_results[
        (bayesian_results['posterior_beat_prob'] > 0.7) &
        (bayesian_results['model_confidence'] > 0.5)
    ]
    if not high_conf.empty:
        industry_counts = high_conf['industry'].value_counts().head(15)
        fig.add_trace(
            go.Bar(
                x=industry_counts.values,
                y=industry_counts.index,
                orientation='h',
                marker_color='#9b59b6',
                name='High Conf Count'
            ),
            row=2, col=2
        )

    # Update layout
    fig.update_layout(
        height=800,
        title_text="🎯 Bayesian Earnings Beat Probability Model - Comprehensive Analysis",
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        template=PLOTLY_TEMPLATE
    )

    # Update axes labels
    fig.update_xaxes(title_text="EPS Positive Quarters (Last 5)", row=1, col=1)
    fig.update_yaxes(title_text="P(Beat Next Quarter)", row=1, col=1)
    fig.update_xaxes(title_text="Posterior Beat Probability", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Model Confidence", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_xaxes(title_text="Number of Stocks", row=2, col=2)
    fig.update_yaxes(title_text="Industry", row=2, col=2)



    # Summary statistics
    print("\n📈 Model Summary Statistics:")
    print(f"   Total stocks analyzed: {len(bayesian_results):,}")
    print(f"   Median beat probability: {bayesian_results['posterior_beat_prob'].median():.3f}")
    print(f"   High probability (>70%): {(bayesian_results['posterior_beat_prob'] > 0.7).sum():,} stocks")
    print(f"   High confidence (>50%): {(bayesian_results['model_confidence'] > 0.5).sum():,} stocks")

    # Top candidates table
    print("\n🏆 Top 10 Highest Probability Earnings Beat Candidates:")
    display(bayesian_results.nlargest(10, 'posterior_beat_prob')[
        ['ticker', 'name', 'industry', 'eps_positive_streak',
         'posterior_beat_prob', 'model_confidence', 'map_estimate']
    ].round(3))

    fig.show()

# %% [markdown]
# ### 4.3 Financial Distress Risk Distribution Analysis
# 
# %%
# Analyze distress risk distribution with tail risk metrics
fig = analyze_distress_distribution(all_stocks_features)
fig.show()

# %% [markdown]
# ---
# ## 5. Composite Quality Scoring
# 
# %%
# Generate composite quality scores
composite_df = create_composite_quality_score(all_stocks_features)

# Visualize distribution
fig = px.histogram(
    composite_df,
    x='composite_quality_score',
    color='quality_tier',
    nbins=50,
    title='📊 Composite Quality Score Distribution',
    labels={'composite_quality_score': 'Composite Quality Score (0-100)'},
    color_discrete_map={'Low': '#e74c3c', 'Below Avg': '#f39c12', 
                        'Above Avg': '#3498db', 'High': '#00bc8c'},
    height=450
)
fig.update_layout(template=PLOTLY_TEMPLATE)


# Top quality stocks
print("\n🏆 Top 15 Stocks by Composite Quality Score:")
display(composite_df[['ticker', 'name', 'industry', 'composite_quality_score', 
                      'quality_tier']].head(15).round(2))

# Quality tier summary
print("\n📊 Quality Tier Distribution:")
print(composite_df['quality_tier'].value_counts().to_string())

fig.show()

# %% [markdown]
# ---
# ## 6. Summary Dashboard
# 
# %%
# Generate KPI summary dashboard
fig = create_summary_dashboard(all_stocks_features)
fig.show()

# %% [markdown]
# ---
# ## 7. Static Visualizations (Matplotlib)
# 
# %%
# Ensure matplotlib is properly configured for Jupyter output
# %matplotlib inline
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.autolayout'] = True

# %%
# Visualization: Price Momentum Analysis (Fixed subplot layout)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

momentum_cols = ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m']
momentum_labels = ['1-Month', '3-Month', '6-Month']
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Histogram of momentum distributions
ax = axes[0, 0]
for col, label, color in zip(momentum_cols, momentum_labels, colors):
    if col in all_stocks_features.columns:
        data = all_stocks_features[col].dropna()
        if len(data) > 0:
            data_clipped = data.clip(-50, 100)
            ax.hist(data_clipped, bins=50, alpha=0.5, label=label, color=color, edgecolor='white')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Price Momentum (%)', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of Price Momentum', fontsize=13, fontweight='bold', pad=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Momentum by sector (boxplot for 3M)
ax = axes[0, 1]
if 'industry' in all_stocks_features.columns and 'price_momentum_3m' in all_stocks_features.columns:
    industries = sorted(all_stocks_features['industry'].dropna().unique())
    mom_data = [
        all_stocks_features[all_stocks_features['industry'] == s]['price_momentum_3m'].dropna().clip(-50, 100)
        for s in industries
    ]
    # Filter out empty series
    valid_data = [(d, s) for d, s in zip(mom_data, industries) if len(d) > 0]
    if valid_data:
        mom_data, industries = zip(*valid_data)
        bp = ax.boxplot(
            mom_data, 
            labels=[s[:12] + '...' if len(s) > 12 else s for s in industries], 
            patch_artist=True,
            showfliers=False
        )
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.6)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.set_ylabel('3-Month Momentum (%)', fontsize=12)
ax.set_title('3-Month Momentum by Industry', fontsize=13, fontweight='bold', pad=10)
ax.tick_params(axis='x', rotation=60, labelsize=8)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 52-week range position distribution
ax = axes[1, 0]
if 'range_52w_position' in all_stocks_features.columns:
    range_data = all_stocks_features['range_52w_position'].dropna()
    if len(range_data) > 0:
        ax.hist(range_data, bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
        ax.axvline(x=range_data.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'Median: {range_data.median():.2f}')
ax.set_xlabel('52-Week Range Position (0=Low, 1=High)', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of 52-Week Range Position', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Scatter: 1M vs 6M momentum
ax = axes[1, 1]
if 'price_momentum_1m' in all_stocks_features.columns and 'price_momentum_6m' in all_stocks_features.columns:
    valid_mask = all_stocks_features['price_momentum_1m'].notna() & all_stocks_features['price_momentum_6m'].notna()
    if valid_mask.sum() > 0:
        x_data = all_stocks_features.loc[valid_mask, 'price_momentum_1m'].clip(-50, 100)
        y_data = all_stocks_features.loc[valid_mask, 'price_momentum_6m'].clip(-50, 200)
        ax.scatter(x_data, y_data, alpha=0.3, s=12, c='#3498db')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('1-Month Momentum (%)', fontsize=12)
ax.set_ylabel('6-Month Momentum (%)', fontsize=12)
ax.set_title('Short-term vs Medium-term Momentum', fontsize=13, fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()
plt.close(fig)  # Explicitly close to prevent state bleeding
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 3: Valuation Metrics Heatmap by Industry
fig, ax = plt.subplots(figsize=(14, 15))

valuation_cols = ['p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio']
val_labels = ['P/E Ratio', 'P/B Ratio', 'EV/EBITDA', 'EV/Sales']

sectors_sorted = sorted(all_stocks_features['industry'].unique())
heatmap_data = []

for sector in sectors_sorted:
    sector_data = all_stocks_features[all_stocks_features['industry'] == sector]
    row = []
    for col in valuation_cols:
        median_val = sector_data[col].median()
        row.append(median_val if not np.isnan(median_val) else 0)
    heatmap_data.append(row)

heatmap_array = np.array(heatmap_data)
heatmap_normalized = (heatmap_array - heatmap_array.min(axis=0)) / (
            heatmap_array.max(axis=0) - heatmap_array.min(axis=0) + 1e-10)

im = ax.imshow(heatmap_normalized, cmap='RdYlGn_r', aspect='auto')

ax.set_xticks(np.arange(len(val_labels)))
ax.set_yticks(np.arange(len(sectors_sorted)))
ax.set_xticklabels(val_labels, fontsize=12, fontweight='bold')
ax.set_yticklabels(sectors_sorted, fontsize=11)

for i in range(len(sectors_sorted)):
    for j in range(len(val_labels)):
        text_color = 'white' if heatmap_normalized[i, j] > 0.6 or heatmap_normalized[i, j] < 0.4 else 'black'
        ax.text(j, i, f'{heatmap_array[i, j]:.1f}', ha='center', va='center', color=text_color, fontsize=11,
                fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Relative Valuation (Normalized)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title('Median Valuation Metrics by Industry\n(Green = Lower/Cheaper, Red = Higher/Expensive)',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Valuation Metric', fontsize=13, fontweight='bold')
ax.set_ylabel('Industry', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt

# Visualization 8: Financial Health Dashboard - Leverage & Liquidity
fig, axes = plt.subplots(2, 2, figsize=(25, 20))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# Debt-to-Equity Distribution
ax = axes[0, 0]
dte_data = all_stocks_features['debt_to_equity'].dropna().clip(0, 3)
ax.hist(dte_data, bins=50, color='#e74c3c', edgecolor='white', alpha=0.7)
ax.axvline(x=dte_data.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {dte_data.median():.2f}')
ax.axvline(x=1, color='green', linestyle=':', linewidth=2, alpha=0.8, label='D/E = 1 (Reference)')
ax.set_xlabel('Debt-to-Equity Ratio', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of Debt-to-Equity Ratio', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')

# Current Ratio Distribution
ax = axes[0, 1]
cr_data = all_stocks_features['current_ratio'].dropna().clip(0, 5)
ax.hist(cr_data, bins=50, color='#2ecc71', edgecolor='white', alpha=0.7)
ax.axvline(x=cr_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {cr_data.median():.2f}')
ax.axvline(x=1.5, color='blue', linestyle=':', linewidth=2, alpha=0.8, label='CR = 1.5 (Healthy)')
ax.set_xlabel('Current Ratio', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of Current Ratio (Liquidity)', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')

# Distress Risk Score by Industry
ax = axes[1, 0]
sectors = sorted(all_stocks_features['industry'].unique())
distress_by_sector = all_stocks_features.groupby('industry')['distress_risk_score'].median().sort_values(ascending=True)
colors = ['#e74c3c' if v < 40 else '#f39c12' if v < 60 else '#2ecc71' for v in distress_by_sector.values]

bars = ax.barh(distress_by_sector.index, distress_by_sector.values, color=colors, edgecolor='white', height=0.7)
ax.axvline(x=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Median Distress Risk Score (Higher = Safer)', fontsize=12)
ax.set_title('Financial Distress Risk by Industry', fontsize=13, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=10)
for bar, val in zip(bars, distress_by_sector.values):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f'{val:.0f}', va='center', fontsize=9, fontweight='bold')

# Leverage vs Liquidity Quadrant
ax = axes[1, 1]
valid = all_stocks_features['debt_to_equity'].notna() & all_stocks_features['current_ratio'].notna()
dte = all_stocks_features.loc[valid, 'debt_to_equity'].clip(0, 3)
cr = all_stocks_features.loc[valid, 'current_ratio'].clip(0, 5)

# Color by distress score
distress = all_stocks_features.loc[valid, 'distress_risk_score'].fillna(50)
scatter = ax.scatter(dte, cr, c=distress, cmap='RdYlGn', alpha=0.5, s=18)

ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Healthy Liquidity')
ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='High Leverage Threshold')
ax.set_xlabel('Debt-to-Equity Ratio', fontsize=12)
ax.set_ylabel('Current Ratio', fontsize=12)
ax.set_title('Leverage vs Liquidity Analysis\n(Color = Distress Risk Score)', fontsize=13, fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(fontsize=9, loc='upper right')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Distress Risk Score', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# Add quadrant labels
ax.text(0.3, 4.5, 'Low Risk', fontsize=11, color='green', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
ax.text(2.5, 0.5, 'High Risk', fontsize=11, color='red', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 7: Analyst Sentiment Analysis
fig, axes = plt.subplots(2, 2, figsize=(30, 25))
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# Sentiment distribution stacked
ax = axes[0, 0]
sentiment_cols = ['analyst_bullish_pct', 'analyst_neutral_pct', 'analyst_bearish_pct']
valid_sent = all_stocks_features[sentiment_cols].dropna()

ax.hist([valid_sent['analyst_bullish_pct'], valid_sent['analyst_neutral_pct'], valid_sent['analyst_bearish_pct']],
        bins=20, stacked=False, alpha=0.6, label=['Bullish', 'Neutral', 'Bearish'],
        color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='white')
ax.set_xlabel('Analyst Sentiment (%)', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of Analyst Sentiment', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, linestyle='--')

# Upside Potential Distribution
ax = axes[0, 1]
upside = all_stocks_features['upside_potential'].dropna().clip(-50, 100)
ax.hist(upside, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax.axvline(x=upside.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {upside.median():.1f}%')
ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax.set_xlabel('Upside Potential (%)', fontsize=12)
ax.set_ylabel('Number of Stocks', fontsize=12)
ax.set_title('Distribution of Analyst Upside Potential', fontsize=13, fontweight='bold', pad=10)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--')

# Analyst Rating by Industry
ax = axes[1, 0]
sectors = sorted(all_stocks_features['industry'].unique())
rating_by_industry = all_stocks_features.groupby('industry')['analyst_rating_normalized'].median().sort_values()
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(rating_by_industry)))

bars = ax.barh(rating_by_industry.index, rating_by_industry.values, color=colors, edgecolor='white', height=0.7)
ax.set_xlabel('Median Analyst Rating (Normalized)', fontsize=12)
ax.set_title('Analyst Rating by Industry', fontsize=13, fontweight='bold', pad=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=11)
for bar, val in zip(bars, rating_by_industry.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}', va='center', fontsize=10, fontweight='bold')

# Bullish vs Upside scatter
ax = axes[1, 1]
valid = all_stocks_features['analyst_bullish_pct'].notna() & all_stocks_features['upside_potential'].notna()
bullish = all_stocks_features.loc[valid, 'analyst_bullish_pct']
upside = all_stocks_features.loc[valid, 'upside_potential'].clip(-50, 150)

scatter = ax.scatter(bullish, upside, alpha=0.4, s=12, c=bullish, cmap='RdYlGn')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Bullish Analyst %', fontsize=12)
ax.set_ylabel('Upside Potential (%)', fontsize=12)
ax.set_title('Bullish Sentiment vs Upside Potential', fontsize=13, fontweight='bold', pad=10)
ax.grid(alpha=0.3, linestyle='--')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
cbar.set_label('Bullish %', fontsize=11)
cbar.ax.tick_params(labelsize=9)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 4: Price Momentum Comparison (1M, 3M, 6M)
fig, axes = plt.subplots(1, 4, figsize=(30, 5))

momentum_cols = ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_1y']
titles = ['1-Month Momentum', '3-Month Momentum', '6-Month Momentum', '1-Year Momentum']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for ax, col, title, color in zip(axes, momentum_cols, titles, colors):
    data = all_stocks_features[col].dropna()
    data_clipped = np.clip(data, -50, 100)

    ax.hist(data_clipped, bins=50, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)

    median_val = data.median()
    mean_val = data.mean()
    ax.axvline(median_val, color='black', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}%')
    ax.axvline(mean_val, color='orange', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.1f}%')
    ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Momentum (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Price Momentum Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 7: Upside Potential Analysis by Industry
fig, ax = plt.subplots(figsize=(25, 15))

upside_data = all_stocks_features[['industry', 'last_price', 'price_target']].dropna()
upside_data = upside_data[upside_data['last_price'] > 0]
upside_pct = ((upside_data['price_target'] - upside_data['last_price']) / upside_data['last_price']) * 100
upside_data = upside_data.copy()
upside_data['upside_pct'] = upside_pct

upside_clipped = np.clip(upside_data['upside_pct'], -50, 100)
upside_data['upside_clipped'] = upside_clipped

sector_stats = upside_data.groupby('industry')['upside_clipped'].agg(['mean', 'median', 'std']).sort_values('median',
                                                                                                            ascending=True)

x = np.arange(len(sector_stats))
width = 0.35

bars1 = ax.bar(x - width / 2, sector_stats['mean'], width, label='Mean Upside', color='#3498db', alpha=0.8,
               edgecolor='white')
bars2 = ax.bar(x + width / 2, sector_stats['median'], width, label='Median Upside', color='#2ecc71', alpha=0.8,
               edgecolor='white')

ax.errorbar(x - width / 2, sector_stats['mean'], yerr=sector_stats['std'] / 2, fmt='none', color='black', capsize=4,
            alpha=0.5)

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(sector_stats.index, rotation=50, ha='right', fontsize=11)
ax.set_ylabel('Upside Potential (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Industry', fontsize=13, fontweight='bold')
ax.set_title('Price Target Upside Potential by Industry\n(Capped at -50% to 100%)', fontsize=15, fontweight='bold',
             pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.tick_params(axis='y', labelsize=11)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 8: Upside Potential vs P/E Ratio Scatter with Industry Coloring
fig, ax = plt.subplots(figsize=(25, 20))

df_scatter = all_stocks_features[['p_e_ratio', 'upside_potential', 'industry', 'market_cap']].dropna()
df_scatter = df_scatter[(df_scatter['p_e_ratio'] > 0) & (df_scatter['p_e_ratio'] < 100)]
df_scatter = df_scatter[(df_scatter['upside_potential'] > -50) & (df_scatter['upside_potential'] < 150)]

sectors = df_scatter['industry'].unique()
colors_map = {s: plt.cm.tab10(i / len(sectors)) for i, s in enumerate(sectors)}

for sector in sectors:
    mask = df_scatter['industry'] == sector
    subset = df_scatter[mask]
    sizes = np.clip(subset['market_cap'] / 1000, 25, 350)
    ax.scatter(subset['p_e_ratio'], subset['upside_potential'],
               s=sizes, alpha=0.55, label=sector, color=colors_map[sector], edgecolors='white', linewidth=0.4)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax.axhline(y=25, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7, label='25% Upside')
ax.axvline(x=25, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7, label='P/E = 25')

undervalued = ((df_scatter['upside_potential'] > 25) & (df_scatter['p_e_ratio'] < 25)).sum()
overvalued = ((df_scatter['upside_potential'] < 0) & (df_scatter['p_e_ratio'] > 40)).sum()
stats_text = f'Potentially Undervalued\n(Upside>25%, P/E<25): {undervalued:,}\n\nPotentially Overvalued\n(Upside<0%, P/E>40): {overvalued:,}'
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel('P/E Ratio', fontsize=13, fontweight='bold')
ax.set_ylabel('Upside Potential (%)', fontsize=13, fontweight='bold')
ax.set_title('Upside Potential vs P/E Ratio by Industry\n(Bubble Size = Market Cap)', fontsize=15, fontweight='bold',
             pad=15)
ax.legend(loc='lower right', fontsize=10, ncol=2, framealpha=0.9, title='Industry', title_fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 3: Price Momentum Heatmap by Industry
fig, ax = plt.subplots(figsize=(25, 20))

momentum_cols = ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_1y']
sectors_list = all_stocks_features['industry'].dropna().unique()

momentum_matrix = []
for sector in sectors_list:
    sector_data = all_stocks_features[all_stocks_features['industry'] == sector][momentum_cols].median()
    momentum_matrix.append(sector_data.values)

momentum_matrix = np.array(momentum_matrix)

im = ax.imshow(momentum_matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=40)

ax.set_xticks(np.arange(len(momentum_cols)))
ax.set_yticks(np.arange(len(sectors_list)))
ax.set_xticklabels(['1 Month', '3 Months', '6 Months', '1 Year'], fontsize=12, fontweight='bold')
ax.set_yticklabels(sectors_list, fontsize=12)

for i in range(len(sectors_list)):
    for j in range(len(momentum_cols)):
        val = momentum_matrix[i, j]
        text_color = 'white' if abs(val) > 15 else 'black'
        ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=text_color, fontsize=11, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Median Price Momentum (%)', fontsize=12)
cbar.ax.tick_params(labelsize=10)

ax.set_title('Price Momentum by Industry and Time Period', fontsize=15, fontweight='bold', pad=15)
ax.set_xlabel('Time Period', fontsize=13, fontweight='bold')
ax.set_ylabel('Industry', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np


def format_pct(value):
    return f"{value:.1f}%"


def plot_stacked_sentiment_bars(ax, x, bullish_vals, neutral_vals, bearish_vals, width):
    ax.bar(x, bullish_vals, width, label='Bullish (Buy + Strong Buy)', color='#2ecc71', edgecolor='black')
    ax.bar(x, neutral_vals, width, bottom=bullish_vals, label='Neutral (Hold)', color='#95a5a6', edgecolor='black')
    ax.bar(
        x,
        bearish_vals,
        width,
        bottom=[b + n for b, n in zip(bullish_vals, neutral_vals)],
        label='Bearish (Sell + Strong Sell)',
        color='#e74c3c',
        edgecolor='black',
    )

    for i, (b, n, be) in enumerate(zip(bullish_vals, neutral_vals, bearish_vals)):
        ax.text(i, b / 2, format_pct(b), ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        ax.text(i, b + n / 2, format_pct(n), ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        ax.text(i, b + n + be / 2, format_pct(be), ha='center', va='center', fontweight='bold', fontsize=12,
                color='white')


# Visualization: Complete Analyst Sentiment Breakdown (Bullish + Neutral + Bearish = 100%)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
plt.subplots_adjust(wspace=0.25)

# Extract sentiment data
bullish = all_stocks_features['analyst_bullish_pct'].dropna()
neutral = all_stocks_features['analyst_neutral_pct'].dropna()
bearish = all_stocks_features['analyst_bearish_pct'].dropna()

# Left plot: Stacked histogram showing distribution
bins = np.linspace(0, 100, 21)

ax1.hist(bullish, bins=bins, alpha=0.7, label='Bullish %', color='#2ecc71', edgecolor='black', linewidth=0.8)
ax1.hist(neutral, bins=bins, alpha=0.7, label='Neutral %', color='#95a5a6', edgecolor='black', linewidth=0.8)
ax1.hist(bearish, bins=bins, alpha=0.7, label='Bearish %', color='#e74c3c', edgecolor='black', linewidth=0.8)

# Add median lines
bullish_median = bullish.median()
neutral_median = neutral.median()
bearish_median = bearish.median()

ax1.axvline(bullish_median, color='#27ae60', linestyle='--', linewidth=2.5,
            label=f'Bullish Median: {bullish_median:.1f}%')
ax1.axvline(neutral_median, color='#7f8c8d', linestyle='--', linewidth=2.5,
            label=f'Neutral Median: {neutral_median:.1f}%')
ax1.axvline(bearish_median, color='#c0392b', linestyle='--', linewidth=2.5,
            label=f'Bearish Median: {bearish_median:.1f}%')

# Statistics text box
strongly_bullish = (bullish >= 75).sum()
mostly_neutral = ((neutral >= 30) & (neutral <= 50)).sum()
strongly_bearish = (bearish >= 25).sum()
stats_text = f'Strongly Bullish (≥75%): {strongly_bullish:,}\nMostly Neutral (30-50%): {mostly_neutral:,}\nStrongly Bearish (≥25%): {strongly_bearish:,}'
ax1.text(0.98, 0.95, stats_text, transform=ax1.transAxes, ha='right', va='top', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax1.set_xlabel('Analyst Sentiment (%)', fontsize=13)
ax1.set_ylabel('Number of Stocks', fontsize=13)
ax1.set_title('Distribution of Analyst Sentiment Components', fontsize=14, fontweight='bold', pad=10)
ax1.legend(loc='upper center', fontsize=10, ncol=2)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)
ax1.tick_params(axis='both', labelsize=11)

# Right plot: Average sentiment breakdown (stacked bar showing they sum to 100%)
# Get stocks with complete sentiment data
mask = bullish.index.intersection(neutral.index).intersection(bearish.index)
avg_bullish = all_stocks_features.loc[mask, 'analyst_bullish_pct'].mean()
avg_neutral = all_stocks_features.loc[mask, 'analyst_neutral_pct'].mean()
avg_bearish = all_stocks_features.loc[mask, 'analyst_bearish_pct'].mean()

categories = ['Average\nSentiment', 'Median\nSentiment']
bullish_vals = [avg_bullish, bullish_median]
neutral_vals = [avg_neutral, neutral_median]
bearish_vals = [avg_bearish, bearish_median]

x = np.arange(len(categories))
width = 0.55

# Stacked bar chart
bars1 = ax2.bar(x, bullish_vals, width, label=f'Bullish (Buy + Strong Buy)', color='#2ecc71', edgecolor='black')
bars2 = ax2.bar(x, neutral_vals, width, bottom=bullish_vals, label=f'Neutral (Hold)', color='#95a5a6',
                edgecolor='black')
bars3 = ax2.bar(x, bearish_vals, width, bottom=[b + n for b, n in zip(bullish_vals, neutral_vals)],
                label=f'Bearish (Sell + Strong Sell)', color='#e74c3c', edgecolor='black')

# Add value labels on bars
for i, (b, n, be) in enumerate(zip(bullish_vals, neutral_vals, bearish_vals)):
    ax2.text(i, b / 2, f'{b:.1f}%', ha='center', va='center', fontweight='bold', fontsize=12, color='white')
    ax2.text(i, b + n / 2, f'{n:.1f}%', ha='center', va='center', fontweight='bold', fontsize=12, color='white')
    ax2.text(i, b + n + be / 2, f'{be:.1f}%', ha='center', va='center', fontweight='bold', fontsize=12, color='white')

ax2.set_ylabel('Percentage (%)', fontsize=13)
ax2.set_title('Sentiment Breakdown (Sums to 100%)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=12)
ax2.legend(loc='upper right', fontsize=11)
ax2.set_ylim(0, 108)
ax2.axhline(y=100, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
ax2.tick_params(axis='y', labelsize=11)

# Add total verification text
total_avg = avg_bullish + avg_neutral + avg_bearish
ax2.text(0.5, -0.1, f'✓ Total: {total_avg:.1f}% (Bullish + Neutral + Bearish)',
         transform=ax2.transAxes, ha='center', fontsize=11, style='italic')

plt.suptitle('Distribution of Analyst Sentiment: Bullish vs Neutral vs Bearish', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 5: Valuation Ratios Comparison Across Industrys
fig, axes = plt.subplots(2, 2, figsize=(25, 20))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

valuation_metrics = [
    ('p_e_ratio', 'P/E Ratio', (0, 80)),
    ('p_b_ratio', 'P/B Ratio', (0, 15)),
    ('ev_ebitda_ratio', 'EV/EBITDA', (0, 40)),
    ('ev_sales_ratio', 'EV/Sales', (0, 15))
]

sectors = all_stocks_features['industry'].dropna().unique()
x_pos = np.arange(len(sectors))
width = 0.65

for idx, (col, title, ylim) in enumerate(valuation_metrics):
    ax = axes[idx // 2, idx % 2]

    medians = all_stocks_features.groupby('industry')[col].median().reindex(sectors)
    q25 = all_stocks_features.groupby('industry')[col].quantile(0.25).reindex(sectors)
    q75 = all_stocks_features.groupby('industry')[col].quantile(0.75).reindex(sectors)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sectors)))
    bars = ax.bar(x_pos, medians.values, width, color=colors, alpha=0.85, edgecolor='white', linewidth=0.8)

    ax.errorbar(x_pos, medians.values,
                yerr=[medians.values - q25.values, q75.values - medians.values],
                fmt='none', color='black', capsize=4, capthick=1.5, linewidth=1.5)

    ax.set_ylabel(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sectors, rotation=55, ha='right', fontsize=10)
    ax.set_ylim(ylim)
    ax.set_title(f'{title} by Industry (Median with IQR)', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

plt.suptitle('Valuation Ratios Comparison Across Industries', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# %%
import matplotlib.pyplot as plt
import numpy as np

# Visualization 4: Piotroski F-Score Distribution
fig, ax = plt.subplots(figsize=(14, 7))

fscore_data = all_stocks_features['piotroski_f_score'].dropna()
bins = np.arange(-0.5, 10.5, 1)
counts, edges, patches = ax.hist(fscore_data, bins=bins, edgecolor='black', linewidth=1.2, alpha=0.85)

colors_map = plt.cm.RdYlGn(np.linspace(0.1, 0.9, 10))
for i, patch in enumerate(patches):
    patch.set_facecolor(colors_map[i])

ax.axvline(x=7, color='#2ecc71', linestyle='--', linewidth=2.5, label='Strong (≥7)')
ax.axvline(x=4, color='#e74c3c', linestyle='--', linewidth=2.5, label='Weak (≤3)')

for i, count in enumerate(counts):
    if count > 0:
        ax.text(i, count + max(counts) * 0.02, f'{int(count)}', ha='center', va='bottom', fontsize=10,
                fontweight='bold')

strong = (fscore_data >= 7).sum()
weak = (fscore_data <= 3).sum()
ax.text(0.98, 0.95, f'Strong (7-9): {strong:,} stocks\nWeak (0-3): {weak:,} stocks',
        transform=ax.transAxes, ha='right', va='top', fontsize=12,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

ax.set_xlabel('Piotroski F-Score', fontsize=13, fontweight='bold')
ax.set_ylabel('Number of Stocks', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Piotroski F-Score (Financial Health Indicator)', fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(range(10))
ax.tick_params(axis='both', labelsize=11)
ax.legend(loc='upper left', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
# %%
print(f"✓ Dataset loaded: {all_stocks_features.shape[0]:,} stocks, {all_stocks_features.shape[1]} features")

# %% [markdown]
# ---
# ## 3. Feature Category Definitions (Updated for 487 columns)
# 
# %%
# Cell 3: Feature Category Definitions (UPDATED for mv_all_stock_features v2)
# Aligned with SQL Feature Registry from mv_feature_registry.sql

FEATURE_CATEGORIES = {
    'Valuation Ratios': [
        'p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio', 'dividend_yield',
        'peg_ratio', 'ev_sales_trend_1y', 'ev_ebitda_momentum', 'p_e_momentum_yoy',
        'p_e_momentum_qoq', 'ev_sales_vs_3y_avg', 'ev_ebitda_vs_3y_avg', 'p_e_vs_3y_avg',
        'ev_sales_forward_discount', 'ev_ebitda_forward_discount', 'p_e_forward_discount',
        'p_b_vs_5y_avg', 'ev_sales_qoq_1q', 'p_e_vs_5y_avg_ext', 'p_b_momentum_yoy',
        'forward_pe_premium', 'price_to_tangible_book', 'tangible_equity_ratio',
        'tangible_book_value', 'tangible_book_per_share'
    ],
    'Momentum & Technical': [
        'price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_1y',
        'price_momentum_5d', 'price_momentum_3y', 'price_momentum_5y',
        'ema_crossover_20_50', 'ema_crossover_50_250', 'price_vs_ema_20d',
        'price_vs_ema_250d', 'price_vs_ema_100d', 'pct_off_52w_high', 'pct_above_52w_low',
        'range_52w_position', 'beta_momentum', 'volatility_regime', 'ema_slope_20d',
        'ema_trend_consistency', 'near_52w_high_flag', 'near_52w_low_flag',
        'volume_momentum_score', 'breakout_signal', 'high_volume_flag', 'low_volume_flag',
        'volatility_compression', 'volatility_term_structure', 'long_term_trend_score',
        'multi_year_high_flag', 'secular_trend_flag'
    ],
    'Profitability': [
        'roe', 'roa', 'gross_margin_pct', 'operating_margin_pct', 'net_margin_pct',
        'ebitda_margin_pct', 'ebitda_margin_ltm', 'ebit_margin_ltm', 'roic',
        'rnd_intensity', 'equity_multiplier', 'gross_margin_trend_yoy',
        'operating_margin_trend', 'net_margin_trend_yoy', 'ebitda_margin_trend',
        'margin_expansion_flag', 'gp_margin_fq', 'gp_margin_trend', 'gp_margin_expansion_temp'
    ],
    'Quality & Risk': [
        'has_goodwill_impairment', 'has_asset_writedown', 'has_restructuring',
        'goodwill_to_assets_pct', 'intangible_intensity', 'exceptional_items_to_ebitda',
        'altman_z_score', 'altman_z_trend', 'current_ratio', 'quick_ratio',
        'distress_risk_score', 'liquidity_stress_score', 'working_capital_trend',
        'cash_runway_months', 'accumulated_deficit_flag', 'adequate_cash_buffer',
        'accounting_quality_score', 'goodwill_change_rate', 'restructuring_intensity',
        'exceptional_items_frequency', 'merger_impact_ratio', 'non_operating_income_share',
        'asset_sale_boost', 'beta_stability_score', 'high_beta_flag', 'low_beta_flag'
    ],
    'Leverage & Liquidity': [
        'debt_to_equity', 'debt_to_assets', 'equity_ratio', 'interest_coverage',
        'interest_coverage_ratio', 'cash_ratio', 'working_capital_ratio',
        'asset_turnover', 'inventory_turnover', 'receivables_days', 'working_capital_turns',
        'cash_to_assets_pct', 'cash_change_qoq', 'cash_vs_5y_avg', 'inventory_change_yoy',
        'inventory_vs_5y_avg', 'working_capital_vs_5y_avg', 'retained_earnings_vs_5y',
        'balance_sheet_strength', 'debt_maturity_risk', 'intangibles_growth_flag',
        'asset_quality_score', 'debt_deleveraging', 'debt_to_equity_trend'
    ],
    'Analyst Sentiment': [
        'analyst_bullish_pct', 'analyst_bearish_pct', 'upside_potential',
        'price_target_spread_pct', 'price_target_revision_1m', 'price_target_revision_3m',
        'eps_revision_momentum', 'analyst_rating_normalized', 'analyst_coverage_quality',
        'pt_momentum_1w', 'pt_momentum_1m', 'pt_momentum_3m', 'pt_momentum_6m',
        'pt_momentum_1y', 'analyst_coverage_change_1m', 'analyst_coverage_change_3m',
        'analyst_coverage_change_1y'
    ],
    'Earnings Quality': [
        'eps_surprise_pct', 'revenue_surprise_pct', 'eps_adjustment_ratio',
        'gaap_adj_eps_gap_pct', 'ebitda_adjustment_ratio', 'eps_quarterly_trend',
        'eps_yoy_growth', 'eps_qoq_growth', 'eps_yoy_quarterly', 'eps_positive_streak',
        'eps_cagr_3y', 'eps_cagr_5y', 'eps_improvement_count', 'eps_trajectory_score',
        'eps_adjustment_spread_ltm', 'eps_adjustment_pct', 'ni_adjustment_ratio',
        'net_income_adjustment_pct', 'ebitda_adjustment_pct_fy', 'earnings_quality_score',
        'earnings_quality_warning', 'forward_eps_gaap_adj_spread',
        'gaap_revision_momentum', 'gaap_revision_1m', 'gaap_revision_3m',
        'gaap_revision_6m', 'gaap_revision_1y', 'gaap_vs_norm_revision_spread',
        'gaap_revision_acceleration', 'gaap_positive_revision_flag',
        'earnings_quality_composite_comp'
    ],
    'Growth Metrics': [
        'revenue_growth_yoy', 'revenue_growth_yoy_temp', 'ebitda_growth_yoy',
        'ebitda_growth_yoy_comp', 'operating_income_growth', 'fcf_growth', 'fcf_growth_yoy',
        'revenue_cagr_5y', 'forward_revenue_growth', 'revenue_vs_5y_avg',
        'revenue_vs_5y_avg_qtr', 'revenue_acceleration', 'ebit_growth_yoy',
        'ebit_cagr_3y', 'ebitda_cagr_3y', 'net_income_growth_yoy', 'gp_yoy_growth'
    ],
    'Revenue Forecasting': [
        'revenue_est_spread', 'revenue_beat_potential', 'revenue_est_revision_trend',
        'ebitda_est_vs_actual', 'forward_revenue_multiple', 'revenue_estimate_count',
        'revenue_guidance_gap', 'consensus_revenue_growth', 'forward_ebitda_margin',
        'estimate_confidence_score', 'revenue_est_avg_fy1e', 'revenue_est_med_fy1e',
        'revenue_est_avg_ntm', 'revenue_est_med_ntm', 'revenue_avg_med_diff_pct',
        'revenue_consensus_strength', 'revenue_vs_current'
    ],
    'Dividend Features': [
        'dividend_streak', 'dividend_yield_ltm', 'dividend_yield_ntm',
        'dividend_payout_ratio', 'fcf_dividend_coverage', 'buyback_yield',
        'total_shareholder_yield', 'dividend_growth_expectation', 'days_since_ex_date',
        'days_to_payment', 'dividend_announced_flag', 'ex_date_approaching_flag',
        'dividend_frequency_score', 'dividend_consistency', 'dividend_yield_vs_5y_avg',
        'div_yield_ind', 'div_yield_1fy_ind', 'div_yield_5y_avg', 'div_yield_vs_5y_avg',
        'div_yield_growth_expected', 'high_yield_flag', 'sustainable_dividend_flag'
    ],
    'Cash Flow': [
        'cfo_to_net_income', 'fcf_to_net_income', 'fcf_margin', 'fcf_margin_pct',
        'cfo_growth_yoy', 'cfo_growth_yoy_comp', 'fcf_positive_ratio', 'fcf_positive_years',
        'fcf_always_positive', 'acquisition_intensity', 'self_funding_ratio',
        'self_funding_flag', 'capex_vs_5y_avg', 'underinvestment_flag',
        'cfo_share_of_cf', 'cfi_share_of_cf', 'cff_share_of_cf',
        'fcf_4q_improvement', 'cash_flow_quality_score', 'cfo_quarterly_trend',
        'cfi_quarterly_trend', 'cff_quarterly_trend', 'fcf_quarterly_trend',
        'cfo_positive_quarters', 'cfo_positive_years', 'cfi_negative_quarters',
        'cash_burn_rate', 'financing_dependency', 'fcf_yield',
        # NEW: CapEx temporal analysis
        'capex_yoy_growth', 'capex_qoq_growth', 'capex_3y_trend', 'capex_volatility',
        'capex_acceleration', 'capex_cut_flag', 'overinvestment_flag',
        # NEW: Cash Acquisitions temporal analysis
        'acquisitions_yoy_growth', 'acquisitions_vs_5y_avg', 'acquisitions_ltm_total',
        'ma_intensity_score', 'serial_acquirer_flag', 'acquisition_pause_flag',
        # NEW: Combined investment metrics
        'total_investment_to_cfo', 'organic_vs_inorganic', 'investment_efficiency'
    ],
    'Temporal Patterns': [
        'fiscal_quarter', 'fiscal_month', 'fiscal_year', 'days_to_earnings',
        'earnings_report_recency', 'reporting_lag', 'fiscal_year_progress',
        'days_since_last_report', 'days_to_fy_end', 'is_quarter_end_month',
        'is_fy_end_month', 'earnings_season_flag', 'pre_earnings_window',
        'post_earnings_window', 'reporting_freshness_score'
    ],
    'Employee Productivity': [
        'revenue_per_employee', 'profit_per_employee', 'ebitda_per_employee',
        'assets_per_employee', 'fte_growth_1y_pct', 'fte_growth_2y_pct',
        'fte_growth_3y_pct', 'workforce_stability', 'layoff_risk_flag',
        'rapid_hiring_flag', 'sustainable_growth_flag'
    ],
    'Cost Structure': [
        'cogs_to_revenue', 'opex_to_revenue', 'sga_to_revenue', 'rnd_to_revenue',
        'interest_to_revenue', 'interest_income_to_revenue', 'interest_expense_to_revenue',
        # NEW: Marketing & SG&A efficiency metrics
        'marketing_to_revenue', 'marketing_trend_yoy', 'marketing_vs_5y_avg',
        'sga_vs_5y_avg', 'sga_efficiency_trend'
    ],
    'EPS Continuing Operations': [
        # NEW: Basic EPS from continuing operations (excludes discontinued ops)
        'eps_cont_ltm', 'eps_cont_fq', 'eps_cont_fy',
        'eps_cont_1fy', 'eps_cont_2fy', 'eps_cont_3fy', 'eps_cont_4fy',
        'eps_cont_qoq_growth', 'eps_cont_yoy_growth', 'eps_cont_vs_total_eps',
        'discontinued_ops_impact', 'eps_cont_positive_streak', 'eps_cont_trajectory_score',
        'eps_cont_cagr_3y', 'core_earnings_stability'
    ],
    'R&D Investment': [
        # NEW: R&D temporal features and intensity metrics
        'rnd_fq', 'rnd_fy', 'rnd_1fy', 'rnd_2fy', 'rnd_3fy', 'rnd_4fy',
        'rnd_intensity_ltm', 'rnd_intensity_fy', 'rnd_intensity_trend',
        'rnd_qoq_growth', 'rnd_yoy_growth', 'rnd_cagr_3y',
        'rnd_per_employee', 'rnd_to_gross_profit', 'rnd_roi_proxy',
        'rnd_increasing_flag', 'rnd_cut_flag', 'high_rnd_intensity_flag'
    ],
    'Inventory Temporal': [
        # NEW: Full inventory historical coverage
        'inventory_1fq', 'inventory_2fq', 'inventory_3fq', 'inventory_4fq',
        'inventory_1fy', 'inventory_2fy', 'inventory_3fy', 'inventory_4fy',
        'inventory_qoq_change', 'inventory_yoy_change', 'inventory_4q_trend',
        'inventory_vs_5y_avg', 'inventory_days', 'inventory_turnover_mv',
        'inventory_to_revenue', 'inventory_to_assets',
        'inventory_buildup_flag', 'inventory_reduction_flag', 'inventory_volatility'
    ],
    'Goodwill & M&A': [
        # NEW: Goodwill temporal features for M&A tracking
        'goodwill_1fq', 'goodwill_2fq', 'goodwill_3fq', 'goodwill_4fq',
        'goodwill_1fy', 'goodwill_2fy', 'goodwill_3fy', 'goodwill_4fy',
        'goodwill_qoq_change', 'goodwill_yoy_change', 'goodwill_3y_growth',
        'goodwill_vs_5y_avg', 'goodwill_concentration', 'goodwill_accumulation_rate',
        'goodwill_to_assets_trend', 'recent_acquisition_flag', 'impairment_risk_score'
    ],
    'Tangible Book Value': [
        # NEW/ENHANCED: Native TBV columns from schema
        'tangible_book_value_fy', 'tangible_book_value_ltm', 'tangible_book_per_share',
        'price_to_tangible_book', 'tangible_equity_ratio', 'tbv_yoy_growth',
        'tbv_vs_calculated', 'tangible_asset_quality'
    ],
    'Balance Sheet Dynamics': [
        'working_capital_ltm', 'working_capital_fq', 'working_capital_fy',
        'wc_to_revenue', 'wc_to_assets', 'wc_change_qoq', 'wc_change_yoy',
        'wc_qoq_change', 'wc_yoy_change', 'wc_4q_trend', 'wc_vs_5y_avg_temp',
        'wc_positive_quarters', 'wc_improving_flag_temp', 'wc_volatility',
        'days_working_capital', 'negative_wc_flag', 'wc_improvement_flag',
        'debt_qoq_change', 'debt_yoy_change', 'debt_4q_trend', 'debt_3y_cagr',
        'assets_qoq_growth', 'assets_yoy_growth', 'assets_3y_cagr',
        'asset_growth_accel', 'asset_base_stable'
    ],
    'Unusual Items': [
        'other_unusual_items_ltm', 'total_unusual_items', 'unusual_items_to_revenue',
        'unusual_items_to_ebitda', 'has_unusual_items_flag', 'goodwill_to_equity',
        'intangibles_to_equity'
    ],
    'Composite Scores': [
        'piotroski_f_score', 'dilution_score', 'accounting_quality_score',
        'asset_quality_score', 'balance_sheet_strength', 'cash_flow_quality_score',
        'earnings_quality_score', 'reporting_freshness_score', 'beta_stability_score',
        'estimate_confidence_score', 'revenue_consensus_strength'
    ]
}

# Filter to features that exist in the dataframe
for category, features in FEATURE_CATEGORIES.items():
    FEATURE_CATEGORIES[category] = [f for f in features if f in all_stocks_features.columns]

print("Feature Categories Summary:")
total_features = 0
for cat, feats in sorted(FEATURE_CATEGORIES.items(), key=lambda x: -len(x[1])):
    print(f"  {cat}: {len(feats)} features")
    total_features += len(feats)
print(f"\nTotal categorized features: {total_features}")
print(f"Total columns in dataset: {len(all_stocks_features.columns)}")

# %%
# Cell: Validate Feature Categories Against Registry (optional database check)
# This ensures notebook feature categories align with the calculated_features_registry

def validate_feature_alignment(df: pd.DataFrame, categories: dict) -> dict:
    """
    Check which features in categories exist in the DataFrame.
    
    Returns dict with 'available', 'missing', and 'coverage_pct' per category.
    """
    # Store original categories for validation (before filtering)
    original_categories = {
        'Price Momentum': ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m', 'price_momentum_12m', 'range_52w_position'],
        'Valuation': ['p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio', 'peg_ratio'],
        'Profitability': ['gross_margin', 'operating_margin', 'net_margin', 'roa', 'roe', 'roic'],
        'Analyst': ['analyst_bullish_pct', 'analyst_bearish_pct', 'analyst_neutral_pct', 'analyst_count'],
        'Financial Health': ['current_ratio', 'quick_ratio', 'debt_to_equity', 'interest_coverage'],
    }
    
    validation_results = {}
    for category, features in original_categories.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        coverage = len(available) / len(features) * 100 if features else 0
        validation_results[category] = {
            'available_count': len(available),
            'missing_count': len(missing),
            'coverage_pct': coverage,
            'missing_features': missing[:5]  # Show first 5 missing
        }
    return validation_results

# Run validation
validation = validate_feature_alignment(all_stocks_features, FEATURE_CATEGORIES)
low_coverage = {k: v for k, v in validation.items() if v['coverage_pct'] < 80}

if low_coverage:
    print("⚠️ Categories with <80% feature coverage:")
    for cat, info in low_coverage.items():
        print(f"  {cat}: {info['coverage_pct']:.1f}% - Missing: {info['missing_features']}")
else:
    print("✓ All feature categories have ≥80% coverage")

# %% [markdown]
# ---
# ## 4. Dataset Overview & Coverage Analysis
# 
# %%
def compute_metric_statistics(series: pd.Series) -> dict:
    """Compute standard statistics for a numeric series."""
    data = pd.to_numeric(series, errors="coerce").dropna()
    if len(data) == 0:
        return None
    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75)),
        "positive_pct": float((data > 0).sum() / len(data) * 100),
        "missing_pct": float((series.isna().sum() / len(series)) * 100),
    }


# Compute overall statistics
numeric_cols = all_stocks_features.select_dtypes(include=[np.number]).columns
print(f"📊 Dataset Overview")
print(f"{'=' * 50}")
print(f"Total Stocks: {len(all_stocks_features):,}")
print(f"Total Columns: {len(all_stocks_features.columns)}")
print(f"Numeric Features: {len(numeric_cols)}")
print(f"Industrys: {all_stocks_features['industry'].nunique()}")
print(f"Regions: {all_stocks_features['region'].nunique()}")
print(f"Countries: {all_stocks_features['country'].nunique()}")
print(f"\n📈 Key Statistics:")
print(f"  Avg Market Cap: ${all_stocks_features['market_cap'].mean() / 1e9:.2f}B")
print(f"  Median P/E Ratio: {all_stocks_features['p_e_ratio'].median():.2f}")
print(f"  Avg Upside Potential: {all_stocks_features['upside_potential'].mean():.1f}%")
# %%
# Cell 5: Feature Coverage Analysis
# Analyze missing values and coverage by feature category

coverage_data = []
for category, features in FEATURE_CATEGORIES.items():
    for feature in features:
        if feature in all_stocks_features.columns:
            coverage = (1 - all_stocks_features[feature].isna().mean()) * 100
            coverage_data.append({
                'Category': category,
                'Feature': feature,
                'Coverage %': coverage
            })

coverage_df = pd.DataFrame(coverage_data)

# Plot coverage by category
fig = px.box(coverage_df, x='Category', y='Coverage %',
             title='Feature Coverage by Category',
             color='Category',
             labels={'Coverage %': 'Data Coverage (%)'},
             height=500)
fig.update_layout(showlegend=False, xaxis_tickangle=-45)


# Summary statistics
print("\n📊 Coverage Summary by Category:")
print(coverage_df.groupby('Category')['Coverage %'].agg(['mean', 'median', 'min', 'max']).round(1).to_string())

fig.show()
# %% [markdown]
# ---
# ## 5. Valuation Analytics
#    - Traditional ratios
#    - Valuation timeseries & momentum
#    - Tangible book features (NEW)
# 
# %%
valuation_features = ['p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'ev_sales_ratio', 'peg_ratio']
valuation_subset = [f for f in valuation_features if f in all_stocks_features.columns]

fig = make_subplots(rows=3, cols=2, subplot_titles=valuation_subset + [''],
                    vertical_spacing=0.12)

for i, feature in enumerate(valuation_subset):
    row = i // 2 + 1
    col = i % 2 + 1
    data = all_stocks_features[feature].dropna()
    # Clip extreme values for visualization
    data_clipped = data.clip(data.quantile(0.01), data.quantile(0.99))

    fig.add_trace(
        go.Histogram(x=data_clipped, name=feature, nbinsx=50,
                     marker_color='#00bc8c', opacity=0.7),
        row=row, col=col
    )

fig.update_layout(height=600, title_text="Valuation Ratios Distribution",
                  showlegend=False, template=PLOTLY_TEMPLATE)


# Statistics table
print("\n📊 Valuation Ratios Statistics:")
stats = all_stocks_features[valuation_subset].describe().T
stats['missing_%'] = all_stocks_features[valuation_subset].isna().mean() * 100
print(stats.round(2).to_string())

fig.show()
# %%
# Cell 7: Momentum Features Analysis
momentum_features = ['price_momentum_1m', 'price_momentum_3m', 'price_momentum_6m',
                     'price_momentum_1y', 'range_52w_position']

fig = make_subplots(rows=2, cols=3, subplot_titles=momentum_features + ['EMA Crossover Distribution'])

for i, feature in enumerate(momentum_features):
    row = i // 3 + 1
    col = i % 3 + 1
    data = all_stocks_features[feature].dropna()
    data_clipped = data.clip(data.quantile(0.01), data.quantile(0.99))

    fig.add_trace(
        go.Histogram(x=data_clipped, name=feature, nbinsx=50,
                     marker_color='#3498db', opacity=0.7),
        row=row, col=col
    )

# EMA Crossover signals distribution
ema_counts = all_stocks_features['ema_crossover_20_50'].value_counts().sort_index()
fig.add_trace(
    go.Bar(x=['Bearish (-1)', 'Neutral (0)', 'Bullish (1)'],
           y=[ema_counts.get(-1, 0), ema_counts.get(0, 0), ema_counts.get(1, 0)],
           marker_color=['#e74c3c', '#adb5bd', '#00bc8c']),
    row=2, col=3
)

fig.update_layout(height=600, title_text="Momentum & Technical Features",
                  showlegend=False, template=PLOTLY_TEMPLATE)


print("\n📊 Momentum Statistics:")
print(all_stocks_features[momentum_features].describe().T.round(2).to_string())

fig.show()
# %%
# Cell 8: Quality Scores Distribution by Industry
quality_scores = ['cash_flow_quality_score', 'beta_stability_score', 'dilution_score',
                  'accounting_quality_score', 'earnings_quality_score']
quality_subset = [f for f in quality_scores if f in all_stocks_features.columns]

fig = px.box(all_stocks_features, x='industry', y='cash_flow_quality_score',
             title='Cash Flow Quality Score Distribution by Industry',
             color='industry', height=800)
fig.update_layout(showlegend=True, xaxis_tickangle=-45)
fig.show()

# %%

# Composite quality scores heatmap by sector
quality_by_sector = all_stocks_features.groupby('industry')[quality_subset].mean()

fig = px.imshow(quality_by_sector.T,
                labels=dict(x="sector", y="Quality Metric", color="Score"),
                title='Average Quality Scores by Industry',
                aspect='auto', height=800,
                color_continuous_scale='RdYlGn')
fig.update_layout(xaxis_tickangle=-45)


print("\n📊 Quality Scores by Industry:")
print(quality_by_sector.round(1).to_string())

fig.show()
# %%
# Cell 9: Analyst Sentiment Analysis
# Update sentiment features list to include new neutral percentage
sentiment_features = [
    'analyst_bullish_pct',
    'analyst_neutral_pct',  # NEW: Hold ratings percentage
    'analyst_bearish_pct',
    'upside_potential',
    'analyst_rating_normalized',
    'eps_revision_momentum'
]

# Scatter plot: Upside Potential vs Analyst Bullish %
fig = px.scatter(all_stocks_features, x='analyst_bullish_pct', y='upside_potential',
                 color='industry', hover_data=['ticker', 'name'],
                 title='Analyst Sentiment: Bullish % vs Upside Potential',
                 labels={'analyst_bullish_pct': 'Analyst Bullish %',
                         'upside_potential': 'Upside Potential %'},
                 height=600)
fig.update_traces(marker=dict(size=5, opacity=0.6))
fig.show()
# %%

# Distribution of upside potential
fig = px.histogram(all_stocks_features, x='upside_potential', nbins=100,
                   title='Distribution of Analyst Upside Potential',
                   labels={'upside_potential': 'Upside Potential %'},
                   color_discrete_sequence=['#f39c12'])
fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Fair Value")
fig.add_vline(x=all_stocks_features['upside_potential'].median(), line_dash="dot",
              line_color="#00bc8c", annotation_text="Median")


# Sentiment Statistics - Updated to include neutral percentage
sentiment_cols = ['analyst_bullish_pct', 'analyst_neutral_pct', 'analyst_bearish_pct',
                  'upside_potential', 'analyst_rating_normalized', 'eps_revision_momentum']

available_sentiment = [c for c in sentiment_cols if c in all_stocks_features.columns]
sentiment_stats = all_stocks_features[available_sentiment].describe().T
sentiment_stats['missing_%'] = (1 - all_stocks_features[available_sentiment].count() / len(all_stocks_features)) * 100

print("\n📊 Sentiment Statistics:")
print(sentiment_stats[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']].round(2))

# NEW: Verify sentiment percentages sum to 100%
required_cols = ['analyst_bullish_pct', 'analyst_neutral_pct', 'analyst_bearish_pct']
if all(col in all_stocks_features.columns for col in required_cols):
    sentiment_sum = (all_stocks_features['analyst_bullish_pct'] +
                     all_stocks_features['analyst_neutral_pct'] +
                     all_stocks_features['analyst_bearish_pct'])
    valid_sums = sentiment_sum.dropna()
    print(f"\n✓ Sentiment breakdown completeness check:")
    print(f"  Stocks with complete sentiment data: {len(valid_sums):,}")
    print(
        f"  Sum equals 100% (±0.01): {(abs(valid_sums - 100) < 0.01).sum():,} ({(abs(valid_sums - 100) < 0.01).mean() * 100:.1f}%)")

    fig.show()
# %%
# Cell 10: Feature Correlation Heatmap (Key Features)

import scipy.cluster.hierarchy as sch

key_features_v2 = [
    'p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'roe', 'roa', 'roic',
    'debt_to_equity', 'current_ratio', 'cash_ratio',
    'price_momentum_3m', 'price_momentum_1y', 'long_term_trend_score',
    'upside_potential', 'piotroski_f_score', 'distress_risk_score',
    'fcf_margin', 'fcf_positive_years', 'cash_flow_quality_score',
    'revenue_growth_yoy', 'eps_trajectory_score', 'earnings_quality_composite_comp'
]
available_features = [f for f in key_features_v2 if f in all_stocks_features.columns]

if available_features:
    corr_matrix = all_stocks_features[available_features].corr().fillna(0)

    # Hierarchical clustering for better ordering
    if not corr_matrix.empty and len(available_features) > 1:
        linkage = sch.linkage(corr_matrix, method='average')
        order = sch.leaves_list(linkage)
        corr_ordered = corr_matrix.iloc[order, order]
    else:
        corr_ordered = corr_matrix

    fig = px.imshow(
        corr_ordered,
        labels=dict(color="Correlation"),
        title='Feature Correlation Matrix (Hierarchically Clustered)',
        aspect='equal',
        height=800,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    fig.update_layout(xaxis_tickangle=-45)
    fig.show()
else:
    print("No features available for correlation matrix")

# %%
# Cell: Industry Quality Profile Radar Chart

quality_metrics = ['piotroski_f_score', 'distress_risk_score',
                   'earnings_quality_composite_comp', 'cash_flow_quality_score',
                   'fcf_positive_years']

available_metrics = [f for f in quality_metrics if f in all_stocks_features.columns]

if len(available_metrics) >= 3:
    # Normalize to 0-100 scale
    sector_col = 'industry' if 'industry' in all_stocks_features.columns else 'industry'
    sector_profiles = all_stocks_features.groupby(sector_col)[available_metrics].mean()
    sector_profiles_norm = sector_profiles.copy()

    for col in available_metrics:
        if col == 'piotroski_f_score':
            sector_profiles_norm[col] = sector_profiles[col] / 9 * 100
        elif col == 'fcf_positive_years':
            sector_profiles_norm[col] = sector_profiles[col] / 5 * 100
        else:
            # Assume already 0-100 or normalize by max
            if sector_profiles[col].max() > 100:
                sector_profiles_norm[col] = sector_profiles[col] / sector_profiles[col].max() * 100

    fig = go.Figure()

    # Top 10 sectors by avg quality
    top_sectors = sector_profiles_norm.mean(axis=1).sort_values(ascending=False).index[:10]

    for sector in top_sectors:
        values = sector_profiles_norm.loc[sector].tolist()
        values.append(values[0])  # Close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            fill='toself',
            name=sector,
            opacity=0.6
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title='Quality Profile Radar by Industry',
        template=PLOTLY_TEMPLATE,
        height=600
    )
    fig.show()
else:
    print(f"Not enough quality metrics available for radar chart. Found: {available_metrics}")
# %%

# Original correlation code for reference or fallback
key_features = [
    'p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'roe', 'roa',
    'debt_to_equity', 'current_ratio', 'price_momentum_3m',
    'upside_potential', 'piotroski_f_score', 'long_term_trend_score',
    'revenue_growth_yoy', 'eps_yoy_growth', 'dividend_yield'
]
key_features_present = [f for f in key_features if f in all_stocks_features.columns]

corr_matrix = all_stocks_features[key_features_present].corr()

fig = px.imshow(corr_matrix,
                labels=dict(color="Correlation"),
                title='Feature Correlation Matrix (Key Features)',
                aspect='equal', height=700,
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# Find highly correlated pairs
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr:
    print("\n🔗 Highly Correlated Feature Pairs (|r| > 0.7):")
    print(pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False).to_string(index=False))
# %%
# Cell 11: Regional Feature Comparison
regional_features = ['p_e_ratio', 'roe', 'debt_to_equity', 'upside_potential',
                     'dividend_yield', 'long_term_trend_score']
regional_subset = [f for f in regional_features if f in all_stocks_features.columns]

fig = make_subplots(rows=2, cols=3, subplot_titles=regional_subset)

for i, feature in enumerate(regional_subset):
    row = i // 3 + 1
    col = i % 3 + 1

    fig.add_trace(
        go.Box(x=all_stocks_features['region'], y=all_stocks_features[feature], name=feature),
        row=row, col=col
    )

fig.update_layout(height=700, title_text="Feature Distributions by Region",
                  showlegend=False, template=PLOTLY_TEMPLATE)
fig.update_xaxes(tickangle=-45)


print("\n📊 Regional Averages:")
print(all_stocks_features.groupby('region')[regional_subset].mean().round(2).to_string())
fig.show()
# %%
# Cell 12: Cash Flow Quality Analysis
cf_features = ['cfo_to_net_income', 'fcf_margin', 'fcf_positive_ratio',
               'cash_flow_quality_score', 'self_funding_ratio']
cf_subset = [f for f in cf_features if f in all_stocks_features.columns]

# Cash Flow Quality Score by Industry
fig = px.box(all_stocks_features, x='industry', y='cash_flow_quality_score',
             title='Cash Flow Quality Score by Industry',
             color='industry', height=500)
fig.update_layout(showlegend=False, xaxis_tickangle=-45)
fig.show()
# %%

# FCF positive ratio distribution
fig = px.histogram(all_stocks_features, x='fcf_positive_ratio', nbins=20,
                   title='Free Cash Flow Positive Ratio (Last 5 Quarters)',
                   labels={'fcf_positive_ratio': 'FCF Positive Ratio'},
                   color_discrete_sequence=['#00bc8c'])

print("\n📊 Cash Flow Quality Statistics:")
print(all_stocks_features[cf_subset].describe().T.round(2).to_string())

fig.show()
# %%
# Cell 13: Top & Bottom Stocks by Key Metrics
def show_top_bottom(df, metric, n=100):
    """Display top and bottom stocks for a given metric."""
    display_cols = ['ticker', 'name', 'trading_country', 'exchange', 'sector', 'industry', 'market_cap',
                    'enterprise_value', 'last_price', 'price_target',
                    'price_target_spread_pct', 'price_target_revision_1m', 'price_target_revision_3m',
                    'eps_revision_momentum', 'analyst_rating_normalized', 'analyst_coverage_quality',
                    'upside_potential', 'analyst_bullish_pct', 'analyst_bearish_pct', 'analyst_neutral_pct',
                    'pt_momentum_1y', 'pt_momentum_3m', 'pt_momentum_1m', 'pt_momentum_6m', 'pt_momentum_1w',
                    metric]
    # Filter to columns that exist in the dataframe
    available_cols = [c for c in display_cols if c in df.columns]
    top = df.nlargest(n, metric)[available_cols]
    bottom = df.nsmallest(n, metric)[available_cols]
    return top, bottom


# Quality Momentum Score Leaders
print("🏆 Top 100 Stocks by Quality Momentum Score:")
top_quality, _ = show_top_bottom(all_stocks_features, 'long_term_trend_score')
display(top_quality)

# Highest Upside Potential (with analyst coverage)
covered_stocks = all_stocks_features[all_stocks_features['analyst_bullish_pct'].notna()]
print("\n📈 Top 100 Stocks by Upside Potential (with analyst coverage):")
top_upside, _ = show_top_bottom(covered_stocks, 'upside_potential')
display(top_upside)

# Best Piotroski F-Score
print("\n⭐ Top 100 Stocks by Piotroski F-Score:")
top_piotroski, _ = show_top_bottom(all_stocks_features, 'piotroski_f_score')
display(top_piotroski)

# %%
# Cell 14: Feature Statistics Summary Export
# Create comprehensive statistics for all engineered features

all_stats = []
for category, features in FEATURE_CATEGORIES.items():
    for feature in features:
        if feature in all_stocks_features.columns:
            stats = compute_metric_statistics(all_stocks_features[feature])
            if stats:
                stats['category'] = category
                stats['feature'] = feature
                all_stats.append(stats)

stats_df = pd.DataFrame(all_stats)
stats_df = stats_df[['category', 'feature', 'count', 'mean', 'median', 'std',
                     'min', 'q25', 'q75', 'max', 'positive_pct', 'missing_pct']]

print("📊 Feature Statistics Summary (all features):")
display(stats_df.round(2))

# Save to CSV
stats_df.to_csv('outputs/feature_statistics_summary.csv', index=False)
print(f"\n✓ Saved statistics to outputs/feature_statistics_summary.csv")
# %%
# Cell 15: Interactive Feature Explorer
# Create an interactive visualization for exploring any feature by sector

def create_feature_explorer(feature_name):
    """Create an interactive box plot for a feature by sector."""
    if feature_name not in all_stocks_features.columns:
        print(f"Feature '{feature_name}' not found")
        return

    fig = px.box(all_stocks_features, x='industry', y=feature_name,
                 color='industry',
                 title=f'{feature_name} Distribution by Industry',
                 hover_data=['ticker', 'name'],
                 height=500)
    fig.update_layout(showlegend=False, xaxis_tickangle=-45)

    # Add statistics annotation
    stats = compute_metric_statistics(all_stocks_features[feature_name])
    stats_text = f"Mean: {stats['mean']:.2f} | Median: {stats['median']:.2f} | Coverage: {100 - stats['missing_pct']:.1f}%"
    fig.add_annotation(text=stats_text, xref="paper", yref="paper",
                       x=0.5, y=1.05, showarrow=False, font=dict(size=12))

    return fig


# Example: Explore different features
fig = create_feature_explorer('long_term_trend_score')


# List all available features for exploration
print("\n📋 Available Features for Exploration:")
all_features = [f for feats in FEATURE_CATEGORIES.values() for f in feats]
for i, f in enumerate(sorted(all_features)[:30], 1):
    print(f"  {i}. {f}")
print(f"  ... and {len(all_features) - 30} more")

fig.show()
# %% [markdown]
# # ---
# # ## Extended Analytics: Net Income & Earnings Quality Deep Dive
# 
# This section analyzes the comprehensive **GAAP vs Adjusted** features (48 new features) covering:
# - EPS, Net Income, EBITDA, EBIT across all periods (LTM, FY, FQ, -1FY to -4FY, -1FQFQ to -4FQFQ, 5YAVGFQ)
# - Adjustment ratios, quality scores, and earnings consistency metrics
# - GAAP revision momentum and spread analysis
# 
# %%
# Cell: GAAP vs Adjusted EPS Analysis
# Comprehensive analysis of EPS adjustments across the dataset

# Define GAAP vs Adjusted EPS columns
gaap_adj_eps_cols = [
    'eps_adjustment_ratio', 'eps_adjustment_spread_ltm', 'eps_adjustment_pct',
    'gaap_adj_eps_gap_pct', 'forward_eps_gaap_adj_spread'
]

available_eps_adj = [c for c in gaap_adj_eps_cols if c in all_stocks_features.columns]

if available_eps_adj:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'EPS Adjustment Ratio (Adj/GAAP)',
            'EPS Adjustment Spread (Adj - GAAP)',
            'EPS Adjustment % Distribution',
            'GAAP vs Adjusted Gap by Industry'
        ]
    )

    # Adjustment Ratio
    if 'eps_adjustment_ratio' in all_stocks_features.columns:
        adj_ratio = all_stocks_features['eps_adjustment_ratio'].dropna()
        adj_ratio_clipped = adj_ratio.clip(adj_ratio.quantile(0.05), adj_ratio.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=adj_ratio_clipped, nbinsx=50, marker_color='#00bc8c', name='Adj Ratio'),
            row=1, col=1
        )

    # Adjustment Spread
    if 'eps_adjustment_spread_ltm' in all_stocks_features.columns:
        adj_spread = all_stocks_features['eps_adjustment_spread_ltm'].dropna()
        adj_spread_clipped = adj_spread.clip(adj_spread.quantile(0.05), adj_spread.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=adj_spread_clipped, nbinsx=50, marker_color='#3498db', name='Adj Spread'),
            row=1, col=2
        )

    # Adjustment Percentage
    if 'eps_adjustment_pct' in all_stocks_features.columns:
        adj_pct = all_stocks_features['eps_adjustment_pct'].dropna()
        adj_pct_clipped = adj_pct.clip(adj_pct.quantile(0.05), adj_pct.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=adj_pct_clipped, nbinsx=50, marker_color='#e74c3c', name='Adj %'),
            row=2, col=1
        )

    # Gap by Industry
    if 'gaap_adj_eps_gap_pct' in all_stocks_features.columns:
        gap_by_sector = all_stocks_features.groupby('industry')['gaap_adj_eps_gap_pct'].median().sort_values()
        fig.add_trace(
            go.Bar(x=gap_by_sector.values, y=gap_by_sector.index, orientation='h',
                   marker_color='#9b59b6', name='Gap %'),
            row=2, col=2
        )

    fig.update_layout(
        height=700,
        title_text="GAAP vs Adjusted EPS Analysis",
        showlegend=False,
        template=PLOTLY_TEMPLATE
    )


    # Summary Statistics
    print("\n📊 GAAP vs Adjusted EPS Statistics:")
    if 'eps_adjustment_ratio' in all_stocks_features.columns:
        ratio_stats = all_stocks_features['eps_adjustment_ratio'].describe()
        print(f"  Median Adjustment Ratio: {ratio_stats['50%']:.2f}")
        print(f"  Stocks with Adj > GAAP: {(all_stocks_features['eps_adjustment_ratio'] > 1).sum():,}")
        print(f"  Stocks with Adj < GAAP: {(all_stocks_features['eps_adjustment_ratio'] < 1).sum():,}")

    fig.show()

# %%
# Cell: Net Income GAAP vs Adjusted Deep Dive
# Analysis of Net Income adjustments and normalization

ni_gaap_cols = [
    'net_income_is_ltm', 'net_income_adj_ltm', 'normalized_ni_ltm',
    'ni_adjustment_ratio', 'net_income_adjustment_pct'
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Net Income Adjustment Ratio Distribution',
        'Net Income Adjustment % Distribution',
        'GAAP vs Adjusted NI Scatter',
        'NI Adjustment by Industry (Median)'
    ]
)

# NI Adjustment Ratio
if 'ni_adjustment_ratio' in all_stocks_features.columns:
    ni_adj = all_stocks_features['ni_adjustment_ratio'].dropna()
    ni_adj_clipped = ni_adj.clip(ni_adj.quantile(0.05), ni_adj.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ni_adj_clipped, nbinsx=50, marker_color='#1abc9c'),
        row=1, col=1
    )

# NI Adjustment %
if 'net_income_adjustment_pct' in all_stocks_features.columns:
    ni_pct = all_stocks_features['net_income_adjustment_pct'].dropna()
    ni_pct_clipped = ni_pct.clip(ni_pct.quantile(0.05), ni_pct.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ni_pct_clipped, nbinsx=50, marker_color='#e67e22'),
        row=1, col=2
    )

# GAAP vs Adjusted Scatter
if 'net_income_is_ltm' in all_stocks_features.columns and 'net_income_adj_ltm' in all_stocks_features.columns:
    scatter_df = all_stocks_features[['net_income_is_ltm', 'net_income_adj_ltm', 'industry']].dropna()
    # Clip for visualization
    scatter_df = scatter_df[
        (scatter_df['net_income_is_ltm'].between(
            scatter_df['net_income_is_ltm'].quantile(0.05),
            scatter_df['net_income_is_ltm'].quantile(0.95)
        )) &
        (scatter_df['net_income_adj_ltm'].between(
            scatter_df['net_income_adj_ltm'].quantile(0.05),
            scatter_df['net_income_adj_ltm'].quantile(0.95)
        ))
        ]
    fig.add_trace(
        go.Scatter(
            x=scatter_df['net_income_is_ltm'],
            y=scatter_df['net_income_adj_ltm'],
            mode='markers',
            marker=dict(size=4, opacity=0.5, color='#3498db'),
            name='GAAP vs Adj'
        ),
        row=2, col=1
    )
    # Add diagonal reference line
    max_val = max(scatter_df['net_income_is_ltm'].max(), scatter_df['net_income_adj_ltm'].max())
    min_val = min(scatter_df['net_income_is_ltm'].min(), scatter_df['net_income_adj_ltm'].min())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                   line=dict(dash='dash', color='white'), name='1:1 Line'),
        row=2, col=1
    )

# Adjustment by Industry
if 'ni_adjustment_ratio' in all_stocks_features.columns:
    sector_adj = all_stocks_features.groupby('industry')['ni_adjustment_ratio'].median().sort_values()
    fig.add_trace(
        go.Bar(x=sector_adj.values, y=sector_adj.index, orientation='h',
               marker_color='#f39c12'),
        row=2, col=2
    )

fig.update_layout(
    height=750,
    title_text="Net Income: GAAP vs Adjusted Analysis",
    showlegend=False,
    template=PLOTLY_TEMPLATE
)
fig.show()

# %%
# Cell: EBITDA Adjustment Analysis
# EBITDA GAAP vs Adjusted comparison

ebitda_adj_cols = ['ebitda_ltm', 'ebitda_adj_ltm', 'ebitda_adjustment_ratio', 'ebitda_adjustment_pct_fy']

fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=[
        'EBITDA Adjustment Ratio',
        'EBITDA Adjustment %',
        'Adjustment Ratio by Industry'
    ]
)

# EBITDA Adjustment Ratio
if 'ebitda_adjustment_ratio' in all_stocks_features.columns:
    ebitda_ratio = all_stocks_features['ebitda_adjustment_ratio'].dropna()
    ebitda_ratio_clipped = ebitda_ratio.clip(ebitda_ratio.quantile(0.05), ebitda_ratio.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ebitda_ratio_clipped, nbinsx=50, marker_color='#00bc8c'),
        row=1, col=1
    )

# EBITDA Adjustment %
if 'ebitda_adjustment_pct_fy' in all_stocks_features.columns:
    ebitda_pct = all_stocks_features['ebitda_adjustment_pct_fy'].dropna()
    ebitda_pct_clipped = ebitda_pct.clip(ebitda_pct.quantile(0.05), ebitda_pct.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ebitda_pct_clipped, nbinsx=50, marker_color='#e74c3c'),
        row=1, col=2
    )

# By Industry
if 'ebitda_adjustment_ratio' in all_stocks_features.columns:
    ebitda_sector = all_stocks_features.groupby('industry')['ebitda_adjustment_ratio'].median().sort_values()
    fig.add_trace(
        go.Bar(x=ebitda_sector.values, y=ebitda_sector.index, orientation='h',
               marker_color='#9b59b6'),
        row=1, col=3
    )

fig.update_layout(
    height=450,
    title_text="EBITDA: GAAP vs Adjusted Analysis",
    showlegend=False,
    template=PLOTLY_TEMPLATE
)


print("\n📊 EBITDA Adjustment Summary:")
if 'ebitda_adjustment_ratio' in all_stocks_features.columns:
    print(f"  Median EBITDA Adj Ratio: {all_stocks_features['ebitda_adjustment_ratio'].median():.2f}")
    print(
        f"  Stocks with significant adjustments (>10%): {(all_stocks_features['ebitda_adjustment_pct_fy'].abs() > 10).sum():,}")

fig.show()

# %%
# Cell: GAAP Revision Momentum Analysis
# Tracking analyst revisions to GAAP EPS estimates

gaap_revision_cols = [
    'gaap_revision_momentum', 'gaap_revision_1m', 'gaap_revision_3m',
    'gaap_revision_6m', 'gaap_revision_1y', 'gaap_vs_norm_revision_spread',
    'gaap_revision_acceleration', 'gaap_positive_revision_flag'
]

available_rev_cols = [c for c in gaap_revision_cols if c in all_stocks_features.columns]

if available_rev_cols:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'GAAP Revision Momentum (Weighted)',
            'Revision Timeline (1M → 1Y)',
            'GAAP vs Normalized Revision Spread',
            'Positive Revision Flags by Industry'
        ]
    )

    # Revision Momentum
    if 'gaap_revision_momentum' in all_stocks_features.columns:
        rev_mom = all_stocks_features['gaap_revision_momentum'].dropna()
        rev_mom_clipped = rev_mom.clip(rev_mom.quantile(0.05), rev_mom.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=rev_mom_clipped, nbinsx=50, marker_color='#00bc8c'),
            row=1, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=1)

    # Revision Timeline Box Plot
    revision_timeline = ['gaap_revision_1m', 'gaap_revision_3m', 'gaap_revision_6m', 'gaap_revision_1y']
    available_timeline = [c for c in revision_timeline if c in all_stocks_features.columns]

    for i, col in enumerate(available_timeline):
        data = all_stocks_features[col].dropna()
        data_clipped = data.clip(data.quantile(0.05), data.quantile(0.95))
        fig.add_trace(
            go.Box(y=data_clipped, name=col.replace('gaap_revision_', '').upper(),
                   marker_color=['#3498db', '#e74c3c', '#f39c12', '#9b59b6'][i]),
            row=1, col=2
        )

    # GAAP vs Norm Spread
    if 'gaap_vs_norm_revision_spread' in all_stocks_features.columns:
        spread = all_stocks_features['gaap_vs_norm_revision_spread'].dropna()
        spread_clipped = spread.clip(spread.quantile(0.05), spread.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=spread_clipped, nbinsx=50, marker_color='#e67e22'),
            row=2, col=1
        )

    # Positive Revision Flags by Industry
    if 'gaap_positive_revision_flag' in all_stocks_features.columns:
        pos_rev_sector = all_stocks_features.groupby('industry')['gaap_positive_revision_flag'].mean() * 100
        pos_rev_sector = pos_rev_sector.sort_values()
        fig.add_trace(
            go.Bar(x=pos_rev_sector.values, y=pos_rev_sector.index, orientation='h',
                   marker_color='#1abc9c'),
            row=2, col=2
        )

    fig.update_layout(
        height=750,
        title_text="GAAP EPS Revision Momentum Analysis",
        showlegend=False,
        template=PLOTLY_TEMPLATE
    )


    # Summary
    print("\n📈 GAAP Revision Summary:")
    if 'gaap_revision_momentum' in all_stocks_features.columns:
        print(
            f"  Stocks with positive revision momentum: {(all_stocks_features['gaap_revision_momentum'] > 0).sum():,}")
        print(
            f"  Stocks with negative revision momentum: {(all_stocks_features['gaap_revision_momentum'] < 0).sum():,}")
    if 'gaap_positive_revision_flag' in all_stocks_features.columns:
        print(
            f"  Stocks with consistently positive GAAP revisions: {all_stocks_features['gaap_positive_revision_flag'].sum():,}")

        fig.show()
    else:
        print("No GAAP revision data available for summary")

# %%
# Cell: Earnings Quality Composite Analysis
# Comprehensive view of earnings quality metrics

quality_cols = [
    'earnings_quality_score', 'earnings_quality_composite_comp',
    'earnings_quality_warning', 'eps_trajectory_score'
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Earnings Quality Score Distribution',
        'Earnings Quality Composite Distribution',
        'Quality Score by Industry',
        'Quality Warning Flags by Industry'
    ],
    specs=[[{"type": "histogram"}, {"type": "histogram"}],
           [{"type": "bar"}, {"type": "bar"}]]
)

# Earnings Quality Score
if 'earnings_quality_score' in all_stocks_features.columns:
    eq_score = all_stocks_features['earnings_quality_score'].dropna()
    fig.add_trace(
        go.Histogram(x=eq_score, nbinsx=30, marker_color='#00bc8c'),
        row=1, col=1
    )

# Earnings Quality Composite
if 'earnings_quality_composite_comp' in all_stocks_features.columns:
    eq_comp = all_stocks_features['earnings_quality_composite_comp'].dropna()
    fig.add_trace(
        go.Histogram(x=eq_comp, nbinsx=30, marker_color='#3498db'),
        row=1, col=2
    )

# Quality by Industry
if 'earnings_quality_composite_comp' in all_stocks_features.columns:
    quality_sector = all_stocks_features.groupby('industry')['earnings_quality_composite_comp'].mean().sort_values()
    colors = ['#e74c3c' if v < 50 else '#f39c12' if v < 70 else '#00bc8c' for v in quality_sector.values]
    fig.add_trace(
        go.Bar(x=quality_sector.values, y=quality_sector.index, orientation='h',
               marker_color=colors),
        row=2, col=1
    )

# Warning Flags by Industry
if 'earnings_quality_warning' in all_stocks_features.columns:
    warning_sector = all_stocks_features.groupby('industry')['earnings_quality_warning'].mean() * 100
    warning_sector = warning_sector.sort_values(ascending=False)
    fig.add_trace(
        go.Bar(x=warning_sector.values, y=warning_sector.index, orientation='h',
               marker_color='#e74c3c'),
        row=2, col=2
    )

fig.update_layout(
    height=750,
    title_text="Earnings Quality Comprehensive Analysis",
    showlegend=False,
    template=PLOTLY_TEMPLATE
)


# Quality Tier Summary
print("\n⭐ Earnings Quality Tier Summary:")
if 'earnings_quality_composite_comp' in all_stocks_features.columns:
    eq = all_stocks_features['earnings_quality_composite_comp']
    print(f"  High Quality (>70): {(eq > 70).sum():,} stocks ({(eq > 70).mean() * 100:.1f}%)")
    print(f"  Medium Quality (50-70): {((eq >= 50) & (eq <= 70)).sum():,} stocks")
    print(f"  Low Quality (<50): {(eq < 50).sum():,} stocks ({(eq < 50).mean() * 100:.1f}%)")
if 'earnings_quality_warning' in all_stocks_features.columns:
    print(f"  ⚠️ Quality Warning Flags: {all_stocks_features['earnings_quality_warning'].sum():,} stocks")

    fig.show()

# %%
# Cell: Historical Net Income Trends (Multi-Period Analysis)
# Analyzing NI across FQ, FY, and historical periods

ni_historical_cols = [
    'net_income_is_fq', 'net_income_is_ltm', 'net_income_is_fy',
    'net_income_is_1fy', 'net_income_is_2fy', 'net_income_is_3fy', 'net_income_is_4fy',
    'net_income_positive_years', 'net_income_growth_yoy', 'net_income_qoq_growth'
]

available_ni_hist = [c for c in ni_historical_cols if c in all_stocks_features.columns]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Net Income YoY Growth Distribution',
        'Net Income QoQ Growth Distribution',
        'Positive Net Income Years (out of 5)',
        'NI Profitability Trend by Industry'
    ]
)

# YoY Growth
if 'net_income_growth_yoy' in all_stocks_features.columns:
    ni_yoy = all_stocks_features['net_income_growth_yoy'].dropna()
    ni_yoy_clipped = ni_yoy.clip(ni_yoy.quantile(0.05), ni_yoy.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ni_yoy_clipped, nbinsx=50, marker_color='#00bc8c'),
        row=1, col=1
    )
    fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=1)

# QoQ Growth
if 'net_income_qoq_growth' in all_stocks_features.columns:
    ni_qoq = all_stocks_features['net_income_qoq_growth'].dropna()
    ni_qoq_clipped = ni_qoq.clip(ni_qoq.quantile(0.05), ni_qoq.quantile(0.95))
    fig.add_trace(
        go.Histogram(x=ni_qoq_clipped, nbinsx=50, marker_color='#3498db'),
        row=1, col=2
    )

# Positive Years Distribution
if 'net_income_positive_years' in all_stocks_features.columns:
    pos_years_dist = all_stocks_features['net_income_positive_years'].value_counts().sort_index()
    colors = ['#e74c3c', '#e74c3c', '#f39c12', '#f39c12', '#00bc8c', '#00bc8c']
    fig.add_trace(
        go.Bar(x=pos_years_dist.index.astype(str), y=pos_years_dist.values,
               marker_color=colors[:len(pos_years_dist)]),
        row=2, col=1
    )

# Profitability by Industry
if 'net_income_positive_years' in all_stocks_features.columns:
    avg_pos_years = all_stocks_features.groupby('industry')['net_income_positive_years'].mean().sort_values()
    fig.add_trace(
        go.Bar(x=avg_pos_years.values, y=avg_pos_years.index, orientation='h',
               marker_color='#9b59b6'),
        row=2, col=2
    )

fig.update_layout(
    height=700,
    title_text="Net Income Historical Trends Analysis",
    showlegend=False,
    template=PLOTLY_TEMPLATE
)
fig.show()

# Summary
print("\n📊 Net Income Profitability Summary:")
if 'net_income_positive_years' in all_stocks_features.columns:
    ni_pos = all_stocks_features['net_income_positive_years']
    print(f"  Always Profitable (5/5 years): {(ni_pos == 5).sum():,} stocks")
    print(f"  Mostly Profitable (4+/5 years): {(ni_pos >= 4).sum():,} stocks")
    print(f"  Never Profitable (0/5 years): {(ni_pos == 0).sum():,} stocks")
if 'net_income_growth_yoy' in all_stocks_features.columns:
    print(f"  Stocks with positive YoY NI growth: {(all_stocks_features['net_income_growth_yoy'] > 0).sum():,}")

fig.show()

# %%
# Cell: GAAP vs Adjusted Interactive Comparison
# Interactive scatter plot comparing GAAP and Adjusted metrics

from plotly.subplots import make_subplots

# Create interactive comparison scatter
fig = px.scatter(
    all_stocks_features.dropna(subset=['eps_adjustment_pct', 'ebitda_adjustment_pct_fy']),
    x='eps_adjustment_pct',
    y='ebitda_adjustment_pct_fy',
    color='industry',
    size='market_cap',
    hover_data=['ticker', 'name', 'sector', 'country', 'exchange', 'earnings_quality_score', 'earnings_quality_warning',
                'accounting_quality_score'],
    title='EPS vs EBITDA Adjustment % (Color = Industry, Size = Market Cap)',
    labels={
        'eps_adjustment_pct': 'EPS Adjustment % (Adj vs GAAP)',
        'ebitda_adjustment_pct_fy': 'EBITDA Adjustment % (Adj vs GAAP)'
    },
    height=600
)

# Clip axes for better visualization
fig.update_xaxes(range=[
    all_stocks_features['eps_adjustment_pct'].quantile(0.05),
    all_stocks_features['eps_adjustment_pct'].quantile(0.95)
])
fig.update_yaxes(range=[
    all_stocks_features['ebitda_adjustment_pct_fy'].quantile(0.05),
    all_stocks_features['ebitda_adjustment_pct_fy'].quantile(0.95)
])

# Add reference lines
fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5)

# Add quadrant annotations
fig.add_annotation(x=20, y=20, text="High Adj > GAAP", showarrow=False,
                   font=dict(color="green", size=10))
fig.add_annotation(x=-20, y=-20, text="GAAP > High Adj", showarrow=False,
                   font=dict(color="red", size=10))

fig.update_traces(marker=dict(opacity=0.6, sizemin=3))
fig.update_layout(template=PLOTLY_TEMPLATE)
fig.show()

# %%
# Cell: Earnings Quality Screening Dashboard
# Interactive widget for screening stocks by earnings quality criteria

def screen_earnings_quality(
        min_quality_score: float = 60,
        max_adjustment_pct: float = 20,
        require_positive_revisions: bool = False,
        min_positive_years: int = 3,
        sector_filter: str = 'All'
) -> pd.DataFrame:
    """
    Screen stocks based on earnings quality criteria.
    
    Args:
        min_quality_score: Minimum earnings quality composite score (0-100)
        max_adjustment_pct: Maximum absolute EPS adjustment percentage
        require_positive_revisions: Only include stocks with positive GAAP revision flag
        min_positive_years: Minimum net income positive years (0-5)
        sector_filter: Filter by sector
    """
    df = all_stocks_features.copy()

    mask = pd.Series([True] * len(df))

    if 'earnings_quality_composite' in df.columns:
        mask &= (df['earnings_quality_composite'] >= min_quality_score)

    if 'eps_adjustment_pct' in df.columns:
        mask &= (df['eps_adjustment_pct'].abs() <= max_adjustment_pct)

    if require_positive_revisions and 'gaap_positive_revision_flag' in df.columns:
        mask &= (df['gaap_positive_revision_flag'] == 1)

    if 'net_income_positive_years' in df.columns:
        mask &= (df['net_income_positive_years'] >= min_positive_years)

    if sector_filter != 'All':
        mask &= (df['industry'] == sector_filter)

    result = df[mask].sort_values('earnings_quality_composite', ascending=False)

    return result


# Example: High quality earnings screen
high_quality_earnings = screen_earnings_quality(
    min_quality_score=50,
    max_adjustment_pct=50,
    min_positive_years=2
)

print(f"✓ Found {len(high_quality_earnings):,} stocks with high earnings quality")

# Display columns
display_cols = [
    'ticker', 'name', 'industry', 'earnings_quality_composite',
    'eps_adjustment_pct', 'net_income_positive_years',
    'gaap_revision_momentum', 'earnings_quality_warning'
]
available_display = [c for c in display_cols if c in high_quality_earnings.columns]

display(high_quality_earnings[available_display].head(100))

# %%
# Cell: GAAP vs Adjusted Summary Statistics Table
# Comprehensive statistics for all GAAP vs Adjusted features

gaap_adj_features = [
    'eps_adjustment_ratio', 'eps_adjustment_pct', 'eps_adjustment_spread_ltm',
    'ni_adjustment_ratio', 'net_income_adjustment_pct',
    'ebitda_adjustment_ratio', 'ebitda_adjustment_pct_fy',
    'earnings_quality_score', 'earnings_quality_composite_comp',
    'gaap_revision_momentum', 'gaap_vs_norm_revision_spread'
]

available_gaap_features = [f for f in gaap_adj_features if f in all_stocks_features.columns]

gaap_stats = []
for feature in available_gaap_features:
    stats = compute_metric_statistics(all_stocks_features[feature])
    if stats:
        stats['feature'] = feature
        gaap_stats.append(stats)

gaap_stats_df = pd.DataFrame(gaap_stats)
gaap_stats_df = gaap_stats_df[['feature', 'count', 'mean', 'median', 'std', 'min', 'max', 'missing_pct']]

print("\n📊 GAAP vs Adjusted Features Summary Statistics:")
display(gaap_stats_df.round(2))

# Save to file
gaap_stats_df.to_csv('outputs/gaap_vs_adjusted_statistics.csv', index=False)
print("\n✓ Statistics saved to outputs/gaap_vs_adjusted_statistics.csv")

# %%
# Cell 16: Profitability & Margin Analysis (Aligned with mv_all_stock_features)
# Using available profitability metrics from the materialized view

profitability_cols = [
    'ticker', 'industry', 'net_margin_pct', 'gross_margin_pct', 'operating_margin_pct',
    'ebitda_margin_pct', 'roe', 'roa', 'roic', 'net_margin_trend_yoy',
    'gross_margin_trend_yoy', 'operating_margin_trend', 'ebitda_margin_trend',
    'margin_expansion_flag'
]

profitability_data = all_stocks_features[
    [c for c in profitability_cols if c in all_stocks_features.columns]
].copy()

# Net Margin Distribution
fig = px.histogram(
    all_stocks_features,
    x='net_margin_pct',
    nbins=50,
    title='Net Income Margin (LTM) Distribution',
    labels={'net_margin_pct': 'Net Margin %'},
    color_discrete_sequence=['#00bc8c']
)
median_val = all_stocks_features['net_margin_pct'].median()
fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Breakeven")
fig.add_vline(x=median_val, line_dash="dot", line_color="yellow",
              annotation_text=f"Median ({median_val:.1f}%)")


# Statistics
print("\n📊 Profitability Statistics:")
margin_cols = ['net_margin_pct', 'gross_margin_pct', 'operating_margin_pct', 'ebitda_margin_pct']
available_margin_cols = [c for c in margin_cols if c in all_stocks_features.columns]
print(all_stocks_features[available_margin_cols].describe().round(2))
profitable = (all_stocks_features['net_margin_pct'] > 0).sum()
high_margin = (all_stocks_features['net_margin_pct'] > 15).sum()
print(f"\nProfitable Stocks (margin > 0): {profitable:,}")
print(f"High Margin Stocks (margin > 15%): {high_margin:,}")

fig.show()

# %%
# Cell 17: Margin Consistency & Trend Analysis
# Analyze profitability trends using available columns

def create_margin_trends_chart(
        data: pd.DataFrame,
        margin_trend_cols: list[str],
        subplot_titles: list[str],
        chart_title: str,
        bar_color: str = '#3498db',
        height: int = 500
) -> Figure | None:
    """
    Create a multi-column visualization for margin trend metrics.
    """
    available_cols = [c for c in margin_trend_cols if c in data.columns]
    available_titles = subplot_titles[:len(available_cols)]

    if not available_cols:
        print(f"No margin trend columns available from: {margin_trend_cols}")
        return None

    fig = make_subplots(rows=1, cols=len(available_cols), subplot_titles=available_titles)

    for i, col in enumerate(available_cols):
        col_data = data[col].dropna()
        col_clipped = col_data.clip(col_data.quantile(0.05), col_data.quantile(0.95))
        fig.add_trace(
            go.Histogram(
                x=col_clipped,
                nbinsx=40,
                marker_color=bar_color,
                name=col
            ),
            row=1,
            col=i + 1
        )

    fig.update_layout(
        height=height,
        title_text=chart_title,
        showlegend=False,
        template=PLOTLY_TEMPLATE
    )
    return fig


# Margin trend analysis using available columns
margin_trend_columns = [
    'net_margin_trend_yoy',
    'gross_margin_trend_yoy',
    'operating_margin_trend',
    'ebitda_margin_trend'
]
margin_trend_titles = [
    'Net Margin YoY Trend',
    'Gross Margin YoY Trend',
    'Operating Margin Trend',
    'EBITDA Margin Trend'
]

fig = create_margin_trends_chart(
    data=all_stocks_features,
    margin_trend_cols=margin_trend_columns,
    subplot_titles=margin_trend_titles,
    chart_title="Margin Trend Analysis (Year-over-Year Changes)"
)
if fig:
    fig.show()
# %%

# Margin Expansion Flag by Industry
if 'margin_expansion_flag' in all_stocks_features.columns:
    expansion_by_sector = all_stocks_features.groupby('industry')['margin_expansion_flag'].agg(['sum', 'mean'])
    expansion_by_sector.columns = ['count', 'pct']
    expansion_by_sector['pct'] = expansion_by_sector['pct'] * 100
    expansion_by_sector = expansion_by_sector.sort_values('pct', ascending=True)

    fig = px.bar(
        x=expansion_by_sector['pct'],
        y=expansion_by_sector.index,
        orientation='h',
        title='Margin Expansion Rate by Industry (%)',
        labels={'x': '% of Stocks with Margin Expansion', 'y': 'industry'},
        color=expansion_by_sector['pct'],
        color_continuous_scale='Greens'
    )


    print(f"\n📈 Margin Expansion Summary:")
    print(f"Total stocks with margin expansion: {all_stocks_features['margin_expansion_flag'].sum():,}")
    print(f"Expansion rate: {all_stocks_features['margin_expansion_flag'].mean() * 100:.1f}%")

    fig.show()

# %%
# Cell 18: Profitability Quality Analysis
# ROE, ROA, ROIC comparison across sectors

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'ROE Distribution',
    'ROA Distribution',
    'ROIC Distribution',
    'Profitability Metrics by Industry'
])

# ROE
roe_data = all_stocks_features['roe'].dropna()
roe_clipped = roe_data.clip(roe_data.quantile(0.05), roe_data.quantile(0.95))
fig.add_trace(go.Histogram(x=roe_clipped, nbinsx=50, marker_color='#00bc8c'), row=1, col=1)

# ROA
roa_data = all_stocks_features['roa'].dropna()
roa_clipped = roa_data.clip(roa_data.quantile(0.05), roa_data.quantile(0.95))
fig.add_trace(go.Histogram(x=roa_clipped, nbinsx=50, marker_color='#3498db'), row=1, col=2)

# ROIC
roic_data = all_stocks_features['roic'].dropna()
roic_clipped = roic_data.clip(roic_data.quantile(0.05), roic_data.quantile(0.95))
fig.add_trace(go.Histogram(x=roic_clipped, nbinsx=50, marker_color='#9b59b6'), row=2, col=1)

# Average by sector
profitability_by_sector = all_stocks_features.groupby('industry')[['roe', 'roa', 'roic']].median()
avg_profitability = profitability_by_sector.mean(axis=1).sort_values()
fig.add_trace(go.Bar(x=avg_profitability.values, y=avg_profitability.index,
                     orientation='h', marker_color='#f39c12'), row=2, col=2)

fig.update_layout(height=700, title_text="Profitability Quality Analysis",
                  showlegend=False, template=PLOTLY_TEMPLATE)

print("\n📊 Profitability Summary:")
print(f"Median ROE: {all_stocks_features['roe'].median():.1f}%")
print(f"Median ROA: {all_stocks_features['roa'].median():.1f}%")
print(f"Median ROIC: {all_stocks_features['roic'].median():.1f}%")

fig.show()

# %% [markdown]
# # ---
# # ## Technical Analysis Features
# 
# %%
# Cell 19: Technical Analysis Dashboard
# Use features from the main dataframe

technical_features = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in technical_features.columns and 'industry' not in technical_features.columns:
    technical_features['industry'] = technical_features['industry']

# EMA Trend Consistency Distribution
trend_counts = technical_features['ema_trend_consistency'].value_counts().sort_index()
colors = ['#e74c3c', '#adb5bd', '#00bc8c']
labels = ['Bearish (-1)', 'Mixed (0)', 'Bullish (1)']

fig = go.Figure(data=[
    go.Pie(labels=labels, values=[trend_counts.get(-1, 0), trend_counts.get(0, 0), trend_counts.get(1, 0)],
           marker_colors=colors, hole=0.4)
])
fig.update_layout(title='EMA Trend Consistency Distribution', template=PLOTLY_TEMPLATE)
fig.show()
# %%

# Breakout Signals by Industry
breakout_by_sector = technical_features.groupby('industry')['breakout_signal'].sum().sort_values(ascending=True)
fig = px.bar(x=breakout_by_sector.values, y=breakout_by_sector.index, orientation='h',
             title='Breakout Signals Count by Industry',
             labels={'x': 'Number of Breakout Signals', 'y': 'industry'},
             color=breakout_by_sector.values, color_continuous_scale='Greens')

print(f"\n🚀 Total Breakout Signals: {technical_features['breakout_signal'].sum()}")
print(f"📊 High Volume Flags: {technical_features['high_volume_flag'].sum()}")
print(f"📉 Low Volume Flags: {technical_features['low_volume_flag'].sum()}")

fig.show()
# %%
# Cell 20: Volume Momentum Score Analysis
fig = px.scatter(
    technical_features,
    x='ema_slope_20d',
    y='volume_momentum_score',
    color='ema_trend_consistency',
    hover_data=['ticker', 'name', 'sector', 'industry', 'country', 'trading_country', 'exchange', 'breakout_signal'],
    title='Volume Momentum vs EMA Slope (colored by trend consistency)',
    labels={'ema_slope_20d': 'EMA Slope (20D)', 'volume_momentum_score': 'Volume Momentum Score'},
    color_continuous_scale='RdYlGn'
)
fig.update_traces(marker=dict(size=6, opacity=0.6))
fig.show()

# %%
# Cell 21: Volatility Analysis
fig = make_subplots(rows=1, cols=2, subplot_titles=[
    'Volatility Compression Distribution',
    'Volatility Term Structure Distribution'
])

vol_comp = technical_features['volatility_compression'].dropna()
vol_comp_clipped = vol_comp.clip(vol_comp.quantile(0.05), vol_comp.quantile(0.95))
fig.add_trace(go.Histogram(x=vol_comp_clipped, nbinsx=50, marker_color='#3498db', name='Compression'), row=1, col=1)

vol_term = technical_features['volatility_term_structure'].dropna()
vol_term_clipped = vol_term.clip(vol_term.quantile(0.05), vol_term.quantile(0.95))
fig.add_trace(go.Histogram(x=vol_term_clipped, nbinsx=50, marker_color='#e74c3c', name='Term Structure'), row=1, col=2)

fig.update_layout(height=400, title_text="Volatility Metrics Analysis",
                  showlegend=False, template=PLOTLY_TEMPLATE)
fig.show()

# %% [markdown]
# # ---
# # ## Financial Distress & Risk Analysis
# 
# %%
# Cell 22: Financial Distress Features
# Use features from the main dataframe

distress_features = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in distress_features.columns and 'industry' not in distress_features.columns:
    distress_features['industry'] = distress_features['industry']

# Distress Risk Score Distribution
fig = px.histogram(
    distress_features,
    x='distress_risk_score',
    nbins=30,
    color='industry',
    title='Distress Risk Score Distribution by Industry',
    labels={'distress_risk_score': 'Distress Risk Score (0=High Risk, 100=Safe)'},
    marginal='box'
)
fig.add_vline(x=30, line_dash="dash", line_color="red", annotation_text="High Risk Zone")
fig.add_vline(x=70, line_dash="dash", line_color="green", annotation_text="Safe Zone")
fig.show()
# %%

# Risk Summary
print("\n⚠️ Financial Distress Summary:")
print(f"High Risk Stocks (score < 30): {(distress_features['distress_risk_score'] < 30).sum():,}")
print(
    f"Moderate Risk (30-70): {((distress_features['distress_risk_score'] >= 30) & (distress_features['distress_risk_score'] < 70)).sum():,}")
print(f"Low Risk (score >= 70): {(distress_features['distress_risk_score'] >= 70).sum():,}")

# %%
# Cell 23: Cash Runway Analysis
fig = px.scatter(
    distress_features,
    x='cash_runway_months',
    y='distress_risk_score',
    color='liquidity_stress_score',
    size='market_cap',
    hover_data=['ticker', 'name', 'industry', 'trading_country', 'country', 'exchange'],
    title='Cash Runway vs Distress Risk (size = Market Cap)',
    labels={'liquidity_stress_score': 'Liquidity Stress Score',
            'distress_risk_score': 'Distress Risk Score'},
    color_continuous_scale='RdYlGn_r'
)
fig.update_traces(marker=dict(opacity=0.7, sizemin=5))
# Limit x-axis for better visualization
fig.update_xaxes(range=[0, distress_features['liquidity_stress_score'].quantile(0.95)])
fig.show()

# %%
# Cell 24 (Fixed): Working Capital Health Indicators
# Using available columns from mv_all_stock_features

wc_available_cols = ['wc_to_revenue', 'wc_to_assets', 'wc_change_qoq', 'wc_change_yoy',
                     'wc_improvement_flag', 'negative_wc_flag', 'adequate_cash_buffer']
wc_cols_present = [c for c in wc_available_cols if c in all_stocks_features.columns]

if wc_cols_present:
    wc_flags = all_stocks_features[['ticker', 'name', 'industry', 'country', 'exchange'] + wc_cols_present].copy()

    # Build flag summary from available columns
    flag_metrics = []
    if 'negative_wc_flag' in wc_flags.columns:
        flag_metrics.append(('Negative Working Capital', wc_flags['negative_wc_flag'].sum(),
                             wc_flags['negative_wc_flag'].mean() * 100))
    if 'wc_improvement_flag' in wc_flags.columns:
        flag_metrics.append(('WC Improving', wc_flags['wc_improvement_flag'].sum(),
                             wc_flags['wc_improvement_flag'].mean() * 100))
    if 'adequate_cash_buffer' in wc_flags.columns:
        flag_metrics.append(('Adequate Cash Buffer', wc_flags['adequate_cash_buffer'].sum(),
                             wc_flags['adequate_cash_buffer'].mean() * 100))

    if flag_metrics:
        flag_summary = pd.DataFrame(flag_metrics, columns=['Metric', 'Count', 'Percentage'])

        fig = px.bar(flag_summary, x='Metric', y='Count', color='Percentage',
                     title='Working Capital Health Indicators',
                     text='Count', color_continuous_scale='RdYlGn')
        fig.update_traces(textposition='outside')

        display(flag_summary)
else:
    print("Working capital flag columns not available in current schema")

    fig.show()

# %% [markdown]
# # ---
# # ## Long-Term Momentum & Secular Trends (NEW)
# 
# %%
# Cell: Long-Term Price Momentum Analysis
# New features: price_momentum_3y, price_momentum_5y, long_term_trend_score, secular_trend_flag

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    '3-Year Price Momentum',
    '5-Year Price Momentum',
    'Long-Term Trend Score',
    'Secular Trend Flags by Industry'
])

# 3Y Momentum
if 'price_momentum_3y' in all_stocks_features.columns:
    mom_3y = all_stocks_features['price_momentum_3y'].dropna()
    if not mom_3y.empty:
        mom_3y_clipped = mom_3y.clip(mom_3y.quantile(0.05), mom_3y.quantile(0.95))
        fig.add_trace(go.Histogram(x=mom_3y_clipped, nbinsx=50, marker_color='#3498db'), row=1, col=1)

# 5Y Momentum
if 'price_momentum_5y' in all_stocks_features.columns:
    mom_5y = all_stocks_features['price_momentum_5y'].dropna()
    if not mom_5y.empty:
        mom_5y_clipped = mom_5y.clip(mom_5y.quantile(0.05), mom_5y.quantile(0.95))
        fig.add_trace(go.Histogram(x=mom_5y_clipped, nbinsx=50, marker_color='#9b59b6'), row=1, col=2)

# Long-term trend score
if 'long_term_trend_score' in all_stocks_features.columns:
    lts = all_stocks_features['long_term_trend_score'].dropna()
    if not lts.empty:
        lts_clipped = lts.clip(lts.quantile(0.05), lts.quantile(0.95))
        fig.add_trace(go.Histogram(x=lts_clipped, nbinsx=50, marker_color='#00bc8c'), row=2, col=1)

# Secular trend by sector
if 'secular_trend_flag' in all_stocks_features.columns:
    secular_by_sector = all_stocks_features.groupby('industry')['secular_trend_flag'].sum().sort_values()
    fig.add_trace(go.Bar(x=secular_by_sector.values, y=secular_by_sector.index,
                         orientation='h', marker_color='#f39c12'), row=2, col=2)

fig.update_layout(height=700, title_text="Long-Term Momentum Analytics (NEW)",
                  showlegend=False, template=PLOTLY_TEMPLATE)


print(f"\n📈 Long-Term Trend Summary:")
if 'price_momentum_3y' in all_stocks_features.columns:
    print(f"Stocks with positive 3Y momentum: {(all_stocks_features['price_momentum_3y'] > 0).sum():,}")
if 'price_momentum_5y' in all_stocks_features.columns:
    print(f"Stocks with positive 5Y momentum: {(all_stocks_features['price_momentum_5y'] > 0).sum():,}")
if 'secular_trend_flag' in all_stocks_features.columns:
    print(f"Stocks in secular uptrend: {all_stocks_features['secular_trend_flag'].sum():,}")
if 'multi_year_high_flag' in all_stocks_features.columns:
    print(f"Multi-year high flags: {all_stocks_features['multi_year_high_flag'].sum():,}")

fig.show()

# %% [markdown]
# # ---
# # ## Gross Profit Temporal Analysis (NEW)
# 
# %%
# Cell: Gross Profit Trends - Full Historical Coverage

gp_cols = ['ticker', 'industry', 'gp_fq', 'gp_fy', 'gp_ltm_temp',
           'gp_qoq_growth', 'gp_yoy_growth', 'gp_margin_fq',
           'gp_margin_trend', 'gp_positive_quarters', 'gp_margin_expansion_temp']

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'Gross Profit QoQ Growth Distribution',
    'Gross Profit YoY Growth Distribution',
    'GP Margin FQ Distribution',
    'GP Margin Expansion by Industry'
])

# QoQ Growth
if 'gp_qoq_growth' in all_stocks_features.columns:
    gp_qoq = all_stocks_features['gp_qoq_growth'].dropna()
    if not gp_qoq.empty:
        gp_qoq_clipped = gp_qoq.clip(gp_qoq.quantile(0.05), gp_qoq.quantile(0.95))
        fig.add_trace(go.Histogram(x=gp_qoq_clipped, nbinsx=50, marker_color='#00bc8c'), row=1, col=1)

# YoY Growth
if 'gp_yoy_growth' in all_stocks_features.columns:
    gp_yoy = all_stocks_features['gp_yoy_growth'].dropna()
    if not gp_yoy.empty:
        gp_yoy_clipped = gp_yoy.clip(gp_yoy.quantile(0.05), gp_yoy.quantile(0.95))
        fig.add_trace(go.Histogram(x=gp_yoy_clipped, nbinsx=50, marker_color='#3498db'), row=1, col=2)

# Margin FQ
if 'gp_margin_fq' in all_stocks_features.columns:
    gp_margin = all_stocks_features['gp_margin_fq'].dropna()
    if not gp_margin.empty:
        gp_margin_clipped = gp_margin.clip(0, 100)
        fig.add_trace(go.Histogram(x=gp_margin_clipped, nbinsx=50, marker_color='#e74c3c'), row=2, col=1)

# Margin expansion by sector
if 'gp_margin_expansion_temp' in all_stocks_features.columns:
    expansion_by_sector = all_stocks_features.groupby('industry')['gp_margin_expansion_temp'].mean() * 100
    fig.add_trace(go.Bar(x=expansion_by_sector.values, y=expansion_by_sector.index,
                         orientation='h', marker_color='#f39c12'), row=2, col=2)

fig.update_layout(height=700, title_text="Gross Profit Temporal Analysis (NEW)",
                  showlegend=False, template=PLOTLY_TEMPLATE)


print(f"\n📊 Gross Profit Summary:")
if 'gp_margin_fq' in all_stocks_features.columns:
    print(f"Avg GP Margin (FQ): {all_stocks_features['gp_margin_fq'].mean():.1f}%")
if 'gp_margin_expansion_temp' in all_stocks_features.columns:
    print(f"Stocks with GP margin expansion: {all_stocks_features['gp_margin_expansion_temp'].sum():,}")
if 'gp_positive_quarters' in all_stocks_features.columns:
    print(f"Stocks with 5 positive GP quarters: {(all_stocks_features['gp_positive_quarters'] == 5).sum():,}")

fig.show()

# %% [markdown]
# # ---
# # ## Debt Dynamics & Deleveraging Analysis (NEW)
# 
# %%
# Cell: Comprehensive Debt Trend Analysis

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'Debt QoQ Change Distribution',
    'Debt YoY Change Distribution',
    'Debt 3Y CAGR Distribution',
    'Deleveraging Stocks by Industry'
])

# Debt QoQ
if 'debt_qoq_change' in all_stocks_features.columns:
    debt_qoq = all_stocks_features['debt_qoq_change'].dropna()
    if not debt_qoq.empty:
        debt_qoq_clipped = debt_qoq.clip(debt_qoq.quantile(0.05), debt_qoq.quantile(0.95))
        fig.add_trace(go.Histogram(x=debt_qoq_clipped, nbinsx=50, marker_color='#e74c3c'), row=1, col=1)

# Debt YoY
if 'debt_yoy_change' in all_stocks_features.columns:
    debt_yoy = all_stocks_features['debt_yoy_change'].dropna()
    if not debt_yoy.empty:
        debt_yoy_clipped = debt_yoy.clip(debt_yoy.quantile(0.05), debt_yoy.quantile(0.95))
        fig.add_trace(go.Histogram(x=debt_yoy_clipped, nbinsx=50, marker_color='#9b59b6'), row=1, col=2)

# Debt CAGR 3Y
if 'debt_3y_cagr' in all_stocks_features.columns:
    debt_cagr = all_stocks_features['debt_3y_cagr'].dropna()
    if not debt_cagr.empty:
        debt_cagr_clipped = debt_cagr.clip(debt_cagr.quantile(0.05), debt_cagr.quantile(0.95))
        fig.add_trace(go.Histogram(x=debt_cagr_clipped, nbinsx=50, marker_color='#3498db'), row=2, col=1)

# Deleveraging by sector
if 'debt_deleveraging' in all_stocks_features.columns:
    deleveraging_by_sector = all_stocks_features.groupby('industry')['debt_deleveraging'].sum().sort_values()
    fig.add_trace(go.Bar(x=deleveraging_by_sector.values, y=deleveraging_by_sector.index,
                         orientation='h', marker_color='#00bc8c'), row=2, col=2)

fig.update_layout(height=700, title_text="Debt Dynamics Analysis (NEW)",
                  showlegend=False, template=PLOTLY_TEMPLATE)

print(f"\n💰 Debt Dynamics Summary:")
if 'debt_deleveraging' in all_stocks_features.columns:
    print(f"Stocks actively deleveraging: {all_stocks_features['debt_deleveraging'].sum():,}")
if 'debt_3y_cagr' in all_stocks_features.columns:
    print(f"Avg Debt 3Y CAGR: {all_stocks_features['debt_3y_cagr'].mean():.1f}%")
if 'debt_to_equity_trend' in all_stocks_features.columns:
    print(f"Stocks with negative debt trend: {(all_stocks_features['debt_to_equity_trend'] < 0).sum():,}")

fig.show()

# %% [markdown]
# # ---
# # ## Composite Quality Scores Deep Dive
# 
# %%
# Cell 25: Piotroski F-Score Analysis
# Use features from the main dataframe

composite_scores = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in composite_scores.columns and 'industry' not in composite_scores.columns:
    composite_scores['industry'] = composite_scores['industry']

# Piotroski F-Score Distribution
fscore_counts = composite_scores['piotroski_f_score'].value_counts().sort_index()

fig = go.Figure(data=[
    go.Bar(x=fscore_counts.index, y=fscore_counts.values,
           marker_color=['#e74c3c' if x <= 3 else '#f39c12' if x <= 6 else '#00bc8c'
                         for x in fscore_counts.index],
           text=fscore_counts.values, textposition='outside')
])
fig.update_layout(
    title='Piotroski F-Score Distribution (9 = Best)',
    xaxis_title='F-Score',
    yaxis_title='Number of Stocks',
    template=PLOTLY_TEMPLATE
)

print("\n📊 Piotroski F-Score Summary:")
print(f"Strong (7-9): {(composite_scores['piotroski_f_score'] >= 7).sum():,} stocks")
print(
    f"Moderate (4-6): {((composite_scores['piotroski_f_score'] >= 4) & (composite_scores['piotroski_f_score'] < 7)).sum():,} stocks")
print(f"Weak (0-3): {(composite_scores['piotroski_f_score'] < 4).sum():,} stocks")

fig.show()

# %%
# Cell 26: Quality-Momentum Composite Interactive Scatter
fig = px.scatter(
    composite_scores,
    x='dilution_score',
    y='long_term_trend_score',
    color='piotroski_f_score',
    size='market_cap',
    hover_data=['ticker', 'name', 'industry', 'eps_trajectory_score'],
    title='Quality-Momentum vs Dilution Score (color = Piotroski F-Score)',
    labels={'dilution_score': 'Dilution Score (100=Buyback, 0=Heavy Dilution)',
            'long_term_trend_score': 'Quality-Momentum Composite'},
    color_continuous_scale='RdYlGn'
)
fig.update_traces(marker=dict(opacity=0.7, sizemin=4))
fig.show()

# %%
# Cell 27: Industry-wise Composite Score Heatmap
score_cols = ['piotroski_f_score', 'eps_trajectory_score', 'dilution_score', 'accounting_quality_score',
              'balance_sheet_strength', 'asset_quality_score', 'long_term_trend_score']
sector_scores = composite_scores.groupby('industry')[score_cols].mean()

fig = px.imshow(
    sector_scores.T,
    labels=dict(x="sector", y="Score Type", color="Average Score"),
    title='Average Composite Scores by Industry',
    aspect='auto',
    height=400,
    color_continuous_scale='RdYlGn'
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# %% [markdown]
# # ---
# # ## Price Target Dynamics & Analyst Revisions
# 
# %%
# Cell 28: Price Target Momentum Analysis
# Use features from the main dataframe

pt_dynamics = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in pt_dynamics.columns and 'industry' not in pt_dynamics.columns:
    pt_dynamics['industry'] = pt_dynamics['industry']

# Price Target Momentum Comparison (1W vs 1M vs 3M)
momentum_cols = ['pt_momentum_1w', 'pt_momentum_1m', 'pt_momentum_3m']
momentum_data = pt_dynamics[momentum_cols].dropna()

fig = go.Figure()
for col, color in zip(momentum_cols, ['#3498db', '#e74c3c', '#00bc8c']):
    data = momentum_data[col].clip(-0.5, 0.5)  # Clip extremes for visualization
    fig.add_trace(go.Box(y=data, name=col.replace('pt_momentum_', '').upper(),
                         marker_color=color))

fig.update_layout(
    title='Price Target Momentum Distribution (1W, 1M, 3M)',
    yaxis_title='Momentum (%)',
    template=PLOTLY_TEMPLATE
)
fig.show()

# %%
# Cell 29: Consensus Convergence Analysis
# Positive convergence = analysts becoming more aligned
fig = px.scatter(
    pt_dynamics,
    x='price_target_spread_pct',
    y='price_momentum_3m',
    color='industry',
    size='market_cap',
    hover_data=['ticker', 'name', 'region', 'country', 'trading_country', 'sector', 'industry', 'exchange',
                'last_price', 'price_target', 'analyst_coverage_change_3m', 'price_target_spread_pct',
                'price_momentum_3m'],
    title='Price Target Spread vs 3-Month Price Momentum',
    labels={'price_target_spread_pct': 'Price Target Spread',
            'price_momentum_3m': '3-Month Price Momentum'}
)
fig.add_hline(y=0, line_dash="dash", line_color="white", annotation_text="No Change")
fig.add_vline(x=0, line_dash="dash", line_color="white")
fig.update_traces(marker=dict(size=8, opacity=0.6))
fig.update_layout(
    template=PLOTLY_TEMPLATE,
    title='Price Target Spread vs 3-Month Price Momentum',
    xaxis_title='Price Target Spread',
    yaxis_title='3-Month Price Momentum'
)
fig.show()

# %%
# Cell 30: Analyst Coverage Changes
coverage_changes = pt_dynamics[['analyst_coverage_change_1m', 'analyst_coverage_change_3m']].dropna()

fig = make_subplots(rows=1, cols=2, subplot_titles=['1M Coverage Change', '3M Coverage Change'])

for i, col in enumerate(['analyst_coverage_change_1m', 'analyst_coverage_change_3m']):
    data = coverage_changes[col]
    fig.add_trace(
        go.Histogram(x=data, nbinsx=30, marker_color='#9b59b6' if i == 0 else '#1abc9c'),
        row=1, col=i + 1
    )

fig.update_layout(height=400, title_text="Analyst Coverage Changes",
                  showlegend=False, template=PLOTLY_TEMPLATE)

print("\n📊 Analyst Coverage Change Summary:")
print(f"Gaining Coverage (3M): {(pt_dynamics['analyst_coverage_change_3m'] > 0).sum():,} stocks")
print(f"Losing Coverage (3M): {(pt_dynamics['analyst_coverage_change_3m'] < 0).sum():,} stocks")

fig.show()

# %% [markdown]
# # ---
# # ## Interactive Multi-Factor Screener
# 
# %%
# Cell: Enhanced Multi-Factor Screener (Updated with new features)

def create_enhanced_screener(
        min_fscore: int = 5,
        min_quality_momentum: float = 40,
        max_distress_risk: float = 70,
        min_eps_trajectory: float = 40,
        min_fcf_positive_years: int = 3,
        require_deleveraging: bool = False,
        require_secular_trend: bool = False,
        sector_filter: str = 'All'
) -> pd.DataFrame:
    """
    Enhanced stock screener with new v2 features.
    
    Args:
        min_fscore: Minimum Piotroski F-Score (0-9)
        min_quality_momentum: Minimum quality momentum score
        max_distress_risk: Maximum distress risk score (inverted: higher = safer)
        min_eps_trajectory: Minimum EPS trajectory score
        min_fcf_positive_years: Minimum FCF positive years (0-5)
        require_deleveraging: Only stocks actively reducing debt
        require_secular_trend: Only stocks in secular uptrend
        sector_filter: Filter by sector ('All' for no filter)
    """
    df = all_stocks_features.copy()

    # Ensure necessary columns exist for filtering
    required_cols = ['piotroski_f_score', 'distress_risk_score', 'eps_trajectory_score', 'fcf_positive_years']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns for screening: {[c for c in required_cols if c not in df.columns]}")
        return pd.DataFrame()

    # Apply filters
    mask = (
            (df['piotroski_f_score'] >= min_fscore) &
            (df['distress_risk_score'] >= (100 - max_distress_risk)) &
            (df['eps_trajectory_score'] >= min_eps_trajectory) &
            (df['fcf_positive_years'] >= min_fcf_positive_years)
    )

    if require_deleveraging and 'debt_deleveraging' in df.columns:
        mask &= (df['debt_deleveraging'] == 1)

    if require_secular_trend and 'secular_trend_flag' in df.columns:
        mask &= (df['secular_trend_flag'] == 1)

    filtered = df[mask]

    if sector_filter != 'All':
        sector_col = 'industry' if 'industry' in filtered.columns else 'industry'
        filtered = filtered[filtered[sector_col] == sector_filter]

    # Sort by composite quality
    return filtered.sort_values('piotroski_f_score', ascending=False)


# Example: Quality + Deleveraging screen
quality_deleveraging = create_enhanced_screener(
    min_fscore=5,
    require_deleveraging=True,
    require_secular_trend=False,
    min_fcf_positive_years=3
)

print(f"✓ Found {len(quality_deleveraging):,} stocks meeting quality + deleveraging criteria")

display_cols = ['ticker', 'name', 'industry', 'country', 'exchange', 'piotroski_f_score',
                'distress_risk_score', 'eps_trajectory_score',
                'fcf_positive_years', 'debt_deleveraging', 'secular_trend_flag']
# Map to existing columns
final_cols = []
for c in display_cols:
    if c in quality_deleveraging.columns:
        final_cols.append(c)
    elif c.capitalize() in quality_deleveraging.columns:
        final_cols.append(c.capitalize())

display(quality_deleveraging[final_cols].head(50))

# %%
# Cell 32: Quality Factor Correlation Analysis
quality_factors = ['piotroski_f_score', 'eps_trajectory_score', 'dilution_score',
                   'long_term_trend_score', 'earnings_quality_composite_comp',
                   'distress_risk_score', 'cash_flow_quality_score']

# Get available factors from merged data
# Use all_stocks_features instead of undefined net_income_data
merged_quality = composite_scores.copy()

# Add earnings_quality_composite_comp if available in all_stocks_features
if 'earnings_quality_composite_comp' in all_stocks_features.columns:
    if 'earnings_quality_composite_comp' not in merged_quality.columns:
        merged_quality = merged_quality.merge(
            all_stocks_features[['ticker', 'earnings_quality_composite_comp']],
            on='ticker', how='left'
        )

# Add distress_risk_score if available
if 'distress_risk_score' in distress_features.columns:
    if 'distress_risk_score' not in merged_quality.columns:
        merged_quality = merged_quality.merge(
            distress_features[['ticker', 'distress_risk_score']],
            on='ticker', how='left'
        )

# Add cash flow quality if available
if 'cash_flow_quality_score' in all_stocks_features.columns:
    if 'cash_flow_quality_score' not in merged_quality.columns:
        merged_quality = merged_quality.merge(
            all_stocks_features[['ticker', 'cash_flow_quality_score']],
            on='ticker', how='left'
        )

available_factors = [f for f in quality_factors if f in merged_quality.columns]
corr_matrix = merged_quality[available_factors].corr()

fig = px.imshow(
    corr_matrix,
    labels=dict(color="Correlation"),
    title='Quality Factor Correlation Matrix',
    aspect='equal',
    height=600,
    color_continuous_scale='RdBu_r',
    zmin=-1, zmax=1
)
fig.show()

# %% [markdown]
# # ---
# # ## Comprehensive EBIT/EBITDA Analysis
# 
# %%
# Cell 33: EBIT/EBITDA Trends
# Use features from the main dataframe

ebit_ebitda_data = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in ebit_ebitda_data.columns and 'industry' not in ebit_ebitda_data.columns:
    ebit_ebitda_data['industry'] = ebit_ebitda_data['industry']

# EBITDA Margin Trends by Industry
fig = px.box(
    ebit_ebitda_data,
    x='industry',
    y='ebitda_margin_ltm',
    title='EBITDA Margin (LTM) Distribution by Industry',
    color='industry'
)
fig.update_layout(showlegend=False, xaxis_tickangle=-45)
fig.update_yaxes(range=[ebit_ebitda_data['ebitda_margin_ltm'].quantile(0.05),
                        ebit_ebitda_data['ebitda_margin_ltm'].quantile(0.95)])
fig.show()

# %%
# Cell 34: EBITDA Growth Analysis
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'EBITDA YoY Growth Distribution',
    'EBITDA QoQ Growth Distribution',
    'EBITDA 3Y CAGR Distribution',
    'EBITDA Positive Years (out of 5)'
])

# YoY Growth
yoy_data = ebit_ebitda_data['ebitda_growth_yoy'].dropna()
yoy_clipped = yoy_data.clip(yoy_data.quantile(0.05), yoy_data.quantile(0.95))
fig.add_trace(go.Histogram(x=yoy_clipped, nbinsx=50, marker_color='#00bc8c'), row=1, col=1)

# QoQ Growth
qoq_data = ebit_ebitda_data['ebitda_qoq_growth'].dropna()
qoq_clipped = qoq_data.clip(qoq_data.quantile(0.05), qoq_data.quantile(0.95))
fig.add_trace(go.Histogram(x=qoq_clipped, nbinsx=50, marker_color='#3498db'), row=1, col=2)

# 3Y CAGR
cagr_data = ebit_ebitda_data['ebitda_cagr_3y'].dropna()
cagr_clipped = cagr_data.clip(cagr_data.quantile(0.05), cagr_data.quantile(0.95))
fig.add_trace(go.Histogram(x=cagr_clipped, nbinsx=50, marker_color='#e74c3c'), row=2, col=1)

# Positive Years
pos_years = ebit_ebitda_data['ebitda_positive_years'].value_counts().sort_index()
fig.add_trace(go.Bar(x=pos_years.index.astype(str), y=pos_years.values, marker_color='#f39c12'), row=2, col=2)

fig.update_layout(height=700, title_text="EBITDA Growth Analytics",
                  showlegend=False, template=PLOTLY_TEMPLATE)
fig.show()

# %% [markdown]
# # ---
# # ## Cash Flow Comprehensive Analysis
# 
# %%
# Cell 35: Cash Flow Quality Deep Dive
# Use features from the main dataframe

cashflow_data = all_stocks_features.copy()

# Rename columns if they don't match the expected lowercase names
if 'industry' in cashflow_data.columns and 'industry' not in cashflow_data.columns:
    cashflow_data['industry'] = cashflow_data['industry']

# FCF Consistency Analysis
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    'FCF Positive Years (5Y)',
    'CFO Positive Quarters (5Q)',
    'FCF Yield Distribution',
    'Self-Funding Ratio Distribution'
])

# FCF Positive Years
fcf_pos = cashflow_data['fcf_positive_years'].value_counts().sort_index()
fig.add_trace(go.Bar(x=fcf_pos.index.astype(str), y=fcf_pos.values,
                     marker_color='#00bc8c'), row=1, col=1)

# CFO Positive Quarters
cfo_pos = cashflow_data['cfo_positive_quarters'].value_counts().sort_index()
fig.add_trace(go.Bar(x=cfo_pos.index.astype(str), y=cfo_pos.values,
                     marker_color='#3498db'), row=1, col=2)

# FCF Yield
fcf_yield = cashflow_data['fcf_yield'].dropna()
fcf_yield_clipped = fcf_yield.clip(fcf_yield.quantile(0.05), fcf_yield.quantile(0.95))
fig.add_trace(go.Histogram(x=fcf_yield_clipped, nbinsx=50, marker_color='#9b59b6'), row=2, col=1)

# Self-Funding Ratio
sf_ratio = cashflow_data['self_funding_ratio'].dropna()
sf_ratio_clipped = sf_ratio.clip(sf_ratio.quantile(0.05), sf_ratio.quantile(0.95))
fig.add_trace(go.Histogram(x=sf_ratio_clipped, nbinsx=50, marker_color='#e74c3c'), row=2, col=2)

fig.update_layout(height=700, title_text="Cash Flow Quality Metrics",
                  showlegend=False, template=PLOTLY_TEMPLATE)

print(f"\n💰 Cash Flow Summary:")
print(f"Always Positive FCF (5Y): {cashflow_data['fcf_positive_years'].eq(5).sum():,} stocks")
print(f"Self-Funding (CFO > CFI): {cashflow_data['self_funding_ratio'].gt(1).sum():,} stocks")
print(f"High Quality CF Score (≥75): {cashflow_data['cash_flow_quality_score'].ge(75).sum():,} stocks")

fig.show()

# %%
# Cell 36: Acquisition & CapEx Intensity
# Use features from the main dataframe

cf_with_sector = cashflow_data.copy()

fig = px.scatter(
    cf_with_sector,
    x='capex_vs_5y_avg',
    y='acquisition_intensity',
    color='industry',
    size='market_cap',
    hover_data=['ticker', 'self_funding_ratio', 'cash_flow_quality_score'],
    title='CapEx vs 5Y Avg vs Acquisition Intensity (LTM)',
    labels={'capex_vs_5y_avg': 'CapEx vs 5Y Average Ratio',
            'acquisition_intensity': 'Acquisition Intensity'}
)
fig.add_vline(x=1.0, line_dash="dash", line_color="white", annotation_text="Historical Avg")
fig.update_traces(marker=dict(opacity=0.6, sizemin=4))

# Handle potential missing columns for visualization
if 'capex_vs_5y_avg' in cf_with_sector.columns:
    fig.update_xaxes(range=[0, cf_with_sector['capex_vs_5y_avg'].quantile(0.95)])
if 'acquisition_intensity' in cf_with_sector.columns:
    fig.update_yaxes(range=[0, cf_with_sector['acquisition_intensity'].quantile(0.95)])
fig.show()

# %% [markdown]
# # ---
# # ## NEW: CapEx & Cash Acquisitions Temporal Analytics
# #
# # Analysis of capital expenditure trends, M&A spending patterns, and investment efficiency
# # using the enhanced calc_enhanced_cashflow_features function.
# 
# %%
# Cell: CapEx Temporal Analysis
# Comprehensive capital expenditure trend analysis

capex_features = ['capex_yoy_growth', 'capex_qoq_growth', 'capex_3y_trend', 'capex_volatility',
                  'capex_acceleration', 'capex_cut_flag', 'overinvestment_flag', 'capex_vs_5y_avg']
capex_available = [f for f in capex_features if f in all_stocks_features.columns]

if len(capex_available) >= 2:
    capex_data = all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'CapEx YoY Growth Distribution (%)',
        'CapEx vs 5Y Average Ratio',
        'CapEx 3Y Trend Distribution (%)',
        'CapEx Investment Flags by Industry'
    ])

    # CapEx YoY Growth
    if 'capex_yoy_growth' in capex_data.columns:
        capex_yoy = capex_data['capex_yoy_growth'].dropna()
        capex_yoy_clipped = capex_yoy.clip(capex_yoy.quantile(0.05), capex_yoy.quantile(0.95))
        fig.add_trace(go.Histogram(x=capex_yoy_clipped, nbinsx=50, marker_color='#3498db', name='YoY %'), row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="white", row=1, col=1)

    # CapEx vs 5Y Average
    if 'capex_vs_5y_avg' in capex_data.columns:
        capex_ratio = capex_data['capex_vs_5y_avg'].dropna()
        capex_ratio_clipped = capex_ratio.clip(0, capex_ratio.quantile(0.95))
        fig.add_trace(go.Histogram(x=capex_ratio_clipped, nbinsx=50, marker_color='#00bc8c', name='vs 5Y'), row=1,
                      col=2)
        fig.add_vline(x=1.0, line_dash="dash", line_color="yellow", annotation_text="Historical Avg", row=1, col=2)

    # CapEx 3Y Trend
    if 'capex_3y_trend' in capex_data.columns:
        capex_3y = capex_data['capex_3y_trend'].dropna()
        capex_3y_clipped = capex_3y.clip(capex_3y.quantile(0.05), capex_3y.quantile(0.95))
        fig.add_trace(go.Histogram(x=capex_3y_clipped, nbinsx=50, marker_color='#9b59b6', name='3Y Trend'), row=2,
                      col=1)

    # Investment Flags by Industry
    flag_cols = ['underinvestment_flag', 'overinvestment_flag', 'capex_acceleration', 'capex_cut_flag']
    available_flags = [c for c in flag_cols if c in capex_data.columns]
    if available_flags:
        flag_summary = capex_data.groupby('industry')[available_flags].sum()
        total_flags = flag_summary.sum(axis=1).sort_values(ascending=True)
        fig.add_trace(go.Bar(x=total_flags.values, y=total_flags.index, orientation='h',
                             marker_color='#f39c12', name='Total Flags'), row=2, col=2)

    fig.update_layout(height=700, title_text="CapEx Temporal Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

  # Summary Statistics
    print("\n📊 CapEx Investment Summary:")
    if 'capex_yoy_growth' in capex_data.columns:
        increasing = (capex_data['capex_yoy_growth'] > 0).sum()
        decreasing = (capex_data['capex_yoy_growth'] < 0).sum()
        print(f"CapEx increasing YoY: {increasing:,} stocks ({increasing / len(capex_data) * 100:.1f}%)")
        print(f"CapEx decreasing YoY: {decreasing:,} stocks ({decreasing / len(capex_data) * 100:.1f}%)")
    if 'underinvestment_flag' in capex_data.columns:
        underinvest = capex_data['underinvestment_flag'].sum()
        print(f"Underinvestment warnings (CapEx <70% of 5Y avg): {underinvest:,}")
    if 'overinvestment_flag' in capex_data.columns:
        overinvest = capex_data['overinvestment_flag'].sum()
        print(f"Overinvestment signals (CapEx >150% of 5Y avg): {overinvest:,}")
    if 'capex_acceleration' in capex_data.columns:
        accelerating = capex_data['capex_acceleration'].sum()
        print(f"CapEx accelerating (3 consecutive years): {accelerating:,}")
    if 'capex_cut_flag' in capex_data.columns:
        cutting = capex_data['capex_cut_flag'].sum()
        print(f"Significant CapEx cuts (>25% decline): {cutting:,}")
else:
    print(f"⚠️ CapEx features not available. Found: {capex_available}")
# %%
# Cell: Cash Acquisitions Temporal Analysis
# M&A spending patterns and acquisition activity tracking

acq_features = ['acquisitions_yoy_growth', 'acquisitions_vs_5y_avg', 'acquisitions_ltm_total',
                'ma_intensity_score', 'serial_acquirer_flag', 'acquisition_pause_flag']
acq_available = [f for f in acq_features if f in all_stocks_features.columns]

if len(acq_available) >= 2:
    acq_data = all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Cash Acquisitions LTM Distribution',
        'M&A Intensity Score (% of Assets)',
        'Acquisitions vs 5Y Average Ratio',
        'M&A Activity Flags by Industry'
    ])

    # Acquisitions LTM Total
    if 'acquisitions_ltm_total' in acq_data.columns:
        acq_ltm = acq_data['acquisitions_ltm_total'].dropna()
        acq_ltm_positive = acq_ltm[acq_ltm > 0]
        if len(acq_ltm_positive) > 0:
            acq_ltm_clipped = acq_ltm_positive.clip(0, acq_ltm_positive.quantile(0.95))
            fig.add_trace(go.Histogram(x=acq_ltm_clipped, nbinsx=50, marker_color='#e74c3c', name='LTM $'), row=1,
                          col=1)

    # M&A Intensity Score
    if 'ma_intensity_score' in acq_data.columns:
        ma_int = acq_data['ma_intensity_score'].dropna()
        ma_int_clipped = ma_int.clip(0, ma_int.quantile(0.95))
        fig.add_trace(go.Histogram(x=ma_int_clipped, nbinsx=50, marker_color='#9b59b6', name='Intensity %'), row=1,
                      col=2)

    # Acquisitions vs 5Y Average
    if 'acquisitions_vs_5y_avg' in acq_data.columns:
        acq_ratio = acq_data['acquisitions_vs_5y_avg'].dropna()
        acq_ratio_clipped = acq_ratio.clip(0, acq_ratio.quantile(0.95))
        fig.add_trace(go.Histogram(x=acq_ratio_clipped, nbinsx=50, marker_color='#3498db', name='vs 5Y'), row=2, col=1)

    # M&A Flags by Industry
    ma_flag_cols = ['serial_acquirer_flag', 'acquisition_pause_flag']
    available_ma_flags = [c for c in ma_flag_cols if c in acq_data.columns]
    if available_ma_flags:
        ma_flag_summary = acq_data.groupby('industry')[available_ma_flags].sum()
        serial_acquirers = ma_flag_summary.get('serial_acquirer_flag', pd.Series()).sort_values(ascending=True)
        if len(serial_acquirers) > 0:
            fig.add_trace(go.Bar(x=serial_acquirers.values, y=serial_acquirers.index, orientation='h',
                                 marker_color='#00bc8c', name='Serial Acquirers'), row=2, col=2)

    fig.update_layout(height=700, title_text="Cash Acquisitions Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

        # Summary Statistics
    print("\n🏢 Cash Acquisitions Summary:")
    if 'acquisitions_ltm_total' in acq_data.columns:
        active_acquirers = (acq_data['acquisitions_ltm_total'] > 0).sum()
        print(f"Companies with LTM acquisitions: {active_acquirers:,}")
    if 'serial_acquirer_flag' in acq_data.columns:
        serial = acq_data['serial_acquirer_flag'].sum()
        print(f"Serial acquirers (M&A in 3+ of last 4 years): {serial:,}")
    if 'acquisition_pause_flag' in acq_data.columns:
        paused = acq_data['acquisition_pause_flag'].sum()
        print(f"M&A pause (no recent activity after history): {paused:,}")
    if 'ma_intensity_score' in acq_data.columns:
        high_intensity = (acq_data['ma_intensity_score'] > 5).sum()
        print(f"High M&A intensity (>5% of assets): {high_intensity:,}")
else:
    print(f"⚠️ Acquisitions features not available. Found: {acq_available}")
# %%
# Cell: Combined Investment Efficiency Analysis
# Organic vs Inorganic growth and investment efficiency metrics

investment_features = ['total_investment_to_cfo', 'organic_vs_inorganic', 'investment_efficiency',
                       'capex_vs_5y_avg', 'acquisition_to_fcf']
investment_available = [f for f in investment_features if f in all_stocks_features.columns]

if len(investment_available) >= 2:
    inv_data = all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Total Investment to CFO Ratio',
        'Organic vs Inorganic Growth (CapEx/Acquisitions)',
        'Investment Efficiency (Revenue Growth per $)',
        'Investment Strategy by Industry'
    ])

    # Total Investment to CFO
    if 'total_investment_to_cfo' in inv_data.columns:
        inv_cfo = inv_data['total_investment_to_cfo'].dropna()
        inv_cfo_clipped = inv_cfo.clip(0, inv_cfo.quantile(0.95))
        fig.add_trace(go.Histogram(x=inv_cfo_clipped, nbinsx=50, marker_color='#e74c3c', name='Inv/CFO'), row=1, col=1)
        fig.add_vline(x=1.0, line_dash="dash", line_color="yellow", annotation_text="100% of CFO", row=1, col=1)

    # Organic vs Inorganic
    if 'organic_vs_inorganic' in inv_data.columns:
        org_inorg = inv_data['organic_vs_inorganic'].dropna()
        org_inorg_clipped = org_inorg.clip(0, org_inorg.quantile(0.95))
        fig.add_trace(go.Histogram(x=org_inorg_clipped, nbinsx=50, marker_color='#00bc8c', name='CapEx/Acq'), row=1,
                      col=2)
        fig.add_vline(x=1.0, line_dash="dash", line_color="yellow", annotation_text="Equal Split", row=1, col=2)

    # Investment Efficiency
    if 'investment_efficiency' in inv_data.columns:
        inv_eff = inv_data['investment_efficiency'].dropna()
        inv_eff_clipped = inv_eff.clip(inv_eff.quantile(0.05), inv_eff.quantile(0.95))
        fig.add_trace(go.Histogram(x=inv_eff_clipped, nbinsx=50, marker_color='#3498db', name='Efficiency'), row=2,
                      col=1)

    # Investment Strategy by Industry (scatter: organic vs inorganic colored by efficiency)
    if 'capex_vs_5y_avg' in inv_data.columns and 'acquisition_to_fcf' in inv_data.columns:
        sector_investment = inv_data.groupby('industry').agg({
            'capex_vs_5y_avg': 'median',
            'acquisition_to_fcf': 'median'
        }).dropna()
        fig.add_trace(go.Bar(x=sector_investment['capex_vs_5y_avg'].values,
                             y=sector_investment.index, orientation='h',
                             marker_color='#9b59b6', name='CapEx Ratio'), row=2, col=2)

    fig.update_layout(height=700, title_text="Investment Efficiency Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary Statistics
    print("\n📈 Investment Efficiency Summary:")
    if 'total_investment_to_cfo' in inv_data.columns:
        high_investment = (inv_data['total_investment_to_cfo'] > 1).sum()
        print(f"High investment (>100% of CFO): {high_investment:,} stocks")
        print(f"Median Investment/CFO ratio: {inv_data['total_investment_to_cfo'].median():.2f}")
    if 'organic_vs_inorganic' in inv_data.columns:
        organic_focused = (inv_data['organic_vs_inorganic'] > 2).sum()
        acq_focused = (inv_data['organic_vs_inorganic'] < 0.5).sum()
        print(f"Organic-focused (CapEx > 2x Acquisitions): {organic_focused:,}")
        print(f"Acquisition-focused (Acquisitions > 2x CapEx): {acq_focused:,}")
    if 'investment_efficiency' in inv_data.columns:
        efficient = (inv_data['investment_efficiency'] > 1).sum()
        print(f"High efficiency (revenue growth > investment): {efficient:,}")
        print(f"Median Investment Efficiency: {inv_data['investment_efficiency'].median():.2f}")
else:
    print(f"⚠️ Investment features not available. Found: {investment_available}")

# %%
# Cell: CapEx & Acquisitions Interactive Scatter
# Investment strategy visualization

if all(col in all_stocks_features.columns for col in ['capex_vs_5y_avg', 'acquisition_to_fcf', 'industry']):
    scatter_data = all_stocks_features[['ticker', 'name', 'industry', 'market_cap',
                                        'capex_vs_5y_avg', 'acquisition_to_fcf',
                                        'fcf_positive_years', 'cash_flow_quality_score']].dropna(
        subset=['capex_vs_5y_avg', 'acquisition_to_fcf'])

    # Clip for visualization
    scatter_data = scatter_data[
        (scatter_data['capex_vs_5y_avg'] > 0) &
        (scatter_data['capex_vs_5y_avg'] < scatter_data['capex_vs_5y_avg'].quantile(0.95)) &
        (scatter_data['acquisition_to_fcf'] < scatter_data['acquisition_to_fcf'].quantile(0.95))
        ]

    fig = px.scatter(
        scatter_data,
        x='capex_vs_5y_avg',
        y='acquisition_to_fcf',
        color='industry',
        size='market_cap',
        hover_data=['ticker', 'name', 'fcf_positive_years', 'cash_flow_quality_score'],
        title='Investment Strategy: CapEx vs M&A Focus (Color = Industry, Size = Market Cap)',
        labels={
            'capex_vs_5y_avg': 'CapEx vs 5Y Average',
            'acquisition_to_fcf': 'Acquisitions / FCF Ratio'
        },
        height=600
    )

    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Sustainable M&A (<50% FCF)")
    fig.add_vline(x=1.0, line_dash="dash", line_color="yellow", annotation_text="Historical CapEx Avg")
    fig.add_vline(x=0.7, line_dash="dot", line_color="red", annotation_text="Underinvestment")
    fig.add_vline(x=1.5, line_dash="dot", line_color="orange", annotation_text="Overinvestment")

    # Add quadrant annotations
    fig.add_annotation(x=0.5, y=0.2, text="Low Investment", showarrow=False,
                       font=dict(color="gray", size=10))
    fig.add_annotation(x=2.0, y=1.5, text="Aggressive Growth", showarrow=False,
                       font=dict(color="orange", size=10))
    fig.add_annotation(x=1.2, y=0.2, text="Organic Focus", showarrow=False,
                       font=dict(color="green", size=10))
    fig.add_annotation(x=0.5, y=1.5, text="M&A Focus", showarrow=False,
                       font=dict(color="blue", size=10))

    fig.update_traces(marker=dict(opacity=0.6, sizemin=4))
    fig.update_layout(template=PLOTLY_TEMPLATE)
    fig.show()

# %% [markdown]
# # ---
# # ## NEW: EPS Continuing Operations Analytics
# #
# # Analysis of earnings from continuing operations vs total EPS, identifying companies with
# # significant discontinued operations impact and core earnings quality.
# 
# %%
# Cell: EPS Continuing Operations Analysis
# Uses the new calc_eps_continuing_features function outputs

eps_cont_features = ['eps_cont_ltm', 'eps_cont_fy', 'eps_cont_yoy_growth',
                     'eps_cont_vs_total_eps', 'discontinued_ops_impact']
eps_cont_available = [f for f in eps_cont_features if f in all_stocks_features.columns]

if len(eps_cont_available) >= 3:
    eps_cont_data = all_stocks_features[['ticker', 'name', 'industry', 'region'] + eps_cont_available].dropna(
        subset=eps_cont_available[:2])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'EPS Continuing vs Total EPS Ratio',
        'Discontinued Operations Impact (%)',
        'EPS Continuing YoY Growth Distribution',
        'EPS Continuing by Industry'
    ])

    # EPS Continuing vs Total ratio
    if 'eps_cont_vs_total_eps' in eps_cont_data.columns:
        ratio_data = eps_cont_data['eps_cont_vs_total_eps'].dropna()
        ratio_clipped = ratio_data.clip(ratio_data.quantile(0.05), ratio_data.quantile(0.95))
        fig.add_trace(go.Histogram(x=ratio_clipped, nbinsx=50, marker_color='#00bc8c', name='Cont/Total'), row=1, col=1)

    # Discontinued ops impact
    if 'discontinued_ops_impact' in eps_cont_data.columns:
        disc_impact = eps_cont_data['discontinued_ops_impact'].dropna()
        disc_clipped = disc_impact.clip(disc_impact.quantile(0.05), disc_impact.quantile(0.95))
        fig.add_trace(go.Histogram(x=disc_clipped, nbinsx=50, marker_color='#e74c3c', name='Disc Ops %'), row=1, col=2)

    # YoY Growth
    if 'eps_cont_yoy_growth' in eps_cont_data.columns:
        yoy_data = eps_cont_data['eps_cont_yoy_growth'].dropna()
        yoy_clipped = yoy_data.clip(yoy_data.quantile(0.05), yoy_data.quantile(0.95))
        fig.add_trace(go.Histogram(x=yoy_clipped, nbinsx=50, marker_color='#3498db', name='YoY Growth'), row=2, col=1)

    # By Industry
    if 'eps_cont_ltm' in eps_cont_data.columns:
        sector_avg = eps_cont_data.groupby('industry')['eps_cont_ltm'].median().sort_values()
        fig.add_trace(go.Bar(x=sector_avg.values, y=sector_avg.index, orientation='h',
                             marker_color='#9b59b6', name='Median EPS Cont'), row=2, col=2)

    fig.update_layout(height=700, title_text="EPS Continuing Operations Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary statistics
    print("\n📊 EPS Continuing Operations Summary:")
    print(f"Stocks with EPS Continuing data: {len(eps_cont_data):,}")
    if 'eps_cont_vs_total_eps' in eps_cont_data.columns:
        high_disc_ops = (eps_cont_data['eps_cont_vs_total_eps'].abs() < 0.8).sum()
        print(f"Stocks with significant discontinued ops (ratio < 0.8): {high_disc_ops:,}")
    if 'discontinued_ops_impact' in eps_cont_data.columns:
        material_impact = (eps_cont_data['discontinued_ops_impact'].abs() > 10).sum()
        print(f"Stocks with material discontinued ops impact (>10%): {material_impact:,}")
    print(f"\n📈 EPS Continuing Statistics:")
    print(eps_cont_data[eps_cont_available].describe().T.round(2).to_string())
else:
    print(f"⚠️ EPS Continuing features not available. Found: {eps_cont_available}")

# %% [markdown]
# # ---
# # ## NEW: R&D Investment Analytics
# #
# # Analysis of R&D spending patterns, intensity metrics, and innovation investment trends
# # using the new calc_rnd_temporal_features function.
# 
# %%
# Cell: R&D Investment Analysis
# Comprehensive R&D spending and intensity analysis

rnd_features = ['rnd_intensity_ltm', 'rnd_yoy_growth', 'rnd_per_employee', 'high_rnd_intensity_flag',
                'rnd_fy', 'rnd_1fy', 'rnd_2fy', 'rnd_3fy']
rnd_available = [f for f in rnd_features if f in all_stocks_features.columns]

if len(rnd_available) >= 2:
    # Filter to companies with R&D data
    rnd_data = all_stocks_features[all_stocks_features['rnd_intensity_ltm'].notna() &
                                   (all_stocks_features[
                                        'rnd_intensity_ltm'] > 0)].copy() if 'rnd_intensity_ltm' in all_stocks_features.columns else all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'R&D Intensity Distribution (% of Revenue)',
        'R&D YoY Growth Distribution',
        'R&D Intensity by Industry',
        'R&D per Employee by Industry'
    ])

    # R&D Intensity distribution
    if 'rnd_intensity_ltm' in rnd_data.columns:
        intensity = rnd_data['rnd_intensity_ltm'].dropna()
        intensity_clipped = intensity.clip(0, intensity.quantile(0.95))
        fig.add_trace(go.Histogram(x=intensity_clipped, nbinsx=50, marker_color='#00bc8c', name='R&D %'), row=1, col=1)

    # R&D YoY Growth
    if 'rnd_yoy_growth' in rnd_data.columns:
        rnd_growth = rnd_data['rnd_yoy_growth'].dropna()
        rnd_growth_clipped = rnd_growth.clip(rnd_growth.quantile(0.05), rnd_growth.quantile(0.95))
        fig.add_trace(go.Histogram(x=rnd_growth_clipped, nbinsx=50, marker_color='#3498db', name='YoY %'), row=1, col=2)

    # R&D Intensity by Industry
    if 'rnd_intensity_ltm' in rnd_data.columns:
        sector_rnd = rnd_data.groupby('industry')['rnd_intensity_ltm'].median().sort_values(ascending=True)
        colors = ['#e74c3c' if v > 10 else '#f39c12' if v > 5 else '#00bc8c' for v in sector_rnd.values]
        fig.add_trace(go.Bar(x=sector_rnd.values, y=sector_rnd.index, orientation='h',
                             marker_color=colors, name='Median R&D %'), row=2, col=1)

    # R&D per Employee by Industry
    if 'rnd_per_employee' in rnd_data.columns:
        sector_rnd_emp = rnd_data.groupby('industry')['rnd_per_employee'].median().sort_values(ascending=True)
        fig.add_trace(go.Bar(x=sector_rnd_emp.values, y=sector_rnd_emp.index, orientation='h',
                             marker_color='#9b59b6', name='R&D/Employee'), row=2, col=2)

    fig.update_layout(height=700, title_text="R&D Investment Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # High R&D intensity companies
    print("\n🔬 R&D Investment Summary:")
    if 'high_rnd_intensity_flag' in rnd_data.columns:
        high_rnd_count = rnd_data['high_rnd_intensity_flag'].sum()
        print(f"High R&D Intensity Companies (>10% of revenue): {high_rnd_count:,}")
    if 'rnd_intensity_ltm' in rnd_data.columns:
        print(f"Average R&D Intensity: {rnd_data['rnd_intensity_ltm'].mean():.2f}%")
        print(f"Median R&D Intensity: {rnd_data['rnd_intensity_ltm'].median():.2f}%")
    if 'rnd_yoy_growth' in rnd_data.columns:
        rnd_increasing = (rnd_data['rnd_yoy_growth'] > 0).sum()
        print(f"Companies increasing R&D YoY: {rnd_increasing:,} ({rnd_increasing / len(rnd_data) * 100:.1f}%)")

    # Top R&D spenders
    if 'rnd_intensity_ltm' in rnd_data.columns:
        print("\n🏆 Top 10 R&D Intensive Companies:")
        top_rnd = rnd_data.nlargest(10, 'rnd_intensity_ltm')[['ticker', 'name', 'industry', 'rnd_intensity_ltm']]
        display(top_rnd)
else:
    print(f"⚠️ R&D features not available. Found: {rnd_available}")

# %% [markdown]
# # ---
# # ## NEW: Inventory Temporal Analytics
# #
# # Analysis of inventory trends, efficiency metrics, and quality flags using the new
# # calc_inventory_temporal_features function.
# 
# %%
# Cell: Inventory Temporal Analysis
# Full inventory historical coverage and efficiency metrics

inventory_features = ['inventory_qoq_change', 'inventory_yoy_change', 'inventory_days',
                      'inventory_turnover_mv', 'inventory_buildup_flag', 'inventory_reduction_flag']
inventory_available = [f for f in inventory_features if f in all_stocks_features.columns]

if len(inventory_available) >= 2:
    # Filter to companies with inventory (exclude service companies)
    inv_data = all_stocks_features[all_stocks_features[
        'inventory_days'].notna()].copy() if 'inventory_days' in all_stocks_features.columns else all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Inventory Days Distribution',
        'Inventory Turnover Distribution',
        'Inventory QoQ Change (%)',
        'Inventory Days by Industry'
    ])

    # Inventory Days
    if 'inventory_days' in inv_data.columns:
        inv_days = inv_data['inventory_days'].dropna()
        inv_days_clipped = inv_days.clip(0, inv_days.quantile(0.95))
        fig.add_trace(go.Histogram(x=inv_days_clipped, nbinsx=50, marker_color='#00bc8c', name='Days'), row=1, col=1)

    # Inventory Turnover
    if 'inventory_turnover_mv' in inv_data.columns:
        turnover = inv_data['inventory_turnover_mv'].dropna()
        turnover_clipped = turnover.clip(0, turnover.quantile(0.95))
        fig.add_trace(go.Histogram(x=turnover_clipped, nbinsx=50, marker_color='#3498db', name='Turnover'), row=1,
                      col=2)

    # QoQ Change
    if 'inventory_qoq_change' in inv_data.columns:
        qoq = inv_data['inventory_qoq_change'].dropna()
        qoq_clipped = qoq.clip(qoq.quantile(0.05), qoq.quantile(0.95))
        fig.add_trace(go.Histogram(x=qoq_clipped, nbinsx=50, marker_color='#e74c3c', name='QoQ %'), row=2, col=1)

    # By Industry
    if 'inventory_days' in inv_data.columns:
        sector_inv = inv_data.groupby('industry')['inventory_days'].median().sort_values(ascending=True)
        fig.add_trace(go.Bar(x=sector_inv.values, y=sector_inv.index, orientation='h',
                             marker_color='#9b59b6', name='Median Days'), row=2, col=2)

    fig.update_layout(height=700, title_text="Inventory Temporal Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary
    print("\n📦 Inventory Analysis Summary:")
    if 'inventory_days' in inv_data.columns:
        print(f"Companies with inventory data: {len(inv_data):,}")
        print(f"Average Inventory Days: {inv_data['inventory_days'].mean():.1f}")
        print(f"Median Inventory Days: {inv_data['inventory_days'].median():.1f}")
    if 'inventory_buildup_flag' in inv_data.columns:
        buildup = inv_data['inventory_buildup_flag'].sum()
        print(f"Inventory Buildup Warning: {buildup:,} companies")
    if 'inventory_reduction_flag' in inv_data.columns:
        reduction = inv_data['inventory_reduction_flag'].sum()
        print(f"Inventory Reduction Trend: {reduction:,} companies")
else:
    print(f"⚠️ Inventory features not available. Found: {inventory_available}")

# %% [markdown]
# # ---
# # ## NEW: Goodwill & M&A Activity Analytics
# #
# # Analysis of goodwill trends, M&A activity tracking, and impairment risk using the new
# # calc_goodwill_temporal_features function.
# 
# %%
# Cell: Goodwill & M&A Analysis
# M&A activity tracking through goodwill changes

goodwill_features = ['goodwill_qoq_change', 'goodwill_yoy_change', 'goodwill_3y_growth',
                     'goodwill_concentration', 'recent_acquisition_flag', 'impairment_risk_score']
goodwill_available = [f for f in goodwill_features if f in all_stocks_features.columns]

if len(goodwill_available) >= 2:
    # Filter to companies with goodwill
    gw_data = all_stocks_features[all_stocks_features[
        'goodwill_concentration'].notna()].copy() if 'goodwill_concentration' in all_stocks_features.columns else all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Goodwill Concentration (% of Equity)',
        'Goodwill YoY Change (%)',
        'Goodwill 3Y Growth (%)',
        'Goodwill Concentration by Industry'
    ])

    # Goodwill Concentration
    if 'goodwill_concentration' in gw_data.columns:
        conc = gw_data['goodwill_concentration'].dropna()
        conc_clipped = conc.clip(0, conc.quantile(0.95))
        fig.add_trace(go.Histogram(x=conc_clipped, nbinsx=50, marker_color='#e74c3c', name='% Equity'), row=1, col=1)

    # YoY Change
    if 'goodwill_yoy_change' in gw_data.columns:
        yoy = gw_data['goodwill_yoy_change'].dropna()
        yoy_clipped = yoy.clip(yoy.quantile(0.05), yoy.quantile(0.95))
        fig.add_trace(go.Histogram(x=yoy_clipped, nbinsx=50, marker_color='#3498db', name='YoY %'), row=1, col=2)

    # 3Y Growth
    if 'goodwill_3y_growth' in gw_data.columns:
        growth_3y = gw_data['goodwill_3y_growth'].dropna()
        growth_3y_clipped = growth_3y.clip(growth_3y.quantile(0.05), growth_3y.quantile(0.95))
        fig.add_trace(go.Histogram(x=growth_3y_clipped, nbinsx=50, marker_color='#00bc8c', name='3Y %'), row=2, col=1)

    # By Industry
    if 'goodwill_concentration' in gw_data.columns:
        sector_gw = gw_data.groupby('industry')['goodwill_concentration'].median().sort_values(ascending=True)
        colors = ['#e74c3c' if v > 50 else '#f39c12' if v > 25 else '#00bc8c' for v in sector_gw.values]
        fig.add_trace(go.Bar(x=sector_gw.values, y=sector_gw.index, orientation='h',
                             marker_color=colors, name='Median %'), row=2, col=2)

    fig.update_layout(height=700, title_text="Goodwill & M&A Activity Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary
    print("\n🏢 Goodwill & M&A Summary:")
    if 'goodwill_concentration' in gw_data.columns:
        print(f"Companies with goodwill: {len(gw_data):,}")
        high_gw = (gw_data['goodwill_concentration'] > 50).sum()
        print(f"High goodwill concentration (>50% equity): {high_gw:,}")
    if 'recent_acquisition_flag' in gw_data.columns:
        recent_acq = gw_data['recent_acquisition_flag'].sum()
        print(f"Recent acquisitions detected (GW +20% QoQ): {recent_acq:,}")
    if 'impairment_risk_score' in gw_data.columns:
        high_risk = (gw_data['impairment_risk_score'] > 50).sum()
        print(f"High impairment risk (score >50): {high_risk:,}")

    # Top acquirers
    if 'goodwill_3y_growth' in gw_data.columns:
        print("\n🏆 Top 10 M&A Active Companies (3Y Goodwill Growth):")
        top_acq = gw_data.nlargest(10, 'goodwill_3y_growth')[
            ['ticker', 'name', 'industry', 'goodwill_3y_growth', 'goodwill_concentration']]
        display(top_acq)
else:
    print(f"⚠️ Goodwill features not available. Found: {goodwill_available}")

# %% [markdown]
# # ---
# # ## NEW: Marketing & Cost Efficiency Analytics
# #
# # Analysis of marketing expenses, SG&A efficiency, and cost structure trends using the
# # enhanced calc_cost_structure_features function.
# 
# %%
# Cell: Marketing & Cost Efficiency Analysis
# Enhanced cost structure with marketing and SG&A metrics

cost_features = ['marketing_to_revenue', 'marketing_trend_yoy', 'marketing_vs_5y_avg',
                 'sga_vs_5y_avg', 'sga_to_revenue', 'cogs_to_revenue', 'opex_to_revenue']
cost_available = [f for f in cost_features if f in all_stocks_features.columns]

if len(cost_available) >= 2:
    cost_data = all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Marketing to Revenue (%)',
        'Marketing Trend YoY (%)',
        'SG&A vs 5Y Average Ratio',
        'Cost Structure by Industry'
    ])

    # Marketing to Revenue
    if 'marketing_to_revenue' in cost_data.columns:
        mkt = cost_data['marketing_to_revenue'].dropna()
        mkt_clipped = mkt.clip(0, mkt.quantile(0.95))
        fig.add_trace(go.Histogram(x=mkt_clipped, nbinsx=50, marker_color='#f39c12', name='Mkt %'), row=1, col=1)

    # Marketing Trend
    if 'marketing_trend_yoy' in cost_data.columns:
        mkt_trend = cost_data['marketing_trend_yoy'].dropna()
        mkt_trend_clipped = mkt_trend.clip(mkt_trend.quantile(0.05), mkt_trend.quantile(0.95))
        fig.add_trace(go.Histogram(x=mkt_trend_clipped, nbinsx=50, marker_color='#3498db', name='YoY %'), row=1, col=2)

    # SG&A vs 5Y Avg
    if 'sga_vs_5y_avg' in cost_data.columns:
        sga_ratio = cost_data['sga_vs_5y_avg'].dropna()
        sga_ratio_clipped = sga_ratio.clip(sga_ratio.quantile(0.05), sga_ratio.quantile(0.95))
        fig.add_trace(go.Histogram(x=sga_ratio_clipped, nbinsx=50, marker_color='#00bc8c', name='vs 5Y'), row=2, col=1)

    # Cost Structure by Industry (stacked)
    if 'cogs_to_revenue' in cost_data.columns and 'sga_to_revenue' in cost_data.columns:
        sector_costs = cost_data.groupby('industry')[['cogs_to_revenue', 'sga_to_revenue']].median().sort_values(
            'cogs_to_revenue')
        fig.add_trace(go.Bar(x=sector_costs['cogs_to_revenue'].values, y=sector_costs.index, orientation='h',
                             marker_color='#e74c3c', name='COGS'), row=2, col=2)

    fig.update_layout(height=700, title_text="Marketing & Cost Efficiency Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary
    print("\n💰 Marketing & Cost Efficiency Summary:")
    if 'marketing_to_revenue' in cost_data.columns:
        mkt_data = cost_data['marketing_to_revenue'].dropna()
        print(f"Companies with marketing data: {len(mkt_data):,}")
        print(f"Average Marketing/Revenue: {mkt_data.mean():.2f}%")
    if 'sga_vs_5y_avg' in cost_data.columns:
        sga_improving = (cost_data['sga_vs_5y_avg'] < 1).sum()
        print(f"SG&A improving vs 5Y avg: {sga_improving:,} companies")
    if 'marketing_trend_yoy' in cost_data.columns:
        mkt_increasing = (cost_data['marketing_trend_yoy'] > 0).sum()
        print(f"Marketing spend increasing YoY: {mkt_increasing:,} companies")
else:
    print(f"⚠️ Cost structure features not available. Found: {cost_available}")

# %% [markdown]
# # ---
# # ## NEW: Tangible Book Value Analytics
# #
# # Analysis of tangible book value metrics using native TBV columns from the schema,
# # as implemented in the enhanced calc_tangible_book_features function.
# 
# %%
# Cell: Tangible Book Value Analysis
# Native TBV columns and enhanced metrics

tbv_features = ['tangible_book_value_ltm', 'tangible_book_value_fy', 'tbv_yoy_growth',
                'price_to_tangible_book', 'tangible_equity_ratio', 'tangible_asset_quality']
tbv_available = [f for f in tbv_features if f in all_stocks_features.columns]

if len(tbv_available) >= 2:
    tbv_data = all_stocks_features[all_stocks_features[
        'tangible_book_value_ltm'].notna()].copy() if 'tangible_book_value_ltm' in all_stocks_features.columns else all_stocks_features.copy()

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        'Price to Tangible Book Distribution',
        'TBV YoY Growth (%)',
        'Tangible Equity Ratio (%)',
        'P/TBV by Industry'
    ])

    # P/TBV
    if 'price_to_tangible_book' in tbv_data.columns:
        ptbv = tbv_data['price_to_tangible_book'].dropna()
        ptbv_clipped = ptbv.clip(ptbv.quantile(0.05), ptbv.quantile(0.95))
        fig.add_trace(go.Histogram(x=ptbv_clipped, nbinsx=50, marker_color='#00bc8c', name='P/TBV'), row=1, col=1)

    # TBV Growth
    if 'tbv_yoy_growth' in tbv_data.columns:
        tbv_growth = tbv_data['tbv_yoy_growth'].dropna()
        tbv_growth_clipped = tbv_growth.clip(tbv_growth.quantile(0.05), tbv_growth.quantile(0.95))
        fig.add_trace(go.Histogram(x=tbv_growth_clipped, nbinsx=50, marker_color='#3498db', name='YoY %'), row=1, col=2)

    # Tangible Equity Ratio
    if 'tangible_equity_ratio' in tbv_data.columns:
        ter = tbv_data['tangible_equity_ratio'].dropna()
        ter_clipped = ter.clip(0, ter.quantile(0.95))
        fig.add_trace(go.Histogram(x=ter_clipped, nbinsx=50, marker_color='#9b59b6', name='TE Ratio'), row=2, col=1)

    # P/TBV by Industry
    if 'price_to_tangible_book' in tbv_data.columns:
        sector_ptbv = tbv_data.groupby('industry')['price_to_tangible_book'].median().sort_values(ascending=True)
        colors = ['#00bc8c' if v < 2 else '#f39c12' if v < 5 else '#e74c3c' for v in sector_ptbv.values]
        fig.add_trace(go.Bar(x=sector_ptbv.values, y=sector_ptbv.index, orientation='h',
                             marker_color=colors, name='Median P/TBV'), row=2, col=2)

    fig.update_layout(height=700, title_text="Tangible Book Value Analysis (NEW)",
                      showlegend=False, template=PLOTLY_TEMPLATE)
    fig.show()

    # Summary
    print("\n📚 Tangible Book Value Summary:")
    if 'tangible_book_value_ltm' in tbv_data.columns:
        print(f"Companies with TBV data: {len(tbv_data):,}")
    if 'price_to_tangible_book' in tbv_data.columns:
        undervalued = (tbv_data['price_to_tangible_book'] < 1).sum()
        print(f"Trading below tangible book (P/TBV < 1): {undervalued:,}")
        print(f"Median P/TBV: {tbv_data['price_to_tangible_book'].median():.2f}")
    if 'tbv_yoy_growth' in tbv_data.columns:
        tbv_growing = (tbv_data['tbv_yoy_growth'] > 0).sum()
        print(f"TBV growing YoY: {tbv_growing:,} companies")
    if 'tangible_asset_quality' in tbv_data.columns:
        high_quality = (tbv_data['tangible_asset_quality'] > 75).sum()
        print(f"High tangible asset quality (>75): {high_quality:,}")

    # Value opportunities
    if 'price_to_tangible_book' in tbv_data.columns:
        print("\n💎 Top 10 Value Opportunities (Lowest P/TBV, positive TBV):")
        value_opps = tbv_data[tbv_data['tangible_book_value_ltm'] > 0].nsmallest(10, 'price_to_tangible_book')[
            ['ticker', 'name', 'industry', 'price_to_tangible_book', 'tbv_yoy_growth']]
        display(value_opps)
else:
    print(f"⚠️ TBV features not available. Found: {tbv_available}")

# %% [markdown]
# # ---
# # ## NEW: Feature Registry Enhancements Summary
# #
# # Overview of all new features added from the feature registry improvements.
# 
# %%
# Cell: Feature Registry Enhancements Summary
# Summary of new feature categories and their coverage

new_categories = {
    'EPS Continuing Operations': FEATURE_CATEGORIES.get('EPS Continuing Operations', []),
    'R&D Investment': FEATURE_CATEGORIES.get('R&D Investment', []),
    'Inventory Temporal': FEATURE_CATEGORIES.get('Inventory Temporal', []),
    'Goodwill & M&A': FEATURE_CATEGORIES.get('Goodwill & M&A', []),
    'Tangible Book Value': FEATURE_CATEGORIES.get('Tangible Book Value', []),
}

# Calculate coverage for new features
new_feature_coverage = []
for category, features in new_categories.items():
    for feature in features:
        if feature in all_stocks_features.columns:
            coverage = (1 - all_stocks_features[feature].isna().mean()) * 100
            new_feature_coverage.append({
                'Category': category,
                'Feature': feature,
                'Coverage %': coverage,
                'Non-Null Count': all_stocks_features[feature].notna().sum()
            })

if new_feature_coverage:
    new_coverage_df = pd.DataFrame(new_feature_coverage)

    # Summary by category
    print("📊 NEW Feature Categories Coverage Summary:")
    print("=" * 60)
    category_summary = new_coverage_df.groupby('Category').agg({
        'Feature': 'count',
        'Coverage %': 'mean',
        'Non-Null Count': 'mean'
    }).round(1)
    category_summary.columns = ['Features Available', 'Avg Coverage %', 'Avg Non-Null']
    print(category_summary.to_string())

    # Visualization
    fig = px.bar(new_coverage_df, x='Feature', y='Coverage %', color='Category',
                 title='NEW Feature Coverage from Registry Enhancements',
                 height=500)
    fig.update_layout(xaxis_tickangle=-45, showlegend=True)
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% threshold")
    fig.show()

    # Total new features
    print(f"\n✅ Total NEW features available: {len(new_coverage_df)}")
    print(f"📈 Features with >80% coverage: {(new_coverage_df['Coverage %'] > 80).sum()}")
    print(f"⚠️ Features with <50% coverage: {(new_coverage_df['Coverage %'] < 50).sum()}")
else:
    print("⚠️ No new features found in the dataset. Ensure mv_all_stock_features is refreshed.")

# %% [markdown]
# # ---
# # ## Markowitz Investor's Ruin & Conditional Risk Analysis
# 
# %%
# Cell: Gambler's Ruin / Investor's Ruin Probability Model
# Estimates probability of capital depletion based on financial metrics

import numpy as np
import pandas as pd
from scipy import stats

def calculate_ruin_probability(
    df: pd.DataFrame,
    initial_capital_col: str = 'market_cap',
    cash_burn_col: str = 'cash_burn_rate',
    volatility_col: str = 'volatility_regime'
) -> pd.DataFrame:
    """
    Calculate investor's ruin probability using a modified Gambler's Ruin framework.
    
    P(ruin) ≈ exp(-2 * μ * W / σ²) for μ > 0 (drift)
    where W = initial wealth, μ = expected return, σ = volatility
    """
    result = df[['ticker', 'name', 'industry', 'market_cap', 'distress_risk_score']].copy()
    
    # Proxy expected return from FCF yield and earnings trajectory
    if 'fcf_yield' in df.columns and 'eps_trajectory_score' in df.columns:
        # Normalize to approximate drift (μ)
        fcf_norm = df['fcf_yield'].clip(-20, 50) / 100
        eps_norm = df['eps_trajectory_score'] / 100
        result['expected_drift'] = (fcf_norm * 0.6 + eps_norm * 0.4).fillna(0)
    else:
        result['expected_drift'] = 0.05  # Default 5% drift
    
    # Volatility proxy from regime or beta
    if volatility_col in df.columns:
        result['volatility'] = df[volatility_col].abs().clip(5, 80) / 100
    elif 'beta_momentum' in df.columns:
        result['volatility'] = (df['beta_momentum'].abs() * 0.2).clip(0.1, 0.8)
    else:
        result['volatility'] = 0.25
    
    # Cash runway as wealth buffer (months → years)
    if 'cash_runway_months' in df.columns:
        result['wealth_buffer'] = df['cash_runway_months'].clip(0, 120) / 12
    else:
        result['wealth_buffer'] = 3  # Default 3 years
    
    # Calculate ruin probability using modified formula
    # P(ruin) = exp(-2μW/σ²) when μ > 0
    mu = result['expected_drift']
    sigma = result['volatility']
    W = result['wealth_buffer']
    
    # Avoid division by zero
    sigma_sq = sigma ** 2 + 1e-6
    
    # For positive drift, use exponential decay formula
    # For negative drift, ruin probability approaches 1
    result['ruin_probability'] = np.where(
        mu > 0,
        np.exp(-2 * mu * W / sigma_sq).clip(0, 1),
        np.minimum(1.0, 0.5 + 0.5 * np.abs(mu) * W)
    )
    
    # Survival probability
    result['survival_probability'] = 1 - result['ruin_probability']
    
    # Risk tier classification
    result['risk_tier'] = pd.cut(
        result['ruin_probability'],
        bins=[0, 0.1, 0.3, 0.6, 1.0],
        labels=['Low Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']
    )
    
    return result

# Calculate ruin probabilities
ruin_analysis = calculate_ruin_probability(all_stocks_features)

# Visualization
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        'Ruin Probability Distribution',
        'Survival Probability by Industry',
        'Drift vs Volatility (Risk Quadrant)',
        'Risk Tier Distribution'
    ],
    specs=[[{"type": "histogram"}, {"type": "bar"}],
           [{"type": "scatter"}, {"type": "pie"}]]
)

# Ruin probability histogram
fig.add_trace(
    go.Histogram(x=ruin_analysis['ruin_probability'], nbinsx=50, 
                 marker_color='#e74c3c', opacity=0.7),
    row=1, col=1
)

# Survival by industry
survival_by_sector = ruin_analysis.groupby('industry')['survival_probability'].mean().sort_values()
fig.add_trace(
    go.Bar(x=survival_by_sector.values, y=survival_by_sector.index,
           orientation='h', marker_color='#00bc8c'),
    row=1, col=2
)

# Risk quadrant scatter
fig.add_trace(
    go.Scatter(
        x=ruin_analysis['expected_drift'],
        y=ruin_analysis['volatility'],
        mode='markers',
        marker=dict(
            size=6,
            color=ruin_analysis['ruin_probability'],
            colorscale='RdYlGn_r',
            opacity=0.6
        ),
        text=ruin_analysis['ticker'],
        hovertemplate='%{text}<br>Drift: %{x:.2%}<br>Vol: %{y:.2%}<extra></extra>'
    ),
    row=2, col=1
)

# Risk tier pie
tier_counts = ruin_analysis['risk_tier'].value_counts()
fig.add_trace(
    go.Pie(labels=tier_counts.index, values=tier_counts.values,
           marker_colors=['#00bc8c', '#f39c12', '#e74c3c', '#8e44ad']),
    row=2, col=2
)

fig.update_layout(
    height=800,
    title_text="📉 Markowitz Investor's Ruin Analysis",
    showlegend=False,
    template=PLOTLY_TEMPLATE
)
fig.show()
# %%

# Summary statistics
print("\n📊 Investor's Ruin Analysis Summary:")
print(f"  Low Risk (P < 10%): {(ruin_analysis['ruin_probability'] < 0.1).sum():,} stocks")
print(f"  Critical Risk (P > 60%): {(ruin_analysis['ruin_probability'] > 0.6).sum():,} stocks")
print(f"  Median Survival Probability: {ruin_analysis['survival_probability'].median():.1%}")

# %%
# Cell: Conditional Probabilities by Feature Category
# P(Distress | Feature Condition)

def calculate_conditional_probabilities(df: pd.DataFrame, feature_categories: dict) -> pd.DataFrame:
    """
    Calculate conditional probability of financial distress given feature conditions.
    
    P(Distress | High Feature) vs P(Distress | Low Feature)
    """
    results = []
    
    # Define distress condition
    distress_threshold = 30
    df['is_distressed'] = df['distress_risk_score'] < distress_threshold
    base_distress_rate = df['is_distressed'].mean()
    
    for category, features in feature_categories.items():
        for feature in features[:5]:  # Top 5 features per category
            if feature not in df.columns:
                continue
            
            data = df[[feature, 'is_distressed']].dropna()
            if len(data) < 100:
                continue
            
            median_val = data[feature].median()
            
            # P(Distress | Feature > Median)
            high_mask = data[feature] > median_val
            p_distress_high = data.loc[high_mask, 'is_distressed'].mean()
            
            # P(Distress | Feature <= Median)
            p_distress_low = data.loc[~high_mask, 'is_distressed'].mean()
            
            # Lift ratio
            lift_high = p_distress_high / base_distress_rate if base_distress_rate > 0 else 1
            lift_low = p_distress_low / base_distress_rate if base_distress_rate > 0 else 1
            
            results.append({
                'category': category,
                'feature': feature,
                'p_distress_high': p_distress_high,
                'p_distress_low': p_distress_low,
                'lift_high': lift_high,
                'lift_low': lift_low,
                'separation': abs(p_distress_high - p_distress_low)
            })
    
    return pd.DataFrame(results).sort_values('separation', ascending=False)

cond_probs = calculate_conditional_probabilities(all_stocks_features, FEATURE_CATEGORIES)

# Visualize top separating features
top_separating = cond_probs.head(20)

fig = go.Figure()
fig.add_trace(go.Bar(
    name='P(Distress | High)',
    x=top_separating['feature'],
    y=top_separating['p_distress_high'],
    marker_color='#e74c3c'
))
fig.add_trace(go.Bar(
    name='P(Distress | Low)',
    x=top_separating['feature'],
    y=top_separating['p_distress_low'],
    marker_color='#00bc8c'
))

fig.update_layout(
    barmode='group',
    title='Conditional Distress Probabilities by Feature Threshold',
    xaxis_tickangle=-45,
    yaxis_title='P(Distress)',
    template=PLOTLY_TEMPLATE,
    height=500
)
fig.show()
# %%

print("\n📊 Top Features for Distress Prediction (by separation):")
display(cond_probs[['category', 'feature', 'p_distress_high', 'p_distress_low', 'separation']].head(15).round(3))

# %% [markdown]
# # ---
# # ## Probabilistic & Statistical Analytics by Feature Category
# 
# %%
# Cell: Bayesian Parameter Estimation per Category
# Estimate posterior distributions for key metrics using conjugate priors

from scipy import stats
import numpy as np

def bayesian_category_analysis(
    df: pd.DataFrame, 
    category_name: str,
    features: list,
    prior_mean: float = 0,
    prior_std: float = 10
) -> dict:
    """
    Bayesian analysis of feature distributions within a category.
    Uses Normal-Normal conjugate prior for continuous features.
    
    Prior: μ ~ N(prior_mean, prior_std²)
    Likelihood: X | μ ~ N(μ, σ²)
    Posterior: μ | X ~ N(posterior_mean, posterior_var)
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
        prior_var = prior_std ** 2
        posterior_var = 1 / (1/prior_var + n/sample_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + n*sample_mean/sample_var)
        posterior_std = np.sqrt(posterior_var)
        
        # 95% Credible Interval
        ci_low = posterior_mean - 1.96 * posterior_std
        ci_high = posterior_mean + 1.96 * posterior_std
        
        # Probability that true mean > 0 (for growth/positive metrics)
        prob_positive = 1 - stats.norm.cdf(0, posterior_mean, posterior_std)
        
        results[feature] = {
            'n_obs': n,
            'sample_mean': sample_mean,
            'sample_std': np.sqrt(sample_var),
            'posterior_mean': posterior_mean,
            'posterior_std': posterior_std,
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'prob_positive': prob_positive
        }
    
    return results

# Analyze each category
bayesian_results_by_category = {}
for category, features in FEATURE_CATEGORIES.items():
    bayesian_results_by_category[category] = bayesian_category_analysis(
        all_stocks_features, category, features[:50]
    )

# Visualize Analyst Sentiment category
if 'Analyst Sentiment' in bayesian_results_by_category:
    sentiment_bayes = bayesian_results_by_category['Analyst Sentiment']
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        'Posterior Mean Estimates with 95% CI',
        'P(μ > 0) for Each Feature'
    ])
    
    features_list = list(sentiment_bayes.keys())
    means = [sentiment_bayes[f]['posterior_mean'] for f in features_list]
    ci_lows = [sentiment_bayes[f]['ci_95_low'] for f in features_list]
    ci_highs = [sentiment_bayes[f]['ci_95_high'] for f in features_list]
    prob_pos = [sentiment_bayes[f]['prob_positive'] for f in features_list]
    
    # Error bar plot
    fig.add_trace(
        go.Scatter(
            x=means, y=features_list,
            mode='markers',
            marker=dict(size=10, color='#3498db'),
            error_x=dict(
                type='data',
                symmetric=False,
                array=[h - m for m, h in zip(means, ci_highs)],
                arrayminus=[m - l for m, l in zip(means, ci_lows)]
            ),
            name='Posterior Mean'
        ),
        row=1, col=1
    )
    
    # Probability bar
    colors = ['#00bc8c' if p > 0.5 else '#e74c3c' for p in prob_pos]
    fig.add_trace(
        go.Bar(x=prob_pos, y=features_list, orientation='h',
               marker_color=colors, name='P(μ > 0)'),
        row=1, col=2
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="white", row=1, col=2)
    
    fig.update_layout(
        height=500,
        title_text="📊 Bayesian Analysis: Analyst Sentiment Category",
        template=PLOTLY_TEMPLATE,
        showlegend=False
    )
    fig.show()

# %%
# Cell: Monte Carlo Distribution Fitting per Category
# Fit parametric distributions and simulate scenarios

from scipy.stats import norm, t, lognorm, skewnorm

def fit_distributions_by_category(
    df: pd.DataFrame,
    category: str,
    features: list,
    n_simulations: int = 10000
) -> dict:
    """
    Fit multiple distributions and select best fit using AIC.
    Simulate future scenarios using best-fit distribution.
    """
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
            fits['normal'] = {'params': params_norm, 'aic': 2*2 - 2*ll_norm}
        except: pass
        
        # Student's t
        try:
            params_t = t.fit(data_clean)
            ll_t = t.logpdf(data_clean, *params_t).sum()
            fits['student_t'] = {'params': params_t, 'aic': 2*3 - 2*ll_t}
        except: pass
        
        # Skew Normal
        try:
            params_skew = skewnorm.fit(data_clean)
            ll_skew = skewnorm.logpdf(data_clean, *params_skew).sum()
            fits['skew_normal'] = {'params': params_skew, 'aic': 2*3 - 2*ll_skew}
        except: pass
        
        if not fits:
            continue
        
        # Select best fit by AIC
        best_dist = min(fits.keys(), key=lambda k: fits[k]['aic'])
        best_params = fits[best_dist]['params']
        
        # Simulate from best distribution
        if best_dist == 'normal':
            simulations = norm.rvs(*best_params, size=n_simulations)
        elif best_dist == 'student_t':
            simulations = t.rvs(*best_params, size=n_simulations)
        else:
            simulations = skewnorm.rvs(*best_params, size=n_simulations)
        
        # Calculate VaR and CVaR
        var_5 = np.percentile(simulations, 5)
        cvar_5 = simulations[simulations <= var_5].mean()
        
        results[feature] = {
            'best_distribution': best_dist,
            'params': best_params,
            'aic': fits[best_dist]['aic'],
            'simulated_mean': simulations.mean(),
            'simulated_std': simulations.std(),
            'var_5_pct': var_5,
            'cvar_5_pct': cvar_5,
            'simulations': simulations
        }
    
    return results

# Fit distributions for Earnings Quality category
eq_dist_fits = fit_distributions_by_category(
    all_stocks_features,
    'Earnings Quality',
    FEATURE_CATEGORIES.get('Earnings Quality', [])[:8]
)

# Visualize distribution fits
if eq_dist_fits:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(eq_dist_fits.keys())[:4]
    )
    
    for i, (feature, result) in enumerate(list(eq_dist_fits.items())[:4]):
        row = i // 2 + 1
        col = i % 2 + 1
        
        # Historical data
        hist_data = all_stocks_features[feature].dropna()
        hist_clipped = hist_data.clip(hist_data.quantile(0.05), hist_data.quantile(0.95))
        
        fig.add_trace(
            go.Histogram(x=hist_clipped, nbinsx=40, opacity=0.6,
                        marker_color='#3498db', name='Historical'),
            row=row, col=col
        )
        
        # Simulated data overlay
        sim_clipped = np.clip(result['simulations'], 
                              hist_data.quantile(0.05), 
                              hist_data.quantile(0.95))
        fig.add_trace(
            go.Histogram(x=sim_clipped, nbinsx=40, opacity=0.4,
                        marker_color='#e74c3c', name='Simulated'),
            row=row, col=col
        )
    
    fig.update_layout(
        height=600,
        title_text=f"📈 Distribution Fitting: Earnings Quality Category",
        template=PLOTLY_TEMPLATE,
        barmode='overlay'
    )

    
    # Summary table
    dist_summary = pd.DataFrame([
        {
            'feature': f,
            'best_dist': r['best_distribution'],
            'var_5%': r['var_5_pct'],
            'cvar_5%': r['cvar_5_pct']
        }
        for f, r in eq_dist_fits.items()
    ])
    print("\n📊 Distribution Fitting Summary (Earnings Quality):")
    display(dist_summary.round(3))

    fig.show()

# %% [markdown]
# # ---
# # ## Feature Category-Specific MCMC Simulations
# 
# %%
# Cell: MCMC Metropolis-Hastings Sampler for Category Analysis
# Simulating posterior distributions for category-level parameters

import numpy as np
from scipy import stats

def metropolis_hastings_sampler(
    data: np.ndarray,
    n_samples: int = 10000,
    burn_in: int = 2000,
    proposal_std: float = 0.5,
    prior_mean: float = 0,
    prior_std: float = 10
) -> np.ndarray:
    """
    Metropolis-Hastings MCMC sampler for estimating posterior of mean parameter.
    
    Assumes: X ~ N(μ, σ²) with σ known from data
             Prior: μ ~ N(prior_mean, prior_std²)
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
        ll = -n/2 * np.log(2*np.pi*data_std**2) - np.sum((data - mu)**2) / (2*data_std**2)
        # Log-prior
        lp = -0.5 * ((mu - prior_mean) / prior_std)**2
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

def run_category_mcmc(
    df: pd.DataFrame,
    category_name: str,
    features: list,
    n_samples: int = 10000
) -> dict:
    """
    Run MCMC analysis for features in a category.
    """
    results = {}
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        data = df[feature].dropna().values
        if len(data) < 100:
            continue
        
        # Remove outliers
        q01, q99 = np.percentile(data, [1, 99])
        data_clean = data[(data >= q01) & (data <= q99)]
        
        samples, acceptance_rate = metropolis_hastings_sampler(
            data_clean, n_samples=n_samples
        )
        
        results[feature] = {
            'posterior_samples': samples,
            'posterior_mean': np.mean(samples),
            'posterior_std': np.std(samples),
            'hdi_95': (np.percentile(samples, 2.5), np.percentile(samples, 97.5)),
            'acceptance_rate': acceptance_rate,
            'n_obs': len(data_clean)
        }
    
    return results

# %%
# Cell: Analyst Sentiment MCMC Analysis
print("🔄 Running MCMC for Analyst Sentiment category...")

analyst_mcmc = run_category_mcmc(
    all_stocks_features,
    'Analyst Sentiment',
    FEATURE_CATEGORIES.get('Analyst Sentiment', [])[:6],
    n_samples=15000
)

if analyst_mcmc:
    # Trace plots and posteriors
    n_features = min(len(analyst_mcmc), 4)
    fig = make_subplots(
        rows=n_features, cols=2,
        subplot_titles=[f'{f} - Trace' if i % 2 == 0 else f'{f} - Posterior' 
                       for f in list(analyst_mcmc.keys())[:n_features] for i in range(2)]
    )
    
    for i, (feature, result) in enumerate(list(analyst_mcmc.items())[:n_features]):
        samples = result['posterior_samples']
        
        # Trace plot
        fig.add_trace(
            go.Scatter(y=samples[::10], mode='lines', 
                      line=dict(width=0.5, color='#3498db'),
                      name=f'{feature} trace'),
            row=i+1, col=1
        )
        
        # Posterior histogram
        fig.add_trace(
            go.Histogram(x=samples, nbinsx=50, 
                        marker_color='#00bc8c', opacity=0.7,
                        name=f'{feature} posterior'),
            row=i+1, col=2
        )
        
        # Add HDI lines
        hdi_low, hdi_high = result['hdi_95']
        fig.add_vline(x=hdi_low, line_dash="dash", line_color="red", 
                     row=i+1, col=2)
        fig.add_vline(x=hdi_high, line_dash="dash", line_color="red",
                     row=i+1, col=2)
    
    fig.update_layout(
        height=250 * n_features,
        title_text="📊 MCMC Posterior Analysis: Analyst Sentiment",
        showlegend=False,
        template=PLOTLY_TEMPLATE
    )
    fig.show()

# %%
    # Summary table
    mcmc_summary = pd.DataFrame([
        {
            'feature': f,
            'posterior_mean': r['posterior_mean'],
            'posterior_std': r['posterior_std'],
            'hdi_2.5%': r['hdi_95'][0],
            'hdi_97.5%': r['hdi_95'][1],
            'acceptance_rate': r['acceptance_rate']
        }
        for f, r in analyst_mcmc.items()
    ])
    print("\n📈 MCMC Summary (Analyst Sentiment):")
    display(mcmc_summary.round(4))

# %%
# Cell: Accounting Quality MCMC with Student's t Likelihood
print("🔄 Simulating Student's t-distribution with MCMC/Metropolis dependent sampling algorithm")

def mcmc_student_t(
    data: np.ndarray,
    n_samples: int = 10000,
    burn_in: int = 2000
) -> tuple:
    """
    MCMC for Student's t location parameter with heavier tails.
    Better for financial data with outliers.
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
        # Propose new mu
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

# Run for Accounting Quality features
aq_features = ['eps_adjustment_pct', 'earnings_quality_score', 'gaap_revision_momentum']
aq_features_available = [f for f in aq_features if f in all_stocks_features.columns]

if aq_features_available:
    fig = make_subplots(
        rows=len(aq_features_available), cols=2,
        subplot_titles=[f'{f} - μ Posterior' if i % 2 == 0 else f'{f} - df Posterior'
                       for f in aq_features_available for i in range(2)]
    )
    
    for i, feature in enumerate(aq_features_available):
        data = all_stocks_features[feature].dropna().values
        q01, q99 = np.percentile(data, [1, 99])
        data_clean = data[(data >= q01) & (data <= q99)]
        
        samples_mu, samples_df = mcmc_student_t(data_clean, n_samples=10000)
        
        fig.add_trace(
            go.Histogram(x=samples_mu, nbinsx=40, marker_color='#9b59b6', opacity=0.7),
            row=i+1, col=1
        )
        fig.add_trace(
            go.Histogram(x=samples_df, nbinsx=40, marker_color='#e67e22', opacity=0.7),
            row=i+1, col=2
        )
    
    fig.update_layout(
        height=250 * len(aq_features_available),
        title_text="📊 MCMC Student's t Analysis: Accounting Quality",
        showlegend=False,
        template=PLOTLY_TEMPLATE
    )
    fig.show()

# %%
# Cell: Profitability Category MCMC with Hierarchical Structure
print("🔄 Running Hierarchical MCMC for Profitability by Industry...")

def hierarchical_mcmc_by_sector(
    df: pd.DataFrame,
    feature: str,
    sector_col: str = 'industry',
    n_samples: int = 8000
) -> dict:
    """
    Hierarchical MCMC: estimate sector-level means with pooling toward global mean.
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
            'raw_mean': sector_mean,
            'posterior_mean': posterior_mean,
            'shrinkage': shrinkage,
            'samples': samples,
            'n_obs': n
        }
    
    return results

if 'roe' in all_stocks_features.columns:
    roe_hierarchical = hierarchical_mcmc_by_sector(all_stocks_features, 'roe')
    
    # Visualize shrinkage effect
    sectors = list(roe_hierarchical.keys())[:15]
    raw_means = [roe_hierarchical[s]['raw_mean'] for s in sectors]
    post_means = [roe_hierarchical[s]['posterior_mean'] for s in sectors]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Raw Sector Mean',
        x=sectors,
        y=raw_means,
        marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        name='Posterior (Shrunk) Mean',
        x=sectors,
        y=post_means,
        marker_color='#00bc8c'
    ))
    
    fig.update_layout(
        barmode='group',
        title='Hierarchical MCMC: ROE by Industry (Raw vs Shrunk Estimates)',
        xaxis_tickangle=-45,
        yaxis_title='ROE',
        template=PLOTLY_TEMPLATE,
        height=500
    )
    fig.show()
    
    print("\n📊 Shrinkage Effect Summary:")
    shrinkage_df = pd.DataFrame([
        {'sector': s, 'raw': r['raw_mean'], 'posterior': r['posterior_mean'], 
         'shrinkage': r['shrinkage'], 'n': r['n_obs']}
        for s, r in roe_hierarchical.items()
    ])
    
    if not shrinkage_df.empty:
        shrinkage_df = shrinkage_df.sort_values('shrinkage')
        display(shrinkage_df.head(10).round(3))
    else:
        print("⚠️ No sectors with sufficient data (n≥30) for hierarchical analysis.")

# %% [markdown]
# # ---
# # ## Final Summary Dashboard
# 
# %%
# Cell 37: Executive Summary Dashboard
from plotly.subplots import make_subplots

# Create summary metrics
total_stocks = len(all_stocks_features)
high_quality_count = (composite_scores['piotroski_f_score'] >= 7).sum()
breakout_count = technical_features['breakout_signal'].sum() if 'breakout_signal' in technical_features.columns else 0
distressed_count = (distress_features['distress_risk_score'] < 5).sum()

# Summary cards data
summary_metrics = {
    'Total Stocks': total_stocks,
    'High Quality (F≥7)': high_quality_count,
    'Breakout Signals': breakout_count,
    'Distressed (<30)': distressed_count,
    'Bullish Trend': (technical_features['ema_trend_consistency'] == 1).sum(),
    'Always +FCF': cashflow_data['fcf_positive_years'].eq(5).sum()
}

# Create indicator cards
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{"type": "indicator"}] * 3] * 2,
    subplot_titles=list(summary_metrics.keys())
)

for i, (name, value) in enumerate(summary_metrics.items()):
    row = i // 3 + 1
    col = i % 3 + 1
    fig.add_trace(
        go.Indicator(
            mode="number",
            value=value,
            number={'font': {'size': 48, 'color': '#00bc8c' if 'Quality' in name or 'Bullish' in name or 'FCF' in name
            else '#e74c3c' if 'Distressed' in name else '#3498db'}}
        ),
        row=row, col=col
    )

fig.update_layout(
    height=400,
    title_text="📊 Feature Analytics Executive Summary",
    template=PLOTLY_TEMPLATE
)
fig.show()

# %%
# Cell: Export Enhanced Analytics (Updated for v2 schema)

# Define comprehensive export columns (Updated with NEW feature registry enhancements)
export_columns = {
    'Identifiers': ['ticker', 'isin', 'name', 'industry', 'region', 'industry'],
    'Market Data': ['market_cap', 'enterprise_value', 'last_price', 'price_target'],
    'Valuation': ['p_e_ratio', 'p_b_ratio', 'ev_ebitda_ratio', 'peg_ratio',
                  'price_to_tangible_book', 'tangible_book_value_ltm'],
    'Quality Scores': ['piotroski_f_score', 'distress_risk_score',
                       'earnings_quality_composite_comp', 'cash_flow_quality_score',
                       'accounting_quality_score', 'beta_stability_score',
                       'tangible_asset_quality', 'core_earnings_stability'],
    'Growth': ['revenue_growth_yoy', 'eps_trajectory_score', 'ebitda_cagr_3y',
               'tbv_yoy_growth', 'rnd_yoy_growth'],
    'Momentum': ['price_momentum_1y', 'long_term_trend_score', 'secular_trend_flag'],
    'Cash Flow': ['fcf_positive_years', 'fcf_yield', 'self_funding_ratio'],
    'Leverage': ['debt_to_equity', 'debt_deleveraging', 'interest_coverage_ratio'],
    'Sentiment': [
        'analyst_bullish_pct',
        'analyst_neutral_pct',
        'analyst_bearish_pct',
        'upside_potential'
    ],
    # NEW: Feature Registry Enhancement Categories
    'EPS Continuing': ['eps_cont_ltm', 'eps_cont_yoy_growth', 'eps_cont_vs_total_eps',
                       'discontinued_ops_impact'],
    'R&D Investment': ['rnd_intensity_ltm', 'rnd_per_employee', 'high_rnd_intensity_flag'],
    'Inventory': ['inventory_days', 'inventory_turnover_mv', 'inventory_yoy_change'],
    'Goodwill & M&A': ['goodwill_concentration', 'goodwill_3y_growth', 'recent_acquisition_flag'],
    'Cost Efficiency': ['marketing_to_revenue', 'sga_vs_5y_avg', 'marketing_trend_yoy'],
    # NEW: CapEx & Acquisitions Temporal
    'CapEx Analysis': ['capex_yoy_growth', 'capex_vs_5y_avg', 'capex_3y_trend',
                       'underinvestment_flag', 'overinvestment_flag', 'capex_acceleration'],
    'M&A Activity': ['acquisitions_ltm_total', 'acquisitions_vs_5y_avg', 'ma_intensity_score',
                     'serial_acquirer_flag', 'acquisition_pause_flag'],
    'Investment Efficiency': ['total_investment_to_cfo', 'organic_vs_inorganic',
                              'investment_efficiency', 'sustainable_ma_flag']
}

# Flatten and filter available columns
all_export_cols = []
for category, cols in export_columns.items():
    all_export_cols.extend([c for c in cols if c in all_stocks_features.columns])

# Ensure unique columns
all_export_cols = list(dict.fromkeys(all_export_cols))

enhanced_export = all_stocks_features[all_export_cols].copy()

# Add calculated fields
if 'piotroski_f_score' in enhanced_export.columns and 'distress_risk_score' in enhanced_export.columns:
    enhanced_export['quality_composite'] = (
            enhanced_export['piotroski_f_score'] / 9 * 25 +
            enhanced_export['distress_risk_score'] / 100 * 25 +
            enhanced_export.get('earnings_quality_composite_comp', 50) / 100 * 25 +
            enhanced_export.get('cash_flow_quality_score', 50) / 100 * 25
    )

# Save
enhanced_export.to_csv('outputs/feature_analytics_v2_export.csv', index=False)
print(f"✓ Enhanced analytics exported: {len(enhanced_export):,} stocks with {len(enhanced_export.columns)} features")
print(f"\n📋 Export Categories:")
for category, cols in export_columns.items():
    available = [c for c in cols if c in all_stocks_features.columns]
    print(f"  {category}: {len(available)} features")

# Build enhanced summary directly from all_stocks_features (no redundant merges needed)
# Updated with NEW feature registry enhancement columns
summary_cols = [
    'ticker', 'name', 'industry', 'region', 'market_cap',
    'piotroski_f_score', 'long_term_trend_score', 'dilution_score',
    'distress_risk_score', 'cash_runway_months',
    'earnings_quality_composite_comp', 'net_income_growth_yoy',
    'cash_flow_quality_score', 'fcf_yield',
    # NEW: Feature Registry Enhancements
    'eps_cont_vs_total_eps', 'discontinued_ops_impact',
    'rnd_intensity_ltm', 'high_rnd_intensity_flag',
    'inventory_days', 'inventory_turnover_mv',
    'goodwill_concentration', 'recent_acquisition_flag',
    'tangible_book_value_ltm', 'tbv_yoy_growth',
    'marketing_to_revenue', 'sga_vs_5y_avg',
    # NEW: CapEx & Acquisitions
    'capex_yoy_growth', 'capex_vs_5y_avg', 'underinvestment_flag',
    'acquisitions_ltm_total', 'serial_acquirer_flag', 'ma_intensity_score',
    'total_investment_to_cfo', 'organic_vs_inorganic', 'investment_efficiency'
]

# Filter to available columns
available_summary_cols = [c for c in summary_cols if c in all_stocks_features.columns]
enhanced_summary = all_stocks_features[available_summary_cols].copy()

# Save enhanced summary
enhanced_summary.to_csv('outputs/enhanced_feature_analytics.csv', index=False)
print(f"✓ Enhanced analytics saved: {len(enhanced_summary):,} stocks with {len(enhanced_summary.columns)} features")

# Display sample
print("\n📋 Enhanced Analytics Sample:")
display(enhanced_summary.head(10))