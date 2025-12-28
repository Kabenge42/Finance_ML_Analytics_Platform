`stock_analytics.ipynb` Enhancements:

Here's the enhanced `stock_analytics.ipynb` with comprehensive analytics dashboards:

```python
# %% md
# Stock Analytics Dashboard

This
notebook
provides
comprehensive
stock
analytics
dashboards
with interactive visualizations,
statistics, and benchmarking
organized
by ** Phase
9.3
Feature
Categories **.

** Version: ** 2.0
.0 | ** Model
Version: ** v9_10 | ** Updated: ** 2025 - 12 - 24

## Dashboard Structure (21 Feature Categories, 294 Features)

### 📊 Core Analytics
1. ** Region & Sector
Overview ** - Distribution
analysis
with feature coverage heatmaps
2. ** Exchange
Analysis ** - Market - level
analytics
by
stock
exchange
3. ** Industry
Deep - Dive ** - Granular
industry - level
benchmarking

### 💹 Financial Analytics (by Feature Category)
4. ** Valuation
Ratios ** - P / E, P / B, EV / EBITDA, PEG
analysis
5. ** Profitability ** - ROE, ROA, ROIC, margin
analysis
6. ** Growth
Metrics ** - Revenue, earnings, EBITDA
growth
7. ** Leverage & Liquidity ** - Debt
ratios, coverage, liquidity
metrics

### 📈 Technical & Market Analytics
8. ** Momentum & Technical ** - Price
momentum, EMA
crossovers, breakout
signals
9. ** Technical
Analysis ** - RSI, 52 - week
range, volume
momentum
10. ** Market
Sentiment ** - Beta
stability, systematic
risk

### 🎯 Quality & Risk Analytics
11. ** Quality & Risk ** - Altman
Z, accounting
quality, distress
signals
12. ** Composite
Scores ** - Piotroski
F, Beneish
M, momentum
scores
13. ** Earnings
Quality ** - EPS
surprises, GAAP
vs
Adjusted
analytics

### 💰 Dividends & Capital Allocation
14. ** Dividend
Reliability ** - Dividend
streaks, safety
scores,
yield analysis
15. ** Capital
Allocation ** - CapEx, reinvestment, shareholder
returns

### 🔮 Forecasting & Analyst Analytics
16. ** Revenue
Forecasting ** - Estimate
spreads, consensus
uncertainty
17. ** Analyst
Sentiment ** - Price
targets, recommendations, conviction

### 👥 Operations & Efficiency
18. ** Employee
Productivity ** - Revenue
per
employee, hiring
intensity
19. ** Efficiency
Ratios ** - Asset
turnover, inventory
efficiency
20. ** Balance
Sheet
Dynamics ** - Asset / debt
growth, working
capital
21. ** Temporal
Patterns ** - Earnings
calendar, fiscal
quarters

## Integrated Artifacts
- `outputs / eda / visualizations / ` - Sector
benchmarking, hypothesis
tests
- `outputs / eda / earnings_analytics / ` - Earnings
surprises, market
movers
- `outputs / eda / dividend_visualizations / ` - Dividend
yield by
sector
- `outputs / eda / advanced_analytics / ` - VaR, risk
attribution

# %%
# ============================================================================
# Cell 1: Configuration & Setup
# ============================================================================
import json
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SQLAlchemy check
try:
    from sqlalchemy import create_engine, text

    HAVE_SQLALCHEMY = True
except ImportError:
    HAVE_SQLALCHEMY = False

# Project paths
PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
CACHE_DIR = PROJECT_ROOT / '.cache'

# Create output directories
for subdir in ['eda/dashboards', 'eda/by_region', 'eda/by_sector',
               'eda/by_industry', 'eda/by_exchange']:
    (OUTPUT_DIR / subdir).mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_ml.notebook_config import NotebookConfig
from finance_ml.ml_workflow.data.schema import (
    PHASE93_FEATURE_CATEGORIES,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
)

# Earnings Dashboard Widgets
from finance_ml.dashboards.earnings_widgets import (
    DATE_DISPLAY_FORMAT,
    PLOTLY_TEMPLATE,
    COLOR_PALETTE,
    resolve_reference_date,
    add_formatted_date_columns,
    get_category_metrics,
    create_earnings_calendar_dashboard,
    display_earnings_dashboard,
    create_earnings_metrics_chart,
    create_earnings_surprise_dashboard,
    create_analyst_recommendation_heatmap,
    create_market_movers_dashboard,
    create_price_target_analytics,
    create_earnings_calendar_analytics,
    analyze_earnings_quality,
    create_gaap_adjusted_comparison_chart,
    create_technical_valuation_dashboard,
    create_dividend_sustainability_scorecard,
    create_employee_productivity_dashboard,
    create_category_correlation_network,
    generate_earnings_quality_alerts,
    EarningsAlertConfig,
)

from finance_ml.ml_workflow.features import engineer_temporal_features

CFG = NotebookConfig(
    have_finance_prediction=True,
    have_database_connection=True,
    have_advanced_analytics=True,
    have_dim_reduction=False,
    debug_mode=False,
)

# Configuration Constants
from finance_ml.ml_workflow.config import (
    TARGET_COL, TARGET_COL_FALLBACK, TEST_SIZE, TRAIN_SIZE,
    CV_FOLDS, QUANTILES, MIN_SECTOR_SAMPLES,
    WINSORIZE_LOWER, WINSORIZE_UPPER, RANDOM_SEED, MODEL_VERSION,
)

np.random.seed(RANDOM_SEED)

# Visualization Configuration
TOP_N_SECTORS = 12
TOP_N_INDUSTRIES = 20
TOP_N_EXCHANGES = 15
TOP_N_FEATURES = 15

# Style Configuration (code_guidelines.md §17)
plt.style.use('dark_background')
sns.set_palette('husl')

# Category-specific color mapping
CATEGORY_COLORS = {
    "Momentum & Technical": "#3498db",
    "Valuation Ratios": "#375a7f",
    "Profitability": "#00bc8c",
    "Quality & Risk": "#e74c3c",
    "Cash Flow": "#f39c12",
    "Capital Allocation": "#9b59b6",
    "Analyst Sentiment": "#1abc9c",
    "Market Sentiment": "#34495e",
    "Leverage & Liquidity": "#2980b9",
    "Temporal Patterns": "#8e44ad",
    "Composite Scores": "#16a085",
    "Growth Metrics": "#27ae60",
    "Efficiency Ratios": "#d35400",
    "Employee Productivity": "#7f8c8d",
    "Balance Sheet Dynamics": "#2c3e50",
    "Revenue Forecasting": "#c0392b",
    "Earnings Quality": "#e67e22",
    "Technical Analysis": "#1abc9c",
    "Valuation Timeseries": "#3498db",
    "Dividend Reliability": "#27ae60",
    "Employment Dynamics": "#34495e",
}

print('=' * 80)
print('STOCK ANALYTICS DASHBOARD - CONFIGURATION')
print('=' * 80)
print(f'PROJECT_ROOT: {PROJECT_ROOT}')
print(f'Python: {sys.version.split()[0]}')
print(f'Model Version: {MODEL_VERSION}')
print(f'Phase 9.3 Categories: {len(PHASE93_FEATURE_CATEGORIES)}')
print(f'Total Phase 9.3 Features: {sum(len(f) for f in PHASE93_FEATURE_CATEGORIES.values())}')
CFG.display_summary()

# %% md
## Cell 2: ETL Pipeline & Data Loading

# %%
# ============================================================================
# Cell 2: ETL Pipeline - Extract, Transform, Load
# ============================================================================

from finance_ml.ml_workflow.preprocessing.etl import (
    DataExtractionConfig, DataSanitizationConfig, DtypeCastingConfig,
    FeatureEngineeringConfig, FeatureSelectionConfig, FinancialMetricsConfig,
    ImputationConfig, ScalingConfig, SchemaValidationConfig,
    SemanticClassificationConfig, SemanticTransformConfig,
    etl_with_features, ETLConfig,
)
from finance_ml.ml_workflow.eda.phase93_categories import (
    categorize_dataframe_columns, get_phase93_coverage_stats, list_all_phase93_features,
)
from finance_ml.ml_workflow.preprocessing.column_semantics import PRICE_COLUMNS

etl_config = ETLConfig(
    extraction=DataExtractionConfig(normalize_column_names=True),
    validation=SchemaValidationConfig(
        validate_schema=True, require_target_column=True,
        drop_rows_with_missing_critical_fields=True,
        validate_schema_alignment=True, schema_alignment_threshold=0.80,
    ),
    dtype_casting=DtypeCastingConfig(apply_dtype_casting=True, track_diagnostics=True),
    semantic_classification=SemanticClassificationConfig(enabled=False, preserve_price_columns=True),
    imputation=ImputationConfig(
        apply_imputation=True, strategy="6step", knn_neighbors=5,
        sector_column="sector", reference_price_column="last_price",
        impute_categorical_columns=True, impute_datetime_columns=True,
    ),
    semantic_transform=SemanticTransformConfig(
        apply_log_transforms=False, exclude_ratios_from_winsorization=True,
        exclude_percentages_from_winsorization=True,
    ),
    sanitization=DataSanitizationConfig(sanitize_data=True, apply_winsorization=False),
    scaling=ScalingConfig(enabled=False, exclude_price_columns=True),
    feature_engineering=FeatureEngineeringConfig(
        enabled=True, preset="comprehensive", engineer_earnings_analytics=True,
    ),
    feature_selection=FeatureSelectionConfig(enabled=False),
    financial_metrics=FinancialMetricsConfig(
        compute_valuation_metrics=True, compute_profitability_metrics=True,
        compute_growth_metrics=True, compute_leverage_metrics=True,
        compute_target_vs_price_metrics=True,
    ),
)

print('=' * 80)
print('ETL PIPELINE EXECUTION')
print('=' * 80)

df, metrics = etl_with_features(source='csv', data_dir=DATA_DIR, return_metrics=True, config=etl_config)

print('\n' + metrics.summary())

# Resolve reference date
REFERENCE_DATE = resolve_reference_date(df, None)
print(f"\nReference date: {REFERENCE_DATE.strftime(DATE_DISPLAY_FORMAT)}")

# Add temporal features
if 'next_earnings' in df.columns:
    df = engineer_temporal_features(df, date_col='next_earnings', reference_date=REFERENCE_DATE)

# Add formatted date columns
date_cols = ['_reference_date', 'next_earnings', 'income_statement_report_date', 'dividend_record_ex_date']
date_cols = [c for c in date_cols if c in df.columns]
add_formatted_date_columns(df, date_cols)

# Get Phase 9.3 coverage stats
coverage_stats = get_phase93_coverage_stats(df)
total_phase93 = sum(coverage_stats.values())

print(f'\n✓ ETL Complete: {df.shape[0]:,} stocks × {df.shape[1]} features')
print(f'✓ Phase 9.3 Features: {total_phase93}/{sum(len(f) for f in PHASE93_FEATURE_CATEGORIES.values())}')

# %% md
## Cell 3: Region & Exchange Distribution Analytics

Interactive
dashboards
analyzing
stock
distribution
by
Region, Sector, Industry, and Exchange.

# %%
# ============================================================================
# Cell 3: Region, Sector, Industry & Exchange Distribution
# ============================================================================

print('=' * 80)
print('📊 REGION, SECTOR, INDUSTRY & EXCHANGE ANALYTICS')
print('=' * 80)

dashboard_dir = OUTPUT_DIR / 'eda' / 'dashboards'

# ============================================================================
# 3.1 Region Distribution with Key Metrics
# ============================================================================
if 'region' in df.columns:
    print('\n📍 Region Distribution Analysis...')

    region_stats = df.groupby('region').agg({
        'ticker': 'count',
        'market_cap': ['sum', 'mean', 'median'],
        'last_price': 'mean',
    }).round(2)
    region_stats.columns = ['Count', 'Total_MCap', 'Mean_MCap', 'Median_MCap', 'Avg_Price']
    region_stats = region_stats.sort_values('Count', ascending=False)
    region_stats['Pct'] = (region_stats['Count'] / region_stats['Count'].sum() * 100).round(1)

    # Create region sunburst with sector breakdown
    fig_region = px.sunburst(
        df, path=['region', 'sector'], values='market_cap',
        title='<b>Market Cap Distribution by Region → Sector</b>',
        template=PLOTLY_TEMPLATE, color='region',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_region.update_layout(height=700, font=dict(family='Segoe UI, Roboto, Arial'))
    fig_region.write_html(dashboard_dir / 'region_sector_sunburst.html')
    fig_region.show()

    print(f'  ✓ Saved: region_sector_sunburst.html')
    print(f'\n  Region Summary:')
    print(region_stats.to_string())

# ============================================================================
# 3.2 Exchange Distribution Analysis
# ============================================================================
exchange_col = None
for col in ['exchange', 'primary_exchange', 'stock_exchange']:
    if col in df.columns:
        exchange_col = col
        break

if exchange_col:
    print(f'\n🏛️ Exchange Distribution Analysis ({exchange_col})...')

    exchange_stats = df.groupby(exchange_col).agg({
        'ticker': 'count',
        'market_cap': ['sum', 'mean'],
        'last_price': 'mean',
    }).round(2)
    exchange_stats.columns = ['Count', 'Total_MCap', 'Mean_MCap', 'Avg_Price']
    exchange_stats = exchange_stats.sort_values('Total_MCap', ascending=False).head(TOP_N_EXCHANGES)
    exchange_stats['Pct'] = (exchange_stats['Count'] / df.shape[0] * 100).round(1)

    # Exchange treemap
    top_exchanges = exchange_stats.index.tolist()
    df_exchange = df[df[exchange_col].isin(top_exchanges)]

    fig_exchange = px.treemap(
        df_exchange, path=[exchange_col, 'sector'],
        values='market_cap', color='market_cap',
        color_continuous_scale='Blues',
        title=f'<b>Market Cap by Exchange → Sector</b><br><sup>Top {TOP_N_EXCHANGES} Exchanges</sup>',
        template=PLOTLY_TEMPLATE,
    )
    fig_exchange.update_layout(height=700)
    fig_exchange.write_html(dashboard_dir / 'exchange_sector_treemap.html')
    fig_exchange.show()

    print(f'  ✓ Saved: exchange_sector_treemap.html')
    print(f'\n  Top Exchanges by Market Cap:')
    print(exchange_stats.head(10).to_string())

# ============================================================================
# 3.3 Industry Distribution Analysis
# ============================================================================
industry_col = None
for col in ['industry', 'industry_group', 'sub_industry']:
    if col in df.columns:
        industry_col = col
        break

if industry_col:
    print(f'\n🏭 Industry Distribution Analysis ({industry_col})...')

    industry_stats = df.groupby(industry_col).agg({
        'ticker': 'count',
        'market_cap': ['sum', 'mean'],
        'roe': 'mean' if 'roe' in df.columns else 'count',
    }).round(2)

    if 'roe' in df.columns:
        industry_stats.columns = ['Count', 'Total_MCap', 'Mean_MCap', 'Avg_ROE']
    else:
        industry_stats.columns = ['Count', 'Total_MCap', 'Mean_MCap', '_count']
        industry_stats = industry_stats.drop('_count', axis=1)

    industry_stats = industry_stats.sort_values('Count', ascending=False).head(TOP_N_INDUSTRIES)

    # Industry bar chart
    fig_industry = px.bar(
        industry_stats.reset_index().head(TOP_N_INDUSTRIES),
        x='Count', y=industry_col, orientation='h',
        title=f'<b>Stock Count by Industry</b><br><sup>Top {TOP_N_INDUSTRIES} Industries</sup>',
        template=PLOTLY_TEMPLATE, color='Total_MCap',
        color_continuous_scale='Viridis',
    )
    fig_industry.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
    fig_industry.write_html(dashboard_dir / 'industry_distribution.html')
    fig_industry.show()

    print(f'  ✓ Saved: industry_distribution.html')

# ============================================================================
# 3.4 Region × Exchange × Sector 3D Analysis
# ============================================================================
if 'region' in df.columns and exchange_col:
    print('\n🌐 Region × Exchange × Sector 3D Analysis...')

    # Create cross-tabulation
    cross_tab = pd.crosstab([df['region'], df['sector']], df[exchange_col])

    # Region-Exchange heatmap
    region_exchange = pd.crosstab(df['region'], df[exchange_col])

    fig_re_heatmap = px.imshow(
        region_exchange, title='<b>Stock Distribution: Region × Exchange</b>',
        template=PLOTLY_TEMPLATE, color_continuous_scale='RdYlGn',
        aspect='auto', text_auto=True,
    )
    fig_re_heatmap.update_layout(height=500)
    fig_re_heatmap.write_html(dashboard_dir / 'region_exchange_heatmap.html')
    fig_re_heatmap.show()

    print(f'  ✓ Saved: region_exchange_heatmap.html')

print('\n✓ Region, Sector, Industry & Exchange Analytics Complete')

# %% md
## Cell 4: Phase 9.3 Feature Category Dashboards

Comprehensive
dashboards
for each of the 21 Phase 9.3 feature categories.

# %%
# ============================================================================
# Cell 4: Phase 9.3 Feature Category Dashboards
# ============================================================================

print('=' * 80)
print('📊 PHASE 9.3 FEATURE CATEGORY DASHBOARDS')
print('=' * 80)

category_viz_dir = OUTPUT_DIR / 'eda' / 'dashboards' / 'categories'
category_viz_dir.mkdir(parents=True, exist_ok=True)

# Get categorized features
categorized = categorize_dataframe_columns(df)

# ============================================================================
# 4.1 Category Coverage Overview
# ============================================================================
print('\n📋 Feature Category Coverage Analysis...')

coverage_data = []
for category, features in PHASE93_FEATURE_CATEGORIES.items():
    present = [f for f in features if f in df.columns]
    avg_completeness = 0
    if present:
        avg_completeness = df[present].notna().mean().mean() * 100

    coverage_data.append({
        'Category': category,
        'Expected': len(features),
        'Present': len(present),
        'Coverage_Pct': len(present) / len(features) * 100 if features else 0,
        'Completeness_Pct': avg_completeness,
        'Color': CATEGORY_COLORS.get(category, '#666666'),
    })

coverage_df = pd.DataFrame(coverage_data).sort_values('Expected', ascending=False)

# Coverage bar chart
fig_coverage = go.Figure()
fig_coverage.add_trace(go.Bar(
    y=coverage_df['Category'], x=coverage_df['Expected'],
    name='Expected', orientation='h', marker_color=COLOR_PALETTE['neutral'],
))
fig_coverage.add_trace(go.Bar(
    y=coverage_df['Category'], x=coverage_df['Present'],
    name='Present', orientation='h', marker_color=COLOR_PALETTE['success'],
))
fig_coverage.update_layout(
    title='<b>Phase 9.3 Feature Coverage by Category</b>',
    template=PLOTLY_TEMPLATE, height=700, barmode='overlay',
    xaxis_title='Feature Count', yaxis_title='Category',
)
fig_coverage.write_html(category_viz_dir / 'phase93_coverage_overview.html')
fig_coverage.show()
print(f'  ✓ Saved: phase93_coverage_overview.html')


# ============================================================================
# 4.2 Generate Dashboard for Each Major Category
# ============================================================================

def create_category_dashboard(df: pd.DataFrame, category: str, features: list,
                              output_dir: Path, group_cols: list = None):
    """Create comprehensive dashboard for a feature category."""
    if group_cols is None:
        group_cols = ['sector', 'region']

    available = [f for f in features if f in df.columns]
    if not available:
        return None

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{category}: Distribution by Sector',
            f'{category}: Top Features Correlation',
            f'{category}: Regional Comparison',
            f'{category}: Key Metrics Box Plots',
        ],
        specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
               [{'type': 'scatter'}, {'type': 'box'}]],
        vertical_spacing=0.12, horizontal_spacing=0.1,
    )

    # 1. Mean by Sector (top 10 sectors)
    if 'sector' in df.columns and available:
        sector_means = df.groupby('sector')[available[:5]].mean()
        top_sectors = df['sector'].value_counts().head(10).index
        sector_means = sector_means.loc[sector_means.index.isin(top_sectors)]

        for i, feat in enumerate(available[:3]):
            if feat in sector_means.columns:
                fig.add_trace(
                    go.Bar(x=sector_means.index, y=sector_means[feat], name=feat[:20]),
                    row=1, col=1
                )

    # 2. Correlation heatmap
    if len(available) >= 3:
        corr = df[available[:10]].corr()
        fig.add_trace(
            go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                       colorscale='RdBu_r', zmin=-1, zmax=1),
            row=1, col=2
        )

    # 3. Regional scatter (first 2 features)
    if 'region' in df.columns and len(available) >= 2:
        for region in df['region'].unique()[:5]:
            region_df = df[df['region'] == region]
            fig.add_trace(
                go.Scatter(x=region_df[available[0]], y=region_df[available[1]],
                           mode='markers', name=str(region)[:15], opacity=0.6),
                row=2, col=1
            )

    # 4. Box plots
    if 'sector' in df.columns and available:
        for feat in available[:2]:
            fig.add_trace(
                go.Box(y=df[feat], x=df['sector'], name=feat[:15]),
                row=2, col=2
            )

    fig.update_layout(
        title=f'<b>{category} Dashboard</b><br><sup>{len(available)} features available</sup>',
        template=PLOTLY_TEMPLATE, height=900, showlegend=True,
    )

    output_path = output_dir / f'{category.lower().replace(" ", "_").replace("&", "and")}_dashboard.html'
    fig.write_html(output_path)
    return output_path


# Generate dashboards for key categories
key_categories = [
    'Momentum & Technical', 'Valuation Ratios', 'Profitability',
    'Quality & Risk', 'Growth Metrics', 'Leverage & Liquidity',
    'Analyst Sentiment', 'Earnings Quality', 'Dividend Reliability',
]

for category in key_categories:
    if category in PHASE93_FEATURE_CATEGORIES:
        features = PHASE93_FEATURE_CATEGORIES[category]
        result = create_category_dashboard(df, category, features, category_viz_dir)
        if result:
            print(f'  ✓ Created: {result.name}')

print('\n✓ Phase 9.3 Category Dashboards Complete')

# %% md
## Cell 5: Regional Benchmarking Dashboard

Comprehensive
regional
analysis
with key financial metrics comparison.

# %%
# ============================================================================
# Cell 5: Regional Benchmarking Dashboard
# ============================================================================

print('=' * 80)
print('🌍 REGIONAL BENCHMARKING DASHBOARD')
print('=' * 80)

region_dir = OUTPUT_DIR / 'eda' / 'by_region'

if 'region' in df.columns:
    # Key metrics for regional comparison
    benchmark_metrics = [
        'roe', 'roa', 'p_e_ratio', 'debt_to_equity', 'price_momentum_1m',
        'earnings_quality_score', 'ema_trend_consistency', 'piotroski_f_score',
        'altman_z_score', 'dividend_yield', 'revenue_growth_yoy',
    ]
    available_metrics = [m for m in benchmark_metrics if m in df.columns]

    # ============================================================================
    # 5.1 Regional Statistics Table
    # ============================================================================
    print('\n📊 Computing Regional Statistics...')

    regional_stats = df.groupby('region')[available_metrics].agg(['mean', 'median', 'std', 'count'])
    regional_stats.columns = ['_'.join(col) for col in regional_stats.columns]

    # Save to JSON
    regional_stats_dict = regional_stats.to_dict()
    with open(region_dir / 'regional_statistics.json', 'w') as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   for k, v in regional_stats_dict.items()}, f, indent=2, default=str)
    print(f'  ✓ Saved: regional_statistics.json')

    # ============================================================================
    # 5.2 Regional Radar Charts
    # ============================================================================
    print('\n🎯 Creating Regional Radar Charts...')

    # Normalize metrics for radar chart
    radar_metrics = available_metrics[:8]  # Limit for readability
    regional_means = df.groupby('region')[radar_metrics].mean()

    # Z-score normalize for comparability
    regional_normalized = (regional_means - regional_means.mean()) / regional_means.std()
    regional_normalized = regional_normalized.fillna(0)

    fig_radar = go.Figure()
    for region in regional_normalized.index:
        fig_radar.add_trace(go.Scatterpolar(
            r=regional_normalized.loc[region].values.tolist() + [regional_normalized.loc[region].values[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill='toself', name=str(region), opacity=0.6,
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        title='<b>Regional Financial Profile Comparison</b><br><sup>Z-Score Normalized Metrics</sup>',
        template=PLOTLY_TEMPLATE, height=700,
    )
    fig_radar.write_html(region_dir / 'regional_radar_comparison.html')
    fig_radar.show()
    print(f'  ✓ Saved: regional_radar_comparison.html')

    # ============================================================================
    # 5.3 Regional Box Plots Grid
    # ============================================================================
    print('\n📦 Creating Regional Box Plot Grid...')

    n_metrics = min(6, len(available_metrics))
    rows = 2
    cols = 3

    fig_boxes = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[m.replace('_', ' ').title() for m in available_metrics[:n_metrics]],
    )

    for idx, metric in enumerate(available_metrics[:n_metrics]):
        row = idx // cols + 1
        col = idx % cols + 1

        for i, region in enumerate(df['region'].unique()):
            region_data = df[df['region'] == region][metric].dropna()
            fig_boxes.add_trace(
                go.Box(y=region_data, name=str(region)[:10],
                       marker_color=px.colors.qualitative.Set2[i % 8],
                       showlegend=(idx == 0)),
                row=row, col=col
            )

    fig_boxes.update_layout(
        title='<b>Key Metrics Distribution by Region</b>',
        template=PLOTLY_TEMPLATE, height=700, showlegend=True,
    )
    fig_boxes.write_html(region_dir / 'regional_boxplots.html')
    fig_boxes.show()
    print(f'  ✓ Saved: regional_boxplots.html')

print('\n✓ Regional Benchmarking Complete')

# %% md
## Cell 6: Sector Deep-Dive Analytics

# %%
# ============================================================================
# Cell 6: Sector Deep-Dive Analytics
# ============================================================================

print('=' * 80)
print('🏢 SECTOR DEEP-DIVE ANALYTICS')
print('=' * 80)

sector_dir = OUTPUT_DIR / 'eda' / 'by_sector'

if 'sector' in df.columns:
    # ============================================================================
    # 6.1 Sector Performance Heatmap
    # ============================================================================
    print('\n🔥 Creating Sector Performance Heatmap...')

    perf_metrics = [
        'roe', 'roa', 'price_momentum_1m', 'price_momentum_3m',
        'revenue_growth_yoy', 'earnings_quality_score', 'debt_to_equity',
    ]
    perf_metrics = [m for m in perf_metrics if m in df.columns]

    sector_perf = df.groupby('sector')[perf_metrics].mean()

    # Z-score normalize
    sector_perf_z = (sector_perf - sector_perf.mean()) / sector_perf.std()

    fig_heatmap = px.imshow(
        sector_perf_z.T, title='<b>Sector Performance Heatmap</b><br><sup>Z-Score Normalized</sup>',
        template=PLOTLY_TEMPLATE, color_continuous_scale='RdYlGn',
        aspect='auto', text_auto='.2f',
    )
    fig_heatmap.update_layout(height=500, xaxis_title='Sector', yaxis_title='Metric')
    fig_heatmap.write_html(sector_dir / 'sector_performance_heatmap.html')
    fig_heatmap.show()
    print(f'  ✓ Saved: sector_performance_heatmap.html')

    # ============================================================================
    # 6.2 Sector Feature Category Coverage
    # ============================================================================
    print('\n📋 Analyzing Feature Coverage by Sector...')

    sector_coverage = []
    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector]
        for category in list(PHASE93_FEATURE_CATEGORIES.keys())[:10]:  # Top 10 categories
            features = PHASE93_FEATURE_CATEGORIES[category]
            available = [f for f in features if f in sector_df.columns]
            if available:
                completeness = sector_df[available].notna().mean().mean() * 100
                sector_coverage.append({
                    'Sector': sector,
                    'Category': category,
                    'Completeness': completeness,
                })

    coverage_matrix = pd.DataFrame(sector_coverage).pivot(
        index='Category', columns='Sector', values='Completeness'
    )

    fig_coverage = px.imshow(
        coverage_matrix,
        title='<b>Phase 9.3 Feature Completeness by Sector × Category</b>',
        template=PLOTLY_TEMPLATE, color_continuous_scale='RdYlGn',
        aspect='auto', text_auto='.0f',
    )
    fig_coverage.update_layout(height=600)
    fig_coverage.write_html(sector_dir / 'sector_category_coverage.html')
    fig_coverage.show()
    print(f'  ✓ Saved: sector_category_coverage.html')

    # ============================================================================
    # 6.3 Sector Valuation Scatter
    # ============================================================================
    print('\n💰 Creating Sector Valuation Scatter...')

    if 'p_e_ratio' in df.columns and 'roe' in df.columns:
        fig_scatter = px.scatter(
            df[df['p_e_ratio'].between(0, 100) & df['roe'].between(-50, 100)],
            x='roe', y='p_e_ratio', color='sector',
            size='market_cap', size_max=30,
            hover_data=['ticker', 'last_price'],
            title='<b>Sector Valuation: ROE vs P/E Ratio</b><br><sup>Size = Market Cap</sup>',
            template=PLOTLY_TEMPLATE,
        )
        fig_scatter.update_layout(height=700)
        fig_scatter.write_html(sector_dir / 'sector_valuation_scatter.html')
        fig_scatter.show()
        print(f'  ✓ Saved: sector_valuation_scatter.html')

print('\n✓ Sector Deep-Dive Analytics Complete')

# %% md
## Cell 7: Earnings & Dividend Analytics (earnings_widgets.py Integration)

Comprehensive
earnings and dividend
dashboards
using
Phase
9.3
widgets.

# %%
# ============================================================================
# Cell 7: Earnings & Dividend Analytics
# ============================================================================

print('=' * 80)
print('📈 EARNINGS & DIVIDEND ANALYTICS')
print('=' * 80)

earnings_dir = OUTPUT_DIR / 'eda' / 'earnings_analytics'
dividend_dir = OUTPUT_DIR / 'eda' / 'dividend_visualizations'

# ============================================================================
# 7.1 Earnings Surprise Dashboard
# ============================================================================
print('\n📊 Creating Earnings Surprise Dashboard...')
fig_surprise = create_earnings_surprise_dashboard(df, reference_date=REFERENCE_DATE)
if fig_surprise:
    fig_surprise.write_html(earnings_dir / 'earnings_surprise_analysis.html')
    fig_surprise.show()
    print(f'  ✓ Saved: earnings_surprise_analysis.html')

# ============================================================================
# 7.2 Analyst Recommendation Heatmap
# ============================================================================
print('\n🎯 Creating Analyst Recommendation Heatmap...')
fig_analyst = create_analyst_recommendation_heatmap(df)
if fig_analyst:
    fig_analyst.write_html(earnings_dir / 'analyst_recommendations.html')
    fig_analyst.show()
    print(f'  ✓ Saved: analyst_recommendations.html')

# ============================================================================
# 7.3 Market Movers Dashboard
# ============================================================================
print('\n📈 Creating Market Movers Dashboard...')
fig_movers = create_market_movers_dashboard(df, reference_date=REFERENCE_DATE)
if fig_movers:
    fig_movers.write_html(earnings_dir / 'market_movers.html')
    fig_movers.show()
    print(f'  ✓ Saved: market_movers.html')

# ============================================================================
# 7.4 Price Target Analytics
# ============================================================================
print('\n🎯 Creating Price Target Analytics...')
fig_targets = create_price_target_analytics(df)
if fig_targets:
    fig_targets.write_html(earnings_dir / 'price_target_analysis.html')
    fig_targets.show()
    print(f'  ✓ Saved: price_target_analysis.html')

# ============================================================================
# 7.5 Technical & Valuation Dashboard
# ============================================================================
print('\n📊 Creating Technical Valuation Dashboard...')
fig_tech = create_technical_valuation_dashboard(df)
if fig_tech:
    fig_tech.write_html(earnings_dir / 'technical_valuation.html')
    fig_tech.show()
    print(f'  ✓ Saved: technical_valuation.html')

# ============================================================================
# 7.6 Dividend Sustainability Scorecard
# ============================================================================
print('\n💰 Creating Dividend Sustainability Scorecard...')
fig_div = create_dividend_sustainability_scorecard(df, output_path=dividend_dir / 'dividend_scorecard.html')
if fig_div:
    fig_div.show()
    print(f'  ✓ Saved: dividend_scorecard.html')

# ============================================================================
# 7.7 Earnings Quality Alerts
# ============================================================================
print('\n⚠️ Generating Earnings Quality Alerts...')
alert_config = EarningsAlertConfig()
alerts = generate_earnings_quality_alerts(df, config=alert_config, reference_date=REFERENCE_DATE)
with open(earnings_dir / 'earnings_alerts.json', 'w') as f:
    json.dump(alerts, f, indent=2, default=str)
print(f'  ✓ Saved: earnings_alerts.json ({len(alerts)} alerts)')

print('\n✓ Earnings & Dividend Analytics Complete')

# %% md
## Cell 8: Employee Productivity & Efficiency Analytics

# %%
# ============================================================================
# Cell 8: Employee Productivity & Efficiency Analytics
# ============================================================================

print('=' * 80)
print('👥 EMPLOYEE PRODUCTIVITY & EFFICIENCY ANALYTICS')
print('=' * 80)

employment_dir = OUTPUT_DIR / 'eda' / 'employment_analytics'

# ============================================================================
# 8.1 Employee Productivity Dashboard
# ============================================================================
print('\n📊 Creating Employee Productivity Dashboard...')
fig_emp = create_employee_productivity_dashboard(df, output_path=employment_dir / 'productivity_dashboard.html')
if fig_emp:
    fig_emp.show()
    print(f'  ✓ Saved: productivity_dashboard.html')

# ============================================================================
# 8.2 Efficiency Metrics by Sector
# ============================================================================
efficiency_metrics = [
    'asset_turnover', 'inventory_turnover', 'receivables_turnover',
    'revenue_per_employee', 'profit_per_employee',
]
efficiency_available = [m for m in efficiency_metrics if m in df.columns]

if efficiency_available and 'sector' in df.columns:
    print('\n⚙️ Analyzing Efficiency Metrics by Sector...')

    sector_efficiency = df.groupby('sector')[efficiency_available].mean()

    fig_eff = px.bar(
        sector_efficiency.reset_index().melt(id_vars='sector', var_name='Metric', value_name='Value'),
        x='sector', y='Value', color='Metric', barmode='group',
        title='<b>Efficiency Metrics by Sector</b>',
        template=PLOTLY_TEMPLATE,
    )
    fig_eff.update_layout(height=500, xaxis_tickangle=45)
    fig_eff.write_html(employment_dir / 'efficiency_by_sector.html')
    fig_eff.show()
    print(f'  ✓ Saved: efficiency_by_sector.html')

print('\n✓ Employee Productivity & Efficiency Analytics Complete')

# %% md
## Cell 9: Hypothesis Testing & Statistical Benchmarking

# %%
# ============================================================================
# Cell 9: Hypothesis Testing & Statistical Benchmarking
# ============================================================================

print('=' * 80)
print('📊 HYPOTHESIS TESTING & STATISTICAL BENCHMARKING')
print('=' * 80)

import scipy.stats as scipy_stats

stats_dir = OUTPUT_DIR / 'eda' / 'advanced_analytics'

# ============================================================================
# 9.1 ANOVA Tests by Sector
# ============================================================================
test_metrics = ['roe', 'price_momentum_1m', 'debt_to_equity', 'earnings_quality_score']
test_metrics = [m for m in test_metrics if m in df.columns]

if test_metrics and 'sector' in df.columns:
    print('\n📈 Running ANOVA Tests by Sector...')

    anova_results = []
    for metric in test_metrics:
        groups = [group[metric].dropna().values for name, group in df.groupby('sector')]
        groups = [g for g in groups if len(g) >= 5]  # Min samples

        if len(groups) >= 2:
            f_stat, p_value = scipy_stats.f_oneway(*groups)
            anova_results.append({
                'Metric': metric,
                'F_Statistic': round(f_stat, 4),
                'P_Value': round(p_value, 6),
                'Significant': p_value < 0.05,
            })

    anova_df = pd.DataFrame(anova_results)
    print(anova_df.to_string(index=False))

    # Save results
    anova_df.to_json(stats_dir / 'anova_by_sector.json', orient='records', indent=2)
    print(f'  ✓ Saved: anova_by_sector.json')

# ============================================================================
# 9.2 Hypothesis Test Heatmap
# ============================================================================
if test_metrics and 'sector' in df.columns:
    print('\n🔥 Creating Hypothesis Test Heatmap...')

    # Kruskal-Wallis tests (non-parametric)
    kw_matrix = pd.DataFrame(index=test_metrics, columns=['H_Statistic', 'P_Value'])

    for metric in test_metrics:
        groups = [group[metric].dropna().values for name, group in df.groupby('sector')]
        groups = [g for g in groups if len(g) >= 5]

        if len(groups) >= 2:
            h_stat, p_val = scipy_stats.kruskal(*groups)
            kw_matrix.loc[metric, 'H_Statistic'] = h_stat
            kw_matrix.loc[metric, 'P_Value'] = p_val

    # Create visualization
    fig_hyp = go.Figure(data=go.Heatmap(
        z=[[float(kw_matrix.loc[m, 'P_Value']) for m in test_metrics]],
        x=test_metrics, y=['Kruskal-Wallis'],
        colorscale='RdYlGn_r', zmin=0, zmax=0.1,
        text=[[f"p={float(kw_matrix.loc[m, 'P_Value']):.4f}" for m in test_metrics]],
        texttemplate='%{text}', textfont={'size': 12},
    ))
    fig_hyp.update_layout(
        title='<b>Hypothesis Test Results: Sector Differences</b><br><sup>Kruskal-Wallis Test (Green = Significant)</sup>',
        template=PLOTLY_TEMPLATE, height=300,
    )
    fig_hyp.write_html(stats_dir / 'hypothesis_test_heatmap.html')
    fig_hyp.show()
    print(f'  ✓ Saved: hypothesis_test_heatmap.html')

print('\n✓ Hypothesis Testing Complete')

# %% md
## Cell 10: Dashboard Summary & Artifact Index

# %%
# ============================================================================
# Cell 10: Dashboard Summary & Artifact Index
# ============================================================================

print('=' * 80)
print('📋 DASHBOARD SUMMARY & ARTIFACT INDEX')
print('=' * 80)

# Collect all generated artifacts
artifact_index = {
    'generated_at': datetime.now().isoformat(),
    'reference_date': REFERENCE_DATE.isoformat(),
    'data_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
    'phase93_coverage': coverage_stats,
    'artifacts': {},
}

# Scan output directories
for subdir in ['dashboards', 'by_region', 'by_sector', 'by_industry', 'by_exchange',
               'earnings_analytics', 'dividend_visualizations', 'employment_analytics',
               'advanced_analytics', 'earnings_visualizations', 'visualizations']:
    dir_path = OUTPUT_DIR / 'eda' / subdir
    if dir_path.exists():
        files = list(dir_path.glob('*.html')) + list(dir_path.glob('*.json'))
        artifact_index['artifacts'][subdir] = [f.name for f in files]

# Save artifact index
with open(OUTPUT_DIR / 'eda' / 'artifact_index.json', 'w') as f:
    json.dump(artifact_index, f, indent=2, default=str)

# Print summary
print('\n📁 Generated Artifacts:')
for category, files in artifact_index['artifacts'].items():
    if files:
        print(f'\n  {category}/:')
        for f in files[:5]:
            print(f'    • {f}')
        if len(files) > 5:
            print(f'    ... and {len(files) - 5} more')

total_artifacts = sum(len(f) for f in artifact_index['artifacts'].values())
print(f'\n✓ Total Artifacts Generated: {total_artifacts}')
print(f'✓ Artifact Index: outputs/eda/artifact_index.json')

# Phase 9.3 Coverage Summary
print('\n' + '=' * 80)
print('📊 PHASE 9.3 FEATURE COVERAGE SUMMARY')
print('=' * 80)

print(f'\nCategories: {len(PHASE93_FEATURE_CATEGORIES)}')
print(f'Total Features Registered: {sum(len(f) for f in PHASE93_FEATURE_CATEGORIES.values())}')
print(f'Features Present in Data: {total_phase93}')
print(f'Overall Coverage: {total_phase93 / sum(len(f) for f in PHASE93_FEATURE_CATEGORIES.values()) * 100:.1f}%')

print('\n✓ Stock Analytics Dashboard Complete!')
```

This enhanced `stock_analytics.ipynb` provides:

## 📊 Key Enhancements

### 1. **Structured by Feature Categories**

All 21 Phase 9.3 feature categories are systematically analyzed with dedicated dashboards.

### 2. **Multi-Dimensional Analysis**

- **By Region**: Sunburst charts, radar comparisons, box plot grids
- **By Sector**: Performance heatmaps, valuation scatter, coverage analysis
- **By Industry**: Distribution charts, market cap treemaps
- **By Exchange**: Market-level analytics with sector breakdowns

### 3. **Integrated Artifacts**

Uses the existing artifacts from `outputs/eda/`:

- `visualizations/` - Sector benchmarking, hypothesis tests
- `earnings_analytics/` - Earnings surprises, market movers
- `dividend_visualizations/` - Dividend yield by sector
- `advanced_analytics/` - VaR, risk attribution, statistical benchmarking

### 4. **`earnings_widgets.py` Integration**

Full integration of all dashboard functions:

- `create_earnings_surprise_dashboard()`
- `create_analyst_recommendation_heatmap()`
- `create_market_movers_dashboard()`
- `create_price_target_analytics()`
- `create_technical_valuation_dashboard()`
- `create_dividend_sustainability_scorecard()`
- `create_employee_productivity_dashboard()`
- `generate_earnings_quality_alerts()`

### 5. **Statistical Benchmarking**

- ANOVA tests by sector
- Kruskal-Wallis non-parametric tests
- Hypothesis test heatmaps
- Regional/sector statistical comparisons

### 6. **Artifact Index**

Generates `artifact_index.json` cataloging all outputs for downstream consumption.