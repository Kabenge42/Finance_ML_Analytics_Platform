#!/usr/bin/env python
# coding: utf-8

# # ETL Data Explorer
#
# This notebook explores the unified ETL (Extract, Transform, Load) Pipeline for stock data.
#
# **Version:** 1.0.0 | **Model Version:** v9_9
#
# ## Pipeline Stages
# 1. **Extract** - Load from DB with CSV fallback
# 2. **Transform** - Normalize, validate, sanitize, impute (6-step)
# 3. **Load** - Quality validation and finalization
#
# ## 16 Feature Categories (196 Features)
# - Momentum & Technical, Valuation Ratios, Profitability, Quality & Risk
# - Cash Flow, Capital Allocation, Analyst Sentiment, Market Sentiment
# - Leverage & Liquidity, Temporal Patterns, Composite Scores, Growth Metrics
# - Efficiency Ratios, Employee Productivity, Balance Sheet, Revenue Forecasting
#
# ## Feature Engineering API (`build_features`)
#
# **Business Goal:** Engineer comprehensive financial features including valuation ratios,
# profitability metrics, quality indicators, and sector-specific features to maximize model predictive power.
#
# **Key Objectives:**
# - Engineer valuation ratios (P/E, P/B, EV/EBITDA, PEG)
# - Engineer profitability features (margins, ROE, ROA, ROIC)
# - Create momentum and technical indicators
# - Engineer analyst quality features
# - Create accounting quality scores (Altman Z, Piotroski F)
# - Build sector-relative features
# - Create interaction features
#

# In[1]:


# ============================================================================
# Cell 1: Configuration & Setup
# ============================================================================
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# NumPy compatibility note:
# Avoid modifying NumPy private attributes to satisfy PyDeprecationInspection and stability concerns.
# If certain libraries expect np._ARRAY_API, prefer updating those libraries rather than mutating NumPy.
# Here, we intentionally do NOT set any private compatibility shims.

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
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"

try:
    (OUTPUT_DIR / "eda").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "preprocessing").mkdir(parents=True, exist_ok=True)
except Exception as dir_err:
    # Graceful handling for permission/disk issues without crashing the notebook
    print(f"Warning: could not ensure output directories exist: {dir_err}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finance_ml.notebook_config import NotebookConfig

CFG = NotebookConfig(
    have_finance_prediction=True,
    have_database_connection=True,
    have_advanced_analytics=True,
    have_dim_reduction=False,
    debug_mode=False,
)

# Section 17 Style Guidelines
plt.style.use("dark_background")
sns.set_palette("husl")
PLOTLY_TEMPLATE = "plotly_dark"
COLOR_PALETTE = {
    "primary": "#375a7f",
    "secondary": "#6c757d",
    "success": "#00bc8c",
    "warning": "#f39c12",
    "danger": "#e74c3c",
    "info": "#3498db",
    "neutral": "#adb5bd",
}


def convert_jdbc_to_sqlalchemy(jdbc_url: str) -> str:
    """Convert a JDBC PostgreSQL URL to a SQLAlchemy URL preserving credentials when present.

    Supported inputs:
    - jdbc:postgresql://host:port/db
    - jdbc:postgresql://user:password@host:port/db
    - jdbc:postgresql://host/db (port optional)

    Returns a postgresql+psycopg2 URL string.
    """
    import re

    pattern = re.compile(
        r"^jdbc:postgresql://(?:(?P<user>[^:@/]+)(?::(?P<pw>[^@/]*))?@)?(?P<host>[^:/?#]+)(?::(?P<port>\d+))?/(?P<db>[^?]+)"
    )
    m = pattern.match(jdbc_url)
    if not m:
        # Fallback: return as-is for caller to handle
        return jdbc_url
    user = m.group("user")
    pw = m.group("pw") or ""
    host = m.group("host")
    port = m.group("port") or "5432"
    db = m.group("db")
    if user:
        return f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    # No credentials embedded - use default postgres user with empty password
    return f"postgresql+psycopg2://postgres:@{host}:{port}/{db}"


def check_db_connection(db_url: str) -> bool:
    """Return True if a quick test connection to the database succeeds, else False.

    Uses SQLAlchemy if available and performs a trivial SELECT 1. Exceptions are not raised.
    """
    if not HAVE_SQLALCHEMY or not db_url:
        return False
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


# Database URL (env var or safe default)
DB_URL_ENV = os.getenv("DB_URL")
if DB_URL_ENV and DB_URL_ENV.startswith("jdbc:"):
    DB_URL = convert_jdbc_to_sqlalchemy(DB_URL_ENV)
elif DB_URL_ENV:
    DB_URL = DB_URL_ENV
else:
    # Safe default with postgres user and empty password; aligns with environment_variables.txt
    DB_URL = "postgresql+psycopg2://postgres:bItcfiTg142!@localhost:5432/postgres"

print("=" * 60)
print("ETL DATA EXPLORER - CONFIGURATION")
print("=" * 60)
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"Python: {sys.version.split()[0]}")
CFG.display_summary()


# ## Cell 2: Import Finance ML Modules
#
# Import unified ETL pipeline, EDA analytics, and feature engineering modules.
#

# In[2]:


# ============================================================================
# Cell 2: Import Finance ML Modules
# ============================================================================

# ETL Pipeline (Phase 9.1)
from finance_ml.ml_workflow.preprocessing.etl import (
    run_etl_pipeline,
    etl_with_features,
    ETLConfig,
    ETLMetrics,
    etl_with_imputation,
    etl_from_csv,
)

# EDA Analytics (Phase 9.2)
from finance_ml.ml_workflow.eda.eda import (
    eda_summary,
    generate_phase93_coverage_report,
    sector_distribution_summary,
    correlation_analysis,
)

# Phase 9.3 Feature Categories
from finance_ml.ml_workflow.eda.phase93_categories import (
    PHASE93_FEATURE_CATEGORIES,
    categorize_dataframe_columns,
    get_phase93_coverage_stats,
    get_category_description,
    list_all_phase93_features,
)

# Feature Engineering API (Phase 9.3)
from finance_ml.ml_workflow.features.api import build_features

# Column Semantics and Safety (Section 8.5)
from finance_ml.ml_workflow.preprocessing.column_semantics import (
    PRICE_COLUMNS,
    get_winsorizable_columns,
    get_scalable_columns,
    classify_columns,
)

print("✓ Finance ML modules imported successfully")
print(f"✓ Phase 9.3 Feature Categories: {len(PHASE93_FEATURE_CATEGORIES)} categories")
print(f"✓ Total Phase 9.3 Features: {len(list_all_phase93_features())} features")
print(f"✓ Price Columns Protected: {len(PRICE_COLUMNS)} columns")
print(f"\nFeature Engineering Presets Available:")
print(f"  - 'basic': core ratios, margins, volatility, revenue CAGR")
print(f"  - 'momentum': momentum & technical indicators")
print(f"  - 'quality': accounting quality and financial distress signals")
print(f"  - 'comprehensive': full advanced feature set (196 features)")


# ## Cell 3: Database Configuration
#
# Configure PostgreSQL database connection with automatic URL format conversion.
#

# In[3]:


# ============================================================================
# Cell 3: Database Configuration
# ============================================================================

# SQL file paths
SQL_SCHEMA = PROJECT_ROOT / "create_equities_schema.sql"
SQL_IMPORT = PROJECT_ROOT / "import_equities_data.sql"

# Detect availability
have_db_url = DB_URL is not None and len(DB_URL) > 0
reachable_db = check_db_connection(DB_URL) if have_db_url else False
CFG.have_database_connection = bool(reachable_db)

print("=" * 60)
print("DATABASE CONFIGURATION")
print("=" * 60)
print(f"SQLAlchemy installed:  {HAVE_SQLALCHEMY}")
print(f"DB_URL configured:     {have_db_url}")
print(f"Database available:    {CFG.have_database_connection}")

if have_db_url:
    # Mask password for security
    try:
        parts = DB_URL.split("@")
        if len(parts) == 2:
            user_part = parts[0].split("//")[-1].split(":")[0]
            masked_url = f"postgresql://{user_part}@{parts[1]}"
        else:
            masked_url = "postgresql://***@localhost:5432/postgres"
    except Exception:
        masked_url = "postgresql://***"
    print(f"Connection:            {masked_url}")
else:
    print("Connection:            Not configured (will use CSV fallback)")

print(f"\nSQL Files:")
print(f"  Schema script:       {'✓' if SQL_SCHEMA.exists() else '✗'} {SQL_SCHEMA.name}")
print(f"  Import script:       {'✓' if SQL_IMPORT.exists() else '✗'} {SQL_IMPORT.name}")
print("=" * 60)


# ## Cell 4: ETL Pipeline - Extract, Transform, Load
#
# Run the unified ETL pipeline with 6-step imputation strategy.
#

# In[4]:


# ============================================================================
# Cell 4: ETL Pipeline - Extract, Transform, Load
# ============================================================================

print("=" * 60)
print("ETL PIPELINE EXECUTION")
print("=" * 60)

# Complete ETL + features + financial metrics
all_stocks_preprocessed, metrics = etl_with_features(
    source="csv",
    data_dir="data/",
    db_url=DB_URL,
    feature_preset="comprehensive",
    return_metrics=True,
)

# Display ETL metrics summary
print("\n" + "=" * 60)
print(metrics.summary())
print("=" * 60)

# Validation checkpoint
assert not all_stocks_preprocessed.empty, "Preprocessed data must not be empty"
assert "ticker" in all_stocks_preprocessed.columns, "ticker column must be present"
assert "sector" in all_stocks_preprocessed.columns, "sector column must be present"

print(f"\n✓ ETL Pipeline Complete")
print(f"  Data shape: {all_stocks_preprocessed.shape}")
print(f"  Source: {metrics.source_type}")
print(f"  Duration: {metrics.total_time_sec:.2f}s")
print(f"  Quality score: {metrics.quality_score:.3f}")


# ## Cell 5: Data Quality Overview
#
# Examine data shape, dtypes, and missing values post-imputation.
#

# In[5]:


# ============================================================================
# Cell 5: Data Quality Overview
# ============================================================================

print("=" * 60)
print("DATA QUALITY OVERVIEW")
print("=" * 60)

# Basic info
print(
    f"\nDataFrame Shape: {all_stocks_preprocessed.shape[0]:,} rows × {all_stocks_preprocessed.shape[1]} columns"
)
print(f"Memory Usage: {all_stocks_preprocessed.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

# Data types summary
dtype_counts = all_stocks_preprocessed.dtypes.value_counts()
print(f"\nData Types:")
for dtype, count in dtype_counts.items():
    print(f"  {dtype}: {count} columns")

# Missing values (post-imputation)
missing = all_stocks_preprocessed.isnull().sum()
missing_pct = (missing / len(all_stocks_preprocessed) * 100).round(2)
missing_df = (
    pd.DataFrame(
        {"Column": missing.index, "Missing": missing.values, "Percent": missing_pct.values}
    )
    .query("Missing > 0")
    .sort_values("Missing", ascending=False)
)

if len(missing_df) > 0:
    print(f"\n⚠ Columns with missing values (Top 10):")
    print(missing_df.head(10).to_string(index=False))
else:
    print(f"\n✓ No missing values - imputation successful!")

# Summary statistics for key columns
print(f"\nSummary Statistics (key numeric columns):")
key_cols = ["last_price", "market_cap", "enterprise_value", "ebitda_ltm", "p_e_ntm"]
available_keys = [c for c in key_cols if c in all_stocks_preprocessed.columns]
if available_keys:
    print(all_stocks_preprocessed[available_keys].describe().round(2).to_string())

# ============================================================================
# Semantic Classification Display
# ============================================================================
print(f'\n{"=" * 60}')
print("SEMANTIC CLASSIFICATION")
print("=" * 60)

semantic_classification = classify_columns(all_stocks_preprocessed.columns.tolist())

print(f"\nColumn Classification by Semantic Category:")
for category, columns in semantic_classification.items():
    if columns:
        print(f"\n{category.upper()}: {len(columns)} columns")
        columns_list = sorted(columns)  # Convert set to sorted list
        for col in columns_list[:3]:  # Now slicing works
            print(f"  - {col}")
        if len(columns) > 3:
            print(f"  ... and {len(columns) - 3} more")

# ============================================================================
# Price Column Preservation Visualization
# ============================================================================
print(f'\n{"=" * 60}')
print("PRICE COLUMN PRESERVATION")
print("=" * 60)

price_cols_in_df = [c for c in all_stocks_preprocessed.columns if c in PRICE_COLUMNS]
print(f"\nPrice columns preserved: {len(price_cols_in_df)}")

for col in price_cols_in_df[:10]:  # Show first 10
    if col in all_stocks_preprocessed.columns:
        min_val = all_stocks_preprocessed[col].min()
        max_val = all_stocks_preprocessed[col].max()
        non_null = all_stocks_preprocessed[col].notna().sum()
        print(f"  - {col}: range=[{min_val:.2f}, {max_val:.2f}], non-null={non_null:,}")

if "last_price" in all_stocks_preprocessed.columns:
    price_range = (
        all_stocks_preprocessed["last_price"].max() - all_stocks_preprocessed["last_price"].min()
    )
    print(f"\n✓ Last Price range: ${price_range:.2f} (original dollar units preserved)")

# ============================================================================
# Log-Transformed Columns Analysis
# ============================================================================
print(f'\n{"=" * 60}')
print("LOG-TRANSFORMED COLUMNS")
print("=" * 60)

log_transformed = [c for c in all_stocks_preprocessed.columns if c.startswith("log_")]
print(f"\nLog-transformed columns: {len(log_transformed)}")

for col in log_transformed[:10]:  # Show first 10
    non_null = all_stocks_preprocessed[col].notna().sum()
    mean_val = all_stocks_preprocessed[col].mean()
    std_val = all_stocks_preprocessed[col].std()
    print(f"  - {col}: mean={mean_val:.2f}, std={std_val:.2f}, non-null={non_null:,}")

if len(log_transformed) > 10:
    print(f"  ... and {len(log_transformed) - 10} more")

print(f"\n✓ Cell 5 Complete: Data quality, semantic classification, and transformations validated")


# In[6]:


all_stocks_preprocessed.head(100)


# ## Cell 7: Region & Sector Analytics with Feature Coverage
#
# Section 17 compliant Plotly visualizations with dark theme.
# Enhanced with Phase 9.3 feature analytics by region and sector.
#

# In[7]:


# ============================================================================
# Cell 7: Region & Sector Analytics with Feature Coverage
# ============================================================================

print("=" * 80)
print("REGION & SECTOR ANALYTICS WITH FEATURE COVERAGE")
print("=" * 80)

# Use all_stocks_features (post feature engineering) for analytics
df_analytics = all_stocks_preprocessed

# Region distribution
if "region" in df_analytics.columns:
    region_counts = df_analytics["region"].value_counts().reset_index()
    region_counts.columns = ["Region", "Count"]
    region_counts["Percentage"] = (
        region_counts["Count"] / region_counts["Count"].sum() * 100
    ).round(1)

    fig_region = px.bar(
        region_counts,
        x="Count",
        y="Region",
        orientation="h",
        title="Stock Distribution by Region (Post Feature Engineering)",
        template=PLOTLY_TEMPLATE,
        color="Count",
        color_continuous_scale="Blues",
        text="Count",
    )
    fig_region.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata[0]:.1f}%",
        customdata=region_counts[["Percentage"]].values,
    )
    fig_region.update_layout(
        xaxis_title="Number of Stocks",
        yaxis_title="Region",
        height=400,
        showlegend=False,
    )
    fig_region.show()

# Sector distribution (Top 15)
if "sector" in df_analytics.columns:
    sector_counts = df_analytics["sector"].value_counts().head(15).reset_index()
    sector_counts.columns = ["Sector", "Count"]
    sector_counts["Percentage"] = (sector_counts["Count"] / len(df_analytics) * 100).round(1)

    fig_sector = px.bar(
        sector_counts.sort_values("Count"),
        x="Count",
        y="Sector",
        orientation="h",
        title="Stock Distribution by Sector (Top 15, Post Feature Engineering)",
        template=PLOTLY_TEMPLATE,
        color="Count",
        color_continuous_scale="Greens",
        text="Count",
    )
    fig_sector.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Count: %{x}<br>Percentage: %{customdata[0]:.1f}%",
        customdata=sector_counts.sort_values("Count")[["Percentage"]].values,
    )
    fig_sector.update_layout(
        xaxis_title="Number of Stocks",
        yaxis_title="Sector",
        height=500,
        showlegend=False,
    )
    fig_sector.show()

# Region-Sector Heatmap
if "region" in df_analytics.columns and "sector" in df_analytics.columns:
    cross_tab = pd.crosstab(df_analytics["sector"], df_analytics["region"]).head(12)

    fig_heatmap = px.imshow(
        cross_tab,
        title="Region-Sector Distribution Heatmap",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Viridis",
        aspect="auto",
    )
    fig_heatmap.update_layout(
        xaxis_title="Region",
        yaxis_title="Sector",
        height=500,
    )
    fig_heatmap.show()

print("\n✓ Distribution visualizations complete")

# ============================================================================
# Feature Coverage Analytics by Region and Sector
# ============================================================================
print("\n" + "=" * 80)
print("📊 PHASE 9.3 FEATURE ANALYTICS BY REGION & SECTOR")
print("=" * 80)

# Define key Phase 9.3 features for analysis
key_phase93_features = [
    "roe",
    "roa",
    "roic",  # Profitability
    "p_e_ratio",
    "p_s_ratio",
    "ev_ebitda_ratio",  # Valuation
    "price_momentum_1m",
    "price_momentum_3m",
    "price_momentum_6m",  # Momentum
    "debt_to_equity",
    "interest_coverage",  # Leverage
    "piotroski_f_score",
    "altman_z_score",  # Composite Scores
]
available_phase93 = [f for f in key_phase93_features if f in df_analytics.columns]

if available_phase93 and "region" in df_analytics.columns:
    print(f"\n📍 Feature Coverage by Region ({len(available_phase93)} key features):")

    # Calculate non-null percentage per region for each feature
    region_coverage_data = []
    for region in df_analytics["region"].unique():
        region_df = df_analytics[df_analytics["region"] == region]
        region_count = len(region_df)

        for feat in available_phase93:
            non_null = region_df[feat].notna().sum()
            coverage_pct = (non_null / region_count * 100) if region_count > 0 else 0
            region_coverage_data.append(
                {
                    "Region": region,
                    "Feature": feat,
                    "Coverage %": round(coverage_pct, 1),
                    "Non-Null": non_null,
                    "Total": region_count,
                }
            )

    region_coverage_df = pd.DataFrame(region_coverage_data)

    # Pivot for heatmap
    region_pivot = region_coverage_df.pivot(index="Feature", columns="Region", values="Coverage %")

    fig_region_coverage = px.imshow(
        region_pivot,
        title="Phase 9.3 Feature Coverage by Region (%)",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        aspect="auto",
    )
    fig_region_coverage.update_layout(
        xaxis_title="Region",
        yaxis_title="Feature",
        height=500,
    )
    fig_region_coverage.show()

if available_phase93 and "sector" in df_analytics.columns:
    print(
        f"\n🏢 Feature Coverage by Sector (Top 10 sectors, {len(available_phase93)} key features):"
    )

    # Get top 10 sectors by count
    top_sectors = df_analytics["sector"].value_counts().head(10).index.tolist()

    # Calculate non-null percentage per sector for each feature
    sector_coverage_data = []
    for sector in top_sectors:
        sector_df = df_analytics[df_analytics["sector"] == sector]
        sector_count = len(sector_df)

        for feat in available_phase93:
            non_null = sector_df[feat].notna().sum()
            coverage_pct = (non_null / sector_count * 100) if sector_count > 0 else 0
            sector_coverage_data.append(
                {
                    "Sector": str(sector)[:25],  # Truncate long names
                    "Feature": feat,
                    "Coverage %": round(coverage_pct, 1),
                    "Non-Null": non_null,
                    "Total": sector_count,
                }
            )

    sector_coverage_df = pd.DataFrame(sector_coverage_data)

    # Pivot for heatmap
    sector_pivot = sector_coverage_df.pivot(index="Feature", columns="Sector", values="Coverage %")

    fig_sector_coverage = px.imshow(
        sector_pivot,
        title="Phase 9.3 Feature Coverage by Sector (%)",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        aspect="auto",
    )
    fig_sector_coverage.update_layout(
        xaxis_title="Sector",
        yaxis_title="Feature",
        height=600,
    )
    fig_sector_coverage.show()

# Feature statistics by region
if available_phase93 and "region" in df_analytics.columns:
    print(f"\n📊 Key Feature Statistics by Region:")

    # Select 3 key features for detailed statistics
    stat_features = ["roe", "price_momentum_1m", "debt_to_equity"]
    stat_features = [f for f in stat_features if f in df_analytics.columns]

    if stat_features:
        region_stats = (
            df_analytics.groupby("region")[stat_features].agg(["mean", "median", "std"]).round(3)
        )
        print(region_stats.to_string())

        # Box plot for ROE by region (if available)
        if "roe" in df_analytics.columns:
            fig_roe_region = px.box(
                df_analytics[df_analytics["roe"].notna()],
                x="region",
                y="roe",
                title="Return on Equity (ROE) Distribution by Region",
                template=PLOTLY_TEMPLATE,
                color="region",
            )
            fig_roe_region.update_layout(
                xaxis_title="Region",
                yaxis_title="ROE",
                height=450,
                showlegend=False,
            )
            fig_roe_region.show()

print("\n✓ Feature analytics by region and sector complete")


# ## Cell 8: Numeric Feature Distributions
#
# Examine distributions of key numeric metrics with statistical annotations (using engineered features).
#

# In[8]:


# ============================================================================
# Cell 8: Numeric Feature Distributions
# ============================================================================

# Key metrics for comprehensive feature store (includes Phase 9.3 engineered features)
key_metrics = [
    "last_price",
    "market_cap",
    "enterprise_value",
    "ebitda_ltm",
    "p_e_ntm",
    "beta_5y",
    "roe",
    "roa",
    "price_momentum_1m",
    "piotroski_f_score",  # Phase 9.3 features
]
available_metrics = [m for m in key_metrics if m in all_stocks_preprocessed.columns]

if available_metrics:
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.ravel() if n_metrics > 1 else [axes]

    for i, metric in enumerate(available_metrics):
        data = all_stocks_preprocessed[metric].dropna()

        if len(data) > 0:
            ax = axes[i]
            sns.histplot(data, bins=50, kde=True, ax=ax, color=COLOR_PALETTE["primary"])

            # Add statistics
            mean_val = data.mean()
            median_val = data.median()

            ax.axvline(
                mean_val,
                color=COLOR_PALETTE["warning"],
                linestyle="--",
                label=f"Mean: {mean_val:,.2f}",
                linewidth=2,
            )
            ax.axvline(
                median_val,
                color=COLOR_PALETTE["success"],
                linestyle="--",
                label=f"Median: {median_val:,.2f}",
                linewidth=2,
            )

            ax.set_title(f"Distribution: {metric}", fontsize=14, fontweight="bold")
            ax.set_xlabel(metric.replace("_", " ").title())
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"✓ Visualized {len(available_metrics)} key metrics")
else:
    print("⚠ No key metrics found in dataset")

# Correlation heatmap for key metrics (using engineered features)
numeric_cols = all_stocks_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 4:
    # Select top correlated features (prioritize Phase 9.3 features)
    corr_subset = numeric_cols[:15]  # Limit for readability
    corr_matrix = all_stocks_preprocessed[corr_subset].corr()

    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Heatmap",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(height=600)
    fig_corr.show()


# ## Cell 9: Phase 9.3 Feature Category Coverage
#
# Analyze coverage of the 16 Phase 9.3 feature categories (196 features) after feature engineering.
#

# In[9]:


# ============================================================================
# Cell 9: Phase 9.3 Feature Category Coverage
# ============================================================================

print("=" * 80)
print("PHASE 9.3 FEATURE CATEGORY COVERAGE (Post Feature Engineering)")
print("=" * 80)

# Generate comprehensive coverage report using engineered features
coverage_report = generate_phase93_coverage_report(all_stocks_preprocessed)

# Build coverage DataFrame
coverage_data = []
for cat, features in PHASE93_FEATURE_CATEGORIES.items():
    present_count = coverage_report.get("category_breakdown", {}).get(cat, 0)
    total_count = len(features)
    coverage_pct = (present_count / total_count * 100) if total_count > 0 else 0
    coverage_data.append(
        {
            "Category": cat,
            "Present": present_count,
            "Total": total_count,
            "Coverage %": round(coverage_pct, 1),
            "Description": get_category_description(cat)[:40] + "...",
        }
    )

coverage_df = pd.DataFrame(coverage_data).sort_values("Present", ascending=False)

print(
    f"\nTotal Phase 9.3 Features Present: {coverage_report.get('total_phase93_features', 0)} / {len(list_all_phase93_features())}"
)
print(f"Overall Coverage: {coverage_report.get('coverage_percentage', 0):.1f}%\n")
print(coverage_df.to_string(index=False))

# Treemap visualization
treemap_data = coverage_df[coverage_df["Present"] > 0].copy()
if not treemap_data.empty:
    fig_treemap = px.treemap(
        treemap_data,
        path=["Category"],
        values="Present",
        title="Phase 9.3 Feature Coverage by Category",
        template=PLOTLY_TEMPLATE,
        color="Coverage %",
        color_continuous_scale="Viridis",
        hover_data=["Description", "Total"],
    )
    fig_treemap.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Present: %{value}<br>Coverage: %{color:.1f}%<extra></extra>",
    )
    fig_treemap.update_layout(height=600)
    fig_treemap.show()

# Bar chart for coverage comparison
fig_coverage = px.bar(
    coverage_df.sort_values("Coverage %", ascending=True),
    x="Coverage %",
    y="Category",
    orientation="h",
    title="Phase 9.3 Feature Coverage by Category",
    template=PLOTLY_TEMPLATE,
    color="Coverage %",
    color_continuous_scale="RdYlGn",
    text="Present",
)
fig_coverage.update_traces(texttemplate="%{text}", textposition="outside")
fig_coverage.update_layout(
    xaxis_title="Coverage Percentage",
    yaxis_title="Feature Category",
    height=600,
    showlegend=False,
)
fig_coverage.show()

print("\n✓ Phase 9.3 coverage analysis complete")


# ## Cell 10: Financial Metrics Analytics
#
# Comprehensive financial analytics with statistical summaries, hypothesis testing, and interactive visualizations.
#
# **Key Objectives:**
# 1. Generate comprehensive statistical summaries and financial analytics reports
# 2. Analyze correlations and multicollinearity between features
# 3. Perform hypothesis testing across features, sectors, industry, country and regions
# 4. Create interactive visualizations for financial metrics distributions and relationships
# 5. Generate benchmarking reports comparing sector and regional financial/market performance
#
# **Outputs:**
# - **JSON Reports (4 files):** eda_summary.json, data_quality_alerts.json, metrics_dashboard.json, hypothesis_tests.json
# - **Interactive Visualizations (7 HTML files):** correlation_heatmap.html, distributions.html, valuation_3d.html, region_sector_heatmap.html, sector_boxplots.html, regional_comparison.html, phase93_category_sector_bubble_chart.html
#

# In[10]:


# ============================================================================
# Cell 10: Financial Metrics Analytics
# ============================================================================

import scipy.stats as scipy_stats
from datetime import datetime

# Import analytics functions
from finance_ml.ml_workflow.analytics.eval import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    perform_comprehensive_hypothesis_tests,
    calculate_correlation_matrix,
    find_top_correlations,
    create_region_sector_heatmap,
)
from finance_ml.ml_workflow.eda.eda import eda_summary
from finance_ml.ml_workflow.eda.reports import generate_benchmarking_report

print("=" * 80)
print("FINANCIAL METRICS ANALYTICS")
print("=" * 80)

# Create output directories
financial_metrics_dir = OUTPUT_DIR / "eda" / "financial_metrics"
financial_metrics_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Generate JSON Reports
# ============================================================================
print("\n📊 Generating JSON Reports...")

# 1.1 EDA Summary Report
print("  → eda_summary.json")
eda_summary_data = eda_summary(
    all_stocks_preprocessed, sector_column="sector", include_correlations=True
)
eda_summary_path = financial_metrics_dir / "eda_summary.json"
with open(eda_summary_path, "w") as f:
    # Convert non-serializable objects
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        return obj

    json.dump(make_serializable(eda_summary_data), f, indent=2, default=str)
print(f"    ✓ Saved: {eda_summary_path}")

# 1.2 Data Quality Alerts Report
print("  → data_quality_alerts.json")
quality_alerts = generate_data_quality_alerts(all_stocks_preprocessed, outlier_threshold=3.0)
quality_alerts_path = financial_metrics_dir / "data_quality_alerts.json"
with open(quality_alerts_path, "w") as f:
    json.dump(make_serializable(quality_alerts), f, indent=2, default=str)
print(f"    ✓ Saved: {quality_alerts_path}")

# 1.3 Financial Metrics Dashboard
print("  → metrics_dashboard.json")
# By sector
metrics_by_sector = calculate_financial_metrics_dashboard(
    all_stocks_preprocessed, group_by="sector"
)
# By region
metrics_by_region = calculate_financial_metrics_dashboard(
    all_stocks_preprocessed, group_by="region"
)
metrics_dashboard = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks": len(all_stocks_preprocessed),
    "by_sector": make_serializable(metrics_by_sector),
    "by_region": make_serializable(metrics_by_region),
}
metrics_dashboard_path = financial_metrics_dir / "metrics_dashboard.json"
with open(metrics_dashboard_path, "w") as f:
    json.dump(metrics_dashboard, f, indent=2, default=str)
print(f"    ✓ Saved: {metrics_dashboard_path}")

# 1.4 Hypothesis Tests Report
print("  → hypothesis_tests.json")
test_metrics = ["roe", "roa", "p_e_ratio", "debt_to_equity", "price_momentum_1m"]
test_metrics = [m for m in test_metrics if m in all_stocks_preprocessed.columns]
hypothesis_results = perform_comprehensive_hypothesis_tests(
    all_stocks_preprocessed, group_column="sector", metrics=test_metrics, alpha=0.05
)
hypothesis_path = financial_metrics_dir / "hypothesis_tests.json"
with open(hypothesis_path, "w") as f:
    json.dump(make_serializable(hypothesis_results), f, indent=2, default=str)
print(f"    ✓ Saved: {hypothesis_path}")

print(f"\n✓ JSON Reports Complete: 4 files generated")

# ============================================================================
# 2. Generate Interactive HTML Visualizations
# ============================================================================
print("\n📈 Generating Interactive HTML Visualizations...")

# 2.1 Correlation Heatmap (Top 50 features)
print("  → correlation_heatmap.html")
numeric_features = all_stocks_preprocessed.select_dtypes(include=[np.number]).columns.tolist()
# Select top 50 most complete numeric features
feature_completeness = (
    all_stocks_preprocessed[numeric_features].notna().sum().sort_values(ascending=False)
)
top_50_features = feature_completeness.head(50).index.tolist()

corr_matrix = all_stocks_preprocessed[top_50_features].corr()

# Cluster the correlation matrix for better visualization
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

# Handle NaN in correlation matrix
corr_matrix_filled = corr_matrix.fillna(0)
try:
    # Compute distance matrix and linkage
    dist_matrix = 1 - np.abs(corr_matrix_filled)
    np.fill_diagonal(dist_matrix.values, 0)
    linkage = hierarchy.linkage(squareform(dist_matrix), method="average")
    order = hierarchy.leaves_list(linkage)
    corr_clustered = corr_matrix_filled.iloc[order, order]
except (ValueError, FloatingPointError):
    corr_clustered = corr_matrix_filled

fig_corr = px.imshow(
    corr_clustered,
    title="<b>Feature Correlation Heatmap</b><br><sup>Top 50 Features (Clustered)</sup>",
    template=PLOTLY_TEMPLATE,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
)
fig_corr.update_layout(
    height=900,
    width=1000,
    font=dict(family="Segoe UI, Roboto, Arial", size=10),
    title_font_size=20,
    xaxis_title="Features",
    yaxis_title="Features",
)
fig_corr.write_html(financial_metrics_dir / "correlation_heatmap.html")
print(f"    ✓ Saved: correlation_heatmap.html")

# 2.2 Feature Distributions by Sector
print("  → distributions.html")
key_distribution_features = [
    "roe",
    "roa",
    "p_e_ratio",
    "debt_to_equity",
    "piotroski_f_score",
    "altman_z_score",
]
key_distribution_features = [
    f for f in key_distribution_features if f in all_stocks_preprocessed.columns
]

if key_distribution_features and "sector" in all_stocks_preprocessed.columns:
    # Create subplots for distributions
    fig_dist = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f.replace("_", " ").title() for f in key_distribution_features[:6]],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    colors = [
        COLOR_PALETTE["primary"],
        COLOR_PALETTE["success"],
        COLOR_PALETTE["warning"],
        COLOR_PALETTE["danger"],
        COLOR_PALETTE["info"],
        COLOR_PALETTE["neutral"],
    ]

    for idx, feat in enumerate(key_distribution_features[:6]):
        row = idx // 3 + 1
        col = idx % 3 + 1

        data = all_stocks_preprocessed[feat].dropna()
        fig_dist.add_trace(
            go.Histogram(
                x=data,
                name=feat,
                marker_color=colors[idx % len(colors)],
                opacity=0.7,
                nbinsx=50,
                hovertemplate=f"<b>{feat}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>",
            ),
            row=row,
            col=col,
        )
        fig_dist.update_xaxes(title_text=feat.replace("_", " ").title(), row=row, col=col)
        fig_dist.update_yaxes(title_text="Frequency", row=row, col=col)

    fig_dist.update_layout(
        title="<b>Financial Metrics Distributions</b><br><sup>Key Features Across All Stocks</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        showlegend=False,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
    )
    fig_dist.write_html(financial_metrics_dir / "distributions.html")
    print(f"    ✓ Saved: distributions.html")

# 2.3 3D Valuation Scatter (Category-Sector-Market Cap)
print("  → valuation_3d.html")
# Select features for 3D visualization
val_features = {
    "x": "roe" if "roe" in all_stocks_preprocessed.columns else "roa",
    "y": (
        "piotroski_f_score"
        if "piotroski_f_score" in all_stocks_preprocessed.columns
        else "altman_z_score"
    ),
    "z": "market_cap" if "market_cap" in all_stocks_preprocessed.columns else "enterprise_value",
}

if all(f in all_stocks_preprocessed.columns or f is None for f in val_features.values()):
    # Prepare data for 3D scatter
    df_3d = all_stocks_preprocessed[
        ["ticker", "sector", "region"] + list(val_features.values())
    ].dropna()

    # Log transform market cap for better visualization
    if "market_cap" in df_3d.columns:
        df_3d["market_cap_log"] = np.log10(df_3d["market_cap"].clip(lower=1))
        z_col = "market_cap_log"
        z_label = "Market Cap (Log10 $)"
    else:
        z_col = val_features["z"]
        z_label = val_features["z"].replace("_", " ").title()

    fig_3d = px.scatter_3d(
        df_3d.head(2000),  # Limit for performance
        x=val_features["x"],
        y=val_features["y"],
        z=z_col,
        color="sector",
        symbol="region",
        hover_data=["ticker"],
        title="<b>Value vs Quality vs Size Trade-offs</b><br><sup>3D Category-Sector-Market Cap Analysis</sup>",
        template=PLOTLY_TEMPLATE,
        opacity=0.7,
    )
    fig_3d.update_layout(
        height=800,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        scene=dict(
            xaxis_title=val_features["x"].replace("_", " ").upper(),
            yaxis_title=val_features["y"].replace("_", " ").title(),
            zaxis_title=z_label,
        ),
    )
    fig_3d.write_html(financial_metrics_dir / "valuation_3d.html")
    print(f"    ✓ Saved: valuation_3d.html")

# 2.4 Region-Sector Heatmap
print("  → region_sector_heatmap.html")
if "region" in all_stocks_preprocessed.columns and "sector" in all_stocks_preprocessed.columns:
    # Create pivot table for region-sector distribution
    region_sector_pivot = pd.crosstab(
        all_stocks_preprocessed["sector"], all_stocks_preprocessed["region"], margins=True
    )

    # Remove margins for heatmap
    region_sector_data = region_sector_pivot.iloc[:-1, :-1]

    fig_rs_heatmap = px.imshow(
        region_sector_data,
        title="<b>Regional Financial Analytics Distribution</b><br><sup>Stock Count by Sector and Region</sup>",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="Blues",
        aspect="auto",
        text_auto=True,
    )
    fig_rs_heatmap.update_layout(
        height=700,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        xaxis_title="Region",
        yaxis_title="Sector",
    )
    fig_rs_heatmap.write_html(financial_metrics_dir / "region_sector_heatmap.html")
    print(f"    ✓ Saved: region_sector_heatmap.html")

# 2.5 Sector Boxplots
print("  → sector_boxplots.html")
boxplot_metrics = ["roe", "roa", "debt_to_equity", "price_momentum_1m"]
boxplot_metrics = [m for m in boxplot_metrics if m in all_stocks_preprocessed.columns]

if boxplot_metrics and "sector" in all_stocks_preprocessed.columns:
    fig_box = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[m.replace("_", " ").title() for m in boxplot_metrics[:4]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for idx, metric in enumerate(boxplot_metrics[:4]):
        row = idx // 2 + 1
        col = idx % 2 + 1

        # Get top 10 sectors by count
        top_sectors = all_stocks_preprocessed["sector"].value_counts().head(10).index.tolist()
        df_filtered = all_stocks_preprocessed[all_stocks_preprocessed["sector"].isin(top_sectors)]

        for i, sector in enumerate(top_sectors):
            sector_data = df_filtered[df_filtered["sector"] == sector][metric].dropna()
            fig_box.add_trace(
                go.Box(
                    y=sector_data,
                    name=str(sector)[:15],
                    marker_color=px.colors.qualitative.Set3[i % 12],
                    showlegend=(idx == 0),
                    hovertemplate=f"<b>{sector}</b><br>{metric}: %{{y:.2f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig_box.update_yaxes(title_text=metric.replace("_", " ").title(), row=row, col=col)

    fig_box.update_layout(
        title="<b>Financial Metrics by Sector</b><br><sup>Box Plots for Key Metrics (Top 10 Sectors)</sup>",
        template=PLOTLY_TEMPLATE,
        height=800,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    fig_box.write_html(financial_metrics_dir / "sector_boxplots.html")
    print(f"    ✓ Saved: sector_boxplots.html")

# 2.6 Regional Comparison
print("  → regional_comparison.html")
if "region" in all_stocks_preprocessed.columns:
    comparison_metrics = ["roe", "roa", "p_e_ratio", "debt_to_equity", "piotroski_f_score"]
    comparison_metrics = [m for m in comparison_metrics if m in all_stocks_preprocessed.columns]

    if comparison_metrics:
        # Calculate mean metrics by region
        regional_means = all_stocks_preprocessed.groupby("region")[comparison_metrics].mean()

        fig_regional = go.Figure()

        for i, region in enumerate(regional_means.index):
            fig_regional.add_trace(
                go.Scatterpolar(
                    r=regional_means.loc[region].values,
                    theta=[m.replace("_", " ").title() for m in comparison_metrics],
                    fill="toself",
                    name=region,
                    opacity=0.6,
                )
            )

        fig_regional.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[regional_means.min().min() * 0.8, regional_means.max().max() * 1.2],
                )
            ),
            title="<b>Financial Metrics by Region</b><br><sup>Radar Chart Comparison</sup>",
            template=PLOTLY_TEMPLATE,
            height=600,
            font=dict(family="Segoe UI, Roboto, Arial"),
            title_font_size=20,
            showlegend=True,
        )
        fig_regional.write_html(financial_metrics_dir / "regional_comparison.html")
        print(f"    ✓ Saved: regional_comparison.html")

# 2.7 Phase 9.3 Category-Sector Bubble Chart
print("  → phase93_category_sector_bubble_chart.html")
if "sector" in all_stocks_preprocessed.columns:
    # Calculate average coverage by sector for each Phase 9.3 category
    bubble_data = []
    top_sectors = all_stocks_preprocessed["sector"].value_counts().head(8).index.tolist()

    for sector in top_sectors:
        sector_df = all_stocks_preprocessed[all_stocks_preprocessed["sector"] == sector]
        sector_count = len(sector_df)

        for category, features in PHASE93_FEATURE_CATEGORIES.items():
            available_features = [f for f in features if f in sector_df.columns]
            if available_features:
                coverage = sector_df[available_features].notna().mean().mean() * 100
                feature_count = len(available_features)
                bubble_data.append(
                    {
                        "Sector": str(sector)[:20],
                        "Category": category,
                        "Coverage %": round(coverage, 1),
                        "Feature Count": feature_count,
                        "Stock Count": sector_count,
                    }
                )

    bubble_df = pd.DataFrame(bubble_data)

    if not bubble_df.empty:
        fig_bubble = px.scatter(
            bubble_df,
            x="Category",
            y="Sector",
            size="Coverage %",
            color="Coverage %",
            color_continuous_scale="RdYlGn",
            hover_data=["Feature Count", "Stock Count"],
            title="<b>Phase 9.3 Feature Coverage by Sector</b><br><sup>Bubble Size = Coverage Percentage</sup>",
            template=PLOTLY_TEMPLATE,
        )
        fig_bubble.update_layout(
            height=700,
            font=dict(family="Segoe UI, Roboto, Arial"),
            title_font_size=20,
            xaxis_title="Phase 9.3 Category",
            yaxis_title="Sector",
            xaxis_tickangle=45,
        )
        fig_bubble.write_html(financial_metrics_dir / "phase93_category_sector_bubble_chart.html")
        print(f"    ✓ Saved: phase93_category_sector_bubble_chart.html")

print(f"\n✓ Interactive HTML Visualizations Complete: 7 files generated")
print(f"\n📁 All outputs saved to: {financial_metrics_dir}")

# ============================================================================
# 3. Display Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("📊 FINANCIAL METRICS SUMMARY")
print("=" * 80)

# Hypothesis test summary
if hypothesis_results and "summary" in hypothesis_results:
    print(f"\n🔬 Hypothesis Testing Results:")
    print(f"   Tests performed: {hypothesis_results.get('n_tests', len(test_metrics))}")
    print(f"   Significance level: α = 0.05")
    if "significant_differences" in hypothesis_results:
        sig_count = sum(1 for v in hypothesis_results["significant_differences"].values() if v)
        print(f"   Significant differences found: {sig_count}/{len(test_metrics)} metrics")

# Quality alerts summary
if quality_alerts:
    print(f"\n⚠️ Data Quality Alerts:")
    if "outlier_counts" in quality_alerts:
        total_outliers = sum(quality_alerts["outlier_counts"].values())
        print(f"   Total outliers detected: {total_outliers:,}")
    if "missing_critical" in quality_alerts:
        print(
            f"   Missing critical values: {len(quality_alerts.get('missing_critical', []))} columns"
        )

# Correlation summary
print(f"\n🔗 Correlation Analysis:")
print(f"   Features analyzed: {len(top_50_features)}")
top_correlations = find_top_correlations(corr_matrix, n_top=5, threshold=0.7)
if len(top_correlations) > 0:  # find_top_correlations returns a list of tuples, not a DataFrame
    print(f"   High correlations (>0.7): {len(top_correlations)} pairs")
    print(
        f"   Top correlation: {top_correlations[0][2]:.3f}"
    )  # Access tuple element (var1, var2, correlation)

print("\n✓ Financial Metrics Analytics complete")


# ## Cell 10.5: Enhanced Financial Metrics & Price Target Analytics
#
# Run the **unified ETL pipeline** with comprehensive valuation, profitability, growth, and leverage metrics.
# Uses the consolidated `etl_with_financial_metrics()` function from `finance_ml.ml_workflow.preprocessing.etl`.
#
# **Key Objectives:**
# 1. Compute comprehensive financial metrics using unified ETL pipeline
# 2. Analyze price target upside/downside across sectors and regions
# 3. Generate interactive visualizations (scatter plots, bar charts with confidence bands)
# 4. Export valuation_opportunities.json and multi_dimensional_valuation_analysis.json
#
# **Visualizations:**
# - **Price Target Scatter Plot:** Last Price vs Price Target with sector coloring
# - **Price Target Distribution Bar Chart:** By sector with confidence bands
# - **EMA Comparison Chart:** 20D, 50D, 100D, 250D EMAs vs Last Price
# - **52W High/Low Analysis:** Position within 52-week range
#
# **Outputs:**
# - **JSON Reports:** valuation_opportunities.json, multi_dimensional_valuation_analysis.json
# - **Interactive HTML Charts:** price_target_scatter.html, price_target_by_sector.html
#

# In[11]:


# ============================================================================
# Cell 10.5: Enhanced Financial Metrics & Price Target Analytics
# ============================================================================
# Import unified ETL pipeline (replaces deprecated financial_metrics_etl)
from finance_ml.ml_workflow.preprocessing.etl import (
    etl_with_financial_metrics,
    ETLConfig,
    compute_valuation_metrics,
    compute_profitability_metrics,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_target_vs_price_metrics,
    generate_metrics_dashboard,
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

print("=" * 80)
print("ENHANCED FINANCIAL METRICS & PRICE TARGET ANALYTICS")
print("Using unified ETL pipeline (etl.py)")
print("=" * 80)

# ============================================================================
# 1. Compute Financial Metrics using Unified ETL Functions
# ============================================================================
print("\n🔧 Computing Financial Metrics...")

# Apply financial metrics to existing preprocessed data
all_stocks_enhanced = all_stocks_preprocessed.copy()

# Initialize fin_metrics dictionary to track metrics added
fin_metrics = {
    "valuation_metrics_added": 0,
    "profitability_metrics_added": 0,
    "growth_metrics_added": 0,
    "leverage_metrics_added": 0,
    "target_vs_price_metrics_added": 0,
    "sector_specific_metrics_added": 0,
}

# Compute valuation metrics (P/E, P/S, EV/EBITDA, EV/Sales)
all_stocks_enhanced = compute_valuation_metrics(all_stocks_enhanced)
valuation_cols_added = ["p_e", "p_s", "ev_ebitda", "ev_sales"]
fin_metrics["valuation_metrics_added"] = len(
    [c for c in valuation_cols_added if c in all_stocks_enhanced.columns]
)
print(f'  ✓ Valuation metrics: {fin_metrics["valuation_metrics_added"]} added')

# Compute profitability metrics (margins, ROE, ROA)
all_stocks_enhanced = compute_profitability_metrics(all_stocks_enhanced)
profit_cols_added = ["gross_margin", "operating_margin", "net_margin", "roe", "roa"]
fin_metrics["profitability_metrics_added"] = len(
    [c for c in profit_cols_added if c in all_stocks_enhanced.columns]
)
print(f'  ✓ Profitability metrics: {fin_metrics["profitability_metrics_added"]} added')

# Compute growth metrics (revenue, EBITDA, earnings growth)
all_stocks_enhanced = compute_growth_metrics(all_stocks_enhanced)
growth_cols_added = ["revenue_growth", "ebitda_growth", "earnings_growth"]
fin_metrics["growth_metrics_added"] = len(
    [c for c in growth_cols_added if c in all_stocks_enhanced.columns]
)
print(f'  ✓ Growth metrics: {fin_metrics["growth_metrics_added"]} added')

# Compute leverage metrics (debt ratios)
all_stocks_enhanced = compute_leverage_metrics(all_stocks_enhanced)
leverage_cols_added = ["debt_to_equity", "debt_to_assets"]
fin_metrics["leverage_metrics_added"] = len(
    [c for c in leverage_cols_added if c in all_stocks_enhanced.columns]
)
print(f'  ✓ Leverage metrics: {fin_metrics["leverage_metrics_added"]} added')

# Compute target vs price metrics
all_stocks_enhanced = compute_target_vs_price_metrics(all_stocks_enhanced)
target_cols_added = ["target_vs_price", "target_vs_price_median", "target_spread_pct"]
fin_metrics["target_vs_price_metrics_added"] = len(
    [c for c in target_cols_added if c in all_stocks_enhanced.columns]
)
print(f'  ✓ Target vs price metrics: {fin_metrics["target_vs_price_metrics_added"]} added')

print(f"\n✓ Financial Metrics Complete: {all_stocks_enhanced.shape[1]} total columns")

# ============================================================================
# 2. Price Target Analytics with Visualizations
# ============================================================================
print("\n📊 Price Target Analytics & Visualizations...")

# Define key price-related columns for analysis
price_cols = [
    "last_price",
    "price_target_ytd_ago",
    "price_target",
    "price_target_low",
    "price_target_median",
    "price_target_high",
    "price_target_count",
    "52w_high_adj",
    "52w_low_adj",
    "ema_20d",
    "ema_50d",
    "ema_100d",
    "ema_250d",
]
available_price_cols = [c for c in price_cols if c in all_stocks_enhanced.columns]
print(f"  Available price columns: {len(available_price_cols)}/{len(price_cols)}")

# ============================================================================
# 2.1 Price Target Scatter Plot: Last Price vs Price Target (Log-Scaled with Confidence Bands)
# ============================================================================
if "last_price" in all_stocks_enhanced.columns and "price_target" in all_stocks_enhanced.columns:
    print("\n  📈 Creating Price Target Scatter Plot (Log-Scaled)...")

    # Filter valid data for visualization - include confidence bounds
    required_cols = ["ticker", "sector", "last_price", "price_target", "target_vs_price"]
    confidence_cols = ["price_target_low", "price_target_high"]
    available_cols = required_cols + [
        col for col in confidence_cols if col in all_stocks_enhanced.columns
    ]

    scatter_data = all_stocks_enhanced[available_cols].dropna(subset=required_cols)

    # Limit to reasonable price range for visualization
    scatter_data = scatter_data[
        (scatter_data["last_price"] > 0)
        & (scatter_data["last_price"] < 1000000)
        & (scatter_data["price_target"] > 0)
        & (scatter_data["price_target"] < 2000000)
    ]

    if len(scatter_data) > 0:
        # Apply log10 transformation
        scatter_data["last_price_log"] = np.log10(scatter_data["last_price"])
        scatter_data["price_target_log"] = np.log10(scatter_data["price_target"])

        # Add log transformations for confidence bounds if available
        has_confidence = (
            "price_target_low" in scatter_data.columns
            and "price_target_high" in scatter_data.columns
        )
        if has_confidence:
            scatter_data["price_target_low_log"] = np.log10(
                scatter_data["price_target_low"].clip(lower=0.01)
            )
            scatter_data["price_target_high_log"] = np.log10(
                scatter_data["price_target_high"].clip(lower=0.01)
            )

        # Create figure with go.Figure for confidence bands
        fig_scatter = go.Figure()

        # Get unique sectors and color palette
        sectors = scatter_data["sector"].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {sector: colors[i % len(colors)] for i, sector in enumerate(sectors)}

        # Add scatter points and confidence bands by sector
        for sector in sectors:
            sector_data = scatter_data[scatter_data["sector"] == sector].copy()

            # Add confidence band if available
            if has_confidence:
                # Sort by last_price_log for proper polygon rendering
                sector_data_sorted = sector_data.sort_values(by="last_price_log")

                # Create confidence band trace (invisible, for fill)
                fig_scatter.add_trace(
                    go.Scatter(
                        x=sector_data_sorted["last_price_log"],
                        y=sector_data_sorted["price_target_low_log"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{sector} Low",
                    )
                )

                fig_scatter.add_trace(
                    go.Scatter(
                        x=sector_data_sorted["last_price_log"],
                        y=sector_data_sorted["price_target_high_log"],
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=f"rgba{tuple(list(int(color_map[sector].lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + [0.15])}",
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{sector} High",
                    )
                )

            # Add scatter points
            fig_scatter.add_trace(
                go.Scatter(
                    x=sector_data["last_price_log"],
                    y=sector_data["price_target_log"],
                    mode="markers",
                    name=sector,
                    marker=dict(
                        color=color_map[sector],
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color="white"),
                    ),
                    customdata=np.column_stack(
                        (
                            sector_data["ticker"],
                            sector_data["target_vs_price"],
                            sector_data["last_price"],
                            sector_data["price_target"],
                        )
                    ),
                    hovertemplate="<b>%{customdata[0]}</b><br>"
                    + "Sector: "
                    + sector
                    + "<br>"
                    + "Last Price: $%{customdata[2]:.2f}<br>"
                    + "Price Target: $%{customdata[3]:.2f}<br>"
                    + "Target vs Price: %{customdata[1]:.1f}%<br>"
                    + "Log10(Last Price): %{x:.3f}<br>"
                    + "Log10(Price Target): %{y:.3f}<br>"
                    + "<extra></extra>",
                )
            )

        # Add diagonal reference line in log space (Price Target = Last Price)
        min_log = scatter_data["last_price_log"].min()
        max_log = max(scatter_data["last_price_log"].max(), scatter_data["price_target_log"].max())
        fig_scatter.add_trace(
            go.Scatter(
                x=[min_log, max_log],
                y=[min_log, max_log],
                mode="lines",
                name="Fair Value Line",
                line=dict(dash="dash", color="gray", width=2),
                showlegend=True,
                hoverinfo="skip",
            )
        )

        fig_scatter.update_layout(
            title="<b>Price Target vs Last Price by Sector (Log-Scaled)</b><br>"
            + "<sup>Log10 transformation | Dots above diagonal = Analyst upside potential"
            + (" | Shaded bands = Confidence intervals" if has_confidence else "")
            + "</sup>",
            xaxis_title="Log10(Last Price)",
            yaxis_title="Log10(Price Target)",
            template=PLOTLY_TEMPLATE,
            height=700,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            hovermode="closest",
        )

        scatter_path = financial_metrics_dir / "price_target_scatter.html"
        fig_scatter.write_html(str(scatter_path))
        print(f"    ✓ Saved: {scatter_path}")
        if has_confidence:
            print(f"    ✓ Confidence bands included from price_target_low and price_target_high")
        else:
            print(
                f"    ⚠ Confidence bands not available (missing price_target_low or price_target_high)"
            )
        fig_scatter.show()
    else:
        print("    ⚠ Insufficient data for scatter plot")

# ============================================================================
# 2.2 Price Target Upside by Sector (Bar Chart with Confidence Bands)
# ============================================================================
if "target_vs_price" in all_stocks_enhanced.columns and "sector" in all_stocks_enhanced.columns:
    print("\n  📊 Creating Price Target by Sector Bar Chart...")

    sector_stats = (
        all_stocks_enhanced.groupby("sector")["target_vs_price"]
        .agg(["mean", "std", "count", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .reset_index()
    )
    sector_stats.columns = ["sector", "mean", "std", "count", "q25", "q75"]

    # Filter sectors with sufficient data
    sector_stats = sector_stats[sector_stats["count"] >= 10].sort_values("mean", ascending=True)

    if len(sector_stats) > 0:
        fig_bar = go.Figure()

        # Add bar chart
        fig_bar.add_trace(
            go.Bar(
                y=sector_stats["sector"],
                x=sector_stats["mean"],
                orientation="h",
                name="Mean Upside (%)",
                marker_color=sector_stats["mean"].apply(lambda x: "green" if x > 0 else "red"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=sector_stats["q75"] - sector_stats["mean"],
                    arrayminus=sector_stats["mean"] - sector_stats["q25"],
                    color="rgba(0,0,0,0.3)",
                ),
                hovertemplate="<b>%{y}</b><br>Mean: %{x:.1f}%<br>Q25-Q75: %{customdata[0]:.1f}% - %{customdata[1]:.1f}%<extra></extra>",
                customdata=sector_stats[["q25", "q75"]].values,
            )
        )

        fig_bar.update_layout(
            title="<b>Price Target Upside by Sector</b><br><sup>With 25th-75th percentile confidence bands</sup>",
            xaxis_title="Target vs Price Upside (%)",
            yaxis_title="Sector",
            template=PLOTLY_TEMPLATE,
            height=500,
            showlegend=False,
        )

        # Add vertical line at 0%
        fig_bar.add_vline(x=0, line_dash="dash", line_color="gray")

        bar_path = financial_metrics_dir / "price_target_by_sector.html"
        fig_bar.write_html(str(bar_path))
        print(f"    ✓ Saved: {bar_path}")
        fig_bar.show()
    else:
        print("    ⚠ Insufficient sector data for bar chart")

# ============================================================================
# 2.3 EMA Comparison Chart (20D, 50D, 100D, 250D vs Last Price)
# ============================================================================
ema_cols = ["ema_20d", "ema_50d", "ema_100d", "ema_250d"]
available_emas = [c for c in ema_cols if c in all_stocks_enhanced.columns]

if "last_price" in all_stocks_enhanced.columns and len(available_emas) >= 2:
    print("\n  📉 Creating EMA Comparison Chart...")

    # Calculate EMA position relative to last price
    ema_analysis = []
    for ema_col in available_emas:
        valid_data = all_stocks_enhanced[["last_price", ema_col]].dropna()
        if len(valid_data) > 0:
            above_ema = (valid_data["last_price"] > valid_data[ema_col]).sum()
            below_ema = (valid_data["last_price"] <= valid_data[ema_col]).sum()
            ema_analysis.append(
                {
                    "EMA": ema_col.replace("_", " ").upper(),
                    "Above EMA (%)": above_ema / len(valid_data) * 100,
                    "Below EMA (%)": below_ema / len(valid_data) * 100,
                    "Count": len(valid_data),
                }
            )

    if ema_analysis:
        ema_df = pd.DataFrame(ema_analysis)

        fig_ema = go.Figure()
        fig_ema.add_trace(
            go.Bar(
                x=ema_df["EMA"], y=ema_df["Above EMA (%)"], name="Above EMA", marker_color="green"
            )
        )
        fig_ema.add_trace(
            go.Bar(x=ema_df["EMA"], y=ema_df["Below EMA (%)"], name="Below EMA", marker_color="red")
        )

        fig_ema.update_layout(
            title="<b>Stock Position vs Exponential Moving Averages</b><br><sup>Percentage of stocks above/below each EMA</sup>",
            xaxis_title="Moving Average",
            yaxis_title="Percentage of Stocks (%)",
            barmode="group",
            template=PLOTLY_TEMPLATE,
            height=400,
        )

        ema_path = financial_metrics_dir / "ema_comparison.html"
        fig_ema.write_html(str(ema_path))
        print(f"    ✓ Saved: {ema_path}")
        fig_ema.show()

# ============================================================================
# 2.4 52-Week High/Low Position Analysis
# ============================================================================
if "last_price" in all_stocks_enhanced.columns and "52w_high_adj" in all_stocks_enhanced.columns:
    print("\n  📊 Creating 52-Week Range Position Chart...")

    range_data = all_stocks_enhanced[
        ["ticker", "sector", "last_price", "52w_high_adj", "52w_low_adj"]
    ].dropna()

    if len(range_data) > 0 and "52w_low_adj" in range_data.columns:
        # Calculate position within 52W range (0% = at low, 100% = at high)
        range_data["position_52w"] = (
            (range_data["last_price"] - range_data["52w_low_adj"])
            / (range_data["52w_high_adj"] - range_data["52w_low_adj"])
            * 100
        ).clip(0, 100)

        # Distribution by sector
        sector_52w = (
            range_data.groupby("sector")["position_52w"].agg(["mean", "std", "count"]).reset_index()
        )
        sector_52w = sector_52w[sector_52w["count"] >= 10].sort_values("mean", ascending=True)

        if len(sector_52w) > 0:
            fig_52w = px.bar(
                sector_52w,
                y="sector",
                x="mean",
                orientation="h",
                error_x="std",
                title="<b>52-Week Range Position by Sector</b><br><sup>0% = At 52W Low, 100% = At 52W High</sup>",
                labels={"mean": "Position within 52W Range (%)", "sector": "Sector"},
                color="mean",
                color_continuous_scale="RdYlGn",
            )

            fig_52w.update_layout(template=PLOTLY_TEMPLATE, height=500, showlegend=False)
            fig_52w.add_vline(
                x=50, line_dash="dash", line_color="gray", annotation_text="Mid-range"
            )

            range_path = financial_metrics_dir / "52w_range_position.html"
            fig_52w.write_html(str(range_path))
            print(f"    ✓ Saved: {range_path}")
            fig_52w.show()

# ============================================================================
# 3. Valuation Opportunities Analysis
# ============================================================================
print("\n📐 Generating Valuation Opportunities Analysis...")

# Categorize stocks by valuation
if "target_vs_price" in all_stocks_enhanced.columns:
    valuation_opportunities = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks_analyzed": len(all_stocks_enhanced),
        "valuation_summary": {
            "mean_target_vs_price": float(all_stocks_enhanced["target_vs_price"].mean()),
            "median_target_vs_price": float(all_stocks_enhanced["target_vs_price"].median()),
            "std_target_vs_price": float(all_stocks_enhanced["target_vs_price"].std()),
        },
        "category_distribution": {},
        "top_undervalued": [],
        "top_overvalued": [],
    }

    # Categorize stocks
    def categorize_valuation(upside):
        if upside > 50:
            return "Deeply Undervalued"
        elif upside > 15:
            return "Undervalued"
        elif upside >= -15:
            return "Fairly Valued"
        elif upside >= -30:
            return "Overvalued"
        else:
            return "Deeply Overvalued"

    all_stocks_enhanced["valuation_category"] = all_stocks_enhanced["target_vs_price"].apply(
        categorize_valuation
    )
    valuation_opportunities["category_distribution"] = (
        all_stocks_enhanced["valuation_category"].value_counts().to_dict()
    )

    # Top undervalued stocks
    top_under = all_stocks_enhanced.nlargest(10, "target_vs_price")[
        ["ticker", "sector", "last_price", "price_target", "target_vs_price"]
    ]
    valuation_opportunities["top_undervalued"] = top_under.to_dict("records")

    # Top overvalued stocks
    top_over = all_stocks_enhanced.nsmallest(10, "target_vs_price")[
        ["ticker", "sector", "last_price", "price_target", "target_vs_price"]
    ]
    valuation_opportunities["top_overvalued"] = top_over.to_dict("records")

    # Save valuation opportunities JSON
    val_opp_path = financial_metrics_dir / "valuation_opportunities.json"
    with open(val_opp_path, "w") as f:
        json.dump(make_serializable(valuation_opportunities), f, indent=2, default=str)
    print(f"  ✓ Saved: {val_opp_path}")

    # Print summary
    print(f"\n  Valuation Category Distribution:")
    for cat, count in valuation_opportunities["category_distribution"].items():
        print(f"    {cat:20s}: {count:,}")

# ============================================================================
# 4. Multi-Dimensional Valuation Analysis
# ============================================================================
print("\n📐 Multi-Dimensional Valuation Analysis...")
valuation_metrics = ["p_e_ratio", "p_s_ratio", "ev_ebitda_ratio", "ev_sales_ratio", "roe", "roa"]
available_val_metrics = [m for m in valuation_metrics if m in all_stocks_enhanced.columns]

if len(available_val_metrics) >= 2:
    multi_dim_valuation = {
        "timestamp": datetime.now().isoformat(),
        "total_stocks_analyzed": len(all_stocks_enhanced),
        "dimensions": {},
        "correlations": {},
    }

    # Compute statistics for each valuation dimension
    for metric in available_val_metrics:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            multi_dim_valuation["dimensions"][metric] = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "std": float(data.std()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
            }

    # Correlation analysis
    if len(available_val_metrics) >= 2:
        val_corr = all_stocks_enhanced[available_val_metrics].corr()
        for i, m1 in enumerate(available_val_metrics):
            for j, m2 in enumerate(available_val_metrics):
                if i < j:
                    corr_val = val_corr.loc[m1, m2]
                    if not np.isnan(corr_val):
                        multi_dim_valuation["correlations"][f"{m1}_vs_{m2}"] = float(corr_val)

    # Save multi-dimensional analysis
    multi_dim_path = financial_metrics_dir / "multi_dimensional_valuation_analysis.json"
    with open(multi_dim_path, "w") as f:
        json.dump(make_serializable(multi_dim_valuation), f, indent=2, default=str)
    print(f"  ✓ Saved: {multi_dim_path}")

# ============================================================================
# 5. Generate Financial Metrics Dashboard
# ============================================================================
print("\n💾 Generating Financial Metrics Dashboard...")
financial_dashboard = generate_metrics_dashboard(all_stocks_enhanced, sector_column="sector")
financial_dashboard["price_target_analytics"] = (
    valuation_opportunities if "valuation_opportunities" in dir() else {}
)

dashboard_path = financial_metrics_dir / "financial_metrics_dashboard.json"
with open(dashboard_path, "w") as f:
    json.dump(make_serializable(financial_dashboard), f, indent=2, default=str)
print(f"  ✓ Saved: {dashboard_path}")

print("\n" + "=" * 80)
print("✓ Enhanced Financial Metrics & Price Target Analytics Complete")
print("=" * 80)
print(f"  Total columns: {all_stocks_enhanced.shape[1]}")
print(f"  Visualizations saved to: {financial_metrics_dir}")


# ## Cell 10.6: Earnings Monitoring
#
# Monitor earnings-related metrics including revenue growth, EBITDA trends, margin analysis, and earnings quality indicators.
#
# **Key Objectives:**
# 1. Track earnings growth trends across sectors and regions
# 2. Analyze margin compression/expansion patterns
# 3. Monitor earnings quality and consistency
# 4. Generate earnings monitoring dashboard
#
# **Outputs:**
# - **JSON Report:** earnings_monitor.json
#

# In[12]:


# ============================================================================
# Cell 10.6: Earnings Monitoring
# ============================================================================

print("=" * 80)
print("EARNINGS MONITORING")
print("=" * 80)

# ============================================================================
# 1. Revenue & Earnings Growth Analysis
# ============================================================================
print("\n📈 Revenue & Earnings Growth Analysis...")

earnings_metrics = {
    "revenue_growth": ["revenues_est_yoy_pct_fy1e"],
    "ebitda_growth": [
        "ev_ebitda_ltm",
        "ev_ebitda_ntm",
        "ev_ebitda_1fyltm",
        "ev_ebitda_1fqltm",
        "ev_ebitda_3yavgltm",
        "ev_ebitda_est_fy1",
    ],
    "earnings_growth": [
        "eps_norm_est_avg_ntm",
        "eps_adj_1fy",
        "eps_adj_fy",
        "eps_adj_ltm",
        "eps_norm_est_avg_fy1e",
    ],
}

earnings_monitor = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks_monitored": len(all_stocks_enhanced),
    "growth_metrics": {},
    "margin_metrics": {},
    "quality_indicators": {},
}

# Analyze growth metrics
for category, metrics in earnings_metrics.items():
    available = [m for m in metrics if m in all_stocks_enhanced.columns]

    if available:
        category_stats = {}
        print(f'\n  {category.replace("_", " ").title()}:')

        for metric in available:
            data = all_stocks_enhanced[metric].dropna()
            if len(data) > 0:
                category_stats[metric] = {
                    "count": int(len(data)),
                    "mean": float(data.mean()),
                    "median": float(data.median()),
                    "positive_growth_pct": float((data > 0).sum() / len(data) * 100),
                    "negative_growth_pct": float((data < 0).sum() / len(data) * 100),
                }

                print(
                    f"    {metric:25s}: mean={data.mean():.2f}%, median={data.median():.2f}%, +growth={((data > 0).sum() / len(data) * 100):.1f}%"
                )

        earnings_monitor["growth_metrics"][category] = category_stats

# ============================================================================
# 2. Margin Analysis
# ============================================================================
print("\n📊 Margin Analysis...")

margin_metrics = [
    "gross_margin_pct",
    "gross_margin_pct_previous_year",
    "ebitda_margin",
    "gross_profit_margin_pct_fy",
    "gross_profit_margin_pct_ltm",
    "net_income_margin_pct_fy",
    "net_income_margin_pct_ltm",
]
available_margins = [m for m in margin_metrics if m in all_stocks_enhanced.columns]

if available_margins:
    margin_stats = {}

    for metric in available_margins:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            margin_stats[metric] = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "positive_pct": float((data > 0).sum() / len(data) * 100),
            }

            print(
                f"  {metric:20s}: mean={data.mean():.2f}%, median={data.median():.2f}%, positive={((data > 0).sum() / len(data) * 100):.1f}%"
            )

    earnings_monitor["margin_metrics"] = margin_stats

    # Sector margin comparison
    if "sector" in all_stocks_enhanced.columns and len(available_margins) > 0:
        top_margin = available_margins[0]
        sector_margins = {}

        for sector in all_stocks_enhanced["sector"].dropna().unique():
            sector_data = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector][
                top_margin
            ].dropna()
            if len(sector_data) >= 5:
                sector_margins[str(sector)] = {
                    "mean": float(sector_data.mean()),
                    "median": float(sector_data.median()),
                    "count": int(len(sector_data)),
                }

        earnings_monitor["margin_by_sector"] = sector_margins

        print(f'\n  Top 5 Sectors by {top_margin.replace("_", " ").title()}:')
        sorted_sectors = sorted(sector_margins.items(), key=lambda x: x[1]["mean"], reverse=True)[
            :5
        ]
        for sector, stats in sorted_sectors:
            print(f'    {sector[:30]:30s}: {stats["mean"]:.2f}%')

# ============================================================================
# 3. Earnings Quality Indicators
# ============================================================================
print("\n🎯 Earnings Quality Indicators...")

quality_metrics = ["roe", "roa", "roic", "asset_turnover", "fcf_to_net_income"]
available_quality = [m for m in quality_metrics if m in all_stocks_enhanced.columns]

if available_quality:
    quality_stats = {}

    for metric in available_quality:
        data = all_stocks_enhanced[metric].dropna()
        if len(data) > 0:
            quality_stats[metric] = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
            }

            print(f"  {metric:25s}: mean={data.mean():.3f}, median={data.median():.3f}")

    earnings_monitor["quality_indicators"] = quality_stats

# ============================================================================
# 4. Regional Earnings Trends
# ============================================================================
if "region" in all_stocks_enhanced.columns:
    print("\n🌍 Regional Earnings Trends...")

    # Use first available revenue growth metric
    rev_growth_cols = [
        c
        for c in ["revenue_growth_1y", "revenue_growth_3y", "revenue_cagr_3y"]
        if c in all_stocks_enhanced.columns
    ]

    if rev_growth_cols:
        rev_col = rev_growth_cols[0]
        regional_earnings = {}

        for region in all_stocks_enhanced["region"].dropna().unique():
            region_data = all_stocks_enhanced[all_stocks_enhanced["region"] == region][
                rev_col
            ].dropna()
            if len(region_data) >= 5:
                regional_earnings[str(region)] = {
                    "count": int(len(region_data)),
                    "mean_growth": float(region_data.mean()),
                    "median_growth": float(region_data.median()),
                    "positive_growth_pct": float((region_data > 0).sum() / len(region_data) * 100),
                }

                print(
                    f"  {region:20s}: mean={region_data.mean():.2f}%, positive={((region_data > 0).sum() / len(region_data) * 100):.1f}%"
                )

        earnings_monitor["by_region"] = regional_earnings

# ============================================================================
# 5. Export Earnings Monitor JSON
# ============================================================================
print("\n💾 Exporting Earnings Monitor Dashboard...")

earnings_monitor_path = financial_metrics_dir / "earnings_monitor.json"
with open(earnings_monitor_path, "w") as f:
    json.dump(make_serializable(earnings_monitor), f, indent=2, default=str)
print(f"  ✓ Saved: {earnings_monitor_path}")

print("\n✓ Earnings Monitoring complete")


# ## Cell 10.7: Analyst Rating & Recommendations Analytics
#
# Analyze analyst ratings, recommendations, and price target consensus across sectors, regions, and market segments.
#
# **Key Objectives:**
# 1. Aggregate analyst recommendation distribution (Buy/Hold/Sell)
# 2. Analyze rating changes and momentum
# 3. Compare price target consensus vs current prices
# 4. Segment analysis by size_class, style_class, sector, industry
#
# **Outputs:**
# - **JSON Report:** analyst_recommendations.json
#

# In[13]:


# ============================================================================
# Cell 10.7: Analyst Rating & Recommendations Analytics
# ============================================================================

print("=" * 80)
print("ANALYST RATING & RECOMMENDATIONS ANALYTICS")
print("=" * 80)

# ============================================================================
# 1. Identify Analyst Rating Columns
# ============================================================================
print("\n🔍 Identifying Analyst Rating Columns...")

# Common analyst rating column patterns
rating_col_patterns = {
    "analyst_rating": ["analyst_rating", "rating", "recommendation", "analyst_recommendation"],
    "buy_ratings": ["buy_ratings", "num_buy", "strong_buy", "buy_count"],
    "hold_ratings": ["hold_ratings", "num_hold", "hold_count"],
    "sell_ratings": ["sell_ratings", "num_sell", "strong_sell", "sell_count"],
    "price_target": [
        "price_target",
        "price_target",
        "analyst_target",
        "price_target_mean",
        "price_target_ytd_ago",
    ],
    "target_high": ["price_target_high", "target_high", "high_target"],
    "target_low": ["price_target_low", "target_low", "low_target"],
    "num_analysts": ["num_analysts", "analyst_count", "coverage_count", "analysts_covering"],
}

# Find available columns
available_rating_cols = {}
for category, patterns in rating_col_patterns.items():
    for pattern in patterns:
        matching_cols = [c for c in all_stocks_enhanced.columns if pattern.lower() in c.lower()]
        if matching_cols:
            available_rating_cols[category] = matching_cols[0]
            break

print(f"  Found {len(available_rating_cols)} analyst rating column categories:")
for cat, col in available_rating_cols.items():
    print(f"    {cat:20s}: {col}")

# ============================================================================
# 2. Analyst Recommendations Summary
# ============================================================================
analyst_recommendations = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks_analyzed": len(all_stocks_enhanced),
    "available_columns": available_rating_cols,
    "rating_distribution": {},
    "by_sector": {},
    "by_region": {},
    "by_size_class": {},
    "by_style_class": {},
}

# Price target analysis
if "price_target" in available_rating_cols:
    target_col = available_rating_cols["price_target"]
    print(f"\n📊 Price Target Analysis ({target_col})...")

    target_data = pd.to_numeric(all_stocks_enhanced[target_col], errors="coerce").dropna()
    if len(target_data) > 0:
        analyst_recommendations["price_target_stats"] = {
            "count": int(len(target_data)),
            "mean": float(target_data.mean()),
            "median": float(target_data.median()),
            "std": float(target_data.std()),
            "min": float(target_data.min()),
            "max": float(target_data.max()),
        }
        print(f"  Stocks with price targets: {len(target_data):,}")
        print(f"  Mean target:               ${target_data.mean():.2f}")
        print(f"  Median target:             ${target_data.median():.2f}")

# Analyst coverage analysis
if "num_analysts" in available_rating_cols:
    coverage_col = available_rating_cols["num_analysts"]
    print(f"\n👥 Analyst Coverage Analysis ({coverage_col})...")

    coverage_data = pd.to_numeric(all_stocks_enhanced[coverage_col], errors="coerce").dropna()
    if len(coverage_data) > 0:
        analyst_recommendations["coverage_stats"] = {
            "count": int(len(coverage_data)),
            "mean_analysts": float(coverage_data.mean()),
            "median_analysts": float(coverage_data.median()),
            "max_analysts": int(coverage_data.max()),
            "uncovered_pct": float((coverage_data == 0).sum() / len(coverage_data) * 100),
        }
        print(f"  Stocks with coverage data: {len(coverage_data):,}")
        print(f"  Mean analysts per stock:   {coverage_data.mean():.1f}")
        print(f"  Max analysts:              {int(coverage_data.max())}")

        # Coverage distribution
        coverage_bins = [0, 1, 5, 10, 20, float("inf")]
        coverage_labels = ["0", "1-5", "6-10", "11-20", "20+"]
        coverage_dist = pd.cut(
            coverage_data, bins=coverage_bins, labels=coverage_labels
        ).value_counts()
        analyst_recommendations["coverage_distribution"] = {
            str(k): int(v) for k, v in coverage_dist.items()
        }

# ============================================================================
# 3. Sector-Level Analyst Analysis
# ============================================================================
if "sector" in all_stocks_enhanced.columns and "price_target" in available_rating_cols:
    print("\n🏢 Sector-Level Analyst Analysis...")
    target_col = available_rating_cols["price_target"]

    sector_analyst_stats = {}
    for sector in all_stocks_enhanced["sector"].dropna().unique():
        sector_df = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector]
        target_data = pd.to_numeric(sector_df[target_col], errors="coerce").dropna()

        if len(target_data) >= 5:
            # Calculate target vs price if available
            upside_data = None
            if "target_vs_price" in sector_df.columns:
                upside_data = sector_df["target_vs_price"].dropna()

            sector_analyst_stats[str(sector)] = {
                "count": int(len(target_data)),
                "mean_target": float(target_data.mean()),
                "median_target": float(target_data.median()),
            }

            if upside_data is not None and len(upside_data) > 0:
                sector_analyst_stats[str(sector)]["mean_upside"] = float(upside_data.mean())
                sector_analyst_stats[str(sector)]["positive_upside_pct"] = float(
                    (upside_data > 0).sum() / len(upside_data) * 100
                )

    analyst_recommendations["by_sector"] = sector_analyst_stats

    # Print top sectors by upside
    sectors_with_upside = {
        k: v.get("mean_upside", 0) for k, v in sector_analyst_stats.items() if "mean_upside" in v
    }
    if sectors_with_upside:
        print("\n  Top 5 Sectors by Mean Analyst Upside:")
        top_sectors = sorted(sectors_with_upside.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, upside in top_sectors:
            print(f"    {sector[:30]:30s}: {upside:+.2f}%")

# ============================================================================
# 4. Size Class & Style Class Analysis
# ============================================================================
print("\n📏 Size Class & Style Class Analysis...")

# Size class analysis
if "size_class" in all_stocks_enhanced.columns and "target_vs_price" in all_stocks_enhanced.columns:
    size_class_stats = {}
    for size_class in all_stocks_enhanced["size_class"].dropna().unique():
        size_df = all_stocks_enhanced[all_stocks_enhanced["size_class"] == size_class]
        upside_data = size_df["target_vs_price"].dropna()

        if len(upside_data) >= 5:
            size_class_stats[str(size_class)] = {
                "count": int(len(upside_data)),
                "mean_upside": float(upside_data.mean()),
                "median_upside": float(upside_data.median()),
                "positive_pct": float((upside_data > 0).sum() / len(upside_data) * 100),
            }
            print(
                f"  {size_class:15s}: n={len(upside_data):4d}, mean={upside_data.mean():+.2f}%, positive={((upside_data > 0).sum() / len(upside_data) * 100):.1f}%"
            )

    analyst_recommendations["by_size_class"] = size_class_stats

# Style class analysis
if (
    "style_class" in all_stocks_enhanced.columns
    and "target_vs_price" in all_stocks_enhanced.columns
):
    style_class_stats = {}
    for style_class in all_stocks_enhanced["style_class"].dropna().unique():
        style_df = all_stocks_enhanced[all_stocks_enhanced["style_class"] == style_class]
        upside_data = style_df["target_vs_price"].dropna()

        if len(upside_data) >= 5:
            style_class_stats[str(style_class)] = {
                "count": int(len(upside_data)),
                "mean_upside": float(upside_data.mean()),
                "median_upside": float(upside_data.median()),
                "positive_pct": float((upside_data > 0).sum() / len(upside_data) * 100),
            }
            print(
                f"  {style_class:15s}: n={len(upside_data):4d}, mean={upside_data.mean():+.2f}%, positive={((upside_data > 0).sum() / len(upside_data) * 100):.1f}%"
            )

    analyst_recommendations["by_style_class"] = style_class_stats

# ============================================================================
# 5. Export Analyst Recommendations JSON
# ============================================================================
print("\n💾 Exporting Analyst Recommendations...")

analyst_recommendations_path = financial_metrics_dir / "analyst_recommendations.json"
with open(analyst_recommendations_path, "w") as f:
    json.dump(make_serializable(analyst_recommendations), f, indent=2, default=str)
print(f"  ✓ Saved: {analyst_recommendations_path}")

print("\n✓ Analyst Rating & Recommendations Analytics complete")


# ## Cell 10.8: Estimated vs. Actual vs. Adjusted Earnings Analytics
#
# Analyze earnings estimates, actual reported earnings, and adjusted earnings metrics across sectors, regions, and market segments.
#
# **Key Objectives:**
# 1. Compare estimated vs actual earnings (earnings surprise analysis)
# 2. Analyze adjusted vs GAAP earnings differences
# 3. Track earnings revision trends
# 4. Segment analysis by sector, region, size_class, style_class, industry, trading country, exchange
#
# **Outputs:**
# - **JSON Report:** earnings_estimates_analysis.json
#

# In[14]:


# ============================================================================
# Cell 10.8: Estimated vs. Actual vs. Adjusted Earnings Analytics
# ============================================================================

print("=" * 80)
print("ESTIMATED VS. ACTUAL VS. ADJUSTED EARNINGS ANALYTICS")
print("=" * 80)

# ============================================================================
# 1. Identify Earnings-Related Columns
# ============================================================================
print("\n🔍 Identifying Earnings-Related Columns...")

# Common earnings column patterns
earnings_col_patterns = {
    "eps_actual": ["eps", "eps_actual", "earnings_per_share", "basic_eps", "diluted_eps"],
    "eps_estimate": ["eps_estimate", "eps_est", "estimated_eps", "consensus_eps", "eps_mean_est"],
    "eps_adjusted": ["eps_adjusted", "adjusted_eps", "non_gaap_eps", "core_eps"],
    "earnings_surprise": ["earnings_surprise", "eps_surprise", "surprise_pct"],
    "revenue_actual": ["revenue", "total_revenue", "sales", "net_sales"],
    "revenue_estimate": ["revenue_estimate", "revenue_est", "sales_estimate"],
    "net_income": ["net_income", "net_earnings", "profit"],
    "ebitda": ["ebitda", "operating_ebitda"],
    "ebit": ["ebit", "operating_income", "operating_profit"],
}

# Find available columns
available_earnings_cols = {}
for category, patterns in earnings_col_patterns.items():
    for pattern in patterns:
        matching_cols = [c for c in all_stocks_enhanced.columns if pattern.lower() in c.lower()]
        if matching_cols:
            available_earnings_cols[category] = matching_cols[0]
            break

print(f"  Found {len(available_earnings_cols)} earnings column categories:")
for cat, col in available_earnings_cols.items():
    print(f"    {cat:20s}: {col}")

# ============================================================================
# 2. Initialize Earnings Estimates Analysis
# ============================================================================
earnings_estimates_analysis = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks_analyzed": len(all_stocks_enhanced),
    "available_columns": available_earnings_cols,
    "eps_analysis": {},
    "earnings_surprise": {},
    "by_sector": {},
    "by_region": {},
    "by_size_class": {},
    "by_style_class": {},
    "by_industry": {},
    "by_trading_country": {},
    "by_exchange": {},
}

# ============================================================================
# 3. EPS Analysis (Actual vs Estimated vs Adjusted)
# ============================================================================
print("\n📊 EPS Analysis...")

eps_metrics = {}

# Actual EPS
if "eps_actual" in available_earnings_cols:
    eps_col = available_earnings_cols["eps_actual"]
    eps_data = pd.to_numeric(all_stocks_enhanced[eps_col], errors="coerce").dropna()
    if len(eps_data) > 0:
        eps_metrics["actual"] = {
            "column": eps_col,
            "count": int(len(eps_data)),
            "mean": float(eps_data.mean()),
            "median": float(eps_data.median()),
            "std": float(eps_data.std()),
            "positive_pct": float((eps_data > 0).sum() / len(eps_data) * 100),
            "negative_pct": float((eps_data < 0).sum() / len(eps_data) * 100),
        }
        print(f"  Actual EPS ({eps_col}):")
        print(f"    Valid values: {len(eps_data):,}")
        print(f"    Mean: ${eps_data.mean():.2f}, Median: ${eps_data.median():.2f}")
        print(f"    Profitable: {((eps_data > 0).sum() / len(eps_data) * 100):.1f}%")

# Estimated EPS
if "eps_estimate" in available_earnings_cols:
    est_col = available_earnings_cols["eps_estimate"]
    est_data = pd.to_numeric(all_stocks_enhanced[est_col], errors="coerce").dropna()
    if len(est_data) > 0:
        eps_metrics["estimated"] = {
            "column": est_col,
            "count": int(len(est_data)),
            "mean": float(est_data.mean()),
            "median": float(est_data.median()),
            "std": float(est_data.std()),
        }
        print(f"  Estimated EPS ({est_col}):")
        print(f"    Valid values: {len(est_data):,}")
        print(f"    Mean: ${est_data.mean():.2f}, Median: ${est_data.median():.2f}")

# Adjusted EPS
if "eps_adjusted" in available_earnings_cols:
    adj_col = available_earnings_cols["eps_adjusted"]
    adj_data = pd.to_numeric(all_stocks_enhanced[adj_col], errors="coerce").dropna()
    if len(adj_data) > 0:
        eps_metrics["adjusted"] = {
            "column": adj_col,
            "count": int(len(adj_data)),
            "mean": float(adj_data.mean()),
            "median": float(adj_data.median()),
            "std": float(adj_data.std()),
        }
        print(f"  Adjusted EPS ({adj_col}):")
        print(f"    Valid values: {len(adj_data):,}")
        print(f"    Mean: ${adj_data.mean():.2f}, Median: ${adj_data.median():.2f}")

earnings_estimates_analysis["eps_analysis"] = eps_metrics

# ============================================================================
# 4. Earnings Surprise Analysis
# ============================================================================
print("\n📈 Earnings Surprise Analysis...")

# Calculate earnings surprise if we have both actual and estimated
if "eps_actual" in available_earnings_cols and "eps_estimate" in available_earnings_cols:
    actual_col = available_earnings_cols["eps_actual"]
    est_col = available_earnings_cols["eps_estimate"]

    # Get aligned data
    mask = all_stocks_enhanced[actual_col].notna() & all_stocks_enhanced[est_col].notna()
    actual = pd.to_numeric(all_stocks_enhanced.loc[mask, actual_col], errors="coerce")
    estimated = pd.to_numeric(all_stocks_enhanced.loc[mask, est_col], errors="coerce")

    # Calculate surprise percentage
    with np.errstate(divide="ignore", invalid="ignore"):
        surprise_pct = ((actual - estimated) / estimated.abs()) * 100
    surprise_pct = surprise_pct.replace([np.inf, -np.inf], np.nan).dropna()

    if len(surprise_pct) > 0:
        earnings_estimates_analysis["earnings_surprise"] = {
            "count": int(len(surprise_pct)),
            "mean_surprise_pct": float(surprise_pct.mean()),
            "median_surprise_pct": float(surprise_pct.median()),
            "beat_pct": float((surprise_pct > 0).sum() / len(surprise_pct) * 100),
            "miss_pct": float((surprise_pct < 0).sum() / len(surprise_pct) * 100),
            "large_beat_pct": float((surprise_pct > 10).sum() / len(surprise_pct) * 100),
            "large_miss_pct": float((surprise_pct < -10).sum() / len(surprise_pct) * 100),
        }
        print(f"  Stocks with surprise data: {len(surprise_pct):,}")
        print(f"  Mean surprise:             {surprise_pct.mean():+.2f}%")
        print(
            f"  Beat estimates:            {((surprise_pct > 0).sum() / len(surprise_pct) * 100):.1f}%"
        )
        print(
            f"  Miss estimates:            {((surprise_pct < 0).sum() / len(surprise_pct) * 100):.1f}%"
        )

        # Store surprise in DataFrame for segment analysis
        all_stocks_enhanced.loc[mask, "calculated_eps_surprise"] = surprise_pct
elif "earnings_surprise" in available_earnings_cols:
    surprise_col = available_earnings_cols["earnings_surprise"]
    surprise_data = pd.to_numeric(all_stocks_enhanced[surprise_col], errors="coerce").dropna()
    if len(surprise_data) > 0:
        earnings_estimates_analysis["earnings_surprise"] = {
            "count": int(len(surprise_data)),
            "mean_surprise_pct": float(surprise_data.mean()),
            "median_surprise_pct": float(surprise_data.median()),
            "beat_pct": float((surprise_data > 0).sum() / len(surprise_data) * 100),
            "miss_pct": float((surprise_data < 0).sum() / len(surprise_data) * 100),
        }
        print(f"  Using existing surprise column: {surprise_col}")
        print(f"  Mean surprise: {surprise_data.mean():+.2f}%")


# ============================================================================
# 5. Segment Analysis Functions
# ============================================================================
def analyze_earnings_by_segment(df, segment_col, eps_col):
    """Analyze earnings metrics by a segment column."""
    segment_stats = {}

    for segment in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment]
        eps_data = pd.to_numeric(segment_df[eps_col], errors="coerce").dropna()

        if len(eps_data) >= 5:
            segment_stats[str(segment)] = {
                "count": int(len(eps_data)),
                "mean_eps": float(eps_data.mean()),
                "median_eps": float(eps_data.median()),
                "positive_pct": float((eps_data > 0).sum() / len(eps_data) * 100),
            }

            # Add surprise if available
            if "calculated_eps_surprise" in segment_df.columns:
                surprise = segment_df["calculated_eps_surprise"].dropna()
                if len(surprise) >= 3:
                    segment_stats[str(segment)]["mean_surprise"] = float(surprise.mean())
                    segment_stats[str(segment)]["beat_pct"] = float(
                        (surprise > 0).sum() / len(surprise) * 100
                    )

    return segment_stats


# Get primary EPS column
primary_eps_col = None
for key in ["eps_actual", "eps_adjusted", "net_income"]:
    if key in available_earnings_cols:
        primary_eps_col = available_earnings_cols[key]
        break

if primary_eps_col:
    # ============================================================================
    # 6. By Sector
    # ============================================================================
    if "sector" in all_stocks_enhanced.columns:
        print("\n🏢 Earnings by Sector...")
        sector_stats = analyze_earnings_by_segment(all_stocks_enhanced, "sector", primary_eps_col)
        earnings_estimates_analysis["by_sector"] = sector_stats

        # Print top 5 sectors by profitability
        profitable_sectors = {k: v["positive_pct"] for k, v in sector_stats.items()}
        top_sectors = sorted(profitable_sectors.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, pct in top_sectors:
            print(f"  {sector[:30]:30s}: {pct:.1f}% profitable")

    # ============================================================================
    # 7. By Region
    # ============================================================================
    if "region" in all_stocks_enhanced.columns:
        print("\n🌍 Earnings by Region...")
        region_stats = analyze_earnings_by_segment(all_stocks_enhanced, "region", primary_eps_col)
        earnings_estimates_analysis["by_region"] = region_stats

        for region, stats in region_stats.items():
            print(f'  {region:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%')

    # ============================================================================
    # 8. By Size Class
    # ============================================================================
    if "size_class" in all_stocks_enhanced.columns:
        print("\n📏 Earnings by Size Class...")
        size_stats = analyze_earnings_by_segment(all_stocks_enhanced, "size_class", primary_eps_col)
        earnings_estimates_analysis["by_size_class"] = size_stats

        for size_class, stats in size_stats.items():
            print(
                f'  {size_class:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%'
            )

    # ============================================================================
    # 9. By Style Class
    # ============================================================================
    if "style_class" in all_stocks_enhanced.columns:
        print("\n🎨 Earnings by Style Class...")
        style_stats = analyze_earnings_by_segment(
            all_stocks_enhanced, "style_class", primary_eps_col
        )
        earnings_estimates_analysis["by_style_class"] = style_stats

        for style_class, stats in style_stats.items():
            print(
                f'  {style_class:15s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%'
            )

    # ============================================================================
    # 10. By Industry
    # ============================================================================
    if "industry" in all_stocks_enhanced.columns:
        print("\n🏭 Earnings by Industry (Top 10)...")
        industry_stats = analyze_earnings_by_segment(
            all_stocks_enhanced, "industry", primary_eps_col
        )
        earnings_estimates_analysis["by_industry"] = industry_stats

        # Show top 10 industries by count
        top_industries = sorted(industry_stats.items(), key=lambda x: x[1]["count"], reverse=True)[
            :10
        ]
        for industry, stats in top_industries:
            print(
                f'  {industry[:35]:35s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%'
            )

    # ============================================================================
    # 11. By Trading Country
    # ============================================================================
    trading_country_cols = ["trading_country", "country", "domicile"]
    trading_country_col = None
    for col in trading_country_cols:
        if col in all_stocks_enhanced.columns:
            trading_country_col = col
            break

    if trading_country_col:
        print(f"\n🌐 Earnings by Trading Country ({trading_country_col})...")
        country_stats = analyze_earnings_by_segment(
            all_stocks_enhanced, trading_country_col, primary_eps_col
        )
        earnings_estimates_analysis["by_trading_country"] = country_stats

        # Show top 10 countries by count
        top_countries = sorted(country_stats.items(), key=lambda x: x[1]["count"], reverse=True)[
            :10
        ]
        for country, stats in top_countries:
            print(
                f'  {country[:25]:25s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%'
            )

    # ============================================================================
    # 12. By Industry
    # ============================================================================
    industry_cols = ["industry", "stock_exchange", "primary_exchange", "listing_exchange"]
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    if industry_col:
        print(f"\n📈 Earnings by Industry ({industry_col})...")
        industry_stats = analyze_earnings_by_segment(
            all_stocks_enhanced, industry_col, primary_eps_col
        )
        earnings_estimates_analysis["by_exchange"] = industry_stats

        # Show top 10 exchanges by count
        top_exchanges = sorted(industry_stats.items(), key=lambda x: x[1]["count"], reverse=True)[
            :10
        ]
        for exchange, stats in top_exchanges:
            print(
                f'  {exchange[:25]:25s}: n={stats["count"]:4d}, profitable={stats["positive_pct"]:.1f}%'
            )

# ============================================================================
# 13. Export Earnings Estimates Analysis JSON
# ============================================================================
print("\n💾 Exporting Earnings Estimates Analysis...")

earnings_estimates_path = financial_metrics_dir / "earnings_estimates_analysis.json"
with open(earnings_estimates_path, "w") as f:
    json.dump(make_serializable(earnings_estimates_analysis), f, indent=2, default=str)
print(f"  ✓ Saved: {earnings_estimates_path}")

print("\n✓ Estimated vs. Actual vs. Adjusted Earnings Analytics complete")

# ============================================================================
# Interactive Dashboard: Earnings Calendar
# ============================================================================
from finance_ml.dashboards.earnings_widgets import display_earnings_dashboard

print("\nINTERACTIVE EARNINGS CALENDAR DASHBOARD (Top 100)")
print("=" * 80)
# Using all_stocks_enhanced if available, else fallback to all_stocks_features
if "all_stocks_enhanced" in locals():
    display_df = all_stocks_enhanced
elif "all_stocks_features" in locals():
    display_df = all_stocks_preprocessed
else:
    display_df = all_stocks_preprocessed

display_earnings_dashboard(display_df, mode="earnings")


# ## Cell 10.9: Dividend Analytics
#
# Comprehensive dividend analysis across sectors, regions, and market segments including yield, payout ratios, dividend growth, and income stock screening.
#
# **Key Objectives:**
# 1. Analyze dividend yield distribution and trends
# 2. Evaluate payout ratios and dividend sustainability
# 3. Track dividend growth and consistency
# 4. Segment analysis by size_class, style_class, sector, industry, trading country, exchange
#
# **Outputs:**
# - **JSON Report:** dividend_analytics.json
#

# In[15]:


# ============================================================================
# Cell 10.9: Dividend Analytics
# ============================================================================

print("=" * 80)
print("DIVIDEND ANALYTICS")
print("=" * 80)

# ============================================================================
# 1. Identify Dividend-Related Columns
# ============================================================================
print("\n🔍 Identifying Dividend-Related Columns...")

# Common dividend column patterns
dividend_col_patterns = {
    "dividend_yield": ["dividend_yield", "div_yield", "yield", "annual_dividend_yield"],
    "dividend_per_share": ["dividend_per_share", "dps", "annual_dividend", "dividends_per_share"],
    "payout_ratio": ["payout_ratio", "dividend_payout", "payout_pct"],
    "dividend_growth": ["dividend_growth", "div_growth", "dividend_growth_rate", "dividend_cagr"],
    "ex_dividend_date": ["ex_dividend_date", "ex_div_date", "next_ex_date"],
    "dividend_frequency": ["dividend_frequency", "div_frequency", "payment_frequency"],
    "years_of_dividend": ["years_of_dividend", "consecutive_dividends", "dividend_streak"],
}

# Find available columns
available_dividend_cols = {}
for category, patterns in dividend_col_patterns.items():
    for pattern in patterns:
        matching_cols = [c for c in all_stocks_enhanced.columns if pattern.lower() in c.lower()]
        if matching_cols:
            available_dividend_cols[category] = matching_cols[0]
            break

print(f"  Found {len(available_dividend_cols)} dividend column categories:")
for cat, col in available_dividend_cols.items():
    print(f"    {cat:25s}: {col}")

# ============================================================================
# 2. Initialize Dividend Analytics
# ============================================================================
dividend_analytics = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks_analyzed": len(all_stocks_enhanced),
    "available_columns": available_dividend_cols,
    "yield_analysis": {},
    "payout_analysis": {},
    "dividend_payers": {},
    "by_sector": {},
    "by_region": {},
    "by_size_class": {},
    "by_style_class": {},
    "by_industry": {},
    "by_trading_country": {},
    "by_exchange": {},
}

# ============================================================================
# 3. Dividend Yield Analysis
# ============================================================================
print("\n💰 Dividend Yield Analysis...")

if "dividend_yield" in available_dividend_cols:
    yield_col = available_dividend_cols["dividend_yield"]
    yield_data = pd.to_numeric(all_stocks_enhanced[yield_col], errors="coerce").dropna()

    if len(yield_data) > 0:
        # Filter for positive yields (dividend payers)
        positive_yields = yield_data[yield_data > 0]

        dividend_analytics["yield_analysis"] = {
            "column": yield_col,
            "total_stocks": int(len(yield_data)),
            "dividend_payers": int(len(positive_yields)),
            "dividend_payer_pct": float(len(positive_yields) / len(yield_data) * 100),
            "mean_yield": float(positive_yields.mean()) if len(positive_yields) > 0 else 0,
            "median_yield": float(positive_yields.median()) if len(positive_yields) > 0 else 0,
            "std_yield": float(positive_yields.std()) if len(positive_yields) > 0 else 0,
            "min_yield": float(positive_yields.min()) if len(positive_yields) > 0 else 0,
            "max_yield": float(positive_yields.max()) if len(positive_yields) > 0 else 0,
            "q25": float(positive_yields.quantile(0.25)) if len(positive_yields) > 0 else 0,
            "q75": float(positive_yields.quantile(0.75)) if len(positive_yields) > 0 else 0,
        }

        print(f"  Total stocks analyzed:    {len(yield_data):,}")
        print(
            f"  Dividend payers:          {len(positive_yields):,} ({len(positive_yields) / len(yield_data) * 100:.1f}%)"
        )
        if len(positive_yields) > 0:
            print(f"  Mean yield:               {positive_yields.mean():.2f}%")
            print(f"  Median yield:             {positive_yields.median():.2f}%")
            print(
                f"  Yield range:              {positive_yields.min():.2f}% - {positive_yields.max():.2f}%"
            )

        # Yield distribution buckets
        yield_bins = [0, 1, 2, 3, 4, 5, 7, 10, float("inf")]
        yield_labels = ["0-1%", "1-2%", "2-3%", "3-4%", "4-5%", "5-7%", "7-10%", "10%+"]
        yield_dist = pd.cut(positive_yields, bins=yield_bins, labels=yield_labels).value_counts()
        dividend_analytics["yield_distribution"] = {str(k): int(v) for k, v in yield_dist.items()}

        print("\n  Yield Distribution:")
        for bucket, count in sorted(
            yield_dist.items(),
            key=lambda x: yield_labels.index(x[0]) if x[0] in yield_labels else 99,
        ):
            pct = count / len(positive_yields) * 100 if len(positive_yields) > 0 else 0
            print(f"    {bucket:10s}: {count:5,} ({pct:5.1f}%)")

# ============================================================================
# 4. Payout Ratio Analysis
# ============================================================================
print("\n📊 Payout Ratio Analysis...")

if "payout_ratio" in available_dividend_cols:
    payout_col = available_dividend_cols["payout_ratio"]
    payout_data = pd.to_numeric(all_stocks_enhanced[payout_col], errors="coerce").dropna()

    # Filter reasonable payout ratios (0-200%)
    valid_payout = payout_data[(payout_data >= 0) & (payout_data <= 200)]

    if len(valid_payout) > 0:
        dividend_analytics["payout_analysis"] = {
            "column": payout_col,
            "count": int(len(valid_payout)),
            "mean_payout": float(valid_payout.mean()),
            "median_payout": float(valid_payout.median()),
            "sustainable_pct": float((valid_payout <= 75).sum() / len(valid_payout) * 100),
            "high_payout_pct": float((valid_payout > 100).sum() / len(valid_payout) * 100),
        }

        print(f"  Stocks with payout data:  {len(valid_payout):,}")
        print(f"  Mean payout ratio:        {valid_payout.mean():.1f}%")
        print(f"  Median payout ratio:      {valid_payout.median():.1f}%")
        print(
            f"  Sustainable (<=75%):      {((valid_payout <= 75).sum() / len(valid_payout) * 100):.1f}%"
        )
        print(
            f"  High payout (>100%):      {((valid_payout > 100).sum() / len(valid_payout) * 100):.1f}%"
        )

# ============================================================================
# 5. Dividend Growth Analysis
# ============================================================================
print("\n📈 Dividend Growth Analysis...")

if "dividend_growth" in available_dividend_cols:
    growth_col = available_dividend_cols["dividend_growth"]
    growth_data = pd.to_numeric(all_stocks_enhanced[growth_col], errors="coerce").dropna()

    if len(growth_data) > 0:
        dividend_analytics["dividend_growth"] = {
            "column": growth_col,
            "count": int(len(growth_data)),
            "mean_growth": float(growth_data.mean()),
            "median_growth": float(growth_data.median()),
            "positive_growth_pct": float((growth_data > 0).sum() / len(growth_data) * 100),
            "growers_pct": float((growth_data > 5).sum() / len(growth_data) * 100),
            "cutters_pct": float((growth_data < -5).sum() / len(growth_data) * 100),
        }

        print(f"  Stocks with growth data:  {len(growth_data):,}")
        print(f"  Mean dividend growth:     {growth_data.mean():+.2f}%")
        print(
            f"  Dividend growers (>5%):   {((growth_data > 5).sum() / len(growth_data) * 100):.1f}%"
        )
        print(
            f"  Dividend cutters (<-5%):  {((growth_data < -5).sum() / len(growth_data) * 100):.1f}%"
        )


# ============================================================================
# 6. Segment Analysis Function for Dividends
# ============================================================================
def analyze_dividends_by_segment(df, segment_col, yield_col):
    """Analyze dividend metrics by a segment column."""
    segment_stats = {}

    for segment in df[segment_col].dropna().unique():
        segment_df = df[df[segment_col] == segment]
        yield_data = pd.to_numeric(segment_df[yield_col], errors="coerce").dropna()
        positive_yields = yield_data[yield_data > 0]

        if len(yield_data) >= 5:
            segment_stats[str(segment)] = {
                "total_stocks": int(len(yield_data)),
                "dividend_payers": int(len(positive_yields)),
                "dividend_payer_pct": float(len(positive_yields) / len(yield_data) * 100),
            }

            if len(positive_yields) >= 3:
                segment_stats[str(segment)]["mean_yield"] = float(positive_yields.mean())
                segment_stats[str(segment)]["median_yield"] = float(positive_yields.median())

    return segment_stats


# Get primary dividend yield column
primary_yield_col = available_dividend_cols.get("dividend_yield")

if primary_yield_col:
    # ============================================================================
    # 7. By Sector
    # ============================================================================
    if "sector" in all_stocks_enhanced.columns:
        print("\n🏢 Dividends by Sector...")
        sector_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, "sector", primary_yield_col
        )
        dividend_analytics["by_sector"] = sector_stats

        # Print top 5 sectors by dividend payer percentage
        payer_sectors = {k: v["dividend_payer_pct"] for k, v in sector_stats.items()}
        top_sectors = sorted(payer_sectors.items(), key=lambda x: x[1], reverse=True)[:5]
        for sector, pct in top_sectors:
            mean_yield = sector_stats[sector].get("mean_yield", 0)
            print(f"  {sector[:30]:30s}: {pct:.1f}% payers, avg yield={mean_yield:.2f}%")

    # ============================================================================
    # 8. By Region
    # ============================================================================
    if "region" in all_stocks_enhanced.columns:
        print("\n🌍 Dividends by Region...")
        region_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, "region", primary_yield_col
        )
        dividend_analytics["by_region"] = region_stats

        for region, stats in region_stats.items():
            mean_yield = stats.get("mean_yield", 0)
            print(
                f'  {region:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%'
            )

    # ============================================================================
    # 9. By Size Class
    # ============================================================================
    if "size_class" in all_stocks_enhanced.columns:
        print("\n📏 Dividends by Size Class...")
        size_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, "size_class", primary_yield_col
        )
        dividend_analytics["by_size_class"] = size_stats

        for size_class, stats in size_stats.items():
            mean_yield = stats.get("mean_yield", 0)
            print(
                f'  {size_class:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%'
            )

    # ============================================================================
    # 10. By Style Class
    # ============================================================================
    if "style_class" in all_stocks_enhanced.columns:
        print("\n🎨 Dividends by Style Class...")
        style_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, "style_class", primary_yield_col
        )
        dividend_analytics["by_style_class"] = style_stats

        for style_class, stats in style_stats.items():
            mean_yield = stats.get("mean_yield", 0)
            print(
                f'  {style_class:15s}: {stats["dividend_payer_pct"]:.1f}% payers, avg yield={mean_yield:.2f}%'
            )

    # ============================================================================
    # 11. By Industry
    # ============================================================================
    if "industry" in all_stocks_enhanced.columns:
        print("\n🏭 Dividends by Industry (Top 10 by Yield)...")
        industry_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, "industry", primary_yield_col
        )
        dividend_analytics["by_industry"] = industry_stats

        # Show top 10 industries by mean yield
        industries_with_yield = {
            k: v.get("mean_yield", 0)
            for k, v in industry_stats.items()
            if v.get("dividend_payers", 0) >= 5
        }
        top_industries = sorted(industries_with_yield.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        for industry, yield_val in top_industries:
            payer_pct = industry_stats[industry]["dividend_payer_pct"]
            print(f"  {industry[:35]:35s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%")

    # ============================================================================
    # 12. By Trading Country
    # ============================================================================
    trading_country_cols = ["trading_country", "country", "domicile"]
    trading_country_col = None
    for col in trading_country_cols:
        if col in all_stocks_enhanced.columns:
            trading_country_col = col
            break

    if trading_country_col:
        print(f"\n🌐 Dividends by Trading Country ({trading_country_col})...")
        country_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, trading_country_col, primary_yield_col
        )
        dividend_analytics["by_trading_country"] = country_stats

        # Show top 10 countries by mean yield
        countries_with_yield = {
            k: v.get("mean_yield", 0)
            for k, v in country_stats.items()
            if v.get("dividend_payers", 0) >= 5
        }
        top_countries = sorted(countries_with_yield.items(), key=lambda x: x[1], reverse=True)[:10]
        for country, yield_val in top_countries:
            payer_pct = country_stats[country]["dividend_payer_pct"]
            print(f"  {country[:25]:25s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%")

    # ============================================================================
    # 13. By Industry
    # ============================================================================
    industry_cols = ["industry", "stock_exchange", "primary_exchange", "listing_exchange"]
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    if industry_col:
        print(f"\n📈 Dividends by Industry ({industry_col})...")
        industry_stats = analyze_dividends_by_segment(
            all_stocks_enhanced, industry_col, primary_yield_col
        )
        dividend_analytics["by_exchange"] = industry_stats

        # Show top 10 exchanges by mean yield
        exchanges_with_yield = {
            k: v.get("mean_yield", 0)
            for k, v in industry_stats.items()
            if v.get("dividend_payers", 0) >= 5
        }
        top_exchanges = sorted(exchanges_with_yield.items(), key=lambda x: x[1], reverse=True)[:10]
        for exchange, yield_val in top_exchanges:
            payer_pct = industry_stats[exchange]["dividend_payer_pct"]
            print(f"  {exchange[:25]:25s}: yield={yield_val:.2f}%, payers={payer_pct:.1f}%")

# ============================================================================
# 14. High Yield Stock Screening
# ============================================================================
print("\n🔝 High Yield Stock Screening...")

if "dividend_yield" in available_dividend_cols:
    yield_col = available_dividend_cols["dividend_yield"]
    yield_data = pd.to_numeric(all_stocks_enhanced[yield_col], errors="coerce")

    # Define high yield threshold
    high_yield_threshold = 4.0
    high_yield_mask = yield_data >= high_yield_threshold
    high_yield_stocks = all_stocks_enhanced[high_yield_mask]

    dividend_analytics["high_yield_screening"] = {
        "threshold": high_yield_threshold,
        "count": int(len(high_yield_stocks)),
        "pct_of_total": float(len(high_yield_stocks) / len(all_stocks_enhanced) * 100),
    }

    print(
        f"  High yield stocks (>={high_yield_threshold}%): {len(high_yield_stocks):,} ({len(high_yield_stocks) / len(all_stocks_enhanced) * 100:.1f}%)"
    )

    # Show sector distribution of high yield stocks
    if "sector" in high_yield_stocks.columns and len(high_yield_stocks) > 0:
        hy_sector_dist = high_yield_stocks["sector"].value_counts().head(5)
        dividend_analytics["high_yield_by_sector"] = {
            str(k): int(v) for k, v in hy_sector_dist.items()
        }

        print("\n  High Yield by Sector (Top 5):")
        for sector, count in hy_sector_dist.items():
            print(f"    {sector[:30]:30s}: {count:5,}")

# ============================================================================
# 15. Export Dividend Analytics JSON
# ============================================================================
print("\n💾 Exporting Dividend Analytics...")

dividend_analytics_path = financial_metrics_dir / "dividend_analytics.json"
with open(dividend_analytics_path, "w") as f:
    json.dump(make_serializable(dividend_analytics), f, indent=2, default=str)
print(f"  ✓ Saved: {dividend_analytics_path}")

print("\n✓ Dividend Analytics complete")

# ============================================================================
# Interactive Dashboard: Dividend Analytics
# ============================================================================
print("\nINTERACTIVE DIVIDEND ANALYTICS DASHBOARD")
print("=" * 80)
display_earnings_dashboard(display_df, mode="dividends")


# ## Cell 10.10: Enhanced Interactive Visualizations
#
# Generate additional interactive HTML visualizations for statistical testing results, earnings analytics,
# and dividend analytics following Section 17 Style Guidelines from code_guidelines.md.
#
# **Key Objectives:**
# 1. Visualize hypothesis testing results with p-value heatmaps
# 2. Create earnings surprise distribution and sector comparison charts
# 3. Generate dividend yield analysis visualizations
# 4. Build analyst recommendation summary charts
# 5. Create benchmarking comparison visualizations
#
# **Outputs:**
# - **HTML Visualizations (6 files):** hypothesis_test_heatmap.html, earnings_surprise_analysis.html,
#   dividend_yield_distribution.html, analyst_recommendations_chart.html, sector_benchmarking.html,
#   financial_metrics_radar.html
#

# ## Cell 10.11: Advanced Benchmarking & Risk Analytics
#
# Enhanced visualizations for statistical benchmarking, risk analytics (VaR), and performance attribution analysis.
#
# **Key Objectives:**
# 1. Generate Value at Risk (VaR) and Expected Shortfall analytics by sector
# 2. Create statistical benchmarking comparison charts (ANOVA, Kruskal-Wallis)
# 3. Build performance attribution sunburst visualization
# 4. Generate cross-sectional risk heatmaps
#
# **Outputs:**
# - **HTML Visualizations (4 files):** var_risk_analytics.html, statistical_benchmarking.html, performance_attribution_sunburst.html, cross_sectional_risk_heatmap.html
#

# In[16]:


# ============================================================================
# Cell 10.10: Enhanced Interactive Visualizations
# ============================================================================

print("=" * 80)
print("ENHANCED INTERACTIVE VISUALIZATIONS")
print("=" * 80)

# Create visualizations output directory
viz_output_dir = OUTPUT_DIR / "eda" / "visualizations"
viz_output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Hypothesis Testing Results Heatmap
# ============================================================================
print("\n📊 Generating Hypothesis Testing Visualizations...")

# Load hypothesis test results if available
hypothesis_tests_file = financial_metrics_dir / "hypothesis_tests.json"
if hypothesis_tests_file.exists():
    with open(hypothesis_tests_file, "r") as f:
        hypothesis_data = json.load(f)

    # Extract sector test results for visualization
    if "sector_tests" in hypothesis_data:
        sector_tests = hypothesis_data["sector_tests"]

        # Build DataFrame for heatmap
        test_metrics = [k for k in sector_tests.keys() if k != "summary"]
        test_types = ["anova", "kruskal_wallis"]

        heatmap_data = []
        for metric in test_metrics:
            for test_type in test_types:
                if test_type in sector_tests.get(metric, {}):
                    test_result = sector_tests[metric][test_type]
                    p_value = test_result.get("p_value", 1.0)
                    significant = test_result.get("significant", "False") == "True"
                    heatmap_data.append(
                        {
                            "Metric": metric.replace("_", " ").title(),
                            "Test": test_type.replace("_", " ").title(),
                            "P-Value": p_value,
                            "Significant": "Yes" if significant else "No",
                            "-log10(p)": -np.log10(max(p_value, 1e-100)),
                        }
                    )

        if heatmap_data:
            heatmap_df = pd.DataFrame(heatmap_data)

            # Create pivot for heatmap
            pivot_df = heatmap_df.pivot(index="Metric", columns="Test", values="-log10(p)")

            fig_hypothesis = px.imshow(
                pivot_df,
                title="<b>Statistical Hypothesis Testing Results</b><br><sup>-log10(p-value) by Metric and Test Type (Higher = More Significant)</sup>",
                template=PLOTLY_TEMPLATE,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=".2f",
            )
            fig_hypothesis.update_layout(
                height=500,
                font=dict(family="Segoe UI, Roboto, Arial"),
                title_font_size=20,
                xaxis_title="Statistical Test",
                yaxis_title="Financial Metric",
            )
            fig_hypothesis.add_hline(
                y=-0.5,
                line_dash="dash",
                line_color="white",
                annotation_text="α=0.05 threshold: -log10(0.05)≈1.3",
            )
            fig_hypothesis.write_html(viz_output_dir / "hypothesis_test_heatmap.html")
            print(f"  ✓ Saved: hypothesis_test_heatmap.html")

            # Display inline
            fig_hypothesis.show()
else:
    print("  ⚠ hypothesis_tests.json not found, skipping visualization")

# ============================================================================
# 2. Earnings Surprise Analysis Visualization
# ============================================================================
print("\n📈 Generating Earnings Surprise Visualizations...")

earnings_file = financial_metrics_dir / "earnings_monitor.json"
if earnings_file.exists():
    with open(earnings_file, "r") as f:
        earnings_data = json.load(f)

    # Create earnings surprise visualization if data available
    if "earnings_surprise_analysis" in earnings_data:
        surprise_data = earnings_data["earnings_surprise_analysis"]

        # Build comparison data
        surprise_metrics = []
        for metric_name, metric_data in surprise_data.items():
            if isinstance(metric_data, dict) and "mean_surprise_pct" in metric_data:
                surprise_metrics.append(
                    {
                        "Metric": metric_name,
                        "Mean Surprise (%)": metric_data.get("mean_surprise_pct", 0),
                        "Median Surprise (%)": metric_data.get("median_surprise_pct", 0),
                        "Positive Surprises": metric_data.get("positive_surprises", 0),
                        "Negative Surprises": metric_data.get("negative_surprises", 0),
                        "Count": metric_data.get("count", 0),
                    }
                )

        if surprise_metrics:
            surprise_df = pd.DataFrame(surprise_metrics)

            # Create grouped bar chart for earnings surprise
            fig_earnings = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["Mean vs Median Surprise (%)", "Positive vs Negative Surprises"],
                specs=[[{"type": "bar"}, {"type": "bar"}]],
            )

            # Mean vs Median
            fig_earnings.add_trace(
                go.Bar(
                    name="Mean Surprise",
                    x=surprise_df["Metric"],
                    y=surprise_df["Mean Surprise (%)"],
                    marker_color=COLOR_PALETTE["primary"],
                    hovertemplate="%{x}<br>Mean: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig_earnings.add_trace(
                go.Bar(
                    name="Median Surprise",
                    x=surprise_df["Metric"],
                    y=surprise_df["Median Surprise (%)"],
                    marker_color=COLOR_PALETTE["success"],
                    hovertemplate="%{x}<br>Median: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

            # Positive vs Negative
            fig_earnings.add_trace(
                go.Bar(
                    name="Positive",
                    x=surprise_df["Metric"],
                    y=surprise_df["Positive Surprises"],
                    marker_color=COLOR_PALETTE["success"],
                    hovertemplate="%{x}<br>Positive: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )
            fig_earnings.add_trace(
                go.Bar(
                    name="Negative",
                    x=surprise_df["Metric"],
                    y=surprise_df["Negative Surprises"],
                    marker_color=COLOR_PALETTE["danger"],
                    hovertemplate="%{x}<br>Negative: %{y}<extra></extra>",
                ),
                row=1,
                col=2,
            )

            fig_earnings.update_layout(
                title="<b>Earnings Surprise Analysis</b><br><sup>Estimated vs Actual Performance by Metric</sup>",
                template=PLOTLY_TEMPLATE,
                height=500,
                font=dict(family="Segoe UI, Roboto, Arial"),
                title_font_size=20,
                barmode="group",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            fig_earnings.write_html(viz_output_dir / "earnings_surprise_analysis.html")
            print(f"  ✓ Saved: earnings_surprise_analysis.html")
            fig_earnings.show()
else:
    print("  ⚠ earnings_monitor.json not found, skipping visualization")

# ============================================================================
# 3. Dividend Yield Distribution Visualization
# ============================================================================
print("\n💰 Generating Dividend Yield Visualizations...")

dividend_file = financial_metrics_dir / "dividend_analytics.json"
if dividend_file.exists():
    with open(dividend_file, "r") as f:
        dividend_data = json.load(f)

    # Create dividend yield distribution visualization
    fig_dividend = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Dividend Yield Distribution",
            "Dividend Payers by Sector",
            "Yield by Size Class",
            "Yield by Style Class",
        ],
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    # Yield distribution buckets
    if "yield_distribution" in dividend_data:
        yield_dist = dividend_data["yield_distribution"]
        buckets = list(yield_dist.keys())
        counts = list(yield_dist.values())

        fig_dividend.add_trace(
            go.Bar(
                x=buckets,
                y=counts,
                marker_color=COLOR_PALETTE["success"],
                hovertemplate="Yield: %{x}<br>Count: %{y}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Sector distribution
    if "by_sector" in dividend_data:
        sector_data = dividend_data["by_sector"]
        sectors = list(sector_data.keys())[:10]  # Top 10
        payer_pcts = [sector_data[s].get("dividend_payer_pct", 0) for s in sectors]

        fig_dividend.add_trace(
            go.Bar(
                x=[s[:15] for s in sectors],
                y=payer_pcts,
                marker_color=COLOR_PALETTE["primary"],
                hovertemplate="%{x}<br>Payer %: %{y:.1f}%<extra></extra>",
            ),
            row=1,
            col=2,
        )

    # Size class distribution
    if "by_size_class" in dividend_data:
        size_data = dividend_data["by_size_class"]
        sizes = list(size_data.keys())
        mean_yields = [size_data[s].get("mean_yield", 0) for s in sizes]

        fig_dividend.add_trace(
            go.Bar(
                x=sizes,
                y=mean_yields,
                marker_color=COLOR_PALETTE["info"],
                hovertemplate="%{x}<br>Mean Yield: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Style class distribution
    if "by_style_class" in dividend_data:
        style_data = dividend_data["by_style_class"]
        styles = list(style_data.keys())
        mean_yields = [style_data[s].get("mean_yield", 0) for s in styles]

        fig_dividend.add_trace(
            go.Bar(
                x=styles,
                y=mean_yields,
                marker_color=COLOR_PALETTE["warning"],
                hovertemplate="%{x}<br>Mean Yield: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig_dividend.update_layout(
        title="<b>Dividend Analytics Dashboard</b><br><sup>Yield Distribution and Segment Analysis</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        showlegend=False,
    )
    fig_dividend.write_html(viz_output_dir / "dividend_yield_distribution.html")
    print(f"  ✓ Saved: dividend_yield_distribution.html")
    fig_dividend.show()
else:
    print("  ⚠ dividend_analytics.json not found, skipping visualization")

# ============================================================================
# 4. Analyst Recommendations Visualization
# ============================================================================
print("\n👥 Generating Analyst Recommendations Visualizations...")

analyst_file = financial_metrics_dir / "analyst_recommendations.json"
if analyst_file.exists():
    with open(analyst_file, "r") as f:
        analyst_data = json.load(f)

    # Create analyst recommendations dashboard
    fig_analyst = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Price Target Upside by Sector",
            "Analyst Coverage Distribution",
            "Upside by Size Class",
            "Upside by Style Class",
        ],
        specs=[[{"type": "bar"}, {"type": "pie"}], [{"type": "bar"}, {"type": "bar"}]],
    )

    # Sector upside analysis
    if "by_sector" in analyst_data:
        sector_stats = analyst_data["by_sector"]
        sectors_with_upside = [
            (k, v.get("mean_upside", 0)) for k, v in sector_stats.items() if "mean_upside" in v
        ]
        sectors_with_upside.sort(key=lambda x: x[1], reverse=True)

        if sectors_with_upside:
            top_sectors = sectors_with_upside[:10]
            sector_names = [s[0][:20] for s in top_sectors]
            upside_values = [s[1] for s in top_sectors]
            colors = [
                COLOR_PALETTE["success"] if v > 0 else COLOR_PALETTE["danger"]
                for v in upside_values
            ]

            fig_analyst.add_trace(
                go.Bar(
                    x=sector_names,
                    y=upside_values,
                    marker_color=colors,
                    hovertemplate="%{x}<br>Upside: %{y:.2f}%<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Coverage distribution (pie chart)
    if "coverage_distribution" in analyst_data:
        coverage = analyst_data["coverage_distribution"]
        labels = list(coverage.keys())
        values = list(coverage.values())

        fig_analyst.add_trace(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker_colors=[
                    COLOR_PALETTE["primary"],
                    COLOR_PALETTE["success"],
                    COLOR_PALETTE["info"],
                    COLOR_PALETTE["warning"],
                    COLOR_PALETTE["neutral"],
                ],
            ),
            row=1,
            col=2,
        )

    # Size class upside
    if "by_size_class" in analyst_data:
        size_stats = analyst_data["by_size_class"]
        sizes = list(size_stats.keys())
        upsides = [size_stats[s].get("mean_upside", 0) for s in sizes]

        fig_analyst.add_trace(
            go.Bar(
                x=sizes,
                y=upsides,
                marker_color=COLOR_PALETTE["info"],
                hovertemplate="%{x}<br>Upside: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Style class upside
    if "by_style_class" in analyst_data:
        style_stats = analyst_data["by_style_class"]
        styles = list(style_stats.keys())
        upsides = [style_stats[s].get("mean_upside", 0) for s in styles]

        fig_analyst.add_trace(
            go.Bar(
                x=styles,
                y=upsides,
                marker_color=COLOR_PALETTE["warning"],
                hovertemplate="%{x}<br>Upside: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=2,
        )

    fig_analyst.update_layout(
        title="<b>Analyst Recommendations Dashboard</b><br><sup>Price Target Analysis by Segment</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        showlegend=False,
    )
    fig_analyst.write_html(viz_output_dir / "analyst_recommendations_chart.html")
    print(f"  ✓ Saved: analyst_recommendations_chart.html")
    fig_analyst.show()
else:
    print("  ⚠ analyst_recommendations.json not found, skipping visualization")

# ============================================================================
# 5. Sector Benchmarking Visualization
# ============================================================================
print("\n📊 Generating Sector Benchmarking Visualizations...")

# Create sector benchmarking comparison
if "sector" in all_stocks_enhanced.columns:
    # Select key metrics for benchmarking
    benchmark_metrics = ["roe", "roa", "debt_to_equity", "price_momentum_1m", "piotroski_f_score"]
    benchmark_metrics = [m for m in benchmark_metrics if m in all_stocks_enhanced.columns]

    if len(benchmark_metrics) >= 3:
        # Calculate sector medians for benchmarking
        top_sectors = all_stocks_enhanced["sector"].value_counts().head(8).index.tolist()
        sector_benchmark_data = []

        for sector in top_sectors:
            sector_df = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector]
            sector_row = {"Sector": str(sector)[:20]}

            for metric in benchmark_metrics:
                median_val = sector_df[metric].median()
                sector_row[metric] = median_val if pd.notna(median_val) else 0

            sector_benchmark_data.append(sector_row)

        benchmark_df = pd.DataFrame(sector_benchmark_data)

        # Normalize for radar chart (0-1 scale)
        for metric in benchmark_metrics:
            col_min = benchmark_df[metric].min()
            col_max = benchmark_df[metric].max()
            if col_max != col_min:
                benchmark_df[f"{metric}_norm"] = (benchmark_df[metric] - col_min) / (
                    col_max - col_min
                )
            else:
                benchmark_df[f"{metric}_norm"] = 0.5

        # Create radar chart
        fig_benchmark = go.Figure()

        colors = px.colors.qualitative.Set2[: len(top_sectors)]
        for idx, row in benchmark_df.iterrows():
            r_values = [row[f"{m}_norm"] for m in benchmark_metrics]
            r_values.append(r_values[0])  # Close the polygon

            theta_values = [m.replace("_", " ").title() for m in benchmark_metrics]
            theta_values.append(theta_values[0])

            fig_benchmark.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill="toself",
                    name=row["Sector"],
                    opacity=0.6,
                    line=dict(color=colors[idx % len(colors)]),
                )
            )

        fig_benchmark.update_layout(
            title="<b>Sector Benchmarking Comparison</b><br><sup>Normalized Median Metrics by Sector (Radar Chart)</sup>",
            template=PLOTLY_TEMPLATE,
            height=600,
            font=dict(family="Segoe UI, Roboto, Arial"),
            title_font_size=20,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        fig_benchmark.write_html(viz_output_dir / "sector_benchmarking.html")
        print(f"  ✓ Saved: sector_benchmarking.html")
        fig_benchmark.show()
else:
    print("  ⚠ Sector column not available for benchmarking")

# ============================================================================
# 6. Financial Metrics Radar by Region
# ============================================================================
print("\n🌍 Generating Financial Metrics Radar by Region...")

if "region" in all_stocks_enhanced.columns:
    radar_metrics = ["roe", "roa", "p_e_ratio", "debt_to_equity", "piotroski_f_score"]
    radar_metrics = [m for m in radar_metrics if m in all_stocks_enhanced.columns]

    if len(radar_metrics) >= 3:
        regions = all_stocks_enhanced["region"].dropna().unique().tolist()

        region_radar_data = []
        for region in regions:
            region_df = all_stocks_enhanced[all_stocks_enhanced["region"] == region]
            region_row = {"Region": region}

            for metric in radar_metrics:
                median_val = region_df[metric].median()
                region_row[metric] = median_val if pd.notna(median_val) else 0

            region_radar_data.append(region_row)

        radar_df = pd.DataFrame(region_radar_data)

        # Normalize for radar chart
        for metric in radar_metrics:
            col_min = radar_df[metric].min()
            col_max = radar_df[metric].max()
            if col_max != col_min:
                radar_df[f"{metric}_norm"] = (radar_df[metric] - col_min) / (col_max - col_min)
            else:
                radar_df[f"{metric}_norm"] = 0.5

        # Create radar chart
        fig_radar = go.Figure()

        region_colors = {
            "US": COLOR_PALETTE["primary"],
            "EU": COLOR_PALETTE["success"],
            "APAC": COLOR_PALETTE["warning"],
            "ROTW": COLOR_PALETTE["info"],
        }

        for idx, row in radar_df.iterrows():
            r_values = [row[f"{m}_norm"] for m in radar_metrics]
            r_values.append(r_values[0])

            theta_values = [m.replace("_", " ").title() for m in radar_metrics]
            theta_values.append(theta_values[0])

            color = region_colors.get(row["Region"], COLOR_PALETTE["neutral"])

            fig_radar.add_trace(
                go.Scatterpolar(
                    r=r_values,
                    theta=theta_values,
                    fill="toself",
                    name=row["Region"],
                    opacity=0.6,
                    line=dict(color=color, width=2),
                )
            )

        fig_radar.update_layout(
            title="<b>Financial Metrics Comparison by Region</b><br><sup>Normalized Median Values (Radar Chart)</sup>",
            template=PLOTLY_TEMPLATE,
            height=550,
            font=dict(family="Segoe UI, Roboto, Arial"),
            title_font_size=20,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        )
        fig_radar.write_html(viz_output_dir / "financial_metrics_radar.html")
        print(f"  ✓ Saved: financial_metrics_radar.html")
        fig_radar.show()
else:
    print("  ⚠ Region column not available for radar chart")

print(f"\n✓ Enhanced Interactive Visualizations Complete")
print(f"  Output directory: {viz_output_dir}")
print(f"  New HTML files: 6 visualizations generated")


# In[17]:


# ============================================================================
# Cell 10.11: Advanced Benchmarking & Risk Analytics
# ============================================================================

print("=" * 80)
print("ADVANCED BENCHMARKING & RISK ANALYTICS")
print("=" * 80)

# Create output directory for advanced analytics
advanced_analytics_dir = OUTPUT_DIR / "eda" / "advanced_analytics"
advanced_analytics_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 1. Value at Risk (VaR) & Expected Shortfall Analytics
# ============================================================================
print("\n📊 Generating VaR & Risk Analytics Visualization...")

# Calculate VaR proxies using available volatility and return metrics
volatility_cols = [
    c for c in all_stocks_enhanced.columns if "volatility" in c.lower() or "beta" in c.lower()
]
return_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m", "price_momentum_12m"]
return_cols = [c for c in return_cols if c in all_stocks_enhanced.columns]

if volatility_cols or return_cols:
    # Build VaR analytics data
    var_data = []
    risk_cols = (volatility_cols + return_cols)[:5]  # Limit to 5 metrics

    if "sector" in all_stocks_enhanced.columns:
        for sector in all_stocks_enhanced["sector"].dropna().unique():
            sector_df = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector]
            if len(sector_df) >= 10:
                sector_row = {"Sector": str(sector)[:25], "Count": len(sector_df)}

                for col in risk_cols:
                    if col in sector_df.columns:
                        data = sector_df[col].dropna()
                        if len(data) > 5:
                            # Calculate VaR at 95% confidence (5th percentile for losses)
                            var_95 = np.percentile(data, 5)
                            # Expected Shortfall (CVaR) - mean of values below VaR
                            es_95 = (
                                data[data <= var_95].mean()
                                if len(data[data <= var_95]) > 0
                                else var_95
                            )
                            sector_row[f"{col}_VaR95"] = float(var_95)
                            sector_row[f"{col}_ES95"] = float(es_95)

                var_data.append(sector_row)

    if var_data:
        var_df = pd.DataFrame(var_data)

        # Create VaR visualization with subplots
        var_metrics = [c for c in var_df.columns if "VaR95" in c][:4]

        if var_metrics:
            fig_var = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=[
                    m.replace("_VaR95", "").replace("_", " ").title() + " VaR Analysis"
                    for m in var_metrics[:4]
                ],
                vertical_spacing=0.15,
                horizontal_spacing=0.12,
            )

            for idx, metric in enumerate(var_metrics[:4]):
                row = idx // 2 + 1
                col = idx % 2 + 1
                es_metric = metric.replace("VaR95", "ES95")

                # Sort by VaR
                plot_df = var_df[["Sector", metric, es_metric]].dropna().sort_values(metric)

                # VaR bars
                fig_var.add_trace(
                    go.Bar(
                        x=plot_df["Sector"],
                        y=plot_df[metric],
                        name="VaR (95%)",
                        marker_color=COLOR_PALETTE["warning"],
                        hovertemplate="<b>%{x}</b><br>VaR 95%: %{y:.3f}<extra></extra>",
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )

                # ES bars
                fig_var.add_trace(
                    go.Bar(
                        x=plot_df["Sector"],
                        y=plot_df[es_metric],
                        name="Expected Shortfall",
                        marker_color=COLOR_PALETTE["danger"],
                        hovertemplate="<b>%{x}</b><br>ES 95%: %{y:.3f}<extra></extra>",
                        showlegend=(idx == 0),
                    ),
                    row=row,
                    col=col,
                )

            fig_var.update_layout(
                title="<b>Value at Risk (VaR) & Expected Shortfall by Sector</b><br><sup>95% Confidence Level Risk Metrics</sup>",
                template=PLOTLY_TEMPLATE,
                height=800,
                font=dict(family="Segoe UI, Roboto, Arial"),
                title_font_size=20,
                barmode="group",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
            )
            fig_var.update_xaxes(tickangle=45)
            fig_var.write_html(advanced_analytics_dir / "var_risk_analytics.html")
            print(f"  ✓ Saved: var_risk_analytics.html")
            fig_var.show()
else:
    print("  ⚠ Insufficient volatility/return columns for VaR analytics")

# ============================================================================
# 2. Statistical Benchmarking Comparison
# ============================================================================
print("\n📈 Generating Statistical Benchmarking Visualization...")

# Load hypothesis test results if available
hypothesis_file = financial_metrics_dir / "hypothesis_tests.json"
benchmark_data = []

if hypothesis_file.exists():
    with open(hypothesis_file, "r") as f:
        hyp_results = json.load(f)

    # Extract test results for visualization
    if "tests" in hyp_results:
        for metric, test_info in hyp_results["tests"].items():
            if isinstance(test_info, dict):
                benchmark_data.append(
                    {
                        "Metric": metric.replace("_", " ").title(),
                        "Test Statistic": test_info.get("statistic", 0),
                        "P-Value": test_info.get("p_value", 1),
                        "Significant": test_info.get("significant", False),
                        "Test Type": test_info.get("test_type", "Unknown"),
                    }
                )

# If no hypothesis file, compute fresh tests
if not benchmark_data and "sector" in all_stocks_enhanced.columns:
    test_metrics = [
        "roe",
        "roa",
        "p_e_ratio",
        "debt_to_equity",
        "price_momentum_1m",
        "piotroski_f_score",
    ]
    test_metrics = [m for m in test_metrics if m in all_stocks_enhanced.columns]

    for metric in test_metrics:
        groups = []
        for sector in all_stocks_enhanced["sector"].dropna().unique():
            sector_data = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector][
                metric
            ].dropna()
            if len(sector_data) >= 5:
                groups.append(sector_data.values)

        if len(groups) >= 2:
            # Kruskal-Wallis test (non-parametric ANOVA)
            try:
                stat, p_val = scipy_stats.kruskal(*groups)
                benchmark_data.append(
                    {
                        "Metric": metric.replace("_", " ").title(),
                        "Test Statistic": float(stat),
                        "P-Value": float(p_val),
                        "Significant": p_val < 0.05,
                        "Test Type": "Kruskal-Wallis",
                    }
                )
            except (ValueError, TypeError):
                pass

if benchmark_data:
    bench_df = pd.DataFrame(benchmark_data)

    # Create statistical benchmarking visualization
    fig_bench = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Test Statistics by Metric", "P-Values (Log Scale)"],
        horizontal_spacing=0.15,
    )

    # Sort by test statistic
    bench_df_sorted = bench_df.sort_values("Test Statistic", ascending=True)

    # Test statistic bars
    colors = [
        COLOR_PALETTE["success"] if sig else COLOR_PALETTE["neutral"]
        for sig in bench_df_sorted["Significant"]
    ]

    fig_bench.add_trace(
        go.Bar(
            x=bench_df_sorted["Test Statistic"],
            y=bench_df_sorted["Metric"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Statistic: %{x:.2f}<extra></extra>",
            name="Test Statistic",
        ),
        row=1,
        col=1,
    )

    # P-value scatter (log scale)
    fig_bench.add_trace(
        go.Scatter(
            x=bench_df_sorted["P-Value"],
            y=bench_df_sorted["Metric"],
            mode="markers",
            marker=dict(
                size=15,
                color=colors,
                line=dict(width=2, color="white"),
            ),
            hovertemplate="<b>%{y}</b><br>P-Value: %{x:.4f}<extra></extra>",
            name="P-Value",
        ),
        row=1,
        col=2,
    )

    # Add significance threshold line
    fig_bench.add_vline(x=0.05, line_dash="dash", line_color=COLOR_PALETTE["danger"], row=1, col=2)
    fig_bench.add_annotation(
        x=0.05,
        y=len(bench_df) - 1,
        text="α=0.05",
        showarrow=False,
        font=dict(color=COLOR_PALETTE["danger"]),
        xanchor="left",
        row=1,
        col=2,
    )

    fig_bench.update_xaxes(type="log", title="P-Value (Log Scale)", row=1, col=2)
    fig_bench.update_xaxes(title="Test Statistic", row=1, col=1)

    fig_bench.update_layout(
        title="<b>Statistical Benchmarking: Sector Differences Analysis</b><br><sup>Kruskal-Wallis Tests (Green = Significant at α=0.05)</sup>",
        template=PLOTLY_TEMPLATE,
        height=500,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        showlegend=False,
    )
    fig_bench.write_html(advanced_analytics_dir / "statistical_benchmarking.html")
    print(f"  ✓ Saved: statistical_benchmarking.html")
    fig_bench.show()
else:
    print("  ⚠ Insufficient data for statistical benchmarking")

# ============================================================================
# 3. Performance Attribution Sunburst
# ============================================================================
print("\n🌐 Generating Performance Attribution Sunburst...")

if "region" in all_stocks_enhanced.columns and "sector" in all_stocks_enhanced.columns:
    # Build hierarchical data for sunburst with 4 levels: Region → Sector → Industry → Name
    sunburst_data = []

    # Calculate mean ROE or available profitability metric as performance proxy
    perf_metric = "roe" if "roe" in all_stocks_enhanced.columns else "roa"

    # Detect industry column
    industry_cols = ["industry"]
    industry_col = None
    for col in industry_cols:
        if col in all_stocks_enhanced.columns:
            industry_col = col
            break

    # Detect market cap column
    market_cap_col = None
    market_cap_cols = ["market_cap", "market_capitalization", "mkt_cap"]
    for col in market_cap_cols:
        if col in all_stocks_enhanced.columns:
            market_cap_col = col
            break

    if perf_metric in all_stocks_enhanced.columns and industry_col and market_cap_col:
        for region in all_stocks_enhanced["region"].dropna().unique():
            region_df = all_stocks_enhanced[all_stocks_enhanced["region"] == region]

            # Level 2: Sector hierarchy
            for sector in region_df["sector"].dropna().unique():
                sector_df = region_df[region_df["sector"] == sector]

                # Level 3: Industry hierarchy within Sector
                for industry in sector_df[industry_col].dropna().unique():
                    industry_df = sector_df[sector_df[industry_col] == industry]

                    if len(industry_df) >= 5:
                        # Level 4: Add top 5 stocks by performance metric AND market cap
                        # Filter stocks with valid performance metric and market cap
                        valid_stocks = industry_df[
                            (industry_df[perf_metric].notna())
                            & (industry_df[market_cap_col].notna())
                        ].copy()

                        if len(valid_stocks) > 0:
                            # Rank by performance metric (descending)
                            valid_stocks["perf_rank"] = valid_stocks[perf_metric].rank(
                                ascending=False, method="average"
                            )
                            # Rank by market cap (descending)
                            valid_stocks["mktcap_rank"] = valid_stocks[market_cap_col].rank(
                                ascending=False, method="average"
                            )
                            # Combined score (lower is better)
                            valid_stocks["combined_score"] = (
                                valid_stocks["perf_rank"] + valid_stocks["mktcap_rank"]
                            )

                            # Sort by combined score and get top 5
                            top_stocks = valid_stocks.nsmallest(5, "combined_score")

                            for idx, row in top_stocks.iterrows():
                                stock_name = str(row.get("name", row.get("ticker", "Unknown")))[
                                    :30
                                ]  # Truncate long names
                                stock_perf = (
                                    float(row[perf_metric]) if pd.notna(row[perf_metric]) else 0
                                )

                                sunburst_data.append(
                                    {
                                        "Region": str(region),
                                        "Sector": str(sector)[:20],
                                        "Industry": str(industry)[:25],
                                        "Name": stock_name,
                                        "Count": 1,  # Each stock counts as 1
                                        "Performance": stock_perf,
                                        "Ticker": str(row.get("ticker", "")),  # For hover data
                                    }
                                )

        if sunburst_data:
            sunburst_df = pd.DataFrame(sunburst_data)

            # Create sunburst with 4-level hierarchy
            fig_sunburst = px.sunburst(
                sunburst_df,
                path=["Region", "Sector", "Industry", "Name"],
                values="Count",
                color="Performance",
                color_continuous_scale="RdYlGn",
                color_continuous_midpoint=0,
                hover_data=["Ticker"],
                title=f"<b>Performance Attribution by Region, Sector, Industry & Top Stocks</b><br><sup>Size = Stock Count, Color = {perf_metric.upper()}, Name = Top 5 Stocks per Industry (by Performance & Market Cap)</sup>",
            )

            fig_sunburst.update_layout(
                template=PLOTLY_TEMPLATE,
                height=700,
                font=dict(family="Segoe UI, Roboto, Arial"),
                title_font_size=20,
            )

            fig_sunburst.write_html(
                advanced_analytics_dir / "performance_attribution_sunburst.html"
            )
            print(f"  ✓ Saved: performance_attribution_sunburst.html")
            print(f"  ✓ Hierarchy: Region → Sector → Industry → Name (Top 5 stocks per industry)")
            print(f"  ✓ Total stocks displayed: {len(sunburst_df)}")
            print(f"  ✓ Performance metric: {perf_metric.upper()}")
            print(
                f"  ✓ Selection criteria: Combined ranking by {perf_metric.upper()} and Market Cap"
            )
            fig_sunburst.show()
    else:
        print(
            "  ⚠ Required columns not available for enhanced sunburst (need: region, sector, industry, performance metric, market_cap)"
        )
else:
    print("  ⚠ Region/Sector columns not available for sunburst")

# ============================================================================
# 4. Cross-Sectional Risk Heatmap
# ============================================================================
print("\n🔥 Generating Cross-Sectional Risk Heatmap...")

risk_metrics = ["beta_5y", "volatility_90d", "debt_to_equity", "altman_z_score"]
risk_metrics = [m for m in risk_metrics if m in all_stocks_enhanced.columns]

if risk_metrics and "sector" in all_stocks_enhanced.columns:
    # Calculate median risk metrics by sector
    risk_by_sector = []
    top_sectors = all_stocks_enhanced["sector"].value_counts().head(12).index.tolist()

    for sector in top_sectors:
        sector_df = all_stocks_enhanced[all_stocks_enhanced["sector"] == sector]
        sector_row = {"Sector": str(sector)[:25]}

        for metric in risk_metrics:
            median_val = sector_df[metric].median()
            sector_row[metric] = float(median_val) if pd.notna(median_val) else 0

        risk_by_sector.append(sector_row)

    risk_df = pd.DataFrame(risk_by_sector).set_index("Sector")

    # Normalize for heatmap (z-scores)
    risk_normalized = (risk_df - risk_df.mean()) / risk_df.std()
    risk_normalized = risk_normalized.fillna(0)

    fig_risk_heat = px.imshow(
        risk_normalized,
        title="<b>Cross-Sectional Risk Profile by Sector</b><br><sup>Z-Score Normalized Risk Metrics (Red = Higher Risk)</sup>",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="RdYlGn_r",  # Reversed so red = high risk
        aspect="auto",
        text_auto=".2f",
    )
    fig_risk_heat.update_layout(
        height=600,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        xaxis_title="Risk Metric",
        yaxis_title="Sector",
    )
    fig_risk_heat.update_xaxes(tickangle=45)
    fig_risk_heat.write_html(advanced_analytics_dir / "cross_sectional_risk_heatmap.html")
    print(f"  ✓ Saved: cross_sectional_risk_heatmap.html")
    fig_risk_heat.show()
else:
    print("  ⚠ Insufficient risk metrics for cross-sectional heatmap")

# ============================================================================
# Export Advanced Analytics Summary
# ============================================================================
advanced_analytics_summary = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks": len(all_stocks_enhanced),
    "visualizations_generated": [
        "var_risk_analytics.html",
        "statistical_benchmarking.html",
        "performance_attribution_sunburst.html",
        "cross_sectional_risk_heatmap.html",
    ],
    "output_directory": str(advanced_analytics_dir),
}

summary_path = advanced_analytics_dir / "advanced_analytics_summary.json"
with open(summary_path, "w") as f:
    json.dump(advanced_analytics_summary, f, indent=2)
print(f"\n✓ Summary saved: {summary_path}")

print(f"\n✓ Advanced Benchmarking & Risk Analytics Complete")
print(f"  Output directory: {advanced_analytics_dir}")
print(f"  New HTML files: 4 visualizations generated")


# ## Cell 10.12: Phase 9.3 Enhanced Category Analytics
#
# Advanced visualizations for Phase 9.3 feature category analysis across regions and sectors.
#
# **Key Visualizations:**
# 1. **Regional Performance Radar Charts** - Category scores (z-scored) by region across 11 feature categories
# 2. **Category Distribution Box Plots** - Distribution of category scores by sector
# 3. **Value vs Quality Bubble Chart** - Strategic positioning scatter with quadrant analysis
#
# **Outputs:**
# - phase93_regional_radar_charts.html
# - phase93_category_distributions_boxplots.html
# - phase93_value_quality_bubble_chart.html
#

# In[18]:


# ============================================================================
# Cell 10.12: Phase 9.3 Enhanced Category Analytics
# ============================================================================

from scipy.stats import zscore

print("=" * 80)
print("PHASE 9.3 ENHANCED CATEGORY ANALYTICS")
print("=" * 80)

# Create output directory for enhanced category analytics
eda_output_dir = OUTPUT_DIR / "eda"
eda_output_dir.mkdir(parents=True, exist_ok=True)

# Define category mapping using PHASE93_FEATURE_CATEGORIES
# Map to metrics available in all_stocks_enhanced
category_mapping = {}
for category, features in PHASE93_FEATURE_CATEGORIES.items():
    available_features = [f for f in features if f in all_stocks_enhanced.columns]
    if available_features:
        category_mapping[category] = available_features

print(f"\n📊 Category Mapping Summary:")
print(f"  Total categories: {len(category_mapping)}")
for cat, feats in list(category_mapping.items())[:5]:
    print(f"    {cat}: {len(feats)} features")
print(f"    ...")

# ============================================================================
# 1. Build Category Score Matrix
# ============================================================================
print("\n📈 Building Category Score Matrix...")

# Create z-scored category scores for each stock
category_score_matrix = pd.DataFrame(index=all_stocks_enhanced.index)

for category_name, category_metrics in category_mapping.items():
    if len(category_metrics) == 0:
        continue

    # Get available metrics
    category_data = all_stocks_enhanced[category_metrics].copy()

    # Convert to numeric
    for col in category_metrics:
        category_data[col] = pd.to_numeric(category_data[col], errors="coerce")

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data.apply(
        lambda x: zscore(x, nan_policy="omit") if x.notna().sum() > 1 else x
    )

    # Average z-scores for category score
    category_score_matrix[category_name] = z_scored_data.mean(axis=1)

print(f"  Category score matrix shape: {category_score_matrix.shape}")
print(f"  Categories computed: {len(category_score_matrix.columns)}")

# ============================================================================
# 2. Regional Performance Radar Charts
# ============================================================================
print("\n📊 Regional Performance Radar Charts:")

# Compute category scores by region
category_region_scores = {}

for category_name, category_metrics in category_mapping.items():
    available_in_category = [m for m in category_metrics if m in all_stocks_enhanced.columns]

    if len(available_in_category) == 0:
        continue

    # Compute z-scores for available metrics and average by region
    category_data = all_stocks_enhanced[available_in_category + ["region"]].copy()

    # Convert to numeric and compute z-scores
    for col in available_in_category:
        category_data[col] = pd.to_numeric(category_data[col], errors="coerce")

    # Compute z-scores (handle NaNs)
    z_scored_data = category_data[available_in_category].apply(
        lambda x: zscore(x, nan_policy="omit") if x.notna().sum() > 1 else x
    )
    category_data["category_score"] = z_scored_data.mean(axis=1)

    # Aggregate by region
    region_scores = category_data.groupby("region")["category_score"].mean()
    category_region_scores[category_name] = region_scores

if category_region_scores:
    # Create radar chart for each region
    radar_df = pd.DataFrame(category_region_scores)

    # Create single figure with all regions
    fig_radar = go.Figure()

    regions = radar_df.index.tolist()
    categories = radar_df.columns.tolist()

    for region in regions:
        values = radar_df.loc[region].tolist()
        values.append(values[0])  # Close the radar chart

        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=region,
                opacity=0.6,
            )
        )

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-1, 1])),
        showlegend=True,
        title="<b>Regional Performance Across Feature Categories (Phase 9.3)</b><br><sup>Z-Score Normalized Category Averages</sup>",
        template=PLOTLY_TEMPLATE,
        height=700,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
    )

    fig_radar.show()
    output_path = eda_output_dir / "phase93_regional_radar_charts.html"
    fig_radar.write_html(output_path)

    print(f"\n✓ Regional radar charts complete")
    print(f"  Regions visualized: {len(regions)}")
    print(f"  Categories analyzed: {len(categories)}")
    print(f"  Output: {output_path}")

    # Display top category per region
    print(f"\n  🌍 Strongest Category by Region:")
    for region in regions[:5]:
        top_category = radar_df.loc[region].idxmax()
        top_score = radar_df.loc[region].max()
        print(f"    {region}: {top_category} (z-score: {top_score:.2f})")
else:
    print("  ⚠️ No regional category data available")

# ============================================================================
# 3. Category Distribution Box Plots
# ============================================================================
print("\n📊 Category Distribution Box Plots:")

# Create box plots for each category showing distribution across sectors
if not category_score_matrix.empty:
    # Add sector information to category scores
    category_scores_with_sector = category_score_matrix.copy()
    category_scores_with_sector["sector"] = all_stocks_enhanced["sector"].values

    # Create subplot grid for all categories
    categories = [col for col in category_score_matrix.columns]
    n_categories = len(categories)
    n_cols = 2
    n_rows = math.ceil(n_categories / n_cols)

    fig_box = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=categories,
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # Get unique sectors for consistent coloring
    unique_sectors = category_scores_with_sector["sector"].dropna().unique()

    for idx, category in enumerate(categories):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        # Create box plot data for this category
        for sector_idx, sector in enumerate(unique_sectors):
            if pd.notna(sector):
                sector_data = category_scores_with_sector[
                    category_scores_with_sector["sector"] == sector
                ][category].dropna()

                if len(sector_data) > 0:
                    fig_box.add_trace(
                        go.Box(
                            y=sector_data,
                            name=str(sector)[:15],
                            showlegend=(idx == 0),  # Only show legend for first subplot
                            marker_color=px.colors.qualitative.Plotly[sector_idx % 10],
                        ),
                        row=row,
                        col=col,
                    )

    fig_box.update_layout(
        title_text="<b>Category Score Distributions by Sector (Phase 9.3)</b><br><sup>Z-Score Normalized Feature Categories</sup>",
        template=PLOTLY_TEMPLATE,
        height=300 * n_rows,
        showlegend=True,
        font=dict(family="Segoe UI, Roboto, Arial"),
        title_font_size=20,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="center", x=0.5),
    )

    fig_box.update_yaxes(title_text="Z-Score")

    fig_box.show()
    output_path = eda_output_dir / "phase93_category_distributions_boxplots.html"
    fig_box.write_html(output_path)

    print(f"\n✓ Category distribution box plots complete")
    print(f"  Categories visualized: {n_categories}")
    print(f"  Grid layout: {n_rows}×{n_cols}")
    print(f"  Output: {output_path}")

    # Identify categories with highest variance
    category_variances = category_score_matrix.var().sort_values(ascending=False)
    print(f"\n  📊 Categories with Highest Variance:")
    for category, variance in category_variances.head(5).items():
        print(f"    {category}: {variance:.2f}")
else:
    print("  ⚠️ No category score data available for box plots")

# ============================================================================
# 4. Value vs Quality Bubble Chart (Strategic Positioning)
# ============================================================================
print("\n📊 Value vs Quality Bubble Chart:")

# Create scatter plot comparing two key categories with sector coloring
if not category_score_matrix.empty:
    # Select two categories for comparison (Valuation vs Quality)
    categories_list = list(category_score_matrix.columns)

    # Default to Valuation Ratios vs Quality & Risk if available
    x_category = "Valuation Ratios" if "Valuation Ratios" in categories_list else categories_list[0]
    y_category = (
        "Quality & Risk"
        if "Quality & Risk" in categories_list
        else (categories_list[1] if len(categories_list) > 1 else categories_list[0])
    )

    # Prepare data for bubble chart
    bubble_data = pd.DataFrame(
        {
            x_category: category_score_matrix[x_category],
            y_category: category_score_matrix[y_category],
            "sector": all_stocks_enhanced["sector"].values,
            "ticker": all_stocks_enhanced.get("ticker", range(len(category_score_matrix))),
            "market_cap": all_stocks_enhanced.get(
                "market_cap", pd.Series([100] * len(category_score_matrix))
            ),
        }
    ).dropna()

    if len(bubble_data) > 0:
        # Create bubble chart
        fig_bubble = px.scatter(
            bubble_data,
            x=x_category,
            y=y_category,
            color="sector",
            size="market_cap",
            hover_data=["ticker"],
            title=f"<b>Strategic Positioning: {x_category} vs {y_category} (Phase 9.3)</b><br><sup>Bubble Size = Market Cap</sup>",
            labels={x_category: f"{x_category} Score (Z)", y_category: f"{y_category} Score (Z)"},
            template=PLOTLY_TEMPLATE,
            size_max=30,
            opacity=0.6,
        )

        # Add quadrant lines
        fig_bubble.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig_bubble.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

        # Add quadrant labels
        fig_bubble.add_annotation(
            text="High Quality,<br>High Valuation",
            x=1.5,
            y=1.5,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
        fig_bubble.add_annotation(
            text="High Quality,<br>Low Valuation",
            x=-1.5,
            y=1.5,
            showarrow=False,
            font=dict(size=10, color="green"),
        )
        fig_bubble.add_annotation(
            text="Low Quality,<br>Low Valuation",
            x=-1.5,
            y=-1.5,
            showarrow=False,
            font=dict(size=10, color="gray"),
        )
        fig_bubble.add_annotation(
            text="Low Quality,<br>High Valuation",
            x=1.5,
            y=-1.5,
            showarrow=False,
            font=dict(size=10, color="red"),
        )

        fig_bubble.update_layout(
            height=700,
            font=dict(family="Segoe UI, Roboto, Arial"),
            title_font_size=20,
        )

        fig_bubble.show()
        output_path = eda_output_dir / "phase93_value_quality_bubble_chart.html"
        fig_bubble.write_html(output_path)

        print(f"\n✓ Value vs Quality bubble chart complete")
        print(f"  X-axis: {x_category}")
        print(f"  Y-axis: {y_category}")
        print(f"  Data points: {len(bubble_data)}")
        print(f"  Output: {output_path}")

        # Identify quadrants
        q1 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] > 0)]
        q2 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] > 0)]
        q3 = bubble_data[(bubble_data[x_category] < 0) & (bubble_data[y_category] < 0)]
        q4 = bubble_data[(bubble_data[x_category] > 0) & (bubble_data[y_category] < 0)]

        print(f"\n  📍 Quadrant Distribution:")
        print(
            f"    Q1 (High Val, High Qual): {len(q1)} stocks ({len(q1) / len(bubble_data) * 100:.1f}%)"
        )
        print(
            f"    Q2 (Low Val, High Qual): {len(q2)} stocks ({len(q2) / len(bubble_data) * 100:.1f}%) - Value opportunities"
        )
        print(
            f"    Q3 (Low Val, Low Qual): {len(q3)} stocks ({len(q3) / len(bubble_data) * 100:.1f}%)"
        )
        print(
            f"    Q4 (High Val, Low Qual): {len(q4)} stocks ({len(q4) / len(bubble_data) * 100:.1f}%) - Risk flags"
        )
    else:
        print("  ⚠️ Insufficient data for bubble chart")
else:
    print("  ⚠️ No category score data available for bubble chart")

# ============================================================================
# Export Enhanced Category Analytics Summary
# ============================================================================
enhanced_category_summary = {
    "timestamp": datetime.now().isoformat(),
    "total_stocks": len(all_stocks_enhanced),
    "categories_analyzed": len(category_mapping),
    "visualizations_generated": [
        "phase93_regional_radar_charts.html",
        "phase93_category_distributions_boxplots.html",
        "phase93_value_quality_bubble_chart.html",
    ],
    "output_directory": str(eda_output_dir),
}

summary_path = eda_output_dir / "phase93_enhanced_category_analytics_summary.json"
with open(summary_path, "w") as f:
    json.dump(enhanced_category_summary, f, indent=2)
print(f"\n✓ Summary saved: {summary_path}")

print(f"\n✓ Phase 9.3 Enhanced Category Analytics Complete")
print(f"  Output directory: {eda_output_dir}")
print(f"  New HTML files: 3 visualizations generated")


# ## Cell 11: Summary & Next Steps
#
# Pipeline execution summary and recommendations for next analysis phases.
#

# In[19]:


# ============================================================================
# Cell 11: Summary & Next Steps
# ============================================================================

print("=" * 80)
print("ETL DATA EXPLORER - EXECUTION SUMMARY")
print("=" * 80)

# Summary statistics (using enhanced features DataFrame)
print(f"\n📊 Dataset Summary:")
print(f"  Total stocks processed:     {len(all_stocks_enhanced):,}")
print(f"  Original columns (ETL):     {len(all_stocks_preprocessed.columns)}")
print(f"  Feature engineered columns: {len(all_stocks_preprocessed.columns)}")
print(f"  Enhanced final columns:     {len(all_stocks_enhanced.columns)}")
print(
    f"  Total features added:       {len(all_stocks_enhanced.columns) - len(all_stocks_preprocessed.columns)}"
)

print(f"\n📄 Generated Outputs:")
print(f"  JSON Reports (13 files):")
print(f"    1. eda_summary.json")
print(f"    2. data_quality_alerts.json")
print(f"    3. metrics_dashboard.json")
print(f"    4. hypothesis_tests.json")
print(f"    5. financial_metrics_dashboard.json")
print(f"    6. multi_dimensional_valuation_analysis.json")
print(f"    7. earnings_monitor.json")
print(f"    8. analyst_recommendations.json")
print(f"    9. earnings_estimates_analysis.json")
print(f"    10. dividend_analytics.json")
print(f"    11. advanced_analytics_summary.json")
print(f"    12. phase93_feature_viz_summary.json")
print(f"    13. phase93_enhanced_category_analytics_summary.json")
print(f"  HTML Visualizations - Cell 6.5 (5 files):")
print(f"    1. phase93_feature_treemap.html")
print(f"    2. phase93_category_coverage.html")
print(f"    3. phase93_category_sector_heatmap.html")
print(f"    4. phase93_feature_radar.html")
print(f"    5. phase93_category_sunburst.html")
print(f"  HTML Visualizations - Cell 10 (7 files):")
print(f"    6. correlation_heatmap.html")
print(f"    7. distributions.html")
print(f"    8. valuation_3d.html")
print(f"    9. region_sector_heatmap.html")
print(f"    10. sector_boxplots.html")
print(f"    11. regional_comparison.html")
print(f"    12. phase93_category_sector_bubble_chart.html")
print(f"  HTML Visualizations - Cell 10.10 (6 files):")
print(f"    13. hypothesis_test_heatmap.html")
print(f"    14. earnings_surprise_analysis.html")
print(f"    15. dividend_yield_distribution.html")
print(f"    16. analyst_recommendations_chart.html")
print(f"    17. sector_benchmarking.html")
print(f"    18. financial_metrics_radar.html")
print(f"  HTML Visualizations - Cell 10.11 (4 files):")
print(f"    19. var_risk_analytics.html")
print(f"    20. statistical_benchmarking.html")
print(f"    21. performance_attribution_sunburst.html")
print(f"    22. cross_sectional_risk_heatmap.html")
print(f"  HTML Visualizations - Cell 10.12 (3 files):")
print(f"    23. phase93_regional_radar_charts.html")
print(f"    24. phase93_category_distributions_boxplots.html")
print(f"    25. phase93_value_quality_bubble_chart.html")
print(f"  Output directories:")
print(f"    • {financial_metrics_dir}")
print(f'    • {OUTPUT_DIR / "eda" / "visualizations"}')
print(f'    • {OUTPUT_DIR / "eda" / "advanced_analytics"}')
print(f'    • {OUTPUT_DIR / "eda" / "phase93_feature_categories"}')

if "region" in all_stocks_enhanced.columns:
    print(f"\n📍 Geographic Distribution:")
    for region, count in all_stocks_preprocessed["region"].value_counts().head(5).items():
        pct = count / len(all_stocks_preprocessed) * 100
        print(f"  {region:20s}: {count:5,} ({pct:5.1f}%)")

if "sector" in all_stocks_preprocessed.columns:
    print(f"\n🏢 Sector Distribution (Top 5):")
    for sector, count in all_stocks_preprocessed["sector"].value_counts().head(5).items():
        pct = count / len(all_stocks_preprocessed) * 100
        print(f"  {str(sector)[:30]:30s}: {count:5,} ({pct:5.1f}%)")

print(f"\n🎯 Recommended Next Steps:")
print(f"  1. Phase 9.4: Classification Modeling")
print(f"     - Event classification for price movements")
print(f"     - Multi-class prediction for market events")
print(f"  2. Phase 9.5: Regression Workflow")
print(f"     - Sector-optimized price target prediction")
print(f"     - Quantile regression for uncertainty bounds")
print(f"  3. Phase 9.6: Model Evaluation & Analytics")
print(f"     - Feature importance analysis")
print(f"     - Model performance by sector/region")
print(f"  4. Phase 9.7: Portfolio Optimization")
print(f"     - Mispricing score calculation")
print(f"     - Stock selection and ranking")
print(f"     - Portfolio construction and backtesting")

print(f"\n" + "=" * 80)
print(f"✓ ETL DATA EXPLORER EXECUTION COMPLETE!")
print("=" * 80)
print(f"  Timestamp: {pd.Timestamp.now()}")
print(f"  Output DataFrames:")
print(f"    • all_stocks_preprocessed  - ETL pipeline output")
print(f"    • all_stocks_features      - Phase 9.3 feature engineering output")
print(f"    • all_stocks_enhanced      - Final enhanced dataset with financial metrics")
print(f"  Total pipeline stages: 19 cells executed (Cells 1-11 + 6.5 + 10.5-10.12)")
print(f"  JSON reports: 13 files | HTML visualizations: 25 files")
print(f"  Phase 9.3 Feature Category Visualizations: Treemap, Coverage, Heatmap, Radar, Sunburst")
print(f"  Phase 9.3 Enhanced Analytics: Regional Radar, Category Boxplots, Value-Quality Bubble")
print(
    f"  Advanced Analytics: VaR, Statistical Benchmarking, Performance Attribution, Risk Heatmaps"
)
print(
    f"  Ready for: Phase 9.4 Classification, Phase 9.5 Regression, Phase 9.7 Portfolio Optimization"
)
print("=" * 80)
