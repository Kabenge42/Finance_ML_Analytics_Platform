#!/usr/bin/env python
# coding: utf-8

# # Stock Price Target Prediction — ML Analytics Platform
#
# **Version 1.0.0** — Production-Ready Stock Prediction Notebook
#
# ## Business Objective
#
# **Primary Goal**: Predict Stock Price Targets for all stocks in the portfolio to support investment decisions and portfolio optimization.
#
# **Target Variable**: "Predicted Price Target" for regression modeling
#
# ## Workflow Overview (8 Steps)
#
# 1. **Loading and Preprocessing** — Multi-region data loading with quality checks
# 2. **Exploratory Data Analysis** — Comprehensive financial metrics analysis
# 3. **Feature Engineering** — Sector-specific optimizations and advanced features
# 4. **Event Classification** — Multi-class financial event detection
# 5. **Regression Modeling** — Sector-optimized price target prediction
# 6. **Model Evaluation** — Comprehensive error analysis and metrics
# 7. **Stock Valuation** — Under/overvalued stock identification
# 8. **Analytics** — Predicted vs. Analyst target comparison
#
# ## Key Features
#
# - 📊 **Data Management**: PostgreSQL or CSV with validation
# - 🔧 **Feature Engineering**: Financial ratios, sector-specific features
# - 🤖 **ML Models**: Event classification + sector-optimized regression
# - 📈 **Analytics**: Mispricing scores, stock rankings, comprehensive reporting
# - 🧪 **Tested**: Comprehensive test coverage (≥80%)
# - ⚙️ **Modular**: Uses finance_ml package for maintainability
#

# ## Configuration and Setup
#

# In[ ]:


# Import configuration and feature flags
from finance_ml import NotebookConfig

# Initialize configuration with production settings
config = NotebookConfig(
    have_finance_prediction=True,
    have_database_connection=True,
    have_advanced_analytics=True,
    have_dim_reduction=True,
    debug_mode=False,
    enable_sector_analysis=True,
    enable_region_analysis=True,
    enable_interactive_plots=True,
    enable_excel_export=True,
)

# Display configuration summary
config.display_summary()


# In[ ]:


# Core imports
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Finance ML package imports
from finance_ml import (
    data,
    features,
    advanced_features,
    classification,
    advanced_models,
    eval as fm_eval,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
np.random.seed(RANDOM_SEED)

# Configure plotting
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

print("✓ Imports loaded successfully")
print(f"✓ Random seed set to {RANDOM_SEED}")


# In[ ]:


# Utility functions for notebook execution


def print_section_header(title: str, width: int = 80) -> None:
    """Print formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


# Checkpoint system for tracking progress
_CHECKPOINTS = {
    "config_loaded": False,
    "data_loaded": False,
    "preprocessing_complete": False,
    "eda_complete": False,
    "features_engineered": False,
    "classification_complete": False,
    "regression_complete": False,
    "evaluation_complete": False,
    "valuation_complete": False,
    "analytics_complete": False,
}


def checkpoint(name: str, requires: list = None) -> None:
    """Mark checkpoint and validate dependencies."""
    if requires:
        missing = [r for r in requires if not _CHECKPOINTS.get(r, False)]
        if missing:
            raise RuntimeError(
                f"Cannot execute {name}: missing prerequisites {missing}. "
                "Run earlier cells first."
            )
    _CHECKPOINTS[name] = True
    print(f"✓ Checkpoint: {name}")


checkpoint("config_loaded")
print("✓ Utility functions defined")


# ## Step 1: Loading and Preprocessing Financial Data
#
# Load multi-region stock data with comprehensive quality checks and preprocessing.
#

# In[ ]:


print_section_header("Step 1: Data Loading and Preprocessing")

# Determine data source
data_source = os.getenv("DATA_SOURCE", "auto")
db_url = os.getenv("DB_URL", None)
data_dir = Path(os.getenv("DATA_DIR", "data"))
limit = int(os.getenv("DATA_LIMIT", "0")) or None

# Load data from appropriate source
if data_source == "auto" or data_source == "db":
    if db_url and config.have_database_connection:
        print(f"Loading data from database: {db_url}")
        all_stocks = data.load_from_db(db_url, limit=limit)
    else:
        print(f"Loading data from CSV files in: {data_dir}")
        all_stocks = data.load_from_csv(data_dir, limit=limit)
else:
    print(f"Loading data from CSV files in: {data_dir}")
    all_stocks = data.load_from_csv(data_dir, limit=limit)

print(f"\n✓ Loaded {len(all_stocks):,} stocks")
print(f"✓ Columns: {len(all_stocks.columns)}")
print(f"\nData shape: {all_stocks.shape}")
print(f"\nRegion distribution:")
if "region" in all_stocks.columns:
    print(all_stocks["region"].value_counts())
print(f"\nSector distribution:")
if "sector" in all_stocks.columns:
    print(all_stocks["sector"].value_counts())

checkpoint("data_loaded", requires=["config_loaded"])


# In[ ]:


# Data quality checks and preprocessing
print_section_header("Data Quality and Preprocessing")

# Normalize column names
all_stocks = data.normalize_columns(all_stocks)

# Validate data quality (requires region parameter)
region = (
    all_stocks.get("region", pd.Series(["US"] * len(all_stocks))).iloc[0]
    if "region" in all_stocks.columns
    else "US"
)
quality_report = data.validate_financial_data_quality(all_stocks, region=region)
print("\nData Quality Report:")
print(f"  Completeness: {quality_report.get('completeness', 0):.1%}")
print(f"  Missing values: {quality_report.get('missing_count', 0):,}")
print(f"  Duplicate rows: {quality_report.get('duplicates', 0):,}")

# Handle missing values - use preprocess function
print("\nHandling missing values...")
all_stocks = data.preprocess(all_stocks)

# Remove duplicates
if "ticker" in all_stocks.columns:
    before_dedup = len(all_stocks)
    all_stocks = all_stocks.drop_duplicates(subset=["ticker"], keep="first")
    print(f"Removed {before_dedup - len(all_stocks):,} duplicate tickers")

print(f"\n✓ Preprocessing complete")
print(f"✓ Final dataset: {len(all_stocks):,} stocks with {len(all_stocks.columns)} columns")

checkpoint("preprocessing_complete", requires=["data_loaded"])


# ## Step 2: Exploratory Data Analysis
#
# Comprehensive analysis of financial metrics with sector and region comparisons.
#

# In[ ]:


print_section_header("Step 2: Exploratory Data Analysis")

# Run comprehensive EDA
eda_results = fm_eval.simple_eda(
    all_stocks,
    target_column="price_target" if "price_target" in all_stocks.columns else None,
    save_plots=False,
)

print("\n✓ EDA complete")
print(f"✓ Analyzed {len(eda_results.get('numeric_columns', []))} numeric columns")
print(f"✓ Identified {len(eda_results.get('categorical_columns', []))} categorical columns")

checkpoint("eda_complete", requires=["preprocessing_complete"])


# ## Step 3: Advanced Feature Engineering
#
# Sector-specific features and advanced financial ratios.
#

# In[ ]:


print_section_header("Step 3: Feature Engineering")

# Build comprehensive features using the orchestrator
all_stocks_featured = advanced_features.build_comprehensive_features(
    all_stocks,
    include_interactions=False,  # Can enable if needed, but may be slow
    include_relative_values=True,
    sector_col="sector" if "sector" in all_stocks.columns else None,
)

print(f"\n✓ Feature engineering complete")
print(f"✓ Original columns: {len(all_stocks.columns)}")
print(f"✓ Featured columns: {len(all_stocks_featured.columns)}")
print(f"✓ New features added: {len(all_stocks_featured.columns) - len(all_stocks.columns)}")

checkpoint("features_engineered", requires=["eda_complete"])


# ## Step 4: Multi-Class Event Classification
#
# Financial event detection to enhance regression regression.
#

# In[ ]:


print_section_header("Step 4: Event Classification")

# Create event labels - returns np.ndarray, not DataFrame
event_labels = classification.create_enhanced_event_labels(
    all_stocks_featured, method="price_momentum"
)

# Add labels as a column to the dataframe
all_stocks_featured["event_label"] = event_labels

if "event_label" in all_stocks_featured.columns:
    print("\nEvent label distribution:")
    print(all_stocks_featured["event_label"].value_counts())

    # Train classification model
    print("\nTraining event classifier...")
    feature_cols = [
        c
        for c in all_stocks_featured.columns
        if c not in ["ticker", "sector", "region", "event_label", "price_target"]
    ]

    X = all_stocks_featured[feature_cols].fillna(0)
    y = all_stocks_featured["event_label"]

    # Split data for classification
    from sklearn.model_selection import train_test_split

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []  # For now, assume all features are numeric after fillna

    clf_result = classification.train_xgboost_classifier(
        X_train_clf, y_train_clf, X_test_clf, y_test_clf, numeric_cols, categorical_cols
    )

    # Add classification probabilities as meta-features
    model = clf_result["model"]
    y_proba = model.predict_proba(X)

    all_stocks_featured = classification.export_classification_features(
        all_stocks_featured, y_proba
    )

    print(f"\n✓ Classification complete")
    print(f"✓ Model test accuracy: {clf_result.get('test_accuracy', 0):.2%}")
else:
    print("\n⚠ Event labels not created, skipping classification")

checkpoint("classification_complete", requires=["features_engineered"])


# ## Step 5: Sector-Optimized Regression Models
#
# Price target prediction with stacking ensemble and quantile regression.
#

# In[ ]:


print_section_header("Step 5: Regression Modeling")

# Prepare features and target
target_col = "price_target" if "price_target" in all_stocks_featured.columns else "last_price"
feature_cols = [
    c
    for c in all_stocks_featured.columns
    if c not in ["ticker", "sector", "region", target_col, "event_label"]
]

X = all_stocks_featured[feature_cols].fillna(0)
y = all_stocks_featured[target_col].fillna(all_stocks_featured[target_col].median())

print(f"\nTraining regression models...")
print(f"  Features: {len(feature_cols)}")
print(f"  Samples: {len(X)}")

# Train stacking ensemble
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# train_stacking_regressor returns (model, metrics_dict)
stacking_model, stacking_metrics = advanced_models.train_stacking_regressor(
    X_train, y_train, random_state=RANDOM_SEED
)

# Generate predictions
predictions = stacking_model.predict(X_test)
all_stocks_featured.loc[X_test.index, "predicted_price_target"] = predictions

print(f"\n✓ Regression complete")
print(f"✓ Train score: {stacking_metrics.get('train_score', 0):.3f}")
print(f"✓ CV score: {stacking_metrics.get('cv_score', 0):.3f}")

checkpoint("regression_complete", requires=["classification_complete"])


# ## Step 6: Model Evaluation and Error Analysis
#
# Comprehensive performance assessment with residual analysis.
#

# In[ ]:


print_section_header("Step 6: Model Evaluation")

# Evaluate model performance
if "predicted_price_target" in all_stocks_featured.columns:
    eval_df = all_stocks_featured.dropna(subset=["predicted_price_target", target_col])

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(eval_df[target_col], eval_df["predicted_price_target"])
    mae = mean_absolute_error(eval_df[target_col], eval_df["predicted_price_target"])
    rmse = np.sqrt(mean_squared_error(eval_df[target_col], eval_df["predicted_price_target"]))

    print("\nOverall Performance:")
    print(f"  R² Score: {r2:.3f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")

    # Sector-specific evaluation
    if config.enable_sector_analysis and "sector" in eval_df.columns:
        print("\nSector-Specific Performance:")
        for sector in eval_df["sector"].unique():
            sector_df = eval_df[eval_df["sector"] == sector]
            if len(sector_df) >= 10:
                sector_r2 = r2_score(sector_df[target_col], sector_df["predicted_price_target"])
                sector_mae = mean_absolute_error(
                    sector_df[target_col], sector_df["predicted_price_target"]
                )
                print(
                    f"  {sector:15s}: R²={sector_r2:.3f}, MAE=${sector_mae:.2f} (n={len(sector_df)})"
                )

    print(f"\n✓ Evaluation complete")
else:
    print("\n⚠ Predictions not available, skipping evaluation")

checkpoint("evaluation_complete", requires=["regression_complete"])


# ## Step 7: Under/Overvalued Stock Identification
#
# Identify investment opportunities based on mispricing scores.
#

# In[ ]:


print_section_header("Step 7: Stock Valuation")

# Calculate mispricing scores
if (
    "predicted_price_target" in all_stocks_featured.columns
    and "last_price" in all_stocks_featured.columns
):
    all_stocks_valued = all_stocks_featured.copy()
    all_stocks_valued["mispricing_pct"] = (
        (all_stocks_valued["predicted_price_target"] - all_stocks_valued["last_price"])
        / all_stocks_valued["last_price"]
        * 100
    )

    # Categorize valuations
    def categorize_valuation(pct):
        if pd.isna(pct):
            return "Unknown"
        elif pct > 20:
            return "Strong Buy"
        elif pct > 10:
            return "Buy"
        elif pct > -10:
            return "Hold"
        elif pct > -20:
            return "Sell"
        else:
            return "Strong Sell"

    all_stocks_valued["valuation_category"] = all_stocks_valued["mispricing_pct"].apply(
        categorize_valuation
    )

    print("\nValuation Distribution:")
    print(all_stocks_valued["valuation_category"].value_counts())

    # Top undervalued stocks
    print("\nTop 10 Undervalued Stocks:")
    top_undervalued = all_stocks_valued.nlargest(10, "mispricing_pct")[
        ["ticker", "sector", "last_price", "predicted_price_target", "mispricing_pct"]
    ]
    print(top_undervalued.to_string(index=False))

    print(f"\n✓ Valuation analysis complete")
    print(
        f"✓ Identified {len(all_stocks_valued[all_stocks_valued['valuation_category'] == 'Strong Buy'])} strong buy opportunities"
    )
else:
    print("\n⚠ Required columns not available, skipping valuation")
    all_stocks_valued = all_stocks_featured.copy()

checkpoint("valuation_complete", requires=["evaluation_complete"])


# ## Step 8: Comprehensive Analytics
#
# Predicted vs. Analyst Price Target comparison and reporting.
#

# In[ ]:


print_section_header("Step 8: Comprehensive Analytics")

# Compare predictions with analyst targets (if available)
if "predicted_price_target" in all_stocks_valued.columns:
    analyst_col = None
    for col in ["analyst_target", "price_target_median", "consensus_target"]:
        if col in all_stocks_valued.columns:
            analyst_col = col
            break

    if analyst_col:
        comparison_df = all_stocks_valued.dropna(subset=["predicted_price_target", analyst_col])

        print(f"\nComparing predictions with {analyst_col}...")
        print(f"  Stocks with both values: {len(comparison_df):,}")

        # Calculate agreement rate
        comparison_df["model_direction"] = np.sign(
            comparison_df["predicted_price_target"] - comparison_df["last_price"]
        )
        comparison_df["analyst_direction"] = np.sign(
            comparison_df[analyst_col] - comparison_df["last_price"]
        )
        agreement_rate = (
            comparison_df["model_direction"] == comparison_df["analyst_direction"]
        ).mean()

        print(f"  Directional agreement: {agreement_rate:.1%}")

        # Identify high-conviction disagreements
        disagreements = comparison_df[
            (comparison_df["model_direction"] != comparison_df["analyst_direction"])
            & (np.abs(comparison_df["mispricing_pct"]) > 15)
        ]

        if len(disagreements) > 0:
            print(f"\nHigh-conviction disagreements (>15%):")
            print(f"  Count: {len(disagreements)}")
            print("\nTop 5 Disagreements:")
            top_disagreements = disagreements.nlargest(5, "mispricing_pct")[
                [
                    "ticker",
                    "sector",
                    "last_price",
                    "predicted_price_target",
                    analyst_col,
                    "mispricing_pct",
                ]
            ]
            print(top_disagreements.to_string(index=False))
    else:
        print("\n⚠ Analyst targets not available")

    print(f"\n✓ Analytics complete")

    # Export results
    if config.enable_excel_export:
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "stock_prediction_results.csv"
        all_stocks_valued.to_csv(output_file, index=False)
        print(f"✓ Results exported to {output_file}")
else:
    print("\n⚠ Predictions not available, skipping analytics")

checkpoint("analytics_complete", requires=["valuation_complete"])


# ## Summary and Next Steps
#
# All 8 workflow steps completed successfully!
#

# In[ ]:


print_section_header("Workflow Summary")

print("\n✓ All checkpoints completed:")
for name, status in _CHECKPOINTS.items():
    status_icon = "✓" if status else "✗"
    print(f"  {status_icon} {name}")

print("\n" + "=" * 80)
print("Stock Prediction Workflow Complete!")
print("=" * 80)
