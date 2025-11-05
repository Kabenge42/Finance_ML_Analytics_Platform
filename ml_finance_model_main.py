# %% md
# # Finance ML Analytics Platform — v8_3
#
# **Version 0.3.0** — Now using modular `finance_ml` package
#
# ## What's New
#
# - All functions are now imported from the `finance_ml` package
# - No need to define functions inline — they're maintained in the package modules
# - Configuration management with `FinanceMLConfig`
# - Better code organization and testability
# - Feature flags for optional functionality control
#
# ## Modules
#
# - `finance_ml.data`: Data loading, normalization, validation
# - `finance_ml.features`: Feature engineering
# - `finance_ml.models`: Classification, regression, ensembles
# - `finance_ml.eval`: Analytics, visualizations, reporting
# - `finance_ml.config`: Configuration management
# - `finance_ml.cli`: Command-line interface
#
# ## Usage
#
# This notebook demonstrates the ML workflow:
# 1. Load and validate data
# 2. Exploratory data analysis
# 3. Feature engineering
# 4. Model training (classification and regression)
# 5. Evaluation and analytics
# %% md
# ## Configuration and Feature Flags
# %%
##%% Feature Availability Flags
# Configuration for optional notebook functionality
# Set these flags to enable/disable features based on your requirements

# Prediction and modeling capabilities
HAVE_FINANCE_PREDICTION = True  # Enable financial prediction features

# Database connectivity
HAVE_DATABASE_CONNECTION = False  # Enable PostgreSQL database connections

# Advanced analytics and visualizations
HAVE_ADVANCED_ANALYTICS = True  # Enable advanced analytics features

# Dimensionality reduction visualizations
HAVE_DIM_REDUCTION = False  # Enable dimensionality reduction visualizations

# Debug and development features
DEBUG_MODE = False  # Enable debug output and additional logging

# Optional feature toggles
ENABLE_SECTOR_ANALYSIS = True  # Enable sector-based analysis
ENABLE_REGION_ANALYSIS = True  # Enable region-based analysis
ENABLE_INTERACTIVE_PLOTS = True  # Enable interactive Plotly visualizations
ENABLE_EXCEL_EXPORT = True  # Enable Excel export functionality

print("=" * 80)
print("FEATURE FLAGS CONFIGURATION")
print("=" * 80)
print(f"Financial Prediction:        {HAVE_FINANCE_PREDICTION}")
print(f"Database Connection:         {HAVE_DATABASE_CONNECTION}")
print(f"Advanced Analytics:          {HAVE_ADVANCED_ANALYTICS}")
print(f"Dimensionality Reduction:    {HAVE_DIM_REDUCTION}")
print(f"Debug Mode:                  {DEBUG_MODE}")
print(f"Sector Analysis:             {ENABLE_SECTOR_ANALYSIS}")
print(f"Region Analysis:             {ENABLE_REGION_ANALYSIS}")
print(f"Interactive Plots:           {ENABLE_INTERACTIVE_PLOTS}")
print(f"Excel Export:                {ENABLE_EXCEL_EXPORT}")
print("=" * 80)
# %%
# Finance ML Analytics Platform — Notebook (v0.3.0)
# This notebook now uses the modular finance_ml package

import warnings

warnings.filterwarnings("ignore")

# Data science libraries
import numpy as np
import pandas as pd

# Import all functions from finance_ml package
from finance_ml import (
    # Version
    __version__,
    # Configuration
    load_config,
    # Utilities
    setup_logging,
    # Data loading and validation
    preprocess,
    # Notebook utilities
    display_config_summary,
    load_stock_data,
    display_data_summary,
    # Feature engineering
    build_features_and_target,
    # Modeling
    create_event_labels,
    train_event_classifier,
    train_and_evaluate_regression,
    # Evaluation and analytics
    calculate_mispricing_score,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    create_sector_heatmap,
    create_interactive_prediction_plot,
    # Week 1 Enhancements - Data Quality & Monitoring
    validate_financial_data_quality,
    sanitize_dataframe_with_logging,
    monitor_ensemble_training,
    perform_early_pipeline_validation,
)

# Setup logging
import logging

setup_logging()
logger = logging.getLogger(__name__)

print(f"Finance ML Analytics Platform v{__version__}")
print("All functions imported from finance_ml package")

# %% md
# ## Configuration
#
# Load configuration from environment variables or config files.
#
# %%
# Load configuration
config = load_config()
display_config_summary(config)

# Create output directory structure
config.create_output_structure()
print(f"\n✓ Output directory structure created")
print(f"  Main subdirectories: analytics, models, eda")
print(
    f"  EDA subdirectories: eda_with_importance, eda_with_multivariate, enhanced_eda, financial_data_quality_reports"
)

# %% md
# ## Sample Data Generator
#
# Create sample financial dataset for demonstration when real data is unavailable.
#
# Note: The generator is now provided by the package as
# `finance_ml.create_sample_financial_dataset` — no inline definition needed here.
#
# %% md
# ## Data Loading
#
# Load stock data from configured data source (database or CSV files) with automatic fallback to sample data.
#
# %%
# Load stock data using package strategy helpers
all_stocks = load_stock_data(config)
if all_stocks is None or len(all_stocks) == 0:
    raise ValueError("Failed to load any stock data")

display_data_summary(all_stocks)

# %% md
# ## Data Validation and Quality Checks
#
# Validate schema and check data quality using finance_ml package functions.
#
# %%
# Unified validation reporting with error handling
try:
    from finance_ml.data import validate_schema, check_missing_values

    # Schema validation
    try:
        is_valid, errors = validate_schema(all_stocks, require_target=False)
        if is_valid:
            print("✓ Schema validation passed")
        else:
            print(f"⚠ Schema validation failed:")
            for error in errors:
                print(f"  - {error}")
    except Exception as e:
        print(f"⚠ Schema validation skipped: {e}")

    # Missing values check
    try:
        missing_report = all_stocks.isnull().sum()
        missing_pct = (missing_report / len(all_stocks) * 100).round(2)
        missing_df = pd.DataFrame(
            {
                "Missing Count": missing_report[missing_report > 0],
                "Missing %": missing_pct[missing_report > 0],
            }
        ).sort_values("Missing Count", ascending=False)

        if len(missing_df) > 0:
            print("\n📊 Missing Values Report:")
            print(missing_df.head(10).to_string())
        else:
            print("✓ No missing values detected")
    except Exception as e:
        print(f"⚠ Missing value check failed: {e}")

except Exception as e:
    logger.error(f"Validation failed: {e}")
    print(f"⚠ Validation checks incomplete")
# %% md
# ## Exploratory Data Analysis
#
# Perform EDA using the simple_eda function.
#
# %%
# Unified EDA display with proper output directory
try:
    from pathlib import Path
    from finance_ml.data import simple_eda

    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Use enhanced EDA directory from config
    simple_eda(all_stocks, out_dir=config.enhanced_eda_dir)
    print(f"✓ EDA completed - outputs saved to {config.enhanced_eda_dir}")

except Exception as e:
    logger.error(f"EDA failed: {e}")
    print(f"⚠ EDA failed: {e}")
# %% md
# ## Data Preprocessing
#
# Preprocess and normalize the data using finance_ml package.
#
# %%
# Preprocess data
try:
    all_stocks_processed = preprocess(all_stocks)
    logger.info(f"Data preprocessed: {all_stocks_processed.shape}")
    print(f"\n✓ Data preprocessed successfully")
    print(f"  Shape after preprocessing: {all_stocks_processed.shape}")

    # Log presence of new price-related columns (Nov 2025 schema update)
    new_price_cols = [
        "price_chg_pct_1m",
        "price_chg_pct_3m",
        "one_day_pct",
        "price_5d_ago",
        "price_1w_ago",
        "price_1m_ago",
        "price_3m_ago",
        "price_6m_ago",
        "price_1y_ago",
        "price_3y_ago",
        "price_5y_ago",
        "price_qtd_ago",
    ]
    detected = [c for c in new_price_cols if c in all_stocks_processed.columns]
    missing = [c for c in new_price_cols if c not in all_stocks_processed.columns]
    print(f"  New price columns detected: {len(detected)}/{len(new_price_cols)}")
    if detected:
        print(f"    Present: {detected[:8]}{' ...' if len(detected) > 8 else ''}")
    if missing:
        print(f"    Missing: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
except Exception as e:
    logger.error(f"Preprocessing failed: {e}")
    print(f"✗ Preprocessing failed: {e}")
    all_stocks_processed = all_stocks.copy()

# %% md
# ## Feature Engineering
#
# Build features using the complete feature pipeline from finance_ml package.
#
# %%
# Build features and target
try:
    X, y, numeric_features, categorical_features = build_features_and_target(all_stocks_processed)
    logger.info(f"Features built: {X.shape}, Target: {y.shape if y is not None else 'None'}")
    print(f"\n✓ Features engineered successfully")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target shape: {y.shape if y is not None else 'None'}")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  First 10 numeric features: {numeric_features[:10]}")
    if categorical_features:
        print(f"  Categorical features: {categorical_features}")
except Exception as e:
    logger.error(f"Feature engineering failed: {e}")
    print(f"✗ Feature engineering failed: {e}")
    raise
# %% md
# ## Model Training - Classification
#
# Train event classifier using finance_ml package.
#
# %%
# Create event labels for classification
try:
    event_labels = create_event_labels(all_stocks_processed)
    print(f"\n✓ Event labels created")
    print(f"  Label distribution: {pd.Series(event_labels).value_counts().to_dict()}")

    # Remove duplicate columns before training
    df_for_classifier = all_stocks_processed.copy()
    duplicate_cols = df_for_classifier.columns[df_for_classifier.columns.duplicated()].unique()

    if len(duplicate_cols) > 0:
        print(f"  Removing duplicate columns: {list(duplicate_cols)}")
        # Keep only first occurrence of each column
        df_for_classifier = df_for_classifier.loc[:, ~df_for_classifier.columns.duplicated()]

    # Train event classifier using the cleaned DataFrame
    classifier_results = train_event_classifier(df_for_classifier, event_labels)
    print(f"\n✓ Event classifier trained")
    print(f"  Accuracy: {classifier_results.get('accuracy', 0):.4f}")
    print(f"  F1 Score (macro): {classifier_results.get('f1_macro', 0):.4f}")

except Exception as e:
    logger.error(f"Classification training failed: {e}")
    print(f"✗ Classification training failed: {e}")
# %% md
# ## Model Training - Regression
#
# Train regression models using finance_ml package.
#
# %%
# Train baseline regression model
try:
    # Use the proper training function with required parameters
    from pathlib import Path

    # Use models directory from config
    regression_results = train_and_evaluate_regression(
        all_stocks_processed,
        out_dir=config.models_output_dir,
        n_jobs=config.n_jobs if hasattr(config, "n_jobs") else -1,
    )

    if regression_results:
        print(f"\n✓ Regression model trained")
        print(f"  MAE: {regression_results.get('mae', 0):.4f}")
        print(f"  RMSE: {regression_results.get('rmse', 0):.4f}")
        print(f"  R²: {regression_results.get('r2', 0):.4f}")

        # Store predictions in dataframe for later use
        if "predictions" in regression_results:
            pred_df = regression_results["predictions"]
            all_stocks_processed.loc[pred_df.index, "predicted_target"] = pred_df["y_pred"].values
    else:
        print("⚠ Regression training skipped (insufficient data or dry run)")
except Exception as e:
    logger.error(f"Regression training failed: {e}")
    print(f"✗ Regression training failed: {e}")
# %% md
# ## Stock Valuation Analysis
#
# Calculate mispricing scores and rank stocks using finance_ml package.
#
# %%
# Calculate mispricing scores
try:
    # Get predictions from regression model
    if "predictions" in regression_results and regression_results["predictions"] is not None:
        predictions = regression_results["predictions"]

        # Properly align predictions with dataframe using index
        if isinstance(predictions, pd.DataFrame):
            # predictions is a DataFrame with y_pred column
            pred_series = predictions["y_pred"]
            all_stocks_processed.loc[pred_series.index, "predicted_target"] = pred_series.values
        elif isinstance(predictions, pd.Series):
            all_stocks_processed.loc[predictions.index, "predicted_target"] = predictions.values
        else:
            print("⚠ Unexpected prediction format")
            raise ValueError("Predictions must be DataFrame or Series")

        # Calculate mispricing only for rows with predictions
        mask = all_stocks_processed["predicted_target"].notna()
        df_with_pred = all_stocks_processed[mask].copy()

        mispricing = calculate_mispricing_score(df_with_pred)
        all_stocks_processed.loc[mask, "mispricing_score"] = mispricing

        print(f"\n✓ Mispricing scores calculated for {mask.sum()} stocks")
        print(f"  Mean mispricing: {mispricing.mean():.4f}")
        print(f"  Std mispricing: {mispricing.std():.4f}")

        # Rank undervalued stocks (only from rows with mispricing scores)
        df_scored = all_stocks_processed[all_stocks_processed["mispricing_score"].notna()].copy()

        if len(df_scored) >= 10:
            undervalued = rank_undervalued_stocks(df_scored, top_n=10)
            print(f"\nTop 10 Undervalued Stocks:")
            display_cols = [
                c
                for c in ["ticker", "name", "sector", "mispricing_score"]
                if c in undervalued.columns
            ]
            print(undervalued[display_cols].to_string())

            # Rank overvalued stocks
            overvalued = rank_overvalued_stocks(df_scored, top_n=10)
            print(f"\nTop 10 Overvalued Stocks:")
            print(overvalued[display_cols].to_string())
        else:
            print(f"⚠ Insufficient scored stocks ({len(df_scored)}) for ranking")
    else:
        print("⚠ No predictions available from regression model")

except Exception as e:
    logger.error(f"Valuation analysis failed: {e}")
    print(f"✗ Valuation analysis failed: {e}")
    import traceback

    traceback.print_exc()
# %% md
# ## Advanced Preprocessing Pipeline Demo
#
# Demonstrate the proper use of separate transformers for numeric vs categorical features using the returned feature lists.
# %%
# Demonstrate proper preprocessing pipeline with separate transformers
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    if y is not None and len(X) > 0:
        print("\n" + "=" * 80)
        print("ADVANCED PREPROCESSING PIPELINE DEMONSTRATION")
        print("=" * 80)

        # Show feature type separation
        print(f"\n📊 Feature Type Analysis:")
        print(f"  Total features: {len(numeric_features) + len(categorical_features)}")
        print(f"  Numeric features: {len(numeric_features)}")
        print(f"  Categorical features: {len(categorical_features)}")

        # Build preprocessing pipeline with separate transformers
        print(f"\n🔧 Building preprocessing pipeline with separate transformers...")

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(with_mean=False), numeric_features),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ],
            remainder="drop",
        )

        # Create full pipeline with regressor
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("regressor", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ]
        )

        # Prepare data (remove NaN targets)
        mask = ~y.isna()
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]

        if len(X_clean) >= 20:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=0.2, random_state=42
            )

            print(f"\n📈 Training pipeline...")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Test samples: {len(X_test)}")

            # Fit pipeline
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)

            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"\n✓ Pipeline training completed")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")

            # Show transformed feature dimensions
            X_transformed = preprocessor.transform(X_test)
            print(f"\n🔄 Transformed feature space:")
            print(f"  Original features: {X_test.shape[1]}")
            print(f"  Transformed features: {X_transformed.shape[1]}")
            print(
                f"  (Numeric: {len(numeric_features)}, One-Hot Encoded Categorical: {X_transformed.shape[1] - len(numeric_features)})"
            )

            print(f"\n✓ Preprocessing pipeline demonstration complete")
        else:
            print(f"\n⚠ Insufficient clean data ({len(X_clean)} samples) for pipeline demo")
    else:
        print("\n⚠ No target variable or features available for pipeline demo")

except Exception as e:
    logger.error(f"Pipeline demo failed: {e}")
    print(f"✗ Pipeline demo failed: {e}")
    import traceback

    traceback.print_exc()
# %% md
# ## Visualization
#
# Create visualizations using finance_ml package functions.
# %%
# Create sector heatmap
try:
    if (
        "sector" in all_stocks_processed.columns
        and "mispricing_score" in all_stocks_processed.columns
    ):
        df_for_viz = all_stocks_processed[all_stocks_processed["mispricing_score"].notna()].copy()
        if len(df_for_viz) > 0:
            fig = create_sector_heatmap(df_for_viz)
            print("\n✓ Sector heatmap created")
        else:
            print("\n⚠ No data with mispricing scores for sector heatmap")
except Exception as e:
    logger.error(f"Sector heatmap failed: {e}")
    print(f"✗ Sector heatmap creation failed: {e}")

# Create interactive prediction plot
try:
    # Remove duplicate columns before plotting
    df_for_plot = all_stocks_processed.copy()
    df_for_plot = df_for_plot.loc[:, ~df_for_plot.columns.duplicated()]

    if "predicted_target" in df_for_plot.columns and "price_target" in df_for_plot.columns:
        # Filter to rows with both values
        mask = df_for_plot["predicted_target"].notna() & df_for_plot["price_target"].notna()
        df_for_plot = df_for_plot[mask]

        if len(df_for_plot) > 0:
            fig = create_interactive_prediction_plot(df_for_plot)
            print("✓ Interactive prediction plot created")
        else:
            print("⚠ No data with both predicted and actual targets for plot")
    else:
        print("⚠ Missing required columns for prediction plot")

except Exception as e:
    logger.error(f"Prediction plot failed: {e}")
    print(f"✗ Prediction plot creation failed: {e}")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
# %%
# Visualize dimensionality reduction results
try:
    _transformed = locals().get("transformed_data", None)
    _dim_res = locals().get("dim_reduction_results", None)
    _viz = locals().get("visualizer", None)
    _y_cls = locals().get("y_class_train", None)
    if HAVE_DIM_REDUCTION and isinstance(_transformed, dict) and len(_transformed) > 0:
        print("\n" + "=" * 80)
        print("DIMENSIONALITY REDUCTION VISUALIZATIONS")
        print("=" * 80)

        # Plot explained variance for PCA if available
        if isinstance(_dim_res, dict) and "PCA" in _dim_res:
            pca_entry = _dim_res.get("PCA", {})
            pca_method = pca_entry.get("fitted_method") if isinstance(pca_entry, dict) else None
            if (
                pca_method is not None
                and hasattr(pca_method, "explained_variance_ratio_")
                and _viz is not None
                and hasattr(_viz, "plot_explained_variance")
            ):
                _viz.plot_explained_variance(pca_method, "PCA Explained Variance Analysis")

        # Create 2D visualizations for methods that support it
        visualization_methods = {}
        for method_name, X_transformed in _transformed.items():
            try:
                if hasattr(X_transformed, "shape") and X_transformed.shape[1] >= 2:
                    visualization_methods[method_name] = X_transformed[:, :2]
            except Exception:
                continue

        # Create comparison plots if we have 2D data and a visualizer
        if (
            visualization_methods
            and _viz is not None
            and _y_cls is not None
            and hasattr(_viz, "plot_component_comparison")
        ):
            _viz.plot_component_comparison(visualization_methods, _y_cls)

        print("Visualization plots generated successfully")
    else:
        print("Dimensionality reduction disabled or not available; skipping.")
except Exception as e:
    logger.error(f"Error in visualization: {e}")
    print(f"Error in visualization: {e}")

# %% md
# ## Key Improvements: Proper Preprocessing Pipelines
#
# This notebook now uses the returned `numeric_features` and `categorical_features` lists from `build_features_and_target()` to create proper preprocessing pipelines with:
#
# 1. **Separate Transformers**:
#    - `StandardScaler` for numeric features
#    - `OneHotEncoder` for categorical features
#
# 2. **Benefits**:
#    - Proper handling of different feature types
#    - Prevents data leakage by fitting transformers only on training data
#    - Automatically handles unknown categories in test data
#    - Cleaner, more maintainable code
#
# 3. **Implementation**:
#    - `build_features_and_target()` returns 4 values: `X, y, numeric_features, categorical_features`
#    - These lists are used in `ColumnTransformer` for proper preprocessing
#    - All model training functions now use this pattern
#
# See the "Advanced Preprocessing Pipeline Demonstration" section above for a working example.
# %%
# Week 1 Enhancement Validation Test Suite
print("=" * 80)
print("WEEK 1 ENHANCEMENT VALIDATION TEST SUITE")
print("=" * 80)


def test_data_quality_validation():
    """Test enhanced data quality validation functionality"""
    print("\n🧪 Testing Data Quality Validation...")

    try:
        # Create test dataset with known issues
        test_data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.inf, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, np.nan, 50.0],
                "feature3": [100.0, -np.inf, 300.0, 400.0, 500.0],
                "Sector": ["Tech", "Finance", "Healthcare", "Tech", "Finance"],
            }
        )

        # Test data quality validation function
        quality_results = validate_financial_data_quality(test_data, "test_region")

        # Validate expected results
        assert quality_results["total_rows"] == 5, "Row count validation failed"
        assert quality_results["infinity_values"] == 2, "Infinity detection failed"  # inf and -inf
        assert quality_results["null_values"] == 1, "Null detection failed"  # 1 NaN
        assert "data_quality_score" in quality_results, "Quality score missing"

        print("✅ Data quality validation tests passed")
        return True

    except Exception as e:
        print(f"❌ Data quality validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sanitization_monitoring():
    """Test comprehensive sanitization logging functionality"""
    print("\n🧪 Testing Sanitization Monitoring...")

    try:
        # Create test dataset with various data quality issues
        test_data = pd.DataFrame(
            {
                "numeric1": [1.0, 2.0, np.inf, 4.0, 5.0, -np.inf],
                "numeric2": [10.0, 20.0, 1000000.0, np.nan, 50.0, 60.0],  # extreme value
                "numeric3": [100.0, 200.0, 300.0, np.nan, np.nan, 600.0],  # multiple NaN
            }
        )

        # Capture original issues for validation
        original_inf_count = np.isinf(test_data.select_dtypes(include=[np.number])).sum().sum()
        original_nan_count = test_data.isnull().sum().sum()

        # Apply sanitization
        cleaned_data = sanitize_dataframe_with_logging(test_data)

        # Validate sanitization results
        post_inf_count = np.isinf(cleaned_data.select_dtypes(include=[np.number])).sum().sum()
        post_nan_count = cleaned_data.isnull().sum().sum()

        assert post_inf_count == 0, "Infinity values not properly removed"
        assert post_nan_count == 0, "NaN values not properly filled"
        assert cleaned_data.shape == test_data.shape, "Data shape changed unexpectedly"

        print("✅ Sanitization monitoring tests passed")
        return True

    except Exception as e:
        print(f"❌ Sanitization monitoring test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_monitoring():
    """Test enhanced training monitoring functionality"""
    print("\n🧪 Testing Training Monitoring...")

    try:
        # Create simple test model and data
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        # Generate test data
        X_test = pd.DataFrame(
            {
                "feature1": np.random.random(100),
                "feature2": np.random.random(100),
                "feature3": np.random.random(100),
            }
        )
        y_test = X_test["feature1"] * 2 + X_test["feature2"] * 3 + np.random.random(100) * 0.1

        X_train, X_val, y_train, y_val = train_test_split(
            X_test, y_test, test_size=0.3, random_state=42
        )

        # Create test model
        test_model = LinearRegression()

        # Apply monitoring
        monitoring_results, y_train_pred, y_val_pred = monitor_ensemble_training(
            test_model, X_train, y_train, X_val, y_val, "Test_Model"
        )

        # Validate monitoring results structure
        required_keys = ["model_name", "timestamp", "training_time_seconds", "performance_metrics"]
        for key in required_keys:
            assert key in monitoring_results, f"Missing key in monitoring results: {key}"

        # Validate performance metrics
        perf_metrics = monitoring_results["performance_metrics"]
        required_perf_keys = ["train_r2", "test_r2", "train_mse", "test_mse"]
        for key in required_perf_keys:
            assert key in perf_metrics, f"Missing performance metric: {key}"

        print("✅ Training monitoring tests passed")
        return True

    except Exception as e:
        print(f"❌ Training monitoring test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_pipeline_validation():
    """Test early pipeline validation functionality"""
    print("\n🧪 Testing Pipeline Validation...")

    try:
        # Create test dataset that mimics financial data
        test_financial_data = pd.DataFrame(
            {
                "p_e_ntm": [15.0, 25.0, -5.0, 1500.0, 20.0],  # negative and extreme values
                "market_cap": [1000000, 2000000, -500000, 5000000, 1500000],  # negative value
                "last_price": [50.0, 100.0, 0.0, 200.0, 75.0],  # zero price
                "price_target": [60.0, 120.0, 50.0, 180.0, 80.0],
                "sector": ["Technology", "Healthcare", "Finance", "Technology", "Energy"],
                "ticker": ["AAPL", "JNJ", "JPM", "MSFT", "XOM"],
                "next_earnings_days": [30, 45, -10, 60, np.nan],  # past date and NaN
            }
        )

        # Test pipeline validation function
        validation_results = perform_early_pipeline_validation(test_financial_data)

        # Validate results structure
        required_keys = [
            "validation_score",
            "total_checks",
            "passed_checks",
            "warnings",
            "recommendations",
        ]
        for key in required_keys:
            assert key in validation_results, f"Missing key in validation results: {key}"

        # Check that warnings were detected for problematic data
        assert (
            len(validation_results["warnings"]) > 0
        ), "Expected warnings for problematic test data"
        assert "validation_score" in validation_results, "Validation score missing"
        assert 0 <= validation_results["validation_score"] <= 1, "Validation score out of range"

        print("✅ Pipeline validation tests passed")
        return True

    except Exception as e:
        print(f"❌ Pipeline validation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# Execute validation test suite
print("\nExecuting Week 1 Enhancement Validation Tests...")
print("-" * 60)

test_results = {
    "data_quality_validation": test_data_quality_validation(),
    "sanitization_monitoring": test_sanitization_monitoring(),
    "training_monitoring": test_training_monitoring(),
    "pipeline_validation": test_pipeline_validation(),
}

# Summary results
print("\n" + "=" * 60)
print("VALIDATION TEST RESULTS SUMMARY")
print("=" * 60)

passed_tests = sum(test_results.values())
total_tests = len(test_results)
success_rate = (passed_tests / total_tests) * 100

for test_name, result in test_results.items():
    status = "✅ PASSED" if result else "❌ FAILED"
    print(f"{test_name.replace('_', ' ').title():<30}: {status}")

print("-" * 60)
print(f"Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")

if success_rate >= 75:
    print("🎉 Week 1 enhancements validation SUCCESSFUL!")
    print("✅ Ready for CI/CD integration")
else:
    print("⚠️ Some Week 1 enhancements need attention")
    print("🔧 Review failed tests before proceeding")

print("=" * 60)

# %% md
# ## Per-Sector Regression, Quantile Bands, Stacking, and Excel Export (Implemented from Examine_theml_finance_model_v8_2.ipynb_f.md)
#
# This section implements key enhancements from the planning document:
# - Per-sector regression metrics
# - Quantile regression by sector for uncertainty bands
# - Stacking ensemble by sector
# - Excel export with predictions and summaries
#
# All steps are guarded to keep the notebook robust on small demo datasets.
# %%
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

try:
    from finance_ml import (
        train_and_evaluate_regression_by_sector,
        train_quantile_regression_by_sector,
        predict_quantile_regression,
        train_stacking_ensemble_by_sector,
        export_predictions_to_excel,
    )

    HAVE_ENHANCED_MODELS = True
except Exception as e:
    print(f"⚠ Enhanced modeling imports unavailable: {e}")
    HAVE_ENHANCED_MODELS = False

# Remove duplicate columns first
df_enhanced = all_stocks_processed.loc[:, ~all_stocks_processed.columns.duplicated()].copy()

# --- Per-sector regression metrics ---
if HAVE_ENHANCED_MODELS:
    try:
        print("\n" + "=" * 80)
        print("PER-SECTOR REGRESSION METRICS")
        print("=" * 80)
        # Use models directory from config for sector models
        sector_metrics = train_and_evaluate_regression_by_sector(
            df_enhanced, config.models_output_dir
        )
        display_cols = [
            c
            for c in ["sector", "n_train", "n_test", "mae", "rmse", "r2"]
            if c in sector_metrics.columns
        ]
        if len(sector_metrics) > 0:
            print(sector_metrics[display_cols].to_string(index=False))
        else:
            print("(No sectors with sufficient data)")
    except Exception as e:
        print(f"⚠ Per-sector regression metrics step skipped: {e}")

# --- Quantile regression by sector (uncertainty bands) ---
quantile_models_by_sector = {}
if HAVE_ENHANCED_MODELS:
    try:
        # Determine target column
        target_col = None
        for cand in ["price_target", "price_target_median"]:
            if cand in df_enhanced.columns:
                target_col = cand
                break

        if target_col is None:
            raise ValueError("No target column found")

        # Select numeric feature columns only
        numeric_cols = [
            c for c in df_enhanced.columns if pd.api.types.is_numeric_dtype(df_enhanced[c])
        ]
        blacklist = {target_col, "predicted_target", "mispricing_score"}
        feature_cols = [c for c in numeric_cols if c not in blacklist]

        if len(feature_cols) < 3:
            raise ValueError(
                f"Insufficient numeric features ({len(feature_cols)}) for quantile regression"
            )

        # Check we have enough clean data
        required_cols = feature_cols + [target_col]
        if "sector" in df_enhanced.columns:
            required_cols.append("sector")

        df_clean = df_enhanced[required_cols].dropna()

        if len(df_clean) < 50:
            raise ValueError(
                f"Insufficient clean data ({len(df_clean)} rows) for quantile regression"
            )

        print("\n" + "=" * 80)
        print("QUANTILE REGRESSION BY SECTOR (q10/q50/q90)")
        print("=" * 80)

        quantile_models_by_sector = train_quantile_regression_by_sector(
            df_enhanced, feature_cols, target_col, quantiles=[0.1, 0.5, 0.9]
        )

        # Generate quantile predictions per sector
        prediction_count = 0
        for sector, model in quantile_models_by_sector.items():
            sec_mask = (
                (df_enhanced.get("sector") == sector)
                if "sector" in df_enhanced.columns
                else pd.Series(False, index=df_enhanced.index)
            )
            if sec_mask.sum() == 0:
                continue

            X_sec = df_enhanced.loc[sec_mask, feature_cols].dropna()
            if len(X_sec) == 0:
                continue

            preds_df = predict_quantile_regression(model, X_sec, quantiles=[0.1, 0.5, 0.9])

            # Assign predictions back to main frame
            for col in preds_df.columns:
                df_enhanced.loc[X_sec.index, col] = preds_df[col].values
            prediction_count += len(X_sec)

        if prediction_count > 0:
            print(
                f"✓ Quantile bands added for {prediction_count} stocks across {len(quantile_models_by_sector)} sectors"
            )
        else:
            print("⚠ No quantile predictions generated")

    except Exception as e:
        print(f"⚠ Quantile regression step skipped: {e}")

# --- Stacking ensemble by sector ---
stacking_models_by_sector = {}
if HAVE_ENHANCED_MODELS:
    try:
        if "feature_cols" in locals() and "target_col" in locals():
            cols_needed = [
                c for c in (feature_cols + [target_col, "sector"]) if c in df_enhanced.columns
            ]
            df_stack = df_enhanced[cols_needed].dropna()

            if len(df_stack) >= 100:  # Increased threshold for stability
                print("\n" + "=" * 80)
                print("STACKING ENSEMBLE BY SECTOR")
                print("=" * 80)
                stacking_models_by_sector = train_stacking_ensemble_by_sector(
                    df_stack, feature_cols, target_col
                )
                print(f"✓ Trained stacking ensembles for {len(stacking_models_by_sector)} sectors")
            else:
                print(
                    f"⚠ Skipping stacking ensemble — insufficient clean rows: {len(df_stack)} (need 100+)"
                )
        else:
            print("⚠ Skipping stacking ensemble — features/target not available")
    except Exception as e:
        print(f"⚠ Stacking ensemble step skipped: {e}")

# --- Excel export ---
if HAVE_ENHANCED_MODELS:
    try:
        if (
            "mispricing_score" in df_enhanced.columns
            and df_enhanced["mispricing_score"].notna().sum() > 0
        ):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use analytics directory from config for Excel reports
            excel_path = config.analytics_dir / f"Stock_Prediction_Analysis_Report_{ts}.xlsx"
            export_predictions_to_excel(df_enhanced, excel_path, include_summary=True)
            print(f"\n✓ Exported predictions and summaries to Excel: {excel_path}")
        else:
            print(
                "⚠ Excel export skipped — mispricing_score not found; run valuation analysis first"
            )
    except ImportError as e:
        print(f"⚠ Excel export skipped (engine missing): {e}")
    except Exception as e:
        print(f"⚠ Excel export failed: {e}")

# Update main dataframe with enhancements
all_stocks_processed = df_enhanced


# =============================================================================
# Script CLI Entry Point (lightweight)
# =============================================================================
def _cli_main() -> int:
    """Minimal CLI entry point mirroring the legacy v8_2 script behavior.

    Notes:
    - This function provides a lightweight batch runner so the file can be
      executed directly: python ml_finance_model_main.py --data-source auto
    - The primary, recommended CLI remains the console script: finance-ml
    - For robust, interactive exploration, open ml_finance_model_main.ipynb
    """
    import argparse
    import logging as _logging
    from pathlib import Path as _Path

    # Defer imports to avoid interfering with notebook execution at import time
    from finance_ml import setup_logging as _setup_logging
    from finance_ml import get_env as _get_env
    from finance_ml import load_from_csv as _load_from_csv
    from finance_ml import load_from_db as _load_from_db
    from finance_ml import preprocess as _preprocess
    from finance_ml import simple_eda as _simple_eda
    from finance_ml import train_and_evaluate_regression as _train_reg

    _setup_logging()

    parser = argparse.ArgumentParser(
            description="Finance ML Analytics Platform — main script (lightweight)")
    parser.add_argument("--data-source", choices=["auto", "csv", "db"], default="auto")
    parser.add_argument("--db-url", default=None, help="SQLAlchemy URL; or use DB_URL env var")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()

    data_dir = _Path(_get_env("DATA_DIR", default="data"))
    out_dir = _Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Decide data source
    db_url = args.db_url or _get_env("DB_URL")
    source = args.data_source
    if source == "auto":
        try:
            # Only attempt DB if URL provided
            from sqlalchemy import create_engine as _ce  # type: ignore
            _ = _ce  # silence linter
            source = "db" if db_url else "csv"
        except Exception:
            source = "csv"

    _logging.info(
            "Configuration: source=%s, limit=%s, out_dir=%s", source, args.limit, out_dir,
            )

    # Load data
    if source == "db":
        if not db_url:
            _logging.error(
                    "--data-source db requested but DB URL is missing. Provide --db-url or DB_URL env var.",
                    )
            return 2
        df_raw = _load_from_db(db_url, limit=args.limit)
    else:
        df_raw = _load_from_csv(data_dir, limit=args.limit)

    # Preprocess, EDA, and baseline regression
    df = _preprocess(df_raw)
    _simple_eda(df, out_dir)

    _ = _train_reg(
            df,
            out_dir,
            n_jobs=(args.n_jobs if args.n_jobs is not None else -1),
            dry_run=args.dry_run,
            )

    _logging.info("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys as _sys

    _sys.exit(_cli_main())
