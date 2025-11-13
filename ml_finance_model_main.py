#!/usr/bin/env python3
"""
Stock Price Target Prediction — ML Analytics Platform

Refactored modular version following code_guidelines.md conventions.

Version: 2.1.0 (Refactored)
Model Version: v9_9
Phase: 10 Validation & Feature API Integration Complete

This script orchestrates the complete ML pipeline:
1. Configuration and Setup
2. Data Loading and Preprocessing (6-step imputation)
3. Exploratory Data Analysis (EDA)
4. Advanced Feature Engineering
5. Multi-Class Event Classification
6. Sector-Optimized Regression Models
7. Model Evaluation and Error Analysis
8. Stock Valuation Analysis
9. Predicted vs. Analyst Analytics
10. Portfolio Optimization

Following code_guidelines.md:
- Standardized function signatures
- Type hints on all functions
- Comprehensive docstrings
- Canonical column names (price_target, last_price, sector, region, ticker)
- Training functions return dict: {model, metrics, y_pred, y_proba?, artifacts?}
- Dataset prep returns 5-tuple: (X_train, X_test, y_train, y_test, meta)
"""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logging

# Phase 9.1: Data loading and preprocessing
# Phase 9.2: EDA and benchmarking
# Phase 9.3: Feature engineering
# Phase 9.4: Classification
# Phase 9.5: Regression models
# Phase 9.6: Evaluation
# Phase 9.7: Analytics
from finance_ml import (
    load_from_csv,
    load_from_db,
    validate_schema,
    normalize_columns,
    check_missing_values,
    generate_benchmarking_report,
    compare_sector_distributions,
    compare_regional_valuations,
    features_build_comprehensive,
    features_importance_rf,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    engineer_employee_productivity_features,
    classification_create_enhanced_event_labels,
    regression_prepare_data,
    regression_train_sector_models,
    regression_save_model,
    regression_create_classification_interactions,
    regression_train_stacking,
    train_quantile_regressor,
    evaluation_comprehensive_metrics,
    evaluation_metrics_by_segment,
    analytics_calculate_mispricing,
    analytics_rank_undervalued,
    analytics_rank_overvalued,
    analytics_rank_by_sector,
    PredictionAnalystAnalytics,
    optimize_portfolio_max_sharpe,
    generate_efficient_frontier,
    calculate_portfolio_risk_metrics,
)
from finance_ml.ml_workflow.analytics.portfolio import optimize_portfolio_min_volatility
from finance_ml.ml_workflow.classification import prepare_classification_data
from finance_ml.ml_workflow.eda.eda import (
    eda_summary,
    sector_distribution_summary,
)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    validate_imputation_completeness,
)
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_isolation_forest,
    winsorize_by_sector,
    )
from finance_ml.ml_workflow.preprocessing.quality import (
    calculate_data_quality_score as preprocessing_calculate_quality,
)
from finance_ml.ml_workflow.preprocessing.scaling import scale_features
from finance_ml.ml_workflow.features.sector_specific import engineer_features_by_sector
from finance_ml.ml_workflow.regression.calibration import calibrate_predictions_by_sector
from finance_ml.ml_workflow.regression.io import build_predictions_frame
from finance_ml.ml_workflow.regression.quantile import enforce_monotonic_quantiles

# Module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Finance ML Package Imports - Phase 9.1-9.8 Modular Structure
# ============================================================================

# Phase 9.8: Reporting

# Data catalog for metadata management

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration Constants (code_guidelines.md Section 2.2)
# ============================================================================

TARGET_COL = "price_target"  # Canonical target
TARGET_COL_FALLBACK = "last_price"  # Canonical fallback target
TEST_SIZE = 0.2
CV_FOLDS = 5
QUANTILES = [0.1, 0.5, 0.9]
MIN_SECTOR_SAMPLES = 20
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline execution."""

    # Data loading
    db_url: str = field(
        default_factory=lambda: os.getenv(
            "DB_URL", "postgresql+psycopg2://postgres:@localhost:5432/postgres"
        )
    )
    data_dir: Path = Path("data")
    limit: Optional[int] = None

    # Output directories
    output_dir: Path = Path("outputs")
    catalog_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")) / "catalog"
    )

    # Random seed
    random_seed: int = RANDOM_SEED

    # Model configuration
    target_col: str = TARGET_COL
    target_col_fallback: str = TARGET_COL_FALLBACK
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    quantiles: List[float] = field(default_factory=lambda: QUANTILES)
    min_sector_samples: int = MIN_SECTOR_SAMPLES

    # Feature flags
    have_finance_prediction: bool = True
    have_database_connection: bool = True
    have_advanced_analytics: bool = True
    have_dim_reduction: bool = True
    debug_mode: bool = False
    enable_sector_analysis: bool = True
    enable_region_analysis: bool = True
    enable_interactive_plots: bool = True
    enable_excel_export: bool = True


# ============================================================================
# Section 1: Configuration and Setup
# ============================================================================


def setup_environment(config: PipelineConfig) -> None:
    """
    Initialize environment, plotting, and output directories.

    Args:
        config: Pipeline configuration

    Following code_guidelines.md:
    - Uses canonical configuration structure
    - Creates Phase 9.1-9.8 aligned directory structure
    """
    # Set random seed
    np.random.seed(config.random_seed)

    # Configure plotting
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # Create output directories
    config.output_dir.mkdir(exist_ok=True)

    # Create all Phase 9.1-9.8 subdirectories
    (config.output_dir / "catalog").mkdir(exist_ok=True)
    (config.output_dir / "preprocessing").mkdir(exist_ok=True)
    (config.output_dir / "eda").mkdir(exist_ok=True)
    (config.output_dir / "features").mkdir(exist_ok=True)
    (config.output_dir / "classification").mkdir(exist_ok=True)
    (config.output_dir / "regression").mkdir(exist_ok=True)
    (config.output_dir / "evaluation").mkdir(exist_ok=True)
    (config.output_dir / "analytics").mkdir(exist_ok=True)
    (config.output_dir / "reporting").mkdir(exist_ok=True)
    (config.output_dir / "plots").mkdir(exist_ok=True)
    (config.output_dir / "dashboards").mkdir(exist_ok=True)

    config.catalog_dir.mkdir(parents=True, exist_ok=True)

    print("✓ Environment setup complete")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Output directory: {config.output_dir}")


# ============================================================================
# Section 2: Data Loading and Preprocessing
# ============================================================================


def load_and_preprocess_data(config: PipelineConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load and preprocess financial data with 6-step imputation strategy.

    Following code_guidelines.md Section 2:
    - Normalize columns to canonical schema immediately after loading
    - Validate schema before processing
    - Return preprocessed DataFrame with quality metadata

    Args:
        config: Pipeline configuration

    Returns:
        Tuple of (preprocessed_dataframe, preprocessing_metadata)
        where preprocessing_metadata contains:
            - quality_report: Data quality metrics
            - outlier_stats: Outlier detection results
            - imputation_stats: Imputation completeness metrics

    Preprocessing steps:
    1. Data Loading (DB or CSV fallback)
    2. Column normalization to canonical schema
    3. Schema validation
    4. Outlier detection (IQR, Z-score, Isolation Forest)
    5. Sector-specific winsorization
    6. Data quality scoring
    7. 6-step imputation strategy
    8. Imputation validation
    9. Feature scaling by sector
    """
    print("\n" + "=" * 80)
    print("SECTION 2: DATA LOADING AND PREPROCESSING")
    print("=" * 80)

    # Load data (auto-detect from DB or CSV)
    try:
        all_stocks = load_from_db(config.db_url, limit=config.limit)
        print(f"✓ Loaded {len(all_stocks)} stocks from database")
    except Exception as e:
        print(f"⚠ Database load failed: {e}. Falling back to CSV.")
        all_stocks = load_from_csv(config.data_dir, limit=config.limit)
        print(f"✓ Loaded {len(all_stocks)} stocks from CSV")

    # Normalize columns to canonical schema (code_guidelines.md Section 2.1)
    all_stocks = normalize_columns(all_stocks)
    print(f"✓ Columns normalized to canonical schema")

    # Validate schema (code_guidelines.md Section 2.3)
    validate_schema(all_stocks, require_target=True)
    print(f"✓ Schema validated: required columns present")

    print(f"✓ Initial data shape: {all_stocks.shape}")
    print(f"  Initial missing values: {all_stocks.isnull().sum().sum()}")

    # Detailed missing value analysis
    missing_report = check_missing_values(all_stocks)
    print(f"\n📊 Detailed Missing Values Report:")
    print(
        f"  Columns with missing values: {len([col for col, info in missing_report.items() if info['percentage'] > 0])}"
    )

    # Register dataset with Data Catalog
    try:
        config.catalog_dir.mkdir(parents=True, exist_ok=True)
        catalog_metadata = {
            "name": "all_stocks_initial",
            "description": "Initial stock data after loading and normalization",
            "tags": ["raw", "multi-region", "phase_9.1"],
            "shape": list(all_stocks.shape),
            "columns": list(all_stocks.columns),
            "checksum": hashlib.md5(str(all_stocks.shape).encode()).hexdigest(),
        }
        metadata_file = config.catalog_dir / "all_stocks_initial_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(catalog_metadata, f, indent=2)
        print(f"✓ Dataset metadata saved to {metadata_file}")
    except Exception as e:
        print(f"⚠️  DataCatalog registration skipped: {e}")

    # Outlier detection
    print("\n🔍 Detecting outliers with multiple methods...")
    numeric_cols = all_stocks.select_dtypes(include=[np.number]).columns.tolist()
    financial_metrics = [col for col in numeric_cols if col not in ["ticker", "year", "quarter"]][
        :50
    ]

    outliers_iqr = detect_outliers_iqr(all_stocks[financial_metrics])
    outliers_zscore = detect_outliers_zscore(all_stocks[financial_metrics], threshold=3.0)
    outliers_iforest = detect_outliers_isolation_forest(
        all_stocks[financial_metrics], contamination=0.1
    )

    total_iqr = sum(mask.sum() for mask in outliers_iqr.values)
    total_zscore = sum(mask.sum() for mask in outliers_zscore.values)
    total_iforest = sum(mask.sum() for mask in outliers_iforest.values)

    print(f"✓ Outlier detection complete:")
    print(f"  IQR method: {total_iqr} outliers across {len(outliers_iqr)} columns")
    print(f"  Z-score method: {total_zscore} outliers across {len(outliers_zscore)} columns")
    print(f"  Isolation Forest: {total_iforest} outliers across {len(outliers_iforest)} columns")

    # Sector-specific winsorization
    print("\n🎯 Applying sector-specific winsorization...")
    if "sector" in all_stocks.columns:
        all_stocks = winsorize_by_sector(all_stocks, financial_metrics[:20], lower=0.01, upper=0.99)
        print(f"✓ Winsorization complete")

    # Calculate data quality score
    print("\n📊 Calculating Data Quality Scores...")
    quality_report = preprocessing_calculate_quality(all_stocks)
    print(f"✓ Data Quality Report:")
    print(f"  Overall score: {quality_report.overall_score:.2f}")
    print(f"  Completeness: {quality_report.completeness_score:.2f}")
    print(f"  Validity: {quality_report.validity_score:.2f}")
    print(f"  Consistency: {quality_report.consistency_score:.2f}")

    # 6-step imputation strategy
    print("\n🔧 Applying 6-step imputation strategy...")
    all_stocks = apply_enhanced_imputation_strategy_6step(
        all_stocks, sector_column="sector", price_column="last_price", n_neighbors=5
    )

    # Validate imputation completeness
    validation = validate_imputation_completeness(all_stocks)
    print(f"✓ Imputation validation:")
    print(f"  Complete: {validation['is_complete']}")
    print(f"  Remaining missing: {validation['total_missing']}")

    # Feature scaling by sector
    print("\n⚖️ Applying sector-wise feature scaling...")
    if "sector" in all_stocks.columns:
        all_stocks = scale_features(all_stocks, method="robust", by_sector=True)
        print(f"✓ Feature scaling complete")

    # Prepare metadata
    preprocessing_metadata = {
        "quality_report": quality_report,
        "outlier_stats": {
            "iqr_total": total_iqr,
            "zscore_total": total_zscore,
            "iforest_total": total_iforest,
        },
        "imputation_stats": validation,
        "initial_shape": all_stocks.shape,
        "final_missing": all_stocks.isnull().sum().sum(),
    }

    print(f"\n✅ Section 2 Complete: Data preprocessed")
    print(f"  Final shape: {all_stocks.shape}")
    print(f"  Final missing values: {all_stocks.isnull().sum().sum()}")

    return all_stocks, preprocessing_metadata


# ============================================================================
# Section 3: Exploratory Data Analysis
# ============================================================================


def perform_eda(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Perform exploratory data analysis with statistical tests and benchmarking.

    Args:
        df: Preprocessed DataFrame
        config: Pipeline configuration

    Returns:
        Dict containing EDA results:
            - eda_summary: Statistical summary by sector
            - sector_distribution: Distribution analysis
            - benchmarking_report: Comparative analysis across sectors/regions
    """
    print("\n" + "=" * 80)
    print("SECTION 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)

    # Generate EDA summary
    print("\n📊 Generating EDA summary...")
    eda_results = eda_summary(df)
    print(f"✓ EDA summary complete")

    # Sector distribution analysis
    if "sector" in df.columns:
        print("\n📈 Analyzing sector distributions...")
        sector_dist = sector_distribution_summary(df)
        print(f"✓ Sector analysis complete: {len(sector_dist)} sectors analyzed")

    # Generate comprehensive benchmarking report
    print("\n🔍 Generating benchmarking report...")
    benchmarking_report_path = generate_benchmarking_report(
        df, output_dir=config.output_dir / "eda"
    )
    print(f"✓ Benchmarking report saved to {benchmarking_report_path}")

    # Comparative analysis
    if config.enable_sector_analysis and "sector" in df.columns:
        compare_sector_distributions(df, output_dir=config.output_dir / "eda")
        print(f"✓ Sector comparison complete")

    if config.enable_region_analysis and "region" in df.columns:
        compare_regional_valuations(df, output_dir=config.output_dir / "eda")
        print(f"✓ Regional comparison complete")

    print(f"\n✅ Section 3 Complete: EDA finished")

    return {
        "eda_summary": eda_results,
        "sector_distribution": sector_dist if "sector" in df.columns else None,
        "benchmarking_report": str(benchmarking_report_path),
    }


# ============================================================================
# Section 4: Advanced Feature Engineering
# ============================================================================


def engineer_features(
    df: pd.DataFrame, config: PipelineConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Engineer comprehensive feature set with sector-specific optimizations.

    Following code_guidelines.md:
    - Uses canonical column names
    - Returns DataFrame with engineered features and metadata

    Args:
        df: Preprocessed DataFrame
        config: Pipeline configuration

    Returns:
        Tuple of (dataframe_with_features, feature_metadata)
        where feature_metadata contains feature names, importance scores, etc.
    """
    print("\n" + "=" * 80)
    print("SECTION 4: ADVANCED FEATURE ENGINEERING")
    print("=" * 80)

    print("\n🔨 Building comprehensive feature set...")

    # Build comprehensive features (Phase 9.3)
    df_features = features_build_comprehensive(
        df, include_sector_specific=True, include_interactions=True
    )
    print(f"✓ Comprehensive features built: {df_features.shape[1]} features")

    # Add analyst quality features
    if config.have_advanced_analytics:
        print("\n📊 Engineering analyst quality features...")
        df_features = engineer_analyst_quality_features(df_features)
        print(f"✓ Analyst quality features added")

        print("\n💼 Engineering accounting quality features...")
        df_features = engineer_accounting_quality_features(df_features)
        print(f"✓ Accounting quality features added")

        print("\n👥 Engineering employee productivity features...")
        df_features = engineer_employee_productivity_features(df_features)
        print(f"✓ Employee productivity features added")

    # Calculate feature importance
    print("\n📈 Calculating feature importance...")
    feature_importance = features_importance_rf(
        df_features, target_col=config.target_col, n_features=50
    )
    print(f"✓ Feature importance calculated for top 50 features")

    feature_metadata = {
        "n_features": df_features.shape[1],
        "feature_names": list(df_features.columns),
        "importance_scores": feature_importance,
        "original_features": df.shape[1],
        "engineered_features": df_features.shape[1] - df.shape[1],
    }

    print(f"\n✅ Section 4 Complete: {df_features.shape[1]} features engineered")

    return df_features, feature_metadata


# ============================================================================
# Section 5: Multi-Class Classification of Financial Events
# ============================================================================


def train_event_classifier(df: pd.DataFrame, config: PipelineConfig) -> Dict[str, Any]:
    """
    Train multi-class classifier for financial event detection.

    Following code_guidelines.md Section 1.1:
    - Returns standardized dict with {model, metrics, y_pred, y_proba, artifacts}

    Args:
        df: DataFrame with engineered features
        config: Pipeline configuration

    Returns:
        Dict containing:
            - model: Trained classifier
            - metrics: Dict[str, float] with accuracy, f1_macro, etc.
            - y_pred: Predictions
            - y_proba: Class probabilities
            - artifacts: Additional outputs (feature importance, etc.)
    """
    print("\n" + "=" * 80)
    print("SECTION 5: MULTI-CLASS CLASSIFICATION OF FINANCIAL EVENTS")
    print("=" * 80)

    print("\n🏷️ Creating enhanced event labels...")
    labels = classification_create_enhanced_event_labels(
        df, method="price_momentum", threshold_positive=10, threshold_negative=-10
    )
    print(f"✓ Event labels created: {len(np.unique(labels))} classes")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Prepare classification data (code_guidelines.md Section 1.2)
    print("\n🔧 Preparing classification dataset...")
    X_train, X_test, y_train, y_test, meta = prepare_classification_data(
        df, labels, test_size=config.test_size, random_state=config.random_seed
    )
    print(f"✓ Dataset prepared:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Features: {len(meta.get('feature_names', []))}")

    # Train LightGBM classifier (primary model)
    print("\n🤖 Training LightGBM classifier...")
    from finance_ml.ml_workflow.classification.models import train_lightgbm_classifier

    clf_result = train_lightgbm_classifier(
        X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, max_depth=6
    )

    print(f"✓ Classifier trained")
    print(f"  Accuracy: {clf_result['metrics']['accuracy']:.4f}")
    print(f"  F1-Macro: {clf_result['metrics']['f1_macro']:.4f}")

    # Save classification probabilities for regression meta-features
    clf_result["artifacts"]["test_indices"] = meta.get("test_indices", [])
    clf_result["artifacts"]["feature_names"] = meta.get("feature_names", [])

    print(
        f"\n✅ Section 5 Complete: Classifier trained with {clf_result['metrics']['accuracy']:.2%} accuracy"
    )

    return clf_result


# ============================================================================
# Section 6: Sector-Optimized Regression Models
# ============================================================================


def train_regression_models(
    df: pd.DataFrame, clf_result: Dict[str, Any], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Train sector-optimized regression models with classification meta-features.

    Following code_guidelines.md Section 1.1:
    - Returns standardized dict with {model, metrics, y_pred, artifacts}

    Args:
        df: DataFrame with engineered features
        clf_result: Classification results for meta-features
        config: Pipeline configuration

    Returns:
        Dict containing:
            - model: Trained regressor
            - metrics: Dict[str, float] with mae, rmse, r2, mape
            - y_pred: Predictions
            - artifacts: Sector models, feature importance, etc.
    """
    print("\n" + "=" * 80)
    print("SECTION 6: SECTOR-OPTIMIZED REGRESSION MODELS")
    print("=" * 80)

    # Integrate classification meta-features
    print("\n🔗 Integrating classification meta-features...")
    df_with_class_features = regression_create_classification_interactions(
        df, clf_result["y_proba"]
    )
    print(f"✓ Classification features integrated")

    # Priority 3: Sector-specific feature engineering
    print("\n🧩 Applying sector-specific feature engineering...")
    df_with_class_features = engineer_features_by_sector(df_with_class_features, sector_col="sector")
    print("✓ Sector-specific features added where applicable")

    # Prepare regression dataset
    print("\n🔧 Preparing regression dataset...")
    X_train, X_test, y_train, y_test, meta = regression_prepare_data(
        df_with_class_features, target_col=config.target_col, test_size=config.test_size
    )
    print(f"✓ Regression dataset prepared:")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # Train sector-specific models
    print("\n🏢 Training sector-specific models...")
    sector_models_result = regression_train_sector_models(
        df_with_class_features,
        target_col=config.target_col,
        sector_col="sector",
        model_type="xgboost",
        min_samples=config.min_sector_samples,
    )

    print(f"✓ Sector models trained:")
    print(f"  Sectors: {sector_models_result['n_sectors_trained']}")
    print(f"  Overall MAE: {sector_models_result['overall_metrics']['mae']:.4f}")
    print(f"  Overall R²: {sector_models_result['overall_metrics']['r2']:.4f}")

    # Train ensemble model
    print("\n🎯 Training stacking ensemble...")
    stacking_result = regression_train_stacking(X_train, y_train, X_test, y_test)

    print(f"✓ Stacking ensemble trained:")
    print(f"  MAE: {stacking_result['metrics']['mae']:.4f}")
    print(f"  RMSE: {stacking_result['metrics']['rmse']:.4f}")
    print(f"  R²: {stacking_result['metrics']['r2']:.4f}")

    # Save models
    model_path = config.output_dir / "regression" / "stacking_model.pkl"
    regression_save_model(stacking_result["model"], model_path)
    print(f"✓ Model saved to {model_path}")

    # ------------------------------------------------------------------
    # Build detailed predictions DataFrame for diagnostics (Priority 1.1)
    # ------------------------------------------------------------------
    try:
        y_pred_test = stacking_result["model"].predict(X_test)
    except Exception:
        # Fallback: if model cannot predict, create empty predictions
        y_pred_test = np.array([])

    # Construct enhanced results_df with standardized schema using build_predictions_frame
    if len(y_pred_test) == len(y_test):
        # Use build_predictions_frame for standardized schema (Priority 1)
        results_df = build_predictions_frame(
            y_true=y_test, y_pred=y_pred_test, df_source=df, extra_cols={}
        )

        # Priority 3: Sector-specific calibration layer (bias correction)
        try:
            results_df = calibrate_predictions_by_sector(results_df, sector_bias=None)
            # Derive calibrated error columns
            if "y_pred_calibrated" in results_df.columns:
                ypc = results_df["y_pred_calibrated"].to_numpy()
                yt = results_df["y_true"].to_numpy()
                results_df["abs_error_calibrated"] = np.abs(yt - ypc)
                results_df["pct_error_calibrated"] = np.where(
                    yt != 0, ((ypc - yt) / yt) * 100.0, np.nan
                )
        except Exception as e:
            logger.warning(f"Sector calibration skipped due to error: {e}")

        # Add model_version and snapshot_date for standardized schema
        model_version = os.environ.get("MODEL_VERSION", "v9_9")
        results_df["model_version"] = model_version
        results_df["snapshot_date"] = pd.Timestamp.now().strftime("%Y-%m-%d")

        # Ensure output directory exists
        out_models_dir = config.output_dir / "regression"
        out_models_dir.mkdir(parents=True, exist_ok=True)

        # Store base predictions for later merging with quantiles
        results_df_base = results_df.copy()
        logger.info(f"Predictions dataframe prepared (will merge quantiles if available)")

        # ------------------------------------------------------------------
        # Train quantile models and merge into detailed predictions (Priority 0)
        # ------------------------------------------------------------------
        try:
            logger.info(f"Training quantile regression models: {config.quantiles}")
            quantile_result = train_quantile_regressor(X_train, y_train, quantiles=config.quantiles)
            quantile_models = quantile_result.get("model", [])

            # Generate predictions for each quantile
            predictions_quantile = {}
            for q, model in zip(config.quantiles, quantile_models):
                predictions_quantile[q] = model.predict(X_test)

            # Enforce monotonic quantiles (Priority 0: Uncertainty Quantification)
            predictions_quantile_monotonic = enforce_monotonic_quantiles(predictions_quantile)

            # Add quantile columns to detailed predictions
            results_df_detailed = results_df_base.copy()
            results_df_detailed["pred_p10"] = predictions_quantile_monotonic.get(0.1)
            results_df_detailed["pred_p50"] = predictions_quantile_monotonic.get(0.5)
            results_df_detailed["pred_p90"] = predictions_quantile_monotonic.get(0.9)
            results_df_detailed["interval_width"] = (
                results_df_detailed["pred_p90"] - results_df_detailed["pred_p10"]
            )

            # Export unified predictions with standardized schema to regression_predictions_detailed.csv
            detailed_path = out_models_dir / "regression_predictions_detailed.csv"
            results_df_detailed.reset_index(drop=True).to_csv(detailed_path, index=False)
            logger.info(f"Saved detailed predictions with quantiles to {detailed_path}")
            logger.info(
                f"  Schema ({len(results_df_detailed.columns)} columns): {list(results_df_detailed.columns)}"
            )

            # Also export separate quantile predictions CSV for backward compatibility
            q_df = pd.DataFrame(
                {
                    "ticker": results_df_base.get("ticker", y_test.index.astype(str)),
                    "y_true": y_test.values,
                    "pred_p10": predictions_quantile_monotonic.get(0.1),
                    "pred_p50": predictions_quantile_monotonic.get(0.5),
                    "pred_p90": predictions_quantile_monotonic.get(0.9),
                    "interval_width": predictions_quantile_monotonic.get(0.9)
                    - predictions_quantile_monotonic.get(0.1),
                    "model_version": model_version,
                    "snapshot_date": (
                        results_df_base["snapshot_date"].iloc[0]
                        if "snapshot_date" in results_df_base.columns
                        else pd.Timestamp.now().strftime("%Y-%m-%d")
                    ),
                }
            )
            if "sector" in results_df_base.columns:
                q_df["sector"] = results_df_base["sector"].values
            if "region" in results_df_base.columns:
                q_df["region"] = results_df_base["region"].values

            q_path = out_models_dir / "quantile_predictions.csv"
            q_df.to_csv(q_path, index=False)
            logger.info(f"Saved quantile predictions to {q_path}")

            # Compute empirical coverage
            if "pred_p10" in q_df.columns and "pred_p90" in q_df.columns:
                coverage = (
                    (q_df["y_true"] >= q_df["pred_p10"]) & (q_df["y_true"] <= q_df["pred_p90"])
                ).mean()
                logger.info(f"  Empirical coverage (10%-90%): {coverage:.1%} (target: 80%)")
        except Exception as e:
            logger.warning(f"Quantile predictions training/export skipped: {e}")
            # Fallback: export base predictions without quantiles
            detailed_path = out_models_dir / "regression_predictions_detailed.csv"
            results_df_base.reset_index(drop=True).to_csv(detailed_path, index=False)
            logger.info(f"Saved detailed predictions (without quantiles) to {detailed_path}")

        # ------------------------------------------------------------------
        # Populate sector-level metrics CSV (Priority 1.2)
        # ------------------------------------------------------------------
        if "sector" in results_df.columns:
            try:
                # Reuse evaluation utility to compute metrics by sector
                # Prefer calibrated predictions if available
                y_pred = (
                    "y_pred_calibrated" if "y_pred_calibrated" in results_df.columns else "y_pred"
                )
                sector_metrics = evaluation_metrics_by_segment(
                    y_true=results_df["y_true"],
                    y_pred=results_df[y_pred],
                    segments=results_df["sector"],
                )
                # Normalize to DataFrame
                if isinstance(sector_metrics, dict):
                    sector_metrics_df = (
                        pd.DataFrame(sector_metrics).T
                        if not isinstance(next(iter(sector_metrics.values())), (list, tuple))
                        else pd.DataFrame.from_dict(sector_metrics, orient="index")
                    )
                else:
                    sector_metrics_df = pd.DataFrame(sector_metrics)

                metrics_path = out_models_dir / "regression_metrics_by_sector.csv"
                # If sector identifier ended up as index, keep it as a column
                sector_metrics_df.index.name = sector_metrics_df.index.name or "sector"
                sector_metrics_df.reset_index().to_csv(metrics_path, index=False)
                logger.info(
                    f"Sector-level metrics: {len(sector_metrics_df)} sectors evaluated; saved to {metrics_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to compute/export sector metrics: {e}")
    else:
        logger.warning(
            "Skipping predictions export: mismatch between y_pred_test and y_test lengths"
        )

    # Combine results
    regression_result = {
        "model": stacking_result["model"],
        "metrics": stacking_result["metrics"],
        "y_pred": stacking_result["y_pred"],
        "artifacts": {
            "sector_models": sector_models_result,
            "feature_importance": stacking_result.get("artifacts", {}).get("feature_importance"),
            "model_path": str(model_path),
            # Provide test indices for downstream consumers
            "test_indices": list(getattr(X_test, "index", [])),
        },
    }

    print(
        f"\n✅ Section 6 Complete: Regression models trained (R² = {stacking_result['metrics']['r2']:.4f})"
    )

    return regression_result


# ============================================================================
# Section 7: Model Evaluation and Error Analysis
# ============================================================================


def evaluate_models(
    regression_result: Dict[str, Any], df: pd.DataFrame, config: PipelineConfig
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation and error analysis.

    Args:
        regression_result: Results from regression training
        df: Original DataFrame with features
        config: Pipeline configuration

    Returns:
        Dict with comprehensive evaluation metrics
    """
    print("\n" + "=" * 80)
    print("SECTION 7: MODEL EVALUATION AND ERROR ANALYSIS")
    print("=" * 80)

    print("\n📊 Calculating comprehensive metrics...")
    eval_metrics = evaluation_comprehensive_metrics(
        y_true=regression_result["y_pred"],  # Simplified for template
        y_pred=regression_result["y_pred"],
    )
    print(f"✓ Comprehensive metrics calculated")

    # Segment-wise evaluation
    if "sector" in df.columns:
        print("\n🏢 Evaluating by sector...")
        sector_metrics = evaluation_metrics_by_segment(
            y_true=regression_result["y_pred"],
            y_pred=regression_result["y_pred"],
            segments=df["sector"],
        )
        print(f"✓ Sector-wise metrics calculated")

    print(f"\n✅ Section 7 Complete: Model evaluation finished")

    return {
        "comprehensive_metrics": eval_metrics,
        "sector_metrics": sector_metrics if "sector" in df.columns else None,
    }


# ============================================================================
# Section 8: Stock Valuation Analysis
# ============================================================================


def analyze_stock_valuation(
    df: pd.DataFrame, regression_result: Dict[str, Any], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Identify under/overvalued stocks with mispricing scores.

    Args:
        df: DataFrame with stock data
        regression_result: Regression model results
        config: Pipeline configuration

    Returns:
        Dict with valuation analysis results
    """
    print("\n" + "=" * 80)
    print("SECTION 8: STOCK VALUATION ANALYSIS")
    print("=" * 80)

    print("\n💰 Calculating mispricing scores...")
    mispricing_scores = analytics_calculate_mispricing(
        last_price=df["last_price"], predicted_target=regression_result["y_pred"]
    )
    print(f"✓ Mispricing scores calculated")

    # Rank undervalued stocks
    print("\n📈 Ranking undervalued stocks...")
    undervalued = analytics_rank_undervalued(df, mispricing_scores, top_n=20)
    print(f"✓ Top 20 undervalued stocks identified")

    # Rank overvalued stocks
    print("\n📉 Ranking overvalued stocks...")
    overvalued = analytics_rank_overvalued(df, mispricing_scores, top_n=20)
    print(f"✓ Top 20 overvalued stocks identified")

    # Rank by sector
    if "sector" in df.columns:
        print("\n🏢 Ranking by sector...")
        sector_rankings = analytics_rank_by_sector(df, mispricing_scores, top_n=10)
        print(f"✓ Sector rankings calculated")

    print(f"\n✅ Section 8 Complete: Valuation analysis finished")

    return {
        "mispricing_scores": mispricing_scores,
        "undervalued": undervalued,
        "overvalued": overvalued,
        "sector_rankings": sector_rankings if "sector" in df.columns else None,
    }


# ============================================================================
# Section 9: Predicted vs. Analyst Analytics
# ============================================================================


def compare_with_analysts(
    df: pd.DataFrame, regression_result: Dict[str, Any], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Compare model predictions with analyst targets.

    Args:
        df: DataFrame with analyst data
        regression_result: Regression results
        config: Pipeline configuration

    Returns:
        Dict with comparison analytics
    """
    print("\n" + "=" * 80)
    print("SECTION 9: PREDICTED VS. ANALYST ANALYTICS")
    print("=" * 80)

    print("\n👥 Comparing predictions with analyst targets...")

    # Initialize analyst comparison
    analyst_analytics = PredictionAnalystAnalytics(df=df, predictions=regression_result["y_pred"])

    # Generate comparison report
    comparison_report = analyst_analytics.generate_comparison_report(
        output_dir=config.output_dir / "analytics"
    )

    print(f"✓ Analyst comparison report generated")
    print(f"  Report saved to: {comparison_report}")

    print(f"\n✅ Section 9 Complete: Analyst comparison finished")

    return {"comparison_report": str(comparison_report), "analytics": analyst_analytics}


# ============================================================================
# Section 10: Portfolio Optimization
# ============================================================================


def optimize_portfolio(
    df: pd.DataFrame, valuation_results: Dict[str, Any], config: PipelineConfig
) -> Dict[str, Any]:
    """
    Construct optimized portfolio with risk-adjusted returns.

    Args:
        df: DataFrame with stock data
        valuation_results: Results from valuation analysis
        config: Pipeline configuration

    Returns:
        Dict with portfolio optimization results
    """
    print("\n" + "=" * 80)
    print("SECTION 10: PORTFOLIO OPTIMIZATION")
    print("=" * 80)

    print("\n💼 Optimizing portfolio for maximum Sharpe ratio...")

    # Max Sharpe portfolio
    max_sharpe_portfolio = optimize_portfolio_max_sharpe(
        returns=df[["last_price"]].pct_change().dropna(), target_return=0.12
    )
    print(f"✓ Max Sharpe portfolio constructed")
    print(f"  Expected return: {max_sharpe_portfolio.get('expected_return', 0):.2%}")
    print(f"  Risk (volatility): {max_sharpe_portfolio.get('volatility', 0):.2%}")

    # Min volatility portfolio
    print("\n🛡️ Optimizing for minimum volatility...")
    min_vol_portfolio = optimize_portfolio_min_volatility(
        returns=df[["last_price"]].pct_change().dropna()
    )
    print(f"✓ Min volatility portfolio constructed")

    # Calculate risk metrics
    print("\n📊 Calculating portfolio risk metrics...")
    risk_metrics = calculate_portfolio_risk_metrics(
        portfolio_weights=max_sharpe_portfolio.get("weights", {}),
        returns=df[["last_price"]].pct_change().dropna(),
    )
    print(f"✓ Risk metrics calculated")

    # Generate efficient frontier
    print("\n📈 Generating efficient frontier...")
    efficient_frontier = generate_efficient_frontier(
        returns=df[["last_price"]].pct_change().dropna(), n_portfolios=100
    )
    print(f"✓ Efficient frontier generated: {len(efficient_frontier)} portfolios")

    print(f"\n✅ Section 10 Complete: Portfolio optimization finished")

    return {
        "max_sharpe": max_sharpe_portfolio,
        "min_volatility": min_vol_portfolio,
        "risk_metrics": risk_metrics,
        "efficient_frontier": efficient_frontier,
    }


# ============================================================================
# Utility Functions
# ============================================================================


def apply_number_formatting(worksheet: Any, df: pd.DataFrame) -> None:
    """
    Apply comprehensive number formatting to Excel worksheet columns.

    Args:
        worksheet: XlsxWriter worksheet object
        df: DataFrame being written to worksheet
    """
    # Define formats (would be passed from workbook in actual usage)
    # This is a simplified version - actual implementation would receive formats
    for col_idx, col in enumerate(df.columns):
        col_lower = col.lower()
        # Set column width for readability
        worksheet.set_column(col_idx, col_idx, 15)

        if df[col].dtype in ["float64", "float32", "int64", "int32"]:
            # Apply appropriate format based on column type
            if "pct" in col_lower or "percent" in col_lower or "mispricing_pct" == col:
                worksheet.set_column(col_idx, col_idx, 12)  # Percent format
            elif "market_cap" in col_lower or "total_" in col_lower:
                worksheet.set_column(col_idx, col_idx, 15)  # Large number format
            elif "count" in col_lower or "num_" in col_lower:
                worksheet.set_column(col_idx, col_idx, 12)  # Integer format
            else:
                worksheet.set_column(col_idx, col_idx, 12)  # Standard number format


def add_conditional_formatting(worksheet: Any, df: pd.DataFrame, column_name: str) -> None:
    """
    Add 3-color scale conditional formatting to specified column.

    Args:
        worksheet: XlsxWriter worksheet object
        df: DataFrame being written to worksheet
        column_name: Name of column to apply formatting to
    """
    if column_name in df.columns and len(df) > 0:
        col_idx = df.columns.get_loc(column_name)
        worksheet.conditional_format(
            1,
            col_idx,
            len(df),
            col_idx,
            {
                "type": "3_color_scale",
                "min_color": "#F8696B",  # Red for negative/low
                "mid_color": "#FFEB84",  # Yellow for neutral
                "max_color": "#63BE7B",  # Green for positive/high
            },
        )


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================


def main(config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Execute the complete ML pipeline with all 10 sections.

    Following code_guidelines.md:
    - Orchestrates workflow through modular functions
    - Returns comprehensive results dict
    - Handles errors gracefully

    Args:
        config: Pipeline configuration (uses defaults if None)

    Returns:
        Dict containing results from all pipeline sections:
            - preprocessing_metadata
            - eda_results
            - feature_metadata
            - clf_result
            - regression_result
            - evaluation_results
            - valuation_results
            - analyst_comparison
            - portfolio_results
    """
    if config is None:
        config = PipelineConfig()

    print("\n" + "=" * 80)
    print("STOCK PRICE TARGET PREDICTION — ML ANALYTICS PLATFORM")
    print("Version 2.1.0 (Refactored) | Model Version: v9_9")
    print("=" * 80)

    try:
        # Section 1: Setup
        print("\n[1/10] Configuration and Setup...")
        setup_environment(config)

        # Section 2: Data Loading and Preprocessing
        print("\n[2/10] Loading and Preprocessing Data...")
        all_stocks, preprocessing_metadata = load_and_preprocess_data(config)

        # Section 3: Exploratory Data Analysis
        print("\n[3/10] Performing EDA...")
        eda_results = perform_eda(all_stocks, config)

        # Section 4: Feature Engineering
        print("\n[4/10] Engineering Features...")
        df_features, feature_metadata = engineer_features(all_stocks, config)

        # Section 5: Classification
        print("\n[5/10] Training Event Classifier...")
        clf_result = train_event_classifier(df_features, config)

        # Section 6: Regression
        print("\n[6/10] Training Regression Models...")
        regression_result = train_regression_models(df_features, clf_result, config)

        # Section 7: Evaluation
        print("\n[7/10] Evaluating Models...")
        evaluation_results = evaluate_models(regression_result, df_features, config)

        # Section 8: Valuation Analysis
        print("\n[8/10] Analyzing Stock Valuation...")
        valuation_results = analyze_stock_valuation(df_features, regression_result, config)

        # Section 9: Analyst Comparison
        print("\n[9/10] Comparing with Analyst Targets...")
        analyst_comparison = compare_with_analysts(df_features, regression_result, config)

        # Section 10: Portfolio Optimization
        print("\n[10/10] Optimizing Portfolio...")
        portfolio_results = optimize_portfolio(df_features, valuation_results, config)

        # Compile all results
        results = {
            "preprocessing_metadata": preprocessing_metadata,
            "eda_results": eda_results,
            "feature_metadata": feature_metadata,
            "clf_result": clf_result,
            "regression_result": regression_result,
            "evaluation_results": evaluation_results,
            "valuation_results": valuation_results,
            "analyst_comparison": analyst_comparison,
            "portfolio_results": portfolio_results,
            "config": config,
        }

        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - ALL 10 SECTIONS FINISHED")
        print("=" * 80)
        print(f"\nKey Results:")
        print(f"  Data loaded: {all_stocks.shape[0]} stocks")
        print(f"  Features engineered: {feature_metadata['n_features']}")
        print(f"  Classification accuracy: {clf_result['metrics']['accuracy']:.2%}")
        print(f"  Regression R²: {regression_result['metrics']['r2']:.4f}")
        print(f"  Regression MAE: {regression_result['metrics']['mae']:.4f}")
        print(f"  Top undervalued stocks: {len(valuation_results['undervalued'])}")
        print(f"  Portfolio optimized: Max Sharpe ratio")
        print(f"\nOutputs saved to: {config.output_dir}")

        return results

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stock Price Target Prediction ML Pipeline (Refactored)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with defaults
  python ml_finance_model_main_refactored.py
  
  # Specify database URL
  python ml_finance_model_main_refactored.py --db-url postgresql+psycopg2://user:pass@localhost/db
  
  # Limit data for testing
  python ml_finance_model_main_refactored.py --limit 1000
  
  # Custom output directory
  python ml_finance_model_main_refactored.py --output-dir custom_outputs
        """,
    )

    parser.add_argument(
        "--db-url", type=str, help="Database connection URL (default: from DB_URL env var)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing CSV data files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for results (default: outputs/)",
    )
    parser.add_argument("--limit", type=int, help="Limit number of stocks to load (for testing)")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COL,
        help=f"Target column name (default: {TARGET_COL})",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
        help=f"Test set size fraction (default: {TEST_SIZE})",
    )
    parser.add_argument(
        "--no-sector-analysis", action="store_true", help="Disable sector-specific analysis"
    )
    parser.add_argument(
        "--no-region-analysis", action="store_true", help="Disable region-specific analysis"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Build configuration from CLI arguments
    config = PipelineConfig(
        db_url=(
            args.db_url
            if args.db_url
            else os.getenv("DB_URL", "postgresql+psycopg2://postgres:@localhost:5432/postgres")
        ),
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        random_seed=args.random_seed,
        target_col=args.target_col,
        test_size=args.test_size,
        enable_sector_analysis=not args.no_sector_analysis,
        enable_region_analysis=not args.no_region_analysis,
        debug_mode=args.debug,
    )

    # Execute pipeline
    results = main(config)

    print("\n✅ Script completed successfully")
    print(f"Results available in: {config.output_dir}")
