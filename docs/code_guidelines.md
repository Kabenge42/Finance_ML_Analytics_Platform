# Finance ML Analytics Platform — Code Guidelines

These guidelines codify conventions for function signatures, return types, dataset preparation, column naming/schema,
and general Python best practices. They align with the project’s business objectives and the Phase 9.1–9.8 modular
design described in README.md and the improvement plans.

Goals

- Maximize code quality, maintainability, and testability
- Ensure consistent APIs across modules and phases
- Guarantee schema and naming consistency end-to-end
- Make downstream analytics, notebooks, and CLI predictable and robust

1) Standardized Function Signatures and Return Types

1.1 Training functions (train_*)

- Contract: All model-training functions return a dict with these keys:
    - model: The fitted estimator or pipeline
    - metrics: Dict[str, float] — evaluation metrics (e.g., accuracy, f1_macro for classification; mae, rmse, r2 for
      regression)
    - y_pred: 1D array-like or pandas Series/DataFrame of predictions aligned to input indices
    - y_proba: Optional 2D array-like or DataFrame of class probabilities (classification only). Omit or set to None for
      regressors.
    - artifacts: Optional Dict[str, Any] — auxiliary items (e.g., feature_importance, confusion_matrix, oof_predictions,
      cv_results)

- Examples:

```python
# Classification
res = train_event_classifier(X, y, model="lightgbm")
assert set(res).issuperset({"model", "metrics", "y_pred"})
acc = res["metrics"].get("accuracy")
f1m = res["metrics"].get("f1_macro")
y_proba = res.get("y_proba")  # May be None if estimator has no predict_proba

# Regression
res = train_and_evaluate_regression(df)
mae = res["metrics"].get("mae")
r2 = res["metrics"].get("r2")
y_pred = res["y_pred"]  # Series/DataFrame aligned to df index
```

- Backward compatibility: Where legacy code expects top-level metric keys (e.g., res["mae"]) provide shims during
  transition, but write new code to use res["metrics"]["mae"].

1.2 Dataset preparation

- Contract: Dataset prep functions return a 5-tuple or a dataclass:
    - (X_train, X_test, y_train, y_test, meta)
    - Where meta is a dict or small dataclass including feature_names, categorical_features, target_name, indices, and
      any scalers/encoders if applicable.
- Dataclass option:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class DatasetSplit:
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    meta: Dict[str, Any]
```

- Rationale: A consistent shape across phases simplifies tests and integration between classification, regression, and
  analytics.

1.3 Comprehensive Function Signatures by Module

This section provides detailed function signatures from the Finance ML Analytics Platform package modules.

**Phase 9.1 — Preprocessing**

```python
# finance_ml.ml_workflow.preprocessing.pipeline
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data

prepared_df, quality_stats = prepare_phase91_data(
        df: pd.DataFrame,
sector_column: str = "sector",
price_column: str = "last_price",
n_neighbors: int = 5,
return_stats: bool = True
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]
# Returns: (preprocessed_df, quality_statistics_dict)

# Phase 9.1 Enhanced Imputation (6-step strategy as of v0.6.1)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_4step,  # Core 6-step (zero, KNN, price, median)
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_price_imputation,
    apply_median_imputation
    )

# Note: CHANGELOG refers to "6-step imputation" in v0.6.1 (505+ lines of improvements)
# Current implementation provides modular 6-step base with extensibility for additional steps
df_imputed = apply_enhanced_imputation_strategy_4step(
        df: pd.DataFrame,
zero_fill_columns: Optional[List[str]] = None,
knn_neighbors: int = 5,
price_columns: Optional[List[str]] = None
) -> pd.DataFrame
# Returns: DataFrame with all missing values imputed
```

**Phase 9.3 — Features**

```python
# finance_ml.ml_workflow.features.core
from finance_ml.ml_workflow.features.core import preprocess_for_lightgbm

df_processed, encoders = preprocess_for_lightgbm(
        df: pd.DataFrame,
categorical_columns: Optional[List[str]] = None,
datetime_columns: Optional[List[str]] = None,
drop_columns: Optional[List[str]] = None,
return_encoders: bool = False
) -> Tuple[pd.DataFrame, Optional[Dict[str, LabelEncoder]]]
# Returns: (numeric_df, label_encoders_dict or None)

# finance_ml.ml_workflow.features.api (Phase 9.3 API - RECOMMENDED)
from finance_ml.ml_workflow.features.api import build_features, PresetName

df_with_features = build_features(
        df: pd.DataFrame,
preset: PresetName = "comprehensive",
include_interactions: bool = True,
include_relative: bool = True,
sector_col: str = "sector"
) -> pd.DataFrame
# Returns: df with features based on preset
# Presets (v9_9):
#   "basic" - Core ratios, margins, volatility, revenue CAGR
#   "momentum" - Price momentum, RSI, moving averages, return stability
#   "quality" - Accounting quality, financial distress signals (Altman Z)
#   "comprehensive" - Full advanced feature set
#   "full_enhanced" - Alias for comprehensive preset

# finance_ml.ml_workflow.features.advanced
from finance_ml.ml_workflow.features.advanced import (
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_momentum_features,
    engineer_analyst_quality_features,
    engineer_accounting_quality_features,
    build_comprehensive_features
    )

df_with_ratios = engineer_valuation_ratios(
        df: pd.DataFrame,
fillna: bool = True
) -> pd.DataFrame
# Returns: df with p_e, p_b, p_s, ev_ebitda, peg_ratio, etc.

df_with_momentum = engineer_momentum_features(
        df: pd.DataFrame
) -> pd.DataFrame
# Returns: df with price_momentum_1m/3m/6m, rsi_14d/30d, ma_crossover_signal, return_stability_score

df_comprehensive = build_comprehensive_features(
        df: pd.DataFrame,
include_interactions: bool = True,
include_relative_values: bool = True,
sector_col: str = "sector",
preset: Optional[str] = None
) -> pd.DataFrame
# Returns: df with all advanced features
# Optional preset: "momentum", "quality", "comprehensive" for targeted feature sets
```

**Phase 9.4 — Classification**

```python
# finance_ml.ml_workflow.classification.labels
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels

labels = create_enhanced_event_labels(
        df: pd.DataFrame,
method: Literal[
    'price_momentum', 'valuation', 'fundamental', 'volatility',
    'analyst_rating', 'market_events', 'combined_signals',
    'profitability_event', 'leverage_event', 'liquidity_event',
    'growth_event', 'efficiency_event', 'quality_event'
] = 'price_momentum',
threshold_positive: float = 10.0,
threshold_negative: float = -10.0,
use_sector_adjustment: bool = True
) -> np.ndarray
# Returns: array of labels (0=Strong Negative, 1=Negative, 2=Neutral, 3=Positive, 4=Strong Positive)
# 
# Phase 9.4 enhancements (v9_9): All 13 methods support Phase 9.3 engineered columns
# Method 1 (price_momentum): Uses price_momentum_1m/3m/6m, rsi_14d/30d, ma_crossover_signal
# Method 2 (valuation): Uses p_e_ratio, p_b_ratio, ev_ebitda_ratio, peg_ratio
# Method 3 (fundamental): Uses gross_margin_pct, operating_margin_pct, roe, roa, roic
# Method 4 (volatility): Uses return_stability_score, sharpe_proxy
# Method 5 (analyst_rating): Uses upside_potential, analyst_bullish_pct, analyst_coverage_quality
# Method 6 (market_events): Uses short_interest_ratio, systematic_risk_trend, sector-relative metrics
# Method 7 (combined_signals): Multi-metric composite
# Methods 8-13: Profitability, leverage, liquidity, growth, efficiency, quality events
# All methods backward compatible with original columns

# finance_ml.ml_workflow.classification.tuning
from finance_ml.ml_workflow.classification.tuning import optimize_classifier_hyperparameters

result = optimize_classifier_hyperparameters(
        X_train: pd.DataFrame,
y_train: np.ndarray,
classifier_type: Literal['xgboost', 'lightgbm', 'catboost', 'random_forest'] = 'xgboost',
n_trials: int = 50,
cv_folds: int = 5,
random_state: int = 42,
verbose: bool = True
) -> Dict[str, Any]
# Returns: {
#   'best_params': Dict[str, Any],
#   'best_score': float,
#   'study': optuna.Study,
#   'model': fitted_classifier
# }

# finance_ml.ml_workflow.classification.models - Data Preparation
from finance_ml.ml_workflow.classification import prepare_classification_data

X_train, X_test, y_train, y_test, numeric_cols, categorical_cols = prepare_classification_data(
        df: pd.DataFrame,
labels: np.ndarray,
test_size: float = 0.2,
random_state: int = 42,
feature_groups: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]
# Returns: (X_train, X_test, y_train, y_test, numeric_features, categorical_features)
# Handles train/test split, categorical encoding, and Phase 9.3 feature group integration

# finance_ml.ml_workflow.classification.models - Model Training and Comparison
from finance_ml.ml_workflow.classification import (
    compare_classifiers,
    fit_classifier,
    train_xgboost_classifier,
    train_lightgbm_classifier,
    train_catboost_classifier
    )

# High-level orchestrator API (recommended)
result = fit_classifier(
        X_train: pd.DataFrame,
y_train: np.ndarray,
X_test: Optional[pd.DataFrame] = None,
y_test: Optional[np.ndarray] = None,
model: Union[str, List[str]] = "xgboost",
params: Optional[Dict[str, Any]] = None,
tuning: Optional[Dict[str, Any]] = None,
cv: Optional[Dict[str, Any]] = None,
class_weighting: Optional[str] = None,
compare: bool = False
) -> Dict[str, Any]
# Returns: {
#   'model': fitted_classifier,
#   'metrics': {'accuracy': float, 'f1_macro': float, 'precision': float, 'recall': float},
#   'y_pred': np.ndarray,
#   'y_proba': np.ndarray,
#   'artifacts': {'feature_importance': pd.DataFrame, 'confusion_matrix': np.ndarray}
# }

# Compare multiple classifiers
comparison_results = compare_classifiers(
        X_train: pd.DataFrame,
y_train: np.ndarray,
X_test: pd.DataFrame,
y_test: np.ndarray,
numeric_cols: List[str],
categorical_cols: List[str]
) -> Dict[str, Dict[str, Any]]
# Returns: {model_name: {model, metrics, y_pred, y_proba}} for each classifier

# finance_ml.ml_workflow.classification.evaluation - Metrics and Evaluation
from finance_ml.ml_workflow.classification.evaluation import (
    evaluate_classification,
    evaluate_classification_by_sector,
    cross_validate_classifier
    )

metrics = evaluate_classification(
        y_true: np.ndarray,
y_pred: np.ndarray,
y_proba: Optional[np.ndarray] = None,
class_names: Optional[List[str]] = None
) -> Dict[str, Any]
# Returns: {accuracy, precision, recall, f1_score, confusion_matrix, classification_report}

sector_metrics = evaluate_classification_by_sector(
        y_true: np.ndarray,
y_pred: np.ndarray,
sectors: pd.Series
) -> pd.DataFrame
# Returns: DataFrame with per-sector accuracy, precision, recall, f1

cv_results = cross_validate_classifier(
        model: ClassifierMixin,
X: pd.DataFrame,
y: np.ndarray,
cv: int = 5,
stratify_by: Optional[str] = None
) -> Dict[str, Any]
# Returns: {scores, mean_score, std_score, fold_details}

# finance_ml.ml_workflow.classification.evaluation - Visualization
from finance_ml.ml_workflow.classification.evaluation import (
    plot_confusion_matrices,
    plot_learning_curves
    )

plot_confusion_matrices(
        models_results: Dict[str, Dict[str, Any]],
class_names: Optional[List[str]] = None
) -> None
# Plots confusion matrices for multiple models side-by-side

plot_learning_curves(
        model: ClassifierMixin,
X: pd.DataFrame,
y: np.ndarray,
cv: int = 5,
train_sizes: Optional[np.ndarray] = None,
scoring: str = "accuracy"
) -> None
# Plots learning curves showing train/validation scores vs training size

# finance_ml.ml_workflow.classification.evaluation - SHAP Analysis
from finance_ml.ml_workflow.classification.evaluation import (
    compute_shap_values,
    analyze_per_class_feature_importance,
    analyze_shap_by_feature_groups
    )

shap_values = compute_shap_values(
        model: Any,
X_train: pd.DataFrame,
X_test: pd.DataFrame,
max_samples: int = 100
) -> Any
# Returns: SHAP values for interpretability analysis

class_importance = analyze_per_class_feature_importance(
        model: ClassifierMixin,
X: pd.DataFrame,
y: np.ndarray,
feature_names: Optional[List[str]] = None,
top_n: int = 10
) -> Dict[int, pd.DataFrame]
# Returns: {class_id: DataFrame of top features} for each class

group_shap = analyze_shap_by_feature_groups(
        shap_values: Any,
feature_names: List[str],
top_n_per_group: int = 10
) -> Dict[str, pd.DataFrame]
# Returns: SHAP analysis grouped by feature categories (Phase 9.3 groups)

# finance_ml.ml_workflow.classification.evaluation - Feature Importance
from finance_ml.ml_workflow.classification.evaluation import (
    compare_feature_importance,
    analyze_feature_importance_by_groups,
    analyze_feature_importance_by_sector
    )

importance_comparison = compare_feature_importance(
        models_dict: Dict[str, Dict[str, Any]],
feature_names: List[str],
top_n: int = 20
) -> pd.DataFrame
# Returns: DataFrame comparing feature importance across models

group_importance = analyze_feature_importance_by_groups(
        importance_dict: Dict[str, float],
feature_names: Optional[List[str]] = None,
top_n_per_group: int = 10
) -> Dict[str, pd.DataFrame]
# Returns: Feature importance grouped by Phase 9.3 categories

sector_importance = analyze_feature_importance_by_sector(
        model: Any,
X: pd.DataFrame,
y: np.ndarray,
sector_col: str = "sector",
top_n: int = 15
) -> Dict[str, pd.DataFrame]
# Returns: {sector: DataFrame of top features} for sector-specific analysis

# finance_ml.ml_workflow.classification.evaluation - Calibration
from finance_ml.ml_workflow.classification.evaluation import analyze_calibration

calibration_metrics = analyze_calibration(
        y_true: np.ndarray,
y_proba: np.ndarray,
n_bins: int = 10
) -> Dict[str, Any]
# Returns: {brier_score, log_loss, calibration_curve_data, ece, mce}

# finance_ml.ml_workflow.classification.tuning - Sector-Stratified Cross-Validation
from finance_ml.ml_workflow.classification.tuning import cross_validate_with_sector_stratification

cv_results = cross_validate_with_sector_stratification(
        model: Any,
X: pd.DataFrame,
y: np.ndarray,
sector_col: str = "sector",
n_splits: int = 5,
random_state: int = 42
) -> Dict[str, Any]
# Returns: {scores, mean_score, std_score, per_sector_scores}
# Uses StratifiedGroupKFold to maintain class and sector distribution
```

**Phase 9.5 — Regression**

```python
# finance_ml.ml_workflow.regression.dataset
from finance_ml.ml_workflow.regression.dataset import prepare_regression_data

X_train, X_test, y_train, y_test, meta = prepare_regression_data(
        df: pd.DataFrame,
target_col: str = 'price_target',
test_size: float = 0.2,
random_state: int = 42,
feature_groups: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]
# Returns: (X_train, X_test, y_train, y_test, metadata)

# finance_ml.ml_workflow.regression.models
from finance_ml.ml_workflow.regression.models import (
    train_xgboost_regressor,
    train_lightgbm_regressor,
    compare_regressors
    )

result = train_xgboost_regressor(
        X_train: pd.DataFrame,
y_train: pd.Series,
X_test: Optional[pd.DataFrame] = None,
y_test: Optional[pd.Series] = None,
** kwargs
) -> Dict[str, Any]
# Returns: {
#   'model': fitted_model,
#   'metrics': {'mae': float, 'rmse': float, 'r2': float, 'mape': float},
#   'y_pred': np.ndarray,
#   'feature_importance': pd.DataFrame
# }

comparison_df = compare_regressors(
        X_train: pd.DataFrame,
y_train: pd.Series,
X_test: pd.DataFrame,
y_test: pd.Series,
models: Optional[List[str]] = None
) -> pd.DataFrame
# Returns: DataFrame with model comparison metrics (mae, rmse, r2, mape per model)

# finance_ml.ml_workflow.regression.constraints
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper

model = NonNegativeRegressionWrapper(base_estimator=your_regressor)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # Guaranteed >= 0

# Phase 9.5 Classification Meta-Features (v0.6.1+)
# Extract classification probabilities to enhance regression models
from finance_ml.ml_workflow.regression.dataset import extract_classification_features

df_with_meta = extract_classification_features(
        df: pd.DataFrame,
classifier: Any,
feature_cols: List[str]
) -> pd.DataFrame
# Returns: df with event_prob_neutral, event_prob_positive, event_prob_negative columns
# These probabilities provide sentiment and event likelihood signals to regression models

# Phase 9.5 Modular Regression Pipelines (v0.6.1+)
# Ridge, Lasso, ElasticNet, Bayesian Ridge, Gradient Boosting with improved abstraction
from finance_ml.ml_workflow.regression.models import (
    train_ridge_regressor,
    train_lasso_regressor,
    train_elasticnet_regressor,
    train_bayesian_ridge_regressor
    )
# All follow standardized return signature: {model, metrics, y_pred, feature_importance}

# finance_ml.ml_workflow.regression.dataset - Data Validation and Preparation
from finance_ml.ml_workflow.regression import (
    validate_training_data,
    prepare_features_for_training,
    extract_numeric_feature_columns,
    integrate_classification_features_into_dataframe,
    create_classification_interactions,
    train_sector_specific_models
    )

validation_result = validate_training_data(
        X: pd.DataFrame,
y: pd.Series,
strict: bool = True
) -> Dict[str, Any]
# Returns: {is_valid, issues, warnings, feature_stats}
# Validates data quality before training

X_prepared, y_prepared = prepare_features_for_training(
        df: pd.DataFrame,
feature_cols: List[str],
target_col: str,
apply_imputation: bool = True,
sector_column: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.Series]
# Returns: (X, y) with imputation applied if requested

numeric_cols = extract_numeric_feature_columns(
        df: pd.DataFrame,
exclude_cols: Optional[List[str]] = None
) -> List[str]
# Returns: List of numeric column names, excluding specified columns

df_enhanced = integrate_classification_features_into_dataframe(
        df: pd.DataFrame,
class_features: Dict[str, np.ndarray]
) -> pd.DataFrame
# Returns: df with classification probability columns added

df_interactions = create_classification_interactions(
        df: pd.DataFrame,
class_proba_cols: List[str],
value_cols: List[str]
) -> pd.DataFrame
# Returns: df with interaction features (class_prob * value)

sector_models, results = train_sector_specific_models(
        df: pd.DataFrame,
feature_cols: List[str],
target_col: str,
sector_col: str = "sector",
model_type: str = "random_forest",
min_samples: int = 20,
ensure_nonnegative: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any]]
# Returns: ({sector: model}, {metrics, n_sectors_trained, sectors_trained, fallback_used})

# finance_ml.ml_workflow.regression.models - All Model Training Functions
from finance_ml.ml_workflow.regression.models import (
    # Linear models
    train_ridge_regressor,
    train_lasso_regressor,
    train_elastic_net_regressor,
    train_bayesian_ridge_regressor,
    train_polynomial_regressor,
    # Gradient boosting
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_histgb_regressor,
    # Tree models
    train_random_forest_regressor,
    train_extra_trees_regressor,
    # Neural network
    train_neural_network_regressor,
    # Ensemble methods
    train_voting_regressor,
    train_stacking_regressor,
    )

# Linear Models - All return: {model, metrics, y_pred, feature_importance}
ridge_result = train_ridge_regressor(X_train, y_train, X_test, y_test, alpha=1.0)
lasso_result = train_lasso_regressor(X_train, y_train, X_test, y_test, alpha=1.0)
elastic_result = train_elastic_net_regressor(X_train, y_train, X_test, y_test, alpha=1.0, l1_ratio=0.5)
bayes_result = train_bayesian_ridge_regressor(X_train, y_train, X_test, y_test)
poly_result = train_polynomial_regressor(X_train, y_train, X_test, y_test, degree=2)

# Gradient Boosting Models - All return: {model, metrics, y_pred, feature_importance}
xgb_result = train_xgboost_regressor(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6)
lgbm_result = train_lightgbm_regressor(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=6)
catboost_result = train_catboost_regressor(X_train, y_train, X_test, y_test, iterations=100, depth=6)
histgb_result = train_histgb_regressor(X_train, y_train, X_test, y_test, max_iter=100, max_depth=6)

# Tree Models - All return: {model, metrics, y_pred, feature_importance}
rf_result = train_random_forest_regressor(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)
et_result = train_extra_trees_regressor(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=10)

# Neural Network - Returns: {model, metrics, y_pred, history}
nn_result = train_neural_network_regressor(
        X_train, y_train, X_test, y_test,
        hidden_layers=[64, 32],
        activation='relu',
        epochs=50,
        batch_size=32
        )

# Ensemble Methods - All return: {model, metrics, y_pred, feature_importance}
voting_result = train_voting_regressor(
        X_train, y_train, X_test, y_test,
        estimators=[('rf', rf_model), ('xgb', xgb_model), ('lgbm', lgbm_model)]
        )
stacking_result = train_stacking_regressor(
        X_train, y_train, X_test, y_test,
        base_estimators=[rf_model, xgb_model, lgbm_model],
        final_estimator=Ridge()
        )

# finance_ml.ml_workflow.regression.quantile - Quantile Regression
from finance_ml.ml_workflow.regression.quantile import train_quantile_regressor

quantile_result = train_quantile_regressor(
        X_train: pd.DataFrame,
y_train: pd.Series,
X_test: Optional[pd.DataFrame] = None,
y_test: Optional[pd.Series] = None,
quantiles: List[float] = [0.1, 0.5, 0.9],
** kwargs
) -> Dict[str, Any]
# Returns: {models: {quantile: model}, predictions: {quantile: y_pred}, metrics: {quantile: metrics}}
# Provides prediction intervals for uncertainty estimation

# finance_ml.ml_workflow.regression.tuning - Hyperparameter Optimization
from finance_ml.ml_workflow.regression.tuning import optimize_hyperparameters_optuna

tuning_result = optimize_hyperparameters_optuna(
        X_train: pd.DataFrame,
y_train: pd.Series,
model_type: str = "xgboost",
n_trials: int = 100,
cv_folds: int = 5,
optimization_metric: str = "neg_mean_absolute_error",
** kwargs
) -> Dict[str, Any]
# Returns: {best_params, best_score, study, model}
# Uses Bayesian optimization (Optuna) to find optimal hyperparameters

# finance_ml.ml_workflow.regression.io - Model Persistence
from finance_ml.ml_workflow.regression.io import save_model, load_model

save_model(
        model: Any,
filepath: Union[str, Path],
metadata: Optional[Dict[str, Any]] = None
) -> None
# Saves model with metadata (feature names, training date, version, etc.)

loaded_model, metadata = load_model(
        filepath: Union[str, Path]
) -> Tuple[Any, Dict[str, Any]]
# Returns: (model, metadata)
# Loads model and associated metadata

# finance_ml.ml_workflow.regression.constraints - Non-Negative Predictions
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper

wrapped_model = NonNegativeRegressionWrapper(base_estimator=Ridge(alpha=1.0))
wrapped_model.fit(X_train, y_train)
predictions = wrapped_model.predict(X_test)  # All predictions >= 0
# Ensures predictions are non-negative (critical for price predictions)
```

**Phase 9.6 — Evaluation**

```python
# finance_ml.ml_workflow.evaluation
from finance_ml.ml_workflow.evaluation import (
    comprehensive_regression_metrics,
    compute_metrics_by_segment
    )

metrics = comprehensive_regression_metrics(
        y_true: np.ndarray,
y_pred: np.ndarray
) -> Dict[str, float]
# Returns: {'mae': float, 'rmse': float, 'r2': float, 'mape': float, 'median_ae': float}

segment_metrics = compute_metrics_by_segment(
        y_true: pd.Series,
y_pred: pd.Series,
segment_col: pd.Series,
metrics: Optional[List[str]] = None
) -> pd.DataFrame
# Returns: DataFrame with metrics per segment
```

**Phase 9.7 — Analytics**

```python
# finance_ml.ml_workflow.analytics
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_score,
    rank_undervalued_stocks
    )

mispricing = calculate_mispricing_score(
        last_price: pd.Series,
predicted_target: pd.Series
) -> pd.Series
# Returns: (predicted_target - last_price) / last_price

undervalued_df = rank_undervalued_stocks(
        df: pd.DataFrame,
mispricing_col: str = 'mispricing_score',
top_n: int = 20
) -> pd.DataFrame
# Returns: Top N undervalued stocks sorted by mispricing score
```

**Phase 9.8 — Reporting and Analyst Comparison**

```python
# finance_ml.ml_workflow.reporting
from finance_ml.ml_workflow.reporting import (
    prepare_plotly_dashboard_data,
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts
    )

dashboard_data = prepare_plotly_dashboard_data(
        predictions_df: pd.DataFrame,
metrics_df: pd.DataFrame
) -> Dict[str, Any]
# Returns: dict with plotly-ready data structures

financial_metrics = calculate_financial_metrics_dashboard(
        df: pd.DataFrame
) -> Dict[str, Any]
# Returns: automated KPI reporting metrics (Phase 9.2 enhanced EDA)

quality_alerts = generate_data_quality_alerts(
        df: pd.DataFrame,
thresholds: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]
# Returns: data validation and quality monitoring alerts

# Phase 9.7 Analyst Comparison (v0.6.0+)
from finance_ml.ml_workflow.analytics.analyst_comparison import (
    compare_prediction_vs_analyst_targets,
    calculate_agreement_rate,
    calculate_directional_accuracy,
    analyze_systematic_bias,
    generate_prediction_analyst_excel_report
    )

comparison_df = compare_prediction_vs_analyst_targets(
        predictions: pd.Series,
analyst_targets: pd.Series,
last_prices: pd.Series,
tickers: pd.Series
) -> pd.DataFrame
# Returns: comprehensive comparison with agreement metrics

excel_report = generate_prediction_analyst_excel_report(
        comparison_df: pd.DataFrame,
output_path: str
) -> None
# Generates Excel report with formatted comparison tables and charts
```

2) Column Naming and Schema (all_stocks dataframe)

2.1 Normalized column names based on `create_equities_schema.sql schema`

- Always normalize DataFrame column names early via `finance_ml.data.normalize_columns`.
- Canonical names (must exist if relevant in your workflow):
    - last_price
    - price_target (preferred target)
    - price_target_median (optional, fallback target)
    - sector
    - region
    - ticker (identifier)
- Normalization rules:
    - Lowercase snake_case
    - Replace non-alphanumeric with underscores
    - Trim leading/trailing underscores
    - Preserve data types

2.2 Phase 9.3 Schema Version 1.3 (310 columns)

**Schema Expansion:** As of 2025-11-20, the equities schema has been expanded from 262 to 310 columns (+48, +18.3%) to
support advanced feature engineering.

**48 New Columns by Category:**

1. **Revenue Forecasting Estimates (4 columns):**
    - revenues_est_avg_ntm, revenues_est_avg_fy1e, revenues_est_med_ntm, revenues_est_med_fy1e

2. **EV/Sales Time-Series (11 columns):**
    - ev_sales_est_fy1, ev_sales_ltm, ev_sales_ntm, ev_sales_1fyltm through ev_sales_4fqltm

3. **Employment Metrics (2 columns):**
    - total_employees_fy, total_employees_fq

4. **Technical Indicators (6 columns):**
    - 52w_high_adj, 52w_low_adj, ema_20d, ema_50d, ema_100d, ema_250d

5. **EV/EBITDA Extended Time-Series (6 columns):**
    - ev_ebitda_ltm, ev_ebitda_ntm, ev_ebitda_1fyltm, ev_ebitda_1fqltm, ev_ebitda_3yavgltm, ev_ebitda_est_fy1

6. **P/E Extended Time-Series (11 columns):**
    - p_e_est_fy1, p_e_2fyltm, p_e_3fyltm, p_e_3yavgltm, plus 7 quarterly/YoY variants

7. **Dividend Record Information (8 columns):**
    - dividend_record_announce_date, dividend_record_ex_date, dividend_record_payable_date, dividend_record_record_date
    - dividend_record_frequency, dividend_record_currency, dividend_record_amount, dividend_streak

**5 New Feature Engineering Functions:**

These functions leverage the new Schema 1.3 columns and are integrated into `build_comprehensive_features()`:

1. `engineer_technical_analysis_features()` - EMA crossovers, 52W position, volume momentum, breakout signals
2. `engineer_valuation_timeseries_features()` - Valuation momentum, mean reversion, forward/trailing spreads, quarterly
   stability
3. `engineer_revenue_forecast_features()` - Analyst consensus, estimate quality, growth expectations
4. `engineer_dividend_reliability_features()` - Dividend consistency, coverage, safety scores, growth metrics
5. `engineer_employment_dynamics_features()` - Employee growth, productivity, workforce indicators

**Usage:**

```python
from finance_ml.ml_workflow.features.advanced import (
    build_comprehensive_features,
    engineer_technical_analysis_features,
    engineer_valuation_timeseries_features,
    engineer_revenue_forecast_features,
    engineer_dividend_reliability_features,
    engineer_employment_dynamics_features
    )

# Comprehensive pipeline (includes all 5 new functions)
df_features = build_comprehensive_features(df, preset="comprehensive")

# Or use individual functions
df = engineer_technical_analysis_features(df)
df = engineer_valuation_timeseries_features(df)
# ... etc
```

**Reference Documentation:**

- Complete column mapping: `phase93_new_columns_mapping.md`
- Feature enhancement plan: `docs/improvement_plan/Phase_9.3_feature_enhancement_plan.md`
- Implementation summary: `PHASE93_FEATURE_IMPLEMENTATION_SUMMARY.md`
- Feature categories table: See section 3.1 "Phase 9.3 Feature Categories (v9_9)" below

2.3 Downstream assumptions

- All modules must assume normalized names. Do not mix raw CSV header style (e.g., "Last Price" or "Price Target").
- When joining/merging, preserve index alignment and canonical names.
- Tests assume normalized columns for loaders and downstream utilities.

2.4 Validation

- Use `validate_schema(df, require_target: bool)` to assert required fields.
- For notebook/script workflows, validate after normalization and before heavy processing.

2.5 Schema Module and Helper Functions (v0.8.2+)

**Module**: `finance_ml.ml_workflow.data.schema`

**Purpose**: Centralized column schema registry providing datatype and role information for all 310+ columns, enabling
schema-aware preprocessing, validation, and feature engineering.

**COLUMN_SCHEMA Registry**: Dictionary mapping normalized column names to metadata:

```python
COLUMN_SCHEMA: Dict[str, Dict[str, str]] = {
    "ticker": {"dtype": "string", "role": "id"},
    "sector": {"dtype": "category", "role": "categorical"},
    "last_price": {"dtype": "float", "role": "feature"},
    "price_target": {"dtype": "float", "role": "target"},
    "last_updated": {"dtype": "datetime64[ns]", "role": "date"},
    # ... 310+ total columns
    }
```

**Helper Functions**:

```python
from finance_ml.ml_workflow.data.schema import (
    list_categorical_cols,  # Get all categorical column names
    list_date_cols,  # Get all date/datetime column names
    list_numeric_feature_cols,  # Get all numeric feature columns
    get_expected_dtype,  # Get expected dtype for a column
    get_column_role,  # Get role for a column (id, feature, target, etc.)
    normalize_column_name,  # Normalize column name to schema format
    COLUMN_SCHEMA,  # Access full schema registry
    PHASE93_FEATURE_INPUTS  # Phase 9.3 feature categorization
    )

# Example usage in preprocessing
categorical_columns = list_categorical_cols()
# Returns: ['sector', 'industry', 'region', 'country', 'trading_country', 'exchange', ...]

date_columns = list_date_cols()
# Returns: ['last_updated', 'income_statement_report_date', 'next_earnings', ...]

numeric_features = list_numeric_feature_cols()
# Returns: ['last_price', 'market_cap', 'enterprise_value', 'p_e_ntm', ...]

# Get metadata for specific column
dtype = get_expected_dtype("sector")  # Returns: "category"
role = get_column_role("price_target")  # Returns: "target"

# Normalize column names
normalized = normalize_column_name("Last Price")  # Returns: "last_price"
```

**Schema-Aware Preprocessing Pattern** (see Section 8.2 for notebook integration):

```python
from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_date_cols
from finance_ml.ml_workflow.features.core import preprocess_for_lightgbm

# Extract schema-based column lists
categorical_columns = list_categorical_cols()
datetime_columns = list_date_cols()

# Apply preprocessing with schema columns
X_processed, encoders = preprocess_for_lightgbm(
        df,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        drop_columns=['name', 'description'],
        return_encoders=True
        )
```

**PHASE93_FEATURE_INPUTS** (Feature Categorization):

The schema module also provides `PHASE93_FEATURE_INPUTS` dictionary categorizing columns by feature type:

```python
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS

# Available categories:
# - 'momentum': Price momentum, RSI, moving averages, return stability
# - 'valuation': P/E, P/B, EV/EBITDA, PEG ratios
# - 'profitability': Margins, ROE, ROA, ROIC
# - 'quality_risk': Altman Z-Score, beta, volatility
# - 'cash_flow': FCF, operating cash flow, capex
# - 'growth': Revenue growth, earnings growth, CAGR

momentum_cols = PHASE93_FEATURE_INPUTS['momentum']
valuation_cols = PHASE93_FEATURE_INPUTS['valuation']
```

**Benefits**:

1. **Single Source of Truth**: All column metadata centralized in one place
2. **Schema-Driven Preprocessing**: Automatic column type detection and handling
3. **Type Safety**: Consistent dtype casting based on schema definitions
4. **Testability**: Schema functions are fully tested (100% coverage, 44 tests)
5. **Maintainability**: Schema changes propagate automatically to all consumers

**Test Coverage**: See `tests/test_notebook_schema_integration.py` for 44 comprehensive tests covering all schema helper
functions and integration patterns.

**Related Documentation**:

- Schema implementation: `docs/TDD_IMPLEMENTATION_SUMMARY.md`
- Test coverage: `tests/test_data_types_detection.py`, `tests/test_enhanced_imputation_phase93.py`
- Usage examples: See Section 8.2 "Schema-Aware Preprocessing Pattern"

3) DataFrame Shape and Feature References

- Keep `all_stocks` as the single, unified DataFrame across regions.
- Clearly separate identifiers/targets from features:
    - Identifiers: ticker, sector, region
    - Targets: price_target (and optional price_target_median)
    - Features: everything else numeric/categorical after filtering
- Use helper utilities to extract feature columns; avoid hard-coded lists spread across files.

3.1 Variable Mapping Reference Table

This table documents standard variable names used across the Finance ML Analytics Platform codebase.

**Core DataFrames:**

| Variable Name                    | Type         | Description                                | Source/Usage                |
|----------------------------------|--------------|--------------------------------------------|-----------------------------|
| `all_stocks`                     | pd.DataFrame | Primary unified dataset across all regions | Data loading, main workflow |
| `all_stocks_features`            | pd.DataFrame | Dataset with engineered features           | After feature engineering   |
| `all_stocks_with_classification` | pd.DataFrame | Dataset with classification probabilities  | After classification step   |

**Dataset Splits (Classification):**

| Variable Name       | Type         | Description                                    | Source/Usage                      |
|---------------------|--------------|------------------------------------------------|-----------------------------------|
| `X_train_cls`       | pd.DataFrame | Classification training features (raw)         | `prepare_classification_data()`   |
| `X_test_cls`        | pd.DataFrame | Classification test features (raw)             | `prepare_classification_data()`   |
| `X_train_processed` | pd.DataFrame | Classification training features (numeric)     | After `preprocess_for_lightgbm()` |
| `X_test_processed`  | pd.DataFrame | Classification test features (numeric)         | After `preprocess_for_lightgbm()` |
| `y_train_cls`       | np.ndarray   | Classification training labels (0, 1, 2, 3, 4) | `prepare_classification_data()`   |
| `y_test_cls`        | np.ndarray   | Classification test labels (0, 1, 2, 3, 4)     | `prepare_classification_data()`   |
| `labels`            | np.ndarray   | Event labels for entire dataset                | `create_enhanced_event_labels()`  |

**Dataset Splits (Regression):**

| Variable Name | Type           | Description                              | Source/Usage                |
|---------------|----------------|------------------------------------------|-----------------------------|
| `X_train`     | pd.DataFrame   | Regression training features             | `prepare_regression_data()` |
| `X_test`      | pd.DataFrame   | Regression test features                 | `prepare_regression_data()` |
| `y_train`     | pd.Series      | Regression training targets              | `prepare_regression_data()` |
| `y_test`      | pd.Series      | Regression test targets                  | `prepare_regression_data()` |
| `meta`        | Dict[str, Any] | Metadata (feature names, encoders, etc.) | Dataset prep functions      |

**Model Training Results:**

| Variable Name | Type           | Description                                      | Source/Usage                                   |
|---------------|----------------|--------------------------------------------------|------------------------------------------------|
| `result`      | Dict[str, Any] | Training result with model, metrics, predictions | `train_*()` functions                          |
| `cls_model`   | Classifier     | Trained classification model                     | `result['model']` from classification          |
| `reg_model`   | Regressor      | Trained regression model                         | `result['model']` from regression              |
| `y_pred`      | np.ndarray     | Predicted values                                 | `result['y_pred']` or `model.predict()`        |
| `y_proba`     | np.ndarray     | Class probabilities (classification)             | `result['y_proba']` or `model.predict_proba()` |
| `y_proba_all` | np.ndarray     | Probabilities for entire dataset                 | After full dataset prediction                  |

**Column Names (Canonical):**

| Variable Name         | Type  | Description                            | Source/Usage        |
|-----------------------|-------|----------------------------------------|---------------------|
| `last_price`          | float | Current stock price                    | Database/CSV column |
| `price_target`        | float | Analyst price target (primary)         | Database/CSV column |
| `price_target_median` | float | Median price target (fallback)         | Database/CSV column |
| `ticker`              | str   | Stock ticker symbol                    | Database/CSV column |
| `sector`              | str   | Stock sector                           | Database/CSV column |
| `region`              | str   | Geographic region (US, EU, APAC, ROTW) | Database/CSV column |
| `industry`            | str   | Stock industry                         | Database/CSV column |
| `market_cap`          | float | Market capitalization                  | Database/CSV column |

**Feature Categories:**

| Variable Name         | Type      | Description                        | Source/Usage                                                                                                                       |
|-----------------------|-----------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `numeric_cols`        | List[str] | Numeric feature column names       | `prepare_classification_data()`                                                                                                    |
| `categorical_cols`    | List[str] | Categorical feature column names   | `prepare_classification_data()`                                                                                                    |
| `classification_cols` | List[str] | Classification probability columns | `['event_prob_strong_negative', 'event_prob_negative', 'event_prob_neutral', 'event_prob_positive', 'event_prob_strong_positive']` |
| `valuation_cols`      | List[str] | Valuation metric columns           | `['p_e', 'p_b', 'ev_ebitda', 'market_cap']`                                                                                        |
| `financial_metrics`   | List[str] | Financial metric columns           | Various analysis contexts                                                                                                          |

**Phase 9.3 Feature Categories (v9_9):**

| Category                 | Example Features                                                                                  | Description                                 |
|--------------------------|---------------------------------------------------------------------------------------------------|---------------------------------------------|
| **Momentum & Technical** | `price_momentum_1m`, `price_momentum_3m`, `rsi_14d`, `rsi_30d`, `ma_crossover_signal`             | Price momentum, RSI, moving averages        |
| **Valuation Ratios**     | `p_e_ratio`, `p_b_ratio`, `ev_ebitda_ratio`, `peg_ratio`, `p_s_ratio`                             | Comprehensive valuation metrics             |
| **Profitability**        | `gross_margin_pct`, `operating_margin_pct`, `net_margin_pct`, `roe`, `roa`, `roic`                | Margin and return metrics                   |
| **Quality & Risk**       | `altman_z_score`, `accounting_quality_score`, `exceptional_items_to_ebitda`, `return_stability`   | Financial distress and quality signals      |
| **Analyst Sentiment**    | `upside_potential`, `analyst_bullish_pct`, `analyst_coverage_quality`, `price_target_range`       | Analyst consensus and price target features |
| **Market Sentiment**     | `short_interest_ratio`, `systematic_risk_trend`, `beta_stability`                                 | Market and systematic risk indicators       |
| **Cash Flow**            | `cfo_to_net_income`, `fcf_margin`, `cfo_growth_yoy`, `cash_conversion_cycle`                      | Cash flow quality and efficiency            |
| **Capital Allocation**   | `capex_intensity`, `total_shareholder_return_yield`, `reinvestment_rate`, `acquisition_intensity` | Capital deployment metrics                  |
| **Leverage & Liquidity** | `debt_to_equity`, `current_ratio`, `cash_ratio`, `working_capital_ratio`                          | Balance sheet strength indicators           |
| **Temporal Patterns**    | `ltm_vs_5yavg_revenue`, `fq_vs_5yavg_ebitda`, `days_to_earnings`, `reporting_lag`                 | Time-series and seasonality features        |
| **Composite Scores**     | `quality_score_composite`, `growth_quality_score`, `value_score`, `momentum_score`                | Multi-metric aggregated signals             |

**Configuration and Metadata:**

| Variable Name   | Type                    | Description                  | Source/Usage                                    |
|-----------------|-------------------------|------------------------------|-------------------------------------------------|
| `encoders`      | Dict[str, LabelEncoder] | Categorical feature encoders | `preprocess_for_lightgbm(return_encoders=True)` |
| `quality_stats` | Dict[str, Any]          | Data quality statistics      | `prepare_phase91_data(return_stats=True)`       |
| `config`        | FinanceMLConfig         | Configuration object         | `get_config()` from finance_ml.config           |

### Schema and Datatype Management (v1.3+)

All data loading and preprocessing **MUST** respect a centralized column schema defined in
`finance_ml.ml_workflow.data.schema`.

**Core Schema Components:**

- **`COLUMN_SCHEMA`**: Authoritative registry mapping 350+ normalized column names to their expected dtype and role
    - `dtype`: one of `float`, `int`, `string`, `category`, `datetime64[ns]`
    - `role`: one of `id`, `feature`, `target`, `target_fallback`, `date`, `auxiliary`
    - Derived directly from `create_equities_schema.sql` with normalized column names

- **`PHASE93_FEATURE_INPUTS`**: Phase 9.3 feature engineering categorization
    - `momentum`: Price changes, EMAs, returns, technical indicators
    - `valuation`: P/E, P/B, EV ratios, market cap, valuation metrics
    - `profitability`: Margins, EBITDA, EBIT, net income, profitability ratios
    - `quality_risk`: Altman Z-Score, ROE, ROA, beta, volatility, quality indicators
    - `cash_flow`: CFO, FCF, CFI, CFF, capex, cash flow quality metrics
    - `growth`: Revenue CAGR, return CAGR, growth trajectories

**Helper Functions:**

```python
from finance_ml.ml_workflow.data.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_INPUTS,
    get_expected_dtype,
    get_column_role,
    list_numeric_feature_cols,
    list_categorical_cols,
    list_date_cols,
    normalize_column_name
    )

# Get expected dtype for a column
dtype = get_expected_dtype('last_price')  # Returns 'float'

# Get column role
role = get_column_role('sector')  # Returns 'categorical'

# Get all numeric feature columns
numeric_cols = list_numeric_feature_cols()

# Get Phase 9.3 momentum features
momentum_features = PHASE93_FEATURE_INPUTS['momentum']
```

**Datatype Detection and Casting:**

All CSV and DB imports **MUST** call `detect_and_cast_dtypes()` before any modeling logic or imputation:

```python
from finance_ml.ml_workflow.preprocessing.dtypes import detect_and_cast_dtypes

# Cast DataFrame to schema-compliant dtypes
df_cast, diagnostics = detect_and_cast_dtypes(df)

# Diagnostics include:
# - inferred_dtypes: {col: str} - dtypes before casting
# - cast_applied: {col: str} - columns that were cast
# - coercion_warnings: {col: int} - number of values coerced to NaN
# - unknown_columns: list[str] - columns not in schema
# - missing_expected_columns: list[str] - expected columns not in df
```

**Metadata Requirements:**

- All type coercions (e.g., invalid numerics converted to NaN) must be tracked via diagnostics
- Coercion counts and unknown columns must be logged and surfaced in metadata (`*_metadata.json`)
- Metadata files must include `dtypes` and `missing_counts` sections aligned with schema expectations

**Testing Requirements:**

- Schema compliance tests: `tests/test_data_types_detection.py`
- Metadata validation tests: `tests/test_metadata_catalog_quality.py`
- All loaders must have tests verifying schema-aware dtype casting

**Common Parameters:**

| Parameter Name | Type  | Default        | Description                                  |
|----------------|-------|----------------|----------------------------------------------|
| `test_size`    | float | 0.2            | Train/test split ratio                       |
| `random_state` | int   | 42             | Random seed for reproducibility              |
| `n_trials`     | int   | 50             | Number of hyperparameter optimization trials |
| `cv_folds`     | int   | 5              | Cross-validation folds                       |
| `target_col`   | str   | 'price_target' | Target column name for regression            |

4) Typing, Logging, and Errors

- Typing: add type hints for public APIs and internal utilities.
- Logging: prefer `logging` over prints. For notebooks, prints are ok for user feedback, but underlying package
  functions should log.
- Errors: raise specific exceptions; avoid broad `except Exception` unless re-raising with context. Provide actionable
  messages.

5) Reproducibility and Configuration

- Respect environment variables and config objects (e.g., RANDOM_SEED, N_JOBS, DATA_DIR, DB_URL).
- Avoid hard-coded paths; use `pathlib.Path` and config.
- Document default behavior in docstrings.

6) Testing Conventions

- Use unittest; keep tests deterministic and fast where possible.
- Mock external services (DB) for unit tests.
- Provide small sample data for functional tests.
- Ensure coverage for normalization, dataset prep returns, and train_* result schema.

7) Notebook and CLI Alignment

- Notebooks and CLI should:
    - Normalize columns immediately after loading using `normalize_columns()`
    - Validate schema with `validate_schema()` using canonical names
    - Expect training functions to return the standardized dict and read metrics via `res["metrics"]`
- Maintain light wrapper logic in notebooks; delegate work to `finance_ml` package APIs.

8) Notebook Best Practices and TDD Conventions

This section codifies best practices for notebook development following Test-Driven Development (TDD) principles,
ensuring maintainability, testability, and consistency across the project.

8.1) Centralized Configuration Constants (Single Source of Truth)

**Policy**: All configuration constants must be defined once in a dedicated configuration cell at the top of the
notebook. No magic numbers or duplicate constant definitions are allowed in subsequent cells.

**Required Configuration Constants**:

```python
# ========== CONFIGURATION CONSTANTS ==========
# Section 8.1: Single Source of Truth - All constants defined once

# Target and fallback (Section 2.2)
TARGET_COL = 'price_target'  # Canonical target (code_guidelines.md Section 2.2)
TARGET_COL_FALLBACK = 'last_price'  # Canonical fallback target

# Data splits
TEST_SIZE = 0.2  # Train/test split ratio
TRAIN_SIZE = 1 - TEST_SIZE  # Training set size (computed from TEST_SIZE)
CV_FOLDS = 5  # Cross-validation folds

# Quantile regression
QUANTILES = [0.1, 0.5, 0.9]  # Lower, median, upper quantiles
LOWER_QUANTILE = QUANTILES[0]
MEDIAN_QUANTILE = QUANTILES[1]
UPPER_QUANTILE = QUANTILES[2]

# Sector constraints
MIN_SECTOR_SAMPLES = 20  # Minimum samples required per sector

# Portfolio constraints
MAX_SECTOR_WEIGHT = 0.25  # Maximum portfolio weight per sector (25%)
MAX_SINGLE_POSITION = 0.10  # Maximum weight for single position (10%)

# Outlier thresholds
IQR_MULTIPLIER = 1.5  # IQR multiplier for outlier detection
ZSCORE_THRESHOLD = 3.0  # Z-score threshold for outliers
WINSORIZE_LOWER = 0.01  # Lower percentile for winsorization (1%)
WINSORIZE_UPPER = 0.99  # Upper percentile for winsorization (99%)

# Confidence scoring
CONFIDENCE_LOW_THRESHOLD = 0.50  # Low confidence threshold
CONFIDENCE_MEDIUM_THRESHOLD = 0.75  # Medium confidence threshold

# Random seed
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))
np.random.seed(RANDOM_SEED)
```

**Validation Function** (recommended):

```python
def validate_configuration():
    """
    Validate notebook configuration constants.
    
    Ensures all configuration values are properly defined and within valid ranges.
    This provides a single source of truth validation for all configuration.
    
    Returns:
        bool: True if all validations pass
    
    Raises:
        ValueError: If any configuration value is invalid
    """
    # Validate target columns
    if not TARGET_COL or not isinstance(TARGET_COL, str):
        raise ValueError(f"TARGET_COL must be a non-empty string, got: {TARGET_COL}")
    if not TARGET_COL_FALLBACK or not isinstance(TARGET_COL_FALLBACK, str):
        raise ValueError(f"TARGET_COL_FALLBACK must be a non-empty string, got: {TARGET_COL_FALLBACK}")

    # Validate test size
    if not (0 < TEST_SIZE < 1):
        raise ValueError(f"TEST_SIZE must be between 0 and 1, got: {TEST_SIZE}")
    if not (0 < TRAIN_SIZE < 1):
        raise ValueError(f"TRAIN_SIZE must be between 0 and 1, got: {TRAIN_SIZE}")

    # Validate CV folds
    if not isinstance(CV_FOLDS, int) or CV_FOLDS < 2:
        raise ValueError(f"CV_FOLDS must be an integer >= 2, got: {CV_FOLDS}")

    # Validate quantiles
    if not all(0 <= q <= 1 for q in QUANTILES):
        raise ValueError(f"All QUANTILES must be between 0 and 1, got: {QUANTILES}")

    # Validate portfolio constraints
    if not (0 < MAX_SECTOR_WEIGHT <= 1):
        raise ValueError(f"MAX_SECTOR_WEIGHT must be between 0 and 1, got: {MAX_SECTOR_WEIGHT}")
    if not (0 < MAX_SINGLE_POSITION <= 1):
        raise ValueError(f"MAX_SINGLE_POSITION must be between 0 and 1, got: {MAX_SINGLE_POSITION}")

    return True


# Execute validation
validate_configuration()
```

**Usage in Downstream Cells**:

```python
# ✅ CORRECT: Use configuration constants
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,  # Not 0.2
        random_state=RANDOM_SEED,  # Not 42
        stratify=sectors
        )

# ✅ CORRECT: Use target column constants
target_col = TARGET_COL if TARGET_COL in df.columns else TARGET_COL_FALLBACK
y = df[target_col]

# ✅ CORRECT: Use quantile constants
quantile_predictions = predict_quantiles(
        model, X,
        quantiles=QUANTILES,  # Not [0.05, 0.5, 0.95]
        random_state=RANDOM_SEED
        )

# ✅ CORRECT: Use portfolio constraints
optimized_weights = optimize_portfolio(
        returns, cov_matrix,
        max_sector_weight=MAX_SECTOR_WEIGHT,  # Not 0.25
        max_single_position=MAX_SINGLE_POSITION  # Not 0.10
        )

# ❌ WRONG: Hardcoded values (magic numbers)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Violation
y = df['price_target']  # Violation: use TARGET_COL
quantiles = [0.05, 0.5, 0.95]  # Violation: use QUANTILES
max_weight = 0.25  # Violation: use MAX_SECTOR_WEIGHT
```

**Benefits**:

- Single source of truth: Changes propagate automatically
- Testability: Constants can be validated programmatically
- Reproducibility: Clear documentation of all tunable parameters
- Maintainability: Easy to audit and update configuration

8.2) Schema-Aware Preprocessing Pattern (v0.8.2+)

**Policy**: Use the schema module helper functions to extract categorical and date columns for preprocessing pipelines.
This ensures consistency with the centralized COLUMN_SCHEMA registry and eliminates hardcoded column lists.

**Required Pattern for Notebook Preprocessing**:

```python
# Import schema helper functions at the top of preprocessing section
from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_date_cols

# Extract schema-based column lists
categorical_columns_from_schema = list_categorical_cols()
datetime_cols_from_schema = list_date_cols()

# Use in preprocessing pipeline
prep_params = {
    'cat_cols': categorical_columns_from_schema,
    'date_cols': datetime_cols_from_schema,
    'drop_cols': auxiliary_cols_to_drop,  # ['name', 'description', etc.]
    'encoders': encoders,
    'ref_date': reference_date
    }

# Apply preprocessing
X_processed, encoders = preprocess_for_lightgbm(
        df,
        categorical_columns=prep_params['cat_cols'],
        datetime_columns=prep_params['date_cols'],
        drop_columns=prep_params['drop_cols'],
        encoders=prep_params['encoders'],
        reference_date=prep_params['ref_date']
        )
```

**Complete Example (Notebook Classification Section)**:

```python
#%% Classification Preprocessing
from finance_ml.ml_workflow.data.schema import list_categorical_cols, list_date_cols
from finance_ml.ml_workflow.features.core import preprocess_for_lightgbm

# 1. Extract schema-based column lists
categorical_columns_from_schema = list_categorical_cols()
datetime_cols_from_schema = list_date_cols()

print(f"[INFO] Schema integration:")
print(f"  Categorical columns from schema: {len(categorical_columns_from_schema)}")
print(f"  Date columns from schema: {len(datetime_cols_from_schema)}")

# 2. Define auxiliary columns to drop
auxiliary_cols_to_drop = ['name', 'description', 'flag', 'unit']

# 3. Prepare data for model prediction/inference
def _prepare_inference_data(df_raw, raw_train_columns, model_feature_names, prep_params):
    """
    Preprocess raw data and align exactly to model features for inference.
    
    Args:
        df_raw: DataFrame containing all raw features
        raw_train_columns: List of columns expected by preprocessing (from training)
        model_feature_names: List of features expected by trained model
        prep_params: Dict with cat_cols, date_cols, drop_cols, encoders, ref_date
    """
    # Step 1: Align raw columns to training structure
    X_raw = df_raw.reindex(columns=raw_train_columns)
    
    # Step 2: Apply preprocessing using training encoders (Inference Mode)
    X_processed, _ = preprocess_for_lightgbm(
        X_raw.copy(),
        categorical_columns=prep_params['cat_cols'],
        datetime_columns=prep_params['date_cols'],
        drop_columns=prep_params['drop_cols'],
        encoders=prep_params['encoders'],
        reference_date=prep_params['ref_date']
    )
    
    # Step 3: Align processed data to model schema (handle missing/extra columns)
    missing_cols = set(model_feature_names) - set(X_processed.columns)
    extra_cols = set(X_processed.columns) - set(model_feature_names)
    
    if missing_cols:
        for col in missing_cols:
            X_processed[col] = 0
    
    if extra_cols:
        X_processed = X_processed.drop(columns=list(extra_cols))
    
    # Step 4: Final reorder to match model expectation
    return X_processed[model_feature_names]

# 4. Prepare preprocessing parameters
prep_params = {
    'cat_cols': categorical_columns_from_schema,
    'date_cols': datetime_cols_from_schema,
    'drop_cols': auxiliary_cols_to_drop,
    'encoders': encoders,  # From training
    'ref_date': reference_date
}

# 5. Apply preprocessing for inference
X_all_processed = _prepare_inference_data(
    all_stocks_features,
    X_train_cls.columns,  # Columns from training
    model_feature_names,   # Features expected by model
    prep_params
)

print(f"  [OK] Final shape aligned to model: {X_all_processed.shape}")
```

**Benefits**:

1. **Schema Consistency**: Automatic synchronization with COLUMN_SCHEMA registry
2. **Maintainability**: Column list changes propagate automatically from schema
3. **Type Safety**: Schema-aware preprocessing ensures correct dtype handling
4. **Testability**: Schema functions are fully tested (100% coverage)
5. **Reduced Errors**: Eliminates hardcoded column lists that can become stale

**Anti-Patterns (Avoid)**:

```python
# ❌ WRONG: Hardcoded categorical columns list
categorical_columns = ['sector', 'industry', 'region', 'country']  # Becomes stale

# ❌ WRONG: Undefined variable usage
prep_params = {
    'cat_cols': categorical_columns_from_schema,  # NameError if not defined!
    'date_cols': datetime_cols_from_schema
    }

# ❌ WRONG: Direct column specification without schema
X_processed, _ = preprocess_for_lightgbm(
        df,
        categorical_columns=['sector', 'industry'],  # Incomplete, hardcoded
        datetime_columns=['last_updated']  # Missing other date columns
        )
```

**Test Coverage**: The schema integration pattern is validated by 44 tests in
`tests/test_notebook_schema_integration.py`, covering:

- Schema helper function behavior
- Integration with preprocessing pipeline
- Edge cases (empty lists, unknown columns)
- Consistency across multiple calls

**Related Sections**:

- Section 2.5: Schema Module and Helper Functions (complete API documentation)
- Section 1.3: Phase 9.3 Features module (preprocessing functions)

8.3) DataFrame Stage Naming Convention

**Policy**: Use descriptive, stage-based naming for DataFrames instead of in-place mutations. Each transformation
stage should produce a new DataFrame with a name that clearly indicates the pipeline stage.

**Standard Stage Naming Pattern**:

```python
# ✅ CORRECT: Stage-based naming with clear progression
# Data Loading Options:
# - load_from_csv(): Load from CSV files in data/ directory
# - load_from_db(): Load from equities table with Region filter
# - load_from_all_stocks(): Load from unified all_stocks table (RECOMMENDED)
all_stocks_raw = load_from_all_stocks(db_url)  # Initial load from unified table
all_stocks_normalized = normalize_columns(all_stocks_raw)  # After normalization
all_stocks_typed = detect_and_cast_dtypes(all_stocks_normalized)[0]  # After type detection
all_stocks_winsorized = winsorize_by_sector(all_stocks_typed, ...)  # After winsorization
all_stocks_imputed = apply_enhanced_imputation_strategy_6step(all_stocks_winsorized, ...)  # After imputation
all_stocks_scaled = scale_features(all_stocks_imputed, ...)  # After scaling
all_stocks_features = build_comprehensive_features(all_stocks_scaled, ...)  # After feature engineering
all_stocks_enhanced = add_classification_probabilities(all_stocks_features, ...)  # After classification features

# ❌ WRONG: In-place mutations (unclear pipeline progression)
all_stocks = load_from_all_stocks(db_url)
all_stocks = normalize_columns(all_stocks)  # Lost reference to raw data
all_stocks = winsorize_by_sector(all_stocks, ...)  # Can't rollback to normalized
all_stocks = apply_imputation(all_stocks, ...)  # Can't compare pre/post imputation
```

**Required Stage Names** (in order):

1. `all_stocks_raw` — Initial data loaded from CSV or database (before any transformations)
2. `all_stocks_normalized` — After column name normalization (`normalize_columns()`)
3. `all_stocks_typed` — After datatype detection and casting (`detect_and_cast_dtypes()`)
4. `all_stocks_winsorized` — After outlier winsorization (`winsorize_by_sector()`)
5. `all_stocks_imputed` — After missing value imputation (6-step strategy)
6. `all_stocks_scaled` — After feature scaling (`scale_features()`)
7. `all_stocks_features` — After feature engineering (`build_comprehensive_features()`)
8. `all_stocks_enhanced` — After adding classification probabilities or meta-features

**Validation Checkpoints** (recommended after each stage):

```python
# After preprocessing
assert 'all_stocks_scaled' in globals(), "❌ Missing all_stocks_scaled"
assert all_stocks_scaled.shape[0] > 0, "❌ Empty DataFrame after preprocessing"
assert all_stocks_scaled[TARGET_COL].notna().sum() > 0, f"❌ No valid targets in {TARGET_COL}"

# After feature engineering
assert 'all_stocks_features' in globals(), "❌ Missing all_stocks_features"
expected_features = ['p_e_ratio', 'ev_ebitda', 'roe', 'roa']
missing_features = [f for f in expected_features if f not in all_stocks_features.columns]
assert len(missing_features) == 0, f"❌ Missing features: {missing_features}"
```

**Benefits**:

- Debugging: Can inspect intermediate stages (e.g., compare `all_stocks_imputed` vs `all_stocks_scaled`)
- Rollback: Can revert to previous stage without re-running entire pipeline
- Testing: Each stage can be validated independently
- Documentation: Pipeline progression is self-documenting in code

8.3) Magic Numbers Policy

**Policy**: All numeric literals with semantic meaning must be defined as named constants. Only use inline literals
for universally understood values (e.g., 0, 1, 100 for percentage conversions) or when the value is used exactly once
in a highly localized context.

**Prohibited Magic Numbers** (must use constants):

```python
# ❌ WRONG: Magic numbers without semantic names
random_state = 42  # Use RANDOM_SEED
test_size = 0.2  # Use TEST_SIZE
split_idx = int(len(X) * 0.8)  # Use TRAIN_SIZE or (1 - TEST_SIZE)
max_sector_weight = 0.25  # Use MAX_SECTOR_WEIGHT
quantiles = [0.05, 0.5, 0.95]  # Use QUANTILES
threshold = 1.5  # Use IQR_MULTIPLIER
lower = 0.01, upper = 0.99  # Use WINSORIZE_LOWER, WINSORIZE_UPPER

# ✅ CORRECT: Use named constants
random_state = RANDOM_SEED
test_size = TEST_SIZE
split_idx = int(len(X) * TRAIN_SIZE)
max_sector_weight = MAX_SECTOR_WEIGHT
quantiles = QUANTILES
threshold = IQR_MULTIPLIER
lower = WINSORIZE_LOWER, upper = WINSORIZE_UPPER
```

**Allowed Inline Literals** (universally understood or highly localized):

```python
# ✅ ALLOWED: Universal mathematical constants or type conversions
percentage = value * 100  # Converting to percentage
zero_fill = np.zeros(len(df))  # Creating zero array
identity_matrix = np.eye(n)  # Identity matrix
correlation_base = 0.8  # Matrix construction (with comment explaining purpose)

# ✅ ALLOWED: Highly localized, single-use values with clear context
if len(sector_group) < 5:  # Minimum group size for this specific check
    warnings.warn(f"Small sector group: {len(sector_group)} samples")

# Note: If the value appears more than once or has domain-specific meaning, use a constant
```

**Special Case: Correlation Matrix Construction**:

```python
# Correlation matrices often use mathematical weights that are part of the algorithm
# These should be documented with comments explaining their purpose
corr_matrix = np.eye(n_stocks) * 0.8 + np.ones((n_stocks, n_stocks)) * 0.2
# ^^ 0.8 = diagonal weight (self-correlation), 0.2 = off-diagonal weight (cross-correlation)
# These are algorithm parameters, not configuration values
```

**Testing Constants Usage**:

The project includes `tests/test_notebook_tdd_compliance.py` which validates:

- No hardcoded `random_state=42` (must use `RANDOM_SEED`)
- No hardcoded `test_size=0.2` (must use `TEST_SIZE`)
- No hardcoded train size `0.8` (must use `TRAIN_SIZE`)
- No hardcoded quantile lists (must use `QUANTILES`)
- No hardcoded sector weights (must use `MAX_SECTOR_WEIGHT`)
- No hardcoded column names like `'price_target'` (must use `TARGET_COL`)

**Benefits**:

- Global tuning: Change one constant to affect all usages
- Documentation: Constants serve as self-documenting parameters
- Testing: Can validate constant usage programmatically
- Code review: Easier to identify and audit configuration values

9) Comprehensive Import Examples by Phase

This section provides complete import patterns for each phase of the ML workflow, showing both new modular imports and
backward-compatible top-level imports.

**Phase 9.1 — Data Preprocessing and Quality**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_4step,
    apply_zero_imputation,
    apply_knn_imputation_enhanced,
    apply_median_imputation
    )
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    winsorize_by_sector
    )
from finance_ml.ml_workflow.preprocessing.quality import (
    DataQualityReport,
    calculate_data_quality_score
    )
from finance_ml.ml_workflow.preprocessing.scaling import (
    create_scaler_pipeline,
    scale_features
    )

# Backward-compatible top-level imports
from finance_ml import (
    prepare_phase91_data,
    apply_enhanced_imputation_strategy_4step,
    DataQualityReport,
    calculate_data_quality_score
    )

# Data loading and validation
from finance_ml.ml_workflow.data import (
    load_from_csv,
    load_from_db,
    normalize_columns,
    validate_schema,
    preprocess
    )

# Data catalog and versioning
from finance_ml.ml_workflow.data_catalog import (
    DataCatalog,
    extract_schema_info,
    create_statistical_profile
    )
from finance_ml.ml_workflow.data_versioning import (
    DataVersionManager,
    create_version_snapshot
    )
```

**Phase 9.2 — Exploratory Data Analysis**

```python
# EDA and benchmarking
from finance_ml.ml_workflow.eda import (
    generate_eda_report,
    calculate_correlation_matrix,
    perform_pca
    )
from finance_ml.ml_workflow.benchmarking import (
    compare_sector_distributions,
    compare_regional_valuations,
    find_peer_group,
    generate_benchmarking_report
    )

# Advanced EDA
from finance_ml.ml_workflow.advanced_eda import (
    EDAReport,
    test_normality,
    calculate_skewness_kurtosis,
    detect_outliers_statistical,
    calculate_mutual_information,
    compare_sector_means
    )

# Backward-compatible imports
from finance_ml import (
    generate_eda_report,
    compare_sector_distributions,
    simple_eda
    )
```

**Phase 9.3 — Feature Engineering**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.features.core import (
    preprocess_for_lightgbm,
    engineer_basic_ratios,
    engineer_margin_features,
    engineer_volatility_features,
    engineer_revenue_cagr,
    build_features_and_target
    )
from finance_ml.ml_workflow.features.advanced import (
    engineer_valuation_ratios,
    engineer_profitability_ratios,
    engineer_leverage_ratios,
    engineer_sector_specific_features,
    engineer_analyst_quality_features,
    build_comprehensive_features
    )
from finance_ml.ml_workflow.features.selection import (
    calculate_feature_importance_mutual_info,
    calculate_feature_importance_rf,
    calculate_feature_importance_shap,
    calculate_feature_importance_rfe
    )

# Backward-compatible imports (with aliases)
from finance_ml import (
    features_basic_ratios as engineer_basic_ratios,
    features_build_comprehensive as build_comprehensive_features,
    features_importance_rf as calculate_feature_importance_rf
    )
```

**Phase 9.3 Data Prerequisites (v1.3+)**

Before calling any Phase 9.3 feature engineering functions (e.g., `engineer_momentum_features`,
`engineer_cash_flow_quality_features`, `build_comprehensive_features`), input dataframes **MUST**:

1. **Schema Compliance:**
    - Conform to `COLUMN_SCHEMA` dtypes via `detect_and_cast_dtypes()`
    - All required columns for target feature buckets must be present with correct dtypes
    - Numeric features must be `float` or `int`, dates must be `datetime64[ns]`, categoricals must be `string` or
      `category`

2. **Imputation Completeness:**
    - Be fully imputed via `apply_enhanced_imputation_strategy_6step()`
    - Zero missing values in Phase 9.3 core input features (momentum, valuation, profitability, quality/risk, cash flow,
      growth)
    - Validated via `validate_imputation_completeness()` before feature engineering

3. **Safety Rails:**
    - Satisfy non-negativity constraints for price, market cap, revenues, cash flows
    - Satisfy outlier safety rails (winsorization, clipping) for all Phase 9.3 core inputs
    - No infinite values (replaced with NaN during imputation)

4. **Metadata Availability:**
    - `quality_stats` including per-column missingness and imputation volume
    - Catalog metadata (`*_metadata.json`) with `dtypes` and `missing_counts` sections
    - Schema validation diagnostics from `detect_and_cast_dtypes()`

**Testing Requirements:**

- New tests for Phase 9.3 features must confirm that feature functions **do not perform their own imputation**
- Instead, assume the standardized preprocessing pipeline (Phase 9.1) has already been executed
- Feature engineering tests should use pre-imputed fixtures or mock data
- Tests: `tests/test_features.py`, `tests/test_advanced_features.py`, `tests/test_finance_ml_features.py`

**Phase 9.4 — Classification**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.classification.labels import (
    create_enhanced_event_labels
    )
from finance_ml.ml_workflow.classification.tuning import (
    optimize_classifier_hyperparameters,
    cross_validate_with_sector_stratification
    )
from finance_ml.ml_workflow.classification.models import (
    train_xgboost_classifier,
    train_lightgbm_classifier,
    train_catboost_classifier,
    train_voting_classifier,
    train_stacking_classifier,
    prepare_classification_data
    )
from finance_ml.ml_workflow.classification.evaluation import (
    evaluate_classification,
    plot_confusion_matrices,
    analyze_calibration
    )

# Backward-compatible imports (with aliases)
from finance_ml import (
    classification_create_enhanced_event_labels as create_enhanced_event_labels,
    classification_optimize_hyperparameters as optimize_classifier_hyperparameters,
    classification_cross_validate_sector as cross_validate_with_sector_stratification
    )

# Legacy imports (deprecated but still working)
from finance_ml.ml_workflow.classification import (
    create_enhanced_event_labels,
    train_xgboost_classifier,
    compare_classifiers
    )
```

**Phase 9.5 — Regression**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data,
    create_classification_interactions,
    extract_numeric_feature_columns,
    train_sector_specific_models
    )
from finance_ml.ml_workflow.regression.models import (
    train_ridge_regressor,
    train_lasso_regressor,
    train_xgboost_regressor,
    train_lightgbm_regressor,
    train_catboost_regressor,
    train_random_forest_regressor,
    train_voting_regressor,
    train_stacking_regressor,
    compare_regressors
    )
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper
    )
from finance_ml.ml_workflow.regression.quantile import (
    train_quantile_regressor
    )
from finance_ml.ml_workflow.regression.tuning import (
    optimize_hyperparameters_optuna
    )
from finance_ml.ml_workflow.regression.io import (
    save_model,
    load_model
    )

# Backward-compatible imports (with aliases)
from finance_ml import (
    regression_prepare_data as prepare_regression_data,
    regression_train_xgboost as train_xgboost_regressor,
    regression_nonnegative_wrapper as NonNegativeRegressionWrapper,
    regression_compare_regressors as compare_regressors
    )

# Legacy imports (deprecated)
from finance_ml.ml_workflow.advanced_models import (
    prepare_regression_data,
    train_xgboost_regressor,
    compare_regressors,
    save_model
    )
```

**Phase 9.6 — Evaluation**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.evaluation import (
    comprehensive_regression_metrics,
    compute_metrics_by_segment,
    compute_sector_region_metrics
    )

# Backward-compatible imports
from finance_ml import (
    evaluation_comprehensive_metrics as comprehensive_regression_metrics,
    evaluation_metrics_by_segment as compute_metrics_by_segment
    )
```

**Phase 9.7 — Analytics**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_score,
    calculate_risk_adjusted_mispricing,
    rank_undervalued_stocks,
    rank_overvalued_stocks,
    rank_stocks_by_sector,
    )
from finance_ml.ml_workflow.analytics.eval import (
    simple_eda,
    create_sector_heatmap,
    create_interactive_prediction_plot,
    )
from finance_ml.ml_workflow.analytics.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
    calculate_portfolio_sharpe_ratio,
    generate_efficient_frontier,
    optimize_portfolio_max_sharpe,
    optimize_portfolio_min_volatility,
    optimize_portfolio_target_return,
    rebalance_portfolio,
    )
from finance_ml.ml_workflow.analytics.risk import (
    calculate_var_historical,
    calculate_var_parametric,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_portfolio_risk_metrics,
    )

# Backward-compatible imports
from finance_ml import (
    analytics_calculate_mispricing as calculate_mispricing_score,
    analytics_rank_undervalued as rank_undervalued_stocks,
    calculate_mispricing_score,
    rank_undervalued_stocks
    )
```

**Phase 9.8 — Reporting and Analyst Comparison**

```python
# New modular imports (preferred)
from finance_ml.ml_workflow.reporting import (
    calculate_financial_metrics_dashboard,
    generate_data_quality_alerts,
    prepare_plotly_dashboard_data
    )
from finance_ml.ml_workflow.analyst_comparison import (
    compare_prediction_vs_analyst_targets,
    calculate_agreement_rate,
    calculate_directional_accuracy,
    analyze_systematic_bias,
    generate_prediction_analyst_excel_report
    )

# Backward-compatible imports
from finance_ml import (
    reporting_financial_metrics as calculate_financial_metrics_dashboard,
    compare_prediction_vs_analyst_targets,
    generate_prediction_analyst_excel_report
    )
```

**Complete Workflow Import Pattern (Recommended)**

```python
# Single import block for complete ML workflow
import pandas as pd
import numpy as np

# Data loading and preprocessing (Phase 9.1)
from finance_ml.ml_workflow.data import load_from_csv, normalize_columns, validate_schema
from finance_ml.ml_workflow.preprocessing.pipeline import prepare_phase91_data

# Feature engineering (Phase 9.3)
from finance_ml.ml_workflow.features.core import preprocess_for_lightgbm
from finance_ml.ml_workflow.features.advanced import build_comprehensive_features

# Classification (Phase 9.4)
from finance_ml.ml_workflow.classification.labels import create_enhanced_event_labels
from finance_ml.ml_workflow.classification.tuning import optimize_classifier_hyperparameters
from finance_ml.ml_workflow.classification.models import prepare_classification_data

# Regression (Phase 9.5)
from finance_ml.ml_workflow.regression.dataset import (
    prepare_regression_data,
    create_classification_interactions
    )
from finance_ml.ml_workflow.regression.models import (
    train_xgboost_regressor,
    compare_regressors
    )
from finance_ml.ml_workflow.regression.constraints import NonNegativeRegressionWrapper

# Evaluation (Phase 9.6)
from finance_ml.ml_workflow.evaluation import comprehensive_regression_metrics

# Analytics (Phase 9.7)
from finance_ml.ml_workflow.analytics import (
    calculate_mispricing_score,
    rank_undervalued_stocks
    )

# Reporting (Phase 9.8)
from finance_ml.ml_workflow.reporting import prepare_plotly_dashboard_data
```

**Backward-Compatible Top-Level Imports (For Legacy Code)**

```python
# All major functions available at top level for backward compatibility
from finance_ml import (
    # Data & Preprocessing
    load_from_csv,
    normalize_columns,
    prepare_phase91_data,
    apply_enhanced_imputation_strategy_4step,

    # Features
    preprocess_for_lightgbm,
    build_comprehensive_features,

    # Classification
    classification_create_enhanced_event_labels,
    classification_optimize_hyperparameters,

    # Regression
    regression_prepare_data,
    regression_train_xgboost,
    regression_compare_regressors,
    regression_nonnegative_wrapper,

    # Analytics
    analytics_calculate_mispricing,
    analytics_rank_undervalued,

    # Config
    get_config
    )
```

Appendix A — Quick Reference

- Train result schema: {model, metrics, y_pred, y_proba?, artifacts?}
- Dataset prep return: (X_train, X_test, y_train, y_test, meta) or DatasetSplit dataclass
- Canonical columns: last_price, price_target, price_target_median, sector, region, ticker
- Normalize early: `df = normalize_columns(df)`
- Validate before modeling: `validate_schema(df, require_target=True)`

Appendix B — Migration Notes

- Legacy code reading top-level metrics (e.g., `res['rmse']`) should be updated to `res['metrics']['rmse']`. Provide
  shims in training functions during the migration window.
- Replace references to raw column names (e.g., `"Last Price"`) with canonical names after `normalize_columns()`.

Appendix C — Preprocessing Function Parameter Differences (Phase 9.1 Migration)

**IMPORTANT**: During Phase 9.1 refactoring, preprocessing functions were moved from `data.py` to
`finance_ml.ml_workflow.preprocessing/` with updated signatures. The package-level imports may route to OLD versions
with DIFFERENT parameter names.

**Issue**: The old `data.py` functions expect singular `column` parameter, while new preprocessing functions accept
plural `columns`. Calling with wrong parameter names causes
`TypeError: unexpected keyword argument 'columns'. Did you mean 'column'?`

**Affected Functions**:

1. **detect_outliers_iqr**
    - OLD (data.py): `def detect_outliers_iqr(df, column: str, multiplier: float = 1.5)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_iqr(df, columns: Optional[List[str]] = None, by_sector: bool = True, iqr_multiplier: float = 1.5)`
    - **Solution**: Loop through each column individually when using old version:
      ```python
      outliers_iqr = {}
      for col in financial_metrics[:20]:
          outliers_iqr[col] = detect_outliers_iqr(all_stocks, column=col, multiplier=1.5)
      ```

2. **detect_outliers_zscore**
    - OLD (data.py): `def detect_outliers_zscore(df, column: str, threshold: float = 3.0)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_zscore(df, columns: Optional[List[str]] = None, threshold: float = 3.0, by_sector: bool = True)`
    - **Solution**: Loop through each column individually:
      ```python
      outliers_zscore = {}
      for col in financial_metrics[:20]:
          outliers_zscore[col] = detect_outliers_zscore(all_stocks, column=col, threshold=3.0)
      ```

3. **detect_outliers_isolation_forest**
    - OLD (data.py): `def detect_outliers_isolation_forest(df, column: str, contamination: float = 0.1)`
    - NEW (preprocessing/outliers.py):
      `def detect_outliers_isolation_forest(df, columns: Optional[List[str]] = None, contamination: float = 0.1)`
    - **Solution**: Loop through each column individually:
      ```python
      outliers_iforest = {}
      for col in financial_metrics[:20]:
          outliers_iforest[col] = detect_outliers_isolation_forest(all_stocks, column=col, contamination=0.1, random_state=42)
      ```

4. **winsorize_by_sector**
    - OLD (data.py):
      `def winsorize_by_sector(df, columns: List[str], sector_column: str = "sector", lower: float = 0.01, upper: float = 0.99)`
    - NEW (preprocessing/outliers.py):
      `def winsorize_by_sector(df, columns: Optional[List[str]] = None, lower_percentile: float = 0.01, upper_percentile: float = 0.99, by_sector: bool = True)`
    - **Solution**: Use correct parameter names for old version:
      ```python
      all_stocks = winsorize_by_sector(
          all_stocks,
          columns=financial_metrics[:20],
          lower=0.01,  # NOT lower_percentile
          upper=0.99,  # NOT upper_percentile
          sector_column='sector'  # NOT by_sector=True
      )
      ```

**Best Practices**:

1. **Explicit imports**: Import directly from the subpackage you intend to use:
   ```python
   # Use new Phase 9.1 version explicitly
   from finance_ml.ml_workflow.preprocessing.outliers import detect_outliers_iqr
   ```

2. **Check function signature**: Before calling, verify parameter names match the version you're using:
   ```python
   import inspect
   print(inspect.signature(detect_outliers_iqr))
   ```

3. **Per-column processing**: When using old data.py versions, always loop through columns individually rather than
   passing a list.

4. **Migration path**: Update imports to use new Phase 9.1 preprocessing modules once they're stable and fully tested.

**Testing**: Add unit tests that verify function signatures match expected parameter names to catch these issues early.

---

## Appendix D — Version History and Recent Updates

**Model Version: v9_9** (as of 2025-11-10, version 0.7.0)

### Phase 9.3 Feature Engineering Enhancements (v9_9)

**New Feature Categories (50-75 new features):**

1. **Momentum & Technical Features**
    - Price momentum indicators (1m, 3m, 6m, 1y)
    - RSI (14-day, 30-day)
    - Moving average crossover signals
    - Return stability score and Sharpe proxy

2. **Quality & Risk Signals**
    - Altman Z-Score trends and volatility
    - Financial distress composite scores
    - Accounting quality metrics (exceptional items, goodwill impairment, restructuring)
    - Asset quality indicators

3. **Cash Flow & Capital Allocation**
    - Cash flow quality (CFO/Net Income, FCF margin)
    - Capital intensity and efficiency
    - Total shareholder return yield
    - Reinvestment and acquisition metrics

4. **Market Sentiment & Analyst Features**
    - Analyst consensus (bullish/bearish percentages)
    - Price target features (upside potential, range, revisions)
    - Short interest ratio and beta stability
    - Analyst coverage quality

5. **Profitability Trends**
    - Margin evolution (EBITDA, gross, operating)
    - Operating leverage
    - Earnings quality scores
    - Forward-looking profitability estimates

6. **Balance Sheet Strength**
    - Growth rates (debt, equity, assets)
    - Liquidity trends and ratios
    - Retained earnings patterns

7. **Temporal & Composite Features**
    - LTM vs 5Y average comparisons
    - Earnings date features
    - Quality/growth/value/momentum composite scores

**API Enhancements:**

- `build_features()` with presets: "basic", "momentum", "quality", "comprehensive", "full_enhanced"
- `build_comprehensive_features()` accepts optional `preset` parameter
- All features support Phase 9.3 column naming conventions

### Phase 9.4 Classification Label Enhancements (v9_9)

**13 Event Label Creation Methods:**

1. `price_momentum` - Enhanced with Phase 9.3 momentum features
2. `valuation` - Multi-metric valuation composite
3. `fundamental` - Profitability and quality metrics
4. `volatility` - Stability indicators
5. `analyst_rating` - Analyst consensus with coverage quality
6. `market_events` - Sentiment and sector-relative signals
7. `combined_signals` - Multi-metric composite
8. `profitability_event` - Margin and return events
9. `leverage_event` - Debt and capital structure
10. `liquidity_event` - Working capital and cash
11. `growth_event` - Revenue and earnings growth
12. `efficiency_event` - Asset turnover and productivity
13. `quality_event` - Accounting quality and distress

**Key Improvements:**

- `_get_column()` helper for Phase 9.3 column support
- Backward compatible with original columns
- Meaningful class distributions across all methods
- All 29 classification tests passing

**5-Class Labeling System (Enhanced Granularity):**

All event label creation methods now use a 5-class classification system instead of the previous 3-class system:

| Label | Class Name      | Interpretation                                     | Use Case                                |
|-------|-----------------|----------------------------------------------------|-----------------------------------------|
| 0     | Strong Negative | Significantly unfavorable signal (bottom quintile) | High-conviction short/avoid signals     |
| 1     | Negative        | Moderately unfavorable signal                      | Cautious/underweight positions          |
| 2     | Neutral         | Mixed or insufficient signal                       | Hold/market-weight positions            |
| 3     | Positive        | Moderately favorable signal                        | Opportunistic long positions            |
| 4     | Strong Positive | Significantly favorable signal (top quintile)      | High-conviction long/overweight signals |

**Threshold Mechanics:**

- **Percentile-based**: Strong labels (0, 4) typically use 20th/80th percentiles
- **Sector-adjusted**: Optional sector normalization for cross-sector comparability
- **Method-specific**: Each method defines its own thresholds based on metric characteristics
- **Backward compatible**: Existing 3-class code automatically benefits from finer granularity

**Benefits of 5-Class System:**

1. **Improved Signal Strength**: Distinguishes high-conviction from moderate signals
2. **Better Risk Management**: Enables tiered position sizing (e.g., 2x allocation for class 4 vs class 3)
3. **Enhanced Model Training**: More granular labels improve classifier performance
4. **Flexible Aggregation**: Can collapse back to 3-class (0-1→Negative, 2→Neutral, 3-4→Positive) if needed

**Implementation Notes:**

- XGBoost/LightGBM `num_class` updated from 3 to 5
- All 29 classification tests updated and passing
- Notebook and submodules (models.py, tuning.py, evaluation.py) fully compatible
- Classification probability columns now include 5 probabilities per sample

### Phase 9.5 Regression Enhancements (v0.6.1+)

**Classification Meta-Features:**

- `extract_classification_features()` adds event probabilities to regression datasets
- Provides sentiment and event likelihood signals
- Enhances prediction accuracy with multi-stage modeling

**Modular Regression Pipelines:**

- Ridge, Lasso, ElasticNet, Bayesian Ridge models
- Standardized return signatures: {model, metrics, y_pred, feature_importance}
- Improved abstraction and testability
- NonNegativeRegressionWrapper for constraint enforcement

### Phase 9.1 Preprocessing Updates (v0.5.1+, v1.3+)

**Enhanced 6-Step Imputation Strategy:**

The standardized imputation pipeline is implemented in
`finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step`.
It operates on a schema-validated dataframe and guarantees **zero missing values** for all Phase 9.x required features.

**Six Steps:**

1. **Zero Imputation (Schema-Driven)**
    - Apply zero imputation to curated event and count metrics defined by `get_zero_imputation_columns()`
    - Column selection derived from `COLUMN_SCHEMA` and Phase 9.3 feature inputs
    - Only columns with natural zero (exceptional items, counts, certain ratios) are eligible
    - Validation ensures no non-numeric columns in zero-impute list

2. **Sector-Aware KNN Imputation (Core Metrics)**
    - Use `impute_missing_values_knn_sector()` / `apply_knn_imputation_enhanced()` for core financial metrics
    - Imputation performed within sector (and optionally region) groups to preserve domain patterns
    - Column selection via `get_knn_imputation_columns()`, consistent with Phase 9.3 feature requirements
    - Tests validate sector-aware neighbor selection

3. **Price Imputation (Targets)**
    - Apply domain-specific rules to fill `price_target` and related columns
    - Uses `last_price`, `price_target_median`, and valuation metrics
    - Preserves monotonicity and deterministic behavior
    - Future: provenance flags (e.g., `price_target_imputed`) for all imputed target values

4. **Median Imputation (Residual Numerics)**
    - Robust median imputation for all remaining numeric columns
    - After this step: **no missing numeric values** for modeling/analytics columns

5. **Categorical Imputation (Groupwise + Global)**
    - Groupwise imputation for selected categoricals (e.g., `size_class` within `sector`, `country` within `region`)
    - Followed by global most-frequent or constant strategies
    - Configuration defined in `get_categorical_imputation_config()` and overridable via `FinanceMLConfig`
    - Tests validate mode-based sector-grouped imputation

6. **Datetime Imputation and Formatting (Temporal Readiness)**
    - Convert all date columns to `datetime64[ns]`
    - Per-column strategies: forward-fill, median, or constants (defined in `get_datetime_imputation_config()`)
    - Column-specific policies:
        - `last_updated`: forward-fill within ticker
        - `income_statement_report_date`: median or nearest valid date
        - `next_earnings`: constant (NaT) or median with flag
    - Future: imputation flags (e.g., `next_earnings_imputed`) for temporal features

**Post-Conditions:**

- `validate_imputation_completeness()` confirms zero missing values across numeric, categorical, and date columns
- Non-negativity and outlier safety rails hold after imputation (no negative prices/market cap)
- Imputation diagnostics (missingness before/after, coercion counts, sector/region summaries) recorded in
  `quality_stats`
- Metadata catalog persistence in `*_metadata.json`

**Schema Integration (v1.3+):**

- Column selection for each step driven by `COLUMN_SCHEMA` role and dtype metadata
- All imputation functions receive schema-validated dataframes via `detect_and_cast_dtypes()`
- Future enhancement: dynamic column derivation from schema to eliminate hard-coded lists

**Testing Coverage:**

- Core tests: `tests/test_enhanced_imputation.py` (21 tests)
- Phase 9.3 integration: `tests/test_enhanced_imputation_phase93.py` (8 tests, 7 passed, 1 skipped)
- Metadata validation: `tests/test_metadata_catalog_quality.py` (4 tests)
- Coverage: ≥80% for imputation modules

### Phase 9.2 EDA Enhancements (v0.5.0+)

**New Analysis Functions:**

- `calculate_financial_metrics_dashboard()` - Automated KPI reporting
- `generate_data_quality_alerts()` - Quality monitoring
- `perform_comprehensive_hypothesis_tests()` - Statistical testing
- Benchmarking module with sector/regional comparisons

### Phase 9.7 Analytics Updates (v0.6.0+)

**Analyst Comparison:**

- `compare_prediction_vs_analyst_targets()` - Comprehensive comparison
- Agreement rate and directional accuracy metrics
- Systematic bias analysis
- Excel report generation with charts

### Testing and Quality (Current Status)

- **Test Suite:** 83 test modules (updated 2025-11-21)
    - Includes TDD v0.8.2 additions: schema integration (44 tests), datatype detection, Phase 9.3 enhancements
    - Portfolio Optimization: 5 modules, 23 tests covering ML-based returns, advanced optimization, risk management
- **Coverage Target:** ≥85% for new code, ≥80% overall
- **Fast Tests:** < 100 lines, pure functions (coverage_smoke, loaders, validation, repository_setup)
- **Medium Tests:** 100-500 lines, integration (imputation, risk_metrics, logging, notebook_schema_integration)
- **Slow Tests:** > 500 lines, heavy ML (classification_phase94, advanced_models_phase95, finance_ml_eval)

**Recommended Testing Workflow:**

1. Development: Run only affected module tests
2. Before commit: Run fast + medium tests (~1-3 minutes)
3. CI/CD: Run full suite with timeout protection

---

## Uncertainty and Prediction Intervals (Standards)

Purpose: Establish a single, consistent approach to predictive uncertainty for regression models (price targets) that is
compatible with both notebook and CLI, and fully testable (TDD).

1) Quantile Regression + Conformal Calibration

- Primary approach: train quantile regressors at q ∈ {0.1, 0.5, 0.9} and apply distribution-free conformal calibration
  on a calibration split to achieve target coverage.
- Required package APIs (to be implemented in `finance_ml.ml_workflow.regression.quantiles`):
    - `train_quantile_models(X_train, y_train, quantiles=(0.1, 0.5, 0.9), **kwargs) -> Dict[float, Any]`
    - `predict_quantiles(models, X) -> Dict[float, np.ndarray]`
    - `conformal_calibrate_intervals(y_cal, y_cal_pred, y_test_pred, alpha=0.2) -> Tuple[np.ndarray, np.ndarray]`
- Monotonicity: enforce p10 ≤ p50 ≤ p90 post-prediction; if violated, sort the triplet per row.
- Non-negativity: when the target is non-negative (price), clip lower bounds at 0.
- Coverage target: 80% ± 5% overall; also compute per-sector coverage within ±10% of overall.

2) Return Schema Extensions for Regression Functions

- In addition to the standard return contract, extend with optional keys when uncertainty is requested:
    - `y_quantiles`: Dict[str, np.ndarray] — keys: `p10`, `p50`, `p90`
    - `intervals`: Dict[str, np.ndarray] — keys: `lower`, `upper`, `width`
    - `calibration`: Dict[str, Any] — residual quantile used, coverage metrics by segment
- Tests must verify presence and shapes when enabled.

3) Artifact and File Output Contract

- Quantile predictions CSV: `outputs/regression/quantile_predictions.csv` with columns:
  -
  `ticker, isin, sector, region, last_price, y_true, pred_p10, pred_p50, pred_p90, interval_width, model_version, snapshot_date`
- Validation: no negative `pred_p10` when last_price ≥ 0; `pred_p10 ≤ pred_p50 ≤ pred_p90` for every row.

4) TDD for Uncertainty

- Add `tests/test_uncertainty_calibration.py`:
    - Synthetic monotonic dataset; verify monotonic quantiles and coverage (75–85%) after conformal calibration.
    - Verify lower bound non-negativity when target ≥ 0.
    - Include per-sector synthetic split to validate segment coverage stability (±10%).

---

## Outlier Safety Rails Policy

To mitigate catastrophic errors that disproportionately impact mean metrics:

- Target winsorization: cap `y_train` at [1st, 99th] percentiles before training robust models; parameterize limits.
- Robust loss: prefer Huber loss (or quantile) for baseline gradient boosting.
- Post-prediction clipping: clip `y_pred` to mean ± 3·std of training target, lower bound 0 for prices.
- Negative prediction guard: enforce non-negative outputs via wrapper where appropriate.
- Diagnostics: compute counts above 100% and 1000% absolute percentage error; log top-k outliers for inspection.

TDD: `tests/test_outlier_safety_rails.py` covers winsorization bounds, clipping behavior, and non-negativity.

---

## Data Split and Leakage Policy

Choose the most leakage-safe split available given the dataset:

1) If a time column exists (`as_of_date`, `snapshot_date`, `date`): use `TimeSeriesSplit(n_splits=5)` or explicit
   pre/post temporal split; no shuffling.
2) Else, if an entity identifier exists (`ticker`): use `GroupKFold`/`GroupShuffleSplit` grouping by `ticker`.
3) Else, stratify by `sector` to preserve distribution balance.
4) Else, use random split with fixed `RANDOM_SEED` from env.

Provide utility:
`finance_ml.ml_workflow.validation.splits.time_series_cv_or_grouped_split(df, date_col=None, group_col=None, stratify_col=None) -> Splitter`.

TDD: `tests/test_data_splits_policy.py` verifies behavior for each scenario on synthetic data.

---

## Standardized Predictions Schema (Contract)

All regression prediction outputs must include the following columns when available:

- Core identifiers: `ticker, isin, name (optional), sector, region, snapshot_date`
- Price columns: `last_price, y_true` (if in-sample/validation), `y_pred`
- Calibrated prediction (optional): `y_pred_calibrated`
- Uncertainty (optional): `pred_p10, pred_p50, pred_p90, interval_width`
- Errors (if `y_true` present): `abs_error, pct_error`
- Metadata: `model_version`

Primary file path: `outputs/regression/regression_predictions_detailed.csv`.

TDD: `tests/test_predictions_schema.py` asserts presence/dtypes of required columns and basic invariants (
non-negativity, monotonic intervals).

---

## Sector Metrics and Calibration

- The pipeline must compute and persist per-sector metrics to `outputs/models/regression_metrics_by_sector.csv` (MAE,
  RMSE, R², MAPE, count).
- Sector-specific bias calibration may be applied as a post-processing layer using
  `finance_ml.ml_workflow.regression.calibration.calibrate_predictions_by_sector`.

TDD:

- `tests/test_regression_sector_metrics.py` ensures non-empty sector metrics on synthetic data.
- `tests/test_sector_bias_calibration.py` verifies additive adjustments only for mapped sectors and optional
  non-negativity.

---

## TDD Conventions and Selective Test Execution

- Naming: all new tests live under `tests/` and follow `test_*.py` naming.
- Fast tests (<100 lines): synthetic datasets, pure functions/utilities (include new safety rails, splits, schema
  checks).
- Medium tests (100–500 lines): integration across small pipelines (quantiles + conformal on synthetic; sector metrics).
- Slow tests (>500 lines): heavy ML training; avoid adding new slow tests unless necessary.

Selective execution examples:

- Fast only:
  -
  `python -m unittest tests.test_predictions_schema tests.test_uncertainty_calibration tests.test_data_splits_policy -v`
- Medium set:
    - `python -m unittest tests.test_regression_sector_metrics tests.test_sector_bias_calibration -v`

---

## Section 8: Notebook Implementation Guidelines (Phase 10)

### 8.1 Configuration Management

**Single Source of Truth Principle**:

- Define all configuration constants in a single cell at the top of the notebook (after imports)
- Avoid redefining constants in later sections
- Use descriptive variable names matching code_guidelines.md conventions

**Required Configuration Constants**:

```python
# Regression configuration
TARGET_COL = 'price_target'  # Canonical target (Section 2.2)
TARGET_COL_FALLBACK = 'last_price'  # Canonical fallback
TEST_SIZE = 0.2
CV_FOLDS = 5
QUANTILES = [0.1, 0.5, 0.9]
MIN_SECTOR_SAMPLES = 20
RANDOM_SEED = int(os.getenv('RANDOM_SEED', '42'))

# Output directories
OUTPUT_DIR = Path("outputs")
```

**Configuration Validation**:

- Add validation cell to check all required constants are defined
- Verify OUTPUT_DIR structure exists (all Phase 9.1-9.8 subdirectories)
- Log configuration summary for reproducibility

### 8.2 Version Alignment Conventions

**Notebook Version vs. Package Version**:

- Notebook version should match or exceed package version
- Document version in notebook title cell: `**Version X.Y.Z** — **Model Version: vN_M**`
- Use environment variable `MODEL_VERSION` for consistency: `os.environ.get('MODEL_VERSION', 'v9_9')`

**Version Number Format**:

- Notebook version: Semantic versioning `X.Y.Z` (e.g., `2.1.0`)
- Model version: Phase-aligned `vN_M` (e.g., `v9_9` for Phase 9.9)
- Package version: Defined in `finance_ml/config.py` and `pyproject.toml`

**Version Update Triggers**:

- Major workflow changes → increment notebook major version
- New features/sections → increment notebook minor version
- Bug fixes/refactoring → increment notebook patch version
- Model architecture changes → increment model version

### 8.3 Import Organization Best Practices

**Import Structure** (following Phase 9.1-9.8 modular design):

1. **Standard library imports** (os, warnings, pathlib, datetime)
2. **Third-party core imports** (numpy, pandas, matplotlib, seaborn)
3. **Visualization imports** (plotly.express, plotly.graph_objects)
4. **Phase 9.1-9.8 package imports** (organized by phase)

**Package Import Pattern**:

```python
# Phase 9.1: Data loading and preprocessing
from finance_ml import (
    load_from_csv, load_from_db, validate_schema,
    normalize_columns, check_missing_values,
    )
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_enhanced_imputation_strategy_6step,
    validate_imputation_completeness
    )

# Phase 9.3: Feature engineering
from finance_ml import (
    features_build_comprehensive,
    features_importance_rf,
    engineer_valuation_ratios,
    )

# Phase 9.5: Regression models
from finance_ml import (
    regression_prepare_data,
    regression_train_xgboost,
    regression_train_quantile,
    )
```

**Import Best Practices**:

- Use package-level convenience imports when available (avoid deep module paths)
- Import all required functions once at the top (avoid scattered imports)
- Add comments indicating which phase each import block corresponds to
- Use descriptive prefixes for package functions (e.g., `preprocessing_*`, `features_*`, `regression_*`)

### 8.4 Cell Execution Order Dependencies

**Required Execution Order**:

1. Configuration and Setup
2. Data Loading and Preprocessing (Section 2)
3. Exploratory Data Analysis (Section 3)
4. Feature Engineering (Section 4)
5. Classification (Section 5)
6. Regression (Section 6)
7. Evaluation (Section 7)
8. Analytics (Section 8-10)

**Dependency Validation**:

- Add assertions to check required variables exist before use
- Example: `assert 'all_stocks' in globals(), "Run Section 2 first to load data"`
- Use meaningful variable names to track pipeline stages (e.g., `all_stocks`, `all_stocks_scaled`,
  `all_stocks_features`, `all_stocks_enhanced`)

**State Management**:

- Avoid in-place modifications when possible; create new DataFrames with descriptive names
- Document which cells modify state vs. create new objects
- Clear variables that are no longer needed to free memory: `del large_dataframe`

### 8.5 Output Artifact Validation

**Required Output Artifacts** (Phase 9.1-9.8 aligned):

- `outputs/eda/` — EDA visualizations and reports
- `outputs/features/` — Feature importance and selection results
- `outputs/classification/` — Classification models and metrics
- `outputs/regression/regression_predictions_detailed.csv` — Standardized predictions schema
- `outputs/regression/quantile_predictions.csv` — Quantile predictions
- `outputs/regression/regression_metrics_by_sector.csv` — Sector-level metrics
- `outputs/analytics/` — Mispricing scores, analyst comparison
- `outputs/plots/` — Interactive visualizations (HTML)

**Validation Checklist**:

```python
# After Section 6 (Regression)
required_files = [
    OUTPUT_DIR / "regression" / "regression_predictions_detailed.csv",
    OUTPUT_DIR / "regression" / "quantile_predictions.csv",
    OUTPUT_DIR / "regression" / "regression_metrics_by_sector.csv",
    ]
for filepath in required_files:
    assert filepath.exists(), f"Missing required output: {filepath}"
    assert filepath.stat().st_size > 0, f"Empty output file: {filepath}"
print("✓ All required regression artifacts validated")
```

**Schema Validation**:

- Verify standardized predictions schema columns (Section 2.2)
- Check for non-negative intervals: `pred_p10 ≤ pred_p50 ≤ pred_p90`
- Validate sector metrics CSV has expected columns and non-zero rows

---

## Section 9: Performance Optimization Guidelines (Phase 10)

### 9.1 Quantile Calibration Standards

**Coverage Target**: 75-85% empirical coverage (target: 80%)

**Calibration Procedure**:

1. Train quantile regression models with proper loss function:
    - GradientBoostingRegressor with `loss='quantile'` and `alpha` parameter
    - Use TimeSeriesSplit to prevent leakage
2. Apply conformal calibration on separate calibration set
3. Enforce monotonicity: `pred_p10 ≤ pred_p50 ≤ pred_p90` post-prediction
4. Clip lower bounds at 0 for price predictions (non-negativity)
5. Validate coverage per sector (within ±10% of overall target)

**Validation Metrics**:

```python
# Compute empirical coverage
coverage = ((y_true >= pred_p10) & (y_true <= pred_p90)).mean()
assert 0.75 <= coverage <= 0.85, f"Coverage {coverage:.1%} outside target range 75-85%"

# Check monotonicity
assert (pred_p10 <= pred_p50).all(), "Monotonicity violated: p10 > p50"
assert (pred_p50 <= pred_p90).all(), "Monotonicity violated: p50 > p90"

# Verify non-negativity for prices
assert (pred_p10 >= 0).all(), "Negative lower bound detected in price predictions"
```

**TDD Requirement**: `tests/test_quantile_calibration_coverage.py` must validate these properties

### 9.2 Outlier Handling Policies

**Outlier Definition**:

- **Mild outliers**: Absolute percentage error > 100%
- **Severe outliers**: Absolute percentage error > 500%
- **Catastrophic outliers**: Absolute percentage error > 1,000%

**Detection Thresholds**:

- Mean/median error ratio > 3x indicates outlier problem
- Max error > 1,000% requires investigation
- > 1% predictions with >500% error is unacceptable

**Filtering Strategy**:

1. **Pre-training**: Winsorize target at [1st, 99th] percentiles
2. **Post-prediction**: Clip predictions to `mean ± 3·std` of training target
3. **Confidence scoring**: Assign confidence based on feature completeness and prediction uncertainty
4. **Quality flagging**: Add `prediction_quality` column: {high, medium, low}

**Confidence Score Calculation**:

```python
def calculate_prediction_confidence(df, feature_cols, interval_width):
    """
    Calculate confidence score (0-1) based on:
    - Feature completeness (% non-null)
    - Prediction interval width (lower is better)
    - Sector-specific volatility adjustment
    """
    completeness = df[feature_cols].notna().mean(axis=1)
    interval_score = 1 - np.clip(interval_width / interval_width.median(), 0, 1)
    confidence = (completeness * 0.6) + (interval_score * 0.4)
    return confidence


# Apply quality flags
df['confidence_score'] = calculate_prediction_confidence(df, feature_cols, df['interval_width'])
df['prediction_quality'] = pd.cut(
        df['confidence_score'],
        bins=[0, 0.5, 0.75, 1.0],
        labels=['low', 'medium', 'high']
        )
```

**Reporting Strategy**:

- Always report metrics for both **all predictions** and **high-confidence only**
- Identify and log top-k outliers for manual inspection
- Export outlier diagnostics: `outputs/evaluation/outlier_analysis.csv`

**TDD Requirement**: `tests/test_outlier_prediction_filtering.py` must validate filtering logic

### 9.3 Sector-Specific Modeling Criteria

**When to Use Sector-Specific Models**:

- Sector has ≥100 samples in training set (MIN_SECTOR_SAMPLES threshold)
- Global model achieves >150% mean absolute percentage error for sector
- Sector has unique feature importance patterns (>30% different from global)

**Sector Model Training Procedure**:

1. Filter training data by sector
2. Apply sector-specific feature engineering (e.g., commodity prices for Energy)
3. Tune hyperparameters separately using Optuna (≥30 trials)
4. Validate performance vs. global model on hold-out set
5. Use sector model only if it improves MAE by ≥10%

**Fallback Strategy**:

```python
# Train global model first
global_model = train_global_model(X_train, y_train)

# Train sector-specific models
sector_models = {}
for sector in sectors_needing_optimization:
    if len(X_train[X_train['sector'] == sector]) >= MIN_SECTOR_SAMPLES:
        sector_model = train_sector_model(X_train, y_train, sector)
        if sector_model.mae < global_model.mae * 0.9:  # 10% improvement
            sector_models[sector] = sector_model
        else:
            print(f"Sector {sector}: Using global model (no improvement)")


# Prediction routing
def predict_with_sector_routing(X, global_model, sector_models):
    predictions = np.zeros(len(X))
    for i, row in X.iterrows():
        sector = row['sector']
        model = sector_models.get(sector, global_model)
        predictions[i] = model.predict(row.values.reshape(1, -1))
    return predictions
```

**Performance Thresholds**:

- **Best**: <60% mean absolute percentage error
- **Good**: 60-100% mean absolute percentage error
- **Acceptable**: 100-150% mean absolute percentage error
- **Needs optimization**: >150% mean absolute percentage error

**TDD Requirement**: `tests/test_sector_specific_models.py` must validate sector model selection logic

### 9.4 Bias Correction Procedures

**When to Apply Bias Correction**:

- Systematic over-prediction or under-prediction detected (mean bias > ±10)
- Sector-specific bias patterns identified (>±20 difference between sectors)
- Temporal drift detected (predictions increasingly biased over time)

**Bias Correction Methods**:

**1. Additive Sector Bias Correction** (simplest):

```python
# Calculate sector-wise bias on validation set
sector_bias = val_df.groupby('sector').apply(lambda x: (x['y_pred'] - x['y_true']).mean())

# Apply correction
df['y_pred_calibrated'] = df.apply(
        lambda row: row['y_pred'] - sector_bias.get(row['sector'], 0),
        axis=1
        )
```

**2. Isotonic Regression Calibration** (recommended):

```python
from sklearn.isotonic import IsotonicRegression

# Train isotonic calibrator per sector
isotonic_models = {}
for sector in sectors:
    sector_data = val_df[val_df['sector'] == sector]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(sector_data['y_pred'], sector_data['y_true'])
    isotonic_models[sector] = iso

# Apply calibration
df['y_pred_calibrated'] = df.apply(
        lambda row: isotonic_models[row['sector']].transform([row['y_pred']])[0],
        axis=1
        )
```

**3. Market Cap Bucket Correction**:

- Separate bias correction for small-cap (<$2B), mid-cap ($2-10B), large-cap (>$10B)
- Small-caps often have higher prediction errors; apply stronger correction

**Validation Requirements**:

- Bias correction must not break monotonicity of quantile predictions
- Non-negativity must be preserved (clip at 0 for prices)
- Document bias reduction achieved: `(original_bias - corrected_bias) / original_bias`

**TDD Requirement**: `tests/test_bias_correction_isotonic.py` must validate calibration logic

### 9.5 Integration Testing for Optimization Features

**End-to-End Validation**:

```python
# After training and applying all optimizations
def validate_optimization_success(results_df):
    """Validate that all Phase 10 optimizations were successfully applied."""

    # 1. Quantile coverage validation
    coverage = ((results_df['y_true'] >= results_df['pred_p10']) &
                (results_df['y_true'] <= results_df['pred_p90'])).mean()
    assert 0.75 <= coverage <= 0.85, f"Coverage {coverage:.1%} outside 75-85% target"

    # 2. Outlier reduction validation
    mean_error = results_df['pct_error'].abs().mean()
    median_error = results_df['pct_error'].abs().median()
    ratio = mean_error / median_error
    assert ratio < 3, f"Mean/median error ratio {ratio:.1f}x still too high (target: <3x)"

    # 3. Sector performance validation
    sector_errors = results_df.groupby('sector')['pct_error'].apply(lambda x: x.abs().mean())
    worst_sectors = sector_errors.nlargest(3)
    for sector, error in worst_sectors.items():
        if sector == 'Real Estate':
            assert error < 200, f"{sector} error {error:.1f}% exceeds 200% target"
        elif sector in ['Materials', 'Energy']:
            assert error < 150, f"{sector} error {error:.1f}% exceeds 150% target"

    # 4. Bias reduction validation
    sector_bias = results_df.groupby('sector').apply(
            lambda x: (x['y_pred_calibrated'] - x['y_true']).mean()
            )
    max_bias = sector_bias.abs().max()
    assert max_bias < 30, f"Max sector bias {max_bias:.1f} exceeds ±30 target"

    print("✅ All Phase 10 optimization targets validated successfully")
```

**Performance Baseline Tracking**:

- Record baseline metrics before optimization
- Track improvement percentages for each optimization
- Export comparison report: `outputs/evaluation/optimization_impact_report.csv`

---

**Document Version:** 1.3 (Phase 10 Enhanced)  
**Last Updated:** 2025-11-13  
**Synchronized with:** CHANGELOG.md v0.7.0, README.md v0.7.0, finance_ml_improvement_plan.md Phase 10
