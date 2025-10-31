"""
Finance ML Models Module

Machine learning model functions including classification, regression,
quantile regression, and stacking ensembles.

Phase 7 TDD refactoring: Extracted from ml_finance_model_v8_2.py with
cleaner API and comprehensive test coverage.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    StackingRegressor,
    )
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    classification_report,
    )
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Import dependencies from finance_ml modules
from finance_ml.data import normalize_columns, validate_schema


def create_event_labels(df: pd.DataFrame, use_volatility: bool = False) -> np.ndarray:
    """Create event classification labels from price targets and other features.

    Labels:
    - 0: Neutral (price_target within ±10% of last_price)
    - 1: Positive catalyst (price_target > last_price by >=10%)
    - 2: Negative catalyst (price_target < last_price by >=10%)

    Args:
        df: DataFrame with last_price and price_target columns
        use_volatility: If True, consider volatility spikes as additional signal

    Returns:
        numpy array of labels (0, 1, or 2)
    """
    labels = np.zeros(len(df), dtype=int)

    # Calculate price target uplift/downlift percentage
    price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0

    # Classify based on thresholds
    labels[price_diff_pct >= 10.0] = 1  # Positive (>=10% upside)
    labels[price_diff_pct <= -10.0] = 2  # Negative (>=10% downside)
    # Everything else stays 0 (Neutral)

    # Optional: incorporate volatility spikes
    if use_volatility and "volatility_1m" in df.columns:
        vol_col = df["volatility_1m"]
        # High volatility (>0.5) could be treated as negative signal
        high_vol_mask = vol_col > 0.5
        # Downgrade neutral or positive to negative if high volatility
        labels[high_vol_mask & (labels != 2)] = 2

    return labels


def train_event_classifier(
    df: pd.DataFrame, labels: np.ndarray, random_state: int = 42
) -> Dict[str, Any]:
    """Train an event classifier and return model + metrics.

    Args:
        df: DataFrame with features (ticker, sector excluded automatically)
        labels: Event labels (0/1/2)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with keys:
        - 'model': trained classifier
        - 'accuracy': accuracy score
        - 'classification_report': detailed classification metrics
        - 'probabilities': predicted probabilities on training data
    """
    # Prepare features: drop identifiers and target-related columns
    X = df.copy()
    drop_cols = ["ticker", "isin", "name", "description", "price_target", "price_target_median"]
    drop_cols = [c for c in drop_cols if c in X.columns]
    X = X.drop(columns=drop_cols)

    # Remove any duplicate columns to avoid downstream transformer issues
    if X.columns.duplicated().any():
        dup_count = int(X.columns.duplicated().sum())
        logging.warning("train_event_classifier: removing %d duplicate column(s)", dup_count)
        X = X.loc[:, ~X.columns.duplicated(keep="first")]

    # Split categorical and numeric
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # Build preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    # Build and train classifier
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=random_state, class_weight="balanced"
    )

    # Fit preprocessing + model
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    model.fit(X_train_prep, y_train)

    # Evaluate
    y_pred = model.predict(X_test_prep)
    accuracy = float(accuracy_score(y_test, y_pred))

    # Get probabilities for all data
    X_all_prep = preprocessor.transform(X)
    probabilities = model.predict_proba(X_all_prep)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "accuracy": accuracy,
        "classification_report": report,
        "probabilities": probabilities,
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def build_regression_pipeline(
    numeric_features: List[str], categorical_features: List[str], n_jobs: int = 1
) -> Pipeline:
    """Build sklearn pipeline for regression with preprocessing.

    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        n_jobs: Number of parallel jobs for RandomForestRegressor

    Returns:
        sklearn Pipeline with preprocessor and regressor steps
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    regressor = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    return pipeline


def train_and_evaluate_regression(
    df: pd.DataFrame, out_dir: Path, n_jobs: int = 1, dry_run: bool = False
) -> Optional[Dict[str, Any]]:
    """Train and evaluate regression model.

    Args:
        df: DataFrame with features and target
        out_dir: Directory to save outputs
        n_jobs: Number of parallel jobs
        dry_run: If True, skip training

    Returns:
        Dictionary with metrics (mae, rmse, r2) or None if dry_run or insufficient data
    """
    from finance_ml.features import build_features_and_target

    X, y, num_cols, cat_cols = build_features_and_target(df)
    if y is None:
        logging.warning(
            "No suitable numeric target found (price_target or _median). Skipping regression."
        )
        return None

    # Drop rows with NaN target
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if len(X) < 50:
        logging.warning("Too few samples (%d) for meaningful regression. Skipping.", len(X))
        return None

    if dry_run:
        logging.info("Dry run enabled — skipping model fit.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = build_regression_pipeline(num_cols, cat_cols, n_jobs=n_jobs)

    logging.info(
        "Fitting regression model on %d samples, %d features (num=%d, cat=%d)",
        len(X_train),
        X_train.shape[1],
        len(num_cols),
        len(cat_cols),
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))

    logging.info("Regression metrics — MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)

    # Save predictions
    results_df = pd.DataFrame(
        {
            "y_true": y_test.values,
            "y_pred": preds,
            "residual": y_test.values - preds,
        },
        index=y_test.index,
    )
    results_path = out_dir / "regression_predictions.csv"
    results_df.to_csv(results_path, index=False)
    logging.info("Saved regression predictions to %s", results_path)

    return {"model": pipe, "mae": mae, "rmse": rmse, "r2": r2, "predictions": results_df}


def train_and_evaluate_regression_by_sector(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Train and evaluate regression models separately for each sector.

    Computes baseline regression metrics per sector by predicting the
    training-mean of the target on the test split.

    Args:
        df: DataFrame with sector and target columns
        out_dir: Directory to save outputs

    Returns:
        DataFrame with per-sector metrics
    """
    df = normalize_columns(df)
    is_valid, errors = validate_schema(df, require_target=True)
    if not is_valid:
        raise ValueError(f"Schema validation failed: {'; '.join(errors)}")

    # Identify target name
    target_candidates = ["price_target", "price_target_median"]
    y_name = next((t for t in target_candidates if t in df.columns), None)
    if not y_name:
        raise ValueError("No target column found among: price_target, price_target_median")

    rows = []
    for sector, g in df.groupby("sector"):
        y = pd.to_numeric(g[y_name], errors="coerce")
        mask = ~y.isna()
        g = g.loc[mask]
        y = y.loc[mask]
        if len(g) < 10:
            logging.warning("Skipping sector %s due to too few samples: %d", sector, len(g))
            continue

        # baseline split
        idx_train, idx_test = train_test_split(g.index, test_size=0.2, random_state=42)
        y_train = y.loc[idx_train]
        y_test = y.loc[idx_test]
        if len(y_test) == 0 or len(y_train) == 0:
            logging.warning("Skipping sector %s due to empty split.", sector)
            continue
        y_pred = np.full(shape=len(y_test), fill_value=float(y_train.mean()))

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))
        rows.append(
            {
                "sector": sector,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

    metrics = pd.DataFrame(rows)
    out_path = out_dir / "regression_metrics_by_sector.csv"
    metrics.to_csv(out_path, index=False)
    logging.info("Saved per-sector metrics to %s", out_path)
    return metrics


def train_quantile_regression(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    quantiles: List[float] = None,
    random_state: int = 42,
):
    """Train quantile regression models for uncertainty quantification.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
        random_state: Random seed for reproducibility

    Returns:
        QuantileRegressionModel object with predict method
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    # Prepare data
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Remove NaN values
    mask = ~y.isna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 10:
        raise ValueError(f"Insufficient data after cleaning: {len(X)} samples")

    # Train a model for each quantile
    models = {}
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=q,
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
        model.fit(X, y)
        models[q] = model

    # Return object with predict method that delegates to models dict
    class QuantileRegressionModel:
        """Wrapper class for multiple quantile regression models.

        Provides a unified interface to predict multiple quantiles simultaneously.
        """
        def __init__(self, models_dict):
            self.models = models_dict

        def predict(self, X, quantiles=None):
            """Predict method for compatibility with sklearn API"""
            if quantiles is None:
                quantiles = list(self.models.keys())
            return {q: self.models[q].predict(X) for q in quantiles}

    return QuantileRegressionModel(models)


def predict_quantile_regression(
    model, X: pd.DataFrame, quantiles: List[float] = None
) -> pd.DataFrame:
    """Generate predictions for all quantiles.

    Args:
        model: Trained QuantileRegressionModel
        X: Feature DataFrame
        quantiles: List of quantiles to predict (default: use model's quantiles)

    Returns:
        DataFrame with columns q_{quantile} for each quantile
    """
    if quantiles is None:
        quantiles = list(model.models.keys())

    predictions_dict = model.predict(X, quantiles)

    # Create DataFrame with q_{quantile} columns
    result = pd.DataFrame()
    for q in sorted(quantiles):
        col_name = f"q_{q}"
        result[col_name] = predictions_dict[q]

    return result


def train_quantile_regression_by_sector(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    quantiles: List[float] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train separate quantile regression models for each sector.

    Args:
        df: DataFrame with features, target, and sector column
        feature_cols: List of feature column names
        target_col: Target column name
        quantiles: List of quantiles to predict (default: [0.1, 0.5, 0.9])
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping sector names to trained QuantileRegressionModel objects
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    models_by_sector = {}

    for sector, group_df in df.groupby("sector"):
        if len(group_df) < 20:
            logging.warning(
                f"Skipping sector {sector} due to insufficient data: {len(group_df)} samples"
            )
            continue

        try:
            model = train_quantile_regression(
                group_df, feature_cols, target_col, quantiles=quantiles, random_state=random_state
            )
            models_by_sector[sector] = model
            logging.info(
                f"Trained quantile regression for sector {sector} with {len(group_df)} samples"
            )
        except ValueError as e:
            logging.warning(f"Could not train quantile regression for sector {sector}: {e}")
            continue

    return models_by_sector


def train_stacking_ensemble(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, random_state: int = 42
):
    """Train a stacking ensemble with multiple base models and a meta-learner.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        random_state: Random seed for reproducibility

    Returns:
        StackingEnsembleModel with base_models and meta_model attributes
    """
    # Prepare data
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")

    # Remove NaN values
    mask = ~y.isna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 20:
        raise ValueError(f"Insufficient data after cleaning: {len(X)} samples")

    # Define base models (diverse model types for better ensemble)
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state)),
        ("gb", GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state)),
    ]

    # Define meta-learner (simple linear model to combine base predictions)
    meta_model = Ridge(alpha=1.0)

    # Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=3,  # Use cross-validation to generate meta-features
    )

    # Train the stacking ensemble
    stacking_regressor.fit(X, y)

    # Wrap in custom class to expose base_models and meta_model attributes
    class StackingEnsembleModel:
        """Wrapper class for stacking ensemble regressor.

        Exposes base models and meta-learner for inspection and provides
        a unified prediction interface.
        """
        def __init__(self, stacking_reg):
            self.stacking_regressor = stacking_reg
            # estimators_ is a list of fitted base estimators
            self.base_models = stacking_reg.estimators_
            self.meta_model = stacking_reg.final_estimator_

        def predict(self, X):
            """Predict using the stacking ensemble"""
            return self.stacking_regressor.predict(X)

    return StackingEnsembleModel(stacking_regressor)


def train_stacking_ensemble_by_sector(
    df: pd.DataFrame, feature_cols: List[str], target_col: str, random_state: int = 42
) -> Dict[str, Any]:
    """Train separate stacking ensembles for each sector.

    Args:
        df: DataFrame with features, target, and sector column
        feature_cols: List of feature column names
        target_col: Target column name
        random_state: Random seed for reproducibility

    Returns:
        Dictionary mapping sector names to trained StackingEnsembleModel objects
    """
    models_by_sector = {}

    for sector, group_df in df.groupby("sector"):
        if len(group_df) < 30:
            logging.warning(
                f"Skipping sector {sector} due to insufficient data: {len(group_df)} samples"
            )
            continue

        try:
            model = train_stacking_ensemble(
                group_df, feature_cols, target_col, random_state=random_state
            )
            models_by_sector[sector] = model
            logging.info(
                f"Trained stacking ensemble for sector {sector} with {len(group_df)} samples"
            )
        except ValueError as e:
            logging.warning(f"Could not train stacking ensemble for sector {sector}: {e}")
            continue

    return models_by_sector


def monitor_ensemble_training(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Monitor ensemble training with comprehensive logging.

    Args:
        model: sklearn model instance
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for logging

    Returns:
        Tuple of (monitoring_results, y_train_pred, y_test_pred)
        monitoring_results contains:
        - model_name: Name of the model
        - timestamp: Training timestamp
        - training_time_seconds: Time taken to train
        - performance_metrics: dict with train/test metrics
    """
    import time
    from datetime import datetime

    results = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
    }

    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    results["training_time_seconds"] = training_time

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    results["performance_metrics"] = {
        "train_mse": float(train_mse),
        "test_mse": float(test_mse),
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "train_mae": float(train_mae),
        "test_mae": float(test_mae),
        "train_rmse": float(np.sqrt(train_mse)),
        "test_rmse": float(np.sqrt(test_mse)),
    }

    logging.info(
        "Model '%s' trained in %.2fs: train_r2=%.3f, test_r2=%.3f, test_rmse=%.3f",
        model_name,
        training_time,
        train_r2,
        test_r2,
        np.sqrt(test_mse),
    )

    return results, y_train_pred, y_test_pred
