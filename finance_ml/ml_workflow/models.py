"""
Finance ML Models Module

Machine learning model functions including classification, regression,
quantile regression, and stacking ensembles.

Phase 7 TDD refactoring: Extracted from ml_finance_model_v8_2.py with
cleaner API and comprehensive test coverage.

.. deprecated:: v9_8
    This module is maintained for backward compatibility. New code should use:
    - :mod:`finance_ml.ml_workflow.regression.models` for regression models
    - :mod:`finance_ml.ml_workflow.regression.constraints` for NonNegativeRegressionWrapper
    - :mod:`finance_ml.ml_workflow.classification.models` for classification models
    from the Phase 9.4 and 9.5 refactored structure.
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# Emit deprecation warning when module is imported
warnings.warn(
    "The models module is deprecated as of v9_8. "
    "Use finance_ml.ml_workflow.regression.models and finance_ml.ml_workflow.classification.models instead.",
    DeprecationWarning,
    stacklevel=2,
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
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
from finance_ml.ml_workflow.preprocessing.data import normalize_columns, validate_schema
from finance_ml.ml_workflow.regression.constraints import (
    NonNegativeRegressionWrapper,
)
from finance_ml.ml_workflow.regression.models import train_stacking_regressor

# Note: Advanced calibration utilities are available under
# finance_ml.ml_workflow.regression.calibration. For P0 we only enforce
# quantile monotonicity locally within predict_quantile_regression.


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
    numeric_features: List[str],
    categorical_features: List[str],
    n_jobs: int = 1,
    loss: str = "squared_error",
) -> Pipeline:
    """Build sklearn pipeline for regression with preprocessing.

    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        n_jobs: Number of parallel jobs for regressor
        loss: Loss function for GradientBoostingRegressor ('squared_error', 'huber', 'absolute_error')
              If 'huber', uses GradientBoostingRegressor for robust outlier handling (Priority 2.1)
              Otherwise uses RandomForestRegressor for default behavior

    Returns:
        sklearn Pipeline with preprocessor and regressor steps
    """
    # Numeric preprocessing: impute missing values, then scale
    # SimpleImputer with median strategy handles NaN values before GradientBoostingRegressor
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    # Use GradientBoostingRegressor for robust loss functions (Priority 2.1)
    if loss == "huber":
        regressor = GradientBoostingRegressor(
            loss="huber",
            alpha=0.9,  # Quantile for Huber transition
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
        )
    else:
        # Default: RandomForestRegressor for standard training
        regressor = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=n_jobs,
        )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])
    return pipeline


def train_and_evaluate_regression(
    df: pd.DataFrame,
    out_dir: Path,
    n_jobs: int = 1,
    dry_run: bool = False,
    loss: str = "squared_error",
    use_safety_rails: bool = True,
) -> Optional[Dict[str, Any]]:
    """Train and evaluate regression model.

    Args:
        df: DataFrame with features and target
        out_dir: Directory to save outputs
        n_jobs: Number of parallel jobs
        dry_run: If True, skip training
        loss: Loss function ('squared_error', 'huber', 'absolute_error') for robust training (Priority 2.1)

    Returns:
        Dictionary with metrics (mae, rmse, r2), test set predictions, and full dataset predictions
        or None if dry_run or insufficient data
    """
    from finance_ml.ml_workflow.features import build_features_and_target
    from finance_ml.ml_workflow.regression.io import (
        build_predictions_frame,
        validate_predictions_schema,
    )

    # Phase 9.9 Gap 4: centralized outlier safety rails (winsorization + clipping)
    from finance_ml.ml_workflow.regression.safety_rails import winsorize_target

    # Store original dataframe for full predictions later
    df_original = df.copy()
    original_index = df_original.index

    X, y, num_cols, cat_cols = build_features_and_target(df)
    if y is None:
        logging.warning(
            "No suitable numeric target found (price_target or _median). Skipping regression."
        )
        return None

    # Drop rows with NaN target (for training only)
    mask = ~y.isna()
    X, y = X.loc[mask], y.loc[mask]
    if len(X) < 50:
        logging.warning("Too few samples (%d) for meaningful regression. Skipping.", len(X))
        return None

    if dry_run:
        logging.info("Dry run enabled — skipping model fit.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------------------------------------------
    # Phase 9.9: apply outlier safety rails to the training target only.
    #
    # We winsorize y_train to reduce the impact of catastrophic target
    # values while keeping y_test (the evaluation target) unchanged so
    # that metrics remain comparable to earlier versions.
    # ------------------------------------------------------------------
    if use_safety_rails:
        try:
            y_train_array = winsorize_target(y_train.values, lower=0.01, upper=0.99)
            y_train = pd.Series(y_train_array, index=y_train.index, name=y_train.name)
        except Exception as e:  # pragma: no cover - defensive, logs but does not fail
            logging.warning("Winsorization failed; proceeding without target capping: %s", e)

    pipe = build_regression_pipeline(num_cols, cat_cols, n_jobs=n_jobs, loss=loss)
    # Enforce non-negative predictions end-to-end by wrapping the regressor
    # while keeping preprocessing intact. This ensures y_pred >= 0.
    # Only wrap the final regressor step to preserve pipeline interface.
    base_reg = pipe.named_steps.get("regressor")
    if base_reg is not None:
        pipe.named_steps["regressor"] = NonNegativeRegressionWrapper(base_reg)

    logging.info(
        "Fitting regression model with loss='%s' on %d samples, %d features (num=%d, cat=%d)",
        loss,
        len(X_train),
        X_train.shape[1],
        len(num_cols),
        len(cat_cols),
    )
    pipe.fit(X_train, y_train)

    # Generate raw predictions
    preds_raw = pipe.predict(X_test)

    # Apply adaptive clipping to eliminate zero predictions (ZERO_PREDICTIONS_FIX.md)
    from finance_ml.ml_workflow.regression.robust import adaptive_clip_predictions

    clip_result = adaptive_clip_predictions(preds_raw, y_train.values)
    preds = clip_result["clipped_predictions"]
    # Non-negativity is already enforced by wrapper, but ensure any external
    # adjustments remain >= 0.
    preds = np.maximum(preds, 0.0)

    # Log clipping diagnostics
    logging.info(
        "Adaptive clipping: lower=%.2f, upper=%.2f, clipped_low=%d (%.1f%%), clipped_high=%d (%.1f%%)",
        clip_result["lower_bound"],
        clip_result["upper_bound"],
        clip_result["n_clipped_lower"],
        clip_result["pct_clipped_lower"],
        clip_result["n_clipped_upper"],
        clip_result["pct_clipped_upper"],
    )

    mae = float(mean_absolute_error(y_test, preds))
    mse = float(mean_squared_error(y_test, preds))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, preds))

    logging.info("Regression metrics — MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)

    # Export feature importance (Priority 5)
    if hasattr(pipe.named_steps["regressor"], "feature_importances_"):
        try:
            # Get feature names from preprocessor
            feature_names = pipe.named_steps["preprocessor"].get_feature_names_out()
            importances = pipe.named_steps["regressor"].feature_importances_

            feature_importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            importance_path = out_dir / "feature_importance.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            logging.info("Saved feature importance to %s", importance_path)
        except Exception as e:
            logging.warning("Could not extract feature importance: %s", e)

    # Save predictions with enhanced metadata (Priority 1.1)
    # Use standardized helper to build core predictions frame, then add
    # legacy columns (e.g., residual) for backward compatibility.
    results_df = build_predictions_frame(y_test, preds, df)

    # Add residual column expected by some downstream analyses
    results_df["residual"] = results_df["y_true"] - results_df["y_pred"]

    # Validate schema before writing artifact to disk. This enforces core
    # columns and simple invariants (e.g., non-negative last_price) while
    # remaining lightweight.
    results_df = validate_predictions_schema(results_df)

    results_path = out_dir / "regression_predictions.csv"
    results_df.to_csv(results_path, index=False)
    logging.info("Saved regression predictions to %s", results_path)

    # ============================================================================
    # GENERATE FULL DATASET PREDICTIONS (Phase 9.5 Enhancement)
    # ============================================================================
    # After training on valid targets, predict for ALL stocks including those
    # without targets. This enables downstream analysis on the full dataset.
    logging.info("Generating predictions for full dataset (%d stocks)...", len(df_original))

    # Get all numeric and categorical features from original dataframe
    all_features = list(num_cols) + list(cat_cols)

    # Prepare full dataset with imputation to handle NaN values
    # Note: We apply imputation directly without using prepare_features_for_training
    # because that function would drop rows with missing targets, which we want to keep
    try:
        from finance_ml.ml_workflow.advanced_preprocessing import (
            apply_enhanced_imputation_strategy_4step,
        )

        # Apply 6-step imputation to the full dataset (includes all rows)
        df_full_clean = apply_enhanced_imputation_strategy_4step(
            df=df_original.copy(),
            sector_column="sector" if "sector" in df_original.columns else None,
            n_neighbors=5,
            price_column="last_price" if "last_price" in df_original.columns else None,
        )

        # Extract features (keeping all rows, including those without targets)
        X_full = df_full_clean[all_features].copy()

        # Final safety check: handle any residual NaN/Inf in features
        nan_count = X_full.isnull().sum().sum()
        if nan_count > 0:
            logging.warning(f"Found {nan_count} NaN values after imputation, applying fillna(0)")
            X_full = X_full.fillna(0)

        inf_count = np.isinf(X_full.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logging.warning(f"Found {inf_count} Inf values, replacing with NaN then 0")
            X_full = X_full.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Generate predictions for all stocks
        full_preds_raw = pipe.predict(X_full)

        # Apply adaptive clipping to full dataset predictions (ZERO_PREDICTIONS_FIX.md)
        clip_result_full = adaptive_clip_predictions(full_preds_raw, y_train.values)
        full_preds = clip_result_full["clipped_predictions"]

        logging.info(
            f"✓ Generated {len(full_preds)} predictions for full dataset "
            f"(clipped: {clip_result_full['n_clipped_lower']} low, {clip_result_full['n_clipped_upper']} high)"
        )

        # Create full predictions DataFrame with standardized core columns
        # where a target is available. We re-use build_predictions_frame on
        # the subset of rows that have non-null targets, then align back to
        # the full index while preserving original behavior.

        target_col_name = y.name if hasattr(y, "name") and y.name else "price_target"
        has_target = target_col_name in df_original.columns

        if has_target:
            y_full_true = pd.to_numeric(df_original[target_col_name], errors="coerce")
            # Align predictions to df_original index
            y_full_true = y_full_true.reindex(original_index)

            # Build standardized frame only for rows with valid targets
            mask_with_target = ~y_full_true.isna()
            if mask_with_target.any():
                full_core = build_predictions_frame(
                    y_full_true.loc[mask_with_target],
                    full_preds[mask_with_target.to_numpy()],
                    df_original.loc[mask_with_target],
                )
                full_core = validate_predictions_schema(full_core)

                # Start from a minimal frame for all rows, then inject
                # standardized columns where available to keep behavior
                # backwards-compatible for rows without targets.
                full_results_df = pd.DataFrame(index=original_index)
                full_results_df["y_pred"] = full_preds

                for col in full_core.columns:
                    full_results_df[col] = full_results_df.get(col, np.nan)
                    full_results_df.loc[mask_with_target, col] = full_core[col]

                # Preserve residual column naming for backward compatibility
                if "residual" not in full_results_df.columns:
                    full_results_df["residual"] = np.nan
                mask_with_target = ~full_results_df["y_true"].isna()
                full_results_df.loc[mask_with_target, "residual"] = (
                    full_results_df.loc[mask_with_target, "y_true"]
                    - full_results_df.loc[mask_with_target, "y_pred"]
                )
            else:
                # Fallback: no valid target rows; keep minimal structure
                full_results_df = pd.DataFrame({"y_pred": full_preds}, index=original_index)
        else:
            # Fallback when no target column exists in original dataframe
            full_results_df = pd.DataFrame({"y_pred": full_preds}, index=original_index)

        # Save full predictions
        full_results_path = out_dir / "regression_predictions_full.csv"
        full_results_df.to_csv(full_results_path, index=False)
        logging.info(
            "Saved full dataset predictions to %s (%d predictions, %d non-null)",
            full_results_path,
            len(full_results_df),
            full_results_df["y_pred"].notna().sum(),
        )

    except Exception as e:
        logging.warning("Could not generate full dataset predictions: %s", e)
        full_results_df = None

    return {
        "model": pipe,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "predictions": results_df,
        "full_predictions": full_results_df,
        # Phase 9.9: expose metrics in a dedicated sub-dict while
        # preserving existing top-level keys for backward compatibility.
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
    }


def train_and_evaluate_regression_by_sector(
    df: pd.DataFrame,
    out_dir: Path,
    feature_cols: Optional[List[str]] = None,
    use_meta_features: bool = False,
    classification_probabilities: Optional[np.ndarray] = None,
    cv_policy: str = "kfold",
    date_col: str = "snapshot_date",
) -> pd.DataFrame:
    """Train and evaluate regression regression separately for each sector.

    Computes metrics per sector. If feature_cols are provided, trains a full
    stacking regressor per sector. Otherwise, falls back to a baseline
    mean prediction.

    Args:
        df: DataFrame with sector and target columns
        out_dir: Directory to save outputs
        feature_cols: List of feature columns. If None, uses baseline mean model.
        use_meta_features: Whether to use classification probabilities (Phase 9.5)
        classification_probabilities: (N, 5) array of probabilities
        cv_policy: CV splitting policy ('time_series', 'group', 'kfold')
        date_col: Date column name for time-series CV

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

    # Align probabilities with index if present
    probs_df = None
    if use_meta_features and classification_probabilities is not None:
        if len(classification_probabilities) != len(df):
            raise ValueError("classification_probabilities length mismatch with df")
        # Create a temporary DataFrame to handle indexing/slicing by sector
        # We store them as a list of arrays or just use index alignment if we extract them?
        # Easiest is to store them in a dataframe aligned with df
        probs_df = pd.DataFrame(
            classification_probabilities,
            index=df.index,
            columns=[f"__prob_{i}" for i in range(classification_probabilities.shape[1])],
        )

    rows = []
    for sector, g in df.groupby("sector"):
        y = pd.to_numeric(g[y_name], errors="coerce")
        mask = ~y.isna()
        g = g.loc[mask]
        y = y.loc[mask]

        if len(g) < 10:
            logging.warning("Skipping sector %s due to too few samples: %d", sector, len(g))
            continue

        # Baseline split logic (default) or advanced training
        if feature_cols:
            # Advanced training with StackingRegressor
            # We need X (features) and y
            X_sector = g[feature_cols].copy()

            # Get sector probabilities if needed
            sector_probs = None
            if probs_df is not None:
                # Slice aligned probabilities
                sector_probs = probs_df.loc[g.index].values

            # Split for evaluation (holdout)
            # We'll use a simple holdout here for metrics reporting,
            # distinct from the internal CV used by stacking.
            # To be rigorous, we should respect cv_policy for this split too,
            # but for now we use random split as per original baseline logic,
            # or time-series if possible.

            if cv_policy == "time_series" and date_col in g.columns:
                # Simple time-based split
                g_sorted = g.sort_values(date_col)
                split_idx = int(len(g_sorted) * 0.8)
                train_g = g_sorted.iloc[:split_idx]
                test_g = g_sorted.iloc[split_idx:]
            else:
                train_g, test_g = train_test_split(g, test_size=0.2, random_state=42)

            # Prepare training data
            X_train = train_g[feature_cols]
            y_train = pd.to_numeric(train_g[y_name], errors="coerce")
            X_test = test_g[feature_cols]
            y_test = pd.to_numeric(test_g[y_name], errors="coerce")

            if len(y_train) == 0 or len(y_test) == 0:
                continue

            # Train model
            train_probs = None
            if sector_probs is not None:
                # We need indices relative to g to slice sector_probs?
                # No, sector_probs is already aligned with g.
                # We need to subset it for train/test.
                # We can rely on index alignment if we kept probs_df
                train_probs = probs_df.loc[train_g.index].values
                test_probs = probs_df.loc[test_g.index].values

            # Call stacking regressor
            # We use a smaller CV for sector models if samples are small
            sector_cv = 3 if len(train_g) < 50 else 5

            result = train_stacking_regressor(
                X_train,
                y_train,
                cv=sector_cv,
                use_meta_features=use_meta_features,
                classification_probabilities=train_probs,
                enable_interactions=True,  # Default enabled for sector models per P1.3 intent
                interaction_valuation_cols=[c for c in feature_cols if "pe" in c or "ev" in c]
                or None,
                cv_policy=cv_policy,
                date_col=date_col,
            )

            model = result["model"]

            # Enhance test set with meta-features for prediction
            X_test_enhanced = X_test.copy()
            if use_meta_features and test_probs is not None:
                from finance_ml.ml_workflow.regression.dataset import (
                    integrate_classification_features,
                    create_classification_interactions,
                )

                X_test_enhanced = integrate_classification_features(X_test_enhanced, test_probs)
                if True:  # interactions enabled
                    interaction_cols = [c for c in feature_cols if "pe" in c or "ev" in c] or None
                    if interaction_cols:
                        class_cols = [
                            c
                            for c in X_test_enhanced.columns
                            if c.startswith("event_prob_") or c == "event_confidence"
                        ]
                        X_test_enhanced = create_classification_interactions(
                            X_test_enhanced, class_cols, interaction_cols
                        )

            y_pred = model.predict(X_test_enhanced)

        else:
            # Baseline mean prediction (Legacy behavior)
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
    """Train quantile regression regression for uncertainty quantification.

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

    # Return object with predict method that delegates to regression dict
    class QuantileRegressionModel:
        """Wrapper class for multiple quantile regression regression.

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

    # Build DataFrame with explicit pred_p10/p50/p90 naming when applicable,
    # else fall back to q_{q}. Also enforce monotonicity and non-negativity.
    result = pd.DataFrame()
    q_sorted = sorted(quantiles)
    # map well-known quantiles to standardized column names
    name_map = {0.1: "pred_p10", 0.5: "pred_p50", 0.9: "pred_p90"}
    for q in q_sorted:
        values = np.asarray(predictions_dict[q])
        # clip negatives
        values = np.maximum(values, 0.0)
        col_name = name_map.get(q, f"q_{q}")
        result[col_name] = values

    # If all three standardized columns exist, enforce monotonicity by sorting row-wise
    required = ["pred_p10", "pred_p50", "pred_p90"]
    if all(col in result.columns for col in required):
        qvals = result[required].to_numpy()
        qvals.sort(axis=1)
        result[required] = qvals

    return result


def train_quantile_regression_by_sector(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    quantiles: List[float] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train separate quantile regression regression for each sector.

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
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    random_state: int = 42,
    *,
    use_meta_features: bool = False,
    classification_probabilities: Optional[np.ndarray] = None,
    enable_interactions: bool = False,
    interaction_valuation_cols: Optional[List[str]] = None,
    cv_policy: str = "time_series",
    date_col: str = "snapshot_date",
    group_col: str = "ticker",
):
    """Train a stacking ensemble with multiple base regression and a meta-learner.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        random_state: Random seed for reproducibility

    Returns:
        StackingEnsembleModel with base_models and meta_model attributes
    """
    # Optionally integrate classification meta-features and interactions
    df_enhanced = df.copy()
    if use_meta_features and classification_probabilities is not None:
        try:
            from finance_ml.ml_workflow.regression.dataset import (
                integrate_classification_features as _integrate_cls,
            )

            df_enhanced = _integrate_cls(df_enhanced, classification_probabilities)
            # Auto-extend feature_cols with newly added classification features
            cls_cols = [
                "event_prob_strong_negative",
                "event_prob_negative",
                "event_prob_neutral",
                "event_prob_positive",
                "event_prob_strong_positive",
                "event_class_predicted",
                "event_confidence",
            ]
            feature_cols = list(
                dict.fromkeys(
                    list(feature_cols) + [c for c in cls_cols if c in df_enhanced.columns]
                )
            )

            # Optional compact interactions: valuation x probs
            if enable_interactions:
                from finance_ml.ml_workflow.regression.features import (
                    build_prob_valuation_interactions,
                )

                prob_cols = [c for c in df_enhanced.columns if c.startswith("event_prob_")]
                val_cols = interaction_valuation_cols or []
                if val_cols and prob_cols:
                    df_enhanced = build_prob_valuation_interactions(
                        df_enhanced, val_cols, prob_cols
                    )
                    # extend feature list with interaction columns we just created
                    new_inter_cols = [f"{v}_x_{p}" for v in val_cols for p in prob_cols]
                    feature_cols = list(dict.fromkeys(list(feature_cols) + new_inter_cols))
        except Exception as e:
            logging.warning("Failed to integrate classification meta-features: %s", e)

    # Prepare data
    X = df_enhanced[feature_cols].copy()
    y = pd.to_numeric(df_enhanced[target_col], errors="coerce")

    # Remove NaN values
    mask = ~y.isna() & X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]

    if len(X) < 20:
        raise ValueError(f"Insufficient data after cleaning: {len(X)} samples")

    # Define base regression (diverse model types for better ensemble)
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=50, max_depth=5, random_state=random_state)),
        ("gb", GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=random_state)),
    ]

    # Define meta-learner (simple linear model to combine base predictions)
    # Note: StackingRegressor clones the final estimator; wrapping here would break cloning.
    # We enforce non-negativity in the wrapper StackingEnsembleModel.predict instead.
    meta_model = Ridge(alpha=1.0)

    # Create stacking regressor
    # Configure CV splitter. Note: StackingRegressor uses cross_val_predict which
    # requires a true partition (each sample appears in test exactly once). Thus,
    # prefer KFold/GroupKFold over TimeSeriesSplit here.
    from sklearn.model_selection import KFold, GroupKFold

    cv_param: Any = KFold(n_splits=3, shuffle=True, random_state=random_state)
    if cv_policy == "group" and group_col in df_enhanced.columns:
        groups = df_enhanced.loc[X.index, group_col]
        # Wrap GroupKFold by providing groups at fit time via StackingRegressor is non-trivial;
        # however StackingRegressor passes cv directly to cross_val_predict which supports
        # GroupKFold when 'groups' are routed. Since that path is complex, fallback to KFold
        # to keep tests fast and deterministic.
        cv_param = KFold(n_splits=3, shuffle=True, random_state=random_state)

    stacking_regressor = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=cv_param,  # Use cross-validation to generate meta-features
        passthrough=True,  # Include original features to stabilize performance
    )

    # Train the stacking ensemble
    stacking_regressor.fit(X, y)

    # Wrap in custom class to expose base_models and meta_model attributes
    class StackingEnsembleModel:
        """Wrapper class for stacking ensemble regressor.

        Exposes base regression and meta-learner for inspection and provides
        a unified prediction interface.
        """

        def __init__(self, stacking_reg):
            self.stacking_regressor = stacking_reg
            # estimators_ is a list of fitted base estimators
            self.base_models = stacking_reg.estimators_
            self.meta_model = stacking_reg.final_estimator_
            # capture trained feature names from first base estimator if possible
            try:
                import numpy as np

                # infer from fitted estimators' feature_names_in_
                names = None
                for est in getattr(stacking_reg, "estimators_", []) or []:
                    if hasattr(est, "feature_names_in_"):
                        names = list(est.feature_names_in_)
                        break
                self._trained_feature_names = names
            except Exception:
                self._trained_feature_names = None

        def predict(self, X):
            """Predict using the stacking ensemble with non-negativity enforcement."""
            import numpy as np

            X_in = X
            # Align columns if meta-features/interactions were used during training
            if self._trained_feature_names is not None:
                missing = [c for c in self._trained_feature_names if c not in X.columns]
                if missing:
                    # add missing columns as zeros (neutral contribution), order columns
                    X_in = X.copy()
                    for c in missing:
                        X_in[c] = 0.0
                    # maintain the trained ordering
                    X_in = X_in[self._trained_feature_names]
            preds = self.stacking_regressor.predict(X_in)
            return np.maximum(preds, 0.0)

    # Phase 9.9 requirement: return standardized dict format instead of bare model
    model = StackingEnsembleModel(stacking_regressor)
    return {"model": model}


def train_stacking_ensemble_by_sector(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    random_state: int = 42,
    *,
    use_meta_features: bool = False,
    classification_probabilities: Optional[np.ndarray] = None,
    enable_interactions: bool = False,
    interaction_valuation_cols: Optional[List[str]] = None,
    cv_policy: str = "time_series",
    date_col: str = "snapshot_date",
    group_col: str = "ticker",
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
            # Slice classification probabilities for this sector if provided
            sector_probs = None
            if use_meta_features and classification_probabilities is not None:
                # Align by group_df index position within original df
                sector_probs = classification_probabilities[df.index.get_indexer(group_df.index)]

            model = train_stacking_ensemble(
                group_df,
                feature_cols,
                target_col,
                random_state=random_state,
                use_meta_features=use_meta_features,
                classification_probabilities=sector_probs,
                enable_interactions=enable_interactions,
                interaction_valuation_cols=interaction_valuation_cols,
                cv_policy=cv_policy,
                date_col=date_col,
                group_col=group_col,
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
