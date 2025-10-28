"""
Phase 9.4: Multi-Class Classification Module

Sophisticated event classification with multiple model architectures:
- Gradient Boosting: XGBoost, LightGBM, CatBoost
- Neural Networks: Feedforward DNN with batch normalization
- Ensemble Methods: Voting and Stacking classifiers
- Class Imbalance Handling: SMOTE, ADASYN, class weights
- Model Interpretation: SHAP values, feature importance

Classes:
- 0: Neutral (price_target within ±10% of last_price)
- 1: Positive Catalyst (price_target > last_price by >=10%)
- 2: Negative Catalyst (price_target < last_price by >=10%)
"""

import logging
import warnings
from typing import Dict, Any, Optional, List, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    )
from sklearn.model_selection import train_test_split

# Optional imports with fallback handling
try:
    import xgboost as xgb
    HAVE_XGBOOST = True
except ImportError:
    HAVE_XGBOOST = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except ImportError:
    HAVE_LIGHTGBM = False
    warnings.warn("LightGBM not available. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
    HAVE_CATBOOST = True
except ImportError:
    HAVE_CATBOOST = False
    warnings.warn("CatBoost not available. Install with: pip install catboost")

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False
    warnings.warn("imbalanced-learn not available. Install with: pip install imbalanced-learn")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAVE_TENSORFLOW = True
except ImportError:
    HAVE_TENSORFLOW = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    warnings.warn("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


def _prepare_categorical_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, categorical_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare categorical features using one-hot encoding (pd.get_dummies).

    This function replaces LabelEncoder with one-hot encoding for robust handling
    of categorical variables, including unseen categories in test set.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        categorical_cols: List of categorical column names

    Returns:
        Tuple of (X_train_encoded, X_test_encoded) with one-hot encoded categoricals
    """
    if not categorical_cols:
        return X_train.copy(), X_test.copy()

    # Separate numeric and categorical columns
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    # One-hot encode training set
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False)

    # One-hot encode test set
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False)

    # Align test set columns with training set
    # Add missing columns (fill with 0)
    for col in X_train_encoded.columns:
        if col not in X_test_encoded.columns:
            X_test_encoded[col] = 0

    # Remove extra columns in test set that weren't in training
    X_test_encoded = X_test_encoded[X_train_encoded.columns]

    return X_train_encoded, X_test_encoded


def create_enhanced_event_labels(
    df: pd.DataFrame,
    method: str = "price_momentum",
    threshold_positive: float = 10.0,
    threshold_negative: float = -10.0,
    use_sector_adjustment: bool = False,
) -> np.ndarray:
    """Create sophisticated event classification labels.

    Multiple methods for event detection:
    1. price_momentum: Based on price target vs current price
    2. valuation: Based on valuation metric percentiles (P/E, P/B)
    3. fundamental: Based on margin expansion/contraction
    4. volatility: Based on price volatility spikes

    Args:
        df: DataFrame with required columns
        method: Event detection method ('price_momentum', 'valuation', 'fundamental', 'volatility')
        threshold_positive: Threshold for positive catalyst (%)
        threshold_negative: Threshold for negative catalyst (%)
        use_sector_adjustment: If True, adjust thresholds by sector volatility

    Returns:
        numpy array of labels (0=Neutral, 1=Positive, 2=Negative)
    """
    labels = np.zeros(len(df), dtype=int)

    if method == "price_momentum":
        # Price target momentum
        if "price_target" not in df.columns or "last_price" not in df.columns:
            logger.warning("price_target or last_price not available, returning all neutral")
            return labels

        price_diff_pct = (df["price_target"] - df["last_price"]) / df["last_price"] * 100.0

        # Sector-specific adjustment
        if use_sector_adjustment and "sector" in df.columns:
            for sector in df["sector"].unique():
                sector_mask = df["sector"] == sector
                sector_vol = price_diff_pct[sector_mask].std()
                # Adjust thresholds based on sector volatility
                adj_positive = threshold_positive * (1 + sector_vol / 50.0)
                adj_negative = threshold_negative * (1 + sector_vol / 50.0)

                labels[sector_mask & (price_diff_pct >= adj_positive)] = 1
                labels[sector_mask & (price_diff_pct <= adj_negative)] = 2
        else:
            labels[price_diff_pct >= threshold_positive] = 1
            labels[price_diff_pct <= threshold_negative] = 2

    elif method == "valuation":
        # Valuation-based events (undervalued = positive, overvalued = negative)
        if "p_e" not in df.columns:
            logger.warning("p_e not available for valuation method, returning all neutral")
            return labels

        # Calculate percentiles within sector
        if "sector" in df.columns:
            df["p_e_percentile"] = df.groupby("sector")["p_e"].rank(pct=True)
        else:
            df["p_e_percentile"] = df["p_e"].rank(pct=True)

        # Low P/E (undervalued) = positive, High P/E (overvalued) = negative
        labels[df["p_e_percentile"] <= 0.25] = 1  # Bottom quartile = positive
        labels[df["p_e_percentile"] >= 0.75] = 2  # Top quartile = negative

    elif method == "fundamental":
        # Fundamental events based on margin trends
        margin_cols = [c for c in ["gross_margin", "operating_margin", "net_margin"] if c in df.columns]
        if not margin_cols:
            logger.warning("No margin columns available, returning all neutral")
            return labels

        # Calculate average margin score
        margin_data = df[margin_cols].fillna(0)
        avg_margin = margin_data.mean(axis=1)

        # High margins = positive, low margins = negative
        labels[avg_margin >= avg_margin.quantile(0.7)] = 1
        labels[avg_margin <= avg_margin.quantile(0.3)] = 2

    elif method == "volatility":
        # Volatility-based events
        vol_cols = [c for c in df.columns if "volatility" in c.lower()]
        if not vol_cols:
            logger.warning("No volatility columns available, returning all neutral")
            return labels

        volatility = df[vol_cols[0]]
        # High volatility = negative (risk), low volatility = positive (stable)
        labels[volatility <= volatility.quantile(0.3)] = 1
        labels[volatility >= volatility.quantile(0.7)] = 2

    else:
        logger.error(f"Unknown method: {method}")

    logger.info(
        f"Created labels with method={method}: Neutral={np.sum(labels == 0)}, "
        f"Positive={np.sum(labels == 1)}, Negative={np.sum(labels == 2)}"
    )

    return labels


def prepare_classification_data(
    df: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, List[str], List[str]]:
    """Prepare data for classification with train/test split.

    Args:
        df: DataFrame with features
        labels: Target labels
        test_size: Proportion of test set
        random_state: Random seed

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, numeric_cols, categorical_cols)
    """
    # Drop non-feature columns
    drop_cols = [
        "ticker", "isin", "name", "description", "price_target", "price_target_median",
        "last_updated", "income_statement_report_date", "p_e_percentile"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Handle duplicate columns
    if X.columns.duplicated().any():
        logger.warning(f"Removing {X.columns.duplicated().sum()} duplicate columns")
        X = X.loc[:, ~X.columns.duplicated(keep="first")]

    # Identify numeric and categorical columns
    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    # Fill NaN values
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in categorical_cols:
        X[col] = X[col].fillna("Unknown")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    logger.info(
        f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples, "
        f"Features: {len(numeric_cols)} numeric + {len(categorical_cols)} categorical"
    )

    return X_train, X_test, y_train, y_test, numeric_cols, categorical_cols


def train_xgboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train XGBoost classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional XGBoost parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_XGBOOST:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")

    # Default parameters
    default_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "eval_metric": "mlogloss",
    }
    if params:
        default_params.update(params)

    # Prepare data: encode categoricals and scale numerics
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Get updated numeric columns list (after one-hot encoding)
    encoded_numeric_cols = [
        col for col in X_train_proc.columns if col not in categorical_cols or "_" in col
    ]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc[encoded_numeric_cols] = scaler.fit_transform(X_train_proc[encoded_numeric_cols])
    X_test_proc[encoded_numeric_cols] = scaler.transform(X_test_proc[encoded_numeric_cols])

    # Train model
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)], verbose=False)

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
    }


def train_lightgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train LightGBM classifier.

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional LightGBM parameters

    Returns:
        Dictionary with model, predictions, and metrics
    """
    if not HAVE_LIGHTGBM:
        raise ImportError("LightGBM not available. Install with: pip install lightgbm")

    # Default parameters
    default_params = {
        "objective": "multiclass",
        "num_class": 3,
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }
    if params:
        default_params.update(params)

    # Prepare data
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Get updated numeric columns list (after one-hot encoding)
    encoded_numeric_cols = [
        col for col in X_train_proc.columns if col not in categorical_cols or "_" in col
    ]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc[encoded_numeric_cols] = scaler.fit_transform(X_train_proc[encoded_numeric_cols])
    X_test_proc[encoded_numeric_cols] = scaler.transform(X_test_proc[encoded_numeric_cols])

    # Train model
    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train_proc, y_train, eval_set=[(X_test_proc, y_test)])

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"LightGBM - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
    }


def train_catboost_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train CatBoost classifier.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional CatBoost parameters

    Returns:
        Dictionary with model, predictions, and metrics
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
    """
    if not HAVE_CATBOOST:
        raise ImportError("CatBoost not available. Install with: pip install catboost")

    # Default parameters
    default_params = {
        "iterations": 200,
        "depth": 6,
        "learning_rate": 0.1,
        "loss_function": "MultiClass",
        "random_seed": 42,
        "verbose": False,
    }
    if params:
        default_params.update(params)

    # Prepare data - CatBoost handles categoricals natively
    from sklearn.preprocessing import StandardScaler

    X_train_proc = X_train.copy()
    X_test_proc = X_test.copy()

    # Scale numeric features only
    scaler = StandardScaler()
    X_train_proc[numeric_cols] = scaler.fit_transform(X_train_proc[numeric_cols])
    X_test_proc[numeric_cols] = scaler.transform(X_test_proc[numeric_cols])

    # Train model with categorical features
    model = CatBoostClassifier(**default_params)
    model.fit(
        X_train_proc,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_test_proc, y_test),
    )

    # Predictions
    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)

    logger.info(f"CatBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "feature_importance": dict(zip(X_train.columns, model.feature_importances_)),
    }


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Comprehensive classification evaluation.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        class_names: Names for classes (default: ['Neutral', 'Positive', 'Negative'])

    Returns:
        Dictionary with metrics, confusion matrix, and classification report
    """
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    # ROC-AUC (if probabilities available)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            logger.warning("Could not compute ROC-AUC (likely due to missing classes)")

    return {
        "accuracy": accuracy,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
    }


def apply_smote(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    numeric_cols: List[str],
    sampling_strategy: str = "auto",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Apply SMOTE for class imbalance.

    Args:
        X_train: Training features
        y_train: Training labels
        numeric_cols: Numeric columns (SMOTE requires numeric data)
        sampling_strategy: Sampling strategy ('auto', 'minority', or dict)
        random_state: Random seed

    Returns:
        Tuple of (X_resampled, y_resampled)
    """
    if not HAVE_IMBLEARN:
        logger.warning("imbalanced-learn not available, returning original data")
        return X_train, y_train

    # SMOTE requires numeric data
    X_numeric = X_train[numeric_cols].copy()

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_numeric, y_train)

    logger.info(
        f"SMOTE applied: {len(y_train)} -> {len(y_resampled)} samples. "
        f"Class distribution: {np.bincount(y_resampled)}"
    )

    return pd.DataFrame(X_resampled, columns=numeric_cols), y_resampled


def train_neural_network_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train Neural Network classifier with TensorFlow/Keras.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        params: Optional NN parameters

    Returns:
        Dictionary with model, predictions, and metrics
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
    """
    if not HAVE_TENSORFLOW:
        raise ImportError("TensorFlow not available. Install with: pip install tensorflow")

    from sklearn.preprocessing import StandardScaler

    # Default parameters
    default_params = {
        "hidden_layers": [256, 128, 64],
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "epochs": 50,
        "batch_size": 32,
        "validation_split": 0.2,
    }
    if params:
        default_params.update(params)

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Scale all features for neural network
    scaler = StandardScaler()
    X_train_proc = pd.DataFrame(
        scaler.fit_transform(X_train_proc), columns=X_train_proc.columns, index=X_train_proc.index
    )
    X_test_proc = pd.DataFrame(
        scaler.transform(X_test_proc), columns=X_test_proc.columns, index=X_test_proc.index
    )

    # Build neural network
    model = keras.Sequential()
    model.add(layers.Input(shape=(X_train_proc.shape[1],)))

    for units in default_params["hidden_layers"]:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(default_params["dropout_rate"]))

    model.add(layers.Dense(3, activation="softmax"))

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=default_params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train model
    history = model.fit(
        X_train_proc.values,
        y_train,
        epochs=default_params["epochs"],
        batch_size=default_params["batch_size"],
        validation_split=default_params["validation_split"],
        verbose=0,
    )

    # Predictions
    y_proba = model.predict(X_test_proc.values, verbose=0)
    y_pred = np.argmax(y_proba, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Neural Network - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "history": history.history,
    }


def train_voting_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
    voting: Literal["hard", "soft"] = "soft",
) -> Dict[str, Any]:
    """Train Voting classifier ensemble.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names
        voting: 'soft' or 'hard' voting

    Returns:
        Dictionary with model, predictions, and metrics
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
    """
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Get updated numeric columns list (after one-hot encoding)
    encoded_numeric_cols = [
        col for col in X_train_proc.columns if col not in categorical_cols or "_" in col
    ]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc[encoded_numeric_cols] = scaler.fit_transform(X_train_proc[encoded_numeric_cols])
    X_test_proc[encoded_numeric_cols] = scaler.transform(X_test_proc[encoded_numeric_cols])

    # Create base estimators
    estimators = []

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    estimators.append(("rf", rf))

    # XGBoost
    if HAVE_XGBOOST:
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
        )
        estimators.append(("xgb", xgb_clf))

    # LightGBM
    if HAVE_LIGHTGBM:
        lgb_clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgb", lgb_clf))

    # Create voting classifier
    voting_clf = VotingClassifier(estimators=estimators, voting=voting)
    voting_clf.fit(X_train_proc, y_train)

    # Predictions
    y_pred = voting_clf.predict(X_test_proc)
    y_proba = voting_clf.predict_proba(X_test_proc) if voting == "soft" else None

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Voting Classifier ({voting}) - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": voting_clf,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def train_stacking_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Train Stacking classifier ensemble.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names

    Returns:
        Dictionary with model, predictions, and metrics
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Get updated numeric columns list (after one-hot encoding)
    encoded_numeric_cols = [
        col for col in X_train_proc.columns if col not in categorical_cols or "_" in col
    ]

    # Scale numeric features
    scaler = StandardScaler()
    X_train_proc[encoded_numeric_cols] = scaler.fit_transform(X_train_proc[encoded_numeric_cols])
    X_test_proc[encoded_numeric_cols] = scaler.transform(X_test_proc[encoded_numeric_cols])

    # Create base estimators
    estimators = []

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, class_weight="balanced"
    )
    estimators.append(("rf", rf))

    # XGBoost
    if HAVE_XGBOOST:
        xgb_clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
        )
        estimators.append(("xgb", xgb_clf))

    # LightGBM
    if HAVE_LIGHTGBM:
        lgb_clf = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            random_state=42,
            verbose=-1,
        )
        estimators.append(("lgb", lgb_clf))

    # Create stacking classifier with Logistic Regression as meta-learner
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
    )
    stacking_clf.fit(X_train_proc, y_train)

    # Predictions
    y_pred = stacking_clf.predict(X_test_proc)
    y_proba = stacking_clf.predict_proba(X_test_proc)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    logger.info(f"Stacking Classifier - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return {
        "model": stacking_clf,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 100,
) -> Dict[str, Any]:
    """Compute SHAP values for model interpretation.

    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data (for SHAP values)
        max_samples: Maximum samples for SHAP computation

    Returns:
        Dictionary with SHAP values and explainer
    """
    if not HAVE_SHAP:
        logger.warning("SHAP not available, skipping interpretation")
        return {}

    # Sample data for performance
    X_train_sample = X_train.sample(min(max_samples, len(X_train)), random_state=42)
    X_test_sample = X_test.sample(min(max_samples, len(X_test)), random_state=42)

    try:
        # Create explainer based on model type
        if hasattr(model, "predict_proba"):
            explainer = shap.Explainer(model.predict_proba, X_train_sample)
        else:
            explainer = shap.Explainer(model.predict, X_train_sample)

        # Compute SHAP values
        shap_values = explainer(X_test_sample)

        logger.info(f"SHAP values computed for {len(X_test_sample)} samples")

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "X_test_sample": X_test_sample,
        }

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {}


def export_classification_features(
    df: pd.DataFrame,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Export classification probabilities as meta-features for regression.

    Args:
        df: Original DataFrame
        y_proba: Predicted probabilities (n_samples, n_classes)
        class_names: Names for classes

    Returns:
        DataFrame with added classification meta-features
    """
    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]

    df_with_features = df.copy()

    # Add probability columns
    for i, class_name in enumerate(class_names):
        df_with_features[f"event_prob_{class_name.lower()}"] = y_proba[:, i]

    # Add predicted class
    df_with_features["event_class_predicted"] = np.argmax(y_proba, axis=1)

    # Add confidence (max probability)
    df_with_features["event_confidence"] = np.max(y_proba, axis=1)

    logger.info(f"Added {len(class_names) + 2} classification meta-features")

    return df_with_features


def validate_data_quality(X: pd.DataFrame, feature_names: Optional[List[str]] = None) -> bool:
    """Validate data quality and report issues.

    Args:
        X: Input dataframe to validate
        feature_names: Optional list of feature names to validate (defaults to all columns)

    Returns:
        True if no issues detected, False otherwise
    """
    issues = []

    cols_to_check = feature_names if feature_names is not None else X.columns

    for col in cols_to_check:
        if col not in X.columns:
            continue

        col_data = X[col]

        # Check for infinities
        if np.any(np.isinf(col_data)):
            inf_count = np.sum(np.isinf(col_data))
            issues.append(f"Column {col}: {inf_count} infinite values")

        # Check for extremely large values
        max_val = np.nanmax(np.abs(col_data))
        if max_val > 1e10:
            issues.append(f"Column {col}: extremely large values (max={max_val:.2e})")

    if issues:
        logger.warning("⚠️ Data Quality Issues Detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    return len(issues) == 0


def compare_classifiers(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> pd.DataFrame:
    """Compare multiple classifiers and return results table.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        numeric_cols: Numeric feature names
        categorical_cols: Categorical feature names

    Returns:
        DataFrame with comparison results
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
    """
    results = []

    # Random Forest (baseline)
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")

    # Prepare data for sklearn models
    from sklearn.preprocessing import StandardScaler

    # One-hot encode categorical features
    X_train_proc, X_test_proc = _prepare_categorical_features(X_train, X_test, categorical_cols)

    # Get updated numeric columns list (after one-hot encoding)
    encoded_numeric_cols = [
        col for col in X_train_proc.columns if col not in categorical_cols or "_" in col
    ]

    # Clean infinite and extreme values before scaling
    logger.info("Cleaning infinite and extreme values before scaling...")

    # Replace infinities with NaN first
    X_train_proc[encoded_numeric_cols] = X_train_proc[encoded_numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )
    X_test_proc[encoded_numeric_cols] = X_test_proc[encoded_numeric_cols].replace(
        [np.inf, -np.inf], np.nan
    )

    # Fill NaN values with column median (more robust than mean for outliers)
    for col in encoded_numeric_cols:
        median_val = X_train_proc[col].median()
        if np.isnan(median_val):  # If entire column is NaN, use 0
            median_val = 0
        X_train_proc[col].fillna(median_val, inplace=True)
        X_test_proc[col].fillna(median_val, inplace=True)

    # Cap extreme outliers (values beyond 3 standard deviations)
    for col in encoded_numeric_cols:
        mean_val = X_train_proc[col].mean()
        std_val = X_train_proc[col].std()
        if std_val > 0:  # Avoid division by zero
            lower_bound = mean_val - 3 * std_val
            upper_bound = mean_val + 3 * std_val
            X_train_proc[col] = X_train_proc[col].clip(lower_bound, upper_bound)
            X_test_proc[col] = X_test_proc[col].clip(lower_bound, upper_bound)

    # Now safely scale the cleaned data
    scaler = StandardScaler()
    X_train_proc[encoded_numeric_cols] = scaler.fit_transform(X_train_proc[encoded_numeric_cols])
    X_test_proc[encoded_numeric_cols] = scaler.transform(X_test_proc[encoded_numeric_cols])

    rf.fit(X_train_proc, y_train)
    y_pred_rf = rf.predict(X_test_proc)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = precision_recall_fscore_support(y_test, y_pred_rf, average="macro", zero_division=0)[2]
    results.append({"Model": "Random Forest", "Accuracy": acc_rf, "F1-Score": f1_rf})

    # XGBoost
    if HAVE_XGBOOST:
        logger.info("Training XGBoost...")
        xgb_result = train_xgboost_classifier(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
        results.append({"Model": "XGBoost", "Accuracy": xgb_result["accuracy"], "F1-Score": xgb_result["f1_score"]})

    # LightGBM
    if HAVE_LIGHTGBM:
        logger.info("Training LightGBM...")
        lgb_result = train_lightgbm_classifier(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
        results.append({"Model": "LightGBM", "Accuracy": lgb_result["accuracy"], "F1-Score": lgb_result["f1_score"]})

    # CatBoost
    if HAVE_CATBOOST:
        logger.info("Training CatBoost...")
        cb_result = train_catboost_classifier(X_train, y_train, X_test, y_test, numeric_cols, categorical_cols)
        results.append({"Model": "CatBoost", "Accuracy": cb_result["accuracy"], "F1-Score": cb_result["f1_score"]})

    # Neural Network
    if HAVE_TENSORFLOW:
        logger.info("Training Neural Network...")
        nn_result = train_neural_network_classifier(
            X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
        )
        results.append(
            {
                "Model": "Neural Network",
                "Accuracy": nn_result["accuracy"],
                "F1-Score": nn_result["f1_score"],
            }
        )

    # Voting Classifier
    logger.info("Training Voting Classifier...")
    voting_result = train_voting_classifier(
        X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
    )
    results.append(
        {
            "Model": "Voting Ensemble",
            "Accuracy": voting_result["accuracy"],
            "F1-Score": voting_result["f1_score"],
        }
    )

    # Stacking Classifier
    logger.info("Training Stacking Classifier...")
    stacking_result = train_stacking_classifier(
        X_train, y_train, X_test, y_test, numeric_cols, categorical_cols
    )
    results.append(
        {
            "Model": "Stacking Ensemble",
            "Accuracy": stacking_result["accuracy"],
            "F1-Score": stacking_result["f1_score"],
        }
    )

    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
    logger.info(f"Model comparison complete. Best model: {results_df.iloc[0]['Model']}")

    return results_df


from sklearn.base import ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Any, Optional, Union


def cross_validate_classifier(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int = 5,
    stratify_by: Optional[str] = None,
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """Perform stratified cross-validation for classifier.

    Args:
        model: Classifier to evaluate
        X: Feature data
        y: Labels
        cv: Number of folds
        stratify_by: Column to stratify by (e.g., 'sector')

    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_validate

    # Setup cross-validation strategy
    if stratify_by and stratify_by in X.columns:

        groups = X[stratify_by]
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Perform cross-validation
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    # Aggregate results
    results = {
        "test_accuracy": cv_results["test_accuracy"].mean(),
        "test_accuracy_std": cv_results["test_accuracy"].std(),
        "test_precision": cv_results["test_precision"].mean(),
        "test_precision_std": cv_results["test_precision"].std(),
        "test_recall": cv_results["test_recall"].mean(),
        "test_recall_std": cv_results["test_recall"].std(),
        "test_f1": cv_results["test_f1"].mean(),
        "test_f1_std": cv_results["test_f1"].std(),
        "train_accuracy": cv_results["train_accuracy"].mean(),
        "cv_scores": cv_results,
    }

    logger.info(
        f"Cross-validation ({cv} folds): Accuracy={results['test_accuracy']:.4f}±{results['test_accuracy_std']:.4f}, "
        f"F1={results['test_f1']:.4f}±{results['test_f1_std']:.4f}"
    )

    return results


def compare_feature_importance(
    models_dict: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Compare feature importance across multiple models.

    Args:
        models_dict: Dictionary of model results (from train_*_classifier functions)
        feature_names: List of feature names
        top_n: Number of top features to display

    Returns:
        DataFrame with feature importance comparison
    """
    importance_data = {}

    for model_name, model_result in models_dict.items():
        if "feature_importance" in model_result:
            importance_dict = model_result["feature_importance"]
            importance_data[model_name] = pd.Series(importance_dict)

    if not importance_data:
        logger.warning("No feature importance available")
        return pd.DataFrame()

    # Create comparison dataframe
    importance_df = pd.DataFrame(importance_data)

    # Add average importance
    importance_df["Average"] = importance_df.mean(axis=1)

    # Sort by average and take top N
    importance_df = importance_df.sort_values("Average", ascending=False).head(top_n)

    logger.info(f"Feature importance comparison for top {top_n} features")

    return importance_df


def plot_confusion_matrices(
    models_results: Dict[str, Dict[str, Any]],
    class_names: Optional[List[str]] = None,
) -> None:
    """Plot confusion matrices for multiple models.

    Args:
        models_results: Dictionary of model results with y_test and y_pred
        class_names: Names for classes
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not available for plotting")
        return

    if class_names is None:
        class_names = ["Neutral", "Positive", "Negative"]

    n_models = len(models_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for idx, (model_name, results) in enumerate(models_results.items()):
        if "y_pred" in results and "y_test" in results:
            cm = confusion_matrix(results["y_test"], results["y_pred"])
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=axes[idx],
            )
            axes[idx].set_title(f"{model_name}")
            axes[idx].set_ylabel("True Label")
            axes[idx].set_xlabel("Predicted Label")

    plt.tight_layout()
    plt.show()


def evaluate_classification_by_sector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sectors: pd.Series,
) -> pd.DataFrame:
    """Evaluate classification performance by sector.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sectors: Sector labels for each sample

    Returns:
        DataFrame with per-sector metrics
    """
    sector_results = []

    for sector in sectors.unique():
        mask = sectors == sector
        if mask.sum() > 0:
            y_true_sector = y_true[mask]
            y_pred_sector = y_pred[mask]

            accuracy = accuracy_score(y_true_sector, y_pred_sector)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true_sector, y_pred_sector, average="macro", zero_division=0
            )

            sector_results.append(
                {
                    "Sector": sector,
                    "Samples": mask.sum(),
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            )

    results_df = pd.DataFrame(sector_results).sort_values("F1-Score", ascending=False)

    logger.info(f"Evaluated classification performance across {len(results_df)} sectors")

    return results_df
