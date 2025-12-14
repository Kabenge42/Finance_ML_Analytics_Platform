"""
Classification Evaluation Module - Phase 9.4.2

This module provides comprehensive evaluation capabilities for classification models:
- Performance metrics and reporting
- SHAP-based interpretation
- Cross-validation utilities
- Feature importance analysis
- Confusion matrices and learning curves
- Per-sector and per-class evaluation
- Calibration analysis

Extracted from classification.py and classification_enhanced.py as part of Phase 9.4.2 refactor.

Author: Finance ML Team
Date: 2025-11-09
Version: 9.4.2
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# Optional imports
try:
    import shap

    HAVE_SHAP = True
except ImportError:
    shap = None
    HAVE_SHAP = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAVE_MATPLOTLIB = True
except ImportError:
    plt = None
    sns = None
    HAVE_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def export_classification_probabilities(
    y_true: "np.ndarray",
    y_pred: "np.ndarray",
    y_proba: "np.ndarray",
    index: "Optional[pd.Index]" = None,
) -> "pd.DataFrame":
    """Export standardized classification probabilities for meta-features.

    This helper converts raw classifier outputs into a standardized
    probabilities DataFrame that can be used both for diagnostics and
    as input to the regression meta-feature pipeline (see
    :mod:`finance_ml.ml_workflow.regression.dataset`).

    The function is aligned with the **5-class event labeling system**
    introduced in Phase 9.4 / 9.9 with the following class semantics
    (per ``labels.py`` and ``code_guidelines.md``):

    - 0 → Strong Negative
    - 1 → Negative
    - 2 → Neutral
    - 3 → Positive
    - 4 → Strong Positive

    It expects ``y_proba`` with shape ``(n_samples, 5)`` ordered exactly
    as above and returns seven columns:

    - ``event_prob_strong_negative``  – probability of class 0
    - ``event_prob_negative``         – probability of class 1
    - ``event_prob_neutral``          – probability of class 2
    - ``event_prob_positive``         – probability of class 3
    - ``event_prob_strong_positive``  – probability of class 4
    - ``event_class_predicted``       – predicted class label (from ``y_pred``)
    - ``event_confidence``            – max probability across the five classes

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (used for shape validation; values 0–4).
    y_pred : array-like of shape (n_samples,)
        Predicted class labels from the classifier.
    y_proba : array-like of shape (n_samples, 5)
        Predicted class probabilities in 5-class order.
    index : pandas.Index, optional
        Optional index to apply to the returned DataFrame so that it
        aligns with the main stock universe.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 7 standardized columns and ``n_samples`` rows.

    Raises
    ------
    ValueError
        If ``y_proba`` does not have shape ``(n_samples, 5)`` or if
        ``y_true`` / ``y_pred`` lengths do not match ``y_proba``.
    """

    import numpy as np
    import pandas as pd

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    proba_arr = np.asarray(y_proba)

    if proba_arr.ndim != 2 or proba_arr.shape[1] != 5:
        raise ValueError(
            "export_classification_probabilities expects probabilities with "
            f"shape (n_samples, 5); got {proba_arr.shape}"
        )

    n_samples = proba_arr.shape[0]
    if y_true_arr.shape[0] != n_samples or y_pred_arr.shape[0] != n_samples:
        raise ValueError(
            "y_true, y_pred, and y_proba must have the same number of samples "
            f"(got y_true={y_true_arr.shape[0]}, y_pred={y_pred_arr.shape[0]}, "
            f"y_proba={n_samples})"
        )

    # Confidence is simply the max probability; we rely on the provided
    # y_pred for the predicted class to stay consistent with upstream
    # classifier outputs.
    event_confidence = proba_arr.max(axis=1)

    probs_df = pd.DataFrame(
        {
            "event_prob_strong_negative": proba_arr[:, 0],
            "event_prob_negative": proba_arr[:, 1],
            "event_prob_neutral": proba_arr[:, 2],
            "event_prob_positive": proba_arr[:, 3],
            "event_prob_strong_positive": proba_arr[:, 4],
            "event_class_predicted": y_pred_arr,
            "event_confidence": event_confidence,
        }
    )

    if index is not None:
        probs_df.index = index

    return probs_df


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
        class_names: Names for classes. If None, inferred based on number of classes.
            - 3 classes → ["Neutral", "Positive", "Negative"]
            - 5 classes → ["Strong Negative", "Negative", "Neutral", "Positive", "Strong Positive"]

    Returns:
        Dictionary with metrics, confusion matrix, and classification report
    """

    # Infer label set to ensure consistency across metrics and reports
    if y_proba is not None and hasattr(y_proba, "shape") and len(y_proba.shape) == 2:
        n_classes = int(y_proba.shape[1])
        labels = list(range(n_classes))
    else:
        # Use union of observed labels to maintain backward compatibility
        y_true_arr = np.asarray(y_true).ravel()
        y_pred_arr = np.asarray(y_pred).ravel()
        labels = sorted(list(set(np.unique(y_true_arr)).union(np.unique(y_pred_arr))))
        # Ensure labels are int-like and contiguous starting at 0 if possible
        try:
            labels = [int(x) for x in labels]
        except Exception:  # keep as-is if not castable
            pass
        n_classes = len(labels)

    # Determine class names
    default_3 = ["Neutral", "Positive", "Negative"]
    default_5 = [
        "Strong Negative",
        "Negative",
        "Neutral",
        "Positive",
        "Strong Positive",
    ]

    inferred_defaults = (
        default_5
        if n_classes == 5
        else default_3 if n_classes == 3 else [f"Class {i}" for i in range(n_classes)]
    )

    if class_names is None:
        class_names_use = inferred_defaults
    else:
        class_names_use = class_names

    # If provided class_names length doesn't match the label count, reconcile to avoid ValueError
    if len(class_names_use) != n_classes:
        # Prefer using probability-derived class count when available, otherwise fall back to labels observed
        logger.warning(
            "Mismatch between number of classes (%d) and class_names (%d). Adjusting labels/names to align.",
            n_classes,
            len(class_names_use),
        )
        # If y_proba dictates number of classes and class_names provided match that, align labels accordingly
        if (
            y_proba is not None
            and hasattr(y_proba, "shape")
            and y_proba.shape[1] == len(class_names_use)
        ):
            labels = list(range(len(class_names_use)))
            n_classes = len(labels)
        else:
            # Otherwise, override class_names with sensible defaults matching n_classes
            class_names_use = inferred_defaults

    # Compute metrics with explicit labels to ensure fixed-length outputs
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    # Confusion matrix with explicit labels for stable NxN shape
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Classification report with aligned labels and names
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names_use,
        output_dict=True,
        zero_division=0,
    )

    # ROC-AUC (if probabilities available)
    roc_auc = None
    if y_proba is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            logger.warning("Could not compute ROC-AUC (likely due to missing classes)")

    return {
        "accuracy": float(accuracy),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_auc": roc_auc,
    }


def compute_shap_values(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    max_samples: int = 100,
    max_evals: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute SHAP values for model interpretation.

    Args:
        model: Trained model
        X_train: Training data (for background)
        X_test: Test data (for SHAP values)
        max_samples: Maximum samples for SHAP computation
        max_evals: Maximum evaluations for SHAP explainer (auto-calculated if None)
                   Required for PermutationExplainer: must be >= 2 * n_features + 1

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

        # Calculate max_evals if not provided (required for PermutationExplainer)
        # PermutationExplainer needs: max_evals >= 2 * num_features + 1
        n_features = X_test_sample.shape[1]
        if max_evals is None:
            max_evals = max(500, 2 * n_features + 1)
            logger.info(f"Auto-calculated max_evals={max_evals} for {n_features} features")

        # Compute SHAP values with explicit max_evals
        shap_values = explainer(X_test_sample, max_evals=max_evals)

        logger.info(f"SHAP values computed for {len(X_test_sample)} samples")

        return {
            "explainer": explainer,
            "shap_values": shap_values,
            "X_test_sample": X_test_sample,
        }

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {}


def cross_validate_classifier(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: Union[int, Any] = 5,
    stratify_by: Optional[str] = None,
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, np.ndarray, Dict[str, np.ndarray]]]:
    """Perform stratified cross-validation for classifier.

    Args:
        model: Classifier to evaluate
        X: Feature data
        y: Labels
        cv: Number of folds (int) or a CV splitter object (e.g., GroupKFold, StratifiedKFold)
        stratify_by: Column to stratify by (e.g., 'sector')
        groups: Group labels for GroupKFold-based CV strategies

    Returns:
        Dictionary with cross-validation results

    Note:
        When passing a CV splitter object that requires groups (e.g., GroupKFold),
        you must also provide the groups parameter. If cv is a splitter object,
        it will be used directly; otherwise, a StratifiedKFold is created.
    """
    from sklearn.model_selection import cross_validate, BaseCrossValidator

    # Setup cross-validation strategy
    # Accept either an integer (create StratifiedKFold) or a CV splitter object
    if isinstance(cv, int):
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    elif hasattr(cv, "split"):
        # cv is already a CV splitter object (e.g., GroupKFold, StratifiedGroupKFold)
        cv_strategy = cv
    else:
        # Fallback: try to use as integer
        try:
            n_splits = int(cv)
            cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid cv parameter type: {type(cv)}. Using default 5-fold StratifiedKFold."
            )
            cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Remove stratify_by column if present (it's not a feature)
    X_for_cv = X.copy()
    if stratify_by and stratify_by in X.columns:
        X_for_cv = X_for_cv.drop(columns=[stratify_by])

    # Perform cross-validation
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    # Extract groups from stratify_by column if not explicitly provided
    cv_groups = groups
    if cv_groups is None and stratify_by and stratify_by in X.columns:
        cv_groups = X[stratify_by].values

    try:
        cv_results = cross_validate(
            model,
            X_for_cv,
            y,
            cv=cv_strategy,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            groups=cv_groups,
        )
    except TypeError as e:
        # Some CV strategies don't accept groups parameter
        if "groups" in str(e):
            logger.warning("CV strategy doesn't support groups parameter, retrying without groups")
            cv_results = cross_validate(
                model,
                X_for_cv,
                y,
                cv=cv_strategy,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
            )
        else:
            raise

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

    # Determine number of folds for logging
    n_folds = cv if isinstance(cv, int) else getattr(cv_strategy, "n_splits", "N/A")
    logger.info(
        f"Cross-validation ({n_folds} folds): Accuracy={results['test_accuracy']:.4f}±{results['test_accuracy_std']:.4f}, "
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
    if not HAVE_MATPLOTLIB:
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
            # Ensure confusion matrix shape matches class_names length when provided
            if class_names is not None:
                labels = list(range(len(class_names)))
            else:
                # Infer labels from observed data
                labels = sorted(
                    list(set(np.unique(results["y_test"])).union(np.unique(results["y_pred"])))
                )
                # Coerce to int when possible
                try:
                    labels = [int(x) for x in labels]
                except Exception:
                    pass

            cm = confusion_matrix(results["y_test"], results["y_pred"], labels=labels)
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


def plot_learning_curves(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: Optional[np.ndarray] = None,
    scoring: str = "accuracy",
) -> Dict[str, Any]:
    """Generate learning curves to diagnose bias/variance.

    Args:
        model: Classifier to evaluate
        X: Feature data
        y: Labels
        cv: Number of cross-validation folds
        train_sizes: Array of training set sizes (default: np.linspace(0.1, 1.0, 10))
        scoring: Scoring metric ('accuracy', 'f1_macro', etc.)

    Returns:
        Dictionary with train_sizes, train_scores, and test_scores
    """
    from sklearn.model_selection import learning_curve

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    try:
        logger.info(f"Computing learning curves with {cv}-fold CV...")
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )

        # Calculate mean and std
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        logger.info(
            f"Learning curves computed. Final train score: {train_scores_mean[-1]:.4f}, "
            f"Final test score: {test_scores_mean[-1]:.4f}"
        )

        # Optionally plot if matplotlib available
        if HAVE_MATPLOTLIB:
            try:
                plt.figure(figsize=(10, 6))
                plt.title("Learning Curves")
                plt.xlabel("Training Examples")
                plt.ylabel("Score")
                plt.grid()

                plt.fill_between(
                    train_sizes_abs,
                    train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std,
                    alpha=0.1,
                    color="r",
                )
                plt.fill_between(
                    train_sizes_abs,
                    test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std,
                    alpha=0.1,
                    color="g",
                )
                plt.plot(
                    train_sizes_abs,
                    train_scores_mean,
                    "o-",
                    color="r",
                    label="Training score",
                )
                plt.plot(
                    train_sizes_abs,
                    test_scores_mean,
                    "o-",
                    color="g",
                    label="Cross-validation score",
                )
                plt.legend(loc="best")
                plt.tight_layout()
            except Exception as e:
                logger.info(f"Could not plot learning curves: {e}")

        return {
            "train_sizes": train_sizes_abs,
            "train_scores": train_scores,
            "test_scores": test_scores,
            "train_scores_mean": train_scores_mean,
            "train_scores_std": train_scores_std,
            "test_scores_mean": test_scores_mean,
            "test_scores_std": test_scores_std,
        }

    except Exception as e:
        logger.error(f"Failed to compute learning curves: {e}")
        return {
            "train_sizes": np.array([]),
            "train_scores": np.array([]),
            "test_scores": np.array([]),
        }


def analyze_per_class_feature_importance(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Analyze feature importance for each class separately.

    This uses a one-vs-rest approach to train class-specific models
    and extract feature importance for each class.

    Args:
        model: Base classifier (should have feature_importances_ attribute)
        X: Feature data
        y: Labels
        feature_names: List of feature names
        top_n: Number of top features to return per class

    Returns:
        DataFrame with per-class feature importance
    """
    from sklearn.multiclass import OneVsRestClassifier

    if feature_names is None:
        feature_names = (
            list(X.columns)
            if hasattr(X, "columns")
            else [f"feature_{i}" for i in range(X.shape[1])]
        )

    try:
        logger.info("Analyzing per-class feature importance...")

        # Use OneVsRestClassifier to get class-specific models
        if not hasattr(model, "estimators_"):
            # If model is not already a multi-class ensemble, wrap it
            ovr_model = OneVsRestClassifier(model)
            ovr_model.fit(X, y)
        else:
            ovr_model = model
            if not hasattr(ovr_model, "estimators_"):
                ovr_model.fit(X, y)

        unique_classes = np.unique(y)
        importance_data = []

        # Extract importance from each class-specific estimator
        if hasattr(ovr_model, "estimators_"):
            for class_idx, estimator in enumerate(ovr_model.estimators_):
                if hasattr(estimator, "feature_importances_"):
                    importances = estimator.feature_importances_
                    # Get top N features for this class
                    top_indices = np.argsort(importances)[-top_n:][::-1]
                    for idx in top_indices:
                        importance_data.append(
                            {
                                "Class": (
                                    unique_classes[class_idx]
                                    if class_idx < len(unique_classes)
                                    else class_idx
                                ),
                                "Feature": feature_names[idx],
                                "Importance": importances[idx],
                            }
                        )
        elif hasattr(model, "feature_importances_"):
            # If model has global feature importance, use that for all classes
            importances = model.feature_importances_
            for class_label in unique_classes:
                top_indices = np.argsort(importances)[-top_n:][::-1]
                for idx in top_indices:
                    importance_data.append(
                        {
                            "Class": class_label,
                            "Feature": feature_names[idx],
                            "Importance": importances[idx],
                        }
                    )
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame(columns=["Class", "Feature", "Importance"])

        importance_df = pd.DataFrame(importance_data)
        logger.info(f"Per-class feature importance computed for {len(unique_classes)} classes")

        return importance_df

    except Exception as e:
        logger.error(f"Failed to analyze per-class feature importance: {e}")
        return pd.DataFrame(columns=["Class", "Feature", "Importance"])


def analyze_calibration(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> Dict[str, Any]:
    """
    Analyze prediction calibration quality.

    Calibration measures how well predicted probabilities match actual outcomes.
    Well-calibrated models have predicted probabilities that match true frequencies.

    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_proba : np.ndarray
        Predicted probabilities (n_samples, n_classes)
    n_bins : int, default=10
        Number of bins for calibration curve

    Returns:
    --------
    dict : Calibration metrics including Brier score, log loss, and binned statistics

    Example:
    --------
    >>> from finance_ml.ml_workflow.classification.evaluation import analyze_calibration
    >>> calibration = analyze_calibration(y_test, y_proba_test)
    >>> print(f"Brier Score: {calibration['brier_score']:.4f}")
    >>> print(f"Log Loss: {calibration['log_loss']:.4f}")
    """
    from sklearn.calibration import calibration_curve

    results = {}

    # Overall metrics
    try:
        # For multi-class, compute one-vs-rest Brier scores
        brier_scores = []
        for class_idx in range(y_proba.shape[1]):
            y_binary = (y_true == class_idx).astype(int)
            brier = brier_score_loss(y_binary, y_proba[:, class_idx])
            brier_scores.append(brier)
        results["brier_score"] = float(np.mean(brier_scores))
        results["brier_score_per_class"] = brier_scores

        results["log_loss"] = float(log_loss(y_true, y_proba))
    except Exception as e:
        logger.warning(f"Failed to compute calibration metrics: {e}")
        results["brier_score"] = None
        results["log_loss"] = None

    # Calibration curves per class
    calibration_curves = {}
    for class_idx in range(y_proba.shape[1]):
        try:
            y_binary = (y_true == class_idx).astype(int)
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_proba[:, class_idx], n_bins=n_bins
            )
            calibration_curves[f"class_{class_idx}"] = {
                "fraction_of_positives": fraction_of_positives.tolist(),
                "mean_predicted_value": mean_predicted_value.tolist(),
            }
        except Exception as e:
            logger.warning(f"Failed to compute calibration curve for class {class_idx}: {e}")

    results["calibration_curves"] = calibration_curves

    return results


def analyze_feature_importance_by_groups(
    importance_dict: Dict[str, float],
    feature_names: Optional[List[str]] = None,
    top_n_per_group: int = 10,
) -> Dict[str, Any]:
    """
    Analyze feature importance grouped by Phase 9.3 feature categories.

    Categorizes features into Phase 9.3 groups and aggregates importance:
    - analyst_quality: analyst coverage, consensus, ratings, price targets
    - accounting_quality: exceptional items, goodwill, intangibles
    - employee_productivity: revenue/profit/assets per employee
    - basic: all other features

    Args:
        importance_dict: Dictionary mapping feature names to importance values
        feature_names: Optional list of feature names (if not in importance_dict)
        top_n_per_group: Number of top features to return per group

    Returns:
        Dictionary containing:
        - group_totals: Total importance per group
        - group_percentages: Percentage of total importance per group
        - top_features_per_group: Top N features for each group
        - feature_groups: Mapping of each feature to its group

    Example:
        >>> from finance_ml.ml_workflow.classification.evaluation import analyze_feature_importance_by_groups
        >>> importance = {'analyst_coverage': 0.15, 'revenue_per_employee': 0.12, 'p_e_ratio': 0.20}
        >>> result = analyze_feature_importance_by_groups(importance)
        >>> print(result['group_totals'])
        {'analyst_quality': 0.15, 'employee_productivity': 0.12, 'basic': 0.20}
    """
    # Phase 9.3 feature group keywords
    analyst_keywords = [
        "analyst_coverage",
        "analyst_consensus",
        "price_target_spread",
        "rating_buy_ratio",
        "rating_sell_ratio",
        "analyst_rating",
        "price_target_ytd",
        "price_target_median",
    ]

    accounting_keywords = [
        "exceptional_items_intensity",
        "goodwill_intensity",
        "intangibles_ratio",
        "accounting_quality_score",
        "exceptional_items",
        "goodwill",
        "intangibles",
    ]

    employee_keywords = [
        "revenue_per_employee",
        "profit_per_employee",
        "assets_per_employee",
        "employee_growth_rate",
        "employee_growth",
        "employees",
    ]

    # Categorize features
    feature_groups = {}
    group_importance = {
        "analyst_quality": [],
        "accounting_quality": [],
        "employee_productivity": [],
        "basic": [],
    }

    for feature, importance in importance_dict.items():
        feature_lower = feature.lower()

        # Check which group the feature belongs to
        if any(keyword in feature_lower for keyword in analyst_keywords):
            group = "analyst_quality"
        elif any(keyword in feature_lower for keyword in accounting_keywords):
            group = "accounting_quality"
        elif any(keyword in feature_lower for keyword in employee_keywords):
            group = "employee_productivity"
        else:
            group = "basic"

        feature_groups[feature] = group
        group_importance[group].append((feature, importance))

    # Calculate group totals
    group_totals = {
        group: sum(imp for _, imp in features) for group, features in group_importance.items()
    }

    total_importance = sum(group_totals.values())

    # Calculate percentages
    group_percentages = {
        group: (total / total_importance * 100) if total_importance > 0 else 0.0
        for group, total in group_totals.items()
    }

    # Get top N features per group
    top_features_per_group = {}
    for group, features in group_importance.items():
        if features:
            sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
            top_features_per_group[group] = [
                {"feature": feat, "importance": imp}
                for feat, imp in sorted_features[:top_n_per_group]
            ]
        else:
            top_features_per_group[group] = []

    logger.info(
        f"Feature importance by groups: "
        f"analyst={group_percentages['analyst_quality']:.1f}%, "
        f"accounting={group_percentages['accounting_quality']:.1f}%, "
        f"employee={group_percentages['employee_productivity']:.1f}%, "
        f"basic={group_percentages['basic']:.1f}%"
    )

    return {
        "group_totals": group_totals,
        "group_percentages": group_percentages,
        "top_features_per_group": top_features_per_group,
        "feature_groups": feature_groups,
    }


def analyze_feature_importance_by_sector(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    sector_col: str = "sector",
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Analyze feature importance separately for each sector.

    Trains sector-specific models and extracts feature importance for each sector,
    enabling identification of sector-specific predictive features.

    Args:
        model: Base classifier (should have feature_importances_ or coef_ attribute)
        X: Feature data (must include sector column)
        y: Labels
        sector_col: Name of sector column
        top_n: Number of top features to return per sector

    Returns:
        DataFrame with columns: Sector, Feature, Importance, Rank

    Example:
        >>> from finance_ml.ml_workflow.classification.evaluation import analyze_feature_importance_by_sector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier(random_state=42)
        >>> result = analyze_feature_importance_by_sector(model, X_train, y_train, sector_col='sector')
        >>> print(result.groupby('Sector')['Feature'].count())
    """
    if sector_col not in X.columns:
        logger.warning(f"Sector column '{sector_col}' not found in data")
        return pd.DataFrame(columns=["Sector", "Feature", "Importance", "Rank"])

    sectors = X[sector_col].unique()
    X_features = X.drop(columns=[sector_col])
    feature_names = list(X_features.columns)

    sector_results = []

    for sector in sectors:
        try:
            # Filter data for this sector
            sector_mask = X[sector_col] == sector
            X_sector = X_features[sector_mask]
            y_sector = y[sector_mask]

            if len(y_sector) < 10:  # Skip sectors with too few samples
                logger.warning(f"Skipping sector {sector} - only {len(y_sector)} samples")
                continue

            # Clone and train model for this sector
            from sklearn.base import clone

            sector_model = clone(model)
            sector_model.fit(X_sector, y_sector)

            # Extract feature importance
            if hasattr(sector_model, "feature_importances_"):
                importances = sector_model.feature_importances_
            elif hasattr(sector_model, "coef_"):
                # For linear models, use absolute coefficient values
                importances = np.abs(sector_model.coef_).mean(axis=0)
            else:
                logger.warning(f"Model for sector {sector} has no feature importance")
                continue

            # Get top N features for this sector
            top_indices = np.argsort(importances)[-top_n:][::-1]

            for rank, idx in enumerate(top_indices, start=1):
                sector_results.append(
                    {
                        "Sector": sector,
                        "Feature": feature_names[idx],
                        "Importance": importances[idx],
                        "Rank": rank,
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to analyze sector {sector}: {e}")
            continue

    results_df = pd.DataFrame(sector_results)

    if not results_df.empty:
        logger.info(f"Sector-specific feature importance computed for {len(sectors)} sectors")
    else:
        logger.warning("No sector-specific feature importance could be computed")

    return results_df


def analyze_shap_by_feature_groups(
    shap_values: Any,
    feature_names: List[str],
    top_n_per_group: int = 10,
) -> Dict[str, Any]:
    """
    Analyze SHAP values grouped by Phase 9.3 feature categories.

    Groups SHAP importance values by Phase 9.3 feature categories to understand
    which feature groups contribute most to model predictions.

    Args:
        shap_values: SHAP values object (from shap.Explainer)
        feature_names: List of feature names
        top_n_per_group: Number of top features to return per group

    Returns:
        Dictionary containing:
        - group_mean_abs_shap: Mean absolute SHAP value per group
        - group_percentages: Percentage of total SHAP importance per group
        - top_features_per_group: Top N features by mean |SHAP| for each group
        - feature_groups: Mapping of each feature to its group

    Example:
        >>> from finance_ml.ml_workflow.classification.evaluation import analyze_shap_by_feature_groups
        >>> # After computing SHAP values
        >>> result = analyze_shap_by_feature_groups(shap_values, feature_names)
        >>> print(result['group_percentages'])
    """
    if not HAVE_SHAP:
        logger.warning("SHAP not available, cannot analyze by feature groups")
        return {}

    # Phase 9.3 feature group keywords
    analyst_keywords = [
        "analyst_coverage",
        "analyst_consensus",
        "price_target_spread",
        "rating_buy_ratio",
        "rating_sell_ratio",
        "analyst_rating",
    ]

    accounting_keywords = [
        "exceptional_items_intensity",
        "goodwill_intensity",
        "intangibles_ratio",
        "accounting_quality_score",
    ]

    employee_keywords = [
        "revenue_per_employee",
        "profit_per_employee",
        "assets_per_employee",
        "employee_growth_rate",
    ]

    try:
        # Extract SHAP values array
        if hasattr(shap_values, "values"):
            shap_array = shap_values.values
        else:
            shap_array = shap_values

        # Handle multi-class SHAP values (take mean across classes)
        if shap_array.ndim == 3:
            shap_array = np.abs(shap_array).mean(axis=2)
        else:
            shap_array = np.abs(shap_array)

        # Calculate mean absolute SHAP per feature
        mean_abs_shap = shap_array.mean(axis=0)

        # Categorize features
        feature_groups = {}
        group_shap = {
            "analyst_quality": [],
            "accounting_quality": [],
            "employee_productivity": [],
            "basic": [],
        }

        for idx, feature in enumerate(feature_names):
            feature_lower = feature.lower()
            shap_val = mean_abs_shap[idx] if idx < len(mean_abs_shap) else 0.0

            # Determine group
            if any(keyword in feature_lower for keyword in analyst_keywords):
                group = "analyst_quality"
            elif any(keyword in feature_lower for keyword in accounting_keywords):
                group = "accounting_quality"
            elif any(keyword in feature_lower for keyword in employee_keywords):
                group = "employee_productivity"
            else:
                group = "basic"

            feature_groups[feature] = group
            group_shap[group].append((feature, shap_val))

        # Calculate group totals
        group_mean_abs_shap = {
            group: np.mean([val for _, val in features]) if features else 0.0
            for group, features in group_shap.items()
        }

        total_shap = sum(group_mean_abs_shap.values())

        # Calculate percentages
        group_percentages = {
            group: (val / total_shap * 100) if total_shap > 0 else 0.0
            for group, val in group_mean_abs_shap.items()
        }

        # Get top N features per group
        top_features_per_group = {}
        for group, features in group_shap.items():
            if features:
                sorted_features = sorted(features, key=lambda x: x[1], reverse=True)
                top_features_per_group[group] = [
                    {"feature": feat, "mean_abs_shap": val}
                    for feat, val in sorted_features[:top_n_per_group]
                ]
            else:
                top_features_per_group[group] = []

        logger.info(
            f"SHAP importance by groups: "
            f"analyst={group_percentages['analyst_quality']:.1f}%, "
            f"accounting={group_percentages['accounting_quality']:.1f}%, "
            f"employee={group_percentages['employee_productivity']:.1f}%, "
            f"basic={group_percentages['basic']:.1f}%"
        )

        return {
            "group_mean_abs_shap": group_mean_abs_shap,
            "group_percentages": group_percentages,
            "top_features_per_group": top_features_per_group,
            "feature_groups": feature_groups,
        }

    except Exception as e:
        logger.error(f"Failed to analyze SHAP by feature groups: {e}")
        return {}


# Export all evaluation functions
__all__ = [
    "evaluate_classification",
    "compute_shap_values",
    "cross_validate_classifier",
    "compare_feature_importance",
    "plot_confusion_matrices",
    "evaluate_classification_by_sector",
    "plot_learning_curves",
    "analyze_per_class_feature_importance",
    "analyze_calibration",
    # Phase 9.4.3 - Enhanced feature importance analysis
    "analyze_feature_importance_by_groups",
    "analyze_feature_importance_by_sector",
    "analyze_shap_by_feature_groups",
]
