"""
Explainability utilities (SHAP, LIME) extracted from analytics/eval.py.

This module provides functions for model interpretation using SHAP and LIME.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Any, Union


def compute_shap_values(model, X, model_type="auto", n_samples=100):
    """
    Compute SHAP values for a given model and dataset.

    Args:
        model: Trained model (scikit-learn compatible)
        X: Feature data (DataFrame or array)
        model_type: Type of explainer to use
            - "auto": Automatically detect best explainer
            - "tree": Use TreeExplainer (for tree-based regression)
            - "kernel": Use KernelExplainer (model-agnostic, slower)
            - "linear": Use LinearExplainer (for linear regression only)
        n_samples: Number of background samples for KernelExplainer

    Returns:
        dict with keys:
            - "shap_values": SHAP values array
            - "expected_value": Base value
            - "feature_names": List of feature names
    """
    import shap

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Fix Float64 dtype incompatibility with SHAP (Phase 9.5 audit fix)
    # SHAP uses numpy's isfinite which doesn't work with pandas nullable Float64 dtype
    # Convert all numeric columns to standard numpy float64
    feature_names = list(X.columns)
    X_converted = X.copy()
    for col in X_converted.columns:
        # Check for nullable Float64 or other extension dtypes
        if hasattr(X_converted[col].dtype, "numpy_dtype") or str(
            X_converted[col].dtype
        ) in [
            "Float64",
            "Int64",
        ]:
            X_converted[col] = X_converted[col].astype("float64")
        elif X_converted[col].dtype == "object":
            # Try to convert object columns to numeric
            X_converted[col] = pd.to_numeric(X_converted[col], errors="coerce").astype(
                "float64"
            )

    # Ensure all columns are float64 for SHAP compatibility
    try:
        X_converted = X_converted.astype(np.float64)
    except (ValueError, TypeError):
        # If conversion fails, handle NaN/inf values first
        X_converted = X_converted.replace([np.inf, -np.inf], np.nan)
        X_converted = X_converted.fillna(0.0)
        X_converted = X_converted.astype(np.float64)

    X = X_converted

    # Get base model if this is a stacking ensemble
    base_model = getattr(model, "final_estimator_", model)

    # Auto-detect model type if requested
    if model_type == "auto":
        model_type = _detect_model_type(base_model)

    # Compute SHAP values based on model type
    if model_type == "tree":
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X)
        except Exception as e:
            print(f"TreeExplainer failed: {e}. Falling back to KernelExplainer.")
            model_type = "kernel"

    if model_type == "linear":
        try:
            # For stacking regression, we need to check if final estimator is truly linear
            if hasattr(base_model, "coef_"):
                n_coef = len(base_model.coef_)
                n_features = X.shape[1]

                if n_coef != n_features:
                    print(
                        f"Warning: Coefficient count ({n_coef}) doesn't match feature count ({n_features}). Using KernelExplainer instead."
                    )
                    model_type = "kernel"
                else:
                    explainer = shap.LinearExplainer(base_model, X)
                    shap_values = explainer.shap_values(X)
            else:
                print(
                    "Model doesn't have coef_ attribute. Using KernelExplainer instead."
                )
                model_type = "kernel"
        except Exception as e:
            print(f"LinearExplainer failed: {e}. Falling back to KernelExplainer.")
            model_type = "kernel"

    if model_type == "kernel":
        # Use a subset for background distribution
        background = shap.sample(X, min(n_samples, len(X)))
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X)

    # Handle multi-output case (for multi-class classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]  # Use first class for visualization

    return {
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "feature_names": list(X.columns) if hasattr(X, "columns") else None,
    }


def _detect_model_type(model):
    """Detect appropriate SHAP explainer type for a model."""
    model_class = type(model).__name__

    # Tree-based regression
    tree_models = [
        "RandomForest",
        "GradientBoosting",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "ExtraTrees",
        "DecisionTree",
    ]
    if any(tree in model_class for tree in tree_models):
        return "tree"

    # Linear regression
    linear_models = ["Linear", "Ridge", "Lasso", "ElasticNet", "SGD"]
    if any(linear in model_class for linear in linear_models):
        return "linear"

    # Default to kernel for everything else
    return "kernel"


def create_shap_summary_plot(
    model, X, output_path=None, model_type="auto", n_samples=100
):
    """
    Create SHAP summary plot showing feature importance.

    Args:
        model: Trained model
        X: Feature data
        output_path: Path to save plot (optional)
        model_type: Type of explainer ("auto", "tree", "kernel", "linear")
        n_samples: Number of samples for KernelExplainer
    """
    import shap
    import matplotlib.pyplot as plt

    # Convert to DataFrame if needed
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values with automatic fallback
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]

    # Create summary plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"SHAP summary plot saved to {output_path}")

    plt.close()

    print("✓ SHAP analysis complete")


def create_shap_waterfall_plot(
    model, X, sample_idx=0, output_path=None, model_type="tree", n_samples=100
):
    """
    Create SHAP waterfall plot for individual prediction explanation.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        output_path: Path to save plot
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        None (saves plot to file)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]
    expected_value = result["expected_value"]

    # Create waterfall plot for specific sample
    if isinstance(shap_values, list):
        # Multi-output model, use first output
        shap_values_sample = shap_values[0][sample_idx]
    else:
        shap_values_sample = shap_values[sample_idx]

    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values_sample,
        base_values=expected_value,
        data=X.iloc[sample_idx].values,
        feature_names=list(X.columns),
    )

    shap.plots.waterfall(explanation, show=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def create_shap_dependence_plot(
    model, X, feature, output_path=None, model_type="tree", n_samples=100
):
    """
    Create SHAP dependence plot showing feature interactions.

    Args:
        model: Trained model
        X: Feature matrix
        feature: Feature name or index for dependence plot
        output_path: Path to save plot
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        None (saves plot to file)
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Compute SHAP values
    result = compute_shap_values(model, X, model_type=model_type, n_samples=n_samples)
    shap_values = result["shap_values"]

    # Create dependence plot
    shap.dependence_plot(feature, shap_values, X, show=False)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close()


def analyze_shap_by_sector(model, X, sectors, model_type="tree", n_samples=100):
    """
    Analyze SHAP values separately by sector.

    Args:
        model: Trained model
        X: Feature matrix
        sectors: Series with sector labels
        model_type: Type of model
        n_samples: Number of background samples

    Returns:
        dict: SHAP analysis for each sector
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP library is required. Install with: pip install shap")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    results = {}

    for sector in sectors.unique():
        sector_mask = sectors == sector
        X_sector = X[sector_mask]

        if len(X_sector) > 0:
            result = compute_shap_values(
                model, X_sector, model_type=model_type, n_samples=n_samples
            )

            # Compute mean absolute SHAP values for feature importance
            shap_values = result["shap_values"]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

            results[sector] = {
                "shap_values": shap_values,
                "expected_value": result["expected_value"],
                "feature_importance": dict(zip(result["feature_names"], mean_abs_shap)),
                "n_samples": len(X_sector),
            }

    return results


def explain_with_lime(model, X, sample_idx=0, output_path=None, n_features=10):
    """
    Generate LIME explanation for a single prediction.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        output_path: Optional path to save HTML explanation
        n_features: Number of features to show in explanation

    Returns:
        dict: LIME explanation with feature weights
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        raise ImportError("LIME library is required. Install with: pip install lime")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X.values, feature_names=list(X.columns), mode="regression", random_state=42
    )

    # Generate explanation for sample
    explanation = explainer.explain_instance(
        X.iloc[sample_idx].values, model.predict, num_features=n_features
    )

    # Extract feature weights
    feature_weights = dict(explanation.as_list())

    # Save HTML if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        explanation.save_to_file(str(output_path))

    return {
        "feature_weights": feature_weights,
        "prediction": float(model.predict(X.iloc[[sample_idx]])[0]),
        "intercept": explanation.intercept[0]
        if hasattr(explanation, "intercept")
        else 0.0,
        "score": explanation.score if hasattr(explanation, "score") else None,
    }


def compare_lime_shap_consistency(
    model, X, sample_idx=0, model_type="tree", n_features=10
):
    """
    Compare LIME and SHAP explanations for consistency.

    Args:
        model: Trained model
        X: Feature matrix
        sample_idx: Index of sample to explain
        model_type: Type of model for SHAP
        n_features: Number of features to compare

    Returns:
        dict: Comparison results with correlation metric
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    # Get LIME explanation
    lime_result = explain_with_lime(
        model, X, sample_idx=sample_idx, n_features=n_features
    )
    lime_weights = lime_result["feature_weights"]

    # Get SHAP explanation
    shap_result = compute_shap_values(
        model, X.iloc[[sample_idx]], model_type=model_type
    )
    shap_values = shap_result["shap_values"]

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_dict = dict(
        zip(X.columns, shap_values[0] if len(shap_values.shape) > 1 else shap_values)
    )

    # Compare feature importances
    common_features = set(lime_weights.keys()) & set(shap_dict.keys())

    if len(common_features) > 0:
        lime_vals = [lime_weights[f] for f in common_features]
        shap_vals = [shap_dict[f] for f in common_features]

        # Calculate correlation
        correlation = np.corrcoef(lime_vals, shap_vals)[0, 1]
    else:
        correlation = np.nan

    return {
        "lime_weights": lime_weights,
        "shap_values": shap_dict,
        "correlation": float(correlation) if not np.isnan(correlation) else None,
        "common_features": list(common_features),
    }
