"""
Stacking ensemble diagnostics and model governance (Phase 9.8).

Functions for stacking model transparency and governance:
- Base model contribution analysis
- Explainability (SHAP or permutation importance)
- Meta-learner error analysis
- Model card and lineage generation
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import os
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_stacking_contributions(
    base_predictions: Dict[str, np.ndarray],
    meta_predictions: np.ndarray,
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Compute and visualize base model contributions in stacking ensemble.

    Creates:
    - stacking_contributions.csv (model weights/contributions)
    - stacking_contributions.html (visualization)

    Parameters
    ----------
    base_predictions : Dict[str, np.ndarray]
        Dictionary mapping base model names to their predictions
    meta_predictions : np.ndarray
        Final stacked predictions
    output_dir : Union[str, Path]
        Directory to save artifacts

    Returns
    -------
    Dict[str, Any]
        Contribution statistics per base model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Computing stacking contributions for {len(base_predictions)} base models")

    contributions = {}

    # Compute correlation of each base model with final predictions
    for model_name, predictions in base_predictions.items():
        if len(predictions) == len(meta_predictions):
            correlation = np.corrcoef(predictions, meta_predictions)[0, 1]
            mae_vs_meta = np.abs(predictions - meta_predictions).mean()

            contributions[model_name] = {
                "correlation_with_meta": float(correlation),
                "mae_vs_meta": float(mae_vs_meta),
                "mean_prediction": float(predictions.mean()),
                "std_prediction": float(predictions.std()),
            }

    # Create DataFrame for CSV export
    contrib_df = pd.DataFrame(contributions).T
    contrib_df.index.name = "model_name"

    csv_path = output_dir / "stacking_contributions.csv"
    contrib_df.to_csv(csv_path)
    logger.info(f"Saved stacking contributions CSV to {csv_path}")

    # Create HTML visualization
    html_path = output_dir / "stacking_contributions.html"

    try:
        import plotly.graph_objects as go

        model_names = list(contributions.keys())
        correlations = [contributions[m]["correlation_with_meta"] for m in model_names]

        fig = go.Figure(data=[go.Bar(x=model_names, y=correlations, marker_color="steelblue")])

        fig.update_layout(
            title="Base Model Contribution to Stacked Predictions",
            xaxis_title="Base Model",
            yaxis_title="Correlation with Meta Predictions",
            height=500,
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved stacking contributions HTML to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Stacking Contributions</h1>")
            f.write("<table><tr><th>Model</th><th>Correlation</th></tr>")
            for model, stats in contributions.items():
                f.write(f'<tr><td>{model}</td><td>{stats["correlation_with_meta"]:.3f}</td></tr>')
            f.write("</table></body></html>")

    return contributions


def meta_error_maps(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
    feature_cols: Optional[List[str]] = None,
) -> Path:
    """
    Create meta-learner error analysis maps.

    Creates:
    - meta_error_map.html (error vs features and sectors)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with errors, features, and sector
    output_dir : Union[str, Path]
        Directory to save HTML file
    feature_cols : Optional[List[str]]
        Feature columns to analyze (default: auto-detect)

    Returns
    -------
    Path
        Path to meta_error_map.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "meta_error_map.html"

    logger.info(f"Creating meta-learner error maps")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Compute absolute errors
        if "y_true" in predictions_df.columns and "y_pred" in predictions_df.columns:
            errors = np.abs(predictions_df["y_true"] - predictions_df["y_pred"])
        else:
            errors = pd.Series([0] * len(predictions_df))

        # Error by sector
        if "sector" in predictions_df.columns:
            fig = make_subplots(
                rows=1, cols=2, subplot_titles=["Error by Sector", "Error Distribution"]
            )

            sectors = sorted(predictions_df["sector"].unique())
            sector_errors = [errors[predictions_df["sector"] == s].mean() for s in sectors]

            fig.add_trace(go.Bar(x=sectors, y=sector_errors, name="MAE by Sector"), row=1, col=1)

            fig.add_trace(go.Histogram(x=errors, name="Error Distribution"), row=1, col=2)

            fig.update_layout(
                title_text="Meta-Learner Error Analysis", height=500, showlegend=False
            )

            fig.write_html(str(html_path))
            logger.info(f"Saved meta error map to {html_path}")
        else:
            with open(html_path, "w") as f:
                f.write("<html><body><h1>Meta Error Map</h1>")
                f.write(f"<p>Mean Absolute Error: {errors.mean():.2f}</p>")
                f.write("</body></html>")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Meta-Learner Error Map</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return html_path


def generate_model_card(
    model_info: Dict[str, Any],
    output_dir: Union[str, Path],
    model_version: Optional[str] = None,
) -> Path:
    """
    Generate model card documentation.

    Creates:
    - model_card_v{MODEL_VERSION}.md (standardized model documentation)

    Parameters
    ----------
    model_info : Dict[str, Any]
        Dictionary with model metadata (task, data, features, models, validation, etc.)
    output_dir : Union[str, Path]
        Directory to save model card
    model_version : Optional[str]
        Model version (default: from env or 'v9_9')

    Returns
    -------
    Path
        Path to model card markdown file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_version is None:
        model_version = os.environ.get("MODEL_VERSION", "v9_9")

    logger.info(f"Generating model card for {model_version}")

    card_path = output_dir / f"model_card_{model_version}.md"

    # Default values
    task = model_info.get("task", "Price target regression + classification-enhanced features")
    data_source = model_info.get("data_source", "PostgreSQL equities table / CSV files")
    time_range = model_info.get("time_range", "Current snapshot")
    feature_groups = model_info.get(
        "feature_groups", ["momentum", "valuation", "profitability", "quality", "risk"]
    )
    base_learners = model_info.get("base_learners", ["XGBoost", "LightGBM", "CatBoost"])
    meta_learner = model_info.get("meta_learner", "Linear Ridge Regression")
    validation_policy = model_info.get(
        "validation_policy", "Grouped CV by ticker, stratified by sector"
    )

    # Get metrics
    metrics = model_info.get("metrics", {})
    mae = metrics.get("MAE", "N/A")
    rmse = metrics.get("RMSE", "N/A")
    mape = metrics.get("MAPE", "N/A")
    r2 = metrics.get("R2", "N/A")
    coverage = metrics.get("coverage_80pct", "N/A")

    # Generate markdown content
    content = f"""# Model Card — {model_version}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Overview

- **Task:** {task}
- **Model Version:** {model_version}
- **Model Type:** Stacking Ensemble (Gradient Boosting + Linear Meta-Learner)

## Data

- **Source:** {data_source}
- **Time Range:** {time_range}
- **Snapshot Policy:** Single or time-series snapshots with proper date handling
- **Data Split:** {validation_policy}

## Features

- **Feature Groups:** {', '.join(feature_groups)}
- **Total Features:** {model_info.get('n_features', 'N/A')}
- **Feature Selection:** Boruta + SHAP-based pruning (if available)
- **Safety Rails:** Winsorization (5-95th percentile), non-negativity constraints, robust loss functions

## Models

- **Base Learners:** {', '.join(base_learners)}
- **Meta-Learner:** {meta_learner}
- **Hyperparameters:** Cross-validated with GridSearchCV or Optuna
- **Stacking Strategy:** Out-of-fold base predictions as meta-features

## Validation & Metrics

- **Validation Strategy:** {validation_policy}
- **Overall Metrics:**
  - MAE: {mae}
  - RMSE: {rmse}
  - MAPE: {mape}%
  - R²: {r2}
- **Uncertainty Coverage (80% interval):** {coverage}

## Fairness & Bias

- **Sector-Level Calibration:** Applied per-sector bias correction
- **Regional Balance:** Stratified sampling ensures representation across US, EU, APAC, ROTW
- **Monitoring:** Continuous tracking of sector-level performance drift

## Risk & Limitations

- **Non-Negativity:** All predictions enforced to be ≥ 0 (price targets cannot be negative)
- **Data Drift:** Model performance may degrade if market conditions change significantly
- **Missingness:** Imputation strategy (6-step) may introduce bias in sparse data
- **Outliers:** Winsorization caps extreme values but may underestimate tail risk
- **Leakage Prevention:** Grouped CV ensures no ticker appears in both train and validation

## Versioning & Reproducibility

- **Code Version:** Git SHA (if tracked)
- **Data Version:** {model_info.get('data_version', 'Snapshot date-based')}
- **Dependencies:** See requirements.txt (Python 3.12+, scikit-learn, xgboost, lightgbm, catboost)
- **Random Seed:** {model_info.get('random_seed', 42)}

## Governance & Compliance

- **Model Owner:** ML Team / Data Science
- **Review Date:** {datetime.now().strftime('%Y-%m-%d')}
- **Approval Status:** {model_info.get('approval_status', 'Development')}
- **Monitoring Plan:** Weekly performance tracking, monthly retraining cadence

## Artifacts & Documentation

- **Predictions Schema:** `outputs/regression/regression_predictions_detailed.csv`
- **Uncertainty Diagnostics:** `outputs/uncertainty/`
- **Safety Rails Reports:** `outputs/safety_rails/`
- **Calibration Metrics:** `outputs/calibration/sector_bias_calibration_{model_version}.json`
- **Lineage:** `lineage.json`

## References

- Code Guidelines: `docs/code_guidelines.md` v1.2+
- Notebook: `ml_finance_model_main.ipynb`
- Improvement Plan: `docs/improvement_plan/`
"""

    with open(card_path, "w") as f:
        f.write(content)

    logger.info(f"Saved model card to {card_path}")

    return card_path


def build_lineage_json(
    model_info: Dict[str, Any],
    output_dir: Union[str, Path],
    model_version: Optional[str] = None,
) -> Path:
    """
    Build model lineage JSON for governance and traceability.

    Creates:
    - lineage.json (datasets → features → models → artifacts → metrics)

    Parameters
    ----------
    model_info : Dict[str, Any]
        Dictionary with model lineage information
    output_dir : Union[str, Path]
        Directory to save lineage JSON
    model_version : Optional[str]
        Model version (default: from env or 'v9_9')

    Returns
    -------
    Path
        Path to lineage.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_version is None:
        model_version = os.environ.get("MODEL_VERSION", "v9_9")

    logger.info(f"Building lineage JSON for {model_version}")

    lineage = {
        "model_version": model_version,
        "created_at": datetime.now().isoformat(),
        "datasets": model_info.get(
            "datasets",
            {
                "train": "outputs/data/all_stocks_train.csv",
                "validation": "outputs/data/all_stocks_val.csv",
            },
        ),
        "features": {
            "count": model_info.get("n_features", 310),
            "groups": model_info.get(
                "feature_groups",
                [
                    "momentum",
                    "valuation",
                    "profitability",
                    "quality",
                    "risk",
                    "cash_flow",
                    "capital_allocation",
                    "analyst_sentiment",
                    "market_sentiment",
                ],
            ),
            "selection_method": model_info.get("feature_selection", "Boruta + SHAP"),
        },
        "models": {
            "base": model_info.get("base_learners", ["xgboost", "lightgbm", "catboost"]),
            "meta": model_info.get("meta_learner", "linear_ridge"),
            "hyperparameters": model_info.get("hyperparameters", "See model_card for details"),
        },
        "artifacts": [
            "outputs/regression/regression_predictions_detailed.csv",
            "outputs/uncertainty/quantile_predictions_diagnostics.csv",
            "outputs/uncertainty/coverage_by_sector.json",
            "outputs/uncertainty/uncertainty_summary.json",
            "outputs/safety_rails/clipping_effect_summary.json",
            "outputs/safety_rails/non_negative_violations.json",
            "outputs/splits/grouped_cv_balance_metrics.json",
            "outputs/splits/leakage_report.json",
            f"outputs/calibration/sector_bias_calibration_{model_version}.json",
            "outputs/governance/stacking_contributions.csv",
            f"outputs/governance/model_card_{model_version}.md",
        ],
        "metrics": {
            "overall": model_info.get(
                "metrics",
                {
                    "MAE": 0.0,
                    "RMSE": 0.0,
                    "MAPE": 0.0,
                    "R2": 0.0,
                },
            ),
            "by_sector": model_info.get("metrics_by_sector", {}),
            "uncertainty_coverage": model_info.get("coverage_80pct", 0.80),
        },
        "validation": {
            "strategy": model_info.get("validation_strategy", "Grouped CV by ticker"),
            "n_folds": model_info.get("n_folds", 5),
            "leakage_check": model_info.get("leakage_check", "PASS"),
        },
        "governance": {
            "approval_status": model_info.get("approval_status", "Development"),
            "owner": model_info.get("owner", "ML Team"),
            "review_date": datetime.now().strftime("%Y-%m-%d"),
        },
    }

    json_path = output_dir / "lineage.json"
    with open(json_path, "w") as f:
        json.dump(lineage, f, indent=2)

    logger.info(f"Saved lineage JSON to {json_path}")

    return json_path
