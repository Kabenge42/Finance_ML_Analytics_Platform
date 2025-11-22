"""
Safety rails reporting and visualization (Phase 9.5).

Functions for generating notebook-friendly safety rails diagnostics:
- Winsorization effects summary (JSON + HTML)
- Constraint violation tracking (JSON + HTML)
- Interactive sensitivity dashboard (HTML)
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def summarize_winsorization_effects(
    features_raw: pd.DataFrame,
    features_winsorized: pd.DataFrame,
    output_dir: Union[str, Path],
    feature_cols: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Summarize winsorization effects on features.

    Creates:
    - clipping_effect_summary.json (per-feature statistics)
    - pre_post_winsorization_distributions.html (visualization)

    Parameters
    ----------
    features_raw : pd.DataFrame
        Raw features before winsorization
    features_winsorized : pd.DataFrame
        Features after winsorization
    output_dir : Union[str, Path]
        Directory to save artifacts
    feature_cols : Optional[list]
        List of feature columns to analyze. If None, auto-detect numeric columns.

    Returns
    -------
    Dict[str, Any]
        Summary statistics per feature
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect numeric feature columns
    if feature_cols is None:
        feature_cols = features_raw.select_dtypes(include=[np.number]).columns.tolist()
        # Remove non-feature columns
        exclude_cols = ["ticker", "isin", "id", "index"]
        feature_cols = [col for col in feature_cols if col.lower() not in exclude_cols]

    logger.info(f"Analyzing winsorization effects for {len(feature_cols)} features")

    summary = {}

    for col in feature_cols:
        if col not in features_raw.columns or col not in features_winsorized.columns:
            continue

        raw_values = features_raw[col].dropna()
        wins_values = features_winsorized[col].dropna()

        if len(raw_values) == 0 or len(wins_values) == 0:
            continue

        # Compute statistics
        raw_mean = float(raw_values.mean())
        wins_mean = float(wins_values.mean())
        raw_std = float(raw_values.std())
        wins_std = float(wins_values.std())

        # Detect changed values
        if len(raw_values) == len(wins_values):
            values_changed = (raw_values.values != wins_values.values).sum()
            pct_changed = float(values_changed / len(raw_values) * 100)
        else:
            pct_changed = 0.0

        summary[col] = {
            "raw_mean": raw_mean,
            "winsorized_mean": wins_mean,
            "raw_std": raw_std,
            "winsorized_std": wins_std,
            "raw_min": float(raw_values.min()),
            "raw_max": float(raw_values.max()),
            "winsorized_min": float(wins_values.min()),
            "winsorized_max": float(wins_values.max()),
            "pct_values_changed": pct_changed,
        }

    # Save JSON summary
    json_path = output_dir / "clipping_effect_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved winsorization summary to {json_path}")

    # Create HTML visualization
    html_path = output_dir / "pre_post_winsorization_distributions.html"

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create subplots for each feature
        n_features = min(len(feature_cols), 6)  # Limit to 6 features for readability
        rows = (n_features + 1) // 2

        fig = make_subplots(
            rows=rows, cols=2, subplot_titles=[f"{col[:20]}" for col in feature_cols[:n_features]]
        )

        for idx, col in enumerate(feature_cols[:n_features]):
            if col not in features_raw.columns or col not in features_winsorized.columns:
                continue

            row = (idx // 2) + 1
            col_num = (idx % 2) + 1

            # Raw distribution
            fig.add_trace(
                go.Histogram(
                    x=features_raw[col].dropna(),
                    name=f"{col} (raw)",
                    opacity=0.5,
                    marker_color="blue",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col_num,
            )

            # Winsorized distribution
            fig.add_trace(
                go.Histogram(
                    x=features_winsorized[col].dropna(),
                    name=f"{col} (winsorized)",
                    opacity=0.5,
                    marker_color="red",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col_num,
            )

        fig.update_layout(
            title_text="Pre- vs Post-Winsorization Distributions",
            height=300 * rows,
            showlegend=True,
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved winsorization HTML to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Winsorization Effects</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return summary


def track_constraint_violations(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
    prediction_col: str = "y_pred_raw",
) -> Dict[str, Any]:
    """
    Track constraint violations (e.g., negative predictions).

    Creates:
    - non_negative_violations.json (violation counts and details)
    - violation_heatmap_by_feature_sector.html (visualization)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with sector and prediction columns
    output_dir : Union[str, Path]
        Directory to save artifacts
    prediction_col : str
        Name of prediction column to check for violations

    Returns
    -------
    Dict[str, Any]
        Violation summary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Tracking constraint violations in {len(predictions_df)} predictions")

    # Detect negative predictions
    if prediction_col not in predictions_df.columns:
        logger.warning(f"Prediction column '{prediction_col}' not found")
        violations_df = pd.DataFrame()
        total_violations = 0
    else:
        violations_mask = predictions_df[prediction_col] < 0
        violations_df = predictions_df[violations_mask].copy()
        total_violations = int(violations_mask.sum())

    # Group violations by sector
    violations_by_sector = {}
    if "sector" in predictions_df.columns and total_violations > 0:
        for sector in violations_df["sector"].unique():
            sector_violations = violations_df[violations_df["sector"] == sector]
            violations_by_sector[sector] = {
                "count": int(len(sector_violations)),
                "min_value": float(sector_violations[prediction_col].min()),
                "mean_value": float(sector_violations[prediction_col].mean()),
            }

    violations_summary = {
        "total_violations": total_violations,
        "total_predictions": int(len(predictions_df)),
        "violation_rate": (
            float(total_violations / len(predictions_df)) if len(predictions_df) > 0 else 0.0
        ),
        "violations_by_sector": violations_by_sector,
    }

    # Save JSON
    json_path = output_dir / "non_negative_violations.json"
    with open(json_path, "w") as f:
        json.dump(violations_summary, f, indent=2)
    logger.info(f"Saved violations summary to {json_path}")

    # Create heatmap HTML
    html_path = output_dir / "violation_heatmap_by_feature_sector.html"

    try:
        import plotly.graph_objects as go

        if "sector" in predictions_df.columns and total_violations > 0:
            # Create violation counts by sector
            sector_counts = []
            sectors = sorted(predictions_df["sector"].unique())

            for sector in sectors:
                sector_df = predictions_df[predictions_df["sector"] == sector]
                sector_violations = (sector_df[prediction_col] < 0).sum()
                sector_counts.append(sector_violations)

            fig = go.Figure(data=[go.Bar(x=sectors, y=sector_counts, marker_color="red")])

            fig.update_layout(
                title="Constraint Violations by Sector",
                xaxis_title="Sector",
                yaxis_title="Number of Violations",
                height=500,
            )

            fig.write_html(str(html_path))
            logger.info(f"Saved violation heatmap to {html_path}")
        else:
            with open(html_path, "w") as f:
                f.write("<html><body><h1>No Violations Found</h1></body></html>")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Constraint Violations</h1>")
            f.write(f"<p>Total violations: {total_violations}</p>")
            f.write("</body></html>")

    return violations_summary


def safety_rails_sensitivity_app(
    data_df: pd.DataFrame,
    output_dir: Union[str, Path],
    default_lower_pct: float = 0.05,
    default_upper_pct: float = 0.95,
) -> Path:
    """
    Create interactive sensitivity dashboard for safety rails thresholds.

    Allows exploration of how different winsorization thresholds affect
    feature distributions and downstream metrics.

    Parameters
    ----------
    data_df : pd.DataFrame
        Data with features and optional target
    output_dir : Union[str, Path]
        Directory to save HTML dashboard
    default_lower_pct : float
        Default lower percentile for winsorization
    default_upper_pct : float
        Default upper percentile for winsorization

    Returns
    -------
    Path
        Path to safety_rails_sensitivity_dashboard.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "safety_rails_sensitivity_dashboard.html"

    logger.info(f"Creating sensitivity dashboard for {len(data_df)} samples")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Get numeric features
        numeric_cols = data_df.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out non-feature columns
        exclude_cols = ["ticker", "isin", "id", "index"]
        feature_cols = [col for col in numeric_cols if col.lower() not in exclude_cols]

        if len(feature_cols) == 0:
            logger.warning("No numeric features found for sensitivity analysis")
            with open(html_path, "w") as f:
                f.write("<html><body><h1>Sensitivity Dashboard</h1>")
                f.write("<p>No numeric features found</p></body></html>")
            return html_path

        # Use first feature for demo (in real app, this would be interactive)
        feature_col = feature_cols[0]
        feature_values = data_df[feature_col].dropna()

        # Create figure with multiple percentile scenarios
        percentile_scenarios = [
            (0.01, 0.99, "Mild (1-99%)"),
            (0.05, 0.95, "Standard (5-95%)"),
            (0.10, 0.90, "Aggressive (10-90%)"),
        ]

        fig = make_subplots(
            rows=len(percentile_scenarios),
            cols=1,
            subplot_titles=[f"Winsorization: {label}" for _, _, label in percentile_scenarios],
        )

        for idx, (lower, upper, label) in enumerate(percentile_scenarios):
            lower_bound = feature_values.quantile(lower)
            upper_bound = feature_values.quantile(upper)

            winsorized = feature_values.clip(lower=lower_bound, upper=upper_bound)

            fig.add_trace(go.Histogram(x=winsorized, name=label, opacity=0.7), row=idx + 1, col=1)

        fig.update_layout(
            title_text=f"Safety Rails Sensitivity: {feature_col}",
            height=300 * len(percentile_scenarios),
            showlegend=True,
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved sensitivity dashboard to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Safety Rails Sensitivity Dashboard</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write(f"<p>Default thresholds: {default_lower_pct} - {default_upper_pct}</p>")
            f.write("</body></html>")

    return html_path
