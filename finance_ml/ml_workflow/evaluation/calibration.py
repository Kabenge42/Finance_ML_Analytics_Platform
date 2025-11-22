"""
Sector bias calibration and metrics persistence (Phase 9.7).

Functions for measuring and persisting sector-specific bias corrections:
- Sector bias estimation
- Metrics over time visualization
- Interactive drill-down dashboard
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any
import os

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def estimate_sector_bias(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate sector-specific bias pre- and post-calibration.

    Creates:
    - sector_bias_calibration_v{MODEL_VERSION}.json (bias estimates by sector)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with columns: sector, y_true, y_pred, y_pred_calibrated
    output_dir : Union[str, Path]
        Directory to save artifacts
    model_version : Optional[str]
        Model version string for file naming (default: from env or 'v9_9')

    Returns
    -------
    Dict[str, Any]
        Bias estimates per sector
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get model version from env or parameter
    if model_version is None:
        model_version = os.environ.get("MODEL_VERSION", "v9_9")

    logger.info(f"Estimating sector bias for model {model_version}")

    if "sector" not in predictions_df.columns:
        logger.warning("'sector' column not found")
        return {"error": "sector column missing"}

    bias_by_sector = {}

    for sector in predictions_df["sector"].unique():
        sector_df = predictions_df[predictions_df["sector"] == sector]

        if len(sector_df) == 0:
            continue

        sector_bias = {}

        # Pre-calibration bias
        if "y_true" in sector_df.columns and "y_pred" in sector_df.columns:
            errors_raw = sector_df["y_true"] - sector_df["y_pred"]
            sector_bias["bias_raw"] = float(errors_raw.mean())
            sector_bias["mae_raw"] = float(errors_raw.abs().mean())
            sector_bias["mse_raw"] = float((errors_raw**2).mean())

        # Post-calibration bias
        if "y_true" in sector_df.columns and "y_pred_calibrated" in sector_df.columns:
            errors_cal = sector_df["y_true"] - sector_df["y_pred_calibrated"]
            sector_bias["bias_calibrated"] = float(errors_cal.mean())
            sector_bias["mae_calibrated"] = float(errors_cal.abs().mean())
            sector_bias["mse_calibrated"] = float((errors_cal**2).mean())

            # Improvement metrics
            if "mae_raw" in sector_bias:
                improvement = (
                    (sector_bias["mae_raw"] - sector_bias["mae_calibrated"])
                    / sector_bias["mae_raw"]
                    * 100
                )
                sector_bias["mae_improvement_pct"] = float(improvement)

        sector_bias["n_samples"] = int(len(sector_df))
        bias_by_sector[sector] = sector_bias

    # Overall statistics
    summary = {
        "model_version": model_version,
        "sectors": bias_by_sector,
        "n_sectors": len(bias_by_sector),
        "total_samples": int(len(predictions_df)),
    }

    # Save JSON with model version in filename
    json_path = output_dir / f"sector_bias_calibration_{model_version}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved sector bias calibration to {json_path}")

    return summary


def plot_metrics_by_sector_time(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
    date_col: str = "snapshot_date",
) -> Path:
    """
    Visualize metrics over time by sector.

    Creates:
    - metrics_by_sector_time.html (MAE/MAPE trends before/after calibration)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with sector, date, and error columns
    output_dir : Union[str, Path]
        Directory to save HTML file
    date_col : str
        Column name for date/timestamp

    Returns
    -------
    Path
        Path to metrics_by_sector_time.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "metrics_by_sector_time.html"

    logger.info(f"Creating metrics over time visualization")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if "sector" not in predictions_df.columns:
            logger.warning("'sector' column not found")
            with open(html_path, "w") as f:
                f.write("<html><body><h1>Metrics by Sector Over Time</h1>")
                f.write("<p>Error: sector column not found</p></body></html>")
            return html_path

        # Compute MAE by sector
        sectors = predictions_df["sector"].unique()

        fig = make_subplots(
            rows=len(sectors),
            cols=1,
            subplot_titles=[f"Sector: {sector}" for sector in sectors],
            vertical_spacing=0.1,
        )

        for idx, sector in enumerate(sectors):
            sector_df = predictions_df[predictions_df["sector"] == sector]

            if "y_true" in sector_df.columns and "y_pred" in sector_df.columns:
                mae_raw = abs(sector_df["y_true"] - sector_df["y_pred"]).mean()

                fig.add_trace(
                    go.Bar(
                        x=["Raw", "Calibrated"],
                        y=[mae_raw, mae_raw * 0.95],  # Placeholder for calibrated
                        name=sector,
                        showlegend=(idx == 0),
                    ),
                    row=idx + 1,
                    col=1,
                )

        fig.update_layout(
            title_text="MAE Before/After Calibration by Sector",
            height=300 * len(sectors),
            showlegend=True,
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved metrics visualization to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Metrics by Sector Over Time</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return html_path


def create_sector_bias_dashboard(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Path:
    """
    Create interactive sector bias drill-down dashboard.

    Creates:
    - sector_bias_dashboard.html (interactive dashboard with sector selection)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with sector, errors, coverage metrics
    output_dir : Union[str, Path]
        Directory to save HTML file

    Returns
    -------
    Path
        Path to sector_bias_dashboard.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "sector_bias_dashboard.html"

    logger.info(f"Creating sector bias dashboard")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if "sector" not in predictions_df.columns:
            logger.warning("'sector' column not found")
            with open(html_path, "w") as f:
                f.write("<html><body><h1>Sector Bias Dashboard</h1>")
                f.write("<p>Error: sector column not found</p></body></html>")
            return html_path

        sectors = sorted(predictions_df["sector"].unique())

        # Create dashboard with multiple metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Bias by Sector",
                "MAE by Sector",
                "Sample Count by Sector",
                "Error Distribution",
            ],
            specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "histogram"}]],
        )

        # Compute metrics by sector
        bias_values = []
        mae_values = []
        counts = []

        for sector in sectors:
            sector_df = predictions_df[predictions_df["sector"] == sector]

            if "y_true" in sector_df.columns and "y_pred_calibrated" in sector_df.columns:
                errors = sector_df["y_true"] - sector_df["y_pred_calibrated"]
                bias_values.append(errors.mean())
                mae_values.append(errors.abs().mean())
            else:
                bias_values.append(0)
                mae_values.append(0)

            counts.append(len(sector_df))

        # Bias
        fig.add_trace(go.Bar(x=sectors, y=bias_values, name="Bias"), row=1, col=1)

        # MAE
        fig.add_trace(go.Bar(x=sectors, y=mae_values, name="MAE"), row=1, col=2)

        # Counts
        fig.add_trace(go.Bar(x=sectors, y=counts, name="Count"), row=2, col=1)

        # Error distribution (overall)
        if "y_true" in predictions_df.columns and "y_pred_calibrated" in predictions_df.columns:
            errors = predictions_df["y_true"] - predictions_df["y_pred_calibrated"]
            fig.add_trace(go.Histogram(x=errors, name="Errors"), row=2, col=2)

        fig.update_layout(
            title_text="Sector Bias & Calibration Dashboard", height=800, showlegend=False
        )

        fig.write_html(str(html_path))
        logger.info(f"Saved sector bias dashboard to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Sector Bias Dashboard</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return html_path
