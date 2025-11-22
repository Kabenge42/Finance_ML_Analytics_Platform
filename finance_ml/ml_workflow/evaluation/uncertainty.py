"""
Uncertainty quantification reporting and visualization (Phase 9.4).

Functions for generating notebook-friendly uncertainty diagnostics:
- Quantile prediction diagnostics (CSV)
- Coverage statistics by sector (JSON)
- Summary metrics and validation (JSON)
- Interactive HTML visualizations
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_quantile_diagnostics(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
    target_coverage: float = 0.80,
    tolerance: float = 0.10,
) -> Path:
    """
    Build quantile prediction diagnostics and export artifacts.

    Creates:
    - quantile_predictions_diagnostics.csv (detailed diagnostics per prediction)
    - coverage_by_sector.json (aggregated coverage stats by sector)
    - uncertainty_summary.json (overall summary with validation)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with columns: ticker, sector, region, y_true,
        pred_p10, pred_p50, pred_p90, and optionally interval_width
    output_dir : Union[str, Path]
        Directory to save artifacts
    target_coverage : float, default=0.80
        Target coverage rate for 80% prediction intervals
    tolerance : float, default=0.10
        Tolerance band for identifying under/over-covered sectors

    Returns
    -------
    Path
        Path to quantile_predictions_diagnostics.csv
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building quantile diagnostics for {len(predictions_df)} predictions")

    # Create diagnostics DataFrame
    diagnostics_df = predictions_df.copy()

    # Compute coverage flag (y_true within [p10, p90])
    if "y_true" in diagnostics_df.columns:
        diagnostics_df["coverage_flag_p90"] = (
            (diagnostics_df["y_true"] >= diagnostics_df["pred_p10"])
            & (diagnostics_df["y_true"] <= diagnostics_df["pred_p90"])
        ).astype(int)
    else:
        diagnostics_df["coverage_flag_p90"] = 0

    # Compute interval width if not present
    if "interval_width" not in diagnostics_df.columns:
        diagnostics_df["interval_width"] = diagnostics_df["pred_p90"] - diagnostics_df["pred_p10"]

    # Compute calibration error (distance from y_true to predicted median)
    if "y_true" in diagnostics_df.columns and "pred_p50" in diagnostics_df.columns:
        diagnostics_df["calibration_error"] = abs(
            diagnostics_df["y_true"] - diagnostics_df["pred_p50"]
        )
    else:
        diagnostics_df["calibration_error"] = 0.0

    # Save diagnostics CSV
    csv_path = output_dir / "quantile_predictions_diagnostics.csv"
    diagnostics_df.to_csv(csv_path, index=False)
    logger.info(f"Saved diagnostics CSV to {csv_path}")

    # Build coverage_by_sector.json
    if "sector" in diagnostics_df.columns and "coverage_flag_p90" in diagnostics_df.columns:
        coverage_by_sector = {}
        for sector in diagnostics_df["sector"].unique():
            sector_df = diagnostics_df[diagnostics_df["sector"] == sector]
            coverage_rate = sector_df["coverage_flag_p90"].mean()
            coverage_by_sector[sector] = {
                "coverage_rate": float(coverage_rate),
                "count": int(len(sector_df)),
                "mean_interval_width": float(sector_df["interval_width"].mean()),
            }

        coverage_json_path = output_dir / "coverage_by_sector.json"
        with open(coverage_json_path, "w") as f:
            json.dump(coverage_by_sector, f, indent=2)
        logger.info(f"Saved coverage by sector to {coverage_json_path}")

    # Build uncertainty_summary.json
    overall_coverage = diagnostics_df["coverage_flag_p90"].mean()
    mean_interval_width = diagnostics_df["interval_width"].mean()

    # Identify under/over-covered sectors
    sectors_under_covered = []
    sectors_over_covered = []

    if "sector" in diagnostics_df.columns:
        for sector in diagnostics_df["sector"].unique():
            sector_df = diagnostics_df[diagnostics_df["sector"] == sector]
            sector_coverage = sector_df["coverage_flag_p90"].mean()

            if sector_coverage < (target_coverage - tolerance):
                sectors_under_covered.append(
                    {
                        "sector": sector,
                        "coverage": float(sector_coverage),
                        "delta": float(target_coverage - sector_coverage),
                    }
                )
            elif sector_coverage > (target_coverage + tolerance):
                sectors_over_covered.append(
                    {
                        "sector": sector,
                        "coverage": float(sector_coverage),
                        "delta": float(sector_coverage - target_coverage),
                    }
                )

    # Validation status
    validation_status = (
        "PASS"
        if (target_coverage - tolerance <= overall_coverage <= target_coverage + tolerance)
        else "WARNING"
    )

    summary = {
        "overall_coverage": float(overall_coverage),
        "mean_interval_width": float(mean_interval_width),
        "target_coverage": target_coverage,
        "tolerance": tolerance,
        "sectors_under_covered": sectors_under_covered,
        "sectors_over_covered": sectors_over_covered,
        "validation_status": validation_status,
        "total_predictions": int(len(diagnostics_df)),
    }

    summary_json_path = output_dir / "uncertainty_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved uncertainty summary to {summary_json_path}")

    return csv_path


def plot_interval_coverage(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> List[Path]:
    """
    Create HTML visualizations for interval coverage analysis.

    Generates:
    - interval_width_by_bucket.html (width distribution by price buckets)
    - coverage_heatmap_region_sector.html (coverage pivot table)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with interval_width, sector, region, last_price
    output_dir : Union[str, Path]
        Directory to save HTML files

    Returns
    -------
    List[Path]
        List of paths to created HTML files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_paths = []

    try:
        import plotly.graph_objects as go
        import plotly.express as px

        # 1. Interval width by price bucket
        if "last_price" in predictions_df.columns and "interval_width" in predictions_df.columns:
            df_plot = predictions_df.copy()
            df_plot["price_bucket"] = pd.cut(
                df_plot["last_price"],
                bins=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )

            fig = px.box(
                df_plot,
                x="price_bucket",
                y="interval_width",
                title="Interval Width by Price Bucket",
                labels={"price_bucket": "Price Bucket", "interval_width": "Interval Width"},
            )

            width_html_path = output_dir / "interval_width_by_bucket.html"
            fig.write_html(str(width_html_path))
            html_paths.append(width_html_path)
            logger.info(f"Saved interval width plot to {width_html_path}")

        # 2. Coverage heatmap by region and sector
        if all(
            col in predictions_df.columns
            for col in ["sector", "region", "pred_p10", "pred_p90", "y_true"]
        ):
            df_plot = predictions_df.copy()
            df_plot["covered"] = (
                (df_plot["y_true"] >= df_plot["pred_p10"])
                & (df_plot["y_true"] <= df_plot["pred_p90"])
            ).astype(int)

            # Create pivot table
            pivot_coverage = df_plot.pivot_table(
                index="sector", columns="region", values="covered", aggfunc="mean"
            ).fillna(0)

            fig = px.imshow(
                pivot_coverage,
                labels=dict(x="Region", y="Sector", color="Coverage Rate"),
                title="Coverage Heatmap: Region vs Sector",
                color_continuous_scale="RdYlGn",
                aspect="auto",
            )

            heatmap_html_path = output_dir / "coverage_heatmap_region_sector.html"
            fig.write_html(str(heatmap_html_path))
            html_paths.append(heatmap_html_path)
            logger.info(f"Saved coverage heatmap to {heatmap_html_path}")

    except ImportError:
        logger.warning("Plotly not available, skipping HTML visualizations")

    return html_paths


def plot_reliability_diagram(
    predictions_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Path:
    """
    Create reliability diagram comparing pre- and post-calibration predictions.

    Shows calibration quality by comparing predicted vs actual coverage rates.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with y_true, y_pred, y_pred_calibrated, pred_p10, pred_p90
    output_dir : Union[str, Path]
        Directory to save HTML file

    Returns
    -------
    Path
        Path to reliability_diagram_conformal.html
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "reliability_diagram_conformal.html"

    try:
        import plotly.graph_objects as go

        # Compute coverage at different confidence levels
        df = predictions_df.copy()

        # Compute actual coverage for the 80% interval
        if all(col in df.columns for col in ["y_true", "pred_p10", "pred_p90"]):
            df["covered"] = (
                (df["y_true"] >= df["pred_p10"]) & (df["y_true"] <= df["pred_p90"])
            ).astype(int)

            actual_coverage = df["covered"].mean()

            # Create reliability diagram
            fig = go.Figure()

            # Perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(dash="dash", color="gray"),
                )
            )

            # Actual calibration point
            fig.add_trace(
                go.Scatter(
                    x=[0.80],  # Target 80% coverage
                    y=[actual_coverage],
                    mode="markers",
                    name=f"Actual (80% interval)",
                    marker=dict(size=12, color="blue"),
                )
            )

            fig.update_layout(
                title="Reliability Diagram: Conformal Calibration",
                xaxis_title="Predicted Coverage",
                yaxis_title="Actual Coverage",
                showlegend=True,
                width=800,
                height=600,
            )

            fig.write_html(str(html_path))
            logger.info(f"Saved reliability diagram to {html_path}")

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Reliability Diagram</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return html_path
