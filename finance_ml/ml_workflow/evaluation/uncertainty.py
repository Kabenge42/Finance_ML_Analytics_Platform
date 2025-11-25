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
    y_true_col: str = "y_true",
    pred_cols: Optional[Dict[str, str]] = None,
    sector_col: str = "sector",
    region_col: str = "region",
    target_coverage: float = 0.80,
    tolerance: float = 0.10,
) -> pd.DataFrame:
    """Build quantile prediction diagnostics and export artifacts.

    This function is notebook- and CLI-friendly and supports flexible
    column naming while preserving the original artifact filenames used
    throughout the repository.

    Creates
    -------
    - ``quantile_predictions_diagnostics.csv`` (detailed diagnostics
      per prediction)
    - ``coverage_by_sector.json`` (aggregated coverage stats by
      sector)
    - ``uncertainty_summary.json`` (overall summary with validation
      flags)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions dataframe. By default this should contain columns
        ``y_true``, ``pred_p10``, ``pred_p50``, ``pred_p90``,
        ``sector`` and ``region``. When working with different
        schemas, use the flexible column-name arguments to map these
        roles.
    output_dir : Union[str, Path]
        Directory where diagnostics artifacts will be written.
    y_true_col : str, default "y_true"
        Name of the true target column in ``predictions_df``.
    pred_cols : Dict[str, str], optional
        Mapping of quantile labels to column names. Expected keys are
        "p10", "p50", and "p90". If ``None``, defaults to
        ``{"p10": "pred_p10", "p50": "pred_p50", "p90": "pred_p90"}``.
    sector_col : str, default "sector"
        Name of the sector column used for coverage aggregation.
    region_col : str, default "region"
        Name of the region column (reserved for future extensions).
    target_coverage : float, default 0.80
        Target coverage rate for the central prediction interval.
    tolerance : float, default 0.10
        Tolerance band around ``target_coverage`` used to flag
        under- and over-covered sectors.

    Returns
    -------
    pd.DataFrame
        Diagnostics dataframe with additional columns such as
        ``coverage_flag_p90``, ``interval_width``, and
        ``calibration_error``. This is the same content that is
        written to ``quantile_predictions_diagnostics.csv``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default prediction column mapping
    if pred_cols is None:
        pred_cols = {"p10": "pred_p10", "p50": "pred_p50", "p90": "pred_p90"}

    col_p10 = pred_cols.get("p10", "pred_p10")
    col_p50 = pred_cols.get("p50", "pred_p50")
    col_p90 = pred_cols.get("p90", "pred_p90")

    logger.info(
        "Building quantile diagnostics for %d predictions (y_true_col=%s, p10=%s, p50=%s, p90=%s)",
        len(predictions_df),
        y_true_col,
        col_p10,
        col_p50,
        col_p90,
    )

    # Create diagnostics DataFrame
    diagnostics_df = predictions_df.copy()

    # Compute coverage flag (y_true within [p10, p90])
    if y_true_col in diagnostics_df.columns and all(
        col in diagnostics_df.columns for col in (col_p10, col_p90)
    ):
        diagnostics_df["coverage_flag_p90"] = (
            (diagnostics_df[y_true_col] >= diagnostics_df[col_p10])
            & (diagnostics_df[y_true_col] <= diagnostics_df[col_p90])
        ).astype(int)
    else:
        # If we cannot compute coverage, fall back to zeros to keep
        # downstream logic robust and tests simple.
        logger.warning(
            "Unable to compute coverage_flag_p90 - missing columns among %s, %s, %s",
            y_true_col,
            col_p10,
            col_p90,
        )
        diagnostics_df["coverage_flag_p90"] = 0

    # Compute interval width if not present
    if "interval_width" not in diagnostics_df.columns and all(
        col in diagnostics_df.columns for col in (col_p10, col_p90)
    ):
        diagnostics_df["interval_width"] = diagnostics_df[col_p90] - diagnostics_df[col_p10]
    elif "interval_width" not in diagnostics_df.columns:
        # Fallback to NaN interval width when required columns are missing
        logger.warning(
            "interval_width column missing and cannot be computed because %s or %s is absent",
            col_p10,
            col_p90,
        )
        diagnostics_df["interval_width"] = np.nan

    # Compute calibration error (distance from y_true to predicted median)
    if y_true_col in diagnostics_df.columns and col_p50 in diagnostics_df.columns:
        diagnostics_df["calibration_error"] = (
            (diagnostics_df[y_true_col] - diagnostics_df[col_p50]).abs()
        )
    else:
        logger.warning(
            "calibration_error cannot be computed - missing columns %s or %s",
            y_true_col,
            col_p50,
        )
        diagnostics_df["calibration_error"] = 0.0

    # Save diagnostics CSV (path is stable for tests and notebook)
    csv_path = output_dir / "quantile_predictions_diagnostics.csv"
    diagnostics_df.to_csv(csv_path, index=False)
    logger.info("Saved diagnostics CSV to %s", csv_path)

    # Build coverage_by_sector.json (only if sector_col is present)
    if sector_col in diagnostics_df.columns and "coverage_flag_p90" in diagnostics_df.columns:
        coverage_by_sector: Dict[str, Any] = {}
        for sector in diagnostics_df[sector_col].dropna().unique():
            sector_df = diagnostics_df[diagnostics_df[sector_col] == sector]
            coverage_rate = sector_df["coverage_flag_p90"].mean()
            coverage_by_sector[str(sector)] = {
                "coverage_rate": float(coverage_rate),
                "count": int(len(sector_df)),
                "mean_interval_width": float(sector_df["interval_width"].mean()),
            }

        coverage_json_path = output_dir / "coverage_by_sector.json"
        with open(coverage_json_path, "w") as f:
            json.dump(coverage_by_sector, f, indent=2)
        logger.info("Saved coverage by sector to %s", coverage_json_path)

    # Build uncertainty_summary.json
    overall_coverage = float(diagnostics_df["coverage_flag_p90"].mean())
    mean_interval_width = float(diagnostics_df["interval_width"].mean())

    # Identify under/over-covered sectors
    sectors_under_covered: List[Dict[str, Any]] = []
    sectors_over_covered: List[Dict[str, Any]] = []

    if sector_col in diagnostics_df.columns:
        for sector in diagnostics_df[sector_col].dropna().unique():
            sector_df = diagnostics_df[diagnostics_df[sector_col] == sector]
            sector_coverage = float(sector_df["coverage_flag_p90"].mean())

            if sector_coverage < (target_coverage - tolerance):
                sectors_under_covered.append(
                    {
                        "sector": str(sector),
                        "coverage": sector_coverage,
                        "delta": float(target_coverage - sector_coverage),
                    }
                )
            elif sector_coverage > (target_coverage + tolerance):
                sectors_over_covered.append(
                    {
                        "sector": str(sector),
                        "coverage": sector_coverage,
                        "delta": float(sector_coverage - target_coverage),
                    }
                )

    # Validation status and high-level flags expected by notebook
    within_tolerance = bool(
        target_coverage - tolerance <= overall_coverage <= target_coverage + tolerance
    )
    validation_status = "PASS" if within_tolerance else "WARNING"

    summary = {
        "overall_coverage": overall_coverage,
        "mean_interval_width": mean_interval_width,
        "target_coverage": float(target_coverage),
        "tolerance": float(tolerance),
        "sectors_under_covered": sectors_under_covered,
        "sectors_over_covered": sectors_over_covered,
        "validation_status": validation_status,
        "within_tolerance": within_tolerance,
        "under_covered_sectors": [s["sector"] for s in sectors_under_covered],
        "over_covered_sectors": [s["sector"] for s in sectors_over_covered],
        "total_predictions": int(len(diagnostics_df)),
    }

    summary_json_path = output_dir / "uncertainty_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved uncertainty summary to %s", summary_json_path)

    # Notebook and CLI expect the diagnostics DataFrame, but the
    # diagnostics CSV path remains stable for external consumers/tests
    # that load from disk.
    return diagnostics_df


def plot_interval_coverage(
    output_dir: Union[str, Path],
    predictions_df: Optional[pd.DataFrame] = None,
    *,
    diagnostics_df: Optional[pd.DataFrame] = None,
    last_price_col: str = "last_price",
    interval_width_col: str = "interval_width",
    sector_col: str = "sector",
    region_col: str = "region",
    y_true_col: str = "y_true",
    pred_cols: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Create HTML visualizations for interval coverage analysis.

    This helper is used both in the notebook and tests. It accepts
    either ``predictions_df`` (original API) or ``diagnostics_df``
    (notebook API); if both are provided, ``diagnostics_df`` takes
    precedence.

    Generates
    ---------
    - ``interval_width_by_bucket.html`` (width distribution by price
      buckets)
    - ``coverage_heatmap_region_sector.html`` (coverage pivot table)

    Parameters
    ----------
    predictions_df : pd.DataFrame, optional
        Input dataframe when using the original API.
    output_dir : Union[str, Path]
        Directory to save HTML files.
    diagnostics_df : pd.DataFrame, optional
        Input dataframe when using the notebook API.
    last_price_col : str, default "last_price"
        Name of the last price column used for bucketing.
    interval_width_col : str, default "interval_width"
        Name of the interval width column.
    sector_col : str, default "sector"
        Name of the sector column.
    region_col : str, default "region"
        Name of the region column.
    y_true_col : str, default "y_true"
        Name of the true target column.
    pred_cols : Dict[str, str], optional
        Mapping of quantile labels to column names. Expected keys are
        "p10" and "p90". If ``None``, defaults to
        ``{"p10": "pred_p10", "p90": "pred_p90"}``.

    Returns
    -------
    List[Path]
        List of paths to created HTML files.
    """

    # Prefer diagnostics_df (notebook) over predictions_df (legacy)
    df_plot = diagnostics_df if diagnostics_df is not None else predictions_df
    if df_plot is None:
        raise ValueError("Either predictions_df or diagnostics_df must be provided")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_paths: List[Path] = []

    # Default prediction columns for coverage heatmap
    if pred_cols is None:
        pred_cols = {"p10": "pred_p10", "p90": "pred_p90"}

    col_p10 = pred_cols.get("p10", "pred_p10")
    col_p90 = pred_cols.get("p90", "pred_p90")

    try:
        import plotly.express as px

        df_plot = df_plot.copy()

        # 1. Interval width by price bucket
        if last_price_col in df_plot.columns and interval_width_col in df_plot.columns:
            df_plot["price_bucket"] = pd.cut(
                df_plot[last_price_col],
                bins=5,
                labels=["Very Low", "Low", "Medium", "High", "Very High"],
            )

            fig = px.box(
                df_plot,
                x="price_bucket",
                y=interval_width_col,
                title="Interval Width by Price Bucket",
                labels={"price_bucket": "Price Bucket", interval_width_col: "Interval Width"},
                template="plotly_dark",
                color_discrete_sequence=["#375a7f"],
            )
            fig.update_layout(font_family="Arial")

            width_html_path = output_dir / "interval_width_by_bucket.html"
            fig.write_html(str(width_html_path))
            html_paths.append(width_html_path)
            logger.info("Saved interval width plot to %s", width_html_path)

        # 2. Coverage heatmap by region and sector
        required_for_heatmap = [sector_col, region_col, col_p10, col_p90, y_true_col]
        if all(col in df_plot.columns for col in required_for_heatmap):
            df_plot["covered"] = (
                (df_plot[y_true_col] >= df_plot[col_p10])
                & (df_plot[y_true_col] <= df_plot[col_p90])
            ).astype(int)

            # Create pivot table
            pivot_coverage = df_plot.pivot_table(
                index=sector_col, columns=region_col, values="covered", aggfunc="mean"
            ).fillna(0)

            fig = px.imshow(
                pivot_coverage,
                labels=dict(x="Region", y="Sector", color="Coverage Rate"),
                title="Coverage Heatmap: Region vs Sector",
                color_continuous_scale="RdYlGn",
                aspect="auto",
                text_auto=".2%",
                template="plotly_dark",
            )
            fig.update_layout(font_family="Arial")

            heatmap_html_path = output_dir / "coverage_heatmap_region_sector.html"
            fig.write_html(str(heatmap_html_path))
            html_paths.append(heatmap_html_path)
            logger.info("Saved coverage heatmap to %s", heatmap_html_path)

    except ImportError:
        logger.warning("Plotly not available, skipping HTML visualizations")

    return html_paths


def plot_reliability_diagram(
    output_dir: Union[str, Path],
    predictions_df: Optional[pd.DataFrame] = None,
    *,
    diagnostics_df: Optional[pd.DataFrame] = None,
    pre_calibration_df: Optional[pd.DataFrame] = None,
    y_true_col: str = "y_true",
    pred_cols: Optional[Dict[str, str]] = None,
) -> Path:
    """Create reliability diagram comparing post-calibration predictions.

    Similar to :func:`plot_interval_coverage`, this function accepts
    both ``predictions_df`` (legacy API) and ``diagnostics_df``
    (notebook API). The optional ``pre_calibration_df`` argument is
    accepted for API compatibility but is not yet used.

    Returns
    -------
    Path
        Path to ``reliability_diagram_conformal.html``.
    """

    # Prefer diagnostics_df (notebook) over predictions_df (legacy)
    df = diagnostics_df if diagnostics_df is not None else predictions_df
    if df is None:
        raise ValueError("Either predictions_df or diagnostics_df must be provided")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html_path = output_dir / "reliability_diagram_conformal.html"

    # Default prediction columns for coverage computation
    if pred_cols is None:
        pred_cols = {"p10": "pred_p10", "p90": "pred_p90"}

    col_p10 = pred_cols.get("p10", "pred_p10")
    col_p90 = pred_cols.get("p90", "pred_p90")

    try:
        import plotly.graph_objects as go

        df = df.copy()

        if all(col in df.columns for col in [y_true_col, col_p10, col_p90]):
            df["covered"] = (
                (df[y_true_col] >= df[col_p10]) & (df[y_true_col] <= df[col_p90])
            ).astype(int)

            actual_coverage = float(df["covered"].mean())

            # Create reliability diagram
            fig = go.Figure()

            # Perfect calibration line
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Perfect Calibration",
                    line=dict(dash="dash", color="#adb5bd"),
                )
            )

            # Actual calibration point
            fig.add_trace(
                go.Scatter(
                    x=[0.80],  # Target 80% coverage
                    y=[actual_coverage],
                    mode="markers",
                    name="Actual (80% interval)",
                    marker=dict(size=12, color="#375a7f"),
                )
            )

            fig.update_layout(
                title="Reliability Diagram: Conformal Calibration",
                xaxis_title="Predicted Coverage",
                yaxis_title="Actual Coverage",
                showlegend=True,
                width=800,
                height=600,
                template="plotly_dark",
            )
            fig.update_layout(font_family="Arial")

            fig.write_html(str(html_path))
            logger.info("Saved reliability diagram to %s", html_path)

    except ImportError:
        logger.warning("Plotly not available, creating minimal HTML")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>Reliability Diagram</h1>")
            f.write("<p>Plotly required for interactive visualization</p>")
            f.write("</body></html>")

    return html_path
