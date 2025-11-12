"""
Phase 9.8: Reporting Export Module

Export predictions and results to various formats (CSV, Excel, JSON).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def export_predictions(
    predictions: Union[pd.DataFrame, Dict[str, Any]],
    out_path: Union[str, Path] = "outputs/regression_predictions.csv",
    file_format: str = "auto",
    include_metadata: bool = True,
) -> Path:
    """
    Export predictions to file.

    Args:
        predictions: Predictions dataframe or dict with 'predictions' key
        out_path: Output file path
        file_format: Export format ('auto', 'csv', 'excel', 'json')
        include_metadata: Whether to include metadata sheet (Excel only)

    Returns:
        Path to exported file
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract dataframe from dict if needed
    if isinstance(predictions, dict):
        if "predictions" in predictions:
            df = predictions["predictions"]
            metadata = {k: v for k, v in predictions.items() if k != "predictions"}
        else:
            # Assume it's a regression result with various keys
            df = pd.DataFrame(predictions)
            metadata = {}
    else:
        df = predictions
        metadata = {}

    # Auto-detect format from extension
    if file_format == "auto":
        ext = out_path.suffix.lower()
        if ext in [".xlsx", ".xls"]:
            file_format = "excel"
        elif ext == ".json":
            file_format = "json"
        else:
            file_format = "csv"

    # Export based on format
    if file_format == "csv":
        df.to_csv(out_path, index=False)
        logger.info(f"Predictions exported to CSV: {out_path}")

    elif file_format == "excel":
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Predictions", index=False)

            # Add metadata sheet if requested
            if include_metadata and metadata:
                metadata_df = pd.DataFrame(
                    [{"Key": k, "Value": str(v)} for k, v in metadata.items()]
                )
                metadata_df.to_excel(writer, sheet_name="Metadata", index=False)

        logger.info(f"Predictions exported to Excel: {out_path}")

    elif file_format == "json":
        if isinstance(df, pd.DataFrame):
            df.to_json(out_path, orient="records", indent=2)
        else:
            import json

            with open(out_path, "w") as f:
                json.dump(predictions, f, indent=2, default=str)
        logger.info(f"Predictions exported to JSON: {out_path}")

    else:
        raise ValueError(f"Unsupported format: {file_format}")

    return out_path


def export_model_results(
    results: Dict[str, Any],
    out_dir: Union[str, Path] = "outputs/models",
    prefix: str = "model_results",
) -> Dict[str, Path]:
    """
    Export comprehensive model results including predictions, metrics, and artifacts.

    Args:
        results: Dictionary containing model results
        out_dir: Output directory
        prefix: Filename prefix

    Returns:
        Dictionary mapping result type to file path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export predictions
    if "predictions" in results:
        pred_path = out_dir / f"{prefix}_predictions_{timestamp}.csv"
        if isinstance(results["predictions"], pd.DataFrame):
            results["predictions"].to_csv(pred_path, index=False)
        else:
            pd.DataFrame({"predictions": results["predictions"]}).to_csv(pred_path, index=False)
        exported_files["predictions"] = pred_path

    # Export metrics
    if "metrics" in results:
        metrics_path = out_dir / f"{prefix}_metrics_{timestamp}.json"
        import json

        with open(metrics_path, "w") as f:
            json.dump(results["metrics"], f, indent=2, default=str)
        exported_files["metrics"] = metrics_path

    # Export feature importance if available
    if "feature_importance" in results:
        fi_path = out_dir / f"{prefix}_feature_importance_{timestamp}.csv"
        if isinstance(results["feature_importance"], pd.DataFrame):
            results["feature_importance"].to_csv(fi_path, index=False)
        elif isinstance(results["feature_importance"], dict):
            pd.DataFrame(
                [{"feature": k, "importance": v} for k, v in results["feature_importance"].items()]
            ).to_csv(fi_path, index=False)
        exported_files["feature_importance"] = fi_path

    logger.info(f"Model results exported to {out_dir}: {list(exported_files.keys())}")
    return exported_files


def create_summary_report(
    results: Dict[str, Any],
    out_path: Union[str, Path] = "outputs/summary_report.xlsx",
) -> Path:
    """
    Create a comprehensive Excel summary report.

    Args:
        results: Dictionary containing various result components
        out_path: Output file path

    Returns:
        Path to created report
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Overview sheet
        overview_data = []
        if "metrics" in results:
            for key, value in results["metrics"].items():
                overview_data.append({"Metric": key, "Value": value})

        if overview_data:
            pd.DataFrame(overview_data).to_excel(writer, sheet_name="Overview", index=False)

        # Predictions sheet
        if "predictions" in results:
            if isinstance(results["predictions"], pd.DataFrame):
                results["predictions"].to_excel(writer, sheet_name="Predictions", index=False)

        # Feature importance sheet
        if "feature_importance" in results:
            if isinstance(results["feature_importance"], pd.DataFrame):
                results["feature_importance"].to_excel(
                    writer, sheet_name="Feature Importance", index=False
                )

        # By sector analysis if available
        if "by_sector" in results:
            for sector, sector_data in results["by_sector"].items():
                sheet_name = f"Sector_{sector}"[:31]  # Excel sheet name limit
                if isinstance(sector_data, pd.DataFrame):
                    sector_data.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Summary report created: {out_path}")
    return out_path
