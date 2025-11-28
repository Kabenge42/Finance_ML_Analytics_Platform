"""
Enhanced Excel report generation for Phase 9.7 Reporting & Analytics.

Canonical implementation lives here under reporting/. Analytics module imports
from this module as a shim to avoid circular dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from finance_ml.ml_workflow.reporting.report_config import (
    ExcelReportConfig,
    RISK_ZSCORE_THRESHOLD,
    DISTRESS_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sheet Creators
# =============================================================================


def create_executive_summary_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """
    Create Executive_Summary sheet with key metrics and findings.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock predictions and metrics.
        metrics: Dictionary with model performance metrics (r2, mae, rmse, mape).
    """
    if df.empty:
        summary_data = [
            {"metric": "Status", "value": "No Data", "interpretation": "Empty dataset"},
        ]
    else:
        total_stocks = len(df)
        sectors = df["sector"].nunique() if "sector" in df.columns else 0
        regions = df["region"].nunique() if "region" in df.columns else 0

        # Calculate buy/sell based on mispricing
        buy_count = sell_count = 0
        if "mispricing_score" in df.columns:
            buy_count = (df["mispricing_score"] > 0).sum()
            sell_count = (df["mispricing_score"] < 0).sum()

        # Extract model metrics
        r2 = metrics.get("r2", 0) if metrics else 0
        mae = metrics.get("mae", 0) if metrics else 0
        rmse = metrics.get("rmse", 0) if metrics else 0
        mape = metrics.get("mape", 0) if metrics else 0

        summary_data = [
            {"metric": "Total Stocks", "value": total_stocks, "interpretation": "Universe size"},
            {"metric": "Sectors", "value": sectors, "interpretation": "Sector diversification"},
            {"metric": "Regions", "value": regions, "interpretation": "Geographic coverage"},
            {
                "metric": "Buy Opportunities",
                "value": buy_count,
                "interpretation": (
                    f"{buy_count/total_stocks*100:.1f}% undervalued" if total_stocks > 0 else "N/A"
                ),
            },
            {
                "metric": "Sell Candidates",
                "value": sell_count,
                "interpretation": (
                    f"{sell_count/total_stocks*100:.1f}% overvalued" if total_stocks > 0 else "N/A"
                ),
            },
            {
                "metric": "R² Score",
                "value": f"{r2:.4f}",
                "interpretation": "Model explanatory power",
            },
            {
                "metric": "MAE ($)",
                "value": f"{mae:.2f}",
                "interpretation": "Average prediction error",
            },
            {
                "metric": "RMSE ($)",
                "value": f"{rmse:.2f}",
                "interpretation": "Root mean squared error",
            },
            {
                "metric": "MAPE (%)",
                "value": f"{mape*100:.2f}",
                "interpretation": "Mean absolute percentage error",
            },
        ]

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_excel(writer, sheet_name="Executive_Summary", index=False)

    logger.debug("Created Executive_Summary sheet")


def create_quality_opportunities_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    quality_threshold: float = 0.5,
) -> None:
    """
    Create Quality_Opportunities sheet with quality-filtered stocks.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock data including quality_score.
        quality_threshold: Minimum quality score for inclusion.
    """
    if df.empty:
        empty_df = pd.DataFrame(columns=["ticker", "sector", "mispricing_score", "quality_score"])
        empty_df.to_excel(writer, sheet_name="Quality_Opportunities", index=False)
        return

    # Filter by quality score if available
    if "quality_score" in df.columns:
        filtered_df = df[df["quality_score"] >= quality_threshold].copy()
    else:
        filtered_df = df.copy()

    # Sort by mispricing (most undervalued first)
    if "mispricing_score" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("mispricing_score", ascending=False)

    # Select key columns
    key_cols = [
        "ticker",
        "sector",
        "region",
        "mispricing_score",
        "quality_score",
        "profitability_score",
        "last_price",
        "price_target",
        "market_cap",
    ]
    available_cols = [c for c in key_cols if c in filtered_df.columns]

    output_df = filtered_df[available_cols].head(50)
    output_df.to_excel(writer, sheet_name="Quality_Opportunities", index=False)

    logger.debug(f"Created Quality_Opportunities sheet with {len(output_df)} stocks")


def create_sector_leaders_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    top_n: int = 5,
) -> None:
    """
    Create Sector_Leaders sheet with top performers per sector.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock data.
        top_n: Number of top stocks per sector.
    """
    if df.empty or "sector" not in df.columns:
        empty_df = pd.DataFrame(columns=["sector", "ticker", "mispricing_score", "rank"])
        empty_df.to_excel(writer, sheet_name="Sector_Leaders", index=False)
        return

    leaders_list = []

    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector].copy()

        # Sort by mispricing (highest = most undervalued = leader)
        if "mispricing_score" in sector_df.columns:
            sector_df = sector_df.sort_values("mispricing_score", ascending=False)

        top_stocks = sector_df.head(top_n)

        for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
            leaders_list.append(
                {
                    "sector": sector,
                    "ticker": row.get("ticker", "N/A"),
                    "mispricing_score": row.get("mispricing_score", 0),
                    "quality_score": row.get("quality_score", 0),
                    "last_price": row.get("last_price", 0),
                    "rank": rank,
                }
            )

    leaders_df = pd.DataFrame(leaders_list)
    leaders_df = leaders_df.sort_values(["sector", "rank"])
    leaders_df.to_excel(writer, sheet_name="Sector_Leaders", index=False)

    logger.debug(f"Created Sector_Leaders sheet with {len(leaders_df)} entries")


def create_sector_laggards_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    bottom_n: int = 5,
) -> None:
    """
    Create Sector_Laggards sheet with bottom performers per sector.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock data.
        bottom_n: Number of bottom stocks per sector.
    """
    if df.empty or "sector" not in df.columns:
        empty_df = pd.DataFrame(columns=["sector", "ticker", "mispricing_score", "rank"])
        empty_df.to_excel(writer, sheet_name="Sector_Laggards", index=False)
        return

    laggards_list = []

    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector].copy()

        # Sort by mispricing (lowest = most overvalued = laggard)
        if "mispricing_score" in sector_df.columns:
            sector_df = sector_df.sort_values("mispricing_score", ascending=True)

        bottom_stocks = sector_df.head(bottom_n)

        for rank, (_, row) in enumerate(bottom_stocks.iterrows(), 1):
            laggards_list.append(
                {
                    "sector": sector,
                    "ticker": row.get("ticker", "N/A"),
                    "mispricing_score": row.get("mispricing_score", 0),
                    "quality_score": row.get("quality_score", 0),
                    "last_price": row.get("last_price", 0),
                    "rank": rank,
                }
            )

    laggards_df = pd.DataFrame(laggards_list)
    laggards_df = laggards_df.sort_values(["sector", "rank"])
    laggards_df.to_excel(writer, sheet_name="Sector_Laggards", index=False)

    logger.debug(f"Created Sector_Laggards sheet with {len(laggards_df)} entries")


def create_risk_assessment_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
) -> None:
    """
    Create Risk_Assessment sheet with high-risk stock indicators.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock data including risk metrics.
    """
    if df.empty:
        empty_df = pd.DataFrame(
            columns=["ticker", "risk_type", "z_score", "distress_score", "volatility"]
        )
        empty_df.to_excel(writer, sheet_name="Risk_Assessment", index=False)
        return

    risk_list = []

    for _, row in df.iterrows():
        risk_types = []

        # Check high volatility
        vol = row.get("volatility", 0)
        if vol > 0.5:
            risk_types.append("High Volatility")

        # Check z-score outliers
        z = row.get("z_score", 0)
        if abs(z) > RISK_ZSCORE_THRESHOLD:
            risk_types.append("Statistical Outlier")

        # Check financial distress
        distress = row.get("distress_score", 0)
        if distress > DISTRESS_SCORE_THRESHOLD:
            risk_types.append("Financial Distress")

        if risk_types:
            risk_list.append(
                {
                    "ticker": row.get("ticker", "N/A"),
                    "sector": row.get("sector", "N/A"),
                    "risk_type": ", ".join(risk_types),
                    "z_score": z,
                    "distress_score": distress,
                    "volatility": vol,
                }
            )

    risk_df = pd.DataFrame(risk_list)

    # Sort by number of risk types (most risky first)
    if not risk_df.empty:
        risk_df["risk_count"] = risk_df["risk_type"].str.count(",") + 1
        risk_df = risk_df.sort_values("risk_count", ascending=False)
        risk_df = risk_df.drop(columns=["risk_count"])

    risk_df.to_excel(writer, sheet_name="Risk_Assessment", index=False)

    logger.debug(f"Created Risk_Assessment sheet with {len(risk_df)} high-risk stocks")


def create_phase93_analysis_sheet(
    writer: pd.ExcelWriter,
    df: pd.DataFrame,
    category_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Create Phase93_Analysis sheet with feature category breakdown.

    Args:
        writer: pandas ExcelWriter object.
        df: DataFrame with stock data.
        category_stats: Dictionary with category statistics.
    """
    if not category_stats:
        empty_df = pd.DataFrame(columns=["category", "feature_count", "coverage_pct", "avg_value"])
        empty_df.to_excel(writer, sheet_name="Phase93_Analysis", index=False)
        return

    analysis_list = []

    for category, stats in category_stats.items():
        analysis_list.append(
            {
                "category": category.replace("_", " ").title(),
                "feature_count": stats.get("feature_count", 0),
                "coverage_pct": stats.get("coverage_pct", 0),
                "avg_value": stats.get("avg_value", 0),
            }
        )

    analysis_df = pd.DataFrame(analysis_list)
    analysis_df.to_excel(writer, sheet_name="Phase93_Analysis", index=False)

    logger.debug(f"Created Phase93_Analysis sheet with {len(analysis_df)} categories")


# =============================================================================
# Main Report Generator
# =============================================================================


def generate_enhanced_excel_report(
    df: pd.DataFrame,
    output_path: Path,
    config: Optional[ExcelReportConfig] = None,
    model_metrics: Optional[Dict[str, float]] = None,
    category_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Generate full enhanced Excel analysis report.

    Args:
        df: DataFrame with stock predictions and metrics.
        output_path: Path to save the Excel report.
        config: ExcelReportConfig with sheet toggles and parameters.
        model_metrics: Dictionary with model performance metrics.
        category_stats: Dictionary with Phase 9.3 category statistics.

    Returns:
        Path to the generated Excel report file.

    Example:
        >>> config = ExcelReportConfig(top_n_per_sector=10)
        >>> path = generate_enhanced_excel_report(df, Path("report.xlsx"), config)
    """
    if config is None:
        config = ExcelReportConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating enhanced Excel report: {output_path}")

    sheets_created = 0

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if config.include_executive_summary:
            create_executive_summary_sheet(writer, df, model_metrics)
            sheets_created += 1

        if config.include_quality_opportunities:
            create_quality_opportunities_sheet(
                writer, df, quality_threshold=config.quality_threshold
            )
            sheets_created += 1

        if config.include_sector_leaders:
            create_sector_leaders_sheet(writer, df, top_n=config.top_n_per_sector)
            create_sector_laggards_sheet(writer, df, bottom_n=config.top_n_per_sector)
            sheets_created += 2

        if config.include_risk_assessment:
            create_risk_assessment_sheet(writer, df)
            sheets_created += 1

        if config.include_phase93_analysis:
            create_phase93_analysis_sheet(writer, df, category_stats)
            sheets_created += 1

        # Always create at least one sheet with basic data summary
        if sheets_created == 0:
            summary_df = pd.DataFrame([{"info": "Report generated with minimal configuration"}])
            summary_df.to_excel(writer, sheet_name="Info", index=False)

    logger.info(f"Excel report generated successfully: {output_path} ({sheets_created} sheets)")

    return output_path


__all__ = [
    "create_executive_summary_sheet",
    "create_quality_opportunities_sheet",
    "create_sector_leaders_sheet",
    "create_sector_laggards_sheet",
    "create_risk_assessment_sheet",
    "create_phase93_analysis_sheet",
    "generate_enhanced_excel_report",
]
