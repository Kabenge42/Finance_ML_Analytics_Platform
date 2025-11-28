"""
Enhanced HTML report generation for Phase 9.7 Reporting & Analytics.

Canonical implementation lives here under reporting/. Analytics module imports
from this module as a shim to avoid circular dependencies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from finance_ml.ml_workflow.reporting.report_config import (
    HTMLReportConfig,
    RISK_ZSCORE_THRESHOLD,
    DISTRESS_SCORE_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# HTML Templates
# =============================================================================

HTML_HEADER = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Valuation Analysis Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-card {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metric-item {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            margin: 15px 0;
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .risk-alerts {{ display: flex; flex-wrap: wrap; gap: 10px; }}
        .alert {{
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: 500;
        }}
        .alert.high-risk {{ background: #e74c3c; color: white; }}
        .alert.warning {{ background: #f39c12; color: white; }}
        .alert.info {{ background: #3498db; color: white; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
    </style>
</head>
<body>
<h1>📊 Stock Valuation Analysis Summary</h1>
"""

HTML_FOOTER = """
</body>
</html>
"""


# =============================================================================
# Section Generators
# =============================================================================


def generate_executive_summary_html(
    df: pd.DataFrame,
    model_metrics: Optional[Dict[str, float]] = None,
) -> str:
    """
    Generate executive summary HTML section with key findings.

    Args:
        df: DataFrame with stock predictions and metrics.
        model_metrics: Dictionary with model performance metrics (r2, mae, rmse, mape).

    Returns:
        HTML string for executive summary section.
    """
    if df.empty:
        return "<h2>📋 Executive Summary</h2><p>No data available.</p>"

    total_stocks = len(df)
    sectors = df["sector"].nunique() if "sector" in df.columns else 0

    # Calculate buy/sell opportunities based on mispricing
    mispricing_col = "mispricing_score"
    if mispricing_col in df.columns:
        buy_count = (df[mispricing_col] > 0).sum()
        sell_count = (df[mispricing_col] < 0).sum()
    else:
        buy_count = sell_count = 0

    buy_pct = (buy_count / total_stocks * 100) if total_stocks > 0 else 0
    sell_pct = (sell_count / total_stocks * 100) if total_stocks > 0 else 0

    # Model metrics
    r2 = model_metrics.get("r2", 0) if model_metrics else 0
    mae = model_metrics.get("mae", 0) if model_metrics else 0

    html = f"""
<h2>📋 Executive Summary</h2>
<div class="summary-card">
    <h3>Key Findings</h3>
    <ul>
        <li>Total Universe: <strong>{total_stocks}</strong> stocks across <strong>{sectors}</strong> sectors</li>
        <li>Buy Opportunities: <strong>{buy_count}</strong> stocks (<strong>{buy_pct:.1f}%</strong>)</li>
        <li>Sell Candidates: <strong>{sell_count}</strong> stocks (<strong>{sell_pct:.1f}%</strong>)</li>
        <li>Model Accuracy: R² = <strong>{r2:.3f}</strong>, MAE = $<strong>{mae:.2f}</strong></li>
    </ul>
</div>
<div class="summary-card">
    <h3>Model Performance</h3>
    <div class="metric-grid">
        <div class="metric-item">
            <div class="metric-value">{r2:.3f}</div>
            <div class="metric-label">R² Score</div>
        </div>
        <div class="metric-item">
            <div class="metric-value">${mae:.2f}</div>
            <div class="metric-label">MAE</div>
        </div>
"""

    if model_metrics:
        if "rmse" in model_metrics:
            html += f"""
        <div class="metric-item">
            <div class="metric-value">${model_metrics['rmse']:.2f}</div>
            <div class="metric-label">RMSE</div>
        </div>
"""
        if "mape" in model_metrics:
            html += f"""
        <div class="metric-item">
            <div class="metric-value">{model_metrics['mape']*100:.1f}%</div>
            <div class="metric-label">MAPE</div>
        </div>
"""

    html += """
    </div>
</div>
"""
    return html


def generate_sector_breakdown_html(
    df: pd.DataFrame,
    sector_col: str = "sector",
) -> str:
    """
    Generate sector breakdown HTML section with metrics table.

    Args:
        df: DataFrame with stock data.
        sector_col: Column name for sector classification.

    Returns:
        HTML string for sector breakdown section.
    """
    if df.empty or sector_col not in df.columns:
        return "<h2>📊 Sector Analysis</h2><p>No sector data available.</p>"

    # Calculate sector metrics
    sector_stats = []
    for sector in df[sector_col].unique():
        sector_df = df[df[sector_col] == sector]
        stats = {
            "Sector": sector,
            "Stocks": len(sector_df),
        }

        if "mispricing_score" in df.columns:
            stats["Avg Mispricing"] = f"{sector_df['mispricing_score'].mean():.2%}"
            # Top performer (highest mispricing = most undervalued)
            top_idx = sector_df["mispricing_score"].idxmax()
            stats["Leader"] = sector_df.loc[top_idx, "ticker"] if "ticker" in df.columns else "N/A"
            # Bottom performer
            bottom_idx = sector_df["mispricing_score"].idxmin()
            stats["Laggard"] = (
                sector_df.loc[bottom_idx, "ticker"] if "ticker" in df.columns else "N/A"
            )
        else:
            stats["Avg Mispricing"] = "N/A"
            stats["Leader"] = "N/A"
            stats["Laggard"] = "N/A"

        sector_stats.append(stats)

    # Sort by number of stocks
    sector_stats.sort(key=lambda x: x["Stocks"], reverse=True)

    # Build table HTML
    html = """
<h2>📊 Sector Analysis</h2>
<div class="summary-card">
<table>
    <tr>
        <th>Sector</th>
        <th>Stocks</th>
        <th>Avg Mispricing</th>
        <th>Leader</th>
        <th>Laggard</th>
    </tr>
"""
    for stats in sector_stats:
        html += f"""
    <tr>
        <td><strong>{stats['Sector']}</strong></td>
        <td>{stats['Stocks']}</td>
        <td>{stats['Avg Mispricing']}</td>
        <td class="positive">{stats['Leader']}</td>
        <td class="negative">{stats['Laggard']}</td>
    </tr>
"""
    html += """
</table>
</div>
"""
    return html


def generate_quality_filtered_html(
    df: pd.DataFrame,
    quality_threshold: float = 0.5,
    top_n: int = 20,
) -> str:
    """
    Generate quality-filtered rankings HTML section.

    Args:
        df: DataFrame with stock data including quality_score.
        quality_threshold: Minimum quality score for inclusion.
        top_n: Number of top stocks to display.

    Returns:
        HTML string for quality-filtered rankings section.
    """
    if df.empty:
        return "<h2>🏆 Quality-Filtered Opportunities</h2><p>No data available.</p>"

    # Filter by quality score if available
    if "quality_score" in df.columns:
        filtered_df = df[df["quality_score"] >= quality_threshold].copy()
    else:
        filtered_df = df.copy()

    # Sort by mispricing (undervalued first)
    if "mispricing_score" in filtered_df.columns:
        filtered_df = filtered_df.sort_values("mispricing_score", ascending=False)

    # Take top N
    display_df = filtered_df.head(top_n)

    html = f"""
<h2>🏆 Quality-Filtered Opportunities</h2>
<div class="summary-card">
    <p>Showing top {len(display_df)} stocks with quality score ≥ {quality_threshold:.0%}</p>
<table>
    <tr>
        <th>Rank</th>
        <th>Ticker</th>
        <th>Sector</th>
        <th>Mispricing</th>
        <th>Quality Score</th>
        <th>Last Price</th>
    </tr>
"""

    for i, (_, row) in enumerate(display_df.iterrows(), 1):
        ticker = row.get("ticker", "N/A")
        sector = row.get("sector", "N/A")
        mispricing = row.get("mispricing_score", 0)
        quality = row.get("quality_score", 0)
        price = row.get("last_price", 0)

        mispricing_class = "positive" if mispricing > 0 else "negative"

        html += f"""
    <tr>
        <td>{i}</td>
        <td><strong>{ticker}</strong></td>
        <td>{sector}</td>
        <td class="{mispricing_class}">{mispricing:.2%}</td>
        <td>{quality:.2f}</td>
        <td>${price:.2f}</td>
    </tr>
"""

    html += """
</table>
</div>
"""
    return html


def generate_risk_warnings_html(df: pd.DataFrame) -> str:
    """
    Generate risk warnings HTML section.

    Args:
        df: DataFrame with stock data including risk metrics.

    Returns:
        HTML string for risk warnings section.
    """
    if df.empty:
        return "<h2>⚠️ Risk Warnings</h2><p>No data available.</p>"

    warnings = []

    # High volatility stocks
    if "volatility" in df.columns:
        high_vol = df[df["volatility"] > 0.5]
        if len(high_vol) > 0:
            warnings.append(
                {
                    "type": "high-risk",
                    "message": f"High Volatility: {len(high_vol)} stocks with volatility > 50%",
                }
            )

    # High z-score (outliers)
    if "z_score" in df.columns:
        outliers = df[df["z_score"].abs() > RISK_ZSCORE_THRESHOLD]
        if len(outliers) > 0:
            warnings.append(
                {
                    "type": "warning",
                    "message": f"Statistical Outliers: {len(outliers)} stocks with |z-score| > {RISK_ZSCORE_THRESHOLD}",
                }
            )

    # Financial distress
    if "distress_score" in df.columns:
        distressed = df[df["distress_score"] > DISTRESS_SCORE_THRESHOLD]
        if len(distressed) > 0:
            warnings.append(
                {
                    "type": "high-risk",
                    "message": f"Financial Distress: {len(distressed)} stocks with distress score > {DISTRESS_SCORE_THRESHOLD}",
                }
            )

    html = """
<h2>⚠️ Risk Warnings</h2>
<div class="summary-card">
    <div class="risk-alerts">
"""

    if not warnings:
        html += '<div class="alert info">No significant risk alerts detected.</div>'
    else:
        for warning in warnings:
            html += f'<div class="alert {warning["type"]}">{warning["message"]}</div>\n'

    html += """
    </div>
</div>
"""

    # Add detailed risk table if there are high-risk stocks
    if "volatility" in df.columns:
        high_risk_df = df.nlargest(10, "volatility")
        html += """
<div class="summary-card">
    <h3>Top 10 High Volatility Stocks</h3>
    <table>
        <tr>
            <th>Ticker</th>
            <th>Sector</th>
            <th>Volatility</th>
            <th>Z-Score</th>
            <th>Distress Score</th>
        </tr>
"""
        for _, row in high_risk_df.iterrows():
            ticker = row.get("ticker", "N/A")
            sector = row.get("sector", "N/A")
            vol = row.get("volatility", 0)
            z = row.get("z_score", 0)
            distress = row.get("distress_score", 0)

            html += f"""
        <tr>
            <td><strong>{ticker}</strong></td>
            <td>{sector}</td>
            <td class="negative">{vol:.2%}</td>
            <td>{z:.2f}</td>
            <td>{distress:.2f}</td>
        </tr>
"""
        html += """
    </table>
</div>
"""

    return html


def generate_phase93_summary_html(
    df: pd.DataFrame,
    category_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """
    Generate Phase 9.3 feature category summary HTML section.

    Args:
        df: DataFrame with stock data.
        category_stats: Dictionary with category statistics (feature_count, coverage_pct, avg_value).

    Returns:
        HTML string for Phase 9.3 summary section.
    """
    if not category_stats:
        return "<h2>🔬 Feature Quality Analysis</h2><p>No category statistics available.</p>"

    html = """
<h2>🔬 Feature Quality Analysis</h2>
<div class="summary-card">
    <p>Phase 9.3 feature category breakdown and coverage analysis</p>
    <table>
        <tr>
            <th>Category</th>
            <th>Features</th>
            <th>Coverage</th>
            <th>Avg Score</th>
        </tr>
"""

    for category, stats in category_stats.items():
        feature_count = stats.get("feature_count", 0)
        coverage = stats.get("coverage_pct", 0)
        avg_value = stats.get("avg_value", 0)

        # Color code coverage
        coverage_class = "positive" if coverage >= 0.9 else ("negative" if coverage < 0.7 else "")

        html += f"""
        <tr>
            <td><strong>{category.replace('_', ' ').title()}</strong></td>
            <td>{feature_count}</td>
            <td class="{coverage_class}">{coverage:.1%}</td>
            <td>{avg_value:.2f}</td>
        </tr>
"""

    html += """
    </table>
</div>
"""
    return html


# =============================================================================
# Main Report Generator
# =============================================================================


def generate_enhanced_analysis_html(
    df: pd.DataFrame,
    output_path: Path,
    config: Optional[HTMLReportConfig] = None,
    model_metrics: Optional[Dict[str, float]] = None,
    category_stats: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Path:
    """
    Generate full enhanced HTML analysis report.

    Args:
        df: DataFrame with stock predictions and metrics.
        output_path: Path to save the HTML report.
        config: HTMLReportConfig with section toggles and parameters.
        model_metrics: Dictionary with model performance metrics.
        category_stats: Dictionary with Phase 9.3 category statistics.

    Returns:
        Path to the generated HTML report file.

    Example:
        >>> config = HTMLReportConfig(top_n_stocks=50)
        >>> path = generate_enhanced_analysis_html(df, Path("report.html"), config)
    """
    if config is None:
        config = HTMLReportConfig()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating enhanced HTML report: {output_path}")

    # Build report sections
    sections = [HTML_HEADER]

    if config.include_executive_summary:
        sections.append(generate_executive_summary_html(df, model_metrics))

    if config.include_sector_breakdown:
        sections.append(generate_sector_breakdown_html(df))

    if config.include_quality_filtered:
        sections.append(
            generate_quality_filtered_html(
                df,
                quality_threshold=config.quality_threshold,
                top_n=config.top_n_stocks,
            )
        )

    if config.include_risk_warnings:
        sections.append(generate_risk_warnings_html(df))

    if config.include_phase93_summary:
        sections.append(generate_phase93_summary_html(df, category_stats))

    sections.append(HTML_FOOTER)

    # Write report
    html_content = "\n".join(sections)
    output_path.write_text(html_content, encoding="utf-8")

    logger.info(f"HTML report generated successfully: {output_path}")

    return output_path


__all__ = [
    "generate_executive_summary_html",
    "generate_sector_breakdown_html",
    "generate_quality_filtered_html",
    "generate_risk_warnings_html",
    "generate_phase93_summary_html",
    "generate_enhanced_analysis_html",
]
