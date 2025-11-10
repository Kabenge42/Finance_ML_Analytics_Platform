"""
Phase 9.2: EDA Reports Module

HTML and static report generation for EDA results.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import logging

from finance_ml.ml_workflow.eda.eda import eda_summary

logger = logging.getLogger(__name__)


def generate_eda_report(
    df: pd.DataFrame,
    out_dir: Union[str, Path] = "outputs/eda",
    sector_column: str = "sector",
    include_correlations: bool = True,
    report_name: Optional[str] = None,
) -> Path:
    """
    Generate HTML EDA report from dataframe.

    Args:
        df: Input dataframe
        out_dir: Output directory for report
        sector_column: Name of sector column
        include_correlations: Whether to include correlation matrix
        report_name: Custom report name (defaults to timestamped name)

    Returns:
        Path to generated HTML report
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate EDA summary
    summary = eda_summary(
        df, sector_column=sector_column, include_correlations=include_correlations
    )

    # Generate report filename
    if report_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"eda_report_{timestamp}.html"
    elif not report_name.endswith(".html"):
        report_name = f"{report_name}.html"

    report_path = out_dir / report_name

    # Generate HTML content
    html_content = _generate_html_content(df, summary, sector_column)

    # Write to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"EDA report generated: {report_path}")
    return report_path


def _generate_html_content(df: pd.DataFrame, summary: dict, sector_column: str) -> str:
    """Generate HTML content for EDA report"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>EDA Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
        }}
        h3 {{
            color: #666;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px 10px 0;
        }}
        .metric-label {{
            font-weight: bold;
            color: #555;
        }}
        .metric-value {{
            color: #4CAF50;
            font-size: 1.2em;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Exploratory Data Analysis Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
"""

    # Basic information
    html += f"""
        <h2>Dataset Overview</h2>
        <div class="metric">
            <span class="metric-label">Rows:</span>
            <span class="metric-value">{summary['shape'][0]:,}</span>
        </div>
        <div class="metric">
            <span class="metric-label">Columns:</span>
            <span class="metric-value">{summary['shape'][1]}</span>
        </div>
"""

    # Missing values
    if summary["missing_values"]["counts"]:
        html += "<h2>Missing Values</h2>"
        missing_data = [
            (col, count) for col, count in summary["missing_values"]["counts"].items() if count > 0
        ]

        if missing_data:
            html += "<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>"
            for col, count in missing_data:
                pct = summary["missing_values"]["percentages"].get(col, 0)
                html += f"<tr><td>{col}</td><td>{count}</td><td>{pct:.2f}%</td></tr>"
            html += "</table>"
        else:
            html += "<p>No missing values detected.</p>"

    # Numeric summary
    if summary["numeric_summary"]:
        html += "<h2>Numeric Columns Summary</h2>"
        html += "<table><tr><th>Column</th><th>Count</th><th>Mean</th><th>Std</th><th>Min</th><th>25%</th><th>50%</th><th>75%</th><th>Max</th></tr>"

        for col, stats in summary["numeric_summary"].items():
            if isinstance(stats, dict):
                html += f"<tr><td><strong>{col}</strong></td>"
                html += f"<td>{stats.get('count', 'N/A'):.0f}</td>"
                html += f"<td>{stats.get('mean', 0):.2f}</td>"
                html += f"<td>{stats.get('std', 0):.2f}</td>"
                html += f"<td>{stats.get('min', 0):.2f}</td>"
                html += f"<td>{stats.get('25%', 0):.2f}</td>"
                html += f"<td>{stats.get('50%', 0):.2f}</td>"
                html += f"<td>{stats.get('75%', 0):.2f}</td>"
                html += f"<td>{stats.get('max', 0):.2f}</td>"
                html += "</tr>"
        html += "</table>"

    # Categorical summary
    if summary["categorical_summary"]:
        html += "<h2>Categorical Columns Summary</h2>"
        for col, stats in summary["categorical_summary"].items():
            html += f"<h3>{col}</h3>"
            html += f"<p><strong>Unique values:</strong> {stats['unique_count']}</p>"
            if stats["top_values"]:
                html += "<table><tr><th>Value</th><th>Count</th></tr>"
                for value, count in list(stats["top_values"].items())[:10]:
                    html += f"<tr><td>{value}</td><td>{count}</td></tr>"
                html += "</table>"

    # Sector distribution
    if "sector_distribution" in summary:
        html += f"<h2>Sector Distribution</h2>"
        html += "<table><tr><th>Sector</th><th>Count</th></tr>"
        for sector, count in summary["sector_distribution"].items():
            html += f"<tr><td>{sector}</td><td>{count}</td></tr>"
        html += "</table>"

    # Correlations
    if "correlations" in summary:
        html += "<h2>Correlation Matrix</h2>"
        html += "<p><em>Showing strong correlations (&gt; 0.5 or &lt; -0.5)</em></p>"
        corr_data = summary["correlations"]
        if corr_data:
            html += "<table><tr><th>Variable 1</th><th>Variable 2</th><th>Correlation</th></tr>"
            for col1, corr_dict in corr_data.items():
                for col2, corr_val in corr_dict.items():
                    if col1 < col2 and abs(corr_val) > 0.5 and abs(corr_val) < 0.999:
                        html += f"<tr><td>{col1}</td><td>{col2}</td><td>{corr_val:.3f}</td></tr>"
            html += "</table>"

    html += """
    </div>
</body>
</html>
"""

    return html


def generate_benchmarking_report(
    df: pd.DataFrame,
    out_dir: Union[str, Path] = "outputs/eda",
    sector_column: str = "sector",
    region_column: str = "region",
) -> Path:
    """
    Generate benchmarking report (wrapper for backward compatibility).

    This is a simplified wrapper that generates an EDA report.
    For full benchmarking functionality, use finance_ml.ml_workflow.eda.benchmarking.generate_benchmarking_report

    Args:
        df: Input dataframe
        out_dir: Output directory
        sector_column: Sector column name
        region_column: Region column name

    Returns:
        Path to generated report
    """
    from finance_ml.ml_workflow.eda.benchmarking import (
        generate_benchmarking_report as gen_bench_report,
    )

    return gen_bench_report(
        df=df,
        metrics=["last_price", "market_cap", "pe_ratio"] if "last_price" in df.columns else [],
        sector_column=sector_column,
        region_column=region_column,
    )
