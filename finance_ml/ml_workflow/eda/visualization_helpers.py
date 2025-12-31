# finance_ml/ml_workflow/eda/visualization_helpers.py
"""Reusable visualization helper functions."""
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE


def load_json_file(file_path: Path) -> Optional[dict]:
    """Load JSON file and return data, or None if file doesn't exist."""
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        return json.load(f)


def save_and_display_figure(fig, output_path: Path, display_inline: bool = True):
    """Save figure to HTML and optionally display inline."""
    fig.write_html(str(output_path))
    print(f"  ✓ Saved: {output_path.name}")
    if display_inline:
        fig.show()


def create_hypothesis_heatmap(sector_tests: dict) -> Optional[go.Figure]:
    """Create hypothesis testing heatmap from sector test results."""
    test_metrics = [k for k in sector_tests.keys() if k != "summary"]
    test_types = ["anova", "kruskal_wallis"]
    heatmap_data = []

    for metric in test_metrics:
        for test_type in test_types:
            test_result = sector_tests[metric].get(test_type)
            if not test_result:
                continue

            p_value = test_result.get("p_value", 1.0)
            significant = test_result.get("significant", "False") == "True"
            heatmap_data.append(
                {
                    "Metric": metric.replace("_", " ").title(),
                    "Test": test_type.replace("_", " ").title(),
                    "P-Value": p_value,
                    "Significant": "Yes" if significant else "No",
                    "-log10(p)": -np.log10(max(p_value, 1e-100)),
                }
            )

    if not heatmap_data:
        return None

    heatmap_df = pd.DataFrame(heatmap_data)
    pivot_df = heatmap_df.pivot(index="Metric", columns="Test", values="-log10(p)")

    fig = px.imshow(
        pivot_df,
        title="<b>Statistical Hypothesis Testing Results</b>",
        template=PLOTLY_TEMPLATE,
        color_continuous_scale="RdYlGn",
        aspect="auto",
        text_auto=".2f",
    )
    fig.update_layout(height=500, title_font_size=20)
    return fig


def create_regional_radar(
    df: pd.DataFrame, radar_metrics: list, region_col: str = "region"
) -> Optional[go.Figure]:
    """Create radar chart for regional financial metrics comparison."""
    if region_col not in df.columns:
        return None

    regions = df[region_col].dropna().unique().tolist()
    region_data = []

    for region in regions:
        region_df = df[df[region_col] == region]
        row = {"Region": region}
        for metric in radar_metrics:
            median_val = region_df[metric].median() if metric in region_df.columns else 0
            row[metric] = median_val if pd.notna(median_val) else 0
        region_data.append(row)

    radar_df = pd.DataFrame(region_data)

    # Normalize for radar chart
    for metric in radar_metrics:
        col_min, col_max = radar_df[metric].min(), radar_df[metric].max()
        if float(col_max) != float(col_min):
            radar_df[f"{metric}_norm"] = (radar_df[metric] - col_min) / (col_max - col_min)
        else:
            radar_df[f"{metric}_norm"] = 0.5

    fig = go.Figure()
    for idx, row in radar_df.iterrows():
        r_values = [row[f"{m}_norm"] for m in radar_metrics] + [row[f"{radar_metrics[0]}_norm"]]
        theta_values = [m.replace("_", " ").title() for m in radar_metrics] + [
            radar_metrics[0].replace("_", " ").title()
        ]
        fig.add_trace(
            go.Scatterpolar(
                r=r_values, theta=theta_values, fill="toself", name=row["Region"], opacity=0.6
            )
        )

    fig.update_layout(
        title="<b>Financial Metrics by Region</b>",
        template=PLOTLY_TEMPLATE,
        height=550,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    )
    return fig
