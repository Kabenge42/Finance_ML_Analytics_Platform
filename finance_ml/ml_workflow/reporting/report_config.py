"""Reporting configuration dataclasses for Phase 9.7 Reporting & Analytics.

This module defines the canonical configuration objects and constants used by
report generation code. Analytics shims import from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Constants
REPORT_TOP_N_DEFAULT: int = 50
"""Default number of top stocks to include in reports."""

QUALITY_THRESHOLD_DEFAULT: float = 0.5
"""Default quality score threshold for filtering stocks."""

RISK_ZSCORE_THRESHOLD: float = 2.0
"""Z-score threshold for identifying high-risk stocks."""

DISTRESS_SCORE_THRESHOLD: float = 0.7
"""Threshold for financial distress score alerts."""


@dataclass
class HTMLReportConfig:
    """Configuration for HTML report generation.

    Attributes:
        include_executive_summary: Whether to include executive summary section.
        include_sector_breakdown: Whether to include sector analysis section.
        include_quality_filtered: Whether to include quality-filtered rankings.
        include_risk_warnings: Whether to include risk warnings section.
        include_phase93_summary: Whether to include Phase 9.3 feature analysis.
        top_n_stocks: Number of top stocks to display in rankings.
        quality_threshold: Minimum quality score for filtered rankings.
        template: Report template style ("modern", "minimal", "detailed").
    """

    include_executive_summary: bool = True
    include_sector_breakdown: bool = True
    include_quality_filtered: bool = True
    include_risk_warnings: bool = True
    include_phase93_summary: bool = True
    top_n_stocks: int = 20
    quality_threshold: float = 0.5
    template: Literal["modern", "minimal", "detailed"] = "modern"


@dataclass
class ExcelReportConfig:
    """Configuration for Excel report generation.

    Attributes:
        include_executive_summary: Whether to include Executive_Summary sheet.
        include_quality_opportunities: Whether to include Quality_Opportunities sheet.
        include_sector_leaders: Whether to include Sector_Leaders sheet.
        include_risk_assessment: Whether to include Risk_Assessment sheet.
        include_phase93_analysis: Whether to include Phase93_Analysis sheet.
        top_n_per_sector: Number of top stocks per sector for leaders/laggards.
        quality_threshold: Minimum quality score for filtered opportunities.
        embed_visualizations: Whether to embed visualization PNGs.
    """

    include_executive_summary: bool = True
    include_quality_opportunities: bool = True
    include_sector_leaders: bool = True
    include_risk_assessment: bool = True
    include_phase93_analysis: bool = True
    top_n_per_sector: int = 5
    quality_threshold: float = 0.5
    embed_visualizations: bool = True


__all__ = [
    "HTMLReportConfig",
    "ExcelReportConfig",
    "REPORT_TOP_N_DEFAULT",
    "QUALITY_THRESHOLD_DEFAULT",
    "RISK_ZSCORE_THRESHOLD",
    "DISTRESS_SCORE_THRESHOLD",
]
