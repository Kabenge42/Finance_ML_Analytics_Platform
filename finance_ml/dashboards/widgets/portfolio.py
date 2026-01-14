"""portfolio.py - Dashboard widgets for portfolio analytics.

All scorecard creation functions follow a consistent pattern:
- Accept `output_path` as Optional[Union[str, Path]]
- Return pd.DataFrame with scores and grades
- Caller can pass either a directory OR a complete file path
- Each function has a well-defined default filename (CSV for scorecards)

See Section 20 of code_guidelines.md for output artifact standards.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from finance_ml.core.constants import PLOTLY_TEMPLATE, COLOR_PALETTE
from finance_ml.core.schema import (
    PHASE93_FEATURE_CATEGORIES,
    COLUMN_SCHEMA,
)
from .base import _write_html_artifact

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS (Section 2 Guidelines)
# =============================================================================

# Default filenames for each dashboard type (Section 20 - Output Artifact Standards)
DEFAULT_FILENAMES = {
    "dividend_reliability": "dividend_reliability_scorecard.csv",
    "dividend_sustainability": "dividend_sustainability_scorecard.csv",
    "analyst_recommendation": "analyst_recommendation_heatmap.html",
    "analyst_consensus": "analyst_consensus_scorecard.csv",
    "leverage_liquidity": "leverage_liquidity_scorecard.csv",
    "market_movers": "market_movers_dashboard.html",
    "price_target": "price_target_scorecard.csv",
    "employee_productivity": "employee_productivity_scorecard.csv",
    "earnings_quality": "earnings_quality_scorecard.csv",
    "revenue_forecast": "revenue_forecast_scorecard.csv",
    "cash_flow_quality": "cash_flow_quality_scorecard.csv",
    "momentum_technical": "momentum_technical_scorecard.csv",
    "valuation_timeseries": "valuation_timeseries_scorecard.csv",
}

# =============================================================================
# SCORECARD THRESHOLDS (A-F Grade Distribution)
# =============================================================================

# Grade thresholds: A >= 80, B >= 65, C >= 50, D >= 35, F < 35
GRADE_THRESHOLDS = {
    "A": 80,
    "B": 65,
    "C": 50,
    "D": 35,
}

# Single unified threshold config for all scorecards
SCORECARD_QUALITY_THRESHOLDS = {
    "excellent": 80,
    "good": 65,
    "moderate": 50,
    "poor": 35,
}

# =============================================================================
# SCHEMA-DRIVEN COLUMN MAPPINGS
# =============================================================================


def _build_column_mapping_from_schema(
    category: str,
    role_filter: Optional[str] = None,
) -> Dict[str, str]:
    """Build column mapping from PHASE93_FEATURE_CATEGORIES.

    Args:
        category: Feature category name from schema
        role_filter: Optional role filter (e.g., 'ratio', 'percentage')

    Returns:
        Dict mapping short names to full column names
    """
    features = PHASE93_FEATURE_CATEGORIES.get(category, [])

    if role_filter:
        features = [
            f for f in features if COLUMN_SCHEMA.get(f, {}).get("role") == role_filter
        ]

    # Create short name mapping (remove common suffixes)
    mapping = {}
    for feature in features:
        short_name = (
            feature.replace("_score", "").replace("_ratio", "").replace("_pct", "")
        )
        mapping[short_name] = feature

    return mapping


# Schema-aligned mappings - derive from PHASE93_FEATURE_CATEGORIES
DIVIDEND_RELIABILITY_COLS = _build_column_mapping_from_schema("Dividend Reliability")
LEVERAGE_LIQUIDITY_COLS = _build_column_mapping_from_schema("Leverage & Liquidity")
EMPLOYEE_PRODUCTIVITY_COLS = _build_column_mapping_from_schema("Employee Productivity")
ANALYST_CONSENSUS_COLS = _build_column_mapping_from_schema("Analyst Sentiment")
EARNINGS_QUALITY_COLS = _build_column_mapping_from_schema("Earnings Quality")

# For backward compatibility, maintain explicit mappings where needed
DIVIDEND_RELIABILITY_COLS.update(
    {
        "reliability": "dividend_reliability_score",
        "streak": "dividend_streak",
        "yield_stability": "dividend_yield_stability",
        "fcf_coverage": "fcf_dividend_coverage",
        "payout_consistency": "payout_consistency_score",
        "sustainable_flag": "sustainable_dividend_flag",
        "payout_ratio": "dividend_payout_ratio",
        "div_growth_3y": "dividend_growth_3y",
        "div_growth_5y": "dividend_growth_5y",
    }
)

# Maintain specific UI mappings
MARKET_MOVERS_COLS = {
    "1D Change": "one_day_pct",
    "1M Change": "price_chg_pct_1m",
    "3M Change": "price_chg_pct_3m",
}

ANALYST_RATING_COLS = {
    "Strong Buy": "num_strong_buys_ratings",
    "Buy": "num_buys_ratings",
    "Hold": "num_hold_ratings",
    "Sell": "num_sell_ratings",
    "Strong Sell": "num_strong_sell_ratings",
}

REVENUE_FORECAST_COLS = {
    "revenue_growth": "revenue_growth",
    "revenue_est_revision": "revenues_est_avg_ntm",
    "revenue_actual": "total_revenues_ltm",
    "revenue_surprise": "revenue_beat_indicator",
    "eps_est_revision_1m": "eps_est_avg_rev_pct_fy1e_1m",
    "eps_est_revision_3m": "eps_est_avg_rev_pct_fy1e_3m",
}

PRICE_TARGET_COLS = {
    "price_target": "price_target",
    "price_target_high": "price_target_high",
    "price_target_low": "price_target_low",
    "price_target_median": "price_target_median",
    "target_vs_price": "target_vs_price",
    "last_price": "last_price",
    "analyst_rating": "analyst_rating",
    "price_target_num": "price_target_num",
}


# =============================================================================
# PRIVATE HELPER FUNCTIONS
# =============================================================================


def _get_available_columns(
    df: pd.DataFrame,
    column_mapping: Dict[str, str],
) -> Dict[str, str]:
    """Filter column mapping to only include columns present in DataFrame."""
    return {k: v for k, v in column_mapping.items() if v in df.columns}


def _get_phase93_category_columns(
    df: pd.DataFrame,
    category: str,
) -> List[str]:
    """Get available columns from a Phase 9.3 feature category."""
    category_cols = PHASE93_FEATURE_CATEGORIES.get(category, [])
    return [col for col in category_cols if col in df.columns]


def _prepare_numeric_columns(
    df: pd.DataFrame,
    columns: Dict[str, str],
) -> pd.DataFrame:
    """Copy DataFrame and convert specified columns to numeric."""
    df_local = df.copy()
    for col in columns.values():
        if col in df_local.columns:
            df_local[col] = pd.to_numeric(df_local[col], errors="coerce")
    return df_local


def _assign_letter_grade(score: float) -> str:
    """Assign letter grade A-F based on score (0-100 scale)."""
    if pd.isna(score):
        return "F"
    if score >= GRADE_THRESHOLDS["A"]:
        return "A"
    elif score >= GRADE_THRESHOLDS["B"]:
        return "B"
    elif score >= GRADE_THRESHOLDS["C"]:
        return "C"
    elif score >= GRADE_THRESHOLDS["D"]:
        return "D"
    return "F"


def _save_scorecard_csv(
    scorecard: pd.DataFrame,
    output_path: Optional[Union[str, Path]],
    default_filename: str,
) -> None:
    """Save scorecard DataFrame to CSV."""
    if output_path is not None:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / default_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scorecard.to_csv(output_path, index=False)
        logger.info(f"Saved scorecard to {output_path}")


def _get_top_sectors(df: pd.DataFrame, top_n: int) -> pd.Index:
    """Get top N sectors by count from DataFrame."""
    return df["sector"].value_counts().head(int(top_n)).index


def _normalize_to_score(
    series: pd.Series,
    clip_min: float,
    clip_max: float,
    invert: bool = False,
) -> pd.Series:
    """Normalize a series to 0-100 score with optional inversion."""
    clipped = series.clip(clip_min, clip_max)
    normalized = (clipped - clip_min) / (clip_max - clip_min)
    if invert:
        normalized = 1 - normalized
    return (normalized * 100).fillna(50)


def _percentile_score(series: pd.Series) -> pd.Series:
    """Convert values to percentile-based scores (0-100)."""
    return series.rank(pct=True, na_option="keep") * 100


# =============================================================================
# PUBLIC SCORECARD FUNCTIONS
# =============================================================================


def create_dividend_sustainability_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate dividend sustainability scorecard with A-F grades.

    Uses percentile-based scoring for better grade distribution.
    Evaluates dividend sustainability using multiple factors:
    - Payout Ratio Analysis (lower is more sustainable)
    - FCF Coverage (higher coverage = more sustainable)
    - Dividend Streak (consistency indicator)
    - Dividend Growth (positive growth = healthy)

    Args:
        df: DataFrame with dividend-paying stocks.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with sustainability scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["dividend_sustainability"]

    # Initialize scorecard with base columns
    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()

    sustainability_cols = {
        "payout_ratio": "dividend_payout_ratio",
        "fcf_coverage": "fcf_dividend_coverage",
        "streak": "dividend_streak",
        "div_growth": "dividend_growth_rate",
        "div_growth_5y": "dividend_growth_5y",
        "debt_to_equity": "debt_to_equity",
        "debt_to_assets": "debt_to_assets",
        "reliability": "dividend_reliability_score",
    }

    available = _get_available_columns(df, sustainability_cols)

    # --- Payout Score (0-100, lower payout ratio = higher score) ---
    if "payout_ratio" in available:
        payout = pd.to_numeric(df[available["payout_ratio"]], errors="coerce")
        scorecard["payout_score"] = _normalize_to_score(payout, 0, 2, invert=True)
    else:
        scorecard["payout_score"] = 50.0

    # --- FCF Coverage Score (0-100, higher coverage = higher score) ---
    if "fcf_coverage" in available:
        fcf = pd.to_numeric(df[available["fcf_coverage"]], errors="coerce")
        scorecard["fcf_coverage_score"] = _normalize_to_score(fcf, 0, 5, invert=False)
    else:
        scorecard["fcf_coverage_score"] = 50.0

    # --- Dividend Growth Score (0-100, positive growth = higher score) ---
    div_growth_col = available.get("div_growth") or available.get("div_growth_5y")
    if div_growth_col:
        growth = pd.to_numeric(df[div_growth_col], errors="coerce")
        scorecard["div_growth_score"] = _normalize_to_score(
            growth, -0.2, 0.3, invert=False
        )
    else:
        scorecard["div_growth_score"] = 50.0

    # --- Balance Sheet Score (0-100, lower debt = higher score) ---
    debt_col = available.get("debt_to_equity") or available.get("debt_to_assets")
    if debt_col:
        debt = pd.to_numeric(df[debt_col], errors="coerce")
        scorecard["balance_sheet_score"] = _normalize_to_score(debt, 0, 3, invert=True)
    else:
        scorecard["balance_sheet_score"] = 50.0

    # --- Streak Score (0-100, longer streak = higher score) ---
    if "streak" in available:
        streak = pd.to_numeric(df[available["streak"]], errors="coerce")
        scorecard["streak_score"] = _normalize_to_score(streak, 0, 25, invert=False)
    else:
        scorecard["streak_score"] = 50.0

    # --- Composite Sustainability Score (weighted average) ---
    weights = {
        "payout_score": 0.25,
        "fcf_coverage_score": 0.25,
        "div_growth_score": 0.20,
        "balance_sheet_score": 0.15,
        "streak_score": 0.15,
    }

    scorecard["dividend_sustainability_score"] = (
        scorecard["payout_score"] * weights["payout_score"]
        + scorecard["fcf_coverage_score"] * weights["fcf_coverage_score"]
        + scorecard["div_growth_score"] * weights["div_growth_score"]
        + scorecard["balance_sheet_score"] * weights["balance_sheet_score"]
        + scorecard["streak_score"] * weights["streak_score"]
    ).round(1)

    # --- Assign Letter Grades (A-F) ---
    scorecard["sustainability_grade"] = scorecard[
        "dividend_sustainability_score"
    ].apply(_assign_letter_grade)

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_leverage_liquidity_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate leverage and liquidity scorecard with A-F grades.

    Uses columns from PHASE93_FEATURE_CATEGORIES["Leverage & Liquidity"]:
    - cash_ratio, current_ratio, quick_ratio (higher = better)
    - debt_to_assets, debt_to_equity (lower = better)
    - interest_coverage (higher = better)

    Args:
        df: DataFrame with leverage/liquidity metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with leverage/liquidity scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["leverage_liquidity"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, LEVERAGE_LIQUIDITY_COLS)

    # --- Liquidity Score (current ratio, quick ratio, cash ratio) ---
    liquidity_components = []

    if "current_ratio" in available:
        current = pd.to_numeric(df[available["current_ratio"]], errors="coerce")
        scorecard["current_ratio_score"] = _normalize_to_score(
            current, 0.5, 3, invert=False
        )
        liquidity_components.append(scorecard["current_ratio_score"])
    else:
        scorecard["current_ratio_score"] = 50.0

    if "quick_ratio" in available:
        quick = pd.to_numeric(df[available["quick_ratio"]], errors="coerce")
        scorecard["quick_ratio_score"] = _normalize_to_score(
            quick, 0.3, 2, invert=False
        )
        liquidity_components.append(scorecard["quick_ratio_score"])
    else:
        scorecard["quick_ratio_score"] = 50.0

    if "cash_ratio" in available:
        cash = pd.to_numeric(df[available["cash_ratio"]], errors="coerce")
        scorecard["cash_ratio_score"] = _normalize_to_score(cash, 0, 1, invert=False)
        liquidity_components.append(scorecard["cash_ratio_score"])
    else:
        scorecard["cash_ratio_score"] = 50.0

    # --- Leverage Score (debt ratios - lower is better) ---
    leverage_components = []

    if "debt_to_equity" in available:
        dte = pd.to_numeric(df[available["debt_to_equity"]], errors="coerce")
        scorecard["debt_equity_score"] = _normalize_to_score(dte, 0, 3, invert=True)
        leverage_components.append(scorecard["debt_equity_score"])
    else:
        scorecard["debt_equity_score"] = 50.0

    if "debt_to_assets" in available:
        dta = pd.to_numeric(df[available["debt_to_assets"]], errors="coerce")
        scorecard["debt_assets_score"] = _normalize_to_score(dta, 0, 0.8, invert=True)
        leverage_components.append(scorecard["debt_assets_score"])
    else:
        scorecard["debt_assets_score"] = 50.0

    # --- Interest Coverage Score (higher is better) ---
    if "interest_coverage" in available:
        ic = pd.to_numeric(df[available["interest_coverage"]], errors="coerce")
        scorecard["interest_coverage_score"] = _normalize_to_score(
            ic, 0, 15, invert=False
        )
    else:
        scorecard["interest_coverage_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "current_ratio_score": 0.15,
        "quick_ratio_score": 0.15,
        "cash_ratio_score": 0.10,
        "debt_equity_score": 0.20,
        "debt_assets_score": 0.20,
        "interest_coverage_score": 0.20,
    }

    scorecard["leverage_liquidity_score"] = (
        scorecard["current_ratio_score"] * weights["current_ratio_score"]
        + scorecard["quick_ratio_score"] * weights["quick_ratio_score"]
        + scorecard["cash_ratio_score"] * weights["cash_ratio_score"]
        + scorecard["debt_equity_score"] * weights["debt_equity_score"]
        + scorecard["debt_assets_score"] * weights["debt_assets_score"]
        + scorecard["interest_coverage_score"] * weights["interest_coverage_score"]
    ).round(1)

    scorecard["leverage_grade"] = scorecard["leverage_liquidity_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_employee_productivity_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate employee productivity scorecard with A-F grades.

    Analyzes workforce efficiency metrics from PHASE93_FEATURE_CATEGORIES:
    - Revenue per employee, Profit per employee (higher = better)
    - Employee growth trends (moderate positive = best)
    - Hiring intensity and productivity ratios

    Args:
        df: DataFrame with employee productivity metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with productivity scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["employee_productivity"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, EMPLOYEE_PRODUCTIVITY_COLS)

    # --- Revenue per Employee Score (percentile-based) ---
    if "revenue_per_employee" in available:
        rev_emp = pd.to_numeric(df[available["revenue_per_employee"]], errors="coerce")
        scorecard["revenue_per_emp_score"] = _percentile_score(rev_emp)
    else:
        scorecard["revenue_per_emp_score"] = 50.0

    # --- Profit per Employee Score (percentile-based) ---
    if "profit_per_employee" in available:
        prof_emp = pd.to_numeric(df[available["profit_per_employee"]], errors="coerce")
        scorecard["profit_per_emp_score"] = _percentile_score(prof_emp)
    else:
        scorecard["profit_per_emp_score"] = 50.0

    # --- EBITDA per Employee Score ---
    if "ebitda_per_employee" in available:
        ebitda_emp = pd.to_numeric(
            df[available["ebitda_per_employee"]], errors="coerce"
        )
        scorecard["ebitda_per_emp_score"] = _percentile_score(ebitda_emp)
    else:
        scorecard["ebitda_per_emp_score"] = 50.0

    # --- Employee Growth Score (moderate growth = best) ---
    if "employee_growth_yoy" in available:
        growth = pd.to_numeric(df[available["employee_growth_yoy"]], errors="coerce")
        # Optimal growth is 5-15%, penalize extremes
        optimal_growth = 0.10
        deviation = np.abs(growth - optimal_growth)
        scorecard["emp_growth_score"] = _normalize_to_score(
            deviation, 0, 0.5, invert=True
        )
    else:
        scorecard["emp_growth_score"] = 50.0

    # --- Hiring Intensity Score ---
    if "hiring_intensity" in available:
        hiring = pd.to_numeric(df[available["hiring_intensity"]], errors="coerce")
        scorecard["hiring_intensity_score"] = hiring.clip(0, 100).fillna(50)
    else:
        scorecard["hiring_intensity_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "revenue_per_emp_score": 0.30,
        "profit_per_emp_score": 0.25,
        "ebitda_per_emp_score": 0.20,
        "emp_growth_score": 0.15,
        "hiring_intensity_score": 0.10,
    }

    scorecard["employee_productivity_score"] = (
        scorecard["revenue_per_emp_score"] * weights["revenue_per_emp_score"]
        + scorecard["profit_per_emp_score"] * weights["profit_per_emp_score"]
        + scorecard["ebitda_per_emp_score"] * weights["ebitda_per_emp_score"]
        + scorecard["emp_growth_score"] * weights["emp_growth_score"]
        + scorecard["hiring_intensity_score"] * weights["hiring_intensity_score"]
    ).round(1)

    scorecard["productivity_grade"] = scorecard["employee_productivity_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_analyst_consensus_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate analyst consensus scorecard with A-F grades.

    Evaluates analyst sentiment using multiple factors:
    - Analyst Rating (1-5 scale, higher = more bullish)
    - Price Target Upside (higher = more upside)
    - Consensus Strength (agreement among analysts)
    - Price Target Momentum (positive revisions = bullish)

    Args:
        df: DataFrame with analyst consensus data.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with consensus scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["analyst_consensus"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, ANALYST_CONSENSUS_COLS)

    # --- Analyst Rating Score (1-5 scale to 0-100) ---
    if "analyst_rating" in available:
        rating = pd.to_numeric(df[available["analyst_rating"]], errors="coerce")
        scorecard["rating_score"] = _normalize_to_score(rating, 1, 5, invert=False)
    else:
        scorecard["rating_score"] = 50.0

    # --- Target Upside Score ---
    if "target_vs_price" in available:
        upside = pd.to_numeric(df[available["target_vs_price"]], errors="coerce")
        scorecard["upside_score"] = _normalize_to_score(upside, -30, 50, invert=False)
    else:
        scorecard["upside_score"] = 50.0

    # --- Consensus Strength Score ---
    if "consensus_strength" in available:
        strength = pd.to_numeric(df[available["consensus_strength"]], errors="coerce")
        scorecard["consensus_strength_score"] = strength.clip(0, 100).fillna(50)
    else:
        scorecard["consensus_strength_score"] = 50.0

    # --- Analyst Conviction Score ---
    if "analyst_conviction" in available:
        conviction = pd.to_numeric(df[available["analyst_conviction"]], errors="coerce")
        scorecard["conviction_score"] = conviction.clip(0, 100).fillna(50)
    else:
        scorecard["conviction_score"] = 50.0

    # --- PT Momentum Score (revisions) ---
    pt_momentum_col = available.get("pt_momentum_1m") or available.get(
        "price_target_revision"
    )
    if pt_momentum_col:
        momentum = pd.to_numeric(df[pt_momentum_col], errors="coerce")
        scorecard["pt_momentum_score"] = _normalize_to_score(
            momentum, -20, 20, invert=False
        )
    else:
        scorecard["pt_momentum_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "rating_score": 0.30,
        "upside_score": 0.25,
        "consensus_strength_score": 0.20,
        "conviction_score": 0.15,
        "pt_momentum_score": 0.10,
    }

    scorecard["analyst_consensus_score"] = (
        scorecard["rating_score"] * weights["rating_score"]
        + scorecard["upside_score"] * weights["upside_score"]
        + scorecard["consensus_strength_score"] * weights["consensus_strength_score"]
        + scorecard["conviction_score"] * weights["conviction_score"]
        + scorecard["pt_momentum_score"] * weights["pt_momentum_score"]
    ).round(1)

    scorecard["consensus_grade"] = scorecard["analyst_consensus_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_earnings_quality_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate earnings quality scorecard with A-F grades.

    Analyzes earnings quality using multiple factors:
    - Piotroski F-Score (0-9, higher = better)
    - Altman Z-Score (>2.99 = safe, <1.81 = distress)
    - Earnings Surprise (positive = beat expectations)
    - Accounting Quality indicators

    Args:
        df: DataFrame with earnings quality metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with quality scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["earnings_quality"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, EARNINGS_QUALITY_COLS)

    # --- Piotroski F-Score (0-9 to 0-100) ---
    if "piotroski_f_score" in available:
        f_score = pd.to_numeric(df[available["piotroski_f_score"]], errors="coerce")
        scorecard["piotroski_score"] = _normalize_to_score(f_score, 0, 9, invert=False)
    else:
        scorecard["piotroski_score"] = 50.0

    # --- Altman Z-Score (bankruptcy risk) ---
    if "altman_z_score" in available:
        z_score = pd.to_numeric(df[available["altman_z_score"]], errors="coerce")
        # <1.81 = distress (0), 1.81-2.99 = gray (50), >2.99 = safe (100)
        scorecard["altman_score"] = _normalize_to_score(z_score, 0, 5, invert=False)
    else:
        scorecard["altman_score"] = 50.0

    # --- Beneish M-Score (accounting manipulation, lower = better) ---
    if "beneish_m_score" in available:
        m_score = pd.to_numeric(df[available["beneish_m_score"]], errors="coerce")
        # M-score > -2.22 suggests manipulation
        scorecard["beneish_score"] = _normalize_to_score(m_score, -4, -1, invert=True)
    else:
        scorecard["beneish_score"] = 50.0

    # --- Earnings Surprise Score ---
    if "eps_surprise_pct" in available:
        surprise = pd.to_numeric(df[available["eps_surprise_pct"]], errors="coerce")
        scorecard["surprise_score"] = _normalize_to_score(
            surprise, -20, 30, invert=False
        )
    else:
        scorecard["surprise_score"] = 50.0

    # --- Earnings Beat Indicator ---
    if "earnings_beat_indicator" in available:
        beat = pd.to_numeric(df[available["earnings_beat_indicator"]], errors="coerce")
        scorecard["beat_score"] = (beat * 100).fillna(50)
    else:
        scorecard["beat_score"] = 50.0

    # --- Adjustment Consistency Score ---
    if "adjustment_consistency_score" in available:
        adj = pd.to_numeric(
            df[available["adjustment_consistency_score"]], errors="coerce"
        )
        scorecard["adjustment_score"] = adj.clip(0, 100).fillna(50)
    else:
        scorecard["adjustment_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "piotroski_score": 0.25,
        "altman_score": 0.20,
        "beneish_score": 0.15,
        "surprise_score": 0.15,
        "beat_score": 0.15,
        "adjustment_score": 0.10,
    }

    scorecard["earnings_quality_score"] = (
        scorecard["piotroski_score"] * weights["piotroski_score"]
        + scorecard["altman_score"] * weights["altman_score"]
        + scorecard["beneish_score"] * weights["beneish_score"]
        + scorecard["surprise_score"] * weights["surprise_score"]
        + scorecard["beat_score"] * weights["beat_score"]
        + scorecard["adjustment_score"] * weights["adjustment_score"]
    ).round(1)

    scorecard["quality_grade"] = scorecard["earnings_quality_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_revenue_forecast_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate revenue forecast momentum scorecard with A-F grades.

    Analyzes revenue forecasting trends:
    - Revenue Growth (higher = better)
    - Estimate Revisions (positive = bullish)
    - Revenue Surprise (beat = positive)

    Args:
        df: DataFrame with revenue forecast data.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with momentum scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["revenue_forecast"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, REVENUE_FORECAST_COLS)

    # --- Revenue Growth Score ---
    if "revenue_growth" in available:
        growth = pd.to_numeric(df[available["revenue_growth"]], errors="coerce")
        scorecard["revenue_growth_score"] = _normalize_to_score(
            growth, -0.2, 0.4, invert=False
        )
    else:
        scorecard["revenue_growth_score"] = 50.0

    # --- EPS Estimate Revision Score (1M) ---
    if "eps_est_revision_1m" in available:
        rev_1m = pd.to_numeric(df[available["eps_est_revision_1m"]], errors="coerce")
        scorecard["est_revision_1m_score"] = _normalize_to_score(
            rev_1m, -10, 10, invert=False
        )
    else:
        scorecard["est_revision_1m_score"] = 50.0

    # --- EPS Estimate Revision Score (3M) ---
    if "eps_est_revision_3m" in available:
        rev_3m = pd.to_numeric(df[available["eps_est_revision_3m"]], errors="coerce")
        scorecard["est_revision_3m_score"] = _normalize_to_score(
            rev_3m, -15, 15, invert=False
        )
    else:
        scorecard["est_revision_3m_score"] = 50.0

    # --- Revenue Beat Score ---
    if "revenue_surprise" in available:
        beat = pd.to_numeric(df[available["revenue_surprise"]], errors="coerce")
        scorecard["revenue_beat_score"] = (beat * 100).fillna(50)
    else:
        scorecard["revenue_beat_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "revenue_growth_score": 0.35,
        "est_revision_1m_score": 0.25,
        "est_revision_3m_score": 0.20,
        "revenue_beat_score": 0.20,
    }

    scorecard["revenue_momentum_score"] = (
        scorecard["revenue_growth_score"] * weights["revenue_growth_score"]
        + scorecard["est_revision_1m_score"] * weights["est_revision_1m_score"]
        + scorecard["est_revision_3m_score"] * weights["est_revision_3m_score"]
        + scorecard["revenue_beat_score"] * weights["revenue_beat_score"]
    ).round(1)

    scorecard["momentum_grade"] = scorecard["revenue_momentum_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_price_target_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate price target analytics scorecard with A-F grades.

    Evaluates price target quality and upside potential:
    - Target Upside (higher = more potential)
    - Consensus Spread (tighter = more confident)
    - Analyst Coverage (more analysts = more reliable)
    - Target vs Historical Range

    Args:
        df: DataFrame with price target columns.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with price target scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["price_target"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, PRICE_TARGET_COLS)

    # --- Target Upside Score ---
    if "target_vs_price" in available:
        upside = pd.to_numeric(df[available["target_vs_price"]], errors="coerce")
        scorecard["upside_score"] = _normalize_to_score(upside, -30, 60, invert=False)
    elif "price_target" in available and "last_price" in available:
        pt = pd.to_numeric(df[available["price_target"]], errors="coerce")
        lp = pd.to_numeric(df[available["last_price"]], errors="coerce")
        upside = ((pt - lp) / lp * 100).replace([np.inf, -np.inf], np.nan)
        scorecard["upside_score"] = _normalize_to_score(upside, -30, 60, invert=False)
    else:
        scorecard["upside_score"] = 50.0

    # --- Consensus Spread Score (tighter = better, so inverted) ---
    if (
        "price_target_high" in available
        and "price_target_low" in available
        and "last_price" in available
    ):
        pt_high = pd.to_numeric(df[available["price_target_high"]], errors="coerce")
        pt_low = pd.to_numeric(df[available["price_target_low"]], errors="coerce")
        lp = pd.to_numeric(df[available["last_price"]], errors="coerce")
        spread_pct = ((pt_high - pt_low) / lp * 100).replace([np.inf, -np.inf], np.nan)
        scorecard["spread_score"] = _normalize_to_score(spread_pct, 0, 100, invert=True)
    else:
        scorecard["spread_score"] = 50.0

    # --- Analyst Coverage Score (more analysts = more reliable) ---
    if "price_target_num" in available:
        coverage = pd.to_numeric(df[available["price_target_num"]], errors="coerce")
        scorecard["coverage_score"] = _normalize_to_score(coverage, 0, 30, invert=False)
    else:
        scorecard["coverage_score"] = 50.0

    # --- Analyst Rating Score ---
    if "analyst_rating" in available:
        rating = pd.to_numeric(df[available["analyst_rating"]], errors="coerce")
        scorecard["rating_score"] = _normalize_to_score(rating, 1, 5, invert=False)
    else:
        scorecard["rating_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "upside_score": 0.35,
        "spread_score": 0.25,
        "coverage_score": 0.20,
        "rating_score": 0.20,
    }

    scorecard["price_target_score"] = (
        scorecard["upside_score"] * weights["upside_score"]
        + scorecard["spread_score"] * weights["spread_score"]
        + scorecard["coverage_score"] * weights["coverage_score"]
        + scorecard["rating_score"] * weights["rating_score"]
    ).round(1)

    scorecard["price_target_grade"] = scorecard["price_target_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_dividend_reliability_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate dividend reliability scorecard with A-F grades.

    Leverages dividend reliability metrics from PHASE93_FEATURE_CATEGORIES:
    - dividend_reliability_score (0-100)
    - dividend_streak (years of consecutive dividends)
    - dividend_yield_stability
    - fcf_dividend_coverage

    Args:
        df: DataFrame with dividend metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with reliability scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["dividend_reliability"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()
    available = _get_available_columns(df, DIVIDEND_RELIABILITY_COLS)

    # --- Reliability Score ---
    if "reliability" in available:
        reliability = pd.to_numeric(df[available["reliability"]], errors="coerce")
        scorecard["reliability_score"] = reliability.clip(0, 100).fillna(50)
    else:
        scorecard["reliability_score"] = 50.0

    # --- Streak Score ---
    if "streak" in available:
        streak = pd.to_numeric(df[available["streak"]], errors="coerce")
        scorecard["streak_score"] = _normalize_to_score(streak, 0, 25, invert=False)
    else:
        scorecard["streak_score"] = 50.0

    # --- Yield Stability Score ---
    if "yield_stability" in available:
        stability = pd.to_numeric(df[available["yield_stability"]], errors="coerce")
        scorecard["yield_stability_score"] = stability.clip(0, 100).fillna(50)
    else:
        scorecard["yield_stability_score"] = 50.0

    # --- FCF Coverage Score ---
    if "fcf_coverage" in available:
        fcf = pd.to_numeric(df[available["fcf_coverage"]], errors="coerce")
        scorecard["fcf_coverage_score"] = _normalize_to_score(fcf, 0, 5, invert=False)
    else:
        scorecard["fcf_coverage_score"] = 50.0

    # --- Payout Consistency Score ---
    if "payout_consistency" in available:
        consistency = pd.to_numeric(
            df[available["payout_consistency"]], errors="coerce"
        )
        scorecard["payout_consistency_score"] = consistency.clip(0, 100).fillna(50)
    else:
        scorecard["payout_consistency_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "reliability_score": 0.30,
        "streak_score": 0.25,
        "yield_stability_score": 0.15,
        "fcf_coverage_score": 0.20,
        "payout_consistency_score": 0.10,
    }

    scorecard["dividend_reliability_score"] = (
        scorecard["reliability_score"] * weights["reliability_score"]
        + scorecard["streak_score"] * weights["streak_score"]
        + scorecard["yield_stability_score"] * weights["yield_stability_score"]
        + scorecard["fcf_coverage_score"] * weights["fcf_coverage_score"]
        + scorecard["payout_consistency_score"] * weights["payout_consistency_score"]
    ).round(1)

    scorecard["reliability_grade"] = scorecard["dividend_reliability_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


# =============================================================================
# VISUALIZATION-ONLY FUNCTIONS (Return go.Figure)
# =============================================================================


def create_cash_flow_quality_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate cash flow quality scorecard with A-F grades.

    Leverages Phase 9.3 v1.15 Cash Flow category (17 features):
    - CFO/FCF stability and trends
    - Cash conversion efficiency
    - Investment intensity patterns
    - Acquisition activity trends

    Args:
        df: DataFrame with cash flow metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with cash flow scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["cash_flow_quality"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()

    # Get Cash Flow features from schema
    cash_flow_features = PHASE93_FEATURE_CATEGORIES.get("Cash Flow", [])
    available = {f: f for f in cash_flow_features if f in df.columns}

    # --- FCF Stability Score ---
    if "fcf_stability" in available:
        fcf_stab = pd.to_numeric(df["fcf_stability"], errors="coerce")
        scorecard["fcf_stability_score"] = fcf_stab.clip(0, 100).fillna(50)
    elif "fcf_quarterly_volatility" in available:
        # Invert volatility - lower volatility = higher stability
        vol = pd.to_numeric(df["fcf_quarterly_volatility"], errors="coerce")
        scorecard["fcf_stability_score"] = _normalize_to_score(vol, 0, 1, invert=True)
    else:
        scorecard["fcf_stability_score"] = 50.0

    # --- CFO Trend Score ---
    if "cfo_5y_trend" in available:
        trend = pd.to_numeric(df["cfo_5y_trend"], errors="coerce")
        scorecard["cfo_trend_score"] = _normalize_to_score(
            trend, -0.5, 0.5, invert=False
        )
    elif "cfo_quarterly_trend" in available:
        trend = pd.to_numeric(df["cfo_quarterly_trend"], errors="coerce")
        scorecard["cfo_trend_score"] = _normalize_to_score(
            trend, -0.3, 0.3, invert=False
        )
    else:
        scorecard["cfo_trend_score"] = 50.0

    # --- FCF Margin Score ---
    if "fcf_margin" in available:
        margin = pd.to_numeric(df["fcf_margin"], errors="coerce")
        scorecard["fcf_margin_score"] = _normalize_to_score(
            margin, -0.1, 0.3, invert=False
        )
    else:
        scorecard["fcf_margin_score"] = 50.0

    # --- Cash Conversion Score ---
    if "cfo_to_net_income" in available:
        conversion = pd.to_numeric(df["cfo_to_net_income"], errors="coerce")
        # Good conversion is > 1 (CFO exceeds reported earnings)
        scorecard["cash_conversion_score"] = _normalize_to_score(
            conversion, 0.5, 1.5, invert=False
        )
    else:
        scorecard["cash_conversion_score"] = 50.0

    # --- FCF Positive Ratio Score ---
    if "fcf_positive_ratio" in available:
        pos_ratio = pd.to_numeric(df["fcf_positive_ratio"], errors="coerce")
        scorecard["fcf_consistency_score"] = (pos_ratio * 100).clip(0, 100).fillna(50)
    else:
        scorecard["fcf_consistency_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "fcf_stability_score": 0.25,
        "cfo_trend_score": 0.20,
        "fcf_margin_score": 0.20,
        "cash_conversion_score": 0.20,
        "fcf_consistency_score": 0.15,
    }

    scorecard["cash_flow_quality_score"] = (
        scorecard["fcf_stability_score"] * weights["fcf_stability_score"]
        + scorecard["cfo_trend_score"] * weights["cfo_trend_score"]
        + scorecard["fcf_margin_score"] * weights["fcf_margin_score"]
        + scorecard["cash_conversion_score"] * weights["cash_conversion_score"]
        + scorecard["fcf_consistency_score"] * weights["fcf_consistency_score"]
    ).round(1)

    scorecard["cash_flow_grade"] = scorecard["cash_flow_quality_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_momentum_technical_scorecard(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate momentum & technical analysis scorecard with A-F grades.

    Leverages Phase 9.3 v1.15 Momentum & Technical category (25 features):
    - Price momentum across timeframes
    - EMA crossover signals
    - 52-week range positioning
    - Return stability metrics

    Args:
        df: DataFrame with momentum/technical metrics.
        output_path: Optional path to save CSV output.

    Returns:
        DataFrame with momentum scores and grades.
    """
    default_filename = DEFAULT_FILENAMES["momentum_technical"]

    base_cols = ["ticker"]
    if "sector" in df.columns:
        base_cols.append("sector")
    if "region" in df.columns:
        base_cols.append("region")

    scorecard = df[base_cols].copy()

    momentum_features = PHASE93_FEATURE_CATEGORIES.get("Momentum & Technical", [])
    available = {f: f for f in momentum_features if f in df.columns}

    # --- Price Momentum Score (multi-timeframe composite) ---
    momentum_cols = ["price_momentum_1m", "price_momentum_3m", "price_momentum_6m"]
    available_mom = [c for c in momentum_cols if c in available]
    if available_mom:
        mom_scores = []
        for col in available_mom:
            mom = pd.to_numeric(df[col], errors="coerce")
            mom_scores.append(_normalize_to_score(mom, -30, 50, invert=False))
        scorecard["price_momentum_score"] = pd.concat(mom_scores, axis=1).mean(axis=1)
    else:
        scorecard["price_momentum_score"] = 50.0

    # --- Trend Consistency Score ---
    if "ema_trend_consistency" in available:
        trend = pd.to_numeric(df["ema_trend_consistency"], errors="coerce")
        scorecard["trend_score"] = _normalize_to_score(trend, -1, 1, invert=False)
    else:
        scorecard["trend_score"] = 50.0

    # --- 52W Range Position Score ---
    if "52w_range_position" in available:
        range_pos = pd.to_numeric(df["52w_range_position"], errors="coerce")
        # Higher position (closer to 52W high) is generally bullish
        scorecard["range_position_score"] = (range_pos * 100).clip(0, 100).fillna(50)
    else:
        scorecard["range_position_score"] = 50.0

    # --- Return Stability Score ---
    if "return_stability_score" in available:
        stability = pd.to_numeric(df["return_stability_score"], errors="coerce")
        scorecard["stability_score"] = stability.clip(0, 100).fillna(50)
    elif "sharpe_proxy" in available:
        sharpe = pd.to_numeric(df["sharpe_proxy"], errors="coerce")
        scorecard["stability_score"] = _normalize_to_score(sharpe, -1, 2, invert=False)
    else:
        scorecard["stability_score"] = 50.0

    # --- Breakout Signal Score ---
    if "breakout_signal" in available:
        breakout = pd.to_numeric(df["breakout_signal"], errors="coerce")
        scorecard["breakout_score"] = (breakout * 100).clip(0, 100).fillna(50)
    else:
        scorecard["breakout_score"] = 50.0

    # --- Composite Score ---
    weights = {
        "price_momentum_score": 0.30,
        "trend_score": 0.25,
        "range_position_score": 0.20,
        "stability_score": 0.15,
        "breakout_score": 0.10,
    }

    scorecard["momentum_score"] = (
        scorecard["price_momentum_score"] * weights["price_momentum_score"]
        + scorecard["trend_score"] * weights["trend_score"]
        + scorecard["range_position_score"] * weights["range_position_score"]
        + scorecard["stability_score"] * weights["stability_score"]
        + scorecard["breakout_score"] * weights["breakout_score"]
    ).round(1)

    scorecard["momentum_grade"] = scorecard["momentum_score"].apply(
        _assign_letter_grade
    )

    _save_scorecard_csv(scorecard, output_path, default_filename)
    return scorecard


def create_analyst_recommendation_heatmap(
    df: pd.DataFrame,
    top_n_sectors: int = 12,
    output_path: Optional[Union[str, Path]] = None,
) -> go.Figure:
    """Create a heatmap of analyst recommendations by sector.

    Args:
        df: DataFrame containing analyst rating count columns.
        top_n_sectors: Number of sectors to display.
        output_path: Optional path to save HTML (directory or file).

    Returns:
        go.Figure: Plotly figure.
    """
    default_filename = DEFAULT_FILENAMES["analyst_recommendation"]
    available_ratings = _get_available_columns(df, ANALYST_RATING_COLS)

    if not available_ratings or "sector" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Required analyst rating columns not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path, default_filename=default_filename)
        return fig

    df_local = df.copy()
    top_sectors = _get_top_sectors(df_local, top_n_sectors)

    heatmap_data: List[Dict[str, float]] = []
    for sector in top_sectors:
        sector_df = df_local[df_local["sector"] == sector]
        row: Dict[str, float] = {"Sector": str(sector)[:25]}
        for rating_name, col in available_ratings.items():
            row[rating_name] = float(
                pd.to_numeric(sector_df[col], errors="coerce").sum()
            )
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data).set_index("Sector")
    row_sums = heatmap_df.sum(axis=1).replace(0, np.nan)
    heatmap_normalized = heatmap_df.div(row_sums, axis=0) * 100
    heatmap_normalized = heatmap_normalized.fillna(0)

    fig = px.imshow(
        heatmap_normalized,
        labels=dict(x="Rating Type", y="Sector", color="% of Ratings"),
        x=list(available_ratings.keys()),
        y=heatmap_normalized.index.tolist(),
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=20,
        aspect="auto",
        text_auto=".1f",
        title="<b>Analyst Recommendation Distribution by Sector</b><br><sup>Percentage of Total Ratings per Sector</sup>",
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=600,
        font=dict(family="Arial, sans-serif", size=12),
        title_font_size=20,
        xaxis_title="Rating Type",
        yaxis_title="Sector",
    )

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


def create_market_movers_dashboard(
    df: pd.DataFrame,
    top_n: int = 10,
    time_period: str = "1D",
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> go.Figure:
    """Create dashboard showing top market movers (gainers and losers).

    Uses price change columns from COLUMN_SCHEMA with role='percentage':
    - one_day_pct (1-Day %)
    - price_chg_pct_1m (Price Chg. % 1M)
    - price_chg_pct_3m (Price Chg. % 3M)

    Args:
        df: DataFrame with price change metrics and ticker/sector columns.
        top_n: Number of top gainers/losers to display per panel.
        time_period: Time period for ranking - "1D", "1M", or "3M".
        output_path: Optional path to save HTML (directory or file).
        **kwargs: Additional arguments for backward compatibility.

    Returns:
        go.Figure: Two-panel dashboard with gainers (left) and losers (right).
    """
    default_filename = DEFAULT_FILENAMES["market_movers"]
    period_col_map = {
        "1D": "one_day_pct",
        "1M": "price_chg_pct_1m",
        "3M": "price_chg_pct_3m",
    }

    change_col = period_col_map.get(time_period.upper(), "one_day_pct")

    if change_col not in df.columns:
        if time_period.upper() == "1D" and "1_day_pct" in df.columns:
            change_col = "1_day_pct"
        else:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Required column '{change_col}' not found for {time_period} movers",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16),
            )
            fig.update_layout(template=PLOTLY_TEMPLATE)
            _write_html_artifact(fig, output_path, default_filename=default_filename)
            return fig

    ticker_col = "ticker" if "ticker" in df.columns else None
    if ticker_col is None:
        fig = go.Figure()
        fig.add_annotation(
            text="Required 'ticker' column not found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path, default_filename=default_filename)
        return fig

    df_local = df.copy()
    df_local[change_col] = pd.to_numeric(df_local[change_col], errors="coerce")
    df_valid = df_local.dropna(subset=[change_col, ticker_col])

    if len(df_valid) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Insufficient data for market movers",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(template=PLOTLY_TEMPLATE)
        _write_html_artifact(fig, output_path, default_filename=default_filename)
        return fig

    df_sorted = df_valid.sort_values(change_col, ascending=False)
    top_gainers = df_sorted.head(top_n).copy()
    top_losers = df_sorted.tail(top_n).sort_values(change_col, ascending=True).copy()

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Top {top_n} Gainers ({time_period})",
            f"Top {top_n} Losers ({time_period})",
        ],
        horizontal_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(
            x=top_gainers[change_col],
            y=top_gainers[ticker_col],
            orientation="h",
            marker_color=COLOR_PALETTE["success"],
            text=[f"{x:+.2f}%" for x in top_gainers[change_col]],
            textposition="outside",
            name="Gainers",
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=top_losers[change_col],
            y=top_losers[ticker_col],
            orientation="h",
            marker_color=COLOR_PALETTE["danger"],
            text=[f"{x:+.2f}%" for x in top_losers[change_col]],
            textposition="outside",
            name="Losers",
            hovertemplate="<b>%{y}</b><br>Change: %{x:+.2f}%<extra></extra>",
        ),
        row=1,
        col=2,
    )

    period_labels = {"1D": "1-Day", "1M": "1-Month", "3M": "3-Month"}
    period_label = period_labels.get(time_period.upper(), time_period)

    fig.update_layout(
        title=f"<b>Market Movers Dashboard</b><br><sup>{period_label} Price Changes</sup>",
        template=PLOTLY_TEMPLATE,
        height=max(400, top_n * 35 + 100),
        showlegend=False,
    )

    fig.update_xaxes(title_text="Change %", row=1, col=1)
    fig.update_xaxes(title_text="Change %", row=1, col=2)
    fig.update_yaxes(categoryorder="total ascending", row=1, col=1)
    fig.update_yaxes(categoryorder="total descending", row=1, col=2)

    _write_html_artifact(fig, output_path, default_filename=default_filename)
    return fig


# =============================================================================
# SCORECARD REGISTRY (Mirrors Feature Registry Pattern)
# =============================================================================

SCORECARD_REGISTRY = {
    "dividend_sustainability": {
        "function": create_dividend_sustainability_scorecard,
        "category": "Dividend Reliability",
        "output_type": "csv",
    },
    "dividend_reliability": {
        "function": create_dividend_reliability_scorecard,
        "category": "Dividend Reliability",
        "output_type": "csv",
    },
    "leverage_liquidity": {
        "function": create_leverage_liquidity_scorecard,
        "category": "Leverage & Liquidity",
        "output_type": "csv",
    },
    "employee_productivity": {
        "function": create_employee_productivity_scorecard,
        "category": "Employee Productivity",
        "output_type": "csv",
    },
    "analyst_consensus": {
        "function": create_analyst_consensus_scorecard,
        "category": "Analyst Sentiment",
        "output_type": "csv",
    },
    "earnings_quality": {
        "function": create_earnings_quality_scorecard,
        "category": "Earnings Quality",
        "output_type": "csv",
    },
    "revenue_forecast": {
        "function": create_revenue_forecast_scorecard,
        "category": "Revenue Forecasting",
        "output_type": "csv",
    },
    "price_target": {
        "function": create_price_target_scorecard,
        "category": "Analyst Sentiment",
        "output_type": "csv",
    },
    "cash_flow_quality": {
        "function": create_cash_flow_quality_scorecard,
        "category": "Cash Flow",
        "output_type": "csv",
    },
    "momentum_technical": {
        "function": create_momentum_technical_scorecard,
        "category": "Momentum & Technical",
        "output_type": "csv",
    },
    # Visualization-only entries
    "analyst_recommendation": {
        "function": create_analyst_recommendation_heatmap,
        "category": "Analyst Sentiment",
        "output_type": "html",
    },
    "market_movers": {
        "function": create_market_movers_dashboard,
        "category": "Market Sentiment",
        "output_type": "html",
    },
}


def get_scorecard_generators() -> Dict[str, Dict]:
    """Returns the registry of scorecard generation functions."""
    return SCORECARD_REGISTRY


def generate_all_scorecards(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    categories: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Generate all scorecards for specified categories.

    Args:
        df: DataFrame with financial data.
        output_dir: Directory for output files.
        categories: Optional list of categories to generate.
            If None, generates all scorecards.

    Returns:
        Dict mapping scorecard names to resulting DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, config in SCORECARD_REGISTRY.items():
        if categories and config["category"] not in categories:
            continue

        if config["output_type"] == "csv":
            try:
                scorecard = config["function"](df, output_path=output_dir)
                results[name] = scorecard
                logger.info(f"Generated {name} scorecard: {len(scorecard)} rows")
            except Exception as e:
                logger.warning(f"Failed to generate {name} scorecard: {e}")

    return results
