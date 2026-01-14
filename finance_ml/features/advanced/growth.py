"""Growth-related feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer growth metrics.

    Metrics computed:
    - Revenue CAGR (if multi-year data available)
    - EPS Growth %
    - EBITDA Growth %
    - Book Value Growth %

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with growth metrics added
    """
    result = df.copy()

    # Revenue Growth (YoY if available)
    if "revenue" in df.columns and "revenue_previous_year" in df.columns:
        result["revenue_growth_yoy"] = (
            _safe_div(
                (df["revenue"] - df["revenue_previous_year"]),
                df["revenue_previous_year"],
            )
            * 100
        )
        # Backward-compatible alias for event labels
        result["revenue_growth"] = result["revenue_growth_yoy"]

    # EPS Growth
    if "eps" in df.columns and "eps_previous_year" in df.columns:
        result["eps_growth_yoy"] = (
            _safe_div((df["eps"] - df["eps_previous_year"]), df["eps_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["earnings_growth"] = result["eps_growth_yoy"]

    # EBITDA Growth
    if "ebitda" in df.columns and "ebitda_previous_year" in df.columns:
        result["ebitda_growth_yoy"] = (
            _safe_div((df["ebitda"] - df["ebitda_previous_year"]), df["ebitda_previous_year"]) * 100
        )
        # Backward-compatible alias for event labels
        result["ebitda_growth"] = result["ebitda_growth_yoy"]

    # 1. Book Value Growth (using TBV snapshots)
    if "tbv_ltm" in df.columns and "tbv_fy" in df.columns:
        result["book_value_growth"] = _safe_div(
            (df["tbv_ltm"] - df["tbv_fy"]), df["tbv_fy"]
        ) * 100

    # 2. Operating Income Growth
    if "operating_income_ltm" in df.columns and "operating_income_fy" in df.columns:
        result["operating_income_growth"] = _safe_div(
            (df["operating_income_ltm"] - df["operating_income_fy"]), df["operating_income_fy"]
        ) * 100

    # 3. FCF Growth
    if "fcf_ltm" in df.columns and "fcf_fy" in df.columns:
        result["fcf_growth"] = _safe_div(
            (df["fcf_ltm"] - df["fcf_fy"]), df["fcf_fy"].abs()
        ) * 100

    # 4. Forward Growth Indicators
    if "revenues_est_yoy_pct_fy1e" in df.columns:
        result["forward_revenue_growth"] = pd.to_numeric(
            df["revenues_est_yoy_pct_fy1e"], errors="coerce"
        )

    # 5. Long-Term Consistency
    if "total_revenues_cagr_5y_fy" in df.columns:
        result["revenue_cagr_5y"] = pd.to_numeric(df["total_revenues_cagr_5y_fy"], errors="coerce")

    # Growth Persistence Score (Current Growth vs Long-Term Trend)
    if "revenue_growth_yoy" in result.columns and "revenue_cagr_5y" in result.columns:
        result["growth_persistence_score"] = (
            result["revenue_growth_yoy"] - result["revenue_cagr_5y"]
        )

    # Current vs 5Y average revenue (growth indicator)
    if "total_revenues_ltm" in df.columns and "total_revenues_5yavgltm" in df.columns:
        result["revenue_vs_5y_avg"] = _safe_div(
            df["total_revenues_ltm"], df["total_revenues_5yavgltm"]
        )
        # Growth acceleration signal
        result["revenue_above_5y_avg_flag"] = (
            df["total_revenues_ltm"] > df["total_revenues_5yavgltm"] * 1.1
        ).astype(int)

    # Operating income growth
    if "operating_income_ltm" in df.columns and "operating_income_fy" in df.columns:
        result["operating_income_growth_yoy"] = (
            _safe_div(
                df["operating_income_ltm"] - df["operating_income_fy"],
                df["operating_income_fy"].abs(),
            )
            * 100
        )

    # EBITDA vs 5Y growth
    if "ebitda_ltm" in df.columns and "ebitda_5yavgltm" in df.columns:
        result["ebitda_vs_5y_avg"] = _safe_div(df["ebitda_ltm"], df["ebitda_5yavgltm"])

    logger.info("Engineered growth metrics (enhanced coverage)")
    return result
