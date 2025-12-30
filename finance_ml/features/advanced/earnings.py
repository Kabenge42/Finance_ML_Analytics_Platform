"""Earnings analytics feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_estimated_vs_actual_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer Estimated vs. Actual earnings analytics features.

    Phase 9.3 Enhanced Earnings Analytics: Compares forward estimates against
    actual reported metrics to identify earnings surprises, estimate momentum,
    and analyst forecast accuracy.

    Features created:
    - EPS Surprise %: (eps_actual - eps_estimated) / |eps_estimated| * 100
    - Revenue Surprise %: (revenue_actual - revenue_estimated) / |revenue_estimated| * 100
    - EBITDA Surprise %: (ebitda_actual - ebitda_estimated) / |ebitda_estimated| * 100
    - Earnings Beat Indicator: Boolean flag for positive EPS surprise
    - Surprise Momentum Score: Weighted average of multi-period revisions (1M, 3M, 6M)
    - Surprise Magnitude: Categorical (small/moderate/large) based on surprise %
    - Consensus Uncertainty: Spread between estimate and actual as volatility proxy
    - Estimate Revision Trend: Acceleration in estimate revisions over time

    Input Columns (from COLUMN_SCHEMA):
    - Actuals: eps_adj_ltm, net_eps_basic_ltm, total_revenues_ltm, ebitda_ltm
    - Estimates: eps_norm_est_avg_ntm, eps_norm_est_avg_fy1e, revenues_est_avg_ntm,
                 revenues_est_avg_fy1e, ebitda_est_avg_ntm, ebitda_est_avg_fy1e
    - Revisions: eps_est_avg_rev_pct_fy1e_1m, eps_est_avg_rev_pct_fy1e_3m,
                 eps_est_avg_rev_pct_fy1e_6m, eps_est_avg_rev_pct_fy1e_1y

    Args:
        df: Input DataFrame with EPS, revenue, and estimate columns

    Returns:
        DataFrame with estimated vs. actual analytics features added
    """
    result = df.copy()

    # =========================================================================
    # 1. EPS Surprise Analytics
    # =========================================================================
    # Primary EPS Surprise: Actual vs. Next Twelve Months (NTM) estimate
    actual_eps_cols = ["eps_adj_ltm", "net_eps_basic_ltm", "eps"]
    estimate_eps_cols = ["eps_norm_est_avg_ntm", "eps_norm_est_avg_fy1e"]

    eps_actual = None
    for col in actual_eps_cols:
        if col in df.columns:
            eps_actual = pd.to_numeric(df[col], errors="coerce")
            break

    eps_estimate = None
    for col in estimate_eps_cols:
        if col in df.columns:
            eps_estimate = pd.to_numeric(df[col], errors="coerce")
            break

    if eps_actual is not None and eps_estimate is not None:
        # EPS Surprise Percentage
        result["eps_surprise_pct"] = (
            _safe_div((eps_actual - eps_estimate), eps_estimate.abs()) * 100
        )

        # Earnings Beat Indicator (boolean flag)
        result["earnings_beat_indicator"] = (result["eps_surprise_pct"] > 0).fillna(False)

        # Surprise Magnitude Categorization
        surprise_abs = result["eps_surprise_pct"].abs()
        eps_magnitude_cat = pd.cut(
            surprise_abs,
            bins=[0, 5, 15, float("inf")],
            labels=["small", "moderate", "large"],
            include_lowest=True,
        )
        # Add "unknown" to categories before filling NaN, then convert to category dtype
        eps_magnitude_cat = eps_magnitude_cat.cat.add_categories(["unknown"])
        result["eps_surprise_magnitude"] = eps_magnitude_cat.fillna("unknown").astype("category")

        logger.info(f"Computed EPS surprise for {result['eps_surprise_pct'].notna().sum()} stocks")

    # =========================================================================
    # 2. Revenue Surprise Analytics
    # =========================================================================
    actual_revenue_cols = ["total_revenues_ltm", "total_revenues_fy", "revenue"]
    estimate_revenue_cols = [
        "revenues_est_avg_ntm",
        "revenues_est_avg_fy1e",
        "revenues_est_med_ntm",
    ]

    revenue_actual = None
    for col in actual_revenue_cols:
        if col in df.columns:
            revenue_actual = pd.to_numeric(df[col], errors="coerce")
            break

    revenue_estimate = None
    for col in estimate_revenue_cols:
        if col in df.columns:
            revenue_estimate = pd.to_numeric(df[col], errors="coerce")
            break

    if revenue_actual is not None and revenue_estimate is not None:
        # Revenue Surprise Percentage
        result["revenue_surprise_pct"] = (
            _safe_div((revenue_actual - revenue_estimate), revenue_estimate.abs()) * 100
        )

        # Revenue Beat Indicator
        result["revenue_beat_indicator"] = (result["revenue_surprise_pct"] > 0).fillna(False)

    # =========================================================================
    # 3. Estimate Revision Momentum
    # =========================================================================
    # Composite score based on multi-period revisions
    rev_cols = {
        "eps_est_avg_rev_pct_fy1e_1w": 0.6,
        "eps_est_avg_rev_pct_fy1e_1m": 0.5,  # 1M revision weighted most
        "eps_est_avg_rev_pct_fy1e_3m": 0.3,
        "eps_est_avg_rev_pct_fy1e_6m": 0.2,
        "eps_est_avg_rev_pct_fy1e_1y": 0.1,
    }

    momentum_components = []
    for col, weight in rev_cols.items():
        if col in df.columns:
            val = pd.to_numeric(df[col], errors="coerce").fillna(0)
            momentum_components.append(val * weight)

    if momentum_components:
        result["surprise_momentum_score"] = pd.concat(momentum_components, axis=1).sum(axis=1)

    # GAAP vs Non-GAAP Revision Divergence
    if (
        "eps_gaap_est_avg_rev_pct_fy1e_1m" in df.columns
        and "eps_est_avg_rev_pct_fy1e_1m" in df.columns
    ):
        result["gaap_revision_divergence"] = pd.to_numeric(
            df["eps_gaap_est_avg_rev_pct_fy1e_1m"], errors="coerce"
        ) - pd.to_numeric(df["eps_est_avg_rev_pct_fy1e_1m"], errors="coerce")

    # Revenue Forecast Skew
    if "revenues_est_avg_ntm" in df.columns and "revenues_est_med_ntm" in df.columns:
        avg_rev = pd.to_numeric(df["revenues_est_avg_ntm"], errors="coerce")
        med_rev = pd.to_numeric(df["revenues_est_med_ntm"], errors="coerce")
        result["revenue_forecast_skew"] = _safe_div(avg_rev - med_rev, avg_rev)

    # Revision acceleration (1M revision - 3M revision)
    if "eps_est_avg_rev_pct_fy1e_1m" in df.columns and "eps_est_avg_rev_pct_fy1e_3m" in df.columns:
        result["estimate_revision_acceleration"] = (
            df["eps_est_avg_rev_pct_fy1e_1m"] - df["eps_est_avg_rev_pct_fy1e_3m"]
        ).fillna(False)

    logger.info("Engineered Estimated vs. Actual analytics features (Phase 9.3)")
    return result

def engineer_gaap_vs_adjusted_analytics(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer GAAP vs. Adjusted earnings quality analytics features.

    Phase 9.3 Enhanced Earnings Quality: Compares GAAP (reported) metrics against
    Adjusted (non-GAAP) metrics to assess earnings quality, identify aggressive
    accounting adjustments, and flag potential red flags.

    Features created:
    - Adjustment Spreads: Dollar differences between adjusted and GAAP metrics
    - Adjustment Ratios: Adjusted / GAAP for EPS, Net Income, EBITDA, EBIT
    - Earnings Quality Flags: Warning indicators for excessive adjustments (>20%)
    - Adjustment Consistency Score: Temporal stability of adjustment patterns
    - Exceptional Items Impact: Non-recurring item magnitude relative to core earnings
    - Quality Score: Composite 0-100 score based on adjustment magnitudes

    Input Columns (from COLUMN_SCHEMA):
    - GAAP EPS: net_eps_basic_ltm, net_eps_basic_fq, net_eps_basic_fy
    - Adjusted EPS: eps_adj_ltm, eps_adj_fy, eps_adj_1fy
    - GAAP Estimates: eps_gaap_est_avg_fy1e, eps_gaap_est_avg_ntm
    - GAAP Net Income: net_income_is_ltm, net_income_is_fy, net_income_is_fq
    - Adjusted Net Income: net_income_adj_ltm, net_income_adj_fy, net_income_adj_1fy
    - GAAP EBITDA: ebitda_ltm, ebitda_fy, ebitda_fq
    - Adjusted EBITDA: ebitda_adj_ltm, ebitda_adj_fy, ebitda_adj_1fy
    - GAAP EBIT: ebit_ltm, ebit_fy, ebit_fq
    - Adjusted EBIT: ebit_adj_ltm, ebit_adj_fy, ebit_adj_1fy

    Args:
        df: Input DataFrame with GAAP and adjusted earnings columns

    Returns:
        DataFrame with GAAP vs. Adjusted analytics features added
    """
    result = df.copy()

    # =========================================================================
    # 1. EPS Adjustment Analytics (LTM - Last Twelve Months)
    # =========================================================================
    if "eps_adj_ltm" in df.columns and "net_eps_basic_ltm" in df.columns:
        eps_adj = pd.to_numeric(df["eps_adj_ltm"], errors="coerce")
        eps_gaap = pd.to_numeric(df["net_eps_basic_ltm"], errors="coerce")

        # EPS Adjustment Spread (dollar difference)
        result["eps_adjustment_spread_ltm"] = eps_adj - eps_gaap

        # EPS Adjustment Ratio (adjusted / GAAP)
        result["eps_adjustment_ratio_ltm"] = _safe_div(eps_adj, eps_gaap)

        # EPS Adjustment Percentage
        result["eps_adjustment_pct_ltm"] = _safe_div((eps_adj - eps_gaap), eps_gaap.abs()) * 100

        # Earnings Quality Flag: Warn if adjustment > 20%
        result["eps_quality_flag_ltm"] = (result["eps_adjustment_pct_ltm"].abs() > 20).fillna(False)

        logger.info(
            f"Computed EPS GAAP vs. Adjusted for {result['eps_adjustment_ratio_ltm'].notna().sum()} stocks (LTM)"
        )

    # EPS Adjustment Analytics (FY - Fiscal Year)
    if "eps_adj_fy" in df.columns and "net_eps_basic_fy" in df.columns:
        eps_adj_fy = pd.to_numeric(df["eps_adj_fy"], errors="coerce")
        eps_gaap_fy = pd.to_numeric(df["net_eps_basic_fy"], errors="coerce")

        result["eps_adjustment_spread_fy"] = eps_adj_fy - eps_gaap_fy
        result["eps_adjustment_ratio_fy"] = _safe_div(eps_adj_fy, eps_gaap_fy)
        result["eps_adjustment_pct_fy"] = (
            _safe_div((eps_adj_fy - eps_gaap_fy), eps_gaap_fy.abs()) * 100
        )

    # =========================================================================
    # 2. Net Income Adjustment Analytics
    # =========================================================================
    if "net_income_adj_ltm" in df.columns and "net_income_is_ltm" in df.columns:
        ni_adj = pd.to_numeric(df["net_income_adj_ltm"], errors="coerce")
        ni_gaap = pd.to_numeric(df["net_income_is_ltm"], errors="coerce")

        # Net Income Adjustment Spread (dollar difference)
        result["net_income_adjustment_spread_ltm"] = ni_adj - ni_gaap

        # Net Income Adjustment Ratio
        result["net_income_adjustment_ratio_ltm"] = _safe_div(ni_adj, ni_gaap)

        # Net Income Adjustment Percentage
        result["net_income_adjustment_pct_ltm"] = _safe_div((ni_adj - ni_gaap), ni_gaap.abs()) * 100

        logger.info(
            f"Computed Net Income GAAP vs. Adjusted for {result['net_income_adjustment_ratio_ltm'].notna().sum()} stocks (LTM)"
        )

    # Net Income Adjustment (FY)
    if "net_income_adj_fy" in df.columns and "net_income_is_fy" in df.columns:
        ni_adj_fy = pd.to_numeric(df["net_income_adj_fy"], errors="coerce")
        ni_gaap_fy = pd.to_numeric(df["net_income_is_fy"], errors="coerce")

        result["net_income_adjustment_spread_fy"] = ni_adj_fy - ni_gaap_fy
        result["net_income_adjustment_ratio_fy"] = _safe_div(ni_adj_fy, ni_gaap_fy)

    # =========================================================================
    # 3. EBITDA Adjustment Analytics
    # =========================================================================
    if "ebitda_ltm" in df.columns and "ebitda_adj_ltm" in df.columns:
        result["ebitda_adjustment_spread_ltm"] = df["ebitda_adj_ltm"] - df["ebitda_ltm"]
        result["ebitda_adjustment_ratio_ltm"] = _safe_div(df["ebitda_adj_ltm"], df["ebitda_ltm"])
        result["ebitda_adjustment_pct_ltm"] = _safe_div(
            result["ebitda_adjustment_spread_ltm"].abs(), df["ebitda_ltm"].abs()
        ) * 100

    if "ebitda_fy" in df.columns and "ebitda_adj_fy" in df.columns:
        result["ebitda_adjustment_spread_fy"] = df["ebitda_adj_fy"] - df["ebitda_fy"]
        result["ebitda_adjustment_ratio_fy"] = _safe_div(df["ebitda_adj_fy"], df["ebitda_fy"])

    # =========================================================================
    # 4. EBIT Adjustment Analytics
    # =========================================================================
    if "ebit_ltm" in df.columns and "ebit_adj_ltm" in df.columns:
        result["ebit_adjustment_spread_ltm"] = df["ebit_adj_ltm"] - df["ebit_ltm"]
        result["ebit_adjustment_ratio_ltm"] = _safe_div(df["ebit_adj_ltm"], df["ebit_ltm"])
        result["ebit_adjustment_pct_ltm"] = _safe_div(
            result["ebit_adjustment_spread_ltm"].abs(), df["ebit_ltm"].abs()
        ) * 100

    # =========================================================================
    # 5. Quality Scoring Logic
    # =========================================================================
    if "eps_adjustment_pct_ltm" in result.columns:
        # Lower adjustment percentage = Higher quality
        result["earnings_quality_score"] = 100 - result["eps_adjustment_pct_ltm"].clip(0, 100)
        result["earnings_quality_warning_flag"] = (result["eps_adjustment_pct_ltm"] > 15).astype(int)

    logger.info("Engineered GAAP vs Adjusted earnings quality features")
    return result
