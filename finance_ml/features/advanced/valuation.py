"""Valuation-related feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_valuation_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer comprehensive valuation ratios.

    Ratios computed:
    - P/E (Price/Earnings)
    - P/B (Price/Book)
    - P/S (Price/Sales)
    - EV/EBITDA
    - EV/Sales
    - PEG (P/E to Growth)
    - Dividend Yield

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with valuation ratios added
    """
    result = df.copy()

    # Book Value per Share
    if "total_equity" in df.columns and "shares_outstanding" in df.columns:
        result["book_value_per_share"] = _safe_div(df["total_equity"], df["shares_outstanding"])

    # P/E ratio
    if "last_price" in df.columns and "eps" in df.columns:
        result["p_e_ratio"] = _safe_div(df["last_price"], df["eps"])

    # P/B ratio
    if "last_price" in df.columns and "book_value_per_share" in df.columns:
        result["p_b_ratio"] = _safe_div(df["last_price"], df["book_value_per_share"])

    # P/S ratio (Price to Sales per share)
    if (
        "last_price" in df.columns
        and "revenue" in df.columns
        and "shares_outstanding" in df.columns
    ):
        sales_per_share = _safe_div(df["revenue"], df["shares_outstanding"])
        result["p_s_ratio"] = _safe_div(df["last_price"], sales_per_share)

    # EV/EBITDA
    if "enterprise_value" in df.columns and "ebitda" in df.columns:
        result["ev_ebitda_ratio"] = _safe_div(df["enterprise_value"], df["ebitda"])

    # EV/Sales
    if "enterprise_value" in df.columns and "revenue" in df.columns:
        result["ev_sales_ratio"] = _safe_div(df["enterprise_value"], df["revenue"])

    # PEG ratio (P/E to Growth)
    if "p_e" in result.columns and "earnings_growth" in df.columns:
        result["peg_ratio"] = _safe_div(result["p_e"], df["earnings_growth"])

    # Dividend Yield
    if "dividend_per_share" in df.columns and "last_price" in df.columns:
        result["dividend_yield"] = _safe_div(df["dividend_per_share"], df["last_price"]) * 100

    logger.info("Engineered valuation ratios")
    return result

def engineer_valuation_timeseries_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer valuation time-series features using extended EV/Sales, EV/EBITDA, and P/E data.

    Phase 9.3 Schema Version 1.3: Leverages new valuation multiples time-series columns
    (EV/Sales variants, EV/EBITDA extended, P/E extended history).

    Features created:
    - Valuation momentum indicators (1Y, 3Y trends)
    - Mean reversion metrics (current vs 3Y average)
    - Forward vs trailing valuation spreads
    - Quarterly valuation stability metrics

    Args:
        df: Input DataFrame with valuation time-series columns

    Returns:
        DataFrame with valuation valuation time-series features added
    """
    result = df.copy()

    # 1. Valuation Momentum Indicators
    # EV/Sales trend (1Y)
    if "ev_sales_ltm" in df.columns and "ev_sales_1fyltm" in df.columns:
        result["ev_sales_trend_1y"] = _safe_div(
            df["ev_sales_ltm"] - df["ev_sales_1fyltm"], df["ev_sales_1fyltm"]
        )

    # EV/Sales trend (3Y) - using 3-year lookback
    if all(
        c in df.columns
        for c in [
            "ev_sales_ltm",
            "ev_sales_1fyltm",
            "ev_sales_2fyltm",
            "ev_sales_3fyltm",
        ]
    ):
        # Linear trend slope approximation
        y_vals = pd.concat(
            [
                df["ev_sales_3fyltm"],
                df["ev_sales_2fyltm"],
                df["ev_sales_1fyltm"],
                df["ev_sales_ltm"],
            ],
            axis=1,
        )
        result["ev_sales_trend_3y"] = y_vals.apply(
            lambda row: (
                (row.iloc[-1] - row.iloc[0]) / (row.iloc[0] + 1e-9)
                if row.notna().sum() >= 2
                else np.nan
            ),
            axis=1,
        )

    # EV/EBITDA momentum
    if "ev_ebitda_ltm" in df.columns and "ev_ebitda_1fyltm" in df.columns:
        result["ev_ebitda_momentum"] = _safe_div(
            df["ev_ebitda_ltm"] - df["ev_ebitda_1fyltm"], df["ev_ebitda_1fyltm"]
        )

    # P/E momentum (YoY)
    if "p_e_ltm" in df.columns and "p_e_1fyltm" in df.columns:
        result["p_e_momentum_yoy"] = _safe_div(df["p_e_ltm"] - df["p_e_1fyltm"], df["p_e_1fyltm"])

    # P/E momentum (QoQ)
    if "p_e_ltm" in df.columns and "p_e_1fqltm" in df.columns:
        result["p_e_momentum_qoq"] = _safe_div(df["p_e_ltm"] - df["p_e_1fqltm"], df["p_e_1fqltm"])

    # 2. Valuation Mean Reversion Features
    if "ev_sales_ltm" in df.columns and "ev_sales_3yavgltm" in df.columns:
        # Z-score: (current - mean) / std, approximated using deviation from 3Y avg
        deviation = df["ev_sales_ltm"] - df["ev_sales_3yavgltm"]
        result["ev_sales_vs_3y_avg"] = _safe_div(deviation, df["ev_sales_3yavgltm"])

    if "ev_ebitda_ltm" in df.columns and "ev_ebitda_3yavgltm" in df.columns:
        deviation = df["ev_ebitda_ltm"] - df["ev_ebitda_3yavgltm"]
        result["ev_ebitda_vs_3y_avg"] = _safe_div(deviation, df["ev_ebitda_3yavgltm"])

    if "p_e_ltm" in df.columns and "p_e_3yavgltm" in df.columns:
        deviation = df["p_e_ltm"] - df["p_e_3yavgltm"]
        result["p_e_vs_3y_avg"] = _safe_div(deviation, df["p_e_3yavgltm"])

    # Valuation extreme flag (>2 std dev from mean, approximated as >200% deviation)
    if "ev_sales_vs_3y_avg" in result.columns:
        result["valuation_extreme_flag"] = (
            (result["ev_sales_vs_3y_avg"].abs() > 2.0).fillna(False).astype(int)
        )

    # 3. Forward vs Trailing Valuation
    if "ev_sales_ntm" in df.columns and "ev_sales_ltm" in df.columns:
        result["ev_sales_forward_discount"] = _safe_div(
            df["ev_sales_ntm"] - df["ev_sales_ltm"], df["ev_sales_ltm"]
        )

    if "ev_ebitda_ntm" in df.columns and "ev_ebitda_ltm" in df.columns:
        result["ev_ebitda_forward_discount"] = _safe_div(
            df["ev_ebitda_ntm"] - df["ev_ebitda_ltm"], df["ev_ebitda_ltm"]
        )

    if "p_e_est_fy1" in df.columns and "p_e_ltm" in df.columns:
        result["p_e_forward_discount"] = _safe_div(df["p_e_est_fy1"] - df["p_e_ltm"], df["p_e_ltm"])

    # Growth implied by valuation (forward discount as proxy for growth expectations)
    if "ev_sales_forward_discount" in result.columns:
        # Negative discount implies growth expectations
        result["growth_implied_by_valuation"] = -result["ev_sales_forward_discount"]

    # 4. Valuation Stability Metrics
    ev_sales_cols = [c for c in df.columns if "ev_sales_" in c and "ltm" in c]
    if len(ev_sales_cols) >= 4:
        result["ev_sales_quarterly_volatility"] = df[ev_sales_cols].std(axis=1) / df[
            ev_sales_cols
        ].mean(axis=1)

    # 5. Valuation Consistency (monotonicity of forward multiples)
    if all(c in df.columns for c in ["ev_sales_ntm", "ev_sales_ltm", "ev_sales_1fyltm"]):

        def check_monotonicity(row):
            vals = row.dropna()
            if len(vals) < 2:
                return 0
            return int(vals.is_monotonic_increasing or vals.is_monotonic_decreasing)

        result["valuation_trend_consistency"] = df[
            ["ev_sales_ntm", "ev_sales_ltm", "ev_sales_1fyltm"]
        ].apply(check_monotonicity, axis=1)

    # Valuation Stability Score (composite)
    if "ev_sales_quarterly_volatility" in result.columns:
        result["valuation_stability_score"] = 1.0 / (
            1.0 + result["ev_sales_quarterly_volatility"]
        )

    # EV/Sales quarterly trajectory
    EV_SALES_FQ_COLS = [
        "ev_sales_ltm",
        "ev_sales_1fqltm",
        "ev_sales_2fqltm",
        "ev_sales_3fqltm",
        "ev_sales_4fqltm",
    ]
    if all(c in df.columns for c in EV_SALES_FQ_COLS[:3]):
        # Quarterly volatility
        ev_sales_mat = df[EV_SALES_FQ_COLS[:3]].astype(float)
        result["ev_sales_quarterly_volatility"] = ev_sales_mat.std(axis=1, ddof=0)

        # Trend direction consistency
        ev_q1_vs_q2 = np.sign(df["ev_sales_ltm"] - df["ev_sales_1fqltm"])
        ev_q2_vs_q3 = np.sign(df["ev_sales_1fqltm"] - df["ev_sales_2fqltm"])
        result["ev_sales_trend_consistency"] = (ev_q1_vs_q2 == ev_q2_vs_q3).astype(int)

    # P/E sequential momentum
    if "p_e_0fqqoqltm" in df.columns:
        result["p_e_qoq_momentum"] = df["p_e_0fqqoqltm"]
    if "p_e_0fyyoyltm" in df.columns:
        result["p_e_yoy_momentum"] = df["p_e_0fyyoyltm"]

    # P/B vs 5Y average (mean reversion signal)
    if "p_b_ltm" in df.columns and "p_b_5yavg" in df.columns:
        result["p_b_vs_5y_avg"] = _safe_div(df["p_b_ltm"], df["p_b_5yavg"])
        result["p_b_mean_reversion_signal"] = (
            df["p_b_ltm"] > df["p_b_5yavg"] * 1.2
        ).astype(int) - (df["p_b_ltm"] < df["p_b_5yavg"] * 0.8).astype(int)

    logger.info("Engineered valuation time-series features (Phase 9.3 Schema 1.3)")
    return result
