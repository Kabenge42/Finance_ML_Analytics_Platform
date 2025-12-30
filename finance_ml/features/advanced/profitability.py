"""Profitability-related feature engineering."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_profitability_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer profitability ratios.

    Ratios computed:
    - ROE (Return on Equity)
    - ROA (Return on Assets)
    - ROIC (Return on Invested Capital)
    - Gross Margin %
    - Operating Margin %
    - Net Margin %
    - EBITDA/EBIT adjustment ratios (adj/LTM)

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with profitability ratios added
    """
    result = df.copy()

    # ROE
    if "net_income" in df.columns and "total_equity" in df.columns:
        result["roe"] = _safe_div(df["net_income"], df["total_equity"]) * 100

    # ROA
    if "net_income" in df.columns and "total_assets" in df.columns:
        result["roa"] = _safe_div(df["net_income"], df["total_assets"]) * 100

    # ROIC (simplified: Net Income / (Total Equity + Total Debt))
    if "net_income" in df.columns and "total_equity" in df.columns and "total_debt" in df.columns:
        invested_capital = df["total_equity"] + df["total_debt"]
        result["roic"] = _safe_div(df["net_income"], invested_capital) * 100

    # Gross Margin %
    if "gross_profit" in df.columns and "revenue" in df.columns:
        result["gross_margin_pct"] = _safe_div(df["gross_profit"], df["revenue"]) * 100

    # Operating Margin %
    if "operating_income" in df.columns and "revenue" in df.columns:
        result["operating_margin_pct"] = _safe_div(df["operating_income"], df["revenue"]) * 100

    # Net Margin %
    if "net_income" in df.columns and "revenue" in df.columns:
        result["net_margin_pct"] = _safe_div(df["net_income"], df["revenue"]) * 100

    # Adjustment ratios (adj/LTM/FY) as robustness/quality proxies
    if "ebitda_adj_ltm" in df.columns and "ebitda_ltm" in df.columns:
        result["ebitda_adjustment_ratio_ltm"] = _safe_div(
            df["ebitda_adj_ltm"].abs(), df["ebitda_ltm"].abs()
        )

    if "ebitda_adj_fy" in df.columns and "ebitda_fy" in df.columns:
        result["ebitda_adjustment_ratio_fy"] = _safe_div(
            df["ebitda_adj_fy"].abs(), df["ebitda_fy"].abs()
        )

    if "ebit_adj_ltm" in df.columns and "ebit_ltm" in df.columns:
        result["ebit_adjustment_ratio_ltm"] = _safe_div(
            df["ebit_adj_ltm"].abs(), df["ebit_ltm"].abs()
        )

    if "ebit_adj_fy" in df.columns and "ebit_fy" in df.columns:
        result["ebit_adjustment_ratio_fy"] = _safe_div(df["ebit_adj_fy"].abs(), df["ebit_fy"].abs())

    if "net_income_adj_ltm" in df.columns and "net_income_is_ltm" in df.columns:
        result["net_income_adjustment_ratio_ltm"] = _safe_div(
            df["normalized_net_income_fq"].abs() if "normalized_net_income_fq" in df.columns else df["net_income_adj_ltm"].abs(), 
            df["net_income_is_ltm"].abs()
        )

    if "net_income_adj_fy" in df.columns and "net_income_is_fy" in df.columns:
        result["net_income_adjustment_ratio_fy"] = _safe_div(
            df["net_income_adj_fy"].abs(), df["net_income_is_fy"].abs()
        )

    # 5. Operational Efficiency Ratios
    # R&D Intensity
    if "randd_expenses_ltm" in df.columns and "total_revenues_ltm" in df.columns:
        result["rnd_intensity"] = _safe_div(df["randd_expenses_ltm"], df["total_revenues_ltm"])

    # Marketing Efficiency (Revenue / Marketing Expenses) - using FY data
    if "marketing_expenses_fy" in df.columns:
        rev_col = "total_revenues_fy" if "total_revenues_fy" in df.columns else "total_revenues"
        if rev_col in df.columns:
            result["marketing_efficiency"] = _safe_div(df[rev_col], df["marketing_expenses_fy"])

    # SG&A Ratio
    if "selling_general_and_admin_expenses_total_fy" in df.columns:
        rev_col = "total_revenues_fy" if "total_revenues_fy" in df.columns else "total_revenues"
        if rev_col in df.columns:
            result["sga_ratio"] = _safe_div(
                df["selling_general_and_admin_expenses_total_fy"], df[rev_col]
            )

    # 6. Dupont Analysis Components (Equity Multiplier)
    # ROE = Net Margin * Asset Turnover * Equity Multiplier
    # Equity Multiplier = Total Assets / Total Equity
    if "total_assets" in df.columns and "total_equity" in df.columns:
        result["equity_multiplier"] = _safe_div(df["total_assets"], df["total_equity"])

    logger.info("Engineered profitability ratios")
    return result

def engineer_margin_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer margin trend features comparing current vs historical periods.

    Features computed:
    - gross_margin_trend: (gross_margin_pct - gross_margin_pct_1fy) / gross_margin_pct_1fy
    - operating_margin_trend: (operating_margin_pct - operating_margin_pct_1fy) / ...
    - net_margin_trend: (net_margin_pct - net_margin_pct_1fy) / ...
    - ebitda_margin_trend: (ebitda_margin_pct - ebitda_margin_pct_1fy) / ...
    - operating_leverage: (%ΔEBIT) / (%ΔRevenue) using ltm vs 1fy
    - earnings_quality_score: 0–100 score based on adjustment ratios

    Returns:
        DataFrame with margin trend features added.
    """
    result = df.copy()

    # Gross margin trend
    if "gross_profit_margin_pct_ltm" in df.columns and "gross_profit_margin_pct_fy" in df.columns:
        result["gross_margin_trend"] = _safe_div(
            df["gross_profit_margin_pct_ltm"] - df["gross_profit_margin_pct_fy"],
            df["gross_profit_margin_pct_fy"].abs(),
        )

    # Operating margin trend
    if "operating_margin_pct" in result.columns:
        # Try to compute from EBIT margins if available
        if "ebit_ltm" in df.columns and "ebit_1fy" in df.columns:
            if "total_revenues_ltm" in df.columns and "total_revenues_1fy" in df.columns:
                current_margin = _safe_div(df["ebit_ltm"], df["total_revenues_ltm"]) * 100
                prev_margin = _safe_div(df["ebit_1fy"], df["total_revenues_1fy"]) * 100
                result["operating_margin_trend"] = _safe_div(
                    current_margin - prev_margin, prev_margin.abs()
                )

    # Net margin trend
    if "net_income_margin_pct_ltm" in df.columns and "net_income_margin_pct_fy" in df.columns:
        result["net_margin_trend"] = _safe_div(
            df["net_income_margin_pct_ltm"] - df["net_income_margin_pct_fy"],
            df["net_income_margin_pct_fy"].abs(),
        )

    # EBITDA margin trend
    if "ebitda_ltm" in df.columns and "ebitda_1fy" in df.columns:
        if "total_revenues_ltm" in df.columns and "total_revenues_1fy" in df.columns:
            current_ebitda_margin = _safe_div(df["ebitda_ltm"], df["total_revenues_ltm"]) * 100
            prev_ebitda_margin = _safe_div(df["ebitda_1fy"], df["total_revenues_1fy"]) * 100
            result["ebitda_margin_trend"] = _safe_div(
                current_ebitda_margin - prev_ebitda_margin, prev_ebitda_margin.abs()
            )

    # Operating leverage = (%ΔEBIT)/(%ΔRevenue)
    if all(
        c in df.columns
        for c in ("ebit_ltm", "ebit_1fy", "total_revenues_ltm", "total_revenues_1fy")
    ):
        delta_ebit = _safe_div(
            df["ebit_ltm"].astype(float) - df["ebit_1fy"].astype(float),
            df["ebit_1fy"].astype(float).abs(),
        )
        delta_rev = _safe_div(
            df["total_revenues_ltm"].astype(float) - df["total_revenues_1fy"].astype(float),
            df["total_revenues_1fy"].astype(float).abs(),
        )
        result["operating_leverage"] = _safe_div(delta_ebit, delta_rev)

    # Earnings quality score based on adjustment ratios
    ebitda_adj_ratio = None
    ebit_adj_ratio = None

    # Try preferred columns first (created by engineer_profitability_ratios)
    if "ebitda_adjustment_ratio_ltm" in result.columns:
        ebitda_adj_ratio = result["ebitda_adjustment_ratio_ltm"].astype(float)
    elif "ebitda_adjustment_ratio" in df.columns:
        ebitda_adj_ratio = df["ebitda_adjustment_ratio"].astype(float)
    elif all(c in df.columns for c in ("ebitda_adj_ltm", "ebitda_ltm")):
        ebitda_adj_ratio = _safe_div(df["ebitda_adj_ltm"].abs(), df["ebitda_ltm"].abs())

    if "ebit_adjustment_ratio_ltm" in result.columns:
        ebit_adj_ratio = result["ebit_adjustment_ratio_ltm"].astype(float)
    elif "ebit_adjustment_ratio" in df.columns:
        ebit_adj_ratio = df["ebit_adjustment_ratio"].astype(float)
    elif all(c in df.columns for c in ("ebit_adj_ltm", "ebit_ltm")):
        ebit_adj_ratio = _safe_div(df["ebit_adj_ltm"].abs(), df["ebit_ltm"].abs())

    if ebitda_adj_ratio is not None or ebit_adj_ratio is not None:
        a = (
            ebitda_adj_ratio
            if ebitda_adj_ratio is not None
            else pd.Series(0.0, index=result.index)
        )
        b = ebit_adj_ratio if ebit_adj_ratio is not None else pd.Series(0.0, index=result.index)
        score = 100.0 - 50.0 * a - 30.0 * b
        result["earnings_quality_score"] = score.clip(lower=0.0, upper=100.0)

    logger.info("Engineered margin trend features")
    return result
