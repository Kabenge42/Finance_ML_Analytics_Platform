"""Quality and risk-related feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_accounting_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer accounting quality and red flag features.

    Features computed:
    - Exceptional items flags and aggregation (goodwill impairment, asset writedowns, restructuring)
    - Exceptional items scaling ratios (to EBITDA/Net Income) and trend (YoY if available)
    - Goodwill to assets ratio (+ change rate), Intangibles intensity
    - Restructuring intensity to total assets
    - Composite accounting quality score (0-100, higher is better)

    Args:
        df: Input DataFrame (normalized column names expected)

    Returns:
        DataFrame with accounting quality features added
    """
    result = df.copy()

    # Goodwill impairment flag (red flag if present)
    if "impairment_of_goodwill_ltm" in df.columns:
        result["has_goodwill_impairment"] = (
            df["impairment_of_goodwill_ltm"].fillna(0) != 0
        ).astype(int)
        # Alias for compatibility with tests/plan wording
        result["goodwill_impairment_flag"] = result["has_goodwill_impairment"]

    # Asset writedown flag
    if "asset_writedown_ltm" in df.columns:
        result["has_asset_writedown"] = (df["asset_writedown_ltm"].fillna(0) != 0).astype(int)

    # Restructuring charges flag
    if "restructuring_charges_ltm" in df.columns:
        result["has_restructuring"] = (df["restructuring_charges_ltm"].fillna(0) != 0).astype(int)

    # Aggregate exceptional items (LTM)
    if all(
        c in df.columns
        for c in [
            "impairment_of_goodwill_ltm",
            "asset_writedown_ltm",
            "restructuring_charges_ltm",
        ]
    ):
        exceptional_items_ltm = (
            df["impairment_of_goodwill_ltm"].fillna(0).abs()
            + df["asset_writedown_ltm"].fillna(0).abs()
            + df["restructuring_charges_ltm"].fillna(0).abs()
        )
        result["total_exceptional_items_ltm"] = exceptional_items_ltm

        # Scale to EBITDA if available
        if "ebitda_ltm" in df.columns:
            result["exceptional_items_to_ebitda"] = _safe_div(
                exceptional_items_ltm, df["ebitda_ltm"].abs()
            )

        # Backward compatible ratio to Net Income (percent)
        if "net_income_ltm" in df.columns:
            result["exceptional_items_to_ni_pct"] = (
                _safe_div(exceptional_items_ltm, df["net_income_ltm"].abs()) * 100
            )

    # Exceptional items trend YoY if -1FY columns exist
    if all(
        c in df.columns
        for c in [
            "impairment_of_goodwill_1fy",
            "asset_writedown_1fy",
            "restructuring_charges_1fy",
        ]
    ):
        exceptional_items_1fy = (
            df["impairment_of_goodwill_1fy"].fillna(0).abs()
            + df["asset_writedown_1fy"].fillna(0).abs()
            + df["restructuring_charges_1fy"].fillna(0).abs()
        )
        if "total_exceptional_items_ltm" in result.columns:
            result["exceptional_items_trend"] = _safe_div(
                result["total_exceptional_items_ltm"] - exceptional_items_1fy,
                exceptional_items_1fy,
            )

    # Goodwill to total assets ratio (high ratio can be risky)
    if "goodwill_ltm" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["goodwill_ltm"], df["total_assets_ltm"])
        result["goodwill_to_assets_pct"] = ratio * 100
        # Aliases (fractional forms)
        result["goodwill_to_assets"] = ratio

    # Intangibles intensity
    if "intangible_assets" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["intangible_assets"], df["total_assets_ltm"])
        result["intangibles_to_assets_pct"] = ratio * 100
        result["intangible_intensity"] = ratio

    # Goodwill change rate (YoY)
    if "goodwill_ltm" in df.columns and "goodwill_1fy" in df.columns:
        result["goodwill_change_rate"] = _safe_div(
            df["goodwill_ltm"] - df["goodwill_1fy"], df["goodwill_1fy"]
        )

    # Restructuring intensity to total assets
    if "restructuring_charges_ltm" in df.columns and "total_assets_ltm" in df.columns:
        result["restructuring_intensity"] = _safe_div(
            df["restructuring_charges_ltm"], df["total_assets_ltm"]
        )

    # Merger Impact Ratio (Merger Charges / Market Cap)
    if "merger_and_restructuring_charges_ltm" in df.columns and "market_cap" in df.columns:
        result["merger_impact_ratio"] = _safe_div(
            df["merger_and_restructuring_charges_ltm"], df["market_cap"]
        )

    # Non-Operating Income Share (Interest Income / Net Income)
    if "interest_income_on_investments_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["non_operating_income_share"] = _safe_div(
            df["interest_income_on_investments_ltm"], df["net_income_ltm"].abs()
        )

    # Asset Sale Boost (Flag if gain on sale of assets > 0)
    if "gain_loss_on_sale_of_assets_ltm" in df.columns:
        result["asset_sale_boost"] = (df["gain_loss_on_sale_of_assets_ltm"].fillna(0) > 0).astype(
            int
        )

    # Accounting quality score (lower is better, 0-100 scale)
    # High exceptional items, high goodwill, presence of impairments = lower quality
    quality_components = []
    if "has_goodwill_impairment" in result.columns:
        quality_components.append(result["has_goodwill_impairment"] * 30)
    if "exceptional_items_to_ni_pct" in result.columns:
        quality_components.append((result["exceptional_items_to_ni_pct"] > 10).astype(int) * 20)
    if "goodwill_to_assets_pct" in result.columns:
        quality_components.append((result["goodwill_to_assets_pct"] > 20).astype(int) * 20)

    if quality_components:
        total_penalties = sum(quality_components)
        result["accounting_quality_score"] = (100 - total_penalties).clip(lower=0, upper=100)

    logger.info("Engineered accounting quality features")
    return result

def engineer_financial_distress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer financial distress indicators using Altman Z-Score variants.

    Features:
    - altman_z_trend: FY vs 1FY (fallback to FQ vs LTM if FY/1FY missing)
    - distress_risk_score: Composite 0–100 using available z-scores (higher = healthier)
    - z_score_volatility: Std deviation across available z-score periods (FQ, FY, LTM)

    Notes:
    - For Financials sector (sector == 'Financials'), returns NaN for all features due to sector-specific model caveats.
    - Missing inputs yield NaNs; function is robust to absent columns.
    """
    result = df.copy()

    # Mask out Financials sector for these metrics
    fin_mask = (
        (result.get("sector").astype(str).str.lower() == "financials")
        if "sector" in result.columns
        else pd.Series(False, index=result.index)
    )

    # Columns for different periods
    z_fy = result.get("altman_z_score_fy")
    z_1fy = result.get("altman_z_score_1fy")
    z_fq = result.get("altman_z_score_fq")
    z_ltm = result.get("altman_z_score_ltm")

    # Trend: prefer FY vs 1FY; fallback to FQ vs LTM
    trend = pd.Series(np.nan, index=result.index, dtype=float)
    if z_fy is not None and z_1fy is not None:
        trend = z_fy.astype(float) - z_1fy.astype(float)
    elif z_fq is not None and z_ltm is not None:
        trend = z_fq.astype(float) - z_ltm.astype(float)
    result["altman_z_trend"] = trend

    # Volatility across available periods
    z_stack = []
    for s in (z_fq, z_fy, z_ltm):
        if s is not None:
            z_stack.append(s.astype(float))
    if z_stack:
        z_mat = np.vstack([s.to_numpy(copy=False) for s in z_stack]).astype(float)
        # Std dev across rows (axis=0)
        z_vol = np.nanstd(z_mat, axis=0)
        result["z_score_volatility"] = pd.Series(z_vol, index=result.index)
    else:
        result["z_score_volatility"] = np.nan

    # Distress risk score: map average z-score to 0–100
    # Use simple clipping: z<=1.8 -> 0; z>=3.0 -> 100; linear in between
    z_components = []
    for s in (z_fq, z_fy, z_ltm):
        if s is not None:
            z_components.append(s.astype(float))
    if z_components:
        z_mean = pd.concat(z_components, axis=1).mean(axis=1)
        score = ((z_mean - 1.8) / (3.0 - 1.8) * 100.0).clip(lower=0.0, upper=100.0)
        result["distress_risk_score"] = score
    else:
        result["distress_risk_score"] = np.nan

    # Apply Financials mask -> NaN
    if fin_mask.any():
        for col in ["altman_z_trend", "z_score_volatility", "distress_risk_score"]:
            result.loc[fin_mask, col] = np.nan

    logger.info("Engineered financial distress features (Altman Z-Score)")
    return result

def engineer_cash_flow_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer cash flow quality and conversion metrics.

    Features computed (added when inputs exist):
    - cfo_to_net_income: CFO / Net Income (accruals quality)
    - fcf_to_net_income: FCF / Net Income
    - fcf_margin: FCF / Total Revenues (LTM)
    - cfo_growth_yoy: (CFO_LTM - CFO_1FY) / CFO_1FY
    - fcf_stability: Std deviation of available FCF periods (ltm, fy, 1fy)

    Notes:
    - Uses normalized column names (e.g., cfo_ltm, net_income_ltm, fcf_ltm, total_revenues_1fy)
    - Safe divisions via _safe_div; returns NaNs when inputs missing.
    """
    result = df.copy()

    # Core ratios
    if "cfo_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["cfo_to_net_income"] = _safe_div(df["cfo_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "net_income_ltm" in df.columns:
        result["fcf_to_net_income"] = _safe_div(df["fcf_ltm"], df["net_income_ltm"])
    if "fcf_ltm" in df.columns and "total_revenues_1fy" in df.columns:
        result["fcf_margin"] = _safe_div(df["fcf_ltm"], df["total_revenues_1fy"])

    # Growth YoY
    if "cfo_ltm" in df.columns and "cfo_1fy" in df.columns:
        result["cfo_growth_yoy"] = _safe_div(df["cfo_ltm"] - df["cfo_1fy"], df["cfo_1fy"])

    # Stability of FCF across available periods
    fcf_cols = [c for c in ("fcf_ltm", "fcf_fy", "fcf_1fy") if c in df.columns]
    if fcf_cols:
        fcf_mat = pd.concat([df[c].astype(float) for c in fcf_cols], axis=1)
        result["fcf_stability"] = fcf_mat.std(axis=1, ddof=0)

    logger.info("Engineered cash flow quality features")
    return result

def engineer_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer composite scores (quality, value, momentum) and keep within [0,100].

    Composite scores computed:
    - piotroski_f_score: 9-point fundamental strength score (0-9)
    - altman_z_score: Bankruptcy prediction score (higher is better)
    - beneish_m_score: Earnings manipulation detection (< -1.78 unlikely manipulator)
    - composite_quality_score: mean of available {distress_risk_score, accounting_quality_score}
    - momentum_score: normalized return_stability_score scaled to 0-100
    """
    result = df.copy()

    # Piotroski F-Score (0-9): 9 binary signals for fundamental strength
    f_score_components = []

    # F1: Positive ROA
    if "roa" in df.columns:
        f_score_components.append((df["roa"].fillna(0) > 0).astype(int))

    # F2: Positive Operating Cash Flow
    if "cfo_ltm" in df.columns:
        f_score_components.append((df["cfo_ltm"].fillna(0) > 0).astype(int))

    # F3: Change in ROA (positive)
    if "roa" in df.columns and "roa_previous_year" in df.columns:
        delta_roa = df["roa"].fillna(0) - df["roa_previous_year"].fillna(0)
        f_score_components.append((delta_roa > 0).astype(int))

    # F4: Quality of Earnings (CFO > Net Income)
    if "cfo_ltm" in df.columns and "net_income" in df.columns:
        f_score_components.append(
            (df["cfo_ltm"].fillna(0) > df["net_income"].fillna(0)).astype(int)
        )

    # F5: Decrease in Leverage (Long-term debt ratio)
    if "debt_to_equity" in df.columns:
        if "debt_to_equity_previous_year" in df.columns:
            delta_lev = df["debt_to_equity"].fillna(0) - df["debt_to_equity_previous_year"].fillna(0)
            f_score_components.append((delta_lev < 0).astype(int))
        else:
            f_score_components.append((df["debt_to_equity"].fillna(0) < 1.0).astype(int))

    # F6: Increase in Liquidity (Current Ratio)
    if "current_ratio" in df.columns:
        if "current_ratio_previous_year" in df.columns:
            delta_liq = df["current_ratio"].fillna(0) - df["current_ratio_previous_year"].fillna(0)
            f_score_components.append((delta_liq > 0).astype(int))
        else:
            f_score_components.append((df["current_ratio"].fillna(0) > 1.5).astype(int))

    # F7: No new equity issuance (shares outstanding decreased or stable)
    if "shares_outstanding" in df.columns and "shares_outstanding_previous_year" in df.columns:
        delta_shares = df["shares_outstanding"].fillna(0) - df["shares_outstanding_previous_year"].fillna(0)
        f_score_components.append((delta_shares <= 0).astype(int))

    # F8: Increase in Gross Margin
    if "gross_margin_pct" in df.columns:
        if "gross_margin_pct_previous_year" in df.columns:
            delta_margin = df["gross_margin_pct"].fillna(0) - df["gross_margin_pct_previous_year"].fillna(0)
            f_score_components.append((delta_margin > 0).astype(int))
        else:
            f_score_components.append((df["gross_margin_pct"].fillna(0) > 30).astype(int))

    # F9: Increase in Asset Turnover
    if "asset_turnover" in df.columns:
        if "asset_turnover_previous_year" in df.columns:
            delta_turn = df["asset_turnover"].fillna(0) - df["asset_turnover_previous_year"].fillna(0)
            f_score_components.append((delta_turn > 0).astype(int))
        else:
            f_score_components.append((df["asset_turnover"].fillna(0) > 0.5).astype(int))

    if f_score_components:
        result["piotroski_f_score"] = pd.concat(f_score_components, axis=1).sum(axis=1)

    # Altman Z-Score: Bankruptcy prediction model
    z_components = {}
    if "working_capital" in df.columns and "total_assets" in df.columns:
        z_components["x1"] = _safe_div(df["working_capital"], df["total_assets"]) * 1.2
    if "retained_earnings" in df.columns and "total_assets" in df.columns:
        z_components["x2"] = _safe_div(df["retained_earnings"], df["total_assets"]) * 1.4
    if "ebit" in df.columns and "total_assets" in df.columns:
        z_components["x3"] = _safe_div(df["ebit"], df["total_assets"]) * 3.3
    if "market_cap" in df.columns and "total_liabilities" in df.columns:
        z_components["x4"] = _safe_div(df["market_cap"], df["total_liabilities"]) * 0.6
    if "revenue" in df.columns and "total_assets" in df.columns:
        z_components["x5"] = _safe_div(df["revenue"], df["total_assets"]) * 1.0

    if z_components:
        result["altman_z_score"] = pd.concat(z_components.values(), axis=1).sum(axis=1)

    # Beneish M-Score: Earnings manipulation detection
    # (Simplified for now if all columns not available)
    if all(c in df.columns for c in ["cfo_ltm", "net_income", "total_assets"]):
        accruals = _safe_div(df["net_income"] - df["cfo_ltm"], df["total_assets"])
        result["beneish_m_score"] = accruals # This is just one component (TATA)

    # Composite scores
    scores_to_avg = []
    if "distress_risk_score" in result.columns:
        scores_to_avg.append(result["distress_risk_score"])
    if "accounting_quality_score" in result.columns:
        scores_to_avg.append(result["accounting_quality_score"])
    if scores_to_avg:
        result["composite_quality_score"] = pd.concat(scores_to_avg, axis=1).mean(axis=1)

    if "return_stability_score" in result.columns:
        # Scale return stability to 0-100 momentum score
        s = result["return_stability_score"]
        result["momentum_score"] = ((s - s.min()) / (s.max() - s.min() + 1e-9) * 100).clip(0, 100)

    logger.info("Engineered composite quality and fundamental scores")
    return result

def engineer_capital_allocation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer capital allocation efficiency and working capital metrics."""
    result = df.copy()

    # Reinvestment rate
    if all(c in df.columns for c in ["capex_ltm", "net_income_ltm"]):
        result["reinvestment_rate"] = _safe_div(-df["capex_ltm"], df["net_income_ltm"])

    # Cash conversion cycle (approximation)
    if all(c in df.columns for c in ["inventory_ltm", "receivables_ltm", "accounts_payable_ltm", "revenue_ltm"]):
        days_sales = _safe_div(df["receivables_ltm"], df["revenue_ltm"]) * 365
        days_inventory = _safe_div(df["inventory_ltm"], df["revenue_ltm"]) * 365 # Should use COGS
        days_payable = _safe_div(df["accounts_payable_ltm"], df["revenue_ltm"]) * 365
        result["cash_conversion_cycle"] = days_sales + days_inventory - days_payable

    logger.info("Engineered capital allocation features")
    return result
