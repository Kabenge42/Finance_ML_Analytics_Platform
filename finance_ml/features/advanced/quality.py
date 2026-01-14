"""Quality and risk-related feature engineering."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)


def _safe_bool_to_int(series: pd.Series, default: bool = False) -> pd.Series:
    """Safely convert a boolean series to int, filling NA values.

    Args:
        series: Boolean series that may contain NA values
        default: Value to use for NA entries (default False → 0)

    Returns:
        Integer series with 0/1 values, no NA
    """
    return series.fillna(default).astype(int)


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
        result["has_goodwill_impairment"] = _safe_bool_to_int(
            df["impairment_of_goodwill_ltm"] != 0
        )
        # Alias for compatibility with tests/plan wording
        result["goodwill_impairment_flag"] = result["has_goodwill_impairment"]

    # Asset writedown flag
    if "asset_writedown_ltm" in df.columns:
        result["has_asset_writedown"] = _safe_bool_to_int(
            df["asset_writedown_ltm"] != 0
        )

    # Restructuring charges flag
    if "restructuring_charges_ltm" in df.columns:
        result["has_restructuring"] = _safe_bool_to_int(
            df["restructuring_charges_ltm"] != 0
        )

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
            df["impairment_of_goodwill_ltm"].abs()
            + df["asset_writedown_ltm"].abs()
            + df["restructuring_charges_ltm"].abs()
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
            df["impairment_of_goodwill_1fy"].abs()
            + df["asset_writedown_1fy"].abs()
            + df["restructuring_charges_1fy"].abs()
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

    # Intangibles intensity - ENHANCED with gross intangibles fallback
    if "gross_intangible_assets_ltm" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["gross_intangible_assets_ltm"], df["total_assets_ltm"])
        result["intangibles_to_assets_pct"] = ratio * 100
        result["intangible_intensity"] = ratio
    elif "intangible_assets" in df.columns and "total_assets_ltm" in df.columns:
        ratio = _safe_div(df["intangible_assets"], df["total_assets_ltm"])
        result["intangibles_to_assets_pct"] = ratio * 100
        result["intangible_intensity"] = ratio

    # NEW: Intangibles trend (FY vs 5Y avg) - stability indicator
    if (
        "gross_intangible_assets_fy" in df.columns
        and "gross_intangible_assets_5yavgfq" in df.columns
    ):
        result["intangibles_vs_5y_avg"] = _safe_div(
            df["gross_intangible_assets_fy"],
            df["gross_intangible_assets_5yavgfq"].clip(lower=1),
        )
        # Flag companies with rapidly growing intangibles (potential acquisition spree)
        result["intangibles_growth_flag"] = _safe_bool_to_int(
            result["intangibles_vs_5y_avg"] > 1.5
        )

    # NEW: Goodwill concentration risk (multi-period analysis)
    if all(
        c in df.columns
        for c in ["goodwill_fq", "goodwill_fy", "goodwill_1fy", "goodwill_5yavgfq"]
    ):
        goodwill_cols = [df["goodwill_fq"], df["goodwill_fy"], df["goodwill_1fy"]]
        goodwill_volatility = pd.concat(goodwill_cols, axis=1).std(axis=1)
        result["goodwill_volatility"] = _safe_div(
            goodwill_volatility, df["goodwill_5yavgfq"].clip(lower=1)
        )

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
        result["asset_sale_boost"] = _safe_bool_to_int(
            df["gain_loss_on_sale_of_assets_ltm"] > 0
        )

    # NEW: SG&A efficiency trend (cost discipline indicator)
    if all(
        c in df.columns
        for c in [
            "selling_general_and_admin_expenses_total_fq",
            "selling_general_and_admin_expenses_total_5yavgfq",
            "total_revenues_fq",
            "total_revenues_5yavgfq",
        ]
    ):
        current_sga_ratio = _safe_div(
            df["selling_general_and_admin_expenses_total_fq"], df["total_revenues_fq"]
        )
        avg_sga_ratio = _safe_div(
            df["selling_general_and_admin_expenses_total_5yavgfq"],
            df["total_revenues_5yavgfq"],
        )
        result["sga_efficiency_trend"] = _safe_div(
            current_sga_ratio - avg_sga_ratio, avg_sga_ratio
        )
        # Negative = improving cost discipline
        result["improving_cost_discipline"] = _safe_bool_to_int(
            result["sga_efficiency_trend"] < -0.05
        )

    # NEW: Marketing intensity (discretionary spending quality signal)
    if "marketing_expenses_fy" in df.columns and "total_revenues_fy" in df.columns:
        result["marketing_intensity"] = _safe_div(
            df["marketing_expenses_fy"], df["total_revenues_fy"]
        )
    if (
        "marketing_expenses_5yavgltm" in df.columns
        and "marketing_expenses_fy" in df.columns
    ):
        result["marketing_vs_5y_avg"] = _safe_div(
            df["marketing_expenses_fy"], df["marketing_expenses_5yavgltm"].clip(lower=1)
        )

    # ENHANCED: Accounting quality score with more granular components
    # High exceptional items, high goodwill, presence of impairments = lower quality
    quality_components = []
    penalty_weights = {}

    if "has_goodwill_impairment" in result.columns:
        quality_components.append(result["has_goodwill_impairment"].fillna(0) * 25)
        penalty_weights["goodwill_impairment"] = 25
    if "has_asset_writedown" in result.columns:
        quality_components.append(result["has_asset_writedown"].fillna(0) * 10)
        penalty_weights["asset_writedown"] = 10
    if "has_restructuring" in result.columns:
        quality_components.append(result["has_restructuring"].fillna(0) * 15)
        penalty_weights["restructuring"] = 15
    if "exceptional_items_to_ni_pct" in result.columns:
        quality_components.append(
            _safe_bool_to_int(result["exceptional_items_to_ni_pct"] > 10) * 15
        )
        penalty_weights["high_exceptional_items"] = 15
    if "goodwill_to_assets_pct" in result.columns:
        quality_components.append(
            _safe_bool_to_int(result["goodwill_to_assets_pct"] > 30) * 15
        )
        penalty_weights["high_goodwill"] = 15
    # NEW: Penalize volatile goodwill
    if "goodwill_volatility" in result.columns:
        quality_components.append(
            _safe_bool_to_int(result["goodwill_volatility"] > 0.3) * 10
        )
        penalty_weights["volatile_goodwill"] = 10
    # NEW: Reward improving cost discipline
    if "improving_cost_discipline" in result.columns:
        quality_components.append(
            -result["improving_cost_discipline"].fillna(0) * 10
        )  # Bonus
        penalty_weights["cost_discipline_bonus"] = -10

    if quality_components:
        total_penalties = sum(quality_components)
        result["accounting_quality_score"] = (100 - total_penalties).clip(lower=0, upper=100)

    # Historical exceptional items pattern (5Y avg baseline)
    EXCEPTIONAL_5Y_COLS = {
        "impairment_of_goodwill_5yavgfq": "impairment_of_goodwill_fq",
        "asset_writedown_5yavgfq": "asset_writedown_fq",
        "restructuring_charges_5yavgfq": "restructuring_charges_fq",
        "merger_and_restructuring_charges_5yavgfq": "merger_and_restructuring_charges_fq",
    }

    for avg_col, current_col in EXCEPTIONAL_5Y_COLS.items():
        if avg_col in df.columns and current_col in df.columns:
            feature_name = avg_col.replace("_5yavgfq", "_vs_5y_avg")
            result[feature_name] = _safe_div(
                df[current_col].abs(), df[avg_col].abs().clip(lower=1)
            )

    # Other unusual items impact
    if "other_unusual_items_total_ltm" in df.columns and "ebitda_ltm" in df.columns:
        result["other_unusual_to_ebitda"] = _safe_div(
            df["other_unusual_items_total_ltm"].abs(), df["ebitda_ltm"].abs()
        )

    # Composite exceptional items frequency score
    exceptional_flags = []
    for col in [
        "impairment_of_goodwill_fq",
        "asset_writedown_fq",
        "restructuring_charges_fq",
    ]:
        if col in df.columns:
            exceptional_flags.append((df[col].abs() > 0).astype(int))
    if exceptional_flags:
        result["exceptional_items_frequency"] = sum(exceptional_flags)

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

    # NEW: Liquidity stress indicator (combines Z-score with current ratio)
    if "current_ratio_ltm" in df.columns:
        current_ratio = df["current_ratio_ltm"].astype(float)
        liquidity_stress = pd.Series(0.0, index=result.index, dtype=float)

        # Liquidity stress levels
        liquidity_stress = np.where(
            current_ratio < 1.0, 30.0, liquidity_stress
        )  # Severe
        liquidity_stress = np.where(
            (current_ratio >= 1.0) & (current_ratio < 1.5), 15.0, liquidity_stress
        )  # Moderate
        result["liquidity_stress_score"] = pd.Series(
            liquidity_stress, index=result.index
        )

        # Combined distress indicator
        if "distress_risk_score" in result.columns:
            result["combined_distress_score"] = (
                result["distress_risk_score"] * 0.7
                + (100 - result["liquidity_stress_score"]) * 0.3
            ).clip(0, 100)

    # NEW: Working capital trend (early warning signal)
    if all(
        c in df.columns
        for c in ["working_capital_fq", "working_capital_fy", "working_capital_5yavgfy"]
    ):
        wc_trend = _safe_div(
            df["working_capital_fq"] - df["working_capital_fy"],
            df["working_capital_fy"].abs().clip(lower=1),
        )
        result["working_capital_trend"] = wc_trend
        # Deteriorating working capital flag
        result["wc_deteriorating_flag"] = _safe_bool_to_int(wc_trend < -0.2)

        # Working capital vs 5Y average
        result["wc_vs_5y_avg"] = _safe_div(
            df["working_capital_fq"], df["working_capital_5yavgfy"]
        )

    # NEW: Retained earnings trajectory (long-term solvency signal)
    if all(
        c in df.columns
        for c in [
            "retained_earnings_fq",
            "retained_earnings_fy",
            "retained_earnings_5yavgfq",
        ]
    ):
        re_growth = _safe_div(
            df["retained_earnings_fq"] - df["retained_earnings_fy"],
            df["retained_earnings_fy"].abs().clip(lower=1),
        )
        result["retained_earnings_growth"] = re_growth

        # Negative retained earnings flag (accumulated deficit)
        result["accumulated_deficit_flag"] = _safe_bool_to_int(
            df["retained_earnings_fq"] < 0
        )

    # NEW: Cash buffer adequacy (survival runway)
    if all(
        c in df.columns
        for c in ["cash_and_equivalents_fq", "total_operating_expenses_ltm"]
    ):
        # Months of runway = Cash / (Monthly OpEx)
        monthly_opex = df["total_operating_expenses_ltm"] / 12
        result["cash_runway_months"] = _safe_div(
            df["cash_and_equivalents_fq"], monthly_opex.clip(lower=1)
        )
        result["adequate_cash_buffer"] = _safe_bool_to_int(
            result["cash_runway_months"] > 6
        )

    # ENHANCED: Distress risk score with liquidity components
    risk_components = []
    if z_components:
        z_mean = pd.concat(z_components, axis=1).mean(axis=1)
        z_score_contrib = ((z_mean - 1.8) / (3.0 - 1.8) * 60.0).clip(
            lower=0.0, upper=60.0
        )
        risk_components.append(z_score_contrib)

    if "liquidity_stress_score" in result.columns:
        risk_components.append((30 - result["liquidity_stress_score"]))  # Max 30 points

    if "adequate_cash_buffer" in result.columns:
        risk_components.append(result["adequate_cash_buffer"] * 10)  # Max 10 points

    if risk_components:
        result["distress_risk_score_enhanced"] = sum(risk_components).clip(0, 100)

    # Apply Financials mask -> NaN
    distress_cols = [
        "altman_z_trend",
        "z_score_volatility",
        "distress_risk_score",
        "liquidity_stress_score",
        "combined_distress_score",
        "distress_risk_score_enhanced",
    ]
    if fin_mask.any():
        for col in distress_cols:
            if col in result.columns:
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
    - cfo_quarterly_volatility, cfo_consistency_score: Quarterly CFO analysis
    - fcf_quarterly_volatility, fcf_4q_improvement: FCF quarterly trends
    - fcf_5y_stability, fcf_positive_years: Multi-year FCF stability
    - acquisition_intensity_4q, acquisition_to_fcf: M&A activity analysis
    - capex_vs_5y_avg, underinvestment_flag: CapEx efficiency
    - cfo_share_of_cf, self_funding_ratio: Cash flow composition
    - cash_flow_quality_score: Composite quality score

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

    # ENHANCED: FCF margin using current period revenues
    if "fcf_ltm" in df.columns and "total_revenues_ltm" in df.columns:
        result["fcf_margin"] = _safe_div(df["fcf_ltm"], df["total_revenues_ltm"])
    elif "fcf_ltm" in df.columns and "total_revenues_1fy" in df.columns:
        result["fcf_margin"] = _safe_div(df["fcf_ltm"], df["total_revenues_1fy"])

    # Growth YoY
    if "cfo_ltm" in df.columns and "cfo_1fy" in df.columns:
        result["cfo_growth_yoy"] = _safe_div(df["cfo_ltm"] - df["cfo_1fy"], df["cfo_1fy"])

    # Stability of FCF across available periods
    fcf_cols = [c for c in ("fcf_ltm", "fcf_fy", "fcf_1fy") if c in df.columns]
    if fcf_cols:
        fcf_mat = pd.concat([df[c].astype(float) for c in fcf_cols], axis=1)
        result["fcf_stability"] = fcf_mat.std(axis=1, ddof=0)

    # NEW: Quarterly CFO consistency (uses -1FQFQ to -4FQFQ columns)
    cfo_quarterly_cols = [
        c
        for c in ["cfo_fq", "cfo_1fqfq", "cfo_2fqfq", "cfo_3fqfq", "cfo_4fqfq"]
        if c in df.columns
    ]
    if len(cfo_quarterly_cols) >= 3:
        cfo_mat = pd.concat([df[c].astype(float) for c in cfo_quarterly_cols], axis=1)
        result["cfo_quarterly_volatility"] = cfo_mat.std(axis=1, ddof=0)
        result["cfo_quarterly_mean"] = cfo_mat.mean(axis=1)
        result["cfo_consistency_score"] = (
            1
            - _safe_div(
                result["cfo_quarterly_volatility"],
                result["cfo_quarterly_mean"].abs().clip(lower=1),
            )
        ).clip(0, 1) * 100

        # Positive CFO streak (consecutive positive quarters)
        positive_quarters = (cfo_mat > 0).sum(axis=1)
        result["cfo_positive_streak"] = positive_quarters

    # NEW: FCF quarterly trend analysis
    fcf_quarterly_cols = [
        c
        for c in ["fcf_fq", "fcf_1fqfq", "fcf_2fqfq", "fcf_3fqfq", "fcf_4fqfq"]
        if c in df.columns
    ]
    if len(fcf_quarterly_cols) >= 3:
        fcf_q_mat = pd.concat([df[c].astype(float) for c in fcf_quarterly_cols], axis=1)
        result["fcf_quarterly_volatility"] = fcf_q_mat.std(axis=1, ddof=0)

        # FCF improvement trend (most recent vs oldest in sequence)
        if "fcf_fq" in df.columns and "fcf_4fqfq" in df.columns:
            result["fcf_4q_improvement"] = _safe_div(
                df["fcf_fq"] - df["fcf_4fqfq"], df["fcf_4fqfq"].abs().clip(lower=1)
            )

    # NEW: Multi-year FCF stability (uses -2FY, -3FY, -4FY)
    fcf_annual_cols = [
        c
        for c in ["fcf_fy", "fcf_1fy", "fcf_2fy", "fcf_3fy", "fcf_4fy"]
        if c in df.columns
    ]
    if len(fcf_annual_cols) >= 3:
        fcf_annual_mat = pd.concat(
            [df[c].astype(float) for c in fcf_annual_cols], axis=1
        )
        result["fcf_5y_stability"] = fcf_annual_mat.std(axis=1, ddof=0)
        result["fcf_5y_mean"] = fcf_annual_mat.mean(axis=1)

        # Consistent positive FCF years
        positive_years = (fcf_annual_mat > 0).sum(axis=1)
        result["fcf_positive_years"] = positive_years
        result["fcf_always_positive"] = _safe_bool_to_int(
            positive_years == len(fcf_annual_cols)
        )

    # NEW: Acquisition activity analysis (uses Cash Acquisitions columns)
    acq_cols = [
        c
        for c in [
            "cash_acquisitions_fq",
            "cash_acquisitions_1fqfq",
            "cash_acquisitions_2fqfq",
            "cash_acquisitions_3fqfq",
            "cash_acquisitions_4fqfq",
        ]
        if c in df.columns
    ]
    if acq_cols:
        acq_mat = pd.concat([df[c].abs().astype(float) for c in acq_cols], axis=1)
        result["acquisition_intensity_4q"] = acq_mat.sum(axis=1)

        # Acquisition to FCF ratio (sustainability of M&A)
        if "fcf_ltm" in df.columns:
            result["acquisition_to_fcf"] = _safe_div(
                result["acquisition_intensity_4q"], df["fcf_ltm"].abs()
            )
            result["sustainable_ma_flag"] = _safe_bool_to_int(
                result["acquisition_to_fcf"] < 0.5
            )

    # Trend vs 5Y average
    if all(
        c in df.columns for c in ["cash_acquisitions_fq", "cash_acquisitions_5yavgfq"]
    ):
        result["acquisition_vs_5y_avg"] = _safe_div(
            df["cash_acquisitions_fq"].abs(),
            df["cash_acquisitions_5yavgfq"].abs().clip(lower=1),
        )

    # NEW: CapEx efficiency and investment quality
    if all(
        c in df.columns
        for c in ["capital_expenditure_fq", "capital_expenditure_5yavgfq"]
    ):
        result["capex_vs_5y_avg"] = _safe_div(
            df["capital_expenditure_fq"].abs(),
            df["capital_expenditure_5yavgfq"].abs().clip(lower=1),
        )
        # Underinvestment flag (CapEx significantly below historical)
        result["underinvestment_flag"] = _safe_bool_to_int(
            result["capex_vs_5y_avg"] < 0.7
        )

    # NEW: Cash flow composition analysis (CFO vs CFI vs CFF balance)
    if all(c in df.columns for c in ["cfo_ltm", "cfi_ltm", "cff_ltm"]):
        total_cf = df["cfo_ltm"].abs() + df["cfi_ltm"].abs() + df["cff_ltm"].abs()
        result["cfo_share_of_cf"] = _safe_div(df["cfo_ltm"].abs(), total_cf)
        result["cfi_share_of_cf"] = _safe_div(df["cfi_ltm"].abs(), total_cf)
        result["cff_share_of_cf"] = _safe_div(df["cff_ltm"].abs(), total_cf)

        # Self-funding ratio (CFO covers CFI needs)
        result["self_funding_ratio"] = _safe_div(df["cfo_ltm"], df["cfi_ltm"].abs())
        result["self_funding_flag"] = _safe_bool_to_int(
            result["self_funding_ratio"] > 1
        )

    # NEW: Composite cash flow quality score
    cf_quality_components = []
    if "cfo_to_net_income" in result.columns:
        # CFO/NI > 1 is good (high accrual quality)
        cf_quality_components.append(
            _safe_bool_to_int(result["cfo_to_net_income"] > 1) * 25
        )
    if "fcf_always_positive" in result.columns:
        cf_quality_components.append(result["fcf_always_positive"] * 25)
    if "cfo_positive_streak" in result.columns and len(cfo_quarterly_cols) > 0:
        cf_quality_components.append(
            (result["cfo_positive_streak"] / len(cfo_quarterly_cols) * 25).clip(0, 25)
        )
    if "self_funding_flag" in result.columns:
        cf_quality_components.append(result["self_funding_flag"] * 25)

    if cf_quality_components:
        result["cash_flow_quality_score"] = sum(cf_quality_components).clip(0, 100)

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

    # ENHANCED: Piotroski F-Score using available columns
    f_score_components = []

    # F1: Positive ROA (use ROA % LTM)
    if "return_on_assets_roa_pct_ltm" in df.columns:
        f_score_components.append(
            _safe_bool_to_int(df["return_on_assets_roa_pct_ltm"].fillna(0) > 0)
        )
    elif "roa" in df.columns:
        f_score_components.append(_safe_bool_to_int(df["roa"].fillna(0) > 0))

    # F2: Positive Operating Cash Flow
    if "cfo_ltm" in df.columns:
        f_score_components.append(_safe_bool_to_int(df["cfo_ltm"] > 0))

    # F3: Change in ROA (FY vs -1FY)
    if (
        "return_on_assets_roa_pct_fy" in df.columns
        and "return_on_assets_roa_pct_1fy" in df.columns
    ):
        delta_roa = (
            df["return_on_assets_roa_pct_fy"] - df["return_on_assets_roa_pct_1fy"]
        )
        f_score_components.append(_safe_bool_to_int(delta_roa > 0))
    elif "roa" in df.columns and "roa_previous_year" in df.columns:
        delta_roa = df["roa"].fillna(0) - df["roa_previous_year"].fillna(0)
        f_score_components.append(_safe_bool_to_int(delta_roa > 0))

    # F4: Quality of Earnings (CFO > Net Income)
    if "cfo_ltm" in df.columns and "net_income_is_ltm" in df.columns:
        f_score_components.append(
            _safe_bool_to_int(df["cfo_ltm"] > df["net_income_is_ltm"])
        )
    elif "cfo_ltm" in df.columns and "net_income" in df.columns:
        f_score_components.append(_safe_bool_to_int(df["cfo_ltm"] > df["net_income"]))

    # F5: Decrease in Leverage (Total Debt / Total Equity)
    if "total_debt_ltm" in df.columns and "total_equity_ltm" in df.columns:
        current_leverage = _safe_div(df["total_debt_ltm"], df["total_equity_ltm"])
        if "total_debt_fy" in df.columns and "total_equity_fy" in df.columns:
            prev_leverage = _safe_div(df["total_debt_fy"], df["total_equity_fy"])
            f_score_components.append(
                _safe_bool_to_int(current_leverage < prev_leverage)
            )
        else:
            f_score_components.append(_safe_bool_to_int(current_leverage < 1.0))
    elif "debt_to_equity" in df.columns:
        if "debt_to_equity_previous_year" in df.columns:
            delta_lev = df["debt_to_equity"].fillna(0) - df[
                "debt_to_equity_previous_year"
            ].fillna(0)
            f_score_components.append(_safe_bool_to_int(delta_lev < 0))
        else:
            f_score_components.append(
                _safe_bool_to_int(df["debt_to_equity"].fillna(0) < 1.0)
            )

    # F6: Increase in Liquidity (Current Ratio LTM vs FY)
    if "current_ratio_ltm" in df.columns:
        if "current_ratio_fy" in df.columns:
            f_score_components.append(
                _safe_bool_to_int(df["current_ratio_ltm"] > df["current_ratio_fy"])
            )
        else:
            f_score_components.append(_safe_bool_to_int(df["current_ratio_ltm"] > 1.5))
    elif "current_ratio" in df.columns:
        if "current_ratio_previous_year" in df.columns:
            delta_liq = df["current_ratio"].fillna(0) - df["current_ratio_previous_year"].fillna(0)
            f_score_components.append(_safe_bool_to_int(delta_liq > 0))
        else:
            f_score_components.append(
                _safe_bool_to_int(df["current_ratio"].fillna(0) > 1.5)
            )

    # F7: No new equity issuance (Shares Out vs -1FY)
    if "shrs_out" in df.columns and "shrs_out_1fy" in df.columns:
        delta_shares = df["shrs_out"] - df["shrs_out_1fy"]
        f_score_components.append(_safe_bool_to_int(delta_shares <= 0))
    elif (
        "shares_outstanding" in df.columns
        and "shares_outstanding_previous_year" in df.columns
    ):
        delta_shares = df["shares_outstanding"] - df["shares_outstanding_previous_year"]
        f_score_components.append(_safe_bool_to_int(delta_shares <= 0))

    # F8: Increase in Gross Margin
    if (
        "gross_profit_margin_pct_ltm" in df.columns
        and "gross_profit_margin_pct_fy" in df.columns
    ):
        delta_margin = (
            df["gross_profit_margin_pct_ltm"] - df["gross_profit_margin_pct_fy"]
        )
        f_score_components.append(_safe_bool_to_int(delta_margin > 0))
    elif "gross_profit_margin_pct_ltm" in df.columns:
        f_score_components.append(
            _safe_bool_to_int(df["gross_profit_margin_pct_ltm"] > 30)
        )
    elif "gross_margin_pct" in df.columns:
        if "gross_margin_pct_previous_year" in df.columns:
            delta_margin = df["gross_margin_pct"].fillna(0) - df[
                "gross_margin_pct_previous_year"
            ].fillna(0)
            f_score_components.append(_safe_bool_to_int(delta_margin > 0))
        else:
            f_score_components.append(
                _safe_bool_to_int(df["gross_margin_pct"].fillna(0) > 30)
            )

    # F9: Increase in Asset Turnover
    if "asset_turnover_ltm" in df.columns and "asset_turnover_fy" in df.columns:
        f_score_components.append(
            _safe_bool_to_int(df["asset_turnover_ltm"] > df["asset_turnover_fy"])
        )
    elif "asset_turnover_ltm" in df.columns:
        f_score_components.append(_safe_bool_to_int(df["asset_turnover_ltm"] > 0.5))
    elif "asset_turnover" in df.columns:
        if "asset_turnover_previous_year" in df.columns:
            delta_turn = df["asset_turnover"].fillna(0) - df[
                "asset_turnover_previous_year"
            ].fillna(0)
            f_score_components.append(_safe_bool_to_int(delta_turn > 0))
        else:
            f_score_components.append(
                _safe_bool_to_int(df["asset_turnover"].fillna(0) > 0.5)
            )

    if f_score_components:
        result["piotroski_f_score"] = pd.concat(f_score_components, axis=1).sum(axis=1)

    # NEW: EPS trajectory score (using historical EPS columns)
    eps_cols = [
        c
        for c in [
            "net_eps_basic_fy",
            "net_eps_basic_1fy",
            "net_eps_basic_2fy",
            "net_eps_basic_3fy",
            "net_eps_basic_4fy",
            "net_eps_basic_5fy",
        ]
        if c in df.columns
    ]
    if len(eps_cols) >= 3:
        eps_mat = pd.concat([df[c].astype(float) for c in eps_cols], axis=1)

        # EPS growth consistency (count of YoY increases)
        eps_changes = eps_mat.diff(axis=1).iloc[:, 1:]  # Skip first NaN column
        result["eps_improvement_count"] = (eps_changes > 0).sum(axis=1)
        result["eps_trajectory_score"] = (
            result["eps_improvement_count"] / (len(eps_cols) - 1) * 100
        ).clip(0, 100)

        # EPS stability (coefficient of variation)
        eps_mean = eps_mat.mean(axis=1)
        eps_std = eps_mat.std(axis=1)
        result["eps_stability"] = (
            1 - _safe_div(eps_std, eps_mean.abs().clip(lower=0.01))
        ).clip(0, 1)

    # NEW: Dilution score (share count trajectory)
    if "shrs_out" in df.columns and "shrs_out_1fy" in df.columns:
        share_growth = _safe_div(
            df["shrs_out"] - df["shrs_out_1fy"], df["shrs_out_1fy"]
        )
        result["share_dilution_rate"] = share_growth
        # Score: 100 = buyback, 0 = heavy dilution
        result["dilution_score"] = (50 - share_growth * 100).clip(0, 100)

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
        result["beneish_m_score"] = accruals  # This is just one component (TATA)

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
    if all(
        c in df.columns
        for c in ["inventory_ltm", "receivables_ltm", "accounts_payable_ltm", "revenue_ltm"]
    ):
        days_sales = _safe_div(df["receivables_ltm"], df["revenue_ltm"]) * 365
        days_inventory = _safe_div(df["inventory_ltm"], df["revenue_ltm"]) * 365  # Should use COGS
        days_payable = _safe_div(df["accounts_payable_ltm"], df["revenue_ltm"]) * 365
        result["cash_conversion_cycle"] = days_sales + days_inventory - days_payable

    logger.info("Engineered capital allocation features")
    return result


def engineer_earnings_quality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer earnings quality and estimate accuracy features.

    Leverages:
    - EPS actual vs estimates (beat/miss patterns)
    - GAAP vs Adjusted EPS divergence
    - EPS revision momentum
    - Revenue estimate accuracy

    Features computed:
    - eps_surprise, eps_surprise_pct, eps_beat_flag: EPS beat/miss analysis
    - gaap_adj_eps_gap, gaap_adj_eps_gap_pct, large_adjustment_flag: GAAP vs Adjusted divergence
    - eps_revision_momentum, positive_revision_trend, revision_positive_count: Revision momentum
    - revenue_surprise_pct, revenue_beat_flag: Revenue estimate accuracy
    - earnings_quality_composite: Composite score (0-100)

    Returns:
        DataFrame with earnings quality features added
    """
    result = df.copy()

    # EPS beat/miss analysis (actual vs estimates)
    if "net_eps_basic_fq" in df.columns and "eps_norm_est_avg_fy1e" in df.columns:
        eps_surprise = df["net_eps_basic_fq"] - df["eps_norm_est_avg_fy1e"]
        result["eps_surprise"] = eps_surprise
        result["eps_surprise_pct"] = (
            _safe_div(eps_surprise, df["eps_norm_est_avg_fy1e"].abs()) * 100
        )
        result["eps_beat_flag"] = _safe_bool_to_int(eps_surprise > 0)

    # GAAP vs Adjusted EPS divergence (earnings quality signal)
    if "eps_gaap_est_avg_fy1e" in df.columns and "eps_norm_est_avg_fy1e" in df.columns:
        gaap_adj_diff = df["eps_gaap_est_avg_fy1e"] - df["eps_norm_est_avg_fy1e"]
        result["gaap_adj_eps_gap"] = gaap_adj_diff
        result["gaap_adj_eps_gap_pct"] = (
            _safe_div(gaap_adj_diff, df["eps_norm_est_avg_fy1e"].abs()) * 100
        )
        # Large gap = potential earnings quality concern
        result["large_adjustment_flag"] = _safe_bool_to_int(
            result["gaap_adj_eps_gap_pct"].abs() > 20
        )

    # EPS revision momentum (1W, 1M, 3M, 6M, 1Y revisions)
    revision_cols = {
        "eps_est_avg_rev_pct_fy1e_1w": 0.3,  # Most recent, highest weight
        "eps_est_avg_rev_pct_fy1e_1m": 0.25,
        "eps_est_avg_rev_pct_fy1e_3m": 0.2,
        "eps_est_avg_rev_pct_fy1e_6m": 0.15,
        "eps_est_avg_rev_pct_fy1e_1y": 0.1,
    }
    available_revisions = {k: v for k, v in revision_cols.items() if k in df.columns}

    if available_revisions:
        revision_values = []
        weights = []
        for col, weight in available_revisions.items():
            revision_values.append(df[col].fillna(0) * weight)
            weights.append(weight)

        result["eps_revision_momentum"] = sum(revision_values) / sum(weights)
        result["positive_revision_trend"] = _safe_bool_to_int(
            result["eps_revision_momentum"] > 0
        )

        # Count of positive revisions across timeframes
        revision_positives = [
            _safe_bool_to_int(df[col] > 0) for col in available_revisions.keys()
        ]
        result["revision_positive_count"] = sum(revision_positives)

    # Revenue estimate accuracy
    if "total_revenues_ltm" in df.columns and "revenues_est_avg_fy1e" in df.columns:
        revenue_surprise = (
            _safe_div(
                df["total_revenues_ltm"] - df["revenues_est_avg_fy1e"],
                df["revenues_est_avg_fy1e"],
            )
            * 100
        )
        result["revenue_surprise_pct"] = revenue_surprise
        result["revenue_beat_flag"] = _safe_bool_to_int(revenue_surprise > 0)

    # Composite earnings quality score
    eq_components = []
    if "eps_beat_flag" in result.columns:
        eq_components.append(result["eps_beat_flag"] * 25)
    if "large_adjustment_flag" in result.columns:
        eq_components.append(
            (1 - result["large_adjustment_flag"]) * 25
        )  # Penalty for large adjustments
    if "positive_revision_trend" in result.columns:
        eq_components.append(result["positive_revision_trend"] * 25)
    if "revenue_beat_flag" in result.columns:
        eq_components.append(result["revenue_beat_flag"] * 25)

    if eq_components:
        result["earnings_quality_composite"] = sum(eq_components).clip(0, 100)

    logger.info("Engineered earnings quality features")
    return result
