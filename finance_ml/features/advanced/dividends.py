"""Dividend reliability and capital allocation features."""

from __future__ import annotations

import logging

import pandas as pd

from .utils import _safe_div

logger = logging.getLogger(__name__)

def engineer_dividend_reliability_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer dividend reliability and coverage features using LTM and FY snapshots.
    
    Features aligned with Phase 9.3 Category Registry:
    - dividend_streak: Number of consecutive years of dividend increases
    - dividend_payout_ratio: Common dividends paid / Net Income
    - fcf_dividend_coverage: Free Cash Flow / Common Dividends
    - dividend_yield_stability: Inverse of yield volatility
    - sustainable_dividend_flag: Boolean flag for coverage > 1.2x
    """
    result = df.copy()

    # 1. Core Dividend Reliability (Direct from Dataset)
    if "dividend_streak" in df.columns:
        result["dividend_streak"] = df["dividend_streak"]

    # 2. Payout and Coverage Ratios
    # Dividend Payout Ratio: Dividends Paid / Net Income
    if "common_dividends_paid_ltm" in df.columns and "net_income_adj_ltm" in df.columns:
        result["dividend_payout_ratio"] = _safe_div(
            df["common_dividends_paid_ltm"].abs(), df["net_income_adj_ltm"]
        )
        result["payout_consistency_score"] = 1.0 / (1.0 + result["dividend_payout_ratio"].fillna(0))

    # FCF Dividend Coverage: FCF / Dividends Paid
    if "fcf_ltm" in df.columns and "common_dividends_paid_ltm" in df.columns:
        result["fcf_dividend_coverage"] = _safe_div(
            df["fcf_ltm"], df["common_dividends_paid_ltm"].abs()
        )
        # Sustainable flag: Coverage > 1.2x and Positive FCF
        result["sustainable_dividend_flag"] = (
            (result["fcf_dividend_coverage"] > 1.2) & (df["fcf_ltm"] > 0)
        ).astype(int)

    # 3. Dividend Yield Stability (if multi-period yields exist)
    yield_cols = [c for c in df.columns if "div_yield" in c and "fy" in c]
    if len(yield_cols) >= 2:
        result["dividend_yield_stability"] = 1.0 / (1.0 + df[yield_cols].std(axis=1))

    # Placeholder logic updated to a composite reliability score
    if "dividend_streak" in result.columns:
        # Scale streak (0-25 years) to 0-100 score
        result["dividend_reliability_score"] = (result["dividend_streak"].clip(0, 25) / 25.0) * 100

    # 4. Buyback and Total Shareholder Yield
    if "buyback_yield_ltm" in df.columns:
        result["buyback_yield"] = pd.to_numeric(df["buyback_yield_ltm"], errors="coerce")

    # Use div_yield_ltm (source) or dividend_yield (computed)
    div_yield = None
    if "div_yield_ltm" in df.columns:
        div_yield = pd.to_numeric(df["div_yield_ltm"], errors="coerce")
    elif "dividend_yield" in df.columns:
        div_yield = pd.to_numeric(df["dividend_yield"], errors="coerce")

    if "buyback_yield" in result.columns and div_yield is not None:
        result["total_shareholder_yield"] = result["buyback_yield"] + div_yield

    # 5. Dividend Yield Term Structure
    if "div_yield_ntm" in df.columns and "div_yield_ltm" in df.columns:
        result["dividend_growth_expectation"] = pd.to_numeric(
            df["div_yield_ntm"], errors="coerce"
        ) - pd.to_numeric(df["div_yield_ltm"], errors="coerce")

    logger.info("Engineered dividend reliability features")
    return result
