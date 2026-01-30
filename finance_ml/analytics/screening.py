"""
Stock screening and filtering utilities for feature analytics.

This module provides functions for:
- Multi-factor stock screening
- Quality scoring and ranking
- Feature-based filtering
- Investment strategy screening
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def create_enhanced_screener(
    df: pd.DataFrame,
    min_fscore: int = 5,
    min_quality_momentum: float = 40,
    max_distress_risk: float = 70,
    min_eps_trajectory: float = 40,
    min_fcf_positive_years: int = 3,
    require_deleveraging: bool = False,
    require_secular_trend: bool = False,
    sector_filter: str = "All",
) -> pd.DataFrame:
    """
    Enhanced stock screener with multiple quality and momentum criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with stock features
    min_fscore : int, default 5
        Minimum Piotroski F-Score (0-9)
    min_quality_momentum : float, default 40
        Minimum quality momentum score
    max_distress_risk : float, default 70
        Maximum distress risk score (inverted: higher = safer)
    min_eps_trajectory : float, default 40
        Minimum EPS trajectory score
    min_fcf_positive_years : int, default 3
        Minimum FCF positive years (0-5)
    require_deleveraging : bool, default False
        Only stocks actively reducing debt
    require_secular_trend : bool, default False
        Only stocks in secular uptrend
    sector_filter : str, default 'All'
        Filter by sector ('All' for no filter)

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted by quality

    Examples
    --------
    >>> screened = create_enhanced_screener(df, min_fscore=7, require_deleveraging=True)
    >>> print(f"Found {len(screened)} high-quality stocks")
    """
    # Ensure necessary columns exist
    required_cols = [
        "piotroski_f_score",
        "distress_risk_score",
        "eps_trajectory_score",
        "fcf_positive_years",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return pd.DataFrame()

    # Apply filters
    mask = (
        (df["piotroski_f_score"] >= min_fscore)
        & (df["distress_risk_score"] >= (100 - max_distress_risk))
        & (df["eps_trajectory_score"] >= min_eps_trajectory)
        & (df["fcf_positive_years"] >= min_fcf_positive_years)
    )

    if require_deleveraging and "debt_deleveraging" in df.columns:
        mask &= df["debt_deleveraging"] == 1

    if require_secular_trend and "secular_trend_flag" in df.columns:
        mask &= df["secular_trend_flag"] == 1

    filtered = df[mask].copy()

    if sector_filter != "All":
        sector_col = "industry" if "industry" in filtered.columns else "sector"
        if sector_col in filtered.columns:
            filtered = filtered[filtered[sector_col] == sector_filter]

    # Sort by composite quality
    if "piotroski_f_score" in filtered.columns:
        filtered = filtered.sort_values("piotroski_f_score", ascending=False)

    return filtered


def screen_earnings_quality(
    df: pd.DataFrame,
    min_quality_score: float = 60,
    max_adjustment_pct: float = 20,
    require_positive_revisions: bool = False,
    min_positive_years: int = 3,
    sector_filter: str = "All",
) -> pd.DataFrame:
    """
    Screen stocks based on earnings quality criteria.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_quality_score : float, default 60
        Minimum earnings quality composite score (0-100)
    max_adjustment_pct : float, default 20
        Maximum absolute EPS adjustment percentage
    require_positive_revisions : bool, default False
        Only include stocks with positive GAAP revision flag
    min_positive_years : int, default 3
        Minimum net income positive years (0-5)
    sector_filter : str, default 'All'
        Filter by sector

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame sorted by earnings quality

    Examples
    --------
    >>> high_quality = screen_earnings_quality(df, min_quality_score=70)
    >>> print(f"Found {len(high_quality)} high-quality earnings stocks")
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if "earnings_quality_composite_comp" in df.columns:
        mask &= df["earnings_quality_composite_comp"] >= min_quality_score

    if "eps_adjustment_pct" in df.columns:
        mask &= df["eps_adjustment_pct"].abs() <= max_adjustment_pct

    if require_positive_revisions and "gaap_positive_revision_flag" in df.columns:
        mask &= df["gaap_positive_revision_flag"] == 1

    if "net_income_positive_years" in df.columns:
        mask &= df["net_income_positive_years"] >= min_positive_years

    if sector_filter != "All":
        sector_col = "industry" if "industry" in df.columns else "sector"
        if sector_col in df.columns:
            mask &= df[sector_col] == sector_filter

    result = df[mask].copy()

    if "earnings_quality_composite_comp" in result.columns:
        result = result.sort_values("earnings_quality_composite_comp", ascending=False)

    return result


def screen_value_opportunities(
    df: pd.DataFrame,
    max_pe_ratio: float = 25,
    min_upside_potential: float = 25,
    max_price_to_tangible_book: float = 2.0,
    min_quality_score: float = 50,
    require_positive_fcf: bool = True,
) -> pd.DataFrame:
    """
    Screen for value investment opportunities.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    max_pe_ratio : float, default 25
        Maximum P/E ratio
    min_upside_potential : float, default 25
        Minimum analyst upside potential (%)
    max_price_to_tangible_book : float, default 2.0
        Maximum price to tangible book ratio
    min_quality_score : float, default 50
        Minimum quality score
    require_positive_fcf : bool, default True
        Require positive free cash flow

    Returns
    -------
    pd.DataFrame
        Value opportunities sorted by upside potential

    Examples
    --------
    >>> value_stocks = screen_value_opportunities(df, max_pe_ratio=20)
    >>> print(f"Found {len(value_stocks)} value opportunities")
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if "p_e_ratio" in df.columns:
        mask &= (df["p_e_ratio"] > 0) & (df["p_e_ratio"] <= max_pe_ratio)

    if "upside_potential" in df.columns:
        mask &= df["upside_potential"] >= min_upside_potential

    if "price_to_tangible_book" in df.columns:
        mask &= (df["price_to_tangible_book"] > 0) & (
            df["price_to_tangible_book"] <= max_price_to_tangible_book
        )

    if "piotroski_f_score" in df.columns:
        mask &= df["piotroski_f_score"] >= min_quality_score / 10

    if require_positive_fcf and "fcf_yield" in df.columns:
        mask &= df["fcf_yield"] > 0

    result = df[mask].copy()

    if "upside_potential" in result.columns:
        result = result.sort_values("upside_potential", ascending=False)

    return result


def screen_growth_momentum(
    df: pd.DataFrame,
    min_revenue_growth: float = 10,
    min_eps_growth: float = 10,
    min_price_momentum_1y: float = 0,
    min_secular_trend_score: float = 60,
    require_rnd_investment: bool = False,
) -> pd.DataFrame:
    """
    Screen for growth and momentum stocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_revenue_growth : float, default 10
        Minimum revenue YoY growth (%)
    min_eps_growth : float, default 10
        Minimum EPS YoY growth (%)
    min_price_momentum_1y : float, default 0
        Minimum 1-year price momentum (%)
    min_secular_trend_score : float, default 60
        Minimum long-term trend score
    require_rnd_investment : bool, default False
        Require R&D investment

    Returns
    -------
    pd.DataFrame
        Growth stocks sorted by momentum

    Examples
    --------
    >>> growth_stocks = screen_growth_momentum(df, min_revenue_growth=15)
    >>> print(f"Found {len(growth_stocks)} growth stocks")
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if "revenue_growth_yoy" in df.columns:
        mask &= df["revenue_growth_yoy"] >= min_revenue_growth

    if "eps_yoy_growth" in df.columns:
        mask &= df["eps_yoy_growth"] >= min_eps_growth

    if "price_momentum_1y" in df.columns:
        mask &= df["price_momentum_1y"] >= min_price_momentum_1y

    if "long_term_trend_score" in df.columns:
        mask &= df["long_term_trend_score"] >= min_secular_trend_score

    if require_rnd_investment and "rnd_intensity_ltm" in df.columns:
        mask &= df["rnd_intensity_ltm"] > 0

    result = df[mask].copy()

    if "long_term_trend_score" in result.columns:
        result = result.sort_values("long_term_trend_score", ascending=False)

    return result


def screen_dividend_quality(
    df: pd.DataFrame,
    min_dividend_yield: float = 2.0,
    min_dividend_streak: int = 3,
    max_payout_ratio: float = 80,
    min_fcf_coverage: float = 1.2,
    require_dividend_growth: bool = True,
) -> pd.DataFrame:
    """
    Screen for quality dividend stocks.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_dividend_yield : float, default 2.0
        Minimum dividend yield (%)
    min_dividend_streak : int, default 3
        Minimum consecutive years of dividends
    max_payout_ratio : float, default 80
        Maximum dividend payout ratio (%)
    min_fcf_coverage : float, default 1.2
        Minimum FCF dividend coverage ratio
    require_dividend_growth : bool, default True
        Require expected dividend growth

    Returns
    -------
    pd.DataFrame
        Dividend stocks sorted by yield

    Examples
    --------
    >>> dividend_stocks = screen_dividend_quality(df, min_dividend_yield=3.0)
    >>> print(f"Found {len(dividend_stocks)} quality dividend stocks")
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if "dividend_yield_ltm" in df.columns:
        mask &= df["dividend_yield_ltm"] >= min_dividend_yield
    elif "dividend_yield" in df.columns:
        mask &= df["dividend_yield"] >= min_dividend_yield

    if "dividend_streak" in df.columns:
        mask &= df["dividend_streak"] >= min_dividend_streak

    if "dividend_payout_ratio" in df.columns:
        mask &= df["dividend_payout_ratio"] <= max_payout_ratio

    if "fcf_dividend_coverage" in df.columns:
        mask &= df["fcf_dividend_coverage"] >= min_fcf_coverage

    if require_dividend_growth and "dividend_growth_expectation" in df.columns:
        mask &= df["dividend_growth_expectation"] > 0

    result = df[mask].copy()

    # Sort by yield
    yield_col = "dividend_yield_ltm" if "dividend_yield_ltm" in result.columns else "dividend_yield"
    if yield_col in result.columns:
        result = result.sort_values(yield_col, ascending=False)

    return result


def screen_financial_health(
    df: pd.DataFrame,
    min_distress_score: float = 70,
    max_debt_to_equity: float = 1.0,
    min_current_ratio: float = 1.5,
    min_interest_coverage: float = 3.0,
    require_positive_wc: bool = True,
) -> pd.DataFrame:
    """
    Screen for financially healthy companies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    min_distress_score : float, default 70
        Minimum distress risk score (higher = safer)
    max_debt_to_equity : float, default 1.0
        Maximum debt-to-equity ratio
    min_current_ratio : float, default 1.5
        Minimum current ratio
    min_interest_coverage : float, default 3.0
        Minimum interest coverage ratio
    require_positive_wc : bool, default True
        Require positive working capital

    Returns
    -------
    pd.DataFrame
        Financially healthy stocks

    Examples
    --------
    >>> healthy_stocks = screen_financial_health(df, min_distress_score=80)
    >>> print(f"Found {len(healthy_stocks)} financially healthy stocks")
    """
    mask = pd.Series([True] * len(df), index=df.index)

    if "distress_risk_score" in df.columns:
        mask &= df["distress_risk_score"] >= min_distress_score

    if "debt_to_equity" in df.columns:
        mask &= df["debt_to_equity"] <= max_debt_to_equity

    if "current_ratio" in df.columns:
        mask &= df["current_ratio"] >= min_current_ratio

    if "interest_coverage_ratio" in df.columns:
        mask &= df["interest_coverage_ratio"] >= min_interest_coverage

    if require_positive_wc and "working_capital_ltm" in df.columns:
        mask &= df["working_capital_ltm"] > 0

    result = df[mask].copy()

    if "distress_risk_score" in result.columns:
        result = result.sort_values("distress_risk_score", ascending=False)

    return result


def rank_stocks_by_composite_score(
    df: pd.DataFrame, weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    Rank stocks by composite quality score with customizable weights.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    weights : dict, optional
        Dictionary of score weights. Default weights:
        - piotroski_f_score: 0.25
        - distress_risk_score: 0.25
        - earnings_quality_composite_comp: 0.25
        - cash_flow_quality_score: 0.25

    Returns
    -------
    pd.DataFrame
        DataFrame with composite_score column, sorted by score

    Examples
    --------
    >>> ranked = rank_stocks_by_composite_score(df)
    >>> top_10 = ranked.head(10)
    """
    if weights is None:
        weights = {
            "piotroski_f_score": 0.25,
            "distress_risk_score": 0.25,
            "earnings_quality_composite_comp": 0.25,
            "cash_flow_quality_score": 0.25,
        }

    result = df.copy()
    result["composite_score"] = 0

    for score_col, weight in weights.items():
        if score_col in result.columns:
            # Normalize to 0-100 scale
            if score_col == "piotroski_f_score":
                normalized = result[score_col] / 9 * 100
            else:
                normalized = result[score_col]

            result["composite_score"] += normalized.fillna(50) * weight

    result = result.sort_values("composite_score", ascending=False)

    return result


def create_sector_relative_ranking(
    df: pd.DataFrame, metric: str, sector_col: str = "industry"
) -> pd.DataFrame:
    """
    Create sector-relative rankings for a metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    metric : str
        Metric column name to rank
    sector_col : str, default 'industry'
        Sector grouping column

    Returns
    -------
    pd.DataFrame
        DataFrame with sector_rank and sector_percentile columns

    Examples
    --------
    >>> ranked = create_sector_relative_ranking(df, 'roe')
    >>> top_in_sector = ranked[ranked['sector_percentile'] > 75]
    """
    if metric not in df.columns or sector_col not in df.columns:
        return df

    result = df.copy()

    # Rank within sector
    result["sector_rank"] = result.groupby(sector_col)[metric].rank(ascending=False, method="min")

    # Calculate percentile within sector
    result["sector_percentile"] = result.groupby(sector_col)[metric].rank(pct=True) * 100

    return result
