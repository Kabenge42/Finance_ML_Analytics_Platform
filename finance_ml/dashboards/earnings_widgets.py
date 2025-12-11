import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List
from datetime import timedelta

# Schema-driven Phase 9.3 feature categorization (code_guidelines.md §9.3)
from finance_ml.ml_workflow.data.schema import PHASE93_FEATURE_INPUTS


def create_earnings_calendar_dashboard(
    df: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    top_n: int = 100,
    mode: str = "all",  # 'all', 'earnings', 'dividends'
) -> pd.DataFrame:
    """
    Creates a dashboard (styled DataFrame) for Earnings and Dividend Analytics.
    Filters for companies with upcoming or recent earnings (t +/- 10 days).

    **Phase 9.3 Schema-Driven Alignment (code_guidelines.md §9.3):**
    Uses PHASE93_FEATURE_INPUTS categories for metric selection:
    - Earnings metrics: profitability + growth + momentum categories
    - Dividend metrics: cash_flow category (dividend sustainability)
    - Additional domain-specific metrics supplementing Phase 9.3 categories

    Args:
        df: Input DataFrame containing stock data.
        reference_date: Date to compare next_earnings against. Defaults to today.
        top_n: Number of top companies (by Market Cap) to include.
        mode: 'all', 'earnings', or 'dividends' to filter displayed columns.

    Returns:
        pd.DataFrame: A styled DataFrame (or just the filtered DataFrame if styling is done in NB).
    """

    if reference_date is None:
        reference_date = pd.Timestamp.now()

    # Ensure date columns are datetime
    date_cols = ["next_earnings", "dividend_record_ex_date", "dividend_record_payable_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Filter logic: next_earnings within +/- 10 days
    if "next_earnings" not in df.columns:
        print("Warning: 'next_earnings' column not found. returning empty dataframe.")
        return pd.DataFrame()

    mask = (df["next_earnings"] - reference_date).abs() <= timedelta(days=10)
    filtered_df = df[mask].copy()

    # Sort by Market Cap (assuming 'market_cap' or similar exists)
    # Check for likely market cap columns
    mcap_col = None
    for col in ["market_cap", "market_cap_usd", "market_cap_curr"]:
        if col in df.columns:
            mcap_col = col
            break

    if mcap_col:
        filtered_df = filtered_df.sort_values(by=mcap_col, ascending=False)

    filtered_df = filtered_df.head(top_n)

    # Define columns to display
    # Identity
    display_cols = ["ticker", "sector", "next_earnings"]
    if mcap_col:
        display_cols.append(mcap_col)

    # Earnings Metrics (Schema-driven from Phase 9.3 categories - code_guidelines.md §9.3)
    # Combine profitability, valuation, growth, and momentum categories for earnings context
    earnings_candidates = (
        PHASE93_FEATURE_INPUTS.get("profitability", [])  # Margins, EBITDA, EBIT, net income
        + PHASE93_FEATURE_INPUTS.get("growth", [])  # Revenue CAGR, growth estimates
        + PHASE93_FEATURE_INPUTS.get("momentum", [])  # Price changes, returns
        + [
            # Additional earnings-specific metrics not in Phase 9.3 categories
            "net_income_adj_1fy",
            "ebitda_adj_fy",
            "ebitda_adj_1fy",
            "ebit_adj_1fy",
            "ebit_adj_fy",
            "net_income_adj_fy",
            "net_income_adj_fq",
            "net_income_adj_5yavgfq",
            "eps_adj_1fy",
            "eps_adj_fy",
            "eps_adj_ltm",
            "ebit_est_med_fy1e",
            "ebit_est_med_ntm",
            "eps_norm_est_avg_ntm",
            "eps_norm_est_avg_fy1e",
            "revenues_est_avg_ntm",
            "revenues_est_avg_fy1e",
            "revenues_est_med_ntm",
            "revenues_est_med_fy1e",
        ]
    )

    # Dividend Metrics (Schema-driven from Phase 9.3 categories - code_guidelines.md §9.3)
    # Include cash_flow category (dividend payment capacity) plus dividend-specific metrics
    dividend_candidates = PHASE93_FEATURE_INPUTS.get(
        "cash_flow", []
    ) + [  # CFO, FCF (dividend sustainability)
        # Dividend-specific metrics not in Phase 9.3 categories
        "dividend_record_announce_date",
        "dividend_record_ex_date",
        "dividend_record_payable_date",
        "dividend_record_record_date",
        "dividend_record_frequency",
        "dividend_record_currency",
        "dividend_record_amount",
        "dividend_streak",
        "div_yield_ltm",
        "div_yield_ind",
        "div_yield_1fyind",
        "div_yield_ttm",
        "div_yield_ntm",
        "div_yield_5yavgltm",
        "dividend_per_share_ltm",
        "dividend_per_share",
        "common_dividends_paid_ltm",
        "common_dividends_paid_fy",
        "dividends_paid",
        "dividends_paid_ltm",
    ]

    # Select existing columns based on mode
    final_cols = display_cols.copy()

    if mode in ["all", "earnings"]:
        existing_earnings_cols = [c for c in earnings_candidates if c in df.columns]
        final_cols.extend(existing_earnings_cols)

    if mode in ["all", "dividends"]:
        existing_dividend_cols = [c for c in dividend_candidates if c in df.columns]
        final_cols.extend(existing_dividend_cols)

    # Remove duplicates just in case
    final_cols = list(dict.fromkeys(final_cols))

    dashboard_df = filtered_df[final_cols].copy()

    # Add computed columns
    dashboard_df["days_to_earnings"] = (dashboard_df["next_earnings"] - reference_date).dt.days

    # Reorder: Put days_to_earnings near next_earnings
    cols = list(dashboard_df.columns)
    cols.remove("days_to_earnings")
    idx = cols.index("next_earnings") + 1
    cols.insert(idx, "days_to_earnings")
    dashboard_df = dashboard_df[cols]

    return dashboard_df


def display_earnings_dashboard(df: pd.DataFrame, mode: str = "all"):
    """
    Displays the earnings dashboard using Pandas Styler or similar.
    """
    dashboard_df = create_earnings_calendar_dashboard(df, mode=mode)

    if dashboard_df.empty:
        print("No companies found with earnings within +/- 10 days.")
        return

    # Basic styling
    format_dict = {
        "market_cap": "${:,.0f}",
        "next_earnings": "{:%Y-%m-%d}",
        "dividend_record_ex_date": "{:%Y-%m-%d}",
        "dividend_record_payable_date": "{:%Y-%m-%d}",
        "eps_adj_ltm": "${:.2f}",
        "eps_norm_est_avg_ntm": "${:.2f}",
        "div_yield_ltm": "{:.2%}",
        "div_yield_ntm": "{:.2%}",
        "one_day_pct": "{:+.2%}",
        "price_chg_pct_1m": "{:+.2%}",
        "price_chg_pct_3m": "{:+.2%}",
    }
    # Only format columns that exist
    format_dict = {k: v for k, v in format_dict.items() if k in dashboard_df.columns}

    styler = dashboard_df.style.format(format_dict, na_rep="-")

    # Highlight days_to_earnings
    def color_days(val):
        if pd.isna(val):
            return ""
        if val < 0:
            return "color: red"  # Past
        if val == 0:
            return "background-color: yellow; color: black"  # Today
        if val > 0:
            return "color: green"  # Future
        return ""

    styler = styler.map(color_days, subset=["days_to_earnings"])

    return styler
