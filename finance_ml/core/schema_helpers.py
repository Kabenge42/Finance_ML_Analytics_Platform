"""Schema helper utilities for notebook usage.

Provides convenient wrappers around COLUMN_SCHEMA and PHASE93_FEATURE_CATEGORIES.
"""

from typing import Optional

import pandas as pd

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    list_required_schema_columns_for_etl,
    list_non_recurring_cols,
    list_knn_imputable_cols,
    list_count_cols,
    list_price_cols,
    normalize_column_name,
    get_sql_column_name,
)


def get_columns_by_role(role: str) -> list[str]:
    """Get all columns with a specific role."""
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == role]


def get_columns_by_dtype(dtype: str) -> list[str]:
    """Get all columns with a specific dtype."""
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("dtype") == dtype]


def get_features_for_category(category: str) -> list[str]:
    """Get Phase 9.3 features for a specific category."""
    return PHASE93_FEATURE_CATEGORIES.get(category, [])


def get_all_feature_categories() -> list[str]:
    """Get all available Phase 9.3 feature category names."""
    return list(PHASE93_FEATURE_CATEGORIES.keys())


def validate_column_exists(col_name: str) -> bool:
    """Check if column exists in canonical schema."""
    return col_name in COLUMN_SCHEMA


def get_analyst_rating_columns() -> dict[str, str]:
    """Get analyst rating columns from canonical schema."""
    rating_mapping = {
        "strong_sell_ratings": "num_strong_sell_ratings",
        "strong_buys_ratings": "num_strong_buys_ratings",
        "num_hold_ratings": "num_hold_ratings",
        "num_buys_ratings": "num_buys_ratings",
        "num_sell_ratings": "num_sell_ratings",
        "analyst_rating": "analyst_rating",
        "price_target": "price_target",
        "price_target_num": "price_target_num",
        "price_target_count": "price_target_count",
    }

    # Filter to only columns that exist in schema
    return {k: v for k, v in rating_mapping.items() if v in COLUMN_SCHEMA}


def get_price_target_historical_columns() -> dict[str, list[str]]:
    """Get price target historical columns organized by metric type.

    Returns:
        Dict with keys: 'mean', 'median', 'high', 'low', 'count'
        Each containing list of historical column names.
    """
    historical_cols = {
        "mean": [],
        "median": [],
        "high": [],
        "low": [],
        "count": [],
    }

    time_periods = [
        "1w_ago",
        "1m_ago",
        "3m_ago",
        "6m_ago",
        "1y_ago",
        "mtd_ago",
        "qtd_ago",
        "ytd_ago",
    ]

    for col_name in COLUMN_SCHEMA.keys():
        if not col_name.startswith("price_target"):
            continue

        for period in time_periods:
            if period in col_name:
                if "median" in col_name:
                    historical_cols["median"].append(col_name)
                elif "high" in col_name:
                    historical_cols["high"].append(col_name)
                elif "low" in col_name:
                    historical_cols["low"].append(col_name)
                elif "num" in col_name or "count" in col_name:
                    historical_cols["count"].append(col_name)
                elif "price_target_" in col_name and "_ago" in col_name:
                    # Mean price target (no qualifier)
                    if (
                        "median" not in col_name
                        and "high" not in col_name
                        and "low" not in col_name
                    ):
                        historical_cols["mean"].append(col_name)
                break

    return historical_cols


def get_cash_flow_temporal_columns() -> dict[str, list[str]]:
    """Get cash flow temporal columns organized by metric type.

    Returns:
        Dict with keys: 'cfo', 'cfi', 'cff', 'fcf', 'acquisitions'
        Each containing list of historical/temporal column names.
    """
    temporal_cols = {
        "cfo": [],
        "cfi": [],
        "cff": [],
        "fcf": [],
        "acquisitions": [],
    }

    for col_name, meta in COLUMN_SCHEMA.items():
        if meta.get("role") != "cash_flow":
            continue

        if col_name.startswith("cfo_"):
            temporal_cols["cfo"].append(col_name)
        elif col_name.startswith("cfi_"):
            temporal_cols["cfi"].append(col_name)
        elif col_name.startswith("cff_"):
            temporal_cols["cff"].append(col_name)
        elif col_name.startswith("fcf_"):
            temporal_cols["fcf"].append(col_name)
        elif "acquisition" in col_name:
            temporal_cols["acquisitions"].append(col_name)

    return temporal_cols


def get_eps_trajectory_columns() -> list[str]:
    """Get EPS trajectory and historical columns.

    Returns columns related to EPS trends, quarterly data, and CAGR metrics.
    """
    eps_cols = []
    eps_patterns = [
        "eps_quarterly",
        "eps_yoy",
        "eps_qoq",
        "eps_cagr",
        "eps_annual",
        "eps_positive",
        "eps_vs_5y",
        "eps_growth_acceleration",
        "net_eps_basic_",
        "eps_adj_",
        "eps_norm_",
        "eps_gaap_",
    ]

    for col_name in COLUMN_SCHEMA.keys():
        for pattern in eps_patterns:
            if pattern in col_name:
                eps_cols.append(col_name)
                break

    return list(set(eps_cols))


def get_dividend_timing_columns() -> list[str]:
    """Get dividend timing and cycle columns.

    Returns columns for dividend date calculations and cycle positioning.
    """
    timing_cols = [
        "days_to_dividend_ex_date",
        "days_to_dividend_record_date",
        "days_to_dividend_payable_date",
        "approaching_ex_date",
        "recently_ex_dividend",
        "dividend_cycle_days",
        "dividend_cycle_position",
        "dividend_announcement_recency",
        "days_to_dividend",
    ]
    return [col for col in timing_cols if col in COLUMN_SCHEMA]


def get_fiscal_calendar_columns() -> list[str]:
    """Get fiscal calendar and temporal pattern columns.

    Returns columns for fiscal timing, reporting lag, and earnings windows.
    """
    fiscal_cols = [
        "fiscal_year_progress",
        "days_to_quarter_end",
        "fiscal_half",
        "reporting_lag_zscore",
        "late_reporter_flag",
        "days_since_fy_end",
        "days_to_next_fy_end",
        "earnings_imminent",
        "pre_earnings_window",
        "fiscal_month",
        "fiscal_quarter",
        "fiscal_year",
        "reporting_lag",
        "reporting_interval",
    ]
    return [col for col in fiscal_cols if col in COLUMN_SCHEMA]


def get_analyst_momentum_columns() -> list[str]:
    """Get analyst sentiment momentum and dynamics columns.

    Returns columns for price target momentum, coverage changes, and revisions.
    """
    momentum_cols = []
    momentum_patterns = [
        "pt_momentum_",
        "pt_median_momentum_",
        "pt_high_momentum_",
        "pt_low_momentum_",
        "pt_acceleration",
        "pt_consensus",
        "pt_spread_trend",
        "pt_skew",
        "analyst_coverage_change_",
        "analyst_coverage_acceleration",
        "analyst_interest_score",
        "analyst_rating_normalized",
        "analyst_rating_conviction",
        "eps_revision_momentum",
        "eps_revision_acceleration",
        "eps_gaap_revision_momentum",
    ]

    for col_name in COLUMN_SCHEMA.keys():
        for pattern in momentum_patterns:
            if col_name.startswith(pattern) or pattern in col_name:
                momentum_cols.append(col_name)
                break

    return list(set(momentum_cols))


def get_balance_sheet_dynamics_columns() -> list[str]:
    """Get balance sheet dynamics and trend columns.

    Returns columns for asset/debt growth, working capital ratios, and retention.
    """
    balance_cols = [
        "asset_growth_rate",
        "balance_sheet_expansion",
        "current_ratio_trend",
        "debt_growth_rate",
        "equity_growth_rate",
        "earnings_retention_rate",
        "retained_earnings_growth",
        "working_capital_ratio",
        "working_capital_vs_5y_avg",
        "cash_stability_ratio",
        "inventory_vs_5y_avg",
        "goodwill_stability",
    ]
    return [col for col in balance_cols if col in COLUMN_SCHEMA]


def get_profitability_enhancement_columns() -> list[str]:
    """Get profitability enhancement and stability columns.

    Returns columns for EBITDA/EBIT comparisons and margin consistency.
    """
    profit_cols = [
        "ebitda_vs_5y_avg",
        "ebitda_stability_score",
        "ebit_vs_5y_avg",
        "operating_leverage_ratio",
        "gross_margin_consistency",
        "normalized_vs_gaap_spread",
        "normalized_vs_gaap_ratio",
        "forward_eps_gaap_adjusted_spread",
        "earnings_stability_score",
    ]
    return [col for col in profit_cols if col in COLUMN_SCHEMA]


def get_workforce_analytics_columns() -> list[str]:
    """Get workforce analytics and employee trend columns.

    Returns columns for FTE growth, productivity metrics, and workforce stability.
    """
    workforce_cols = [
        "fte_growth_1y_pct",
        "fte_growth_2y_pct",
        "fte_growth_3y_pct",
        "fte_cagr_3y_pct",
        "fte_vs_5y_avg",
        "workforce_stability_score",
        "revenue_per_employee",
        "revenue_per_employee_ltm",
        "revenue_per_employee_fy",
        "revenue_per_employee_1fy",
        "revenue_per_employee_trend",
        "revenue_per_employee_vs_5y_pct",
        "employee_growth_yoy",
        "employee_growth_yoy_pct",
        "employee_growth_qoq",
        "employee_growth_cagr_5y",
        "employee_growth_acceleration",
        "workforce_volatility",
        "workforce_volatility_pct",
        "hiring_intensity_score",
    ]
    return [col for col in workforce_cols if col in COLUMN_SCHEMA]


def get_accounting_quality_columns() -> list[str]:
    """Get accounting quality and risk indicator columns.

    Returns columns for exceptional items, impairments, and restructuring vs 5Y averages.
    """
    quality_cols = [
        "impairment_of_goodwill_vs_5y_avg",
        "asset_writedown_vs_5y_avg",
        "restructuring_charges_vs_5y_avg",
        "merger_and_restructuring_charges_vs_5y_avg",
        "other_unusual_to_ebitda",
        "exceptional_items_frequency",
        "exceptional_items_impact_ratio",
        "adjustment_consistency_score",
        "earnings_quality_warning_flag",
        "earnings_quality_score",
        "earnings_quality_score_composite",
    ]
    return [col for col in quality_cols if col in COLUMN_SCHEMA]


def get_momentum_technical_columns() -> list[str]:
    """Get momentum and technical indicator columns.

    Returns columns for price momentum, volatility regime, and volume indicators.
    """
    momentum_cols = [
        "price_momentum_5d",
        "price_vs_ema_100d",
        "volatility_regime",
        "volatility_compression",
        "volatility_term_structure",
        "high_volume_flag",
        "low_volume_flag",
        "return_acceleration",
        "rsi_14d",
        "rsi_30d",
        "momentum_20d",
    ]
    return [col for col in momentum_cols if col in COLUMN_SCHEMA]


def get_valuation_timeseries_columns() -> list[str]:
    """Get valuation timeseries and trend columns.

    Returns columns for P/E, EV/Sales momentum and mean reversion signals.
    """
    valuation_cols = [
        "ev_sales_quarterly_volatility",
        "ev_sales_trend_consistency",
        "p_e_qoq_momentum",
        "p_e_yoy_momentum",
        "p_b_vs_5y_avg",
        "p_b_mean_reversion_signal",
        "valuation_extreme_flag",
        "valuation_stability_score",
        "valuation_trend_consistency",
    ]
    return [col for col in valuation_cols if col in COLUMN_SCHEMA]


def get_dividend_reliability_columns() -> list[str]:
    """Get dividend reliability and sustainability columns.

    Returns columns for dividend yield trends, coverage, and growth metrics.
    """
    dividend_cols = [
        "dividend_yield_volatility",
        "dividend_yield_trend",
        "dividend_yield_vs_5y_avg",
        "dividend_payout_growth",
        "dividend_consistency_years",
        "dividend_yield_cagr_5y",
        "dividend_coverage_ratio",
        "dividend_growth_3y",
        "dividend_growth_5y",
        "dividend_yield_stability",
        "fcf_dividend_coverage",
        "payout_consistency_score",
        "sustainable_dividend_flag",
    ]
    return [col for col in dividend_cols if col in COLUMN_SCHEMA]


def get_revenue_forecast_columns() -> list[str]:
    """Get revenue forecasting and estimate alignment columns.

    Returns columns for revenue estimates, skew, and analyst coverage metrics.
    """
    forecast_cols = [
        "revenue_estimate_skew",
        "ebitda_margin_improvement_expected",
        "forward_ebit_margin",
        "analyst_estimate_coverage",
        "high_coverage_flag",
        "revenue_estimate_alignment",
        "revenue_forecast_accuracy",
        "revenue_cagr_5y",
        "revenue_vs_5y_avg",
        "revenue_above_5y_avg_flag",
    ]
    return [col for col in forecast_cols if col in COLUMN_SCHEMA]


def build_earnings_surprise_pairs(df_columns: list[str]) -> dict[str, tuple[str, str]]:
    """Build earnings surprise calculation pairs from available columns.

    Uses COLUMN_SCHEMA to validate column existence and semantics.
    """
    # Canonical pairs defined by schema roles and naming conventions
    canonical_pairs = {
        "Revenue": ("total_revenues_ltm", "revenues_est_avg_ntm"),
        "EBITDA": ("ebitda_ltm", "ebitda_est_avg_fy1e"),
        "EBIT": ("ebit_ltm", "ebit_est_med_ntm"),
        "Net Income": ("net_income_is_ltm", "net_income_adj_1fy"),
        "EPS": ("eps_adj_ltm", "eps_norm_est_avg_ntm"),
        "EPS GAAP": ("net_eps_basic_ltm", "eps_gaap_est_avg_ntm"),
    }

    # Filter to only include pairs where both columns exist in schema AND dataframe
    available_pairs = {}
    for metric, (actual, estimate) in canonical_pairs.items():
        # Validate both columns are in COLUMN_SCHEMA (Single Source of Truth)
        if actual in COLUMN_SCHEMA and estimate in COLUMN_SCHEMA:
            if actual in df_columns and estimate in df_columns:
                available_pairs[metric] = (actual, estimate)

    return available_pairs


def get_key_features_by_category(
    df_columns: list[str],
    categories: Optional[list[str]] = None,
) -> list[str]:
    """Get available Phase 9.3 features filtered by category.

    Args:
        df_columns: List of columns present in the DataFrame
        categories: List of category names, or None for all categories

    Returns:
        List of feature column names present in both schema and DataFrame
    """
    if categories is None:
        categories = list(PHASE93_FEATURE_CATEGORIES.keys())

    features = []
    for cat in categories:
        cat_features = PHASE93_FEATURE_CATEGORIES.get(cat, [])
        features.extend([f for f in cat_features if f in df_columns])

    return list(set(features))


def get_feature_aliases() -> dict[str, str]:
    """Get feature aliases from ALIAS_SCHEMA in central schema.

    Returns mapping of verbose/canonical column names to their short aliases.
    Note: ALIAS_SCHEMA is defined as alias -> canonical, so this returns
    canonical -> short_alias where applicable for reporting.
    """

    # Reverse ALIAS_SCHEMA to get canonical -> common_name for some key features
    # This is used for display purposes in reports/dashboards
    display_mapping = {
        "return_on_equity_pct_ltm": "roe",
        "return_on_assets_roa_pct_ltm": "roa",
        "p_e_ltm": "p_e_ratio",
        "p_e_ntm": "p_e_forward",
        "ev_ebitda_ltm": "ev_ebitda_ratio",
        "altman_z_score_ltm": "altman_z_score",
        "price_chg_pct_1m": "price_momentum_1m",
        "price_chg_pct_3m": "price_momentum_3m",
        "div_yield_ltm": "dividend_yield",
        "eps_surprise_pct": "eps_surprise",
        "total_revenues_ltm": "revenue_ltm",
        "ebitda_ltm": "ebitda",
        "net_income_is_ltm": "net_income",
    }

    return display_mapping


def get_key_summary_columns() -> list[str]:
    """Get key summary columns based on schema roles.

    Prioritizes market, ratio, and financial_statement roles.
    """
    priority_columns = [
        "last_price",
        "market_cap",
        "enterprise_value",
        "ebitda_ltm",
        "p_e_ntm",
        "p_e_ltm",
        "total_revenues_ltm",
        "price_target",
        "price_target_median",
        "analyst_rating",
    ]

    # Validate all exist in schema
    return [col for col in priority_columns if col in COLUMN_SCHEMA]


def get_dividend_columns_from_schema() -> list[str]:
    """Get dividend-related columns from canonical schema.

    Combines Capital Allocation and Dividend Reliability categories.
    """
    dividend_features = set(PHASE93_FEATURE_CATEGORIES.get("Dividend Reliability", []))
    capital_features = set(PHASE93_FEATURE_CATEGORIES.get("Capital Allocation", []))

    # Also include columns with dividend-related roles from main schema
    for col_name, meta in COLUMN_SCHEMA.items():
        if "dividend" in col_name.lower() or "div_yield" in col_name:
            dividend_features.add(col_name)

    return list(dividend_features | capital_features)


def get_non_recurring_columns() -> list[str]:
    """Get non-recurring exceptional item columns.

    Wrapper around schema.list_non_recurring_cols() for convenience.
    """
    return list_non_recurring_cols()


def get_knn_imputable_columns() -> list[str]:
    """Get columns suitable for KNN imputation.

    Wrapper around schema.list_knn_imputable_cols() for convenience.
    """
    return list_knn_imputable_cols()


def get_count_columns() -> list[str]:
    """Get discrete count columns (use median imputation).

    Wrapper around schema.list_count_cols() for convenience.
    """
    return list_count_cols()


def get_price_related_columns() -> list[str]:
    """Get price-related columns.

    Wrapper around schema.list_price_cols() for convenience.
    """
    return list_price_cols()


def validate_columns_against_schema(df_columns: list[str]) -> dict:
    """Validate DataFrame columns against canonical COLUMN_SCHEMA.

    Returns:
        Dict with 'valid', 'missing', 'extra', and 'alignment_score' keys.
    """
    schema_columns = set(COLUMN_SCHEMA.keys())
    df_cols = set(df_columns)

    valid = df_cols & schema_columns
    missing = schema_columns - df_cols
    extra = df_cols - schema_columns

    # Calculate alignment score
    required = set(list_required_schema_columns_for_etl())
    required_present = required & df_cols
    alignment_score = len(required_present) / len(required) if required else 0

    return {
        "valid": len(valid),
        "missing": len(missing),
        "extra": len(extra),
        "alignment_score": alignment_score,
        "missing_required": list(required - df_cols)[:10],  # First 10
    }


def validate_phase93_coverage(
    df: pd.DataFrame,
    min_coverage_pct: float = 90.0,
    verbose: bool = True,
) -> dict:
    """Validate Phase 9.3 feature coverage against schema definition.

    Args:
        df: DataFrame to validate
        min_coverage_pct: Minimum coverage percentage required
        verbose: Print detailed report

    Returns:
        Dict with coverage stats and missing features by category
    """
    results = {
        "total_defined": 0,
        "total_present": 0,
        "coverage_pct": 0.0,
        "by_category": {},
        "missing_features": [],
        "passed": False,
    }

    for category, features in PHASE93_FEATURE_CATEGORIES.items():
        present = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]

        results["by_category"][category] = {
            "defined": len(features),
            "present": len(present),
            "coverage_pct": len(present) / len(features) * 100 if features else 0,
            "missing": missing,
        }
        results["total_defined"] += len(features)
        results["total_present"] += len(present)
        results["missing_features"].extend(missing)

    results["coverage_pct"] = (
        results["total_present"] / results["total_defined"] * 100
        if results["total_defined"] > 0
        else 0
    )
    results["passed"] = results["coverage_pct"] >= min_coverage_pct

    if verbose:
        print(f"\n{'=' * 60}")
        print("PHASE 9.3 FEATURE COVERAGE VALIDATION")
        print(f"{'=' * 60}")
        print(
            f"Total Coverage: {results['coverage_pct']:.1f}% "
            f"({results['total_present']}/{results['total_defined']})"
        )
        print(f"Status: {'✓ PASSED' if results['passed'] else '✗ FAILED'}")

        if not results["passed"]:
            print(f"\n⚠️ Categories below threshold:")
            for cat, stats in results["by_category"].items():
                if stats["coverage_pct"] < min_coverage_pct:
                    missing_preview = ", ".join(stats["missing"][:3])
                    if len(stats["missing"]) > 3:
                        missing_preview += "..."
                    print(
                        f"  - {cat}: {stats['coverage_pct']:.1f}% "
                        f"(missing: {missing_preview})"
                    )

    return results


def get_schema_role_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a summary of column roles based on COLUMN_SCHEMA.

    Useful for ETL debugging and schema alignment verification.

    Args:
        df: DataFrame to analyze

    Returns:
        Summary DataFrame with role counts and coverage
    """
    role_stats = {}

    for col in df.columns:
        normalized = normalize_column_name(col)
        meta = COLUMN_SCHEMA.get(normalized, {})
        role = meta.get("role", "unknown")

        if role not in role_stats:
            role_stats[role] = {"count": 0, "columns": []}
        role_stats[role]["count"] += 1
        role_stats[role]["columns"].append(col)

    summary = pd.DataFrame(
        [
            {
                "role": role,
                "count": stats["count"],
                "sample_columns": ", ".join(stats["columns"][:3]),
            }
            for role, stats in sorted(role_stats.items(), key=lambda x: -x[1]["count"])
        ]
    )

    return summary


def get_column_sql_mapping(columns: list[str]) -> dict[str, str]:
    """Get SQL column name mapping for a list of normalized column names.

    Args:
        columns: List of normalized Python column names

    Returns:
        Dict mapping normalized names to SQL names
    """
    return {col: get_sql_column_name(col) for col in columns if col in COLUMN_SCHEMA}


def get_temporal_feature_columns() -> dict[str, list[str]]:
    """Get all temporal/time-series feature columns organized by category.

    Returns:
        Dict with temporal feature categories and their columns.
    """
    return {
        "price_target_historical": list(
            set(
                col
                for cols in get_price_target_historical_columns().values()
                for col in cols
            )
        ),
        "cash_flow_temporal": list(
            set(
                col
                for cols in get_cash_flow_temporal_columns().values()
                for col in cols
            )
        ),
        "eps_trajectory": get_eps_trajectory_columns(),
        "fiscal_calendar": get_fiscal_calendar_columns(),
        "dividend_timing": get_dividend_timing_columns(),
        "analyst_momentum": get_analyst_momentum_columns(),
        "valuation_timeseries": get_valuation_timeseries_columns(),
        "momentum_technical": get_momentum_technical_columns(),
    }


def get_feature_count_by_category() -> dict[str, int]:
    """Get count of features defined in each Phase 9.3 category.

    Returns:
        Dict mapping category name to feature count.
    """
    return {cat: len(features) for cat, features in PHASE93_FEATURE_CATEGORIES.items()}
