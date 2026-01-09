"""Schema helper utilities for notebook usage.

Provides convenient wrappers around COLUMN_SCHEMA and PHASE93_FEATURE_CATEGORIES.
"""

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    normalize_column_name,
    list_required_schema_columns_for_etl,
)


def get_columns_by_role(role: str) -> list[str]:
    """Get all columns with a specific role."""
    return [col for col, meta in COLUMN_SCHEMA.items() if meta.get("role") == role]


def get_features_for_category(category: str) -> list[str]:
    """Get Phase 9.3 features for a specific category."""
    return PHASE93_FEATURE_CATEGORIES.get(category, [])


def validate_column_exists(col_name: str) -> bool:
    """Check if column exists in canonical schema."""
    return col_name in COLUMN_SCHEMA


def get_analyst_rating_columns() -> dict[str, str]:
    """Get analyst rating columns from canonical schema."""
    rating_mapping = {}

    # Map semantic names to schema columns with role='count' that relate to analyst ratings
    analyst_keywords = ["analyst", "rating", "buy", "sell", "hold", "target", "strong"]

    for col_name, meta in COLUMN_SCHEMA.items():
        if meta.get("role") == "count" or "target" in col_name:
            for keyword in analyst_keywords:
                if keyword in col_name.lower():
                    # Derive semantic category
                    if "strong_sell" in col_name:
                        rating_mapping["strong_sell_ratings"] = col_name
                    elif "strong_buy" in col_name:
                        rating_mapping["strong_buys_ratings"] = col_name
                    elif "hold" in col_name:
                        rating_mapping["num_hold_ratings"] = col_name
                    elif "buy" in col_name and "strong" not in col_name:
                        rating_mapping["num_buys_ratings"] = col_name
                    elif "sell" in col_name and "strong" not in col_name:
                        rating_mapping["num_sell_ratings"] = col_name
                    elif "analyst_rating" == col_name:
                        rating_mapping["analyst_rating"] = col_name
                    elif "price_target" == col_name:
                        rating_mapping["price_target"] = col_name
                    break

    return rating_mapping


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
    }

    # Filter to only include pairs where both columns exist in schema AND dataframe
    available_pairs = {}
    for metric, (actual, estimate) in canonical_pairs.items():
        # Validate both columns are in COLUMN_SCHEMA (Single Source of Truth)
        if actual in COLUMN_SCHEMA and estimate in COLUMN_SCHEMA:
            if actual in df_columns and estimate in df_columns:
                available_pairs[metric] = (actual, estimate)

    return available_pairs


def get_key_features_by_category(df_columns: list[str], categories: list[str] = None) -> list[str]:
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
    """Get feature aliases from COLUMN_SCHEMA.

    Returns mapping of verbose column names to their short aliases.
    """
    # Define canonical short names mapped from schema entries
    alias_patterns = {
        "return_on_equity": "roe",
        "return_on_assets": "roa",
        "p_e_ltm": "p_e_ratio",
        "p_e_ntm": "p_e_ratio",
        "ev_ebitda_ltm": "ev_ebitda_ratio",
        "altman_z_score": "altman_z_score",
        "price_chg_pct_1m": "price_momentum_1m",
        "price_chg_pct_3m": "price_momentum_3m",
        "dividend_payout_ratio": "payout_ratio",
        "eps_surprise_pct": "eps_surprise_pct",
    }

    # Build from schema where aliases exist
    mapping = {}
    for col_name, meta in COLUMN_SCHEMA.items():
        for pattern, alias in alias_patterns.items():
            if pattern in col_name:
                mapping[col_name] = alias
                break

    return mapping


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
        "total_revenues_ltm",
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
