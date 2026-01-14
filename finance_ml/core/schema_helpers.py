"""Schema helper utilities for notebook usage.

Provides convenient wrappers around COLUMN_SCHEMA and PHASE93_FEATURE_CATEGORIES.
"""

import pandas as pd

from finance_ml.core.schema import (
    COLUMN_SCHEMA,
    PHASE93_FEATURE_CATEGORIES,
    list_required_schema_columns_for_etl,
    normalize_column_name,
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
