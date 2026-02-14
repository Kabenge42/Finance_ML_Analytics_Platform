"""
Data loading and preprocessing utilities for feature analytics.

This module provides functions for loading feature data from databases,
preprocessing, validation, and backfilling missing columns.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

if TYPE_CHECKING:
    from sqlalchemy import Engine

from collections import defaultdict

# ---------------------------------------------------------------------------
# Probability export aggregation & row-limit helpers
# ---------------------------------------------------------------------------

# Maximum rows per prob_vw_features_* table before aggregation kicks in
_PROB_EXPORT_ROW_LIMIT: int = 5_000


@dataclass
class ProbExportPolicy:
    """
    Controls how per-feature probability results are exported.

    Parameters
    ----------
    max_rows : int
        Hard cap on exported rows.  When the raw result exceeds this,
        the ``aggregation`` strategy is applied automatically.
    aggregation : str
        One of ``"none"``, ``"by_feature"``, ``"by_isin"``, ``"by_sector"``.
        ``"none"`` exports the raw per-ISIN × per-feature table (subject to
        ``max_rows`` LIMIT).
    top_n_isins : int or None
        When set, only the top-N ISINs (by mean ``prob_above_median``) are
        kept *before* any aggregation.  Useful for large universes.
    """

    max_rows: int = _PROB_EXPORT_ROW_LIMIT
    aggregation: str = "none"  # "none" | "by_feature" | "by_isin" | "by_sector"
    top_n_isins: int | None = None


def aggregate_probability_results(
    df: pd.DataFrame,
    policy: ProbExportPolicy | None = None,
) -> pd.DataFrame:
    """
    Apply aggregation and row-limit policy to a per-feature probability
    DataFrame before database export.

    The input is expected to have columns produced by
    ``CategoryProbabilityAnalyzer.analyze_view`` or
    ``export_probability_view_results``, including at least:

    - ``feature``, ``value``, ``percentile``, ``z_score``, ``prob_above_median``
    - identifier columns (``isin``, ``sector``, ``industry``, …)

    Parameters
    ----------
    df : pd.DataFrame
        Raw per-ISIN × per-feature probability results.
    policy : ProbExportPolicy, optional
        Export policy.  Defaults to auto-aggregate by feature when the
        row count exceeds ``_PROB_EXPORT_ROW_LIMIT``.

    Returns
    -------
    pd.DataFrame
        Aggregated (or truncated) DataFrame ready for export.
    """
    if df.empty:
        return df

    policy = policy or ProbExportPolicy()

    metric_cols = ["value", "percentile", "z_score", "prob_above_median"]
    available_metrics = [c for c in metric_cols if c in df.columns]

    # ── Optional: keep only top-N ISINs by mean prob_above_median ──
    if policy.top_n_isins and "isin" in df.columns and "prob_above_median" in available_metrics:
        isin_rank = (
            df.groupby("isin")["prob_above_median"].mean().nlargest(policy.top_n_isins).index
        )
        df = df[df["isin"].isin(isin_rank)]

    # ── Decide aggregation strategy ──
    aggregation = policy.aggregation

    # Auto-promote to "by_feature" when row count exceeds limit and
    # the caller hasn't explicitly requested a different strategy.
    if aggregation == "none" and len(df) > policy.max_rows:
        aggregation = "by_feature"
        logging.info(
            "prob export: %d rows exceeds limit %d — auto-aggregating by feature",
            len(df),
            policy.max_rows,
        )

    if aggregation == "by_feature" and "feature" in df.columns:
        agg_dict = {m: ["mean", "median", "std", "min", "max", "count"] for m in available_metrics}
        result = df.groupby("feature").agg(agg_dict)
        # Flatten MultiIndex columns  →  "value_mean", "percentile_std", …
        result.columns = ["_".join(col).strip() for col in result.columns]
        result = result.reset_index()
        logging.info("Aggregated by feature: %d → %d rows", len(df), len(result))
        return result

    if aggregation == "by_isin" and "isin" in df.columns:
        agg_dict = {m: ["mean", "median", "std"] for m in available_metrics}
        group_cols = ["isin"]
        # Keep first occurrence of useful identifier columns
        id_first = {}
        for c in ["ticker", "name", "sector", "industry", "region", "country", "exchange"]:
            if c in df.columns:
                id_first[c] = "first"
        result = df.groupby(group_cols).agg({**agg_dict, **id_first})
        result.columns = [
            "_".join(col).strip() if isinstance(col, tuple) else col for col in result.columns
        ]
        result = result.reset_index()
        logging.info("Aggregated by isin: %d → %d rows", len(df), len(result))
        return result

    if aggregation == "by_sector" and "sector" in df.columns and "feature" in df.columns:
        agg_dict = {m: ["mean", "median", "std", "count"] for m in available_metrics}
        result = df.groupby(["sector", "feature"]).agg(agg_dict)
        result.columns = ["_".join(col).strip() for col in result.columns]
        result = result.reset_index()
        logging.info("Aggregated by sector×feature: %d → %d rows", len(df), len(result))
        return result

    # ── Fallback: apply hard LIMIT ──
    if len(df) > policy.max_rows:
        logging.warning(
            "prob export: truncating %d rows to %d (LIMIT)",
            len(df),
            policy.max_rows,
        )
        df = df.head(policy.max_rows)

    return df


try:
    from sqlalchemy import create_engine, text
except ImportError:  # pragma: no cover
    create_engine = None  # type: ignore
    text = None  # type: ignore


# Default identifier columns (fallback when DB is unreachable)
_DEFAULT_IDENTIFIER_COLS = [
    "isin",
    "ticker",
    "name",
    "region",
    "country",
    "trading_country",
    "exchange",
    "sector",
    "industry",
    "dividend_record_frequency",
    "earnings_report_frequency",
    "fy_end",
    "next_earnings_report",
    "next_earnings_status",
    "next_earnings_when",
    "next_fiscal_quarter",
    "reporting_interval",
    "size_class",
    "style_class",
    "unit",
    "dividend_record_announce_date",
    "dividend_record_ex_date",
    "dividend_record_payable_date",
    "dividend_record_record_date",
    "fy_end_date",
    "income_statement_report_date",
    "last_updated",
    "next_earnings",
    "next_fy_end_date",
    "next_income_statement_report_date",
    "reference_date",
]

# Module-level cache for identifier columns loaded from the DB
_identifier_cols_cache: list[str] | None = None


def load_identifier_columns(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> list[str]:
    """
    Load identifier column names from postgres.public.vw_identifier_columns.

    Queries the view's column metadata and returns the ordered list of
    column names.  Falls back to a hardcoded default when the database
    is unreachable or SQLAlchemy is not installed.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL.  If *None*, reads from ``DB_URL``
        environment variable.
    schema : str, default "public"
        Schema containing the ``vw_identifier_columns`` view.

    Returns
    -------
    list[str]
        Ordered list of identifier column names.
    """
    global _identifier_cols_cache
    if _identifier_cols_cache is not None:
        return _identifier_cols_cache

    if create_engine is None or text is None:
        logging.warning("SQLAlchemy not available, using default identifier columns")
        _identifier_cols_cache = list(_DEFAULT_IDENTIFIER_COLS)
        return _identifier_cols_cache

    resolved_url = db_url or os.environ.get("DB_URL")
    if not resolved_url:
        logging.warning("DB_URL not configured, using default identifier columns")
        _identifier_cols_cache = list(_DEFAULT_IDENTIFIER_COLS)
        return _identifier_cols_cache

    try:
        engine = create_engine(resolved_url)
        query = text(f"SELECT * FROM {schema}.vw_identifier_columns LIMIT 0")
        with engine.connect() as conn:
            result = conn.execute(query)
            cols = list(result.keys())

        logging.info(
            "Loaded %d identifier columns from %s.vw_identifier_columns",
            len(cols),
            schema,
        )
        _identifier_cols_cache = cols
        return _identifier_cols_cache

    except Exception as e:
        logging.warning(
            "Could not load identifier columns from DB: %s. " "Using defaults.",
            e,
        )
        _identifier_cols_cache = list(_DEFAULT_IDENTIFIER_COLS)
        return _identifier_cols_cache


def get_identifier_cols_set() -> set[str]:
    """Return the identifier columns as a set (convenience helper)."""
    return set(load_identifier_columns())


VW_FEATURES_VIEWS = [
    "vw_features_analyst_sentiment",
    "vw_features_balance_sheet",
    "vw_features_cashflow",
    "vw_features_composite_scores",
    "vw_features_cost_structure",
    "vw_features_dividends",
    "vw_features_earnings",
    "vw_features_employment",
    "vw_features_growth",
    "vw_features_leverage_liquidity",
    "vw_features_momentum",
    "vw_features_profitability",
    "vw_features_quality_risk",
    "vw_features_technical_analysis",
    "vw_features_temporal",
    "vw_features_unusual_items",
    "vw_features_valuation_ratios",
]


def get_analytics_engine() -> "Engine":
    """
    Create SQLAlchemy engine for analytics database.

    Returns
    -------
    Engine
        SQLAlchemy engine for analytics exports
    """
    if create_engine is None:
        raise ImportError("SQLAlchemy not available.")

    db_url = os.environ.get("DB_URL")
    if not db_url:
        raise ValueError("DB_URL environment variable not set.")

    return create_engine(db_url)


def export_to_analytics_db(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace",
) -> int | None:
    """
    Export DataFrame to PostgreSQL analytics schema.

    Replaces CSV exports with database table exports using the
    DB_URL and DB_ANALYTICS_SCHEMA environment variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export
    table_name : str
        Target table name (without schema prefix)
    if_exists : str, default "replace"
        Behavior when table exists: 'fail', 'replace', 'append', 'delete_rows'

    Returns
    -------
    int or None
        Number of rows affected

    Examples
    --------
    >>> export_to_analytics_db(feature_stats_df, "feature_statistics")
    >>> export_to_analytics_db(mc_results, "monte_carlo_simulation", if_exists="append")
    """
    engine = get_analytics_engine()
    schema = os.environ.get("DB_ANALYTICS_SCHEMA", "analytics")

    logging.info("Exporting %d rows to %s.%s", len(df), schema, table_name)

    result = df.to_sql(
        name=table_name,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
    )

    logging.info("Export complete: %s.%s", schema, table_name)
    return result


@dataclass
class ExportConfig:
    """
    Configuration for DataFrame export operations.

    Centralises settings for database, CSV, and JSON exports so that
    callers can build a single config object and pass it to any of the
    ``export_to_*`` helpers.

    Parameters
    ----------
    table_name : str
        Logical name used as the DB table name or the base file name.
    if_exists : str, default "replace"
        Behaviour when a DB table already exists: 'fail', 'replace',
        'append', or 'delete_rows'.
    output_dir : str, default "outputs/analytics/views"
        Base directory for file-based exports (CSV / JSON).
    orient : str, default "records"
        Pandas ``to_json`` *orient* parameter.
    json_indent : int, default 2
        Indentation level for JSON output.
    csv_sep : str, default ","
        Column separator for CSV output.
    include_index : bool, default False
        Whether to persist the DataFrame index.

    Examples
    --------
    >>> cfg = ExportConfig(table_name="feature_statistics")
    >>> export_to_db(df, cfg)
    >>> export_to_csv(df, cfg)
    >>> export_to_json(df, cfg)
    """

    table_name: str = ""
    if_exists: str = "replace"
    output_dir: str = "outputs/analytics/views"
    orient: str = "records"
    json_indent: int = 2
    csv_sep: str = ","
    include_index: bool = False


def export_to_db(
    df: pd.DataFrame,
    config: ExportConfig | None = None,
    table_name: str | None = None,
    if_exists: str = "replace",
) -> int | None:
    """
    Export DataFrame to the PostgreSQL analytics schema.

    Thin wrapper around :func:`export_to_analytics_db` that also accepts
    an :class:`ExportConfig`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    config : ExportConfig, optional
        Export configuration. Values in *config* are used as defaults and
        can be overridden by the explicit keyword arguments.
    table_name : str, optional
        Target table name (without schema prefix). Overrides
        ``config.table_name`` when provided.
    if_exists : str, default "replace"
        Behaviour when the table exists.

    Returns
    -------
    int or None
        Number of rows affected.
    """
    cfg = config or ExportConfig()
    resolved_table = table_name or cfg.table_name
    if not resolved_table:
        raise ValueError("table_name must be provided either directly or via ExportConfig.")
    resolved_if_exists = if_exists if table_name else cfg.if_exists

    return export_to_analytics_db(df, resolved_table, if_exists=resolved_if_exists)


def export_to_csv(
    df: pd.DataFrame,
    config: ExportConfig | None = None,
    table_name: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """
    Export DataFrame to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    config : ExportConfig, optional
        Export configuration.
    table_name : str, optional
        Base file name (without extension). Overrides ``config.table_name``.
    output_dir : str, optional
        Target directory. Overrides ``config.output_dir``.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    cfg = config or ExportConfig()
    resolved_name = table_name or cfg.table_name
    if not resolved_name:
        raise ValueError("table_name must be provided either directly or via ExportConfig.")
    resolved_dir = Path(output_dir or cfg.output_dir)

    resolved_dir.mkdir(parents=True, exist_ok=True)
    file_path = resolved_dir / f"{resolved_name}.csv"

    df.to_csv(file_path, sep=cfg.csv_sep, index=cfg.include_index)
    logging.info("Exported %d rows to %s", len(df), file_path)
    return file_path


def export_to_json(
    df: pd.DataFrame,
    config: ExportConfig | None = None,
    table_name: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """
    Export DataFrame to a JSON file.

    The default output directory is ``outputs/analytics/views``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    config : ExportConfig, optional
        Export configuration.
    table_name : str, optional
        Base file name (without extension). Overrides ``config.table_name``.
    output_dir : str, optional
        Target directory. Overrides ``config.output_dir``.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    cfg = config or ExportConfig()
    resolved_name = table_name or cfg.table_name
    if not resolved_name:
        raise ValueError("table_name must be provided either directly or via ExportConfig.")
    resolved_dir = Path(output_dir or cfg.output_dir)

    resolved_dir.mkdir(parents=True, exist_ok=True)
    file_path = resolved_dir / f"{resolved_name}.json"

    df.to_json(file_path, orient=cfg.orient, indent=cfg.json_indent, default_handler=str)
    logging.info("Exported %d rows to %s", len(df), file_path)
    return file_path


_VIEW_NAME = "mv_all_stock_features"
_EARNINGS_LOOKAHEAD_DAYS = 20


def _resolve_db_url(db_url: Optional[str]) -> str:
    """Return an explicit DB URL, falling back to the DB_URL environment variable."""
    if db_url is not None:
        return db_url
    db_url = os.environ.get("DB_URL")
    if db_url is None:
        raise ValueError(
            "db_url parameter not provided and DB_URL environment variable not set. "
            "Please provide a database URL or set the DB_URL environment variable."
        )
    return db_url


def _resolve_schema(schema: Optional[str]) -> str:
    """Return an explicit schema, falling back to DB_EQUITIES_SCHEMA or 'public'."""
    if schema is not None:
        return schema
    return os.environ.get("DB_EQUITIES_SCHEMA", "public")


def _build_feature_query(
    view_ref: str,
    limit: Optional[int] = None,
) -> text:
    """Build a parameterised SQL query for the feature materialized view."""
    base_sql = f"""
        SELECT *
        FROM {view_ref}
        WHERE next_earnings >= CURRENT_DATE - (INTERVAL '1 months')
          AND next_earnings <= CURRENT_DATE + (INTERVAL '1 months') AND size_class <> 'Small Cap'
        ORDER BY next_earnings ASC
    """
    if limit is not None:
        base_sql += f" LIMIT {int(limit)}"
    return text(base_sql)


def load_feature_data_from_db(
    db_url: Optional[str] = None,
    earnings_date_filter: str = "2026-01-01",
    limit: Optional[int] = None,
    schema: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load feature data from PostgreSQL database materialized view.

    Loads data from public.mv_all_stock_features with optional filtering
    by next_earnings date.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. If None, reads from DB_URL environment variable
    earnings_date_filter : str, default "2026-01-01"
        Filter stocks with next_earnings >= this date (ISO format: YYYY-MM-DD)
    limit : int, optional
        Maximum number of rows to return
    schema : str, optional
        Database schema name. If None, reads from DB_EQUITIES_SCHEMA environment variable
        or defaults to 'public'

    Returns
    -------
    pd.DataFrame
        DataFrame with feature data from mv_all_stock_features

    Raises
    ------
    ImportError
        If SQLAlchemy or psycopg2 not available
    ValueError
        If db_url is not provided and DB_URL environment variable is not set

    Examples
    --------
    >>> df = load_feature_data_from_db()
    >>> df = load_feature_data_from_db(db_url="postgresql+psycopg2://user:pass@host:5432/db")
    """
    if create_engine is None:
        raise ImportError(
            "SQLAlchemy not available. Install psycopg2-binary and SQLAlchemy to use database loading."
        )

    db_url = _resolve_db_url(db_url)
    schema = _resolve_schema(schema)
    view_ref = f"{schema}.{_VIEW_NAME}"

    safe_db_url = db_url.split("@")[-1] if "@" in db_url else db_url
    logging.info(
        "Loading feature data from %s (view: %s, earnings_date_filter: %s)",
        safe_db_url,
        view_ref,
        earnings_date_filter,
    )

    engine = create_engine(db_url)
    query = _build_feature_query(view_ref, limit)

    df = pd.read_sql(
        query,
        engine,
        params={
            "earnings_date": earnings_date_filter,
            "lookahead_days": _EARNINGS_LOOKAHEAD_DAYS,
        },
    )

    logging.info("Loaded %d rows from %s", len(df), view_ref)
    return df


def backfill_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backfill expected columns for charts and analysis.

    This function normalizes SQL results and creates missing columns
    by mapping from alternative column names or calculating from
    existing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with feature data

    Returns
    -------
    pd.DataFrame
        DataFrame with backfilled columns

    Examples
    --------
    >>> df = backfill_feature_columns(df)
    >>> print(f"Columns: {len(df.columns)}")
    """
    # Ensure DataFrame type
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return df

    # Backfill analyst_neutral_pct if missing
    if "analyst_neutral_pct" not in df.columns:
        bullish = df.get("analyst_bullish_pct")
        bearish = df.get("analyst_bearish_pct")
        if bullish is not None and bearish is not None:
            neutral = 100 - bullish - bearish
            df["analyst_neutral_pct"] = neutral.clip(lower=0, upper=100)

    # Map inventory_turnover to expected column name
    if "inventory_turnover_mv" not in df.columns:
        if "inventory_turnover" in df.columns:
            df["inventory_turnover_mv"] = df["inventory_turnover"]

    # Calculate inventory_days from turnover
    if "inventory_days" not in df.columns:
        turnover_col = df.get("inventory_turnover_mv")
        if turnover_col is not None:
            turnover = turnover_col.replace(0, pd.NA)
            df["inventory_days"] = 365 / turnover

    # Map R&D intensity columns
    if "rnd_intensity_ltm" not in df.columns:
        for src_col in ["rnd_intensity", "rnd_to_revenue"]:
            if src_col in df.columns:
                df["rnd_intensity_ltm"] = df[src_col]
                break

    # Map tangible book value columns
    if "tangible_book_value_ltm" not in df.columns:
        if "tangible_book_value" in df.columns:
            df["tangible_book_value_ltm"] = df["tangible_book_value"]

    # Map goodwill concentration
    if "goodwill_concentration" not in df.columns:
        for src_col in ["goodwill_to_equity", "goodwill_to_assets_pct"]:
            if src_col in df.columns:
                df["goodwill_concentration"] = df[src_col]
                break

    # Ensure industry column exists
    if "industry" not in df.columns and "sector" in df.columns:
        df["industry"] = df["sector"]

    logging.info("Backfill complete. Columns: %d", len(df.columns))

    return df


def compute_metric_statistics(series: pd.Series) -> Optional[dict]:
    """
    Compute standard statistics for a numeric series.

    Parameters
    ----------
    series : pd.Series
        Input series with numeric data

    Returns
    -------
    dict or None
        Dictionary with statistics (count, mean, median, std, min, max, quartiles, etc.)
        Returns None if series is empty or non-numeric

    Examples
    --------
    >>> stats = compute_metric_statistics(df['p_e_ratio'])
    >>> print(f"Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
    """
    data = pd.to_numeric(series, errors="coerce").dropna()
    if len(data) == 0:
        return None

    return {
        "count": int(len(data)),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "q25": float(data.quantile(0.25)),
        "q75": float(data.quantile(0.75)),
        "positive_pct": float((data > 0).sum() / len(data) * 100),
        "missing_pct": float((series.isna().sum() / len(series)) * 100),
    }


def validate_feature_alignment(df: pd.DataFrame, categories: dict) -> dict:
    """
    Check which features in categories exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature data
    categories : dict
        Dictionary mapping category names to lists of feature names

    Returns
    -------
    dict
        Dictionary with 'available', 'missing', and 'coverage_pct' per category

    Examples
    --------
    >>> validation = validate_feature_alignment(df, FEATURE_CATEGORIES)
    >>> low_coverage = {k: v for k, v in validation.items() if v['coverage_pct'] < 80}
    """
    validation_results = {}

    for category, features in categories.items():
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        coverage = len(available) / len(features) * 100 if features else 0

        validation_results[category] = {
            "available_count": len(available),
            "missing_count": len(missing),
            "coverage_pct": coverage,
            "missing_features": missing[:5],  # Show first 5 missing
        }

    return validation_results


def load_all_feature_views(
    db_url: Optional[str] = None,
    earnings_date_filter: str = "2026-01-01",
    schema: str = "public",
    views: Optional[list[str]] = None,
    return_dict: bool = False,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    """
    Load all vw_features tables from postgres.public database.

    Parameters
    ----------
    db_url : str, optional
        SQLAlchemy database URL. If None, reads from DB_URL environment variable
    schema : str, default "public"
        Database schema containing the views
    views : list[str], optional
        Specific views to load. If None, loads all 17 vw_features views
    return_dict : bool, default False
        If True, returns dict of DataFrames keyed by view name.
        If False, returns merged DataFrame on identifier columns.

    Returns
    -------
    dict[str, pd.DataFrame] or pd.DataFrame
        Dictionary of DataFrames by view name, or single merged DataFrame

    Examples
    --------
    >>> # Load all views as dictionary
    >>> views_dict = load_all_feature_views(return_dict=True)
    >>> print(views_dict.keys())

    >>> # Load all views merged into single DataFrame
    >>> df = load_all_feature_views()
    >>> print(f"Columns: {len(df.columns)}")

    >>> # Load specific views only
    >>> df = load_all_feature_views(views=["vw_features_momentum", "vw_features_valuation_ratios"])
    """
    if create_engine is None:
        raise ImportError("SQLAlchemy not available.")

    if db_url is None:
        db_url = os.environ.get("DB_URL")
        if db_url is None:
            raise ValueError("DB_URL environment variable not set.")

    engine = create_engine(db_url)
    target_views = views or VW_FEATURES_VIEWS

    result_dict: dict[str, pd.DataFrame] = {}
    identifier_cols = load_identifier_columns()

    for view_name in target_views:
        view_ref = f"{schema}.{view_name}"
        logging.info("Loading view: %s", view_ref)

        try:
            query = f"""
        SELECT *
        FROM {view_ref}
        WHERE next_earnings >= CURRENT_DATE - (INTERVAL '1 months')
          AND next_earnings <= CURRENT_DATE + (INTERVAL '1 months') AND size_class <> 'Small Cap'
        ORDER BY next_earnings ASC
    """
            df_view = pd.read_sql(query, engine)
            result_dict[view_name] = df_view
            logging.info("Loaded %d rows from %s", len(df_view), view_name)
        except Exception as e:
            logging.warning("Failed to load %s: %s", view_name, e)
            result_dict[view_name] = pd.DataFrame()

    if return_dict:
        return result_dict

    # Merge all views on identifier columns
    if not result_dict:
        return pd.DataFrame()

    merged_df = None
    for view_name, df_view in result_dict.items():
        if df_view.empty:
            continue
        if merged_df is None:
            merged_df = df_view
        else:
            # Get feature columns (exclude identifiers already present)
            feature_cols = [c for c in df_view.columns if c not in merged_df.columns]
            merge_cols = [
                c for c in identifier_cols if c in df_view.columns and c in merged_df.columns
            ]
            if merge_cols and feature_cols:
                merged_df = merged_df.merge(
                    df_view[merge_cols + feature_cols], on=merge_cols, how="outer"
                )

    return merged_df if merged_df is not None else pd.DataFrame()


def get_view_category_mapping() -> dict[str, dict[str, str | list[str]]]:
    """
    Return mapping of view names to feature categories and their feature columns.

    Each entry includes the category label and the list of feature columns
    (non-identifier columns) produced by the underlying vw_features view.

    Returns
    -------
    dict[str, dict[str, str | list[str]]]
        Mapping from view name to dict with 'category' label and 'feature_cols' list
    """
    return {
        "vw_features_analyst_sentiment": {
            "category": "Analyst Sentiment",
            "feature_cols": [
                # calc_sentiment_features
                "analyst_bullish_pct",
                "analyst_bearish_pct",
                "analyst_neutral_pct",
                "analyst_conviction",
                "upside_potential",
                "price_target_spread_pct",
                "price_target_revision_1m",
                "price_target_revision_3m",
                "eps_revision_momentum",
                "analyst_rating_normalized",
                "analyst_coverage_quality",
                # calc_price_target_dynamics
                "pt_momentum_1w",
                "pt_momentum_1m",
                "pt_momentum_3m",
                "pt_momentum_6m",
                "pt_momentum_1y",
                "pt_median_momentum_1m",
                "pt_median_momentum_3m",
                "pt_acceleration_short",
                "pt_acceleration_long",
                "pt_consensus_convergence",
                "analyst_coverage_change_1m",
                "analyst_coverage_change_3m",
                "analyst_coverage_change_1y",
                "pt_vs_price_momentum",
                "analyst_coverage_trend",
                # calc_price_target_achievement_features
                # calc_forward_consensus_features
                # calc_revenue_estimate_consensus
            ],
        },
        "vw_features_balance_sheet": {
            "category": "Balance Sheet",
            "feature_cols": [
                # calc_balance_sheet_dynamics
                "cash_to_assets_pct",
                "cash_change_qoq",
                "cash_vs_5y_avg",
                "inventory_change_yoy",
                "inventory_vs_5y_avg",
                "receivables_change_yoy",
                "receivables_vs_5y_avg",
                "working_capital_vs_5y_avg",
                "retained_earnings_vs_5y",
                "intangibles_growth_flag",
                "asset_quality_score",
                "balance_sheet_strength",
                "debt_maturity_risk",
                # calc_total_assets_temporal
                "assets_fq",
                "assets_fy",
                "assets_ltm",
                "assets_1fq",
                "assets_2fq",
                "assets_3fq",
                "assets_4fq",
                "assets_1fy",
                "assets_2fy",
                "assets_3fy",
                "assets_4fy",
                "assets_qoq_growth",
                "assets_yoy_growth",
                "assets_3y_cagr",
                "asset_growth_accel",
                "asset_base_stable",
                # calc_total_debt_temporal
                "debt_fq",
                "debt_fy",
                "debt_ltm",
                "debt_1fq",
                "debt_2fq",
                "debt_3fq",
                "debt_4fq",
                "debt_1fy",
                "debt_2fy",
                "debt_3fy",
                "debt_4fy",
                "debt_qoq_change",
                "debt_yoy_change",
                "debt_4q_trend",
                "debt_3y_cagr",
                "debt_deleveraging",
                "debt_to_equity_trend",
                # calc_working_capital_temporal
                "wc_fq",
                "wc_fy",
                "wc_ltm",
                "wc_5yavgfy",
                "wc_1fq",
                "wc_2fq",
                "wc_3fq",
                "wc_4fq",
                "wc_1fy",
                "wc_2fy",
                "wc_3fy",
                "wc_4fy",
                "wc_qoq_change",
                "wc_yoy_change",
                "wc_4q_trend",
                "wc_vs_5y_avg",
                "wc_positive_quarters",
                "wc_improving_flag",
                "wc_volatility",
                # calc_tangible_book_features
                # calc_working_capital_deep_features
                # calc_goodwill_temporal_features
                # calc_inventory_temporal_features
            ],
        },
        "vw_features_cashflow": {
            "category": "Cash Flow",
            "feature_cols": [
                # calc_cashflow_features
                "cfo_to_net_income",
                "fcf_to_net_income",
                "fcf_margin",
                "cfo_growth_yoy",
                "fcf_positive_ratio",
                "acquisition_intensity",
                "self_funding_ratio",
                # calc_enhanced_cashflow_features
                "fcf_positive_years",
                "fcf_always_positive",
                "capex_vs_5y_avg",
                "underinvestment_flag",
                "cfo_share_of_cf",
                "cfi_share_of_cf",
                "cff_share_of_cf",
                "self_funding_flag",
                "acquisition_to_fcf",
                "sustainable_ma_flag",
                "fcf_4q_improvement",
                "cash_flow_quality_score",
                "capex_yoy_growth",
                "capex_qoq_growth",
                "capex_3y_trend",
                "capex_volatility",
                "capex_acceleration",
                "capex_cut_flag",
                "overinvestment_flag",
                "acquisitions_yoy_growth",
                "acquisitions_vs_5y_avg",
                "acquisitions_ltm_total",
                "ma_intensity_score",
                "serial_acquirer_flag",
                "acquisition_pause_flag",
                "total_investment_to_cfo",
                "organic_vs_inorganic",
                "investment_efficiency",
                # calc_cashflow_temporal_features
                "cfo_quarterly_trend",
                "cfo_yoy_quarterly",
                "cfi_quarterly_trend",
                "cff_quarterly_trend",
                "fcf_quarterly_trend",
                "cfo_positive_quarters",
                "cfi_negative_quarters",
                "cff_pattern_score",
                "cash_burn_rate",
                "cf_volatility_score",
                "operating_cf_momentum",
                "financing_dependency",
                # calc_cashflow_comprehensive
                "cfo_fq",
                "cfo_ltm",
                "cfo_fy",
                "fcf_fq",
                "fcf_ltm",
                "fcf_fy",
                "fcf_growth_yoy",
                "fcf_yield",
                "cfo_positive_years",
            ],
        },
        "vw_features_composite_scores": {
            "category": "Composite Scores",
            "feature_cols": [
                # calc_piotroski_f_score
                "piotroski_f_score",
                # calc_shareholder_dilution_features
                "dilution_score",
                # calc_quality_momentum_composite
                "quality_momentum_score",
            ],
        },
        "vw_features_cost_structure": {
            "category": "Cost Structure",
            "feature_cols": [
                # calc_cost_structure_features
                # calc_rnd_temporal_features
                # calc_interest_income_features
                # calc_investment_income_temporal
            ],
        },
        "vw_features_dividends": {
            "category": "Dividend Features",
            "feature_cols": [
                # calc_dividend_features
                "dividend_streak",
                "dividend_yield_ltm",
                "dividend_yield_ntm",
                "dividend_payout_ratio",
                "fcf_dividend_coverage",
                "buyback_yield",
                "total_shareholder_yield",
                "dividend_growth_expectation",
                # calc_dividend_timing
                "days_since_ex_date",
                "days_to_payment",
                "dividend_announced_flag",
                "ex_date_approaching_flag",
                "dividend_frequency_score",
                "dividend_consistency",
                "recent_dividend_change",
                "dividend_yield_vs_5y_avg",
                # calc_dividend_history_features
                # calc_dividend_yield_comprehensive
            ],
        },
        "vw_features_earnings": {
            "category": "Earnings Quality",
            "feature_cols": [
                # calc_earnings_features
                "eps_surprise_pct",
                "revenue_surprise_pct",
                "eps_adjustment_ratio",
                "gaap_adj_eps_gap_pct",
                "ebitda_adjustment_ratio",
                "eps_quarterly_trend",
                "eps_yoy_growth",
                # calc_eps_trajectory_features
                "eps_qoq_growth",
                "eps_yoy_quarterly",
                "eps_positive_streak",
                "eps_cagr_3y",
                "eps_cagr_5y",
                "eps_growth_accel",
                "eps_vs_5y_avg",
                "eps_improvement_count",
                "eps_trajectory_score",
                "eps_stability",
                # calc_gaap_adjusted_analytics
                "eps_adjustment_spread_ltm",
                "eps_adjustment_spread_fy",
                "eps_adjustment_spread_1fy",
                "eps_adjustment_spread_fq",
                "eps_adjustment_spread_1fqfq",
                "eps_adjustment_spread_2fqfq",
                "eps_adjustment_spread_3fqfq",
                "eps_adjustment_spread_4fqfq",
                "eps_adjustment_spread_2fy",
                "eps_adjustment_spread_3fy",
                "eps_adjustment_spread_4fy",
                "eps_adjustment_pct",
                "net_income_adjustment_ratio_ltm",
                "net_income_adjustment_ratio_fy",
                "net_income_adjustment_ratio_1fy",
                "net_income_adjustment_ratio_fq",
                "net_income_adjustment_ratio_5yavgfq",
                "net_income_adjustment_ratio_1fqfq",
                "net_income_adjustment_ratio_2fqfq",
                "net_income_adjustment_ratio_3fqfq",
                "net_income_adjustment_ratio_4fqfq",
                "net_income_adjustment_ratio_2fy",
                "net_income_adjustment_ratio_3fy",
                "net_income_adjustment_ratio_4fy",
                "net_income_adjustment_pct",
                "ebitda_adjustment_pct_ltm",
                "ebitda_adjustment_pct_fy",
                "ebitda_adjustment_pct_1fy",
                "ebitda_adjustment_pct_fq",
                "ebitda_adjustment_pct_1fqfq",
                "ebitda_adjustment_pct_2fqfq",
                "ebitda_adjustment_pct_3fqfq",
                "ebitda_adjustment_pct_4fqfq",
                "ebitda_adjustment_pct_2fy",
                "ebitda_adjustment_pct_3fy",
                "ebitda_adjustment_pct_4fy",
                "ebit_adjustment_pct_ltm",
                "ebit_adjustment_pct_fy",
                "ebit_adjustment_pct_1fy",
                "ebit_adjustment_pct_fq",
                "ebit_adjustment_pct_1fqfq",
                "ebit_adjustment_pct_2fqfq",
                "ebit_adjustment_pct_3fqfq",
                "ebit_adjustment_pct_4fqfq",
                "ebit_adjustment_pct_2fy",
                "ebit_adjustment_pct_3fy",
                "ebit_adjustment_pct_4fy",
                "earnings_quality_score",
                "earnings_quality_warning",
                "forward_eps_gaap_adj_spread",
                # calc_gaap_revision_features
                "gaap_revision_momentum",
                "gaap_revision_1m",
                "gaap_revision_3m",
                "gaap_revision_6m",
                "gaap_revision_1y",
                "gaap_vs_norm_revision_spread",
                "gaap_revision_acceleration",
                "gaap_positive_revision_flag",
                "revision_quality_divergence",
                # calc_eps_comprehensive
                "eps_basic_fq",
                "eps_basic_ltm",
                "eps_basic_fy",
                "eps_adj_ltm",
                "eps_norm_est_fy1e",
                "eps_positive_years",
                # calc_eps_continuing_features
                "eps_cont_ltm",
                "eps_cont_fq",
                "eps_cont_fy",
                "eps_cont_1fqfq",
                "eps_cont_2fqfq",
                "eps_cont_3fqfq",
                "eps_cont_4fqfq",
                "eps_cont_1fy",
                "eps_cont_2fy",
                "eps_cont_3fy",
                "eps_cont_4fy",
                "eps_cont_qoq_growth",
                "eps_cont_yoy_growth",
                "eps_cont_cagr_3y",
                "eps_cont_vs_total_eps",
                "eps_cont_positive_streak",
                "eps_cont_trajectory_score",
                "discontinued_ops_impact",
                "core_earnings_stability",
                # calc_net_income_comprehensive
                "net_income_is_fq",
                "net_income_is_ltm",
                "net_income_is_fy",
                "net_income_adj_ltm",
                "normalized_ni_ltm",
                "net_income_is_1fqfq",
                "net_income_is_2fqfq",
                "net_income_is_3fqfq",
                "net_income_is_4fqfq",
                "net_income_is_1fy",
                "net_income_is_2fy",
                "net_income_is_3fy",
                "net_income_is_4fy",
                "net_income_is_5yavgfq",
                "net_income_is_5yavgltm",
                "normalized_ni_5yavgfq",
                "normalized_ni_5yavgltm",
                "net_income_growth_yoy",
                "net_income_margin_ltm",
                "ni_adjustment_ratio",
                "net_income_positive_years",
                "earnings_quality_composite",
                "net_income_qoq_growth",
                "net_income_yoy_quarterly",
                "net_income_vs_5y_avg",
                "normalized_ni_vs_5y_avg",
            ],
        },
        "vw_features_employment": {
            "category": "Employment Metrics",
            "feature_cols": [
                # calc_employment_features
                "revenue_per_employee",
                "profit_per_employee",
                "ebitda_per_employee",
                "assets_per_employee",
                "fte_growth_1y_pct",
                "fte_growth_3y_pct",
                "workforce_stability",
                # calc_employment_dynamics
                "fte_growth_2y_pct",
                "fte_acceleration",
                "workforce_volatility",
                "hiring_intensity",
                "productivity_trend",
                "headcount_vs_revenue",
                "workforce_efficiency_gain",
                "layoff_risk_flag",
                "rapid_hiring_flag",
                "sustainable_growth_flag",
            ],
        },
        "vw_features_growth": {
            "category": "Growth Metrics",
            "feature_cols": [
                # calc_growth_features
                "revenue_growth_yoy",
                "ebitda_growth_yoy",
                "operating_income_growth",
                "fcf_growth",
                "revenue_cagr_5y",
                "forward_revenue_growth",
                "revenue_vs_5y_avg",
                # calc_revenue_forecast_features
                "revenue_est_spread",
                "revenue_beat_potential",
                "revenue_est_revision_trend",
                "ebitda_est_vs_actual",
                "forward_revenue_multiple",
                "revenue_estimate_count",
                "revenue_guidance_gap",
                "consensus_revenue_growth",
                "ebit_estimate_spread",
                "forward_ebitda_margin",
                "revenue_acceleration",
                "estimate_confidence_score",
                # calc_total_revenues_temporal
                "revenue_fq",
                "revenue_ltm",
                "revenue_fy",
                "revenue_1fy",
                "revenue_5yavgfq",
                "revenue_5yavgltm",
                "revenue_vs_5y_avg_fq",
                "revenue_vs_5y_avg_ltm",
                "revenue_fq_vs_avg",
                "revenue_momentum",
                # calc_gross_profit_temporal
                "gp_fq",
                "gp_fy",
                "gp_ltm",
                "gp_1fqfq",
                "gp_2fqfq",
                "gp_3fqfq",
                "gp_4fqfq",
                "gp_1fy",
                "gp_2fy",
                "gp_3fy",
                "gp_4fy",
                "gp_qoq_growth",
                "gp_yoy_growth",
                "gp_margin_fq",
                "gp_margin_trend",
                "gp_positive_quarters",
                "gp_margin_expansion",
                # calc_ebit_ebitda_comprehensive
                "ebit_fq",
                "ebit_ltm",
                "ebit_fy",
                "ebit_1fy",
                "ebitda_fq",
                "ebitda_ltm",
                "ebitda_fy",
                "ebitda_1fy",
                "ebit_2fy",
                "ebit_3fy",
                "ebit_4fy",
                "ebitda_2fy",
                "ebitda_3fy",
                "ebitda_4fy",
                "ebit_1fqfq",
                "ebit_2fqfq",
                "ebit_3fqfq",
                "ebit_4fqfq",
                "ebitda_1fqfq",
                "ebitda_2fqfq",
                "ebitda_3fqfq",
                "ebitda_4fqfq",
                "ebit_5yavgfq",
                "ebit_5yavgltm",
                "ebitda_5yavgfq",
                "ebitda_5yavgltm",
                "ebit_adj_fq",
                "ebit_adj_ltm",
                "ebit_adj_fy",
                "ebitda_adj_fq",
                "ebitda_adj_ltm",
                "ebitda_adj_fy",
                "ebit_growth_yoy",
                "ebitda_growth_yoy",
                "ebit_margin_ltm",
                "ebitda_margin_ltm",
                "ebit_positive_years",
                "ebitda_positive_years",
                "ebit_qoq_growth",
                "ebitda_qoq_growth",
                "ebit_cagr_3y",
                "ebitda_cagr_3y",
                "ebit_vs_5y_avg",
                "ebitda_vs_5y_avg",
                # calc_revenue_quarterly_features
            ],
        },
        "vw_features_leverage_liquidity": {
            "category": "Leverage & Liquidity",
            "feature_cols": [
                # calc_leverage_features
                "debt_to_equity",
                "debt_to_assets",
                "equity_ratio",
                "interest_coverage",
                "current_ratio",
                "cash_ratio",
                "working_capital_ratio",
                # calc_efficiency_ratios
                "asset_turnover",
                "inventory_turnover",
                "receivables_days",
                "working_capital_turns",
                # calc_financial_distress_features
                "distress_risk_score",
                "liquidity_stress_score",
                "working_capital_trend",
                "cash_runway_months",
                "combined_distress_score",
                "wc_deteriorating_flag",
                "retained_earnings_growth",
                "accumulated_deficit_flag",
                "adequate_cash_buffer",
                # calc_size_liquidity_features
            ],
        },
        "vw_features_momentum": {
            "category": "Momentum",
            "feature_cols": [
                # calc_momentum_features
                "price_momentum_1m",
                "price_momentum_3m",
                "price_momentum_6m",
                "price_momentum_1y",
                "price_momentum_5d",
                "ema_crossover_20_50",
                "ema_crossover_50_250",
                "price_vs_ema_20d",
                "price_vs_ema_250d",
                "pct_off_52w_high",
                "pct_above_52w_low",
                "range_52w_position",
                "beta_momentum",
                "volatility_regime",
                # calc_long_term_momentum_features
                # calc_volatility_surface_features
                # calc_beta_risk_features
            ],
        },
        "vw_features_profitability": {
            "category": "Profitability",
            "feature_cols": [
                # calc_profitability_features
                "roe",
                "roa",
                "gross_margin_pct",
                "operating_margin_pct",
                "net_margin_pct",
                "ebitda_margin_pct",
                "roic",
                "rnd_intensity",
                "equity_multiplier",
                # calc_margin_trends
                "gross_margin_trend_yoy",
                "operating_margin_trend",
                "net_margin_trend_yoy",
                "ebitda_margin_trend",
                "margin_expansion_flag",
                "margin_stability_score",
            ],
        },
        "vw_features_quality_risk": {
            "category": "Quality & Risk",
            "feature_cols": [
                # calc_quality_features
                "has_goodwill_impairment",
                "has_asset_writedown",
                "has_restructuring",
                "goodwill_to_assets_pct",
                "intangible_intensity",
                "exceptional_items_to_ebitda",
                "altman_z_score",
                "altman_z_trend",
                "quick_ratio",
                # calc_accounting_quality_features
                "goodwill_change_rate",
                "restructuring_intensity",
                "exceptional_items_frequency",
                "merger_impact_ratio",
                "non_operating_income_share",
                "asset_sale_boost",
                "accounting_quality_score",
                # calc_quality_features_comprehensive
                "goodwill_impairment_ltm",
                "asset_writedown_ltm",
                "restructuring_ltm",
                "has_goodwill_impairment_ltm",
                "goodwill_impairment_frequency",
                "asset_writedown_frequency",
                "restructuring_frequency",
                "exceptional_items_total_ltm",
                "quality_issues_count_5y",
            ],
        },
        "vw_features_technical_analysis": {
            "category": "Technical Analysis",
            "feature_cols": [
                # calc_technical_analysis_features
                "ema_slope_20d",
                "ema_trend_consistency",
                "price_vs_ema_100d",
                "near_52w_high_flag",
                "near_52w_low_flag",
                "volume_momentum_score",
                "breakout_signal",
                "high_volume_flag",
                "low_volume_flag",
                "volatility_compression",
                "volatility_term_structure",
            ],
        },
        "vw_features_temporal": {
            "category": "Temporal Features",
            "feature_cols": [
                # calc_temporal_features
                "fiscal_quarter",
                "fiscal_month",
                "fiscal_year",
                "days_to_earnings",
                "earnings_report_recency",
                "reporting_lag",
                "fiscal_year_progress",
                # calc_fiscal_calendar_features
                "days_since_last_report",
                "days_to_fy_end",
                "is_quarter_end_month",
                "is_fy_end_month",
                "earnings_season_flag",
                "pre_earnings_window",
                "post_earnings_window",
                "reporting_freshness_score",
                "fiscal_quarter_progress",
            ],
        },
        "vw_features_unusual_items": {
            "category": "Unusual Items",
            "feature_cols": [
                # calc_unusual_items_features
            ],
        },
        "vw_features_valuation_ratios": {
            "category": "Valuation Ratios",
            "feature_cols": [
                # calc_valuation_features
                "p_e_ratio",
                "p_b_ratio",
                "ev_ebitda_ratio",
                "ev_sales_ratio",
                "dividend_yield",
                "peg_ratio",
                # calc_valuation_timeseries_features
                "ev_sales_trend_1y",
                "ev_ebitda_momentum",
                "p_e_momentum_yoy",
                "p_e_momentum_qoq",
                "ev_sales_vs_3y_avg",
                "ev_ebitda_vs_3y_avg",
                "p_e_vs_3y_avg",
                "ev_sales_forward_discount",
                "ev_ebitda_forward_discount",
                "p_e_forward_discount",
                "p_b_vs_5y_avg",
                # calc_extended_valuation_timeseries
                "ev_sales_qoq_1q",
                "ev_sales_qoq_2q",
                "ev_sales_qoq_3q",
                "ev_sales_qoq_4q",
                "p_e_vs_5y_avg",
                "p_e_percentile_proxy",
                "valuation_mean_reversion",
                "ev_ebitda_qoq_trend",
                "p_b_momentum_yoy",
                "valuation_compression",
                "forward_pe_premium",
            ],
        },
    }


def get_view_category_labels() -> dict[str, str]:
    """
    Return flat mapping of view names to category labels (backward-compatible).

    Returns
    -------
    dict[str, str]
        Mapping from view name to category label string
    """
    return {view: info["category"] for view, info in get_view_category_mapping().items()}


def get_view_feature_cols(view_name: str) -> list[str]:
    """
    Return the feature columns for a specific view.

    Parameters
    ----------
    view_name : str
        Name of the vw_features view

    Returns
    -------
    list[str]
        List of feature column names (non-identifier columns)

    Raises
    ------
    KeyError
        If view_name is not in the mapping
    """
    mapping = get_view_category_mapping()
    return mapping[view_name]["feature_cols"]


def export_view_analytics_results(
    analytics_results: dict[str, dict],
    output_schema: str = "analytics",
) -> dict[str, int]:
    """
    Export view-based analytics results to database.

    Parameters
    ----------
    analytics_results : dict[str, dict]
        Results from run_all_views_probability_analytics
    output_schema : str, default "analytics"
        Target schema for output tables

    Returns
    -------
    dict[str, int]
        Row counts exported per table
    """
    import numpy as np

    identifier_cols = load_identifier_columns()
    export_counts = {}

    for view_name, results in analytics_results.items():
        # Export summary statistics
        if results.get("summary_statistics"):
            stats_df = pd.DataFrame(results["summary_statistics"]).T
            stats_df["view_name"] = view_name
            stats_df["category"] = results.get("category", view_name)

            # Populate feature_cols with the actual feature name per row
            # (the index is the feature column name after transposing)
            stats_df["feature_cols"] = stats_df.index

            # Prepend identifier columns from the source view if available
            source_df = results.get("source_df")
            if source_df is not None and not source_df.empty:
                id_cols_present = [c for c in identifier_cols if c in source_df.columns]
                if id_cols_present:
                    id_df = source_df[id_cols_present].drop_duplicates().head(1)
                    for col in id_cols_present:
                        stats_df[col] = id_df[col].iloc[0] if len(id_df) > 0 else None

            table_name = f"{view_name.replace('vw_features_', '')}_statistics"
            count = export_to_analytics_db(stats_df, table_name)
            export_counts[table_name] = count

        # Export distribution fits
        if results.get("distribution_fits"):
            # Clean distribution fits: remove numpy arrays and convert numpy scalars
            clean_fits = {}
            for feature, fit_data in results["distribution_fits"].items():
                clean_fits[feature] = {}
                for key, value in fit_data.items():
                    if isinstance(value, np.ndarray):
                        # Skip large arrays (like simulations) or convert small ones to lists
                        if value.size > 100:
                            continue  # Skip simulation arrays
                        else:
                            clean_fits[feature][key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        clean_fits[feature][key] = float(value)
                    elif isinstance(value, tuple):
                        # Convert tuples with numpy types (like params)
                        clean_fits[feature][key] = tuple(
                            float(v) if isinstance(v, (np.integer, np.floating)) else v
                            for v in value
                        )
                    else:
                        clean_fits[feature][key] = value

            fits_df = pd.DataFrame(clean_fits)
            table_name = f"{view_name.replace('vw_features_', '')}_distributions"
            count = export_to_analytics_db(fits_df, table_name)
            export_counts[table_name] = count

    logging.info("Exported analytics for %d views", len(export_counts))
    return export_counts


def safe_get_column(df: pd.DataFrame, *column_names: str, default=None):
    """
    Safely get a column from DataFrame, trying multiple names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    *column_names : str
        Column names to try in order
    default : any, optional
        Default value if no column found

    Returns
    -------
    pd.Series or default
        First found column or default value

    Examples
    --------
    >>> col = safe_get_column(df, 'industry', 'sector', default=pd.Series())
    """
    for col_name in column_names:
        if col_name in df.columns:
            return df[col_name]
    return default


def load_feature_categories_from_db(
    connection_string: Optional[str] = None,
) -> dict[str, list[str]]:
    """
    Load feature categories from the calculated_features_registry table.

    Parameters
    ----------
    connection_string : str, optional
        Database connection string. If None, reads from DB_URL environment variable.

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping category names to lists of feature aliases.
    """
    if connection_string is None:
        connection_string = os.environ.get("DB_URL")

    # If no connection string is available, use fallback immediately
    if not connection_string:
        logging.info("DB_URL not configured, using fallback feature categories")
        return _get_fallback_feature_categories()

    if create_engine is None or text is None:
        logging.warning("SQLAlchemy not available, using fallback feature categories")
        return _get_fallback_feature_categories()

    query = text("""
                 SELECT category, feature_alias
                 FROM public.calculated_features_registry
                 ORDER BY category, feature_alias
                 """)

    # Attempts to load feature categories from database
    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(query)
            rows = result.fetchall()

        categories = defaultdict(list)
        for row in rows:
            category, feature_alias = row[0], row[1]
            categories[category].append(feature_alias)

        logging.info(f"Loaded {len(categories)} feature categories from database")
        return dict(categories)

    except Exception as e:
        logging.warning(f"Could not load categories from database: {e}")
        logging.warning("Falling back to hardcoded FEATURE_CATEGORIES")
        return _get_fallback_feature_categories()


def _get_fallback_feature_categories() -> dict[str, list[str]]:
    """Fallback hardcoded categories if database is unavailable."""
    return {
        "Valuation Ratios": [
            "p_e_ratio",
            "p_b_ratio",
            "ev_ebitda_ratio",
            "ev_sales_ratio",
            "dividend_yield",
            "peg_ratio",
            "price_to_tangible_book",
            "tangible_book_value_ltm",
        ],
        "Momentum & Technical": [
            "price_momentum_1m",
            "price_momentum_3m",
            "price_momentum_6m",
            "price_momentum_1y",
            "price_momentum_3y",
            "price_momentum_5y",
            "range_52w_position",
            "long_term_trend_score",
            "secular_trend_flag",
        ],
        "Profitability": [
            "roe",
            "roa",
            "gross_margin_pct",
            "operating_margin_pct",
            "net_margin_pct",
            "ebitda_margin_pct",
            "roic",
            "net_margin_trend_yoy",
        ],
        "Quality & Risk": [
            "piotroski_f_score",
            "distress_risk_score",
            "altman_z_score",
            "accounting_quality_score",
            "earnings_quality_composite",
            "cash_flow_quality_score",
            "beta_stability_score",
        ],
        "Leverage & Liquidity": [
            "debt_to_equity",
            "current_ratio",
            "quick_ratio",
            "interest_coverage",
            "cash_ratio",
            "working_capital_ratio",
            "debt_deleveraging",
        ],
        "Analyst Sentiment": [
            "analyst_bullish_pct",
            "analyst_neutral_pct",
            "analyst_bearish_pct",
            "upside_potential",
            "analyst_rating_normalized",
            "eps_revision_momentum",
        ],
        "Earnings Quality": [
            "eps_surprise_pct",
            "eps_adjustment_ratio",
            "gaap_adj_eps_gap_pct",
            "eps_trajectory_score",
            "earnings_quality_score",
            "gaap_revision_momentum",
        ],
        "Growth Metrics": [
            "revenue_growth_yoy",
            "ebitda_growth_yoy",
            "eps_yoy_growth",
            "fcf_growth_yoy",
            "revenue_cagr_5y",
        ],
        "Cash Flow": [
            "fcf_positive_years",
            "fcf_margin",
            "fcf_yield",
            "cfo_to_net_income",
            "self_funding_ratio",
            "cash_flow_quality_score",
        ],
        "Dividend Features": [
            "dividend_streak",
            "dividend_yield_ltm",
            "dividend_payout_ratio",
            "fcf_dividend_coverage",
            "total_shareholder_yield",
        ],
        "R&D Investment": [
            "rnd_intensity_ltm",
            "rnd_yoy_growth",
            "rnd_per_employee",
            "high_rnd_intensity_flag",
        ],
        "Inventory Temporal": [
            "inventory_days",
            "inventory_turnover_mv",
            "inventory_yoy_change",
            "inventory_buildup_flag",
        ],
        "Goodwill & M&A": [
            "goodwill_concentration",
            "goodwill_3y_growth",
            "recent_acquisition_flag",
            "impairment_risk_score",
        ],
        "CapEx & Investment": [
            "capex_yoy_growth",
            "capex_vs_5y_avg",
            "acquisitions_ltm_total",
            "ma_intensity_score",
            "investment_efficiency",
        ],
    }


def validate_feature_registry_alignment(
    db_url: Optional[str] = None,
    schema: str = "public",
) -> dict[str, Any]:
    """
    Cross-validate calculated_features_registry against equities_schema_metadata.

    Checks that every ``primary_source_col`` in the feature registry
    has a corresponding entry in ``equities_schema_metadata``, and that
    every ``source_function`` maps to ``feature_registry_metadata``.

    Parameters
    ----------
    db_url : str, optional
        Database URL. Falls back to DB_URL env var.
    schema : str
        Database schema.

    Returns
    -------
    dict
        Validation report with keys:
        - orphan_source_cols: features whose primary_source_col is missing
        - orphan_functions: features whose source_function has no metadata
        - category_coverage: dict of category → feature count
        - total_features: int
    """
    if create_engine is None or text is None:
        return {"error": "SQLAlchemy not available"}

    url = db_url or os.environ.get("DB_URL")
    if not url:
        return {"error": "DB_URL not configured"}

    engine = create_engine(url)

    try:
        with engine.connect() as conn:
            # All feature registry entries
            features = pd.read_sql(
                f"SELECT feature_key, category, source_function, primary_source_col "
                f"FROM {schema}.calculated_features_registry",
                conn,
            )
            # All equities metadata column aliases
            eq_cols = pd.read_sql(
                f"SELECT column_alias FROM {schema}.equities_schema_metadata",
                conn,
            )
            # All function metadata
            fn_meta = pd.read_sql(
                f"SELECT function_name FROM {schema}.feature_registry_metadata",
                conn,
            )

        eq_col_set = set(eq_cols["column_alias"].dropna())
        fn_set = set(fn_meta["function_name"].dropna())

        orphan_cols = features[
            features["primary_source_col"].notna()
            & ~features["primary_source_col"].isin(eq_col_set)
        ][["feature_key", "primary_source_col"]].to_dict("records")

        orphan_fns = features[
            features["source_function"].notna() & ~features["source_function"].isin(fn_set)
        ][["feature_key", "source_function"]].to_dict("records")

        category_coverage = features.groupby("category").size().to_dict()

        return {
            "orphan_source_cols": orphan_cols,
            "orphan_functions": orphan_fns,
            "category_coverage": category_coverage,
            "total_features": len(features),
        }
    except Exception as e:
        logging.warning("Feature registry validation failed: %s", e)
        return {"error": str(e)}


def compare_registry_with_local(
    db_categories: dict[str, list[str]], local_categories: dict[str, list[str]]
) -> dict:
    """
    Compare database registry with local/fallback categories.

    Useful for identifying missing features or new additions.

    Parameters
    ----------
    db_categories : dict[str, list[str]]
        Categories loaded from database
    local_categories : dict[str, list[str]]
        Local/fallback categories

    Returns
    -------
    dict
        Report with categories and features only in db or local
    """
    report = {
        "categories_only_in_db": [],
        "categories_only_in_local": [],
        "features_only_in_db": {},
        "features_only_in_local": {},
    }

    db_cats = set(db_categories.keys())
    local_cats = set(local_categories.keys())

    report["categories_only_in_db"] = list(db_cats - local_cats)
    report["categories_only_in_local"] = list(local_cats - db_cats)

    for cat in db_cats & local_cats:
        db_features = set(db_categories[cat])
        local_features = set(local_categories[cat])

        only_in_db = db_features - local_features
        only_in_local = local_features - db_features

        if only_in_db:
            report["features_only_in_db"][cat] = list(only_in_db)
        if only_in_local:
            report["features_only_in_local"][cat] = list(only_in_local)

    return report
