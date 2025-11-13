"""
finance_ml.advanced_preprocessing - Advanced preprocessing for Phase 9

DEPRECATION NOTICE (Phase 9.1 refactor):
Many functions from this module have been moved to finance_ml.ml_workflow.preprocessing subpackage:
- Outlier detection: preprocessing.outliers
- Scaling: preprocessing.scaling
- Quality: preprocessing.quality (future)
- Pipeline: preprocessing.pipeline (future)

This module provides backward compatibility shims. Please update imports to use the new structure.

Part of Phase 9.1 implementation.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)

# Phase 9.1 refactor: Import moved functions for backward compatibility
from finance_ml.ml_workflow.preprocessing.outliers import (
    detect_outliers_iqr as _new_detect_outliers_iqr,
    detect_outliers_zscore as _new_detect_outliers_zscore,
    detect_outliers_isolation_forest as _new_detect_outliers_isolation_forest,
    winsorize_by_sector as _new_winsorize_by_sector,
)
from finance_ml.ml_workflow.preprocessing.scaling import (
    create_scaler_pipeline as _new_create_scaler_pipeline,
    scale_features as _new_scale_features,
)
from finance_ml.ml_workflow.preprocessing.quality import (
    DataQualityReport as _new_DataQualityReport,
    calculate_data_quality_score as _new_calculate_data_quality_score,
)
from finance_ml.ml_workflow.preprocessing.imputation import (
    apply_zero_imputation as _new_apply_zero_imputation,
    apply_knn_imputation_enhanced as _new_apply_knn_imputation_enhanced,
    apply_price_imputation as _new_apply_price_imputation,
    apply_median_imputation as _new_apply_median_imputation,
    apply_enhanced_imputation_strategy_4step as _new_apply_enhanced_imputation_strategy_4step,
    apply_enhanced_imputation_strategy_6step as _new_apply_enhanced_imputation_strategy_6step,
)


# Phase 9.1: DataQualityReport moved to preprocessing.quality
# Re-export for backward compatibility
DataQualityReport = _new_DataQualityReport


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    by_sector: bool = True,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers using Interquartile Range (IQR) method.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_iqr` instead.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        by_sector: Apply IQR separately by sector (default: True)
        iqr_multiplier: IQR multiplier for bounds (default: 1.5)

    Returns:
        DataFrame with boolean columns indicating outliers (col_outlier)
    """
    warnings.warn(
        "detect_outliers_iqr from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_iqr instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_detect_outliers_iqr(df, columns, by_sector, iqr_multiplier)


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0,
    by_sector: bool = True,
) -> pd.DataFrame:
    """Detect outliers using z-score method.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_zscore` instead.

    Args:
        df: Input DataFrame
        columns: Columns to check (default: all numeric)
        threshold: Z-score threshold (default: 3.0)
        by_sector: Apply z-score separately by sector (default: True)

    Returns:
        DataFrame with boolean columns indicating outliers (col_zscore_outlier)
    """
    warnings.warn(
        "detect_outliers_zscore from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_zscore instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_detect_outliers_zscore(df, columns, threshold, by_sector)


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest algorithm.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_isolation_forest` instead.

    Args:
        df: Input DataFrame
        columns: Columns to use for detection (default: all numeric)
        contamination: Expected proportion of outliers (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Boolean Series indicating outliers (True = outlier)
    """
    warnings.warn(
        "detect_outliers_isolation_forest from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.outliers.detect_outliers_isolation_forest instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_detect_outliers_isolation_forest(df, columns, contamination, random_state)


def winsorize_by_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    by_sector: bool = True,
) -> pd.DataFrame:
    """Winsorize extreme values by replacing with percentile bounds.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.outliers.winsorize_by_sector` instead.

    Args:
        df: Input DataFrame
        columns: Columns to winsorize (default: all numeric)
        lower_percentile: Lower percentile bound (default: 0.01 = 1%)
        upper_percentile: Upper percentile bound (default: 0.99 = 99%)
        by_sector: Apply winsorization separately by sector (default: True)

    Returns:
        DataFrame with winsorized values
    """
    warnings.warn(
        "winsorize_by_sector from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.outliers.winsorize_by_sector instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_winsorize_by_sector(df, columns, lower_percentile, upper_percentile, by_sector)


def calculate_data_quality_score(df: pd.DataFrame) -> DataQualityReport:
    """Calculate comprehensive data quality metrics.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.quality.calculate_data_quality_score` instead.

    Args:
        df: Input DataFrame

    Returns:
        DataQualityReport with detailed quality metrics
    """
    warnings.warn(
        "calculate_data_quality_score from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.quality.calculate_data_quality_score instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_calculate_data_quality_score(df)


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "sector_median",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Impute missing values using various strategies.

    Args:
        df: Input DataFrame
        strategy: Imputation strategy:
            - 'sector_median': Median by sector (default)
            - 'sector_mean': Mean by sector
            - 'median': Global median
            - 'mean': Global mean
            - 'knn': KNN imputation
        columns: Columns to impute (default: all numeric)

    Returns:
        DataFrame with imputed values
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if strategy in ["sector_median", "sector_mean"]:
        # Sector-specific imputation
        if "sector" not in df.columns:
            logger.warning("Sector column not found, falling back to global imputation")
            strategy = "median" if strategy == "sector_median" else "mean"
        else:
            agg_func = "median" if strategy == "sector_median" else "mean"

            for col in columns:
                if col not in df.columns or not df[col].isna().any():
                    continue

                # Calculate sector aggregates
                sector_values = df.groupby("sector")[col].transform(agg_func)

                # Fill missing values
                result[col] = df[col].fillna(sector_values)

                # Fill any remaining NaN with global aggregate
                if result[col].isna().any():
                    global_value = df[col].median() if agg_func == "median" else df[col].mean()
                    result[col] = result[col].fillna(global_value)

    elif strategy in ["median", "mean"]:
        # Global imputation
        imputer = SimpleImputer(strategy=strategy)
        result[columns] = imputer.fit_transform(df[columns])

    elif strategy == "knn":
        # KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        result[columns] = imputer.fit_transform(df[columns])

    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    logger.info(f"Imputed missing values using '{strategy}' strategy for {len(columns)} columns")
    return result


def impute_missing_values_knn_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sector_column: str = "sector",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using sector-aware KNN imputation.

    This enhanced KNN imputation performs imputation separately within each sector,
    ensuring that missing values are filled using only neighbors from the same sector.
    This preserves sector-specific characteristics and improves imputation quality.

    Args:
        df: Input DataFrame
        columns: Columns to impute (default: all numeric columns)
        sector_column: Name of the sector column for grouping (default: 'sector')
        n_neighbors: Number of neighbors to use for KNN (default: 5)

    Returns:
        DataFrame with imputed values

    Examples:
        >>> df_imputed = impute_missing_values_knn_sector(
        ...     df,
        ...     columns=['revenue', 'ebitda'],
        ...     sector_column='sector',
        ...     n_neighbors=5
        ... )
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove sector column from imputation if present
    columns = [col for col in columns if col != sector_column]

    if not columns:
        logger.warning("No numeric columns to impute")
        return result

    # Check if sector column exists
    if sector_column not in df.columns:
        logger.warning(
            f"Sector column '{sector_column}' not found, falling back to global KNN imputation"
        )
        # Fall back to global KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        result[columns] = imputer.fit_transform(df[columns])
        logger.info(f"Applied global KNN imputation (k={n_neighbors}) to {len(columns)} columns")
        return result

    # Perform sector-aware KNN imputation
    sectors = df[sector_column].dropna().unique()
    imputed_count = 0

    for sector in sectors:
        sector_mask = df[sector_column] == sector
        sector_data = df.loc[sector_mask, columns].copy()

        # Check if sector has enough samples for KNN
        n_samples = sector_data.shape[0]
        if n_samples < 2:
            logger.warning(
                f"Sector '{sector}' has only {n_samples} sample(s), skipping KNN imputation"
            )
            continue

        # Adjust n_neighbors if sector has fewer samples
        k = min(n_neighbors, n_samples - 1)

        # Check if sector has any missing values
        if not sector_data.isna().any().any():
            continue

        # Apply KNN imputation to this sector
        imputer = KNNImputer(n_neighbors=k)
        try:
            sector_imputed = imputer.fit_transform(sector_data)
            # Convert back to DataFrame to preserve column alignment
            sector_imputed_df = pd.DataFrame(
                sector_imputed, index=sector_data.index, columns=sector_data.columns
            )
            result.loc[sector_mask, columns] = sector_imputed_df
            imputed_count += 1
        except Exception as e:
            logger.warning(f"KNN imputation failed for sector '{sector}': {e}. Skipping.")
            continue

    # Handle rows with missing sector values using global imputation
    missing_sector_mask = df[sector_column].isna()
    if missing_sector_mask.any():
        missing_sector_data = df.loc[missing_sector_mask, columns].copy()
        if missing_sector_data.isna().any().any():
            k = min(n_neighbors, missing_sector_data.shape[0] - 1)
            if k > 0:
                imputer = KNNImputer(n_neighbors=k)
                try:
                    missing_imputed = imputer.fit_transform(missing_sector_data)
                    # Convert back to DataFrame to preserve column alignment
                    missing_imputed_df = pd.DataFrame(
                        missing_imputed,
                        index=missing_sector_data.index,
                        columns=missing_sector_data.columns,
                    )
                    result.loc[missing_sector_mask, columns] = missing_imputed_df
                    logger.info(
                        f"Applied global KNN to {missing_sector_mask.sum()} rows with missing sector"
                    )
                except Exception as e:
                    logger.warning(f"KNN imputation failed for missing sectors: {e}")

    logger.info(
        f"Applied sector-aware KNN imputation (k={n_neighbors}) to {imputed_count} sectors "
        f"across {len(columns)} columns"
    )
    return result


def create_scaler_pipeline(
    scaler_type: str = "robust",
    by_sector: bool = True,
) -> Tuple[Any, bool]:
    """Create a scaler for feature scaling.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.scaling.create_scaler_pipeline` instead.

    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Whether to scale separately by sector

    Returns:
        Tuple of (scaler, by_sector flag)
    """
    warnings.warn(
        "create_scaler_pipeline from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.scaling.create_scaler_pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_create_scaler_pipeline(scaler_type, by_sector)


def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler_type: str = "robust",
    by_sector: bool = True,
) -> pd.DataFrame:
    """Scale features using specified scaler.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.scaling.scale_features` instead.

    Args:
        df: Input DataFrame
        columns: Columns to scale (default: all numeric)
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Scale separately by sector (default: True)

    Returns:
        DataFrame with scaled features
    """
    warnings.warn(
        "scale_features from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.scaling.scale_features instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_scale_features(df, columns, scaler_type, by_sector)


# Phase 9.1: Enhanced Two-Step Imputation Strategy


def get_zero_imputation_columns() -> List[str]:
    """Return list of columns for zero imputation (Step 1 of 6-step strategy).

    These columns represent rare/exceptional events (impairments, restructuring,
    acquisitions, etc.) where missing values typically mean the event did not occur.
    Zero is the economically correct imputation.

    Returns:
        List of 48 column names for zero imputation
    """
    return [
        # Impairment of Goodwill (5 columns)
        "impairment_of_goodwill_fq",
        "impairment_of_goodwill_ltm",
        "impairment_of_goodwill_1fy",
        "impairment_of_goodwill_fy",
        "impairment_of_goodwill_5yavgfq",
        # Asset writedown (5 columns)
        "asset_writedown_fq",
        "asset_writedown_ltm",
        "asset_writedown_fy",
        "asset_writedown_1fy",
        "asset_writedown_5yavgfq",
        # Merger & restructuring charges (5 columns)
        "merger_restructuring_charges_fq",
        "merger_restructuring_charges_fy",
        "merger_restructuring_charges_ltm",
        "merger_restructuring_charges_5yavgfq",
        "interest_expense_total_ltm",
        # Restructuring charges (5 columns)
        "restructuring_charges_ltm",
        "restructuring_charges_fq",
        "restructuring_charges_1fy",
        "restructuring_charges_fy",
        "restructuring_charges_5yavgfq",
        # Cash acquisitions (5 columns)
        "cash_acquisitions_fq",
        "cash_acquisitions_ltm",
        "cash_acquisitions_fy",
        "cash_acquisitions_1fy",
        "cash_acquisitions_5yavgfq",
        # Capital expenditure (5 columns)
        "capital_expenditure_ltm",
        "capital_expenditure_1fy",
        "capital_expenditure_fy",
        "capital_expenditure_fq",
        "capital_expenditure_5yavgfq",
        # R&D and Other (6 columns)
        "r_d_expenses_ltm",
        "other_unusual_items_total_ltm",
        "interest_income_on_investments_ltm",
        "volume_shrs",
        "short_int",
        "gain_loss_on_sale_of_assets_ltm",
        # Additional exceptional events (4 columns) - to reach 48 total
        "merger_restructuring_charges_1fy",
        "r_d_expenses_fy",
        "r_d_expenses_fq",
        "r_d_expenses_5yavgfq",
        # Goodwill (5 columns)
        "goodwill_fq",
        "goodwill_ltm",
        "goodwill_fy",
        "goodwill_1fy",
        "goodwill_5yavgfq",
        # Gross intangible assets (3 columns)
        "gross_intangible_assets_ltm",
        "gross_intangible_assets_fy",
        "gross_intangible_assets_5yavgfq",
    ]


def get_knn_imputation_columns() -> List[str]:
    """Return list of columns for KNN imputation (Step 2 of 6-step strategy).

    These are core financial metrics where KNN can leverage sector relationships
    and correlations to provide better estimates than simple statistics.

    Returns:
        List of 148 column names for KNN imputation
    """
    return [
        # Market metrics (3 columns)
        "market_cap",
        "enterprise_value",
        "market_cap_country_r",
        # Analyst ratings (6 columns)
        "analyst_rating",
        "strong_sell_ratings",
        "strong_buys_ratings",
        "hold_ratings",
        "buys_ratings",
        "sell_ratings",
        # Returns (4 columns) - removed tot_return_cagr_10y (redundant with total_return_10y)
        "total_return_ytd",
        "total_return_5y",
        "total_return_10y",
        "tot_return_cagr_3y",
        # Valuation ratios (8 columns)
        "p_e_ntm",
        "p_e_ltm",
        "p_e_1fyltm",
        "p_e_5yavgltm",
        "p_b_ltm",
        "p_b_1fy",
        "p_b_5yavg",
        "p_tbv_ltm",
        # Altman Z-Score (3 columns)
        "altman_z_score_fy",
        "altman_z_score_fq",
        "altman_z_score_ltm",
        # Beta (3 columns)
        "beta_1y",
        "beta_2y",
        "beta_5y",
        # Revenue metrics (8 columns)
        "total_revenues_cagr_5y_fy",
        "total_revenues_fq",
        "total_revenues_1fy",
        "total_revenues_fy",
        "total_revenues_ltm",
        "total_revenues_5yavgfq",
        "total_revenues_5yavgltm",
        "revenues_est_yoy_fy1e",
        # Operating expenses (1 column)
        "total_operating_expenses_ltm",
        # Tangible book value (2 columns)
        "tbv_fy",
        "tbv_ltm",
        # Cash flow metrics (16 columns)
        "cff_ltm",
        "cff_fy",
        "cff_1fy",
        "cff_fq",
        "cfi_ltm",
        "cfi_fy",
        "cfi_1fy",
        "cfi_fq",
        "fcf_ltm",
        "fcf_fy",
        "fcf_fq",
        "fcf_5yavgfq",
        "cfo_ltm",
        "cfo_fy",
        "cfo_1fy",
        "cfo_fq",
        # EBITDA metrics (8 columns) - removed ebitda_5yavgfq (less critical)
        "ebitda_fq",
        "ebitda_ltm",
        "ebitda_fy",
        "ebitda_1fy",
        "ebitda_5yavgltm",
        "ebitda_adj_ltm",
        "ebitda_adj_fy",
        "ebitda_adj_1fy",
        # EBIT metrics (10 columns) - removed ebit_5yavgfq (less critical)
        "ebit_fq",
        "ebit_ltm",
        "ebit_fy",
        "ebit_1fy",
        "ebit_5yavgltm",
        "ebit_adj_1fy",
        "ebit_adj_fy",
        "ebit_adj_ltm",
        "ebit_est_med_fy1e",
        "ebit_est_med_ntm",
        # Profitability metrics (4 columns)
        "return_on_equity_ltm",
        "return_on_equity_fy",
        "return_on_assets_roa_ltm",
        "return_on_assets_roa_fy",
        # Net income metrics (15 columns) - removed net_income_is_5yavgfq and normalized_net_income_5yavgfq (less critical)
        "net_income_is_fy",
        "net_income_is_ltm",
        "net_income_is_1fy",
        "net_income_is_fq",
        "net_income_is_5yavgltm",
        "normalized_net_income_fy",
        "normalized_net_income_ltm",
        "normalized_net_income_1fy",
        "normalized_net_income_fq",
        "normalized_net_income_5yavgltm",
        "net_income_adj_fy",
        "net_income_adj_ltm",
        "net_income_adj_1fy",
        "net_income_adj_fq",
        "net_income_adj_5yavgfq",
        # Margins (2 columns)
        "net_income_margin_fy",
        "net_income_margin_ltm",
        # Volatility (4 columns)
        "volatility_1m",
        "volatility_3m",
        "volatility_6m",
        "volatility_1y",
        # Dividends (5 columns) - removed div_yield_1fyind and div_yield_5yavgltm (less critical)
        "dividend_per_share_ltm",
        "div_yield_ind",
        "div_yield_ltm",
        "div_yield_ttm",
        "div_yield_ntm",
        # Balance sheet items (10 columns)
        "total_debt_fy",
        "total_equity_fy",
        "total_equity_ltm",
        "total_debt_ltm",
        "total_assets_ltm",
        "total_assets_fy",
        "cash_and_equivalents_ltm",
        "cash_and_equivalents_fq",
        "cash_and_equivalents_fy",
        "cash_and_equivalents_5yavgfq",
        # Liquidity ratios (2 columns)
        "current_ratio_fy",
        "current_ratio_ltm",
        # Margins (2 columns)
        "gross_profit_margin_fy",
        "gross_profit_margin_ltm",
        # Turnover (2 columns)
        "asset_turnover_fy",
        "asset_turnover_ltm",
        # Gross profit (2 columns)
        "gross_profit_ltm",
        "gross_profit_fy",
        # EPS metrics (5 columns)
        "eps_norm_est_avg_ntm",
        "eps_adj_1fy",
        "eps_adj_fy",
        "eps_adj_ltm",
        "eps_norm_est_avg_fy1e",
        # Cost and inventory (5 columns)
        "cost_of_revenues_ltm",
        "inventory_ltm",
        "inventory_fq",
        "inventory_fy",
        "inventory_5yavgfq",
        # Operating income (4 columns)
        "operating_income_ltm",
        "operating_income_fy",
        "operating_income_fq",
        "operating_income_5yavgfq",
        # Retained earnings (4 columns)
        "retained_earnings_ltm",
        "retained_earnings_fq",
        "retained_earnings_fy",
        "retained_earnings_5yavgfq",
        # Current assets/liabilities (2 columns)
        "total_current_assets_ltm",
        "total_current_liabilities_ltm",
        # Working capital (4 columns)
        "working_capital_ltm",
        "working_capital_fq",
        "working_capital_fy",
        "working_capital_5yavgfy",
        # Other metrics (4 columns)
        "buyback_yield_ltm",
        "avg_employees_ltm",
        "avg_employees_fy",
        "avg_employees_5yavgfy",
    ]


def apply_zero_imputation(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply zero imputation to specified columns.

    This imputation strategy is appropriate for columns representing rare/exceptional
    events where missing values typically indicate the event did not occur.

    Args:
        df: Input DataFrame
        columns: Columns to zero-impute (default: auto-detect from schema)

    Returns:
        DataFrame with zero-imputed values
    """
    result = df.copy()

    if columns is None:
        columns = get_zero_imputation_columns()

    # Normalize column names to match dataframe
    available_cols = [col for col in columns if col in result.columns]

    if not available_cols:
        logger.warning("No zero-imputation columns found in dataframe")
        return result

    # Apply zero imputation
    for col in available_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            result[col] = result[col].fillna(0)
            logger.debug(f"Zero-imputed {n_missing} values in column '{col}'")

    logger.info(f"Applied zero imputation to {len(available_cols)} columns")
    return result


def apply_knn_imputation_enhanced(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    sector_column: str = "sector",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Apply enhanced KNN imputation with sector awareness.

    This is a wrapper around impute_missing_values_knn_sector that works with
    the predefined KNN imputation column list.

    Args:
        df: Input DataFrame
        columns: Columns for KNN imputation (default: auto-detect from schema)
        sector_column: Name of sector column for grouping
        n_neighbors: Number of neighbors for KNN

    Returns:
        DataFrame with KNN-imputed values
    """
    if columns is None:
        columns = get_knn_imputation_columns()

    # Normalize column names and filter to available columns
    available_cols = [col for col in columns if col in df.columns]

    if not available_cols:
        logger.warning("No KNN-imputation columns found in dataframe")
        return df.copy()

    logger.info(f"Applying KNN imputation to {len(available_cols)} columns")

    # Use existing sector-aware KNN imputation
    return impute_missing_values_knn_sector(
        df=df,
        columns=available_cols,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )


def apply_enhanced_imputation_strategy(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Apply complete two-step imputation strategy from Phase 9.1.

    Step 1: Zero imputation for exceptional event columns
    Step 2: Sector-aware KNN imputation for core financial metrics

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation

    Returns:
        DataFrame with all imputation strategies applied

    Examples:
        >>> # Apply full imputation pipeline
        >>> df_imputed = apply_enhanced_imputation_strategy(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5
        ... )
    """
    logger.info("Starting Phase 9.1 enhanced imputation strategy")

    # Step 1: Zero imputation for exceptional events
    logger.info("Step 1: Applying zero imputation for exceptional event columns")
    result = apply_zero_imputation(df)

    # Step 2: KNN imputation for core financial metrics
    logger.info("Step 2: Applying sector-aware KNN imputation for financial metrics")
    result = apply_knn_imputation_enhanced(
        result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )

    # Log summary statistics
    total_missing_before = df.select_dtypes(include=[np.number]).isna().sum().sum()
    total_missing_after = result.select_dtypes(include=[np.number]).isna().sum().sum()
    reduction = total_missing_before - total_missing_after

    logger.info(
        f"Imputation complete: Reduced missing values from {total_missing_before} "
        f"to {total_missing_after} (reduction: {reduction})"
    )

    return result


# Phase 9.1: Enhanced Four-Step Imputation Strategy


def apply_price_imputation(
    df: pd.DataFrame,
    price_column: str = "last_price",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply price imputation (Step 3 of 6-step strategy).

    Imputes price target columns using the current last_price as the best
    available estimate when analyst targets are missing.

    Args:
        df: Input DataFrame
        price_column: Column to use for imputation (default: "last_price")
        columns: Price target columns to impute (default: all 5 price target columns)

    Returns:
        DataFrame with price-imputed values

    Examples:
        >>> # Impute missing price targets from last_price
        >>> df_imputed = apply_price_imputation(df, price_column='last_price')
    """
    result = df.copy()

    if columns is None:
        columns = [
            "price_target",
            "price_target_low",
            "price_target_median",
            "price_target_high",
            "price_target_ytd_ago",
            "price_5d_ago",
            "price_1w_ago",
            "price_1m_ago",
            "price_3m_ago",
            "price_6m_ago",
            "price_1y_ago",
            "price_3y_ago",
            "price_5y_ago",
            "price_qtd_ago",
        ]

    # Check if price column exists
    if price_column not in result.columns:
        logger.warning(f"Price column '{price_column}' not found in dataframe")
        return result

    # Apply price imputation to available columns
    available_cols = [col for col in columns if col in result.columns]

    if not available_cols:
        logger.warning("No price target columns found in dataframe")
        return result

    for col in available_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            result[col] = result[col].fillna(result[price_column])
            logger.debug(
                f"Price-imputed {n_missing} values in column '{col}' from '{price_column}'"
            )

    logger.info(f"Applied price imputation to {len(available_cols)} columns using '{price_column}'")
    return result


def apply_median_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply median imputation (Step 4 of 6-step strategy).

    Fallback imputation strategy that fills any remaining missing values
    in numerical columns with their median values.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with median-imputed values for all remaining missing numerical data

    Examples:
        >>> # Fill remaining missing values with column medians
        >>> df_complete = apply_median_imputation(df)
    """
    result = df.copy()

    # Get all numeric columns
    numeric_cols = result.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        logger.warning("No numeric columns found in dataframe")
        return result

    total_imputed = 0

    for col in numeric_cols:
        if result[col].isna().any():
            n_missing = result[col].isna().sum()
            median_val = result[col].median()
            result[col] = result[col].fillna(median_val)
            total_imputed += n_missing
            logger.debug(
                f"Median-imputed {n_missing} values in column '{col}' with {median_val:.4f}"
            )

    logger.info(f"Applied median imputation to {total_imputed} total missing values")
    return result


def apply_enhanced_imputation_strategy_4step(
    df: pd.DataFrame,
    sector_column: str = "sector",
    n_neighbors: int = 5,
    price_column: str = "last_price",
) -> pd.DataFrame:
    """Apply complete imputation strategy from Phase 9.1.

    .. deprecated:: Phase 9.1
        Use :func:`finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step` instead.
        This function now calls the 6-step strategy for full coverage.

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation

    Returns:
        DataFrame with complete 6-step imputation applied (zero missing values)
    """
    warnings.warn(
        "apply_enhanced_imputation_strategy_4step from advanced_preprocessing is deprecated. "
        "Use finance_ml.ml_workflow.preprocessing.imputation.apply_enhanced_imputation_strategy_6step instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _new_apply_enhanced_imputation_strategy_6step(
        df, sector_column, n_neighbors, price_column
    )


def prepare_phase95_data(
    df: pd.DataFrame,
    sector_column: str = "sector",
    price_column: str = "last_price",
    n_neighbors: int = 5,
    return_stats: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare data for Phase 9.5 sector-specific regression regression with comprehensive imputation.

    This function applies the complete 6-step imputation strategy with additional validation
    and emergency fallback mechanisms to ensure ZERO NaN and infinite values before model training.
    Designed to prevent the Ridge regression failure caused by 171+ columns containing NaN values.

    The preparation pipeline:
    1. Validates required columns exist
    2. Validates DataFrame is not empty
    3. Logs NaN statistics before imputation
    4. Applies 6-step imputation strategy:
       - Step 1: Zero imputation for exceptional events (48 columns)
       - Step 2: Sector-aware KNN imputation for financial metrics (148 columns)
       - Step 3: Price imputation for price targets (5 columns)
       - Step 4: Median imputation for remaining numeric columns
       - Step 5: Categorical imputation for string/object columns
       - Step 6: Datetime imputation and formatting for date columns
    5. Handles infinite values (replaces with NaN, then re-imputes)
    6. Emergency fallback: fills any remaining NaN with 0
    7. Final validation: confirms zero NaN and infinite values

    Args:
        df: Input DataFrame with financial data (must contain sector and price columns)
        sector_column: Name of sector column for KNN grouping (default: 'sector')
        price_column: Column to use for price target imputation (default: 'last_price')
        n_neighbors: Number of neighbors for KNN imputation (default: 5)
        return_stats: If True, return tuple of (DataFrame, statistics_dict) (default: False)

    Returns:
        If return_stats=False: DataFrame with zero NaN and infinite values (ready for training)
        If return_stats=True: Tuple of (DataFrame, dict with statistics including nan_before,
                              nan_after, inf_count, cols_with_nan_before)

    Raises:
        ValueError: If DataFrame is empty, or required columns are missing

    Examples:
        >>> # Basic usage - prepare Phase 9.5 data
        >>> df_ready = prepare_phase95_data(
        ...     df=all_stocks_phase95,
        ...     sector_column='sector',
        ...     price_column='last_price'
        ... )
        >>> # Verify zero NaN
        >>> assert df_ready.isnull().sum().sum() == 0

        >>> # Get preparation statistics
        >>> df_ready, stats = prepare_phase95_data(
        ...     df=all_stocks_phase95,
        ...     sector_column='sector',
        ...     price_column='last_price',
        ...     return_stats=True
        ... )
        >>> print(f"NaN before: {stats['nan_before']}, after: {stats['nan_after']}")

    Phase 9.5 Integration:
        Insert this at the beginning of Phase 9.5 (before model training):

        ```python
        # Apply comprehensive data preparation with 6-step imputation
        all_stocks_phase95 = prepare_phase95_data(
            df=all_stocks_phase95,
            sector_column='sector',
            price_column='last_price'
        )
        ```
    """
    logger.info("=" * 80)
    logger.info("Phase 9.5 Data Preparation - Comprehensive Imputation Pipeline")
    logger.info("=" * 80)

    # Validation 1: Check for empty DataFrame
    if df.empty:
        raise ValueError("Cannot prepare empty DataFrame for Phase 9.5 training")

    # Validation 2: Check required columns exist
    if sector_column not in df.columns:
        raise ValueError(
            f"Required sector column '{sector_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    if price_column not in df.columns:
        raise ValueError(
            f"Required price column '{price_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Create a copy to avoid modifying original
    result = df.copy()

    # Log NaN statistics BEFORE imputation
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    nan_before = result[numeric_cols].isnull().sum().sum()
    nan_by_col_before = result[numeric_cols].isnull().sum()
    cols_with_nan_before = (nan_by_col_before > 0).sum()

    logger.info(f"\n📊 Missing Values BEFORE Imputation:")
    logger.info(f"  Total NaN values: {nan_before:,}")
    logger.info(f"  Columns with NaN: {cols_with_nan_before}")

    if cols_with_nan_before > 0:
        top_nan_cols = nan_by_col_before[nan_by_col_before > 0].nlargest(10)
        logger.info(f"  Top 10 columns with most NaN:")
        for col, count in top_nan_cols.items():
            logger.info(f"    - {col}: {count} NaN values")

    # Apply comprehensive 6-step imputation strategy
    logger.info("\n🔧 Applying 6-step imputation strategy...")
    result = _new_apply_enhanced_imputation_strategy_6step(
        df=result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
        price_column=price_column,
    )

    # Check NaN after 6-step imputation
    nan_after_imputation = result[numeric_cols].isnull().sum().sum()
    logger.info(f"\n📊 Missing Values AFTER 6-Step Imputation: {nan_after_imputation:,}")

    # Handle infinite values
    logger.info("\n🔧 Checking for infinite values...")
    inf_count_before = np.isinf(result[numeric_cols]).sum().sum()

    if inf_count_before > 0:
        logger.warning(
            f"  Found {inf_count_before} infinite values - replacing with NaN and re-imputing"
        )
        result = result.replace([np.inf, -np.inf], np.nan)

        # Re-apply median imputation to handle converted infinite values
        result = apply_median_imputation(result)

        inf_count_after = np.isinf(result[numeric_cols]).sum().sum()
        logger.info(f"  Infinite values after handling: {inf_count_after}")
    else:
        logger.info("  ✓ No infinite values detected")

    # Emergency fallback: Fill any remaining NaN with 0
    nan_before_fallback = result[numeric_cols].isnull().sum().sum()

    if nan_before_fallback > 0:
        logger.warning(
            f"\n⚠ Emergency Fallback: {nan_before_fallback} NaN values remain after imputation"
        )
        remaining_nan_cols = result[numeric_cols].isnull().sum()
        remaining_nan_cols = remaining_nan_cols[remaining_nan_cols > 0]
        logger.warning(f"  Columns with remaining NaN: {list(remaining_nan_cols.index[:10])}")
        logger.warning("  Applying emergency fillna(0) to ensure training compatibility")

        result = result.fillna(0)

    # Final validation
    nan_final = result[numeric_cols].isnull().sum().sum()
    inf_final = np.isinf(result[numeric_cols]).sum().sum()

    logger.info("\n" + "=" * 80)
    logger.info("Phase 9.5 Data Preparation Complete")
    logger.info("=" * 80)
    logger.info(f"  Final dataset shape: {result.shape}")
    logger.info(f"  Final NaN count: {nan_final}")
    logger.info(f"  Final Inf count: {inf_final}")

    if nan_final == 0 and inf_final == 0:
        logger.info("  ✓ Zero NaN and infinite values confirmed - data ready for model training")
    else:
        logger.error(
            f"  ✗ Data validation FAILED: {nan_final} NaN and {inf_final} Inf values remain"
        )
        raise ValueError(
            f"Phase 9.5 data preparation failed: {nan_final} NaN and {inf_final} Inf remain"
        )

    # Return with or without statistics
    if return_stats:
        stats = {
            "nan_before": int(nan_before),
            "nan_after": int(nan_final),
            "inf_count": int(inf_count_before),
            "cols_with_nan_before": int(cols_with_nan_before),
            "shape": result.shape,
        }
        return result, stats
    else:
        return result
