"""
finance_ml.advanced_preprocessing - Advanced preprocessing for Phase 9

This module implements sophisticated preprocessing techniques including:
- Robust outlier detection (IQR, z-score, isolation forest)
- Sector-specific winsorization
- Data quality scoring and monitoring
- Temporal validation and time-aware splits
- Advanced imputation strategies

Part of Phase 9.1 implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Container for data quality metrics."""

    completeness_score: float  # % of non-null values
    consistency_score: float  # % of values within expected ranges
    validity_score: float  # % of valid data types and formats
    overall_score: float  # Weighted average of above
    issues: List[str]  # List of detected issues
    metrics: Dict[str, Any]  # Detailed metrics

    def __str__(self) -> str:
        """String representation of quality report."""
        return (
            f"Data Quality Report:\n"
            f"  Overall Score: {self.overall_score:.2%}\n"
            f"  Completeness: {self.completeness_score:.2%}\n"
            f"  Consistency: {self.consistency_score:.2%}\n"
            f"  Validity: {self.validity_score:.2%}\n"
            f"  Issues: {len(self.issues)}"
        )


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    by_sector: bool = True,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Detect outliers using Interquartile Range (IQR) method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers (default: all numeric)
        by_sector: Apply IQR separately by sector (default: True)
        iqr_multiplier: IQR multiplier for bounds (default: 1.5)

    Returns:
        DataFrame with boolean columns indicating outliers (col_outlier)
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        outlier_col = f"{col}_outlier"

        if by_sector and "sector" in df.columns:
            # Sector-specific IQR detection
            result[outlier_col] = False

            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                sector_data = df.loc[mask, col].dropna()

                if len(sector_data) < 4:
                    continue

                q1 = sector_data.quantile(0.25)
                q3 = sector_data.quantile(0.75)
                iqr = q3 - q1

                lower_bound = q1 - iqr_multiplier * iqr
                upper_bound = q3 + iqr_multiplier * iqr

                outliers = (df.loc[mask, col] < lower_bound) | (df.loc[mask, col] > upper_bound)
                result.loc[mask, outlier_col] = outliers
        else:
            # Global IQR detection
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            result[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)

    logger.info(f"Detected outliers for {len(columns)} columns using IQR method")
    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    threshold: float = 3.0,
    by_sector: bool = True,
) -> pd.DataFrame:
    """Detect outliers using z-score method.

    Args:
        df: Input DataFrame
        columns: Columns to check (default: all numeric)
        threshold: Z-score threshold (default: 3.0)
        by_sector: Apply z-score separately by sector (default: True)

    Returns:
        DataFrame with boolean columns indicating outliers (col_zscore_outlier)
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        outlier_col = f"{col}_zscore_outlier"

        if by_sector and "sector" in df.columns:
            # Sector-specific z-score
            result[outlier_col] = False

            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                sector_data = df.loc[mask, col].dropna()

                if len(sector_data) < 3:
                    continue

                mean = sector_data.mean()
                std = sector_data.std()

                if std == 0:
                    continue

                z_scores = np.abs((df.loc[mask, col] - mean) / std)
                result.loc[mask, outlier_col] = z_scores > threshold
        else:
            # Global z-score
            mean = df[col].mean()
            std = df[col].std()

            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                result[outlier_col] = z_scores > threshold
            else:
                result[outlier_col] = False

    logger.info(f"Detected outliers for {len(columns)} columns using z-score method")
    return result


def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest algorithm.

    Args:
        df: Input DataFrame
        columns: Columns to use for detection (default: all numeric)
        contamination: Expected proportion of outliers (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        Boolean Series indicating outliers (True = outlier)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select and clean data
    data = df[columns].copy()

    # Fill missing values with median for algorithm compatibility
    for col in data.columns:
        if data[col].isna().any():
            data[col].fillna(data[col].median(), inplace=True)

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)

    predictions = iso_forest.fit_predict(data)
    outliers = predictions == -1  # -1 indicates outlier

    logger.info(f"Detected {outliers.sum()} outliers using Isolation Forest")
    return pd.Series(outliers, index=df.index)


def winsorize_by_sector(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99,
    by_sector: bool = True,
) -> pd.DataFrame:
    """Winsorize extreme values by replacing with percentile bounds.

    Args:
        df: Input DataFrame
        columns: Columns to winsorize (default: all numeric)
        lower_percentile: Lower percentile bound (default: 0.01 = 1%)
        upper_percentile: Upper percentile bound (default: 0.99 = 99%)
        by_sector: Apply winsorization separately by sector (default: True)

    Returns:
        DataFrame with winsorized values
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue

        if by_sector and "sector" in df.columns:
            # Sector-specific winsorization
            for sector in df["sector"].dropna().unique():
                mask = df["sector"] == sector
                sector_data = df.loc[mask, col]

                if sector_data.isna().all():
                    continue

                lower_bound = sector_data.quantile(lower_percentile)
                upper_bound = sector_data.quantile(upper_percentile)

                result.loc[mask, col] = sector_data.clip(lower=lower_bound, upper=upper_bound)
        else:
            # Global winsorization
            lower_bound = df[col].quantile(lower_percentile)
            upper_bound = df[col].quantile(upper_percentile)

            result[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    logger.info(
        f"Winsorized {len(columns)} columns (percentiles: {lower_percentile:.2%}-{upper_percentile:.2%})"
    )
    return result


def calculate_data_quality_score(df: pd.DataFrame) -> DataQualityReport:
    """Calculate comprehensive data quality metrics.

    Args:
        df: Input DataFrame

    Returns:
        DataQualityReport with detailed quality metrics
    """
    issues = []
    metrics = {}

    # 1. Completeness: % of non-null values
    total_cells = df.shape[0] * df.shape[1]
    non_null_cells = df.count().sum()
    completeness = non_null_cells / total_cells if total_cells > 0 else 0

    metrics["total_cells"] = total_cells
    metrics["non_null_cells"] = non_null_cells
    metrics["missing_cells"] = total_cells - non_null_cells

    if completeness < 0.8:
        issues.append(f"Low completeness: {completeness:.2%} (expected >80%)")

    # 2. Consistency: Check numeric ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    consistency_checks = 0
    consistency_passes = 0

    for col in numeric_cols:
        if col in df.columns and not df[col].isna().all():
            consistency_checks += 1

            # Check for infinite values
            if np.isinf(df[col]).any():
                issues.append(f"Column '{col}' contains infinite values")
            else:
                consistency_passes += 1

            # Check for negative values in typically positive columns
            if any(x in col.lower() for x in ["market_cap", "price", "revenue", "assets"]):
                if (df[col] < 0).any():
                    issues.append(f"Column '{col}' contains unexpected negative values")

    consistency = consistency_passes / consistency_checks if consistency_checks > 0 else 1.0
    metrics["consistency_checks"] = consistency_checks
    metrics["consistency_passes"] = consistency_passes

    # 3. Validity: Check data types and formats
    validity_checks = 0
    validity_passes = 0

    # Check if expected columns exist
    expected_cols = ["ticker", "sector", "region"]
    for col in expected_cols:
        validity_checks += 1
        if col in df.columns:
            validity_passes += 1
        else:
            issues.append(f"Missing expected column: '{col}'")

    validity = validity_passes / validity_checks if validity_checks > 0 else 0
    metrics["validity_checks"] = validity_checks
    metrics["validity_passes"] = validity_passes

    # Overall score (weighted average)
    overall = completeness * 0.4 + consistency * 0.3 + validity * 0.3

    report = DataQualityReport(
        completeness_score=completeness,
        consistency_score=consistency,
        validity_score=validity,
        overall_score=overall,
        issues=issues,
        metrics=metrics,
    )

    logger.info(f"Data quality assessment complete: {overall:.2%} overall score")
    return report


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

    Args:
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Whether to scale separately by sector

    Returns:
        Tuple of (scaler, by_sector flag)
    """
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "robust":
        scaler = RobustScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    logger.info(f"Created {scaler_type} scaler (sector-specific: {by_sector})")
    return scaler, by_sector


def scale_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    scaler_type: str = "robust",
    by_sector: bool = True,
) -> pd.DataFrame:
    """Scale features using specified scaler.

    Args:
        df: Input DataFrame
        columns: Columns to scale (default: all numeric)
        scaler_type: Type of scaler ('standard', 'robust', 'minmax')
        by_sector: Scale separately by sector (default: True)

    Returns:
        DataFrame with scaled features
    """
    result = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    scaler, _ = create_scaler_pipeline(scaler_type, by_sector)

    if by_sector and "sector" in df.columns:
        # Sector-specific scaling
        for sector in df["sector"].dropna().unique():
            mask = df["sector"] == sector
            sector_data = df.loc[mask, columns]

            if sector_data.shape[0] > 1:
                result.loc[mask, columns] = scaler.fit_transform(sector_data)
    else:
        # Global scaling
        result[columns] = scaler.fit_transform(df[columns])

    logger.info(f"Scaled {len(columns)} features using {scaler_type} scaler")
    return result


# Phase 9.1: Enhanced Two-Step Imputation Strategy


def get_zero_imputation_columns() -> List[str]:
    """Return list of columns for zero imputation (Step 1 of 4-step strategy).

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
    """Return list of columns for KNN imputation (Step 2 of 4-step strategy).

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
    """Apply price imputation (Step 3 of 4-step strategy).

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
    """Apply median imputation (Step 4 of 4-step strategy).

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
    """Apply complete 4-step imputation strategy from Phase 9.1.

    Step 1: Zero imputation for exceptional event columns (48 columns)
    Step 2: Sector-aware KNN imputation for core financial metrics (148 columns)
    Step 3: Price imputation for price target columns (5 columns)
    Step 4: Median imputation for all remaining numerical columns

    This ensures zero missing values in the output dataframe.

    Args:
        df: Input DataFrame with financial data
        sector_column: Name of sector column for KNN grouping
        n_neighbors: Number of neighbors for KNN imputation
        price_column: Column to use for price target imputation

    Returns:
        DataFrame with complete 4-step imputation applied (zero missing values)

    Examples:
        >>> # Apply complete 4-step imputation pipeline
        >>> df_complete = apply_enhanced_imputation_strategy_4step(
        ...     all_stocks,
        ...     sector_column='sector',
        ...     n_neighbors=5,
        ...     price_column='last_price'
        ... )
        >>> # Verify no missing values remain
        >>> assert df_complete.select_dtypes(include=[np.number]).isna().sum().sum() == 0
    """
    logger.info("Starting Phase 9.1 enhanced 4-step imputation strategy")

    # Track missing values at each step
    missing_initial = df.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"Initial missing values: {missing_initial}")

    # Step 1: Zero imputation for exceptional events
    logger.info("Step 1: Applying zero imputation for exceptional event columns (48 cols)")
    result = apply_zero_imputation(df)
    missing_after_step1 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 1: {missing_after_step1} missing values remain")

    # Step 2: KNN imputation for core financial metrics
    logger.info("Step 2: Applying sector-aware KNN imputation for financial metrics (148 cols)")
    result = apply_knn_imputation_enhanced(
        result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
    )
    missing_after_step2 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 2: {missing_after_step2} missing values remain")

    # Step 3: Price imputation for price targets
    logger.info("Step 3: Applying price imputation for price target columns (5 cols)")
    result = apply_price_imputation(result, price_column=price_column)
    missing_after_step3 = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 3: {missing_after_step3} missing values remain")

    # Step 4: Median imputation for remaining columns
    logger.info("Step 4: Applying median imputation for remaining columns")
    result = apply_median_imputation(result)
    missing_final = result.select_dtypes(include=[np.number]).isna().sum().sum()
    logger.info(f"After Step 4: {missing_final} missing values remain")

    # Log summary
    total_reduction = missing_initial - missing_final
    logger.info(
        f"4-step imputation complete: Reduced missing values from {missing_initial} "
        f"to {missing_final} (reduction: {total_reduction})"
    )

    if missing_final > 0:
        logger.warning(
            f"Warning: {missing_final} missing values still remain after 4-step imputation"
        )

    return result


def prepare_phase95_data(
    df: pd.DataFrame,
    sector_column: str = "sector",
    price_column: str = "last_price",
    n_neighbors: int = 5,
    return_stats: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """Prepare data for Phase 9.5 sector-specific regression models with comprehensive imputation.

    This function applies the complete 4-step imputation strategy with additional validation
    and emergency fallback mechanisms to ensure ZERO NaN and infinite values before model training.
    Designed to prevent the Ridge regression failure caused by 171+ columns containing NaN values.

    The preparation pipeline:
    1. Validates required columns exist
    2. Validates DataFrame is not empty
    3. Logs NaN statistics before imputation
    4. Applies 4-step imputation strategy:
       - Step 1: Zero imputation for exceptional events (48 columns)
       - Step 2: Sector-aware KNN imputation for financial metrics (148 columns)
       - Step 3: Price imputation for price targets (5 columns)
       - Step 4: Median imputation for remaining columns
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
        # Apply comprehensive data preparation with 4-step imputation
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

    # Apply comprehensive 4-step imputation strategy
    logger.info("\n🔧 Applying 4-step imputation strategy...")
    result = apply_enhanced_imputation_strategy_4step(
        df=result,
        sector_column=sector_column,
        n_neighbors=n_neighbors,
        price_column=price_column,
    )

    # Check NaN after 4-step imputation
    nan_after_imputation = result[numeric_cols].isnull().sum().sum()
    logger.info(f"\n📊 Missing Values AFTER 4-Step Imputation: {nan_after_imputation:,}")

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
