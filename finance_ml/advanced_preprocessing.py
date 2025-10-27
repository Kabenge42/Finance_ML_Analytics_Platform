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
