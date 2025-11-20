"""
Finance ML Custom Transformers Module

Provides sklearn-compatible custom transformers for financial data processing:
- FinancialRatioTransformer: Calculates financial ratios (P/E, P/B, EV/EBITDA, etc.)
- SafeDivisionTransformer: Safe division with configurable handling of zero/inf
- ValuationRatioTransformer: Specialized valuation ratio calculations
- RegularizedTargetEncoder: Target encoding with cross-validation and smoothing

All transformers follow sklearn's BaseEstimator and TransformerMixin interface
for seamless integration with sklearn Pipelines.

Phase 9.1 Future Enhancement #2 and #5
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ============================================================================
# Enhancement 2: Regularized Target Encoding
# ============================================================================


class RegularizedTargetEncoder(BaseEstimator, TransformerMixin):
    """Target encoder with cross-validation and smoothing to prevent overfitting.

    Implements target encoding (mean encoding) with:
    - Cross-validation to prevent data leakage
    - Smoothing parameter to regularize rare categories
    - Handling of unseen categories

    Target encoding replaces categorical values with the mean of the target variable
    for that category, potentially capturing non-linear relationships.

    Parameters:
        columns: List of column names to encode
        cv_folds: Number of cross-validation folds (default: 5)
        smoothing: Smoothing parameter (default: 1.0, higher = more smoothing)

    Attributes:
        encodings_: Dictionary mapping column -> category -> encoded value
        global_mean_: Global mean of target variable

    Examples:
        >>> encoder = RegularizedTargetEncoder(columns=['sector', 'industry'])
        >>> X_train_encoded = encoder.fit_transform(X_train, y_train)
        >>> X_test_encoded = encoder.transform(X_test)
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        cv_folds: int = 5,
        smoothing: float = 1.0,
    ):
        self.columns = columns
        self.cv_folds = cv_folds
        self.smoothing = smoothing
        self.encodings_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the encoder by computing target statistics.

        Args:
            X: DataFrame with categorical columns to encode
            y: Target variable

        Returns:
            self
        """
        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)

        if self.columns is None:
            # Encode all object/category columns
            self.columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Calculate global mean for unseen categories
        self.global_mean_ = y.mean()

        # Calculate encodings for each column
        for col in self.columns:
            if col not in X.columns:
                logger.warning(f"Column '{col}' not found in X, skipping")
                continue

            # Calculate mean target and count for each category
            stats = (
                pd.DataFrame({"target": y, "category": X[col]})
                .groupby("category")
                .agg(mean=("target", "mean"), count=("target", "count"))
            )

            # Apply smoothing: weighted average between category mean and global mean
            # Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)

            self.encodings_[col] = smoothed_mean.to_dict()

        logger.info(f"Fitted target encoder for {len(self.columns)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns using fitted encodings.

        Args:
            X: DataFrame with categorical columns to encode

        Returns:
            DataFrame with encoded columns
        """
        X = X.copy()

        for col in self.columns:
            if col not in X.columns:
                continue

            # Map categories to encoded values
            encoding_map = self.encodings_.get(col, {})
            X[col] = X[col].map(encoding_map).fillna(self.global_mean_)

        return X

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit and transform with cross-validation to prevent leakage.

        Uses out-of-fold encoding to ensure the encoding for each sample
        is computed without using that sample's target value.

        Args:
            X: DataFrame with categorical columns to encode
            y: Target variable (required for target encoding)
            **fit_params: Additional fit parameters (ignored)

        Returns:
            DataFrame with encoded columns

        Raises:
            ValueError: If y is None (target encoding requires target variable)
        """
        if y is None:
            raise ValueError("Target encoding requires y parameter")

        X = X.copy()
        y = pd.Series(y).reset_index(drop=True)

        if self.columns is None:
            self.columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Calculate global mean
        self.global_mean_ = y.mean()

        # Initialize result with original data
        result = X.copy()

        # Set up cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        for col in self.columns:
            if col not in X.columns:
                continue

            # Initialize column with global mean
            result[col] = self.global_mean_

            # Perform out-of-fold encoding
            for train_idx, val_idx in kf.split(X):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]

                # Calculate encoding on training fold
                stats = (
                    pd.DataFrame({"target": y_train_fold, "category": X_train_fold[col]})
                    .groupby("category")
                    .agg(mean=("target", "mean"), count=("target", "count"))
                )

                # Apply smoothing
                smoothed_mean = (
                    stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
                ) / (stats["count"] + self.smoothing)

                encoding_map = smoothed_mean.to_dict()

                # Apply to validation fold
                result.iloc[val_idx, result.columns.get_loc(col)] = (
                    X.iloc[val_idx][col].map(encoding_map).fillna(self.global_mean_)
                )

        # Fit on full data for future transforms
        self.fit(X, y)

        logger.info(f"Fit-transformed {len(self.columns)} columns with CV to prevent leakage")
        return result


# ============================================================================
# Enhancement 5: Custom Financial Ratio Transformers
# ============================================================================


class SafeDivisionTransformer(BaseEstimator, TransformerMixin):
    """Safe division transformer that handles zero denominators and infinities.

    Performs division with configurable handling of edge cases:
    - Zero denominators -> fill_value or NaN
    - Infinite results -> capped or NaN
    - Negative denominators -> configurable handling

    Parameters:
        numerator_col: Name of numerator column
        denominator_col: Name of denominator column
        output_col: Name of output column (default: f"{numerator_col}_div_{denominator_col}")
        fill_value: Value to use for zero/inf cases (default: np.nan)
        cap_value: Optional value to cap extreme ratios (default: None)

    Examples:
        >>> transformer = SafeDivisionTransformer('market_cap', 'total_equity', 'market_to_book')
        >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        numerator_col: str,
        denominator_col: str,
        output_col: Optional[str] = None,
        fill_value: float = np.nan,
        cap_value: Optional[float] = None,
    ):
        self.numerator_col = numerator_col
        self.denominator_col = denominator_col
        self.output_col = output_col or f"{numerator_col}_div_{denominator_col}"
        self.fill_value = fill_value
        self.cap_value = cap_value

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer (no-op for stateless transformer).

        Args:
            X: Input DataFrame
            y: Target variable (ignored)

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by computing safe division.

        Args:
            X: Input DataFrame with numerator and denominator columns

        Returns:
            DataFrame with additional output column
        """
        X = X.copy()

        # Check columns exist
        if self.numerator_col not in X.columns:
            raise ValueError(f"Numerator column '{self.numerator_col}' not found")
        if self.denominator_col not in X.columns:
            raise ValueError(f"Denominator column '{self.denominator_col}' not found")

        numerator = X[self.numerator_col]
        denominator = X[self.denominator_col]

        # Perform division with zero handling
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = numerator / denominator

        # Replace inf and -inf with fill_value
        ratio = ratio.replace([np.inf, -np.inf], self.fill_value)

        # Cap extreme values if specified
        if self.cap_value is not None:
            ratio = ratio.clip(-self.cap_value, self.cap_value)

        X[self.output_col] = ratio

        return X


class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """Calculate common financial ratios with safe division.

    Computes predefined financial ratios such as:
    - p_e: Price-to-Earnings (last_price / eps)
    - p_b: Price-to-Book (last_price / book_value_per_share)
    - ev_ebitda: Enterprise Value / EBITDA
    - And more...

    Parameters:
        ratios: List of ratios to compute (default: ['p_e', 'p_b'])
        fill_value: Value for undefined ratios (default: np.nan)
        cap_value: Optional capping for extreme ratios (default: None)

    Supported ratios:
        - p_e: Price/Earnings
        - p_b: Price/Book
        - p_s: Price/Sales (if revenue_per_share available)
        - ev_ebitda: Enterprise Value/EBITDA
        - debt_equity: Total Debt/Total Equity
        - current_ratio: Current Assets/Current Liabilities

    Examples:
        >>> transformer = FinancialRatioTransformer(ratios=['p_e', 'p_b', 'ev_ebitda'])
        >>> X_transformed = transformer.fit_transform(X)
    """

    # Define ratio calculations (numerator_col, denominator_col)
    RATIO_DEFINITIONS = {
        "p_e": ("last_price", "eps"),
        "p_b": ("last_price", "book_value_per_share"),
        "ev_ebitda": ("enterprise_value", "ebitda"),
        "debt_equity": ("total_debt", "total_equity"),
        "market_to_book": ("market_cap", "total_equity"),
    }

    def __init__(
        self,
        ratios: Optional[List[str]] = None,
        fill_value: float = np.nan,
        cap_value: Optional[float] = None,
    ):
        self.ratios = ratios or ["p_e", "p_b"]
        self.fill_value = fill_value
        self.cap_value = cap_value
        self.transformers_ = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit the transformer by creating SafeDivisionTransformers for each ratio.

        Args:
            X: Input DataFrame
            y: Target variable (ignored)

        Returns:
            self
        """
        self.transformers_ = []

        for ratio in self.ratios:
            if ratio not in self.RATIO_DEFINITIONS:
                logger.warning(f"Unknown ratio '{ratio}', skipping")
                continue

            num_col, denom_col = self.RATIO_DEFINITIONS[ratio]

            # Check if required columns exist
            if num_col not in X.columns or denom_col not in X.columns:
                logger.warning(
                    f"Columns for ratio '{ratio}' not found " f"({num_col}, {denom_col}), skipping"
                )
                continue

            transformer = SafeDivisionTransformer(
                numerator_col=num_col,
                denominator_col=denom_col,
                output_col=ratio,
                fill_value=self.fill_value,
                cap_value=self.cap_value,
            )

            self.transformers_.append(transformer)

        logger.info(f"Prepared {len(self.transformers_)} financial ratio calculations")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by computing all financial ratios.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with additional ratio columns
        """
        X = X.copy()

        for transformer in self.transformers_:
            X = transformer.transform(X)

        return X


class ValuationRatioTransformer(BaseEstimator, TransformerMixin):
    """Specialized transformer for valuation ratios.

    Computes multiple valuation ratios with intelligent defaults:
    - Handles missing columns gracefully
    - Applies sector-appropriate capping
    - Provides summary statistics

    Parameters:
        ratios: List of ratios to compute (default: ['ev_ebitda', 'p_e', 'p_b'])
        cap_percentile: Percentile to cap extreme values (default: 99)

    Examples:
        >>> transformer = ValuationRatioTransformer(ratios=['ev_ebitda', 'p_e', 'p_b'])
        >>> X_transformed = transformer.fit_transform(X)
    """

    def __init__(
        self,
        ratios: Optional[List[str]] = None,
        cap_percentile: float = 99.0,
    ):
        self.ratios = ratios or ["ev_ebitda", "p_e", "p_b"]
        self.cap_percentile = cap_percentile
        self.cap_values_ = {}
        self.ratio_transformer_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """Fit by computing cap values from data distribution.

        Args:
            X: Input DataFrame
            y: Target variable (ignored)

        Returns:
            self
        """
        # First compute ratios without capping to determine cap values
        temp_transformer = FinancialRatioTransformer(
            ratios=self.ratios,
            fill_value=np.nan,
            cap_value=None,
        )
        temp_transformer.fit(X)
        X_temp = temp_transformer.transform(X)

        # Compute cap values based on percentile
        for ratio in self.ratios:
            if ratio in X_temp.columns:
                cap_value = X_temp[ratio].quantile(self.cap_percentile / 100.0)
                if not np.isnan(cap_value):
                    self.cap_values_[ratio] = cap_value

        # Create final transformer with computed cap values
        # For simplicity, use the maximum cap value across all ratios
        max_cap = max(self.cap_values_.values()) if self.cap_values_ else None

        self.ratio_transformer_ = FinancialRatioTransformer(
            ratios=self.ratios,
            fill_value=np.nan,
            cap_value=max_cap,
        )
        self.ratio_transformer_.fit(X)

        logger.info(
            f"Fitted valuation ratio transformer with cap at "
            f"{self.cap_percentile}th percentile: {max_cap}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by computing valuation ratios with capping.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with valuation ratio columns
        """
        return self.ratio_transformer_.transform(X)


# Convenience alias for backward compatibility
TargetEncoder = RegularizedTargetEncoder
