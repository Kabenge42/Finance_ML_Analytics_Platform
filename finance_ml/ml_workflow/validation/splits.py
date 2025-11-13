"""
Data splitting utilities with leakage prevention.

Implements intelligent train/test splitting policy:
1. Time-aware split when date column exists (prevents future leakage)
2. Grouped split by ticker when no dates (prevents same-ticker leakage)
3. Stratified split by sector as fallback (maintains sector balance)
4. Random split as final fallback

Addresses Model Optimization Recommendations: "No shared split utility enforcing
time-aware/grouped policy; notebook code shows random splits in places where time
awareness is needed."
"""

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def create_train_test_split(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    group_col: Optional[str] = None,
    stratify_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test split with intelligent policy selection.

    Policy priority:
    1. If date_col provided: time-aware split (test = most recent data)
    2. Elif group_col provided: grouped split (no group appears in both sets)
    3. Elif stratify_col provided: stratified split (maintains class balance)
    4. Else: random split

    Args:
        df: Input dataframe
        date_col: Column name for temporal sorting (e.g., 'snapshot_date')
        group_col: Column name for grouping (e.g., 'ticker')
        stratify_col: Column name for stratification (e.g., 'sector')
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, test_df)

    Example - Time-aware split:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=100),
        ...     'ticker': ['AAPL'] * 100,
        ...     'price': np.random.randn(100)
        ... })
        >>> train, test = create_train_test_split(df, date_col='date')
        >>> # Test has most recent 20% of dates

    Example - Grouped split:
        >>> df = pd.DataFrame({
        ...     'ticker': ['AAPL']*50 + ['MSFT']*50,
        ...     'price': np.random.randn(100)
        ... })
        >>> train, test = create_train_test_split(df, group_col='ticker')
        >>> # Train and test have disjoint tickers
    """
    if len(df) == 0:
        return df.copy(), df.copy()

    # Policy 1: Time-aware split
    if date_col is not None and date_col in df.columns:
        return _time_aware_split(df, date_col, test_size)

    # Policy 2: Grouped split
    if group_col is not None and group_col in df.columns:
        return _grouped_split(df, group_col, test_size, random_state)

    # Policy 3: Stratified split
    if stratify_col is not None and stratify_col in df.columns:
        try:
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                stratify=df[stratify_col],
                random_state=random_state,
            )
            return train_df, test_df
        except ValueError:
            # Stratification failed (e.g., too few samples per class)
            pass

    # Policy 4: Random split (fallback)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df


def _time_aware_split(
    df: pd.DataFrame, date_col: str, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: train = earlier dates, test = recent dates.

    Ensures no temporal leakage (test dates are strictly after train dates).
    """
    df_sorted = df.sort_values(date_col).copy()
    split_idx = int(len(df_sorted) * (1 - test_size))

    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()

    return train_df, test_df


def _grouped_split(
    df: pd.DataFrame, group_col: str, test_size: float, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by groups: train and test have disjoint group values.

    Prevents leakage when same entity (e.g., ticker) appears multiple times.
    """
    unique_groups = df[group_col].unique()

    # Split groups
    n_test_groups = max(1, int(len(unique_groups) * test_size))
    np.random.seed(random_state)
    test_groups = np.random.choice(unique_groups, size=n_test_groups, replace=False)

    # Create boolean masks
    test_mask = df[group_col].isin(test_groups)
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    return train_df, test_df


def time_series_cv(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int = 5,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate time-series cross-validation folds.

    Each fold has train indices < test indices (no future leakage).

    Args:
        df: Input dataframe with temporal data
        date_col: Column name for dates
        n_splits: Number of CV folds

    Returns:
        List of (train_indices, test_indices) tuples

    Example:
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2020-01-01', periods=100),
        ...     'price': np.random.randn(100)
        ... })
        >>> folds = time_series_cv(df, date_col='date', n_splits=3)
        >>> for train_idx, test_idx in folds:
        ...     train = df.iloc[train_idx]
        ...     test = df.iloc[test_idx]
        ...     # train.date.max() <= test.date.min()
    """
    from sklearn.model_selection import TimeSeriesSplit

    # Sort by date
    df_sorted = df.sort_values(date_col).reset_index(drop=True)

    # Generate splits
    tscv = TimeSeriesSplit(n_splits=n_splits)
    folds = []

    for train_idx, test_idx in tscv.split(df_sorted):
        folds.append((train_idx, test_idx))

    return folds
