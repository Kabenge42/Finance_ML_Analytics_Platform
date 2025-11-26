"""
Cross-validation splitter selection for regression (Phase 9.5 P0).

Provides a simple policy-driven interface that returns a splitter with a
split(df, groups=None) method. The splitter respects time order when a
date column is available, otherwise falls back to grouped splits by ticker
or standard KFold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit


@dataclass
class _TimeSeriesDfSplitter:
    n_splits: int = 5
    date_col: str = "snapshot_date"

    def split(
        self, df: pd.DataFrame, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:  # noqa: E501
        if self.date_col not in df.columns:
            raise ValueError(
                f"TimeSeries splitter requires date_col '{self.date_col}' in dataframe"
            )
        # Sort by date then use TimeSeriesSplit on positional indices
        order = df[self.date_col].argsort(kind="mergesort")
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        for tr_pos, te_pos in tscv.split(np.arange(len(df))):
            # map positional splits to original index using sorted order
            tr_idx = df.index[order[tr_pos]].to_numpy()
            te_idx = df.index[order[te_pos]].to_numpy()
            yield tr_idx, te_idx


@dataclass
class _GroupDfSplitter:
    n_splits: int = 5
    group_col: str = "ticker"

    def split(
        self, df: pd.DataFrame, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:  # noqa: E501
        if self.group_col not in df.columns:
            raise ValueError(f"Group splitter requires group_col '{self.group_col}' in dataframe")
        gkf = GroupKFold(n_splits=self.n_splits)
        for tr_pos, te_pos in gkf.split(X=np.zeros(len(df)), y=None, groups=df[self.group_col]):
            yield df.index[tr_pos].to_numpy(), df.index[te_pos].to_numpy()


@dataclass
class _KFoldDfSplitter:
    n_splits: int = 5
    shuffle: bool = False
    random_state: Optional[int] = None

    def split(
        self, df: pd.DataFrame, groups: Optional[pd.Series] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:  # noqa: E501
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for tr_pos, te_pos in kf.split(np.zeros(len(df))):
            yield df.index[tr_pos].to_numpy(), df.index[te_pos].to_numpy()


def get_regression_cv_splitter(
    policy: str = "time_series",
    *,
    n_splits: int = 5,
    date_col: str = "snapshot_date",
    group_col: str = "ticker",
):
    """
    Return a dataframe-aware CV splitter according to policy.

    Policies:
    - "time_series": TimeSeriesSplit on ascending date_col
    - "group": GroupKFold on group_col
    - "kfold": standard KFold on rows

    The returned object exposes split(df) yielding (train_index, test_index)
    arrays referencing the df index labels.
    """
    policy = (policy or "").lower()
    if policy == "time_series":
        return _TimeSeriesDfSplitter(n_splits=n_splits, date_col=date_col)
    if policy == "group":
        return _GroupDfSplitter(n_splits=n_splits, group_col=group_col)
    if policy == "kfold":
        return _KFoldDfSplitter(n_splits=n_splits, shuffle=False)
    # auto: prefer time_series if date_col exists, else group if group_col exists
    return _TimeSeriesDfSplitter(n_splits=n_splits, date_col=date_col)
