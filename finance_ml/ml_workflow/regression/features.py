"""
Compact interaction generators for regression features (Phase 9.5 P2).

Provides utilities to create interaction features between classification
probabilities (event_prob_*) and valuation metrics in a deterministic and
lightweight way, aligned with docs/code_guidelines.md v1.4.
"""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def build_prob_valuation_interactions(
    df: pd.DataFrame,
    valuation_cols: Sequence[str],
    prob_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Create pairwise interaction features: valuation_col x prob_col.

    Naming convention: "{valuation_col}_x_{prob_col}" to keep valuation terms
    first for easier scanning and consistency with docs. The function does not
    introduce NaN values beyond those already present in the input columns
    (multiplication propagates existing NaNs only).

    Args:
        df: Input DataFrame.
        valuation_cols: List/sequence of valuation metric column names.
        prob_cols: Optional list/sequence of classification probability columns. If
                   None, autodetects columns starting with "event_prob_".

    Returns:
        A new DataFrame with added interaction columns.
    """

    out = df.copy()
    if prob_cols is None:
        prob_cols = [c for c in out.columns if c.startswith("event_prob_")]

    if not valuation_cols or not prob_cols:
        return out

    # Debug: Log input shapes for troubleshooting
    logger.debug(
        f"Creating {len(valuation_cols) * len(prob_cols)} interactions: "
        f"{len(valuation_cols)} valuation × {len(prob_cols)} probability columns"
    )

    # Deterministic order: keep original input order for both lists
    for v in valuation_cols:
        if v not in out.columns:
            continue
        for p in prob_cols:
            if p not in out.columns:
                continue

            name = f"{v}_x_{p}"

            # Extract valuation values (should always be 1D)
            v_values = out[v].values
            if v_values.ndim != 1:
                raise ValueError(
                    f"Valuation column '{v}' has unexpected shape {v_values.shape}. "
                    f"Expected 1D array with shape (n_samples,)"
                )

            # Extract probability values with robust 2D handling
            p_values = out[p].values

            # Handle 2D probability arrays (e.g., from duplicate columns or predict_proba)
            if p_values.ndim == 2:
                if p_values.shape[1] == 1:
                    # Single column wrapped in 2D array - flatten
                    p_values = p_values[:, 0]
                    logger.debug(f"Flattened 2D column '{p}' with shape (n, 1) to 1D")
                elif p_values.shape[1] == 2:
                    # Binary classification: take positive class probability (index 1)
                    p_values = p_values[:, 1]
                    logger.warning(
                        f"Probability column '{p}' has shape {out[p].values.shape}. "
                        f"Assuming binary classification and selecting positive class (index 1). "
                        f"If this is incorrect, ensure DataFrame columns are properly structured."
                    )
                else:
                    raise ValueError(
                        f"Probability column '{p}' has shape {p_values.shape}. "
                        f"Expected 1D array or 2D array with shape (n_samples, 1) or (n_samples, 2). "
                        f"For multi-class (>2) probabilities, columns should be separate."
                    )
            elif p_values.ndim != 1:
                raise ValueError(
                    f"Probability column '{p}' has unexpected dimensionality {p_values.ndim}. "
                    f"Expected 1D or 2D array, got shape {p_values.shape}"
                )

            # Validate shapes match
            if len(v_values) != len(p_values):
                raise ValueError(
                    f"Shape mismatch: '{v}' has {len(v_values)} rows, "
                    f"'{p}' has {len(p_values)} rows after reshaping"
                )

            # Create interaction feature
            out[name] = v_values * p_values

    # Normalize all column names to Python str (not numpy.str_)
    # This prevents scikit-learn TypeError about mixed column name types
    out.columns = out.columns.astype(str)

    return out
