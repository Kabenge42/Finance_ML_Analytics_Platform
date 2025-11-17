"""Public API for feature engineering presets (Phase 9.3 Week 9).

Exposes a single entry point :func:`build_features` that composes feature
groups from core and advanced modules using named presets, while preserving
backward compatibility with existing orchestrators.

Presets
-------

- ``"basic"``: core ratios, margins, volatility, and revenue CAGR.
- ``"momentum"``: momentum & technical indicators only.
- ``"quality"``: accounting quality and financial distress signals.
- ``"cashflow"``: cash flow quality and capital allocation metrics.
- ``"comprehensive"``: full advanced feature set (same as
  :func:`advanced.build_comprehensive_features`).
- ``"full_enhanced"``: alias for ``"comprehensive"``.

Backwards compatibility
-----------------------

The :func:`advanced.build_comprehensive_features` orchestrator remains
available and unchanged in its default behavior. This API simply provides a
user-friendly front end that delegates to the underlying core/advanced
functions.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

from finance_ml.ml_workflow.features import core as core_feats, advanced as adv_feats

logger = logging.getLogger(__name__)

PresetName = Literal[
    "basic",
    "momentum",
    "quality",
    "cashflow",
    "comprehensive",
    "comprehensive_v2",
    "full_enhanced",
]


def build_features(
    df: pd.DataFrame,
    preset: PresetName = "comprehensive",
    *,
    include_interactions: bool = True,
    include_relative: bool = True,
    sector_col: str = "sector",
) -> pd.DataFrame:
    """Build features using a named preset.

    Args:
        df: Input DataFrame.
        preset: One of {"basic", "momentum", "quality", "cashflow",
            "comprehensive", "full_enhanced"}.
        include_interactions: For comprehensive presets, whether to add
            interaction features.
        include_relative: For comprehensive presets, whether to add
            relative/sector-based features.
        sector_col: Sector column name (used by some feature groups).

    Returns:
        DataFrame with engineered features added.
    """
    preset_norm = (preset or "comprehensive").lower()

    if preset_norm == "basic":
        result = df.copy()
        result = core_feats.engineer_basic_ratios(result)
        result = core_feats.engineer_margin_features(result)
        # Keep volatility window default to avoid heavy computation
        result = core_feats.engineer_volatility_features(result)
        result = core_feats.engineer_revenue_cagr(result)
        logger.info("Built BASIC features preset")
        return result

    if preset_norm == "momentum":
        result = df.copy()
        result = adv_feats.engineer_momentum_features(result)
        # Hygiene: replace any possible infinities
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built MOMENTUM features preset")
        return result

    if preset_norm == "quality":
        result = df.copy()
        # Quality & risk signals
        result = adv_feats.engineer_accounting_quality_features(result)
        result = adv_feats.engineer_financial_distress_features(result)
        # Include analyst quality signals as part of broader quality theme (harmless if columns absent)
        result = adv_feats.engineer_analyst_quality_features(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built QUALITY features preset")
        return result

    if preset_norm == "cashflow":
        result = df.copy()
        result = adv_feats.engineer_cash_flow_quality_features(result)
        result = adv_feats.engineer_capital_allocation_features(result)
        result = result.replace([float("inf"), float("-inf")], pd.NA)
        logger.info("Built CASHFLOW features preset")
        return result

    if preset_norm in ("comprehensive", "comprehensive_v2", "full_enhanced"):
        logger.info("Building COMPREHENSIVE features via advanced.build_comprehensive_features")
        return adv_feats.build_comprehensive_features(
            df,
            include_interactions=include_interactions,
            include_relative_values=include_relative,
            sector_col=sector_col,
        )

    raise ValueError(
        f"Unknown preset '{preset}'. Expected one of: basic, momentum, quality, cashflow, comprehensive, comprehensive_v2, full_enhanced"
    )
