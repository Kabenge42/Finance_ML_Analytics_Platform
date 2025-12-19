from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from finance_ml.ml_workflow.preprocessing.etl import etl_with_features

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DASHBOARD_ROOT = PROJECT_ROOT / "outputs" / "dashboards" / "equities_dashboard"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_EXPORT_PATH = DASHBOARD_ROOT / "equities_dash_df.csv"
DEFAULT_METADATA_PATH = DASHBOARD_ROOT / "metadata.json"
ARTIFACTS_DIR = DASHBOARD_ROOT / "artifacts"
ARTIFACTS_METADATA_PATH = DASHBOARD_ROOT / "artifacts_metadata.json"
DEFAULT_ALERTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "eda"
    / "earnings_analytics"
    / "earnings_quality_alerts.json"
)

DEFAULT_EXPLORER_COLUMNS = [
    "ticker",
    "name",
    "sector",
    "region",
    "last_price",
    "price_target",
    "market_cap",
]


def load_alerts_payload(path: str | Path = DEFAULT_ALERTS_PATH) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_explorer_columns(df: pd.DataFrame, source_label: str) -> None:
    """Log warnings for missing DEFAULT_EXPLORER_COLUMNS in loaded data."""
    if df is None or df.empty:
        return

    missing_cols = [c for c in DEFAULT_EXPLORER_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"Data source '{source_label}' is missing explorer columns: {missing_cols}. "
            "Some Data Explorer features may be limited."
        )
    if "name" not in df.columns:
        logger.warning(
            f"Data source '{source_label}' is missing 'name' column. "
            "Stock names will not be displayed in the Data Explorer."
        )
