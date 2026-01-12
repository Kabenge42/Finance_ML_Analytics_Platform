"""
Currency Conversion Stage for ETL Pipeline.

Stage 8: Converts monetary columns from local currencies to target currency.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import pandas as pd

from finance_ml.etl.config import CurrencyConversionConfig
from finance_ml.etl.currency import (
    CurrencyConverter,
    CurrencyConversionMetrics,
    MONETARY_COLUMNS,
)

logger = logging.getLogger(__name__)


@dataclass
class CurrencyStageResult:
    """Result container for currency conversion stage."""

    df: pd.DataFrame
    metrics: CurrencyConversionMetrics
    columns_added: int


def run_currency_conversion_stage(
    df: pd.DataFrame,
    config: CurrencyConversionConfig,
) -> Tuple[pd.DataFrame, CurrencyConversionMetrics]:
    """
    Execute currency conversion as ETL Stage 8.

    Args:
        df: DataFrame from previous ETL stage (post-imputation)
        config: Currency conversion configuration

    Returns:
        Tuple of (converted DataFrame, conversion metrics)

    Raises:
        ValueError: If currency_column not found in DataFrame
    """
    if not config.enabled:
        logger.info("Currency conversion stage skipped (disabled)")
        return df, CurrencyConversionMetrics()

    logger.info(f"Stage 8: Currency conversion to {config.target_currency}")

    converter = CurrencyConverter(
        target_currency=config.target_currency,
        cache_rates=config.cache_rates,
        max_fallback_days=config.max_fallback_days,
        use_business_day_fallback=config.use_business_day_fallback,
    )

    columns_to_convert = config.columns or MONETARY_COLUMNS

    df_converted = converter.convert_dataframe(
        df,
        currency_column=config.currency_column,
        date_column=config.date_column,
        columns_to_convert=columns_to_convert,
        suffix=config.suffix,
        inplace=False,
    )

    metrics = converter.get_metrics()
    logger.info(metrics.summary())

    return df_converted, metrics
