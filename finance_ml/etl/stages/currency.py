"""Currency conversion stage for ETL."""

import logging
from typing import List, Optional

import pandas as pd

from finance_ml.etl.currency import CurrencyConverter

logger = logging.getLogger(__name__)


def run_currency_conversion_stage(
    df: pd.DataFrame,
    enabled: bool = True,
    target_currency: str = "USD",
    currency_column: str = "unit",
    date_column: str = "reference_date",
    columns_to_convert: Optional[List[str]] = None,
    suffix: str = "_usd",
    cache_rates: bool = True,
    max_fallback_days: int = 7,
    use_business_day_fallback: bool = True,
) -> pd.DataFrame:
    """
    Stage 5.5: Currency conversion.

    Converts monetary columns from local currency to a target currency.
    """
    if not enabled:
        return df

    logger.info(f"Stage 5.5: Applying currency conversion to {target_currency}")

    converter = CurrencyConverter(
        target_currency=target_currency,
        cache_rates=cache_rates,
        max_fallback_days=max_fallback_days,
        use_business_day_fallback=use_business_day_fallback,
    )

    return converter.convert_dataframe(
        df,
        currency_column=currency_column,
        date_column=date_column,
        columns_to_convert=columns_to_convert,
        suffix=suffix,
    )
