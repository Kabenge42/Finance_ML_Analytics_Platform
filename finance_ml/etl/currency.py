"""Foreign Currency Conversion Module for Equities Data."""

from typing import List, Optional, Dict, Any, Tuple, Set, FrozenSet
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np

# Use internal converter instead of forex-python
from finance_ml.etl.converter import (
    CurrencyRates,
    RatesNotAvailableError,
    get_rate,
    convert,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration Constants
# =============================================================================

# Maximum number of business days to search backwards for available rates
MAX_FALLBACK_DAYS: int = 7

# Default currencies that are always available (no conversion needed)
NO_CONVERSION_CURRENCIES: FrozenSet[str] = frozenset({"USD"})

# Supported currency codes for validation
SUPPORTED_CURRENCIES: FrozenSet[str] = frozenset(
    {
        "USD",
        "EUR",
        "GBP",
        "JPY",
        "CHF",
        "CAD",
        "AUD",
        "NZD",
        "CNY",
        "HKD",
        "SGD",
        "KRW",
        "INR",
        "BRL",
        "MXN",
        "ZAR",
        "SEK",
        "NOK",
        "DKK",
        "PLN",
        "CZK",
        "HUF",
        "TRY",
        "ILS",
        "THB",
        "MYR",
        "IDR",
        "PHP",
        "TWD",
        "RUB",
    }
)

def find_most_recent_business_day(
    target_date: datetime, max_lookback_days: int = MAX_FALLBACK_DAYS
) -> datetime:
    """Find the most recent business day on or before the target date."""
    current = target_date
    # Iterates backward to find most recent business day
    for _ in range(max_lookback_days):
        if current.weekday() < 5:
            month_day = (current.month, current.day)
            common_holidays = {(1, 1), (12, 25), (12, 26)}
            if month_day not in common_holidays:
                return current
        current = current - timedelta(days=1)
    return current


def load_monetary_columns() -> List[str]:
    """Load monetary columns from configuration file."""
    config_path = Path(__file__).parent / "config" / "monetary_columns.json"
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            return [
                col
                for category_cols in config.get("categories", {}).values()
                for col in category_cols
            ]
        except Exception as e:
            logger.error(f"Error loading monetary_columns.json: {e}")

    logger.warning("monetary_columns.json not found or invalid, using empty list")
    return []


MONETARY_COLUMNS: List[str] = load_monetary_columns()


@dataclass
class CurrencyConversionMetrics:
    """Metrics tracking for currency conversion operations."""

    rows_processed: int = 0
    rows_converted: int = 0
    rows_failed: int = 0
    fallback_used_count: int = 0
    unique_currencies: Set[str] = field(default_factory=set)
    columns_converted: int = 0
    conversion_errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary of conversion metrics."""
        success_rate = (
            (self.rows_converted / self.rows_processed * 100) if self.rows_processed > 0 else 0.0
        )
        return (
            f"Currency Conversion: {self.rows_converted}/{self.rows_processed} rows "
            f"({success_rate:.1f}%), {self.columns_converted} columns, "
            f"{len(self.unique_currencies)} currencies, "
            f"{self.fallback_used_count} fallbacks, {len(self.conversion_errors)} errors"
        )


class RateFetchingService:
    """Service for fetching exchange rates with fallback and caching."""

    def __init__(
        self,
        converter: CurrencyRates,
        cache_rates: bool = True,
        max_fallback_days: int = MAX_FALLBACK_DAYS,
        use_business_day_fallback: bool = True,
    ):
        self._converter = converter
        self._cache_rates = cache_rates
        self._max_fallback_days = max_fallback_days
        self._use_fallback = use_business_day_fallback
        self._rate_cache: Dict[str, float] = {}
        self._fallback_count = 0

    def get_rate(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime] = None,
    ) -> Tuple[Optional[float], bool]:
        """Fetch exchange rate with fallback strategy."""
        from_currency = from_currency.upper().strip()
        if from_currency == to_currency:
            return 1.0, False
        if from_currency not in SUPPORTED_CURRENCIES:
            logger.warning(f"Unsupported currency: {from_currency}")
            return None, False

        cache_key = (
            f"{from_currency}_{to_currency}_{date.strftime('%Y-%m-%d') if date else 'latest'}"
        )
        if self._cache_rates and cache_key in self._rate_cache:
            return self._rate_cache[cache_key], False

        return self._fetch_with_fallback(from_currency, to_currency, date, cache_key)

    def _fetch_with_fallback(
        self,
        from_currency: str,
        to_currency: str,
        date: Optional[datetime],
        cache_key: str,
    ) -> Tuple[Optional[float], bool]:
        attempt_date = date
        fallback_used = False
        for attempt in range(self._max_fallback_days + 1):
            try:
                rate = self._converter.get_rate(from_currency, to_currency, attempt_date)
                if self._cache_rates:
                    self._rate_cache[cache_key] = rate
                if fallback_used:
                    self._fallback_count += 1
                return float(rate), fallback_used
            except RatesNotAvailableError:
                if self._use_fallback and attempt_date and attempt < self._max_fallback_days:
                    attempt_date = find_most_recent_business_day(
                        attempt_date - timedelta(days=1), max_lookback_days=1
                    )
                    fallback_used = True
                else:
                    return None, fallback_used
            except Exception as e:
                logger.error(f"Error fetching rate for {from_currency}: {e}")
                return None, fallback_used
        return None, fallback_used

    @property
    def fallback_count(self) -> int:
        return self._fallback_count


class CurrencyConverter:
    """Currency converter for equities data using RateFetchingService."""

    def __init__(
        self,
        target_currency: str = "USD",
        cache_rates: bool = True,
        max_fallback_days: int = MAX_FALLBACK_DAYS,
        use_business_day_fallback: bool = True,
        force_decimal: bool = False,
    ):
        self.target_currency = target_currency.upper()
        self.cache_rates = cache_rates
        self.max_fallback_days = max_fallback_days
        self.use_business_day_fallback = use_business_day_fallback
        self._converter = CurrencyRates(force_decimal=force_decimal)
        self._metrics = CurrencyConversionMetrics()
        self._rate_service = RateFetchingService(
            converter=self._converter,
            cache_rates=cache_rates,
            max_fallback_days=max_fallback_days,
            use_business_day_fallback=use_business_day_fallback,
        )

    def _get_exchange_rate(
        self, from_currency: str, date: Optional[datetime] = None
    ) -> Tuple[Optional[float], bool]:
        """Delegate to rate service."""
        return self._rate_service.get_rate(from_currency, self.target_currency, date)

    def get_metrics(self) -> CurrencyConversionMetrics:
        return self._metrics

    def convert_dataframe(
        self,
        df: pd.DataFrame,
        currency_column: str = "unit",
        date_column: str = "reference_date",
        columns_to_convert: Optional[List[str]] = None,
        suffix: str = "_usd",
        inplace: bool = False,
        alternative_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if not inplace:
            df = df.copy()

        self._metrics = CurrencyConversionMetrics()
        self._metrics.rows_processed = len(df)

        if currency_column not in df.columns:
            raise ValueError(f"Currency column '{currency_column}' not found in DataFrame")

        if columns_to_convert is None:
            columns_to_convert = MONETARY_COLUMNS

        cols_to_convert = [c for c in columns_to_convert if c in df.columns]
        self._metrics.columns_converted = len(cols_to_convert)

        if not cols_to_convert:
            logger.warning("No monetary columns found to convert")
            return df

        unique_currencies = df[currency_column].dropna().unique()
        self._metrics.unique_currencies = set(str(c) for c in unique_currencies)

        conversion_rates: Dict[Any, Optional[float]] = {}
        for idx, row in df.iterrows():
            currency = row.get(currency_column)
            if pd.isna(currency):
                continue

            if alternative_date:
                rate_date = alternative_date
            elif date_column in df.columns and pd.notna(row.get(date_column)):
                rate_date = pd.to_datetime(row[date_column])
                if hasattr(rate_date, "to_pydatetime"):
                    rate_date = rate_date.to_pydatetime()
            else:
                rate_date = None

            rate, fallback_used = self._get_exchange_rate(str(currency), rate_date)
            conversion_rates[idx] = rate
            if fallback_used:
                self._metrics.fallback_used_count += 1
            if rate is None and str(currency) not in NO_CONVERSION_CURRENCIES:
                self._metrics.conversion_errors.append(
                    f"Rate not found for {currency} on {rate_date}"
                )

        for col in cols_to_convert:
            new_col_name = f"{col}{suffix}"
            df[new_col_name] = [
                (
                    val * conversion_rates.get(idx)
                    if pd.notna(val) and conversion_rates.get(idx) is not None
                    else np.nan
                )
                for idx, val in zip(df.index, df[col])
            ]

        self._metrics.rows_converted = sum(1 for r in conversion_rates.values() if r is not None)
        self._metrics.rows_failed = self._metrics.rows_processed - self._metrics.rows_converted
        return df


def get_alternative_reference_date(
    df: pd.DataFrame, date_column: str = "reference_date", fallback_days: int = 1
) -> Optional[datetime]:
    if date_column not in df.columns:
        return None
    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return None
    most_common_date = dates.mode().iloc[0] if not dates.mode().empty else dates.max()
    alternative = most_common_date - timedelta(days=fallback_days)
    if hasattr(alternative, "to_pydatetime"):
        alternative = alternative.to_pydatetime()
    return find_most_recent_business_day(alternative)


def convert_to_usd(
    df: pd.DataFrame,
    currency_column: str = "unit",
    date_column: str = "reference_date",
    columns: Optional[List[str]] = None,
    use_fallback: bool = True,
    alternative_date: Optional[datetime] = None,
) -> pd.DataFrame:
    converter = CurrencyConverter(target_currency="USD", use_business_day_fallback=use_fallback)
    return converter.convert_dataframe(
        df,
        currency_column=currency_column,
        date_column=date_column,
        columns_to_convert=columns,
        alternative_date=alternative_date,
    )

def convert_with_fallback_date(
    df: pd.DataFrame,
    fallback_days: int = 1,
    currency_column: str = "unit",
    date_column: str = "reference_date",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert monetary columns using an alternative reference date.

    This is a convenience wrapper that implements Option 3: using a fallback
    date derived from the DataFrame when the primary reference date has no
    exchange rate data available.

    Args:
        df: DataFrame with equities data
        fallback_days: Days to subtract from the most common reference date
        currency_column: Column containing currency codes
        date_column: Column for reference dates
        columns: Specific columns to convert

    Returns:
        DataFrame with USD-converted columns
    """
    alt_date = get_alternative_reference_date(df, date_column, fallback_days)

    if alt_date:
        logger.info(f"Using alternative reference date: {alt_date.strftime('%Y-%m-%d')}")

    return convert_to_usd(
        df,
        currency_column=currency_column,
        date_column=date_column,
        columns=columns,
        alternative_date=alt_date,
    )
