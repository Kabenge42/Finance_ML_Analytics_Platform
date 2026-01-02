"""Foreign Currency Conversion Module for Equities Data."""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Install: pip install forex-python
try:
    from forex_python.converter import CurrencyRates, RatesNotAvailableError

    FOREX_AVAILABLE = True
except ImportError:
    FOREX_AVAILABLE = False
    logger.warning("forex-python not installed. Run: pip install forex-python")


# =============================================================================
# Configuration Constants (aligned with code_guidelines.md Section 2)
# =============================================================================

# Maximum number of business days to search backwards for available rates
MAX_FALLBACK_DAYS: int = 7

# Default currencies that are always available (no conversion needed)
NO_CONVERSION_CURRENCIES: frozenset = frozenset({"USD"})

# Supported currency codes for validation
SUPPORTED_CURRENCIES: frozenset = frozenset(
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


# Monetary columns from your equities schema that should be converted
MONETARY_COLUMNS: List[str] = [
    # Price columns
    "last_price",
    "price_target",
    "price_target_ytd_ago",
    "price_target_low",
    "price_target_median",
    "price_target_high",
    "price_5d_ago",
    "price_1w_ago",
    "price_1m_ago",
    "price_3m_ago",
    "price_6m_ago",
    "price_1y_ago",
    "price_3y_ago",
    "price_5y_ago",
    "price_qtd_ago",
    "52w_high_adj",
    "52w_low_adj",
    "ema_20d",
    "ema_50d",
    "ema_100d",
    "ema_250d",
    # Market value columns
    "market_cap",
    "enterprise_value",
    "market_cap_country_r",
    # Dividend columns
    "dividend_record_amount",
    "dividend_per_share_ltm",
    "common_dividends_paid_ltm",
    "common_dividends_paid_fy",
    # Revenue & Income columns
    "total_revenues_fq",
    "total_revenues_1fy",
    "total_revenues_fy",
    "total_revenues_ltm",
    "total_revenues_5yavgfq",
    "total_revenues_5yavgltm",
    "revenues_est_avg_ntm",
    "revenues_est_avg_fy1e",
    "revenues_est_med_ntm",
    "revenues_est_med_fy1e",
    # EBITDA columns
    "ebitda_fq",
    "ebitda_ltm",
    "ebitda_fy",
    "ebitda_1fy",
    "ebitda_adj_ltm",
    "ebitda_adj_fy",
    "ebitda_adj_1fy",
    "ebitda_5yavgfq",
    "ebitda_5yavgltm",
    "ebitda_est_avg_ntm",
    "ebitda_est_avg_fy1e",
    # EBIT columns
    "ebit_fq",
    "ebit_ltm",
    "ebit_fy",
    "ebit_1fy",
    "ebit_adj_1fy",
    "ebit_adj_fy",
    "ebit_adj_ltm",
    "ebit_5yavgfq",
    "ebit_5yavgltm",
    "ebit_est_med_fy1e",
    "ebit_est_med_ntm",
    # Net Income columns
    "net_income_is_fy",
    "net_income_is_ltm",
    "net_income_is_fq",
    "net_income_is_1fy",
    "net_income_is_5yavgfq",
    "net_income_is_5yavgltm",
    "normalized_net_income_fy",
    "normalized_net_income_ltm",
    "normalized_net_income_fq",
    "normalized_net_income_1fy",
    "normalized_net_income_5yavgfq",
    "normalized_net_income_5yavgltm",
    "net_income_adj_fy",
    "net_income_adj_ltm",
    "net_income_adj_fq",
    "net_income_adj_1fy",
    "net_income_adj_5yavgfq",
    # Operating & Gross columns
    "operating_income_ltm",
    "operating_income_fy",
    "operating_income_fq",
    "operating_income_5yavgfq",
    "total_operating_expenses_ltm",
    "gross_profit_ltm",
    "gross_profit_fy",
    "cost_of_revenues_ltm",
    "randd_expenses_ltm",
    # SG&A and Marketing
    "selling_general_and_admin_expenses_total_fq",
    "selling_general_and_admin_expenses_total_fy",
    "selling_general_and_admin_expenses_total_1fy",
    "selling_general_and_admin_expenses_total_5yavgfq",
    "marketing_expenses_fq",
    "marketing_expenses_fy",
    "marketing_expenses_1fy",
    "marketing_expenses_5yavgltm",
    # Balance sheet columns
    "total_assets_ltm",
    "total_assets_fy",
    "total_equity_fy",
    "total_equity_ltm",
    "total_debt_fy",
    "total_debt_ltm",
    "total_current_assets_ltm",
    "total_current_liabilities_ltm",
    "working_capital_ltm",
    "working_capital_fq",
    "working_capital_fy",
    "working_capital_5yavgfy",
    "tbv_fy",
    "tbv_ltm",
    "cash_and_equivalents_ltm",
    "cash_and_equivalents_fq",
    "cash_and_equivalents_fy",
    "cash_and_equivalents_5yavgfq",
    "retained_earnings_ltm",
    "retained_earnings_fq",
    "retained_earnings_fy",
    "retained_earnings_5yavgfq",
    "inventory_ltm",
    "inventory_fq",
    "inventory_fy",
    "inventory_5yavgfq",
    "goodwill_fq",
    "goodwill_ltm",
    "goodwill_fy",
    "goodwill_1fy",
    "goodwill_5yavgfq",
    "gross_intangible_assets_ltm",
    "gross_intangible_assets_fy",
    "gross_intangible_assets_5yavgfq",
    "accounts_receivable_total_fy",
    "accounts_receivable_total_1fy",
    "accounts_receivable_total_5yavgfq",
    # Cash flow columns
    "cff_ltm",
    "cff_fy",
    "cff_fq",
    "cff_1fy",
    "cfi_ltm",
    "cfi_fy",
    "cfi_fq",
    "cfi_1fy",
    "fcf_ltm",
    "fcf_fy",
    "fcf_fq",
    "fcf_5yavgfq",
    "cfo_ltm",
    "cfo_fy",
    "cfo_fq",
    "cfo_1fy",
    "capital_expenditure_ltm",
    "capital_expenditure_1fy",
    "capital_expenditure_fy",
    "capital_expenditure_fq",
    "capital_expenditure_5yavgfq",
    "cash_acquisitions_ltm",
    "cash_acquisitions_fy",
    "cash_acquisitions_fq",
    "cash_acquisitions_1fy",
    "cash_acquisitions_5yavgfq",
    # Non-recurring items
    "restructuring_charges_ltm",
    "restructuring_charges_fq",
    "restructuring_charges_1fy",
    "restructuring_charges_fy",
    "restructuring_charges_5yavgfq",
    "merger_and_restructuring_charges_ltm",
    "merger_and_restructuring_charges_fq",
    "merger_and_restructuring_charges_fy",
    "merger_and_restructuring_charges_5yavgfq",
    "asset_writedown_ltm",
    "asset_writedown_fy",
    "asset_writedown_fq",
    "asset_writedown_1fy",
    "asset_writedown_5yavgfq",
    "impairment_of_goodwill_fq",
    "impairment_of_goodwill_ltm",
    "impairment_of_goodwill_1fy",
    "impairment_of_goodwill_fy",
    "impairment_of_goodwill_5yavgfq",
    "other_unusual_items_total_ltm",
    "gain_loss_on_sale_of_assets_ltm",
    "interest_expense_total_ltm",
    "interest_income_on_investments_ltm",
]


@dataclass
class CurrencyConversionMetrics:
    """
    Metrics tracking for currency conversion operations.

    Attributes:
        rows_processed: Total rows in DataFrame
        rows_converted: Rows successfully converted
        rows_failed: Rows where conversion failed (rate unavailable)
        fallback_used_count: Number of times fallback date was used
        unique_currencies: Set of unique source currencies encountered
        columns_converted: Number of monetary columns converted
    """

    rows_processed: int = 0
    rows_converted: int = 0
    rows_failed: int = 0
    fallback_used_count: int = 0
    unique_currencies: set = field(default_factory=set)
    columns_converted: int = 0

    def summary(self) -> str:
        """Return human-readable summary of conversion metrics."""
        success_rate = (
            (self.rows_converted / self.rows_processed * 100) if self.rows_processed > 0 else 0
        )
        return (
            f"Currency Conversion: {self.rows_converted}/{self.rows_processed} rows ({success_rate:.1f}%), "
            f"{self.columns_converted} columns, {len(self.unique_currencies)} currencies, "
            f"{self.fallback_used_count} fallbacks used"
        )


def find_most_recent_business_day(
    target_date: datetime, max_lookback_days: int = MAX_FALLBACK_DAYS
) -> datetime:
    """
    Find the most recent business day on or before the target date.

    Handles weekends and common banking holidays by iterating backwards.
    This is a heuristic approach - it does not have a complete holiday calendar.

    Args:
        target_date: The date to start searching from
        max_lookback_days: Maximum days to search backwards (default: 7)

    Returns:
        datetime: Most recent likely business day

    Example:
        >>> find_most_recent_business_day(datetime(2026, 1, 1))  # New Year's Day
        datetime(2025, 12, 31)  # Falls back to Dec 31
    """
    current = target_date

    for _ in range(max_lookback_days):
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            # Skip common banking holidays (simplified check)
            month_day = (current.month, current.day)
            common_holidays = {
                (1, 1),  # New Year's Day
                (12, 25),  # Christmas
                (12, 26),  # Boxing Day (UK/EU)
            }
            if month_day not in common_holidays:
                return current

        current = current - timedelta(days=1)

    # If we've exhausted lookback, return the last attempted date
    return current


def get_alternative_reference_date(
    df: pd.DataFrame, date_column: str = "reference_date", fallback_days: int = 1
) -> Optional[datetime]:
    """
    Get an alternative reference date from the DataFrame for currency conversion.

    This implements Option 3: using a different reference date when the primary
    date has no exchange rate data available.

    Args:
        df: DataFrame containing date information
        date_column: Column containing reference dates
        fallback_days: Number of days to subtract from the most common date

    Returns:
        Alternative datetime or None if no valid date found

    Example:
        >>> alt_date = get_alternative_reference_date(df, fallback_days=1)
        >>> # If reference_date is 2026-01-01, returns 2025-12-31
    """
    if date_column not in df.columns:
        return None

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return None

    # Use the most common date minus fallback_days
    most_common_date = dates.mode().iloc[0] if not dates.mode().empty else dates.max()
    alternative = most_common_date - timedelta(days=fallback_days)

    # Convert to Python datetime
    if hasattr(alternative, "to_pydatetime"):
        alternative = alternative.to_pydatetime()

    return find_most_recent_business_day(alternative)


class CurrencyConverter:
    """
    Currency converter for equities data using forex-python.

    Converts monetary columns from local currency (specified in 'unit' column)
    to a target currency (default: USD) using historical exchange rates.

    Features:
        - Automatic fallback to most recent business day for holidays/weekends
        - Configurable maximum fallback days
        - Rate caching for performance
        - Detailed conversion metrics

    Example:
        >>> converter = CurrencyConverter(target_currency="USD")
        >>> df_converted = converter.convert_dataframe(df)
        >>> print(converter.get_metrics().summary())
    """

    def __init__(
        self,
        target_currency: str = "USD",
        cache_rates: bool = True,
        max_fallback_days: int = MAX_FALLBACK_DAYS,
        use_business_day_fallback: bool = True,
    ):
        """
        Initialize the currency converter.

        Args:
            target_currency: Target currency code (ISO 4217), default USD
            cache_rates: Whether to cache exchange rates for performance
            max_fallback_days: Maximum days to search for available rates (default: 7)
            use_business_day_fallback: Enable automatic fallback for holidays/weekends
        """
        if not FOREX_AVAILABLE:
            raise ImportError("forex-python is required. Install with: pip install forex-python")

        self.target_currency = target_currency.upper()
        self.cache_rates = cache_rates
        self.max_fallback_days = max_fallback_days
        self.use_business_day_fallback = use_business_day_fallback
        self._rate_cache: Dict[str, float] = {}
        self._converter = CurrencyRates()
        self._metrics = CurrencyConversionMetrics()
        self._fallback_dates_used: Dict[str, datetime] = {}

    def _get_exchange_rate(
        self, from_currency: str, date: Optional[datetime] = None
    ) -> Tuple[Optional[float], bool]:
        """
        Get exchange rate from source currency to target currency.

        Implements automatic fallback to previous business days when rates
        are unavailable (e.g., holidays, weekends, future dates).

        Args:
            from_currency: Source currency code (e.g., 'EUR', 'GBP')
            date: Date for historical rate (None for latest)

        Returns:
            Tuple of (exchange_rate, fallback_used) where:
                - exchange_rate: Float rate or None if unavailable
                - fallback_used: True if a fallback date was used
        """
        from_currency = from_currency.upper().strip()

        # No conversion needed for target currency
        if from_currency == self.target_currency:
            return 1.0, False

        # Validate currency code
        if from_currency not in SUPPORTED_CURRENCIES:
            logger.warning(f"Unsupported currency code: {from_currency}")
            return None, False

        # Check cache first
        cache_key = f"{from_currency}_{date.strftime('%Y-%m-%d') if date else 'latest'}"
        if self.cache_rates and cache_key in self._rate_cache:
            return self._rate_cache[cache_key], False

        # Try to get rate with fallback strategy
        attempt_date = date
        fallback_used = False

        for attempt in range(self.max_fallback_days + 1):
            try:
                if attempt_date:
                    rate = self._converter.get_rate(
                        from_currency, self.target_currency, attempt_date
                    )
                else:
                    rate = self._converter.get_rate(from_currency, self.target_currency)

                # Cache the successful rate
                if self.cache_rates:
                    self._rate_cache[cache_key] = rate

                # Track fallback usage
                if fallback_used:
                    self._fallback_dates_used[cache_key] = attempt_date
                    logger.debug(
                        f"Used fallback date {attempt_date.strftime('%Y-%m-%d')} for {from_currency} "
                        f"(original: {date.strftime('%Y-%m-%d') if date else 'latest'})"
                    )

                return rate, fallback_used

            except RatesNotAvailableError:
                if (
                    self.use_business_day_fallback
                    and attempt_date
                    and attempt < self.max_fallback_days
                ):
                    # Try the previous business day
                    attempt_date = attempt_date - timedelta(days=1)
                    attempt_date = find_most_recent_business_day(attempt_date, max_lookback_days=1)
                    fallback_used = True
                    continue
                else:
                    # Log only on final failure to reduce noise
                    if attempt == self.max_fallback_days or not self.use_business_day_fallback:
                        logger.warning(
                            f"Exchange rate not available for {from_currency} on {date} "
                            f"(tried {attempt + 1} dates)"
                        )
                    return None, fallback_used

            except Exception as e:
                logger.error(f"Error fetching rate for {from_currency}: {e}")
                return None, False

        return None, fallback_used

    def get_metrics(self) -> CurrencyConversionMetrics:
        """Return conversion metrics from the last operation."""
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
        """
        Convert monetary columns in a DataFrame to target currency.

        Args:
            df: DataFrame with equities data
            currency_column: Column containing currency codes (default: 'unit')
            date_column: Column containing dates for historical rates (default: 'reference_date')
            columns_to_convert: List of columns to convert (default: MONETARY_COLUMNS)
            suffix: Suffix for converted columns (default: '_usd')
            inplace: Whether to modify DataFrame in place
            alternative_date: Override date for all conversions (Option 3 support)

        Returns:
            DataFrame with converted columns added

        Raises:
            ValueError: If currency_column not found in DataFrame
        """
        if not inplace:
            df = df.copy()

        # Reset metrics for this operation
        self._metrics = CurrencyConversionMetrics()
        self._metrics.rows_processed = len(df)

        # Validate currency column exists
        if currency_column not in df.columns:
            raise ValueError(f"Currency column '{currency_column}' not found in DataFrame")

        # Determine columns to convert
        if columns_to_convert is None:
            columns_to_convert = MONETARY_COLUMNS

        # Filter to columns that exist in DataFrame
        cols_to_convert = [c for c in columns_to_convert if c in df.columns]
        self._metrics.columns_converted = len(cols_to_convert)

        if not cols_to_convert:
            logger.warning("No monetary columns found to convert")
            return df

        logger.info(f"Converting {len(cols_to_convert)} monetary columns to {self.target_currency}")

        # Get unique currencies
        unique_currencies = df[currency_column].dropna().unique()
        self._metrics.unique_currencies = set(str(c) for c in unique_currencies)
        logger.info(
            f"Found {len(unique_currencies)} unique currencies: {list(unique_currencies)[:10]}..."
        )

        # Pre-fetch exchange rates for all currency-date combinations
        conversion_rates: Dict[Any, Optional[float]] = {}

        for idx, row in df.iterrows():
            currency = row.get(currency_column)
            if pd.isna(currency):
                continue

            # Determine the date for rate lookup
            if alternative_date:
                rate_date = alternative_date
            elif date_column in df.columns and pd.notna(row.get(date_column)):
                rate_date = pd.to_datetime(row[date_column])
                # forex-python needs datetime, not Timestamp
                if hasattr(rate_date, "to_pydatetime"):
                    rate_date = rate_date.to_pydatetime()
            else:
                rate_date = None

            rate, fallback_used = self._get_exchange_rate(str(currency), rate_date)
            conversion_rates[idx] = rate

            if fallback_used:
                self._metrics.fallback_used_count += 1

        # Apply conversion to each column
        for col in cols_to_convert:
            new_col_name = f"{col}{suffix}"

            def convert_value(row_idx: Any, value: Any) -> float:
                if pd.isna(value):
                    return np.nan
                rate = conversion_rates.get(row_idx)
                if rate is None:
                    return np.nan
                return value * rate

            df[new_col_name] = [convert_value(idx, val) for idx, val in zip(df.index, df[col])]

        # Calculate final metrics
        self._metrics.rows_converted = sum(1 for r in conversion_rates.values() if r is not None)
        self._metrics.rows_failed = self._metrics.rows_processed - self._metrics.rows_converted

        logger.info(self._metrics.summary())

        return df


def convert_to_usd(
    df: pd.DataFrame,
    currency_column: str = "unit",
    date_column: str = "reference_date",
    columns: Optional[List[str]] = None,
    use_fallback: bool = True,
    alternative_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Convenience function to convert monetary columns to USD.

    Args:
        df: DataFrame with equities data
        currency_column: Column containing currency codes
        date_column: Column for historical exchange rates
        columns: Specific columns to convert (default: all monetary columns)
        use_fallback: Enable business day fallback for holidays/weekends
        alternative_date: Override date for all conversions

    Returns:
        DataFrame with USD-converted columns added (suffix: _usd)

    Example:
        >>> # Basic usage with automatic fallback
        >>> df_converted = convert_to_usd(all_stocks_features)

        >>> # With explicit alternative date (Option 3)
        >>> from datetime import datetime
        >>> df_converted = convert_to_usd(
        ...     all_stocks_features,
        ...     alternative_date=datetime(2025, 12, 31)
        ... )

        >>> # Check what columns were added
        >>> usd_cols = [c for c in df_converted.columns if c.endswith('_usd')]
        >>> print(f"Added {len(usd_cols)} USD columns")
    """
    converter = CurrencyConverter(target_currency="USD", use_business_day_fallback=use_fallback)

    result = converter.convert_dataframe(
        df,
        currency_column=currency_column,
        date_column=date_column,
        columns_to_convert=columns,
        alternative_date=alternative_date,
    )

    return result


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

    Example:
        >>> # If reference_date is 2026-01-01 (holiday), use 2025-12-31
        >>> df_converted = convert_with_fallback_date(df, fallback_days=1)
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
