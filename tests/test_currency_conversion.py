import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from finance_ml.etl.currency import (
    CurrencyConverter,
    convert_to_usd,
    find_most_recent_business_day,
    get_alternative_reference_date,
    convert_with_fallback_date,
    RatesNotAvailableError,
)


class TestCurrencyConversion(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "unit": ["USD", "EUR", "GBP", "USD"],
                "reference_date": [
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 1),
                    datetime(2023, 1, 2),
                ],
                "last_price": [100.0, 100.0, 100.0, 150.0],
                "market_cap": [1000.0, 2000.0, 3000.0, 4000.0],
            }
        )

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_currency_converter_usd_to_usd(self, mock_rates):
        # No conversion needed for USD to USD
        converter = CurrencyConverter(target_currency="USD")
        rate, fallback = converter._get_exchange_rate("USD", datetime(2023, 1, 1))
        self.assertEqual(rate, 1.0)
        self.assertFalse(fallback)
        mock_rates.return_value.get_rate.assert_not_called()

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_currency_converter_eur_to_usd(self, mock_rates):
        # Mocking EUR to USD rate
        mock_instance = mock_rates.return_value
        mock_instance.get_rate.return_value = 1.1

        converter = CurrencyConverter(target_currency="USD")
        rate, fallback = converter._get_exchange_rate("EUR", datetime(2023, 1, 1))

        self.assertEqual(rate, 1.1)
        self.assertFalse(fallback)
        mock_instance.get_rate.assert_called_with("EUR", "USD", datetime(2023, 1, 1))

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_convert_dataframe(self, mock_rates):
        mock_instance = mock_rates.return_value

        # Mock rates: EUR->USD=1.1, GBP->USD=1.2
        def side_effect(from_curr, to_curr, date=None):
            if from_curr == "EUR":
                return 1.1
            if from_curr == "GBP":
                return 1.2
            return 1.0

        mock_instance.get_rate.side_effect = side_effect

        converter = CurrencyConverter(target_currency="USD")
        result_df = converter.convert_dataframe(
            self.df,
            currency_column="unit",
            date_column="reference_date",
            columns_to_convert=["last_price", "market_cap"],
        )

        # USD rows
        self.assertEqual(result_df.loc[0, "last_price_usd"], 100.0)
        self.assertEqual(result_df.loc[3, "last_price_usd"], 150.0)

        # EUR row: 100 * 1.1 = 110.0
        self.assertAlmostEqual(result_df.loc[1, "last_price_usd"], 110.0)
        # GBP row: 100 * 1.2 = 120.0
        self.assertAlmostEqual(result_df.loc[2, "last_price_usd"], 120.0)

        # Check metrics
        metrics = converter.get_metrics()
        self.assertEqual(metrics.rows_processed, 4)
        self.assertEqual(metrics.rows_converted, 4)
        self.assertEqual(metrics.columns_converted, 2)

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_business_day_fallback(self, mock_rates):
        mock_instance = mock_rates.return_value

        # Saturday, 2026-01-03
        holiday_date = datetime(2026, 1, 3)
        # Should fallback to Friday, 2026-01-02
        fallback_date = datetime(2026, 1, 2)

        def side_effect(from_curr, to_curr, date=None):
            if date == holiday_date:
                raise RatesNotAvailableError("Weekend")
            if date == fallback_date:
                return 1.1
            return 1.0

        mock_instance.get_rate.side_effect = side_effect

        converter = CurrencyConverter(use_business_day_fallback=True)
        rate, fallback_used = converter._get_exchange_rate("EUR", holiday_date)

        self.assertEqual(rate, 1.1)
        self.assertTrue(fallback_used)
        self.assertEqual(mock_instance.get_rate.call_count, 2)

    def test_find_most_recent_business_day(self):
        # Jan 1st 2026 is Thursday (but it's a holiday in our heuristic)
        # Heuristic skips Jan 1, so it should go to Dec 31
        jan1 = datetime(2026, 1, 1)
        bus_day = find_most_recent_business_day(jan1)
        self.assertEqual(bus_day, datetime(2025, 12, 31))

        # Sunday Jan 4 2026 -> should go to Friday Jan 2
        jan4 = datetime(2026, 1, 4)
        bus_day = find_most_recent_business_day(jan4)
        self.assertEqual(bus_day, datetime(2026, 1, 2))

    def test_get_alternative_reference_date(self):
        df = pd.DataFrame({"reference_date": [datetime(2026, 1, 2), datetime(2026, 1, 2)]})
        # Common date is Jan 2. Fallback 1 day is Jan 1 (Holiday).
        # find_most_recent_business_day(Jan 1) -> Dec 31
        alt_date = get_alternative_reference_date(df, fallback_days=1)
        self.assertEqual(alt_date, datetime(2025, 12, 31))

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_convert_with_fallback_date(self, mock_rates):
        mock_instance = mock_rates.return_value
        mock_instance.get_rate.return_value = 1.2

        df = pd.DataFrame(
            {"unit": ["EUR"], "reference_date": [datetime(2026, 1, 2)], "last_price": [100.0]}
        )

        # Uses alternative date Jan 1 -> Dec 31
        result_df = convert_with_fallback_date(df, fallback_days=1, columns=["last_price"])
        self.assertEqual(result_df.loc[0, "last_price_usd"], 120.0)
        # Verify get_rate was called with Dec 31 2025
        mock_instance.get_rate.assert_called_with("EUR", "USD", datetime(2025, 12, 31))

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_currency_not_found(self, mock_rates):
        mock_instance = mock_rates.return_value
        mock_instance.get_rate.side_effect = RatesNotAvailableError("Not found")

        converter = CurrencyConverter(use_business_day_fallback=False)
        rate, fallback = converter._get_exchange_rate("EUR", datetime(2023, 1, 1))
        self.assertIsNone(rate)
        self.assertFalse(fallback)

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_unsupported_currency(self, mock_rates):
        converter = CurrencyConverter()
        rate, fallback = converter._get_exchange_rate("XYZ", datetime(2023, 1, 1))
        self.assertIsNone(rate)
        self.assertFalse(fallback)
        mock_rates.return_value.get_rate.assert_not_called()

    @patch("finance_ml.etl.currency.CurrencyRates")
    def test_metrics_tracking(self, mock_rates):
        mock_instance = mock_rates.return_value

        # One success, one failure
        def side_effect(from_curr, to_curr, date=None):
            if from_curr == "EUR":
                return 1.1
            raise RatesNotAvailableError("FAIL")

        mock_instance.get_rate.side_effect = side_effect

        df = pd.DataFrame({"unit": ["EUR", "GBP"], "last_price": [100.0, 100.0]})

        converter = CurrencyConverter(use_business_day_fallback=False)
        converter.convert_dataframe(
            df, columns_to_convert=["last_price"], date_column="non_existent"
        )

        metrics = converter.get_metrics()
        self.assertEqual(metrics.rows_processed, 2)
        self.assertEqual(metrics.rows_converted, 1)
        self.assertEqual(metrics.rows_failed, 1)
        self.assertIn("EUR", metrics.unique_currencies)
        self.assertIn("GBP", metrics.unique_currencies)


if __name__ == "__main__":
    unittest.main()
