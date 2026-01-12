"""
Internal Currency Converter module.
Ported from forex-python for internal use.
"""

from decimal import Decimal
from typing import Dict, Any, Union

import requests
import simplejson as json


class RatesNotAvailableError(Exception):
    """Custom exception when rates are not available."""

    pass


class DecimalFloatMismatchError(Exception):
    """A float has been supplied when force_decimal was set to True."""

    pass


class Common:
    def __init__(self, force_decimal: bool = False):
        self._force_decimal = force_decimal

    def _source_url(self) -> str:
        return "https://theratesapi.com/api/"

    def _get_date_string(self, date_obj) -> str:
        if date_obj is None:
            return "latest"
        if hasattr(date_obj, "strftime"):
            return date_obj.strftime("%Y-%m-%d")
        return str(date_obj)

    def _decode_rates(
        self, response, use_decimal: bool = False, date_str: str = None
    ) -> Dict[str, Any]:
        if self._force_decimal or use_decimal:
            decoded_data = json.loads(response.text, use_decimal=True)
        else:
            decoded_data = response.json()
        return decoded_data.get("rates", {})

    def _get_decoded_rate(
        self, response, dest_cur: str, use_decimal: bool = False, date_str: str = None
    ) -> Any:
        return self._decode_rates(response, use_decimal=use_decimal, date_str=date_str).get(
            dest_cur, None
        )


class CurrencyRates(Common):
    def get_rates(self, base_cur: str, date_obj=None) -> Dict[str, Any]:
        date_str = self._get_date_string(date_obj)
        payload = {"base": base_cur, "rtype": "fpy"}
        source_url = self._source_url() + date_str
        response = requests.get(source_url, params=payload)
        if response.status_code == 200:
            rates = self._decode_rates(response, date_str=date_str)
            return rates
        raise RatesNotAvailableError("Currency Rates Source Not Ready")

    def get_rate(self, base_cur: str, dest_cur: str, date_obj=None) -> Union[float, Decimal]:
        if base_cur == dest_cur:
            if self._force_decimal:
                return Decimal(1)
            return 1.0
        date_str = self._get_date_string(date_obj)
        payload = {"base": base_cur, "symbols": dest_cur, "rtype": "fpy"}
        source_url = self._source_url() + date_str
        response = requests.get(source_url, params=payload)
        if response.status_code == 200:
            rate = self._get_decoded_rate(response, dest_cur, date_str=date_str)
            if not rate:
                raise RatesNotAvailableError(
                    f"Currency Rate {base_cur} => {dest_cur} not available for Date {date_str}"
                )
            return rate
        raise RatesNotAvailableError("Currency Rates Source Not Ready")

    def convert(
        self, base_cur: str, dest_cur: str, amount: Union[float, Decimal], date_obj=None
    ) -> Union[float, Decimal]:
        if isinstance(amount, Decimal):
            use_decimal = True
        else:
            use_decimal = self._force_decimal

        if base_cur == dest_cur:
            if use_decimal:
                return Decimal(amount)
            return float(amount)

        date_str = self._get_date_string(date_obj)
        payload = {"base": base_cur, "symbols": dest_cur, "rtype": "fpy"}
        source_url = self._source_url() + date_str
        response = requests.get(source_url, params=payload)
        if response.status_code == 200:
            rate = self._get_decoded_rate(
                response, dest_cur, use_decimal=use_decimal, date_str=date_str
            )
            if not rate:
                raise RatesNotAvailableError(
                    f"Currency {base_cur} => {dest_cur} rate not available for Date {date_str}."
                )
            if isinstance(rate, str):
                rate = Decimal(rate) if use_decimal else float(rate)
            try:
                converted_amount = rate * amount
                return converted_amount
            except TypeError:
                raise DecimalFloatMismatchError(
                    "convert requires amount parameter is of type Decimal when force_decimal=True"
                )
        raise RatesNotAvailableError("Currency Rates Source Not Ready")


_CURRENCY_FORMATTER = CurrencyRates()
get_rates = _CURRENCY_FORMATTER.get_rates
get_rate = _CURRENCY_FORMATTER.get_rate
convert = _CURRENCY_FORMATTER.convert
