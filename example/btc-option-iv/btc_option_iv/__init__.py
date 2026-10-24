"""Deribit BTC option implied-volatility example.

The WebSocket client is not part of ``itofin``. Import the offline quote
normalization from this package; run ``python -m btc_option_iv`` for the live stream.
"""

from btc_option_iv.market import (
    evaluate_quote,
    normalize_deribit_quote,
    parse_option_instruments,
    parse_subscription_message,
    solve_quote_iv,
    year_fraction_actual_365_fixed,
)

__all__ = [
    "evaluate_quote",
    "normalize_deribit_quote",
    "parse_option_instruments",
    "parse_subscription_message",
    "solve_quote_iv",
    "year_fraction_actual_365_fixed",
]
