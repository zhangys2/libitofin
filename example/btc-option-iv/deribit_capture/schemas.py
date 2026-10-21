"""Arrow schemas for the Deribit tick capture.

Everything is stored as Deribit sent it: option prices in coin (BTC), IVs in
percent, sizes as quoted. A price of ``0.0`` with a size of ``0.0`` means "no
quote on that side" and is kept as is so no information is lost.
"""

from __future__ import annotations

import pyarrow as pa

_INSTRUMENT = pa.dictionary(pa.int16(), pa.string())
_SIDE = pa.dictionary(pa.int8(), pa.string())

QUOTES = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("instrument", _INSTRUMENT),
        ("bid_px", pa.float64()),
        ("bid_amt", pa.float64()),
        ("ask_px", pa.float64()),
        ("ask_amt", pa.float64()),
        ("bid_iv", pa.float64()),
        ("ask_iv", pa.float64()),
        ("mark_iv", pa.float64()),
        ("mark_px", pa.float64()),
        ("index_px", pa.float64()),
        ("underlying_px", pa.float64()),
    ]
)

OPTION_STATE = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("instrument", _INSTRUMENT),
        ("mark_px", pa.float64()),
        ("mark_iv", pa.float64()),
        ("bid_iv", pa.float64()),
        ("ask_iv", pa.float64()),
        ("delta", pa.float64()),
        ("gamma", pa.float64()),
        ("vega", pa.float64()),
        ("theta", pa.float64()),
        ("rho", pa.float64()),
        ("open_interest", pa.float64()),
        ("last_px", pa.float64()),
        ("volume", pa.float64()),
        ("volume_usd", pa.float64()),
        ("index_px", pa.float64()),
        ("underlying_px", pa.float64()),
        ("underlying_index", pa.string()),
        ("state", _SIDE),
    ]
)

TRADES = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("instrument", _INSTRUMENT),
        ("trade_id", pa.string()),
        ("trade_seq", pa.int64()),
        ("price", pa.float64()),
        ("amount", pa.float64()),
        ("direction", _SIDE),
        ("index_px", pa.float64()),
        ("mark_px", pa.float64()),
        ("iv", pa.float64()),
        ("tick_direction", pa.int8()),
        ("liquidation", pa.string()),
        ("block_trade_id", pa.string()),
        ("extra_json", pa.string()),
    ]
)

TICKER_OTHER = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("instrument", _INSTRUMENT),
        ("best_bid_px", pa.float64()),
        ("best_bid_amt", pa.float64()),
        ("best_ask_px", pa.float64()),
        ("best_ask_amt", pa.float64()),
        ("mark_px", pa.float64()),
        ("index_px", pa.float64()),
        ("last_px", pa.float64()),
        ("open_interest", pa.float64()),
        ("funding_8h", pa.float64()),
        ("current_funding", pa.float64()),
        ("state", _SIDE),
    ]
)

PERP_BOOK = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("change_id", pa.int64()),
        ("instrument", _INSTRUMENT),
        ("bid_px", pa.list_(pa.float64())),
        ("bid_amt", pa.list_(pa.float64())),
        ("ask_px", pa.list_(pa.float64())),
        ("ask_amt", pa.list_(pa.float64())),
    ]
)

OPTION_BOOK = PERP_BOOK

INDEX = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("recv_ns", pa.int64()),
        ("index_name", _SIDE),
        ("price", pa.float64()),
    ]
)

SESSIONS = pa.schema(
    [
        ("ts_ms", pa.int64()),
        ("event", _SIDE),
        ("detail", pa.string()),
    ]
)

SCHEMAS: dict[str, pa.Schema] = {
    "quotes": QUOTES,
    "option_state": OPTION_STATE,
    "trades": TRADES,
    "ticker_other": TICKER_OTHER,
    "perp_book": PERP_BOOK,
    "option_book": OPTION_BOOK,
    "index": INDEX,
    "sessions": SESSIONS,
}
