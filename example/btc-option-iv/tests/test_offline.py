"""Offline fixture tests for Deribit parsing, year fraction, and quote rejection."""

# standard library
import json
import math
from pathlib import Path

# pypi/conda library
import pytest

# itofin library
from itofin.pricingengines import black_formula_implied_volatility
from itofin.instruments import OptionType

from btc_option_iv.market import (
    FORWARD_SOURCE_MARK,
    FORWARD_SOURCE_MID,
    MODEL,
    PRICE_UNIT,
    TIME_BASIS,
    CROSSED,
    EXPIRED,
    MISSING_FORWARD,
    MISSING_SIDE,
    NOT_INVERSE,
    OUTSIDE_NO_ARBITRAGE,
    STALE,
    ForwardData,
    FutureQuote,
    Instrument,
    MarketBook,
    OptionQuote,
    Rejection,
    evaluate_quote,
    format_iv_row,
    forward_from_future_quote,
    future_name_for,
    normalize_deribit_quote,
    options_with_listed_futures,
    parse_future_instruments,
    parse_option_instruments,
    parse_subscription_message,
    subscription_channels,
    year_fraction_actual_365_fixed,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_NOW_MS = 1_700_000_000_000
_EXPIRY_MS = 1_735_286_400_000
_INDEX = 100_000.0
_FORWARD = 101_250.0


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _deribit_coin_premium(index, strike, forward, expiry, vol, option_type: str) -> float:
    """Deribit's published inverse Black-Scholes premium, in BTC."""
    rate = math.log(forward / index) / expiry
    sqrt_t = math.sqrt(expiry)
    d1 = (math.log(index / strike) + (rate + 0.5 * vol * vol) * expiry) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-rate * expiry)
    if option_type == "call":
        usd = index * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    else:
        usd = strike * discount * _norm_cdf(-d2) - index * _norm_cdf(-d1)
    return usd / index


def _instrument(**overrides) -> Instrument:
    fields = {
        "instrument_name": "BTC-27DEC24-100000-C",
        "option_type": "call",
        "strike": 100_000.0,
        "expiration_timestamp_ms": _EXPIRY_MS,
        "future_name": "BTC-27DEC24",
    }
    fields.update(overrides)
    return Instrument(**fields)


def _quote(**overrides) -> OptionQuote:
    fields = {
        "instrument_name": "BTC-27DEC24-100000-C",
        "timestamp_ms": _NOW_MS,
        "bid_price": 0.04,
        "ask_price": 0.05,
        "bid_amount": 1.5,
        "ask_amount": 2.0,
    }
    fields.update(overrides)
    return OptionQuote(**fields)


def _forward(**overrides) -> ForwardData:
    fields = {
        "forward": _FORWARD,
        "forward_timestamp_ms": _NOW_MS,
        "index": _INDEX,
        "index_timestamp_ms": _NOW_MS,
        "source": FORWARD_SOURCE_MARK,
    }
    fields.update(overrides)
    return ForwardData(**fields)


def test_parse_instruments_keeps_live_inverse_options_only():
    instruments, rejections = parse_option_instruments(_load("instruments.json"), now_ms=_NOW_MS)
    assert [item.instrument_name for item in instruments] == [
        "BTC-27DEC24-100000-C",
        "BTC-27DEC24-100000-P",
    ]
    assert instruments[0].option_type == "call"
    assert instruments[0].strike == 100_000.0
    assert instruments[0].future_name == "BTC-27DEC24"
    assert instruments[1].option_type == "put"
    reasons = {item.reason for item in rejections}
    assert reasons == {EXPIRED, NOT_INVERSE}


def test_parse_futures_skips_perpetual():
    futures = parse_future_instruments(_load("futures.json"), now_ms=_NOW_MS)
    assert [item.instrument_name for item in futures] == ["BTC-27DEC24"]


def test_options_without_a_listed_future_are_rejected():
    options, _rejections = parse_option_instruments(_load("instruments.json"), now_ms=_NOW_MS)
    kept, missing = options_with_listed_futures(options, [])
    assert kept == []
    assert {item.reason for item in missing} == {MISSING_FORWARD}


def test_parse_option_ticker_ignores_exchange_mark_iv_and_underlying_price():
    event = parse_subscription_message(_load("option_ticker.json"))
    assert isinstance(event, OptionQuote)
    assert event.bid_price == 0.04
    assert event.ask_price == 0.05
    assert event.bid_amount == 1.5
    assert not hasattr(event, "mark_iv")
    assert future_name_for(event.instrument_name) == "BTC-27DEC24"


def test_parse_future_and_index_channels():
    future = parse_subscription_message(_load("future_ticker.json"))
    index = parse_subscription_message(_load("index.json"))
    assert isinstance(future, FutureQuote)
    assert future.mark_price == 101_250.0
    assert index is not None
    assert index.price == 100_000.0


def test_subscription_channels_cover_index_future_and_option():
    instrument = _instrument()
    channels = subscription_channels([instrument])
    assert channels[0] == "deribit_price_index.btc_usd"
    assert "ticker.BTC-27DEC24.100ms" in channels
    assert "ticker.BTC-27DEC24-100000-C.100ms" in channels
    assert all("PERPETUAL" not in channel for channel in channels)


def test_year_fraction_uses_the_intraday_timestamp():
    quote = _NOW_MS
    one_day_17_hours = 147_600_000
    assert year_fraction_actual_365_fixed(quote, quote + one_day_17_hours) == pytest.approx(
        (1 + 17 / 24) / 365
    )
    assert year_fraction_actual_365_fixed(quote, quote + 12 * 60 * 60 * 1000) == pytest.approx(
        0.5 / 365
    )


def test_normalize_converts_coin_premium_with_explicit_forward_and_index():
    book = MarketBook()
    book.note_index(parse_subscription_message(_load("index.json")))
    book.note_future(parse_subscription_message(_load("future_ticker.json")))
    event = parse_subscription_message(_load("option_ticker.json"))
    instrument = _instrument()
    forward_data = book.forward_data(instrument)
    normalized = normalize_deribit_quote(event, instrument, forward_data, as_of_ms=_NOW_MS)
    assert not isinstance(normalized, Rejection)
    assert normalized.forward == _FORWARD
    assert normalized.index == _INDEX
    assert normalized.bid_black_price == pytest.approx(0.04 * _INDEX)
    assert normalized.ask_black_price == pytest.approx(0.05 * _INDEX)
    assert normalized.mid_black_price == pytest.approx(0.045 * _INDEX)
    assert normalized.discount == pytest.approx(_INDEX / _FORWARD)
    assert normalized.forward_source == FORWARD_SOURCE_MARK
    assert normalized.time_basis == TIME_BASIS


def test_future_mid_is_used_only_when_the_mark_is_missing():
    quote = FutureQuote("BTC-27DEC24", _NOW_MS, None, 101_000.0, 101_200.0)
    price, source = forward_from_future_quote(quote)
    assert price == pytest.approx(101_100.0)
    assert source == FORWARD_SOURCE_MID


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"timestamp_ms": _NOW_MS - 6_000}, STALE),
        ({"bid_price": 0.0}, MISSING_SIDE),
        ({"bid_amount": 0.0}, MISSING_SIDE),
        ({"ask_price": None}, MISSING_SIDE),
        ({"bid_price": 0.06, "ask_price": 0.04}, CROSSED),
    ],
)
def test_invalid_quotes_are_rejected(overrides, reason):
    outcome = normalize_deribit_quote(
        _quote(**overrides),
        _instrument(),
        _forward(),
        as_of_ms=_NOW_MS,
        stale_after_ms=5_000,
    )
    assert isinstance(outcome, Rejection)
    assert outcome.reason == reason


def test_expired_and_missing_forward_are_rejected():
    expired = normalize_deribit_quote(
        _quote(timestamp_ms=_EXPIRY_MS),
        _instrument(),
        _forward(forward_timestamp_ms=_EXPIRY_MS, index_timestamp_ms=_EXPIRY_MS),
        as_of_ms=_EXPIRY_MS,
    )
    assert isinstance(expired, Rejection)
    assert expired.reason == EXPIRED
    missing = normalize_deribit_quote(
        _quote(),
        _instrument(),
        _forward(forward=None, forward_timestamp_ms=None),
        as_of_ms=_NOW_MS,
    )
    assert isinstance(missing, Rejection)
    assert missing.reason == MISSING_FORWARD


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_normalized_premium_recovers_known_volatility(option_type):
    expiry = year_fraction_actual_365_fixed(_NOW_MS, _EXPIRY_MS)
    vol = 0.55
    strike = 100_000.0
    premium = _deribit_coin_premium(_INDEX, strike, _FORWARD, expiry, vol, option_type)
    row = evaluate_quote(
        _quote(
            instrument_name=f"BTC-27DEC24-100000-{'C' if option_type == 'call' else 'P'}",
            bid_price=premium,
            ask_price=premium,
        ),
        _instrument(
            instrument_name=f"BTC-27DEC24-100000-{'C' if option_type == 'call' else 'P'}",
            option_type=option_type,
        ),
        _forward(),
        as_of_ms=_NOW_MS,
    )
    assert not isinstance(row, Rejection)
    assert row.bid_iv == pytest.approx(vol, abs=1e-8)
    assert row.ask_iv == pytest.approx(vol, abs=1e-8)
    assert row.mid_iv == pytest.approx(vol, abs=1e-8)
    assert row.mid_iv != pytest.approx(0.99, abs=1e-2)
    text = format_iv_row(row)
    assert MODEL in text
    assert PRICE_UNIT in text
    assert FORWARD_SOURCE_MARK in text
    assert TIME_BASIS in text


def test_mid_iv_inverts_the_mid_premium():
    expiry = year_fraction_actual_365_fixed(_NOW_MS, _EXPIRY_MS)
    row = evaluate_quote(
        _quote(bid_price=0.02, ask_price=0.08),
        _instrument(),
        _forward(),
        as_of_ms=_NOW_MS,
    )
    assert not isinstance(row, Rejection)
    mid_price = 0.05 * _INDEX
    expected_mid = black_formula_implied_volatility(
        OptionType.Call,
        strike=100_000.0,
        forward=_FORWARD,
        expiry=expiry,
        black_price=mid_price,
        discount=_INDEX / _FORWARD,
    )
    assert row.mid_iv == pytest.approx(expected_mid, abs=1e-12)
    assert row.mid_iv != pytest.approx(0.5 * (row.bid_iv + row.ask_iv), abs=1e-4)


def test_arbitrage_inconsistent_price_is_rejected_without_an_iv():
    outcome = evaluate_quote(
        _quote(bid_price=0.001, ask_price=0.001),
        _instrument(strike=80_000.0),
        _forward(),
        as_of_ms=_NOW_MS,
    )
    assert isinstance(outcome, Rejection)
    assert outcome.reason == OUTSIDE_NO_ARBITRAGE
