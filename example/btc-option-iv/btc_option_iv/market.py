"""Normalize Deribit inverse BTC option quotes and invert them with itofin.

Deribit quotes inverse options in BTC. The published Black-Scholes formula
(Deribit support article "Inverse Options") uses the index ``X``, the expiry
forward ``F``, and ``R = ln(F/X)/T``:

    USD = X N(d1) - K exp(-R T) N(d2)

That premium is Black-76 in USD present value:

    black_price = coin_premium * index_price
    discount = exp(-interest_rate * expiry)
    forward = option ticker underlying_price

When the ticker omits its model inputs, the separate index and dated-future
feeds are used as a fallback, with ``discount = index / forward``. Year fractions
are Actual/365 Fixed measured from the quote
timestamp to the exchange expiry timestamp, including the intraday fraction.
The exchange ``mark_iv`` is never an input to the solver.
"""

from __future__ import annotations

# standard library
import math
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

# itofin library
from itofin import ItofinError
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_volatility

MODEL = "Black-76"
PRICE_UNIT = "USD present value"
FORWARD_SOURCE_MARK = "Deribit dated future mark_price"
FORWARD_SOURCE_MID = "Deribit dated future mid"
TIME_BASIS = "Actual/365 Fixed"
STALE_AFTER_MS = 5_000
_MS_PER_YEAR = 365 * 24 * 60 * 60 * 1000
_MIN_TIMESTAMP_MS = 1_000_000_000_000
_OPTION_NAME = re.compile(r"^BTC-(\d{1,2}[A-Z]{3}\d{2})-(\d+(?:\.\d+)?)-([CP])$")
_FUTURE_NAME = re.compile(r"^BTC-\d{1,2}[A-Z]{3}\d{2}$")
_INDEX_CHANNEL = "deribit_price_index.btc_usd"

EXPIRED = "expired"
STALE = "stale"
MISSING_INDEX = "missing_index"
MISSING_FORWARD = "missing_forward"
INVALID_INDEX = "invalid_index"
INVALID_FORWARD = "invalid_forward"
MISSING_SIDE = "missing_side"
CROSSED = "crossed"
OUTSIDE_NO_ARBITRAGE = "outside_no_arbitrage"
SOLVER_FAILED = "solver_failed"
INVALID_INSTRUMENT = "invalid_instrument"
NOT_INVERSE = "not_inverse"
NOT_OPTION = "not_option"
INACTIVE = "inactive"


@dataclass(frozen=True)
class Rejection:
    """A quote or instrument dropped before an IV is emitted."""

    instrument_name: str
    reason: str
    detail: str
    timestamp_ms: int | None


@dataclass(frozen=True)
class Instrument:
    """One active inverse BTC option."""

    instrument_name: str
    option_type: str
    strike: float
    expiration_timestamp_ms: int
    future_name: str | None


@dataclass(frozen=True)
class FutureContract:
    """One dated BTC future used as an expiry forward."""

    instrument_name: str
    expiration_timestamp_ms: int


@dataclass(frozen=True)
class OptionQuote:
    """A ticker update. Market inputs are retained; exchange IV is intentionally absent."""

    instrument_name: str
    timestamp_ms: int
    bid_price: float | None
    ask_price: float | None
    bid_amount: float | None
    ask_amount: float | None
    index_price: float | None = None
    underlying_price: float | None = None
    interest_rate: float | None = None
    underlying_index: str | None = None


@dataclass(frozen=True)
class FutureQuote:
    """A dated-future ticker update."""

    instrument_name: str
    timestamp_ms: int
    mark_price: float | None
    bid_price: float | None
    ask_price: float | None


@dataclass(frozen=True)
class IndexQuote:
    """A BTC index update."""

    timestamp_ms: int
    price: float


@dataclass(frozen=True)
class FutureState:
    """Last accepted forward for one expiry."""

    price: float
    timestamp_ms: int
    source: str


@dataclass(frozen=True)
class ForwardData:
    """Index and expiry forward aligned to one option quote."""

    forward: float | None
    forward_timestamp_ms: int | None
    index: float | None
    index_timestamp_ms: int | None
    source: str


@dataclass
class MarketBook:
    """Latest index and dated-future forwards. Option tickers do not update it."""

    index_price: float | None = None
    index_timestamp_ms: int | None = None
    futures: dict[str, FutureState] = field(default_factory=dict)

    def note_index(self, quote: IndexQuote) -> None:
        """Record a positive finite index, ignoring out-of-order updates."""
        if (
            math.isfinite(quote.price)
            and quote.price > 0.0
            and (self.index_timestamp_ms is None or quote.timestamp_ms > self.index_timestamp_ms)
        ):
            self.index_price = quote.price
            self.index_timestamp_ms = quote.timestamp_ms

    def note_future(self, quote: FutureQuote) -> None:
        """Record a future mark, or its mid when the mark is missing."""
        resolved = forward_from_future_quote(quote)
        if resolved is None:
            return
        price, source = resolved
        previous = self.futures.get(quote.instrument_name)
        if previous is None or quote.timestamp_ms > previous.timestamp_ms:
            self.futures[quote.instrument_name] = FutureState(price, quote.timestamp_ms, source)

    def forward_data(self, instrument: Instrument) -> ForwardData:
        """Forward for this option's dated future, never the option ticker."""
        state = self.futures.get(instrument.future_name) if instrument.future_name else None
        if state is None:
            return ForwardData(
                None,
                None,
                self.index_price,
                self.index_timestamp_ms,
                FORWARD_SOURCE_MARK,
            )
        return ForwardData(
            state.price,
            state.timestamp_ms,
            self.index_price,
            self.index_timestamp_ms,
            state.source,
        )


@dataclass(frozen=True)
class NormalizedQuote:
    """Black-76 inputs in USD present value for the available quote sides."""

    instrument_name: str
    timestamp_ms: int
    option_type: str
    strike: float
    expiry: float
    forward: float
    discount: float
    index: float
    bid_premium_btc: float | None
    ask_premium_btc: float | None
    bid_black_price: float | None
    ask_black_price: float | None
    mid_black_price: float | None
    price_unit: str
    forward_source: str
    time_basis: str
    conversion: str


@dataclass(frozen=True)
class IvRow:
    """Available bid/ask/mid implied vols and side-level rejection details."""

    timestamp_ms: int
    instrument_name: str
    option_type: str
    strike: float
    expiry: float
    bid_iv: float | None
    ask_iv: float | None
    mid_iv: float | None
    index: float
    forward: float
    discount: float
    bid_premium_btc: float | None
    ask_premium_btc: float | None
    model: str
    price_unit: str
    forward_source: str
    time_basis: str
    conversion: str
    side_rejections: tuple[tuple[str, Rejection], ...]


def year_fraction_actual_365_fixed(quote_timestamp_ms: int, expiration_timestamp_ms: int) -> float:
    """Actual/365 Fixed year fraction from quote time to exchange expiry.

    The fraction includes hours and seconds. It is not the difference of
    calendar dates.
    """
    return (expiration_timestamp_ms - quote_timestamp_ms) / _MS_PER_YEAR


def future_name_for(instrument_name: str) -> str | None:
    """Dated future symbol implied by an inverse BTC option name."""
    match = _OPTION_NAME.match(instrument_name)
    if match is None:
        return None
    return f"BTC-{match.group(1)}"


def forward_from_future_quote(quote: FutureQuote) -> tuple[float, str] | None:
    """Prefer the future mark. Fall back to a non-crossed positive mid."""
    mark = quote.mark_price
    if mark is not None and math.isfinite(mark) and mark > 0.0:
        return mark, FORWARD_SOURCE_MARK
    bid, ask = quote.bid_price, quote.ask_price
    if (
        bid is not None
        and ask is not None
        and math.isfinite(bid)
        and math.isfinite(ask)
        and bid > 0.0
        and ask > 0.0
        and bid <= ask
    ):
        return (bid + ask) / 2.0, FORWARD_SOURCE_MID
    return None


def subscription_channels(instruments: list[Instrument]) -> list[str]:
    """Index, dated-future, and option ticker channels for one universe."""
    channels = [_INDEX_CHANNEL]
    futures = sorted({item.future_name for item in instruments if item.future_name})
    channels.extend(f"ticker.{name}.100ms" for name in futures)
    channels.extend(f"ticker.{item.instrument_name}.100ms" for item in instruments)
    return channels


def parse_option_instruments(
    payload: dict, *, now_ms: int
) -> tuple[list[Instrument], list[Rejection]]:
    """Parse ``public/get_instruments`` and drop anything that is not a live inverse option."""
    instruments: list[Instrument] = []
    rejections: list[Rejection] = []
    for row in _result_rows(payload):
        parsed = _parse_option_row(row, now_ms)
        if isinstance(parsed, Rejection):
            rejections.append(parsed)
        else:
            instruments.append(parsed)
    return instruments, rejections


def parse_future_instruments(payload: dict, *, now_ms: int) -> list[FutureContract]:
    """Parse dated BTC futures. Perpetuals and expired contracts are omitted."""
    futures: list[FutureContract] = []
    for row in _result_rows(payload):
        if not isinstance(row, dict):
            continue
        name = row.get("instrument_name")
        if not isinstance(name, str) or _FUTURE_NAME.match(name) is None:
            continue
        if row.get("settlement_currency") not in (None, "BTC"):
            continue
        if row.get("is_active") is False:
            continue
        expiration = _timestamp_ms(row.get("expiration_timestamp"))
        if expiration is None or expiration <= now_ms:
            continue
        futures.append(FutureContract(name, expiration))
    return futures


def attach_listed_futures(
    options: list[Instrument], futures: list[FutureContract]
) -> list[Instrument]:
    """Attach matching dated futures for fallback; keep options with ticker forwards too."""
    listed = {item.instrument_name for item in futures}
    return [
        option if option.future_name in listed else replace(option, future_name=None)
        for option in options
    ]


def parse_subscription_message(
    message: dict,
) -> OptionQuote | FutureQuote | IndexQuote | None:
    """Parse one Deribit subscription notification. Other RPC messages return None."""
    if not isinstance(message, dict) or message.get("method") != "subscription":
        return None
    params = message.get("params")
    if not isinstance(params, dict):
        return None
    channel = params.get("channel")
    data = params.get("data")
    if not isinstance(channel, str) or not isinstance(data, dict):
        return None
    if channel == _INDEX_CHANNEL:
        return _parse_index(data)
    if not (channel.startswith("ticker.") and channel.endswith(".100ms")):
        return None
    name = channel[len("ticker.") : -len(".100ms")]
    if _OPTION_NAME.match(name):
        return _parse_option_ticker(name, data)
    if _FUTURE_NAME.match(name):
        return _parse_future_ticker(name, data)
    return None


def normalize_deribit_quote(
    event: OptionQuote,
    instrument: Instrument,
    forward_data: ForwardData,
    *,
    as_of_ms: int,
    stale_after_ms: int = STALE_AFTER_MS,
) -> NormalizedQuote | Rejection:
    """Convert one inverse quote into Black-76 inputs, or a reasoned rejection."""
    name = instrument.instrument_name
    timestamp = event.timestamp_ms
    expiry = year_fraction_actual_365_fixed(timestamp, instrument.expiration_timestamp_ms)
    if expiry <= 0.0:
        return _reject(name, EXPIRED, "quote is at or after expiration", timestamp)
    if _stale(timestamp, as_of_ms, stale_after_ms):
        return _reject(name, STALE, "option quote is older than the stale window", timestamp)
    index = event.index_price if event.index_price is not None else forward_data.index
    index_timestamp = (
        timestamp if event.index_price is not None else forward_data.index_timestamp_ms
    )
    if index is None or index_timestamp is None:
        return _reject(name, MISSING_INDEX, "BTC index has not been received", timestamp)
    if not math.isfinite(index) or index <= 0.0:
        return _reject(name, INVALID_INDEX, "BTC index must be positive", timestamp)
    if index_timestamp != timestamp and _stale(index_timestamp, as_of_ms, stale_after_ms):
        return _reject(name, STALE, "BTC index is older than the stale window", timestamp)

    forward = event.underlying_price if event.underlying_price is not None else forward_data.forward
    forward_timestamp = (
        timestamp if event.underlying_price is not None else forward_data.forward_timestamp_ms
    )
    if forward is None or forward_timestamp is None:
        return _reject(name, MISSING_FORWARD, "expiry forward has not been received", timestamp)
    if not math.isfinite(forward) or forward <= 0.0:
        return _reject(name, INVALID_FORWARD, "expiry forward must be positive", timestamp)
    if forward_timestamp != timestamp and _stale(forward_timestamp, as_of_ms, stale_after_ms):
        return _reject(name, STALE, "expiry forward is older than the stale window", timestamp)

    if event.interest_rate is not None:
        if not math.isfinite(event.interest_rate):
            return _reject(name, "invalid_interest_rate", "interest rate must be finite", timestamp)
        try:
            discount = math.exp(-event.interest_rate * expiry)
        except OverflowError:
            return _reject(name, "invalid_discount", "discount factor overflowed", timestamp)
        conversion = f"black_price = coin_premium * index; discount = exp(-{event.interest_rate} * expiry)"
    else:
        discount = index / forward
        conversion = "black_price = coin_premium * index; discount = index / forward"
    if not math.isfinite(discount) or discount <= 0.0:
        return _reject(name, "invalid_discount", "discount factor must be positive and finite", timestamp)

    bid = _live_side(event.bid_price, event.bid_amount)
    ask = _live_side(event.ask_price, event.ask_amount)
    if bid is None and ask is None:
        return _reject(name, MISSING_SIDE, "both bid and ask are missing or non-positive", timestamp)
    if bid is not None and ask is not None and bid > ask:
        return _reject(name, CROSSED, "bid price is above the ask price", timestamp)

    bid_black = bid * index if bid is not None else None
    ask_black = ask * index if ask is not None else None
    mid = (bid + ask) / 2.0 if bid is not None and ask is not None else None
    mid_black = mid * index if mid is not None else None
    forward_source = (
        f"Deribit option ticker underlying_price ({event.underlying_index or 'unspecified'})"
        if event.underlying_price is not None
        else forward_data.source
    )
    return NormalizedQuote(
        instrument_name=name,
        timestamp_ms=timestamp,
        option_type=instrument.option_type,
        strike=instrument.strike,
        expiry=expiry,
        forward=forward,
        discount=discount,
        index=index,
        bid_premium_btc=bid,
        ask_premium_btc=ask,
        bid_black_price=bid_black,
        ask_black_price=ask_black,
        mid_black_price=mid_black,
        price_unit=PRICE_UNIT,
        forward_source=forward_source,
        time_basis=TIME_BASIS,
        conversion=conversion,
    )


def solve_quote_iv(normalized_quote: NormalizedQuote, side: str) -> float | Rejection:
    """Invert one of ``bid``, ``ask``, or ``mid``. Solver failures become rejections."""
    prices = {
        "bid": normalized_quote.bid_black_price,
        "ask": normalized_quote.ask_black_price,
        "mid": normalized_quote.mid_black_price,
    }
    if side not in prices:
        raise ValueError(f"side must be bid, ask, or mid, got {side!r}")
    price = prices[side]
    if price is None:
        return _reject(
            normalized_quote.instrument_name,
            MISSING_SIDE,
            f"{side} premium is unavailable",
            normalized_quote.timestamp_ms,
        )
    try:
        return black_formula_implied_volatility(
            _option_type(normalized_quote.option_type),
            strike=normalized_quote.strike,
            forward=normalized_quote.forward,
            expiry=normalized_quote.expiry,
            black_price=price,
            discount=normalized_quote.discount,
        )
    except ItofinError as exc:
        message = str(exc)
        reason = OUTSIDE_NO_ARBITRAGE if _is_no_arbitrage(message) else SOLVER_FAILED
        return _reject(
            normalized_quote.instrument_name,
            reason,
            f"{side}: {message}",
            normalized_quote.timestamp_ms,
        )


def evaluate_quote(
    event: OptionQuote,
    instrument: Instrument,
    forward_data: ForwardData,
    *,
    as_of_ms: int,
    stale_after_ms: int = STALE_AFTER_MS,
) -> IvRow | Rejection:
    """Normalize a quote and invert bid, ask, and the mid premium separately."""
    normalized = normalize_deribit_quote(
        event,
        instrument,
        forward_data,
        as_of_ms=as_of_ms,
        stale_after_ms=stale_after_ms,
    )
    if isinstance(normalized, Rejection):
        return normalized
    ivs: dict[str, float | None] = {}
    side_rejections: list[tuple[str, Rejection]] = []
    for side in ("bid", "ask", "mid"):
        solved = solve_quote_iv(normalized, side)
        if isinstance(solved, Rejection):
            side_rejections.append((side, solved))
        else:
            ivs[side] = solved
    if not ivs:
        failed = "; ".join(f"{side}: {rejection.detail}" for side, rejection in side_rejections)
        first = side_rejections[0][1]
        return _reject(normalized.instrument_name, first.reason, failed, normalized.timestamp_ms)
    for side in ("bid", "ask", "mid"):
        ivs.setdefault(side, None)
    return IvRow(
        timestamp_ms=normalized.timestamp_ms,
        instrument_name=normalized.instrument_name,
        option_type=normalized.option_type,
        strike=normalized.strike,
        expiry=normalized.expiry,
        bid_iv=ivs["bid"],
        ask_iv=ivs["ask"],
        mid_iv=ivs["mid"],
        index=normalized.index,
        forward=normalized.forward,
        discount=normalized.discount,
        bid_premium_btc=normalized.bid_premium_btc,
        ask_premium_btc=normalized.ask_premium_btc,
        model=MODEL,
        price_unit=PRICE_UNIT,
        forward_source=normalized.forward_source,
        time_basis=TIME_BASIS,
        conversion=normalized.conversion,
        side_rejections=tuple(side_rejections),
    )


def format_timestamp(timestamp_ms: int) -> str:
    """UTC timestamp with millisecond precision."""
    moment = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{timestamp_ms % 1000:03d}Z"


def format_iv_row(row: IvRow) -> str:
    """One text row carrying the model convention used to produce the IVs."""
    return (
        f"{format_timestamp(row.timestamp_ms)} {row.instrument_name} "
        f"option_type={row.option_type} strike={row.strike:.2f} expiry={row.expiry:.8f} "
        f"bid_iv={_format_iv(row.bid_iv)} ask_iv={_format_iv(row.ask_iv)} "
        f"mid_iv={_format_iv(row.mid_iv)} "
        f"index={row.index:.2f} forward={row.forward:.2f} discount={row.discount:.8f} "
        f"bid_btc={_format_number(row.bid_premium_btc)} "
        f"ask_btc={_format_number(row.ask_premium_btc)} "
        f"model={row.model} price_unit={row.price_unit!r} "
        f"forward_source={row.forward_source!r} time_basis={row.time_basis!r} "
        f"conversion={row.conversion!r} "
        f"side_rejections={_format_side_rejections(row.side_rejections)!r}"
    )


def _format_iv(value: float | None) -> str:
    return f"{value:.6f}" if value is not None else "NA"


def _format_number(value: float | None) -> str:
    return f"{value:.8f}" if value is not None else "NA"


def _format_side_rejections(rejections: tuple[tuple[str, Rejection], ...]) -> str:
    return "; ".join(f"{side}:{rejection.reason}={rejection.detail}" for side, rejection in rejections)


def format_rejection(rejection: Rejection) -> str:
    """One diagnostic row for a quote that was not inverted."""
    when = format_timestamp(rejection.timestamp_ms) if rejection.timestamp_ms is not None else "-"
    return (
        f"REJECT {when} {rejection.instrument_name} "
        f"reason={rejection.reason} detail={rejection.detail!r}"
    )


def _parse_option_row(row: object, now_ms: int) -> Instrument | Rejection:
    if not isinstance(row, dict):
        return _reject("?", INVALID_INSTRUMENT, "instrument row is not an object", None)
    name = row.get("instrument_name")
    label = name if isinstance(name, str) else "?"
    if row.get("kind") not in (None, "option"):
        return _reject(label, NOT_OPTION, "instrument kind is not option", None)
    if row.get("is_active") is False:
        return _reject(label, INACTIVE, "instrument is inactive", None)
    settlement = row.get("settlement_currency")
    if settlement != "BTC":
        return _reject(label, NOT_INVERSE, f"settlement_currency is {settlement!r}", None)
    option_type = row.get("option_type")
    if not isinstance(option_type, str) or option_type.lower() not in ("call", "put"):
        return _reject(label, INVALID_INSTRUMENT, "option_type must be call or put", None)
    if not isinstance(name, str) or future_name_for(name) is None:
        return _reject(label, INVALID_INSTRUMENT, "instrument name is not an inverse BTC option", None)
    try:
        strike = float(row["strike"])
    except (KeyError, TypeError, ValueError):
        return _reject(label, INVALID_INSTRUMENT, "strike is missing or not numeric", None)
    if not math.isfinite(strike) or strike <= 0.0:
        return _reject(label, INVALID_INSTRUMENT, "strike must be positive", None)
    expiration = _timestamp_ms(row.get("expiration_timestamp"))
    if expiration is None:
        return _reject(label, INVALID_INSTRUMENT, "expiration_timestamp must be milliseconds", None)
    if expiration <= now_ms:
        return _reject(label, EXPIRED, "instrument is expired", expiration)
    return Instrument(name, option_type.lower(), strike, expiration, future_name_for(name))


def _parse_index(data: dict) -> IndexQuote | None:
    price = _optional_float(data, "price")
    timestamp = _timestamp_ms(data.get("timestamp"))
    if price is None or timestamp is None:
        return None
    return IndexQuote(timestamp, price)


def _parse_option_ticker(name: str, data: dict) -> OptionQuote | None:
    timestamp = _timestamp_ms(data.get("timestamp"))
    if timestamp is None:
        return None
    reported = data.get("instrument_name")
    if isinstance(reported, str) and reported != name:
        return None
    return OptionQuote(
        name,
        timestamp,
        _optional_float(data, "best_bid_price"),
        _optional_float(data, "best_ask_price"),
        _optional_float(data, "best_bid_amount"),
        _optional_float(data, "best_ask_amount"),
        index_price=_optional_float(data, "index_price"),
        underlying_price=_optional_float(data, "underlying_price"),
        interest_rate=_optional_float(data, "interest_rate"),
        underlying_index=(
            data.get("underlying_index") if isinstance(data.get("underlying_index"), str) else None
        ),
    )


def _parse_future_ticker(name: str, data: dict) -> FutureQuote | None:
    timestamp = _timestamp_ms(data.get("timestamp"))
    if timestamp is None:
        return None
    return FutureQuote(
        name,
        timestamp,
        _optional_float(data, "mark_price"),
        _optional_float(data, "best_bid_price"),
        _optional_float(data, "best_ask_price"),
    )


def _result_rows(payload: dict) -> list:
    if not isinstance(payload, dict) or "result" not in payload:
        raise ValueError("Deribit payload has no result")
    rows = payload["result"]
    if not isinstance(rows, list):
        raise ValueError("Deribit result is not a list")
    return rows


def _timestamp_ms(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    timestamp = int(value)
    if timestamp < _MIN_TIMESTAMP_MS:
        return None
    return timestamp


def _optional_float(data: dict, key: str) -> float | None:
    value = data.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _live_side(price: float | None, amount: float | None) -> float | None:
    if price is None or not math.isfinite(price) or price <= 0.0:
        return None
    if amount is not None and (not math.isfinite(amount) or amount <= 0.0):
        return None
    return price


def _stale(timestamp_ms: int, as_of_ms: int, stale_after_ms: int) -> bool:
    return as_of_ms - timestamp_ms > stale_after_ms


def _reject(name: str, reason: str, detail: str, timestamp_ms: int | None) -> Rejection:
    return Rejection(name, reason, detail, timestamp_ms)


def _option_type(name: str) -> OptionType:
    if name == "call":
        return OptionType.Call
    if name == "put":
        return OptionType.Put
    raise ValueError(f"option_type must be call or put, got {name!r}")


def _is_no_arbitrage(message: str) -> bool:
    text = message.lower()
    return "complementary" in text or "no solution" in text
