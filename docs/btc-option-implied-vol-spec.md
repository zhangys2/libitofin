# BTC option market-data and implied-volatility function spec

**Status:** Proposal  
**Scope:** Python-facing `itofin` API plus a separate Deribit live-data example  
**Goal:** Let a Python user consume live BTC option quotes, calculate bid/ask/mid Black implied volatility with `itofin`, and retain the market inputs needed to interpret each result.

## Current state

The Rust core already exports `pricingengines::black_formula_implied_std_dev`. It returns total standard deviation (`volatility * sqrt(expiry)`), not annualized volatility. The generic vanilla-option instrument does not currently expose an implied-volatility method, and the existing core inversion function is not bound into the Python `itofin.pricingengines` module.

Therefore, the narrow library gap is a Python binding and an annualized-volatility convenience function. **The Deribit WebSocket client is not part of the `itofin` library**: it belongs in an example/application, which should pass normalized, correctly-denominated market inputs to `itofin`.

## User workflow

1. Fetch active BTC option instrument metadata from Deribit.
2. Subscribe to option tickers and the BTC index; obtain expiry-specific forward prices from the corresponding futures or another explicit forward source.
3. For each valid option quote, normalize the option premium and pricing inputs to one consistent currency/convention.
4. Invert bid, ask, and (when meaningful) mid prices to annualized IV using `itofin`.
5. Emit rows keyed by instrument and quote timestamp. Surface fitting/interpolation is a later concern and is not part of this spec.

## Proposed Python API

Expose two module-level functions in `itofin.pricingengines`:

```python
black_formula_implied_std_dev(
    option_type: instruments.OptionType,
    strike: float,
    forward: float,
    black_price: float,
    discount: float = 1.0,
    displacement: float = 0.0,
    guess: float | None = None,
    accuracy: float = 1e-8,
    max_iterations: int = 100,
) -> float

black_formula_implied_volatility(
    option_type: instruments.OptionType,
    strike: float,
    forward: float,
    expiry: float,
    black_price: float,
    discount: float = 1.0,
    displacement: float = 0.0,
    accuracy: float = 1e-8,
    max_iterations: int = 100,
) -> float
```

`OptionType` here is the existing Python enum. Names and argument order may be adjusted to match existing binding conventions, but the contracts below are normative.

### Contracts

- `black_formula_implied_std_dev` binds the existing Rust core function with the same meaning and units. `guess=None` selects the core's approximation-seeded path. It returns total standard deviation, not annualized volatility.
- `black_formula_implied_volatility` accepts `expiry` as a positive year fraction, calls the core standard-deviation solver, and returns `std_dev / sqrt(expiry)` as a decimal annualized volatility (e.g. `0.65`, not `65`).
- `strike`, `forward`, `black_price`, `discount`, `displacement`, `expiry`, `accuracy`, and optional `guess` must be finite; strike/forward constraints, price bounds, and positive discount are validated by the core. The convenience function rejects `expiry <= 0`, non-positive accuracy, and `max_iterations <= 0` with `itofin.ItofinError`.
- Solver failures and invalid/no-arbitrage-solution prices propagate as `itofin.ItofinError`; do not return NaN, zero, or a silently clamped IV for bad input.
- `black_price` must be in the same currency and payoff convention expected by the Black formula. The function does not infer or convert exchange quote units.

### Intended use

```python
from itofin.instruments import OptionType
from itofin.pricingengines import black_formula_implied_volatility

iv = black_formula_implied_volatility(
    OptionType.Call,
    strike=100_000.0,
    forward=101_250.0,
    expiry=30 / 365,
    black_price=4_100.0,
    discount=0.999,
)
```

The example values are illustrative only. Call the function separately for bid, ask, and mid premium to produce bid-IV, ask-IV, and mid-IV; averaging bid-IV and ask-IV is not equivalent to inverting the mid premium.

## Deribit live-data adapter requirements (example/application, not core API)

The example should use public market-data APIs only and require no API key:

- Bootstrap active BTC options with `public/get_instruments` (`currency=BTC`, `kind=option`, exclude expired instruments).
- Subscribe over `wss://www.deribit.com/ws/api/v2` to each active option's `ticker.<instrument_name>.100ms` channel and `deribit_price_index.btc_usd`. Track relevant dated futures to estimate/read the forward for each expiry.
- Retain the source timestamp, instrument name, option type, strike, expiration timestamp, bid/ask prices and sizes, index/underlying/forward inputs, and the currency/model conversion used.
- Refresh the instrument universe periodically and after reconnect. For order-book channels, follow snapshot/change-ID recovery rules; do not assume separate channels are synchronized.
- Drop expired instruments, missing/zero-sided quotes, crossed markets, stale updates, and prices outside model no-arbitrage bounds. Preserve rejected rows with a reason in diagnostics rather than fabricating an IV.
- Deribit inverse options are quoted in coin units, while the Black inversion inputs must share a consistent pricing currency. Convert the option premium and use the correct expiry forward/discount convention before calling `itofin`; document that conversion in the example and test it independently. Do not pass the exchange's `mark_iv` as an input to the solver.
- Compute year fraction from the quote timestamp to the exchange expiry timestamp using an explicit convention (the example should default to Actual/365 Fixed and state it). Never calculate time-to-expiry from the calendar date alone.
- Build the output as a live **per-contract IV grid** only. Surface smoothing, SVI calibration, arbitrage-free interpolation, and trading decisions are out of scope.

Suggested example-level functions (names are not public library API):

```text
fetch_active_btc_options() -> instrument metadata
subscribe_btc_option_market_data(instruments) -> async quote events
normalize_deribit_quote(event, instrument, forward_data) -> normalized quote
solve_quote_iv(normalized_quote, side) -> IV or a typed/reasoned rejection
run_live_btc_iv_stream() -> stream/print or persist IV rows
```

No exchange networking, venue symbol parsing, data persistence, or market-specific quote conversion should be added to `itofin` itself.

## Implementation boundaries

### In `itofin`

- Add a small PyO3 binding module/function for the core `black_formula_implied_std_dev`.
- Add the annualized convenience wrapper, preferably in the same binding surface; reuse the Rust core solver rather than implementing another root finder.
- Register the function(s) in `itofin.pricingengines`, generate/update the `.pyi` stub, and document parameters/units/errors.
- Do not add an options WebSocket client or Deribit-specific argument types to the generic quant library.

### In a runnable example

- Add a separately installable/uv-managed example only after the Python binding is implemented.
- Keep WebSocket parsing, feed lifecycle, reconnect/backoff, instrument discovery, quote filtering, and Deribit price-unit conversion outside core.
- Live network checks are manual smoke tests, not CI tests. Core and binding tests must be deterministic/offline.

## Acceptance criteria

1. Rust unit tests continue to cover the existing core implied-standard-deviation solver, including call/put and in-/out-of-the-money cases, and invalid prices.
2. Python tests verify that `black_formula_implied_std_dev` matches the existing core result and that the binding maps core errors to `ItofinError`.
3. Python tests generate Black prices from known volatility and recover the same volatility through the convenience API for calls and puts, several strikes/maturities, positive discount, and nonzero displacement; tolerance `1e-8` absolute for well-conditioned cases.
4. Python tests reject zero/negative expiry, non-finite inputs, invalid solver settings, and arbitrage-inconsistent option prices without returning a numeric IV.
5. The Deribit example has offline fixture tests for instrument/quote parsing, expiry year-fraction, normalization, and stale/invalid quote rejection. No CI test requires live exchange connectivity.
6. A manual live smoke run prints (or writes) timestamped BTC option bid-IV/ask-IV/mid-IV rows and labels model convention, price unit, forward source, and time basis. Compare calculated IV with Deribit's published IV as a diagnostic only; discrepancies are investigated, not patched by substituting the exchange value.

## Explicit non-goals

- A new IV-surface calibration/interpolation API.
- A generic exchange connectivity or market-data abstraction in `itofin`.
- Implementing Deribit's coin-premium conversion inside the library.
- Guaranteeing exact parity with Deribit's proprietary mark-price/IV conventions for every product type.
- Automatically fitting a risk-free or forward curve from exchange data.
