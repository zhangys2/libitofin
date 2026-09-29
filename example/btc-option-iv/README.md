# BTC option implied volatility

Live per-contract Black implied vols for Deribit inverse BTC options. This
example is not part of the `itofin` library. It uses public market data only
and does not need an API key.

## Quote conversion

Deribit quotes these inverse options in BTC. When present, the option ticker's
own `index_price`, `underlying_price`, and `interest_rate` are used together;
they are the synchronized model inputs for that contract. The conversion
matches Deribit's inverse Black-Scholes formula:

- `black_price = coin_premium * index_price` (USD present value)
- `discount = exp(-interest_rate * expiry)`
- `forward = underlying_price` (the option's expiry underlying)
- If the option ticker omits those fields, use the separate BTC index and
  matching dated-future ticker as a fallback, with `discount = index / forward`.
- Year fraction is Actual/365 Fixed from the quote timestamp to the exchange
  expiry timestamp, including hours and seconds.

Bid, ask, and mid premiums are inverted separately. If one side has no valid
Black IV, the other sides are still emitted and the failed side is annotated;
when an option ticker arrives before fallback reference data, it is held until
the references arrive. The exchange `mark_iv` is not passed to the solver.
Calculated IVs can be compared with Deribit's published IV as a diagnostic;
do not replace the calculated value with it.

## Run

From the repo root, with the local `itofin` extension installed
(`maturin develop -m crates/itofin-py/Cargo.toml`):

```bash
PYTHONPATH=example/btc-option-iv python -m btc_option_iv --max-rows 20
```

Or, from this directory, let uv build the binding and the example:

```bash
uv sync --group dev
uv run btc-option-iv --max-rows 20
```

Each accepted row is labeled with the model (`Black-76`), price unit
(`USD present value`), forward source, and time basis. Rejected quotes are
printed to stderr with a reason. A live run is a manual smoke test, not a CI
check.

## Offline tests

```bash
pytest example/btc-option-iv/tests -v
```
