# BTC option implied volatility

Live per-contract Black implied vols for Deribit inverse BTC options. This
example is not part of the `itofin` library. It uses public market data only
and does not need an API key.

## Quote conversion

Deribit quotes these options in BTC. Implied volatility uses the expiry
forward, not the spot index. The conversion below matches Deribit's published
inverse Black-Scholes formula (`R = ln(forward / index) / expiry`):

- `black_price = coin_premium * index` (USD present value)
- `discount = index / forward`
- `forward` is the dated future's `mark_price` (or the bid/ask mid when the
  mark is missing)
- year fraction is Actual/365 Fixed from the quote timestamp to the exchange
  expiry timestamp, including hours and seconds

Bid, ask, and mid premiums are inverted separately. The exchange `mark_iv` is
not passed to the solver. Calculated IVs can be compared with Deribit's
published IV as a diagnostic; do not replace the calculated value with it.

## Run

From a virtualenv that already has the local `itofin` extension installed
(`maturin develop -m crates/itofin-py/Cargo.toml`):

```bash
python -m btc_option_iv --max-rows 20
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
