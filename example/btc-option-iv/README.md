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

## Live marimo surface

The notebook is [`live_vol_surface.py`](live_vol_surface.py): edit it in a
regular Python editor or open it in marimo's web editor. Install the optional
tools and launch from this directory:

```bash
uv sync --group dev --group notebook
uv run --group dev --group notebook marimo edit live_vol_surface.py --host 127.0.0.1 --port 8888 --headless
```

For a VPS, keep the server on localhost and open its printed URL through an SSH
tunnel (for port 8888: `ssh -N -L 8888:127.0.0.1:8888 <your-vps-login>`).
On headless Linux, set `PLOTLY_RENDERER=json` before launching or exporting if
Plotly hangs while probing desktop browsers; marimo still renders the interactive
figures in the browser.
Switch **Connect to live Deribit quotes** on to start the public feed. The app
updates an interactive Plotly surface from fresh mid-IVs and a searchable
DataFrame with the 100 most recently updated option quotes, BTC bid/ask
premiums, amounts, reference prices, and solved bid/mid/ask IVs. Switch the
feed off to stop it. The surface shows observed points until there is enough
data to interpolate; interpolation is for visualization only, not an
arbitrage-free fit.

### Kalman smile reader

The selected-expiry chart overlays the **current regularized fit** (dashed) and
**Kalman-filtered smile** (solid). Both use exactly nine natural-cubic knots at
`[-3, -1.5, -1, -0.6, 0, 0.6, 1, 1.5, 3]`; seven quotes still give nine knot IVs
and eight segments. The regularized fit minimizes mean squared quote error plus
a curvature penalty (`smoothing=0.01`), permitting initialization from two
fresh, distinct in-range strikes. It is used for bootstrap/reference only, not
as a repeated filter pseudo-measurement. `smoothing=0` in `CubicSmileSection`
still requests an unregularized fit requiring full observation rank.

The feed controller updates **every expiry independently on source events**, not
on UI refreshes. The expiry dropdown updates only when the expiry catalog changes,
retains your choice across quote refreshes, and falls back to the first available
expiry only if the selected expiry disappears. The reader displays its selected
UTC expiry; changing expiry resets chart zoom, not filter history. At a shared
strike, the tighter bid–ask IV spread wins; ties prefer the out-of-the-money
option, then instrument name. Amount/timestamp-only heartbeats and index/forward
revaluations do not create new premium versions. An unconsumed version can become
usable once; selecting an already-consumed call/put version never replays it.
Precision is inverse **IV spread**, not inverse premium spread or squared spread.
A diagonal measurement covariance is an approximation, not independent-market
truth.

The global controls use IV **percentage points**:

- **Process IV rate:** default `0.1` points/√second (decimal `0.001`).
- **Measurement SD:** default `0.5` points at a one-point IV spread (`0.005`).
- **Spread floor:** fixed `0.1` points (`0.001`); inverse-spread precision.
- **Reset filters:** clears all estimates and bootstraps from eligible cached
  quotes. Changing noise controls preserves history and affects subsequent time.

Forward/ATM/time changes transport the curve and covariance in strike space.
Endpoint extension is linear, slope-capped, and limited to `0.25` standardized
units; larger moves reset. Five-SD innovations are gated before their ATM IV can
change normalization. Same-direction breaches at three distinct strikes within
three seconds trigger a regime-change reset. This specifically covers broad
level jumps, not every skew shift. Filters become stale after ten seconds without
an accepted measurement and reinitialize after sixty. Short gaps retain the
last observed-support curve with a stale label. With the feed switched off, the
view is explicitly frozen; elapsed time/gap policy is applied when resumed.
Server restart loses all states; nothing is persisted.

The reader displays forward, time, ATM reference IV, unmodified filtered knot
values and coefficients, selected quote IV spreads, raw/filtered residuals,
innovation statistics, premium versions, measurement status, freshness, resets,
and warnings. Status/innovation records refer to the last processing of that
premium version, not a new acceptance on each heartbeat. Reference residuals
include the static fitter's display floor; filtered residuals use the unclipped
internal cubic. `used_in_fit` marks current reference-fit domain eligibility,
not a new filter update.
Curves stay inside observed support and the knot domain. Out-of-range quotes
are marked and do not update the fit. Negative internal IVs/overshoot are
**floored only for plotting**, with a warning; no covariance is silently changed.
Unsampled regions are model-dependent, not market measurements. Residuals and
covariance are not calibrated uncertainty, there are no confidence bands, and
neither curve is arbitrage-free or a trading signal.

Implementation: `btc_option_iv/kalman.py` (NumPy/SciPy math and expiry manager),
`live.py` (source events and three-second immutable snapshot publication), and
`views.py` (pure charts/tables). These optional notebook modules do not change
the Rust API or the dependency-light streaming CLI.

## Offline tests

```bash
pytest example/btc-option-iv/tests -v
```

The deterministic replay uses seed 430, seven quotes/second, independent `0.005`
decimal-IV noise, `0.01` IV spreads, fixed physical strikes/forward/time, and the
default controls. It samples every three seconds after 30 seconds of bootstrap.
The stationary replay reduces frame-to-frame RMS jitter **64.9%** relative to
the same quotes' regularized fit (`0.001576` vs `0.004489` decimal IV). A sustained
five-point level step reaches 90% at the first post-step sample (10 ms of
simulated receipt time, after the seven-quote batch/regime reset). Targets are
≥30% jitter reduction and ≤10 seconds step response. These are synthetic
scenario results, not live guarantees. Tests also cover sparse/bootstrap cases,
transport/Jacobians, expiry isolation, deduplication/reference revaluation,
solver-failure diagnostics, negativity, stale gaps, outliers, and a two-sided
skew replay without promising automatic skew-reset behavior.
