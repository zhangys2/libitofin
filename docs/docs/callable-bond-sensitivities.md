# Callable bond OAS sensitivities

Both `CallableFixedRateBond` and `CallableZeroCouponBond` expose
`effective_duration` and `effective_convexity` in Rust core. Existing APIs and
bindings are unchanged. These methods use an explicit immutable
`&TreeCallableFixedRateBondEngine`, with its Hull-White curve and time steps.
They do not require, replace or calculate through the installed instrument engine.

```rust
let duration = bond.effective_duration(
    spread, &engine, day_counter.clone(), compounding, frequency, None,
)?;
let convexity = bond.effective_convexity(
    spread, &engine, day_counter, compounding, frequency, Some(0.001),
)?;
```

- `spread` and `bump` are conventional OAS rates, not basis-point counts.
- `None` selects `h = 0.0002`, or two basis points. Each of `s`, `s+h` and `s-h`
  is independently converted using the requested day count and compounding.
- `P(s)` is the **clean settlement price**, per 100, from `clean_price_oas`, and
  `D = P(s) + accrued` is the dirty center price at the default settlement.
- Duration: `(P(s-h) - P(s+h)) / (2 D h)`.
- Convexity: `(P(s+h) + P(s-h) - 2 P(s)) / (D h²)`.
- This dirty-price denominator follows QuantLib 1.44's callable-bond methods
  (`callablebond.cpp:362-369,397-404`); 1.43 divided by the clean price.
  Accrued is read after the three prices, so pricing errors come first.
- A zero dirty center returns zero, after both bumped prices are evaluated.
- Nonfinite, nonpositive and unrepresentable bumps return errors. Pricing errors
  propagate; nonfinite results or arithmetic overflow/underflow are rejected.
- Successful and failed probes preserve the installed engine, its results,
  instrument cache, observer notifications and frozen state.

Native QuantLib 1.44 fixtures cover 36 fixed/zero, call/put/noncallable,
continuous/compounded/simple and default/custom-bump cases. Nonzero accrued
fixed-rate cases distinguish dirty normalization from clean normalization.
`callablebonds.cpp:1113` is ported with its OAS, duration and convexity pinned.
Settlement-normalized OAS quotes retain the underlying OAS method's documented
correction for nonzero settlement lag.
