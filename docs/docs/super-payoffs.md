# Superfund and supershare payoffs

`SuperFundPayoff` and `SuperSharePayoff` are additive Rust-core terminal
payoffs matching QuantLib's `ql/instruments/payoffs.hpp` and `payoffs.cpp`.
Both are call-only corridors: the lower strike is inclusive and the upper
strike is exclusive. QuantLib has no suite case for either payoff.

| Payoff | Payment inside `[K1, K2)` | Otherwise |
| --- | --- | --- |
| Superfund | `S / K1` | `0` |
| Supershare | cash amount | `0` |

- `strike()` returns `K1`. `second_strike()` returns `K2`.
- Supershare also exposes `cash_payoff()`.
- Superfund requires `K1 > 0` and `K2 > K1`. Supershare requires only `K2 > K1`.
- A NaN spot fails both comparisons and pays zero.
- `Payoff`, `TypePayoff` and `StrikedTypePayoff` are implemented.
- `BlackCalculator` rejects both with `unsupported payoff type`, as in QuantLib.
  No analytic, finite-difference, or Monte Carlo engine is added.

```rust
use libitofin::instruments::{SuperFundPayoff, SuperSharePayoff};
use libitofin::payoff::Payoff;

let fund = SuperFundPayoff::new(100.0, 120.0).unwrap();
assert_eq!(fund.value(100.0), 1.0);
assert_eq!(fund.value(110.0), 1.1);
assert_eq!(fund.value(120.0), 0.0);

let share = SuperSharePayoff::new(90.0, 110.0, 15.0).unwrap();
assert_eq!(share.value(90.0), 15.0);
assert_eq!(share.value(110.0), 0.0);
```

## Scope

These are scalar payoffs, not a new exotic instrument or pricing engine.
No Python, C or Go binding, existing API change, or release is included.
