# American implied forward validation quotes

`scenarios.tsv` holds synthetic American option chains and the exact forward each was
generated from. `tests/american_forward.rs` bakes the file in (`include_str!`) and checks that
`ImpliedForward` recovers the forward; there is no runtime Python, QuantLib or JSON dependency.

- 15 scenarios, each a 17-strike chain (80 to 120 in steps of 2.5) on a spot of 100:
  nine without dividends (rates 0% to 8%, yields 0% to 6%, 36 to 720 days, one skewed smile),
  six with discrete cash dividends (quarterly 0.5 and 1.0, a 3.00 special dividend, a skewed smile).
- Prices come from `FdBlackScholesVanillaEngine` on an 800 x 1600 Douglas grid (spot cash-dividend
  model), far finer than anything the estimator uses (closed forms, a 400-step tree, a 100 x 200
  grid). The half spread is one cent plus 0.2% of the mid.
- Times are `days / 360` (Act/360). The exact forward is `S e^((r - q) T)`, or
  `(S - PV(dividends)) e^(r T)` with dividends.

Regenerate (about 45 s in a release build):

```sh
cargo run --release -p libitofin --example american_forward_oracle \
  > crates/libitofin/tests/fixtures/american_forward/scenarios.tsv
```

## What the fixtures established

Forward error in basis points over the 15 chains, with the default settings:

| Estimator | Worst | Typical |
| --- | --- | --- |
| Plain European parity (median of per-strike forwards) | 114 | 20 to 100 where early exercise matters |
| Fast level (Ju closed form), no dividends | 6.0 | 2 to 3 |
| Accurate level (400-step tree, or grid with a schedule) | 0.82 | 0.3 |

- The synthetic bid/ask interval from quotes corrected at the mid contains the true forward in all
  15 chains. A unit test also compares the mid correction with a per-side correction at 3% spreads:
  the largest difference is 0.024 of a half spread.
- The corridor width (`corridor_kappa`) does not move the estimate: errors are identical from 0.25 to
  3.0 on these noise-free chains, since the strikes nearest the money drive the carry solve and the
  median sits at the money. Wider windows only add strikes (mean 5 at 0.25, 16 at 1.0, 16.4 at 3.0)
  and, at the Accurate level, accurate-pricer evaluations. The default of 1.0 is the smallest value
  that already admits nearly all of the strikes; the fixtures cannot say whether a wider window helps
  against quote noise.
- A schedule matters: ignoring it (Fast level) costs 46 bp on the 3.00 special dividend.
