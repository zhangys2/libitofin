---
title: American Pricer Inventory, Accuracy & Latency Budget
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: "Claude (session 2026-10-25)"
map: ../maps/implied-forward-from-options.md
resolution: "Inventory and measurement done 2026-10-25. Closed-form Ju/BAW price in 1 to 2 microseconds and recover the forward to within about 6 bp on synthetic American quotes; CRR 400 steps and FD 100x200 recover it to under 2 bp but cost 2 ms per price (about 250 to 900 ms per full IV-matching solve). Only the FD engine handles cash dividends. Vanilla implied-volatility solving does not exist yet. The current corridor path is biased by up to 114 bp on a 2-year 5%-rate case."
---

# American Pricer Inventory, Accuracy & Latency Budget

## Question

Which American-option pricers already exist in `libitofin` (analytic approximations, binomial/lattice, finite-difference engines), what are their accuracy and per-quote cost near the money for equity-style inputs (short to 2-year maturities, discrete dividends), and is any of them fast and accurate enough to sit inside a Newton or root-finding loop over 20 to 50 strikes in a live feed?

Deliverable: a table of candidate pricers with error versus a high-resolution reference, microseconds per price, and per implied-vol solve, plus whether Greeks (vega) are available for the IV solve. Report the cheapest pricer that keeps EEP error below the typical bid-ask spread.

## Context

- Feeds the choice in [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md) and the latency discussion in the earlier rejection of PDE inversion.
- Existing live-feed code path: `example/btc-option-iv/spx_option_iv/feed.py` (`parity_forward`).

## Resolution

Resolved 2026-10-25. Findings only; the choice of method stays with [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md).

### 1. Inventory (`crates/libitofin/src/pricingengines/vanilla/`)

| Pricer | Kind | Cash dividends | Vega | Notes |
|---|---|---|---|---|
| `BaroneAdesiWhaleyApproximationEngine` | closed form (BAW 1987) | no (flat q from the process) | European Black vega | |
| `BjerksundStenslandApproximationEngine` | closed form (1993, one flat boundary) | no | analytic Greeks | the 1993 version, not the 2002 two-step |
| `JuQuadraticApproximationEngine` | closed form (Ju 1999) | no | European Black vega, delta/gamma American | |
| `BinomialVanillaEngine` | CRR lattice | no (flat r, q, sigma extracted at maturity) | none filled | |
| `FdBlackScholesVanillaEngine` | finite difference | yes, spot or escrowed model (`CashDividendModel`) | delta/gamma/theta | `FdmAmericanEngine` is the dividend-free American path |
| `McAmericanEngine` | Monte Carlo | not examined | not examined | not benchmarked, too noisy for a root-finder |

There is no `implied_volatility` on `OneAssetOption`/`VanillaOption`; `instruments/oneassetoption.rs` lists it as follow-up work. Any IV-matching solver has to be built on `math/solvers1d` (Brent and others exist) with the pricer inside the loop. Because the closed forms report the European vega, a derivative-free solver is the natural fit.

### 2. Accuracy

Setup: S = 100, vol 25%, K in {90, 95, 100, 105, 110}, 36 to 720 days, (r, q) in {(4%, 1.5%), (5%, 0%), (1%, 3%)}, puts and calls, 120 cases. Reference: `FdBlackScholesVanillaEngine` Douglas 800x1600. The reference differs from CRR 8000 by up to 1.3e-3 in price at 2 years (about 0.003 vol points), so engine errors below that are not resolved. Largest early-exercise premium in the grid: 2.02.

| Engine | max abs price error | max error, vol points | 95th pct, vol points |
|---|---|---|---|
| BAW | 0.098 | 0.487 | 0.203 |
| Bjerksund-Stensland 1993 | 0.143 | 0.392 | 0.319 |
| Ju quadratic | 0.060 | 0.110 | 0.087 |
| CRR 100 | 0.037 | 0.084 | 0.067 |
| CRR 400 | 0.0093 | 0.018 | 0.016 |
| CRR 1000 | 0.0038 | 0.0074 | 0.0069 |
| FD 50x100 | 0.034 | 0.140 | 0.047 |
| FD 100x200 | 0.015 | 0.034 | 0.021 |
| FD 200x400 | 0.0063 | 0.011 | 0.0078 |

Caveat: the FD candidates share a scheme with the FD reference, so their errors are optimistic. CRR is an independent family.

### 3. Latency (release build, one thread, ATM put, 180 days)

| Engine | per price | per IV solve (6 evaluations) |
|---|---|---|
| BAW | 1.4 us | 28 us |
| Ju | 2.2 us | 29 us |
| Bjerksund-Stensland | 5.1 us | 54 us |
| CRR 100 / 400 / 1000 | 0.15 / 2.1 / 9.7 ms | 8.5 ms at 400 |
| FD 50x100 / 100x200 / 200x400 | 0.50 / 1.8 / 6.1 ms | 9.6 ms at 100x200 |

IV solve = regula falsi on [0.02, 1.5] to 1e-9 in price.

### 4. End-to-end forward recovery

This is the figure of merit for the parent ticket. Quotes: the FD reference at known (r, q), 13 strikes 85 to 115, tiny spread, D fixed to truth. Error is in basis points of F. IV matching solves q so that IV_call(K) equals IV_put(K), mean over K = 95, 100, 105, by secant.

| Scenario | current parity | current corridor | BAW | BjS | Ju | CRR 400 | FD 100x200 |
|---|---|---|---|---|---|---|---|
| r5% q0 180d | -23.1 | -23.1 | -1.1 | -8.7 | -1.8 | +0.2 | -0.5 |
| r5% q0 720d | -113.8 | -113.8 | +6.0 | -14.7 | -3.7 | +0.3 | -1.6 |
| r4% q1.5% 360d | -27.2 | -30.1 | +0.5 | -8.4 | -2.8 | +0.3 | -0.6 |
| r1% q3% 360d | +20.5 | +18.9 | +0.6 | +7.4 | +2.4 | -0.3 | +0.4 |
| r3% q6% 720d | +98.6 | +98.6 | -2.7 | +13.6 | +6.2 | -0.2 | +1.2 |
| r5% q0 36d | -3.6 | -4.9 | -1.0 | -2.2 | -0.3 | +0.1 | -0.1 |

Wall time of the whole IV-matching solve (3 strikes, call and put, about 300 to 330 price evaluations): BAW 0.3 to 0.6 ms, Ju 0.3 to 0.5 ms, BjS 0.6 to 1.1 ms, CRR 400 240 to 650 ms, FD 100x200 330 to 900 ms.

### 5. Findings that bear on the other tickets

- **Current behaviour is a median, so the corridor barely matters.** With `discount_factor` given, `calculate_linear` takes the median of per-strike $F_i = K + (C-P)/D$, not the WLS fit. Symmetric strike sets keep the ATM strike as the median either way, so the corridor and the plain European run agree in four of six rows. The bias is the ATM early-exercise premium divided by $D$.
- **The Merton check is a status flag only.** It can set `DegradedAmericanBounds` but never moves the point estimate. The closed American ticket also describes dividend adjustment and `Confidence` values that do not exist in the implemented code; `ForwardStatus` is the only flag set.
- **Dividends rule out the closed forms as primary.** Only FD takes a cash dividend. On a 2.0 dividend at day 90 (put, 1 year) the spot and escrowed FD models differ by 0.055 in price, larger than FD 100x200 grid error (0.004 to 0.005). Closed forms on $S - \text{PV(Div)}$ matched the escrowed FD for the call exactly and for the put to within 0.006 (BAW), 0.026 (Ju), 0.112 (BjS). Which dividend model to use is an input decision, see [American Carry, Dividend & Rate Inputs](forward-american-carry-dividend-and-rate-inputs.md).
- **Bjerksund-Stensland 1993 is the weakest closed form here** and should not be the default.
- **A two-stage scheme is untested but plausible:** solve with Ju or BAW (sub-millisecond), then compute the early-exercise premium once with CRR or FD at the solution and re-solve. This would fit the iterative-subtraction method; it is a suggestion for the method ticket, not something measured.

### 6. Limits of the measurement

Single flat vol, flat curves, S = 100, no skew; quotes are synthetic with no noise, so quote noise and strike selection are not exercised. The benchmark source is `american_pricer_bench.rs` in the session scratchpad (a temporary example, since removed from the repo); it can be restored to a branch on request.
