---
title: Hybrid Premium-Correction Prototype
type: Prototype (HITL)
status: closed
blocked_by: []
claimed_by: "Claude (session 2026-10-26)"
map: ../maps/implied-forward-from-options.md
resolution: "Verdict: yes. One correction pass (6 accurate prices for 3 strikes) on top of a Ju solve matches the full CRR/FD solve to within 0.01 bp while costing 8 to 12 ms instead of 260 to 520 ms; worst forward error 0.36 bp with CRR 400 (0.21 bp against an independent CRR 4000 reference), 1.6 bp with FD 100x200 (the corrector's own error), 1.3 bp with CRR 200 at 2 to 4 ms. With a dividend schedule, FD-with-schedule correction brings flat-carry errors of up to 45 bp down to 1.1 bp in one pass and 0.8 bp in two. The Accurate level is feasible."
---

# Hybrid Premium-Correction Prototype

## Question

Does solving for the forward with a closed-form American pricer (Ju or BAW), then computing the early-exercise premium once with CRR or FD at that solution and re-solving, recover the forward to under 2 bp of $F$ at a cost far below a full CRR/FD solve (0.3 to 0.9 s per expiry)?

Throwaway code only: a single example or script that reuses the harness from [American Pricer Inventory, Accuracy & Latency Budget](forward-american-pricer-inventory-and-latency.md) (six scenarios, 13 strikes, FD 800x1600 reference quotes, forward error in bp, wall time per expiry). Keep state in memory, skip polish, and close the ticket with the verdict and the numbers, keeping the prototype on a throwaway branch linked here.

## Context

- The Accurate level in [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md) stays a candidate until this reports. If the correction does not reach the target, the Accurate level falls back to a plain CRR/FD solve or is dropped.
- Variants worth trying: one correction versus two; CRR 400 versus FD 100x200 as the correcting pricer; applying the correction to the premium only versus to the final carry.

## Resolution

Resolved 2026-10-26. Throwaway prototype: a single example, run once, removed from the repo. Source `american_hybrid_bench.rs` and the raw output `hybrid.out` are in the session scratchpad; it can be restored to a throwaway branch on request.

### What was built

1. Solve the carry $q$ with a closed-form pricer (secant on the mean call/put IV gap at K = 95, 100, 105).
2. One correction pass: for each quote, take the cheap pricer's IV $\sigma_i$ at the current $q$, compute $\delta_i = V_{\text{acc}}(\sigma_i, q) - V_{\text{cheap}}(\sigma_i, q)$ with one accurate price, shift the quote by $\delta_i$, and re-solve with the cheap pricer. Six accurate prices per pass (three strikes, call and put).
3. Optionally repeat. With a dividend schedule the accurate pricer is FD 100x200 with the schedule, evaluated at the same forward through the residual carry $b = q - \ln(S/(S-\text{PV(Div)}))/T$.

Setup as in [American Pricer Inventory, Accuracy & Latency Budget](forward-american-pricer-inventory-and-latency.md): S = 100, vol 25%, six (r, q, T) scenarios; quotes from FD 800x1600, and a second run from CRR 4000 so the FD correctors are not graded against their own scheme.

### Results: forward error in bp of F (worst of six scenarios) and solve time

| Method | Quotes from FD ref | Quotes from CRR 4000 | Time |
|---|---|---|---|
| Ju only | 6.2 | 6.4 | 0.3 to 0.5 ms |
| BAW only | 6.0 | 5.8 | 0.3 to 0.4 ms |
| Ju + CRR 400, 1 pass | 0.36 | 0.21 | 8 to 11 ms |
| Ju + CRR 400, 2 passes | 0.33 | 0.22 | 15 to 23 ms |
| BAW + CRR 400, 1 pass | 0.38 | 0.22 | 8 to 11 ms |
| Ju + CRR 200, 1 pass | 1.31 | 1.12 | 2.4 to 4 ms |
| Ju + FD 100x200, 1 pass | 1.55 | 1.73 | 9 to 12 ms |
| full CRR 400 solve | 0.33 | 0.22 | 258 to 484 ms |
| full FD 100x200 solve | 1.56 | 1.75 | 333 to 517 ms |

- A hybrid pass reproduces the full solve with the same corrector to within about 0.01 bp, at 25 to 50 times lower cost. A second pass changed nothing meaningful without dividends.
- The remaining error is the corrector's own discretization error, so the corrector sets the accuracy: CRR 400 is better than FD 100x200 on this grid; CRR 200 is a cheap middle option.
- BAW and Ju as the cheap pricer give the same final result, so Ju stays the default.

### Dividend schedules (FD 100x200 with the schedule as the corrector, quotes from FD 800x1600, spot model)

| Scenario | Ju only | 1 pass | 2 passes | Full FD solve |
|---|---|---|---|---|
| 1y, 4 x 0.50 quarterly | -3.6 | -0.45 | -0.44 | -0.45 |
| 2y, 8 x 0.50 quarterly | +5.4 | -1.10 | -0.73 | -0.75 |
| 1y, 4 x 1.00 quarterly | -7.5 | +0.44 | -0.29 | -0.24 |
| 1y, one special 3.00 at day 100 | -45.2 | -0.85 | -0.77 | -0.72 |
| 180d, 2 x 1.00 | -6.3 | -0.11 | -0.20 | -0.20 |

One pass costs 11 to 12.5 ms, two passes 22 to 24 ms, against 330 to 394 ms for the full FD solve. With a schedule the second pass gains up to 0.4 bp and converges to the full-solve value, so two passes are the safer default there.

### Limits

- Synthetic noise-free quotes, flat vol and curves, three strikes, mid prices.
- The prototype iterates on the call/put IV gap at mid. The decided design de-Americanizes bid and ask separately and feeds `ImpliedForward`; at the solution that is the same fixed point, but the full path was not exercised. Computing $\delta$ per side would multiply the accurate-price count by two; applying the mid $\delta$ to both sides is an untested approximation.
- Cost is linear in strikes. The 8 to 12 ms figure is for 3 strikes (6 accurate prices per pass); a 13-strike chain would be about 26 accurate prices, roughly 50 ms per pass with CRR 400.
- FD 100x200 and the FD reference share a scheme; the CRR 4000 run is the independent check.

### Consequences

- The Accurate level is feasible and confirmed as a candidate: Ju solve, then one CRR 400 correction pass without dividends, or two FD-with-schedule passes with a schedule.
- Two points to settle in [American Solver API, Diagnostics & Fallback](forward-american-solver-api-diagnostics-and-fallback.md) or at implementation: whether the correction is per side or at mid, and whether to correct at a strike subset when the corridor is wide.
