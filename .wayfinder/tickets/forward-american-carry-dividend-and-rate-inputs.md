---
title: American Carry, Dividend & Rate Inputs
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Claude + Yuan (grilling session 2026-10-25)"
map: ../maps/implied-forward-from-options.md
resolution: "Model-based path requires a caller-supplied discount_factor (None runs the flagged corridor tier). Optional discrete cash-dividend schedule, used only by the FD correction pass in the Accurate level and ignored with a flag at Fast; FD spot model, fixed. A schedule makes Accurate the default level. Carry search range [-0.10, 0.25] and implausible-carry flag range [-0.05, 0.15] are config fields. Unknown is flat yield q, or residual carry b on top of the schedule."
---

# American Carry, Dividend & Rate Inputs

## Question

For the de-Americanization solver, what is the unknown and what is fixed? Specifically: is the solved quantity $F$ or the borrow/carry rate $q$ (with $F = (S - \text{PV(Div)})e^{rT}$ derived)? How are rates supplied (external OIS discount factor, as today via `ImpliedForwardConfig::discount_factor`), and how are discrete dividends supplied (schedule of dates and amounts, escrowed versus proportional)?

Why it matters: the American pricer needs $r$ and the dividends to compute the EEP. Letting parity fit both $D$ and $F$ on noisy American quotes is poorly conditioned, so the usual choice is to fix $D$ and solve one parameter.

## Context

- Today `discount_factor` is optional and the Merton bounds use escrowed dividend PV in the American ticket's resolution.
- Method settled in [De-Americanization Method & Tier Structure](forward-deamericanization-method-and-tiering.md): per-quote American IV under a trial carry $q$, feeding the existing `ImpliedForward` back-end, so the solver's unknown is the carry (equivalently $F$) with $D$ taken from the curve. Only FD takes cash dividends (see the pricer ticket).

## Resolution

Resolved 2026-10-26 via grilling rounds 1 and 2 (five questions; every answer matched the recommendation).

### Evidence used

Dividend test, 2026-10-25 (quotes from FD 800x1600 with the schedule shown, S = 100, r = 4%, vol 25%, forward error in bp of F):

| Scenario | Current parity | Flat q (Ju) | Flat q (CRR 400) | FD with schedule |
|---|---|---|---|---|
| 1y, 4 x 0.50 quarterly | -24.3 | -3.6 | -0.6 | -0.4 |
| 2y, 8 x 0.50 quarterly | -48.7 | +5.4 | +11.2 | -0.7 |
| 1y, 4 x 1.00 quarterly | -6.7 | -7.5 | -7.8 | -0.2 |
| 1y, one special 3.00 at day 100 | -50.7 | -45.2 | -43.4 | -0.7 |
| 180d, 2 x 1.00 | -6.0 | -6.3 | -6.5 | -0.2 |

The spot and escrowed FD models recovered the forward within about 1.2 bp of each other. The truth model was the spot model, so the figures are not independent of that choice. The benchmark source is in the session scratchpad and was removed from the repo.

### 1. Rates

The model-based path requires `ImpliedForwardConfig::discount_factor`, a single $D$ per expiry that the caller derives from an OIS curve. With `None` the engine runs the corridor tier, which still solves $(F, D)$ jointly, and flags the tier. A yield-curve handle can be added later without breaking callers.

### 2. Dividends

An optional discrete cash-dividend schedule (dates and amounts). Only the FD correction pass of the Accurate level uses it. The Fast level ignores it and flags the result. Without a schedule, dividends are absorbed in the solved flat yield $q$.

### 3. Carry range

The solver searches carry in $[-0.10, 0.25]$ per year and flags a result outside $[-0.05, 0.15]$ as an implausible-carry warning. The result is reported, not failed. Both ranges are config fields with these defaults. This replaces the unimplemented repo-rate flag in the earlier American ticket.

### 4. Dividend model

FD spot cash-dividend model, fixed, no config field. A field can be added later if the validation fixtures show the choice matters.

### 5. Level selection

A supplied schedule makes Accurate the default level. A caller can force Fast, in which case the schedule is ignored and the result is flagged. A latency overrun falls back to Fast automatically, with a flag.

### Derived, not decided

The solver's unknown is the flat yield $q$ when no schedule is given, or a residual continuous carry $b$ on top of the schedule, with $F = (S - \text{PV(Div)})\,e^{(r-b)T}$. The carry range in item 3 applies to $q$ or $b$.

### Handed to the API ticket

The names and representation of the tier, schedule-ignored and implausible-carry flags; where the ATM vol for the widened corridor comes from (the model path uses per-quote IVs and does not need it, the corridor tier does); and the spot being required on the model path (already required by `AmericanEquity { spot }`).

### Contingency

The schedule path relies on [Hybrid Premium-Correction Prototype](forward-hybrid-premium-correction-prototype.md). If that prototype fails, schedule handling falls back to a plain FD solve instead of the one-pass correction.
