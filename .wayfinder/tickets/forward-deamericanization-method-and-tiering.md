---
title: De-Americanization Method & Tier Structure
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Claude + Yuan (grilling session 2026-10-25)"
map: ../maps/implied-forward-from-options.md
resolution: "De-Americanize each quote to European-equivalent bid/ask under a trial carry q (per-quote American IV), feed them to the existing ImpliedForward EuropeanVanilla back-end, close the loop on q by secant/fixed point, and report the call/put IV gap as a diagnostic. Model-based is the default tier; the corridor is the fallback. Fast level = closed-form only (Ju default, BAW alternative); Accurate level = closed-form solve plus one CRR/FD premium correction, pending a prototype. Level chosen by config enum, automatic fall back to Fast on a latency overrun. Strike set: a wider corridor than today, kappa fixed after validation fixtures exist."
---

# De-Americanization Method & Tier Structure

## Question

Which model-based method(s) should `ImpliedForward` add for `MarketConvention::AmericanEquity`, and how do they relate to the existing model-free ATM corridor filter?

Candidates to choose between (or combine):

1. **Iterative EEP subtraction.** Provisional $(F, \sigma)$; price each quote with an American pricer to get its early-exercise premium $\text{EEP}_i = V^{\text{Amer}}_i - V^{\text{Euro}}_i$; subtract it to get European-equivalent prices; rerun the existing WLS parity regression for a new $F$; repeat to convergence.
2. **Call/put IV matching.** Treat $F$ (equivalently the borrow/carry rate $q$) as the single unknown; for a trial $F$ compute American implied vols of the call and the put at each strike; solve $\text{IV}_{\text{call}}(K) = \text{IV}_{\text{put}}(K)$ in a vega- or spread-weighted least-squares sense (1D root-find; the IV gap is monotone in $F$).
3. **Both**, with one as the primary solver and the other as a cross-check.

Sub-questions: does the model-based path replace the corridor filter, widen it (more strikes usable once the EEP is removed), or run as a higher-fidelity tier with the corridor as the fast fallback? Is the choice a config flag or automatic by latency budget?

## Context

- Current behaviour (see [American Option Strike Filtering & Early Exercise Bounds](forward-american-strike-filtering-and-bounds.md)): corridor $|\ln(K/S)| \le \kappa\sigma_{\text{atm}}\sqrt{T}$ plus Merton bounds, no EEP removal, so the bias grows with rates, dividends and maturity.
- The previous decision rejected PDE/lattice inversion for latency. This ticket revisits that with the latency question made explicit, the findings are in [American Pricer Inventory, Accuracy & Latency Budget](forward-american-pricer-inventory-and-latency.md) (closed 2026-10-25): closed forms (Ju, BAW) recover the forward to about 6 bp in 1 to 2 microseconds per price, CRR 400 or FD 100x200 to under 2 bp at about 2 ms per price, only FD handles cash dividends, and the current parity path is off by up to 114 bp.
- Reuses the WLS regression from [Implied Forward Mathematical Formulation & WLS Objective](forward-math-wls-and-bounds-formulation.md).

## Resolution

Resolved 2026-10-25 via grilling round 1 (four questions; every answer matched the recommendation).

### 1. Outer loop and output contract

The two candidate methods share an inner step: an American implied vol per quote under a trial carry $q$. Subtracting the early-exercise premium at that vol gives European-equivalent prices; at the solution the call and put IVs at each strike agree, so both methods reach the same fixed point. They differ only in how the outer loop closes and in the output contract.

Adopted: de-Americanize each quote's bid and ask to European-equivalent prices under trial $q$ and run the existing `ImpliedForward` (`EuropeanVanilla` back-end) on them. Update $q$ by secant or fixed point until the forward is self-consistent. The synthetic interval $[F_{\text{bid}}, F_{\text{ask}}]$, WLS, outlier pruning and diagnostics are reused unchanged. The residual $\text{IV}_C(K) - \text{IV}_P(K)$ is reported as a diagnostic, since the per-quote IVs are already computed.

### 2. Tier structure

Model-based is the default tier when its inputs are present (discount factor, volatility, dividends). The corridor path in [American Option Strike Filtering & Early Exercise Bounds](forward-american-strike-filtering-and-bounds.md) is the fallback on missing inputs, non-convergence or a pricer error. The result is flagged with the tier used. How the flag is represented is a question for [American Solver API, Diagnostics & Fallback](forward-american-solver-api-diagnostics-and-fallback.md).

### 3. Accuracy levels and pricers

Numbers from [American Pricer Inventory, Accuracy & Latency Budget](forward-american-pricer-inventory-and-latency.md):

- **Fast:** closed-form only. Ju is the default, BAW the alternative; Bjerksund-Stensland 1993 is not offered as a default. About 1 ms per expiry and up to about 6 bp of forward error on synthetic quotes.
- **Accurate:** a closed-form solve followed by one CRR or FD premium correction and a re-solve. Target under 2 bp at a cost far below a full CRR/FD solve (those take 0.3 to 0.9 s). Confirmed feasible by [Hybrid Premium-Correction Prototype](forward-hybrid-premium-correction-prototype.md) (2026-10-26): one correction pass matches the full CRR/FD solve (worst 0.36 bp) at 8 to 12 ms instead of 260 to 520 ms.
- The level is a config enum. If a latency budget is exceeded the solver falls back to Fast automatically. The budget setting belongs to the API ticket.

### 4. Strike set

Use a corridor wider than the current $|\ln(K/S)| \le \kappa\sigma_{\text{atm}}\sqrt{T}$ with $\kappa = 0.5$. The ITM-side premium is largest where parity is most biased, and pricer error is largest there too, so the corridor widens moderately rather than admitting all liquid strikes. The final $\kappa$ is set against the validation fixtures (see the map's Not yet specified).

### Consequences

- Dividends and rates are inputs to the American pricer: see [American Carry, Dividend & Rate Inputs](forward-american-carry-dividend-and-rate-inputs.md), now unblocked.
- A vanilla `implied_volatility` is needed inside the loop and does not exist yet; it is part of the implementation, not a separate decision.
