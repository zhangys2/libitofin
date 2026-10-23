---
title: Hazard-rate acceptance oracle
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "Andersen–Buffum Table 2 is not an oracle: it uses face recovery and a stock jump to zero. The hazard check is mesh self-convergence."
map: ../maps/convertible-bond-equity-credit.md
---

# Hazard-rate acceptance oracle

## Question

Once the hazard and recovery convention are fixed, is Andersen and Buffum Table 2 (naïve column, to 0.1) an acceptance oracle for \(\lambda(S)\propto S^{-\alpha}\), or is mesh self-convergence the only hazard check?

## Context

That column prices two contracts with voluntary conversion, a hard call, dated puts, and fixed coupons. The intensity is \(\lambda(S)=s(S(0)/S)^{p}\) with recovery of 40% of notional and a stock jump to 0. It is an oracle only if [Hazard recovery convention](cb-hazard-recovery-convention.md) uses face recovery and a stock jump to zero. Ayache–Forsyth–Vetzal Figure 5 uses the same exponent family and prints no price.

## Resolution

Closed 2026-10-19 because [Hazard recovery convention](cb-hazard-recovery-convention.md) selected partial default with no stock jump. Andersen and Buffum Table 2 uses face recovery and a stock jump to zero, so it is not an acceptance oracle. No published price matches this convention. The hazard check is mesh self-convergence.
