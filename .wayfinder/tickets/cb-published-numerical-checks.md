---
title: Published numerical checks
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
resolution: "Inventory only. Flat-spread bar is the QuantLib zero/fixed testBond tolerances and the zero-spread testOption. No recoverable 1998 price. AFV TF column is 123.97 but q is unprinted. Andersen–Buffum Table 2 naïve column is the reproducible λ(S)∝S^{-α} price, at 0.1, with 40% notional recovery and a stock jump to 0."
map: ../maps/convertible-bond-equity-credit.md
---

# Published numerical checks

## Question

Which published numerical tables or closed-form limits exist for the flat-spread Tsiveriotis–Fernandes system and for \(\lambda(S)\propto S^{-\alpha}\) that can sit beside QuantLib binomial agreement and mesh self-convergence?

## Context

QuantLib’s `convertiblebonds.cpp` oracles cover the constant-spread binomial engine only. They do not cover a stock-dependent hazard or a gamma-scalping identity. Prefer cases whose contract is voluntary conversion, issuer call, holder put, and fixed coupons, with flat \(r\), \(q\), and \(\sigma\). This ticket inventories candidate checks and the assumptions each one uses. Choosing which case is the acceptance oracle is [Flat-spread acceptance bar](cb-tf-acceptance-bar.md) and [Hazard-rate acceptance oracle](cb-hazard-acceptance-oracle.md).

## Resolution

Resolved 2026-10-19 by [Published numerical checks](172e0811-033a-4902-a923-aadaf41ad68d). Inventory only; no acceptance oracle was selected.

- Flat-spread binomial bar: QuantLib `testBond` zero and fixed legs (tolerances 0.01 and 0.02 at 1001 steps) and `testOption` (tolerance 0.05 at 2001 steps, spread 0). The floating leg is outside this engine. The at-the-money inequality is not a price.
- Tsiveriotis–Fernandes (1998) has no publicly recoverable price.
- Ayache–Forsyth–Vetzal (2003) Tables 2–4, TF column, partially implicit, \(S=100\): 123.97050 at 6400 nodes. Call, put, and fixed coupons match this engine. Dividend yield \(q\) is not printed.
- The same paper’s Figure 5 uses \(\lambda(S)\propto S^{-\alpha}\) and prints no price.
- Andersen and Buffum Table 2, naïve column, is the reproducible power-law price, to 0.1. It uses recovery of 40% of notional and a stock jump to 0. The calibrated column is time-dependent and outside the flat-coefficient fence.
