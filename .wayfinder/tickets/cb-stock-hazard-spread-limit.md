---
title: Stock-dependent hazard and constant-spread limit
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
resolution: "Hazard is λ(S)=λ0(S/S0)^{-α} for S>0, S0>0, with α=0 the constant hazard and α>0 the direction both papers treat as realistic. The map from (λ0, R) to the binomial spread s is not agreed. See the recovery-convention grilling ticket."
map: ../maps/convertible-bond-equity-credit.md
---

# Stock-dependent hazard and constant-spread limit

## Question

What exact hazard \(\lambda(S)=\lambda_0(S/S_0)^{-\alpha}\), recovery assumption, and \(\alpha=0\) map onto the binomial credit spread \(s\) should the engine use?

## Context

The binomial engine takes a constant spread quote, not a hazard. The destination requires the power-law hazard to sit beside that spread and to recover a constant hazard at \(\alpha=0\), with an explicit meeting point so the existing oracles still check the flat case. If the literature disagrees on the map from \((\lambda, R)\) to \(s\), record the disagreement and open a grilling ticket rather than picking a formula.

## Resolution

Resolved 2026-10-19 by [Hazard and spread limit](9832eb61-ea0e-4fb1-a6ec-1a5d7b5d69fa).

The hazard is

\[
\lambda(S)=\lambda_0\left(\frac{S}{S_0}\right)^{-\alpha},\qquad S>0,\; S_0>0.
\]

\(\alpha=0\) is the constant hazard \(\lambda_0\). \(\alpha>0\) is the direction Ayache–Forsyth–Vetzal and Andersen–Buffum both treat as realistic: hazard rises as the stock falls. \(S_0\) is the level at which the hazard equals \(\lambda_0\).

The map from \((\lambda_0, R)\) to the binomial spread \(s\) is not shared. Three conventions:

1. Partial default, no stock jump, recover \(R\) times the pre-default bond component. Then \(s=\lambda_0(1-R)\) and the equity/cash split matches the existing tree.
2. Face recovery and a stock jump to zero. The discount is \(r+\lambda\) plus a recovery source. The spread identity is only approximate.
3. Recover \(R\) times the pre-default convertible, so the whole claim, including conversion, is discounted at \(r+\lambda_0(1-R)\).

The choice is [Hazard recovery convention](cb-hazard-recovery-convention.md). No formula was written into the engine.
