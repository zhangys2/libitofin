---
title: Hazard recovery convention
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "Partial default, no stock jump, recover R times the pre-default bond component. At α=0, s=λ0(1-R) in the existing equity/cash split."
map: ../maps/convertible-bond-equity-credit.md
---

# Hazard recovery convention

## Question

Which recovery convention maps the constant hazard \(\lambda_0\) onto the binomial credit spread \(s\)?

## Context

[Stock-dependent hazard and constant-spread limit](cb-stock-hazard-spread-limit.md) fixed \(\lambda(S)=\lambda_0(S/S_0)^{-\alpha}\) and found three conventions. Only the first puts \(s=\lambda_0(1-R)\) into the existing equity/cash split. The other two are hazard-rate PDEs and do not meet the binomial quote exactly.

1. Partial default, no stock jump, recover \(R\) times the pre-default bond component, \(s=\lambda_0(1-R)\).
2. Face recovery and a stock jump to zero, with no exact spread identity.
3. Recover \(R\) times the pre-default convertible, so the whole claim is discounted at \(r+\lambda_0(1-R)\).

## Resolution

Resolved 2026-10-19 by agreement with option 1.

Partial default, no stock jump. Recovery is \(R\) times the pre-default bond component. At \(\alpha=0\), \(s=\lambda_0(1-R)\) enters the cash discount of the existing equity/cash split. Face recovery, a stock jump to zero, and recovery of the whole convertible are out of this destination.
