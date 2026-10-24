---
title: Default-jump term in the gamma identity
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
resolution: "The cash-gamma function returns only the no-default wealth change. There is no realized default-jump term."
map: ../maps/convertible-bond-equity-credit.md
---

# Default-jump term in the gamma identity

## Question

Does the library cash-gamma function return a realized default-jump term, or only the no-default wealth change whose credit carry already sits inside grid theta?

## Context

[Cash-gamma scalping identity](cb-cash-gamma-identity.md) fixes the no-default identity and leaves the jump out of that formula. Ayache–Forsyth–Vetzal put the jump inside the hedge increment. The Tsiveriotis–Fernandes tree has no jump. The choice waits on whether [Hazard recovery convention](cb-hazard-recovery-convention.md) includes a default jump.

## Resolution

Closed 2026-10-19 because [Hazard recovery convention](cb-hazard-recovery-convention.md) selected partial default with no stock jump. The cash-gamma function returns only the no-default wealth change. Credit carry stays inside grid theta. A realized default-jump term is out of this destination.
