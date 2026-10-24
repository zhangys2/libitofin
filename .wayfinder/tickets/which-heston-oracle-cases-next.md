---
title: Which Heston oracle cases next
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Which Heston oracle cases next

## Question

For the selected Equity Heston partial (`hestonmodel.cpp` / coverage “Core done”), which **concrete** QuantLib cases (or small case group) should be the next oracle ambition to move `has_matching_oracle` toward `full`?

## Context

Parent decision: [Next rates+equity gap under the lens](next-gap-under-lens.md) → Equity Heston analytic/FD/MC suite. Coverage already pins some Black/DAX/cached analytic/MC cases; many BOOST cases remain. This ticket sets the coverage ambition slice before any port/oracle implementation.

## Resolution

Confirmed 2026-09-19:

- **Ambition:** `hestonmodel.cpp` FD cached cluster — all four:
  - `testFdBarrierVsCached`
  - `testFdVanillaVsCached`
  - `testFdVanillaWithDividendsVsCached`
  - `testFdAmerican`
- **Done bar:** those cases pass at QuantLib tolerances and appear as covered rows (or an updated Heston row) on `docs/oracle-coverage.md`.
- **Deferred:** analytic refs (Lewis/Kahl–Jaeckel), COS/AP/integrals, piecewise TD, and other `hestonmodel.cpp` cases — later ambitions.
- **Implemented 2026-09-19:** all four FD-cached oracles + coverage rows. Mesher: keep successful chi-square `v` and nudge non-strict `p` for Rust `LinearInterpolation` (QL `Error` catch is construction-only; not the uniform CIR fallback).
