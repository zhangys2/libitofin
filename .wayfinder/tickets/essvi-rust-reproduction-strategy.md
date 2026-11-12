---
title: eSSVI reproduction strategy
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI reproduction strategy

## Question

Must the implementation plan specify a transliteration of the C++ rho-window search, psi bounds, calendar-headroom retry, and warm-update state machine, or only the outputs the in-scope tests assert?

## Context

Cold `fitSlice` searches rho on fixed windows, keeps psi inside the Roger-Lee and Martini–Mingone bounds, anchors theta near the ATM quote, tries a few theta multipliers, and retries with calendar headroom. `calibrate_warm` adds a local refinement, a restricted cold fallback, a full cold fallback, and a covariance update in `(log theta, atanh rho, log psi)`. This is not an NLopt or SLSQP call.

Unblocked 2026-11-12 by [eSSVI in-scope C++ test assertions](essvi-rust-test-assertions.md). Those tests lock exact slice values and also exact path, ages, solver counts, and bitwise covariance. They also lock `runtime_seconds` only as a finite value, plus millisecond budgets in the scenario file, and CSV helper behavior already left out of the Rust API. The decision here is what the plan must specify, not the Rust code.

## Resolution

Confirmed 2026-11-12.

The implementation plan specifies a transliteration of the rho-window search, psi bounds, calendar-headroom retry, and warm-update state machine. The Rust acceptance bar is the exact slice values, path, ages, solver counts, covariance, and quote-quality assertions from the in-scope tests. Scenario millisecond budgets and the CSV helper assertions are not part of that bar. Whether the rho search itself runs sequentially or on the C++ worker pool is a separate ticket.
