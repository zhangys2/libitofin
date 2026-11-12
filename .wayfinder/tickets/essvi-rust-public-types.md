---
title: eSSVI public Rust types
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI public Rust types

## Question

Which public Rust types should the implementation plan specify for the evaluator and the calibrator, and should any of them implement `SmileSection` or `BlackVolTermStructure`?

## Context

`essvi::Slice` is `(expiry, theta, psi, rho)`. `essvi::Surface` interpolates slices with the Corbetta rule and evaluates total variance and implied volatility at `(expiry, k)` where `k = log(strike / forward)`. `calibrate` and `calibrate_warm` take `Quote` values and `WarmOptions`, and return `CalibrationResult`.

libitofin prices through `SmileSection` (one expiry, strike argument) and `BlackVolTermStructure`. The C++ QuantLib adapter in `cpp/essvi_ql.*` is out of scope for this map. The plan still has to say whether the Rust surface is a standalone total-variance type, or whether it also satisfies an existing volatility trait so pricing engines can call it.

This ticket chooses the public types and the trait boundary. It does not choose file layout, and it does not choose how a failed fit is returned.

## Resolution

Confirmed 2026-11-12.

The plan specifies standalone `Slice`, `Surface`, `Quote`, `WarmOptions`, and `CalibrationResult`. `SmileSection` and `BlackVolTermStructure` stay out of this plan.
