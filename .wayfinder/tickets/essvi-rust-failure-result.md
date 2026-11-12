---
title: eSSVI calibration failure result
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI calibration failure result

## Question

Should a calibration that does not converge return one structured result that keeps `converged`, the error text, partial slices, and diagnostics, or should it return `QlResult` and drop that partial state?

## Context

`essvi::CalibrationResult` always carries the surface, the objective, solver counts, warm-path ages, posterior covariances, a `CalibrationPath`, and `error`. `converged` is false when validation fails or the feasible set is empty. A failed cold fit may still hold partial slices for inspection. libitofin usually surfaces failure as `QlResult` / `Err`.

The public type list is a separate ticket. This ticket decides only the failure channel and which diagnostic fields that channel must preserve.

## Resolution

Confirmed 2026-11-12.

A fit that does not converge still returns one `CalibrationResult` with `converged == false`. That result keeps the error text, any partial slices, posterior covariance, `CalibrationPath`, reuse and degraded ages, and the integer solver counts. Wall-clock timings (`timings`, `runtime_seconds`) are omitted. The band-loss objective and the quote-quality ratios were not in this decision.
