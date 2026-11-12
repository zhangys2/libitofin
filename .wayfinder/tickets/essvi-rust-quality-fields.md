---
title: eSSVI objective and quote-quality fields
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI objective and quote-quality fields

## Question

Should `CalibrationResult` also carry the band-loss objective and the quote-quality ratios, on top of the fields already agreed?

## Context

[eSSVI calibration failure result](essvi-rust-failure-result.md) keeps partial slices, posterior covariance, `CalibrationPath`, reuse and degraded ages, and the integer solver counts. It omits wall-clock timings. The C++ result also carries `objective`, the excess ratios, warm and reference loss, loss scale, and loss tolerance. Those floats were not named in that decision. This ticket decides whether the Rust result includes them.

## Resolution

Confirmed 2026-11-12.

`CalibrationResult` carries the band-loss `objective`, the excess ratios, warm and reference loss, loss scale, and loss tolerance, in addition to the fields already agreed on [eSSVI calibration failure result](essvi-rust-failure-result.md).
