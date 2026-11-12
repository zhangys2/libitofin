---
title: eSSVI quote input boundary
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI quote input boundary

## Question

Should the Rust library accept only in-memory quotes, leaving `readQuotes`, `writeParameters`, and `writeDiagnostics` out of the port?

## Context

The C++ header exposes CSV helpers beside `calibrate` and `calibrate_warm`. The Python package also builds `Quotes` in memory. The charter leaves the CVI scoring scripts and the benchmark harness in `ql-essvi`. Those scripts are the callers of the CSV writers. This ticket decides whether the Rust API still includes those file helpers.

## Resolution

Confirmed 2026-11-12.

The Rust API takes in-memory quotes only. `readQuotes`, `writeParameters`, and `writeDiagnostics` stay in the C++ tree.
