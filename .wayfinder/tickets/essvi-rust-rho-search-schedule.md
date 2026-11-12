---
title: eSSVI rho-search schedule
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: "Grok (grilling session 2026-11-12)"
map: ../maps/essvi-rust-port.md
---

# eSSVI rho-search schedule

## Question

Should the plan specify a sequential rho search, with the same windows and the same total bound-call accounting, and leave the C++ worker pool out of the port?

## Context

[eSSVI reproduction strategy](essvi-rust-reproduction-strategy.md) transliterates the rho-window search. In the C++ calibrator, `RhoWorkerPool` splits that search across threads and sums each thread's Martini–Mingone bound calls into one total. The in-scope tests lock that total and require two repeats to match bitwise. They do not lock a thread count. The scenario file's millisecond budgets are outside the Rust acceptance bar. This ticket decides only the schedule.

## Resolution

Confirmed 2026-11-12.

The plan specifies a sequential rho search with the C++ windows and the same total bound-call accounting. The C++ worker pool stays out of the port.
