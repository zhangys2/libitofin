---
title: Rates+equity surface module inventory
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Rates+equity surface module inventory

## Question

Which QuantLib rates+equity public surfaces under `ql/instruments`, `ql/pricingengines`, and `ql/models` (plus closely related processes/indexes) have no corresponding libitofin module/API, once tagged per the classification scheme?

## Context

Oracle absence ≠ surface absence (and vice versa). Equity/rates engines already present in Rust may still lack oracles; some QL headers may have no Rust counterpart at all (e.g. GSR/LMM/market models called out in coverage `Not started`).

## Deliverable

Tagged still-missing surface rows written into [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md) (merged with oracle inventory rows where the same feature slice applies).

## Resolution

Closed 2026-09-19. QL pin `v1.43`. Surface-absent and surface-partial rows merged into [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md) with the oracle inventory. Method: `gh api` listing of `ql/{instruments,pricingengines,models}` (+ experimental/legacy) vs `crates/libitofin/src/{instruments,pricingengines,models,processes}`; credit/inflation/bindings excluded.
