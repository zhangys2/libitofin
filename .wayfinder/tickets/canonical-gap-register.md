---
title: Canonical gap register
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Canonical gap register

## Question

Where should the durable rates+equity gap inventory live after research tickets produce drafts?

## Context

`docs/oracle-coverage.md` is the current human map of covered oracles but its `Not started` section is too coarse. GitHub Issues is the product tracker for execution. This Wayfinder map must not become an execution backlog.

## Options (for grilling)

- **A.** Extend `docs/oracle-coverage.md` with a structured Missing section (same doc, dual purpose)
- **B.** New `docs/quantlib-gaps.md` (or similar) linked from coverage; coverage stays “done” only
- **C.** GitHub epic + child issues as the register; docs only summarize
- **D.** Keep drafts under `.wayfinder/` until prioritization, then promote winners to GitHub issues + a thin doc summary

## Resolution

Confirmed 2026-09-19:

- Canonical register: `docs/quantlib-gaps.md`.
- `docs/oracle-coverage.md` stays covered/done only; its former “Not started” section is a pointer to the gaps doc.
- Gaps doc lists **only** still-missing rows (`has_surface = false` or `has_matching_oracle ≠ full`).
- One row = one feature slice (coverage-doc style); cite QL suite files / headers in columns.
- Columns: Domain | Feature | QL surface | QL oracle(s) | `has_surface` | `has_matching_oracle` | Notes.
- GitHub Issues remain for prioritized execution after the prioritization lens — not the inventory itself.
