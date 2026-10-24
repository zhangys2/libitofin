---
title: Gap classification scheme
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Gap classification scheme

## Question

What fixed tag set should every rates+equity gap carry so surface-absent, oracle-absent, and partial-slice cases stay distinguishable in the inventory?

## Context

Destination already counts “missing” as surface **or** oracle. Survey shows many `oracle-coverage.md` rows are partial (“slice”, “identity-verified”, “Core done”) while `Not started` lists only four coarse domains. Without tags, research inventories will collapse unlike cases.

## Options (for grilling)

- **A.** Two tags only: `surface-gap` | `oracle-gap` (partial counts as oracle-gap)
- **B.** Three tags: `surface-gap` | `oracle-gap` | `partial-oracle` (partial/identity/core-done tracked separately)
- **C.** Dual boolean columns: `has_surface` × `has_matching_oracle` (derive status; partial = surface yes, oracle incomplete)

## Resolution

Confirmed 2026-09-19:

- Dual columns: `has_surface` × `has_matching_oracle` (not a single enum tag).
- `has_surface`: `true` iff an end-to-end priceable path exists (instrument + ≥1 engine wired). Bindings do not count.
- `has_matching_oracle`: ternary `none` | `partial` | `full`.
  - `full` = every case named on the `oracle-coverage.md` ambition for that feature.
  - `partial` = ≥1 such case covered, not all.
  - `none` = zero covered.
  - Suite cases not yet on the ambition stay fog until added.
- Derived “still missing”: `has_surface = false` **or** `has_matching_oracle ≠ full`.
