---
title: Next rates+equity gap under the lens
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Next rates+equity gap under the lens

## Question

Applying the closed [Prioritization lens for next ports](prioritization-lens.md) to [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md), which **one** feature row should be the next port/oracle target?

## Context

Lens order: desk demand → close partials → dependency → greenfield. Desk demand may be a GitHub issue/epic or a named ask recorded in this ticket’s resolution. This ticket chooses the next target; it does not implement it. After resolution, open a GitHub issue if execution is desired.

## Resolution

Confirmed 2026-09-19 by named ask:

- **Next target:** Equity — **Heston analytic/FD/MC full suite** (`docs/quantlib-gaps.md` row; QL `hestonmodel.cpp`; `has_surface=true`, `has_matching_oracle=partial`).
- **Lens fit:** desk demand (this named ask) + close-partial preference (surface already exists).
- **Sibling (not selected as the primary row):** Equity — Heston FD scheme grid (`fdheston.cpp`) remains a related partial; fold in only when it unblocks or completes the same ambition.
- **Out of this pick:** Hybrid Heston–HW, Heston SLV, piecewise TD Heston, rough Heston, FD Heston double-barrier (greenfield / other rows).

Execution of this pick landed 2026-09-19: the narrowed FD-cached cluster
([Which Heston oracle cases next](which-heston-oracle-cases-next.md)) is
implemented with coverage rows on `docs/oracle-coverage.md`. Further
`hestonmodel.cpp` cases remain later ambitions.
