---
title: Prioritization lens for next ports
type: Grilling (HITL)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Prioritization lens for next ports

## Question

Once inventories exist in the canonical register, what rule ranks the next port/oracle targets?

## Context

Without a lens, the inventories become an undifferentiated feature backlog (Wayfinder anti-pattern). Desk need, dependency order, oracle density, and slice-completion are competing lenses. This decision enables later *gap-decision* tickets; it does not itself schedule work.

Inventories now live in `docs/quantlib-gaps.md` (QL `v1.43`).

## Options (for grilling)

- **A.** Desk / issue demand first; inventories only constrain feasibility
- **B.** Dependency order (foundations before exotics; shared engines before one-offs)
- **C.** Close partial coverage rows before new surfaces
- **D.** Weighted mix (state weights explicitly in the resolution)

## Resolution

Confirmed 2026-09-19. Weighted mix (D), order high → low:

1. **Desk demand** — open GitHub issue/epic **or** explicit named ask recorded on the gap-decision ticket.
2. **Close partials** — prefer `has_matching_oracle=partial` over greenfield `has_surface=false`.
3. **Dependency** — shared foundations / engines before one-off exotics.
4. **Greenfield surface** — new `has_surface=false` last among otherwise equal candidates.

Ranks candidates from `docs/quantlib-gaps.md` only; does not schedule or implement ports.
