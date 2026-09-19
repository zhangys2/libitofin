---
title: Rates+equity oracle case inventory
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
map: ../maps/libitofin-vs-quantlib-gaps.md
---

# Rates+equity oracle case inventory

## Question

Against a pinned QuantLib `test-suite`, which rates+equity BOOST cases (or case groups) lack a matching libitofin row in `docs/oracle-coverage.md`, once tagged per the classification scheme?

## Context

Survey snapshot: many suite files (e.g. `binaryoption`, `digitaloption`, `compoundoption`, `margrabeoption`, `everestoption`, `himalayaoption`, `pagodaoption`, `gsr`, `libormarketmodel*`, `marketmodel*`, `roughhestonmodel`, `hybridhestonhullwhiteprocess`, `amortizingbond`, `catbonds`, `equitytotalreturnswap`, …) are not represented as covered rows. This ticket produces facts for later prioritization — not a ranked backlog.

## Deliverable

Tagged still-missing rows written into [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md) (and/or a draft diff ready to merge there), using the classification scheme columns.

## Resolution

Closed 2026-09-19. QL pin `v1.43`. Oracle-side gaps (and partials where surface exists) merged into [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md) together with the surface inventory. Method: coverage doc vs `BOOST_AUTO_TEST_CASE` names on tag `v1.43`; credit/inflation/bindings excluded.
