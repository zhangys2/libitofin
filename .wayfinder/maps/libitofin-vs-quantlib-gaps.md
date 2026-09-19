# libitofin vs C++ QuantLib gaps

## Destination

Decide *how* to inventory and prioritize what libitofin still lacks versus C++ QuantLib (rates + equity; surface gap or oracle gap), so later sessions can pick sharp port targets without re-deriving the gap picture. This planning effort ends when that method and the first frontier of gap-decision tickets exist — not when the ports are implemented.

**Status: Destination met (2026-09-19).** Method + register + inventories + prioritization lens are in place; first gap-decision ticket is on the frontier.

## Notes

- Domain: rates + equity. Credit and inflation stay demoted fog (see `docs/oracle-coverage.md` scope note; upstream credit epic #676).
- Classification (see [Gap classification scheme](../tickets/gap-classification-scheme.md)): `has_surface` × `has_matching_oracle` (`none`|`partial`|`full`). Surface = instrument + ≥1 engine. Oracle ambition = cases named on `oracle-coverage.md` for that feature. Still missing: `has_surface = false` or `has_matching_oracle ≠ full`.
- Canonical register: [`docs/quantlib-gaps.md`](../../docs/quantlib-gaps.md) (missing only; coverage-style feature rows). `oracle-coverage.md` stays done-only and points here.
- QL inventory pin: QuantLib tag `v1.43` (fixtures-aligned).
- Prioritization (see [Prioritization lens for next ports](../tickets/prioritization-lens.md)): desk demand → close partials → dependency → greenfield. Desk demand = GH issue/epic or named ask on the gap-decision ticket.
- Standing preference: reuse QuantLib oracles as the done bar; do not invent alternate numerical truth.
- Skills: `grilling` for HITL decisions; research tickets for AFK diffs against QL + this repo. Do **not** turn this map into an execution backlog unless Notes here explicitly say so. Product work that survives prioritization should land as GitHub issues afterward; this map stays decision-only.

## Decisions so far

- Destination confirmed via grilling (2026-09-19): direction = libitofin vs QL; missing = surface **or** oracle; fence = rates+equity with credit/inflation deferred.
- [Gap classification scheme](../tickets/gap-classification-scheme.md) — dual columns `has_surface` × ternary `has_matching_oracle`; surface = E2E priceable path; full = coverage-doc ambition complete.
- [Canonical gap register](../tickets/canonical-gap-register.md) — `docs/quantlib-gaps.md` for still-missing feature rows; coverage “Not started” → pointer; GH issues after prioritization only.
- [Rates+equity oracle case inventory](../tickets/oracle-case-inventory.md) — v1.43 suite vs coverage; merged into gaps doc.
- [Rates+equity surface module inventory](../tickets/surface-module-inventory.md) — v1.43 `ql/` vs libitofin modules; merged into gaps doc.
- [Prioritization lens for next ports](../tickets/prioritization-lens.md) — weighted mix: desk demand → partials → dependency → greenfield; desk demand = GH or named ask.
- [Next rates+equity gap under the lens](../tickets/next-gap-under-lens.md) — Equity Heston analytic/FD/MC full suite (`hestonmodel.cpp`, partial); named ask 2026-09-19.
- [Which Heston oracle cases next](../tickets/which-heston-oracle-cases-next.md) — ambition = `hestonmodel.cpp` FD cached four: Barrier/Vanilla/Vanilla+divs/American vs cached @ QL tols.

## Not yet specified

- Whether Python/Go binding gaps count in a later sibling map (out of this Destination’s core fence unless promoted).
- Criteria to promote credit or inflation out of demotion.
- How suite cases not yet on an `oracle-coverage.md` ambition get promoted onto that ambition (this Heston FD-cached slice is the worked example once implemented).
- When/how to open a GitHub execution issue for further Heston ambitions (FD-cached slice implemented 2026-09-19).

## Out of scope

- Implementing ports or writing oracles (execution), until explicitly requested after a gap-decision.
- Gaps *inside* upstream C++ QuantLib (features QL lacks).
- Full credit / inflation delivery while demoted.
- Full cbindgen C ABI (`libitofin-ffi`).
- Go/Python binding parity trackers (separate docs/issues).

## Frontier (open, unblocked, unclaimed)

_(none — Heston FD-cached ambition locked; start implementation only on explicit request)_

## Blocked (not frontier)

_(none)_
