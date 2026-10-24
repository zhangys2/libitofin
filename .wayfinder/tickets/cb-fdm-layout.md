---
title: Finite-difference layout for the coupled system
type: Research (AFK)
status: closed
blocked_by: []
claimed_by: null
resolution: "Two successive one-dimensional solves on one shared log-spot mesh, then one joint projection. E uses FdmBlackScholesOp at r. C reuses that operator on a process with rates (r+s, q+s). Douglas θ=0.5. No 2N unknown and no blended discount."
map: ../maps/convertible-bond-equity-credit.md
---

# Finite-difference layout for the coupled system

## Question

How should the two fields \(E\) and \(C\) from [Tsiveriotis–Fernandes coupled PDE statement](cb-tf-coupled-pde-statement.md) be represented on the existing one-dimensional finite-difference stack (`fdm1dimsolver`, the Black–Scholes operator, mesh, and time scheme)?

## Context

The continuous problem is settled: two unknowns, equity discounted at \(r\) and cash at \(r+s\). The blended one-value equation in `tf_stepback` is not the system to discretize. The stack already solves one-dimensional Black–Scholes problems with a single unknown. Coefficients are the flat \(r\), \(q\), and \(\sigma\) freeze from [Flat versus term-structure coefficients](cb-flat-versus-term-structure.md). This ticket decides the field layout only. It does not implement the solver.

## Resolution

Resolved 2026-10-19 by [FDM layout for two fields](d6448491-d4f3-4b1b-8aed-f31815bb9f5c).

Two successive one-dimensional solves on one shared `fdm_black_scholes_mesher` (log-spot, concentrated at redemption/\(\kappa\)), then one joint projection. Do not pack \(E\) and \(C\) into one unknown, and do not run two maturity-to-zero `Fdm1DimSolver` instances.

- \(E\) reuses `FdmBlackScholesOp` at discount \(r\).
- \(C\) reuses the same operator on a second flat process with rates \((r+s, q+s)\) and the same \(\sigma\), so the drift stays \(r-q\) and the reaction is \(-(r+s)\).
- Each field has its own `FdmBackwardSolver`. The scheme is Douglas with \(\theta=0.5\).
- Stopping times cut the grid at call, put, coupon, and conversion dates. American conversion cuts every layer.
- After both fields arrive, one projection applies, in order: call, put, coupon added to \(C\) only, then conversion.
- Boundaries: \(S\to 0\) gives \(E=0\) and \(C\) the straight bond at \(r+s\); \(S\to\infty\) gives \(C=0\) and \(E=\kappa S\). The lower Dirichlet for \(C\) excludes a coupon paid on the layer being projected.

Not built: a second mesher, a length-\(2N\) unknown, `tf_stepback`, or a conversion probability inside the time step.
