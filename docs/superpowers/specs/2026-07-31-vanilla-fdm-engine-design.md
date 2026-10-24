# Vanilla European FDM Engine Design

**Goal:** Add a public one-dimensional Black-Scholes finite-difference engine
for European vanilla options and validate it against the existing analytic
engine.

## Scope

The first engine slice supports European `PlainVanillaPayoff` options on a
`GeneralizedBlackScholesProcess`. It uses the existing `FdmBlackScholesMesher`,
`FdmBlackScholesOp`, `FdmBackwardSolver`, and Crank-Nicolson scheme. The engine
will initialize the terminal payoff on the log-spot grid, roll it back to time
zero, and interpolate the result at the current spot.

The slice includes optional initial implicit-Euler damping, configurable grid
size, and configurable rollback steps. It does not include early exercise,
discrete dividends, local volatility, Heston FDM, multidimensional grids, or
finite-difference Greeks.

## Architecture

Add `FdmEuropeanEngine` under `pricingengines/vanilla`. The engine follows the
existing `PricingEngine` and `GenericEngine` contracts, registers with the
process observable, validates European/plain-vanilla arguments, and stores the
computed NPV in `OneAssetOptionResults`.

The calculation builds a log-spot mesher and Black-Scholes operator, creates
Dirichlet edge conditions from the analytical asymptotic boundary values, and
uses `FdmBackwardSolver` with Crank-Nicolson. The terminal array is evaluated
directly from the payoff at each grid spot. The time-zero result is obtained by
linear interpolation over the mesher locations.

## Boundary behavior

For a call, the lower boundary is zero and the upper boundary is
`S - K * exp(-r * tau)`; for a put, the lower boundary is
`K * exp(-r * tau) - S` and the upper boundary is zero. The boundary value is
recomputed at each rollback time through a small time-aware boundary type.
The existing FDM boundary trait remains value-oriented; matrix-row mutation is
out of scope for this engine slice.

## Validation

Add unit tests for terminal payoff construction and interpolation, plus an
end-to-end call and put regression comparing FDM prices with the analytic
European engine within a documented grid tolerance. Run the focused pricing
tests, all methods tests, the workspace test suite, and the workspace build.
