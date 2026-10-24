# Vanilla European FDM Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a public one-dimensional Black-Scholes finite-difference engine for European vanilla options and validate it against the analytic engine.

**Architecture:** Build a `PricingEngine` beside the existing analytic vanilla engines. The engine will create the existing Black-Scholes mesher/operator, initialize terminal payoff values on the log-spot grid, roll back with `FdmBackwardSolver` using Crank-Nicolson and optional implicit-Euler damping, then linearly interpolate the time-zero value at the process spot. Keep the first slice limited to European plain-vanilla payoffs and value-only boundary enforcement.

**Tech Stack:** Rust 2024, Cargo, `PricingEngine`, `GenericEngine`, `OneAssetOptionResults`, `GeneralizedBlackScholesProcess`, `FdmBlackScholesMesher`, `FdmBlackScholesOp`, `FdmBackwardSolver`, `DirichletBoundary`, and existing interpolation/math utilities.

---

### Task 1: Define the FDM engine API and payoff-grid helpers

**Files:**
- Create: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`
- Modify: `crates/libitofin/src/pricingengines/vanilla/mod.rs`
- Test: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`

- [ ] **Step 1: Write failing helper tests**

Add tests for a log-grid payoff helper and interpolation helper:

```rust
#[test]
fn terminal_payoff_is_evaluated_at_exp_log_grid_points() {
    let payoff = PlainVanillaPayoff::new(OptionType::Call, 100.0);
    let x = Array::from([4.0, (100.0_f64).ln(), 5.0]);
    let values = terminal_payoff(&payoff, &x);
    assert_eq!(values[1], 0.0);
    assert!(values[2] > values[0]);
}

#[test]
fn interpolation_uses_adjacent_log_grid_points() {
    let x = Array::from([0.0, 1.0, 2.0]);
    let values = Array::from([0.0, 10.0, 20.0]);
    assert!((interpolate(&x, &values, 0.25) - 2.5).abs() < 1e-14);
}
```

- [ ] **Step 2: Run the focused tests and confirm they fail**

Run:

```text
cargo test -p libitofin pricingengines::vanilla::fdmeuropeanengine -- --nocapture
```

Expected: compilation failure because the module and helpers do not exist.

- [ ] **Step 3: Implement the helpers and public engine configuration**

Define:

```rust
pub struct FdmEuropeanEngine {
    base: OneAssetOptionEngine,
    process: Shared<GeneralizedBlackScholesProcess>,
    grid_points: Size,
    time_steps: Size,
    damping_steps: Size,
    mesher_eps: Real,
    mesher_scale_factor: Real,
}
```

Add `new(process)`, `with_grid(process, grid_points, time_steps)`, and
`with_damping_steps`. Reject zero grid points, zero time steps, and non-positive
spot values with `QlResult` errors. Export `FdmEuropeanEngine` from the vanilla
pricing-engine module.

Implement `terminal_payoff` by evaluating `payoff.value(x.exp())`, and
`interpolate` by clamping to the endpoint values and linearly interpolating
between adjacent grid coordinates.

- [ ] **Step 4: Run the helper tests**

Run the focused command from Step 2. Expected: PASS.

- [ ] **Step 5: Commit the API/helper slice**

```text
git add crates/libitofin/src/pricingengines/vanilla
git commit -m "feat: add vanilla FDM engine scaffolding"
```

### Task 2: Implement rollback and time-aware boundaries

**Files:**
- Modify: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`
- Modify: `crates/libitofin/src/methods/finitedifferences/boundaryconditions.rs`
- Test: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`

- [ ] **Step 1: Add a failing boundary-time test**

Add a test that constructs a time-aware call boundary, sets its current time,
and verifies the lower and upper values for a known spot, strike, rate, and
remaining time. The boundary must update only its selected edge.

- [ ] **Step 2: Run the test to confirm the missing API**

Run the focused pricing-engine test command. Expected: compilation failure for
the time-aware boundary constructor or setter.

- [ ] **Step 3: Add a time-aware Dirichlet boundary**

Add `TimeDependentDirichletBoundary` with:

```rust
pub fn new(side: BoundarySide, value: impl Fn(Time) -> Real + 'static) -> Self
pub fn set_time(&self, t: Time)
```

Store the current time behind the repository’s existing interior-mutability
pattern. Apply the computed value in `apply_after_applying` and
`apply_after_solving`; leave operator hooks as no-ops because the current
`FdmLinearOp` abstraction has no row-mutation API.

- [ ] **Step 4: Implement the engine calculation**

In `PricingEngine::calculate`:

1. Validate European exercise and plain-vanilla payoff.
2. Read maturity from the exercise and compute time to maturity from the
   process.
3. Build the log-spot mesher with `fdm_black_scholes_mesher`.
4. Build `FdmBlackScholesOp` over the mesher.
5. Evaluate terminal payoff values over `mesher.locations(0)`.
6. Create time-dependent call/put asymptotic Dirichlet boundaries using the
   process discount curve and payoff strike.
7. Construct `FdmBackwardSolver` with `FdmSchemeDesc::crank_nicolson()`.
8. Roll back from maturity to zero, using the configured damping steps.
9. Interpolate at `process.x0()?.ln()` and store the result in
   `OneAssetOptionResults.instrument.value`.

Surface all curve, exercise, payoff, mesher, operator, and rollback errors.
Do not add broad catches or silently substitute analytic values.

- [ ] **Step 5: Add an engine-level rollback test**

Use a flat Black-Scholes process and a call payoff. Assert that the calculated
result is finite, positive, and changes when the grid/time-step configuration
changes.

- [ ] **Step 6: Run focused tests**

```text
cargo test -p libitofin pricingengines::vanilla::fdmeuropeanengine -- --nocapture
```

Expected: PASS.

- [ ] **Step 7: Commit the calculation slice**

```text
git add crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs crates/libitofin/src/methods/finitedifferences/boundaryconditions.rs
git commit -m "feat: implement European vanilla FDM rollback"
```

### Task 3: Add end-to-end analytic parity regressions

**Files:**
- Modify: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`
- Test: `crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs`

- [ ] **Step 1: Add call and put parity tests**

Build the same flat process and European option arguments for both
`AnalyticEuropeanEngine` and `FdmEuropeanEngine`. With at least 201 grid points,
400 rollback steps, and four damping steps, assert the FDM NPV is within the
documented finite-difference tolerance of the analytic NPV for both call and
put options.

- [ ] **Step 2: Run the parity tests**

```text
cargo test -p libitofin pricingengines::vanilla::fdmeuropeanengine -- --nocapture
```

Expected: PASS with both option types and no NaN/Inf values.

- [ ] **Step 3: Commit the parity regressions**

```text
git add crates/libitofin/src/pricingengines/vanilla/fdmeuropeanengine.rs
git commit -m "test: compare vanilla FDM with analytic pricing"
```

### Task 4: Validate, document, and publish

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-31-vanilla-fdm-engine-design.md` only if implementation behavior differs from the approved scope.

- [ ] **Step 1: Update the status documentation**

Replace the finite-difference-pending wording with a precise statement that
one-dimensional European vanilla FDM pricing is available, while Heston FDM,
early exercise, local volatility, and multidimensional schemes remain outside
the current coverage.

- [ ] **Step 2: Run targeted and full validation**

```text
cargo test -p libitofin pricingengines::vanilla::fdmeuropeanengine -- --nocapture
cargo test -p libitofin methods:: -- --nocapture
cargo test --workspace --locked
cargo build --workspace --locked
```

Expected: all commands exit successfully.

- [ ] **Step 3: Commit documentation**

```text
git add README.md
git commit -m "docs: document vanilla FDM coverage"
```

- [ ] **Step 4: Push the verified commits**

```text
git push origin HEAD:main
```
