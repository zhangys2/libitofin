# FDM Schemes and Boundary Conditions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit Euler, Crank-Nicolson, and FDM-native Dirichlet/Neumann boundary support.

**Architecture:** Reuse the existing `Scheme` trait, `BoundaryConditionSchemeHelper`,
and `FdmLinearOpComposite` contracts. Crank-Nicolson delegates the existing
Douglas splitting with `theta = 0.5` for the currently supported one-dimensional
operator path; explicit Euler is a direct operator application. Boundary
conditions remain separate from deprecated legacy tridiagonal conditions.

**Tech Stack:** Rust 2024, Cargo, `Array`, `FdmLinearOp`, `FdmLinearOpComposite`,
`FdmBackwardSolver`, existing finite-difference test fixtures.

---

### Task 1: Adding explicit Euler and Crank-Nicolson schemes

**Files:**
- Create: `crates/libitofin/src/methods/finitedifferences/schemes/expliciteulerscheme.rs`
- Create: `crates/libitofin/src/methods/finitedifferences/schemes/cranknicolsonscheme.rs`
- Modify: `crates/libitofin/src/methods/finitedifferences/schemes/mod.rs`
- Modify: `crates/libitofin/src/methods/finitedifferences/solvers/fdmschemedesc.rs`
- Modify: `crates/libitofin/src/methods/finitedifferences/solvers/fdmbackwardsolver.rs`

- [ ] **Step 1: Add failing scheme descriptor tests**

Add factories and tests asserting:

```rust
assert_eq!(FdmSchemeDesc::explicit_euler().scheme_type, FdmSchemeType::ExplicitEuler);
assert_eq!(FdmSchemeDesc::crank_nicolson().scheme_type, FdmSchemeType::CrankNicolson);
assert_eq!(FdmSchemeDesc::crank_nicolson().theta, 0.5);
```

Run:

```text
cargo test -p libitofin solvers::fdmschemedesc -- --nocapture
```

Expected: compilation failure because the factories do not exist.

- [ ] **Step 2: Add the explicit Euler test**

Create a one-direction diagonal-operator fixture using the existing
`scaled_composite` helper. Set `dt = 0.1`, use coefficient `0.3`, and assert
that one step transforms `u` into:

```rust
&u * (1.0 + 0.1 * 0.3)
```

Also assert that stepping before `set_step` returns an error.

- [ ] **Step 3: Implement explicit Euler**

Implement `ExplicitEulerScheme { dt, map, bc_set }` and its `Scheme` methods:

```rust
let start = (t - dt).max(0.0);
map.set_time(start, t)?;
bc_set.set_time(start);
bc_set.apply_before_applying(&mut *map);
let mut next = a + &(dt * &map.apply(a));
bc_set.apply_after_applying(&mut next);
bc_set.apply_after_solving(&mut next);
*a = next;
```

Reject unset timesteps and negative-time steps using the same error style as
`DouglasScheme` and `ImplicitEulerScheme`.

- [ ] **Step 4: Implement Crank-Nicolson**

Implement `CrankNicolsonScheme` as a thin wrapper around the existing Douglas
algorithm with `theta = 0.5`, preserving the `Scheme` interface and boundary
helper call ordering. Do not duplicate the split-step arithmetic.

- [ ] **Step 5: Wire scheme exports and solver dispatch**

Export both schemes from `schemes/mod.rs`; add `explicit_euler()` and
`crank_nicolson()` factories to `FdmSchemeDesc`; add `ExplicitEuler` and
`CrankNicolson` match arms to `FdmBackwardSolver::rollback`.

- [ ] **Step 6: Run focused scheme tests**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods::finitedifferences::schemes -- --nocapture
```

Expected: all existing and new scheme tests pass.

- [ ] **Step 7: Commit the scheme slice**

```text
git add crates/libitofin/src/methods/finitedifferences
git commit -m "feat: add explicit Euler and Crank-Nicolson FDM schemes"
```

### Task 2: Adding FDM-native Dirichlet and Neumann conditions

**Files:**
- Create: `crates/libitofin/src/methods/finitedifferences/boundaryconditions.rs`
- Modify: `crates/libitofin/src/methods/finitedifferences/mod.rs`
- Test: `crates/libitofin/src/methods/finitedifferences/boundaryconditions.rs`

- [ ] **Step 1: Add failing boundary API tests**

Define tests for a lower or upper boundary with a constant value and a
one-dimensional tridiagonal fixture. The Dirichlet test must assert the edge
grid value equals the prescribed value after `apply_after_solving`; the
Neumann test must assert the edge first difference equals the prescribed slope
after `apply_after_solving`.

- [ ] **Step 2: Implement Dirichlet condition**

Add a public `DirichletBoundary` holding `BoundarySide`, direction, and a
time-independent value. Its `apply_after_applying`, `apply_before_solving`, and
`apply_after_solving` methods must enforce the selected edge without changing
the opposite edge. `set_time` is a no-op for this constant condition.

- [ ] **Step 3: Implement Neumann condition**

Add a public `NeumannBoundary` holding `BoundarySide`, direction, and slope.
For a uniform one-dimensional grid, enforce:

```rust
upper_value - previous_value = slope * dx
previous_value - lower_value = slope * dx
```

Use the operator’s existing boundary hooks where available and return explicit
errors when the selected direction or array size cannot be supported.

- [ ] **Step 4: Add boundary-set integration tests**

Run each condition through `FdmBackwardSolver` with the new
Crank-Nicolson scheme and assert the boundary remains fixed across rollback
steps while interior values evolve.

- [ ] **Step 5: Export and run focused boundary tests**

Export the new types from `finitedifferences/mod.rs`, then run:

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods::finitedifferences -- --nocapture
```

Expected: all finite-difference tests pass.

- [ ] **Step 6: Commit the boundary slice**

```text
git add crates/libitofin/src/methods/finitedifferences
git commit -m "feat: add FDM Dirichlet and Neumann boundaries"
```

### Task 3: Full validation and publication

**Files:**
- No additional source files.

- [ ] **Step 1: Run methods tests**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods:: -- --nocapture
```

- [ ] **Step 2: Build the workspace**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo build --workspace --locked
```

- [ ] **Step 3: Push the verified commits**

```text
git push origin HEAD:main
```
