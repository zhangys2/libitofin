# Methods Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified finite-difference, lattice, and Monte Carlo method coverage while preserving the existing public APIs.

**Architecture:** Reuse the existing `FdmBlackScholesOp`, `UniformGridMesher`, `FdmBackwardSolver`, `TrinomialTree`, and `PathGenerator` abstractions. Add regression fixtures in the owning modules first; only modify method implementation files when a focused regression demonstrates missing behavior.

**Tech Stack:** Rust, Cargo, existing `libitofin` finite-difference, lattice, Monte Carlo, stochastic-process, and Black-Scholes APIs.

---

### Task 1: Establish a finite-difference European-option regression

**Files:**
- Modify: `crates/libitofin/src/methods/finitedifferences/solvers/fdmbackwardsolver.rs` (only if the regression exposes solver glue)
- Modify: `crates/libitofin/src/methods/finitedifferences/solvers/fdmschemedesc.rs` (only if scheme selection is incomplete)
- Modify: `crates/libitofin/src/methods/finitedifferences/schemes/douglasscheme.rs` (only if Douglas stepping is incomplete)
- Modify: `crates/libitofin/src/methods/finitedifferences/schemes/impliciteulerscheme.rs` (only if implicit stepping is incomplete)
- Modify: `crates/libitofin/src/methods/finitedifferences/operators/fdmblackscholesop.rs` (only if operator setup is incomplete)
- Test: `crates/libitofin/src/methods/finitedifferences/` module tests

- [ ] **Step 1: Identify the public constructor path**

Use the existing symbols:

```rust
let layout = shared(FdmLinearOpLayout::new(vec![grid_points]));
let mesher: Shared<dyn FdmMesher> =
    shared(UniformGridMesher::new(layout, &[(log_s_min, log_s_max)]).unwrap());
let process = GeneralizedBlackScholesProcess::new(
    spot_handle,
    dividend_curve,
    risk_free_curve,
    volatility_curve,
);
let op = FdmBlackScholesOp::new(mesher, &process, strike, 0)?;
let mut solver = FdmBackwardSolver::new(
    shared_mut(op),
    Vec::new(),
    None,
    FdmSchemeDesc::douglas(),
);
```

Use a payoff array over the mesher’s `locations(0)` and roll it from `maturity`
to zero. Keep the fixture one-dimensional and use flat curves and volatility so
the expected value is the analytic Black-Scholes call price.

- [ ] **Step 2: Add the failing oracle test**

Add a test in the finite-difference module that:

1. builds a 101-point log-price grid spanning `ln(20)` through `ln(400)`;
2. initializes `max(spot - strike, 0)` at maturity;
3. calls `FdmBackwardSolver::rollback` from `1.0` to `0.0` with 100 steps and
   a small damping segment;
4. interpolates the grid value at `spot = 100`;
5. compares it with the existing Black-Scholes analytic formula using an
   absolute tolerance of `0.15`.

The test must also assert the result is finite and positive.

- [ ] **Step 3: Run only the new regression**

Run:

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods::finitedifferences -- --nocapture
```

Expected: the new test either passes immediately or fails with a concrete
operator/scheme/solver error. Do not broaden the implementation before seeing
that failure.

- [ ] **Step 4: Implement the smallest exposed gap**

If the test fails, fix only the failing method path using existing contracts:

- keep `FdmBlackScholesOp::set_time` responsible for curve coefficients;
- keep `FdmBackwardSolver::rollback` responsible for time segmentation;
- keep schemes responsible for applying or solving one timestep;
- propagate `QlResult` errors rather than replacing them with defaults.

- [ ] **Step 5: Re-run the focused finite-difference tests**

Run the same command and require all existing finite-difference tests plus the
new oracle to pass.

- [ ] **Step 6: Commit the finite-difference slice**

```text
git add crates/libitofin/src/methods/finitedifferences
git commit -m "test: cover finite-difference European pricing"
```

### Task 2: Add a trinomial-tree recombination regression

**Files:**
- Modify: `crates/libitofin/src/methods/lattices/trinomialtree.rs`
- Test: `crates/libitofin/src/methods/lattices/trinomialtree.rs`

- [ ] **Step 1: Build a deterministic process fixture**

Use an existing `StochasticProcess1D` implementation with constant variance and
a regular `TimeGrid`. Reuse the process and grid construction patterns already
used by the `TrinomialTree` tests.

- [ ] **Step 2: Add the regression**

For each node on the first non-root slice, collect the descendants reached by
the three branches. Assert that:

```rust
assert!(tree.descendant(0, node, 0) <= tree.descendant(0, node, 1));
assert!(tree.descendant(0, node, 1) <= tree.descendant(0, node, 2));
assert!(tree.probability(0, node, 0).is_finite());
assert!(tree.probability(0, node, 1).is_finite());
assert!(tree.probability(0, node, 2).is_finite());
```

Then assert the middle descendant from adjacent parent nodes is shared, proving
recombination rather than merely checking monotonic indices.

- [ ] **Step 3: Run the lattice tests**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods::lattices -- --nocapture
```

Expected: all lattice tests pass. If the invariant fails, correct only the
branch-index calculation in `TrinomialTree`; do not alter probability formulas
without a failing numerical oracle.

- [ ] **Step 4: Commit the lattice slice**

```text
git add crates/libitofin/src/methods/lattices/trinomialtree.rs
git commit -m "test: cover trinomial tree recombination"
```

### Task 3: Add seeded Monte Carlo reproducibility coverage

**Files:**
- Modify: `crates/libitofin/src/methods/montecarlo/pathgenerator.rs`
- Test: `crates/libitofin/src/methods/montecarlo/pathgenerator.rs`

- [ ] **Step 1: Reuse the existing seeded fixture**

Use the existing `BlackScholesMertonProcess`, `PseudoRandom::make_sequence_generator`,
and `PathGenerator::new` helpers already present in `pathgenerator.rs`.

- [ ] **Step 2: Add the regression**

Construct two generators with identical process parameters, dimension, and seed:

```rust
let mut first = PathGenerator::new(gbs_process(), 1.0, 12, generator(12, 42), false).unwrap();
let mut second = PathGenerator::new(gbs_process(), 1.0, 12, generator(12, 42), false).unwrap();
let first_path = first.next().unwrap();
let second_path = second.next().unwrap();

assert_eq!(first_path.weight, second_path.weight);
assert_eq!(first_path.value.length(), second_path.value.length());
for i in 0..first_path.value.length() {
    assert_eq!(first_path.value[i], second_path.value[i]);
}
```

Also retain the existing invariant assertions for initial spot and grid length.

- [ ] **Step 3: Run the Monte Carlo tests**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods::montecarlo -- --nocapture
```

Expected: all Monte Carlo tests pass. If reproducibility fails, inspect sequence
generator ownership and cloning; do not add global random state.

- [ ] **Step 4: Commit the Monte Carlo slice**

```text
git add crates/libitofin/src/methods/montecarlo/pathgenerator.rs
git commit -m "test: cover seeded Monte Carlo reproducibility"
```

### Task 4: Run the complete methods validation

**Files:**
- No additional files unless a focused regression identifies a directly coupled defect.

- [ ] **Step 1: Run all methods tests**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo test -p libitofin methods:: -- --nocapture
```

- [ ] **Step 2: Build the workspace**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo build --workspace --locked
```

- [ ] **Step 3: Check formatting**

```text
call "C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cargo fmt --check
```

- [ ] **Step 4: Commit any formatting-only changes separately**

```text
git add crates/libitofin/src/methods
git commit -m "style: format methods coverage"
```

- [ ] **Step 5: Push the verified task-3 commits**

```text
git push origin HEAD:main
```
