# Task 3 Methods Coverage Design

## Scope

Task 3 closes the most concrete methods-layer gaps without attempting a broad
QuantLib rewrite. The work covers:

- a finite-difference European-option rollback path using the existing mesh,
  operator, scheme, condition, and solver abstractions;
- a trinomial-tree recombination regression;
- a seeded Monte Carlo path-generation reproducibility regression.

The finite-difference path is the implementation focus. Lattice and Monte Carlo
work is limited to proving that the existing public primitives preserve their
core invariants.

## Design

The finite-difference test will construct a simple Black-Scholes European call
on a one-dimensional grid, apply the existing boundary conditions and
Crank–Nicolson-style timestep machinery, and assert a stable value against the
analytic Black-Scholes price within a practical discretization tolerance.
Missing glue will be implemented in the existing `operators`, `schemes`,
`solvers`, and `utilities` modules rather than introducing a parallel solver
API.

The lattice test will build a short trinomial tree and assert that equivalent
paths recombine to the same node index/value. It will use the existing
`Tree`, `TrinomialTree`, and `TreeLattice1D` contracts.

The Monte Carlo test will use the existing seeded path generator and assert
that two runs with identical seed and parameters produce the same path shape
and sample values. It will not add a new random-number abstraction.

## Error handling and compatibility

Existing `QlResult` and `require!` validation paths remain authoritative.
Unsupported configurations must return errors rather than silently selecting a
fallback. Public APIs remain unchanged unless a missing adapter is required to
connect already-public primitives.

## Validation

Run the focused methods tests:

```text
cargo test -p libitofin methods:: -- --nocapture
```

Then run the workspace build:

```text
cargo build --workspace --locked
```

The implementation is complete when the new regressions and existing methods
tests pass, formatting remains clean, and no unrelated files are changed.
