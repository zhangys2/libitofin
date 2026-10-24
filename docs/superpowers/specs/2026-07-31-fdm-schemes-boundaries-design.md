# Finite-Difference Schemes and Boundary Conditions Design

**Goal:** Extend the FDM stack with explicit Euler, Crank-Nicolson, and
FDM-native Dirichlet/Neumann boundary conditions.

**Scope:** Add the two schemes to the existing `Scheme` contract and wire them
through `FdmBackwardSolver`. Add boundary conditions that operate on
`FdmLinearOp` and `Array`, not the deprecated legacy tridiagonal API.

**Design:**

- `ExplicitEulerScheme` performs `a <- a + dt * A(a)` with the existing
  before-applying, after-applying, and after-solving hooks.
- `CrankNicolsonScheme` uses the existing Douglas theta-splitting algorithm with
  `theta = 0.5`; this is mathematically equivalent for the current
  one-dimensional operators and avoids duplicating the split-step logic.
- `FdmSchemeDesc` gains `explicit_euler()` and `crank_nicolson()` factories.
- `FdmBackwardSolver` dispatches both new scheme types and retains explicit
  errors for the other unsupported families.
- Dirichlet and Neumann conditions are represented as concrete
  `BoundaryCondition` implementations. They modify the edge row/value through
  the existing `FdmLinearOp` abstraction and are tested with a small
  tridiagonal fixture.

**Validation:** Focused scheme and boundary tests, methods tests, and
`cargo build --workspace --locked`.
