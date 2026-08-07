//! Backward solvers over a finite-difference grid.
//!
//! Port of `ql/methods/finitedifferences/solvers/`. The descriptor a solver
//! picks its scheme from landed first, then the backward solver that switches
//! on it, then the dimensioned solvers (`Fdm2DimSolver`, `FdmG2Solver`).

mod fdm2dimsolver;
mod fdmbackwardsolver;
mod fdmg2solver;
mod fdmschemedesc;
mod fdmsolverdesc;

pub use fdm2dimsolver::Fdm2DimSolver;
pub use fdmbackwardsolver::FdmBackwardSolver;
pub use fdmg2solver::FdmG2Solver;
pub use fdmschemedesc::{FdmSchemeDesc, FdmSchemeType};
pub use fdmsolverdesc::FdmSolverDesc;
