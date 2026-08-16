//! One-dimensional Black–Scholes finite-difference solver.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmblackscholessolver.{hpp,cpp}`:
//! builds an [`FdmBlackScholesOp`] and delegates interpolation to
//! [`Fdm1DimSolver`]. Local-vol and quanto branches are omitted with the
//! operator (`fdmblackscholesop.rs`).

use crate::errors::QlResult;
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::types::Real;

use super::fdm1dimsolver::Fdm1DimSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Lazy Black–Scholes 1-D solver (`fdmblackscholessolver.hpp`).
pub struct FdmBlackScholesSolver {
    solver: Fdm1DimSolver,
}

impl FdmBlackScholesSolver {
    /// `FdmBlackScholesSolver(process, strike, solverDesc, schemeDesc)`
    /// (`cpp:32-45`), without the unported local-vol / quanto arguments.
    pub fn new(
        process: &GeneralizedBlackScholesProcess,
        strike: Real,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
    ) -> QlResult<Self> {
        let op = shared_mut(FdmBlackScholesOp::new(
            Shared::clone(&solver_desc.mesher),
            process,
            strike,
            0,
        )?);
        Ok(Self {
            solver: Fdm1DimSolver::new(
                solver_desc,
                scheme_desc,
                op as SharedMut<dyn FdmLinearOpComposite>,
            ),
        })
    }

    /// Option value at spot `s` (`cpp:59-62`).
    pub fn value_at(&self, s: Real) -> QlResult<Real> {
        self.solver.interpolate_at(s.ln())
    }

    /// Delta at spot `s` (`cpp:64-67`).
    pub fn delta_at(&self, s: Real) -> QlResult<Real> {
        Ok(self.solver.derivative_x(s.ln())? / s)
    }

    /// Theta at spot `s` (`cpp:75-77`).
    pub fn theta_at(&self, s: Real) -> QlResult<Real> {
        self.solver.theta_at(s.ln())
    }
}
