//! One-dimensional Black–Scholes finite-difference solver.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmblackscholessolver.{hpp,cpp}`:
//! builds an [`FdmBlackScholesOp`] and delegates interpolation to
//! [`Fdm1DimSolver`]. The local-vol branch of the operator is wired through
//! [`with_local_vol`](FdmBlackScholesSolver::with_local_vol);
//! quanto is [`with_quanto`](FdmBlackScholesSolver::with_quanto).

use crate::errors::QlResult;
use crate::methods::finitedifferences::operators::{FdmBlackScholesOp, FdmLinearOpComposite};
use crate::methods::finitedifferences::utilities::FdmQuantoHelper;
use crate::processes::GeneralizedBlackScholesProcess;
use crate::shared::{Shared, SharedMut, shared_mut};
use crate::types::Real;
use crate::utilities::null::Null;

use super::fdm1dimsolver::Fdm1DimSolver;
use super::fdmschemedesc::FdmSchemeDesc;
use super::fdmsolverdesc::FdmSolverDesc;

/// Lazy Black–Scholes 1-D solver (`fdmblackscholessolver.hpp`).
pub struct FdmBlackScholesSolver {
    solver: Fdm1DimSolver,
}

impl FdmBlackScholesSolver {
    /// `FdmBlackScholesSolver(process, strike, solverDesc, schemeDesc)`
    /// (`cpp:32-45`), with local-vol off.
    pub fn new(
        process: &GeneralizedBlackScholesProcess,
        strike: Real,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
    ) -> QlResult<Self> {
        Self::with_local_vol(
            process,
            strike,
            solver_desc,
            scheme_desc,
            false,
            -Real::null(),
        )
    }

    /// As [`new`](Self::new), with the C++ `localVol` /
    /// `illegalLocalVolOverwrite` arguments (`cpp:32-45`).
    pub fn with_local_vol(
        process: &GeneralizedBlackScholesProcess,
        strike: Real,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        local_vol: bool,
        illegal_local_vol_overwrite: Real,
    ) -> QlResult<Self> {
        Self::with_quanto(
            process,
            strike,
            solver_desc,
            scheme_desc,
            local_vol,
            illegal_local_vol_overwrite,
            None,
        )
    }

    /// As [`with_local_vol`](Self::with_local_vol), with the C++ `quantoHelper`.
    #[allow(clippy::too_many_arguments)]
    pub fn with_quanto(
        process: &GeneralizedBlackScholesProcess,
        strike: Real,
        solver_desc: FdmSolverDesc,
        scheme_desc: FdmSchemeDesc,
        local_vol: bool,
        illegal_local_vol_overwrite: Real,
        quanto: Option<Shared<FdmQuantoHelper>>,
    ) -> QlResult<Self> {
        let op = shared_mut(FdmBlackScholesOp::with_quanto(
            Shared::clone(&solver_desc.mesher),
            process,
            strike,
            local_vol,
            illegal_local_vol_overwrite,
            0,
            quanto,
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
