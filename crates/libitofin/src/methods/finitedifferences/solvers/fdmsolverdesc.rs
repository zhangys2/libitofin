//! Descriptor bundling the inputs a finite-difference solver needs.
//!
//! Port of `ql/methods/finitedifferences/solvers/fdmsolverdesc.hpp:35`.

use crate::methods::finitedifferences::meshers::FdmMesher;
use crate::methods::finitedifferences::stepconditions::FdmStepConditionComposite;
use crate::methods::finitedifferences::utilities::{
    FdmBoundaryConditionSet, FdmInnerValueCalculator,
};
use crate::shared::Shared;
use crate::types::{Size, Time};

/// Inputs shared by the dimensioned FD solvers (`fdmsolverdesc.hpp:35`).
#[derive(Clone)]
pub struct FdmSolverDesc {
    /// Spatial mesher (layout + locations).
    pub mesher: Shared<dyn FdmMesher>,
    /// Boundary conditions applied during the rollback.
    pub bc_set: FdmBoundaryConditionSet,
    /// Step conditions (Bermudan, American, …).
    pub condition: Shared<FdmStepConditionComposite>,
    /// Terminal / exercise payoff sampler.
    pub calculator: Shared<dyn FdmInnerValueCalculator>,
    /// Maturity of the roll (rollback starts here).
    pub maturity: Time,
    /// Number of scheme steps from maturity to 0.
    pub time_steps: Size,
    /// Leading fully-implicit damping steps.
    pub damping_steps: Size,
}
