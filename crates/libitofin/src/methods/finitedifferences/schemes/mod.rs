//! Time-stepping schemes for the finite-difference solver.
//!
//! Port of `ql/methods/finitedifferences/schemes/`. The boundary-condition
//! helper the schemes all hold landed first; the [`Scheme`] contract they meet
//! drives the backward solver.

mod boundaryconditionschemehelper;
mod craigsneydscheme;
mod cranknicolsonscheme;
mod douglasscheme;
mod expliciteulerscheme;
mod hundsdorferscheme;
mod impliciteulerscheme;
mod modifiedcraigsneydscheme;
mod scheme;
#[cfg(test)]
pub(crate) mod testops;
mod trbdf2scheme;

pub use boundaryconditionschemehelper::BoundaryConditionSchemeHelper;
pub use craigsneydscheme::CraigSneydScheme;
pub use cranknicolsonscheme::CrankNicolsonScheme;
pub use douglasscheme::DouglasScheme;
pub use expliciteulerscheme::ExplicitEulerScheme;
pub use hundsdorferscheme::HundsdorferScheme;
pub use impliciteulerscheme::ImplicitEulerScheme;
pub use modifiedcraigsneydscheme::ModifiedCraigSneydScheme;
pub use scheme::Scheme;
pub use trbdf2scheme::TrBDF2Scheme;
