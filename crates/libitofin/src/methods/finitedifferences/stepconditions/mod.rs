//! The concrete step conditions a solver applies between steps.
//!
//! Port of `ql/methods/finitedifferences/stepconditions/`. C++ keeps the
//! `StepCondition` trait one level up in `stepcondition.hpp` and its
//! implementations here, so
//! [`stepcondition`](super::StepCondition) and this directory coexist by
//! design.
//!
//! `FdmStepConditionComposite::vanillaComposite` (`cpp:80-145`) still omits the
//! dividend handler (`FdmDividendHandler`, `cpp:104`); American and Bermudan
//! step conditions are ported below.

mod fdmamericanstepcondition;
mod fdmbermudanstepcondition;
mod fdmsnapshotcondition;
mod fdmstepconditioncomposite;

pub use fdmamericanstepcondition::FdmAmericanStepCondition;
pub use fdmbermudanstepcondition::FdmBermudanStepCondition;
pub use fdmsnapshotcondition::FdmSnapshotCondition;
pub use fdmstepconditioncomposite::FdmStepConditionComposite;
