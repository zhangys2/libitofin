//! Forward term structure representations and implied forward estimation.

pub mod american_model;
// Wired into `ImpliedForward` by the next commit.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) mod american_pricer;
pub mod implied_forward;

pub use american_model::{
    AmericanLevel, AmericanModelConfig, AmericanModelDiagnostics, ForwardFlags, ForwardTier,
};

pub use implied_forward::{
    ForwardDiagnostics, ForwardStatus, ImpliedForward, ImpliedForwardConfig, ImpliedForwardResult,
    MarketConvention, OptionQuotePair,
};
