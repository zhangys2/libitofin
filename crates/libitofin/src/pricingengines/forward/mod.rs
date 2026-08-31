//! Forward (strike-resetting) vanilla pricing engines.
//!
//! Port of `ql/pricingengines/forward/` plus the quanto specialisation
//! `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.

mod analyticforwardvanillaengine;
mod quantoforwardengine;

pub use analyticforwardvanillaengine::{
    set_analytic_forward_vanilla_engine, AnalyticForwardVanillaEngine,
};
pub use quantoforwardengine::{set_quanto_forward_european_engine, QuantoForwardEuropeanEngine};
