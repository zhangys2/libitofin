//! Forward (strike-resetting) vanilla pricing engines.
//!
//! Port of `ql/pricingengines/forward/` plus the quanto specialisation
//! `QuantoEngine<ForwardVanillaOption, ForwardVanillaEngine<AnalyticEuropeanEngine>>`.

mod analyticforwardperformancevanillaengine;
mod analyticforwardvanillaengine;
mod quantoforwardengine;
mod quantoforwardperformanceengine;

pub use analyticforwardperformancevanillaengine::{
    AnalyticForwardPerformanceVanillaEngine, set_analytic_forward_performance_vanilla_engine,
};
pub use analyticforwardvanillaengine::{
    AnalyticForwardVanillaEngine, set_analytic_forward_vanilla_engine,
};
pub use quantoforwardengine::{QuantoForwardEuropeanEngine, set_quanto_forward_european_engine};
pub use quantoforwardperformanceengine::{
    QuantoForwardPerformanceEuropeanEngine, set_quanto_forward_performance_european_engine,
};
