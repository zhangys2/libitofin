//! Barrier-option pricing engines.
//!
//! Port of `ql/pricingengines/barrier/`. The analytic Haug engine lives with
//! the instrument; the finite-difference, binomial, Monte Carlo, and analytic
//! quanto barrier engines are here.

mod analyticdoublebarrierengine;
mod analyticpartialtimebarrieroptionengine;
mod analyticsoftbarrierengine;
mod binomialbarrierengine;
mod fdblackscholesbarrierengine;
mod fdblackscholesrebateengine;
mod fdhestonbarrierengine;
mod fdhestonrebateengine;
mod mcbarrierengine;
mod mcdoublebarrierengine;
mod quantobarrierengine;
mod quantodoublebarrierengine;
mod vannavolgadoublebarrierengine;
mod vannavolgainterpolation;

pub use analyticdoublebarrierengine::{
    AnalyticDoubleBarrierEngine, set_analytic_double_barrier_engine,
};
pub use analyticpartialtimebarrieroptionengine::{
    AnalyticPartialTimeBarrierOptionEngine, set_analytic_partial_time_barrier_engine,
};
pub use analyticsoftbarrierengine::{AnalyticSoftBarrierEngine, set_analytic_soft_barrier_engine};
pub use binomialbarrierengine::{BinomialBarrierEngine, set_binomial_barrier_engine};
pub use fdblackscholesbarrierengine::{
    FdBlackScholesBarrierEngine, set_fd_black_scholes_barrier_engine,
};
pub use fdblackscholesrebateengine::FdBlackScholesRebateEngine;
pub use fdhestonbarrierengine::{FdHestonBarrierEngine, set_fd_heston_barrier_engine};
pub use fdhestonrebateengine::FdHestonRebateEngine;
pub use mcbarrierengine::{
    BarrierPathPricer, BiasedBarrierPathPricer, MCBarrierEngine, MakeMcBarrierEngine,
    set_mc_barrier_engine,
};
pub use mcdoublebarrierengine::{
    DoubleBarrierPathPricer, MCDoubleBarrierEngine, MakeMcDoubleBarrierEngine,
    set_mc_double_barrier_engine,
};
pub use quantobarrierengine::{QuantoBarrierEngine, set_quanto_barrier_engine};
pub use quantodoublebarrierengine::{QuantoDoubleBarrierEngine, set_quanto_double_barrier_engine};
pub use vannavolgadoublebarrierengine::{
    VannaVolgaDoubleBarrierEngine, set_vanna_volga_double_barrier_engine,
};
