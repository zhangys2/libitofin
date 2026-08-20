//! Barrier-option pricing engines.
//!
//! Port of `ql/pricingengines/barrier/`. The analytic Haug engine lives with
//! the instrument; the finite-difference, binomial, and Monte Carlo barrier
//! engines are here.

mod binomialbarrierengine;
mod fdblackscholesbarrierengine;
mod fdblackscholesrebateengine;
mod fdhestonbarrierengine;
mod fdhestonrebateengine;
mod mcbarrierengine;

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
