//! Barrier-option pricing engines.
//!
//! Port of `ql/pricingengines/barrier/`. The analytic Haug engine lives with
//! the instrument; the finite-difference barrier and rebate engines are here.

mod binomialbarrierengine;
mod fdblackscholesbarrierengine;
mod fdblackscholesrebateengine;

pub use binomialbarrierengine::{BinomialBarrierEngine, set_binomial_barrier_engine};
pub use fdblackscholesbarrierengine::{
    FdBlackScholesBarrierEngine, set_fd_black_scholes_barrier_engine,
};
pub use fdblackscholesrebateengine::FdBlackScholesRebateEngine;
