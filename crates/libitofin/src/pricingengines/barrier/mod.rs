//! Barrier-option pricing engines.
//!
//! Port of `ql/pricingengines/barrier/`. The analytic Haug engine lives with
//! the instrument; the finite-difference barrier and rebate engines are here.

mod fdblackscholesbarrierengine;
mod fdblackscholesrebateengine;

pub use fdblackscholesbarrierengine::{
    FdBlackScholesBarrierEngine, set_fd_black_scholes_barrier_engine,
};
pub use fdblackscholesrebateengine::FdBlackScholesRebateEngine;
