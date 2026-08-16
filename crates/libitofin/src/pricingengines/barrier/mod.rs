//! Barrier-option pricing engines.
//!
//! Port of `ql/pricingengines/barrier/`. The analytic Haug engine lives with
//! the instrument; the finite-difference knock-out engine is here.

mod fdblackscholesbarrierengine;

pub use fdblackscholesbarrierengine::{
    FdBlackScholesBarrierEngine, set_fd_black_scholes_barrier_engine,
};
