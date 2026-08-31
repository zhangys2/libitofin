//! Forward (strike-resetting) vanilla-option engines.
//!
//! Port of `ql/pricingengines/forward/forwardengine.hpp`.

mod forwardvanillaengine;

pub use forwardvanillaengine::{ForwardVanillaEngine, forward_vanilla_calculate};
