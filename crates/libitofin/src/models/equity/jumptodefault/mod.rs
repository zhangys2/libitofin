//! Carr–Madan jump-to-default: a local-volatility stock that jumps to zero.
//!
//! P. Carr and D. B. Madan, "Local Volatility Enhanced by a Jump to Default",
//! *SIAM Journal on Financial Mathematics* 1(1), 2–15 (2010). Equation numbers
//! refer to the 2007 working paper. The intensity is `f(t) S^p` with `p <= 0`
//! (working-paper eq. 1).

mod callsurface;
mod hazardcalibration;
mod powerhazard;
mod powermoments;
mod slvdiagnostics;
mod slvparticlemodel;

pub use hazardcalibration::calibrate_power_hazard;

#[cfg(test)]
pub(crate) use callsurface::ConstantHazardCallSurface;
pub use callsurface::{BlackVolCallSurface, CallSurface};
pub use powerhazard::PowerHazard;
pub use powermoments::{MomentGrid, power_price, truncated_power_price};
pub use slvdiagnostics::{CrossAssetGreeks, compute_cross_asset_greeks};
pub use slvparticlemodel::{
    SlvJtdCalibrationResult, SlvJtdParticleConfig, SlvJtdParticleSnapshot,
    calibrate_slv_jump_to_default,
};
pub(crate) use slvparticlemodel::{defaultable_option_value, simulate_fixed_leverage};
