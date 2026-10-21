//! Carr–Madan jump-to-default: a local-volatility stock that jumps to zero.
//!
//! P. Carr and D. B. Madan, "Local Volatility Enhanced by a Jump to Default",
//! *SIAM Journal on Financial Mathematics* 1(1), 2–15 (2010). Equation numbers
//! refer to the 2007 working paper. The intensity is `f(t) S^p` with `p <= 0`
//! (working-paper eq. 1).

mod powerhazard;

pub use powerhazard::PowerHazard;
