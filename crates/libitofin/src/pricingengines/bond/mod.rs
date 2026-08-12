//! Bond pricing helpers.
//!
//! Port of `ql/pricingengines/bond/`: the free-function analytics a [`Bond`]
//! is priced through.
//!
//! [`Bond`]: crate::instruments::Bond

mod binomialconvertibleengine;
mod blackcallablebondengine;
mod bondfunctions;
mod discountingbondengine;
mod discretizedcallablebond;
pub(crate) mod discretizedconvertible;
mod treecallablebondengine;

pub use binomialconvertibleengine::BinomialConvertibleEngine;
pub use blackcallablebondengine::{
    BlackCallableFixedRateBondEngine, BlackCallableZeroCouponBondEngine,
};
pub use bondfunctions::BondFunctions;
pub use discountingbondengine::DiscountingBondEngine;
pub use discretizedconvertible::DividendSchedule;
pub use treecallablebondengine::TreeCallableFixedRateBondEngine;

/// Lattice engine for callable zero-coupon bonds.
///
/// QuantLib's `TreeCallableZeroCouponBondEngine` is a thin alias of
/// [`TreeCallableFixedRateBondEngine`]; the same engine prices zeros (empty
/// coupon schedule, redemption-only cash flows).
pub type TreeCallableZeroCouponBondEngine = TreeCallableFixedRateBondEngine;
