//! Concrete bond instruments.
//!
//! Port of `ql/instruments/bonds/`: the derived bonds built on the
//! [`Bond`](crate::instruments::Bond) base.

mod convertiblebond;
mod fixedratebond;
mod floatingratebond;
mod zerocouponbond;

pub use convertiblebond::{
    soft_callability, ConvertibleBondArguments, ConvertibleFixedCouponBond,
    ConvertibleZeroCouponBond,
};
pub use fixedratebond::FixedRateBond;
pub use floatingratebond::FloatingRateBond;
pub use zerocouponbond::ZeroCouponBond;
