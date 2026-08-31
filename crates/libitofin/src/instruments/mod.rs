//! Financial instruments.
//!
//! Port of `ql/instruments/`: the payoff subset and the vanilla-option
//! instruments needed by the European-option slice.

mod asianoption;
mod assetswap;
mod barrieroption;
mod bond;
mod bondforward;
mod bonds;
mod callablebond;
mod capfloor;
mod cmsswap;
mod fixedvsfloatingswap;
mod floatfloatswap;
mod forwardrateagreement;
mod forwardvanillaoption;
mod futures;
mod makecapfloor;
mod makeois;
mod makeswaption;
mod makevanillaswap;
mod oneassetoption;
mod overnightindexedswap;
mod payoffs;
mod swap;
mod swaption;
mod vanillaswap;
mod xccybasisswap;

pub use crate::pricingengines::{
    BinomialBarrierEngine, FdBlackScholesBarrierEngine, MCBarrierEngine, MakeMcBarrierEngine,
    set_binomial_barrier_engine, set_fd_black_scholes_barrier_engine, set_mc_barrier_engine,
};
pub use asianoption::geometric_average_price_asian;
pub use assetswap::AssetSwap;
pub use barrieroption::{
    AnalyticBarrierEngine, BarrierArguments, BarrierOption, BarrierType, barrier_price,
    set_analytic_barrier_engine,
};
pub use bond::{Bond, BondArguments, BondEngine, BondPrice, BondPriceType, BondResults};
pub use bondforward::BondForward;
pub use bonds::{
    ConvertibleBondArguments, ConvertibleFixedCouponBond, ConvertibleFloatingRateBond,
    ConvertibleZeroCouponBond, FixedRateBond, FloatingRateBond, ZeroCouponBond, soft_callability,
};
pub use callablebond::{
    Callability, CallabilitySchedule, CallabilityType, CallableBondArguments,
    CallableFixedRateBond, CallableZeroCouponBond,
};
pub use capfloor::{CapFloor, CapFloorArguments, CapFloorType};
pub use cmsswap::CmsSwap;
pub use fixedvsfloatingswap::{
    FixedVsFloatingSwap, FixedVsFloatingSwapArguments, FixedVsFloatingSwapEngine,
    FixedVsFloatingSwapResults, FloatingArgumentsFn,
};
pub use floatfloatswap::FloatFloatSwap;
pub use forwardrateagreement::{ForwardRateAgreement, Position};
pub use forwardvanillaoption::{ForwardOptionArguments, ForwardVanillaOption};
pub use futures::FuturesType;
pub use makecapfloor::MakeCapFloor;
pub use makeois::MakeOis;
pub use makeswaption::MakeSwaption;
pub use makevanillaswap::MakeVanillaSwap;
pub use oneassetoption::{
    EuropeanOption, Greeks, MoreGreeks, OneAssetOption, OneAssetOptionEngine,
    OneAssetOptionResults, OptionArguments, VanillaOption,
};
pub use overnightindexedswap::OvernightIndexedSwap;
pub use payoffs::{CashOrNothingPayoff, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff};
pub use swap::{Swap, SwapArguments, SwapEngine, SwapResults, SwapType};
pub use swaption::{
    SettlementMethod, SettlementType, Swaption, SwaptionArguments, SwaptionEngine,
    check_type_and_method_consistency,
};
pub use vanillaswap::VanillaSwap;
pub use xccybasisswap::XccyBasisSwap;
