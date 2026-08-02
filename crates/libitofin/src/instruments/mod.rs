//! Financial instruments.
//!
//! Port of `ql/instruments/`: the payoff subset and the vanilla-option
//! instruments needed by the European-option slice.

mod asianoption;
mod barrieroption;
mod bond;
mod bondforward;
mod bonds;
mod capfloor;
mod fixedvsfloatingswap;
mod forwardrateagreement;
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

pub use asianoption::geometric_average_price_asian;
pub use barrieroption::{
    AnalyticBarrierEngine, BarrierArguments, BarrierOption, BarrierType, barrier_price,
    set_analytic_barrier_engine,
};
pub use bond::{Bond, BondArguments, BondEngine, BondPrice, BondPriceType, BondResults};
pub use bondforward::BondForward;
pub use bonds::{FixedRateBond, FloatingRateBond, ZeroCouponBond};
pub use capfloor::{CapFloor, CapFloorArguments, CapFloorType};
pub use fixedvsfloatingswap::{
    FixedVsFloatingSwap, FixedVsFloatingSwapArguments, FixedVsFloatingSwapEngine,
    FixedVsFloatingSwapResults, FloatingArgumentsFn,
};
pub use forwardrateagreement::{ForwardRateAgreement, Position};
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
