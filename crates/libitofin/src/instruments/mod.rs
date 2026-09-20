//! Financial instruments.
//!
//! Port of `ql/instruments/`: the payoff subset and the vanilla-option
//! instruments needed by the European-option slice.

mod asianoption;
mod assetswap;
mod barrieroption;
mod basketoption;
mod bond;
mod bondforward;
mod bonds;
mod callablebond;
mod capfloor;
mod claim;
mod cliquetoption;
mod cmsswap;
mod complexchooseroption;
mod compoundoption;
mod continuousaveragingasianoption;
mod creditdefaultswap;
mod discreteaveragingasianoption;
mod doublebarrieroption;
mod everestoption;
mod fixedvsfloatingswap;
mod floatfloatswap;
mod forwardrateagreement;
mod forwardvanillaoption;
mod futures;
mod holderextensibleoption;
mod inflationcapfloor;
mod lookbackoption;
mod makecapfloor;
mod makecds;
mod makeois;
mod makeswaption;
mod makevanillaswap;
mod makeyoyinflationcapfloor;
mod margrabeoption;
mod oneassetoption;
mod overnightindexedswap;
mod partialtimebarrieroption;
mod payoffs;
mod protection;
mod simplechooseroption;
mod softbarrieroption;
mod swap;
mod swaption;
mod twoassetbarrieroption;
mod twoassetcorrelationoption;
mod vanillaswap;
mod writerextensibleoption;
mod xccybasisswap;
mod yearonyearinflationswap;
mod zerocouponinflationswap;

pub use crate::pricingengines::{
    AnalyticBinaryBarrierEngine, AnalyticDoubleBarrierEngine, BinomialBarrierEngine,
    FdBlackScholesBarrierEngine, MCBarrierEngine, MCDoubleBarrierEngine, MakeMcBarrierEngine,
    MakeMcDoubleBarrierEngine, set_analytic_binary_barrier_engine,
    set_analytic_double_barrier_engine, set_binomial_barrier_engine,
    set_fd_black_scholes_barrier_engine, set_mc_barrier_engine, set_mc_double_barrier_engine,
};
pub use asianoption::geometric_average_price_asian;
pub use assetswap::AssetSwap;
pub use barrieroption::{
    AnalyticBarrierEngine, BarrierArguments, BarrierOption, BarrierType, barrier_price,
    set_analytic_barrier_engine,
};
pub use basketoption::{
    AverageBasketPayoff, BasketArguments, BasketEngine, BasketOption, BasketResults,
    SpreadBasketPayoff,
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
pub use claim::{Claim, FaceValueAccrualClaim, FaceValueClaim};
pub use cliquetoption::{CliquetArguments, CliquetOption, CliquetResults};
pub use cmsswap::CmsSwap;
pub use complexchooseroption::{
    ComplexChooserArguments, ComplexChooserOption, ComplexChooserResults,
};
pub use compoundoption::{CompoundArguments, CompoundOption, CompoundResults};
pub use continuousaveragingasianoption::{
    AverageType, ContinuousAveragingAsianArguments, ContinuousAveragingAsianOption,
    ContinuousAveragingAsianResults,
};
pub use creditdefaultswap::{
    CdsArguments, CdsEngine, CdsResults, CdsTerms, CreditDefaultSwap, PricingModel, cds_maturity,
};
pub use discreteaveragingasianoption::{
    DiscreteAveragingAsianArguments, DiscreteAveragingAsianOption, DiscreteAveragingAsianResults,
};
pub use doublebarrieroption::{
    DoubleBarrierArguments, DoubleBarrierOption, DoubleBarrierType, double_barrier_triggered,
};
pub use everestoption::{EverestArguments, EverestOption, EverestResults};
pub use fixedvsfloatingswap::{
    FixedVsFloatingSwap, FixedVsFloatingSwapArguments, FixedVsFloatingSwapEngine,
    FixedVsFloatingSwapResults, FloatingArgumentsFn,
};
pub use floatfloatswap::FloatFloatSwap;
pub use forwardrateagreement::ForwardRateAgreement;
pub use forwardvanillaoption::{ForwardOptionArguments, ForwardVanillaOption};
pub use futures::FuturesType;
pub use holderextensibleoption::{
    HolderExtensibleArguments, HolderExtensibleOption, HolderExtensibleResults,
};
pub use inflationcapfloor::{YoYInflationCapFloor, YoYInflationCapFloorArguments};
pub use lookbackoption::{
    ContinuousFixedLookbackArguments, ContinuousFixedLookbackOption,
    ContinuousFixedLookbackResults, ContinuousFloatingLookbackArguments,
    ContinuousFloatingLookbackOption, ContinuousFloatingLookbackResults,
    ContinuousPartialFixedLookbackArguments, ContinuousPartialFixedLookbackOption,
    ContinuousPartialFixedLookbackResults, ContinuousPartialFloatingLookbackArguments,
    ContinuousPartialFloatingLookbackOption, ContinuousPartialFloatingLookbackResults,
};
pub use makecapfloor::MakeCapFloor;
pub use makecds::MakeCreditDefaultSwap;
pub use makeois::MakeOis;
pub use makeswaption::MakeSwaption;
pub use makevanillaswap::MakeVanillaSwap;
pub use makeyoyinflationcapfloor::MakeYoYInflationCapFloor;
pub use margrabeoption::{MargrabeArguments, MargrabeOption, MargrabeResults};
pub use oneassetoption::{
    EuropeanOption, Greeks, MoreGreeks, OneAssetOption, OneAssetOptionEngine,
    OneAssetOptionResults, OptionArguments, VanillaOption,
};
pub use overnightindexedswap::OvernightIndexedSwap;
pub use partialtimebarrieroption::{
    PartialBarrierRange, PartialTimeBarrierArguments, PartialTimeBarrierOption,
    PartialTimeBarrierResults,
};
pub use payoffs::{
    AssetOrNothingPayoff, CashOrNothingPayoff, FloatingTypePayoff, GapPayoff, NullPayoff,
    PercentageStrikePayoff, PlainVanillaPayoff, StrikedTypePayoff, TypePayoff,
};
pub use protection::ProtectionSide;
pub use simplechooseroption::{SimpleChooserArguments, SimpleChooserOption, SimpleChooserResults};
pub use softbarrieroption::{SoftBarrierArguments, SoftBarrierOption, SoftBarrierResults};
pub use swap::{Swap, SwapArguments, SwapEngine, SwapResults, SwapType};
pub use swaption::{
    SettlementMethod, SettlementType, Swaption, SwaptionArguments, SwaptionEngine,
    SwaptionPriceType, check_type_and_method_consistency,
};
pub use twoassetbarrieroption::{
    TwoAssetBarrierArguments, TwoAssetBarrierOption, TwoAssetBarrierResults,
};
pub use twoassetcorrelationoption::{
    TwoAssetCorrelationArguments, TwoAssetCorrelationOption, TwoAssetCorrelationResults,
};
pub use vanillaswap::VanillaSwap;
pub use writerextensibleoption::{
    WriterExtensibleArguments, WriterExtensibleOption, WriterExtensibleResults,
};
pub use xccybasisswap::XccyBasisSwap;
pub use yearonyearinflationswap::YearOnYearInflationSwap;
pub use zerocouponinflationswap::ZeroCouponInflationSwap;

pub use crate::position::Position;
