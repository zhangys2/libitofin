//! Pricing engines and their numeric cores.
//!
//! Port of the `ql/pricingengines` layer: the Black 1976 formula family,
//! the [`BlackCalculator`] greeks core, and the analytic vanilla engines
//! built on them.

pub mod asian;
pub mod barrier;
pub mod basket;
pub mod blackcalculator;
pub mod blackdeltacalculator;
pub mod blackformula;
pub mod bond;
pub mod capfloor;
pub mod cliquet;
pub mod credit;
pub mod exotic;
pub mod forward;
pub mod greeks;
pub mod inflation;
pub mod lookback;
pub mod mclongstaffschwartzengine;
pub mod swap;
pub mod swaption;
pub mod vanilla;

pub use asian::{
    AnalyticContinuousGeometricAveragePriceAsianEngine,
    AnalyticContinuousGeometricAveragePriceAsianHestonEngine,
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
    AnalyticDiscreteGeometricAveragePriceAsianHestonEngine,
    AnalyticDiscreteGeometricAverageStrikeAsianEngine, ChoiAsianEngine,
    ContinuousArithmeticAsianLevyEngine, ContinuousArithmeticAsianVecerEngine,
    MCDiscreteArithmeticAveragePriceAsianEngine, MCDiscreteArithmeticAveragePriceAsianHestonEngine,
    MCDiscreteArithmeticAverageStrikeAsianEngine, MCDiscreteGeometricAveragePriceAsianEngine,
    MCDiscreteGeometricAveragePriceAsianHestonEngine, MakeMcDiscreteArithmeticApEngine,
    MakeMcDiscreteArithmeticApHestonEngine, MakeMcDiscreteArithmeticAsEngine,
    MakeMcDiscreteGeometricApEngine, MakeMcDiscreteGeometricApHestonEngine,
    TurnbullWakemanAsianEngine, set_analytic_continuous_geometric_average_price_asian_engine,
    set_analytic_continuous_geometric_average_price_asian_heston_engine,
    set_analytic_discrete_geometric_average_price_asian_engine,
    set_analytic_discrete_geometric_average_price_asian_heston_engine,
    set_analytic_discrete_geometric_average_strike_asian_engine, set_choi_asian_engine,
    set_continuous_arithmetic_asian_levy_engine, set_continuous_arithmetic_asian_vecer_engine,
    set_mc_discrete_arithmetic_average_price_asian_engine,
    set_mc_discrete_arithmetic_average_price_asian_heston_engine,
    set_mc_discrete_arithmetic_average_strike_asian_engine,
    set_mc_discrete_geometric_average_price_asian_engine,
    set_mc_discrete_geometric_average_price_asian_heston_engine, set_turnbull_wakeman_asian_engine,
};
pub use barrier::{
    AnalyticBinaryBarrierEngine, AnalyticDoubleBarrierEngine,
    AnalyticPartialTimeBarrierOptionEngine, AnalyticSoftBarrierEngine,
    AnalyticTwoAssetBarrierEngine, BarrierPathPricer, BiasedBarrierPathPricer,
    BinomialBarrierEngine, FdBlackScholesBarrierEngine, FdBlackScholesRebateEngine,
    FdHestonBarrierEngine, FdHestonDoubleBarrierEngine, FdHestonRebateEngine, MCBarrierEngine,
    MCDoubleBarrierEngine, MakeMcBarrierEngine, MakeMcDoubleBarrierEngine, QuantoBarrierEngine,
    QuantoDoubleBarrierEngine, VannaVolgaBarrierEngine, VannaVolgaDoubleBarrierEngine,
    set_analytic_binary_barrier_engine, set_analytic_double_barrier_engine,
    set_analytic_partial_time_barrier_engine, set_analytic_soft_barrier_engine,
    set_analytic_two_asset_barrier_engine, set_binomial_barrier_engine,
    set_fd_black_scholes_barrier_engine, set_fd_heston_barrier_engine,
    set_fd_heston_double_barrier_engine, set_mc_barrier_engine, set_mc_double_barrier_engine,
    set_quanto_barrier_engine, set_quanto_double_barrier_engine, set_vanna_volga_barrier_engine,
    set_vanna_volga_double_barrier_engine,
};
pub use basket::{
    BjerksundStenslandSpreadEngine, ChoiBasketEngine, KirkEngine, OperatorSplittingOrder,
    OperatorSplittingSpreadEngine, PearsonSpreadEngine, SingleFactorBsmBasketEngine, StulzEngine,
    SumExponentialsRootSolver, bjerksund_stensland_spread_option_value, kirk_spread_option_value,
    operator_splitting_spread_option_value, pearson_spread_option_value,
    pearson_spread_option_value_with_config, set_bjerksund_stensland_engine, set_kirk_engine,
    set_operator_splitting_engine, set_operator_splitting_engine_with_order, set_pearson_engine,
    set_pearson_engine_with_config, set_stulz_engine,
};
pub use blackcalculator::BlackCalculator;
pub use blackdeltacalculator::BlackDeltaCalculator;
pub use bond::{BinomialConvertibleEngine, BondFunctions, DiscountingBondEngine, DividendSchedule};
pub use capfloor::{
    AnalyticCapFloorEngine, BachelierCapFloorEngine, BlackCapFloorEngine, TreeCapFloorEngine,
};
pub use cliquet::{
    AnalyticCliquetEngine, AnalyticPerformanceEngine, set_analytic_cliquet_engine,
    set_analytic_performance_engine,
};
pub use credit::{IntegralCdsEngine, MidPointCdsEngine};
pub use exotic::{
    AnalyticAmericanMargrabeEngine, AnalyticComplexChooserEngine, AnalyticCompoundOptionEngine,
    AnalyticEuropeanMargrabeEngine, AnalyticHolderExtensibleOptionEngine,
    AnalyticSimpleChooserEngine, AnalyticTwoAssetCorrelationEngine,
    AnalyticWriterExtensibleOptionEngine, MCEverestEngine, MakeMcEverestEngine,
    set_analytic_american_margrabe_engine, set_analytic_complex_chooser_engine,
    set_analytic_compound_option_engine, set_analytic_european_margrabe_engine,
    set_analytic_holder_extensible_option_engine, set_analytic_simple_chooser_engine,
    set_analytic_two_asset_correlation_engine, set_analytic_writer_extensible_option_engine,
    set_mc_everest_engine,
};
pub use forward::{
    AnalyticForwardPerformanceVanillaEngine, AnalyticForwardVanillaEngine,
    AnalyticHestonForwardEuropeanEngine, BinomialForwardVanillaEngine, ForwardEuropeanBsPathPricer,
    ForwardEuropeanHestonPathPricer, MakeMcForwardEuropeanBsEngine,
    MakeMcForwardEuropeanHestonEngine, McForwardEuropeanBsEngine, McForwardEuropeanHestonEngine,
    QuantoForwardEuropeanEngine, QuantoForwardPerformanceEuropeanEngine,
    ReplicatingVarianceSwapEngine, set_analytic_forward_performance_vanilla_engine,
    set_analytic_forward_vanilla_engine, set_mc_forward_european_bs_engine,
    set_quanto_forward_european_engine, set_quanto_forward_performance_european_engine,
    set_replicating_variance_swap_engine,
};
pub use greeks::{black_scholes_theta, default_theta_per_day};
pub use inflation::{YoYInflationCapFloorEngine, yoy_optionlet_price};
pub use lookback::{
    AnalyticContinuousFixedLookbackEngine, AnalyticContinuousFloatingLookbackEngine,
    AnalyticContinuousPartialFixedLookbackEngine, AnalyticContinuousPartialFloatingLookbackEngine,
    MCLookbackEngine, MakeMcLookbackEngine, McContinuousFixedLookbackEngine,
    McContinuousFloatingLookbackEngine, McContinuousPartialFixedLookbackEngine,
    McContinuousPartialFloatingLookbackEngine, set_analytic_continuous_fixed_lookback_engine,
    set_analytic_continuous_floating_lookback_engine,
    set_analytic_continuous_partial_fixed_lookback_engine,
    set_analytic_continuous_partial_floating_lookback_engine,
    set_mc_continuous_fixed_lookback_engine, set_mc_continuous_floating_lookback_engine,
    set_mc_continuous_partial_fixed_lookback_engine,
    set_mc_continuous_partial_floating_lookback_engine,
};
pub use mclongstaffschwartzengine::McLongstaffSchwartzEngineBase;
pub use swap::DiscountingSwapEngine;
pub use swaption::{
    BachelierSpec, BachelierSwaptionEngine, Black76Spec, BlackStyleSpec, BlackStyleSwaptionEngine,
    BlackSwaptionEngine, CashAnnuityModel, DiscretizedSwap, FdG2SwaptionEngine,
    FdHullWhiteSwaptionEngine, G2SwaptionEngine, JamshidianSwaptionEngine, TreeG2SwaptionEngine,
    TreeSwaptionEngine,
};
pub use vanilla::{
    AnalyticDigitalAmericanEngine, AnalyticDigitalAmericanKOEngine, AnalyticEuropeanEngine,
    BaroneAdesiWhaleyApproximationEngine, BjerksundStenslandApproximationEngine, CashDividendModel,
    FdBlackScholesVanillaEngine, FdHestonVanillaEngine, FdSimpleBSSwingEngine,
    JuQuadraticApproximationEngine, JumpDiffusionEngine, QuantoEuropeanEngine,
    set_fd_simple_bs_swing_engine,
};

pub use blackformula::{
    bachelier_black_formula_implied_vol, black_formula, black_formula_asset_itm_probability,
    black_formula_cash_itm_probability, black_formula_forward_derivative,
    black_formula_implied_std_dev, black_formula_implied_std_dev_approximation,
    black_formula_std_dev_derivative, black_formula_std_dev_second_derivative,
    black_formula_vol_derivative,
};

#[cfg(test)]
pub(crate) mod hull_fixture {
    //! Hull's S=42, K=40, r=10%, q=0, sigma=20%, T=0.5 European option,
    //! shared by the blackformula and blackcalculator oracle tests.

    use crate::types::{Real, Time};

    pub(crate) const SPOT: Real = 42.0;
    pub(crate) const STRIKE: Real = 40.0;
    pub(crate) const MATURITY: Time = 0.5;
    pub(crate) const FORWARD: Real = 44.15338604779301;
    pub(crate) const DISCOUNT: Real = 0.951229424500714;
    pub(crate) const STD_DEV: Real = 0.14142135623730953;
}
