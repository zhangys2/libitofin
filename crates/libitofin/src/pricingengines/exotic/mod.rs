//! Exotic-option pricing engines.
//!
//! Port of `ql/pricingengines/exotic/`.

mod analyticcomplexchooserengine;
mod analyticcompoundoptionengine;
mod analyticeuropeanmargrabeengine;
mod analyticsimplechooserengine;
mod analytictwoassetcorrelationengine;

pub use analyticcomplexchooserengine::{
    AnalyticComplexChooserEngine, set_analytic_complex_chooser_engine,
};
pub use analyticcompoundoptionengine::{
    AnalyticCompoundOptionEngine, set_analytic_compound_option_engine,
};
pub use analyticeuropeanmargrabeengine::{
    AnalyticEuropeanMargrabeEngine, set_analytic_european_margrabe_engine,
};
pub use analyticsimplechooserengine::{
    AnalyticSimpleChooserEngine, set_analytic_simple_chooser_engine,
};
pub use analytictwoassetcorrelationengine::{
    AnalyticTwoAssetCorrelationEngine, set_analytic_two_asset_correlation_engine,
};
