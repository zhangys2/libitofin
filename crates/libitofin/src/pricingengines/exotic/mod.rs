//! Exotic-option pricing engines.
//!
//! Port of `ql/pricingengines/exotic/`.

mod analyticamericanmargrabeengine;
mod analyticcomplexchooserengine;
mod analyticcompoundoptionengine;
mod analyticeuropeanmargrabeengine;
mod analyticholderextensibleoptionengine;
mod analyticsimplechooserengine;
mod analytictwoassetcorrelationengine;
mod analyticwriterextensibleoptionengine;
mod mceverestengine;

pub use analyticamericanmargrabeengine::{
    AnalyticAmericanMargrabeEngine, set_analytic_american_margrabe_engine,
};
pub use analyticcomplexchooserengine::{
    AnalyticComplexChooserEngine, set_analytic_complex_chooser_engine,
};
pub use analyticcompoundoptionengine::{
    AnalyticCompoundOptionEngine, set_analytic_compound_option_engine,
};
pub use analyticeuropeanmargrabeengine::{
    AnalyticEuropeanMargrabeEngine, set_analytic_european_margrabe_engine,
};
pub use analyticholderextensibleoptionengine::{
    AnalyticHolderExtensibleOptionEngine, set_analytic_holder_extensible_option_engine,
};
pub use analyticsimplechooserengine::{
    AnalyticSimpleChooserEngine, set_analytic_simple_chooser_engine,
};
pub use analytictwoassetcorrelationengine::{
    AnalyticTwoAssetCorrelationEngine, set_analytic_two_asset_correlation_engine,
};
pub use analyticwriterextensibleoptionengine::{
    AnalyticWriterExtensibleOptionEngine, set_analytic_writer_extensible_option_engine,
};
pub use mceverestengine::{
    EverestMultiPathPricer, MCEverestEngine, MakeMcEverestEngine, set_mc_everest_engine,
};
