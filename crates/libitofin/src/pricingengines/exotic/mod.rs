//! Exotic-option pricing engines.
//!
//! Port of `ql/pricingengines/exotic/`.

mod analyticsimplechooserengine;

pub use analyticsimplechooserengine::{
    AnalyticSimpleChooserEngine, set_analytic_simple_chooser_engine,
};
