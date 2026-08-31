//! Asian-option pricing engines.

mod analyticcontinuousgeometricaveragepriceasianengine;
mod analyticdiscretegeometricaveragepriceasianengine;

pub use analyticcontinuousgeometricaveragepriceasianengine::{
    AnalyticContinuousGeometricAveragePriceAsianEngine,
    set_analytic_continuous_geometric_average_price_asian_engine,
};
pub use analyticdiscretegeometricaveragepriceasianengine::{
    AnalyticDiscreteGeometricAveragePriceAsianEngine,
    set_analytic_discrete_geometric_average_price_asian_engine,
};
