//! Legacy QuantLib modules.
//!
//! Port of `ql/legacy/`: APIs kept for oracle parity with deprecated QL
//! suites. The first resident is the Libor market model covariance spine.

pub mod libormarketmodels;

pub use libormarketmodels::{
    LfmCovarianceProxy, LmExponentialCorrelationModel, LmLinearExponentialVolatilityModel,
};
