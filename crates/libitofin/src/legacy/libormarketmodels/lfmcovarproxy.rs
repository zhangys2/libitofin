//! Covariance proxy combining LMM vol and correlation models.
//!
//! Port of `ql/legacy/libormarketmodels/lfmcovarproxy.{hpp,cpp}` diffusion and
//! covariance (integrated covariance deferred).

use crate::errors::QlResult;
use crate::legacy::libormarketmodels::lmexpcorrmodel::LmExponentialCorrelationModel;
use crate::legacy::libormarketmodels::lmlinexpvolmodel::LmLinearExponentialVolatilityModel;
use crate::math::matrix::Matrix;
use crate::require;
use crate::types::{Size, Time};

/// `LfmCovarianceProxy(volaModel, corrModel)` for the linear-exp / exponential pair.
pub struct LfmCovarianceProxy<'a> {
    size: Size,
    factors: Size,
    vola: &'a LmLinearExponentialVolatilityModel,
    corr: &'a LmExponentialCorrelationModel,
}

impl<'a> LfmCovarianceProxy<'a> {
    /// Builds the proxy; sizes must match.
    ///
    /// # Errors
    ///
    /// Fails when volatility and correlation sizes differ.
    pub fn new(
        vola: &'a LmLinearExponentialVolatilityModel,
        corr: &'a LmExponentialCorrelationModel,
    ) -> QlResult<Self> {
        require!(
            vola.size() == corr.size(),
            "different size for the volatility ({}) and correlation ({}) models",
            vola.size(),
            corr.size()
        );
        Ok(Self {
            size: corr.size(),
            factors: corr.factors(),
            vola,
            corr,
        })
    }

    /// Number of forward rates.
    pub fn size(&self) -> Size {
        self.size
    }

    /// Number of Brownian factors.
    pub fn factors(&self) -> Size {
        self.factors
    }

    /// Diffusion matrix `σ(t)` (`lfmcovarproxy.cpp:47`).
    pub fn diffusion(&self, t: Time) -> Matrix {
        let mut pca = self.corr.pseudo_sqrt(t);
        let vol = self.vola.volatility(t);
        for i in 0..self.size {
            for j in 0..pca.columns() {
                pca[(i, j)] *= vol[i];
            }
        }
        pca
    }

    /// Instantaneous covariance `diag(σ) ρ diag(σ)`.
    pub fn covariance(&self, t: Time) -> Matrix {
        let volatility = self.vola.volatility(t);
        let correlation = self.corr.correlation(t);
        let mut tmp = Matrix::with_size(self.size, self.size);
        for i in 0..self.size {
            for j in 0..self.size {
                tmp[(i, j)] = volatility[i] * correlation[(i, j)] * volatility[j];
            }
        }
        tmp
    }
}
