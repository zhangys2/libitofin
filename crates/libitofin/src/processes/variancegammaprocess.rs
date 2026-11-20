//! Variance-gamma spot process.
//!
//! Port of `ql/experimental/variancegamma/variancegammaprocess.{hpp,cpp}`.
//! The process carries the spot and the two yield curves the analytic engine
//! discounts with. Path drift and diffusion are the C++ failures: Monte Carlo
//! and finite-difference variance-gamma engines are not provided.

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::quotes::Quote;
use crate::require;
use crate::shared::{Shared, SharedMut, shared};
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::types::Real;

/// Spot, dividend, and risk-free inputs of a variance-gamma model.
pub struct VarianceGammaProcess {
    spot: Handle<dyn Quote>,
    dividend_yield: Handle<dyn YieldTermStructure>,
    risk_free_rate: Handle<dyn YieldTermStructure>,
    sigma: Real,
    nu: Real,
    theta: Real,
    observable: Shared<Observable>,
    _listener: SharedMut<ResetThenNotify>,
}

impl VarianceGammaProcess {
    /// `VarianceGammaProcess(s0, dividendYield, riskFreeRate, sigma, nu, theta)`.
    ///
    /// # Errors
    ///
    /// Rejects a non-positive volatility or gamma variance rate, a non-finite
    /// drift, or a compensator `1 - θν - σ²ν/2` that is not strictly positive.
    /// Those are the inputs for which the analytic integrand is not a real price.
    pub fn new(
        spot: Handle<dyn Quote>,
        dividend_yield: Handle<dyn YieldTermStructure>,
        risk_free_rate: Handle<dyn YieldTermStructure>,
        sigma: Real,
        nu: Real,
        theta: Real,
    ) -> QlResult<Self> {
        require!(
            sigma.is_finite() && sigma > 0.0,
            "variance-gamma sigma must be finite and positive"
        );
        require!(
            nu.is_finite() && nu > 0.0,
            "variance-gamma nu must be finite and positive"
        );
        require!(theta.is_finite(), "variance-gamma theta must be finite");
        let compensator = 1.0 - theta * nu - (sigma * sigma * nu) / 2.0;
        require!(
            compensator.is_finite() && compensator > 0.0,
            "variance-gamma compensator must be positive"
        );
        let observable = shared(Observable::new());
        let listener = ResetThenNotify::broadcasting(Shared::clone(&observable), || {});
        let observer = listener.clone() as SharedMut<dyn Observer>;
        spot.register_observer(&observer);
        dividend_yield.register_observer(&observer);
        risk_free_rate.register_observer(&observer);
        Ok(Self {
            spot,
            dividend_yield,
            risk_free_rate,
            sigma,
            nu,
            theta,
            observable,
            _listener: listener,
        })
    }

    /// Brownian volatility of the variance-gamma time change.
    pub fn sigma(&self) -> Real {
        self.sigma
    }

    /// Variance rate of the gamma clock.
    pub fn nu(&self) -> Real {
        self.nu
    }

    /// Drift of the Brownian motion evaluated on the gamma clock.
    pub fn theta(&self) -> Real {
        self.theta
    }

    /// Current spot.
    ///
    /// # Errors
    ///
    /// Rejects a missing or non-positive spot.
    pub fn x0(&self) -> QlResult<Real> {
        let spot = self.spot.current_link()?.value()?;
        require!(
            spot.is_finite() && spot > 0.0,
            "negative or null underlying given"
        );
        Ok(spot)
    }

    /// Dividend yield handle.
    pub fn dividend_yield(&self) -> Handle<dyn YieldTermStructure> {
        self.dividend_yield.clone()
    }

    /// Risk-free yield handle.
    pub fn risk_free_rate(&self) -> Handle<dyn YieldTermStructure> {
        self.risk_free_rate.clone()
    }

    /// Path drift is the C++ `QL_FAIL("not implemented yet")`.
    ///
    /// # Errors
    ///
    /// Always returns that failure.
    pub fn drift(&self) -> QlResult<Real> {
        fail!("not implemented yet")
    }

    /// Path diffusion is the C++ `QL_FAIL("not implemented yet")`.
    ///
    /// # Errors
    ///
    /// Always returns that failure.
    pub fn diffusion(&self) -> QlResult<Real> {
        fail!("not implemented yet")
    }
}

impl AsObservable for VarianceGammaProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}
