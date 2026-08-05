//! Bates jump-diffusion process (Heston + compound Poisson log-normal jumps).
//!
//! Port of `ql/processes/batesprocess.{hpp,cpp}`: [`BatesProcess`] extends the
//! Heston square-root stochastic-volatility process with a compound Poisson jump
//! component whose jump sizes are log-normal. The analytic Bates engine
//! (`BatesEngine`) reads only the market handles, Heston parameters, and the
//! three jump parameters (`lambda`, `nu`, `delta`); Monte Carlo `evolve` is
//! deferred (needs `InverseCumulativePoisson`).

use crate::errors::QlResult;
use crate::fail;
use crate::handle::Handle;
use crate::math::array::Array;
use crate::math::matrix::Matrix;
use crate::patterns::observable::{AsObservable, Observable};
use crate::processes::hestonprocess::{Discretization, HestonProcess};
use crate::quotes::Quote;
use crate::stochasticprocess::StochasticProcess;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::types::{Real, Size, Time};

/// Square-root stochastic-volatility Bates process
/// (`batesprocess.hpp:49`): Heston plus log-normal jumps.
///
/// ```text
/// dS = (r-d-λm) S dt + √v S dW₁ + (e^J - 1) S dN
/// dv = κ (θ - v) dt + σ √v dW₂
/// dW₁ dW₂ = ρ dt
/// ω(J) = (2πδ²)^(-1/2) exp(-(J-ν)²/(2δ²))
/// ```
///
/// with `m = exp(ν + ½δ²) - 1`.
pub struct BatesProcess {
    heston: HestonProcess,
    lambda: Real,
    nu: Real,
    delta: Real,
    m: Real,
}

impl BatesProcess {
    /// `BatesProcess(...)` (`batesprocess.cpp:26-38`): builds on a Heston process
    /// with the C++ default variance scheme [`Discretization::FullTruncation`].
    ///
    /// # Note
    ///
    /// [`Discretization::FullTruncation`] `evolve` is deferred on
    /// [`HestonProcess`] (#410); Bates `evolve` is likewise deferred. The
    /// analytic engine never evolves the process.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        risk_free_rate: Handle<dyn YieldTermStructure>,
        dividend_yield: Handle<dyn YieldTermStructure>,
        s0: Handle<dyn Quote>,
        v0: Real,
        kappa: Real,
        theta: Real,
        sigma: Real,
        rho: Real,
        lambda: Real,
        nu: Real,
        delta: Real,
    ) -> BatesProcess {
        Self::with_discretization(
            risk_free_rate,
            dividend_yield,
            s0,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            lambda,
            nu,
            delta,
            Discretization::FullTruncation,
        )
    }

    /// Builds with an explicit Heston variance-discretization scheme.
    #[allow(clippy::too_many_arguments)]
    pub fn with_discretization(
        risk_free_rate: Handle<dyn YieldTermStructure>,
        dividend_yield: Handle<dyn YieldTermStructure>,
        s0: Handle<dyn Quote>,
        v0: Real,
        kappa: Real,
        theta: Real,
        sigma: Real,
        rho: Real,
        lambda: Real,
        nu: Real,
        delta: Real,
        discretization: Discretization,
    ) -> BatesProcess {
        let heston = HestonProcess::with_discretization(
            risk_free_rate,
            dividend_yield,
            s0,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            discretization,
        );
        BatesProcess {
            heston,
            lambda,
            nu,
            delta,
            m: (nu + 0.5 * delta * delta).exp() - 1.0,
        }
    }

    /// Underlying Heston process (market handles + SV parameters).
    pub fn heston(&self) -> &HestonProcess {
        &self.heston
    }

    /// Jump intensity `λ`.
    pub fn lambda(&self) -> Real {
        self.lambda
    }

    /// Mean jump size `ν` (log-space).
    pub fn nu(&self) -> Real {
        self.nu
    }

    /// Jump-size volatility `δ`.
    pub fn delta(&self) -> Real {
        self.delta
    }

    /// Compensator `m = exp(ν + ½δ²) - 1`.
    pub fn m(&self) -> Real {
        self.m
    }

    /// Initial variance `v0`.
    pub fn v0(&self) -> Real {
        self.heston.v0()
    }

    /// Mean-reversion speed `κ`.
    pub fn kappa(&self) -> Real {
        self.heston.kappa()
    }

    /// Long-run variance `θ`.
    pub fn theta(&self) -> Real {
        self.heston.theta()
    }

    /// Vol-of-vol `σ`.
    pub fn sigma(&self) -> Real {
        self.heston.sigma()
    }

    /// Spot/variance correlation `ρ`.
    pub fn rho(&self) -> Real {
        self.heston.rho()
    }

    /// Spot quote handle.
    pub fn s0(&self) -> Handle<dyn Quote> {
        self.heston.s0()
    }

    /// Dividend-yield curve handle.
    pub fn dividend_yield(&self) -> Handle<dyn YieldTermStructure> {
        self.heston.dividend_yield()
    }

    /// Risk-free-rate curve handle.
    pub fn risk_free_rate(&self) -> Handle<dyn YieldTermStructure> {
        self.heston.risk_free_rate()
    }
}

impl AsObservable for BatesProcess {
    fn observable(&self) -> &Observable {
        self.heston.observable()
    }
}

impl StochasticProcess for BatesProcess {
    fn size(&self) -> Size {
        self.heston.size()
    }

    /// `factors` (`batesprocess.cpp:65-67`): Heston factors plus two jump
    /// uniforms (Poisson count + jump size).
    fn factors(&self) -> Size {
        self.heston.factors() + 2
    }

    fn initial_values(&self) -> QlResult<Array> {
        self.heston.initial_values()
    }

    /// `drift` (`batesprocess.cpp:40-44`): Heston drift with the jump
    /// compensator `-λm` on the spot factor.
    fn drift(&self, t: Time, x: &Array) -> QlResult<Array> {
        let mut ret = self.heston.drift(t, x)?;
        ret[0] -= self.lambda * self.m;
        Ok(ret)
    }

    fn diffusion(&self, t: Time, x: &Array) -> QlResult<Matrix> {
        self.heston.diffusion(t, x)
    }

    fn apply(&self, x0: &Array, dx: &Array) -> Array {
        self.heston.apply(x0, dx)
    }

    /// `evolve` needs `InverseCumulativePoisson` (`batesprocess.cpp:46-62`);
    /// deferred with the FullTruncation Heston scheme (#410).
    fn evolve(&self, _t0: Time, _x0: &Array, _dt: Time, _dw: &Array) -> QlResult<Array> {
        fail!(
            "BatesProcess::evolve is deferred (needs InverseCumulativePoisson); analytic \
             BatesEngine never evolves the process"
        )
    }

    fn time(&self, date: &Date) -> QlResult<Time> {
        self.heston.time(date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interestrate::Compounding;
    use crate::quotes::make_quote_handle;
    use crate::shared::shared;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual360::Actual360;
    use crate::time::frequency::Frequency;

    fn reference() -> Date {
        Date::new(15, Month::June, 2026)
    }

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            reference(),
            rate,
            Actual360::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as crate::shared::Shared<dyn YieldTermStructure>)
    }

    #[test]
    fn compensator_m_matches_closed_form() {
        let nu = 0.1;
        let delta = 0.2;
        let p = BatesProcess::new(
            flat(0.05),
            flat(0.02),
            make_quote_handle(100.0).handle(),
            0.04,
            1.0,
            0.04,
            0.3,
            -0.5,
            0.5,
            nu,
            delta,
        );
        let expected = (nu + 0.5 * delta * delta).exp() - 1.0;
        assert!((p.m() - expected).abs() < 1e-15);
    }

    #[test]
    fn drift_subtracts_jump_compensator() {
        let lambda = 0.5;
        let nu = 0.0;
        let delta = 0.1;
        let p = BatesProcess::new(
            flat(0.05),
            flat(0.02),
            make_quote_handle(100.0).handle(),
            0.04,
            1.0,
            0.04,
            0.3,
            -0.5,
            lambda,
            nu,
            delta,
        );
        let x = Array::from([100.0_f64.ln(), 0.04]);
        let bates_drift = p.drift(0.0, &x).unwrap();
        let heston_drift = p.heston().drift(0.0, &x).unwrap();
        assert!((bates_drift[0] - (heston_drift[0] - lambda * p.m())).abs() < 1e-14);
        assert_eq!(bates_drift[1], heston_drift[1]);
    }

    #[test]
    fn factors_are_heston_plus_two() {
        let p = BatesProcess::new(
            flat(0.05),
            flat(0.02),
            make_quote_handle(100.0).handle(),
            0.04,
            1.0,
            0.04,
            0.3,
            -0.5,
            0.1,
            0.0,
            0.1,
        );
        assert_eq!(p.factors(), 4);
        assert_eq!(p.size(), 2);
    }
}
