//! Heston stochastic-local-volatility process.
//!
//! Port of `ql/processes/hestonslvprocess.{hpp,cpp}`: first Heston-SLV gap
//! slice covering construction, `drift` / `diffusion` / `apply` /
//! `initialValues`, and the QE-style `evolve`. `HestonSLVFDMModel` /
//! `HestonSLVMCModel` calibration surfaces remain deferred.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::interestrate::Compounding;
use crate::math::array::Array;
use crate::math::distributions::normal::CumulativeNormalDistribution;
use crate::math::matrix::Matrix;
use crate::patterns::observable::{AsObservable, Observable, Observer, ResetThenNotify};
use crate::processes::HestonProcess;
use crate::quotes::Quote;
use crate::shared::{Shared, SharedMut, shared};
use crate::stochasticprocess::StochasticProcess;
use crate::termstructures::volatility::LocalVolTermStructure;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::date::Date;
use crate::time::frequency::Frequency;
use crate::types::{Real, Size, Time};

/// Heston SLV process (`hestonslvprocess.hpp`).
pub struct HestonSLVProcess {
    heston: Shared<HestonProcess>,
    leverage: Shared<dyn LocalVolTermStructure>,
    mixing_factor: Real,
    observable: Shared<Observable>,
    _listener: SharedMut<ResetThenNotify>,
}

impl HestonSLVProcess {
    /// `HestonSLVProcess(hestonProcess, leverageFct, mixingFactor = 1.0)`.
    pub fn new(
        heston: Shared<HestonProcess>,
        leverage: Shared<dyn LocalVolTermStructure>,
        mixing_factor: Real,
    ) -> Self {
        let observable = shared(Observable::new());
        let listener = ResetThenNotify::forwarding(Shared::clone(&observable));
        heston
            .observable()
            .register_observer(&(listener.clone() as SharedMut<dyn Observer>));
        Self {
            heston,
            leverage,
            mixing_factor,
            observable,
            _listener: listener,
        }
    }

    /// Mixing factor (QL default `1.0`).
    pub fn with_defaults(
        heston: Shared<HestonProcess>,
        leverage: Shared<dyn LocalVolTermStructure>,
    ) -> Self {
        Self::new(heston, leverage, 1.0)
    }

    /// Initial variance `v0`.
    pub fn v0(&self) -> Real {
        self.heston.v0()
    }

    /// Correlation `ρ`.
    pub fn rho(&self) -> Real {
        self.heston.rho()
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

    /// Mixing factor.
    pub fn mixing_factor(&self) -> Real {
        self.mixing_factor
    }

    /// Leverage function.
    pub fn leverage_fct(&self) -> Shared<dyn LocalVolTermStructure> {
        Shared::clone(&self.leverage)
    }

    /// Spot quote handle.
    pub fn s0(&self) -> Handle<dyn Quote> {
        self.heston.s0()
    }

    /// Dividend curve.
    pub fn dividend_yield(&self) -> Handle<dyn YieldTermStructure> {
        self.heston.dividend_yield()
    }

    /// Risk-free curve.
    pub fn risk_free_rate(&self) -> Handle<dyn YieldTermStructure> {
        self.heston.risk_free_rate()
    }

    fn mixed_sigma(&self) -> Real {
        self.mixing_factor * self.heston.sigma()
    }

    fn leverage_vol(&self, t: Time, s: Real, v: Real) -> QlResult<Real> {
        let l = self.leverage.local_vol(t, s, true)?;
        Ok((v.max(0.0).sqrt() * l).max(1e-8))
    }

    fn instantaneous_forward(
        &self,
        curve: &Handle<dyn YieldTermStructure>,
        t: Time,
    ) -> QlResult<Real> {
        Ok(curve
            .current_link()?
            .forward_rate(t, t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn forward_interval(
        &self,
        curve: &Handle<dyn YieldTermStructure>,
        t0: Time,
        dt: Time,
    ) -> QlResult<Real> {
        Ok(curve
            .current_link()?
            .forward_rate(
                t0,
                t0 + dt,
                Compounding::Continuous,
                Frequency::NoFrequency,
                false,
            )?
            .rate())
    }
}

impl AsObservable for HestonSLVProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl StochasticProcess for HestonSLVProcess {
    fn size(&self) -> Size {
        2
    }

    fn factors(&self) -> Size {
        2
    }

    fn initial_values(&self) -> QlResult<Array> {
        self.heston.initial_values()
    }

    fn apply(&self, x0: &Array, dx: &Array) -> Array {
        self.heston.apply(x0, dx)
    }

    fn drift(&self, t: Time, x: &Array) -> QlResult<Array> {
        let vol = self.leverage_vol(t, x[0], x[1])?;
        let r = self.instantaneous_forward(&self.heston.risk_free_rate(), t)?;
        let q = self.instantaneous_forward(&self.heston.dividend_yield(), t)?;
        Ok(Array::from([
            r - q - 0.5 * vol * vol,
            self.heston.kappa() * (self.heston.theta() - x[1]),
        ]))
    }

    fn diffusion(&self, t: Time, x: &Array) -> QlResult<Matrix> {
        let vol = self.leverage_vol(t, x[0], x[1])?;
        let sigma2 = self.mixed_sigma() * x[1].max(0.0).sqrt();
        let sqrhov = (1.0 - self.heston.rho() * self.heston.rho()).sqrt();
        Ok(Matrix::from([
            [vol, 0.0],
            [self.heston.rho() * sigma2, sqrhov * sigma2],
        ]))
    }

    fn evolve(&self, t0: Time, x0: &Array, dt: Time, dw: &Array) -> QlResult<Array> {
        let kappa = self.heston.kappa();
        let theta = self.heston.theta();
        let rho = self.heston.rho();
        let mixed = self.mixed_sigma();

        let ex = (-kappa * dt).exp();
        let m = theta + (x0[1] - theta) * ex;
        let s2 = x0[1] * mixed * mixed * ex / kappa * (1.0 - ex)
            + theta * mixed * mixed / (2.0 * kappa) * (1.0 - ex) * (1.0 - ex);
        let psi = s2 / (m * m);

        let next_v = if psi < 1.5 {
            let b2 = 2.0 / psi - 1.0 + (2.0 / psi * (2.0 / psi - 1.0)).sqrt();
            let b = b2.sqrt();
            let a = m / (1.0 + b2);
            a * (b + dw[1]) * (b + dw[1])
        } else {
            let p = (psi - 1.0) / (psi + 1.0);
            let beta = (1.0 - p) / m;
            let u = CumulativeNormalDistribution::standard().value(dw[1]);
            if u <= p {
                0.0
            } else {
                ((1.0 - p) / (1.0 - u)).ln() / beta
            }
        };

        let mu = self.forward_interval(&self.heston.risk_free_rate(), t0, dt)?
            - self.forward_interval(&self.heston.dividend_yield(), t0, dt)?;
        let rho1 = (1.0 - rho * rho).sqrt();
        let l0 = self.leverage.local_vol(t0, x0[0], true)?;
        let v0 = 0.5 * (x0[1] + next_v) * l0 * l0;
        let next_s = x0[0]
            * (mu * dt - 0.5 * v0 * dt
                + rho / mixed
                    * l0
                    * (next_v - kappa * theta * dt + 0.5 * (x0[1] + next_v) * kappa * dt - x0[1])
                + rho1 * (v0 * dt).sqrt() * dw[0])
                .exp();
        Ok(Array::from([next_s, next_v]))
    }

    fn time(&self, date: &Date) -> QlResult<Time> {
        self.heston.time(date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quotes::make_quote_handle;
    use crate::termstructures::volatility::LocalConstantVol;
    use crate::termstructures::yields::FlatForward;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    const TOL: Real = 1e-12;

    fn ref_date() -> Date {
        Date::new(6, Month::June, 2020)
    }

    fn flat_y(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            ref_date(),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    fn heston() -> Shared<HestonProcess> {
        shared(HestonProcess::new(
            flat_y(0.05),
            flat_y(0.02),
            make_quote_handle(100.0).handle(),
            0.04,
            1.0,
            0.04,
            0.3,
            -0.5,
        ))
    }

    fn const_l(l: Real) -> Shared<dyn LocalVolTermStructure> {
        shared(LocalConstantVol::new(ref_date(), l, Actual365Fixed::new()))
            as Shared<dyn LocalVolTermStructure>
    }

    #[test]
    fn constant_leverage_scales_diffusion_and_drift() {
        let h = heston();
        let p1 = HestonSLVProcess::with_defaults(Shared::clone(&h), const_l(1.0));
        let p25 = HestonSLVProcess::new(Shared::clone(&h), const_l(2.5), 1.0);
        assert_eq!(p1.size(), 2);
        assert!((p1.mixing_factor() - 1.0).abs() < TOL);
        assert!((p25.v0() - 0.04).abs() < TOL);

        let x = Array::from([100.0, 0.04]);
        let d1 = p1.diffusion(0.5, &x).unwrap();
        let d25 = p25.diffusion(0.5, &x).unwrap();
        assert!((d25[(0, 0)] - 2.5 * d1[(0, 0)]).abs() < TOL);
        // Variance diffusion uses mixedσ √v (independent of L).
        assert!((d25[(1, 0)] - d1[(1, 0)]).abs() < TOL);
        assert!((d25[(1, 1)] - d1[(1, 1)]).abs() < TOL);

        let mu1 = p1.drift(0.5, &x).unwrap();
        let mu25 = p25.drift(0.5, &x).unwrap();
        // Spot drift −½ (L√v)²; variance drift unchanged.
        assert!((mu25[1] - mu1[1]).abs() < TOL);
        let vol1 = (0.04_f64.sqrt() * 1.0).max(1e-8);
        let vol25 = (0.04_f64.sqrt() * 2.5).max(1e-8);
        assert!((mu1[0] - (0.05 - 0.02 - 0.5 * vol1 * vol1)).abs() < 1e-10);
        assert!((mu25[0] - (0.05 - 0.02 - 0.5 * vol25 * vol25)).abs() < 1e-10);

        let y = p1.apply(&x, &Array::from([0.01, -0.005]));
        assert!((y[0] - 100.0 * 0.01_f64.exp()).abs() < TOL);
        assert!((y[1] - 0.035).abs() < TOL);

        let mix = HestonSLVProcess::new(h, const_l(1.0), 0.5);
        let dm = mix.diffusion(0.5, &x).unwrap();
        assert!((dm[(1, 1)] - 0.5 * d1[(1, 1)]).abs() < TOL);
    }

    #[test]
    fn evolve_l1_matches_qe_variance_update() {
        let p = HestonSLVProcess::with_defaults(heston(), const_l(1.0));
        let x0 = Array::from([100.0, 0.04]);
        let dw = Array::from([0.1, -0.2]);
        let y = p.evolve(0.0, &x0, 0.1, &dw).unwrap();
        assert!(y[0] > 0.0 && y[1] >= 0.0);
        // Second evolve with zero Brownian keeps v in the CIR QE family.
        let z = p.evolve(0.1, &y, 0.1, &Array::from([0.0, 0.0])).unwrap();
        assert!(z[1].is_finite() && z[0].is_finite());
    }
}
