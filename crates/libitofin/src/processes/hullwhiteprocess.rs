//! Hull–White forward-measure process (`ql/processes/hullwhiteprocess`).
//!
//! First Hybrid Heston×HW gap slice: [`HullWhiteForwardProcess`] only. Spot
//! `HullWhiteProcess` is deferred with hybrid `evolve` / engines.

use crate::errors::QlResult;
use crate::handle::Handle;
use crate::interestrate::Compounding;
use crate::patterns::observable::{AsObservable, Observable};
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::processes::forwardmeasureprocess::{ForwardMeasureProcess1D, ForwardMeasureTime};
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::termstructures::yieldtermstructure::YieldTermStructure;
use crate::time::frequency::Frequency;
use crate::types::{Real, Time};

/// Forward-measure Hull–White process (`hullwhiteprocess.hpp`).
pub struct HullWhiteForwardProcess {
    process: OrnsteinUhlenbeckProcess,
    h: Handle<dyn YieldTermStructure>,
    a: Real,
    sigma: Real,
    measure: ForwardMeasureTime,
    observable: Shared<Observable>,
}

impl HullWhiteForwardProcess {
    /// `HullWhiteForwardProcess(Handle, a, sigma)`.
    ///
    /// # Errors
    ///
    /// Fails when the curve forward at `(0,0)` is unavailable or σ is negative.
    pub fn new(h: Handle<dyn YieldTermStructure>, a: Real, sigma: Real) -> QlResult<Self> {
        let x0 = Self::fwd(&h, 0.0)?;
        Ok(Self {
            process: OrnsteinUhlenbeckProcess::new(a, sigma, x0, 0.0)?,
            h,
            a,
            sigma,
            measure: ForwardMeasureTime::new(0.0),
            observable: shared(Observable::new()),
        })
    }

    /// Mean-reversion speed `a`.
    pub fn a(&self) -> Real {
        self.a
    }

    /// Short-rate volatility `σ`.
    pub fn sigma(&self) -> Real {
        self.sigma
    }

    /// Fitting shift `α(t)`.
    pub fn alpha(&self, t: Time) -> QlResult<Real> {
        let mut alfa = if self.a > f64::EPSILON {
            (self.sigma / self.a) * (1.0 - (-self.a * t).exp())
        } else {
            self.sigma * t
        };
        alfa = 0.5 * alfa * alfa;
        Ok(alfa + Self::fwd(&self.h, t)?)
    }

    /// Affine factor `B(t, T)`.
    pub fn b(&self, t: Time, t_end: Time) -> Real {
        if self.a > f64::EPSILON {
            (1.0 - (-self.a * (t_end - t)).exp()) / self.a
        } else {
            t_end - t
        }
    }

    /// Forward-measure adjustment `M_T(s, t, T)`.
    pub fn m_t(&self, s: Real, t: Real, t_measure: Real) -> Real {
        if self.a > f64::EPSILON {
            let coeff = (self.sigma * self.sigma) / (self.a * self.a);
            let exp1 = (-self.a * (t - s)).exp();
            let exp2 = (-self.a * (t_measure - t)).exp();
            let exp3 = (-self.a * (t_measure + t - 2.0 * s)).exp();
            coeff * (1.0 - exp1) - 0.5 * coeff * (exp2 - exp3)
        } else {
            let coeff = (self.sigma * self.sigma) / 2.0;
            coeff * (t - s) * (2.0 * t_measure - t - s)
        }
    }

    fn fwd(h: &Handle<dyn YieldTermStructure>, t: Time) -> QlResult<Real> {
        Ok(h.current_link()?
            .forward_rate(t, t, Compounding::Continuous, Frequency::NoFrequency, false)?
            .rate())
    }

    fn alpha_drift(&self, t: Time) -> QlResult<Real> {
        // QL `hullwhiteprocess.cpp` uses unguarded σ²/(2a)(1−e^{−2at}); `a = 0`
        // makes this NaN/Inf while `alpha`/`b`/`m_t` stay finite (Ho–Lee footgun).
        let mut alpha_drift =
            self.sigma * self.sigma / (2.0 * self.a) * (1.0 - (-2.0 * self.a * t).exp());
        let shift = 0.0001;
        let f = Self::fwd(&self.h, t)?;
        let f_up = Self::fwd(&self.h, t + shift)?;
        alpha_drift += self.a * f + (f_up - f) / shift;
        Ok(alpha_drift)
    }
}

impl AsObservable for HullWhiteForwardProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl ForwardMeasureProcess1D for HullWhiteForwardProcess {
    fn forward_measure_time(&self) -> Time {
        self.measure.get()
    }

    fn set_forward_measure_time(&mut self, t: Time) {
        self.measure.set(t);
        self.observable.notify_observers();
    }
}

impl StochasticProcess1D for HullWhiteForwardProcess {
    fn x0(&self) -> QlResult<Real> {
        self.process.x0()
    }

    fn drift(&self, t: Time, x: Real) -> QlResult<Real> {
        Ok(self.process.drift(t, x)? + self.alpha_drift(t)?
            - self.b(t, self.measure.get()) * self.sigma * self.sigma)
    }

    fn diffusion(&self, t: Time, x: Real) -> QlResult<Real> {
        self.process.diffusion(t, x)
    }

    fn expectation(&self, t0: Time, x0: Real, dt: Time) -> QlResult<Real> {
        let t_m = self.measure.get();
        Ok(self.process.expectation(t0, x0, dt)? + self.alpha(t0 + dt)?
            - self.alpha(t0)? * (-self.a * dt).exp()
            - self.m_t(t0, t0 + dt, t_m))
    }

    fn std_deviation(&self, t0: Time, x0: Real, dt: Time) -> QlResult<Real> {
        self.process.std_deviation(t0, x0, dt)
    }

    fn variance(&self, t0: Time, x0: Real, dt: Time) -> QlResult<Real> {
        self.process.variance(t0, x0, dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::termstructures::yields::FlatForward;
    use crate::test_support::{Flag, as_observer};
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;

    fn flat(rate: Real) -> Handle<dyn YieldTermStructure> {
        Handle::new(shared(FlatForward::with_rate(
            Date::new(19, Month::September, 2026),
            rate,
            Actual365Fixed::new(),
            Compounding::Continuous,
            Frequency::Annual,
        )) as Shared<dyn YieldTermStructure>)
    }

    /// Independent HW-forward E with flat zero curve (`gsrprocess` helper).
    fn hw_forward_e(a: Real, sigma: Real, t_m: Time, w: Time, xw: Real, dt: Time) -> Real {
        let t = w + dt;
        let alpha = |u: Time| {
            let x = (sigma / a) * (1.0 - (-a * u).exp());
            0.5 * x * x
        };
        let c = (sigma * sigma) / (a * a);
        let m_t = c * (1.0 - (-a * dt).exp())
            - 0.5 * c * ((-a * (t_m - t)).exp() - (-a * (t_m + t - 2.0 * w)).exp());
        xw * (-a * dt).exp() + alpha(t) - alpha(w) * (-a * dt).exp() - m_t
    }

    #[test]
    fn forward_measure_pins_and_notify() {
        let a = 0.1;
        let sigma = 0.01;
        let mut p = HullWhiteForwardProcess::new(flat(0.0), a, sigma).unwrap();
        let flag = Flag::new();
        p.observable().register_observer(&as_observer(&flag));
        p.set_forward_measure_time(5.0);
        assert!(Flag::is_up(&flag));

        assert!((p.b(1.0, 5.0) - (1.0 - (-0.4_f64).exp()) / a).abs() < 1e-15);
        let mut alfa = (sigma / a) * (1.0 - (-a).exp());
        alfa = 0.5 * alfa * alfa;
        assert!((p.alpha(1.0).unwrap() - alfa).abs() < 1e-12);

        // T-forward drift shift: μ(T₂) − μ(T₁) = −(B(t,T₂) − B(t,T₁)) σ².
        let t = 1.0;
        let x = 0.02;
        p.set_forward_measure_time(4.0);
        let d4 = p.drift(t, x).unwrap();
        p.set_forward_measure_time(6.0);
        let d6 = p.drift(t, x).unwrap();
        assert!((d6 - d4 + (p.b(t, 6.0) - p.b(t, 4.0)) * sigma * sigma).abs() < 1e-12);

        let m = p.m_t(1.0, 2.0, 5.0);
        let c = (sigma * sigma) / (a * a);
        let expected_m =
            c * (1.0 - (-a).exp()) - 0.5 * c * ((-a * 3.0).exp() - (-a * (5.0 + 2.0 - 2.0)).exp());
        assert!((m - expected_m).abs() < 1e-15);

        p.set_forward_measure_time(10.0);
        let w = 1.0;
        let dt = 2.0;
        let xw = 0.03;
        let e = p.expectation(w, xw, dt).unwrap();
        assert!((e - hw_forward_e(a, sigma, 10.0, w, xw, dt)).abs() < 1e-12);
        let v = p.variance(w, xw, dt).unwrap();
        assert!((v - 0.5 * sigma * sigma / a * (1.0 - (-2.0 * a * dt).exp())).abs() < 1e-15);

        let z = HullWhiteForwardProcess::new(flat(0.0), 0.0, 0.02).unwrap();
        assert!((z.m_t(1.0, 2.0, 5.0) - (0.02 * 0.02) / 2.0 * 1.0 * 7.0).abs() < 1e-15);
    }
}
