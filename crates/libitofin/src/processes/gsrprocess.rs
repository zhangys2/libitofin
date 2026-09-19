//! GSR (Gaussian short-rate) state process in a `T`-forward measure.
//!
//! Port of `ql/processes/gsrprocess.{hpp,cpp}` for **constant** reversion and
//! volatility (first rates GSR gap slice). Piecewise `GsrProcessCore` is
//! deferred; the constructor still accepts QL-shaped grids but requires
//! constant σ and a across intervals.

use crate::errors::QlResult;
use crate::patterns::observable::{AsObservable, Observable};
use crate::processes::forwardmeasureprocess::{ForwardMeasureProcess1D, ForwardMeasureTime};
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Time};

/// GSR stochastic process (`gsrprocess.hpp`) — constant a/σ slice.
pub struct GsrProcess {
    a: Real,
    sigma: Real,
    measure: ForwardMeasureTime,
    observable: Shared<Observable>,
}

impl GsrProcess {
    /// `GsrProcess(times, vols, reversions, T)`.
    ///
    /// # Errors
    ///
    /// Fails on length mismatch, non-increasing times, or non-constant params.
    pub fn new(
        times: Vec<Real>,
        vols: Vec<Real>,
        reversions: Vec<Real>,
        t_measure: Time,
    ) -> QlResult<Self> {
        require!(
            times.len() + 1 == vols.len(),
            "number of volatilities ({}) compared to number of times ({}) must be bigger by one",
            vols.len(),
            times.len()
        );
        require!(
            times.len() + 1 == reversions.len() || reversions.len() == 1,
            "number of reversions compared to number of times must be bigger by one, or exactly 1 reversion must be given"
        );
        for i in 0..times.len().saturating_sub(1) {
            require!(times[i] < times[i + 1], "times must be increasing");
        }
        let sigma = vols[0];
        require!(
            vols.iter().all(|&v| v == sigma),
            "piecewise GSR volatilities deferred (constant σ only)"
        );
        let a = reversions[0];
        require!(
            reversions.iter().all(|&r| r == a),
            "piecewise GSR reversions deferred (constant a only)"
        );
        Ok(Self {
            a,
            sigma,
            measure: ForwardMeasureTime::new(t_measure),
            observable: shared(Observable::new()),
        })
    }

    /// Instantaneous volatility at `t`.
    pub fn sigma(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(self.sigma)
    }

    /// Instantaneous mean reversion at `t`.
    pub fn reversion(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(self.a)
    }

    /// Auxiliary variance function `y(t)`.
    pub fn y(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(self.y_impl(t))
    }

    /// Kernel `G(t, w)` used by the forward-measure drift.
    pub fn g(&self, t: Time, t_end: Time) -> QlResult<Real> {
        require!(
            t_end >= t,
            "G(t,w) should be called with w not lesser than t"
        );
        require!(
            t >= 0.0 && t_end <= self.measure.get(),
            "G(t,w) outside [0, T]"
        );
        Ok(self.g_impl(t, t_end))
    }

    fn check_t(&self, t: Time) -> QlResult<()> {
        require!(
            t <= self.measure.get() && t >= 0.0,
            "t ({t}) must not be greater than forward measure time ({}) and non-negative",
            self.measure.get()
        );
        Ok(())
    }

    fn y_impl(&self, t: Time) -> Real {
        if self.a.abs() < 1.0e-4 {
            self.sigma * self.sigma * t
        } else {
            self.sigma * self.sigma / (2.0 * self.a) * (1.0 - (-2.0 * self.a * t).exp())
        }
    }

    fn g_impl(&self, t: Time, w: Time) -> Real {
        if self.a.abs() < 1.0e-4 {
            w - t
        } else {
            (1.0 - (-self.a * (w - t)).exp()) / self.a
        }
    }

    /// Single-regime `expectation_rn_part` + `expectation_tf_part` (`gsrprocesscore.cpp`).
    fn expectation_rn_tf(&self, w: Time, dt: Time) -> Real {
        let t = w + dt;
        let t_m = self.measure.get();
        let a = self.a;
        let s2 = self.sigma * self.sigma;
        if a.abs() < 1.0e-4 {
            let rn = s2 / 2.0 * (t * t - w * w);
            let tf = -s2
                * ((-(t - t_m).powi(2) - 2.0 * (t - w).powi(2) + (2.0 * w - t_m - t).powi(2))
                    / 4.0);
            rn + tf
        } else {
            let rn = s2 / (2.0 * a * a)
                * ((-2.0 * a * t).exp() + 1.0 - ((-a * (w + t)).exp() + (-a * (t - w)).exp()));
            let tf = -s2
                * (2.0
                    - (a * (t - t_m)).exp()
                    - (2.0 * (-a * (t - w)).exp() - (a * (2.0 * w - t_m - t)).exp()))
                / (2.0 * a * a);
            rn + tf
        }
    }
}

impl ForwardMeasureProcess1D for GsrProcess {
    fn forward_measure_time(&self) -> Time {
        self.measure.get()
    }

    fn set_forward_measure_time(&mut self, t: Time) {
        self.measure.set(t);
    }
}

impl AsObservable for GsrProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl StochasticProcess1D for GsrProcess {
    fn x0(&self) -> QlResult<Real> {
        Ok(0.0)
    }

    fn drift(&self, t: Time, x: Real) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(self.y_impl(t)
            - self.g_impl(t, self.measure.get()) * self.sigma * self.sigma
            - self.a * x)
    }

    fn diffusion(&self, t: Time, _x: Real) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(self.sigma)
    }

    fn expectation(&self, w: Time, xw: Real, dt: Time) -> QlResult<Real> {
        self.check_t(w + dt)?;
        Ok((-self.a * dt).exp() * xw + self.expectation_rn_tf(w, dt))
    }

    fn variance(&self, w: Time, _xw: Real, dt: Time) -> QlResult<Real> {
        self.check_t(w + dt)?;
        if self.a.abs() < 1.0e-4 {
            Ok(self.sigma * self.sigma * dt)
        } else {
            Ok(self.sigma * self.sigma / (2.0 * self.a) * (1.0 - (-2.0 * self.a * dt).exp()))
        }
    }

    fn std_deviation(&self, t0: Time, x0: Real, dt: Time) -> QlResult<Real> {
        Ok(self.variance(t0, x0, dt)?.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hull–White forward E with flat zero curve (independent of GSR algebra).
    fn hw_forward_expectation(
        a: Real,
        sigma: Real,
        t_m: Time,
        w: Time,
        xw: Real,
        dt: Time,
    ) -> Real {
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

    fn hw_forward_variance(a: Real, sigma: Real, dt: Time) -> Real {
        0.5 * sigma * sigma / a * (1.0 - (-2.0 * a * dt).exp())
    }

    /// `gsr.cpp` `testGsrProcess`: constant a/σ GSR vs Hull–White forward.
    #[test]
    fn gsr_process_matches_hull_white_forward_constant_case() {
        let reversion = 0.01;
        let modelvol = 0.01;
        let tol = 1.0e-8;
        let mut t_measure = 10.0;
        while t_measure <= 30.0 {
            let gsr = GsrProcess::new(vec![], vec![modelvol], vec![reversion], t_measure).unwrap();
            let step_times: Vec<Real> = (1..60)
                .map(|i| i as Real * 0.5)
                .filter(|&t| t < t_measure)
                .collect();
            let n = step_times.len() + 1;
            let gsr2 =
                GsrProcess::new(step_times, vec![modelvol; n], vec![reversion; n], t_measure)
                    .unwrap();
            let mut t = 0.5;
            while t <= t_measure - 0.1 {
                let mut w = 0.0;
                while w <= t - 0.1 {
                    let mut xw = -0.1;
                    while xw <= 0.1 {
                        let dt = t - w;
                        let hw_e =
                            hw_forward_expectation(reversion, modelvol, t_measure, w, xw, dt);
                        let hw_v = hw_forward_variance(reversion, modelvol, dt);
                        for (label, p) in [("gsr", &gsr), ("gsr2", &gsr2)] {
                            let e = p.expectation(w, xw, dt).unwrap();
                            let v = p.variance(w, xw, dt).unwrap();
                            assert!(
                                (hw_e - e).abs() <= tol,
                                "E {label} T={t_measure} t={t} w={w} x={xw}: HW={hw_e} got={e}"
                            );
                            assert!(
                                (hw_v - v).abs() <= tol,
                                "V {label} T={t_measure} t={t} w={w} x={xw}: HW={hw_v} got={v}"
                            );
                        }
                        xw += 0.01;
                    }
                    w += t / 5.0;
                }
                t += t_measure / 20.0;
            }
            t_measure += 10.0;
        }
    }
}
