//! Constant-parameter Gaussian short-rate state process in a forward measure.
//!
//! This is the standalone process, not a GSR model or calibration engine.

use crate::errors::QlResult;
use crate::fail;
use crate::patterns::observable::{AsObservable, Observable};
use crate::processes::forwardmeasureprocess::{ForwardMeasureProcess1D, ForwardMeasureTime};
use crate::processes::gsrprocesscore::GsrProcessCore;
use crate::require;
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Time};

enum Law {
    Constant { a: Real, sigma: Real },
    Piecewise(GsrProcessCore),
}

/// Constant-volatility, constant-reversion GSR process under a `T`-forward measure.
///
/// Dynamics are `dx = (y(t) - sigma² G(t,T) - a x) dt + sigma dW`, with
/// `y(t) = sigma² (1-exp(-2at))/(2a)` and `G(t,T) = (1-exp(-a(T-t)))/a`.
/// Zero reversion uses the continuous limits, without a small-reversion cutoff.
/// All dynamics require finite state and times in `[0,T]`. Date conversion is
/// unsupported. Constant grids keep the closed form. Varying volatility or
/// reversion uses the piecewise core. Changing the horizon through
/// [`ForwardMeasureProcess1D`] immediately changes drift and transition mean.
pub struct GsrProcess {
    law: Law,
    measure: ForwardMeasureTime,
    observable: Observable,
}

impl GsrProcess {
    /// Constructs the constant-parameter subset of QuantLib's grid-shaped API.
    ///
    /// Repeated identical parameters use the constant closed form. Varying
    /// finite nonnegative volatilities and finite reversions use the piecewise
    /// core. Redundant constant-grid boundaries do not restrict subsequent
    /// horizon changes, including changes to zero.
    ///
    /// # Errors
    /// Rejects invalid grid lengths, nonfinite/negative or non-increasing times,
    /// nonfinite/negative volatility, nonfinite reversion, and a nonfinite or
    /// negative measure maturity. Signed reversion is supported.
    pub fn new(
        times: Vec<Time>,
        vols: Vec<Real>,
        reversions: Vec<Real>,
        t_measure: Time,
    ) -> QlResult<Self> {
        require!(
            vols.len().checked_sub(1) == Some(times.len()),
            "number of volatilities must exceed number of times by one"
        );
        require!(
            reversions.len() == 1 || reversions.len() == vols.len(),
            "reversions must contain one value or match the volatility count"
        );
        require!(
            times.iter().all(|t| t.is_finite() && *t >= 0.0)
                && times.windows(2).all(|p| p[0] < p[1]),
            "grid times must be finite, nonnegative and strictly increasing"
        );
        let sigma = vols[0];
        let a = reversions[0];
        require!(
            vols.iter().all(|v| v.is_finite() && *v >= 0.0),
            "volatility must be finite and nonnegative"
        );
        require!(
            reversions.iter().all(|v| v.is_finite()),
            "reversion must be finite"
        );
        let constant = vols.iter().all(|v| *v == sigma) && reversions.iter().all(|v| *v == a);
        let law = if constant {
            Law::Constant { a, sigma }
        } else {
            Law::Piecewise(GsrProcessCore::new(times, vols, reversions))
        };
        Ok(Self {
            law,
            measure: ForwardMeasureTime::new(t_measure)?,
            observable: Observable::new(),
        })
    }

    /// Returns constant instantaneous volatility.
    ///
    /// # Errors
    /// Rejects time outside the finite interval `[0,T]`.
    pub fn sigma(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(match &self.law {
            Law::Constant { sigma, .. } => *sigma,
            Law::Piecewise(core) => core.sigma(t),
        })
    }

    /// Returns signed mean reversion, piecewise when the grid varies.
    ///
    /// # Errors
    /// Rejects time outside the finite interval `[0,T]`.
    pub fn reversion(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        Ok(match &self.law {
            Law::Constant { a, .. } => *a,
            Law::Piecewise(core) => core.reversion(t),
        })
    }

    /// Returns the auxiliary variance `y(t)`.
    ///
    /// # Errors
    /// Rejects invalid time or a nonfinite result.
    pub fn y(&self, t: Time) -> QlResult<Real> {
        self.check_t(t)?;
        match &self.law {
            Law::Constant { .. } => self.integrated_variance(t),
            Law::Piecewise(core) => finite(core.y(t, self.measure.get())),
        }
    }

    /// Returns the affine kernel `G(t,end)` independently of state.
    ///
    /// # Errors
    /// Requires `0 <= t <= end <= T` and a finite result.
    pub fn g(&self, t: Time, end: Time) -> QlResult<Real> {
        self.check_t(t)?;
        self.check_t(end)?;
        require!(
            end.partial_cmp(&t).is_some_and(|order| order.is_ge()),
            "G requires ordered endpoints"
        );
        match &self.law {
            Law::Constant { a, .. } => finite(kernel(*a, end - t)),
            Law::Piecewise(core) => finite(core.g(t, end, self.measure.get())),
        }
    }

    fn check_t(&self, t: Time) -> QlResult<()> {
        require!(
            t.is_finite() && t >= 0.0 && t <= self.measure.get(),
            "time must be finite and within the forward-measure horizon"
        );
        Ok(())
    }

    fn state(&self, t: Time, x: Real) -> QlResult<()> {
        self.check_t(t)?;
        require!(x.is_finite(), "state must be finite");
        Ok(())
    }

    fn step(&self, t: Time, x: Real, dt: Time) -> QlResult<Time> {
        self.state(t, x)?;
        require!(
            dt.is_finite() && dt >= 0.0,
            "step must be finite and nonnegative"
        );
        let end = t + dt;
        self.check_t(end)?;
        Ok(end)
    }

    fn constant(&self) -> Option<(Real, Real)> {
        match &self.law {
            Law::Constant { a, sigma } => Some((*a, *sigma)),
            Law::Piecewise(_) => None,
        }
    }

    fn integrated_variance(&self, dt: Time) -> QlResult<Real> {
        let Some((a, sigma)) = self.constant() else {
            fail!("integrated variance is only the constant closed form");
        };
        if dt == 0.0 || sigma == 0.0 {
            return Ok(0.0);
        }
        let z = (-a * dt) * 2.0;
        let log_integral = if a == 0.0 || z == 0.0 {
            dt.ln()
        } else if a > 0.0 {
            (-z.exp_m1()).ln() - a.ln() - 2.0_f64.ln()
        } else {
            z + (-(-z).exp_m1()).ln() - (-a).ln() - 2.0_f64.ln()
        };
        let integral = if z == 0.0 {
            dt
        } else {
            (-z.exp_m1() / a) * 0.5
        };
        let value = sigma * (sigma * integral);
        if value.is_normal() && integral.is_normal() && (sigma * integral).is_normal() {
            return Ok(value);
        }
        finite((2.0 * sigma.ln() + log_integral).exp())
    }
}

impl AsObservable for GsrProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl ForwardMeasureProcess1D for GsrProcess {
    fn forward_measure_state(&self) -> &ForwardMeasureTime {
        &self.measure
    }
}

impl StochasticProcess1D for GsrProcess {
    fn x0(&self) -> QlResult<Real> {
        Ok(0.0)
    }

    fn drift(&self, t: Time, x: Real) -> QlResult<Real> {
        self.state(t, x)?;
        let horizon = self.measure.get();
        match &self.law {
            Law::Constant { a, sigma } => {
                let adjustment = if *sigma == 0.0 {
                    0.0
                } else {
                    sigma * (sigma * self.g(t, horizon)?)
                };
                finite(self.y(t)? - adjustment - a * x)
            }
            Law::Piecewise(core) => {
                let sigma = core.sigma(t);
                finite(
                    core.y(t, horizon)
                        - sigma * sigma * core.g(t, horizon, horizon)
                        - core.reversion(t) * x,
                )
            }
        }
    }

    fn diffusion(&self, t: Time, x: Real) -> QlResult<Real> {
        self.state(t, x)?;
        self.sigma(t)
    }

    fn expectation(&self, t: Time, x: Real, dt: Time) -> QlResult<Real> {
        self.step(t, x, dt)?;
        if dt == 0.0 {
            return Ok(x);
        }
        let horizon = self.measure.get();
        match &self.law {
            Law::Constant { a, sigma } => {
                let decay = (-a * dt).exp();
                let initial = if x == 0.0 { 0.0 } else { x * decay };
                if *sigma == 0.0 {
                    return finite(initial);
                }
                let end = t + dt;
                let duration = kernel(*a, dt);
                let remaining = self.g(end, horizon)?;
                let adjustment = (sigma * duration) * (sigma * remaining) * (0.5 + 0.5 * decay);
                finite(initial + duration * (decay * self.y(t)?) - adjustment)
            }
            Law::Piecewise(core) => finite(core.expectation(t, x, dt, horizon)),
        }
    }

    fn variance(&self, t: Time, x: Real, dt: Time) -> QlResult<Real> {
        self.step(t, x, dt)?;
        match &self.law {
            Law::Constant { .. } => self.integrated_variance(dt),
            Law::Piecewise(core) => finite(core.variance(t, dt, self.measure.get())),
        }
    }

    fn std_deviation(&self, t: Time, x: Real, dt: Time) -> QlResult<Real> {
        Ok(self.variance(t, x, dt)?.sqrt())
    }

    fn evolve(&self, t: Time, x: Real, dt: Time, dw: Real) -> QlResult<Real> {
        require!(dw.is_finite(), "Gaussian increment must be finite");
        finite(self.expectation(t, x, dt)? + self.std_deviation(t, x, dt)? * dw)
    }
}

fn kernel(a: Real, dt: Time) -> Real {
    let z = -a * dt;
    if a == 0.0 || z == 0.0 {
        dt
    } else {
        -z.exp_m1() / a
    }
}

fn finite(value: Real) -> QlResult<Real> {
    require!(
        value.is_finite(),
        "process result is not representable as finite binary64"
    );
    Ok(value)
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
