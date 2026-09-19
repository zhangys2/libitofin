//! Markov-functional state process.
//!
//! Port of `ql/processes/mfstateprocess.{hpp,cpp}`: piecewise-constant
//! volatility process used by the Markov functional model. The SDE is
//! notionally `dx = σ(t) e^{a t} dW(t)`, but matching QuantLib,
//! [`diffusion`](StochasticProcess1D::diffusion) returns the piecewise σ(t)
//! only — the `e^{a t}` factor is carried by the analytic
//! [`variance`](StochasticProcess1D::variance) / `std_deviation` path.
//! Empty `times` use QL's unit-σ closed form (the supplied `vols[0]` is not
//! applied). First Markov-functional / Gaussian-1D gap slice; model, smile,
//! and engines remain deferred.

use crate::errors::QlResult;
use crate::patterns::observable::{AsObservable, Observable};
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::StochasticProcess1D;
use crate::types::{Real, Size, Time};

/// Markov functional state process (`mfstateprocess.hpp`).
pub struct MfStateProcess {
    reversion: Real,
    reversion_zero: bool,
    times: Vec<Real>,
    vols: Vec<Real>,
    observable: Shared<Observable>,
}

impl MfStateProcess {
    /// `MfStateProcess(reversion, times, vols)`.
    ///
    /// # Errors
    ///
    /// Fails when `vols.len() != times.len() + 1`, times are not strictly
    /// increasing, or any volatility is negative.
    pub fn new(reversion: Real, times: Vec<Real>, vols: Vec<Real>) -> QlResult<Self> {
        let reversion_zero = reversion.abs() < Real::EPSILON;
        let process = Self {
            reversion,
            reversion_zero,
            times,
            vols,
            observable: shared(Observable::new()),
        };
        process.check_times_vols()?;
        Ok(process)
    }

    fn check_times_vols(&self) -> QlResult<()> {
        require!(
            self.times.len() + 1 == self.vols.len(),
            "number of volatilities ({}) compared to number of times ({}) must be bigger by one",
            self.vols.len(),
            self.times.len()
        );
        for i in 0..self.times.len().saturating_sub(1) {
            require!(
                self.times[i] < self.times[i + 1],
                "times must be increasing ({}@{}, {}@{})",
                self.times[i],
                i,
                self.times[i + 1],
                i + 1
            );
        }
        for (i, &v) in self.vols.iter().enumerate() {
            require!(v >= 0.0, "volatilities must be non negative ({v}@{i})");
        }
        Ok(())
    }

    /// Index `i` such that `times[i-1] ≤ t < times[i]` via C++ `upper_bound`.
    fn bucket(&self, t: Time) -> Size {
        self.times.partition_point(|&x| x <= t)
    }

    fn segment_start(&self, k: Size) -> Time {
        if k > 0 { self.times[k - 1] } else { 0.0 }
    }
}

impl AsObservable for MfStateProcess {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl StochasticProcess1D for MfStateProcess {
    fn x0(&self) -> QlResult<Real> {
        Ok(0.0)
    }

    fn drift(&self, _t: Time, _x: Real) -> QlResult<Real> {
        Ok(0.0)
    }

    /// Instantaneous σ(t) (piecewise); does **not** include `e^{a t}`.
    fn diffusion(&self, t: Time, _x: Real) -> QlResult<Real> {
        Ok(self.vols[self.bucket(t)])
    }

    fn expectation(&self, _t0: Time, x0: Real, _dt: Time) -> QlResult<Real> {
        Ok(x0)
    }

    fn std_deviation(&self, t0: Time, x0: Real, dt: Time) -> QlResult<Real> {
        Ok(self.variance(t0, x0, dt)?.sqrt())
    }

    fn variance(&self, t: Time, _x0: Real, dt: Time) -> QlResult<Real> {
        if dt < Real::EPSILON {
            return Ok(0.0);
        }
        // Empty times: QL closed form assumes unit σ (vols[0] is ignored).
        if self.times.is_empty() {
            return Ok(if self.reversion_zero {
                dt
            } else {
                (1.0 / (2.0 * self.reversion))
                    * ((2.0 * self.reversion * (t + dt)).exp() - (2.0 * self.reversion * t).exp())
            });
        }

        let i = self.bucket(t);
        let j = self.bucket(t + dt);
        let mut v = 0.0;

        for k in i..j {
            let left = self.segment_start(k).max(t);
            let right = self.times[k];
            let sigma = self.vols[k];
            if self.reversion_zero {
                v += sigma * sigma * (right - left);
            } else {
                v += (1.0 / (2.0 * self.reversion))
                    * sigma
                    * sigma
                    * ((2.0 * self.reversion * right).exp() - (2.0 * self.reversion * left).exp());
            }
        }

        let left = self.segment_start(j).max(t);
        let right = t + dt;
        let sigma = self.vols[j];
        if self.reversion_zero {
            v += sigma * sigma * (right - left);
        } else {
            v += (1.0 / (2.0 * self.reversion))
                * sigma
                * sigma
                * ((2.0 * self.reversion * right).exp() - (2.0 * self.reversion * left).exp());
        }
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `markovfunctional.cpp` `testMfStateProcess`.
    #[test]
    fn mf_state_process() {
        const TOL: Real = 1.0e-10;

        let sp1 = MfStateProcess::new(0.00, vec![], vec![1.0]).unwrap();
        let var11 = sp1.variance(0.0, 0.0, 1.0).unwrap();
        let var12 = sp1.variance(0.0, 0.0, 2.0).unwrap();
        assert!(
            (var11 - 1.0).abs() <= TOL,
            "process 1 variance dt=1.0: {var11}"
        );
        assert!(
            (var12 - 2.0).abs() <= TOL,
            "process 1 variance dt=2.0: {var12}"
        );

        let times2 = vec![1.0, 2.0];
        let vols2 = vec![1.0, 2.0, 3.0];
        let sp2 = MfStateProcess::new(0.00, times2.clone(), vols2.clone()).unwrap();
        let diffs = [
            (0.0, 1.0),
            (0.99, 1.0),
            (1.0, 2.0),
            (1.9, 2.0),
            (2.0, 3.0),
            (3.0, 3.0),
            (5.0, 3.0),
        ];
        for (t, expected) in diffs {
            let d = sp2.diffusion(t, 0.0).unwrap();
            assert!(
                (d - expected).abs() <= TOL,
                "process 2 diffusion at {t}: {d} vs {expected}"
            );
        }
        let vars2 = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.5),
            (0.0, 1.0, 1.0),
            (0.0, 1.5, 3.0),
            (0.0, 3.0, 14.0),
            (0.0, 5.0, 32.0),
            (1.2, 1.0, 5.0),
        ];
        for (t0, dt, expected) in vars2 {
            let v = sp2.variance(t0, 0.0, dt).unwrap();
            assert!(
                (v - expected).abs() <= TOL,
                "process 2 variance t0={t0} dt={dt}: {v} vs {expected}"
            );
        }

        let sp3 = MfStateProcess::new(0.01, times2, vols2).unwrap();
        for (t, expected) in diffs {
            let d = sp3.diffusion(t, 0.0).unwrap();
            assert!(
                (d - expected).abs() <= TOL,
                "process 3 diffusion at {t}: {d} vs {expected} (raw σ, no e^{{at}})"
            );
        }
        let vars3 = [
            (0.0, 0.0, 0.0),
            (0.0, 0.5, 0.502508354208),
            (0.0, 1.0, 1.01006700134),
            (0.0, 1.5, 3.06070578669),
            (0.0, 3.0, 14.5935513933),
            (0.0, 5.0, 34.0940185819),
            (1.2, 1.0, 5.18130257358),
        ];
        for (t0, dt, expected) in vars3 {
            let v = sp3.variance(t0, 0.0, dt).unwrap();
            assert!(
                (v - expected).abs() <= TOL,
                "process 3 variance t0={t0} dt={dt}: {v} vs {expected}"
            );
        }

        // Empty times: QL variance ignores vols[0] (unit-σ closed form).
        let sp_unit = MfStateProcess::new(0.0, vec![], vec![2.0]).unwrap();
        let v_unit = sp_unit.variance(0.0, 0.0, 1.0).unwrap();
        assert!(
            (v_unit - 1.0).abs() <= TOL,
            "empty-times variance must stay unit-σ: {v_unit}"
        );
    }
}
