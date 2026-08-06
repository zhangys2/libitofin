//! G2++ two-factor Gaussian short-rate process.
//!
//! Port of `ql/processes/g2process.{hpp,cpp}`: [`G2Process`] is the
//! multi-factor [`StochasticProcess`] for the state `(x, y)` in
//!
//! ```text
//! dx = -a x dt + σ dW¹
//! dy = -b y dt + η dW²
//! dW¹ dW² = ρ dt
//! ```
//!
//! Each factor is an exact [`OrnsteinUhlenbeckProcess`] (level 0). Instantaneous
//! diffusion uses the lower-triangular correlation root
//!
//! ```text
//! | σ           0                 |
//! | ρ σ   √(1-ρ²) η               |
//! ```
//!
//! while `std_deviation` / `covariance` use the exact OU transition stdevs and
//! the integrated correlation `newRho = H / (σ₁ σ₂)` from `g2process.cpp`.
//!
//! ## Deferred (omitted, not stubbed)
//!
//! - [`G2ForwardProcess`](https://www.quantlib.org/reference/class_quant_lib_1_1_g2_forward_process.html)
//!   (needs `ForwardMeasureProcess`).
//! - Optional term-structure / `φ(t)` state shift present on some upstream
//!   tips; the classic process keeps bare `(x, y)` and leaves `φ` on
//!   [`G2`](crate::models::shortrate::G2).

use crate::errors::QlResult;
use crate::math::array::Array;
use crate::math::matrix::Matrix;
use crate::patterns::observable::{AsObservable, Observable};
use crate::processes::OrnsteinUhlenbeckProcess;
use crate::require;
use crate::shared::{Shared, shared};
use crate::stochasticprocess::{StochasticProcess, StochasticProcess1D};
use crate::types::{Real, Size, Time};

/// Two-factor Gaussian G2++ process (`g2process.hpp:34`).
pub struct G2Process {
    x0: Real,
    y0: Real,
    a: Real,
    sigma: Real,
    b: Real,
    eta: Real,
    rho: Real,
    x_process: OrnsteinUhlenbeckProcess,
    y_process: OrnsteinUhlenbeckProcess,
    observable: Shared<Observable>,
}

impl G2Process {
    /// `G2Process(a, sigma, b, eta, rho)` (`g2process.cpp`): factors start at
    /// `(0, 0)` with OU levels 0, matching the classic QuantLib constructor.
    ///
    /// # Errors
    ///
    /// Fails when `|ρ| > 1` (so `√(1-ρ²)` is real) or when either factor
    /// volatility is negative (OU constructor).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub fn new(a: Real, sigma: Real, b: Real, eta: Real, rho: Real) -> QlResult<G2Process> {
        require!(
            (-1.0..=1.0).contains(&rho),
            "rho ({rho}) must be in [-1, 1]"
        );
        Ok(G2Process {
            x0: 0.0,
            y0: 0.0,
            a,
            sigma,
            b,
            eta,
            rho,
            x_process: OrnsteinUhlenbeckProcess::new(a, sigma, 0.0, 0.0)?,
            y_process: OrnsteinUhlenbeckProcess::new(b, eta, 0.0, 0.0)?,
            observable: shared(Observable::new()),
        })
    }

    /// Initial value of the first factor.
    pub fn x0(&self) -> Real {
        self.x0
    }

    /// Initial value of the second factor.
    pub fn y0(&self) -> Real {
        self.y0
    }

    /// Mean-reversion speed of `x`.
    pub fn a(&self) -> Real {
        self.a
    }

    /// Volatility of `x`.
    pub fn sigma(&self) -> Real {
        self.sigma
    }

    /// Mean-reversion speed of `y`.
    pub fn b(&self) -> Real {
        self.b
    }

    /// Volatility of `y`.
    pub fn eta(&self) -> Real {
        self.eta
    }

    /// Instantaneous factor correlation.
    pub fn rho(&self) -> Real {
        self.rho
    }
}

impl AsObservable for G2Process {
    fn observable(&self) -> &Observable {
        &self.observable
    }
}

impl StochasticProcess for G2Process {
    fn size(&self) -> Size {
        2
    }

    fn initial_values(&self) -> QlResult<Array> {
        Ok(Array::from([self.x0, self.y0]))
    }

    fn drift(&self, t: Time, x: &Array) -> QlResult<Array> {
        Ok(Array::from([
            self.x_process.drift(t, x[0])?,
            self.y_process.drift(t, x[1])?,
        ]))
    }

    /// Instantaneous diffusion root (`g2process.cpp`): Cholesky of the
    /// correlation matrix scaled by `(σ, η)`.
    fn diffusion(&self, _t: Time, _x: &Array) -> QlResult<Matrix> {
        Ok(Matrix::from([
            [self.sigma, 0.0],
            [
                self.rho * self.sigma,
                (1.0 - self.rho * self.rho).sqrt() * self.eta,
            ],
        ]))
    }

    fn expectation(&self, t0: Time, x0: &Array, dt: Time) -> QlResult<Array> {
        Ok(Array::from([
            self.x_process.expectation(t0, x0[0], dt)?,
            self.y_process.expectation(t0, x0[1], dt)?,
        ]))
    }

    /// Exact transition std-deviation matrix (`g2process.cpp`).
    fn std_deviation(&self, t0: Time, x0: &Array, dt: Time) -> QlResult<Matrix> {
        let sigma1 = self.x_process.std_deviation(t0, x0[0], dt)?;
        let sigma2 = self.y_process.std_deviation(t0, x0[1], dt)?;
        let expa = (-self.a * dt).exp();
        let expb = (-self.b * dt).exp();
        let h = (self.rho * self.sigma * self.eta) / (self.a + self.b) * (1.0 - expa * expb);
        let den = (0.5 * self.sigma * self.eta)
            * ((1.0 - expa * expa) * (1.0 - expb * expb) / (self.a * self.b)).sqrt();
        let new_rho = h / den;
        Ok(Matrix::from([
            [sigma1, 0.0],
            [new_rho * sigma2, (1.0 - new_rho * new_rho).sqrt() * sigma2],
        ]))
    }

    fn covariance(&self, t0: Time, x0: &Array, dt: Time) -> QlResult<Matrix> {
        let sigma = self.std_deviation(t0, x0, dt)?;
        Ok(&sigma * &sigma.transpose())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Real = 0.1;
    const SIGMA: Real = 0.01;
    const B: Real = 0.2;
    const ETA: Real = 0.008;
    const RHO: Real = -0.75;
    const DT: Time = 0.25;
    const TOL: Real = 1e-12;

    fn process() -> G2Process {
        G2Process::new(A, SIGMA, B, ETA, RHO).unwrap()
    }

    fn assert_close(actual: Real, expected: Real) {
        assert!(
            (actual - expected).abs() < TOL,
            "{actual} != {expected} (tol {TOL})"
        );
    }

    #[test]
    fn accessors_and_shape() {
        let p = process();
        assert_eq!(p.size(), 2);
        assert_eq!(p.factors(), 2);
        assert_eq!(p.a(), A);
        assert_eq!(p.sigma(), SIGMA);
        assert_eq!(p.b(), B);
        assert_eq!(p.eta(), ETA);
        assert_eq!(p.rho(), RHO);
        assert_eq!(p.x0(), 0.0);
        assert_eq!(p.y0(), 0.0);
        let x0 = p.initial_values().unwrap();
        assert_eq!(x0[0], 0.0);
        assert_eq!(x0[1], 0.0);
    }

    #[test]
    fn drift_is_mean_reverting_ou() {
        let p = process();
        let x = Array::from([0.02, -0.01]);
        let d = p.drift(0.0, &x).unwrap();
        assert_close(d[0], -A * x[0]);
        assert_close(d[1], -B * x[1]);
    }

    #[test]
    fn diffusion_is_cholesky_scaled_by_factor_vols() {
        let p = process();
        let d = p.diffusion(0.0, &Array::from([0.0, 0.0])).unwrap();
        assert_close(d[(0, 0)], SIGMA);
        assert_close(d[(0, 1)], 0.0);
        assert_close(d[(1, 0)], RHO * SIGMA);
        assert_close(d[(1, 1)], (1.0 - RHO * RHO).sqrt() * ETA);
    }

    #[test]
    fn expectation_matches_exact_ou_transition() {
        let p = process();
        let x0 = Array::from([0.015, -0.007]);
        let e = p.expectation(0.0, &x0, DT).unwrap();
        assert_close(e[0], x0[0] * (-A * DT).exp());
        assert_close(e[1], x0[1] * (-B * DT).exp());
    }

    #[test]
    fn covariance_matches_exact_ou_variances_and_integrated_cov() {
        let p = process();
        let x0 = Array::from([0.0, 0.0]);
        let c = p.covariance(0.0, &x0, DT).unwrap();

        let var_x = 0.5 * SIGMA * SIGMA / A * (1.0 - (-2.0 * A * DT).exp());
        let var_y = 0.5 * ETA * ETA / B * (1.0 - (-2.0 * B * DT).exp());
        let cov_xy = (RHO * SIGMA * ETA) / (A + B) * (1.0 - (-(A + B) * DT).exp());

        assert_close(c[(0, 0)], var_x);
        assert_close(c[(1, 1)], var_y);
        assert_close(c[(0, 1)], cov_xy);
        assert_close(c[(1, 0)], cov_xy);
    }

    #[test]
    fn zero_correlation_diagonalises_covariance() {
        let p = G2Process::new(A, SIGMA, B, ETA, 0.0).unwrap();
        let c = p.covariance(0.0, &Array::from([0.0, 0.0]), DT).unwrap();
        assert_close(c[(0, 1)], 0.0);
        assert_close(c[(1, 0)], 0.0);
        assert!(c[(0, 0)] > 0.0 && c[(1, 1)] > 0.0);
    }

    #[test]
    fn evolve_with_zero_correlation_is_independent_ou() {
        let p = G2Process::new(A, SIGMA, B, ETA, 0.0).unwrap();
        let x0 = Array::from([0.01, -0.02]);
        let dw = Array::from([0.5, -0.3]);
        let out = p.evolve(0.0, &x0, DT, &dw).unwrap();

        let ex = x0[0] * (-A * DT).exp();
        let ey = x0[1] * (-B * DT).exp();
        let sx = (0.5 * SIGMA * SIGMA / A * (1.0 - (-2.0 * A * DT).exp())).sqrt();
        let sy = (0.5 * ETA * ETA / B * (1.0 - (-2.0 * B * DT).exp())).sqrt();
        assert_close(out[0], ex + sx * dw[0]);
        assert_close(out[1], ey + sy * dw[1]);
    }

    #[test]
    fn rejects_rho_outside_unit_interval() {
        let err = G2Process::new(A, SIGMA, B, ETA, 1.1).err().unwrap();
        assert!(err.message().contains("rho"));
        let err = G2Process::new(A, SIGMA, B, ETA, -1.1).err().unwrap();
        assert!(err.message().contains("rho"));
    }

    #[test]
    fn rejects_negative_factor_volatility() {
        let err = G2Process::new(A, -1e-9, B, ETA, RHO).err().unwrap();
        assert_eq!(err.message(), "negative volatility given");
    }

    #[test]
    fn trait_is_object_safe() {
        let p = process();
        let dynamic: &dyn StochasticProcess = &p;
        assert_eq!(dynamic.size(), 2);
    }
}
