//! Abcd instantaneous volatility for market models.
//!
//! Port of `ql/termstructures/volatility/abcd.{hpp,cpp}`: [`AbcdFunction`]
//! extends [`AbcdMathFunction`] with covariance / variance helpers used by
//! `marketmodel.cpp` Abcd oracles. First Generic market-model gap slice.

use crate::errors::QlResult;
use crate::math::abcdmathfunction::AbcdMathFunction;
use crate::math::comparison::close;
use crate::require;
use crate::types::{Real, Time};

/// Abcd instantaneous-volatility function (`abcd.hpp`).
#[derive(Clone, Debug)]
pub struct AbcdFunction {
    math: AbcdMathFunction,
}

impl AbcdFunction {
    /// `AbcdFunction(a, b, c, d)`.
    ///
    /// # Errors
    ///
    /// As [`AbcdMathFunction::new`].
    pub fn new(a: Real, b: Real, c: Real, d: Real) -> QlResult<Self> {
        Ok(Self {
            math: AbcdMathFunction::new(a, b, c, d)?,
        })
    }

    /// QuantLib default coefficients (`a=-0.06, b=0.17, c=0.54, d=0.17`).
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new).
    pub fn with_defaults() -> QlResult<Self> {
        Self::new(-0.06, 0.17, 0.54, 0.17)
    }

    pub fn a(&self) -> Real {
        self.math.a()
    }
    pub fn b(&self) -> Real {
        self.math.b()
    }
    pub fn c(&self) -> Real {
        self.math.c()
    }
    pub fn d(&self) -> Real {
        self.math.d()
    }

    /// `f(t)`.
    pub fn value(&self, t: Time) -> Real {
        self.math.value(t)
    }

    /// Instantaneous covariance `f(T-t) f(S-t)`.
    pub fn instantaneous_covariance(&self, t: Time, t_fix: Time, s_fix: Time) -> Real {
        self.value(t_fix - t) * self.value(s_fix - t)
    }

    /// Integrated covariance on `[t1, t2]` for fixing times `T`, `S`.
    ///
    /// # Errors
    ///
    /// Fails when `t1 > t2`.
    pub fn covariance(&self, t1: Time, t2: Time, t_fix: Time, s_fix: Time) -> QlResult<Real> {
        require!(
            t1 <= t2,
            "integration bounds ({t1},{t2}) are in reverse order"
        );
        let mut cut_off = t_fix.min(s_fix);
        if t1 >= cut_off {
            Ok(0.0)
        } else {
            cut_off = t2.min(cut_off);
            Ok(self.primitive(cut_off, t_fix, s_fix) - self.primitive(t1, t_fix, s_fix))
        }
    }

    /// Integrated variance of the `T`-fixing rate on `[t_min, t_max]`
    /// (`∫ f²`; QL `variance`). Duration normalization belongs on
    /// `volatility` when that method is ported.
    ///
    /// # Errors
    ///
    /// As [`covariance`](Self::covariance).
    pub fn variance(&self, t_min: Time, t_max: Time, t_fix: Time) -> QlResult<Real> {
        self.covariance(t_min, t_max, t_fix, t_fix)
    }

    /// Indefinite integral of instantaneous covariance (`abcd.cpp` `primitive`).
    fn primitive(&self, t: Time, t_fix: Time, s_fix: Time) -> Real {
        if t_fix < t || s_fix < t {
            return 0.0;
        }
        let a = self.math.a();
        let b = self.math.b();
        let c = self.math.c();
        let d = self.math.d();
        if close(c, 0.0) {
            let v = a + d;
            return t
                * (v * v + v * b * s_fix + v * b * t_fix - v * b * t + b * b * s_fix * t_fix
                    - 0.5 * b * b * t * (s_fix + t_fix)
                    + b * b * t * t / 3.0);
        }
        let k1 = (c * t).exp();
        let k2 = (c * s_fix).exp();
        let k3 = (c * t_fix).exp();
        (b * b
            * (-1.0 - 2.0 * c * c * s_fix * t_fix - c * (s_fix + t_fix)
                + k1 * k1
                    * (1.0
                        + c * (s_fix + t_fix - 2.0 * t)
                        + 2.0 * c * c * (s_fix - t) * (t_fix - t)))
            + 2.0
                * c
                * c
                * (2.0 * d * a * (k2 + k3) * (k1 - 1.0)
                    + a * a * (k1 * k1 - 1.0)
                    + 2.0 * c * d * d * k2 * k3 * t)
            + 2.0
                * b
                * c
                * (a * (-1.0 - c * (s_fix + t_fix)
                    + k1 * k1 * (1.0 + c * (s_fix + t_fix - 2.0 * t)))
                    - 2.0
                        * d
                        * (k3 * (1.0 + c * s_fix) + k2 * (1.0 + c * t_fix)
                            - k1 * k3 * (1.0 + c * (s_fix - t))
                            - k1 * k2 * (1.0 + c * (t_fix - t)))))
            / (4.0 * c * c * c * k2 * k3)
    }
}

/// Instantaneous covariance integrand `t ↦ f(T-t)f(S-t)` (`AbcdSquared`).
pub struct AbcdSquared {
    abcd: AbcdFunction,
    t_fix: Time,
    s_fix: Time,
}

impl AbcdSquared {
    /// `AbcdSquared(a, b, c, d, T, S)`.
    ///
    /// # Errors
    ///
    /// As [`AbcdFunction::new`].
    pub fn new(a: Real, b: Real, c: Real, d: Real, t_fix: Time, s_fix: Time) -> QlResult<Self> {
        Ok(Self {
            abcd: AbcdFunction::new(a, b, c, d)?,
            t_fix,
            s_fix,
        })
    }

    /// Instantaneous covariance at `t`.
    pub fn value(&self, t: Time) -> Real {
        self.abcd
            .instantaneous_covariance(t, self.t_fix, self.s_fix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::abcdmathfunction::AbcdMathFunction;
    use crate::math::integrals::Integrator;
    use crate::math::integrals::segment::SegmentIntegral;
    use crate::types::Size;

    /// Pin distinct QL default coefficient sets for math vs market-model Abcd.
    #[test]
    fn abcd_ql_defaults() {
        let math = AbcdMathFunction::with_defaults().unwrap();
        assert_eq!(
            (math.a(), math.b(), math.c(), math.d()),
            (0.002, 0.001, 0.16, 0.0005)
        );
        let vol = AbcdFunction::with_defaults().unwrap();
        assert_eq!(
            (vol.a(), vol.b(), vol.c(), vol.d()),
            (-0.06, 0.17, 0.54, 0.17)
        );
    }

    /// `marketmodel.cpp` `testAbcdDegenerateCases`.
    #[test]
    fn abcd_degenerate_cases() {
        let f1 = AbcdFunction::new(0.0, 0.0, 1.0e-15, 1.0).unwrap();
        let f2 = AbcdFunction::new(1.0, 0.0, 1.0e-50, 0.0).unwrap();
        let cov1 = f1.covariance(0.0, 1.0, 1.0, 1.0).unwrap();
        assert!(
            (cov1 - 1.0).abs() <= 1.0e-14 && cov1.is_finite(),
            "cov1={cov1}"
        );
        let cov2 = f2.covariance(0.0, 1.0, 1.0, 1.0).unwrap();
        assert!(
            (cov2 - 1.0).abs() <= 1.0e-14 && cov2.is_finite(),
            "cov2={cov2}"
        );
    }

    /// `marketmodel.cpp` `testAbcdVolatilityIntegration`.
    #[test]
    fn abcd_volatility_integration() {
        let (a, b, c, d) = (-0.0597, 0.1677, 0.5403, 0.1710);
        let n: Size = 10;
        let precision = 1.0e-4;
        let inst_vol = AbcdFunction::new(a, b, c, d).unwrap();
        let si = SegmentIntegral::new(20_000).unwrap();
        for i in 0..n {
            let t1 = 0.5 * (1 + i) as Real;
            for k in 0..n - i {
                let t2 = 0.5 * (1 + k) as Real;
                for j in 0..n {
                    let x_min = 0.5 * j as Real;
                    for l in 0..n - j {
                        let x_max = x_min + 0.5 * l as Real;
                        let abcd2 = AbcdSquared::new(a, b, c, d, t1, t2).unwrap();
                        let numerical = si.integrate(|t| abcd2.value(t), x_min, x_max).unwrap();
                        let analytical = inst_vol.covariance(x_min, x_max, t1, t2).unwrap();
                        assert!(
                            (analytical - numerical).abs() <= precision,
                            "T1={t1} T2={t2} xMin={x_min} xMax={x_max}: analytical={analytical} numerical={numerical}"
                        );
                        if (t1 - t2).abs() <= 0.0 {
                            let variance = inst_vol.variance(x_min, x_max, t1).unwrap();
                            assert!(
                                (analytical - variance).abs() <= 1.0e-14,
                                "variance mismatch {variance} vs {analytical}"
                            );
                        }
                    }
                }
            }
        }
    }
}
