//! Beta functions.
//!
//! Port of `ql/math/beta.{hpp,cpp}`: the complete [`beta_function`] and the
//! regularized [`incomplete_beta`] (the CDF of the Beta distribution), the
//! latter via the Numerical Recipes continued-fraction expansion. Needed for
//! the Student-t CDF.
//!
//! Divergences from QuantLib: the continued fraction keeps QuantLib's
//! accuracy (`1e-16`) but its iteration limit grows with the parameters (see
//! [`incomplete_beta`]), where QuantLib's fixed limit of 100 throws for large
//! `a` or `b` (#1463); and its prefactor `x^a y^b / B(a, b)` is Boost's
//! `ibeta_power_terms`, where QuantLib's `exp(lgamma)` form cancels for large
//! `a`, `b` (#1468).
//!
//! The power terms and the Lanczos sum are adapted from Boost.Math 1.92
//! `boost/math/special_functions/beta.hpp` (Copyright John Maddock 2006,
//! Copyright Matt Borland 2024) and `lanczos.hpp` (`lanczos13m53`, Copyright
//! John Maddock 2006), distributed under the Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES.md`.

use std::f64::consts::E;

use crate::errors::QlResult;
use crate::fail;
use crate::math::gammafunction::log_gamma;
use crate::types::Real;

const ACCURACY: Real = 1e-16;
const MIN_ITERATIONS: u32 = 100;
const MAX_ITERATIONS: u32 = 100_000_000;

/// The `g` of Boost's `lanczos13m53` approximation (`lanczos.hpp`).
#[allow(clippy::excessive_precision)]
const LANCZOS_G: Real = 6.024680040776729583740234375;

/// Numerator coefficients of `lanczos13m53::lanczos_sum_expG_scaled`.
#[allow(clippy::excessive_precision)]
const LANCZOS_NUM: [Real; 13] = [
    56906521.91347156388090791033559122686859,
    103794043.1163445451906271053616070238554,
    86363131.28813859145546927288977868422342,
    43338889.32467613834773723740590533316085,
    14605578.08768506808414169982791359218571,
    3481712.15498064590882071018964774556468,
    601859.6171681098786670226533699352302507,
    75999.29304014542649875303443598909137092,
    6955.999602515376140356310115515198987526,
    449.9445569063168119446858607650988409623,
    19.51992788247617482847860966235652136208,
    0.5098416655656676188125178644804694509993,
    0.006061842346248906525783753964555936883222,
];

/// Denominator coefficients of `lanczos13m53` (the expansion of
/// `z (z + 1) ... (z + 11)`).
const LANCZOS_DENOM: [Real; 13] = [
    0.0,
    39916800.0,
    120543840.0,
    150917976.0,
    105258076.0,
    45995730.0,
    13339535.0,
    2637558.0,
    357423.0,
    32670.0,
    1925.0,
    66.0,
    1.0,
];

/// The complete beta function `B(z, w) = Γ(z)·Γ(w) / Γ(z+w)`.
///
/// # Errors
///
/// Returns an error if `z`, `w`, or `z + w` is not a valid `log_gamma`
/// argument (i.e. not strictly positive).
///
/// # Examples
///
/// ```
/// use libitofin::math::beta::beta_function;
/// // B(2, 3) = 1!·2!/4! = 1/12
/// let b = beta_function(2.0, 3.0)?;
/// assert!((b - 1.0 / 12.0).abs() < 1e-12);
/// # Ok::<(), libitofin::errors::QlError>(())
/// ```
pub fn beta_function(z: Real, w: Real) -> QlResult<Real> {
    Ok((log_gamma(z)? + log_gamma(w)? - log_gamma(z + w)?).exp())
}

/// The regularized incomplete beta function `I_x(a, b)` (the Beta(a, b) CDF).
///
/// `a` and `b` must be strictly positive and `x` must lie in `[0, 1]`.
///
/// The continued fraction runs up to
/// `min(max(100, ceil(10·sqrt(max(a, b)))), 10^8)` iterations, where
/// QuantLib's `incompleteBetaFunction` (`beta.hpp`) stops at 100 and throws.
/// Near the branch crossover the fraction needs a number of iterations
/// growing like `sqrt(max(a, b))`, and its `1e-16` accuracy is below half an
/// ulp of 1, so it stops only on an exactly unit step: isolated inputs of the
/// Student-t cdf at `ν = 1000` and `ν = 10000` near `t = -1.73`, and of the
/// Binomial cdf at `n = 100000`, failed at 100 iterations. The `10^8` cap
/// bounds the work for absurd parameters (`a = b = 1e25` errs after it); it is
/// far above the worst iterations-to-limit ratio (0.79) seen over the swept
/// Student-t (`ν` up to `1e8`) and Binomial domains. Every input that
/// converged within 100 iterations exits at the same iteration with the same
/// result. Past iteration 100 a `NaN` running product, which no later step can
/// clear, returns a distinct `NaN` error at once rather than spinning to the
/// limit.
///
/// The prefactor is Boost's `ibeta_power_terms` (#1468), where QuantLib's
/// `exp(lgamma(a+b) - lgamma(a) - lgamma(b) + a ln x + b ln(1-x))` cancels
/// for large `a`, `b` (the Binomial cdf at `n = 1e16` returned only 0 or 1).
/// Deferred: near the saddle `x = a/(a+b)` the continued fraction itself
/// cancels for `a`, `b` beyond about `1e11`, so the Binomial cdf at
/// `n = 1e12, 1e14, 1e16` is within `5e-10`, `1e-8` and `5e-7` of mpmath
/// there, not `1e-10`; a large-parameter asymptotic expansion is tracked
/// as #1475.
///
/// # Errors
///
/// Returns an error if `a <= 0`, `b <= 0`, `x` is outside `[0, 1]`, any
/// argument is `NaN`, or the continued fraction fails to converge within its
/// iteration limit.
///
/// # Examples
///
/// ```
/// use libitofin::math::beta::incomplete_beta;
/// // Beta(1, 1) is uniform, so I_x(1, 1) = x
/// let p = incomplete_beta(1.0, 1.0, 0.25)?;
/// assert!((p - 0.25).abs() < 1e-12);
/// # Ok::<(), libitofin::errors::QlError>(())
/// ```
pub fn incomplete_beta(a: Real, b: Real, x: Real) -> QlResult<Real> {
    incomplete_beta_xy(a, b, x, 1.0 - x)
}

/// [`incomplete_beta`] with the complement `y = 1 - x` supplied by the
/// caller, who can often form it exactly where `1 - x` would round (the
/// Student-t cdf near `x = 1`). `y` enters the power terms and the swapped
/// continued fraction, and `y = 0` (not `x = 1`, which a rounded `x` can
/// reach first) is the upper endpoint; the branch test stays on `x`, as in
/// [`incomplete_beta`].
///
/// # Errors
///
/// As [`incomplete_beta`], and also if `y` is outside `[0, 1]`.
pub(crate) fn incomplete_beta_xy(a: Real, b: Real, x: Real, y: Real) -> QlResult<Real> {
    debug_assert!(
        !(x.is_finite() && y.is_finite()) || (x + y - 1.0).abs() <= 8.0 * Real::EPSILON,
        "incomplete_beta_xy needs y = 1 - x (x = {x}, y = {y})"
    );
    // `is_finite()` is required: a bare `<= 0.0` lets NaN and +infinity through
    // (all NaN comparisons are false, and +infinity is not <= 0), but
    // QuantLib's QL_REQUIRE(a > 0) rejects them
    if !a.is_finite() || a <= 0.0 {
        fail!("incomplete_beta requires a finite a > 0, got a={a}");
    }
    if !b.is_finite() || b <= 0.0 {
        fail!("incomplete_beta requires a finite b > 0, got b={b}");
    }
    // `contains` is false for NaN, so this also rejects NaN
    if !(0.0..=1.0).contains(&x) {
        fail!("incomplete_beta requires x in [0, 1], got x={x}");
    }
    if !(0.0..=1.0).contains(&y) {
        fail!("incomplete_beta requires y in [0, 1], got y={y}");
    }
    if x == 0.0 {
        return Ok(0.0);
    }
    if y == 0.0 {
        return Ok(1.0);
    }

    let prefactor = ibeta_power_terms(a, b, x, y);

    // use the expansion that converges fastest for this x
    if x < (a + 1.0) / (a + b + 2.0) {
        Ok(prefactor * beta_continued_fraction(a, b, x)? / a)
    } else {
        Ok(1.0 - prefactor * beta_continued_fraction(b, a, y)? / b)
    }
}

/// Boost's `lanczos13m53::lanczos_sum_expG_scaled(z)`, the Lanczos sum
/// `Γ(z) e^(z - 1/2) / (z + g - 1/2)^(z - 1/2)` for `z > 0`.
///
/// Evaluates the rational function as Boost's `tools::evaluate_rational`: in
/// `z` for `z <= 1`, and in `1/z` above, so large `z` cannot overflow.
fn lanczos_sum_exp_g_scaled(z: Real) -> Real {
    let (mut num, mut denom);
    if z <= 1.0 {
        num = LANCZOS_NUM[12];
        denom = LANCZOS_DENOM[12];
        for i in (0..12).rev() {
            num = num * z + LANCZOS_NUM[i];
            denom = denom * z + LANCZOS_DENOM[i];
        }
    } else {
        let z = 1.0 / z;
        num = LANCZOS_NUM[0];
        denom = LANCZOS_DENOM[0];
        for i in 1..13 {
            num = num * z + LANCZOS_NUM[i];
            denom = denom * z + LANCZOS_DENOM[i];
        }
    }
    num / denom
}

/// Boost's normalised `ibeta_power_terms(a, b, x, y)`: the leading factor
/// `x^a y^b / B(a, b)` of the incomplete beta, from `y` as given (not
/// `1 - x`), for `a, b > 0` and `x, y` in `(0, 1)`.
///
/// The Lanczos sums carry `Γ(a + b) / (Γ(a) Γ(b))` and the bases of the two
/// powers are rescaled by `(a + b + g - 1/2) / (a + g - 1/2)` and its `b`
/// analogue, so where a base is near 1 the power is formed as
/// `exp(a log1p(l))` rather than from cancelling logarithms. Boost raises an
/// underflow or overflow error where the product is `0` or `∞`; the product
/// is returned instead, so a deep tail evaluates to `0`.
fn ibeta_power_terms(a: Real, b: Real, x: Real, y: Real) -> Real {
    let c = a + b;
    let gh = LANCZOS_G - 0.5;
    let agh = a + gh;
    let bgh = b + gh;
    let cgh = c + gh;
    let log_max = Real::MAX.ln();
    let log_min = Real::MIN_POSITIVE.ln();
    let mut result = if a < Real::MIN_POSITIVE || b < Real::MIN_POSITIVE {
        0.0
    } else {
        lanczos_sum_exp_g_scaled(c) / (lanczos_sum_exp_g_scaled(a) * lanczos_sum_exp_g_scaled(b))
    };
    result *= (bgh / E).sqrt();
    result *= (agh / cgh).sqrt();

    let l1 = ((x * b - y * a) - y * gh) / agh;
    let l2 = ((y * a - x * b) - x * gh) / bgh;
    if l1.abs().min(l2.abs()) < 0.2 {
        if l1 * l2 > 0.0 || a.min(b) < 1.0 {
            result *= if l1.abs() < 0.1 {
                (a * l1.ln_1p()).exp()
            } else {
                (x * cgh / agh).powf(a)
            };
            result *= if l2.abs() < 0.1 {
                (b * l2.ln_1p()).exp()
            } else {
                (y * cgh / bgh).powf(b)
            };
        } else if l1.abs().max(l2.abs()) < 0.5 {
            let ratio = b / a;
            if (a < b && ratio * l2 < 0.1) || (a >= b && l1 / ratio > 0.1) {
                let l3 = (ratio * l2.ln_1p()).exp_m1();
                result *= (a * (l1 + l3 + l3 * l1).ln_1p()).exp();
            } else {
                let l3 = (l1.ln_1p() / ratio).exp_m1();
                result *= (b * (l2 + l3 + l3 * l2).ln_1p()).exp();
            }
        } else {
            let l = if l1.abs() < l2.abs() {
                a * l1.ln_1p() + b * (y * cgh / bgh).ln()
            } else {
                b * l2.ln_1p() + a * (x * cgh / agh).ln()
            };
            result = if l <= log_min || l >= log_max {
                (l + result.ln()).exp()
            } else {
                result * l.exp()
            };
        }
    } else {
        let b1 = x * cgh / agh;
        let b2 = y * cgh / bgh;
        let l1 = a * b1.ln();
        let l2 = b * b2.ln();
        if l1 >= log_max || l1 <= log_min || l2 >= log_max || l2 <= log_min {
            if a < b {
                let p1 = b2.powf(b / a);
                let l3 = if b1 != 0.0 && p1 != 0.0 {
                    a * (b1.ln() + p1.ln())
                } else {
                    Real::MAX
                };
                result = if l3 < log_max && l3 > log_min {
                    result * (p1 * b1).powf(a)
                } else {
                    (l2 + l1 + result.ln()).exp()
                };
            } else {
                let p1 = if b1 < 1.0 && b < 1.0 && Real::MAX * b < a {
                    0.0
                } else {
                    b1.powf(a / b)
                };
                let l3 = if p1 != 0.0 && b2 != 0.0 {
                    (p1.ln() + b2.ln()) * b
                } else {
                    Real::MAX
                };
                if l3 < log_max && l3 > log_min {
                    result *= (p1 * b2).powf(b);
                } else if result != 0.0 {
                    result = (l2 + l1 + result.ln()).exp();
                }
            }
        } else {
            result *= b1.powf(a) * b2.powf(b);
        }
    }
    result
}

/// Lentz's continued-fraction evaluation for [`incomplete_beta`] (NR `betacf`).
fn beta_continued_fraction(a: Real, b: Real, x: Real) -> QlResult<Real> {
    let eps = Real::EPSILON;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < eps {
        d = eps;
    }
    d = 1.0 / d;
    let mut result = d;
    let max_iterations = MIN_ITERATIONS
        .max((10.0 * a.max(b).sqrt()).ceil() as u32)
        .min(MAX_ITERATIONS);

    for iter in 1..=max_iterations {
        let m = iter as Real;
        let m2 = 2.0 * m;

        // even step
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + aa / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        result *= d * c;

        // odd step
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < eps {
            d = eps;
        }
        c = 1.0 + aa / c;
        if c.abs() < eps {
            c = eps;
        }
        d = 1.0 / d;
        let del = d * c;
        result *= del;
        if (del - 1.0).abs() < ACCURACY {
            return Ok(result);
        }
        if iter > MIN_ITERATIONS && result.is_nan() {
            fail!("incomplete_beta continued fraction produced NaN (a={a}, b={b})");
        }
    }
    fail!("incomplete_beta continued fraction did not converge (a={a}, b={b})");
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: Real = 1e-12;

    fn assert_close(got: Real, expected: Real) {
        let tol = TOL * (1.0 + expected.abs());
        assert!(
            (got - expected).abs() <= tol,
            "got {got}, expected {expected}, diff {}",
            (got - expected).abs()
        );
    }

    #[test]
    fn beta_function_known_values() {
        // B(2,3) = 1!·2!/4! = 1/12
        assert_close(beta_function(2.0, 3.0).unwrap(), 1.0 / 12.0);
        // B(1/2,1/2) = Γ(1/2)²/Γ(1) = π
        assert_close(beta_function(0.5, 0.5).unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn boundaries_and_uniform() {
        assert_eq!(incomplete_beta(2.0, 3.0, 0.0).unwrap(), 0.0);
        assert_eq!(incomplete_beta(2.0, 3.0, 1.0).unwrap(), 1.0);
        // Beta(1,1) is uniform: I_x(1,1) = x
        assert_close(incomplete_beta(1.0, 1.0, 0.3).unwrap(), 0.3);
        assert_close(incomplete_beta(1.0, 1.0, 0.5).unwrap(), 0.5);
    }

    #[test]
    fn known_values() {
        // symmetric Beta(2,2): I_{0.5} = 0.5
        assert_close(incomplete_beta(2.0, 2.0, 0.5).unwrap(), 0.5);
        // Beta(2,3) CDF at 0.5 = 0.6875 (ELSE branch, x > 3/7; computed by hand)
        assert_close(incomplete_beta(2.0, 3.0, 0.5).unwrap(), 0.6875);
        // Beta(2,3) CDF at 0.3 = 0.3483 (IF/direct-CF branch, x < 3/7)
        assert_close(incomplete_beta(2.0, 3.0, 0.3).unwrap(), 0.3483);
    }

    #[test]
    fn symmetry_identity() {
        // I_x(a,b) = 1 - I_{1-x}(b,a)
        for &(a, b, x) in &[(2.0, 3.0, 0.3), (0.5, 2.5, 0.7), (4.0, 1.5, 0.2)] {
            let lhs = incomplete_beta(a, b, x).unwrap();
            let rhs = 1.0 - incomplete_beta(b, a, 1.0 - x).unwrap();
            assert!(
                (lhs - rhs).abs() < TOL,
                "symmetry failed at a={a}, b={b}, x={x}"
            );
        }
    }

    /// mpmath at 60 digits (`tests/fixtures/ibeta_power_terms/generate.py`),
    /// on both arms of the rational evaluation (`z <= 1` and `z > 1`).
    #[test]
    fn lanczos_sum_exp_g_scaled_matches_mpmath() {
        let rows: Vec<(Real, Real)> =
            include_str!("../../tests/fixtures/ibeta_power_terms/cases.csv")
                .lines()
                .filter_map(|line| line.strip_prefix("lanczos,"))
                .map(|line| {
                    let (z, value) = line.split_once(',').unwrap();
                    (z.parse().unwrap(), value.parse().unwrap())
                })
                .collect();
        assert_eq!(rows.len(), 14);
        for (z, expected) in rows {
            let got = lanczos_sum_exp_g_scaled(z);
            let rel = ((got - expected) / expected).abs();
            assert!(
                rel <= 1e-15,
                "z={z}: got {got}, expected {expected}, rel {rel:e}"
            );
        }
    }

    #[test]
    fn invalid_args_rejected() {
        assert!(incomplete_beta(0.0, 1.0, 0.5).is_err()); // a <= 0
        assert!(incomplete_beta(1.0, -1.0, 0.5).is_err()); // b <= 0
        assert!(incomplete_beta(1.0, 1.0, 1.5).is_err()); // x out of range
        assert!(incomplete_beta(1.0, 1.0, Real::NAN).is_err()); // NaN x
        assert!(incomplete_beta(Real::NAN, 1.0, 0.5).is_err()); // NaN a
        assert!(incomplete_beta(1.0, Real::NAN, 0.5).is_err()); // NaN b
        assert!(incomplete_beta(Real::INFINITY, 1.0, 0.5).is_err()); // +inf a
        assert!(incomplete_beta(1.0, Real::INFINITY, 0.5).is_err()); // +inf b
        // beta_function propagates through log_gamma, so infinity errors too
        assert!(beta_function(Real::INFINITY, 1.0).is_err());
    }
}
