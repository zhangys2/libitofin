//! Piecewise-constant scale `f(t)` of the Carr–Madan intensity `λ(t, S) = f(t) S^p`.
//!
//! Working-paper eq. 1. `p = 0` is the constant-hazard case the closed-form
//! oracles use. `f` is flat before the first knot and after the last one.

use crate::errors::QlResult;
use crate::require;
use crate::types::{Real, Time};

/// Jump-to-default intensity `λ(t, S) = f(t) S^p` with piecewise-constant `f`.
///
/// Carr–Madan working-paper eq. 1. Knot `i` holds the scale on
/// `(times[i-1], times[i]]`, with `times[-1] = -∞`, and the last scale is
/// used for every later time.
#[derive(Clone, Debug, PartialEq)]
pub struct PowerHazard {
    times: Vec<Time>,
    scales: Vec<Real>,
    exponent: Real,
}

impl PowerHazard {
    /// Piecewise-constant scale on strictly increasing knot times.
    ///
    /// # Errors
    ///
    /// Returns an error when `times` and `scales` differ in length, either is
    /// empty, a time is non-finite or not strictly increasing, the first time
    /// is not positive, a scale is negative or non-finite, or `exponent` is
    /// greater than zero or non-finite. `exponent == 0` is the constant
    /// hazard.
    pub fn new(times: Vec<Time>, scales: Vec<Real>, exponent: Real) -> QlResult<Self> {
        require!(
            !times.is_empty(),
            "jump-to-default hazard needs at least one knot"
        );
        require!(
            times.len() == scales.len(),
            "jump-to-default hazard times and scales differ in length ({} vs {})",
            times.len(),
            scales.len()
        );
        require!(
            times.iter().all(|time| time.is_finite()),
            "jump-to-default hazard times must be finite"
        );
        require!(
            times[0] > 0.0,
            "jump-to-default hazard knots must start after 0, got {}",
            times[0]
        );
        require!(
            times.windows(2).all(|pair| pair[1] > pair[0]),
            "jump-to-default hazard times must be strictly increasing"
        );
        require!(
            scales
                .iter()
                .all(|scale| scale.is_finite() && *scale >= 0.0),
            "jump-to-default hazard scales must be finite and non-negative"
        );
        require!(
            exponent.is_finite() && exponent <= 0.0,
            "jump-to-default exponent must be finite and <= 0, got {exponent}"
        );
        Ok(Self {
            times,
            scales,
            exponent,
        })
    }

    /// A scale that is `scale` at every time.
    ///
    /// # Errors
    ///
    /// Same scale and exponent checks as [`new`](Self::new).
    pub fn constant(scale: Real, exponent: Real) -> QlResult<Self> {
        Self::new(vec![1.0], vec![scale], exponent)
    }

    /// Scale `f(t)`. Flat before the first knot and after the last one.
    ///
    /// The value on `(times[i-1], times[i]]` is `scales[i]`, with
    /// `times[-1] = -∞`. Calibration stores the right-endpoint sample there,
    /// so inside a bucket this is `f` at the bucket's right knot.
    ///
    /// Carr–Madan working-paper eq. 8 writes the survival hazard as
    /// `h_V(t) = f(t) E[S_t^p]`.
    pub fn scale(&self, t: Time) -> Real {
        let index = self.times.partition_point(|knot| *knot < t);
        self.scales[index.min(self.scales.len() - 1)]
    }

    /// Intensity `f(t) S^p` (working-paper eq. 1).
    ///
    /// # Errors
    ///
    /// Returns an error when `spot` is non-finite or not strictly positive.
    /// The model is a diffusion in `ln S` and the power is undefined at zero.
    pub fn intensity(&self, t: Time, spot: Real) -> QlResult<Real> {
        require!(
            spot.is_finite() && spot > 0.0,
            "jump-to-default intensity needs a positive finite spot, got {spot}"
        );
        Ok(self.scale(t) * spot.powf(self.exponent))
    }

    /// The power `p` in `S^p`.
    pub fn exponent(&self) -> Real {
        self.exponent
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::Real;

    #[test]
    fn validation_rejects_bad_knots_scales_and_exponent() {
        assert!(PowerHazard::new(vec![], vec![], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0], vec![0.1, 0.2], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0, 1.0], vec![0.1, 0.2], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0, 0.5], vec![0.1, 0.2], 0.0).is_err());
        assert!(PowerHazard::new(vec![0.0], vec![0.1], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0], vec![-0.1], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0], vec![Real::NAN], 0.0).is_err());
        assert!(PowerHazard::new(vec![1.0], vec![0.1], 0.5).is_err());
        assert!(PowerHazard::new(vec![1.0], vec![0.1], Real::NAN).is_err());
        assert!(PowerHazard::constant(-1.0, 0.0).is_err());
    }

    #[test]
    fn scale_and_intensity_follow_the_knots_and_extrapolate_flat() {
        let hazard = PowerHazard::new(vec![0.5, 1.5], vec![0.2, 0.4], -1.5).unwrap();
        assert_eq!(hazard.exponent(), -1.5);
        assert_eq!(hazard.scale(0.1), 0.2);
        assert_eq!(hazard.scale(0.5), 0.2);
        assert_eq!(hazard.scale(1.0), 0.4);
        assert_eq!(hazard.scale(1.5), 0.4);
        assert_eq!(hazard.scale(4.0), 0.4);

        let spot: Real = 4.0;
        let expected = 0.4 * spot.powf(-1.5);
        assert!((hazard.intensity(2.0, spot).unwrap() - expected).abs() < 1e-14);
        assert!(hazard.intensity(1.0, 0.0).is_err());
        assert!(hazard.intensity(1.0, -2.0).is_err());
    }

    #[test]
    fn zero_exponent_is_a_constant_hazard() {
        let hazard = PowerHazard::constant(0.07, 0.0).unwrap();
        assert_eq!(hazard.scale(0.0), 0.07);
        assert_eq!(hazard.scale(10.0), 0.07);
        assert_eq!(hazard.intensity(3.0, 80.0).unwrap(), 0.07);
    }
}
