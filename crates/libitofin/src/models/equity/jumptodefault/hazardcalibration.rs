//! Hazard scale `f(t)` from a survival curve and the call surface.
//!
//! Working-paper eq. 8: `h_V(t) = f(t) E[S_t^p]`, and the conditional power
//! moment is `M_p(0, t) / (D(t) V(t))`, so
//! `f(t) = h_V(t) D(t) V(t) / M_p(0, t)`.

use crate::errors::QlResult;
use crate::require;
use crate::termstructures::credit::defaulttermstructure::DefaultProbabilityTermStructure;
use crate::types::{Real, Time};

use super::powermoments::{MomentGrid, power_price};
use super::{CallSurface, PowerHazard};

/// Piecewise-constant `f` on `times`, for the power `exponent`.
///
/// Each knot uses the survival curve's hazard rate and survival probability
/// and the call surface's discount. `M_p(0, t)` is the grid's truncated
/// power price.
///
/// # Errors
///
/// Returns an error when a knot is not positive, the power moment is not a
/// positive finite number, or [`PowerHazard::new`](PowerHazard::new) rejects
/// the knots.
pub fn calibrate_power_hazard(
    surface: &dyn CallSurface,
    survival: &dyn DefaultProbabilityTermStructure,
    exponent: Real,
    times: &[Time],
    grid: &MomentGrid,
) -> QlResult<PowerHazard> {
    let mut scales = Vec::with_capacity(times.len());
    for &t in times {
        require!(
            t.is_finite() && t > 0.0,
            "hazard calibration knot must be positive and finite, got {t}"
        );
        let hazard = survival.hazard_rate(t, true)?;
        let survival_prob = survival.survival_probability(t, true)?;
        let discount = surface.discount(t)?;
        let moment = power_price(surface, exponent, t, grid)?;
        require!(
            moment.is_finite() && moment > 0.0,
            "power moment at time {t} is {moment}, so the hazard scale is undefined"
        );
        scales.push(hazard * discount * survival_prob / moment);
    }
    PowerHazard::new(times.to_vec(), scales, exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::termstructures::credit::flathazardrate::FlatHazardRate;
    use crate::time::date::{Date, Month};
    use crate::time::daycounters::actual365fixed::Actual365Fixed;
    use crate::types::Real;

    use super::super::callsurface::ConstantHazardCallSurface;

    fn assert_rel(actual: Real, expected: Real, tol: Real) {
        let scale = expected.abs().max(1e-12);
        let error = (actual - expected).abs() / scale;
        assert!(error <= tol, "{actual} vs {expected} (rel {error})");
    }

    fn survival(hazard: Real) -> FlatHazardRate {
        FlatHazardRate::with_rate(
            Date::new(15, Month::June, 2026),
            hazard,
            Actual365Fixed::new(),
        )
    }

    #[test]
    fn zero_power_constant_hazard_is_recovered() {
        let h = 0.04;
        let surface = ConstantHazardCallSurface::new(100.0, 0.03, 0.01, 0.2, h).unwrap();
        let curve = survival(h);
        let grid = MomentGrid::new(12.0, 64).unwrap();
        let hazard =
            calibrate_power_hazard(&surface, &curve, 0.0, &[0.25, 1.0, 2.5], &grid).unwrap();
        for t in [0.2, 0.25, 1.0, 2.5, 4.0] {
            assert_rel(hazard.scale(t), h, 1e-6);
        }
    }

    #[test]
    fn negative_power_scale_matches_the_conditional_moment() {
        let h = 0.03;
        let vol = 0.22;
        let rate = 0.03;
        let dividend = 0.01;
        let spot = 100.0;
        let exponent = -1.5;
        let surface = ConstantHazardCallSurface::new(spot, rate, dividend, vol, h).unwrap();
        let curve = survival(h);
        let grid = MomentGrid::new(12.0, 160).unwrap();
        let times = [0.5, 1.0, 2.0];
        let hazard = calibrate_power_hazard(&surface, &curve, exponent, &times, &grid).unwrap();
        for t in times {
            let forward = spot * ((rate - dividend + h) * t).exp();
            let variance = vol * vol * t;
            let expected =
                h / (forward.powf(exponent) * (0.5 * exponent * (exponent - 1.0) * variance).exp());
            assert_rel(hazard.scale(t), expected, 1e-6);
        }
    }
}
